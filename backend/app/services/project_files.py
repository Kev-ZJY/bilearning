"""Portable project snapshots. DB commit and filesystem export are separate outcomes."""

import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import RLock
from contextlib import contextmanager
import uuid

from app.core.database import session_scope
from app.models.entities import (
    Project,
    Video,
    Message,
    KnowledgeDoc,
    KnowledgeRevision,
    AppSetting,
    ConversationSummary,
)

_lock = RLock()
_errors: dict[int, str] = {}
TERMINAL = {"pending", "done", "failed", "cancelled", "skipped"}


@contextmanager
def file_guard():
    with _lock:
        yield


def next_video_id(db, root):
    from sqlalchemy import func

    highest = db.query(func.max(Video.id)).scalar() or 0
    for folder in root.glob("video_*"):
        suffix = folder.name.removeprefix("video_")
        if suffix.isdecimal():
            highest = max(highest, int(suffix))
    return highest + 1


def key(pid):
    return f"desktop_project_{pid}"


def atomic(path: Path, content: str):
    fd, tmp = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def identity(db, pid):
    row = db.get(AppSetting, key(pid))
    return row.value if row else None


def check_root(db, p):
    project_uuid = identity(db, p.id)
    if not project_uuid:
        return None
    root = Path(p.storage_root)
    manifest = root / ".video-workbench" / "project.json"
    if not manifest.is_file():
        raise ValueError(
            "项目目录不可用；请连接原磁盘，或使用「打开项目」选择搬迁后的目录"
        )
    if json.loads(manifest.read_text()).get("uuid") != project_uuid:
        raise ValueError("目录中的项目身份不匹配；已停止写入，请重新打开正确目录")
    return root


def status(pid):
    try:
        with session_scope() as db:
            p = db.get(Project, pid)
            if not p:
                return {"status": "missing", "message": "项目不存在"}
            root = check_root(db, p)
            if root is None:
                return {
                    "status": "legacy",
                    "message": "旧开发项目，尚未建立可移动的目录快照",
                }
        if pid in _errors:
            return {"status": "error", "message": _errors[pid]}
        return {"status": "saved", "message": "项目内容已保存到所选目录"}
    except (OSError, ValueError):
        return {
            "status": "unavailable",
            "message": "目录不可用或身份不匹配，请重新连接磁盘或打开项目目录",
        }


def create(root: Path, title: str, subject: str):
    marker = root / ".video-workbench"
    # reopen snapshots queue state before taking the file lock.
    if marker.exists():
        return reopen(root)
    with _lock:
        if marker.exists():
            raise ValueError("目录正在被打开，请重试")
        # Exclusive ownership marker; never overwrite pre-existing user files.
        marker.mkdir()
        project_uuid = uuid.uuid4().hex
        atomic(
            marker / "project.json", json.dumps({"version": 1, "uuid": project_uuid})
        )
        try:
            with session_scope() as db:
                p = Project(title=title, subject=subject, storage_root=str(root))
                db.add(p)
                db.flush()
                db.add(AppSetting(key=key(p.id), value=project_uuid))
                pid = p.id
        except Exception:
            # Leave the marker to avoid silently deleting a user-selected directory.
            raise
        persist(pid)
        return pid


def persist(pid):
    """Export under a lock, loading latest DB after acquiring it. Retain every snapshot."""
    with _lock:
        try:
            with session_scope() as db:
                p = db.get(Project, pid)
                if not p or not identity(db, pid):
                    return
                root = check_root(db, p)
                marker = root / ".video-workbench"
                data = {
                    "version": 1,
                    "uuid": identity(db, pid),
                    "title": p.title,
                    "subject": p.subject,
                    "knowledge": json.loads(p.doc.doc_json) if p.doc else None,
                    "messages": [
                        {
                            "id": m.id,
                            "role": m.role,
                            "content": m.content,
                            "tool_calls_json": m.tool_calls_json,
                            "task_refs_json": m.task_refs_json,
                        }
                        for m in db.query(Message)
                        .filter_by(project_id=pid)
                        .order_by(Message.id.asc())
                        .all()
                    ],
                    "videos": [
                        {
                            "folder": f"video_{v.id}",
                            "bvid": v.bvid,
                            "cid": v.cid,
                            "source": v.source,
                            "title": v.title,
                            "duration": v.duration,
                            "cover": v.cover,
                            "status": v.status,
                            "error": v.error,
                        }
                        for v in p.videos
                    ],
                    "revisions": [
                        r.doc_json
                        for r in db.query(KnowledgeRevision)
                        .filter_by(project_id=pid)
                        .all()
                    ],
                }
                summary = db.query(ConversationSummary).filter_by(
                    project_id=pid
                ).one_or_none()
                data["conversation_summary"] = None if summary is None else {
                    "version": summary.version,
                    "summary": json.loads(summary.summary_json),
                    "source_hash": summary.source_hash,
                    "source_message_ids": json.loads(summary.source_message_ids_json),
                    "source_message_hashes": json.loads(
                        summary.source_message_hashes_json
                    ),
                    "covered_until_user_message_id": summary.covered_until_user_message_id,
                    "snapshot_anchor_message_id": summary.snapshot_anchor_message_id,
                    "updated_at": summary.updated_at.isoformat()
                    if summary.updated_at else None,
                }
                # Local paths are made relative only in generated manifests, never copied from arbitrary input.
                for v in p.videos:
                    local = root / f"video_{v.id}" / "local.json"
                    if v.source == "local" and local.is_file():
                        item = json.loads(local.read_text())
                        data["videos"][p.videos.index(v)]["media"] = str(
                            Path(item["video_path"])
                            .resolve()
                            .relative_to(root.resolve())
                        )
            snapshots = marker / "snapshots"
            snapshots.mkdir(exist_ok=True)
            revision = uuid.uuid4().hex
            folder = snapshots / revision
            folder.mkdir()
            atomic(
                folder / "project.json", json.dumps(data, ensure_ascii=False, indent=2)
            )
            if data["knowledge"]:
                from app.worker.renderer import render_standalone_html, render_markdown

                atomic(
                    folder / "knowledge.html", render_standalone_html(data["knowledge"])
                )
                atomic(folder / "knowledge.md", render_markdown(data["knowledge"]))
                # Convenient current files live inside the owned marker, never clobber user files at root.
                atomic(
                    marker / "knowledge.html", render_standalone_html(data["knowledge"])
                )
                atomic(marker / "knowledge.md", render_markdown(data["knowledge"]))
            atomic(
                marker / "project.json",
                json.dumps({"version": 1, "uuid": data["uuid"], "snapshot": revision}),
            )
            _errors.pop(pid, None)
        except Exception:
            _errors[pid] = (
                "数据库已保留最新内容，但项目文件保存失败；请检查目录权限与磁盘空间后重试保存"
            )


def safe_child(root, name):
    if not isinstance(name, str):
        raise ValueError("项目文件路径无效")
    child = (root / name).resolve()
    if not child.is_relative_to(root.resolve()):
        raise ValueError("项目文件不能指向目录之外")
    return child


def reopen(root):
    from app.core.queue import get_queue

    tasks = get_queue().list_all()
    with _lock:
        marker = root / ".video-workbench"
        manifest = json.loads((marker / "project.json").read_text())
        project_uuid = manifest["uuid"]
        # Reconnecting an indexed project changes only its root, preserving newer DB data.
        with session_scope() as db:
            row = (
                db.query(AppSetting)
                .filter(
                    AppSetting.value == project_uuid,
                    AppSetting.key.like("desktop_project_%"),
                )
                .first()
            )
            if row:
                pid = int(row.key.removeprefix("desktop_project_"))
                if any(
                    t.get("payload", {}).get("project_id") == pid
                    and (
                        t.get("status") not in ("done", "failed", "cancelled")
                        or (
                            t.get("payload", {}).get("kind") == "pipeline"
                            and not t.get("worker_finished")
                        )
                    )
                    for t in tasks
                ):
                    raise ValueError("项目仍有任务运行，请先停止任务再重新定位目录")
                p = db.get(Project, pid)
                old = Path(p.storage_root)
                for v in p.videos:
                    local = root / f"video_{v.id}" / "local.json"
                    if v.source == "local" and local.exists():
                        art = json.loads(local.read_text())
                        media = Path(art["video_path"])
                        rel = media.relative_to(old) if media.is_absolute() else media
                        target = safe_child(root, str(rel))
                        atomic(
                            local,
                            json.dumps(
                                {"video_path": str(target), "audio_path": str(target)}
                            ),
                        )
                p.storage_root = str(root)
                return pid
        revision = manifest.get("snapshot", "")
        if len(revision) != 32 or not all(c in "0123456789abcdef" for c in revision):
            raise ValueError("项目没有完整快照")
        data = json.loads(
            (marker / "snapshots" / revision / "project.json").read_text()
        )
        if data.get("version") != 1 or data.get("uuid") != project_uuid:
            raise ValueError("项目快照身份不匹配")
        # Validate paths before any database writes or copies.
        for v in data.get("videos", []):
            safe_child(root, v["folder"])
            if v.get("media"):
                safe_child(root, v["media"])
        with session_scope() as db:
            p = Project(
                title=data["title"],
                subject=data.get("subject", ""),
                storage_root=str(root),
            )
            db.add(p)
            db.flush()
            db.add(AppSetting(key=key(p.id), value=project_uuid))
            if data.get("knowledge"):
                db.add(
                    KnowledgeDoc(
                        project_id=p.id,
                        doc_json=json.dumps(data["knowledge"], ensure_ascii=False),
                    )
                )
            old_to_new_message_ids = {}
            for m in data.get("messages", []):
                if m.get("role") not in ("user", "assistant", "tool"):
                    continue
                refs = m.get("task_refs_json")
                if refs:
                    try:
                        decoded_refs = json.loads(refs) if isinstance(refs, str) else dict(refs)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        decoded_refs = None
                    if isinstance(decoded_refs, dict):
                        target = decoded_refs.get("reply_to_message_id")
                        if target is not None:
                            try:
                                target_id = int(target)
                            except (TypeError, ValueError):
                                target_id = None
                            mapped = old_to_new_message_ids.get(target_id)
                            if mapped is None:
                                # A stale or missing reference must not be rebound
                                # to a different message after import.
                                decoded_refs.pop("reply_to_message_id", None)
                            else:
                                decoded_refs["reply_to_message_id"] = mapped
                        refs = json.dumps(decoded_refs, ensure_ascii=False)
                row = Message(
                    project_id=p.id,
                    role=m["role"],
                    content=m["content"],
                    tool_calls_json=m.get("tool_calls_json"),
                    task_refs_json=refs,
                )
                db.add(row)
                db.flush()
                old_id = m.get("id")
                if old_id is not None:
                    try:
                        old_to_new_message_ids[int(old_id)] = row.id
                    except (TypeError, ValueError):
                        pass
            # Deliberately do not import conversation_summary: source message
            # IDs and the turn cursor belong to the exported database. The raw
            # transcript and reply links were remapped above, so the next chat
            # can rebuild a fresh summary against its new IDs.
            for raw in data.get("revisions", []):
                db.add(KnowledgeRevision(project_id=p.id, doc_json=raw))
            for item in data.get("videos", []):
                folder_name = Path(item["folder"]).name
                original_id = (
                    int(folder_name.removeprefix("video_"))
                    if folder_name.startswith("video_")
                    and folder_name.removeprefix("video_").isdecimal()
                    else 0
                )
                vid = (
                    original_id
                    if original_id > 0 and db.get(Video, original_id) is None
                    else next_video_id(db, root)
                )
                v = Video(
                    id=vid,
                    project_id=p.id,
                    **{
                        k: item.get(k)
                        for k in ("bvid", "cid", "source", "title", "duration", "cover")
                    },
                    status=item["status"] if item["status"] in TERMINAL else "failed",
                    error=item.get("error"),
                )
                db.add(v)
                db.flush()
                original = safe_child(root, item["folder"])
                dest = root / f"video_{v.id}"
                if original.exists() and original != dest:
                    if dest.exists():
                        raise ValueError("导入的视频目录冲突；原文件已保留")
                    shutil.copytree(original, dest, symlinks=True)
                if item.get("media"):
                    dest.mkdir(exist_ok=True)
                    media = safe_child(root, item["media"])
                    atomic(
                        dest / "local.json",
                        json.dumps(
                            {"video_path": str(media), "audio_path": str(media)}
                        ),
                    )
            return p.id
