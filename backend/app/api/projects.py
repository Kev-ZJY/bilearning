"""项目接口：创建 / 列表 / 详情。"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    title: str
    subject: str = ""
    directory_grant: str | None = None


class ResumeRequest(BaseModel):
    video_id: int
    recovery_id: str = Field(pattern=r"^[a-f0-9]{24}$")


@router.post("")
def create_project(body: ProjectCreate):
    from app.core.config import get_settings
    from app.core.database import session_scope
    from app.models.entities import Project

    from app.services.desktop import enabled, consume

    if enabled():
        if not body.directory_grant:
            raise HTTPException(400, "请先选择项目保存目录")
        from app.services.project_files import create

        try:
            pid = create(
                consume(body.directory_grant, "directory"), body.title, body.subject
            )
            return get_project(pid)
        except (ValueError, KeyError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc
    s = get_settings()
    with session_scope() as db:
        p = Project(
            title=body.title,
            subject=body.subject,
            storage_root=s.storage_root_expanded,
        )
        db.add(p)
        db.commit()
        return {"id": p.id, "title": p.title, "subject": p.subject}


@router.get("")
def list_projects():
    from app.core.database import session_scope
    from app.models.entities import Project

    with session_scope() as db:
        return [
            {
                "id": p.id,
                "title": p.title,
                "subject": p.subject,
                "storage_root": p.storage_root,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in db.query(Project).order_by(Project.updated_at.desc()).all()
        ]


@router.get("/{pid}")
def get_project(pid: int):
    import json

    from app.core.database import session_scope
    from app.models.entities import Project

    with session_scope() as db:
        p = db.get(Project, pid)
        if not p:
            raise HTTPException(404, "项目不存在")
        doc = None
        if p.doc and p.doc.doc_json:
            try:
                doc = json.loads(p.doc.doc_json)
            except json.JSONDecodeError:
                doc = None
        from app.services.project_files import status
        from app.api.messages import _serialize

        return {
            "storage_status": status(pid),
            "existing_document": existing_document_path(p) is not None,
            "id": p.id,
            "title": p.title,
            "subject": p.subject,
            "storage_root": p.storage_root,
            "videos": [
                {
                    "id": v.id,
                    "bvid": v.bvid,
                    "title": v.title,
                    "duration": v.duration,
                    "status": v.status,
                    "error": v.error,
                }
                for v in p.videos
            ],
            "messages": [
                _serialize(m) for m in p.messages
            ],
            "knowledge": doc,
        }


@router.get("/{pid}/revisions")
def list_revisions(pid: int):
    from app.core.database import session_scope
    from app.models.entities import Project, KnowledgeRevision

    with session_scope() as db:
        if db.get(Project, pid) is None:
            raise HTTPException(404, "项目不存在")
        return [
            {"id": r.id, "created_at": r.created_at.isoformat()}
            for r in db.query(KnowledgeRevision)
            .filter_by(project_id=pid)
            .order_by(KnowledgeRevision.id.desc())
            .all()
        ]


@router.get("/{pid}/revisions/{revision_id}")
def read_revision(pid: int, revision_id: int):
    import json
    from app.core.database import session_scope
    from app.models.entities import KnowledgeRevision

    with session_scope() as db:
        revision = db.get(KnowledgeRevision, revision_id)
        if revision is None or revision.project_id != pid:
            raise HTTPException(404, "版本不存在")
        return {"id": revision.id, "knowledge": json.loads(revision.doc_json)}


@router.get("/{pid}/tasks")
def project_tasks(pid: int):
    from datetime import datetime, timezone
    from app.core.queue import get_queue
    from app.core.database import session_scope
    from app.models.entities import Project
    from pathlib import Path
    import hashlib

    with session_scope() as db:
        p = db.get(Project, pid)
        if not p:
            raise HTTPException(404, "项目不存在")
        root = Path(p.storage_root)
    from app.services.recovery import RecoveryError, discover_project_recoveries

    try:
        recoveries = discover_project_recoveries(pid)
    except (RecoveryError, OSError, ValueError):
        recoveries = []
    recovery_map = {
        (row.get("video_id"), row.get("recovery_id")): row for row in recoveries
    }
    result = []
    seen_recoveries = set()
    for t in get_queue().list_all():
        payload = t.get("payload", {})
        if payload.get("project_id") != pid:
            continue
        video_id = payload.get("video_id")
        recovery_id = payload.get("recovery_id")
        if not recovery_id and video_id and payload.get("kind") == "pipeline":
            recovery_id = hashlib.sha256(t["task_id"].encode()).hexdigest()[:24]
        recovery = recovery_map.get((video_id, recovery_id), {})
        if recovery_id:
            seen_recoveries.add((video_id, recovery_id))
        draft = (
            root
            / f"video_{video_id}"
            / "tasks"
            / hashlib.sha256(t["task_id"].encode()).hexdigest()[:24]
            / "draft.json"
        )
        result.append(
            {
                **{
                    k: t.get(k)
                    for k in (
                        "task_id",
                        "status",
                        "progress",
                        "message",
                        "stage",
                        "error",
                    )
                },
                "kind": payload.get("kind"),
                "draft_available": bool(video_id and draft.is_file()),
                "video_id": video_id,
                "recovery_id": recovery_id,
                "source_task_id": recovery.get("source_task_id"),
                "source_recovery_id": recovery.get("source_recovery_id"),
                "resumable": bool(recovery.get("resumable", False)),
                "rebuildable": bool(recovery.get("rebuildable", False)),
                "recovery_reason": recovery.get("recovery_reason"),
                "created_at": recovery.get("created_at") or datetime.fromtimestamp(
                    ((t.get("activity") or {}).get("started_at") or 0) / 1000,
                    tz=timezone.utc,
                ).isoformat(),
                "_sort_at": (_recovery_sort_epoch(recovery["created_at"])
                             if recovery.get("created_at") else
                             ((t.get("activity") or {}).get("started_at") or 0) / 1000),
            }
        )
    # Queue state is process-local. Persisted checkpoints are synthetic task
    # entries after restart and deliberately have no invented old task_id.
    result.extend(
        {**row, "_sort_at": _recovery_sort_epoch(row.get("created_at"))}
        for row in recoveries
        if (row.get("video_id"), row.get("recovery_id")) not in seen_recoveries
    )
    # Keep the full chronological history across queue and disk records.
    result.sort(key=lambda row: row.get("_sort_at", 0))
    for row in result:
        row.pop("_sort_at", None)
    return result


@router.get("/{pid}/activity")
def project_activity(
    pid: int,
    task_id: str | None = None,
    video_id: int | None = None,
    recovery_id: str | None = None,
):
    """Return live task activity or a verified task sidecar for this project."""
    from pathlib import Path
    import re

    from app.core.activity import (
        read_activity_snapshot, sanitize_activity_snapshot, sanitize_activity_text,
        sanitize_activity_value,
    )
    from app.core.database import session_scope
    from app.core.queue import get_queue
    from app.models.entities import Project, Video

    recovery_pattern = re.compile(r"^[a-f0-9]{24}$")
    if task_id is not None:
        if not task_id.strip() or len(task_id) > 200 or video_id is not None or recovery_id is not None:
            raise HTTPException(422, "请仅提供 task_id，且 task_id 格式有效")
        with session_scope() as db:
            if db.get(Project, pid) is None:
                raise HTTPException(404, "项目不存在")
        task = get_queue().get_status(task_id)
        payload = (task or {}).get("payload") or {}
        if task is None or payload.get("project_id") != pid:
            raise HTTPException(404, "任务不存在")
        activity = task.get("activity")
        response = {
            "task_id": task_id,
            "status": sanitize_activity_text(task.get("status"), field_name="status", limit=64),
            "stage": sanitize_activity_text(task.get("stage"), field_name="stage", limit=128) or None,
            "message": sanitize_activity_text(task.get("message"), field_name="message") or None,
            "error": sanitize_activity_text(task.get("error"), field_name="error") or None,
            "activity": sanitize_activity_snapshot(activity) if isinstance(activity, dict) else None,
            "activity_available": isinstance(activity, dict),
            "activity_source": "live" if isinstance(activity, dict) else "unavailable",
            "activity_persistence_error": task.get("activity_persistence_error"),
        }
        if "model_budget" in task:
            response["model_budget"] = sanitize_activity_value(task["model_budget"], field_name="model_budget")
        return response

    if video_id is None or recovery_id is None or not recovery_pattern.fullmatch(recovery_id):
        raise HTTPException(422, "请提供 video_id 与 24 位 recovery_id")

    with session_scope() as db:
        project = db.get(Project, pid)
        video = db.get(Video, video_id)
        if project is None or video is None or video.project_id != pid:
            raise HTTPException(404, "视频不存在")
        root = Path(project.storage_root)

    work = root / f"video_{video_id}" / "tasks" / recovery_id
    receipt = read_activity_snapshot(
        work, root,
        {"project_id": pid, "video_id": video_id, "recovery_id": recovery_id},
    )
    if receipt is None:
        history_status, history_error, sidecar_missing = _verified_activity_checkpoint_history(
            root, work, pid, video_id, recovery_id,
        )
        if history_status is None:
            history_status, history_error = "unknown", None
        return {
            "task_id": None,
            "status": sanitize_activity_text(history_status, field_name="status", limit=64) or None,
            "stage": None,
            "message": "历史活动信息未记录" if sidecar_missing else "历史活动记录不可用",
            "error": sanitize_activity_text(history_error, field_name="error") or None,
            "activity": None,
            "activity_available": False,
            "activity_source": "unavailable",
            "activity_persistence_error": None,
        }
    state = receipt["state"]
    response = {
        "task_id": receipt["identity"]["task_id"],
        "status": state["status"],
        "stage": state.get("stage"),
        "message": state.get("message"),
        "error": state.get("error"),
        "activity": receipt["activity"],
        "activity_available": True,
        "activity_source": "persisted",
        "activity_persistence_error": state.get("activity_persistence_error"),
    }
    if "model_budget" in state:
        response["model_budget"] = state["model_budget"]
    return response


def _verified_activity_checkpoint_history(
    root, work, project_id: int, video_id: int, recovery_id: str,
) -> tuple[str | None, str | None, bool]:
    """Return only checkpoint-bound status/error when an activity sidecar is absent."""
    sidecar_missing = False
    try:
        from pathlib import Path

        from app.core.activity import _path_has_symlink
        from app.services.recovery import _checkpoint_history

        root_path, work_path = Path(root), Path(work)
        activity_path = work_path / "activity.json"
        if (
            root_path.is_symlink() or not root_path.is_dir()
            or work_path.is_symlink() or not work_path.is_dir()
            or _path_has_symlink(root_path) or _path_has_symlink(work_path)
            or activity_path.exists() or activity_path.is_symlink()
        ):
            return None, None, False
        sidecar_missing = True
        root_real = root_path.resolve(strict=True)
        work_real = work_path.resolve(strict=True)
        work_real.relative_to(root_real)

        checkpoint, status_record, _ = _checkpoint_history(
            work_path, project_id, video_id, recovery_id,
        )
        status = status_record.get("status")
        error = status_record.get("error")
        if status not in {"running", "failed", "cancelled", "done"}:
            return None, None, sidecar_missing
        if error is not None and not isinstance(error, str):
            return None, None, sidecar_missing
        # _checkpoint_history verifies checkpoint and recovery-status receipts,
        # including project/video/recovery identity and their SHA binding.
        if (
            checkpoint.get("project_id") != project_id
            or checkpoint.get("video_id") != video_id
            or checkpoint.get("recovery_id") != recovery_id
        ):
            return None, None, sidecar_missing
        return status, error, sidecar_missing
    except Exception:
        # Historical status is optional. Corrupt or untrusted files must not
        # surface the current Video row's error as if it belonged to this task.
        return None, None, sidecar_missing


def _recovery_sort_epoch(value):
    from datetime import datetime

    if not isinstance(value, str):
        return 0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0


@router.post("/{pid}/resume")
def resume_project_video(pid: int, body: ResumeRequest):
    from app.services.recovery import RecoveryNotFound
    from app.services.processing import submit_recovery

    try:
        task_id = submit_recovery(pid, body.video_id, body.recovery_id)
    except RecoveryNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"task_id": task_id}


@router.post("/{pid}/rebuild")
def rebuild_project_video(pid: int, body: ResumeRequest):
    from app.services.recovery import RecoveryNotFound
    from app.services.processing import submit_recovery

    try:
        task_id = submit_recovery(pid, body.video_id, body.recovery_id, rebuild_completed=True)
    except RecoveryNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"task_id": task_id}


@router.get("/{pid}/tasks/{task_id}/draft")
def task_draft(pid: int, task_id: str):
    from app.core.queue import get_queue
    from app.core.database import session_scope
    from app.models.entities import Project
    from pathlib import Path
    import hashlib, json

    task = get_queue().get_status(task_id)
    payload = (task or {}).get("payload", {})
    if payload.get("project_id") != pid or not payload.get("video_id"):
        raise HTTPException(404, "草稿不存在")
    with session_scope() as db:
        p = db.get(Project, pid)
        if not p:
            raise HTTPException(404, "项目不存在")
        path = (
            Path(p.storage_root)
            / f"video_{payload['video_id']}"
            / "tasks"
            / hashlib.sha256(task_id.encode()).hexdigest()[:24]
            / "draft.json"
        )
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            raise HTTPException(404, "草稿尚未生成")


def existing_document_path(project):
    """Only the selected root's regular HTML file; never follow external symlinks."""
    from pathlib import Path
    from app.services.desktop import enabled

    if not enabled():
        return None
    root = Path(project.storage_root).resolve()
    path = root / "knowledge.html"
    if path.is_file() and not path.is_symlink() and path.resolve().parent == root:
        return path
    return None


@router.get("/{pid}/existing-document")
def existing_document(pid: int):
    from fastapi.responses import Response
    from app.core.database import session_scope
    from app.models.entities import Project

    with session_scope() as db:
        project = db.get(Project, pid)
        path = existing_document_path(project) if project else None
        if path is None:
            raise HTTPException(404, "原有知识文档不存在")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise HTTPException(409, "原有知识文档暂时无法读取") from exc
    return Response(
        content,
        media_type="text/html",
        headers={
            "Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "X-Frame-Options": "SAMEORIGIN",
        },
    )
