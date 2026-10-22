"""Desktop-only APIs. Native grants cannot be issued by a web cookie."""

import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.core.database import session_scope
from app.models.entities import Project, Video
from app.services import project_files
from app.services.desktop import grant, consume

router = APIRouter(prefix="/api/desktop", tags=["desktop"])


class GrantRequest(BaseModel):
    path: str
    kind: str


class Selected(BaseModel):
    grant: str


@router.post("/grant")
def authorize(body: GrantRequest, request: Request):
    if not secrets.compare_digest(
        request.headers.get("x-workbench-native", ""),
        os.environ["WORKBENCH_DESKTOP_TOKEN"],
    ):
        raise HTTPException(403, "必须通过系统选择面板授权")
    try:
        return grant(body.path, body.kind)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, "所选路径不可用") from exc


@router.post("/open")
def reopen(body: Selected):
    from app.api.projects import get_project

    try:
        return get_project(project_files.reopen(consume(body.grant, "directory")))
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(
            409, "无法打开项目：目录未包含有效快照，或仍有任务运行；原文件已保留"
        ) from exc


@router.post("/projects/{pid}/save")
def save(pid: int):
    project_files.persist(pid)
    return project_files.status(pid)


@router.get("/projects/{pid}/directory")
def directory(pid: int, area: str = "project"):
    with session_scope() as db:
        p = db.get(Project, pid)
        if not p:
            raise HTTPException(404, "项目不存在")
        try:
            project_files.check_root(db, p)
            if not Path(p.storage_root).is_dir():
                raise ValueError()
        except (ValueError, OSError):
            raise HTTPException(409, "目录不可用，请重新打开项目目录")
        root = Path(p.storage_root)
        if area == "documents":
            root = root / ".video-workbench"
        elif area == "exports":
            root = root / ".video-workbench" / "exports"
        elif area != "project":
            raise HTTPException(400, "未知目录")
        if not root.is_dir():
            raise HTTPException(404, "尚无对应文件")
        return {"path": str(root)}


@router.post("/projects/{pid}/media")
def import_media(pid: int, body: Selected):
    source = consume(body.grant, "media")
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        if duration <= 0:
            raise ValueError()
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(
            400, "无法读取媒体，请选择 ffmpeg 支持的视频或音频文件"
        ) from exc
    with project_files.file_guard():
        with session_scope() as db:
            p = db.get(Project, pid)
            if not p:
                raise HTTPException(404, "项目不存在")
            try:
                root = project_files.check_root(db, p)
            except (ValueError, OSError):
                raise HTTPException(409, "项目目录不可用")
            if root is None:
                raise HTTPException(409, "请先创建有保存目录的桌面项目")
            v = Video(
                id=project_files.next_video_id(db, root),
                project_id=pid,
                title=source.stem,
                source="local",
                duration=duration,
                status="pending",
            )
            db.add(v)
            db.flush()
            dest = root / f"video_{v.id}"
            dest.mkdir(exist_ok=False)
            media = dest / ("source" + source.suffix.lower())
            try:
                shutil.copyfile(source, media)
                project_files.atomic(
                    dest / "local.json",
                    json.dumps({"video_path": str(media), "audio_path": str(media)}),
                )
            except OSError as exc:
                raise HTTPException(
                    409, "复制失败，请检查目录权限和空间；原媒体未修改"
                ) from exc
            return {
                "id": v.id,
                "title": v.title,
                "source": v.source,
                "status": v.status,
                "duration": duration,
            }


@router.post("/projects/{pid}/export/{fmt}")
def export_local(pid: int, fmt: str):
    import uuid
    from app.api.export import export_html, export_md, export_pdf

    handlers = {"html": export_html, "md": export_md, "pdf": export_pdf}
    if fmt not in handlers:
        raise HTTPException(404, "不支持的导出格式")
    with session_scope() as db:
        p = db.get(Project, pid)
        if not p:
            raise HTTPException(404, "项目不存在")
        try:
            root = project_files.check_root(db, p)
        except (ValueError, OSError):
            raise HTTPException(409, "项目目录不可用")
        if root is None:
            raise HTTPException(409, "请先打开有保存目录的桌面项目")
    response = handlers[fmt](pid)
    exports = root / ".video-workbench" / "exports"
    try:
        exports.mkdir(exist_ok=True)
        dest = exports / f"knowledge-{uuid.uuid4().hex[:12]}.{fmt}"
        with dest.open("xb") as stream:
            stream.write(response.body)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise HTTPException(409, "导出写入失败，请检查磁盘权限和空间") from exc
    return {"path": str(dest)}
