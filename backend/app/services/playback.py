"""Resolve registered sources for playback without exposing filesystem paths."""

import json
import mimetypes
from pathlib import Path
import re
import uuid
from urllib.parse import urlencode

from fastapi import HTTPException

from app.core.database import session_scope
from app.models.entities import Project, Video
from app.services.desktop import enabled
from app.sources.bilibili_wbi import BilibiliClient


def _registered(pid: int, vid: int):
    with session_scope() as db:
        project = db.get(Project, pid)
        if project is None:
            raise HTTPException(404, "项目不存在")
        video = db.get(Video, vid)
        if video is None or video.project_id != pid:
            raise HTTPException(404, "视频不存在或不属于当前项目")
        if video.source in ("local", "bilibili") and enabled():
            from app.services.project_files import check_root

            try:
                if check_root(db, project) is None:
                    raise ValueError("项目没有目录身份")
            except (OSError, ValueError):
                raise HTTPException(409, "项目媒体目录不可用或身份不匹配，请重新打开项目") from None
        return project, video


def _bili_ids(video):
    bvid, cid = video.bvid or "", video.cid or ""
    if not re.fullmatch(r"BV[0-9A-Za-z]+", bvid) or not re.fullmatch(r"[1-9][0-9]*", cid):
        raise HTTPException(400, "缺少有效的视频或分P 标识，无法确定播放内容")
    return bvid, cid


def _bilibili(video):
    bvid, cid = _bili_ids(video)
    try:
        meta = BilibiliClient().get_view(bvid)
    except Exception:
        raise HTTPException(409, "无法确认视频分P 信息，请稍后重试") from None
    if not isinstance(meta, dict) or (meta.get("bvid") and meta["bvid"] != bvid):
        raise HTTPException(409, "视频分P 信息不匹配，无法确定播放内容")
    pages = meta.get("pages")
    if not isinstance(pages, list):
        raise HTTPException(409, "无法获取视频分P 信息，无法确定播放内容")
    matches = [page for page in pages if isinstance(page, dict) and str(page.get("cid")) == cid]
    if len(matches) != 1 or type(matches[0].get("page")) is not int or matches[0]["page"] < 1:
        raise HTTPException(409, "无法匹配已登记的分P，未切换到其他视频")
    page = matches[0]["page"]
    query = urlencode({"bvid": bvid, "cid": cid, "page": page, "autoplay": 0})
    return {
        "kind": "embed", "url": "https://player.bilibili.com/player.html?" + query,
        "source_url": f"https://www.bilibili.com/video/{bvid}/?p={page}",
        "title": video.title or matches[0].get("part") or meta.get("title") or bvid,
        "page": page,
    }


def _symlink_in_path(path: Path) -> bool:
    return any(part.is_symlink() for part in (path, *path.parents))


def _scoped_path(raw: str, root: Path, *, legacy_original=False) -> Path:
    if (not isinstance(raw, str) or not raw.strip() or "\x00" in raw
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw.lstrip())):
        raise HTTPException(403, "媒体路径无效，无法播放")
    path = Path(raw).expanduser()
    if ".." in path.parts:
        raise HTTPException(403, "媒体路径超出项目目录，无法播放")
    # The old non-desktop import API records a user-selected original path.
    # Only that exact registration may be external; generated artifacts and
    # desktop imports must stay under the owning project directory.
    external_original = legacy_original and not enabled()
    if not path.is_absolute():
        path = path.absolute() if external_original else root / path
    if _symlink_in_path(path):
        raise HTTPException(403, "媒体路径包含符号链接，无法播放")
    if not external_original and not path.is_relative_to(root):
        raise HTTPException(403, "媒体路径超出项目目录，无法播放")
    return path


def _media_root(project):
    root = Path(project.storage_root).expanduser().absolute()
    if _symlink_in_path(root):
        raise HTTPException(403, "项目媒体目录包含符号链接，无法播放")
    if not root.is_dir():
        raise HTTPException(409, "项目媒体目录不可用，请重新打开项目")
    return root


def _local_media(project, video):
    root = _media_root(project)
    manifest = root / f"video_{video.id}" / "local.json"
    legacy_original = False
    if manifest.exists() or manifest.is_symlink():
        manifest = _scoped_path(str(manifest), root)
        try:
            registry = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(409, "媒体登记信息不可用，请重新导入媒体") from None
    else:
        raw = video.artifacts_json or ""
        try:
            registry = json.loads(raw)
        except ValueError:
            registry = raw
        legacy_original = isinstance(registry, str)
    if isinstance(registry, dict):
        source = registry.get("video_path") or registry.get("audio_path")
    elif isinstance(registry, str) and legacy_original:
        source = registry
    else:
        source = None
    if not source:
        raise HTTPException(404, "没有已登记的本地媒体，请重新导入")
    path = _scoped_path(source, root, legacy_original=legacy_original)
    if not path.is_file():
        raise HTTPException(404, "已登记的本地媒体不存在，请重新导入")
    media_type = mimetypes.guess_type(path.name)[0] or ""
    if not media_type.startswith(("video/", "audio/")):
        raise HTTPException(415, "当前本地文件格式不支持播放器，请导入视频或音频")
    return path, media_type


def _unavailable(video, status, message):
    return {"kind": "unavailable", "url": None, "title": video.title or "视频",
            "cache_status": status, "cache_message": message}


def _ready_reply(video):
    return {"kind": "video", "url": f"/api/projects/{video.project_id}/videos/{video.id}/media",
            "title": video.title or "视频", "storage": "cache", "cache_status": "ready",
            "cache_message": "已下载音画并合并为本地播放缓存"}


def _cache_folder(project, video):
    root = _media_root(project)
    folder = _scoped_path(str(root / f"video_{video.id}" / "playback"), root)
    return root, folder


def _owner_uuid(pid):
    from app.services.project_files import identity

    with session_scope() as db:
        return identity(db, pid)


def _read_ready(project, video):
    from app.worker.media import inspect_media, verify_playable

    bvid, cid = _bili_ids(video)
    root, folder = _cache_folder(project, video)
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None
    manifest_path = _scoped_path(str(manifest_path), root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("version") != 1
                or manifest.get("source") != {"video_id": video.id, "bvid": bvid, "cid": cid}):
            raise ValueError("缓存来源标识不匹配")
        if manifest.get("project_uuid") != _owner_uuid(video.project_id):
            raise ValueError("缓存项目身份不匹配")
        source_paths = []
        for kind, name in (("audio", f"{bvid}_{cid}.m4s"), ("video", f"{bvid}_{cid}.video.m4s")):
            recorded = manifest["inputs"][kind]
            source = _scoped_path(recorded["path"], root)
            if (not source.is_relative_to(root / f"video_{video.id}") or source.name != name
                    or not re.fullmatch(r"[0-9a-f]{64}", recorded["sha256"])
                    or type(recorded["size"]) is not int or recorded["size"] <= 0
                    or type(recorded["mtime_ns"]) is not int):
                raise ValueError("缓存缺少有效的下载来源记录")
            source_paths.append(source)
        if source_paths[0].parent != source_paths[1].parent:
            raise ValueError("缓存音画来源不属于同一任务")
        path = _scoped_path(manifest.get("path"), root)
        if path.parent != folder or not re.fullmatch(re.escape(f"{bvid}_{cid}-") + r"[0-9a-f]{16}\.mp4", path.name):
            raise ValueError("缓存路径不属于本视频")
        identity = manifest["file_identity"]
        stat = path.stat()
        if stat.st_size != identity["size"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError("缓存文件已变化")
        verify_playable(inspect_media(path), video.duration or 0)
        return path, manifest
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        raise HTTPException(409, "本地播放缓存无效，需重新准备") from None


def _download_state(project, video):
    from app.worker.media import inspect_media, verify_pair

    bvid, cid = _bili_ids(video)
    root = _media_root(project)
    video_root = _scoped_path(str(root / f"video_{video.id}"), root)
    folders = [video_root]
    tasks = _scoped_path(str(video_root / "tasks"), root)
    if tasks.is_dir():
        folders += sorted(tasks.iterdir())
    found_audio, found_video = False, False
    for folder in folders:
        folder = _scoped_path(str(folder), root)
        if not folder.is_dir():
            continue
        audio = _scoped_path(str(folder / f"{bvid}_{cid}.m4s"), root)
        image = _scoped_path(str(folder / f"{bvid}_{cid}.video.m4s"), root)
        found_audio |= audio.is_file()
        found_video |= image.is_file()
        if audio.is_file() and image.is_file():
            verify_pair(inspect_media(image), inspect_media(audio), video.duration or 0)
            return _unavailable(video, "unmerged", "同一视频的音频和画面已下载，尚未合并为播放缓存")
    if found_audio and found_video:
        return _unavailable(video, "invalid", "音画文件位于不同任务，不能认定为同一组完整下载")
    if found_audio:
        return _unavailable(video, "audio_only", "目前仅发现音频下载，完整视频播放还需要画面文件")
    if found_video:
        return _unavailable(video, "video_only", "目前仅发现画面下载，完整视频播放还需要音频文件")
    return _unavailable(video, "missing", "尚无本地音画缓存，请准备视频播放缓存")


def prepare_bilibili_cache(pid: int, vid: int, audio_path, video_path, checkpoint=lambda: None):
    """Background-worker entry point; no network, knowledge generation or DB writes."""
    from app.core.cancel import TaskCancelled
    from app.services.documents import project_guard
    from app.services.project_files import atomic
    from app.worker.media import mux_streams

    with project_guard(("playback", pid, vid)):
        checkpoint()
        project, video = _registered(pid, vid)
        if video.source != "bilibili":
            raise HTTPException(415, "当前来源不需要 B站音画合并")
        bvid, cid = _bili_ids(video)
        root, folder = _cache_folder(project, video)
        video_root = root / f"video_{vid}"
        audio, image = _scoped_path(audio_path, root), _scoped_path(video_path, root)
        if audio.parent != image.parent:
            raise ValueError("音画必须来自同一任务目录")
        if (not audio.is_relative_to(video_root) or not image.is_relative_to(video_root)
                or audio.name != f"{bvid}_{cid}.m4s" or image.name != f"{bvid}_{cid}.video.m4s"):
            raise ValueError("下载文件与当前视频标识不匹配")
        try:
            ready = _read_ready(project, video)
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            ready = None
        if ready:
            inputs = ready[1].get("inputs") or {}
            unchanged = True
            for kind, path in (("audio", audio), ("video", image)):
                stat = path.stat()
                saved = inputs.get(kind) or {}
                unchanged &= saved.get("path") == str(path.relative_to(root)) and saved.get("size") == stat.st_size and saved.get("mtime_ns") == stat.st_mtime_ns
            if unchanged:
                return _ready_reply(video)
        folder.mkdir(parents=True, exist_ok=True)
        # Immutable generation + atomic manifest keeps the previous ready cache
        # intact if probing, cancellation, encoding or manifest publication fails.
        output = folder / f"{bvid}_{cid}-{uuid.uuid4().hex[:16]}.mp4"
        committed = False
        try:
            receipt = mux_streams(audio, image, output, expected_duration=video.duration or 0, checkpoint=checkpoint)
            for kind, path in (("audio", audio), ("video", image)):
                receipt["inputs"][kind]["path"] = str(path.relative_to(root))
            manifest = {"version": 1, "project_uuid": _owner_uuid(pid),
                        "source": {"video_id": vid, "bvid": bvid, "cid": cid},
                        "path": str(output.relative_to(root)), **receipt}
            checkpoint()
            atomic(folder / "manifest.json", json.dumps(manifest, ensure_ascii=False))
            committed = True
        except TaskCancelled:
            raise
        except (OSError, ValueError, RuntimeError):
            raise HTTPException(409, "音画合并或播放缓存校验失败；原有缓存已保留") from None
        finally:
            if not committed:
                output.unlink(missing_ok=True)
        return _ready_reply(video)


def resolve_playback(pid: int, vid: int, *, remote=False) -> dict:
    project, video = _registered(pid, vid)
    if video.source == "bilibili":
        try:
            if _read_ready(project, video):
                return _ready_reply(video)
            state = _download_state(project, video)
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            state = _unavailable(video, "invalid", str(exc.detail))
        except (OSError, ValueError, RuntimeError):
            state = _unavailable(video, "invalid", "下载音画文件校验失败，请重新准备缓存")
        if remote:
            return {**_bilibili(video), "cache_status": state["cache_status"], "cache_message": state["cache_message"]}
        return state
    if video.source != "local":
        raise HTTPException(415, "当前媒体来源暂不支持播放")
    _, media_type = _local_media(project, video)
    return {"kind": "video" if media_type.startswith("video/") else "audio",
            "url": f"/api/projects/{pid}/videos/{vid}/media", "title": video.title or "本地媒体"}


def local_media(pid: int, vid: int):
    project, video = _registered(pid, vid)
    if video.source == "bilibili":
        from app.services.documents import project_guard

        with project_guard(("playback", pid, vid)):
            ready = _read_ready(project, video)
            if ready:
                return ready[0], "video/mp4"
            raise HTTPException(409, "本地播放缓存尚未就绪，请准备视频播放缓存")
    if video.source != "local":
        raise HTTPException(415, "当前来源没有可直接播放的本地媒体")
    return _local_media(project, video)
