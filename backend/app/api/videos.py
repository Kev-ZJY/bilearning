"""视频接口：向项目添加视频（本地文件 / B站链接）。

B站链接经 parse_bv_input 提取 BV 号后，调用 BilibiliClient.get_view 解析
标题/时长/封面等元信息。这里对「无效输入」做区分处理（见 _classify_input）：
- b23.tv 短链：无法用正则提取 BV，需要用户先在浏览器展开，给"展开"提示；
- 其它垃圾输入（如 "not-a-url"）：是语法错误，给"请输入完整链接"提示。
两者提示必须分开，否则会答非所问。

网络异常（get_view 抛 BilibiliError 或 requests 异常）一律返回 400，
而非 500，避免把内部异常暴露给前端，也保证 B站抽风时接口行为可控。
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.options import PublicProcessingOptions

from app.sources.bilibili_wbi import BilibiliClient, parse_bv_input_ex

router = APIRouter(tags=["videos"])


class VideoAdd(BaseModel):
    url: str
    reuse_existing: bool = False
    page_end: int | None = Field(default=None, ge=1)
    title: str = ""
    duration: float = 0.0
    path: str = ""


@router.post("/api/projects/{pid}/videos")
def add_video(pid: int, body: VideoAdd):
    from app.core.database import session_scope
    from app.models.entities import Project, Video

    with session_scope() as db:
        project = db.get(Project, pid)
        if not project:
            raise HTTPException(404, "项目不存在")

        # 本地文件：直接登记，不联网
        if body.path:
            from app.services.desktop import enabled

            if enabled():
                raise HTTPException(400, "请通过桌面文件选择面板导入媒体")
            from pathlib import Path

            p = Path(body.path)
            if not p.exists():
                raise HTTPException(400, f"文件不存在：{body.path}")
            v = Video(
                project_id=pid,
                source="local",
                title=body.title or p.name,
                duration=body.duration,
                status="pending",
                artifacts_json=body.path,
            )
            db.add(v)
            db.commit()
            return {
                "id": v.id,
                "title": v.title,
                "source": "local",
                "status": v.status,
            }

        # A21：改用 parse_bv_input_ex，除了 BV 号还解析出 ?p=N 选集参数。
        parsed = parse_bv_input_ex(body.url)
        bvid = parsed["bvid"]
        req_page = parsed["page"]  # 用户 ?p= 指定的分P（1-based）；None=未指定
        page_syntax_invalid = parsed["page_invalid"]  # 用户写了 ?p= 但非数字/<=0

        if bvid.startswith("BV"):
            if page_syntax_invalid:
                raise HTTPException(
                    400, "分P 参数无效，未导入任何视频；请使用正整数 p。"
                )
            try:
                meta = BilibiliClient().get_view(bvid)
            except Exception as e:  # noqa: BLE001 网络/接口异常统一收敛为 400
                raise HTTPException(400, f"获取 B站视频信息失败：{e}") from e

            pages = meta.get("pages") or []
            total_pages = len(pages)

            if not pages:
                raise HTTPException(
                    400, "无法获取视频分P 信息，无法确定单集时长；未导入任何视频。"
                )
            selected = req_page or 1
            end = body.page_end if body.page_end is not None else selected
            if body.page_end is not None and req_page is None:
                raise HTTPException(400, "分P 范围缺少起点，请指定 p。")
            if (
                selected > total_pages
                or end > total_pages
                or end < selected
                or end - selected >= 100
            ):
                raise HTTPException(
                    400, f"分P 范围无效（共 {total_pages} P），未导入任何视频。"
                )
            use_default = req_page is None

            # A20 核心修复：duration 必须与 cid 同源，二者都取自同一个分P 条目，
            # 严禁把"顶层 duration（合集总时长）"和"单集 cid"混用。
            # 多P 视频顶层 duration 是全部 P 加起来的时长（如 45P 合集 43.9h），
            # 而顶层 cid 只是 P1 的 cid（仅 27 分钟），分子分母来自不同层级会让
            # ASR 进度分母失真，进度永远到不了 1%。所以一律用选中分P 条目的 duration。
            results = []
            for selected in range(selected, end + 1):
                if use_default:
                    # 默认 P1：按顶层 cid 定位分P（顶层 cid 即 P1 的 cid）。
                    top_cid = str(meta["cid"])
                    page_entry = next(
                        (p for p in pages if str(p.get("cid")) == top_cid), None
                    )
                    if page_entry is None:
                        # 二次防御：找不到匹配 cid（接口结构变了），明确报错，
                        # 绝不退回顶层 duration，否则进度分母失真。
                        raise HTTPException(
                            400,
                            "无法匹配分P 信息（顶层 cid 不在 pages 中），无法确定单集时长。",
                        )
                else:
                    # 用户指定 ?p=N：按分P 序号定位（优先用 page 字段，兜底下标）。
                    # 由于前面已排除越界，pages[selected-1] 不会越界。
                    page_entry = next(
                        (p for p in pages if p.get("page") == selected), None
                    )
                    if page_entry is None:
                        page_entry = pages[selected - 1]

                cid = str(page_entry["cid"])
                duration = page_entry.get("duration")

                # A22：多P 视频标题用分P标题（如「[中文] 02 真正后端工程师的成长路径」），
                # 否则会显示 45P 合集总标题，用户根本不知道导入的是哪一集。
                if total_pages > 1:
                    title = page_entry.get("part") or meta.get("title")
                else:
                    title = meta.get("title")

                v = Video(
                    project_id=pid,
                    bvid=bvid,
                    cid=cid,
                    source="bilibili",
                    title=title,
                    duration=duration,
                    cover=meta.get("pic"),
                    status="pending",
                )
                if body.reuse_existing:
                    existing = (
                        db.query(Video)
                        .filter_by(project_id=pid, bvid=bvid, cid=cid)
                        .first()
                    )
                    if existing is not None:
                        v = existing
                db.add(v)
                db.flush()

                # A22：返回体附带分P 元信息，方便前端/用户确认导入的是哪一集。
                resp = {
                    "id": v.id,
                    "bvid": v.bvid,
                    "cid": v.cid,
                    "title": v.title,
                    "duration": v.duration,
                    "cover": v.cover,
                    "status": v.status,
                    "page": selected,
                    "total_pages": total_pages,
                }
                results.append(resp)
            db.commit()
            return (
                {**results[0], "videos": results}
                if body.page_end is not None
                else results[0]
            )

        if "b23.tv" in bvid:
            raise HTTPException(
                400,
                "检测到 b23.tv 短链，无法自动解析 BV 号。"
                "请在浏览器打开该短链，复制展开后的完整链接（包含 BV 号）后重试。",
            )
        raise HTTPException(
            400,
            "无法识别为 B站视频链接：请输入包含 BV 号的完整链接，"
            "例如 https://www.bilibili.com/video/BVxxxx。",
        )


@router.get("/api/projects/{pid}/videos")
def list_videos(pid: int):
    """列出项目下全部视频（前端 listVideos 对接此端点）。

    返回字段比 GET /api/projects/{pid} 里的 videos 更全（含 bvid / cid / cover）。
    """
    from app.core.database import session_scope
    from app.models.entities import Project, Video

    with session_scope() as db:
        p = db.get(Project, pid)
        if not p:
            raise HTTPException(404, "项目不存在")
        videos = (
            db.query(Video).filter_by(project_id=pid).order_by(Video.id.asc()).all()
        )
        return [
            {
                "id": v.id,
                "bvid": v.bvid,
                "cid": v.cid,
                "title": v.title,
                "duration": v.duration,
                "cover": v.cover,
                "status": v.status,
                "error": v.error,
            }
            for v in videos
        ]


class ProcessRequest(BaseModel):
    video_id: int
    intent: str = "整理视频知识点"
    terms: list[str] = Field(default_factory=list)
    # Direct HTTP requests are current user requests, not a trusted source of
    # conversation history. Keep the private queue snapshot field off this DTO.
    options: PublicProcessingOptions = Field(default_factory=PublicProcessingOptions)


@router.get("/api/projects/{pid}/videos/{vid}/playback")
def video_playback(pid: int, vid: int, remote: bool = False):
    from app.services.playback import resolve_playback

    return resolve_playback(pid, vid, remote=remote)


@router.get("/api/projects/{pid}/videos/{vid}/media")
def video_media(pid: int, vid: int):
    from fastapi.responses import FileResponse
    from app.services.playback import local_media

    path, media_type = local_media(pid, vid)
    return FileResponse(
        path, media_type=media_type, filename=path.name,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/api/projects/{pid}/process")
def process_video(pid: int, body: ProcessRequest):
    from app.services.processing import ProcessingNotFound, submit_processing

    try:
        task_id = submit_processing(
            pid, body.video_id, body.intent, body.terms, body.options.model_dump()
        )
    except ProcessingNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"task_id": task_id}
