"""FastAPI 入口：初始化数据库并装配路由。"""

from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import secrets
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import events, export, messages, projects, settings, tasks, videos
from app.core.config import Settings
from app.core.database import init_db, is_initialized


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 尊重测试或宿主预先初始化的数据库。
    if not is_initialized():
        init_db()
    Path(Settings().storage_root_expanded).mkdir(parents=True, exist_ok=True)
    if os.getenv("WORKBENCH_DESKTOP_TOKEN"):
        from app.core.database import session_scope
        from app.models.entities import Video

        with session_scope() as db:
            for video in db.query(Video).filter(
                Video.status.notin_(["pending", "done", "failed", "cancelled", "skipped"])
            ):
                video.status = "failed"
                video.error = "上次应用退出时任务中断，可重新整理；已有文档已保留"
    yield


def create_app(frontend_dist: str | None = None) -> FastAPI:
    application = FastAPI(title="视频学习工作台", lifespan=lifespan)
    token = os.getenv("WORKBENCH_DESKTOP_TOKEN", "")
    if token:

        @application.middleware("http")
        async def desktop_boundary(request: Request, call_next):
            provided = request.headers.get("x-workbench-native") or request.cookies.get(
                "workbench_session", ""
            )
            origin = request.headers.get("origin")
            expected_origin = str(request.base_url).rstrip("/")
            if not secrets.compare_digest(provided, token) or (
                origin and origin != expected_origin
            ):
                return JSONResponse(
                    {"detail": "桌面会话无效，请重新打开应用"}, status_code=403
                )
            response = await call_next(request)
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers.setdefault("X-Frame-Options", "DENY")
            return response

        from app.api.desktop import router as desktop_router

        application.include_router(desktop_router)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5180", "http://127.0.0.1:5180"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/health")
    def health():
        # 仅检查 Web 服务存活；不声称模型、B站网络或转写已通过检查。
        return {"status": "ok"}

    for route_module in (projects, videos, events, messages, export, tasks, settings):
        application.include_router(route_module.router)
    directory = frontend_dist or os.getenv("FRONTEND_DIST", "")
    if directory:
        # API 优先；静态根目录只包含前端构建产物，不挂载 STORAGE_ROOT。
        application.mount(
            "/", StaticFiles(directory=directory, html=True), name="frontend"
        )
    return application


app = create_app()
