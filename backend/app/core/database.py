"""SQLAlchemy 2.x 数据库基础设施。

提供 DeclarativeBase 基类、引擎/会话工厂、以及 init_db / session_scope 两个入口。
默认使用 SQLite 文件库，并关闭 check_same_thread 以支持 FastAPI 多线程访问。
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def _default_db_url() -> str:
    """数据库 URL。

    早前这里是写死的模块常量 `sqlite:///./data/workbench.db`，带来两个问题：
    改 STORAGE_ROOT 换不到库（验证时无法隔离，会污染真实数据），
    以及 .env 里配什么都不生效。
    现在 DB_URL 可显式指定；未指定时**跟随 STORAGE_ROOT**，
    保证「换存储根目录 = 换一套数据」这个直觉成立。
    """
    explicit = os.getenv("DB_URL", "").strip()
    if explicit:
        return explicit
    root = os.getenv("STORAGE_ROOT", "./data").strip() or "./data"
    return "sqlite:///" + str(Path(root).expanduser() / "workbench.db")


_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker[Session]] = None


class Base(DeclarativeBase):
    """所有 ORM 实体的声明基类。"""


def _ensure_parent_dir(url: str) -> None:
    """为 sqlite 文件路径创建父目录，避免首次建库因目录不存在而报错。"""
    if url.startswith("sqlite") and "///" in url:
        path_part = url.split("///", 1)[1]
        # 排除内存库 sqlite:///:memory:
        if path_part and not path_part.startswith(":memory:"):
            Path(path_part).parent.mkdir(parents=True, exist_ok=True)


def init_db(url: str | None = None) -> Engine:
    """初始化数据库：建父目录、建引擎、注册实体、建表。

    重复调用会切换到新 url（每个测试/进程用各自的库）。
    """
    global _engine, _SessionLocal

    url = url or _default_db_url()
    _ensure_parent_dir(url)

    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(url, connect_args=connect_args)

    # 延迟导入实体，确保 Base.metadata 注册了全部表后再 create_all
    from app.models import entities  # noqa: F401

    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

    # 换库了，配置缓存必须失效。
    #
    # 为什么要在这里（而不是让每个测试 fixture 自己记得调）：
    # `app.core.config` 把读库结果缓存在进程内，而每个 pytest 用例都调
    # init_db() 指向各自的临时库。缓存不失效的话，A 用例写进库的设置会
    # 被 B 用例读到 —— 症状是「单跑绿、整跑红」，而且红的那个用例看起来
    # 与设置毫无关系。**换引擎是缓存唯一必须失效的时机，把失效挂在
    # 换引擎这个动作上，就不可能有调用方忘记。**
    try:
        from app.core.config import invalidate_settings_cache

        invalidate_settings_cache()
    except Exception:
        # 配置模块还没准备好也不能阻断建库
        pass
    return _engine


def is_initialized() -> bool:
    """数据库是否已初始化。

    供应用启动逻辑判断「是否还需要建库」。公开这个接口，
    好过让调用方直接读模块私有 _engine。
    """
    return _SessionLocal is not None


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务型会话上下文：正常 commit，异常 rollback，最后 close。"""
    if _SessionLocal is None:
        raise RuntimeError("数据库尚未初始化，请先调用 init_db()")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
        pending = session.info.pop("project_exports", set())
        if pending:
            from app.services.project_files import persist
            for pid in pending: persist(pid)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@event.listens_for(Session, "before_flush")
def _track_project_files(session, flush_context, instances):
    from app.services.desktop import enabled
    if not enabled(): return
    from app.models.entities import Project
    touched=session.info.setdefault("project_exports",set())
    for obj in list(session.new)+list(session.dirty)+list(session.deleted):
        pid=obj.id if isinstance(obj,Project) else getattr(obj,"project_id",None)
        if pid: touched.add(pid)
