"""原生宿主授权。网页只接触短期 grant，不获得任意路径访问入口。"""

import os
from pathlib import Path
import secrets
import threading
import time
from fastapi import HTTPException

_grants = {}
_lock = threading.Lock()


def enabled():
    return bool(os.getenv("WORKBENCH_DESKTOP_TOKEN"))


def grant(path: str, kind: str):
    p = Path(path).expanduser().resolve(strict=True)
    if (
        kind not in ("directory", "media")
        or (kind == "directory" and not p.is_dir())
        or (kind == "media" and not p.is_file())
    ):
        raise ValueError("所选路径类型不匹配")
    token = secrets.token_urlsafe(32)
    with _lock:
        now = time.monotonic()
        for key in list(_grants):
            if _grants[key][2] < now:
                _grants.pop(key)
        _grants[token] = (p, kind, now + 600)
    return {
        "grant": token,
        "path": str(p),
        "name": p.name,
        "existing_project": kind == "directory"
        and (p / ".video-workbench/project.json").is_file(),
        "existing_document": kind == "directory" and (p / "knowledge.html").is_file(),
    }


def consume(token: str, kind: str):
    with _lock:
        item = _grants.pop(token, None)
    if not item or item[1] != kind or item[2] < time.monotonic():
        raise HTTPException(400, "目录或文件授权已失效，请重新选择")
    if not item[0].exists():
        raise HTTPException(409, "所选路径已不可用，请重新选择")
    return item[0]
