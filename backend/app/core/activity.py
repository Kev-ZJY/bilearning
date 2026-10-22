"""Sanitized activity snapshots and fail-closed sidecar IO for task history."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import re
import secrets
import stat
import time
from datetime import datetime
from typing import Any


ACTIVITY_SCHEMA_VERSION = 1
MAX_ACTIVITY_BYTES = 4 * 1024 * 1024
MAX_ACTIVITY_TEXT = 512
MAX_ACTIVITY_SEGMENT_TEXT = 8192
MAX_ACTIVITY_DEPTH = 5

_SENSITIVE_NAME = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|key|token|authorization|cookie|password|secret|headers?|"
    r"prompt|context|messages|content|image|base64|credential|auth)(?:$|[_-])",
    re.IGNORECASE,
)
_API_KEY = re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}\b")
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_INLINE_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization|cookie|prompt|context|messages?|"
    r"content|image|base64|headers?)\s*[:=]\s*['\"]?[^\s&;,]+"
)
_LONG_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{128,}(?![A-Za-z0-9+/=_-])")
_URL_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[_-]?key|key|token|secret|auth|authorization|password|signature|sig|credential)=)[^&#\s]*"
)
_URL_USERINFO = re.compile(r"(https?://)[^/@\s:]+:[^/@\s]+@", re.IGNORECASE)
_RECOVERY_ID = re.compile(r"^[a-f0-9]{24}$")

_BASE_STEP_FIELDS = (
    "seq", "step", "label", "status", "detail", "elapsed_ms", "order",
)
_OPTIONAL_STEP_FIELDS = (
    "actor", "agent_id", "parent_task_id", "child_task_id", "stage", "round",
    "tool_name", "arguments", "result_summary", "error", "model",
    "started_at", "finished_at", "execution", "cycle", "attempts", "retry_count",
    "finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "total_tokens", "max_tokens", "http_status",
)
_ACTORS = {"orchestrator", "author", "reviewer"}
_EXECUTIONS = {"isolated_context", "child_task", "workflow"}
_FINISH_REASONS = {"stop", "length", "tool_calls", "function_call", "content_filter", "cancelled", "error", "unknown"}
_TOKEN_COUNT_FIELDS = {
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
}
_COUNT_FIELDS = {"cycle", "attempts", "retry_count"}
_OUTPUT_LIMIT_FIELDS = {"max_tokens"}


def _finite_number(value: Any, default: int | float = 0) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if isinstance(value, int):
        return value
    if not math.isfinite(value):
        return default
    return value


def _safe_text(value: Any, *, limit: int = MAX_ACTIVITY_TEXT) -> str:
    if not isinstance(value, str):
        return ""
    text = _BEARER.sub("Bearer [REDACTED]", value)
    text = _API_KEY.sub("[REDACTED]", text)
    text = _INLINE_SECRET.sub(r"\1=[REDACTED]", text)
    text = _URL_QUERY_SECRET.sub(r"\1[REDACTED]", text)
    text = _URL_USERINFO.sub(r"\1[REDACTED]@", text)
    text = _LONG_BASE64.sub("[REDACTED]", text)
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def _is_iso8601_with_timezone(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 64:
        return False
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except (OverflowError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def sanitize_activity_value(value: Any, *, field_name: str = "", depth: int = 0) -> Any:
    """Return bounded JSON-safe metadata while removing likely private payloads."""
    if field_name and _SENSITIVE_NAME.search(field_name):
        return "[REDACTED]"
    if depth > MAX_ACTIVITY_DEPTH:
        return "[TRUNCATED]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, child in list(value.items())[:64]:
            key = str(raw_key)[:80]
            output[key] = sanitize_activity_value(child, field_name=key, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for child in list(value)[:64]:
            # Bare strings in arrays lack a field name and commonly contain prompts,
            # URLs, or image payload fragments. Keep their positions without content.
            if isinstance(child, str):
                output.append("[REDACTED]")
            else:
                output.append(sanitize_activity_value(child, depth=depth + 1))
        return output
    return "[REDACTED]"


def sanitize_activity_text(value: Any, *, field_name: str = "", limit: int = MAX_ACTIVITY_TEXT) -> str:
    """Sanitize a free-text DTO field, including JSON-encoded metadata strings."""
    if value is None:
        return ""
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            cleaned_json = sanitize_activity_value(parsed, field_name=field_name, depth=1)
            rendered = json.dumps(cleaned_json, ensure_ascii=False, separators=(",", ":"))
            return rendered if len(rendered) <= limit else rendered[:limit] + "…"
    cleaned = sanitize_activity_value(value, field_name=field_name)
    if isinstance(cleaned, str):
        return _safe_text(cleaned, limit=limit)
    try:
        rendered = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = "[REDACTED]"
    return _safe_text(rendered, limit=limit)


def sanitize_step(fields: dict[str, Any], *, seq: int, order: int | None = None) -> dict[str, Any]:
    """Normalize a legacy or new step to the shared AgentStep snapshot shape."""
    safe = {
        "seq": int(seq),
        "step": sanitize_activity_text(fields.get("step") or fields.get("label") or "", field_name="step", limit=160),
        "label": sanitize_activity_text(fields.get("label") or fields.get("step") or "", field_name="label", limit=160),
        "status": sanitize_activity_text(fields.get("status") or "running", field_name="status", limit=64),
        "detail": sanitize_activity_text(fields.get("detail") or "", field_name="detail"),
        "elapsed_ms": max(0, int(_finite_number(fields.get("elapsed_ms"), 0))),
        "order": int(order if order is not None else _finite_number(fields.get("order"), time.time_ns())),
    }
    for key in _OPTIONAL_STEP_FIELDS:
        if key not in fields or fields[key] is None:
            continue
        raw = fields[key]
        if key == "actor":
            if isinstance(raw, str) and raw in _ACTORS:
                safe[key] = raw
        elif key == "execution":
            if isinstance(raw, str) and raw in _EXECUTIONS:
                safe[key] = raw
        elif key == "round":
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                safe[key] = raw
        elif key in {"started_at", "finished_at"}:
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
                safe[key] = raw
            elif _is_iso8601_with_timezone(raw):
                safe[key] = raw
        elif key in _COUNT_FIELDS | _TOKEN_COUNT_FIELDS | _OUTPUT_LIMIT_FIELDS:
            if isinstance(raw, int) and not isinstance(raw, bool) and 0 <= raw <= 1_000_000_000:
                if key != "cycle" or raw >= 1:
                    safe[key] = raw
        elif key == "http_status":
            if isinstance(raw, int) and not isinstance(raw, bool) and 100 <= raw <= 599:
                safe[key] = raw
        elif key == "finish_reason":
            if isinstance(raw, str) and raw in _FINISH_REASONS:
                safe[key] = raw
        elif key == "arguments":
            safe[key] = sanitize_activity_value(raw, field_name=key)
        else:
            safe[key] = sanitize_activity_text(raw, field_name=key)
    return safe


def sanitize_activity_snapshot(activity: Any) -> dict[str, Any]:
    """Keep the ActivitySnapshot envelope while excluding final answer segments."""
    if not isinstance(activity, dict):
        activity = {}
    steps = []
    for raw in activity.get("steps", [])[:2000] if isinstance(activity.get("steps"), list) else []:
        if not isinstance(raw, dict):
            continue
        seq = raw.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
            continue
        steps.append(sanitize_step(raw, seq=seq))
    segments = []
    for raw in activity.get("segments", [])[:200] if isinstance(activity.get("segments"), list) else []:
        if not isinstance(raw, dict):
            continue
        segment_id = _safe_text(raw.get("id") or "", limit=120)
        if not segment_id:
            continue
        phase = _safe_text(raw.get("phase") or "", limit=64)
        text = "" if phase.lower() == "final" else _safe_text(
            raw.get("text") or "", limit=MAX_ACTIVITY_SEGMENT_TEXT,
        )
        segments.append({
            "id": segment_id,
            "text": text,
            "phase": phase,
            "order": int(_finite_number(raw.get("order"), 0)),
        })
    return {
        "steps": steps,
        "segments": segments,
        "started_at": _finite_number(activity.get("started_at"), 0),
        "elapsed_ms": max(0, int(_finite_number(activity.get("elapsed_ms"), 0))),
    }


def _validate_identity(identity: Any) -> bool:
    if not isinstance(identity, dict) or set(identity) != {
        "task_id", "project_id", "video_id", "recovery_id",
    }:
        return False
    valid = (
        isinstance(identity.get("task_id"), str) and bool(identity["task_id"])
        and type(identity.get("project_id")) is int and identity["project_id"] > 0
        and type(identity.get("video_id")) is int and identity["video_id"] > 0
        and isinstance(identity.get("recovery_id"), str)
        and _RECOVERY_ID.fullmatch(identity["recovery_id"]) is not None
    )
    if not valid:
        return False
    try:
        task_bytes = identity["task_id"].encode("utf-8")
    except UnicodeEncodeError:
        return False
    return hashlib.sha256(task_bytes).hexdigest()[:24] == identity["recovery_id"]


def _validate_snapshot(data: Any, expected_identity: dict[str, Any]) -> bool:
    if not isinstance(data, dict) or data.get("schema_version") != ACTIVITY_SCHEMA_VERSION:
        return False
    identity = data.get("identity")
    if not _validate_identity(identity) or any(identity.get(key) != value for key, value in expected_identity.items()):
        return False
    state = data.get("state")
    if not isinstance(state, dict) or not isinstance(state.get("status"), str):
        return False
    if any(key not in {
        "status", "stage", "message", "error", "activity_persistence_error", "model_budget",
    } for key in state):
        return False
    if any(key in state and state[key] is not None and not isinstance(state[key], str)
           for key in ("stage", "message", "error", "activity_persistence_error")):
        return False
    activity = data.get("activity")
    if not isinstance(activity, dict) or set(activity) != {"steps", "segments", "started_at", "elapsed_ms"}:
        return False
    if not isinstance(activity["steps"], list) or not isinstance(activity["segments"], list):
        return False
    if _finite_number(activity["started_at"], None) is None or _finite_number(activity["elapsed_ms"], None) is None:
        return False
    seen: set[int] = set()
    for step in activity["steps"]:
        if not isinstance(step, dict) or not set(_BASE_STEP_FIELDS).issubset(step):
            return False
        if type(step["seq"]) is not int or step["seq"] <= 0 or step["seq"] in seen:
            return False
        seen.add(step["seq"])
        if any(not isinstance(step[key], str) for key in ("step", "label", "status", "detail")):
            return False
        if type(step["elapsed_ms"]) is not int or step["elapsed_ms"] < 0:
            return False
        if isinstance(step["order"], bool) or not isinstance(step["order"], (int, float)):
            return False
        if any(key not in set(_BASE_STEP_FIELDS + _OPTIONAL_STEP_FIELDS) for key in step):
            return False
        if "actor" in step and (not isinstance(step["actor"], str) or step["actor"] not in _ACTORS):
            return False
        if "execution" in step and (not isinstance(step["execution"], str) or step["execution"] not in _EXECUTIONS):
            return False
        if "round" in step and (type(step["round"]) is not int or step["round"] < 0):
            return False
        if "cycle" in step and (type(step["cycle"]) is not int or step["cycle"] < 1):
            return False
        if any(key in step and (type(step[key]) is not int or not 0 <= step[key] <= 1_000_000_000)
               for key in _COUNT_FIELDS | _TOKEN_COUNT_FIELDS | _OUTPUT_LIMIT_FIELDS):
            return False
        if "http_status" in step and (type(step["http_status"]) is not int or not 100 <= step["http_status"] <= 599):
            return False
        if "finish_reason" in step and (
            not isinstance(step["finish_reason"], str) or step["finish_reason"] not in _FINISH_REASONS
        ):
            return False
        if any(key in step and not isinstance(step[key], str)
               for key in ("agent_id", "parent_task_id", "child_task_id", "stage", "tool_name",
                           "result_summary", "error", "model")):
            return False
        if "arguments" in step and not isinstance(step["arguments"], (dict, list, str, int, float, bool, type(None))):
            return False
        for key in ("started_at", "finished_at"):
            if key not in step:
                continue
            value = step[key]
            if isinstance(value, bool):
                return False
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    return False
            elif not _is_iso8601_with_timezone(value):
                return False
    for segment in activity["segments"]:
        if not isinstance(segment, dict) or set(segment) != {"id", "text", "phase", "order"}:
            return False
        if any(not isinstance(segment[key], str) for key in ("id", "text", "phase")):
            return False
        if len(segment["text"]) > MAX_ACTIVITY_SEGMENT_TEXT:
            return False
        if segment["phase"].lower() == "final" and segment["text"]:
            return False
        if isinstance(segment["order"], bool) or not isinstance(segment["order"], (int, float)):
            return False
    return True


def _sanitize_receipt(data: dict[str, Any]) -> dict[str, Any]:
    state = data["state"]
    safe_state = {
        "status": sanitize_activity_text(state.get("status") or "pending", field_name="status", limit=64),
        "stage": sanitize_activity_text(state.get("stage"), field_name="stage", limit=128) or None,
        "message": sanitize_activity_text(state.get("message"), field_name="message") or None,
        "error": sanitize_activity_text(state.get("error"), field_name="error") or None,
        "activity_persistence_error": (
            "活动记录暂时无法保存" if state.get("activity_persistence_error") else None
        ),
    }
    if "model_budget" in state:
        safe_state["model_budget"] = sanitize_activity_value(state["model_budget"], field_name="model_budget")
    return {
        "schema_version": ACTIVITY_SCHEMA_VERSION,
        "identity": dict(data["identity"]),
        "state": safe_state,
        "activity": sanitize_activity_snapshot(data["activity"]),
    }


def _path_has_symlink(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        try:
            if cursor.is_symlink():
                return True
        except OSError:
            return True
    return False


def _bounded_read(path: Path, root: Path) -> bytes | None:
    try:
        root_path = Path(root)
        target = Path(path)
        if root_path.is_symlink() or not root_path.is_dir() or _path_has_symlink(target):
            return None
        root_real = root_path.resolve(strict=True)
        target_real = target.resolve(strict=True)
        target_real.relative_to(root_real)
        if target.is_symlink() or not target.is_file():
            return None
        fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ACTIVITY_BYTES:
                return None
            chunks = []
            remaining = MAX_ACTIVITY_BYTES + 1
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            return raw if len(raw) <= MAX_ACTIVITY_BYTES else None
        finally:
            os.close(fd)
    except (OSError, ValueError):
        return None


def read_activity_snapshot(work: Path, root: Path, expected_identity: dict[str, Any]) -> dict[str, Any] | None:
    """Read a sidecar only inside its project root and only for its recorded identity."""
    work = Path(work)
    activity_path = work / "activity.json"
    raw = _bounded_read(activity_path, root)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not _validate_snapshot(data, expected_identity):
        return None
    return _sanitize_receipt(data)


def write_activity_snapshot(
    work: Path,
    identity: dict[str, Any],
    state: dict[str, Any],
    activity: dict[str, Any],
) -> None:
    """Atomically replace this task's sanitized sidecar; never overwrite another identity."""
    work = Path(work)
    if not _validate_identity(identity) or identity["recovery_id"] != work.name:
        raise ValueError("活动记录身份无效")
    if _path_has_symlink(work) or work.is_symlink() or not work.is_dir():
        raise OSError("活动记录目录不可用")
    destination = work / "activity.json"
    if destination.is_symlink():
        raise OSError("活动记录文件不可用")
    if destination.exists():
        existing = _bounded_read(destination, work)
        if existing is None:
            raise OSError("现有活动记录不可验证")
        try:
            existing_data = json.loads(existing)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OSError("现有活动记录不可验证") from exc
        if not _validate_snapshot(existing_data, identity):
            raise OSError("活动记录身份冲突")

    safe_state = {
        "status": sanitize_activity_text(state.get("status") or "pending", field_name="status", limit=64),
        "stage": sanitize_activity_text(state.get("stage"), field_name="stage", limit=128) or None,
        "message": sanitize_activity_text(state.get("message"), field_name="message") or None,
        "error": sanitize_activity_text(state.get("error"), field_name="error") or None,
        "activity_persistence_error": (
            "活动记录暂时无法保存" if state.get("activity_persistence_error") else None
        ),
    }
    if "model_budget" in state and state["model_budget"] is not None:
        safe_state["model_budget"] = sanitize_activity_value(state["model_budget"], field_name="model_budget")
    payload = {
        "schema_version": ACTIVITY_SCHEMA_VERSION,
        "identity": identity,
        "state": safe_state,
        "activity": sanitize_activity_snapshot(activity),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_ACTIVITY_BYTES:
        raise ValueError("活动记录超过大小限制")

    temporary = work / f".activity-{os.getpid()}-{secrets.token_hex(8)}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(work, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
