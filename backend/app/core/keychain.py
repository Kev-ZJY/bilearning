"""通过随桌面应用编译的 Security.framework helper 访问 Keychain。"""

import json
import os
import subprocess
import threading

_cache_lock = threading.RLock()
_cached_identity: tuple[str, int] | None = None
_cached_value: str | None = None
_failed_identity: tuple[str, int] | None = None
_failed_message: str | None = None
_session_identity: tuple[str, int] | None = None
_session_value: str | None = None


def cached_secret() -> str | None:
    """Return only a value already authorized in this process; never start UI."""
    return cached_secret_info()[0]


def cached_secret_info() -> tuple[str | None, str]:
    """Return the cached value and its non-secret source label atomically."""
    identity = (os.getenv("WORKBENCH_KEYCHAIN_HELPER", ""), os.getpid())
    if not _cache_lock.acquire(blocking=False):
        return None, "unchecked"
    try:
        if identity == _session_identity and _session_value is not None:
            return _session_value, "session"
        if identity == _cached_identity and _cached_value is not None:
            return _cached_value, "saved"
        return None, "unchecked"
    finally:
        _cache_lock.release()


def cached_secret_scope() -> str:
    """Return the source of a value already authorized in this process."""
    return cached_secret_info()[1]


def set_session_secret(value: str) -> None:
    """Authorize a temporary key for this helper/PID without touching Keychain."""
    global _session_identity, _session_value
    if not isinstance(value, str) or not value.strip() or value.startswith("__masked__"):
        raise ValueError("本次运行密钥必须是非空且未掩码的字符串")
    helper = os.environ.get("WORKBENCH_KEYCHAIN_HELPER")
    if not helper:
        raise RuntimeError("钥匙串组件不可用，请重新构建桌面应用")
    with _cache_lock:
        _session_identity = (helper, os.getpid())
        _session_value = value


def secret(action: str, value: str = "", *, retry: bool = False) -> str:
    """Reuse authorized values or read failures; retry explicitly retries a failed read.

    A retry bypasses the failure latch, never an already authorized value. Keep
    only a safe error message in memory, not helper output or exception objects.
    """
    global _cached_identity, _cached_value, _failed_identity, _failed_message
    global _session_identity, _session_value
    helper = os.environ.get("WORKBENCH_KEYCHAIN_HELPER")
    if action not in {"read", "write", "delete"}:
        raise ValueError("未知钥匙串操作")
    with _cache_lock:
        identity = (helper or "", os.getpid())
        if action == "read" and identity == _session_identity:
            return _session_value or ""
        if not helper:
            raise RuntimeError("钥匙串组件不可用，请重新构建桌面应用")
        if action == "read" and identity == _cached_identity:
            return _cached_value or ""
        if action == "read" and identity == _failed_identity and not retry:
            raise RuntimeError(_failed_message) from None
        try:
            try:
                result = subprocess.run(
                    [helper, "--keychain", action],
                    input=json.dumps({"value": value}),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("钥匙串读取超时，已有配置未被删除；请通过测试连接重试") from None
            except (OSError, UnicodeError):
                raise RuntimeError("钥匙串组件调用失败，已有配置未被删除；请通过测试连接重试") from None
            if result.returncode:
                raise RuntimeError("钥匙串访问失败，已有配置未被删除；请通过测试连接重试")
            try:
                payload = json.loads(result.stdout)
            except (TypeError, ValueError):
                raise RuntimeError("钥匙串组件返回格式错误") from None
            if not isinstance(payload, dict) or not isinstance(payload.get("value"), str):
                raise RuntimeError("钥匙串组件返回格式错误")
            returned = payload["value"]
        except RuntimeError as exc:
            if action == "read":
                _failed_identity, _failed_message = identity, str(exc)
            raise
        # Failed operations leave the last authorized value intact. No disk copy.
        _cached_identity = identity
        _cached_value = value if action == "write" else "" if action == "delete" else returned
        _failed_identity, _failed_message = None, None
        if action in {"write", "delete"}:
            _session_identity, _session_value = None, None
        return _cached_value
