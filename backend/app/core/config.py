"""配置优先级：数据库中的 LLM 覆盖值 > 环境变量 > 代码默认值。

数据库未初始化时使用环境值；设置保存或切换数据库时使缓存失效。
.env 由 backend/run.sh 加载，直接导入本模块不会读取文件。
"""
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    # 密钥可保存在本地数据库；API 仅返回掩码。
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    # 保留现有默认预算；须按模型支持的输出上限在设置页调整。
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "200000")))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2")))
    llm_timeout: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "900")))
    asr_device: str = field(default_factory=lambda: os.getenv("ASR_DEVICE", "auto"))
    asr_compute_type: str = field(default_factory=lambda: os.getenv("ASR_COMPUTE_TYPE", "auto"))
    # 预置目录不存在时按模型名使用 HF 缓存；缓存缺失时仍可能联网。
    whisper_model_dir: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL_DIR", ""))
    storage_root: str = field(default_factory=lambda: os.getenv("STORAGE_ROOT", "./data"))
    bilibili_quality: int = field(default_factory=lambda: int(os.getenv("BILIBILI_QUALITY", "64")))

    @property
    def storage_root_expanded(self) -> str:
        return str(Path(self.storage_root).expanduser().resolve())


def get_settings(*, include_secrets: bool = True) -> Settings:
    """读取当前配置，数据库不可用时退回环境变量。"""
    s = Settings()
    for key, value in _load_overrides().items():
        if hasattr(s, key) and value is not None:
            setattr(s, key, value)
    desktop_mode = bool(os.getenv("WORKBENCH_DESKTOP_TOKEN"))
    session_identity = _current_session_profile_identity() if desktop_mode else None
    if session_identity:
        from app.core.keychain import cached_secret, secret
        # Keep the public profile and its session key observation consistent
        # when another profile is installed concurrently.
        with _session_profile_lock:
            if session_identity == _session_profile_identity and _session_profile is not None:
                for key, value in _session_profile.items():
                    setattr(s, key, value)
                s.llm_api_key = secret("read") if include_secrets else (cached_secret() or "")
                return s
    if desktop_mode:
        from app.core.keychain import cached_secret, secret
        s.llm_api_key = secret("read") if include_secrets else (cached_secret() or "")
    return s


_cache: Optional[dict] = None
_cache_lock = threading.Lock()
_session_profile_lock = threading.RLock()
_session_profile_identity: tuple[str, int] | None = None
_session_profile: dict[str, object] | None = None


def _current_session_profile_identity() -> tuple[str, int] | None:
    if not os.getenv("WORKBENCH_DESKTOP_TOKEN"):
        return None
    helper = os.environ.get("WORKBENCH_KEYCHAIN_HELPER")
    if not helper:
        return None
    return helper, os.getpid()


def has_session_profile() -> bool:
    """Whether this process currently has an applicable public profile override."""
    identity = _current_session_profile_identity()
    if identity is None:
        return False
    with _session_profile_lock:
        return identity == _session_profile_identity and _session_profile is not None


def set_session_profile(
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    *,
    llm_max_tokens: int | None = None,
    llm_temperature: float | None = None,
    llm_timeout: int | None = None,
) -> None:
    """Install public LLM profile fields in RAM and delegate key storage to keychain's session slot."""
    if not os.getenv("WORKBENCH_DESKTOP_TOKEN"):
        raise RuntimeError("本次运行配置仅支持桌面应用")
    helper = os.environ.get("WORKBENCH_KEYCHAIN_HELPER")
    if not helper:
        raise RuntimeError("钥匙串组件不可用，请重新构建桌面应用")
    if not isinstance(llm_base_url, str) or not llm_base_url.strip():
        raise ValueError("接口地址必须是非空字符串")
    if not isinstance(llm_model, str) or not llm_model.strip():
        raise ValueError("模型名称必须是非空字符串")
    if not isinstance(llm_api_key, str) or not llm_api_key.strip() or llm_api_key.startswith("__masked__"):
        raise ValueError("密钥必须是非空且未掩码的字符串")

    global _session_profile_identity, _session_profile
    from app.core.keychain import set_session_secret

    identity = (helper, os.getpid())
    with _session_profile_lock:
        # The keychain owns the only in-memory secret copy. Publish public
        # fields only after its existing session-key installation succeeds.
        set_session_secret(llm_api_key)
        _session_profile_identity = identity
        _session_profile = {
            "llm_base_url": llm_base_url.strip(),
            "llm_model": llm_model.strip(),
        }
        if llm_max_tokens is not None:
            _session_profile["llm_max_tokens"] = llm_max_tokens
        if llm_temperature is not None:
            _session_profile["llm_temperature"] = llm_temperature
        if llm_timeout is not None:
            _session_profile["llm_timeout"] = llm_timeout

# 允许被界面覆盖的字段。
#
# 白名单而不是「把库里所有 key 都 setattr 上去」：库里多出一个拼错的
# key（比如 lllm_model）时，无白名单的实现会**静默忽略**它，
# 用户在设置页看到「保存成功」而配置根本没生效 —— 一个零信号的失败。
# 有白名单时至少可以在保存那一刻就明确报错。
OVERRIDABLE_FIELDS = (
    "llm_base_url",
    "llm_api_key",
    "llm_model",
    "llm_max_tokens",
    "llm_temperature",
    "llm_timeout",
)

# 数值字段：DB 里一律存字符串，读取时必须转回类型。
_INT_FIELDS = ("llm_max_tokens", "llm_timeout")
_FLOAT_FIELDS = ("llm_temperature",)


def _coerce(key: str, raw: str):
    """把数据库里的字符串转成 Settings 字段的真实类型。

    为什么不存 JSON：键值表里每行一个设置最省事，而类型转换只有 5 个
    字段、规则固定。转换失败**返回 None 并跳过**，而不是抛异常 ——
    一个手滑写进去的 "abc" 不该让整个后端起不来。
    """
    if key in _INT_FIELDS:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    if key in _FLOAT_FIELDS:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return raw


def _load_overrides() -> dict:
    """读数据库里的配置覆盖，带进程内缓存。读不到就当没有覆盖。"""
    global _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        overrides: dict = {}
        try:
            from app.core.database import is_initialized, session_scope
            from app.models.entities import AppSetting

            if not is_initialized():
                # 不变量 1：数据库还没建，退回环境变量
                _cache = overrides
                return _cache
            with session_scope() as db:
                for row in db.query(AppSetting).all():
                    if row.key in OVERRIDABLE_FIELDS and row.value is not None:
                        val = _coerce(row.key, row.value)
                        if val is not None:
                            overrides[row.key] = val
        except Exception:
            # 读不到表（首次运行还没 migrate）不该让后端起不来
            overrides = {}
        _cache = overrides
        return _cache


def invalidate_settings_cache() -> None:
    """让下一次 :func:`get_settings` 重新读库。

    设置页保存后必须调它，否则「保存成功但配置没生效」——
    缓存只写不失效是这个坑的全部来源。
    """
    global _cache
    with _cache_lock:
        _cache = None


def save_settings(values: dict) -> dict:
    """把配置写进数据库并失效缓存，返回落库后的实际值。

    只接受 :data:`OVERRIDABLE_FIELDS` 里的键；传了白名单外的键直接
    ``KeyError``，由 API 层转成 400。见 :func:`_load_overrides` 的注释 ——
    静默忽略拼错的键是零信号的失败。
    """
    from app.core.database import session_scope
    from app.models.entities import AppSetting

    unknown = [k for k in values if k not in OVERRIDABLE_FIELDS]
    if unknown:
        raise KeyError(f"不可配置的字段：{', '.join(unknown)}")

    values = dict(values)
    if os.getenv("WORKBENCH_DESKTOP_TOKEN") and "llm_api_key" in values:
        from app.core.keychain import secret
        value = values.pop("llm_api_key")
        secret("delete" if value is None or value == "" else "write", value or "")
    with session_scope() as db:
        for key, value in values.items():
            if value is None:
                # None = 「回到环境变量默认值」：删掉这行覆盖，
                # 而不是把 NULL 写进去（那样 get_settings 会跳过它，
                # 效果一样但表里留一行看不懂的记录）。
                db.query(AppSetting).filter(AppSetting.key == key).delete()
                continue
            row = db.get(AppSetting, key)
            text = value if isinstance(value, str) else str(value)
            if row is None:
                db.add(AppSetting(key=key, value=text))
            else:
                row.value = text
    invalidate_settings_cache()
    return get_settings(include_secrets=False).__dict__
