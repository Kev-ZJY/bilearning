"""LLM 设置读写：密钥只返回掩码，null 删除覆盖，掩码保留现值；空字符串是实际保存的空值。

数据存放于本地 AppSetting 表；连接探测使用表单配置，不保存覆盖值。
"""

import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import (
    OVERRIDABLE_FIELDS,
    get_settings,
    has_session_profile,
    save_settings,
    set_session_profile,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])

# 掩码前缀。必须与前端 :func:`maskApiKey` 用同一个常量，
# 否则前端改文案时后端就认不出了（本项目踩过「两边各写一份别名」的坑：
# 测试改了别名、真值没变，守卫就成了恒为真的假守卫）。
_MASK_PREFIX = "__masked__"


class SettingsUpdate(BaseModel):
    """设置更新请求。所有字段可选 —— 只传要改的那几个。

    用 ``Optional[...] = None`` 而不是给每个字段一个哨兵值：
    「不传」与「传 null」在 pydantic 里默认无法区分，所以下面统一用
    ``model_fields_set`` 判断「这个字段这次真的被提交了」。
    漏用它的症状：只想改模型名，却把没传的 base_url 也当成要清空。
    """

    # **extra="forbid" 不是洁癖，是必需的。**
    #
    # pydantic 的默认行为是**静默丢弃**未声明的字段：前端把
    # `llm_base_urls` 拼错一个字符，后端收到 200、什么都不改，
    # 界面回一句「保存成功」。用户改的是一个根本不存在的配置，
    # 而系统从头到尾没有任何信号 —— 这是最难查的一类失败。
    #
    # 本项目实测踩到过：这条测试最初写成「未知字段应 400」，
    # 结果拿到 200。生产代码里那段手写的白名单检查因此永远不会执行 ——
    # 字段在到达它之前就已经被 pydantic 丢掉了。
    # **守卫写在错误的层，就等于没有守卫。**
    model_config = ConfigDict(extra="forbid")

    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_max_tokens: Optional[int] = Field(default=None, ge=1)
    llm_temperature: Optional[float] = None
    llm_timeout: Optional[int] = Field(default=None, ge=1)


def _mask(secret: str) -> str:
    """把密钥变成掩码：够用户认出是自己那把，又拼不出原值。"""
    if not secret:
        return ""
    tail = secret[-4:]
    return f"{_MASK_PREFIX}:{tail}"


def _is_mask(value: str) -> bool:
    """这个值是不是前端回传回来的掩码。

    判定只看前缀，**不看长度**：前端可能对掩码做过截断，
    按长度判定就会在某些实现下失效，而失效的后果是用户的密钥被
    一个字符串覆盖。
    """
    return isinstance(value, str) and value.startswith(_MASK_PREFIX)


def _serialize(s) -> dict:
    """配置 → 响应体。api_key 只给掩码。"""
    public_key = s.llm_api_key
    key_scope = None
    desktop_mode = bool(os.getenv("WORKBENCH_DESKTOP_TOKEN"))
    if desktop_mode:
        from app.core.keychain import cached_secret_info

        public_key, key_scope = cached_secret_info()
    key_set = None if public_key is None else bool(public_key)
    key_status = "unchecked" if public_key is None else "configured" if key_set else "missing"
    payload = {
        "llm_base_url": s.llm_base_url,
        "llm_model": s.llm_model,
        "llm_api_key_masked": _mask(public_key or ""),
        "llm_api_key_set": key_set,
        "llm_api_key_status": key_status,
        "llm_max_tokens": s.llm_max_tokens,
        "llm_temperature": s.llm_temperature,
        "llm_timeout": s.llm_timeout,
    }
    if desktop_mode:
        payload["llm_api_key_scope"] = key_scope
        payload["llm_profile_scope"] = "session" if has_session_profile() else "saved"
    return payload


@router.get("")
def read_settings():
    """读当前生效的配置（api_key 已掩码）。"""
    try:
        return _serialize(get_settings(include_secrets=False))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.put("")
def write_settings(body: SettingsUpdate):
    """更新配置。返回落库后的实际生效值（api_key 仍是掩码）。"""
    payload: dict[str, Any] = {}
    for name in body.model_fields_set:
        if name not in OVERRIDABLE_FIELDS:
            # extra="forbid" 已经在校验层拦下了未声明字段，所以正常情况下
            # 走不到这里。留着是**第二道防线**：将来有人把 extra 改回
            # ignore（或换掉 pydantic 版本导致行为变化）时，这里还能兜住。
            # 两道同样的检查不是冗余 —— 一道在契约层，一道在白名单层。
            raise HTTPException(400, f"不可配置的字段：{name}")
        value = getattr(body, name)
        if name == "llm_api_key" and isinstance(value, str) and _is_mask(value):
            # 决定 2：掩码不是新值，跳过（= 保持原样）
            continue
        payload[name] = value
    try:
        save_settings(payload)
    except (KeyError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return _serialize(get_settings(include_secrets=False))


@router.post("/session-key")
async def set_session_key(request: Request):
    """Install an in-memory desktop key for this process only."""
    if not os.getenv("WORKBENCH_DESKTOP_TOKEN"):
        raise HTTPException(404, "本次运行密钥仅支持桌面应用")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "请求必须是包含 llm_api_key 的 JSON 对象") from None
    if not isinstance(payload, dict) or set(payload) != {"llm_api_key"}:
        raise HTTPException(422, "请求必须只包含 llm_api_key")
    value = payload["llm_api_key"]
    if not isinstance(value, str) or not value.strip() or _is_mask(value):
        raise HTTPException(422, "llm_api_key 必须是非空且未掩码的字符串")

    from app.core.keychain import set_session_secret

    try:
        set_session_secret(value)
        settings = get_settings(include_secrets=True)
    except (RuntimeError, ValueError):
        raise HTTPException(503, "无法安装本次运行密钥") from None
    return _serialize(settings)


@router.post("/session-profile")
async def set_session_profile_for_process(request: Request):
    """Apply a public LLM profile and key in memory for this desktop process only."""
    if not os.getenv("WORKBENCH_DESKTOP_TOKEN"):
        raise HTTPException(404, "本次运行配置仅支持桌面应用")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "请求必须是接口地址、模型名称和密钥") from None
    required_fields = {"llm_base_url", "llm_model", "llm_api_key"}
    optional_fields = {"llm_max_tokens", "llm_temperature", "llm_timeout"}
    if (
        not isinstance(payload, dict)
        or not required_fields.issubset(payload)
        or set(payload) - required_fields - optional_fields
    ):
        raise HTTPException(422, "请求必须包含接口地址、模型名称、密钥及允许的预算字段")
    base_url = payload["llm_base_url"]
    model = payload["llm_model"]
    key = payload["llm_api_key"]
    if not isinstance(base_url, str) or not base_url.strip():
        raise HTTPException(422, "接口地址必须是非空字符串")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(422, "模型名称必须是非空字符串")
    if not isinstance(key, str) or not key.strip() or _is_mask(key):
        raise HTTPException(422, "密钥必须是非空且未掩码的字符串")

    numeric_payload = {field: payload[field] for field in optional_fields if field in payload}
    try:
        numeric = SettingsUpdate.model_validate(numeric_payload)
    except ValidationError:
        raise HTTPException(422, "本次运行配置的预算字段格式无效") from None
    numeric_values = {
        field: getattr(numeric, field)
        for field in optional_fields
        if field in numeric.model_fields_set and getattr(numeric, field) is not None
    }

    try:
        set_session_profile(base_url.strip(), model.strip(), key, **numeric_values)
        settings = get_settings(include_secrets=False)
    except (RuntimeError, ValueError):
        raise HTTPException(503, "无法安装本次运行配置") from None
    return _serialize(settings)


@router.post("/test")
def test_connection(body: SettingsUpdate):
    from app.agent.runtime.llm import LLMClient
    import time

    settings = get_settings(include_secrets=False)
    values = {key: getattr(settings, key) for key in OVERRIDABLE_FIELDS}
    for key in body.model_fields_set:
        value = getattr(body, key)
        if value is not None and not (
            key == "llm_api_key" and (not value or _is_mask(value))
        ):
            values[key] = value
    if not values["llm_base_url"] or not values["llm_model"]:
        raise HTTPException(400, "请填写接口地址和模型名称")
    supplied_key = body.llm_api_key
    uses_saved_key = not (isinstance(supplied_key, str) and supplied_key and not _is_mask(supplied_key))
    if os.getenv("WORKBENCH_DESKTOP_TOKEN") and uses_saved_key:
        from app.core.keychain import secret

        try:
            values["llm_api_key"] = secret("read", retry=True)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
    client = LLMClient(
        values["llm_base_url"],
        values["llm_api_key"],
        values["llm_model"],
        max_tokens=256,
        timeout=min(60, values["llm_timeout"]),
    )
    client.max_retries = 1
    start = time.monotonic()
    # A 256-token connectivity probe needs final content, not a reasoning-only
    # response. Match the same direct-provider boundary as tool compatibility.
    endpoint = values["llm_base_url"].rstrip("/").removesuffix("/chat/completions")
    probe_options = {}
    if endpoint in {"https://api.deepseek.com", "https://api.deepseek.com/v1"} and values["llm_model"] in {"deepseek-flash", "deepseek-v4-pro"}:
        probe_options["thinking"] = {"type": "disabled"}
    try:
        answer = client.chat_text([{"role": "user", "content": "Reply with OK."}], **probe_options)
        if not answer.strip():
            raise ValueError("empty response")
    except Exception as exc:
        # Never return provider bodies: they may echo credentials or request data.
        raise HTTPException(
            502, "连接测试失败，请检查接口地址、密钥和模型名称"
        ) from exc
    return {
        "ok": True,
        "elapsed_ms": round((time.monotonic() - start) * 1000),
        "model": values["llm_model"],
    }
