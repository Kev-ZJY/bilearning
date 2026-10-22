"""兼容 Chat Completions 的 JSON、文本、工具调用与文本流客户端。

只读取 content/tool_calls，不将 reasoning 字段当作回答。
网络异常有界重试；空正文最多恢复一次，截断不提交，不自动增加预算。
非流式请求只能在检查点协作取消；流式请求在收到分块时检查并关闭连接。
关闭本地连接不保证远端停止计算或计费。timeout 也不是整个任务的硬截止时间。
"""

import json
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import requests

from app.agent.runtime.context import ContextBudgetError, estimate_llm_input_tokens
from app.agent.runtime.model_limits import (
    DEFAULT_INPUT_BUDGET_TOKENS,
    INPUT_SAFETY_MARGIN_TOKENS,
    resolve_model_limits,
)
from app.core.cancel import TaskCancelled
from app.agent.runtime.task_budget import ModelAttemptBudgetExceeded
from app.agent.runtime.http_transport import (
    RequestDeadlineExceeded,
    post as request_post,
    request_controls,
)


class _DeterministicFailure(RuntimeError):
    """无可提交结果且本轮不再自动恢复；不代表模型输出是确定性的。"""


class _ProviderStatusFailure(RuntimeError):
    """A structured numeric status inside an HTTP-200 provider error body."""

    def __init__(self, status_code: int):
        self.provider_status_code = status_code
        super().__init__(f"provider status {status_code}")


class _UnclassifiedProviderFailure(RuntimeError):
    """An error body without a structured numeric status; never retain its text."""

    def __init__(self):
        super().__init__("unclassified provider error")


class _InvalidModelOutput(ValueError):
    """Safe parse guidance without retaining the provider's generated body."""

    def __init__(self):
        super().__init__("无法从模型输出中解析 JSON；请检查结构化响应格式。")


def extract_json(text: str) -> Any:
    """从 LLM 输出里提取并解析 JSON。

    容忍三种常见情况（reasoning 模型经常这样输出）：
    - 纯 JSON：``{"a": 1}``
    - 包在代码围栏里：````` ```json\\n{"a": 1}\\n``` `````
    - 前后有解释文字：``好的，结果如下：\\n{"a": 1}\\n以上。``

    策略：先尝试直接 ``json.loads``（去掉代码围栏后）；失败则用正则找第一个
    ``{`` 到最后一个 ``}`` 的片段再解析。全部失败抛 ``ValueError``，错误信息里
    带上原文前 200 字符方便调试。
    """
    if text is None:
        raise ValueError("extract_json: 输入为 None")

    candidate = text.strip()

    # 策略一：去掉 ```json ... ``` / ``` ... ``` 代码围栏后直接解析
    fenced = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass  # 围栏内容也可能不是合法 JSON，继续走正则兜底

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 策略二：正则定位第一个 { 到最后一个 } 的片段（容忍前后解释文字）
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # 全部失败：带原文前 200 字符，方便定位问题
    preview = candidate[:200]
    raise ValueError(f"extract_json: 无法从模型输出中解析 JSON，原文前 200 字符：{preview!r}")


class LLMClient:
    """OpenRouter 兼容接口的 LLM 客户端。

    网络尝试（含首次请求）和空正文恢复均有上限；不把 reasoning 内容当作答案。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 200_000,
        temperature: float = 0.2,
        timeout: int = 900,
        on_retry: Callable[[dict], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        task_budget=None,
        on_response_diagnostic: Callable[[dict], None] | None = None,
        on_call_event: Callable[[dict], None] | None = None,
    ):
        self.base_url = base_url.rstrip("/").removesuffix("/chat/completions")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.response_diagnostics = []
        self.transport_diagnostics = []
        self.on_response_diagnostic = on_response_diagnostic
        self.on_call_event = on_call_event
        self._activity_local = threading.local()
        self._activity_call_lock = threading.Lock()
        self._activity_call_sequence = 0
        # 指数退避：第 1 次重试 sleep 2s，第 2 次 4s（见 _try_once 里的退避公式）
        self.backoff_base = 2.0
        self.max_retries = 3
        # 网络重试回调。attempt 是已发出的请求序号；max_retries 表示总尝试次数。
        # 形如 {"attempt": 1, "reason": "...", "elapsed": 12.3}。
        self.on_retry = on_retry
        # 取消检查点。返回 True 表示用户已请求中止。
        #
        # 这里是**协作式中止**在 LLM 层的落点，也是全项目唯一能真正打断
        # 阻塞请求的地方——但只有流式路径做得到（见 _stream_once 里逐块检查
        # + resp.close()）。非流式路径只能在**请求发出前**检查：Python 线程
        # 无法强杀，requests 也没有「中断一个已在飞的 POST」的 API。
        # 这不是本项目的偷懒，是所有 Python agent harness 的共同处境：
        # **在可中断层真中断，在不可中断层协作等待。**
        self.should_cancel = should_cancel
        self.task_budget = task_budget

    @contextmanager
    def activity_context(self, **values):
        """Attach public workflow identity to observations without touching prompts."""
        allowed = {
            "actor", "agent_id", "parent_task_id", "child_task_id", "stage",
            "round", "cycle", "tool_name", "execution",
        }
        context = {}
        for key, value in values.items():
            if key not in allowed:
                continue
            if key in {"round", "cycle"}:
                if type(value) is int and value > 0:
                    context[key] = value
            elif isinstance(value, str) and value.strip():
                context[key] = value.strip()[:120]
        previous = getattr(self._activity_local, "context", None)
        merged = {**(previous or {}), **context}
        self._activity_local.context = merged
        try:
            yield
        finally:
            if previous is None:
                try:
                    del self._activity_local.context
                except AttributeError:
                    pass
            else:
                self._activity_local.context = previous

    def _safe_call_event(self, event):
        """Best-effort public observation; observer failures never affect the call."""
        if self.on_call_event is None:
            return
        try:
            self.on_call_event(dict(event))
        except Exception:
            pass

    def _next_activity_call_id(self):
        with self._activity_call_lock:
            self._activity_call_sequence += 1
            return self._activity_call_sequence

    def _call_event(self, call, status, *, error=None):
        event = {
            "call_id": call["call_id"],
            "method": call["method"],
            "model": self.model if isinstance(self.model, str) else "unknown",
            "status": status,
            "attempts": call["attempts"],
            "retry_count": max(0, call["attempts"] - 1),
            "started_at": call["started_at"],
            "elapsed_ms": max(0, int((time.monotonic() - call["started_mono"]) * 1000)),
            **call["context"],
        }
        if status in {"done", "error", "cancelled"}:
            event["finished_at"] = datetime.now(timezone.utc).isoformat()
            diagnostic = call.get("diagnostic") or {}
            for key in (
                "finish_reason", "max_tokens", "prompt_tokens",
                "completion_tokens", "reasoning_tokens", "total_tokens",
                "http_status",
            ):
                if key in diagnostic:
                    event[key] = diagnostic[key]
        if error is not None:
            # Exception text can contain provider payloads. Publish only the type.
            event["error"] = type(error).__name__
        return event

    def _observe_call(self, method, callback):
        if self.on_call_event is None:
            return callback()
        previous = getattr(self._activity_local, "call", None)
        if previous is not None:
            return callback()
        call = {
            "call_id": self._next_activity_call_id(),
            "method": method,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "started_mono": time.monotonic(),
            "attempts": 0,
            "diagnostic": None,
            "context": dict(getattr(self._activity_local, "context", {}) or {}),
        }
        self._activity_local.call = call
        self._safe_call_event(self._call_event(call, "running"))
        status, error = "done", None
        try:
            return callback()
        except BaseException as exc:
            error = exc
            status = "cancelled" if isinstance(exc, TaskCancelled) else "error"
            raise
        finally:
            self._safe_call_event(self._call_event(call, status, error=error))
            if previous is None:
                try:
                    del self._activity_local.call
                except AttributeError:
                    pass
            else:
                self._activity_local.call = previous

    def _observe_stream(self, method, iterator_factory):
        if self.on_call_event is None:
            yield from iterator_factory()
            return
        previous = getattr(self._activity_local, "call", None)
        if previous is not None:
            yield from iterator_factory()
            return
        call = {
            "call_id": self._next_activity_call_id(),
            "method": method,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "started_mono": time.monotonic(),
            "attempts": 0,
            "diagnostic": None,
            "context": dict(getattr(self._activity_local, "context", {}) or {}),
        }
        self._activity_local.call = call
        self._safe_call_event(self._call_event(call, "running"))
        status, error = "done", None
        try:
            yield from iterator_factory()
        except BaseException as exc:
            error = exc
            status = "cancelled" if isinstance(exc, TaskCancelled) else "error"
            raise
        finally:
            self._safe_call_event(self._call_event(call, status, error=error))
            if previous is None:
                try:
                    del self._activity_local.call
                except AttributeError:
                    pass
            else:
                self._activity_local.call = previous

    def chat_text(self, messages: list[dict], **kw) -> str:
        """发起一次对话，返回**原始字符串**，不做任何解析。

        为什么必须单独开一条路径（这是本项目 P0 空洞的技术前提）：
        :meth:`chat_json` 结尾是 ``return extract_json(content)``，**强制把回复当
        JSON 解析**。助手对话回的是自然语言，走 chat_json() 会直接抛
        ``ValueError: 无法从模型输出中解析 JSON``——这正是「agent 零回复」在
        LLM 层的具体死因。更隐蔽的是：回复里只要出现 ``{"a": 1}`` 这种片段
        （用户问带请求体的知识点时必然出现），extract_json 就会把那个片段
        截出来当结果返回，用户看到的是一段莫名其妙的 JSON。

        除此之外，两条路径的错误处理**完全一致**——它们共用 :meth:`_run`
        这条主干，只有最后「怎么处理 content」不同。这里刻意不复制重试逻辑：
        复制出来的第二份必然与第一份漂移。
        """
        return self._run(messages, self._identity, diagnostic_method="chat_text", **kw)

    def chat_json(self, messages: list[dict], **kw) -> Any:
        """发起一次对话并把回复解析成结构化数据。

        与 :meth:`chat` 等价，单独起名是为了在调用处表明「这里要 JSON」，
        避免后来者以为 chat_text 才是默认。与 chat() 一样解析失败即抛错，
        不返回 None——静默 None 会让上层把「模型没听懂」当成「用户没需求」。
        """
        return self._run(messages, extract_json, diagnostic_method="chat_json", **kw)

    @staticmethod
    def _identity(content: str) -> str:
        """content 的恒等处理器：原样返回，用于自然语言路径。

        必须是 ``staticmethod``：``chat_text`` 里以 ``self._identity`` 传入，
        若它是普通方法，绑定后签名会变成 ``(self, content)``，被当作
        ``parse(content)`` 调用时实际传了两个位置参数，直接 TypeError。
        """
        return content

    def chat_tools(self, messages: list[dict], tools: list[dict],
                   **kw) -> dict:
        """带工具定义的一轮对话，返回 ``{"content", "tool_calls", "finish_reason"}``。

        agent 的工具循环用它。成功判定与 :meth:`_try_once` **不同**，必须分开：

        ``finish_reason == "tool_calls"`` 是**正常结束**（模型要调工具），而
        :meth:`_try_once` 把所有非 ``"stop"`` 都判成截断。直接套用会把每一次
        工具调用都误判成失败——工具循环第一轮必然返回 tool_calls，所以这个 bug
        表现为「agent 一调工具就卡住」。

        另外 ``content`` 在纯工具轮里通常是空的（模型只发 tool_calls，
        实测 content 是一句「我先检索相关知识点。」），所以不能用
        「content 为空即失败」判定，否则纯工具轮同样被误判。两处都必须
        只看 finish_reason：``stop`` / ``tool_calls`` 都算成功，只有
        ``length`` 才是真截断。
        """
        budget = kw.pop("max_tokens", self.max_tokens)
        extra = {k: v for k, v in kw.items() if k != "max_tokens"}
        extra["tools"] = tools
        extra.setdefault("tool_choice", "auto")
        return self._observe_call(
            "chat_tools", lambda: self._try_once_tools(messages, budget, extra)
        )

    def _check_cancel(self) -> None:
        """取消检查点。已请求中止则抛 :class:`TaskCancelled`。

        放在**发请求之前**：非流式请求一旦发出去就没有中途打断的手段
        （见 :attr:`should_cancel` 的说明），所以这里的检查决定了
        「点了停止之后最多还会多跑多久」——答案是「当前这次调用返回为止」，
        而如果在这里就拦下来，则一次都不用多跑。
        """
        if self.should_cancel is not None and self.should_cancel():
            raise TaskCancelled("LLM 调用前收到取消请求")

    def _post(self, *args, **kwargs):
        """Reserve one task-wide HTTP attempt immediately before sending it."""
        diagnostic_method = kwargs.pop("_diagnostic_method", "unknown")
        self._activity_local.last_request_id = None
        if self.task_budget is not None:
            self.task_budget.reserve_attempt()
        call = getattr(self._activity_local, "call", None)
        if call is not None:
            call["attempts"] += 1
            status = "retrying" if call["attempts"] > 1 else "running"
            self._safe_call_event(self._call_event(call, status))
        with request_controls(
            should_cancel=self.should_cancel,
            on_transport_event=self._record_transport_diagnostic,
            call_kind=diagnostic_method,
        ):
            return request_post(*args, **kwargs)

    def _record_transport_diagnostic(self, event):
        """Keep request/transport receipts separate from model responses."""
        if not isinstance(event, dict):
            return
        row = dict(event)
        allowed_kinds = {"request", "transport"}
        if row.get("kind") not in allowed_kinds:
            return
        self.transport_diagnostics.append(row)
        if row.get("event") == "request_started":
            request_id = row.get("request_id")
            if isinstance(request_id, str):
                self._activity_local.last_request_id = request_id
                call = getattr(self._activity_local, "call", None)
                if call is not None:
                    call["request_id"] = request_id
        if self.on_response_diagnostic is not None:
            try:
                self.on_response_diagnostic(dict(row))
            except Exception:
                # Persistence is best-effort and cannot affect an HTTP attempt.
                pass

    @staticmethod
    def _deadline_failure_message(error, *, streaming=False):
        prefix = "LLM 流式请求超过总截止时间，未自动重试。" if streaming else "LLM 请求超过总截止时间，未自动重试。"
        phase = getattr(error, "transport_phase", None)
        if phase not in {"connect_or_response_headers", "response_headers", "body"}:
            phase = "connect_or_response_headers"
        status = getattr(error, "http_status", None)
        if type(status) is not int or status < 100 or status > 599:
            status = None
        received = getattr(error, "received_bytes", 0)
        if type(received) is not int or received < 0:
            received = 0
        elapsed = getattr(error, "elapsed_ms", 0)
        if type(elapsed) is not int or elapsed < 0:
            elapsed = 0
        status_text = "未知" if status is None else str(status)
        return (
            f"{prefix}阶段={phase}；HTTP状态={status_text}；"
            f"已收响应体字节={received}；耗时={elapsed}ms。"
        )

    def _tool_protocol_options(self) -> dict:
        # DeepSeek defaults to thinking mode, whose tool history requires every
        # reasoning_content field. This client keeps content/tool_calls only, so
        # content-consumption paths request the supported non-thinking protocol.
        # https://api-docs.deepseek.com/guides/thinking_mode/
        if self.base_url in {"https://api.deepseek.com", "https://api.deepseek.com/v1"} and self.model in {"deepseek-flash", "deepseek-v4-pro"}:
            return {"thinking": {"type": "disabled"}}
        return {}

    def _model_request_compatibility_options(self) -> dict:
        """Apply only the verified OpenRouter model-specific reasoning controls."""
        if self.base_url != "https://openrouter.ai/api/v1":
            return {}
        if self.model == "dots-studio/dots-3-note-preview:free":
            return {"reasoning": {"enabled": False}}
        if self.model == "stealth/space-bunny-alpha":
            return {"reasoning": {"effort": "low"}}
        return {}

    def input_budget_tokens(self, reserved_output: int | None = None) -> int:
        """Return the model-aware input budget with the configured output reserve."""
        limits = resolve_model_limits(self.base_url, self.model)
        if limits is None:
            return DEFAULT_INPUT_BUDGET_TOKENS
        reserve = self.max_tokens if reserved_output is None else reserved_output
        if type(reserve) is not int or reserve < 0:
            raise ValueError("reserved_output must be a non-negative integer")
        reserve = min(reserve, limits.max_output_tokens)
        return max(
            0,
            limits.context_window - reserve - INPUT_SAFETY_MARGIN_TOKENS,
        )

    def _effective_output_tokens(self, messages: list[dict], payload: dict) -> int:
        """Clamp a request to its exact input, model window, and output limits."""
        limits = resolve_model_limits(self.base_url, self.model)
        requested = payload.get("max_tokens", self.max_tokens)
        if limits is None:
            # An explicit per-call value may lower the user's configured cap,
            # but it must never raise it. Keep unknown-model behavior otherwise
            # unchanged because we cannot safely infer that model's physical
            # context or output limit.
            if (
                type(requested) is int and requested >= 0
                and type(self.max_tokens) is int and self.max_tokens >= 0
            ):
                return min(requested, self.max_tokens)
            return requested
        if type(requested) is not int or requested < 0:
            raise ContextBudgetError("模型输出 token 上限必须是非负整数。")
        if type(self.max_tokens) is not int or self.max_tokens < 0:
            raise ContextBudgetError("配置的模型输出 token 上限必须是非负整数。")
        requested = min(requested, self.max_tokens)
        fields = {
            key: value for key, value in payload.items()
            if key not in {"model", "messages", "max_tokens", "temperature", "stream", "tools"}
        }
        input_tokens = estimate_llm_input_tokens(
            messages,
            tool_schemas=payload.get("tools"),
            extra_payload=fields,
        )
        available = limits.context_window - input_tokens - INPUT_SAFETY_MARGIN_TOKENS
        if available <= 0:
            raise ContextBudgetError(
                f"模型输入估算为 {input_tokens} token，无法保留 "
                f"{INPUT_SAFETY_MARGIN_TOKENS} token 安全余量。"
            )
        return min(requested, limits.max_output_tokens, available)

    @staticmethod
    def _safe_diagnostic_int(value):
        """Return only non-negative integer metadata, excluding bools and text."""
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    @staticmethod
    def _valid_http_status(value):
        if type(value) is int and 400 <= value < 600:
            return value
        return None

    @classmethod
    def _numeric_error_status(cls, error_value):
        """Read only a numeric status from structured provider error data."""
        direct = cls._valid_http_status(error_value)
        if direct is not None:
            return direct
        if not isinstance(error_value, dict):
            return None
        for key in ("status_code", "http_status", "status", "code"):
            found = cls._valid_http_status(error_value.get(key))
            if found is not None:
                return found
        nested = error_value.get("error")
        if isinstance(nested, dict):
            for key in ("status_code", "http_status", "status", "code"):
                found = cls._valid_http_status(nested.get(key))
                if found is not None:
                    return found
        return None

    @classmethod
    def _hard_quota_kind(cls, error_value, *, status=None):
        """Classify only explicit, bounded provider quota signals.

        ``status`` is an actual HTTP status or an explicitly numeric status in
        a structured provider error. Free-model wording is meaningful only for
        an actual/structured 429; arbitrary prose never manufactures a status.
        """
        if status is not None and status != 429:
            return None
        if not isinstance(error_value, dict):
            return None

        candidates = [error_value]
        nested = error_value.get("error")
        if isinstance(nested, dict):
            candidates.append(nested)

        hard_codes = {
            "insufficient_quota",
            "insufficient_credits",
            "credits_exhausted",
            "billing_hard_limit",
        }
        daily_codes = {"free-models-per-day", "daily_free_quota_exceeded"}
        for candidate in candidates:
            for key in ("code", "error_code", "type"):
                value = candidate.get(key)
                if not isinstance(value, str) or len(value) > 96:
                    continue
                code = value.strip().lower()
                if code in hard_codes:
                    return "hard_quota"
                if status == 429 and code in daily_codes:
                    return "daily_free_quota"

        if status == 429:
            for candidate in candidates:
                message = candidate.get("message")
                if not isinstance(message, str) or len(message) > 512:
                    continue
                lowered = message.lower()
                remaining = candidate.get("remaining")
                if type(remaining) is int and remaining > 0:
                    continue
                if "free-models-per-day" in lowered or "daily free quota exceeded" in lowered:
                    return "daily_free_quota"
        return None

    @classmethod
    def _response_error_payload(cls, response, *, streaming=False):
        """Best-effort bounded JSON error extraction; never return raw body text."""
        if response is None:
            return None

        bounded_reader = getattr(response, "bounded_error_json", None)
        if callable(bounded_reader):
            try:
                data = bounded_reader(max_bytes=32_768)
            except (TaskCancelled, RequestDeadlineExceeded,
                    ModelAttemptBudgetExceeded, ContextBudgetError):
                raise
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            error = data.get("error")
            return error if isinstance(error, dict) else None

        # Compatibility for injected fakes/older response-shaped objects. The
        # production adapter always exposes bounded_error_json above.
        headers = getattr(response, "headers", None)
        content_length = None
        try:
            raw_length = headers.get("content-length") if headers is not None else None
            if isinstance(raw_length, str) and raw_length.isdecimal():
                content_length = int(raw_length)
            elif type(raw_length) is int and raw_length >= 0:
                content_length = raw_length
        except (TaskCancelled, RequestDeadlineExceeded,
                ModelAttemptBudgetExceeded, ContextBudgetError):
            raise
        except Exception:
            content_length = None
        max_error_bytes = 32_768
        if content_length is not None and content_length > max_error_bytes:
            return None

        try:
            body = getattr(response, "content", None)
        except (TaskCancelled, RequestDeadlineExceeded,
                ModelAttemptBudgetExceeded, ContextBudgetError):
            raise
        except Exception:
            body = None
        if isinstance(body, (bytes, bytearray)) and len(body) > max_error_bytes:
            return None
        try:
            data = response.json()
        except (TaskCancelled, RequestDeadlineExceeded,
                ModelAttemptBudgetExceeded, ContextBudgetError):
            raise
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        error = data.get("error")
        if not isinstance(error, dict):
            return None
        return error

    @staticmethod
    def _hard_quota_failure_message(*, actual_http_status=None):
        if actual_http_status == 429:
            return "HTTP 429：服务账户额度或计费已耗尽，本次调用停止且未自动重试。"
        return "服务返回了明确的账户额度或计费耗尽错误，本次调用停止且未自动重试。"

    def _raise_for_hard_quota_response(self, response, *, streaming=False):
        actual_status = self._response_http_status(response)
        if actual_status != 429:
            return
        error = self._response_error_payload(response, streaming=streaming)
        if self._hard_quota_kind(error, status=actual_status):
            raise _DeterministicFailure(
                self._hard_quota_failure_message(actual_http_status=actual_status)
            )

    @staticmethod
    def _response_http_status(response, error=None):
        """Return safe actual HTTP status metadata without parsing exception text."""
        candidates = (
            getattr(response, "status_code", None),
            getattr(getattr(error, "response", None), "status_code", None),
            getattr(error, "status_code", None),
        )
        for candidate in candidates:
            if type(candidate) is int and 100 <= candidate < 600:
                return candidate
        return None

    @classmethod
    def _failure_status(cls, error, response=None):
        if isinstance(error, _ProviderStatusFailure):
            return error.provider_status_code
        return cls._valid_http_status(cls._response_http_status(response, error))

    @staticmethod
    def _retryable_http_status(status):
        return status in {408, 409, 425, 429} or 500 <= status < 600

    @staticmethod
    def _http_failure_message(status):
        if status == 400 or status == 422:
            category, next_step = "请求格式或参数被拒绝", "请检查调用参数和端点要求。"
        elif status == 401:
            category, next_step = "认证失败", "请检查服务端密钥和认证配置。"
        elif status == 402:
            category, next_step = "账户额度或计费受限", "请检查服务账户的可用额度。"
        elif status == 403:
            category, next_step = "访问权限不足", "请检查服务端权限和模型授权。"
        elif status == 404:
            category, next_step = "模型或接口不存在", "请检查模型名称和服务地址。"
        elif status == 413:
            category, next_step = "请求内容过大", "请缩小本次输入后重试。"
        elif status == 408:
            category, next_step = "请求超时", "请稍后重试。"
        elif status == 409:
            category, next_step = "服务请求冲突", "请稍后重试。"
        elif status == 425:
            category, next_step = "服务暂不接受请求", "请稍后重试。"
        elif status == 429:
            category, next_step = "服务限流", "请等待限流窗口后重试。"
        elif 500 <= status < 600:
            category, next_step = "服务暂时不可用", "请稍后重试。"
        else:
            category, next_step = "请求被服务拒绝", "请检查服务地址和请求设置。"
        return f"HTTP {status}：{category}。{next_step}"

    @staticmethod
    def _safe_retry_reason(error, status=None):
        if status is not None:
            return LLMClient._http_failure_message(status)
        if isinstance(error, _InvalidModelOutput):
            return str(error)
        if isinstance(error, _UnclassifiedProviderFailure):
            return "服务返回未分类错误；未从错误文本推断状态码，请检查服务端点。"
        if isinstance(error, requests.exceptions.Timeout):
            return "请求超时，请稍后重试。"
        return "网络请求失败，请检查网络或服务状态。"

    def _record_response_diagnostic(self, method, budget, finish_reason, usage,
                                    http_status=None):
        """Record and observe an allowlisted, content-free response receipt."""
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("completion_tokens_details")
        details = details if isinstance(details, dict) else {}
        allowed_finishes = {"stop", "length", "tool_calls", "content_filter"}
        if isinstance(finish_reason, str):
            finish = finish_reason if finish_reason in allowed_finishes else "other"
        else:
            finish = "unknown" if finish_reason is None else "other"
        row = {
            "method": method,
            "finish_reason": finish,
            "max_tokens": self._safe_diagnostic_int(budget),
            "prompt_tokens": self._safe_diagnostic_int(usage.get("prompt_tokens")),
            "completion_tokens": self._safe_diagnostic_int(usage.get("completion_tokens")),
            "reasoning_tokens": self._safe_diagnostic_int(details.get("reasoning_tokens")),
            "total_tokens": self._safe_diagnostic_int(usage.get("total_tokens")),
            "http_status": self._safe_diagnostic_int(http_status),
        }
        request_id = getattr(self._activity_local, "last_request_id", None)
        if isinstance(request_id, str):
            row["request_id"] = request_id
        self.response_diagnostics.append(row)
        call = getattr(self._activity_local, "call", None)
        if call is not None:
            call["diagnostic"] = dict(row)
        if self.on_response_diagnostic is not None:
            try:
                self.on_response_diagnostic(dict(row))
            except Exception:
                # Diagnostics are best-effort; never mask a model failure or cause a retry.
                pass
        return row

    def _try_once_tools(self, messages: list[dict], budget: int,
                        extra: dict) -> dict:
        """工具轮的有界网络尝试。成功判定只认 finish_reason（见 chat_tools）。"""
        last_reason = "网络请求失败，请检查网络或服务状态。"
        for attempt in range(self.max_retries):
            self._check_cancel()
            resp = None
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": budget,
                    "temperature": self.temperature,
                }
                payload.update(self._model_request_compatibility_options())
                payload.update(self._tool_protocol_options())
                payload.update(extra)
                payload["max_tokens"] = self._effective_output_tokens(messages, payload)
                budget = payload["max_tokens"]

                resp = self._post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    _diagnostic_method="chat_tools",
                    timeout=(30, self.timeout),
                )
                resp.raise_for_status()
                data = resp.json()

                if isinstance(data, dict) and "error" in data:
                    self._record_response_diagnostic(
                        "chat_tools", budget, None, data.get("usage"),
                        self._response_http_status(resp),
                    )
                    status = self._numeric_error_status(data["error"])
                    if self._hard_quota_kind(data["error"], status=status):
                        raise _DeterministicFailure(
                            self._hard_quota_failure_message()
                        )
                    if status is not None:
                        if not self._retryable_http_status(status):
                            raise _DeterministicFailure(self._http_failure_message(status))
                        raise _ProviderStatusFailure(status)
                    raise _UnclassifiedProviderFailure()

                choices = data["choices"]
                if not choices:
                    raise RuntimeError("LLM 响应 choices 为空")
                choice = choices[0]
                message = choice.get("message", {})
                finish_reason = choice.get("finish_reason")
                self._record_response_diagnostic(
                    "chat_tools", budget, finish_reason, data.get("usage"),
                    getattr(resp, "status_code", None),
                )

                # 只有 length 才是真截断。stop 与 tool_calls 都是正常结束。
                #
                # 两者都抛 _DeterministicFailure（截断 / 未知枚举）：都是确定性结果，
                # 重试只会让用户多等几轮同样的失败。
                if finish_reason == "length":
                    raise _DeterministicFailure(
                        f"LLM 工具轮响应被截断（finish_reason='length'，"
                        f"预算 {budget}）。未提交不完整内容，"
                        f"请检查单次输出与推理预算。"
                    )
                if finish_reason not in ("stop", "tool_calls"):
                    # 未知枚举：保守当失败，不当成功（避免把错误当回复返回）
                    raise _DeterministicFailure(
                        f"LLM 返回未预期的 finish_reason：{finish_reason!r}。"
                        f"这是确定性结果，重试没有意义。"
                    )

                # 实测 2：只读 content，绝不碰 reasoning / reasoning_details
                return {
                    "content": message.get("content") or "",
                    "tool_calls": message.get("tool_calls") or [],
                    "finish_reason": finish_reason,
                }
            except ModelAttemptBudgetExceeded:
                raise
            except ContextBudgetError:
                raise
            except TaskCancelled:
                raise
            except RequestDeadlineExceeded as exc:
                raise _DeterministicFailure(
                    self._deadline_failure_message(exc)
                ) from None
            except _DeterministicFailure:
                # 确定性失败：立刻冒泡，绝不重试。
                #
                # 这里刻意用**异常类型**而不是 `if "被截断" in str(e)` 这种
                # 字符串匹配：改一次错误文案就会让判定静默失效，而症状是
                # 「用户又看到 15 分钟静默」——极难定位。类型不会漂移。
                raise
            except Exception as e:
                actual_status = self._response_http_status(resp, e)
                if actual_status is not None and actual_status >= 400:
                    self._record_response_diagnostic(
                        "chat_tools", budget, None, {}, actual_status,
                    )
                if actual_status == 429:
                    error_payload = self._response_error_payload(resp)
                    if self._hard_quota_kind(error_payload, status=actual_status):
                        raise _DeterministicFailure(
                            self._hard_quota_failure_message(
                                actual_http_status=actual_status
                            )
                        ) from None
                status = self._failure_status(e, resp)
                if status is not None and not self._retryable_http_status(status):
                    raise _DeterministicFailure(self._http_failure_message(status)) from None
                last_reason = self._safe_retry_reason(e, status)

            if attempt < self.max_retries - 1:
                time.sleep(self.backoff_base * (2 ** attempt))

        raise RuntimeError(
            f"LLM 工具轮调用失败（网络尝试共 {self.max_retries} 次，含首次）：{last_reason}"
        )

    def chat(self, messages: list[dict], **kw) -> Any:
        """发起一次对话，返回已解析的 dict / list（不是原始字符串）。

        上层要直接用结构化数据。本方法等价于 :meth:`chat_json`，保留是为了
        不改动既有调用方。

        绝对只读 ``message.content``，不碰 ``reasoning`` / ``reasoning_details``（实测 2）。
        """
        return self._run(messages, extract_json, diagnostic_method="chat", **kw)

    # ------------------------------------------------------------------
    # 流式（token 级）—— 只给对话通道用，抽取流水线一律走非流式
    # ------------------------------------------------------------------
    #
    # 为什么不把现有方法改成流式：抽取流水线要的是**一整块完整 JSON**，
    # 流式对它毫无价值，反而要把所有 chunk 拼起来再解析，多一层出错面。
    # 两条路径并存，各取所需，不要为了「统一」把抽取也拖进流式。

    def stream_tools(self, messages, *, tools, on_delta=None, **kw):
        return self._observe_call(
            "stream_tools",
            lambda: self._stream_tools_impl(
                messages, tools=tools, on_delta=on_delta, **kw
            ),
        )

    def _stream_tools_impl(self, messages, *, tools, on_delta=None, **kw):
        """One real streaming generation for both visible text and tool arguments."""
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "max_tokens": self.max_tokens, "temperature": self.temperature}
        payload.update(self._tool_protocol_options())
        if tools:
            payload["tools"] = tools
        payload.update(self._model_request_compatibility_options())
        payload.update(kw)
        self._check_cancel()
        payload["max_tokens"] = self._effective_output_tokens(messages, payload)
        response = None
        content, calls, finish, usage = [], {}, None, {}
        try:
            response = self._post(f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload, stream=True, _diagnostic_method="stream_tools",
                timeout=(30, self.timeout))
            self._raise_for_hard_quota_response(response, streaming=True)
            response.raise_for_status()
            for event in self._iter_sse_data(response):
                self._check_cancel()
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                if event.get("error"):
                    status = self._numeric_error_status(event["error"])
                    if self._hard_quota_kind(event["error"], status=status):
                        raise _DeterministicFailure(
                            self._hard_quota_failure_message()
                        )
                    if status is None:
                        raise _UnclassifiedProviderFailure()
                    if not self._retryable_http_status(status):
                        raise _DeterministicFailure(self._http_failure_message(status))
                    raise _ProviderStatusFailure(status)
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    content.append(piece)
                    if on_delta:
                        on_delta(piece)
                for fragment in delta.get("tool_calls") or []:
                    index = fragment.get("index", 0)
                    call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if fragment.get("id"):
                        call["id"] = fragment["id"]
                    for key in ("name", "arguments"):
                        value = (fragment.get("function") or {}).get(key)
                        if value:
                            call["function"][key] += value
                finish = choice.get("finish_reason") or finish
            if finish not in ("stop", "tool_calls"):
                raise RuntimeError(f"模型流未完整结束（{finish or '连接中断'}），未执行工具")
            if not content and not calls:
                raise RuntimeError("模型未返回正文或工具调用")
            result = [calls[k] for k in sorted(calls)]
            for call in result:
                if not call["id"] or not call["function"]["name"]:
                    raise ValueError("模型工具调用不完整")
                json.loads(call["function"]["arguments"] or "{}")
            return {"content": "".join(content), "tool_calls": result, "finish_reason": finish}
        except (ModelAttemptBudgetExceeded, ContextBudgetError, TaskCancelled, _DeterministicFailure):
            raise
        except RequestDeadlineExceeded as exc:
            raise _DeterministicFailure(
                self._deadline_failure_message(exc, streaming=True)
            ) from None
        except (requests.RequestException, _ProviderStatusFailure, _UnclassifiedProviderFailure) as e:
            status = self._failure_status(e, response)
            if status is not None and not self._retryable_http_status(status):
                raise _DeterministicFailure(self._http_failure_message(status)) from None
            raise RuntimeError(self._safe_retry_reason(e, status)) from None
        finally:
            if response is not None:
                self._record_response_diagnostic(
                    "stream_tools", payload.get("max_tokens"), finish, usage,
                    getattr(response, "status_code", None),
                )
                try:
                    response.close()
                except Exception:
                    # Transport cleanup must not replace the call result or failure.
                    pass

    def stream_text(self, messages: list[dict], **kw) -> Iterator[str]:
        """流式产出正文片段（纯文本路径）。

        与 :meth:`chat_text` 的关系：同一个模型、同一套请求体，只多了
        ``"stream": True``，把非流式的一次性返回换成逐片段产出。
        调用方拿到的是**片段迭代器**，不是完整字符串。

        实现上刻意不写成「内部先把流攒完再 return 字符串」——那样虽然
        接口看起来一样，但流式的意义（边生成边给用户看）就全丢了。

        **on_delta 会被调用很多次**，每次一个短片段（可能只有几个字）。
        调用方若要拼接，自己维护 buffer；不要在这里做去重或缓冲。
        """
        budget = kw.pop("max_tokens", self.max_tokens)
        extra = {k: v for k, v in kw.items() if k != "max_tokens"}
        on_delta = extra.pop("on_delta", None)
        yield from self._observe_stream(
            "stream_text",
            lambda: self._stream_once(
                messages, budget, extra, on_delta, diagnostic_method="stream_text"
            ),
        )

    def _iter_sse_data(self, resp) -> Iterator[dict]:
        """把一条 SSE 响应流解析成 JSON 事件。

        三个坑，全部来自对 openrouter + reasoning 模型的实测，
        少处理一个就是「流式一开就崩」或「上屏全是乱码」：

        1. **注释帧**：reasoning 阶段端点会持续发 ``: OPENROUTER PROCESSING``。
           以 ``:`` 开头的是 SSE 注释，不是数据，必须跳过——对它做
           ``json.loads`` 会抛 JSONDecodeError。
        2. **UTF-8 解码**：**不能**用 ``iter_lines(decode_unicode=True)``。
           实测它按 latin-1 之类处理，中文全部变成乱码
           （「索引」变成 ``ç´¢å¼``）。正确做法是取原始 bytes 自己解码。
        3. **心跳与结束**：`data: [DONE]` 表示流正常结束；空行是事件分隔符。
        """
        for raw in resp.iter_lines():
            if not raw:
                continue
            if isinstance(raw, bytes):
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            else:
                line = raw
            line = line.strip()
            if not line or line.startswith(":"):
                # 坑 1 + 坑 3：注释帧（含 OPENROUTER PROCESSING）跳过
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                # 单条事件解析失败不该中断整个流（端点偶尔会发非 JSON 的噪声帧）
                continue

    def _stream_once(self, messages: list[dict], budget: int,
                     extra: dict, on_delta: Callable[[str], None] | None,
                     diagnostic_method="stream_text"
                     ) -> Iterator[str]:
        """单次流式请求。网络错误重试语义与 :meth:`_try_once` 一致。"""
        last_reason = "网络请求失败，请检查网络或服务状态。"
        started = time.monotonic()

        for attempt in range(self.max_retries):
            # 已经 yield 出去的字符数。流式中途断开时不能重试——重跑一遍
            # 会把同样的内容再推一遍，前端拼接后就是重复文本。
            emitted = 0
            self._check_cancel()
            resp = None
            finish = None
            usage = {}
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": budget,
                    "temperature": self.temperature,
                    "stream": True,
                }
                payload.update(self._tool_protocol_options())
                payload.update(self._model_request_compatibility_options())
                payload.update(extra)
                payload["max_tokens"] = self._effective_output_tokens(messages, payload)
                budget = payload["max_tokens"]

                resp = self._post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    stream=True,
                    _diagnostic_method=diagnostic_method,
                    # (connect, read) 二元组：连接超时短一些，读超时给足。
                    # reasoning 模型可能几十秒不吐字，用单值 timeout 会误判。
                    timeout=(30, self.timeout),
                )
                self._raise_for_hard_quota_response(resp, streaming=True)
                resp.raise_for_status()

                produced = False
                for event in self._iter_sse_data(resp):
                    # 逐块检查取消——**这是全项目唯一能真正打断阻塞 I/O 的地方**。
                    #
                    # 为什么要 `resp.close()`：不关连接的话，requests 的
                    # `iter_lines` 会一直阻塞在 socket 读上，即使我们不再消费，
                    # 远端仍在为这次请求生成 token 并计费。关掉响应等于
                    # 关闭 TCP 连接，对端随即收到客户端断开而停止生成。
                    #
                    # 为什么检查放在 `on_delta` **之后**、`yield` 之前：
                    # 已经推给前端的片段不能反悔（前端会留下半截正文），
                    # 而这一块是要保留的——用户看到「已生成的部分」比什么都看不到好。
                    if self.should_cancel is not None and self.should_cancel():
                        resp.close()
                        raise TaskCancelled(
                            f"流式生成中被取消（已产出 {emitted} 字符）")
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    if event.get("error"):
                        status = self._numeric_error_status(event["error"])
                        if self._hard_quota_kind(event["error"], status=status):
                            raise _DeterministicFailure(
                                self._hard_quota_failure_message()
                            )
                        if status is None:
                            raise _UnclassifiedProviderFailure()
                        if not self._retryable_http_status(status):
                            raise _DeterministicFailure(self._http_failure_message(status))
                        raise _ProviderStatusFailure(status)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    # 关键：只取 content，**绝不碰 reasoning / reasoning_details**
                    # （模块 docstring 实测 2：那是模型内部思考，不是结果）
                    piece = delta.get("content")
                    if not piece:
                        continue
                    produced = True
                    emitted += len(piece)
                    if on_delta is not None:
                        on_delta(piece)
                    yield piece

                if finish == "length":
                    raise _DeterministicFailure(
                        f"流式响应被截断（finish_reason='length'，预算 {budget}）。"
                        "已显示的片段不构成完整回答，未提交最终结果。"
                    )
                if finish != "stop":
                    raise _DeterministicFailure(
                        f"流式响应未以 stop 正常结束（finish_reason={finish!r}），"
                        "未提交最终结果。"
                    )
                if not produced:
                    raise _DeterministicFailure(
                        f"流式响应里没有任何 content 片段（预算 {budget}）。"
                        f"未收到可展示的正文，原因需要结合端点 usage 判断。"
                    )
                return

            except ModelAttemptBudgetExceeded:
                raise
            except ContextBudgetError:
                raise
            except TaskCancelled:
                raise
            except RequestDeadlineExceeded as exc:
                raise _DeterministicFailure(
                    self._deadline_failure_message(exc, streaming=True)
                ) from None
            except _DeterministicFailure:
                raise
            except Exception as e:
                status = self._failure_status(e, resp)
                if status is not None and not self._retryable_http_status(status):
                    raise _DeterministicFailure(self._http_failure_message(status)) from None
                retry_reason = self._safe_retry_reason(e, status)
                # 流式中途断开算网络错误，本可以重试。但只要已经吐出过一个字，
                # 重跑就会把同样的内容再推一遍、前端拼接后变成重复文本 ——
                # 所以「产出过内容」时直接冒泡，由上层决定怎么兜。
                if emitted:
                    raise RuntimeError(
                        f"流式响应中途断开（已产出 {emitted} 字符，重试会造成内容重复）；"
                        f"已保留已展示片段。{retry_reason}"
                    ) from None
                last_reason = retry_reason
                if self.on_retry is not None:
                    self.on_retry({
                        "attempt": attempt + 1,
                        "reason": retry_reason,
                        "elapsed": round(time.monotonic() - started, 1),
                    })
            finally:
                if resp is not None:
                    self._record_response_diagnostic(
                        diagnostic_method, budget, finish, usage,
                        getattr(resp, "status_code", None),
                    )
                    close = getattr(resp, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            # Resource cleanup must not mask a partial stream failure.
                            pass

            if attempt < self.max_retries - 1:
                time.sleep(self.backoff_base * (2 ** attempt))

        raise RuntimeError(
            f"LLM 流式调用失败（网络尝试共 {self.max_retries} 次，含首次）：{last_reason}"
        )

    def _run(self, messages: list[dict], parse, diagnostic_method="chat", **kw) -> Any:
        """三条对外路径（chat / chat_text / chat_json）共用的调用主干。

        ``parse`` 是 content 的处理器：``extract_json`` 出结构化数据，
        :meth:`_identity` 出原始字符串。**错误处理只写在这一处**——
        之前只有 chat() 一条路，重试逻辑写死在它内部；加自然语言路径时
        若另写一份，那三条实测坑（空 content / 截断 / 未知 finish_reason）
        迟早在两条路径上漂移出一条，症状是「对话偶尔返回空串」这种难查的问题。
        """
        # max_tokens 可在 kw 里覆盖
        budget = kw.pop("max_tokens", self.max_tokens)
        extra = {k: v for k, v in kw.items() if k != "max_tokens"}
        return self._observe_call(
            diagnostic_method,
            lambda: self._try_once(messages, budget, extra, parse, diagnostic_method),
        )

    def _try_once(self, messages: list[dict], budget: int,
                  extra: dict, parse=extract_json, diagnostic_method="chat") -> Any:
        """有界网络尝试；空正文最多恢复一次，截断不提交，不提高预算。"""
        last_reason = "网络请求失败，请检查网络或服务状态。"
        started = time.monotonic()

        for attempt in range(self.max_retries):
            self._check_cancel()
            resp = None
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": budget,
                    "temperature": self.temperature,
                }
                payload.update(self._tool_protocol_options())
                payload.update(self._model_request_compatibility_options())
                payload.update(extra)
                payload["max_tokens"] = self._effective_output_tokens(messages, payload)
                budget = payload["max_tokens"]

                resp = self._post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    _diagnostic_method=diagnostic_method,
                    timeout=(30, self.timeout),
                )
                resp.raise_for_status()
                data = resp.json()

                # 响应体里显式带 error 字段（OpenRouter 有时 200 但带 error）
                if isinstance(data, dict) and "error" in data:
                    self._record_response_diagnostic(
                        diagnostic_method, budget, None, data.get("usage"),
                        self._response_http_status(resp),
                    )
                    status = self._numeric_error_status(data["error"])
                    if self._hard_quota_kind(data["error"], status=status):
                        raise _DeterministicFailure(
                            self._hard_quota_failure_message()
                        )
                    if status is not None:
                        if not self._retryable_http_status(status):
                            raise _DeterministicFailure(self._http_failure_message(status))
                        raise _ProviderStatusFailure(status)
                    raise _UnclassifiedProviderFailure()

                choices = data["choices"]
                if not choices:
                    raise RuntimeError("LLM 响应 choices 为空")
                choice = choices[0]
                message = choice.get("message", {})

                # 实测 2：只读 content，绝不碰 reasoning / reasoning_details
                content = message.get("content")
                finish_reason = choice.get("finish_reason")

                usage = data.get("usage") or {}
                diagnostic = self._record_response_diagnostic(
                    diagnostic_method, budget, finish_reason, usage,
                    getattr(resp, "status_code", None),
                )
                # A capped reasoning budget can end before any user-visible content.
                # Treat provider truncation as deterministic regardless of whether
                # content is empty; never mistake it for the separate empty-body
                # recovery path or return reasoning-only output as a result.
                if finish_reason == "length":
                    raise _DeterministicFailure(
                        f"响应被截断（finish_reason='length'，预算 {budget}）。"
                        "未提交不完整结果，请缩小本次输出规模。"
                    )
                if not isinstance(content, str) or not content.strip():
                    reason = "模型返回空 content：" + json.dumps(diagnostic, ensure_ascii=False)
                    if attempt == 0 and self.max_retries > 1:
                        if self.on_retry:
                            self.on_retry({"attempt": 1, "reason": reason, "elapsed": round(time.monotonic() - started, 1)})
                        messages = [*messages, {"role": "user", "content": "上次未返回正文。请直接给出要求的简洁结果。"}]
                        continue
                    raise _DeterministicFailure(reason + "；一次恢复尝试后仍无正文，未使用推理内容代替答案。")
                if finish_reason != "stop":
                    raise _DeterministicFailure(
                        f"响应被截断（finish_reason={finish_reason!r}，预算 {budget}）。"
                        f"未提交不完整结果，请缩小本次输出规模。"
                    )

                try:
                    return parse(content)
                except ValueError as exc:
                    raise _InvalidModelOutput() from exc

            except ModelAttemptBudgetExceeded:
                raise
            except ContextBudgetError:
                raise
            except TaskCancelled:
                raise
            except RequestDeadlineExceeded as exc:
                raise _DeterministicFailure(
                    self._deadline_failure_message(exc)
                ) from None
            except _DeterministicFailure:
                raise  # 确定性失败：立刻冒泡，绝不重试
            except Exception as e:
                # 网络/HTTP 错误：重试有意义
                actual_status = self._response_http_status(resp, e)
                if actual_status is not None and actual_status >= 400:
                    self._record_response_diagnostic(
                        diagnostic_method, budget, None, {}, actual_status,
                    )
                if actual_status == 429:
                    error_payload = self._response_error_payload(resp)
                    if self._hard_quota_kind(error_payload, status=actual_status):
                        raise _DeterministicFailure(
                            self._hard_quota_failure_message(
                                actual_http_status=actual_status
                            )
                        ) from None
                status = self._failure_status(e, resp)
                if status is not None and not self._retryable_http_status(status):
                    raise _DeterministicFailure(self._http_failure_message(status)) from None
                last_reason = self._safe_retry_reason(e, status)
                if self.on_retry is not None:
                    self.on_retry({
                        "attempt": attempt + 1,
                        "reason": last_reason,
                        "elapsed": round(time.monotonic() - started, 1),
                    })

            # 指数退避：attempt 0 -> 2s, attempt 1 -> 4s（最后一次不 sleep）
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff_base * (2 ** attempt))

        raise RuntimeError(
            f"LLM 调用失败（网络尝试共 {self.max_retries} 次，含首次）：{last_reason}"
        )
