"""卡片与助手共享的整理契约。省略选项时沿用用户 intent。

``historical_context`` is an internal, program-owned snapshot carried by chat
jobs to the author prompt. It is deliberately absent from the public tool
schema: model-provided values are replaced by ``ToolRegistry`` before this
contract is constructed.
"""

import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class HistoricalContextMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: int = Field(gt=0)
    role: Literal["user", "assistant", "tool"]
    content: str


class HistoricalContextTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_message_id: int = Field(gt=0)
    missing_explicit_relation_message_ids: list[int] = Field(default_factory=list)
    messages: list[HistoricalContextMessage] = Field(default_factory=list)


class HistoricalContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    summary_source_message_ids: list[int] = Field(default_factory=list)
    recent_turns: list[HistoricalContextTurn] = Field(default_factory=list)


class PublicProcessingOptions(BaseModel):
    document_mode: Literal["recitation", "textbook"] | None = None
    integration: Literal["merge", "replace"] = "merge"
    example_strategy: Literal["as_needed", "course", "analogy"] | None = None
    medium_timeout_seconds: float | None = Field(default=None, gt=0)
    hard_subtitles: bool = False
    subtitle_crop_bottom: float = Field(default=0.3, gt=0, le=1)


class ProcessingOptions(PublicProcessingOptions):
    """Internal queue snapshot; public API options intentionally omit history."""

    historical_context: HistoricalContext | None = None


_QUOTED_OPTION_TEXT = re.compile(
    r"“[^”]*”|「[^」]*」|『[^』]*』|\"[^\"]*\"|'[^']*'|`[^`]*`"
)
_NEGATED_OR_EXAMPLE_PREFIX = re.compile(
    r"(?:不要|不需要|不选|不是|别|禁止|不能|不可|无需|避免|例如|比如|举例|"
    r"假设|假如|如果|引用|don't|do not|not|for example|e\.g\.)"
    r"[^，,。；;！？!?]{0,24}$",
    re.IGNORECASE,
)
_FORBIDDEN_OPTION_PATTERNS = {
    "document_mode": (
        (re.compile(
            r"(?:不要|别|不选|不采用|不使用|不接受|禁止|不能|不可|无需|避免)"
            r"\s*(?:再)?\s*(?:用|采用|使用|选择)?\s*(textbook|recitation)"
            r"(?:\s*(?:体例|模式|格式))?", re.I,
        ), {"textbook": "textbook", "recitation": "recitation"}),
    ),
    "integration": (
        (re.compile(
            r"(?:不要|别|不选择|不使用|禁止|不能|不可|无需|避免)"
            r"\s*(?:选择|使用|设为)?\s*(?:integration\s*[:=]\s*)?(merge|replace)\b", re.I,
        ), {"merge": "merge", "replace": "replace"}),
        (re.compile(
            r"(?:不要|别|不选择|不使用|禁止|不能|不可|无需|避免)"
            r"\s*(?:选择|使用|设为)?\s*(合并|融合|替换)(?:进当前文档)?"
        ), {"合并": "merge", "融合": "merge", "替换": "replace"}),
    ),
}
_OPTION_PATTERNS = {
    "document_mode": (
        (re.compile(r"\bdocument_mode\s*[:=]\s*(textbook|recitation)\b", re.I),
         {"textbook": "textbook", "recitation": "recitation"}),
        (re.compile(
            r"(?:本次|本轮|当前|这次)?\s*(?:明确)?\s*(?:用|采用|使用|选择)\s*"
            r"(textbook|recitation)(?:\s*(?:体例|模式|格式|style))?", re.I,
        ), {"textbook": "textbook", "recitation": "recitation"}),
        (re.compile(r"(?:体系化教材|教材体例)"), {"体系化教材": "textbook", "教材体例": "textbook"}),
        (re.compile(r"(?:背诵清单|记忆清单)"), {"背诵清单": "recitation", "记忆清单": "recitation"}),
    ),
    "integration": (
        (re.compile(r"\bintegration\s*[:=]\s*(merge|replace)\b", re.I),
         {"merge": "merge", "replace": "replace"}),
        (re.compile(
            r"(?:本次|本轮|当前|这次)?\s*(?:明确)?\s*(?:选择|使用|设为)\s*"
            r"(merge|replace)\b", re.I,
        ), {"merge": "merge", "replace": "replace"}),
        (re.compile(r"(?:明确)?(?:选择)?(?:合并|融合)进当前文档"), {"合并进当前文档": "merge", "融合进当前文档": "merge"}),
        (re.compile(r"(?:明确)?(?:选择)?(?:替换|重写替换)当前文档"), {"替换当前文档": "replace", "重写替换当前文档": "replace"}),
    ),
}


def _current_explicit_option_values(field: str, current_request: str) -> set[str]:
    """Recognize a deliberately small set of unquoted positive selectors."""
    text = _QUOTED_OPTION_TEXT.sub(" ", current_request or "")
    # URLs and query parameters are references, not structured user choices.
    text = re.sub(r"https?://\S+", " ", text)
    values: set[str] = set()
    boundaries = "，,。；;！？!?"
    for pattern, mapping in _OPTION_PATTERNS[field]:
        for match in pattern.finditer(text):
            prefix_start = max((text.rfind(mark, 0, match.start()) for mark in boundaries), default=-1)
            prefix = text[prefix_start + 1:match.start()]
            if _NEGATED_OR_EXAMPLE_PREFIX.search(prefix):
                continue
            token = match.group(1) if match.lastindex else match.group(0)
            normalized = next((value for phrase, value in mapping.items()
                               if token.lower().endswith(phrase.lower())), None)
            if normalized is not None:
                values.add(normalized)
    return values


def _current_forbidden_option_values(field: str, current_request: str) -> set[str]:
    """Recognize narrow, direct exclusions so a model proposal cannot undo them."""
    text = _QUOTED_OPTION_TEXT.sub(" ", current_request or "")
    text = re.sub(r"https?://\S+", " ", text)
    values: set[str] = set()
    boundaries = "，,。；;！？!?"
    example_prefix = re.compile(r"(?:例如|比如|举例|假设|假如|如果|引用)\s*[^，,。；;！？!?]{0,24}$")
    for pattern, mapping in _FORBIDDEN_OPTION_PATTERNS[field]:
        for match in pattern.finditer(text):
            prefix_start = max((text.rfind(mark, 0, match.start()) for mark in boundaries), default=-1)
            prefix = text[prefix_start + 1:match.start()]
            if example_prefix.search(prefix):
                continue
            token = match.group(1)
            normalized = next((value for phrase, value in mapping.items()
                               if token.lower().endswith(phrase.lower())), None)
            if normalized is not None:
                values.add(normalized)
    return values


def resolve_current_request_options(proposed: dict, current_request: str) -> dict:
    """Apply only clear current-turn enum selectors over model-proposed options.

    This is intentionally not a general natural-language parser. Quotes,
    examples and negated selectors are ignored. Conflicting clear selectors
    fail closed. ``replace`` is especially sensitive: a model may not select
    it based solely on historical style or a historical instruction.
    """
    if not isinstance(proposed, dict):
        raise ValueError("整理 options 必须是对象")
    resolved = dict(proposed)
    for field in ("document_mode", "integration"):
        values = _current_explicit_option_values(field, current_request)
        if len(values) > 1:
            raise ValueError(f"本轮 {field} 选项互相冲突，请明确选择一个值")
        if values:
            resolved[field] = values.pop()
        if resolved.get(field) in _current_forbidden_option_values(field, current_request):
            raise ValueError(f"模型提出的 {field} 与本轮明确约束冲突，未提交视频整理")
    if (resolved.get("integration") == "replace"
            and _current_explicit_option_values("integration", current_request) != {"replace"}):
        raise ValueError("integration=replace 需要本轮当前请求明确选择，不能仅依据历史或模型提案")
    return resolved
