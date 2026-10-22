"""Verified model context and output limits used by offline budget planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


INPUT_SAFETY_MARGIN_TOKENS = 4_096
DEFAULT_INPUT_BUDGET_TOKENS = 140_000


@dataclass(frozen=True)
class ModelLimits:
    context_window: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if type(self.context_window) is not int or self.context_window <= 0:
            raise ValueError("context_window must be a positive integer")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")


_DEEPSEEK_BASE_URLS = frozenset({
    "https://api.deepseek.com",
    "https://api.deepseek.com/v1",
})
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _normalized_base_url(base_url: Any) -> str | None:
    if not isinstance(base_url, str):
        return None
    value = base_url.strip().rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[:-len("/chat/completions")].rstrip("/")
    return value


def resolve_model_limits(base_url: str, model: str) -> ModelLimits | None:
    """Return only the catalogued limits verified for exact endpoint/model pairs."""
    endpoint = _normalized_base_url(base_url)
    if not isinstance(model, str):
        return None
    if endpoint in _DEEPSEEK_BASE_URLS and model == "deepseek-flash":
        return ModelLimits(context_window=1_048_576, max_output_tokens=393_216)
    if endpoint == _OPENROUTER_BASE_URL and model == "dots-studio/dots-3-note-preview:free":
        return ModelLimits(context_window=512_000, max_output_tokens=460_800)
    return None


def input_budget_for_llm(
    llm: Any,
    reserved_output: int | None = None,
    fallback: int = DEFAULT_INPUT_BUDGET_TOKENS,
) -> int:
    """Get a model-aware input budget, or the conservative fallback for old fakes."""
    if type(fallback) is not int or fallback < 0:
        raise ValueError("fallback must be a non-negative integer")

    # Resolve on the class so dynamic Mock.__getattr__ does not look like a real
    # implementation. Existing injected clients without the new method keep the
    # established fallback unchanged.
    method = getattr(type(llm), "input_budget_tokens", None)
    if not callable(method):
        return fallback
    value = method(llm, reserved_output=reserved_output)
    if type(value) is not int or value < 0:
        raise ValueError("input_budget_tokens must return a non-negative integer")
    return value
