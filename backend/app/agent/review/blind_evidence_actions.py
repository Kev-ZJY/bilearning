"""Strict JSON action plans for the bounded Dots blind-review evidence phase.

This module only describes and validates action data. Existing review code is
responsible for executing approved actions through its normal guarded tools.
"""
from __future__ import annotations

from typing import Any

from app.agent.runtime.llm import LLMClient as _RUNTIME_LLM_CLIENT


BLIND_EVIDENCE_ACTIONS_POLICY = "dots-blind-evidence-actions-v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DOTS_REVIEW_MODEL = "dots-studio/dots-3-note-preview:free"
MAX_ACTIONS_PER_PLAN = 8


def supports_dots_blind_evidence_actions(client: Any) -> bool:
    """Return true only for the loaded client class and verified Dots profile."""
    if not isinstance(client, _RUNTIME_LLM_CLIENT):
        return False
    return (
        getattr(client, "base_url", None) == OPENROUTER_BASE_URL
        and getattr(client, "model", None) == DOTS_REVIEW_MODEL
    )


def _validated_values(values: Any, label: str) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label}必须是字符串列表")
    result = list(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{label}包含空值或非字符串")
    if len(result) != len(set(result)):
        raise ValueError(f"{label}不能重复")
    return result


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _action_schema(
    pending_figure_ids: list[str], required_urls: list[str], *, search_only: bool, read_only: bool,
    allow_read_window: bool,
) -> dict[str, Any]:
    if pending_figure_ids:
        figure_action = _object_schema(
            {
                "type": {"type": "string", "const": "view_candidate_figure"},
                "figure_id": {"type": "string", "enum": pending_figure_ids},
            },
            ["type", "figure_id"],
        )
        count = min(MAX_ACTIONS_PER_PLAN, len(pending_figure_ids))
        actions_schema = {
            "type": "array",
            "items": figure_action,
            "minItems": count,
            "maxItems": count,
            "uniqueItems": True,
        }
    else:
        search_action = _object_schema(
            {
                "type": {"type": "string", "const": "search_web"},
                "query": {"type": "string", "minLength": 1},
            },
            ["type", "query"],
        )
        url_schema: dict[str, Any] = {"type": "string", "minLength": 1}
        if required_urls:
            url_schema["enum"] = required_urls
        read_action = _object_schema(
            {
                "type": {"type": "string", "const": "read_web"},
                "url": url_schema,
                "start": ({"type": "integer", "minimum": 0} if allow_read_window
                          else {"type": "integer", "const": 0}),
                "find": ({"anyOf": [{"type": "null"}, {"type": "string", "minLength": 1, "maxLength": 200}]}
                         if allow_read_window else {"type": "null", "const": None}),
            },
            ["type", "url", "start", "find"],
        )
        if search_only:
            action_item_schema = search_action
        else:
            action_item_schema = read_action if required_urls or read_only else {"anyOf": [search_action, read_action]}
        actions_schema = {
            "type": "array",
            "items": action_item_schema,
            "minItems": 1,
            "maxItems": 1 if search_only else MAX_ACTIONS_PER_PLAN,
        }
    return _object_schema({"actions": actions_schema}, ["actions"])


def build_evidence_action_response_format(
    pending_figure_ids: list[str], required_urls: list[str] | tuple[str, ...] = (),
    *, search_only: bool = False, read_only: bool = False, allow_read_window: bool = False,
) -> dict[str, Any]:
    """Build the strict OpenAI response-format wrapper for the current phase."""
    figures = _validated_values(pending_figure_ids, "待查看配图ID")
    urls = _validated_values(required_urls, "待读取URL")
    if type(search_only) is not bool:
        raise ValueError("search_only必须为bool")
    if type(read_only) is not bool:
        raise ValueError("read_only必须为bool")
    if read_only and (search_only or figures):
        raise ValueError("read_only不能与搜索恢复或配图阶段同时启用")
    if figures and urls:
        raise ValueError("配图查看与网页读取必须分阶段请求")
    if search_only and (figures or urls):
        raise ValueError("search_only不能与配图或待核验URL同时启用")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "blind_evidence_actions",
            "strict": True,
            "schema": _action_schema(figures, urls, search_only=search_only, read_only=read_only,
                                     allow_read_window=allow_read_window),
        },
    }


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"动作{label}必须是非空字符串")
    return value


def validate_evidence_action_plan(
    payload: Any,
    *,
    pending_figure_ids: list[str],
    required_urls: list[str] | tuple[str, ...] = (),
    search_only: bool = False, read_only: bool = False, allow_read_window: bool = False,
) -> list[dict[str, Any]]:
    """Validate a model JSON plan and return its action list without execution."""
    figures = _validated_values(pending_figure_ids, "待查看配图ID")
    urls = _validated_values(required_urls, "待读取URL")
    if type(search_only) is not bool:
        raise ValueError("search_only必须为bool")
    if type(read_only) is not bool:
        raise ValueError("read_only必须为bool")
    if read_only and (search_only or figures):
        raise ValueError("read_only不能与搜索恢复或配图阶段同时启用")
    if figures and urls:
        raise ValueError("配图查看与网页读取必须分阶段请求")
    if search_only and (figures or urls):
        raise ValueError("search_only不能与配图或待核验URL同时启用")
    if not isinstance(payload, dict) or set(payload) != {"actions"}:
        raise ValueError("盲审取证计划必须只包含actions字段")
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("盲审取证计划必须包含至少一个动作")
    if len(actions) > MAX_ACTIONS_PER_PLAN:
        raise ValueError("盲审取证动作超过单批上限")
    if search_only and len(actions) != 1:
        raise ValueError("search_only恢复阶段只允许一次search_web")

    validated: list[dict[str, Any]] = []
    if figures:
        expected_count = min(MAX_ACTIONS_PER_PLAN, len(figures))
        if len(actions) != expected_count:
            raise ValueError("配图动作必须覆盖当前批次规定数量")
        seen: set[str] = set()
        for action in actions:
            if not isinstance(action, dict) or set(action) != {"type", "figure_id"}:
                raise ValueError("配图动作字段无效")
            if action.get("type") != "view_candidate_figure":
                raise ValueError("当前阶段只允许查看候选知识配图")
            figure_id = _nonempty_string(action.get("figure_id"), "figure_id")
            if figure_id not in figures:
                raise ValueError("配图动作引用了非当前候选ID")
            if figure_id in seen:
                raise ValueError("配图动作ID不能重复")
            seen.add(figure_id)
            validated.append({"type": "view_candidate_figure", "figure_id": figure_id})
        return validated

    allowed_keys = {
        "search_web": {"type", "query"},
        "read_web": {"type", "url", "start", "find"},
    }
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("网页取证动作必须是对象")
        action_type = action.get("type")
        if action_type not in allowed_keys:
            raise ValueError("当前阶段只允许search_web或read_web")
        if search_only and action_type != "search_web":
            raise ValueError("search_only恢复阶段只允许search_web")
        if read_only and action_type != "read_web":
            raise ValueError("搜索发现后必须通过read_web读取正文，不能继续search_web")
        if urls and action_type != "read_web":
            raise ValueError("当前待核验URL阶段只允许read_web，不允许search_web")
        keys = set(action)
        if action_type == "search_web":
            if keys != allowed_keys[action_type]:
                raise ValueError("search_web动作字段无效")
            validated.append({
                "type": "search_web",
                "query": _nonempty_string(action.get("query"), "query"),
            })
            continue

        if keys != allowed_keys[action_type]:
            raise ValueError("read_web动作字段无效")
        url = _nonempty_string(action.get("url"), "url")
        if urls and url not in urls:
            raise ValueError("read_web动作URL不在当前待核验范围")
        if type(action.get("start")) is not int or action["start"] < 0:
            raise ValueError("read_web start必须为非负整数")
        if not allow_read_window and action["start"] != 0:
            raise ValueError("blind evidence read_web必须从start=0读取正文首段")
        if action.get("find") is not None and (not allow_read_window or
                not isinstance(action["find"], str) or not 1 <= len(action["find"]) <= 200):
            raise ValueError("blind evidence read_web必须使用find=null读取未过滤正文")
        validated.append({
            "type": "read_web", "url": url,
            "start": action["start"], "find": action["find"],
        })
    return validated
