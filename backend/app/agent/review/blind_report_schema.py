"""Strict Dots response format and local decoder for isolated blind reviews.

This module only validates structured output data. The existing blind-review
validator remains responsible for semantic, citation, evidence, and coverage
gates after this decoder restores the legacy points-array representation.
"""
from __future__ import annotations

import json
from typing import Any


BLIND_REPORT_SCHEMA_POLICY = "dots-blind-report-keyed-v1"

_FACT_STATUSES = ["confirmed", "qualified", "contradicted", "unverified", "not_applicable"]
_FACT_BASES = ["web", "model_knowledge", "not_applicable"]
_RISKS = ["normal", "high"]
_REVIEW_STATUSES = ["adequate", "unnecessary", "missing", "incorrect"]
_KNOWLEDGE_STATUSES = ["knowledge", "not_knowledge", "uncertain"]
_CATEGORIES = ["fact_accuracy", "knowledge_value", "examples", "figures"]
_SEVERITIES = ["critical", "high", "medium", "low"]


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _string(*, minimum: int = 1, maximum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "minLength": minimum}
    if maximum is not None:
        schema["maxLength"] = maximum
    return schema


def _citation_schema() -> dict[str, Any]:
    properties = {
        "url": _string(),
        "quote": _string(minimum=12, maximum=2400),
        "claim": _string(minimum=4),
        "explanation": _string(),
    }
    return _object(properties, list(properties))


def _point_schema() -> dict[str, Any]:
    fact_properties = {
        "status": {"type": "string", "enum": _FACT_STATUSES},
        "basis": {"type": "string", "enum": _FACT_BASES},
        "risk": {"type": "string", "enum": _RISKS},
        "reason": _string(),
        "citations": {"type": "array", "items": _citation_schema()},
    }
    def simple_category(statuses):
        properties = {
            "status": {"type": "string", "enum": statuses},
            "reason": _string(),
        }
        return _object(properties, list(properties))

    properties = {
        "fact_accuracy": _object(fact_properties, list(fact_properties)),
        "knowledge_value": simple_category(_KNOWLEDGE_STATUSES),
        "examples": simple_category(_REVIEW_STATUSES),
        "figures": simple_category(_REVIEW_STATUSES),
    }
    return _object(properties, list(properties))


def _findings_schema(point_ids: list[str]) -> dict[str, Any]:
    item_schema = _object({
        "id": _string(),
        "point_id": {"type": "string", "enum": point_ids},
        "category": {"type": "string", "enum": _CATEGORIES},
        "severity": {"type": "string", "enum": _SEVERITIES},
        "issue": _string(),
        "suggestion": _string(),
        "citations": {"type": "array", "items": _citation_schema()},
    }, ["id", "point_id", "category", "severity", "issue", "suggestion", "citations"])
    return {"type": "array", "items": item_schema}


def _validate_point_ids(point_ids: Any) -> list[str]:
    if not isinstance(point_ids, (list, tuple)):
        raise ValueError("盲审opaque point IDs必须是字符串列表")
    values = list(point_ids)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("盲审opaque point ID不能为空或非字符串")
    if len(values) != len(set(values)):
        raise ValueError("盲审opaque point IDs不能重复")
    if not values:
        raise ValueError("盲审必须至少有一个opaque point ID")
    return values


def build_blind_report_response_format(
    point_ids: list[str] | tuple[str, ...], *, allow_supplemental_reads: bool = True,
) -> dict[str, Any]:
    """Build the Dots strict response format for a report or read-only action plan."""
    ids = _validate_point_ids(point_ids)
    if type(allow_supplemental_reads) is not bool:
        raise ValueError("allow_supplemental_reads必须为bool")
    points_schema = _object({point_id: _point_schema() for point_id in ids}, ids)
    report_schema = _object({
        "points": points_schema,
        "findings": _findings_schema(ids),
    }, ["points", "findings"])
    result_schema = report_schema
    if allow_supplemental_reads:
        from app.agent.review.blind_evidence_actions import build_evidence_action_response_format

        action_schema = build_evidence_action_response_format(
            [], read_only=True, allow_read_window=True,
        )["json_schema"]["schema"]
        result_schema = {"anyOf": [report_schema, action_schema]}
    root_schema = _object({
        "result": result_schema,
    }, ["result"])
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dots_blind_report_or_read_actions",
            "strict": True,
            "schema": root_schema,
        },
    }


def _parse_payload(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"盲审结构化JSON包含重复键：{key}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Dots盲审结构化响应不是有效JSON") from exc


def _expect_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label}字段必须精确为{sorted(keys)!r}")
    return value


def _validate_string(value: Any, label: str, *, minimum: int = 1, maximum: int | None = None) -> None:
    if not isinstance(value, str) or len(value) < minimum or not value.strip():
        raise ValueError(f"{label}必须为非空字符串且至少{minimum}字")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{label}超过{maximum}字上限")


def _validate_citations(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{label}必须为数组")
    for index, citation in enumerate(value):
        prefix = f"{label}[{index}]"
        citation = _expect_object(citation, {"url", "quote", "claim", "explanation"}, prefix)
        _validate_string(citation["url"], prefix + ".url")
        _validate_string(citation["quote"], prefix + ".quote", minimum=12, maximum=2400)
        _validate_string(citation["claim"], prefix + ".claim", minimum=4)
        _validate_string(citation["explanation"], prefix + ".explanation")


def _validate_point(value: Any, opaque_id: str) -> dict[str, Any]:
    row = _expect_object(value, {"fact_accuracy", "knowledge_value", "examples", "figures"},
                         f"points[{opaque_id}]")
    fact = _expect_object(row["fact_accuracy"], {"status", "basis", "risk", "reason", "citations"},
                          f"points[{opaque_id}].fact_accuracy")
    if fact["status"] not in _FACT_STATUSES:
        raise ValueError(f"points[{opaque_id}].fact_accuracy.status无效")
    if fact["basis"] not in _FACT_BASES:
        raise ValueError(f"points[{opaque_id}].fact_accuracy.basis无效")
    if fact["risk"] not in _RISKS:
        raise ValueError(f"points[{opaque_id}].fact_accuracy.risk无效")
    _validate_string(fact["reason"], f"points[{opaque_id}].fact_accuracy.reason")
    _validate_citations(fact["citations"], f"points[{opaque_id}].fact_accuracy.citations")
    for category in ("knowledge_value", "examples", "figures"):
        check = _expect_object(row[category], {"status", "reason"}, f"points[{opaque_id}].{category}")
        allowed = _KNOWLEDGE_STATUSES if category == "knowledge_value" else _REVIEW_STATUSES
        if check["status"] not in allowed:
            raise ValueError(f"points[{opaque_id}].{category}.status无效")
        _validate_string(check["reason"], f"points[{opaque_id}].{category}.reason")
    return row


def _decode_report(result: Any, point_ids: list[str]) -> dict[str, Any]:
    report = _expect_object(result, {"points", "findings"}, "盲审报告")
    keyed_points = report["points"]
    expected = set(point_ids)
    if not isinstance(keyed_points, dict):
        raise ValueError("盲审points必须为对象")
    if any(not isinstance(key, str) for key in keyed_points):
        raise ValueError("盲审points键必须是opaque point ID字符串")
    actual = set(keyed_points)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"盲审points必须精确覆盖当前opaque IDs：缺失={missing!r}；多余={extra!r}")

    rows = []
    for opaque_id in point_ids:
        value = _validate_point(keyed_points[opaque_id], opaque_id)
        rows.append({"point_id": opaque_id, **value})

    findings = report["findings"]
    if not isinstance(findings, list):
        raise ValueError("盲审findings必须为数组")
    if not point_ids and findings:
        raise ValueError("没有候选opaque point IDs时禁止输出findings")
    ids = set()
    normalized_findings = []
    for index, raw in enumerate(findings):
        prefix = f"findings[{index}]"
        finding = _expect_object(raw, {
            "id", "point_id", "category", "severity", "issue", "suggestion", "citations",
        }, prefix)
        _validate_string(finding["id"], prefix + ".id")
        if finding["id"] in ids:
            raise ValueError(f"{prefix}.id重复")
        ids.add(finding["id"])
        if not isinstance(finding["point_id"], str) or finding["point_id"] not in expected:
            raise ValueError(f"{prefix}.point_id不属于当前opaque IDs")
        if finding["category"] not in _CATEGORIES:
            raise ValueError(f"{prefix}.category无效")
        if finding["severity"] not in _SEVERITIES:
            raise ValueError(f"{prefix}.severity无效")
        _validate_string(finding["issue"], prefix + ".issue")
        _validate_string(finding["suggestion"], prefix + ".suggestion")
        _validate_citations(finding["citations"], prefix + ".citations")
        normalized_findings.append(dict(finding))
    return {"points": rows, "findings": normalized_findings}


def decode_blind_report_turn(
    payload: Any,
    point_ids: list[str] | tuple[str, ...],
    *,
    allow_supplemental_reads: bool = True,
) -> dict[str, Any]:
    """Decode a strict envelope to a legacy report or validated actions.

    The return value is the legacy ``{"points": [...], "findings": [...]}``
    report, or the existing ``{"actions": [...]}`` envelope. Reports are structurally
    strict here; the caller must still pass them through
    ``_validate_blind_review`` for semantic and evidence gates.
    """
    ids = _validate_point_ids(point_ids)
    if type(allow_supplemental_reads) is not bool:
        raise ValueError("allow_supplemental_reads必须为bool")
    envelope = _expect_object(_parse_payload(payload), {"result"}, "Dots盲审响应")
    result = envelope["result"]
    if isinstance(result, dict) and set(result) == {"actions"}:
        if not allow_supplemental_reads:
            raise ValueError("当前盲审阶段禁止补读actions")
        from app.agent.review.blind_evidence_actions import validate_evidence_action_plan

        try:
            actions = validate_evidence_action_plan(
                result, pending_figure_ids=[], read_only=True, allow_read_window=True,
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("盲审补读actions结构无效") from exc
        fingerprints = [(row["type"], row["url"], row["start"], row["find"]) for row in actions]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("盲审补读actions不能重复")
        return {"actions": actions}
    return _decode_report(result, ids)
