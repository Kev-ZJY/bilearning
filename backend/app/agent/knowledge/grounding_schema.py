"""Strict provider schemas for source-grounding review responses."""

from __future__ import annotations

from typing import Any


GROUNDING_SCHEMA_POLICY = "dots-grounding-evidence-v1"
COMPLETENESS_SCHEMA_POLICY = "dots-completeness-keyed-evidence-v1"
GROUNDING_EVIDENCE_REFERENCE_POLICY = "minimum-sufficient-evidence-refs-v1"
MAX_GROUNDING_EVIDENCE_IDS_PER_ITEM = 32


def _validated_ids(values: Any, label: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label}必须是字符串列表")
    result = list(values)
    if (not allow_empty and not result) or any(
        not isinstance(value, str) or not value.strip() for value in result
    ):
        raise ValueError(f"{label}包含空值或非字符串")
    if len(result) != len(set(result)):
        raise ValueError(f"{label}不能重复")
    return result


def build_grounding_response_format(
    item_ids: list[str], evidence_ids: list[str],
) -> dict[str, Any]:
    """Build a batch-local schema for the current items and evidence rows.

    An empty evidence enum is useful when the reviewer marks every item
    unsupported. Supported rows still require non-empty, current evidence IDs
    in the existing grounding validator.
    """
    current_items = _validated_ids(item_ids, "item_ids", allow_empty=False)
    current_evidence = _validated_ids(evidence_ids, "evidence_ids", allow_empty=True)

    evidence_array = {
        "type": "array",
        "uniqueItems": True,
    }
    if current_evidence:
        evidence_array["maxItems"] = MAX_GROUNDING_EVIDENCE_IDS_PER_ITEM
        evidence_array["items"] = {"type": "string", "enum": current_evidence}
    else:
        # JSON Schema enum must be non-empty; an empty array is the only
        # permitted response when this source block issued no evidence IDs.
        evidence_array["maxItems"] = 0

    review_variants = []
    for item_id in current_items:
        if item_id == "appendix":
            # The parser treats the appendix as a document-level object, so its
            # replacement has no point ID to preserve.
            replacement_schema = {
                "anyOf": [
                    {"type": "null"},
                    {"type": "object", "additionalProperties": True},
                ],
            }
        else:
            replacement_schema = {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {"id": {"type": "string", "enum": [item_id]}},
                        "required": ["id"],
                        "additionalProperties": True,
                    },
                ],
            }
        review_variants.append({
            "type": "object",
            "properties": {
                "id": {"type": "string", "enum": [item_id]},
                "supported": {"type": "boolean"},
                "evidence_ids": evidence_array,
                "replacement": replacement_schema,
            },
            "required": ["id", "supported", "evidence_ids", "replacement"],
            "additionalProperties": False,
        })
    schema = {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": len(current_items),
                "maxItems": len(current_items),
                "items": {"anyOf": review_variants},
            },
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "video_grounding_reviews",
            "strict": True,
            "schema": schema,
        },
    }


def build_completeness_response_format(module_ids, evidence_ids):
    """Require an explicit model disposition for every batch-local source row."""
    modules = _validated_ids(module_ids, "module_ids", allow_empty=False)
    evidence = _validated_ids(evidence_ids, "evidence_ids", allow_empty=False)
    rows = {
        key: {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["covered", "added", "ignored"]},
                "point_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["action", "point_ids", "reason"],
            "additionalProperties": False,
        }
        for key in evidence
    }
    schema = {
        "type": "object",
        "properties": {
            "project_title": {"type": "string", "minLength": 1, "maxLength": 48},
            "coverage": {"type": "object", "properties": rows, "required": evidence,
                         "additionalProperties": False},
            "additions": {
                "type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "module_id": {"type": "string", "enum": modules},
                        "point": {"type": "object", "additionalProperties": True},
                        "evidence_ids": {"type": "array", "minItems": 1,
                                         "items": {"type": "string", "enum": evidence}},
                    },
                    "required": ["module_id", "point", "evidence_ids"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["project_title", "coverage", "additions"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {
        "name": "video_completeness_rows", "strict": True, "schema": schema,
    }}
