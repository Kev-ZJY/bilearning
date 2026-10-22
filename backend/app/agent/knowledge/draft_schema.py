"""Provider JSON Schema for source-indexed initial draft output."""

from __future__ import annotations

import re


DRAFT_SCHEMA_POLICY = "dots-structured-draft-v1"
_SOURCE_ID_RE = re.compile(r"v[1-9][0-9]*\Z")
_MAX_SOURCE_ID_LENGTH = 64


def _validated_evidence_ids(evidence_ids: list[str]) -> list[str]:
    if type(evidence_ids) is not list or not evidence_ids:
        raise ValueError("evidence_ids 必须是非空的当前来源ID列表")
    if any(
        not isinstance(identity, str)
        or len(identity) > _MAX_SOURCE_ID_LENGTH
        or _SOURCE_ID_RE.fullmatch(identity) is None
        for identity in evidence_ids
    ):
        raise ValueError("evidence_ids 只能包含当前来源索引ID（v1、v2…）")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("evidence_ids 不能重复")
    return list(evidence_ids)


def build_draft_response_format(evidence_ids: list[str]) -> dict:
    """Build a fresh strict JSON-schema response format for one source index.

    Only source IDs already issued by the current task appear in the quote
    binding enum. Point-level ``source_quotes`` is intentionally absent and
    rejected by the point object's closed property set; the program binds the
    accepted IDs to exact source lines after parsing.
    """
    current_ids = _validated_evidence_ids(evidence_ids)

    point_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "anchor": {"type": "string"},
            "body": {"type": "string"},
            "source_evidence_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "enum": current_ids},
            },
            "examples": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
            "figures": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "mastery": {"type": "string"},
        },
        "required": [
            "id", "title", "anchor", "body", "source_evidence_ids",
            "examples", "warnings", "figures", "mastery",
        ],
        "additionalProperties": False,
    }
    module_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "points": {
                "type": "array",
                "minItems": 1,
                "items": point_schema,
            },
        },
        "required": ["id", "title", "points"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "meta": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "positioning": {"type": "string"},
                },
                "required": ["title", "positioning"],
                "additionalProperties": False,
            },
            "modules": {
                "type": "array",
                "minItems": 1,
                "items": module_schema,
            },
            "appendix": {
                "type": "object",
                "properties": {
                    "cheatsheets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    },
                },
                "required": ["cheatsheets"],
                "additionalProperties": False,
            },
            "public_progress": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["plan", "finding", "decision"],
                    },
                    "summary": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1200,
                    },
                },
                "required": ["kind", "summary"],
                "additionalProperties": False,
            },
        },
        "required": ["meta", "modules", "appendix", "public_progress"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "video_learning_draft",
            "strict": True,
            "schema": schema,
        },
    }
