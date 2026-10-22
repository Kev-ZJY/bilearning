"""Validated, bounded model-authored progress summaries.

The tool only records public status text. It never executes a business or
evidence operation, and observer failures are deliberately isolated from the
model/tool loop that produced the summary.
"""
from __future__ import annotations

import copy
import hashlib
import re
from collections import defaultdict
from typing import Any, Callable

MAX_PROGRESS_SUMMARY_CHARS = 1200
MAX_PROGRESS_CALLS_PER_ROUND = 2
MAX_PROGRESS_CALLS_PER_LOOP = 16

REPORT_PROGRESS_TOOL = {
    "type": "function",
    "function": {
        "name": "report_progress",
        "description": (
            "报告一条简短、面向用户的公开工作说明。只写实际准备做的检查、"
            "已发现的情况或已作出的工作决定；不要写私密推理、隐藏思维链、"
            "内部逐步推导或模型/token统计。此工具只记录文字，不执行任务或取证。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["plan", "finding", "decision"],
                },
                "summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PROGRESS_SUMMARY_CHARS,
                    "description": "简短的公开工作说明，最多1200个字符。",
                },
            },
            "required": ["kind", "summary"],
            "additionalProperties": False,
        },
    },
}

_ALLOWED_KINDS = frozenset({"plan", "finding", "decision"})
_ALLOWED_ACTORS = frozenset({"orchestrator", "author", "reviewer"})
_ALLOWED_EXECUTIONS = frozenset({"workflow", "isolated_context", "child_task"})


def _segment_component(value: Any) -> str:
    text = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.") or "context"
    if len(cleaned) > 64:
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:10]
        cleaned = f"{cleaned[:53]}-{digest}"
    return cleaned


def validate_progress_arguments(args: Any) -> dict[str, str]:
    """Validate and return the exact text the model supplied."""
    if not isinstance(args, dict) or set(args) != {"kind", "summary"}:
        raise ValueError("report_progress 只接受 kind 和 summary 两个字段")
    kind = args.get("kind")
    if not isinstance(kind, str) or kind not in _ALLOWED_KINDS:
        raise ValueError("kind 必须是 plan、finding 或 decision")
    summary = args.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary 必须是非空字符串")
    if len(summary) > MAX_PROGRESS_SUMMARY_CHARS:
        raise ValueError(f"summary 不能超过 {MAX_PROGRESS_SUMMARY_CHARS} 个字符")
    if any(ord(char) < 32 and char not in "\r\n\t" for char in summary):
        raise ValueError("summary 不能包含控制字符")
    return {"kind": kind, "summary": summary}


class PublicProgressReporter:
    """Apply independent progress limits and notify an observational callback."""

    def __init__(self, on_event: Callable[[dict], None] | None = None):
        self._on_event = on_event if callable(on_event) else lambda _event: None
        self._attempt_count = 0
        self._accepted_count = 0
        self._round_attempts: dict[int, int] = defaultdict(int)

    @property
    def progress_count(self) -> int:
        """Number of valid progress summaries accepted by this loop."""
        return self._accepted_count

    def report(
        self,
        args: Any,
        *,
        round: int,
        actor: str,
        agent_id: str | None = None,
        stage: str | None = None,
        cycle: int | None = None,
        execution: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        if type(round) is not int or round < 1:
            return {"ok": False, "accepted": False, "error": "round 必须是正整数"}

        self._attempt_count += 1
        self._round_attempts[round] += 1
        if self._attempt_count > MAX_PROGRESS_CALLS_PER_LOOP:
            return {
                "ok": False, "accepted": False,
                "error": "公开工作说明已达到本轮任务上限",
            }
        if self._round_attempts[round] > MAX_PROGRESS_CALLS_PER_ROUND:
            return {
                "ok": False, "accepted": False,
                "error": "每轮最多接受两条公开工作说明",
            }

        try:
            clean = validate_progress_arguments(args)
        except ValueError as exc:
            return {"ok": False, "accepted": False, "error": str(exc)}

        if actor not in _ALLOWED_ACTORS:
            return {"ok": False, "accepted": False, "error": "actor 无效"}
        if agent_id is not None and (not isinstance(agent_id, str) or not agent_id.strip()):
            return {"ok": False, "accepted": False, "error": "agent_id 无效"}
        if cycle is not None and (type(cycle) is not int or cycle < 1):
            return {"ok": False, "accepted": False, "error": "cycle 无效"}
        if execution is not None and execution not in _ALLOWED_EXECUTIONS:
            return {"ok": False, "accepted": False, "error": "execution 无效"}

        cycle_part = f"c{cycle}" if cycle is not None else "c0"
        segment_id = ":".join((
            "public",
            _segment_component(stage or "task"),
            _segment_component(actor),
            _segment_component(agent_id or actor),
            cycle_part,
            _segment_component(scope or "loop"),
            f"r{round}",
            f"n{self._round_attempts[round]}",
        ))
        event: dict[str, Any] = {
            "event": "public_progress",
            "segment_id": segment_id,
            "kind": clean["kind"],
            "text": clean["summary"],
            "actor": actor,
            "round": round,
        }
        if agent_id is not None:
            event["agent_id"] = agent_id
        if stage is not None:
            event["stage"] = stage
        if cycle is not None:
            event["cycle"] = cycle
        if execution is not None:
            event["execution"] = execution

        self._accepted_count += 1
        try:
            self._on_event(copy.deepcopy(event))
        except Exception:
            # Public activity is best-effort. It cannot change the model's
            # tool result, cancellation, business work, or evidence budgets.
            pass
        return {"ok": True, "accepted": True}
