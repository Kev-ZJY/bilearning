"""Bounded, hash-bound review/revision orchestration.

This module owns cycle boundaries and durable workflow receipts. Each call to
``review_and_revise`` creates a new reviewer client, messages and web research
state; only the current candidate and the original evidence enter that cycle.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.review.independent_review import IndependentReviewBlocked, document_hash, review_and_revise
from app.agent.review.blind_review import validate_binding
from app.core.cancel import TaskCancelled

WORKFLOW_SCHEMA_VERSION = 1
MAX_SEMANTIC_REVISIONS = 2
_NON_SEMANTIC_META_KEYS = {
    "created_at", "updated_at", "modified_at", "last_modified_at", "reviewed_at",
    "independent_review", "reviews", "point_id_map", "final_id_map",
}
_NON_SEMANTIC_POINT_KEYS = {"review_point_id", "review_video_id", "review_task_id"}


class ReviewWorkflowBlocked(ValueError):
    """A required review/coverage gate did not pass within its fixed budget."""

    def __init__(self, message: str, kind: str = "blocked"):
        super().__init__(message)
        self.kind = kind


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_content_hash(doc: dict[str, Any]) -> str:
    """Hash knowledge content while excluding only known audit/display metadata.

    IDs, module structure, examples, warnings, figures (including image bytes),
    captions and appendices remain part of the semantic hash.
    """
    clean = copy.deepcopy(doc)
    meta = clean.get("meta")
    if isinstance(meta, dict):
        for key in _NON_SEMANTIC_META_KEYS:
            meta.pop(key, None)
    modules = clean.get("modules")
    if isinstance(modules, list):
        for module in modules:
            if not isinstance(module, dict) or not isinstance(module.get("points"), list):
                continue
            for point in module["points"]:
                if isinstance(point, dict):
                    for key in _NON_SEMANTIC_POINT_KEYS:
                        point.pop(key, None)
    return _json_hash(clean)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str), encoding="utf-8")
    pending.replace(path)


def _failure(exc: BaseException, kind: str | None = None) -> dict[str, str]:
    return {
        "kind": kind or (exc.kind if isinstance(exc, ReviewWorkflowBlocked) else "cancelled" if isinstance(exc, TaskCancelled) else "exception"),
        "type": type(exc).__name__,
        "message": str(exc)[:2000],
    }


def _coverage_errors(receipt: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return ["coverage receipt must be an object"]
    status = receipt.get("status")
    if status != "complete":
        errors.append(f"coverage status must be complete (got {status!r})")
    if receipt.get("complete") is not True:
        errors.append("complete must be true")
    for key in ("passed", "ok", "complete", "coverage_complete"):
        if key in receipt and receipt.get(key) is False:
            errors.append(f"{key}=false")
    for key in ("errors", "issues", "coverage_errors", "unresolved"):
        value = receipt.get(key)
        if value:
            errors.append(f"{key} is non-empty")
    return errors


def _point_ids(doc: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for module in doc.get("modules", []) if isinstance(doc.get("modules"), list) else []:
        if not isinstance(module, dict):
            continue
        for point in module.get("points", []) if isinstance(module.get("points"), list) else []:
            if isinstance(point, dict) and isinstance(point.get("id"), str):
                result.append(point["id"])
    appendix = doc.get("appendix") if isinstance(doc.get("appendix"), dict) else {}
    tables = appendix.get("cheatsheets", []) if isinstance(appendix.get("cheatsheets", []), list) else []
    result.extend(f"appendix:{index}" for index, _table in enumerate(tables))
    return result


def _module_point_ids(doc: dict[str, Any]) -> list[str]:
    result: list[str] = []
    modules = doc.get("modules", [])
    for module in modules if isinstance(modules, list) else []:
        if not isinstance(module, dict):
            continue
        points = module.get("points", [])
        for point in points if isinstance(points, list) else []:
            if isinstance(point, dict) and isinstance(point.get("id"), str):
                result.append(point["id"])
    return result


def _module_points(doc: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    modules = doc.get("modules", [])
    for module in modules if isinstance(modules, list) else []:
        if not isinstance(module, dict):
            continue
        points = module.get("points", [])
        if isinstance(points, list):
            result.extend(point for point in points if isinstance(point, dict))
    return result


def _has_review_provenance(doc: dict[str, Any]) -> bool:
    return any(any(key in point for key in _NON_SEMANTIC_POINT_KEYS) for point in _module_points(doc))


def _unique_stable_points(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index only unambiguous, well-formed stable review identities."""
    points = _module_points(doc)
    visible_id_counts: dict[str, int] = {}
    stable_id_counts: dict[str, int] = {}
    for point in points:
        point_id = point.get("id")
        stable_id = point.get("review_point_id")
        if isinstance(point_id, str) and point_id:
            visible_id_counts[point_id] = visible_id_counts.get(point_id, 0) + 1
        if isinstance(stable_id, str) and stable_id:
            stable_id_counts[stable_id] = stable_id_counts.get(stable_id, 0) + 1

    result: dict[str, dict[str, Any]] = {}
    for point in points:
        point_id = point.get("id")
        stable_id = point.get("review_point_id")
        if not isinstance(point_id, str) or not point_id or visible_id_counts.get(point_id) != 1:
            continue
        if not isinstance(stable_id, str) or not stable_id or stable_id_counts.get(stable_id) != 1:
            continue
        video_id = point.get("review_video_id")
        task_id = point.get("review_task_id")
        if "review_video_id" in point and not (
            (isinstance(video_id, str) and video_id)
            or (type(video_id) is int and video_id >= 0)
        ):
            continue
        if "review_task_id" in point and (not isinstance(task_id, str) or not task_id):
            continue
        result[stable_id] = point
    return result


def _same_review_context(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for key in ("review_video_id", "review_task_id"):
        if (key in before) != (key in after):
            return False
        if key in before and before.get(key) != after.get(key):
            return False
    return True


def _stable_point_lineage(
    before: dict[str, Any], after: dict[str, Any], previous: dict[str, str | None],
    *, rejected_stable_ids: set[str] | None = None,
) -> dict[str, str | None]:
    """Carry origins only across a unique stable ID present in both checked documents."""
    old_points = _unique_stable_points(before)
    new_points = _unique_stable_points(after)
    rejected = rejected_stable_ids or set()
    result: dict[str, str | None] = {}
    for point in _module_points(after):
        point_id = point.get("id")
        if not isinstance(point_id, str):
            continue
        origin = None
        stable_id = point.get("review_point_id")
        new = new_points.get(stable_id) if isinstance(stable_id, str) else None
        old = old_points.get(stable_id) if isinstance(stable_id, str) and stable_id not in rejected else None
        if new is point and old is not None and _same_review_context(old, point):
            old_id = old.get("id")
            old_origin = previous.get(old_id) if isinstance(old_id, str) else None
            if isinstance(old_origin, str):
                origin = old_origin
        result[point_id] = origin
    return result


def _accepted_deleted_stable_ids(before: dict[str, Any], report: dict[str, Any]) -> set[str]:
    findings = {
        row.get("id"): row for row in report.get("findings", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    deleted_point_ids = set()
    decisions = report.get("decisions", [])
    for decision in decisions if isinstance(decisions, list) else []:
        if not isinstance(decision, dict) or decision.get("decision") != "accept" or decision.get("delete") is not True:
            continue
        finding = findings.get(decision.get("finding_id"))
        point_id = finding.get("point_id") if isinstance(finding, dict) else None
        if isinstance(point_id, str):
            deleted_point_ids.add(point_id)
    return {
        point.get("review_point_id") for point in _module_points(before)
        if point.get("id") in deleted_point_ids
        and isinstance(point.get("review_point_id"), str)
        and point.get("review_point_id")
    }


def _appendix_tables(doc: dict[str, Any]) -> list[Any]:
    appendix = doc.get("appendix") if isinstance(doc.get("appendix"), dict) else {}
    tables = appendix.get("cheatsheets", [])
    return copy.deepcopy(tables) if isinstance(tables, list) else []


def _review_decision_appendix_lineage(
    before: dict[str, Any], after: dict[str, Any], report: dict[str, Any], lineage: list[str | None]
) -> list[str | None]:
    """Carry appendix identities through the exact ordered edits in a review report."""
    old_tables = _appendix_tables(before)
    new_tables = _appendix_tables(after)
    if old_tables == new_tables:
        return list(lineage)
    findings = {row.get("id"): row for row in report.get("findings", []) if isinstance(row, dict)}
    deleted: set[int] = set()
    replacements: dict[int, Any] = {}
    decisions = report.get("decisions", [])
    for decision in decisions if isinstance(decisions, list) else []:
        if not isinstance(decision, dict) or decision.get("decision") != "accept":
            continue
        finding = findings.get(decision.get("finding_id"))
        point_id = finding.get("point_id") if isinstance(finding, dict) else None
        if not isinstance(point_id, str) or not point_id.startswith("appendix:"):
            continue
        try:
            index = int(point_id.split(":", 1)[1])
        except ValueError:
            continue
        if not 0 <= index < len(old_tables):
            continue
        if decision.get("delete") is True:
            deleted.add(index)
        else:
            replacement = decision.get("replacement")
            if isinstance(replacement, dict) and isinstance(replacement.get("table"), dict):
                replacements[index] = copy.deepcopy(replacement["table"])
    expected = [replacements.get(index, table) for index, table in enumerate(old_tables) if index not in deleted]
    if expected != new_tables:
        return [None] * len(new_tables)
    return [origin for index, origin in enumerate(lineage) if index not in deleted]


def _coverage_appendix_lineage(
    before: dict[str, Any], after: dict[str, Any], lineage: list[str | None], receipt: dict[str, Any]
) -> list[str | None]:
    """Carry appendix IDs through coverage changes only when identity is explicit or provable."""
    old_tables, new_tables = _appendix_tables(before), _appendix_tables(after)
    if old_tables == new_tables:
        return list(lineage)
    supplied = receipt.get("appendix_id_map")
    if isinstance(supplied, dict):
        if set(supplied) != {f"appendix:{i}" for i in range(len(old_tables))}:
            return [None] * len(new_tables)
        result: list[str | None] = [None] * len(new_tables)
        claimed: set[int] = set()
        for old_index, origin in enumerate(lineage):
            target = supplied.get(f"appendix:{old_index}")
            if target is None:
                continue
            if not isinstance(target, str) or not target.startswith("appendix:"):
                return [None] * len(new_tables)
            try:
                new_index = int(target.split(":", 1)[1])
            except ValueError:
                return [None] * len(new_tables)
            if not 0 <= new_index < len(new_tables) or new_index in claimed:
                return [None] * len(new_tables)
            claimed.add(new_index)
            result[new_index] = origin
        return result
    # Exact table values uniquely identify retained rows across delete/reindex.
    result = [None] * len(new_tables)
    for old_index, old_table in enumerate(old_tables):
        if old_tables.count(old_table) != 1 or new_tables.count(old_table) != 1:
            continue
        new_index = new_tables.index(old_table)
        result[new_index] = lineage[old_index] if old_index < len(lineage) else None
    return result


def _major_review_issues(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    findings = report.get("findings", [])
    if isinstance(findings, list):
        for finding in findings:
            if isinstance(finding, dict) and finding.get("severity") in {"critical", "high"}:
                issues.append(f"unresolved {finding.get('severity')} finding {finding.get('id')!r}")
    dimension_blockers = {
        "fact_accuracy": {"contradicted"},
        "knowledge_value": {"not_knowledge", "uncertain"},
        "examples": {"missing", "incorrect"},
        "figures": {"missing", "incorrect"},
    }
    points = report.get("points", [])
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, dict):
                continue
            point_id = point.get("point_id", "?")
            for dimension, blocked_statuses in dimension_blockers.items():
                assessment = point.get(dimension)
                if isinstance(assessment, dict) and assessment.get("status") in blocked_statuses:
                    issues.append(f"blocked {dimension} for {point_id!r}: {assessment.get('status')}")
            fact = point.get("fact_accuracy")
            if isinstance(fact, dict) and fact.get("risk") == "high" and fact.get("status") in {"qualified", "unverified"}:
                issues.append(f"blocked high-risk fact_accuracy for {point_id!r}: {fact.get('status')}")
    return list(dict.fromkeys(issues))


def review_workflow(
    doc,
    sources,
    visual_report,
    *,
    reviewer_factory,
    author_call,
    author_tools_call=None,
    revalidate,
    work,
    checkpoint,
    progress=lambda message: None,
    on_tool_event=lambda event: None,
    on_activity_event=lambda event: None,
    max_semantic_revisions=2,
):
    """Run up to three isolated reviews and return only a fully reviewed candidate.

    ``revalidate`` must perform the complete source-grounding and coverage check
    and return ``(validated_document, receipt)``. The receipt is accepted only
    when its ``document_hash`` equals the validated candidate's content hash.
    """
    root = Path(work)
    root.mkdir(parents=True, exist_ok=True)
    original = copy.deepcopy(doc)
    immutable_sources = copy.deepcopy(sources)
    immutable_visual_report = copy.deepcopy(visual_report)
    original_hash = document_hash(original)
    sources_hash = _json_hash(immutable_sources)
    visual_hash = _json_hash(immutable_visual_report)
    original_point_ids = _point_ids(original)
    initial_appendix_lineage = [f"appendix:{index}" for index, _table in enumerate(_appendix_tables(original))]
    lineage_by_id: dict[str, str | None] = {
        **{point_id: f"origin:original:{point_id}" for point_id in _module_point_ids(original)},
        **{f"appendix:{index}": f"origin:original:appendix:{index}" for index in range(len(initial_appendix_lineage))},
    }
    cycle_origins: dict[int, dict[str, str | None]] = {}
    stable_point_lineage = {"enabled": _has_review_provenance(original)}
    if type(max_semantic_revisions) is not int or not 0 <= max_semantic_revisions <= MAX_SEMANTIC_REVISIONS:
        raise ValueError("max_semantic_revisions must be an integer from 0 through 2")

    aggregate: dict[str, Any] = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "status": "in_progress",
        "final_review_passed": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope_note": (
            "验证每轮 reviewer 的上下文隔离与内容哈希闭环；不代表不同模型供应商、统计独立性、"
            "真实模型质量或平台验收。每轮只将当前候选、原始来源和当前视觉输入传给审查流程。"
        ),
        "max_semantic_revisions": max_semantic_revisions,
        "max_fresh_reviews": max_semantic_revisions + 1,
        "semantic_revisions_used": 0,
        "input_hash": original_hash,
        "input_semantic_hash": semantic_content_hash(original),
        "source_hash": sources_hash,
        "visual_report_hash": visual_hash,
        "output_hash": None,
        "output_semantic_hash": None,
        "final_hash": None,
        "final_semantic_hash": None,
        "final_coverage_hash": None,
        "final_coverage": None,
        "final_review": None,
        "cycles": [],
        "points": [],
        "findings": [],
        "decisions": [],
        "limitations": [],
        "cycle_points": [],
        "cycle_findings": [],
        "cycle_decisions": [],
        "final_id_map": {"schema_version": 1, "original_to_final": {}, "cycle_to_final": {}},
        "web_sources": [],
        "rounds": [],
        "review_tool_activity": [],
        "tool_activity": [],
        "failure": None,
    }
    _write_json(root / "review_workflow.json", aggregate)

    active_cycle: dict[str, Any] | None = None
    global_tool_seq = 0
    current = copy.deepcopy(original)
    seen_hashes: set[str] = set()
    coverage_receipt: dict[str, Any] | None = None
    reviewer_clients: list[Any] = []

    def notify_activity(event):
        try:
            on_activity_event(dict(event))
        except Exception:
            # Public activity must not affect review, cancellation, or submission.
            pass

    def persist() -> None:
        if active_cycle is not None:
            _write_json(Path(active_cycle["work_dir"]) / "cycle.json", active_cycle)
        _write_json(root / "review_workflow.json", aggregate)

    def refresh_final_id_map(final_doc: dict[str, Any]) -> None:
        origin_to_final: dict[str, str] = {}
        final_appendix_lineage = [lineage_by_id.get(f"appendix:{i}") for i, _ in enumerate(_appendix_tables(final_doc))]
        for point_id in _module_point_ids(final_doc):
            origin = lineage_by_id.get(point_id)
            if isinstance(origin, str):
                origin_to_final.setdefault(origin, point_id)
        for index, origin in enumerate(final_appendix_lineage):
            if isinstance(origin, str):
                origin_to_final.setdefault(origin, f"appendix:{index}")
        aggregate["final_id_map"] = {
            "schema_version": 1,
            "scope": "原始候选与每轮审查 point_id 到最终未融合候选 point_id；null 表示已删除或不能无歧义映射。",
            "original_to_final": {
                point_id: origin_to_final.get(f"origin:original:{point_id}") for point_id in original_point_ids
            },
            "cycle_to_final": {
                f"cycle-{cycle_number:02d}:{point_id}": origin_to_final.get(origin)
                for cycle_number, origins in cycle_origins.items()
                for point_id, origin in origins.items()
            },
        }

    def namespace_cycle_rows(cycle_number: int, report: dict[str, Any]) -> None:
        prefix = f"cycle-{cycle_number:02d}"
        for point in report.get("points", []) if isinstance(report.get("points"), list) else []:
            if not isinstance(point, dict):
                continue
            row = copy.deepcopy(point)
            point_id = row.get("point_id")
            row["source_point_id"] = point_id
            row["point_id"] = f"{prefix}:{point_id}"
            row["cycle"] = cycle_number
            aggregate["cycle_points"].append(row)
        for index, finding in enumerate(report.get("findings", []) if isinstance(report.get("findings"), list) else []):
            if not isinstance(finding, dict):
                continue
            row = copy.deepcopy(finding)
            source_id = row.get("id")
            row["source_id"] = source_id
            row["id"] = f"{prefix}:{source_id}" if isinstance(source_id, str) else f"{prefix}:finding-{index}"
            point_id = row.get("point_id")
            row["source_point_id"] = point_id
            row["point_id"] = f"{prefix}:{point_id}"
            row["cycle"] = cycle_number
            aggregate["cycle_findings"].append(row)
        for index, decision in enumerate(report.get("decisions", []) if isinstance(report.get("decisions"), list) else []):
            if not isinstance(decision, dict):
                continue
            row = copy.deepcopy(decision)
            source_id = row.get("finding_id")
            row["source_finding_id"] = source_id
            row["finding_id"] = f"{prefix}:{source_id}" if isinstance(source_id, str) else f"{prefix}:finding-{index}"
            row["cycle"] = cycle_number
            aggregate["cycle_decisions"].append(row)

    def record_review_report(cycle_number: int, cycle_work: Path, review_report: dict[str, Any]) -> None:
        active_cycle["review_report"] = copy.deepcopy(review_report)
        active_cycle["web_sources"] = copy.deepcopy(review_report.get("web_sources", []))
        active_cycle["rounds"] = copy.deepcopy(review_report.get("rounds", []))
        active_cycle["review_tool_activity"] = copy.deepcopy(review_report.get("tool_activity", []))
        active_cycle["review_input_hash"] = review_report.get("input_hash")
        active_cycle["review_output_hash"] = review_report.get("output_hash")
        active_cycle["blind_review"] = copy.deepcopy(review_report.get("blind_review"))
        active_cycle["review_input_semantic_hash"] = active_cycle.get("input_semantic_hash")
        active_cycle["review_output_semantic_hash"] = active_cycle.get("output_semantic_hash")
        aggregate["points"] = copy.deepcopy(review_report.get("points", []))
        aggregate["findings"] = copy.deepcopy(review_report.get("findings", []))
        aggregate["decisions"] = copy.deepcopy(review_report.get("decisions", []))
        aggregate["limitations"] = copy.deepcopy(review_report.get("limitations", []))
        namespace_cycle_rows(cycle_number, review_report)
        sources = review_report.get("web_sources", [])
        for source in sources if isinstance(sources, list) else []:
            if isinstance(source, dict):
                aggregate["web_sources"].append({**copy.deepcopy(source), "cycle": cycle_number})
        aggregate["rounds"].append({"cycle": cycle_number, "rounds": copy.deepcopy(review_report.get("rounds", []))})
        aggregate["review_tool_activity"].append({
            "cycle": cycle_number, "tool_activity": copy.deepcopy(review_report.get("tool_activity", []))
        })
        _write_json(cycle_work / "review_report.json", review_report)
        persist()

    def set_failure(exc: BaseException, kind: str | None = None) -> None:
        nonlocal active_cycle
        failure = _failure(exc, kind)
        if active_cycle is not None:
            active_cycle["failure"] = failure
            active_cycle["status"] = "cancelled" if isinstance(exc, TaskCancelled) else "blocked" if isinstance(exc, ReviewWorkflowBlocked) else "failed"
            _write_json(Path(active_cycle["work_dir"]) / "cycle.json", active_cycle)
        aggregate["failure"] = failure
        aggregate["status"] = "cancelled" if isinstance(exc, TaskCancelled) else "blocked" if isinstance(exc, ReviewWorkflowBlocked) else "failed"
        aggregate["final_review_passed"] = False
        if isinstance(current, dict):
            aggregate["failure_current_hash"] = document_hash(current)
            aggregate["failure_current_semantic_hash"] = semantic_content_hash(current)
            aggregate.setdefault("output_hash", aggregate["failure_current_hash"])
            aggregate.setdefault("output_semantic_hash", aggregate["failure_current_semantic_hash"])
            aggregate.setdefault("final_hash", aggregate["output_hash"])
            aggregate.setdefault("final_semantic_hash", aggregate["output_semantic_hash"])
            refresh_final_id_map(current)
        persist()

    def stop(message: str, kind: str) -> None:
        raise ReviewWorkflowBlocked(f"{message} (blocked: {kind})", kind)

    def run_revalidation(candidate: dict[str, Any], cycle_work: Path, stage: str):
        checkpoint()
        before_hash = document_hash(candidate)
        revalidation_dir = cycle_work / stage
        revalidation_dir.mkdir(parents=True, exist_ok=True)
        _write_json(revalidation_dir / "input.json", candidate)
        try:
            validated, receipt = revalidate(copy.deepcopy(candidate), cycle_work)
        except TaskCancelled:
            active_cycle["coverage_input_hash"] = before_hash
            active_cycle["coverage_receipt"] = None
            active_cycle["coverage_error"] = {"kind": "cancelled", "type": "TaskCancelled", "message": "coverage cancelled"}
            _write_json(revalidation_dir / "failure.json", active_cycle["coverage_error"])
            persist()
            raise
        except Exception as exc:
            active_cycle["coverage_input_hash"] = before_hash
            active_cycle["coverage_receipt"] = None
            active_cycle["coverage_error"] = _failure(exc, "coverage_exception")
            _write_json(revalidation_dir / "failure.json", active_cycle["coverage_error"])
            persist()
            raise ReviewWorkflowBlocked(f"full source coverage failed: {exc}", "coverage_error") from exc
        if not isinstance(validated, dict):
            receipt_copy = copy.deepcopy(receipt) if isinstance(receipt, dict) else {"raw_receipt": str(receipt)}
            active_cycle["coverage_input_hash"] = before_hash
            active_cycle["coverage_receipt"] = receipt_copy
            _write_json(revalidation_dir / "receipt.json", receipt_copy)
            stop("revalidate did not return a document object", "coverage_contract")
        receipt_copy = copy.deepcopy(receipt) if isinstance(receipt, dict) else {"raw_receipt": str(receipt)}
        validated_hash = document_hash(validated)
        active_cycle["coverage_input_hash"] = before_hash
        active_cycle["coverage_hash"] = validated_hash
        active_cycle["coverage_receipt"] = receipt_copy
        active_cycle["coverage_status"] = "checked"
        _write_json(revalidation_dir / "validated_document.json", validated)
        _write_json(revalidation_dir / "receipt.json", receipt_copy)
        if not isinstance(receipt, dict):
            active_cycle["coverage_status"] = "blocked"
            active_cycle["coverage_errors"] = ["coverage receipt must be an object"]
            _write_json(Path(active_cycle["work_dir"]) / "cycle.json", active_cycle)
            persist()
            stop("coverage receipt must be an object", "coverage_contract")
        if receipt.get("document_hash") != validated_hash:
            active_cycle["coverage_status"] = "blocked"
            _write_json(Path(active_cycle["work_dir"]) / "cycle.json", active_cycle)
            persist()
            stop("coverage receipt hash does not match the validated candidate", "coverage_hash_mismatch")
        errors = _coverage_errors(receipt)
        if errors:
            active_cycle["coverage_status"] = "blocked"
            active_cycle["coverage_errors"] = errors
            _write_json(Path(active_cycle["work_dir"]) / "cycle.json", active_cycle)
            persist()
            stop("full source coverage failed: " + "; ".join(errors), "coverage_error")
        active_cycle["coverage_status"] = "complete"
        active_cycle["coverage_semantic_hash"] = semantic_content_hash(validated)
        checkpoint()
        return copy.deepcopy(validated), receipt_copy, validated_hash

    def lineage_after_review(before: dict[str, Any], after: dict[str, Any], report: dict[str, Any], cycle_number: int):
        new_lineage: dict[str, str | None] = {}
        use_stable_ids = (
            stable_point_lineage["enabled"]
            or _has_review_provenance(before)
            or _has_review_provenance(after)
        )
        if use_stable_ids:
            stable_point_lineage["enabled"] = True
            rejected = _accepted_deleted_stable_ids(before, report)
            new_lineage.update(_stable_point_lineage(
                before, after, lineage_by_id, rejected_stable_ids=rejected,
            ))
        else:
            for point_id in _module_point_ids(after):
                if point_id in lineage_by_id:
                    new_lineage[point_id] = lineage_by_id[point_id]
                else:
                    new_lineage[point_id] = f"origin:cycle-{cycle_number:02d}:output:{point_id}"
        appendix_lineage = _review_decision_appendix_lineage(
            before, after, report, [lineage_by_id.get(f"appendix:{i}") for i, _ in enumerate(_appendix_tables(before))]
        )
        for index, origin in enumerate(appendix_lineage):
            new_lineage[f"appendix:{index}"] = origin if origin is not None else f"origin:cycle-{cycle_number:02d}:output:appendix:{index}"
        return new_lineage

    def lineage_after_coverage(before: dict[str, Any], after: dict[str, Any], previous: dict[str, str | None], receipt: dict[str, Any], cycle_number: int):
        new_lineage: dict[str, str | None] = {}
        use_stable_ids = (
            stable_point_lineage["enabled"]
            or _has_review_provenance(before)
            or _has_review_provenance(after)
        )
        if use_stable_ids:
            stable_point_lineage["enabled"] = True
            new_lineage.update(_stable_point_lineage(before, after, previous))
        else:
            for point_id in _module_point_ids(after):
                if point_id in previous:
                    new_lineage[point_id] = previous[point_id]
                else:
                    new_lineage[point_id] = f"origin:coverage-cycle-{cycle_number:02d}:{point_id}"
        appendix_before = _appendix_tables(before)
        appendix_origins = [previous.get(f"appendix:{i}") for i in range(len(appendix_before))]
        appendix_after = _coverage_appendix_lineage(before, after, appendix_origins, receipt)
        for index, origin in enumerate(appendix_after):
            new_lineage[f"appendix:{index}"] = origin if origin is not None else f"origin:coverage-cycle-{cycle_number:02d}:appendix:{index}"
        return new_lineage

    try:
        for cycle_number in range(1, max_semantic_revisions + 2):
            checkpoint()
            cycle_work = root / f"cycle-{cycle_number:02d}"
            cycle_work.mkdir(parents=True, exist_ok=True)
            active_cycle = {
                "cycle": cycle_number,
                "status": "in_progress",
                "work_dir": str(cycle_work),
                "candidate_hash_before_coverage": document_hash(current),
                "candidate_semantic_hash_before_coverage": semantic_content_hash(current),
                "input_hash": None,
                "input_semantic_hash": None,
                "output_hash": None,
                "output_semantic_hash": None,
                "coverage_input_hash": None,
                "coverage_hash": None,
                "coverage_receipt": None,
                "review_report": None,
                "web_sources": [],
                "rounds": [],
                "tool_activity": [],
                "failure": None,
            }
            aggregate["cycles"].append(active_cycle)
            _write_json(cycle_work / "input.json", current)
            persist()
            progress(f"独立复审：准备第 {cycle_number}/{max_semantic_revisions + 1} 轮候选覆盖")

            if coverage_receipt is None:
                before_coverage = copy.deepcopy(current)
                current, coverage_receipt, current_hash = run_revalidation(current, cycle_work, "coverage-input")
                lineage_by_id = lineage_after_coverage(
                    before_coverage, current, lineage_by_id, coverage_receipt, cycle_number
                )
            else:
                current_hash = document_hash(current)
                if coverage_receipt.get("document_hash") != current_hash:
                    stop("carried coverage receipt does not match the next review candidate", "coverage_hash_mismatch")
                _write_json(cycle_work / "coverage-input-receipt.json", coverage_receipt)
            active_cycle["input_hash"] = current_hash
            current_semantic_hash = semantic_content_hash(current)
            active_cycle["input_semantic_hash"] = current_semantic_hash
            active_cycle["input_coverage_receipt"] = copy.deepcopy(coverage_receipt)
            _write_json(cycle_work / "validated-input.json", current)
            if current_semantic_hash in seen_hashes:
                stop("coverage reconciliation returned a previously reviewed content hash", "content_hash_cycle")
            aggregate["final_hash"] = current_hash
            aggregate["final_semantic_hash"] = current_semantic_hash
            aggregate["final_coverage_hash"] = current_hash
            aggregate["final_coverage"] = copy.deepcopy(coverage_receipt)
            aggregate["output_hash"] = current_hash
            aggregate["output_semantic_hash"] = current_semantic_hash
            cycle_origins[cycle_number] = {
                point_id: lineage_by_id[point_id] if point_id in lineage_by_id else None
                for point_id in _point_ids(current)
            }
            active_cycle["point_origin_ids"] = copy.deepcopy(cycle_origins[cycle_number])
            persist()

            local_to_global: dict[int, int] = {}

            def forward_tool_event(event):
                nonlocal global_tool_seq
                copied = copy.deepcopy(event)
                if copied.get("event") == "public_progress":
                    row = {**copied, "cycle": cycle_number}
                    try:
                        on_tool_event(copy.deepcopy(row))
                    except Exception:
                        # Progress is observational; never let delivery affect review work.
                        pass
                    return
                local_seq = copied.get("seq")
                if type(local_seq) is int:
                    if local_seq not in local_to_global:
                        global_tool_seq += 1
                        local_to_global[local_seq] = global_tool_seq
                    copied["seq"] = local_to_global[local_seq]
                row = {**copy.deepcopy(copied), "cycle": cycle_number}
                active_cycle["tool_activity"].append(row)
                aggregate["tool_activity"].append(row)
                _write_json(cycle_work / "tool_events.json", active_cycle["tool_activity"])
                persist()
                try:
                    on_tool_event(copy.deepcopy(row))
                except Exception:
                    # Callback delivery is observational; the audited row is already saved.
                    pass

            def fresh_reviewer_factory():
                client = reviewer_factory()
                if any(client is previous for previous in reviewer_clients):
                    stop("reviewer_factory reused a client across semantic cycles", "reviewer_client_reused")
                reviewer_clients.append(client)
                return client

            checkpoint()
            seen_hashes.add(current_semantic_hash)
            notify_activity({
                "cycle": cycle_number, "status": "running",
                "detail": "獨立審查開始核對本輪候選",
            })
            try:
                review_kwargs = {
                    "reviewer_factory": fresh_reviewer_factory,
                    "author_call": author_call,
                    "work": cycle_work,
                    "evidence_root": root,
                    "checkpoint": checkpoint,
                    "progress": progress,
                    "on_tool_event": forward_tool_event,
                }
                try:
                    parameters = inspect.signature(review_and_revise).parameters
                    accepts_extra = any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                except (TypeError, ValueError):
                    parameters, accepts_extra = {}, False
                if accepts_extra or "on_activity_event" in parameters:
                    review_kwargs["on_activity_event"] = lambda event: notify_activity({
                        **event, "cycle": cycle_number,
                    }) if isinstance(event, dict) else None
                if accepts_extra or "activity_context" in parameters:
                    review_kwargs["activity_context"] = {
                        "actor": "reviewer",
                        "agent_id": f"reviewer-cycle-{cycle_number:02d}",
                        "stage": "review", "cycle": cycle_number,
                        "execution": "isolated_context",
                    }
                if author_tools_call is not None:
                    review_kwargs["author_tools_call"] = author_tools_call
                result_doc, review_report = review_and_revise(
                    copy.deepcopy(current),
                    copy.deepcopy(immutable_sources),
                    copy.deepcopy(immutable_visual_report),
                    **review_kwargs,
                )
            except TaskCancelled:
                notify_activity({
                    "cycle": cycle_number, "status": "cancelled",
                    "error": "TaskCancelled",
                })
                try:
                    cancelled_report = json.loads((cycle_work / "independent_review.json").read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    cancelled_report = None
                if isinstance(cancelled_report, dict):
                    record_review_report(cycle_number, cycle_work, cancelled_report)
                raise
            except Exception as review_exc:
                notify_activity({
                    "cycle": cycle_number, "status": "error",
                    "error": type(review_exc).__name__,
                })
                failed_report = getattr(review_exc, "report", None)
                if not isinstance(failed_report, dict):
                    try:
                        failed_report = json.loads((cycle_work / "independent_review.json").read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        failed_report = None
                if isinstance(failed_report, dict):
                    record_review_report(cycle_number, cycle_work, failed_report)
                if isinstance(review_exc, IndependentReviewBlocked) or (
                    isinstance(failed_report, dict) and failed_report.get("status") == "blocked"
                ):
                    raise ReviewWorkflowBlocked(str(review_exc), "independent_review_blocked") from review_exc
                raise
            if isinstance(result_doc, dict):
                output_hash = document_hash(result_doc)
                output_semantic_hash = semantic_content_hash(result_doc)
                active_cycle["output_hash"] = output_hash
                active_cycle["output_semantic_hash"] = output_semantic_hash
                aggregate["output_hash"] = output_hash
                aggregate["output_semantic_hash"] = output_semantic_hash
                _write_json(cycle_work / "reviewed_output.json", result_doc)
            if isinstance(review_report, dict):
                record_review_report(cycle_number, cycle_work, review_report)
            persist()
            checkpoint()
            if not isinstance(result_doc, dict) or not isinstance(review_report, dict):
                stop("review_and_revise returned an invalid result", "review_contract")

            reported_input = review_report.get("input_hash")
            reported_output = review_report.get("output_hash")
            if reported_input != current_hash:
                stop("review report input hash does not match its candidate", "review_input_hash_mismatch")
            if reported_output != output_hash:
                stop("review report output hash does not match its candidate", "review_output_hash_mismatch")
            binding_errors = validate_binding(
                current, review_report.get("blind_review"), current_hash,
            )
            if binding_errors:
                stop("blind review receipt is invalid: " + "; ".join(binding_errors), "blind_review_binding")
            status = review_report.get("status")
            if status not in {"reviewed", "revised"}:
                stop(f"independent review did not pass (status={status!r})", "review_not_passed")
            if output_semantic_hash != current_semantic_hash and status != "revised":
                stop("review report says reviewed while semantic content changed", "review_status_mismatch")
            notify_activity({
                "cycle": cycle_number, "status": "done",
                "result_summary": (
                    f"独立审查完成：{len(review_report.get('findings', []))} 条发现，"
                    f"{len(review_report.get('decisions', []))} 条作者裁决"
                ),
            })

            output_lineage = lineage_after_review(current, result_doc, review_report, cycle_number)
            if output_hash == current_hash:
                validated_doc = copy.deepcopy(result_doc)
                validated_coverage = copy.deepcopy(coverage_receipt)
                validated_hash = output_hash
                validated_lineage = output_lineage
            else:
                if output_semantic_hash != current_semantic_hash and aggregate["semantic_revisions_used"] < max_semantic_revisions:
                    coverage_work = root / f"cycle-{cycle_number + 1:02d}"
                    coverage_stage = "coverage-input"
                else:
                    coverage_work = cycle_work
                    coverage_stage = "coverage-output"
                coverage_work.mkdir(parents=True, exist_ok=True)
                validated_doc, validated_coverage, validated_hash = run_revalidation(
                    result_doc, coverage_work, coverage_stage
                )
                validated_lineage = lineage_after_coverage(
                    result_doc, validated_doc, output_lineage, validated_coverage, cycle_number
                )
                active_cycle["validated_output_hash"] = validated_hash
                active_cycle["validated_output_semantic_hash"] = semantic_content_hash(validated_doc)
                active_cycle["output_coverage_receipt"] = copy.deepcopy(validated_coverage)
                _write_json(cycle_work / "validated_output.json", validated_doc)

            validated_semantic_hash = semantic_content_hash(validated_doc)
            active_cycle["final_candidate_hash"] = validated_hash
            active_cycle["final_candidate_semantic_hash"] = validated_semantic_hash
            aggregate["output_hash"] = validated_hash
            aggregate["output_semantic_hash"] = validated_semantic_hash
            aggregate["final_hash"] = validated_hash
            aggregate["final_semantic_hash"] = validated_semantic_hash
            aggregate["final_coverage_hash"] = validated_coverage.get("document_hash")
            aggregate["final_coverage"] = copy.deepcopy(validated_coverage)

            if validated_semantic_hash == current_semantic_hash:
                if output_semantic_hash != current_semantic_hash:
                    stop("coverage reconciliation returned a previously reviewed semantic hash", "content_hash_cycle")
                major_issues = _major_review_issues(review_report)
                if major_issues:
                    active_cycle["unresolved_major_issues"] = major_issues
                    stop("final fresh review still has unresolved major issues: " + "; ".join(major_issues),
                         "unresolved_major_issues")
                current = copy.deepcopy(validated_doc)
                lineage_by_id = validated_lineage
                coverage_receipt = validated_coverage
                aggregate["status"] = "revised" if aggregate["semantic_revisions_used"] else "reviewed"
                aggregate["final_review_passed"] = True
                aggregate["final_hash"] = validated_hash
                aggregate["final_semantic_hash"] = validated_semantic_hash
                aggregate["output_hash"] = validated_hash
                aggregate["output_semantic_hash"] = validated_semantic_hash
                aggregate["final_coverage_hash"] = validated_hash
                aggregate["final_coverage"] = copy.deepcopy(validated_coverage)
                aggregate["final_review"] = copy.deepcopy(review_report)
                aggregate["final_review_binding"] = {
                    "reviewed_document_hash": output_hash,
                    "reviewed_semantic_hash": output_semantic_hash,
                    "final_candidate_document_hash": validated_hash,
                    "final_candidate_semantic_hash": validated_semantic_hash,
                    "semantic_match": output_semantic_hash == validated_semantic_hash,
                }
                aggregate["completed_at"] = datetime.now(timezone.utc).isoformat()
                active_cycle["status"] = "passed"
                active_cycle["final_coverage_hash"] = validated_hash
                active_cycle["final_coverage_semantic_hash"] = validated_semantic_hash
                refresh_final_id_map(current)
                _write_json(cycle_work / "cycle.json", active_cycle)
                persist()
                return current, aggregate

            if validated_semantic_hash in seen_hashes:
                stop("semantic revision returned to a previously reviewed content hash", "content_hash_cycle")
            aggregate["semantic_revisions_used"] += 1
            active_cycle["semantic_revision"] = aggregate["semantic_revisions_used"]
            lineage_by_id = validated_lineage
            current = copy.deepcopy(validated_doc)
            coverage_receipt = copy.deepcopy(validated_coverage)
            if aggregate["semantic_revisions_used"] > max_semantic_revisions:
                stop("semantic revision budget exhausted before a final unchanged review", "semantic_revision_budget")
            active_cycle["status"] = "revised_needs_fresh_review"
            _write_json(cycle_work / "cycle.json", active_cycle)
            persist()
            progress(f"独立复审：第 {cycle_number} 轮产生语义修订，已完成覆盖核验；启动全新审查员")

        stop("fresh review budget exhausted without an unchanged final candidate", "fresh_review_budget")
    except TaskCancelled as exc:
        set_failure(exc, "cancelled")
        raise
    except Exception as exc:
        set_failure(exc)
        raise
