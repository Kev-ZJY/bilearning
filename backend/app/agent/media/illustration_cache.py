"""Pure, fail-closed replay records for illustration review decisions.

This module deliberately does not read checkpoint files, frame files, settings,
or environment variables. The recovery caller supplies stable identities and
frame SHA-256 values, then must route every returned decision through the live
illustration validation and application gates.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any


ILLUSTRATION_REPLAY_CACHE_VERSION = 1
DOTS_ILLUSTRATION_MODEL_ID = "dots-studio/dots-3-note-preview:free"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _identity_hash(value: Any, name: str) -> str:
    if not _valid_sha256(value):
        raise ValueError(f"{name}必须是小写64位SHA-256")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _source_points(source_doc: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(source_doc, dict) or not isinstance(source_doc.get("modules"), list):
        raise ValueError("source-reviewed文档结构无效")
    result: dict[str, dict[str, Any]] = {}
    for module in source_doc["modules"]:
        if not isinstance(module, dict) or not isinstance(module.get("points"), list):
            raise ValueError("source-reviewed模块结构无效")
        for point in module["points"]:
            if not isinstance(point, dict):
                raise ValueError("source-reviewed知识点结构无效")
            point_id = point.get("id")
            if not isinstance(point_id, str) or not point_id:
                raise ValueError("source-reviewed知识点ID无效")
            if point_id in result:
                raise ValueError("source-reviewed知识点ID重复")
            # Validate that the complete point can be persisted and compared as
            # JSON data; do not normalize or project away any knowledge fields.
            _canonical_json(point)
            result[point_id] = copy.deepcopy(point)
    return result


def _seconds(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(type(second) is not int or second < 0 for second in value):
        return None
    if len(value) != len(set(value)):
        return None
    return list(value)


def _hash_for_second(values: Mapping[Any, Any], second: int) -> str | None:
    value = values.get(second)
    if value is None:
        value = values.get(str(second))
    return value if _valid_sha256(value) else None


def _frame_hashes_for_seconds(
    values: Any, seconds: list[int], *, strict: bool,
) -> dict[str, str] | None:
    if not isinstance(values, Mapping):
        return None
    hashes: dict[str, str] = {}
    for second in seconds:
        digest = _hash_for_second(values, second)
        if digest is None:
            if strict:
                return None
            raise ValueError("候选帧缺少有效文件SHA-256")
        hashes[str(second)] = digest
    return hashes


def _candidate_image_hashes(value: Any) -> list[str] | None:
    if not isinstance(value, list) or any(not _valid_sha256(item) for item in value):
        return None
    if value != sorted(set(value)):
        return None
    return list(value)


def _raw_batch_decisions(
    report: Any, source_points: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return decisions only from complete, protocol-accepted full batches."""
    if not isinstance(report, dict) or not isinstance(report.get("batches", []), list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    ambiguous_points: set[str] = set()
    batch_id_counts: dict[str, int] = {}
    for candidate in report.get("batches", []):
        if isinstance(candidate, dict) and isinstance(candidate.get("batch_id"), str):
            batch_id = candidate["batch_id"]
            batch_id_counts[batch_id] = batch_id_counts.get(batch_id, 0) + 1
    for batch in report.get("batches", []):
        if not isinstance(batch, dict):
            continue
        if batch.get("status") != "complete" or batch.get("protocol_status") != "accepted":
            continue
        batch_id = batch.get("batch_id")
        point_ids = batch.get("point_ids")
        if (
            not isinstance(batch_id, str) or not batch_id or batch_id_counts.get(batch_id) != 1
            or not isinstance(point_ids, list) or not point_ids
            or any(not isinstance(point_id, str) or point_id not in source_points for point_id in point_ids)
            or len(point_ids) != len(set(point_ids))
        ):
            continue
        candidate_seconds = batch.get("candidate_seconds")
        image_hashes = batch.get("candidate_image_hashes")
        if (
            not isinstance(candidate_seconds, dict)
            or set(candidate_seconds) != set(point_ids)
            or not isinstance(image_hashes, dict)
            or set(image_hashes) != set(point_ids)
        ):
            continue
        per_point_metadata = {}
        valid_metadata = True
        for point_id in point_ids:
            seconds = _seconds(candidate_seconds[point_id])
            digests = _candidate_image_hashes(image_hashes[point_id])
            if seconds is None or digests is None:
                valid_metadata = False
                break
            per_point_metadata[point_id] = {
                "candidate_seconds": seconds,
                "candidate_image_hashes": digests,
            }
        if not valid_metadata:
            continue

        raw_result = batch.get("raw_result")
        reviews = raw_result.get("reviews") if isinstance(raw_result, dict) else None
        if not isinstance(reviews, list) or len(reviews) != len(point_ids):
            continue
        parsed: dict[str, dict[str, Any]] = {}
        for review in reviews:
            if not isinstance(review, dict):
                parsed = {}
                break
            point_id = review.get("point_id")
            if not isinstance(point_id, str) or point_id not in point_ids or point_id in parsed:
                parsed = {}
                break
            parsed[point_id] = copy.deepcopy(review)
        if set(parsed) != set(point_ids):
            continue

        for point_id in point_ids:
            if point_id in ambiguous_points:
                continue
            raw_review = parsed[point_id]
            decision = {key: copy.deepcopy(value) for key, value in raw_review.items() if key != "point_id"}
            if point_id in result:
                # Multiple accepted batches for one point are ambiguous unless
                # the full decision and candidate identity are byte-for-byte
                # equivalent after canonical JSON serialization.
                prior = result[point_id]
                comparable = {
                    key: prior[key] for key in (
                        "result", "raw_review", "batch_id", "candidate_seconds",
                        "candidate_image_hashes",
                    )
                }
                current = {
                    "result": decision,
                    "raw_review": raw_review,
                    "batch_id": batch_id,
                    **per_point_metadata[point_id],
                }
                if _canonical_json(comparable) != _canonical_json(current):
                    result.pop(point_id, None)
                    ambiguous_points.add(point_id)
                continue
            result[point_id] = {
                "result": decision,
                "raw_review": raw_review,
                "batch_id": batch_id,
                **per_point_metadata[point_id],
            }
    return result


def build_illustration_replay_cache(
    source_doc: Any,
    review_report: Any,
    *,
    source_inputs_sha256: str,
    frame_sha256_by_second: Mapping[Any, Any],
    policy_sha256: str,
    model_id: str,
) -> dict[str, Any]:
    """Build a JSON-safe replay cache from explicit checkpoint data only.

    The caller is responsible for proving that source_inputs_sha256 covers the
    original source inputs and that frame_sha256_by_second comes from the
    original checkpoint's frame files. This function performs no file or model
    access. Only the exact configured Dots model identifier is eligible.
    """
    source_digest = _identity_hash(source_inputs_sha256, "source_inputs_sha256")
    policy_digest = _identity_hash(policy_sha256, "policy_sha256")
    if model_id != DOTS_ILLUSTRATION_MODEL_ID:
        raise ValueError("只有精确匹配的Dots模型可以建立图例重放缓存")
    if not isinstance(frame_sha256_by_second, Mapping):
        raise ValueError("frame_sha256_by_second必须是显式映射")
    source_points = _source_points(source_doc)
    if not isinstance(review_report, dict):
        raise ValueError("illustration_review报告结构无效")

    report_points = review_report.get("points", [])
    if not isinstance(report_points, list):
        raise ValueError("illustration_review.points结构无效")
    point_entries: dict[str, dict[str, Any]] = {}
    ambiguous_point_entries: set[str] = set()
    for entry in report_points:
        if not isinstance(entry, dict):
            continue
        point_id = entry.get("point_id")
        if not isinstance(point_id, str) or point_id not in source_points:
            continue
        if point_id in point_entries:
            point_entries.pop(point_id, None)
            ambiguous_point_entries.add(point_id)
            continue
        if point_id in ambiguous_point_entries:
            continue
        point_entries[point_id] = entry

    batch_decisions = _raw_batch_decisions(review_report, source_points)
    entries: dict[str, dict[str, Any]] = {}

    # A per-point status is set only after the existing necessity, citation,
    # document, and figure gates passed. Store the raw accepted attempt, but the
    # replay caller must still rerun those gates against the current inputs.
    for point_id, entry in point_entries.items():
        if entry.get("status") != "complete":
            continue
        attempts = entry.get("attempts")
        if (
            not isinstance(attempts, list) or not attempts
            or any(
                not isinstance(attempt, dict) or not isinstance(attempt.get("result"), dict)
                for attempt in attempts
            )
        ):
            continue
        final_attempt = attempts[-1]
        if "error" in final_attempt or not isinstance(final_attempt.get("result"), dict):
            continue
        final_result = final_attempt["result"]
        previously_required = {
            key: bool((source_points[point_id].get("figure_decision") or {}).get(key))
            for key in ("needs_figure", "needs_example")
        }
        for attempt in attempts:
            raw_attempt_result = attempt.get("result") if isinstance(attempt, dict) else None
            if isinstance(raw_attempt_result, dict):
                for key in previously_required:
                    previously_required[key] = (
                        previously_required[key] or raw_attempt_result.get(key) is True
                    )
        if any(previously_required[key] and final_result.get(key) is not True
               for key in previously_required):
            continue
        seconds = _seconds(final_attempt.get("candidate_seconds"))
        if seconds is None:
            continue
        hashes = _frame_hashes_for_seconds(frame_sha256_by_second, seconds, strict=True)
        if hashes is None:
            continue
        raw_review = None
        candidate_image_hashes = None
        batch_id = final_attempt.get("batch_call_id")
        if batch_id is not None:
            batch = batch_decisions.get(point_id)
            if batch is None or batch["batch_id"] != batch_id:
                continue
            if (
                batch["candidate_seconds"] != seconds
                or _canonical_json(batch["result"]) != _canonical_json(final_attempt["result"])
            ):
                continue
            raw_review = copy.deepcopy(batch["raw_review"])
            candidate_image_hashes = copy.deepcopy(batch["candidate_image_hashes"])
        else:
            # Individual correction calls do not have batch metadata to bind
            # the exact JPEG payloads sent to the model. Older reports without
            # this per-attempt receipt are deliberately a cache miss.
            candidate_image_hashes = _candidate_image_hashes(
                final_attempt.get("candidate_image_hashes")
            )
            if candidate_image_hashes is None:
                continue
        entries[point_id] = {
            "replay_state": "previously_complete",
            "source_point": copy.deepcopy(source_points[point_id]),
            "candidate_seconds": seconds,
            "frame_sha256_by_second": hashes,
            "candidate_image_hashes": candidate_image_hashes,
            "result": copy.deepcopy(final_attempt["result"]),
            "raw_review": raw_review,
            "attempts": copy.deepcopy(attempts),
            "batch_id": batch_id,
        }

    # A completed provider batch may precede per-point application. It is safe
    # to reuse only its individual raw decision, explicitly marked unapplied;
    # live gates decide whether it is acceptable now.
    for point_id, batch in batch_decisions.items():
        if point_id in entries or point_id in ambiguous_point_entries:
            continue
        entry = point_entries.get(point_id)
        # The initial batch may be replayed only before any per-point attempt
        # has had a chance to strengthen the sticky necessity requirement.
        if (
            entry is None
            or entry.get("status") == "complete"
            or entry.get("attempts") != []
        ):
            continue
        hashes = _frame_hashes_for_seconds(
            frame_sha256_by_second, batch["candidate_seconds"], strict=True,
        )
        if hashes is None:
            continue
        entries[point_id] = {
            "replay_state": "unapplied_batch",
            "source_point": copy.deepcopy(source_points[point_id]),
            "candidate_seconds": copy.deepcopy(batch["candidate_seconds"]),
            "frame_sha256_by_second": hashes,
            "candidate_image_hashes": copy.deepcopy(batch["candidate_image_hashes"]),
            "result": copy.deepcopy(batch["result"]),
            "raw_review": copy.deepcopy(batch["raw_review"]),
            "attempts": [],
            "batch_id": batch["batch_id"],
        }

    cache = {
        "schema_version": ILLUSTRATION_REPLAY_CACHE_VERSION,
        "source_inputs_sha256": source_digest,
        "policy_sha256": policy_digest,
        "model_id": model_id,
        "entries": entries,
    }
    # Reports are normally JSON-loaded. Keep the cache contract explicit and
    # fail closed if a caller supplied non-serializable synthetic objects.
    _canonical_json(cache)
    return cache


def lookup_illustration_replay(
    cache: Any,
    source_point: Any,
    *,
    source_inputs_sha256: str,
    candidate_seconds: Any,
    frame_sha256_by_second: Mapping[Any, Any],
    policy_sha256: str,
    model_id: str,
    candidate_image_hashes: Any = None,
) -> dict[str, Any] | None:
    """Return one raw decision only when every explicit replay identity matches.

    The returned value contains no document or applied point. The caller must
    rerun its live decision, source-quote, necessity, and document validators.
    """
    try:
        source_digest = _identity_hash(source_inputs_sha256, "source_inputs_sha256")
        policy_digest = _identity_hash(policy_sha256, "policy_sha256")
    except ValueError:
        return None
    if model_id != DOTS_ILLUSTRATION_MODEL_ID or not isinstance(cache, dict):
        return None
    if (
        type(cache.get("schema_version")) is not int
        or cache.get("schema_version") != ILLUSTRATION_REPLAY_CACHE_VERSION
        or cache.get("source_inputs_sha256") != source_digest
        or cache.get("policy_sha256") != policy_digest
        or cache.get("model_id") != model_id
        or not isinstance(source_point, dict)
    ):
        return None
    point_id = source_point.get("id")
    entries = cache.get("entries")
    if not isinstance(point_id, str) or not isinstance(entries, dict):
        return None
    entry = entries.get(point_id)
    if not isinstance(entry, dict):
        return None
    if entry.get("replay_state") not in ("previously_complete", "unapplied_batch"):
        return None
    try:
        if _canonical_json(entry.get("source_point")) != _canonical_json(source_point):
            return None
    except (TypeError, ValueError):
        return None

    seconds = _seconds(candidate_seconds)
    cached_seconds = _seconds(entry.get("candidate_seconds"))
    if seconds is None or cached_seconds != seconds:
        return None
    current_hashes = _frame_hashes_for_seconds(frame_sha256_by_second, seconds, strict=True)
    if current_hashes is None or entry.get("frame_sha256_by_second") != current_hashes:
        return None

    cached_image_hashes = _candidate_image_hashes(entry.get("candidate_image_hashes"))
    current_image_hashes = _candidate_image_hashes(candidate_image_hashes)
    if cached_image_hashes is None or current_image_hashes is None:
        return None
    if current_image_hashes != cached_image_hashes:
        return None
    result = entry.get("result")
    attempts = entry.get("attempts")
    raw_review = entry.get("raw_review")
    if not isinstance(result, dict) or not isinstance(attempts, list):
        return None
    if entry.get("replay_state") == "unapplied_batch":
        if not isinstance(raw_review, dict) or raw_review.get("point_id") != point_id:
            return None
        try:
            same_decision = _canonical_json(
                {key: value for key, value in raw_review.items() if key != "point_id"}
            ) == _canonical_json(result)
        except (TypeError, ValueError):
            return None
        if not same_decision or not isinstance(entry.get("batch_id"), str):
            return None
    try:
        # Ensure the returned record is JSON-safe and detached from cache data.
        _canonical_json(result)
        _canonical_json(attempts)
    except (TypeError, ValueError):
        return None
    return {
        "replay_state": entry["replay_state"],
        "result": copy.deepcopy(result),
        "raw_review": copy.deepcopy(raw_review),
        "attempts": copy.deepcopy(attempts),
        "candidate_seconds": list(seconds),
        "frame_sha256_by_second": copy.deepcopy(current_hashes),
        "candidate_image_hashes": copy.deepcopy(cached_image_hashes),
        "batch_id": entry.get("batch_id"),
    }
