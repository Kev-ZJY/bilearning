"""Bind optional raw illustration decisions to verified recovery inputs.

No model requests or credentials. A cache is a replay of a tool result, never
an accepted document; enrich_illustrations must apply every current gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from app.agent.media.illustration_cache import (
    DOTS_ILLUSTRATION_MODEL_ID, build_illustration_replay_cache,
)

_LEGACY_ENRICH = "0dfb7ac0353310f55a870f385def1f0c2e6e3886e68799eb8efa720d6cbf0bdb"
# Set only after proving the new wrapper retains the exact nested decision and
# application gates. Any subsequent enrich code edit invalidates this bridge.
_REPLAY_ENRICH = "9b38a06ab97fd9d7f92c7ab6e88af2eb2656ddd55cde56a26b40f918cd4b4441"


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _policy_components():
    from app.services.recovery_versions import (
        default_candidate_generation_code_components, _loaded_module_callables,
        loaded_code_sha256,
    )
    from app.agent.media import illustration_cache
    components = {k: v for k, v in default_candidate_generation_code_components().items()
                  if k.startswith("illustrations.")}
    cache_code = {k: loaded_code_sha256(v) for k, v in
                  _loaded_module_callables(illustration_cache, "illustration_cache").items()}
    import sys
    bridge_code = {k: loaded_code_sha256(v) for k, v in
                   _loaded_module_callables(sys.modules[__name__], "illustration_replay").items()}
    return components, _digest({"illustrations": components, "cache_code": cache_code,
                               "bridge_code": bridge_code,
                               "legacy_enrich": _LEGACY_ENRICH, "replay_enrich": _REPLAY_ENRICH,
                               "dots_model": DOTS_ILLUSTRATION_MODEL_ID,
                               "cache_version": illustration_cache.ILLUSTRATION_REPLAY_CACHE_VERSION})


def replay_policy_for_source(source_components, report):
    current, policy = _policy_components()
    if not isinstance(source_components, dict):
        return None
    source = {k: v for k, v in source_components.items() if k.startswith("illustrations.")}
    recorded = report.get("replay_policy_sha256")
    if recorded is not None:
        return policy if source == current and recorded == policy else None
    # One explicit, finite compatibility transition for raw pre-cache batches.
    if (source.get("illustrations.enrich_illustrations") != _LEGACY_ENRICH
            or current.get("illustrations.enrich_illustrations") != _REPLAY_ENRICH):
        return None
    source["illustrations.enrich_illustrations"] = _REPLAY_ENRICH
    return policy if source == current else None


def frame_hashes(inputs):
    frames = {}
    for row in inputs:
        path = Path(row["path"])
        if path.suffix.lower() == ".jpg" and path.stem.isdigit():
            second = str(int(path.stem))
            if second in frames:
                raise ValueError("图例重放帧秒数不唯一")
            frames[second] = row["sha256"]
    return frames


def source_identity(sources, visual_report, media_identity, frames):
    def without_paths(value):
        if isinstance(value, dict):
            return {k: without_paths(v) for k, v in value.items() if k != "paths"}
        if isinstance(value, list):
            return [without_paths(v) for v in value]
        return value
    return _digest({"sources": sources, "visual": without_paths(visual_report),
                    "media_identity": media_identity, "frames": frames})


def build_recovery_replay(source_doc, review_report, *, code_components,
                          sources, visual_report, media_identity, inputs, activity,
                          inherited=None):
    models = {s.get("model") for s in activity.get("activity", {}).get("steps", [])
              if s.get("step") == "llm_call" and s.get("stage") == "illustrate"
              and s.get("status") == "done"}
    policy = replay_policy_for_source(code_components, review_report)
    if policy is None:
        return None
    frames = frame_hashes(inputs)
    identity = source_identity(sources, visual_report, media_identity, frames)
    cache = build_illustration_replay_cache(
        source_doc,
        review_report if models == {DOTS_ILLUSTRATION_MODEL_ID} else {"points": []},
        source_inputs_sha256=identity,
        frame_sha256_by_second=frames, policy_sha256=policy,
        model_id=DOTS_ILLUSTRATION_MODEL_ID,
    )
    if (isinstance(inherited, dict) and inherited.get("policy_sha256") == policy
            and inherited.get("source_inputs_sha256") == identity
            and inherited.get("model_id") == DOTS_ILLUSTRATION_MODEL_ID
            and isinstance(inherited.get("entries"), dict)):
        prior = copy.deepcopy(inherited["entries"])
        for point in review_report.get("points", []):
            if point.get("attempts") and point.get("status") != "complete":
                prior.pop(point.get("point_id"), None)
        cache["entries"] = {**prior, **cache["entries"]}
    return cache if cache["entries"] else None


def runtime_replay_context(work, sources, visual_report, model_id):
    if model_id != DOTS_ILLUSTRATION_MODEL_ID:
        return None
    root = Path(work)
    checkpoint = root / "checkpoint.json"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        return None
    cp = json.loads(checkpoint.read_text())
    frames = frame_hashes(cp["inputs"])
    _, policy = _policy_components()
    context = {
        "source_inputs_sha256": source_identity(sources, visual_report, cp["media_identity"], frames),
        "policy_sha256": policy, "frame_sha256_by_second": frames, "model_id": model_id,
        "cache": None,
    }
    receipt = optional_replay_receipt(cp.get("optional_inputs"))
    context["cache"] = load_bound_replay_cache(root, receipt)
    return context


def optional_replay_receipt(rows):
    if not isinstance(rows, list):
        return None
    matches = [r for r in rows if isinstance(r, dict)
               and r.get("path") == "illustration-replay-cache.json"]
    return matches[0] if len(matches) == 1 else None


def load_bound_replay_cache(root, receipt):
    """Discard damaged optional acceleration data; core inputs stay mandatory."""
    if not isinstance(receipt, dict) or receipt.get("path") != "illustration-replay-cache.json":
        return None
    root = Path(root)
    path = root / receipt["path"]
    try:
        if root.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 8*1024*1024:
            return None
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt.get("sha256") or len(raw) != receipt.get("size"):
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None
