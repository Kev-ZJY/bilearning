"""Bind the final merged document, its JSON persistence bytes, and rendered HTML.

The receipt keeps current-source review scope separate from project content that
was already present. The full formal-object hash retains display IDs and stable
review_point_id; only exact named audit metadata is omitted. Current-source
comparison normalizes display IDs under the verified mapping.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from app.worker.renderer import render_standalone_html


SCHEMA_VERSION = 1
FORMAL_JSON_NAME = "formal-output.json"
FORMAL_HTML_NAME = "formal-output.html"
FORMAL_MANIFEST_NAME = "formal-output-manifest.json"

# These are the exact root-meta fields added/maintained as verification/audit
# reports by the current pipeline. Other metadata (including title, positioning,
# source_videos, and unknown application metadata) stays in the semantic hash.
FORMAL_AUDIT_META_KEYS = frozenset({
    "verification",
    "visual_review",
    "independent_review",
    "illustrations",
    "reviews",
    "updated_at",
})

# Existing pipeline review semantics exclude these exact point-level provenance
# fields. Unknown point fields remain semantic and are therefore preserved.
FORMAL_POINT_AUDIT_KEYS = frozenset({
    "review_video_id",
    "review_task_id",
    "evidence_point_id",
    "source_quotes",
})

# ground_document attaches these root-level provenance fields to the candidate
# appendix, while the established document fusion contract carries only its
# tables into the formal project document. Exclude only these exact paths from
# current-source appendix semantics; all other appendix fields remain semantic.
FORMAL_APPENDIX_PROVENANCE_KEYS = frozenset({"evidence_point_id", "source_quotes"})
CURRENT_SOURCE_SCOPE_PROJECTION_VERSION = 2


def _fail(message: str) -> None:
    raise ValueError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _document_shape(document: Any, label: str) -> tuple[dict, list, dict]:
    if not isinstance(document, dict):
        _fail(f"{label}必须是对象")
    modules = document.get("modules")
    if not isinstance(modules, list):
        _fail(f"{label}.modules必须是数组")
    for index, module in enumerate(modules):
        if not isinstance(module, dict) or not isinstance(module.get("points"), list):
            _fail(f"{label}.modules[{index}]结构无效")
        for point_index, point in enumerate(module["points"]):
            if not isinstance(point, dict):
                _fail(f"{label}.modules[{index}].points[{point_index}]必须是对象")
    appendix = document.get("appendix")
    if appendix is None:
        appendix = {}
    if not isinstance(appendix, dict):
        _fail(f"{label}.appendix必须是对象")
    tables = appendix.get("cheatsheets", [])
    if not isinstance(tables, list):
        _fail(f"{label}.appendix.cheatsheets必须是数组")
    return document, modules, appendix


def _source_scope_point(point: dict) -> dict:
    """Normalize only display ID for mapping-based source-scope comparison."""
    return {
        key: copy.deepcopy(value)
        for key, value in point.items()
        if key != "id" and key not in FORMAL_POINT_AUDIT_KEYS
    }


def _semantic_module_metadata(module: dict) -> dict:
    return {
        key: copy.deepcopy(value)
        for key, value in module.items()
        if key not in {"id", "points"}
    }


def formal_object_semantic_projection(document: dict) -> dict:
    """Return full formal meaning, excluding only exact nonsemantic audit fields."""
    _document_shape(document, "正式文档")
    projected = copy.deepcopy(document)
    meta = projected.get("meta")
    if isinstance(meta, dict):
        for key in FORMAL_AUDIT_META_KEYS:
            meta.pop(key, None)
    for module in projected["modules"]:
        module["points"] = [
            {
                key: copy.deepcopy(value)
                for key, value in point.items()
                if key not in FORMAL_POINT_AUDIT_KEYS
            }
            for point in module["points"]
        ]
    return projected


def formal_object_semantic_hash(document: dict) -> str:
    return _sha256(_canonical_bytes(formal_object_semantic_projection(document)))


def _stable_source_points(candidate_modules: list[dict]) -> tuple[list[list[str]], dict[str, dict]]:
    stable_by_module: list[list[str]] = []
    candidate_by_stable: dict[str, dict] = {}
    candidate_display_ids: set[str] = set()
    for module_index, module in enumerate(candidate_modules):
        points = module["points"]
        if not points:
            _fail(f"当前来源候选模块[{module_index}]没有知识点")
        stable_ids = []
        for point in points:
            stable_id = point.get("review_point_id")
            if not isinstance(stable_id, str) or not stable_id.strip():
                _fail("当前来源候选缺少稳定review_point_id")
            if stable_id in candidate_by_stable:
                _fail("当前来源候选review_point_id重复：" + stable_id)
            display_id = point.get("id")
            if isinstance(display_id, str):
                if display_id in candidate_display_ids:
                    _fail("当前来源候选display point ID重复：" + display_id)
                candidate_display_ids.add(display_id)
            stable_ids.append(stable_id)
            candidate_by_stable[stable_id] = point
        stable_by_module.append(stable_ids)
    if not candidate_by_stable:
        _fail("当前来源候选没有可绑定的知识点")
    return stable_by_module, candidate_by_stable


def _validate_stable_map(
    formal_modules: list[dict],
    candidate_by_stable: dict[str, dict],
    stable_map: Mapping[str, str],
) -> tuple[dict[str, tuple[int, dict]], dict[str, str]]:
    if not isinstance(stable_map, Mapping):
        _fail("stable review ID映射必须是对象")
    expected = set(candidate_by_stable)
    if set(stable_map) != expected:
        _fail("stable review ID映射未精确覆盖当前来源候选")
    normalized_map: dict[str, str] = {}
    formal_id_rows: dict[str, list[tuple[int, dict]]] = {}
    formal_stable_rows: dict[str, list[tuple[int, dict]]] = {}
    for module_index, module in enumerate(formal_modules):
        for point in module["points"]:
            display_id = point.get("id")
            if isinstance(display_id, str):
                formal_id_rows.setdefault(display_id, []).append((module_index, point))
            stable_id = point.get("review_point_id")
            if isinstance(stable_id, str) and stable_id in expected:
                formal_stable_rows.setdefault(stable_id, []).append((module_index, point))

    values = []
    for stable_id in sorted(expected):
        formal_id = stable_map[stable_id]
        if not isinstance(formal_id, str) or not formal_id:
            _fail("stable review ID映射包含无效formal display ID")
        rows = formal_id_rows.get(formal_id, [])
        candidate_semantics = _source_scope_point(candidate_by_stable[stable_id])
        stable_rows = [
            row for row in formal_stable_rows.get(stable_id, [])
            if _source_scope_point(row[1]) == candidate_semantics
        ]
        if (len(rows) != 1 or len(stable_rows) != 1
                or rows[0] != stable_rows[0] or rows[0][1].get("id") != formal_id):
            _fail("stable review ID无法唯一绑定到正式知识点：" + stable_id)
        normalized_map[stable_id] = formal_id
        values.append(formal_id)
    if len(values) != len(set(values)):
        _fail("多个stable review ID映射到同一formal display ID")
    by_stable = {
        stable_id: next(
            row for row in formal_stable_rows.get(stable_id, [])
            if row[1].get("id") == formal_id
            and _source_scope_point(row[1]) == _source_scope_point(candidate_by_stable[stable_id])
        )
        for stable_id, formal_id in normalized_map.items()
    }
    return by_stable, normalized_map


def _parse_appendix_id(value: Any, *, label: str) -> int:
    if not isinstance(value, str) or not value.startswith("appendix:"):
        _fail(f"{label}必须使用appendix:index格式")
    raw_index = value[len("appendix:"):]
    if not raw_index.isdigit():
        _fail(f"{label}索引无效")
    return int(raw_index)


def _validate_appendix_map(
    candidate_appendix: dict,
    formal_appendix: dict,
    appendix_map: Mapping[str, str],
) -> tuple[dict, dict]:
    candidate_tables = candidate_appendix.get("cheatsheets", [])
    formal_tables = formal_appendix.get("cheatsheets", [])
    if not isinstance(appendix_map, Mapping):
        _fail("当前来源附录映射必须是对象")
    expected_keys = {f"appendix:{index}" for index in range(len(candidate_tables))}
    if set(appendix_map) != expected_keys:
        _fail("当前来源附录映射未精确覆盖候选速查表")

    formal_indices = []
    normalized: dict[str, str] = {}
    formal_source_tables = []
    for index, candidate_table in enumerate(candidate_tables):
        source_id = f"appendix:{index}"
        formal_id = appendix_map[source_id]
        formal_index = _parse_appendix_id(formal_id, label="正式附录ID")
        if formal_index >= len(formal_tables):
            _fail("当前来源附录映射超出正式速查表范围")
        if formal_tables[formal_index] != candidate_table:
            _fail("正式文档改变了已审查的当前来源附录：" + source_id)
        formal_indices.append(formal_index)
        normalized[source_id] = formal_id
        formal_source_tables.append(copy.deepcopy(formal_tables[formal_index]))
    if len(formal_indices) != len(set(formal_indices)):
        _fail("多个当前来源附录映射到同一正式速查表")
    if any(right <= left for left, right in zip(formal_indices, formal_indices[1:])):
        _fail("正式文档改变了已审查附录顺序")

    # Non-table appendix fields are also current-source semantics, except the
    # two exact root provenance fields that grounding attaches but merge_docs
    # intentionally does not carry into the project document. Existing
    # project-only fields may remain in the formal appendix, but every other
    # field supplied by this source candidate must survive unchanged.
    for key, value in candidate_appendix.items():
        if key == "cheatsheets" or key in FORMAL_APPENDIX_PROVENANCE_KEYS:
            continue
        if formal_appendix.get(key) != value:
            _fail("正式文档改变了已审查附录元数据：" + str(key))
    projected = copy.deepcopy(candidate_appendix)
    for key in FORMAL_APPENDIX_PROVENANCE_KEYS:
        projected.pop(key, None)
    projected["cheatsheets"] = formal_source_tables
    return projected, normalized


def _source_scope_projections(
    reviewed_candidate: dict,
    formal_document: dict,
    stable_map: Mapping[str, str],
    appendix_map: Mapping[str, str],
) -> tuple[dict, dict, dict, dict]:
    _candidate, candidate_modules, candidate_appendix = _document_shape(
        reviewed_candidate, "已审查当前来源候选"
    )
    _formal, formal_modules, formal_appendix = _document_shape(
        formal_document, "正式文档"
    )
    groups, candidate_by_stable = _stable_source_points(candidate_modules)
    formal_by_stable, stable_map_sorted = _validate_stable_map(
        formal_modules, candidate_by_stable, stable_map
    )

    candidate_groups = []
    formal_groups = []
    formal_positions = []
    expected_stable = set(candidate_by_stable)
    for module_index, (candidate_module, stable_ids) in enumerate(zip(candidate_modules, groups)):
        rows = [formal_by_stable[stable_id] for stable_id in stable_ids]
        formal_module_indexes = {row[0] for row in rows}
        if len(formal_module_indexes) != 1:
            _fail("项目融合改变了当前来源知识点的模块分组")
        formal_module_index = next(iter(formal_module_indexes))
        formal_module = formal_modules[formal_module_index]
        formal_positions.append(formal_module_index)
        formal_stable_ids = [point.get("review_point_id") for point in formal_module["points"]]
        if formal_stable_ids != stable_ids:
            _fail("正式模块改变了当前来源知识点的分组或顺序")
        candidate_module_meta = _semantic_module_metadata(candidate_module)
        formal_module_meta = _semantic_module_metadata(formal_module)
        if formal_module_meta != candidate_module_meta:
            _fail("项目融合改变了已审查当前来源模块元数据")
        candidate_groups.append({
            **candidate_module_meta,
            "points": [_source_scope_point(candidate_by_stable[sid]) for sid in stable_ids],
        })
        formal_groups.append({
            **formal_module_meta,
            "points": [_source_scope_point(formal_by_stable[sid][1]) for sid in stable_ids],
        })
        for stable_id in stable_ids:
            if _source_scope_point(candidate_by_stable[stable_id]) != _source_scope_point(formal_by_stable[stable_id][1]):
                _fail("项目融合改变了已审查知识点正文、例子或配图：" + stable_id)

    if any(right <= left for left, right in zip(formal_positions, formal_positions[1:])):
        _fail("项目融合改变了已审查当前来源模块顺序")
    candidate_appendix_projection, appendix_map_sorted = _validate_appendix_map(
        candidate_appendix, formal_appendix, appendix_map
    )
    candidate_scope = {
        "modules": candidate_groups,
        "appendix": candidate_appendix_projection,
    }
    formal_scope = {
        "modules": formal_groups,
        "appendix": copy.deepcopy(candidate_appendix_projection),
    }
    if candidate_scope != formal_scope:
        _fail("正式文档的当前来源语义投影与已审查候选不一致")

    mapped_formal_points = {
        (module_index, id(point)) for module_index, point in formal_by_stable.values()
    }
    outside_points = []
    for module_index, module in enumerate(formal_modules):
        for point in module["points"]:
            if (module_index, id(point)) not in mapped_formal_points:
                display_id = point.get("id")
                if not isinstance(display_id, str) or not display_id:
                    _fail("正式文档范围外知识点缺少display ID")
                outside_points.append(display_id)
    return candidate_scope, formal_scope, stable_map_sorted, {
        "source_appendix_id_to_formal_id": appendix_map_sorted,
        "outside_point_ids": outside_points,
        "current_stable_ids": sorted(expected_stable),
    }


def _sha_record(data: bytes, artifact: str) -> dict[str, Any]:
    return {"artifact": artifact, "sha256": _sha256(data), "utf8_bytes": len(data)}


def _database_json_bytes(formal_document: dict, supplied: bytes) -> bytes:
    if not isinstance(supplied, bytes):
        _fail("正式数据库JSON输入必须为UTF-8 bytes")
    try:
        decoded = supplied.decode("utf-8")
        parsed = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("正式数据库JSON输入不是有效UTF-8 JSON") from exc
    if parsed != formal_document:
        _fail("数据库序列化输入与实际正式合并文档不一致")
    expected = json.dumps(formal_document, ensure_ascii=False).encode("utf-8")
    if supplied != expected:
        _fail("数据库序列化bytes不是正式对象的实际持久化格式")
    return supplied


def _render_bytes(formal_document: dict) -> bytes:
    before = formal_object_semantic_hash(formal_document)
    rendered = render_standalone_html(formal_document)
    if not isinstance(rendered, str):
        _fail("正式HTML renderer必须返回字符串")
    if formal_object_semantic_hash(formal_document) != before:
        _fail("正式HTML renderer改变了正式合并对象")
    return rendered.encode("utf-8")


def _build_manifest(
    reviewed_candidate: dict,
    formal_document: dict,
    stable_map: Mapping[str, str],
    appendix_map: Mapping[str, str],
    database_json_bytes: bytes,
) -> tuple[dict, bytes, bytes]:
    candidate_scope, formal_scope, stable_map_sorted, receipt_scope = _source_scope_projections(
        reviewed_candidate, formal_document, stable_map, appendix_map
    )
    database_bytes = _database_json_bytes(formal_document, database_json_bytes)
    html_bytes = _render_bytes(formal_document)
    candidate_scope_hash = _sha256(_canonical_bytes(candidate_scope))
    formal_scope_hash = _sha256(_canonical_bytes(formal_scope))
    if candidate_scope_hash != formal_scope_hash:
        _fail("当前来源候选与正式文档来源范围hash不一致")
    formal_semantic_hash = formal_object_semantic_hash(formal_document)
    current_ids = set(receipt_scope["current_stable_ids"])
    outside_point_ids = receipt_scope["outside_point_ids"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "database_commit": {"status": "pending"},
        "formal_object": {
            "semantic_sha256": formal_semantic_hash,
            "semantic_projection_version": 1,
            "excluded_metadata": {
                "root_meta_keys": sorted(FORMAL_AUDIT_META_KEYS),
                "module_display_key": "retained in full formal hash",
                "point_display_key": "retained in full formal hash",
                "stable_review_point_id": "retained in full formal and current-source hashes",
                "source_scope_display_id_normalization": "verified stable review ID mapping",
                "point_audit_keys": sorted(FORMAL_POINT_AUDIT_KEYS),
            },
            "database_json": _sha_record(database_bytes, FORMAL_JSON_NAME),
            "html": {
                **_sha_record(html_bytes, FORMAL_HTML_NAME),
                "renderer": "app.worker.renderer.render_standalone_html",
                "render_input_database_json_sha256": _sha256(database_bytes),
                "render_input_semantic_sha256": formal_semantic_hash,
            },
        },
        "current_source_scope": {
            "scope": "freshly reviewed current-source candidate only",
            "projection_version": CURRENT_SOURCE_SCOPE_PROJECTION_VERSION,
            "excluded_appendix_provenance_keys": sorted(
                FORMAL_APPENDIX_PROVENANCE_KEYS
            ),
            "fresh_reviewed": True,
            "point_count": len(current_ids),
            "stable_review_point_ids": receipt_scope["current_stable_ids"],
            "module_count": len(candidate_scope["modules"]),
            "appendix_table_count": len(candidate_scope["appendix"].get("cheatsheets", [])),
            "candidate_semantic_sha256": candidate_scope_hash,
            "formal_semantic_sha256": formal_scope_hash,
            "semantic_sha256": candidate_scope_hash,
            "stable_review_point_id_to_formal_id": stable_map_sorted,
            "source_appendix_id_to_formal_id": receipt_scope[
                "source_appendix_id_to_formal_id"
            ],
        },
        "outside_current_source_scope": {
            "scope": "formal points without a stable current-source review mapping",
            "point_ids": outside_point_ids,
            "point_count": len(outside_point_ids),
            "freshly_reviewed": False,
        },
        "project_history_freshly_reviewed": False,
    }
    return manifest, database_bytes, html_bytes


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bind_formal_output_receipts(
    *,
    reviewed_candidate: dict,
    formal_document: dict,
    stable_review_point_id_to_formal_id: Mapping[str, str],
    source_appendix_id_to_formal_id: Mapping[str, str],
    database_json_bytes: bytes,
    work: str | Path,
) -> dict:
    """Create the final JSON/HTML artifacts and a separate non-self-hashing manifest.

    ``database_json_bytes`` must be the exact bytes the existing document writer
    persists (currently ``json.dumps(formal_document, ensure_ascii=False)``).
    The HTML artifact is rendered here from that same actual merged object.
    """
    manifest, json_bytes, html_bytes = _build_manifest(
        reviewed_candidate,
        formal_document,
        stable_review_point_id_to_formal_id,
        source_appendix_id_to_formal_id,
        database_json_bytes,
    )
    root = Path(work).resolve()
    _atomic_write(root / FORMAL_JSON_NAME, json_bytes)
    _atomic_write(root / FORMAL_HTML_NAME, html_bytes)
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")
    _atomic_write(root / FORMAL_MANIFEST_NAME, manifest_bytes)
    return copy.deepcopy(manifest)


def mark_formal_output_committed(
    *,
    work: str | Path,
    persisted_document_json_bytes: bytes,
) -> dict:
    """Mark the manifest committed only after caller confirms the DB transaction.

    The caller must invoke this after a successful database commit and provide
    the exact ``doc_json`` bytes read back from that persisted row. This method
    validates the prepared JSON/HTML artifacts and their recorded hashes before
    changing the separate manifest; it never changes the hashed formal JSON.
    """
    if not isinstance(persisted_document_json_bytes, bytes):
        _fail("已持久化正式文档必须以原始bytes传入")
    root = Path(work).resolve()
    try:
        manifest = json.loads((root / FORMAL_MANIFEST_NAME).read_text(encoding="utf-8"))
        json_bytes = (root / FORMAL_JSON_NAME).read_bytes()
        html_bytes = (root / FORMAL_HTML_NAME).read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("正式文档提交回执工件缺失或无效") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        _fail("正式文档提交回执manifest版本无效")
    if manifest.get("status") != "prepared" or manifest.get("database_commit") != {"status": "pending"}:
        _fail("正式文档回执不是待提交状态")
    if persisted_document_json_bytes != json_bytes:
        _fail("数据库已持久化doc_json bytes与正式JSON artifact不一致")
    try:
        formal_document = json.loads(persisted_document_json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("数据库已持久化doc_json不是有效UTF-8 JSON") from exc
    _database_json_bytes(formal_document, persisted_document_json_bytes)
    formal_receipt = manifest.get("formal_object")
    if not isinstance(formal_receipt, dict):
        _fail("正式对象hash回执缺失")
    if formal_receipt.get("semantic_sha256") != formal_object_semantic_hash(formal_document):
        _fail("正式对象语义hash与数据库已持久化对象不一致")
    if formal_receipt.get("database_json") != _sha_record(json_bytes, FORMAL_JSON_NAME):
        _fail("正式JSON artifact与数据库提交hash回执不一致")
    html_receipt = formal_receipt.get("html")
    expected_html = _render_bytes(formal_document)
    if html_bytes != expected_html or not isinstance(html_receipt, dict):
        _fail("正式HTML artifact与数据库已持久化对象渲染结果不一致")
    expected_html_receipt = {
        **_sha_record(expected_html, FORMAL_HTML_NAME),
        "renderer": "app.worker.renderer.render_standalone_html",
        "render_input_database_json_sha256": _sha256(json_bytes),
        "render_input_semantic_sha256": formal_object_semantic_hash(formal_document),
    }
    if html_receipt != expected_html_receipt:
        _fail("正式HTML renderer输入/hash回执与数据库已持久化对象不一致")

    manifest["status"] = "committed"
    manifest["database_commit"] = {
        "status": "committed",
        "sha256": _sha256(persisted_document_json_bytes),
        "utf8_bytes": len(persisted_document_json_bytes),
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")
    _atomic_write(root / FORMAL_MANIFEST_NAME, manifest_bytes)
    return copy.deepcopy(manifest)


def verify_formal_output_receipts(
    *,
    reviewed_candidate: dict,
    formal_document: dict,
    stable_review_point_id_to_formal_id: Mapping[str, str],
    source_appendix_id_to_formal_id: Mapping[str, str],
    database_json_bytes: bytes,
    work: str | Path,
) -> dict:
    """Rebind-check DB JSON bytes, semantic projections, and actual renderer output.

    Call inside the formal DB transaction after the row has been written/flushed,
    passing the exact persisted ``doc_json`` bytes before commit. A stale/mutated
    document, serialized payload, artifact, mapping, manifest, or renderer output
    is rejected so the caller can abort the transaction.
    """
    expected_manifest, expected_json, expected_html = _build_manifest(
        reviewed_candidate,
        formal_document,
        stable_review_point_id_to_formal_id,
        source_appendix_id_to_formal_id,
        database_json_bytes,
    )
    root = Path(work).resolve()
    try:
        saved_manifest = json.loads((root / FORMAL_MANIFEST_NAME).read_text(encoding="utf-8"))
        saved_json = (root / FORMAL_JSON_NAME).read_bytes()
        saved_html = (root / FORMAL_HTML_NAME).read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("正式文档回执工件缺失或无效") from exc
    status = saved_manifest.get("status") if isinstance(saved_manifest, dict) else None
    if status == "committed":
        expected_manifest["status"] = "committed"
        expected_manifest["database_commit"] = {
            "status": "committed",
            "sha256": _sha256(database_json_bytes),
            "utf8_bytes": len(database_json_bytes),
        }
    elif status != "prepared":
        _fail("正式文档manifest状态无效")
    if saved_manifest != expected_manifest:
        _fail("正式文档manifest与当前合并对象或来源映射不一致")
    if saved_json != expected_json:
        _fail("正式文档JSON artifact bytes/hash与数据库序列化输入不一致")
    if saved_html != expected_html:
        _fail("正式HTML artifact bytes/hash与当前正式对象renderer输出不一致")
    return copy.deepcopy(saved_manifest)
