"""Pure construction and binding helpers for isolated knowledge review.

The projection intentionally contains only learner-visible candidate text,
examples, and candidate figures. It never resolves paths or external image URLs.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass

from app.agent.knowledge.document_contract import MAX_FIGURE_BYTES

BLIND_REVIEW_MODE = "blind_candidate_v1"
BLIND_REVIEW_TOOL_NAMES = ("search_web", "read_web", "view_candidate_figure")
MAX_CANDIDATE_IMAGE_BYTES = 4_000_000


from app.agent.tooling.tool_definitions import _tool, blind_review_tool_schemas

BLIND_REVIEW_TOOLS = blind_review_tool_schemas()


class BlindProjectionError(ValueError):
    """Candidate cannot be represented without exposing source metadata."""


@dataclass(frozen=True)
class BlindProjection:
    document: dict
    figure_assets: dict[str, dict[str, str]]
    point_id_map: dict[str, str]
    projection_hash: str

    def __iter__(self):
        """Allow simple (projection, assets, point-map, hash) unpacking."""
        yield self.document
        yield self.figure_assets
        yield self.point_id_map
        yield self.projection_hash


def _json_hash(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _image_payload(src: object) -> tuple[str, bytes]:
    if not isinstance(src, str) or not src.startswith("data:image/") or "," not in src:
        raise BlindProjectionError("盲审候选配图必须是内嵌图片，不能读取路径或远程URL")
    header, encoded = src.split(",", 1)
    mime = header.removeprefix("data:").split(";", 1)[0].lower()
    if mime not in {"image/jpeg", "image/png"} or ";base64" not in header.lower():
        raise BlindProjectionError("盲审候选配图类型无效")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise BlindProjectionError("盲审候选配图编码无效") from exc
    if not data or len(data) > MAX_CANDIDATE_IMAGE_BYTES or len(data) > MAX_FIGURE_BYTES:
        raise BlindProjectionError("盲审候选配图为空或超过体积上限")
    if mime == "image/jpeg" and not data.startswith(b"\xff\xd8\xff"):
        raise BlindProjectionError("盲审候选JPEG签名无效")
    if mime == "image/png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise BlindProjectionError("盲审候选PNG签名无效")
    return mime, data


_EXAMPLE_VISIBLE_KEYS = {
    "type", "content", "headers", "rows", "title", "nodes", "layout",
    "alt", "caption", "language", "label",
}


def _visible_example(value: object) -> object:
    if isinstance(value, list):
        return [_visible_example(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _visible_example(child)
        for key, child in value.items()
        if key in _EXAMPLE_VISIBLE_KEYS
    }


def build_projection(doc: dict) -> BlindProjection:
    """Return a source-blind view and opaque lookup for actual embedded figures."""
    if not isinstance(doc, dict):
        raise BlindProjectionError("候选文档必须为对象")
    modules = doc.get("modules")
    if not isinstance(modules, list):
        raise BlindProjectionError("候选文档缺少modules")

    assets: dict[str, dict[str, str]] = {}
    point_map: dict[str, str] = {}
    visible_modules = []
    item_index = 0
    figure_index = 0

    def project_point(point: object) -> dict:
        nonlocal item_index, figure_index
        if not isinstance(point, dict) or not isinstance(point.get("id"), str):
            raise BlindProjectionError("候选知识点缺少稳定ID")
        item_index += 1
        review_id = f"item-{item_index:04d}"
        point_map[review_id] = point["id"]
        figures = point.get("figures", [])
        if not isinstance(figures, list):
            raise BlindProjectionError("候选配图必须为数组")
        visible_figures = []
        for figure in figures:
            if not isinstance(figure, dict):
                raise BlindProjectionError("候选配图必须为对象")
            mime, data = _image_payload(figure.get("src"))
            sha = hashlib.sha256(data).hexdigest()
            figure_index += 1
            figure_id = f"candidate-figure-{figure_index:04d}-{sha[:12]}"
            data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            assets[figure_id] = {"mime_type": mime, "sha256": sha, "data_url": data_url}
            visible_figures.append({
                "figure_id": figure_id,
                "sha256": sha,
                "alt": figure.get("alt", ""),
                "caption": figure.get("caption", ""),
            })
        result = {"point_id": review_id}
        for key in ("title", "anchor", "body", "warnings"):
            if key in point:
                result[key] = point[key]
        examples = point.get("examples", [])
        if not isinstance(examples, list):
            raise BlindProjectionError("候选examples必须为数组")
        result["examples"] = [_visible_example(example) for example in examples]
        result["figures"] = visible_figures
        return result

    for module in modules:
        if not isinstance(module, dict) or not isinstance(module.get("points"), list):
            raise BlindProjectionError("候选模块结构无效")
        visible_modules.append({
            key: module[key] for key in ("title",) if isinstance(module.get(key), str)
        } | {"points": [project_point(point) for point in module["points"]]})

    appendix = doc.get("appendix") or {}
    if not isinstance(appendix, dict):
        raise BlindProjectionError("候选附录结构无效")
    tables = appendix.get("cheatsheets", [])
    if not isinstance(tables, list):
        raise BlindProjectionError("附录表格必须为数组")
    appendix_items = []
    for table in tables:
        if not isinstance(table, dict):
            raise BlindProjectionError("附录表格必须为对象")
        point = {
            "id": f"appendix:{len(appendix_items)}",
            "title": table.get("title", "速查表"),
            "body": "",
            "examples": [{
                key: _visible_example(table[key])
                for key in ("title", "headers", "rows") if key in table
            } | {"type": "table"}],
            "warnings": [],
            "figures": [],
        }
        appendix_items.append(project_point(point))

    projection = {
        "modules": visible_modules,
        "appendix_review_items": appendix_items,
    }
    return BlindProjection(
        document=projection,
        figure_assets=assets,
        point_id_map=point_map,
        projection_hash=_json_hash(projection),
    )


def binding_receipt(doc: dict, projection: BlindProjection, candidate_hash: str) -> dict:
    return {
        "mode": BLIND_REVIEW_MODE,
        "candidate_hash": candidate_hash,
        "projection_hash": projection.projection_hash,
        "tool_allowlist": list(BLIND_REVIEW_TOOL_NAMES),
    }


def validate_binding(doc: dict, receipt: object, candidate_hash: str) -> list[str]:
    if not isinstance(receipt, dict):
        return ["missing blind-review receipt"]
    issues = []
    if receipt.get("mode") != BLIND_REVIEW_MODE:
        issues.append("review mode is missing or not blind")
    if receipt.get("candidate_hash") != candidate_hash:
        issues.append("blind candidate hash does not match")
    try:
        expected = build_projection(doc)
    except BlindProjectionError as exc:
        return [f"blind projection cannot be recomputed: {exc}"]
    if receipt.get("projection_hash") != expected.projection_hash:
        issues.append("blind projection hash does not match")
    if receipt.get("tool_allowlist") != list(BLIND_REVIEW_TOOL_NAMES):
        issues.append("blind tool allowlist does not match")
    return issues
