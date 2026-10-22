import json
import re
from enum import Enum
from difflib import SequenceMatcher


class MergeType(Enum):
    DUPLICATE = "duplicate"
    ENRICH = "enrich"
    NEW_POINT = "new_point"
    NEW_MODULE = "new_module"


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


# 知识点标题是「主题词 + 说明」的结构（"OAuth2 授权码流程" → 主题词 "OAuth2"）。
# 中文标题没有空格，str.split() 完全失效，所以按中英文都能识别的边界切：
# 空格、连字符、以及中文里表示"引出细节"的助词（的/中/如何/为什么/是什么）。
_TOPIC_BOUNDARY = re.compile(
    r"[\s\-_·|]+"                # 空格类
    r"|如何|为什么|是什么|的(?:原理|机制|流程|作用|区别)|"
    r"在|中(?:的)?|与|和|对|及|则"      # 介词/连词
)


def extract_topic(title: str) -> str:
    """从知识点标题里抽出主题词，作为新模块的默认名。

    "OAuth2 授权码流程" -> "OAuth2"
    "幂等性与重试安全"  -> "幂等性"
    "CSRF 攻击原理"     -> "CSRF"

    中文标题没有空格，不能用 str.split()。这里按中英文边界与中文助词切，
    取第一段。全部切不动时回退到原标题。
    """
    t = (title or "").strip()
    if not t:
        return "知识点"
    parts = [p for p in _TOPIC_BOUNDARY.split(t) if p.strip()]
    return parts[0].strip() if parts else t


def classify_merge(new_point: dict, existing: dict,
                   incoming_module: dict | None = None) -> tuple[MergeType, dict]:
    """判断新知识点与已有文档的关系。

    一份项目 = 一份持续演进的 HTML。新增视频时要区分：
    重复（同标题同内容，跳过）、补充（同标题内容更多，合并）、新知识点、新模块。
    """
    modules = (existing or {}).get("modules") or []
    for m in modules:
        for p in m.get("points", []):
            if p.get("title") == new_point.get("title"):
                # 同标题时，只有标题/锚点/正文/例子全部一致才算完全重复；
                # 任一不同（锚点措辞、正文详略、例子）都视为内容有补充，需融合。
                same = (
                    p.get("anchor") == new_point.get("anchor")
                    and p.get("body") == new_point.get("body")
                    and p.get("examples") == new_point.get("examples")
                )
                if same:
                    return MergeType.DUPLICATE, {}
                return MergeType.ENRICH, {
                    "module_id": m["id"],
                    "point_id": p.get("id"),
                    "strategy": "合并内容，保留更全的 body 与 examples 并集",
                }
        if _sim(m.get("title", ""), new_point.get("title", "")) > 0.5:
            return MergeType.NEW_POINT, {"module_id": m["id"]}
    if modules:
        # 主题明显不同 → 建议开新模块。模块名优先用 LLM 自己给的模块标题
        # （它看过整段转写稿，比我们从单个知识点标题猜得准），
        # 没有才从知识点标题抽主题词。
        title = (incoming_module or {}).get("title") or extract_topic(
            new_point.get("title", "")
        )
        return MergeType.NEW_MODULE, {"module_title": title}
    title = (incoming_module or {}).get("title") or extract_topic(
        new_point.get("title", "")
    )
    return MergeType.NEW_MODULE, {"module_title": title}


def _next_module_id(modules: list[dict]) -> str:
    """生成不与现有 id 冲突的模块 id。

    不能用 f"m{len(modules)+1}"——融合过程会边加边算，删过模块的文档
    会出现 id 碰撞，前端按 id 引用就会错位。
    """
    used = {m.get("id") for m in modules}
    n = len(modules) + 1
    while f"m{n}" in used:
        n += 1
    return f"m{n}"


def merge_docs(existing: dict, incoming: dict) -> tuple[dict, list[dict]]:
    """把新抽出的文档合并进已有文档。返回 (合并后文档, 变更报告)。

    meta 的处理：**已有标题优先，为空才用新文档的**。
    首次导入时 existing 是空壳，标题必须从 incoming 取，否则整份文档
    渲染出来没有标题（真实端到端踩过：LLM 产出了「缓存穿透重点清单」，
    合并后 meta 里只剩 source_videos 和 updated_at）。
    已有标题不覆盖，是因为「一个项目 = 一份持续演进的文档」，
    第二个视频的标题不该把第一个的顶掉。
    """
    result = json.loads(json.dumps(existing or {
        "meta": {}, "modules": [], "appendix": {"cheatsheets": []}
    }))
    result.setdefault("modules", [])

    # incoming 的 meta 可能是显式 null，先归一成 dict
    in_meta = (incoming or {}).get("meta")
    if not isinstance(in_meta, dict):
        in_meta = {}
    out_meta = result.get("meta")
    if not isinstance(out_meta, dict):
        out_meta = {}
        result["meta"] = out_meta
    for key, value in in_meta.items():
        # 只在缺失或为空时补，不覆盖已有内容
        if out_meta.get(key) in (None, "", [], {}):
            out_meta[key] = value

    report = []
    for m in (incoming or {}).get("modules", []):
        for p in m.get("points", []):
            mt, action = classify_merge(p, result, incoming_module=m)
            if mt == MergeType.DUPLICATE:
                report.append({"type": "skipped", "title": p["title"]})
            elif mt == MergeType.ENRICH:
                for em in result["modules"]:
                    if em["id"] == action["module_id"]:
                        for ep in em["points"]:
                            if ep.get("id") == action.get("point_id"):
                                ep["body"] = _longer(ep.get("body"), p.get("body"))
                                ep["examples"] = _merge_lists(
                                    ep.get("examples"), p.get("examples"))
                                ep["warnings"] = _merge_lists(
                                    ep.get("warnings"), p.get("warnings"))
                report.append({"type": "enriched", "title": p["title"]})
            elif mt == MergeType.NEW_POINT:
                target = next(x for x in result["modules"]
                              if x["id"] == action["module_id"])
                target.setdefault("points", []).append(p)
                report.append({"type": "added", "title": p["title"]})
            else:
                result["modules"].append({
                    "id": _next_module_id(result["modules"]),
                    "title": action["module_title"],
                    "points": [p],
                })
                report.append({"type": "new_module", "title": p["title"]})
    result.setdefault("appendix", {}).setdefault("cheatsheets", [])
    for cs in (incoming or {}).get("appendix", {}).get("cheatsheets", []):
        if cs not in result["appendix"]["cheatsheets"]:
            result["appendix"]["cheatsheets"].append(cs)
    return result, report


def _longer(a: str | None, b: str | None) -> str:
    return a if len(a or "") >= len(b or "") else (b or "")


def _merge_lists(a, b):
    out = list(a or [])
    for item in (b or []):
        if item not in out:
            out.append(item)
    return out


def assert_review_merge_unchanged(snapshot: dict, latest: dict, draft: dict, incoming: dict) -> None:
    """A reviewed replacement cannot overwrite concurrent edits to its targets."""
    titles = {p.get("title") for d in (draft, incoming) for m in d.get("modules", []) for p in m.get("points", [])}
    table_titles = {t.get("title") for d in (draft, incoming) for t in (d.get("appendix") or {}).get("cheatsheets", [])}

    def targets(doc):
        points = [{k: v for k, v in p.items() if k not in {"id", "evidence_point_id"}}
                  for m in doc.get("modules", []) for p in m.get("points", []) if p.get("title") in titles]
        tables = [t for t in (doc.get("appendix") or {}).get("cheatsheets", []) if t.get("title") in table_titles]
        return sorted(json.dumps(p, sort_keys=True, ensure_ascii=False) for p in points), sorted(json.dumps(t, sort_keys=True, ensure_ascii=False) for t in tables)

    if targets(snapshot) != targets(latest):
        raise ValueError("待替换的知识点或附录在审核期间已被修改；已保留最新文档和本次草稿，请重新整理")


def merge_reviewed_docs(existing: dict, incoming: dict, draft: dict) -> tuple[dict, list[dict]]:
    """Replace superseded content and append reviewed modules without regrouping them.

    A candidate has already passed whole-document review, so its module boundaries
    and module metadata are part of the reviewed result. Existing independent
    knowledge remains in its original modules; matching draft/candidate titles and
    revised tables are replaced by the candidate.
    """
    import copy

    replacement_titles = {
        p.get("title")
        for doc in (draft, incoming)
        for m in doc.get("modules", [])
        for p in m.get("points", [])
        if p.get("title")
    }
    preserved = copy.deepcopy(existing or {
        "meta": {}, "modules": [], "appendix": {"cheatsheets": []}
    })
    if not isinstance(preserved, dict):
        preserved = {"meta": {}, "modules": [], "appendix": {"cheatsheets": []}}
    preserved.setdefault("modules", [])

    # Keep the established meta contract: an existing title/value wins, while
    # incoming values fill fields that are absent or empty.
    in_meta = (incoming or {}).get("meta")
    if not isinstance(in_meta, dict):
        in_meta = {}
    out_meta = preserved.get("meta")
    if not isinstance(out_meta, dict):
        out_meta = {}
        preserved["meta"] = out_meta
    for key, value in in_meta.items():
        if out_meta.get(key) in (None, "", [], {}):
            out_meta[key] = value

    superseded = []
    kept_modules = []
    for module in preserved.get("modules", []):
        if not isinstance(module, dict):
            kept_modules.append(module)
            continue
        kept = []
        for point in module.get("points", []):
            if isinstance(point, dict) and point.get("title") in replacement_titles:
                superseded.append({
                    "type": "reviewed_replacement", "title": point.get("title")
                })
            else:
                kept.append(point)
        if kept:
            module["points"] = kept
            kept_modules.append(module)
    preserved["modules"] = kept_modules

    revised_tables = {
        table.get("title")
        for doc in (draft, incoming)
        for table in (doc.get("appendix") or {}).get("cheatsheets", [])
    }
    if preserved.get("appendix"):
        preserved["appendix"]["cheatsheets"] = [
            table for table in preserved["appendix"].get("cheatsheets", [])
            if table.get("title") not in revised_tables
        ]

    # Copy and append candidate modules as intact units. Keep incoming IDs when
    # available; only rewrite an ID when it would collide with retained content.
    incoming_modules = copy.deepcopy((incoming or {}).get("modules") or [])
    used_module_ids = {
        module.get("id") for module in preserved["modules"]
        if isinstance(module, dict) and module.get("id") is not None
    }
    used_point_ids = {
        point.get("id")
        for module in preserved["modules"] if isinstance(module, dict)
        for point in (module.get("points") or []) if isinstance(point, dict)
        if point.get("id") is not None
    }
    report = []
    for module in incoming_modules:
        if not isinstance(module, dict):
            preserved["modules"].append(module)
            continue

        module_id = module.get("id")
        if module_id is not None and module_id in used_module_ids:
            module["id"] = _next_module_id(preserved["modules"])
            module_id = module["id"]
        if module_id is not None:
            used_module_ids.add(module_id)

        points = module.get("points") or []
        for point_index, point in enumerate(points, 1):
            if not isinstance(point, dict):
                continue
            point_id = point.get("id")
            if point_id is not None and point_id in used_point_ids:
                prefix = module_id if isinstance(module_id, str) and module_id else "reviewed"
                replacement_id = f"{prefix}p{point_index}"
                suffix = point_index
                while replacement_id in used_point_ids:
                    suffix += 1
                    replacement_id = f"{prefix}p{suffix}"
                point["id"] = replacement_id
                point_id = replacement_id
            if point_id is not None:
                used_point_ids.add(point_id)
            report.append({
                "type": "new_module" if point_index == 1 else "added",
                "title": point.get("title"),
            })
        preserved["modules"].append(module)

    appendix = preserved.get("appendix")
    if not isinstance(appendix, dict):
        appendix = {}
        preserved["appendix"] = appendix
    tables = appendix.get("cheatsheets")
    if not isinstance(tables, list):
        tables = []
        appendix["cheatsheets"] = tables
    incoming_tables = ((incoming or {}).get("appendix") or {}).get("cheatsheets") or []
    for table in incoming_tables:
        if table not in tables:
            tables.append(copy.deepcopy(table))

    return preserved, superseded + report
