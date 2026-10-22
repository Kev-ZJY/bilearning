"""Read-only knowledge and video projections; no database, network or provider access."""
from __future__ import annotations

import json
from typing import Any, Optional

MAX_SEARCH_CHARS = 6000


def read_outline(doc: Optional[dict]) -> dict:
    """读文档大纲：模块标题 + 知识点标题/anchor，**不含 body**。

    不含 body 是刻意的：大纲的用途是让 agent 知道「文档里有什么」，
    以便决定下一步检索什么。带上 body 就等于把全文再塞一遍，白花 token。
    """
    doc = doc if isinstance(doc, dict) else {}
    modules = doc.get("modules")
    if not isinstance(modules, list):
        modules = []

    out_modules: list[dict] = []
    point_count = 0
    for m in modules:
        # LLM 偶尔产出非 dict 元素（字符串/null），跳过而不是崩
        if not isinstance(m, dict):
            continue
        title = m.get("title")
        if not isinstance(title, str):
            continue
        points_raw = m.get("points")
        if not isinstance(points_raw, list):
            points_raw = []

        points: list[dict] = []
        for p in points_raw:
            if not isinstance(p, dict):
                continue
            points.append({
                "id": p.get("id") or "",
                "title": p.get("title") or "",
                "anchor": p.get("anchor") or "",
            })
        point_count += len(points)
        out_modules.append({
            "id": m.get("id") or "",
            "title": title,
            "points": points,
        })

    # 防御点 3：meta 可能是显式 null，取标题前必须判类型
    meta = doc.get("meta")
    doc_title = meta.get("title") if isinstance(meta, dict) else None

    result: dict[str, Any] = {
        "title": doc_title,
        "module_count": len(out_modules),
        "point_count": point_count,
        "modules": out_modules,
    }
    if not out_modules:
        result["note"] = (
            "知识文档还是空的。用户在问文档内容时，先提示他粘贴 B 站链接"
            "整理视频，或直接说明当前项目里没有可检索的内容，不要凭空回答。"
        )
    return result

def _iter_points(doc: Optional[dict]):
    """遍历文档里所有 (module, point)，跳过畸形结构。"""
    doc = doc if isinstance(doc, dict) else {}
    modules = doc.get("modules")
    if not isinstance(modules, list):
        return
    for m in modules:
        if not isinstance(m, dict):
            continue
        points = m.get("points")
        if not isinstance(points, list):
            continue
        for p in points:
            if isinstance(p, dict):
                yield m, p

def search_knowledge(doc: Optional[dict], query: str,
                     max_chars: int = MAX_SEARCH_CHARS) -> dict:
    """按关键词检索知识点，返回命中项的**完整 body**。

    检索范围是 title + anchor + body + warnings：用户问的是内容，
    关键词常只出现在正文里（「重试风暴」在标题中就没有）。

    超长时截断并**显式标注 truncated**——见模块文档约束 1。
    """
    q = (query or "").strip()
    if not q:
        return {
            # ok=False 是必须的：空查询是一次失败调用，标成成功会让模型
            # 以为「文档里确实没有内容」，而真实原因是它没给检索词。
            "ok": False,
            "query": query or "",
            "hit_count": 0,
            "hits": [],
            "truncated": False,
            "note": "检索词为空。请给出一个具体的关键词（如「幂等」「退避」），"
                    "不要用空查询把整个文档拉回来。",
        }

    # 多空格查询归一化：实测模型会拟出「野果 食用 安全」这种带空格的检索式，
    # 若按整串匹配会一个都搜不到。这里同时用整串与分词后的单词去匹配。
    tokens = [t for t in q.replace("　", " ").split(" ") if t]
    needles = [q] + tokens

    hits: list[dict] = []
    used_chars = 0
    truncated = False
    for m, p in _iter_points(doc):
        warnings = p.get("warnings")
        blob_parts = [str(p.get("title") or ""), str(p.get("anchor") or ""),
                      str(p.get("body") or "")]
        if isinstance(warnings, list):
            blob_parts.extend(str(w) for w in warnings)
        blob = " ".join(blob_parts)

        if not any(n in blob for n in needles):
            continue

        item = {
            "module_id": m.get("id") or "",
            "module_title": m.get("title") or "",
            "point_id": p.get("id") or "",
            "title": p.get("title") or "",
            "anchor": p.get("anchor") or "",
            "body": p.get("body") or "",
            "warnings": warnings if isinstance(warnings, list) else [],
        }
        # 防御点 1：超长截断，但要让模型知道被截断了
        size = len(json.dumps(item, ensure_ascii=False))
        if used_chars + size > max_chars and hits:
            truncated = True
            break
        hits.append(item)
        used_chars += size

    result = {
        "query": q,
        "hit_count": len(hits),
        "hits": hits,
        "truncated": truncated,
    }
    if truncated:
        result["note"] = (
            f"检索结果超过 {max_chars} 字符已截断，可能还有相关内容没读到。"
            "若信息不足，请换更具体的关键词再检索一次，不要基于残缺内容下结论。"
        )
    elif not hits:
        result["note"] = (
            f"没有命中「{q}」。请先 read_outline 看文档里到底有什么，"
            "再换用文档里实际出现的词检索。**不要因为搜不到就凭空回答**——"
            "应当直接告诉用户文档里没有这部分内容。"
        )
    return result

def get_point(doc: Optional[dict], point_id: str,
              module_id: Optional[str] = None) -> dict:
    """按 id 取知识点全文。

    **返回全部同名 id**（防御点 4）：实测不同模块里会出现同名 point id，
    只取第一个会让 agent 拿到错误内容并当成唯一事实。给了 ``module_id``
    时用它收窄，让调用方能精确指定。
    """
    pid = (point_id or "").strip()
    matches: list[dict] = []
    for m, p in _iter_points(doc):
        if str(p.get("id") or "") != pid:
            continue
        if module_id and str(m.get("id") or "") != module_id:
            continue
        warnings = p.get("warnings")
        matches.append({
            "module_id": m.get("id") or "",
            "module_title": m.get("title") or "",
            "point_id": p.get("id") or "",
            "title": p.get("title") or "",
            "anchor": p.get("anchor") or "",
            "body": p.get("body") or "",
            "warnings": warnings if isinstance(warnings, list) else [],
        })

    result: dict[str, Any] = {
        "point_id": pid,
        "found": bool(matches),
        "match_count": len(matches),
        "matches": matches,
    }
    if not matches:
        result["note"] = (
            f"没有找到 id 为「{pid}」的知识点。请先 read_outline 看实际的 id，"
            "不要拿不存在的 id 继续推断内容。"
        )
    elif len(matches) > 1:
        result["note"] = (
            f"有 {len(matches)} 个知识点的 id 都是「{pid}」（来自不同模块），"
            "已全部返回。回答时请说清是哪个模块里的，避免张冠李戴。"
        )
    return result

def _fmt_duration(seconds: Optional[float]) -> str:
    """把秒数格式化成人类可读文本。

    防御：None / 非数值都要能过。``str(None)`` 会得到字符串 "None"，
    塞进 prompt 会让模型以为视频时长真叫 None。
    """
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return ""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}小时{m}分{s}秒"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"

def list_videos(videos: list[dict]) -> dict:
    """列出项目里的视频：id / 标题 / 状态 / 时长。

    时长转成「3分12秒」这类文本，是因为模型判断「该不该整理这个视频」
    时对人类可读时长的理解远好于裸秒数（实测顶层 duration 对多P 合集
    是全部分P 之和，容易误导——但那属于流水线的问题，这里如实展示）。
    """
    items: list[dict] = []
    for v in (videos or []):
        if not isinstance(v, dict):
            continue
        items.append({
            "video_id": v.get("id"),
            "title": v.get("title") or "",
            "bvid": v.get("bvid") or "",
            "status": v.get("status") or "",
            "duration_text": _fmt_duration(v.get("duration")),
        })
    result: dict[str, Any] = {"count": len(items), "videos": items}
    if not items:
        result["note"] = (
            "这个项目里还没有视频。请提示用户把 B 站链接粘到上方输入框，"
            "整理完成后才会有知识文档。"
        )
    return result
