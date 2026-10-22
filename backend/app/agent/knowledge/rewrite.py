"""知识改写：共享结构和写作门禁，失败不返回可写入文档。"""
from __future__ import annotations

import copy
import json
from typing import Any, Optional

from app.agent.knowledge.extractor import normalize_modules
from app.agent.knowledge.document_contract import validate_document, WRITING_RULES
from app.agent.runtime.context import document_for_model

# 模块数量允许的最大缩水比例。低于这个比例视为「内容丢失」而非「合并冗余」。
#
# 为什么需要：agent 可能误读用户意图（「精简一下」被当成「砍掉一半」）。
# 合并确实重复的模块是合理需求，所以不禁止任何减少；但砍掉一半以上
# 几乎一定是出了问题，必须拦下问用户。
MAX_MODULE_SHRINK_RATIO = 0.5

# 文档至少要保留这么多模块，否则视为清空数据。
MIN_MODULES = 1


class RewriteRejected(Exception):
    """改写结果未通过校验。

    只在 ``require_success=True`` 时抛出。默认返回 ``ok=False`` 的字典，
    让调用方自行决定怎么告知用户。
    """

    def __init__(self, issues: list[str], attempts: int):
        self.issues = issues
        self.attempts = attempts
        super().__init__(
            f"改写未通过校验（尝试 {attempts} 次）：" + "；".join(issues)
        )


def _count_modules(doc: Any) -> int:
    if not isinstance(doc, dict):
        return 0
    mods = doc.get("modules")
    return len(mods) if isinstance(mods, list) else 0


def _check(doc: Any, original_modules: int) -> list[str]:
    """结构与规范校验，返回问题列表（空表示通过）。"""
    issues: list[str] = []

    if not isinstance(doc, dict):
        return [f"返回的不是文档对象（得到 {type(doc).__name__}），"
                "请输出完整的 JSON 文档对象"]

    n = _count_modules(doc)
    if n < MIN_MODULES:
        return [f"文档里一个模块都没有了（原文有 {original_modules} 个）。"
                "改写不能把内容清空，请保留原有模块并只修改文字"]

    if n < original_modules * (1 - MAX_MODULE_SHRINK_RATIO):
        issues.append(
            f"模块数从 {original_modules} 降到 {n}，缩水过多，"
            "这更像是内容丢失而不是合并冗余。请保留原有模块，"
            "只修改文字内容，不要重组结构。"
        )

    issues.extend(validate_document(doc))

    return issues


def _build_prompt(original: dict, intent: str, issues: Optional[list[str]] = None) -> str:
    """构造改写提示词。

    强调「最小改动」是实测踩过的坑：verify 阶段模型把整份文档逐字重写、
    越写越长，四档 token 预算全被截断，挂了近 15 分钟。改写场景同理。
    """
    base = f"""你是知识文档的改写助手。按用户要求修改下面这份知识点文档。

## 用户要求
{intent}

## 统一写作要求
{WRITING_RULES}

## 改写纪律（重要，违反即为失败）

**只改文字，不改结构**：
- 模块数、知识点数、模块 id、知识点 id、顺序必须与原文**完全一致**
- 不要新增或删除任何模块、知识点
- 不要把一个模块拆成两个，也不要把两个合并成一个

**输出规模**：不得超过原文篇幅的 1.2 倍。如果你在大段重写，
说明方向错了，请改用最小改动——只替换确实需要改写的句子。

**保留原文已有的好内容**：反直觉提醒、对照表、具体例子都不要弄丢。
除非用户明确要求删除。
图片的 stored-figure 引用是已有附件，只能保留引用或按用户要求移除，不得编写图片字节或替换来源。

## 原文（完整 JSON）
{json.dumps(document_for_model(original), ensure_ascii=False, indent=2)}

## 输出
只输出改写后的**完整 JSON 文档**，不要任何解释文字、不要代码围栏。
"""
    if issues:
        base += f"""
## 上一版输出的问题（必须逐条修正）
{chr(10).join('- ' + i for i in issues)}
"""
    return base


def rewrite_document(original: dict, current: dict, llm: Any, *,
                     intent: str, max_attempts: int = 2,
                     require_success: bool = False) -> dict:
    """让 LLM 按用户意图改写文档，校验通过才返回结果。

    :param original: 用户的原始文档（**不会被修改**，只读参考）
    :param current: 当前文档（改写基线）
    :param llm: LLMClient 实例，用它的 ``chat()``（要结构化 JSON）
    :param intent: 用户意图原文，如「把第 2 个模块写详细点」
    :param max_attempts: 最多尝试几次（含首次）。默认 2 = 一次重试。
    :param require_success: 失败时抛 :class:`RewriteRejected` 而非返回 ok=False
    :returns: ``{"ok": True, "doc": ..., "attempts": n}`` 或
        ``{"ok": False, "issues": [...], "attempts": n}``（**失败时不含 doc**）
    """
    # 深拷贝：normalize_modules 会就地写 doc["modules"]，作用在原文上就改坏了原文
    base = copy.deepcopy(current) if isinstance(current, dict) else {}
    original_modules = _count_modules(base)
    issues: list[str] = []
    attempts = 0

    for attempts in range(1, max_attempts + 1):
        prompt = _build_prompt(base, intent, issues if attempts > 1 else None)
        # 改写返回的是结构化 JSON，走 chat()（不是 chat_text）
        candidate = llm.chat([{"role": "user", "content": prompt}])
        attachment_issues = []

        # 同样深拷贝：模型返回的对象可能与 base 有共享结构，
        # 而 normalize_modules 会就地改它
        if isinstance(candidate, dict):
            candidate = copy.deepcopy(candidate)
            # The model sees attachment references, never megabytes of base64.
            # Restore bytes only for figures belonging to the same existing point.
            original_points = {p.get("id"): p for m in base.get("modules", []) for p in m.get("points", [])}
            for module in candidate.get("modules", []) if isinstance(candidate.get("modules"), list) else []:
                if not isinstance(module, dict):
                    continue
                for point in module.get("points", []) if isinstance(module.get("points"), list) else []:
                    if not isinstance(point, dict):
                        continue
                    previous = original_points.get(point.get("id"), {})
                    stored = {f.get("id"): f for f in previous.get("figures", [])}
                    if stored and not point.get("figures"):
                        point["figures"] = copy.deepcopy(previous["figures"])
                    if stored and previous.get("figure_decision"):
                        point["figure_decision"] = copy.deepcopy(previous["figure_decision"])
                    if isinstance(point.get("figures"), list) and {f.get("id") for f in point["figures"] if isinstance(f, dict)} != set(stored):
                        attachment_issues.append("文字改写必须保留原有图片附件，不能增加、删减或替换图片")
                    for figure in point.get("figures", []) if isinstance(point.get("figures"), list) else []:
                        if isinstance(figure, dict) and figure.get("id") in stored:
                            # Frame identity and bytes travel together; a model
                            # must not relabel the same pixels as another second.
                            for key in ("src", "kind", "second", "source_ref"):
                                figure[key] = stored[figure["id"]][key]
            if not validate_document(candidate):
                normalize_modules(candidate)

        issues = _check(candidate, original_modules) + attachment_issues
        if not issues:
            return {"ok": True, "doc": candidate, "attempts": attempts}

    # 失败：**故意不返回 doc**，调用方无法误落库
    failed = {"ok": False, "issues": issues, "attempts": attempts}
    if require_success:
        raise RewriteRejected(issues, attempts)
    return failed
