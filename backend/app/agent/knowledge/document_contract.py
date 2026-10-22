"""所有知识文档写入共用的结构与写作门禁。规则通过不等于事实正确。"""

from __future__ import annotations

import copy
import re
import base64
import binascii
import io
import math
from app.agent.knowledge.extractor import normalize_modules, validate_modules, validate_points

WRITING_RULES = """按知识逻辑组织，标题是可独立学习的陈述式技术断言，正文先给记忆锚点。
剔除课堂投票、举手统计、讲师履历、宣传、作业要求、赞助商段落及自己的核查日志；保留可提炼的技术结论。
用户临时提问在对话回答，不把疑问改造成自造的误区表。真实技术误区允许。
例子按需提供，速查表应能追溯到正文。术语、数字、代码和图表应结合实际证据核对；
字幕、OCR、ASR 都可能错误，冲突或证据不足须记录在独立证据报告，不能虚构已核实。
图节点、图说明、替代文字和图注与正文使用同一写作规范；直接表达知识，不叙述来源提供了多少信息或本次核查为什么没有补充内容。
将下列表述视为课程或处理流程元叙述并从知识说明中剔除：“原始材料”“转写稿中”“转写稿里”“作者日常”“课程会”“原计划提供”“按播放列表”“学习安排以”“学习旅程的结尾”。
普通技术术语中的“和”“视频”“分钟”“黑板”不构成违规。"""

SOURCE_AUDIT_NARRATIVE = (
    r"(?:现有|提供的|本次|原始)(?:来源|证据|材料|转写|字幕|画面)[^。；;\n]{0,24}(?:未|没有|不足|无法|只|仅)"
    r"|(?:来源|转写稿)(?:中|里)?(?:未|没有|只|仅)"
    # A source-specific correction or unresolved conflict is review process,
    # even if it is phrased as a clean knowledge claim or placed in another
    # user-facing field. Keep general ASR/OCR accuracy and noise mechanisms
    # available as technical content by requiring a concrete source locator
    # together with a correction, absence, or conflict judgment.
    r"|(?:转写(?:稿)?|字幕|音频(?:转写)?|画面|来源|证据|材料)"
    r"[^。；;\n]{0,40}(?:识别错误|识别有误|误识别|误写|应(?:当)?(?:为|改为|改成|是|以)|"
    r"以.{1,12}为准|不一致|无法(?:唯一)?确定|未(?:显示|出现|识别|提及)|"
    r"没有(?:显示|出现|识别|提及)|需要核实|待核实)"
    r"|(?:只|仅)出现在[^。；;\n]{0,20}(?:转写(?:稿)?|字幕|画面|音频(?:转写)?)(?:中|里)?"
)

META_NARRATIVE = re.compile(
    r"讲师(?:在|说|强调|介绍|履历)|视频中(?:使用|演示|讲)|黑板上(?:画|写)|接下来我们|这部分很重要|课堂投票|举手统计|课程宣传|作业要求|赞助商(?:介绍|段落)|经核查[，,]|核查日志"
    r"|原始材料|转写稿(?:中|里)|作者日常|课程会|原计划提供|按播放列表|学习安排以|学习旅程的结尾"
    + "|" + SOURCE_AUDIT_NARRATIVE
)


def prose_narrative_paths(value):
    """Return paths of visible strings rejected by the shared prose gate."""
    paths = []

    def visit(current, path=""):
        if isinstance(current, dict):
            # These values are evidence/audit provenance, not rendered prose.
            for key, nested in current.items():
                if key in ("source_quotes", "source_quote", "figure_decision", "src"):
                    continue
                child_path = f"{path}.{key}" if path else str(key)
                visit(nested, child_path)
        elif isinstance(current, list):
            for index, nested in enumerate(current):
                visit(nested, f"{path}[{index}]")
        elif isinstance(current, str) and META_NARRATIVE.search(current):
            paths.append(path or "$")

    visit(value)
    return paths


MAX_FIGURE_BYTES = 100 * 1024


def diagram_issues(example, point=None):
    """Validate a small declarative diagram, with no model-authored markup."""
    if not isinstance(example, dict) or example.get("type") != "diagram":
        return ["示意图必须为 diagram 对象"]
    issues = []
    if example.get("layout") not in ("parallel", "sequence"):
        issues.append("示意图布局只能为 parallel 或 sequence")
    nodes = example.get("nodes")
    if not isinstance(nodes, list) or not 2 <= len(nodes) <= 8 or any(not isinstance(node, str) or not node.strip() or len(node) > 120 for node in nodes):
        issues.append("示意图需要 2–8 个非空短文本节点，每个不超过 120 字")
    content = example.get("content")
    if not isinstance(content, str) or not content.strip() or len(content) > 1200:
        issues.append("示意图需要不超过 1200 字的文字说明")
    prose = [content] + (nodes if isinstance(nodes, list) else [])
    if any(isinstance(text, str) and META_NARRATIVE.search(text) for text in prose):
        issues.append("示意图节点和说明包含课堂元叙述或来源核查过程，应直接表达知识")
    if example.get("basis") != "illustrative" or example.get("label") != "解释性例子（非来源原例）":
        issues.append("示意图必须明确标为解释性例子，不能冒充视频原图")
    if any(key in example for key in ("svg", "html", "src", "url", "path")):
        issues.append("示意图不接受自由 HTML、SVG、图片地址或路径")
    if example.get("layout") == "sequence":
        quote = example.get("source_quote")
        source = "\n".join(str((point or {}).get(key, "")) for key in ("anchor", "body"))
        source += "\n" + "\n".join(q for q in (point or {}).get("source_quotes", []) if isinstance(q, str))
        if not isinstance(quote, str) or len(quote.strip()) < 4 or quote not in source or not re.search(r"先.{0,120}(?:再|后)|后.{0,120}再|然后|随后|之后|之前|接着|依次|\b(?:before|after|then|followed by)\b", quote, re.I):
            issues.append("顺序示意图必须逐字引用本点已核对知识中明确先后的来源，不得把并列关系画成流程")
    return issues


def has_illustration(point):
    """Only delivered valid frame images or declarative diagrams satisfy necessity."""
    figures = point.get("figures") if isinstance(point.get("figures"), list) else []
    examples = point.get("examples") if isinstance(point.get("examples"), list) else []
    return any(not figure_issues(figure) for figure in figures) or any(
        isinstance(example, dict) and example.get("type") == "diagram" and not diagram_issues(example, point)
        for example in examples
    )


def figure_issues(figure):
    """Accept only bounded, decodable embedded JPEGs, never model-controlled URLs."""
    if not isinstance(figure, dict):
        return ["插图必须为对象"]
    issues = []
    if figure.get("kind") != "video_frame":
        issues.append("插图只能引用已核对的视频帧")
    if not isinstance(figure.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", figure["id"]):
        issues.append("插图 id 格式无效")
    for field in ("alt", "caption"):
        if not isinstance(figure.get(field), str) or not figure[field].strip():
            issues.append(f"插图 {field} 不能为空")
    second = figure.get("second")
    if type(second) is not int or second < 0:
        issues.append("插图时间必须为非负整数秒")
    elif figure.get("source_ref") != f"frame:{second}":
        issues.append("插图来源引用必须匹配真实时间标签")
    src = figure.get("src")
    prefix = "data:image/jpeg;base64,"
    if not isinstance(src, str) or not src.startswith(prefix) or len(src) > len(prefix) + 4 * math.ceil(MAX_FIGURE_BYTES / 3):
        issues.append("插图必须为大小受限的内嵌 JPEG")
        return issues
    try:
        from PIL import Image
        raw = base64.b64decode(src[len(prefix):], validate=True)
        if len(raw) > MAX_FIGURE_BYTES:
            raise ValueError("image too large")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG" or max(image.size) > 1600 or min(image.size) < 1:
                raise ValueError("invalid JPEG dimensions")
            image.verify()
    except (ValueError, OSError, binascii.Error, Image.DecompressionBombError):
        issues.append("插图 JPEG 数据无效")
    return issues


def structure(doc):
    return [(m.get("id"), [p.get("id") for p in m["points"]]) for m in doc["modules"]]


def validate_document(doc, limits=None):
    if not isinstance(doc, dict):
        return ["文档必须是 JSON 对象"]
    mods = doc.get("modules")
    if not isinstance(mods, list) or not mods:
        return ["文档必须包含非空 modules"]
    issues = []
    points = []
    for m in mods:
        if (
            not isinstance(m, dict)
            or not isinstance(m.get("title"), str)
            or not m["title"].strip()
        ):
            issues.append("模块必须包含非空标题")
            continue
        ps = m.get("points")
        if not isinstance(ps, list) or not ps:
            issues.append("模块必须包含非空 points")
            continue
        for p in ps:
            if not isinstance(p, dict):
                issues.append("知识点必须是对象")
                continue
            if not isinstance(p.get("title"), str) or not p["title"].strip():
                issues.append("知识点标题不能为空")
            for key in ("anchor", "body"):
                if key in p and not isinstance(p[key], str):
                    issues.append(f"{key} 必须为字符串")
            if not (p.get("body") or p.get("anchor")):
                issues.append("知识点正文不能为空")
            for key in ("examples", "warnings", "figures"):
                if key in p and not isinstance(p[key], list):
                    issues.append(f"{key} 必须为数组")
            if isinstance(p.get("warnings"), list) and any(
                not isinstance(w, str) for w in p["warnings"]
            ):
                issues.append("提醒必须为字符串")
            if isinstance(p.get("examples"), list):
                for ex in p["examples"]:
                    if not isinstance(ex, dict):
                        issues.append("示例必须为对象")
                        continue
                    if ex.get("type") == "table":
                        issues.extend(_table_issues(ex))
                    elif ex.get("type") == "diagram":
                        issues.extend(diagram_issues(ex, p))
                    elif not isinstance(ex.get("content"), str):
                        issues.append("非表格示例需要字符串 content")
                    if "nodes" in ex and ex.get("type") != "diagram" and (ex.get("type") != "decision_tree" or not isinstance(ex["nodes"], list) or not ex["nodes"] or any(not isinstance(n, str) or not n.strip() for n in ex["nodes"])):
                        issues.append("简图节点必须为非空字符串数组")
                    if ex.get("basis") not in (None, "source", "illustrative"):
                        issues.append("示例来源类型无效")
                    if ex.get("basis") == "illustrative" and ex.get("label") != "解释性例子（非来源原例）":
                        issues.append("解释性例子必须明确标注，不能冒充来源原例")
            if isinstance(p.get("figures"), list):
                for figure in p["figures"]:
                    issues.extend(figure_issues(figure))
                figure_ids = [f["id"] for f in p["figures"] if isinstance(f, dict) and isinstance(f.get("id"), str)]
                if len(figure_ids) != len(set(figure_ids)):
                    issues.append("同一知识点的插图 id 不能重复")
            if "figure_decision" in p:
                decision = p["figure_decision"]
                if not isinstance(decision, dict) or any(type(decision.get(k)) is not bool for k in ("needs_figure", "needs_example")) or not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                    issues.append("每点图例判断必须包含 needs_figure、needs_example 和理由")
                else:
                    if decision["needs_figure"] and not has_illustration(p):
                        issues.append("必要插图尚未交付")
                    if decision["needs_example"] and not p.get("examples"):
                        issues.append("必要例子尚未交付")
            points.append(p)
    appendix = doc.get("appendix", {})
    if not isinstance(appendix, dict) or not isinstance(
        appendix.get("cheatsheets", []), list
    ):
        issues.append("appendix.cheatsheets 必须为数组")
    else:
        for table in appendix.get("cheatsheets", []):
            issues.extend(_table_issues(table))
    meta = doc.get("meta", {})
    if meta is not None and not isinstance(meta, dict):
        issues.append("meta 必须为对象")
    if isinstance(meta, dict):
        for key in ("title", "positioning"):
            if key in meta and not isinstance(meta[key], str):
                issues.append(f"meta.{key} 必须为字符串")
    if issues:
        return issues
    issues.extend(validate_points(points))
    body = {k: v for k, v in doc.items() if k in ("modules", "appendix")}
    # Document title and positioning are displayed prose too. Leave the other
    # meta fields (review reports and uncertainty receipts) as audit data.
    meta = doc.get("meta")
    if isinstance(meta, dict):
        visible_meta = {k: meta[k] for k in ("title", "positioning") if k in meta}
        if visible_meta:
            body["meta"] = visible_meta
    narrative_paths = prose_narrative_paths(body)
    if narrative_paths:
        for path in narrative_paths[:8]:
            safe_path = re.sub(r"[\x00-\x1f\x7f]", " ", path)
            if len(safe_path) > 160:
                safe_path = safe_path[:159] + "…"
            issues.append(
                f"{safe_path}: 可见字段包含课堂元叙述或来源核查日志；"
                "请改为直接表达知识内容，纠错过程和不确定性记录在独立证据报告中。"
            )
        if len(narrative_paths) > 8:
            issues.append(
                f"另有{len(narrative_paths) - 8}处可见字段包含课堂元叙述或来源核查日志；最多列出8个字段。"
            )
    if limits is not None:
        issues.extend(validate_modules(mods, limits))
    return issues


def _table_issues(table):
    if not isinstance(table, dict):
        return ["表格必须为对象"]
    headers, rows = table.get("headers"), table.get("rows")
    if (
        not isinstance(headers, list)
        or not headers
        or not all(isinstance(h, str) for h in headers)
    ):
        return ["对照表需要非空字符串 headers"]
    if not isinstance(rows, list) or not rows:
        return ["对照表需要非空 rows"]
    if any(
        not isinstance(row, list)
        or len(row) != len(headers)
        or any(not isinstance(c, (str, int, float)) for c in row)
        for row in rows
    ):
        return ["表格行必须与 headers 等宽，单元格必须为文字或数字"]
    return []


def checked_document(doc, limits=None, *, previous=None):
    candidate = copy.deepcopy(doc)
    issues = validate_document(candidate, limits)
    if issues:
        raise ValueError("文档校验失败：" + "；".join(issues))
    if previous is not None and structure(candidate) != structure(previous):
        raise ValueError("文档校验失败：术语校验改变了模块/知识点结构或 ID")
    normalize_modules(candidate)
    if previous is not None and structure(candidate) != structure(previous):
        raise ValueError("文档校验失败：归一化改变了术语校验前的文档结构")
    issues = validate_document(candidate, limits)
    if issues:
        raise ValueError("文档校验失败：" + "；".join(issues))
    return candidate
