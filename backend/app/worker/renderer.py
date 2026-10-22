"""知识点文档渲染器：把 KnowledgeDoc JSON 渲染为 Markdown / 独立 HTML。

设计约束（来自产品需求）：
- 渲染只读 doc_json，不调用 LLM、不联网；纯函数，便于单测。
- 独立 HTML 必须自包含：内联 CSS、无外部 http(s) 的 css/js 依赖，便于导出后双击打开。
- 不引入任何元叙述（讲师 / 视频中 / 黑板 等），保持「可背诵知识点」调性。
"""
import html
import re
from urllib.parse import urlsplit

from app.agent.knowledge.document_contract import diagram_issues, figure_issues


def _esc(s: str) -> str:
    return html.escape(str(s) if s is not None else "", quote=False)


def _md_table(headers: list, rows: list[list]) -> str:
    out = ["| " + " | ".join(_esc(h).replace("|", "&#124;") for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_esc(c).replace("|", "&#124;") for c in r) + " |")
    return "\n".join(out)


def _detect_table(content: str) -> bool:
    lines = [l for l in (content or "").splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    return lines[0].strip().startswith("|") and set(lines[1].strip()) <= set("|-: ")


def _decision_nodes(example):
    nodes = example.get("nodes")
    if isinstance(nodes, list) and nodes:
        return nodes
    text = example.get("content", "")
    lines = text.splitlines() if "\n" in text else text.split("→")
    return [re.sub(r"^\s*(?:[-*]|\d+[.)、])\s*", "", line).strip() for line in lines if line.strip()]


def _safe_figures(point):
    for figure in point.get("figures", []):
        issues = figure_issues(figure)
        if issues:
            raise ValueError("无法渲染不安全插图：" + "；".join(issues))
        yield figure


def _diagram(example, point, *, markdown=False):
    issues = diagram_issues(example, point)
    if issues:
        raise ValueError("无法渲染示意图：" + "；".join(issues))
    nodes = [_esc(node) for node in example["nodes"]]
    label = "解释性示意图（非视频原图）"
    if markdown:
        body = " → ".join(nodes) if example["layout"] == "sequence" else "\n".join("- " + node for node in nodes)
        return f"\n**{label}**\n\n{body}\n\n{_esc(example['content'])}"
    body = []
    for index, node in enumerate(nodes):
        if index and example["layout"] == "sequence":
            body.append('<span class="diagram-arrow" aria-hidden="true">→</span>')
        body.append('<div class="diagram-node">' + node + '</div>')
    return '<figure class="knowledge-diagram ' + example["layout"] + '"><figcaption>' + label + '</figcaption><div class="diagram-nodes">' + ''.join(body) + '</div><p>' + _esc(example["content"]) + '</p></figure>'


def _review_html(report):
    """Only publish concise findings/decisions, never prompts or raw tool activity."""
    if not isinstance(report, dict):
        return ""
    label = {"reviewed": "已审查", "revised": "已审查并修改", "stale": "文档已更新，原审查报告已过期"}.get(report.get("status"), "状态待确认")
    parts = [f'<details class="review-notes"><summary>核验说明 · {label}</summary>']
    if isinstance(report.get("scope"), str) and report["scope"].strip():
        parts.append(f'<p>审查范围：{_esc(report["scope"])}</p>')
    for limitation in report.get("limitations", []):
        if isinstance(limitation, str):
            parts.append(f'<p class="review-limitation">待核实：{_esc(limitation)}</p>')
    for finding in report.get("findings", []):
        if isinstance(finding, dict) and isinstance(finding.get("issue"), str):
            parts.append(f'<p>{_esc(finding["issue"])}')
            if isinstance(finding.get("suggestion"), str) and finding["suggestion"]:
                parts.append(f' · 建议：{_esc(finding["suggestion"])}')
            parts.append('</p>')
    for decision in report.get("decisions", []):
        if isinstance(decision, dict) and isinstance(decision.get("reason"), str):
            action = {"accept": "采纳", "reject": "未采纳", "defer": "待核实"}.get(decision.get("decision"), "处理说明")
            parts.append(f'<p>{action}：{_esc(decision["reason"])}</p>')
    sources = []
    for source in report.get("web_sources", []):
        if not isinstance(source, dict) or not isinstance(source.get("url"), str):
            continue
        try:
            parsed = urlsplit(source["url"])
            valid = parsed.scheme in ("http", "https") and bool(parsed.hostname)
        except ValueError:
            valid = False
        if valid:
            sources.append('<li><a href="' + html.escape(source["url"], quote=True) + '" rel="noopener noreferrer" target="_blank">' + _esc(source.get("title") or source["url"]) + '</a></li>')
    if sources:
        parts.append('<p>参考来源</p><ul>' + ''.join(sources) + '</ul>')
    parts.append('</details>')
    return ''.join(parts)


def render_markdown(doc: dict) -> str:
    out = []
    meta = doc.get("meta") or {}
    out.append(f"# {meta.get('title', '知识点清单')}")
    scope = (meta.get("visual_review") or {}).get("reason")
    if scope:
        out.append(f"\n> 最近一次整理的画面核对范围：{_esc(scope)}")
    if meta.get("positioning"):
        out.append(f"\n> {_esc(meta['positioning'])}")
    if meta.get("independent_review"):
        out.append("\n" + _review_html(meta["independent_review"]))
    for m in doc.get("modules", []):
        out.append(f"\n## {m.get('title', '')}")
        for p in m.get("points", []):
            out.append(f"\n### {p.get('title', '')}")
            if p.get("anchor"):
                out.append(f"\n**{_esc(p['anchor'])}**")
            if p.get("body"):
                out.append(f"\n{_esc(p['body'])}")
            for figure in _safe_figures(p):
                alt = _esc(figure["alt"]).replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("\n", " ")
                out.append(f"\n![{alt}]({figure['src']})\n\n{_esc(figure['caption'])} · 来源 {figure['second']} 秒")
            for ex in p.get("examples", []):
                c = ex.get("content", "")
                if ex.get("type") == "diagram":
                    out.append(_diagram(ex, p, markdown=True))
                    continue
                if ex.get("label"):
                    out.append(f"\n**{_esc(ex['label'])}**")
                if ex.get("type") == "table" and ex.get("headers") and ex.get("rows"):
                    out.append("\n" + _md_table(ex["headers"], ex["rows"]))
                elif ex.get("type") == "decision_tree":
                    out.append("\n" + "\n".join(f"{i + 1}. {_esc(node)}" for i, node in enumerate(_decision_nodes(ex))))
                elif _detect_table(c):
                    out.append(f"\n{c}")
                else:
                    out.append(f"\n```\n{c}\n```")
            for w in p.get("warnings", []):
                out.append(f"\n> ⚠ {_esc(w)}")
    sheets = (doc.get("appendix") or {}).get("cheatsheets") or []
    if sheets:
        out.append("\n## 附录 · 速查表")
        for cs in sheets:
            out.append(f"\n### {cs.get('title', '')}")
            out.append("\n" + _md_table(cs.get("headers") or [],
                                          cs.get("rows") or []))
    return "\n".join(out)


_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","PingFang SC",sans-serif;
line-height:1.7;color:#1d1d1f;background:#fff;max-width:820px;margin:0 auto;padding:48px 28px}
h1{font-size:26px;font-weight:500;margin-bottom:8px;letter-spacing:-.01em}
h2{font-size:19px;font-weight:500;margin:36px 0 14px;padding-bottom:8px;
border-bottom:1px solid rgba(0,0,0,.08)}
h3{font-size:15px;font-weight:500;margin:24px 0 8px}
blockquote{margin:8px 0;padding:2px 0 2px 14px;border-left:2px solid #0071e3;color:#1d1d1f}
pre{background:#f5f5f7;border-radius:8px;padding:12px 14px;overflow-x:auto;
font-family:"SF Mono",Menlo,monospace;font-size:12.5px;margin:10px 0}
code{font-family:"SF Mono",Menlo,monospace}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px}
th,td{border:1px solid rgba(0,0,0,.1);padding:7px 10px;text-align:left}
th{background:#f5f5f7;font-weight:500}
.warn{background:#fff8ed;border-left:2px solid #ff9f0a;padding:9px 12px;
border-radius:8px;margin:10px 0;font-size:13px}
.toc{background:#f5f5f7;border-radius:12px;padding:14px 16px;margin:24px 0}
.toc a{display:block;color:#0071e3;text-decoration:none;font-size:13px;padding:2px 0}
.toc a:hover{text-decoration:underline}
.meta{color:#6e6e73;font-size:13px;margin-bottom:32px}
figure{margin:18px 0}figure img{display:block;max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px}
figcaption,.example-label{font-size:12px;color:#555;margin-top:7px}
.decision-tree{list-style-position:inside;padding:0;margin:12px 0;display:grid;gap:8px}
.decision-tree li{padding:10px 14px;background:#f5f5f7;border-left:3px solid #0071e3;border-radius:6px;white-space:pre-wrap}
.knowledge-diagram{padding:14px;border:1px solid #d8e3ee;border-radius:10px;background:#f8fafc}
.knowledge-diagram figcaption{margin:0 0 12px}.diagram-nodes{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.diagram-node{flex:1 1 140px;min-width:0;padding:12px;border:1px solid #94b3ce;border-radius:7px;background:white;text-align:center;overflow-wrap:anywhere}
.diagram-arrow{color:#51738f;font-size:22px;flex:none}.knowledge-diagram p{font-size:13px;margin-top:12px;overflow-wrap:anywhere}
.review-notes{font-size:13px;background:#f5f5f7;padding:12px 16px;border-radius:8px;margin:16px 0}
.review-notes summary{cursor:pointer}.review-notes p{margin-top:8px}.review-notes ul{padding-left:20px}.review-limitation{color:#805400}
"""


def render_standalone_html(doc: dict) -> str:
    meta = doc.get("meta") or {}
    parts = ["<!DOCTYPE html>", '<html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             f"<title>{_esc(meta.get('title', '知识点清单'))}</title>",
             f"<style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>{_esc(meta.get('title', '知识点清单'))}</h1>")
    scope = (meta.get("visual_review") or {}).get("reason")
    if scope:
        parts.append(f'<aside class="meta">最近一次整理的画面核对范围：{_esc(scope)}</aside>')
    if meta.get("positioning"):
        parts.append(f'<p class="meta">{_esc(meta["positioning"])}</p>')
    if meta.get("independent_review"):
        parts.append(_review_html(meta["independent_review"]))

    modules = doc.get("modules", [])
    if modules:
        parts.append('<nav class="toc">')
        for m in modules:
            parts.append(f'<a href="#{_esc(m.get("id", ""))}">{_esc(m.get("title", ""))}</a>')
        parts.append("</nav>")

    for m in modules:
        parts.append(f'<h2 id="{_esc(m.get("id", ""))}">{_esc(m.get("title", ""))}</h2>')
        for p in m.get("points", []):
            parts.append(f"<h3>{_esc(p.get('title', ''))}</h3>")
            if p.get("anchor"):
                parts.append(f"<blockquote>{_esc(p['anchor'])}</blockquote>")
            if p.get("body"):
                parts.append(f"<p>{_esc(p['body'])}</p>")
            for figure in _safe_figures(p):
                parts.append('<figure><img src="' + figure["src"] + '" alt="' + html.escape(figure["alt"], quote=True) + '" loading="lazy">'
                             + f"<figcaption>{_esc(figure['caption'])} · 来源 {figure['second']} 秒</figcaption></figure>")
            for ex in p.get("examples", []):
                c = ex.get("content", "")
                if ex.get("type") == "diagram":
                    parts.append(_diagram(ex, p))
                    continue
                if ex.get("label"):
                    parts.append(f'<p class="example-label">{_esc(ex["label"])}</p>')
                if ex.get("type") == "table" and ex.get("headers") and ex.get("rows"):
                    parts.append("<table><thead><tr>" + "".join(f"<th>{_esc(h)}</th>" for h in ex["headers"]) + "</tr></thead><tbody>")
                    for row in ex["rows"]:
                        parts.append("<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>")
                    parts.append("</tbody></table>")
                elif ex.get("type") == "decision_tree":
                    parts.append('<ol class="decision-tree">' + "".join(f"<li>{_esc(node)}</li>" for node in _decision_nodes(ex)) + "</ol>")
                elif _detect_table(c):
                    parts.append("<pre>" + _esc(c) + "</pre>")
                else:
                    parts.append(f"<pre>{_esc(c)}</pre>")
            for w in p.get("warnings", []):
                parts.append(f'<div class="warn">⚠ {_esc(w)}</div>')

    sheets = (doc.get("appendix") or {}).get("cheatsheets") or []
    if sheets:
        parts.append("<h2>附录 · 速查表</h2>")
        for cs in sheets:
            parts.append(f"<h3>{_esc(cs.get('title', ''))}</h3>")
            parts.append("<table>")
            headers = cs.get("headers") or []
            if headers:
                parts.append("<thead><tr>" +
                             "".join(f"<th>{_esc(h)}</th>" for h in headers) +
                             "</tr></thead>")
            parts.append("<tbody>")
            for r in cs.get("rows") or []:
                parts.append("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>")
            parts.append("</tbody></table>")

    parts.append("</body></html>")
    return "".join(parts)
