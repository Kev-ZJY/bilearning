"""导出接口：Markdown / 独立 HTML / PDF。

PDF 转换采用可移植策略（修 Task 13 规格里的根因）：
- 规格原本直接 `subprocess.run(["wkhtmltopdf", ...])`，而 wkhtmltopdf 是需要单独
  安装的二进制，缺失时会抛 `FileNotFoundError` 而非走 `returncode != 0` 分支，
  报错信息变成难懂的 "No such file or directory: 'wkhtmltopdf'"。
- 改为先用 `shutil.which()` 依次探测可用转换器（wkhtmltopdf -> cupsfilter）；
  都没有时返回 **503**，并给出清晰的安装指引，而不是 500 或晦涩异常。
- `cupsfilter` 虽为 macOS 自带，但本机实测缺少 text/html 滤镜，无法完成 HTML→PDF，
  因此「存在但调用失败」不能让导出崩溃，必须回退到下一个转换器（reportlab 兜底）。
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from app.worker.renderer import render_standalone_html

router = APIRouter(prefix="/api/projects", tags=["export"])


class NoPdfConverterError(Exception):
    """没有任何可用的 HTML→PDF 转换器时抛出，由路由转译为 503。"""


def _find_wkhtmltopdf() -> str | None:
    return shutil.which("wkhtmltopdf")


def _find_cupsfilter() -> str | None:
    # cupsfilter 通常在 /usr/sbin，可能不在 python 进程的 PATH 中，额外探测常见绝对路径。
    for candidate in ("cupsfilter", "/usr/sbin/cupsfilter"):
        if candidate.startswith("/"):
            if Path(candidate).exists():
                return candidate
        elif shutil.which(candidate):
            return candidate
    return None


def _reportlab_available() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except Exception:
        return False


def _render_pdf_reportlab(doc: dict) -> bytes:
    """用 reportlab 直接由 doc 生成 PDF（无需外部二进制）。

    选用 reportlab 自带的 Adobe CJK CID 字体 STSong-Light，无需任何外部字体文件
    即可正确渲染中文，避免「豆腐块」。作为 wkhtmltopdf / cupsfilter 都不可用时的
    兜底转换器，保证 PDF 导出在纯净 macOS 环境下也能真正产出。
    """
    import html as _html
    from io import BytesIO

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    FONT = "STSong-Light"

    base = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=base["Heading1"], fontName=FONT, fontSize=20)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName=FONT, fontSize=14,
                        spaceBefore=12)
    h3 = ParagraphStyle("h3", parent=base["Heading3"], fontName=FONT, fontSize=12)
    body = ParagraphStyle("body", parent=base["BodyText"], fontName=FONT, fontSize=10,
                          leading=15)
    meta = ParagraphStyle("meta", parent=body, textColor="#6e6e73")
    warn = ParagraphStyle("warn", parent=body, backColor="#fff8ed",
                          leftIndent=8, spaceBefore=4, spaceAfter=4)

    def P(text, style=body):
        return Paragraph(_html.escape(str(text)), style)

    story = []
    meta_d = doc.get("meta") or {}
    story.append(P(meta_d.get("title", "知识点清单"), h1))
    if meta_d.get("positioning"):
        story.append(P(meta_d["positioning"], meta))
    review = meta_d.get("visual_review") or {}
    if review:
        story.append(P("最近一次整理的画面核对范围：" + review.get("reason", "未提供核对范围"), meta))
    story.append(Spacer(1, 8))

    for m in doc.get("modules", []):
        story.append(P(m.get("title", ""), h2))
        for p in m.get("points", []):
            story.append(P(p.get("title", ""), h3))
            if p.get("anchor"):
                story.append(P("“" + p["anchor"] + "”", body))
            if p.get("body"):
                story.append(P(p["body"], body))
            for ex in p.get("examples", []):
                if ex.get("type") == "table" and ex.get("rows"):
                    data = [[P(h, body) for h in ex.get("headers", [])]] + [[P(c, body) for c in row] for row in ex["rows"]]
                    table = Table(data, hAlign="LEFT")
                    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#cccccc")]))
                    story.append(table)
                else:
                    story.append(P(ex.get("content", ""), body))
            for w in p.get("warnings", []):
                story.append(P("⚠ " + w, warn))

    sheets = (doc.get("appendix") or {}).get("cheatsheets") or []
    if sheets:
        story.append(P("附录 · 速查表", h2))
        for cs in sheets:
            story.append(P(cs.get("title", ""), h3))
            headers = cs.get("headers") or []
            rows = cs.get("rows") or []
            data = [[P(h, body) for h in headers]] + \
                   [[P(c, body) for c in r] for r in rows]
            if data:
                t = Table(data, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.5, "#cccccc"),
                    ("BACKGROUND", (0, 0), (-1, 0), "#f5f5f7"),
                ]))
                story.append(t)

    buf = BytesIO()
    SimpleDocTemplate(
        buf, pagesize=A4, title=meta_d.get("title", "知识点清单")
    ).build(story)
    return buf.getvalue()


def _try_wkhtmltopdf(html: str) -> bytes:
    wk = _find_wkhtmltopdf()
    if not wk:
        raise NoPdfConverterError("wkhtmltopdf 未安装")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "doc.html"
        out = Path(td) / "doc.pdf"
        tmp.write_text(html, encoding="utf-8")
        r = subprocess.run(
            [wk, "--enable-local-file-access", "--quiet",
             str(tmp), str(out)],
            capture_output=True,
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out.read_bytes()
        stderr = (r.stderr or b"").decode("utf-8", "replace")[:400]
        raise NoPdfConverterError(
            f"wkhtmltopdf 调用失败（exit={r.returncode}）：{stderr}"
        )


def _try_cupsfilter(html: str) -> bytes:
    cf = _find_cupsfilter()
    if not cf:
        raise NoPdfConverterError("cupsfilter 不可用")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "doc.html"
        out = Path(td) / "doc.pdf"
        tmp.write_text(html, encoding="utf-8")
        with open(out, "wb") as fout:
            r = subprocess.run(
                [cf, "-m", "application/pdf", str(tmp)],
                stdout=fout, stderr=subprocess.PIPE,
            )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out.read_bytes()
        stderr = (r.stderr or b"").decode("utf-8", "replace")[:400]
        # 本机 cupsfilter 可能缺少 text/html 滤镜，这种情况不算成功，继续回退
        raise NoPdfConverterError(
            f"cupsfilter 调用失败（exit={r.returncode}）：{stderr}"
        )


def generate_pdf(doc: dict) -> bytes:
    """生成 PDF 字节。依次尝试 wkhtmltopdf -> cupsfilter -> reportlab。

    前两者是规格指定的外部转换器（优先复用已渲染的 HTML，保真度高）；
    reportlab 是兜底（用内置 CJK 字体，无需外部二进制）。

    关键：某个转换器「存在但调用失败」（如 cupsfilter 缺 text/html 滤镜）
    不能让整个导出崩溃，必须**回退**到下一个转换器；只有当所有候选都不可用/
    都失败时，才抛 NoPdfConverterError，由路由转成 503。
    """
    html = render_standalone_html(doc)
    errors: list[str] = []

    for name, attempt in (
        ("wkhtmltopdf", lambda: _try_wkhtmltopdf(html)),
        ("cupsfilter", lambda: _try_cupsfilter(html)),
    ):
        try:
            return attempt()
        except NoPdfConverterError as e:
            errors.append(f"[{name}] {e}")

    if _reportlab_available():
        try:
            return _render_pdf_reportlab(doc)
        except Exception as e:  # noqa: BLE001 兜底转换器的兜底，避免裸异常暴露成 500
            errors.append(f"[reportlab] {e}")
    else:
        errors.append("[reportlab] 未安装（pip install reportlab）")

    raise NoPdfConverterError(
        "当前环境无法生成 PDF：全部可用转换器均失败。\n"
        + "\n".join(errors)
        + "\n可安装其一后重试：wkhtmltopdf (brew) / cupsfilter (macOS CUPS，需 text/html 滤镜) / reportlab (pip)。"
    )


def _load_doc(pid: int) -> dict:
    from app.core.database import session_scope
    from app.models.entities import Project

    with session_scope() as db:
        p = db.get(Project, pid)
        if not p or not p.doc or not p.doc.doc_json:
            raise HTTPException(404, "该项目尚无知识点文档")
        return json.loads(p.doc.doc_json)


def _doc_title(doc: dict) -> str:
    """取文档标题。

    不能写 ``doc.get("meta", {})``：LLM 产出的 JSON 里 ``meta`` 可能是显式的
    ``null``，此时 ``.get`` 返回 None，再 ``.get("title")`` 就是 AttributeError，
    导出直接 500。用 ``or {}`` 兜住。
    """
    meta = doc.get("meta") or {}
    return meta.get("title", "") if isinstance(meta, dict) else ""


def _safe_filename(title: str, ext: str) -> str:
    base = (title or "knowledge").strip() or "knowledge"
    # 去掉对文件名不安全、以及会破坏 HTTP 头的字符。
    # 控制字符（\r \n \t 等）必须一起清：title 来自 LLM 产出的 JSON，
    # 裸换行进 Content-Disposition 就是 header 注入，浏览器会截断响应头。
    for ch in '/\\:*?"<>|':
        base = base.replace(ch, "_")
    base = "".join("_" if (ord(c) < 0x20 or ord(c) == 0x7F) else c for c in base)
    # 纯点号（"." / ".."）会让某些下载器把文件名当路径或父目录解析
    base = base.strip(".") or "knowledge"
    return f"{base}.{ext}"


def _content_disposition(filename: str) -> str:
    """构造 Content-Disposition，兼容中文文件名。

    HTTP 头只能放 latin-1，直接用中文文件名会抛 UnicodeEncodeError。
    这里用 RFC 5987 的 filename*（UTF-8 百分号编码）承载中文，
    并附带一个 ASCII 回退（filename=），保证老客户端也能下载。
    """
    ascii_fallback = filename.encode("ascii", "ignore").decode() or "knowledge"
    if not Path(ascii_fallback).suffix:
        ascii_fallback += Path(filename).suffix or ""
    return (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{quote(filename)}"
    )


@router.get("/{pid}/export/md")
def export_md(pid: int):
    from app.worker.renderer import render_markdown

    doc = _load_doc(pid)
    # 三个导出格式必须一致地带 Content-Disposition，否则前端拿不到文件名，
    # MD 会退化成固定名 knowledge.md，把项目标题丢掉。
    name = _safe_filename(_doc_title(doc), "md")
    return PlainTextResponse(
        render_markdown(doc),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": _content_disposition(name)},
    )


@router.get("/{pid}/export/html")
def export_html(pid: int):
    from app.worker.renderer import render_standalone_html

    doc = _load_doc(pid)
    name = _safe_filename(_doc_title(doc), "html")
    return Response(
        render_standalone_html(doc),
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": _content_disposition(name)},
    )


@router.get("/{pid}/export/pdf")
def export_pdf(pid: int):
    doc = _load_doc(pid)
    name = _safe_filename(_doc_title(doc), "pdf")
    try:
        data = generate_pdf(doc)
    except NoPdfConverterError as e:
        # 转换器缺失是「服务端当前不可用」，用 503 而不是 500/崩溃
        raise HTTPException(503, str(e)) from e
    return Response(
        data,
        media_type="application/pdf",
        headers={"Content-Disposition": _content_disposition(name)},
    )
