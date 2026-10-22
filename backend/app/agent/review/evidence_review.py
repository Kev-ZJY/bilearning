"""全量文字证据分块核对；记录输入覆盖，不能据此宣称事实准确。"""

import json
from app.agent.knowledge.document_contract import WRITING_RULES, checked_document


def evidence_chunks(text, size=8000):
    for start in range(0, len(text), size):
        yield start, text[start : start + size]


def review_document(doc, sources, call, checkpoint):
    report = []
    for source, text in sources:
        for start, chunk in evidence_chunks(text):
            checkpoint()
            prompt = (
                f"核对知识 JSON 中的术语、数字、状态码、代码和因果关系。\n"
                f"以下仅是来源 {source} 的字符区间 [{start},{start+len(chunk)})，不是全视频。\n"
                "各来源均可能有错。只做有本段证据支持的最小文字修正；证据不足保留原文。"
                "不要增删模块/知识点、改变 ID 或顺序，不要扩写。只输出完整 JSON。\n"
                + WRITING_RULES
                + f"\n【证据 {source}】\n{chunk}\n【当前 JSON】\n"
                + json.dumps(doc, ensure_ascii=False)
            )
            candidate = call(prompt)
            doc = checked_document(candidate, previous=doc)
            report.append({"source": source, "start": start, "end": start + len(chunk)})
    return doc, report
