"""单进程项目写入互斥、旧版留存及改写版本比较。"""

from contextlib import contextmanager
from threading import RLock, Lock
import json

from app.agent.knowledge.document_contract import checked_document
from app.core.database import session_scope
from app.models.entities import Project, KnowledgeDoc, KnowledgeRevision

_locks = {}
_registry = Lock()


@contextmanager
def project_guard(project_id):
    with _registry:
        lock = _locks.setdefault(project_id, RLock())
    with lock:
        yield


def read_doc(project_id):
    with session_scope() as db:
        p = db.get(Project, project_id)
        if p is None:
            raise ValueError("项目不存在")
        return json.loads(p.doc.doc_json) if p.doc else {}


def write_in_session(db, project_id, doc):
    """调用方持 project_guard；旧版与新文档在同一事务提交。"""
    p = db.get(Project, project_id)
    if p is None:
        raise ValueError("项目不存在")
    payload = json.dumps(doc, ensure_ascii=False)
    if p.doc:
        if p.doc.doc_json == payload:
            return
        db.add(KnowledgeRevision(project_id=project_id, doc_json=p.doc.doc_json))
        p.doc.doc_json = payload
    else:
        db.add(KnowledgeDoc(project_id=project_id, doc_json=payload))


def write_generated_in_session(db, project_id, merged, incoming):
    """Persist a validated generated document and its model-selected project name atomically."""
    write_in_session(db, project_id, merged)
    meta = incoming.get("meta") or {}
    title = meta.get("project_title") or meta.get("title")
    if isinstance(title, str) and title.strip():
        db.get(Project, project_id).title = title.strip()[:48]


def replace_doc(project_id, doc, *, expected=None, task_id=None):
    from contextlib import nullcontext
    from app.core.queue import get_queue

    # Direct updates share the same gate; validate before taking locks.
    candidate = checked_document(doc)
    with project_guard(project_id):
        with get_queue().commit_guard(task_id) if task_id else nullcontext():
            latest = read_doc(project_id)
            if expected is not None and latest != expected:
                raise ValueError("文档已被其他任务更新，请重新读取后改写；本次未覆盖")
            # 模型只能改知识文字，不能自报“已核对”或改写来源证据。
            meta = candidate.get("meta") or {}
            candidate["meta"] = meta
            previous_meta = latest.get("meta") or {}
            for key in ("visual_review", "verification", "source_videos", "reviews", "independent_review", "illustrations"):
                if key in previous_meta:
                    meta[key] = previous_meta[key]
                else:
                    meta.pop(key, None)
            # A later conversational edit must not present the old report as a
            # review of new prose. Keep the report for traceability, mark its scope.
            if candidate.get("modules") != latest.get("modules") or candidate.get("appendix") != latest.get("appendix"):
                import copy
                if "independent_review" in meta:
                    meta["independent_review"] = copy.deepcopy(meta["independent_review"])
                    meta["independent_review"]["status"] = "stale"
                    meta["independent_review"]["note"] = "正文在独立审核后已修改，先前报告不覆盖这些改动"
            with session_scope() as db:
                write_in_session(db, project_id, candidate)
    return candidate
