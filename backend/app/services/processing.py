"""统一的视频整理提交入口；卡片 API 与 Agent 工具均委托到这里。"""

from threading import Thread, BoundedSemaphore
from contextlib import contextmanager
import hashlib
import uuid
from pathlib import Path

from app.core.database import session_scope
from app.core.queue import get_queue
from app.models.entities import Project, Video
from app.core.cancel import TaskCancelled
from app.services.documents import project_guard
from app.services.options import ProcessingOptions

# One heavy media/ASR pipeline at a time on a local desktop. Queued jobs stay cancellable.
_PROCESSING_SLOT = BoundedSemaphore(1)


@contextmanager
def processing_slot(queue, task_id):
    while not _PROCESSING_SLOT.acquire(timeout=0.2):
        if queue.is_cancelled(task_id):
            raise TaskCancelled()
    try:
        if queue.is_cancelled(task_id):
            raise TaskCancelled()
        yield
    finally:
        _PROCESSING_SLOT.release()


class ProcessingNotFound(ValueError):
    """项目不存在，或视频不属于该项目。"""


def submit_processing(
    project_id: int,
    video_id: int,
    intent: str = "整理视频知识点",
    terms: list[str] | None = None,
    options: dict | None = None,
) -> str:
    # Serialize registration with preserved-evidence rebuilds as well as with
    # other normal submissions; workers alone taking this guard is too late.
    from app.services.recovery import has_active_video_worker, RecoveryError
    if has_active_video_worker(project_id, video_id):
        raise RecoveryError("该视频仍有未结束 worker，不能并行整理")
    with project_guard(("video", video_id)):
        if has_active_video_worker(project_id, video_id):
            raise RecoveryError("该视频仍有未结束 worker，不能并行整理")
        return _submit_processing(project_id, video_id, intent, terms, options)


def _submit_processing(project_id, video_id, intent, terms, options) -> str:
    terms = terms or []
    options = ProcessingOptions(**(options or {})).model_dump()
    with session_scope() as db:
        project = db.get(Project, project_id)
        if project is None:
            raise ProcessingNotFound("项目不存在")
        video = db.get(Video, video_id)
        if video is None or video.project_id != project_id:
            raise ProcessingNotFound("视频不存在或不属于当前项目")
        from app.services.desktop import enabled

        if enabled():
            from app.services.project_files import check_root

            check_root(db, project)
        video.status = "pending"
        video.error = None
        root = Path(project.storage_root)
        info = {"video_id": video.id, "bvid": video.bvid, "cid": video.cid}

    task_id = f"task-{video_id}-{uuid.uuid4().hex}"
    recovery_work = root / f"video_{video_id}" / "tasks" / hashlib.sha256(task_id.encode()).hexdigest()[:24]
    queue = get_queue()
    # 本服务直接启动线程；不进入没有消费者的待领取队列。
    queue.register(
        task_id,
        {
            "kind": "pipeline",
            "project_id": project_id,
            **info,
            "intent": intent,
            "terms": terms,
            "options": options,
        },
    )

    def cancelled():
        with session_scope() as db:
            video = db.get(Video, video_id)
            if video is not None:
                video.status = "cancelled"
                video.error = None
        queue.update(task_id, status="cancelled", message="已取消，未提交文档")

    def fail(exc: Exception) -> None:
        try:
            # 取消若先到，错误收尾也不能把视频重新标成 failed。
            with queue.commit_guard(task_id):
                with session_scope() as db:
                    video = db.get(Video, video_id)
                    if video is not None:
                        video.status = "failed"
                        video.error = str(exc)
                queue.update(
                    task_id, status="failed", error=str(exc), message=f"失败：{exc}"
                )
        except TaskCancelled:
            cancelled()

    def work() -> None:
        try:
            queue.update(task_id, message="等待前一项视频整理完成", stage="queued")
            with processing_slot(queue, task_id), project_guard(("video", video_id)):
                if queue.is_cancelled(task_id):
                    raise TaskCancelled()
                from app.agent.workflow.pipeline import Pipeline

                Pipeline(project_id, root).run(
                    task_id, intent=intent, terms=terms, options=options, **info
                )
        except TaskCancelled:
            cancelled()
        except Exception as exc:
            fail(exc)
        finally:
            current = queue.get_status(task_id) or {}
            terminal = current.get("status")
            if terminal in ("done", "failed", "cancelled"):
                checkpoint_path = recovery_work / "checkpoint.json"
                if checkpoint_path.is_file() and not checkpoint_path.is_symlink():
                    try:
                        from app.services.recovery import persist_recovery_status

                        persist_recovery_status(
                            recovery_work, terminal, worker_finished=True,
                            error=current.get("error"),
                        )
                    except Exception:
                        # Task completion must not be replaced by an auxiliary
                        # recovery-receipt write failure.
                        pass
            queue.update(task_id, worker_finished=True)

    try:
        Thread(target=work, daemon=True).start()
    except Exception as exc:
        fail(exc)
        queue.update(task_id, worker_finished=True)
        raise
    return task_id


def submit_recovery(project_id: int, video_id: int, recovery_id: str, *, rebuild_completed: bool = False) -> str:
    """Start a new task from verified persisted inputs without repeating ASR/vision."""
    from app.services.recovery import (
        RecoveryError,
        has_active_video_worker,
        load_recovery_inputs,
        persist_recovery_status,
        prepare_recovery_attempt,
    )

    task_id = f"task-{video_id}-resume-{uuid.uuid4().hex}"
    queue = get_queue()
    # Reject a live worker before taking the long-lived video guard. Otherwise
    # a duplicate resume request could wait behind model work instead of
    # returning the expected conflict promptly.
    if has_active_video_worker(project_id, video_id):
        raise RecoveryError("该视频仍有未结束 worker，不能并行恢复")
    # Serialize recovery preparations for this video so two requests cannot
    # both claim the same old failure before either registers its live task.
    with project_guard(("video", video_id)):
        if has_active_video_worker(project_id, video_id):
            raise RecoveryError("该视频仍有未结束 worker，不能并行恢复")
        work, resume_info = prepare_recovery_attempt(
            project_id, video_id, recovery_id, task_id,
            **({"rebuild_completed": True} if rebuild_completed else {}),
        )
        root = work.parents[2]
        video_info = resume_info["video"]
        queue.register(
            task_id,
            {
                "kind": "pipeline",
                "project_id": project_id,
                **video_info,
                "recovery_id": resume_info["recovery_id"],
                "source_recovery_id": recovery_id,
                "resume": True,
                "intent": resume_info["intent"],
                "terms": resume_info["terms"],
                "options": resume_info["options"],
            },
        )

    def cancelled() -> None:
        with session_scope() as db:
            video = db.get(Video, video_id)
            if video is not None:
                video.status = "cancelled"
                video.error = None
        try:
            persist_recovery_status(work, "cancelled", error=None)
        except Exception:
            # A damaged receipt remains non-resumable; it must not mask cancellation.
            pass
        queue.update(task_id, status="cancelled", message="已取消，未提交文档")

    def fail(exc: Exception) -> None:
        try:
            with queue.commit_guard(task_id):
                with session_scope() as db:
                    video = db.get(Video, video_id)
                    if video is not None:
                        video.status = "failed"
                        video.error = str(exc)
                try:
                    persist_recovery_status(work, "failed", error=str(exc))
                except Exception:
                    pass
                queue.update(
                    task_id, status="failed", error=str(exc), message=f"失败：{exc}"
                )
        except TaskCancelled:
            cancelled()

    def work_task() -> None:
        try:
            queue.update(
                task_id,
                message="校验恢复输入并从来源核对继续",
                stage="verify",
            )
            with processing_slot(queue, task_id), project_guard(("video", video_id)):
                if queue.is_cancelled(task_id):
                    raise TaskCancelled()
                inputs = load_recovery_inputs(work, project_id, video_id)
                with session_scope() as db:
                    video = db.get(Video, video_id)
                    if video is None or video.project_id != project_id:
                        raise ProcessingNotFound("视频不存在或不属于当前项目")
                    bvid, cid = video.bvid, video.cid
                from app.agent.workflow.pipeline import Pipeline

                Pipeline(project_id, root).resume(
                    task_id,
                    video_id=video_id,
                    bvid=bvid,
                    cid=cid,
                    work=work,
                    **inputs,
                )
        except TaskCancelled:
            cancelled()
        except Exception as exc:
            fail(exc)
        finally:
            current = queue.get_status(task_id) or {}
            status = current.get("status")
            if status not in ("done", "failed", "cancelled"):
                status = "failed"
                try:
                    queue.update(task_id, status="failed", error="恢复 worker 未正常结束")
                except Exception:
                    pass
            try:
                persist_recovery_status(
                    work,
                    status,
                    worker_finished=True,
                    error=current.get("error"),
                )
            except Exception:
                pass
            queue.update(task_id, worker_finished=True)

    try:
        Thread(target=work_task, daemon=True).start()
    except Exception as exc:
        fail(exc)
        try:
            persist_recovery_status(work, "failed", worker_finished=True, error=str(exc))
        except Exception:
            pass
        queue.update(task_id, worker_finished=True)
        raise
    return task_id
