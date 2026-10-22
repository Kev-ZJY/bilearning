"""完整音画证据 → 图文草稿 → 独立联网审核 → 作者复核 → 原子提交。

任何必需阶段失败都保留任务草稿和审核报告，不覆盖原知识文档。
"""

from __future__ import annotations

import json
import time
import hashlib
import copy
import inspect
import os
import threading
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.agent.knowledge.extractor import (
    _budget_limits,
    build_extraction_prompt,
)
from app.agent.knowledge.fusion import merge_reviewed_docs, assert_review_merge_unchanged
from app.agent.knowledge.document_contract import checked_document, validate_document
from app.agent.workflow.formal_receipts import (
    bind_formal_output_receipts,
    mark_formal_output_committed,
    verify_formal_output_receipts,
)
from app.agent.knowledge.grounding import ground_document, complete_document, validate_final_coverage
from app.agent.media.media_preparation import (
    MediaPreparationCancelled,
    MediaPreparationError,
    MediaPreparationRequest,
    MediaSource,
    prepare_media,
)
from app.agent.runtime.task_budget import ModelAttemptBudgetExceeded, TaskBudget
from app.core.cancel import TaskCancelled
from app.services.documents import project_guard, write_generated_in_session
from app.agent.runtime.heartbeat import chat_with_heartbeat
from app.agent.runtime.llm import LLMClient
_RUNTIME_LLM_CLIENT_TYPE = LLMClient
from app.agent.runtime.context import ContextBudgetError, document_for_model
from app.core.config import get_settings
from app.core.database import session_scope
from app.core.queue import get_queue
from app.models.entities import KnowledgeDoc, Project, Video
from app.sources.bilibili_wbi import BilibiliClient
from app.worker.downloader import download_audio, download_video
from app.worker.evidence import load_subtitles, collect_visual_evidence
from app.worker.transcriber import WhisperTranscriber

# 顺序由真实执行与回归测试共同约束；可用字幕也不能替代音频转写。
STAGES = ["download", "asr", "frames", "draft", "verify", "illustrate", "review", "revise", "render"]

# _emit 在阶段切换时要回写进 Video.status 的英文枚举集合（A16 修复）。
# 不含 "download"：提交服务已经把 status 设成 "downloading"
# （前端同一文案），这里不覆盖以免把库里的值退回到裸的 "download"。
# 也不含 "done"/"failed"：那两处在流水线末尾与异常兜底里单独回写。
_STATUS_SYNC_STAGES = tuple(STAGES[1:])
_LLM_OBSERVER_SEQUENCE_LOCK = threading.Lock()


def _json_default(o):
    """json.dumps 的兜底序列化器：把 Path 等非原生类型转成字符串。

    产物字典会直接 ``json.dumps`` 写进 Video.artifacts_json。只要有任何一个
    字段是 Path，落库那一步就会抛 "Object of type PosixPath is not JSON
    serializable" —— 而且是流水线跑完转写、抽完点、进度到 96% 才炸，
    前面几十秒的算力全白费（真实端到端验证时踩到过）。

    有了这个兜底，新增字段时即使忘了 str() 也不会让整条流水线崩掉。
    """
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"无法序列化的类型: {type(o).__name__}")


def dumps(obj) -> str:
    """项目内统一的 JSON 序列化入口（落库 / 传给前端都走它）。"""
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def _media_preparation_artifact(result):
    """Serialize media identity and per-frame hashes without copying media bytes."""
    identity = result.identity
    transcript = result.transcript
    frames = result.frames
    return {
        "status": result.status,
        "ready_for_joint_analysis": result.ready_for_joint_analysis,
        "branches": [
            {
                "name": branch.name,
                "status": branch.status,
                "elapsed_seconds": branch.elapsed_seconds,
                "media_duration_seconds": branch.media_duration_seconds,
                "error_type": branch.error_type,
                "error_message": branch.error_message,
            }
            for branch in result.branches
        ],
        "identity": ({
            "source_key": identity.source_key,
            "page_index": identity.page_index,
            "audio_sha256": identity.audio_sha256,
            "video_sha256": identity.video_sha256,
            "audio_duration_seconds": identity.audio_duration_seconds,
            "video_duration_seconds": identity.video_duration_seconds,
            "range_start_seconds": identity.range_start_seconds,
            "range_end_seconds": identity.range_end_seconds,
            "timeline_origin_seconds": identity.timeline_origin_seconds,
        } if identity else None),
        "transcript": ({
            "base_path": str(transcript.base_path),
            "medium_path": str(transcript.medium_path),
            "audio_sha256": transcript.audio_sha256,
            "audio_duration_seconds": transcript.audio_duration_seconds,
        } if transcript else None),
        "frames": ({
            "source_key": frames.source_key,
            "video_sha256": frames.video_sha256,
            "duration_seconds": frames.duration_seconds,
            "frames": [
                {"second": row.second, "path": str(row.path), "sha256": row.sha256}
                for row in frames.frames
            ],
        } if frames else None),
        "attempt_dir": str(result.attempt_dir) if result.attempt_dir else None,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


def _persist_transcript_artifact(source, work, name, checkpoint):
    """Atomically persist a stable ASR transcript at the recovery-canonical path."""
    source = Path(source)
    target = Path(work) / name
    if not source.is_file():
        raise ValueError(f"{name} 转写文件缺失，未继续处理")

    def identity(path):
        stat = Path(path).stat()
        return (
            stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns,
        )

    def digest(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while True:
                checkpoint()
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                value.update(chunk)
        return value.hexdigest()

    source_identity = identity(source)
    source_hash = digest(source)
    if identity(source) != source_identity:
        raise ValueError(f"{name} 转写在读取期间发生变化，拒绝继续处理")

    try:
        same_path = source.resolve(strict=True) == target.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"{name} 转写路径无法稳定解析，拒绝继续处理") from exc
    if same_path:
        checkpoint()
        final_hash = digest(source)
        if final_hash != source_hash or identity(source) != source_identity:
            raise ValueError(f"{name} 转写在验证期间发生变化，拒绝继续处理")
        return source

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target_identity = identity(target)
        if digest(target) != source_hash or identity(target) != target_identity:
            raise ValueError(f"规范 {name} 转写已存在且内容不一致，拒绝覆盖")
        final_hash = digest(source)
        if final_hash != source_hash or identity(source) != source_identity:
            raise ValueError(f"{name} 转写在持久化期间发生变化，拒绝继续处理")
        return target

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{name}.", suffix=".tmp",
            dir=target.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            copied_hash = hashlib.sha256()
            with source.open("rb") as stream:
                while True:
                    checkpoint()
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    copied_hash.update(chunk)
            output.flush()

        checkpoint()
        if (
            copied_hash.hexdigest() != source_hash
            or digest(temporary) != source_hash
            or identity(source) != source_identity
            or digest(source) != source_hash
        ):
            raise ValueError(f"{name} 转写在持久化期间发生变化，拒绝继续处理")
        checkpoint()
        # Publish without replacing a destination that appeared concurrently.
        # Both paths are in the task directory, so the hard link is atomic and
        # the temporary file can be removed after the canonical name is bound.
        try:
            os.link(temporary, target)
        except FileExistsError:
            target_identity = identity(target)
            if digest(target) != source_hash or identity(target) != target_identity:
                raise ValueError(f"规范 {name} 转写并发变化，拒绝覆盖")
        target_identity = identity(target)
        if digest(target) != source_hash or identity(target) != target_identity:
            raise ValueError(f"规范 {name} 转写发布后校验失败，拒绝继续处理")
        if digest(target) != source_hash or identity(source) != source_identity:
            raise ValueError(f"{name} 转写发布后校验失败，拒绝继续处理")
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_REVIEW_PROVENANCE_KEYS = {"review_point_id", "review_video_id", "review_task_id"}
_POINT_AUDIT_KEYS = {"evidence_point_id", "source_quotes"}


def _point_semantics(point):
    """Keep all reviewed knowledge, examples, figures and captions for fusion checks."""
    return {
        key: value for key, value in point.items()
        if key not in _REVIEW_PROVENANCE_KEYS and key not in _POINT_AUDIT_KEYS and key != "id"
    }


def _appendix_rows(doc):
    return list(((doc.get("appendix") or {}).get("cheatsheets") or []))


def _assert_reviewed_module_integrity(candidate, formal):
    """Require fusion and normalization to preserve every reviewed module as a group."""
    candidate_groups = []
    for module in candidate.get("modules", []):
        stable_ids = [
            point.get("review_point_id") for point in module.get("points", [])
        ]
        if any(not isinstance(value, str) or not value for value in stable_ids):
            raise ValueError("已审核模块缺少 stable review point ID，未提交知识文档")
        candidate_groups.append((
            {key: copy.deepcopy(value) for key, value in module.items()
             if key not in {"id", "points"}},
            stable_ids,
        ))

    formal_positions = []
    for semantics, expected_ids in candidate_groups:
        matches = []
        for index, module in enumerate(formal.get("modules", [])):
            actual_semantics = {
                key: value for key, value in module.items()
                if key not in {"id", "points"}
            }
            actual_ids = [
                point.get("review_point_id") for point in module.get("points", [])
            ]
            if actual_semantics == semantics and actual_ids == expected_ids:
                matches.append(index)
        if len(matches) != 1:
            raise ValueError(
                "项目融合或归一化改变/合并了已审核模块分组、元数据或顺序，未提交知识文档"
            )
        formal_positions.append(matches[0])

    if any(right <= left for left, right in zip(formal_positions, formal_positions[1:])):
        raise ValueError("项目融合改变了已审核模块顺序，未提交知识文档")


# 转写稿有效内容的下限（去掉空白后的字符数）。
# 低于这个数就认为「没转出东西」，直接失败而不是拿空稿去问 LLM。
MIN_TRANSCRIPT_CHARS = 200


class EmptyTranscriptError(Exception):
    """转写结果为空或过短，无法据此抽取知识点。

    单独一个异常类型，是为了让调用方能把它转成对用户有意义的提示
    （「这个视频没有可识别人声」），而不是笼统的「处理失败」。
    """


class Pipeline:
    def __init__(self, project_id: int, storage_root: Path):
        self.project_id = project_id
        self.storage_root = Path(storage_root)
        self.queue = get_queue()
        self.settings = get_settings()
        # 进度回调：测试可注入以观测 _emit 调用；生产里为 None。
        self.on_progress = None
        # 当前流水线归属的视频 id：_emit 阶段切换时要把 Video.status 同步成
        # 英文阶段枚举，前端再映射成中文文案。run() 一开始就会填上；在此之前
        # _emit 不会回写（video_id 为 None）。
        self.video_id: Optional[int] = None
        self.task_id = None
        self.transcriber = None
        # 上一次回写进库的 stage：避免同一个 stage 内多次 _emit（进度只是 0.05→0.4
        # 这种数字变化）反复写库。只有「切到新阶段」才值得落一次盘。
        self._last_synced_stage: Optional[str] = None
        self.task_budget = None
        self._task_budget_path: Optional[Path] = None
        self._task_budget_lock = threading.Lock()
        self._task_budget_snapshot_order = (-1, -1)
        self._response_diagnostics_path: Optional[Path] = None
        self._response_diagnostics = []
        self._response_diagnostics_lock = threading.Lock()
        self._grounding_response_lock = threading.Lock()
        self._grounding_response_sequence = 0
        self._activity_stage: Optional[str] = None
        self._activity_stage_started_at: Optional[str] = None
        self._activity_stage_started_mono: Optional[float] = None
        self._activity_stage_message = ""
        self._activity_stage_counts: dict[str, int] = {}
        self._activity_stage_key: Optional[str] = None
        self._llm_observer_client_sequence = 0
        self.parent_task_id: Optional[str] = None

    @staticmethod
    def _activity_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _model_input_budget(llm, reserved_output=None):
        from app.agent.runtime.model_limits import input_budget_for_llm
        return input_budget_for_llm(
            llm, reserved_output=reserved_output, fallback=140_000,
        )

    def _input_budget_kwargs(self, function, llm):
        """Pass the model-aware budget when a grounding helper supports it."""
        try:
            parameters = inspect.signature(function).parameters.values()
            supported = any(
                parameter.name == "input_budget_tokens"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            supported = False
        return ({"input_budget_tokens": self._model_input_budget(llm)}
                if supported else {})

    def _record_activity(self, task_id, key, **fields):
        """Persist public activity on a best-effort basis."""
        if not task_id:
            return None
        recorder = getattr(self.queue, "record_activity", None)
        if not callable(recorder):
            return None
        try:
            return recorder(task_id, key, **fields)
        except Exception:
            # Observability must never change a model request or submission result.
            return None

    def _publish_public_progress_segment(self, task_id, event, *, stage, progress):
        if not isinstance(event, dict):
            return
        segment_id = event.get("segment_id")
        text = event.get("text")
        if not isinstance(segment_id, str) or not isinstance(text, str) or not text:
            return
        try:
            self.queue.publish(
                task_id, _write=True, status="running", stage=stage,
                progress=progress, event="token", segment_id=segment_id,
                token=text, phase="working",
            )
        except (TaskCancelled, ContextBudgetError, ModelAttemptBudgetExceeded):
            raise
        except Exception:
            # Public activity is observational; losing a segment cannot stop work.
            pass

    def _activity_task_identity(self, task_id):
        """Return only known parent/child IDs for a pipeline activity row."""
        parent_task_id = getattr(self, "parent_task_id", None)
        if (
            not isinstance(parent_task_id, str)
            or not parent_task_id.strip()
            or not isinstance(task_id, str)
            or not task_id
        ):
            return {}
        return {
            "parent_task_id": parent_task_id,
            "child_task_id": task_id,
        }

    def _bind_activity(self, task_id, work):
        binder = getattr(self.queue, "bind_activity", None)
        if callable(binder):
            try:
                binder(task_id, Path(work))
            except Exception:
                pass

    def _close_activity_stage(self, task_id, status, *, error=None):
        stage = self._activity_stage
        key = self._activity_stage_key
        if not stage or not key:
            return
        elapsed = 0
        if self._activity_stage_started_mono is not None:
            elapsed = max(0, int((time.monotonic() - self._activity_stage_started_mono) * 1000))
        detail = self._activity_stage_message
        if status == "error":
            detail = f"{detail}；阶段失败（{error or 'Error'}）" if detail else f"阶段失败（{error or 'Error'}）"
        elif status == "cancelled":
            detail = f"{detail}；阶段已取消" if detail else "阶段已取消"
        self._record_activity(
            task_id, key, step="pipeline_stage", label=stage, status=status,
            detail=detail, actor="orchestrator", stage=stage,
            execution="workflow", started_at=self._activity_stage_started_at,
            finished_at=self._activity_now(), elapsed_ms=elapsed, error=error,
            **self._activity_task_identity(task_id),
        )

    @contextmanager
    def _resume_activity_error_guard(self, task_id):
        """Close the currently observed stage before the recovery worker finalizes."""
        try:
            yield
        except TaskCancelled:
            self._close_activity_stage(task_id, "cancelled", error="TaskCancelled")
            raise
        except Exception as exc:
            self._close_activity_stage(task_id, "error", error=type(exc).__name__)
            raise

    @staticmethod
    def _draft_tool_arguments_summary(tool_name, args):
        """Expose bounded argument metadata without publishing query or prompt text."""
        args = args if isinstance(args, dict) else {}
        summary = {}
        if tool_name == "search_web":
            query = args.get("query")
            if isinstance(query, str):
                summary["query_length"] = min(len(query), 1000)
            limit = args.get("limit")
            if type(limit) is int:
                summary["limit"] = limit
        elif tool_name == "read_web":
            summary["url_provided"] = isinstance(args.get("url"), str)
            start = args.get("start")
            if type(start) is int:
                summary["start"] = start
            find = args.get("find")
            if isinstance(find, str):
                summary["find_length"] = min(len(find), 200)
        elif tool_name == "read_video_evidence":
            for key in ("start", "count"):
                if type(args.get(key)) is int:
                    summary[key] = args[key]
        elif tool_name == "view_video_frame":
            second = args.get("second")
            if type(second) is int:
                summary["second"] = second
        return summary

    def _draft_activity_callbacks(self, task_id, llm, *, recovery=False):
        """Build the same public author/draft observer for normal and resumed drafts."""
        input_budget = self._model_input_budget(llm)
        round_number = 0
        label_prefix = "恢复初稿作者" if recovery else "初稿作者"

        def draft_chat_tools(messages, *, tools, response_format=None, provider=None):
            nonlocal round_number
            self._checkpoint()
            round_number += 1
            schema_options = {}
            if response_format is not None:
                schema_options["response_format"] = response_format
            if provider is not None:
                schema_options["provider"] = provider
            response = chat_with_heartbeat(
                self._emit, llm, messages, task_id=task_id,
                label=("恢复初稿作者取证并抽取知识点" if recovery
                       else "初稿作者取证并抽取知识点"),
                stage="draft", progress=0.82,
                method="chat_tools", tools=tools,
                max_tokens=self.settings.llm_max_tokens,
                **schema_options,
                activity_context={
                    "actor": "author", "agent_id": "draft-author",
                    "stage": "draft", "round": round_number,
                    "execution": "workflow",
                },
            )
            self._checkpoint()
            return response

        def draft_chat_text(messages, *, max_tokens):
            self._checkpoint()
            response = chat_with_heartbeat(
                self._emit, llm, messages, task_id=task_id,
                label="压缩工作记录", stage="draft", progress=0.82,
                method="chat_text", max_tokens=max_tokens,
                activity_context={
                    "actor": "author", "agent_id": "draft-author",
                    "stage": "draft", "round": round_number + 1,
                    "execution": "workflow",
                },
            )
            self._checkpoint()
            return response

        def draft_tool_event(event):
            event = event if isinstance(event, dict) else {}
            if event.get("event") == "public_progress":
                self._publish_public_progress_segment(
                    task_id, event, stage="draft", progress=0.82,
                )
                return
            if event.get("event") == "context_compaction":
                summary = event.get("summary") if isinstance(event.get("summary"), str) else ""
                seq = event.get("seq") if type(event.get("seq")) is int else 1
                source_refs = event.get("source_refs") if isinstance(event.get("source_refs"), dict) else {}

                def sanitizer_safe_lists(value, key=""):
                    if isinstance(value, dict):
                        return {
                            child_key: sanitizer_safe_lists(child, str(child_key))
                            for child_key, child in value.items()
                        }
                    if isinstance(value, list):
                        singular = {
                            "source_ids": "source_id", "call_ids": "call_id",
                            "tool_names": "tool_name", "removed_message_hashes": "sha256",
                            "history_message_hashes": "sha256", "removed_hashes": "sha256",
                            "locations": "location", "urls": "url",
                        }.get(key, "value")
                        return [
                            ({singular: item} if isinstance(item, str)
                             else sanitizer_safe_lists(item, key))
                            for item in value
                        ]
                    return copy.deepcopy(value)

                arguments = {"compaction": {
                    "before_tokens": event.get("before_tokens"),
                    "after_tokens": event.get("after_tokens"),
                    "before_hash": event.get("before_hash"),
                    "after_hash": event.get("after_hash"),
                    "removed_message_count": event.get("removed_message_count"),
                    "removed_hashes": sanitizer_safe_lists(
                        event.get("removed_message_hashes", []), "removed_message_hashes",
                    ),
                    "source_refs": sanitizer_safe_lists(source_refs),
                }}
                context = self._activity_task_identity(task_id)
                recorded_seq = self._record_activity(
                    task_id, f"draft-context-compaction:{seq}",
                    step="compact", label="压缩工作记录", status="done",
                    detail=summary, result_summary=summary, arguments=arguments,
                    actor="author", agent_id="draft-author", stage="draft",
                    round=event.get("round"), execution="workflow", **context,
                )
                try:
                    self.queue.publish(
                        task_id, _write=True, status="running", stage="draft", progress=0.82,
                        event="step", step={
                            "seq": recorded_seq if type(recorded_seq) is int else seq,
                            "step": "compact", "label": "压缩工作记录",
                            "status": "done", "detail": summary,
                            "result_summary": summary, "arguments": arguments,
                            "actor": "author", "agent_id": "draft-author",
                            "stage": "draft", "round": event.get("round"),
                            "execution": "workflow", **context,
                        },
                        **({"_activity_recorded_seq": recorded_seq}
                           if type(recorded_seq) is int else {}),
                    )
                except Exception:
                    # Compaction observation is best-effort and never changes a model result.
                    pass
                return
            tool_name = event.get("tool") if isinstance(event.get("tool"), str) else "tool"
            tool_status = event.get("status") if isinstance(event.get("status"), str) else "running"
            arguments = self._draft_tool_arguments_summary(tool_name, event.get("args"))
            detail = dumps(arguments) if arguments else (
                "工具已完成" if tool_status == "done" else "正在执行工具"
            )
            event_error = event.get("error")
            error_type = (
                event_error.split(":", 1)[0][:64]
                if isinstance(event_error, str) and tool_status == "error" else None
            )
            recorded_seq = self._record_activity(
                task_id, f"draft-tool:{event.get('seq', 0)}",
                step="tool_call", label=f"{label_prefix}：{tool_name}",
                status=tool_status, detail=detail,
                actor="author", agent_id="draft-author", stage="draft",
                round=round_number, tool_name=tool_name, arguments=arguments,
                execution="workflow", elapsed_ms=event.get("elapsed_ms"),
                error=error_type,
                **self._activity_task_identity(task_id),
            )
            try:
                self.queue.publish(
                    task_id, _write=True, status="running", stage="draft", progress=0.82,
                    event="step", step={
                        "seq": recorded_seq if type(recorded_seq) is int else event.get("seq", 0),
                        "step": "draft_tool", "origin": "author", "stage": "draft",
                        "label": f"{label_prefix}：{tool_name}",
                        "status": tool_status, "detail": detail,
                        "elapsed_ms": event.get("elapsed_ms"),
                        **self._activity_task_identity(task_id),
                    },
                    **({"_activity_recorded_seq": recorded_seq}
                       if type(recorded_seq) is int else {}),
                )
            except (TaskCancelled, ContextBudgetError, ModelAttemptBudgetExceeded):
                raise
            except Exception:
                # Legacy SSE delivery is observational; the tool result/report
                # already exists and must not become a failed tool call.
                pass

        draft_chat_tools.input_budget_tokens = input_budget
        draft_chat_tools.supports_structured_output = bool(
            isinstance(llm, _RUNTIME_LLM_CLIENT_TYPE)
            and llm.base_url == "https://openrouter.ai/api/v1"
            and llm.model == "dots-studio/dots-3-note-preview:free"
        )
        draft_chat_tools.chat_text = draft_chat_text
        draft_chat_tools.summary_input_budget_tokens = self._model_input_budget(
            llm, reserved_output=4096,
        )
        return draft_chat_tools, draft_tool_event

    def _record_llm_activity(self, event, *, actor="author", agent_id="video-author",
                             stage=None, execution="workflow", cycle=None,
                             observer_client_id=None):
        if not isinstance(event, dict) or not self.task_id:
            return
        context = {
            "actor": actor, "agent_id": agent_id, "stage": stage,
            "execution": execution, "cycle": cycle,
        }
        for key in ("actor", "agent_id", "parent_task_id", "child_task_id",
                    "stage", "round", "cycle", "tool_name", "execution"):
            if event.get(key) is not None:
                context[key] = event[key]
        if self.parent_task_id:
            context.setdefault("parent_task_id", self.parent_task_id)
            context.setdefault("child_task_id", self.task_id)
        method = event.get("method") if isinstance(event.get("method"), str) else "chat"
        model = event.get("model") if isinstance(event.get("model"), str) else "模型"
        status = event.get("status", "running")
        labels = {
            "chat": "生成结构化结果", "chat_json": "生成结构化结果",
            "chat_text": "生成文本回复", "chat_tools": "模型工具轮",
            "stream_tools": "流式模型工具轮", "stream_text": "流式文本回复",
        }
        elapsed = event.get("elapsed_ms")
        if status == "running":
            detail = f"请求 {model}：{labels.get(method, method)}"
        elif status == "retrying":
            detail = f"{model} 请求第 {event.get('attempts', 2)} 次，正在重试"
        elif status == "done":
            usage = [
                f"输入 {event['prompt_tokens']} tokens" if type(event.get("prompt_tokens")) is int else None,
                f"输出 {event['completion_tokens']} tokens" if type(event.get("completion_tokens")) is int else None,
                f"推理 token 计数 {event['reasoning_tokens']}" if type(event.get("reasoning_tokens")) is int else None,
            ]
            usage_text = "；".join(item for item in usage if item)
            detail = f"{model} 已返回结果；{elapsed} ms"
            if event.get("finish_reason"):
                detail += f"；结束原因 {event['finish_reason']}"
            if usage_text:
                detail += f"；{usage_text}"
        elif status == "cancelled":
            detail = f"{model} 调用已取消"
        else:
            detail = f"{model} 调用失败（{event.get('error', 'Error')}）"
        call_id = event.get("call_id")
        if type(call_id) is not int:
            call_id = 0
        identity = ":".join(str(value) for value in (
            context.get("actor", "orchestrator"), context.get("agent_id", "agent"),
            context.get("cycle", ""), observer_client_id or "client-0", call_id,
        ))
        fields = {
            "step": "llm_call", "label": labels.get(method, "调用模型"),
            "status": status, "detail": detail, "actor": context.get("actor"),
            "agent_id": context.get("agent_id"), "parent_task_id": context.get("parent_task_id"),
            "child_task_id": context.get("child_task_id"), "stage": context.get("stage"),
            "round": context.get("round"), "cycle": context.get("cycle"),
            "tool_name": context.get("tool_name"), "execution": context.get("execution"),
            "model": model, "started_at": event.get("started_at"),
            "finished_at": event.get("finished_at"), "elapsed_ms": elapsed,
            "result_summary": detail,
        }
        for name in ("attempts", "retry_count", "finish_reason", "max_tokens", "prompt_tokens",
                     "completion_tokens", "reasoning_tokens", "total_tokens"):
            value = event.get(name)
            if type(value) in (int, str):
                fields[name] = value
        if status in {"error", "cancelled"} and isinstance(event.get("error"), str):
            fields["error"] = event["error"]
        self._record_activity(self.task_id, f"llm:{identity}", **fields)

    def _emit(self, task_id: str, stage: str, progress: float, message: str) -> None:
        """向任务队列写进度，并（可选）回调给观察者。

        task_id 必需：队列是进度真值源，SSE 靠 task 状态变化驱动。
        stage 切换时还要把 Video.status 同步成英文枚举（A16 修复），这样视频
        卡片上的状态标签才不永远停在 "downloading"——否则页面一刷新、task 状态
        丢了，卡片和真实进度就脱节。
        """
        if self.queue.is_cancelled(task_id):
            return
        now = self._activity_now()
        if self._activity_stage is not None and self._activity_stage != stage:
            previous_elapsed = 0
            if self._activity_stage_started_mono is not None:
                previous_elapsed = max(0, int((time.monotonic() - self._activity_stage_started_mono) * 1000))
            self._record_activity(
                task_id, self._activity_stage_key or f"pipeline:stage:{self._activity_stage}:1",
                step="pipeline_stage", label=self._activity_stage, status="done",
                detail=self._activity_stage_message, actor="orchestrator",
                stage=self._activity_stage, execution="workflow",
                started_at=self._activity_stage_started_at, finished_at=now,
                elapsed_ms=previous_elapsed,
                **self._activity_task_identity(task_id),
            )
        if self._activity_stage != stage:
            self._activity_stage = stage
            self._activity_stage_counts[stage] = self._activity_stage_counts.get(stage, 0) + 1
            self._activity_stage_key = (
                f"pipeline:stage:{stage}:{self._activity_stage_counts[stage]}"
            )
            self._activity_stage_started_at = now
            self._activity_stage_started_mono = time.monotonic()
        elif self._activity_stage_started_mono is None:
            self._activity_stage_started_mono = time.monotonic()
        self._activity_stage_message = message
        self.queue.update(
            task_id, status="running", progress=progress, message=message, stage=stage
        )
        stage_elapsed = 0
        if self._activity_stage_started_mono is not None:
            stage_elapsed = max(0, int((time.monotonic() - self._activity_stage_started_mono) * 1000))
        self._record_activity(
            task_id, self._activity_stage_key or f"pipeline:stage:{stage}:1",
            step="pipeline_stage", label=stage, status="running", detail=message,
            actor="orchestrator", stage=stage, execution="workflow",
            started_at=self._activity_stage_started_at, elapsed_ms=stage_elapsed,
            **self._activity_task_identity(task_id),
        )
        if self.on_progress:
            self.on_progress({"stage": stage, "progress": progress, "message": message})
        # 只在「切到新阶段」且属于要回写的阶段集合时落库，省掉无意义的重复写。
        if stage in _STATUS_SYNC_STAGES and stage != self._last_synced_stage:
            self._sync_video_status(stage)
            self._last_synced_stage = stage

    def _sync_video_status(self, status_enum: str) -> None:
        """把当前视频的 status 回写成英文阶段枚举。

        后端只存英文枚举，中文文案交给前端映射——数据库里不该出现中文状态，
        否则多端展示时没法复用、也没法做等值比较。
        video_id 还没准备好（None）时直接跳过，避免拿空主键去查库。
        """
        if self.video_id is None:
            return
        with session_scope() as db:
            v = db.get(Video, self.video_id)
            if v:
                v.status = status_enum

    def _asr_progress(self, task_id: str, pct: float | None, message: str) -> None:
        """把 base 转写的 0~100% 映射到流水线 asr 阶段区间 0.45~0.7 并回传。

        base 才是用户等待的主线，它的进度 = 「离出草稿还有多远」。
        ``pct`` 可能为 ``None``（拿不到总时长的降级分支）：此时无法算百分比，
        固定映射到 0.5 并沿用消息，保证界面仍有「在转写」的提示而不崩。
        """
        if pct is None:
            self._emit(task_id, "asr", 0.5, message)
        else:
            mapped = 0.45 + (max(0.0, min(100.0, pct)) / 100.0) * (0.7 - 0.45)
            self._emit(task_id, "asr", mapped, message)

    def _load_existing_doc(self) -> dict:
        with session_scope() as db:
            p = db.get(Project, self.project_id)
            if p and p.doc and p.doc.doc_json:
                try:
                    return json.loads(p.doc.doc_json)
                except json.JSONDecodeError as exc:
                    raise ValueError("已有知识文档无法解析，本次未覆盖") from exc
        return {}

    def _chat_with_heartbeat(
        self,
        llm,
        task_id: str,
        messages: list[dict],
        label: str,
        stage: str,
        progress: float,
        **call_kwargs,
    ) -> dict:
        """调 ``llm.chat``，期间每 30 秒发一条进度心跳。

        实现在 :mod:`app.agent.runtime.heartbeat`（与 agent 工具循环共用）。这里保留一层
        薄封装，是为了让调用点仍然传 ``self._emit``——它除了发事件还会回写
        ``Video.status``，而通用版只管发事件。

        为什么必须有心跳（实测踩过的坑）：``llm.chat`` 是一次阻塞调用。调用期间
        没有任何事件发出去，前端的进度条完全静止 —— 用户看到的就是「卡住了」。
        实测一个 3 分 12 秒的视频，术语校验这一步静默了近 15 分钟。
        """
        call_kwargs.setdefault("activity_context", {
            "actor": "author", "agent_id": "video-author",
            "stage": stage, "execution": "workflow",
        })
        return chat_with_heartbeat(
            self._emit,
            llm,
            messages,
            task_id=task_id,
            label=label,
            stage=stage,
            progress=progress,
            method="chat",
            **call_kwargs,
        )

    def _persist_task_budget(self, snapshot):
        """Expose count-only budget diagnostics in the task snapshot and artifact."""
        path = self._task_budget_path
        if path is not None:
            order = (int(snapshot.get("used", 0)), int(snapshot.get("denied", 0)))
            with self._task_budget_lock:
                if order >= self._task_budget_snapshot_order:
                    self._task_budget_snapshot_order = order
                    temporary = path.with_suffix(path.suffix + ".tmp")
                    temporary.write_text(dumps(snapshot), encoding="utf-8")
                    temporary.replace(path)
        if self.task_id:
            self.queue.update(self.task_id, model_budget=dict(snapshot))

    def _init_task_budget(self, task_id, work, options):
        limit = options.get("max_model_attempts", 200)
        self.task_id = task_id
        self.parent_task_id = None
        try:
            task_state = self.queue.get_status(task_id) or {}
            parent_task_id = task_state.get("parent_task_id")
            if isinstance(parent_task_id, str) and parent_task_id:
                self.parent_task_id = parent_task_id
        except Exception:
            pass
        self._task_budget_path = Path(work) / "task_budget.json"
        self._task_budget_snapshot_order = (-1, -1)
        self._response_diagnostics_path = Path(work) / "llm-response-diagnostics.json"
        self._response_diagnostics = []
        self._grounding_response_sequence = 0
        self._activity_stage = None
        self._activity_stage_started_at = None
        self._activity_stage_started_mono = None
        self._activity_stage_message = ""
        self._activity_stage_counts = {}
        self._activity_stage_key = None
        with _LLM_OBSERVER_SEQUENCE_LOCK:
            self._llm_observer_client_sequence = 0
        self._bind_activity(task_id, work)
        self.task_budget = TaskBudget(
            max_attempts=limit,
            on_change=self._persist_task_budget,
        )
        self._persist_task_budget(self.task_budget.snapshot())

    def _persist_response_diagnostic(self, row):
        """Atomically append one already-sanitized LLM response receipt."""
        path = self._response_diagnostics_path
        if path is None:
            return
        with self._response_diagnostics_lock:
            self._response_diagnostics.append(dict(row))
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                temporary.write_text(
                    dumps({"schema_version": 1, "responses": self._response_diagnostics}),
                    encoding="utf-8",
                )
                temporary.replace(path)
            except Exception:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

    def _persist_grounding_response(self, *, result, label, response_format):
        """Atomically retain a parsed grounding response before parser gates run."""
        diagnostics_path = getattr(self, "_response_diagnostics_path", None)
        if diagnostics_path is None:
            return None
        from app.agent.knowledge.grounding_schema import GROUNDING_SCHEMA_POLICY, COMPLETENESS_SCHEMA_POLICY

        directory = Path(diagnostics_path).parent / "grounding_responses"
        directory.mkdir(parents=True, exist_ok=True)
        schema_sha256 = None
        if response_format is not None:
            canonical = json.dumps(
                response_format, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            schema_sha256 = hashlib.sha256(canonical).hexdigest()
        receipt_lock = getattr(self, "_grounding_response_lock", None)
        if receipt_lock is None:
            receipt_lock = threading.Lock()
            self._grounding_response_lock = receipt_lock
        with receipt_lock:
            sequence = max(0, int(getattr(self, "_grounding_response_sequence", 0)))
            while True:
                sequence += 1
                target = directory / f"{sequence:06d}.json"
                receipt = {
                    "schema_version": 1,
                    "sequence": sequence,
                    "label": str(label)[:160],
                    "schema_policy": (
                        (COMPLETENESS_SCHEMA_POLICY
                         if response_format.get("json_schema", {}).get("name") == "video_completeness_rows"
                         else GROUNDING_SCHEMA_POLICY)
                        if response_format is not None else None
                    ),
                    "schema_sha256": schema_sha256,
                    "result": result,
                }
                fd, temporary_name = tempfile.mkstemp(
                    prefix=".grounding-response-", suffix=".tmp", dir=directory,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        handle.write(dumps(receipt))
                        handle.flush()
                        os.fsync(handle.fileno())
                    try:
                        # A hard-link publishes a complete file and refuses to
                        # replace an earlier task receipt if another writer won.
                        os.link(temporary_name, target)
                    except FileExistsError:
                        continue
                    self._grounding_response_sequence = sequence
                    return target
                finally:
                    try:
                        Path(temporary_name).unlink(missing_ok=True)
                    except OSError:
                        pass

    def _grounding_call(
        self, llm, task_id, content, *, label, stage, progress,
        response_format=None, provider=None,
    ):
        """Use configured tokens, whitelist schema options, and retain parsed output."""
        schema_options = {}
        if response_format is not None:
            schema_options["response_format"] = response_format
        if provider is not None:
            schema_options["provider"] = provider
        result = self._chat_with_heartbeat(
            llm,
            task_id,
            [{"role": "user", "content": content}],
            label=label,
            stage=stage,
            progress=progress,
            max_tokens=self.settings.llm_max_tokens,
            **schema_options,
        )
        self._persist_grounding_response(
            result=result, label=label, response_format=response_format,
        )
        return result

    def _grounding_callback(self, llm, task_id, *, label, stage, progress):
        """Create a task-bound grounding client with an exact Dots capability bit."""
        input_budget = self._model_input_budget(llm)
        supports_structured_output = bool(
            isinstance(llm, _RUNTIME_LLM_CLIENT_TYPE)
            and llm.base_url == "https://openrouter.ai/api/v1"
            and llm.model == "dots-studio/dots-3-note-preview:free"
        )

        def grounding_callback(content, *, response_format=None, provider=None):
            if not supports_structured_output and (
                response_format is not None or provider is not None
            ):
                raise ValueError("当前模型不支持 grounding 结构化输出参数")
            return self._grounding_call(
                llm, task_id, content, label=label, stage=stage, progress=progress,
                response_format=response_format, provider=provider,
            )

        grounding_callback.supports_structured_output = supports_structured_output
        grounding_callback.input_budget_tokens = input_budget
        return grounding_callback

    def _new_llm(self, *, actor="author", agent_id="video-author", stage=None,
                 execution="workflow", cycle=None):
        """独立审查者使用同一用户配置的新客户端，不共享对话历史。"""
        with _LLM_OBSERVER_SEQUENCE_LOCK:
            self._llm_observer_client_sequence = (
                getattr(self, "_llm_observer_client_sequence", 0) + 1
            )
            observer_client_id = f"client-{self._llm_observer_client_sequence}"
        return LLMClient(
            self.settings.llm_base_url,
            self.settings.llm_api_key,
            self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout,
            should_cancel=lambda: self.queue.is_cancelled(self.task_id),
            task_budget=self.task_budget,
            on_response_diagnostic=self._persist_response_diagnostic,
            on_call_event=lambda event: self._record_llm_activity(
                event, actor=actor, agent_id=agent_id, stage=stage,
                execution=execution, cycle=cycle,
                observer_client_id=observer_client_id,
            ),
        )

    def _checkpoint(self):
        if self.task_id and self.queue.is_cancelled(self.task_id):
            raise TaskCancelled()

    def run(
        self,
        task_id: str,
        video_id: int,
        bvid: str,
        cid: str,
        intent: str,
        terms: Optional[list[str]] = None,
        options=None,
    ) -> dict:
        self.task_id, self.video_id = task_id, video_id
        # 同视频任务依次运行，避免重复争用 ASR。不同视频可并行，最终写入按项目互斥。
        with project_guard(("video", video_id)):
            try:
                self._checkpoint()
                return self._run(
                    task_id, video_id, bvid, cid, intent, terms or [], options or {}
                )
            except TaskCancelled:
                self._close_activity_stage(task_id, "cancelled", error="TaskCancelled")
                raise
            except Exception as exc:
                self._close_activity_stage(task_id, "error", error=type(exc).__name__)
                raise
            finally:
                if self.transcriber is not None and hasattr(self.transcriber, "stop"):
                    self.transcriber.stop()

    def _run(self, task_id, video_id, bvid, cid, intent, terms, options):
        work = (
            self.storage_root
            / f"video_{video_id}"
            / "tasks"
            / hashlib.sha256(task_id.encode()).hexdigest()[:24]
        )
        work.mkdir(parents=True, exist_ok=False)
        self._init_task_budget(task_id, work, options)
        # Capture the baseline before any model work. Local recovery checkpoints
        # created after the coverage draft must bind this original document.
        existing = self._load_existing_doc()
        client = BilibiliClient()
        self._emit(task_id, "download", 0.05, "解析视频信息")
        with session_scope() as db:
            video = db.get(Video, video_id)
            duration = video.duration or 0 if video else 0
        local_art = (
            {}
            if bvid
            else json.loads(
                (self.storage_root / f"video_{video_id}" / "local.json").read_text()
            )
        )
        llm = self._new_llm()
        preflight = None
        if bvid:
            from app.agent.media.preflight import inspect_video
            self._emit(task_id, "download", 0.05, "检查视频介绍与预览抽帧，判断是否适合整理")
            preflight = inspect_video(client, bvid, cid, work,
                lambda content: self._chat_with_heartbeat(llm, task_id,
                    [{"role": "user", "content": content}],
                    label="判断视频内容", stage="download", progress=0.05,
                    max_tokens=min(self.settings.llm_max_tokens, 4_000)), self._checkpoint)
            self._checkpoint()
            if preflight['decision'] != 'accept':
                reason = preflight['reason']
                if preflight['decision'] == 'unknown':
                    reason = "现有信息不足以确认是知识视频，已暂停整理：" + reason
                self._close_activity_stage(task_id, "done")
                with self.queue.commit_guard(task_id):
                    with session_scope() as db:
                        video = db.get(Video, video_id)
                        if video:
                            video.status = "skipped"
                            video.error = None
                            video.artifacts_json = dumps({"task_dir": str(work), "preflight": preflight})
                    self.queue.update(task_id, status="done", progress=1.0,
                        message="已跳过：" + reason, result={"skipped": True, "reason": reason})
                return {"skipped": True, "reason": reason}
        transcript, subtitle_report = load_subtitles(
            client,
            bvid,
            cid,
            work,
            duration,
            local_path=local_art.get("video_path") or local_art.get("audio_path"),
            checkpoint=self._checkpoint,
        )
        art = {**local_art, "subtitle": subtitle_report, "duration_sec": duration, "preflight": preflight}
        subtitle_source = transcript
        art["task_dir"] = str(work)
        art["intent"] = intent
        art["options"] = options
        if bvid:
            art.update(download_audio(
                client,
                bvid,
                cid,
                work,
                self.settings.bilibili_quality,
                progress_cb=lambda fraction: self._download_progress(task_id, fraction),
            ))
        elif not art.get("audio_path"):
            raise ValueError("本地媒体缺少音频工件，尚未支持完整本地导入")
        audio_path = Path(art["audio_path"])
        if not audio_path.is_file():
            raise ValueError("音频素材不存在，未开始 ASR 或画面准备")
        if not duration:
            duration = float(art.get("duration_sec") or local_art.get("duration_sec") or 0)
        if not duration > 0:
            from app.agent.media.media_preparation import probe_media_duration
            duration = probe_media_duration(audio_path, self._checkpoint)
        art["duration_sec"] = duration
        self._checkpoint()
        self._emit(task_id, "download", 0.4, "音频已就绪；接下来并行准备 base→medium ASR 与视频逐秒原始帧")

        if bvid:
            source_key = f"bilibili:{bvid}:{cid}"
        else:
            source_identity = hashlib.sha256(
                str(Path(art.get("video_path") or audio_path).resolve()).encode("utf-8")
            ).hexdigest()[:16]
            source_key = f"local:{video_id}:{source_identity}"
        initial_video_path = Path(art["video_path"]) if art.get("video_path") else None
        resolved_video_paths = []
        playback_results = []

        def resolve_bilibili_video(checkpoint):
            checkpoint()
            path = download_video(
                client, bvid, cid, work, progress_cb=lambda _fraction: checkpoint()
            )
            path = Path(path)
            resolved_video_paths.append(path)
            # The cached playback artifact depends only on downloaded audio/video.
            # Create it in the video branch as soon as that source is ready, while
            # medium ASR may still be running; joint analysis still waits for both
            # original-source branches to join below.
            from app.services.playback import prepare_bilibili_cache
            try:
                playback_results.append(prepare_bilibili_cache(
                    self.project_id, video_id, str(audio_path), str(path), checkpoint=checkpoint,
                ))
            except TaskCancelled:
                raise
            except Exception as exc:
                playback_results.append({
                    "cache_status": "invalid", "error_type": type(exc).__name__,
                })
            return MediaSource(source_key=source_key, path=path)

        request = MediaPreparationRequest(
            source_key=source_key,
            audio_path=audio_path,
            video_source=(
                None if (not bvid and initial_video_path is None)
                else MediaSource(source_key, initial_video_path) if initial_video_path is not None
                else None
            ),
            work_dir=work,
            expected_duration_seconds=duration,
        )
        engine_factory = lambda: WhisperTranscriber(
            model_dir=self.settings.whisper_model_dir,
            device=self.settings.asr_device,
            compute_type=self.settings.asr_compute_type,
        )
        last_media_heartbeat = [0.0]

        def media_checkpoint():
            self._checkpoint()
            now = time.monotonic()
            if now - last_media_heartbeat[0] >= 30:
                last_media_heartbeat[0] = now
                self._emit(task_id, "asr", 0.52,
                           "base→medium ASR 与视频下载/原始 1fps 抽帧并行处理中")

        try:
            prepared = prepare_media(
                request,
                terms=terms,
                checkpoint=media_checkpoint,
                asr_engine_factory=engine_factory,
                video_source_resolver=resolve_bilibili_video if bvid else None,
            )
        except (MediaPreparationCancelled, MediaPreparationError) as exc:
            failure_result = getattr(exc, "result", None)
            failure = {
                "status": getattr(failure_result, "status", "cancelled" if isinstance(exc, TaskCancelled) else "failed"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "branches": [
                    {"name": row.name, "status": row.status, "elapsed_seconds": row.elapsed_seconds,
                     "media_duration_seconds": row.media_duration_seconds,
                     "error_type": row.error_type, "error_message": row.error_message}
                    for row in getattr(failure_result, "branches", ())
                ],
            }
            (work / "media_preparation_failure.json").write_text(dumps(failure), encoding="utf-8")
            art["media_preparation"] = failure
            (work / "evidence.json").write_text(dumps(art), encoding="utf-8")
            raise
        except Exception as exc:
            art["media_preparation"] = {
                "status": "failed", "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            (work / "evidence.json").write_text(dumps(art), encoding="utf-8")
            raise

        prepared = prepared
        art["media_preparation"] = _media_preparation_artifact(prepared)
        if resolved_video_paths:
            art["video_path"] = str(resolved_video_paths[-1])
        if playback_results:
            art["playback"] = playback_results[-1]
        if prepared.status == "ready":
            prepared.require_joint_ready()
            base_path = prepared.transcript.base_path
            medium_path = prepared.transcript.medium_path
            art["base_path"] = str(base_path)
            art["medium_path"] = str(medium_path)
            art["visual_review"] = {"status": "pending", "reason": "音频与原始视频帧已按来源身份、哈希及时长汇合"}
        elif prepared.status == "audio_only" and not bvid:
            base_path = prepared.transcript.base_path
            medium_path = prepared.transcript.medium_path
            art["base_path"] = str(base_path)
            art["medium_path"] = str(medium_path)
            art["visual_review"] = {"status": "audio_only", "reason": "本地来源只有音频；本轮保留音频整理语义，不宣称画面审核"}
        else:
            raise ValueError("音画媒体未完整汇合，未进入知识抽取")
        # MediaPreparation keeps branch output under its private attempt
        # directory. Persist exact transcript bytes at the recovery contract's
        # canonical task paths before any knowledge stages can fail. Keep the
        # original branch artifacts untouched for audit and retain cancellation
        # checks plus source/target hash validation during publication.
        base_path = _persist_transcript_artifact(base_path, work, "base.txt", self._checkpoint)
        medium_path = _persist_transcript_artifact(medium_path, work, "medium.txt", self._checkpoint)
        art["base_path"] = str(base_path)
        art["medium_path"] = str(medium_path)
        art["model_attempt_budget"] = self.task_budget.snapshot()
        (work / "evidence.json").write_text(dumps(art), encoding="utf-8")
        transcript = base_path.read_text(encoding="utf-8")
        medium_transcript = medium_path.read_text(encoding="utf-8")
        if len(transcript.strip()) < MIN_TRANSCRIPT_CHARS:
            raise EmptyTranscriptError(
                f"转写结果过短（{len(transcript.strip())} 字符），可能没有可识别人声；未生成知识点"
            )
        if not medium_transcript.strip():
            raise ValueError("medium 未产出有效转写，未校验、未提交文档")
        self._emit(task_id, "asr", 0.70, "base 与 medium 转写完整，媒体分支已汇合")
        self._emit(task_id, "frames", 0.73, art["visual_review"]["reason"])

        video_path = Path(art["video_path"]) if art.get("video_path") else None
        if bvid and video_path is None:
            raise ValueError("未取得视频流，无法按每秒一帧分析，未提交知识文档")
        if art.get("playback", {}).get("cache_status") == "invalid":
            self._emit(task_id, "frames", 0.73, "本地播放准备失败，继续核对原始音画证据")

        art["model_attempt_budget"] = self.task_budget.snapshot()
        (work / "evidence.json").write_text(dumps(art), encoding="utf-8")
        sources = [("base", transcript), ("medium", medium_transcript)]
        if subtitle_source:
            sources.append(("subtitle", subtitle_source))
        visual_text = ""
        if not subtitle_source and options.get("hard_subtitles") and video_path is not None:
            self._emit(task_id, "frames", 0.74, "读取硬字幕（本机 OCR）")
            visual_text, report = collect_visual_evidence(
                video_path,
                work,
                duration,
                hard_subtitles=True,
                crop_bottom=options.get("subtitle_crop_bottom", 0.3),
                checkpoint=self._checkpoint,
            )
            art["hard_subtitles"] = report
            if len(visual_text) >= MIN_TRANSCRIPT_CHARS and report.get("recognized", 0) >= 0.9 * max(report.get("attempted", 1), 1):
                art["subtitle"] = {**report, "status": "usable"}
        if visual_text:
            sources.append(("hard_subtitle_ocr", visual_text))
        if prepared.status == "ready":
            from app.agent.media.visual_review import review_frames
            visual, report = review_frames(
                str(video_path), work, float(duration), sources,
                lambda content: self._chat_with_heartbeat(
                    llm, task_id, [{"role": "user", "content": content}],
                    label="逐秒联合分析画面与音频转写", stage="frames", progress=0.78,
                    max_tokens=self.settings.llm_max_tokens,
                ), self._checkpoint,
                progress=lambda message: self._emit(task_id, "frames", 0.78, message),
                prepared_frames=prepared.frames,
                media_identity=prepared.identity,
            )
            sources.append(("visual_model_observations", visual))
            art["visual_review"] = report
        else:
            art["visual_review"]["media_identity"] = art["media_preparation"].get("identity")
        (work / "evidence.json").write_text(dumps(art), encoding="utf-8")
        self._emit(task_id, "draft", 0.82, "抽取知识点")
        joint_evidence = "\n\n".join(f"【{name}】\n{text}" for name, text in sources)
        prompt_options = dict(options or {})
        historical_context = prompt_options.pop("historical_context", None)
        prompt_kwargs = {"options": prompt_options}
        if historical_context is not None:
            prompt_kwargs["historical_context"] = historical_context
        prompt = build_extraction_prompt(
            joint_evidence,
            terms,
            existing_summary=dumps(document_for_model(existing)),
            intent=intent,
            **prompt_kwargs,
        )
        limits = _budget_limits(joint_evidence)

        from app.agent.knowledge.draft_writer import generate_draft
        from app.services.recovery_versions import (
            capture_initial_draft_intent,
            add_generation_prompt_identity,
            initial_prompt_identity_matches,
        )
        candidate_generation_intent = capture_initial_draft_intent(
            prompt, raw_sources=sources,
        )

        def draft_progress(message):
            self._emit(task_id, "draft", 0.82, message)

        draft_chat_tools, draft_tool_event = self._draft_activity_callbacks(
            task_id, llm,
        )

        doc, draft_evidence = generate_draft(
            prompt=prompt, raw_sources=sources, visual_report=art["visual_review"],
            work=work, chat_tools=draft_chat_tools, checkpoint=self._checkpoint,
            progress=draft_progress, on_tool_event=draft_tool_event,
            input_budget=getattr(draft_chat_tools, "input_budget_tokens", 140_000),
            document_validator=lambda candidate: checked_document(candidate, limits),
        )
        self._checkpoint()
        if not initial_prompt_identity_matches(candidate_generation_intent, draft_evidence):
            raise ValueError("初稿作者报告的实际提示与已捕获恢复指纹不匹配")
        art["draft_evidence"] = {
            "artifact": "draft_evidence.json",
            "status": draft_evidence["status"],
            "candidate_sha256": draft_evidence["candidate_sha256"],
            "tool_budget": copy.deepcopy(draft_evidence["tool_budget"]),
            "format_repairs": copy.deepcopy(draft_evidence["format_repairs"]),
        }
        (work / "extraction.json").write_text(dumps(doc), encoding="utf-8")


        self._emit(task_id, "verify", 0.86, "核对初稿的完整性与视频来源")
        initial_coverage_call_number = 0
        coverage_request = self._grounding_callback(
            llm, task_id, label="逐段检查内容覆盖", stage="verify", progress=0.86,
        )

        def initial_coverage_call(content, **schema_options):
            nonlocal candidate_generation_intent, initial_coverage_call_number
            candidate_generation_intent = add_generation_prompt_identity(
                candidate_generation_intent,
                f"initial_coverage_{initial_coverage_call_number:04d}",
                content,
            )
            initial_coverage_call_number += 1
            return coverage_request(content, **schema_options)

        initial_coverage_call.supports_structured_output = coverage_request.supports_structured_output

        doc, completeness = complete_document(
            doc, sources, initial_coverage_call, self._checkpoint,
            **self._input_budget_kwargs(complete_document, llm),
        )
        (work / "coverage-draft.json").write_text(
            dumps({"doc": doc, "coverage": completeness}), encoding="utf-8"
        )
        if not bvid:
            from app.services.recovery import initialize_local_checkpoint, mark_recovery_unavailable
            recorded = initialize_local_checkpoint(
                self.project_id, video_id, task_id, work,
                intent=intent, terms=terms, options=options,
                baseline_doc=existing, sources=sources,
                visual_report=art.get("visual_review") or {},
                candidate_generation_fingerprint=candidate_generation_intent,
            )
            if not recorded and not (work / "recovery-unavailable.json").exists():
                try:
                    mark_recovery_unavailable(work, "完整本地恢复检查点未能建立")
                except OSError:
                    pass
        return self._continue_after_coverage_draft(
            task_id, video_id, bvid, cid, intent, terms, options, work,
            existing, doc, completeness, sources, art, llm,
            expected_baseline_sha256=None,
        )

    def resume(
        self, task_id, video_id, bvid, cid, intent, terms, options, *,
        work, coverage_draft, visual_report, sources, baseline_doc,
        baseline_doc_sha256, video_path, audio_path,
        source_coverage_draft=None, source_candidate_generation_fingerprint=None,
        target_candidate_generation_intent=None, regenerate_initial_draft=False,
    ):
        """Continue from validated persisted inputs at full source grounding."""
        self.task_id, self.video_id = task_id, video_id
        with project_guard(("video", video_id)), self._resume_activity_error_guard(task_id):
            self._checkpoint()
            work = Path(work)
            self._init_task_budget(task_id, work, options)
            duration = 0.0
            with session_scope() as db:
                video = db.get(Video, video_id)
                if video is not None:
                    duration = float(video.duration or 0)
            art = {
                "task_dir": str(work),
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "duration_sec": duration,
                "visual_review": visual_report,
                "subtitle": {"status": "persisted_audio_transcripts_reused", "source": "recovery"},
                "intent": intent,
                "options": options,
                "recovery": {"resumed_from_stage": "grounding", "reused_asr": True, "reused_visual_review": True},
            }
            llm = self._new_llm()
            if regenerate_initial_draft:
                if coverage_draft is not None:
                    raise ValueError("待重新生成恢复不能把来源旧候选当作当前覆盖候选")
                from app.agent.knowledge.draft_writer import generate_draft
                from app.services.recovery_versions import (
                    add_generation_prompt_identity,
                    candidate_generation_prefix_matches,
                    capture_initial_draft_intent,
                    is_candidate_generation_intent,
                    initial_prompt_identity_matches,
                )
                prompt_options = dict(options or {})
                # Program execution limits are not content instructions and
                # are excluded by current_candidate_generation_intent too.
                prompt_options.pop("max_model_attempts", None)
                historical_context = prompt_options.pop("historical_context", None)
                transcript = "\n\n".join(f"【{name}】\n{text}" for name, text in sources)
                prompt_kwargs = {"options": prompt_options}
                if historical_context is not None:
                    prompt_kwargs["historical_context"] = historical_context
                prompt = build_extraction_prompt(
                    transcript,
                    terms,
                    existing_summary=dumps(document_for_model(baseline_doc)),
                    intent=intent,
                    **prompt_kwargs,
                )
                generation_intent = capture_initial_draft_intent(
                    prompt, raw_sources=sources,
                )
                if not candidate_generation_prefix_matches(
                    target_candidate_generation_intent, generation_intent,
                ):
                    if not is_candidate_generation_intent(target_candidate_generation_intent):
                        mismatch = "排队时目标意图无效"
                    elif target_candidate_generation_intent["code_components"] != generation_intent["code_components"]:
                        expected_components = target_candidate_generation_intent["code_components"]
                        actual_components = generation_intent["code_components"]
                        changed = sorted(
                            name for name in set(expected_components) | set(actual_components)
                            if expected_components.get(name) != actual_components.get(name)
                        )
                        mismatch = "代码依赖摘要变化：" + ",".join(changed)
                    else:
                        mismatch = "精确初稿提示摘要变化"
                    raise ValueError(
                        f"恢复初稿{mismatch}，已保留已验证原始证据"
                    )
                if (work / "extraction.json").exists() or (work / "coverage-draft.json").exists():
                    raise ValueError("待重生成恢复目录已有未登记候选，拒绝覆盖")
                limits = _budget_limits(transcript)

                def draft_progress(message):
                    self._emit(task_id, "draft", 0.82, message)

                draft_chat_tools, draft_tool_event = self._draft_activity_callbacks(
                    task_id, llm, recovery=True,
                )

                doc, draft_evidence = generate_draft(
                    prompt=prompt, raw_sources=sources, visual_report=visual_report,
                    work=work, chat_tools=draft_chat_tools, checkpoint=self._checkpoint,
                    progress=draft_progress, on_tool_event=draft_tool_event,
                    input_budget=getattr(draft_chat_tools, "input_budget_tokens", 140_000),
                    document_validator=lambda candidate: checked_document(candidate, limits),
                )
                self._checkpoint()
                if not initial_prompt_identity_matches(generation_intent, draft_evidence):
                    raise ValueError("恢复初稿作者报告的实际提示与排队目标不匹配")
                (work / "extraction.json").write_text(dumps(doc), encoding="utf-8")
                art["draft_evidence"] = {
                    "artifact": "draft_evidence.json",
                    "status": draft_evidence["status"],
                    "candidate_sha256": draft_evidence["candidate_sha256"],
                    "tool_budget": copy.deepcopy(draft_evidence["tool_budget"]),
                    "format_repairs": copy.deepcopy(draft_evidence["format_repairs"]),
                }
                self._emit(task_id, "verify", 0.86, "从已校验原始音画证据重新生成完整覆盖账本")
                coverage_call_number = 0
                coverage_request = self._grounding_callback(
                    llm, task_id, label="恢复初稿逐段检查内容覆盖", stage="verify", progress=0.86,
                )

                def initial_coverage_call(content, **schema_options):
                    nonlocal generation_intent, coverage_call_number
                    generation_intent = add_generation_prompt_identity(
                        generation_intent,
                        f"initial_coverage_{coverage_call_number:04d}",
                        content,
                    )
                    coverage_call_number += 1
                    return coverage_request(content, **schema_options)

                initial_coverage_call.supports_structured_output = coverage_request.supports_structured_output

                doc, completeness = complete_document(
                    doc, sources, initial_coverage_call, self._checkpoint,
                    **self._input_budget_kwargs(complete_document, llm),
                )
                draft = doc
                coverage_raw = dumps({"doc": doc, "coverage": completeness})
                (work / "coverage-draft.json").write_text(coverage_raw, encoding="utf-8")
                from app.services.recovery import record_regenerated_candidate
                bound_generation = record_regenerated_candidate(
                    work, generation_intent=generation_intent,
                )
                art["recovery"].update({
                    "resumed_from_stage": "initial_draft",
                    "reused_asr": True,
                    "reused_visual_review": True,
                    "regenerated_initial_draft": True,
                    "source_candidate_generation_fingerprint": source_candidate_generation_fingerprint,
                    "candidate_generation_fingerprint": bound_generation,
                })
                (work / "evidence.json").write_text(dumps(art), encoding="utf-8")
            else:
                draft = coverage_draft.get("doc") if isinstance(coverage_draft, dict) else None
                completeness = coverage_draft.get("coverage") if isinstance(coverage_draft, dict) else None
                if not isinstance(draft, dict) or not isinstance(completeness, list):
                    raise ValueError("恢复检查点缺少完整覆盖草稿")
                self._emit(task_id, "verify", 0.87, "从完整覆盖账本继续来源支持核对")
                return self._continue_after_coverage_draft(
                    task_id, video_id, bvid, cid, intent, terms, options, work,
                    baseline_doc, draft, completeness, sources, art, llm,
                    expected_baseline_sha256=baseline_doc_sha256,
                )
            return self._continue_after_coverage_draft(
                task_id, video_id, bvid, cid, intent, terms, options, work,
                baseline_doc, draft, completeness, sources, art, llm,
                expected_baseline_sha256=baseline_doc_sha256,
            )

    def _continue_after_coverage_draft(
        self, task_id, video_id, bvid, cid, intent, terms, options, work,
        existing, doc, completeness, sources, art, llm,
        expected_baseline_sha256=None,
    ):
        """Run source grounding through the guarded project commit for every path."""
        # Draft is not exposed until the complete audio/visual evidence and figures exist.
        from app.agent.media.illustrations import enrich_illustrations
        from app.worker.renderer import render_standalone_html
        self._emit(task_id, "illustrate", 0.90, "逐点检查必要的配图与例子")
        from app.agent.media.illustration_replay import runtime_replay_context
        illustration_options = self._input_budget_kwargs(enrich_illustrations, llm)
        if "replay_context" in inspect.signature(enrich_illustrations).parameters:
            illustration_options["replay_context"] = runtime_replay_context(
                work, sources, art["visual_review"], getattr(llm, "model", None),
            )
        doc, illustration_report = enrich_illustrations(
            doc, art["visual_review"],
            call=lambda content: self._chat_with_heartbeat(
                llm, task_id, [{"role": "user", "content": content}],
                label="补充必要图例", stage="illustrate", progress=0.90,
                max_tokens=self.settings.llm_max_tokens,
            ), work=work, checkpoint=self._checkpoint,
            progress=lambda message: self._emit(task_id, "illustrate", 0.90, message),
            **illustration_options,
        )
        doc = checked_document(doc)

        replayed_count = illustration_report.get("replayed_model_decisions", 0)
        if type(replayed_count) is int and replayed_count > 0:
            self._record_activity(
                task_id, "illustration-decision-replay", step="cache_replay",
                label="复用图例决策", status="done", actor="orchestrator",
                stage="illustrate", execution="workflow",
                detail=f"复用{replayed_count}个已有模型决策并重新通过图例门禁；这些决策未重新调用模型，后续来源与盲审仍须执行。",
            )

        from app.agent.knowledge.grounding import reconcile_final_coverage
        from app.agent.review.independent_review import document_hash

        issued_stable_ids = set()

        def annotate_review_provenance(candidate):
            candidate = copy.deepcopy(candidate)
            points = [
                point
                for module in candidate.get("modules", [])
                for point in module.get("points", [])
            ]
            # Reserve identities already issued for this task before considering
            # any newly added point. A coverage addition may reuse a normalized
            # display ID after grounding deleted and renumbered an earlier point.
            current_task_ids = {
                point.get("review_point_id")
                for point in points
                if point.get("review_task_id") == task_id
                and isinstance(point.get("review_point_id"), str)
                and point.get("review_point_id")
            }
            issued_stable_ids.update(current_task_ids)
            proposed_ids = {
                point.get("id")
                for point in points
                if point.get("review_task_id") != task_id
                and isinstance(point.get("id"), str)
                and point.get("id")
            }
            used_ids = set(current_task_ids)
            reserved_ids = current_task_ids | proposed_ids | issued_stable_ids
            generated_index = 1
            for point in points:
                if point.get("review_task_id") == task_id:
                    stable_id = point.get("review_point_id")
                    if isinstance(stable_id, str) and stable_id:
                        issued_stable_ids.add(stable_id)
                else:
                    stable_id = point.get("id")
                    if (isinstance(stable_id, str)
                            and stable_id in (used_ids | issued_stable_ids)):
                        while True:
                            generated_id = f"coverage:{task_id}:{generated_index}"
                            generated_index += 1
                            if (generated_id not in reserved_ids
                                    and generated_id not in used_ids
                                    and generated_id not in issued_stable_ids):
                                stable_id = generated_id
                                break
                    point["review_point_id"] = stable_id
                    if isinstance(stable_id, str) and stable_id:
                        used_ids.add(stable_id)
                        issued_stable_ids.add(stable_id)
                point["review_video_id"] = video_id
                point["review_task_id"] = task_id
            stable_ids = [
                point.get("review_point_id")
                for module in candidate.get("modules", [])
                for point in module.get("points", [])
            ]
            if any(not isinstance(value, str) or not value for value in stable_ids) or len(stable_ids) != len(set(stable_ids)):
                raise ValueError("审核候选的稳定 point provenance 缺失或重复")
            return candidate

        def full_revalidate(candidate, cycle_work):
            self._checkpoint()
            candidate = annotate_review_provenance(candidate)
            candidate, source_reviews = ground_document(
                candidate, sources,
                self._grounding_callback(
                    llm, task_id, label="核对知识点来源",
                    stage="verify", progress=0.87,
                ),
                self._checkpoint,
                **self._input_budget_kwargs(ground_document, llm),
            )
            candidate, fresh_coverage = complete_document(
                candidate, sources,
                self._grounding_callback(
                    llm, task_id, label="逐段检查内容覆盖",
                    stage="verify", progress=0.88,
                ),
                self._checkpoint,
                **self._input_budget_kwargs(complete_document, llm),
            )
            candidate = annotate_review_provenance(candidate)
            extra_grounding = []
            if any(row.get("additions") for row in fresh_coverage if isinstance(row, dict)):
                candidate, extra_grounding = ground_document(
                    candidate, sources,
                    self._grounding_callback(
                        llm, task_id, label="核对覆盖新增点的来源",
                        stage="verify", progress=0.89,
                    ),
                    self._checkpoint,
                    **self._input_budget_kwargs(ground_document, llm),
                )
            candidate, fresh_coverage, reconciliation = reconcile_final_coverage(
                candidate, fresh_coverage, sources,
                self._grounding_callback(
                    llm, task_id, label="复核最终正文的内容覆盖",
                    stage="verify", progress=0.89,
                ),
                self._checkpoint,
                **self._input_budget_kwargs(reconcile_final_coverage, llm),
            )
            candidate = annotate_review_provenance(candidate)
            validate_final_coverage(candidate, fresh_coverage)
            candidate = checked_document(candidate)
            receipt = {
                "status": "complete",
                "complete": True,
                "document_hash": document_hash(candidate),
                "source_reviews": source_reviews + extra_grounding + reconciliation,
                "completeness": fresh_coverage,
                "reconciliation_reviews": reconciliation,
            }
            cycle_work = Path(cycle_work)
            cycle_work.mkdir(parents=True, exist_ok=True)
            (cycle_work / "pipeline-coverage-receipt.json").write_text(
                dumps(receipt), encoding="utf-8"
            )
            return candidate, receipt

        # The first workflow cycle may reuse this exact hash-bound full ledger;
        # every reviewer-authored change takes the full path above.
        verified_doc, initial_receipt = full_revalidate(doc, work / "initial-coverage")
        initial_hash = initial_receipt["document_hash"]
        (work / "source-grounded.json").write_text(
            dumps({"doc": verified_doc, "receipt": initial_receipt}), encoding="utf-8"
        )
        art["verification"] = {
            "status": "text_reviewed",
            "phase": "hash_bound_before_independent_review",
            "coverage": initial_receipt["source_reviews"],
            "completeness": initial_receipt["completeness"],
            "responses": getattr(llm, "response_diagnostics", []),
            "note": "输入分块已覆盖；不保证事实正确，画面状态单独记录",
        }
        doc = verified_doc
        illustrated_draft = copy.deepcopy(doc)
        (work / "draft.json").write_text(dumps(doc), encoding="utf-8")
        (work / "draft.html").write_text(render_standalone_html(doc), encoding="utf-8")
        art["illustrations"] = illustration_report
        # Retain the pre-review evidence even when research or adjudication fails.
        art["model_attempt_budget"] = self.task_budget.snapshot()
        (work / "evidence.json").write_text(dumps(art), encoding="utf-8")
        from app.agent.review.independent_review import document_hash
        from app.agent.review.review_workflow import review_workflow, semantic_content_hash
        self._emit(task_id, "review", 0.92, "启动独立审查：核验知识准确性与学习价值")

        cached_initial = (copy.deepcopy(doc), copy.deepcopy(initial_receipt))

        def revalidate(candidate, cycle_work):
            candidate = annotate_review_provenance(candidate)
            if document_hash(candidate) == initial_hash:
                # This complete ledger was just produced for the identical raw
                # candidate, so carrying its exact receipt is safe and avoids a
                # second no-op grounding pass before the first review.
                return copy.deepcopy(cached_initial)
            return full_revalidate(candidate, cycle_work)

        def author_call(content):
            self._emit(task_id, "revise", 0.95, "主 agent 结合视频证据逐条复核审核建议")
            return self._chat_with_heartbeat(
                llm, task_id, [{"role": "user", "content": content}],
                label="复核并修订知识点", stage="revise", progress=0.95,
                max_tokens=self.settings.llm_max_tokens,
                activity_context={
                    "actor": "author", "agent_id": "author-review",
                    "stage": "revise", "round": 1,
                    "cycle": review_cycle_number, "execution": "workflow",
                },
            )

        author_tools_round = 0

        def author_tools_call(messages, *, tools):
            nonlocal author_tools_round
            author_tools_round += 1
            self._emit(task_id, "revise", 0.95, "主 agent 使用证据工具复核审核建议")
            return chat_with_heartbeat(
                self._emit, llm, messages, task_id=task_id,
                label="主 agent 取证并复核", stage="revise", progress=0.95,
                method="chat_tools", tools=tools,
                max_tokens=self.settings.llm_max_tokens,
                activity_context={
                    "actor": "author", "agent_id": "author-review",
                    "stage": "revise", "cycle": review_cycle_number,
                    "execution": "workflow",
                },
            )

        def author_chat_text(messages, *, max_tokens):
            self._checkpoint()
            response = chat_with_heartbeat(
                self._emit, llm, messages, task_id=task_id,
                label="压缩工作记录", stage="revise", progress=0.95,
                method="chat_text", max_tokens=max_tokens,
                activity_context={
                    "actor": "author", "agent_id": "author-review",
                    "stage": "revise", "round": author_tools_round + 1,
                    "cycle": review_cycle_number, "execution": "workflow",
                },
            )
            self._checkpoint()
            return response

        author_tools_call.chat_text = author_chat_text
        author_tools_call.input_budget_tokens = self._model_input_budget(llm)
        author_tools_call.summary_input_budget_tokens = self._model_input_budget(
            llm, reserved_output=4096,
        )

        def review_progress(message):
            revising = self._last_synced_stage == "revise" or message.startswith("主agent复核")
            self._emit(task_id, "revise" if revising else "review",
                       0.95 if revising else 0.92, message)

        def review_tool_event(event):
            if isinstance(event, dict) and event.get("event") == "public_progress":
                stage = event.get("stage") if event.get("stage") in {"review", "revise"} else "review"
                self._publish_public_progress_segment(
                    task_id, event, stage=stage,
                    progress=0.95 if stage == "revise" else 0.92,
                )
                return
            labels = {"search_web": "搜索外部依据", "read_web": "读取网页正文",
                      "read_video_evidence": "核对视频证据", "view_video_frame": "查看视频画面"}
            origin = event.get("origin", "reviewer")
            revising = origin == "author"
            tool_name = event.get("tool", "tool")
            tool_status = event.get("status", "running")
            cycle = event.get("cycle")
            actor = "author" if revising else "reviewer"
            agent_id = (
                f"author-review-cycle-{cycle:02d}" if revising and type(cycle) is int
                else f"reviewer-cycle-{cycle:02d}" if type(cycle) is int
                else "author-review" if revising else "reviewer"
            )
            recorded_seq = self._record_activity(
                task_id, f"review-tool:{cycle}:{event.get('seq', 0)}",
                step="tool_call", label=labels.get(tool_name, str(tool_name)),
                status=tool_status,
                detail=event.get("result_summary") or (
                    "工具已完成" if tool_status == "done" else "正在核对证据"
                ),
                actor=actor, agent_id=agent_id,
                parent_task_id=self.parent_task_id,
                child_task_id=task_id if self.parent_task_id else None,
                stage="revise" if revising else "review", cycle=cycle,
                tool_name=str(tool_name), execution="isolated_context" if not revising else "workflow",
                elapsed_ms=event.get("elapsed_ms"),
                error=event.get("error"),
            )
            self.queue.publish(
                task_id, _write=True, status="running", stage="revise" if revising else "review",
                progress=0.95 if revising else 0.92,
                event="step", step={
                    "seq": recorded_seq if type(recorded_seq) is int else event["seq"],
                    "step": "review_tool", "origin": origin,
                    "label": ("作者取证：" if revising else "独立审查：") + labels.get(event["tool"], event["tool"]),
                    "status": event["status"], "detail": dumps(event.get("args", {})),
                    "elapsed_ms": event.get("elapsed_ms"),
                },
                **({"_activity_recorded_seq": recorded_seq}
                   if type(recorded_seq) is int else {}),
            )

        review_cycle_number = 0

        def reviewer_factory():
            nonlocal review_cycle_number
            review_cycle_number += 1
            cycle = review_cycle_number
            return self._new_llm(
                actor="reviewer", agent_id=f"reviewer-cycle-{cycle:02d}",
                stage="review", execution="isolated_context", cycle=cycle,
            )

        def review_cycle_activity(event):
            if not isinstance(event, dict):
                return
            cycle = event.get("cycle")
            status = event.get("status", "running")
            actor = event.get("actor", "reviewer")
            kind = event.get("kind", "reviewer_cycle")
            if kind == "context_compaction":
                seq = event.get("seq") if type(event.get("seq")) is int else 1
                agent_id = event.get("agent_id") or (
                    f"author-review-cycle-{cycle:02d}"
                    if actor == "author" and type(cycle) is int
                    else f"reviewer-cycle-{cycle:02d}" if type(cycle) is int
                    else "author-review" if actor == "author" else "reviewer"
                )
                stage = event.get("stage", "revise" if actor == "author" else "review")
                summary = event.get("result_summary") or ""
                arguments = event.get("arguments")
                recorded_seq = self._record_activity(
                    task_id, f"context-compaction:{cycle}:{actor}:{seq}",
                    step="compact", label="压缩工作记录", status="done",
                    detail=summary, result_summary=summary, arguments=arguments,
                    actor=actor, agent_id=agent_id,
                    parent_task_id=self.parent_task_id,
                    child_task_id=task_id if self.parent_task_id else None,
                    stage=stage, round=event.get("round"), cycle=cycle,
                    execution=event.get("execution", "workflow" if actor == "author" else "isolated_context"),
                )
                try:
                    self.queue.publish(
                        task_id, _write=True, status="running", stage=stage,
                        progress=0.95 if actor == "author" else 0.92,
                        event="step", step={
                            "seq": recorded_seq if type(recorded_seq) is int else seq,
                            "step": "compact", "label": "压缩工作记录",
                            "status": "done", "detail": summary,
                            "result_summary": summary, "arguments": arguments,
                            "actor": actor, "agent_id": agent_id, "stage": stage,
                            "round": event.get("round"), "cycle": cycle,
                            "execution": event.get("execution", "workflow" if actor == "author" else "isolated_context"),
                            **({"parent_task_id": self.parent_task_id,
                               "child_task_id": task_id} if self.parent_task_id else {}),
                        },
                        **({"_activity_recorded_seq": recorded_seq}
                           if type(recorded_seq) is int else {}),
                    )
                except Exception:
                    # Compaction observation cannot affect the model result or cancellation path.
                    pass
                return
            round_no = event.get("round")
            agent_id = event.get("agent_id") or (
                f"reviewer-cycle-{cycle:02d}" if type(cycle) is int else "reviewer"
            )
            detail = event.get("detail") or "独立审查正在核对当前候选"
            if status == "done":
                detail = event.get("result_summary") or "独立审查已完成本轮核验"
            elif status == "error":
                detail = f"独立审查失败（{event.get('error', 'Error')}）"
            elif status == "cancelled":
                detail = "独立审查已取消"
            if kind == "reviewer_round":
                key = f"reviewer-cycle:{cycle}:round:{round_no}"
                step, label = "reviewer_round", f"独立审查第 {round_no} 轮"
            elif kind == "author_review":
                key = f"author-review-cycle:{cycle}:attempt:{round_no}"
                step, label = "author_review", f"作者复核第 {round_no} 轮"
            else:
                key = f"reviewer-cycle:{cycle}:lifecycle"
                step, label = "reviewer_cycle", "独立审查"
            self._record_activity(
                task_id, key, step=step, label=label,
                status=status, detail=detail, actor=actor, agent_id=agent_id,
                parent_task_id=self.parent_task_id,
                child_task_id=task_id if self.parent_task_id else None,
                stage=event.get("stage", "revise" if actor == "author" else "review"),
                round=round_no, cycle=cycle,
                execution=event.get("execution", "workflow" if actor == "author" else "isolated_context"),
                error=event.get("error") if status == "error" else None,
            )

        doc, independent_report = review_workflow(
            doc, sources, art["visual_review"], reviewer_factory=reviewer_factory,
            author_call=author_call, author_tools_call=author_tools_call,
            work=work, checkpoint=self._checkpoint,
            revalidate=revalidate,
            progress=review_progress, on_tool_event=review_tool_event,
            on_activity_event=review_cycle_activity,
        )
        self._checkpoint()
        if independent_report.get("final_review_passed") is not True:
            raise ValueError("最终 fresh review 未通过，保留草稿且未提交知识文档")
        if (independent_report.get("final_hash") != document_hash(doc)
                or independent_report.get("final_coverage_hash") != document_hash(doc)
                or independent_report.get("final_semantic_hash") != semantic_content_hash(doc)):
            raise ValueError("最终审核或覆盖回执未绑定返回候选的完整哈希，未提交知识文档")
        independent_report["scope"] = "仅覆盖本视频候选及其当前来源；不表示审查了项目完整历史"
        (work / "reviewed-output.json").write_text(dumps(doc), encoding="utf-8")
        independent_report["output_artifact"] = "reviewed-output.json"
        independent_report["output_hash"] = document_hash(doc)
        self._emit(task_id, "revise", 0.95, "审核建议已复核，检查最终图文文档")
        doc = checked_document(doc)
        if document_hash(doc) != independent_report.get("final_hash"):
            raise ValueError("文档规范化改变了已审核候选，未提交知识文档")
        independent_report["revised_artifact"] = "revised.json"
        independent_report["revised_hash"] = document_hash(doc)
        art["independent_review"] = independent_report
        art["model_attempt_budget"] = self.task_budget.snapshot()
        (work / "revised.json").write_text(dumps(doc), encoding="utf-8")
        (work / "revised.html").write_text(render_standalone_html(doc), encoding="utf-8")
        (work / "evidence.json").write_text(dumps(art), encoding="utf-8")
        self._emit(task_id, "render", 0.96, "校验并融合进项目文档")
        with project_guard(self.project_id):
            self._checkpoint()
            # Any baseline mutation during model work invalidates this reviewed
            # candidate. Do not merge around a concurrent project update.
            latest = self._load_existing_doc()
            if latest != existing:
                raise ValueError("项目知识文档在审核期间发生变化；保留本轮草稿并要求重新整理")
            if expected_baseline_sha256 is not None:
                from app.services.recovery import document_sha256
                if document_sha256(latest) != expected_baseline_sha256:
                    raise ValueError("项目知识文档已偏离恢复 baseline；本次未覆盖")
            if options.get("integration") == "replace":
                merged, report = copy.deepcopy(doc), []
            else:
                assert_review_merge_unchanged(existing, latest, illustrated_draft, doc)
                merged, report = merge_reviewed_docs(latest, doc, illustrated_draft)
            merged = checked_document(merged)
            _assert_reviewed_module_integrity(doc, merged)
            candidate_points = [
                point for module in doc.get("modules", []) for point in module.get("points", [])
            ]
            formal_points = [
                point for module in merged.get("modules", []) for point in module.get("points", [])
            ]
            stable_to_formal = {}
            candidate_id_to_formal = {}
            for point in candidate_points:
                stable_id = point.get("review_point_id")
                if not isinstance(stable_id, str) or not stable_id:
                    raise ValueError("最终候选缺少 stable review point ID，未提交知识文档")
                semantic = _point_semantics(point)
                matches = [
                    formal for formal in formal_points
                    if formal.get("review_point_id") == stable_id
                    and _point_semantics(formal) == semantic
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "项目融合改变或歧义化了已审核知识点/图片说明，未提交知识文档："
                        + stable_id
                    )
                if stable_id in stable_to_formal:
                    raise ValueError("stable review point ID 重复，未提交知识文档：" + stable_id)
                stable_to_formal[stable_id] = matches[0]["id"]
                candidate_id_to_formal[point["id"]] = matches[0]["id"]

            final_id_map = independent_report.get("final_id_map")
            if not isinstance(final_id_map, dict):
                raise ValueError("fresh review 未提供 final_id_map，未提交知识文档")
            original_ids = final_id_map.get("original_to_final")
            cycle_ids = final_id_map.get("cycle_to_final")
            if not isinstance(original_ids, dict) or not isinstance(cycle_ids, dict):
                raise ValueError("final_id_map 结构无效，未提交知识文档")
            original_to_formal = {
                original_id: (candidate_id_to_formal.get(candidate_id) if candidate_id else None)
                for original_id, candidate_id in original_ids.items()
            }
            cycle_to_formal = {
                review_id: (candidate_id_to_formal.get(candidate_id) if candidate_id else None)
                for review_id, candidate_id in cycle_ids.items()
            }

            candidate_tables = _appendix_rows(doc)
            formal_tables = _appendix_rows(merged)
            appendix_candidate_ids = {
                value
                for mapping in (original_ids, cycle_ids)
                for value in mapping.values()
                if isinstance(value, str) and value.startswith("appendix:")
            }
            appendix_to_formal = {}
            for index, table in enumerate(candidate_tables):
                candidate_id = f"appendix:{index}"
                if candidate_id not in appendix_candidate_ids:
                    raise ValueError("final_id_map 无法证明最终附录来源，未提交知识文档：" + candidate_id)
                matches = [i for i, formal in enumerate(formal_tables) if formal == table]
                if len(matches) != 1:
                    raise ValueError("项目融合改变或歧义化了已审核附录，未提交知识文档：" + candidate_id)
                appendix_to_formal[candidate_id] = f"appendix:{matches[0]}"
            for original_id, candidate_id in original_ids.items():
                if isinstance(candidate_id, str) and candidate_id.startswith("appendix:"):
                    original_to_formal[original_id] = appendix_to_formal.get(candidate_id)
            for review_id, candidate_id in cycle_ids.items():
                if isinstance(candidate_id, str) and candidate_id.startswith("appendix:"):
                    cycle_to_formal[review_id] = appendix_to_formal.get(candidate_id)

            independent_report["point_id_map"] = stable_to_formal
            independent_report["formal_id_map"] = {
                "candidate_point_id_to_formal_id": candidate_id_to_formal,
                "stable_review_point_id_to_formal_id": stable_to_formal,
                "final_id_map_original_to_formal": original_to_formal,
                "final_id_map_cycle_to_formal": cycle_to_formal,
                "appendix_candidate_to_formal": appendix_to_formal,
            }
            independent_report["fusion_integrity"] = {
                "status": "passed",
                "current_video_points_exact": len(candidate_points),
                "reviewed_module_groups_exact": len(doc.get("modules", [])),
                "reviewed_module_order_preserved": True,
                "reviewed_appendices_exact": len(candidate_tables),
                "figures_and_captions_included_in_point_comparison": True,
                "project_history_in_review_scope": False,
            }
            (work / "independent_review.json").write_text(dumps(independent_report), encoding="utf-8")
            meta = merged.setdefault("meta", {}) or {}
            merged["meta"] = meta
            source = {"bvid": bvid, "cid": cid, "video_id": video_id}
            sources = meta.setdefault("source_videos", [])
            if source not in sources:
                sources.append(source)
            meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta["verification"] = art["verification"]
            meta["visual_review"] = art["visual_review"]
            meta["independent_review"] = independent_report
            meta["illustrations"] = illustration_report
            reviews = dict((latest.get("meta") or {}).get("reviews") or {})
            reviews[str(video_id)] = {
                "verification": art["verification"],
                "visual_review": art["visual_review"],
                "subtitle": art["subtitle"],
                "independent_review": independent_report,
                "illustrations": illustration_report,
            }
            meta["reviews"] = reviews
            art["merge_report"] = report
            art["model_attempt_budget"] = self.task_budget.snapshot()
            with self.queue.commit_guard(task_id):
                self._checkpoint()
                formal_database_json_bytes = json.dumps(
                    merged, ensure_ascii=False
                ).encode("utf-8")
                prepared_formal_receipt = bind_formal_output_receipts(
                    reviewed_candidate=doc,
                    formal_document=merged,
                    stable_review_point_id_to_formal_id=stable_to_formal,
                    source_appendix_id_to_formal_id=appendix_to_formal,
                    database_json_bytes=formal_database_json_bytes,
                    work=work,
                )
                formal_output_receipt = {
                    "status": "database_committed",
                    "manifest_artifact": "formal-output-manifest.json",
                    "manifest_status_at_database_commit": prepared_formal_receipt[
                        "status"
                    ],
                    "formal_json": prepared_formal_receipt["formal_object"]["database_json"],
                    "formal_html": prepared_formal_receipt["formal_object"]["html"],
                    "formal_object_semantic_sha256": prepared_formal_receipt[
                        "formal_object"
                    ]["semantic_sha256"],
                    "current_source_scope": prepared_formal_receipt[
                        "current_source_scope"
                    ],
                    "outside_current_source_scope": prepared_formal_receipt[
                        "outside_current_source_scope"
                    ],
                    "project_history_freshly_reviewed": False,
                    "database_commit": {
                        "status": "committed",
                        "sha256": hashlib.sha256(formal_database_json_bytes).hexdigest(),
                        "utf8_bytes": len(formal_database_json_bytes),
                    },
                }
                art["formal_output_receipt"] = formal_output_receipt
                with session_scope() as db:
                    write_generated_in_session(db, self.project_id, merged, doc)
                    db.flush()
                    persisted_doc_json = db.query(KnowledgeDoc.doc_json).filter_by(
                        project_id=self.project_id
                    ).scalar()
                    if not isinstance(persisted_doc_json, str):
                        raise ValueError("正式文档事务写入后无法读回doc_json，未提交")
                    persisted_document_json_bytes = persisted_doc_json.encode("utf-8")
                    verify_formal_output_receipts(
                        reviewed_candidate=doc,
                        formal_document=merged,
                        stable_review_point_id_to_formal_id=stable_to_formal,
                        source_appendix_id_to_formal_id=appendix_to_formal,
                        database_json_bytes=persisted_document_json_bytes,
                        work=work,
                    )
                    v = db.get(Video, video_id)
                    if v:
                        v.status = "done"
                        v.error = None
                        v.artifacts_json = dumps(art)
                        v.extracted_json = dumps(
                            [p for m in doc["modules"] for p in m["points"]]
                        )
                receipt_result = copy.deepcopy(formal_output_receipt)
                receipt_message = "正式文档及输出回执已提交"
                try:
                    committed_formal_receipt = mark_formal_output_committed(
                        work=work,
                        persisted_document_json_bytes=persisted_document_json_bytes,
                    )
                    receipt_result["status"] = "committed"
                    receipt_result["manifest_status"] = committed_formal_receipt["status"]
                except Exception as exc:
                    # The database transaction has committed. A sidecar manifest
                    # failure must not turn the saved formal document into a
                    # reported pipeline failure or invite a duplicate publish.
                    receipt_result["status"] = "receipt_pending"
                    receipt_result["manifest_status"] = "prepared"
                    receipt_result["error"] = f"{type(exc).__name__}: {exc}"
                    receipt_message = "正式文档已保存；输出回执状态为receipt_pending，最终化待补齐"
                    finalization_note = {
                        "status": "receipt_pending",
                        "error": receipt_result["error"],
                        "manifest_artifact": "formal-output-manifest.json",
                        "database_commit": receipt_result["database_commit"],
                    }
                    try:
                        finalization_path = work / "formal-output-finalization.json"
                        temporary_path = finalization_path.with_suffix(".json.tmp")
                        temporary_path.write_text(dumps(finalization_note), encoding="utf-8")
                        temporary_path.replace(finalization_path)
                    except Exception as note_exc:
                        receipt_result["finalization_note_error"] = (
                            f"{type(note_exc).__name__}: {note_exc}"
                        )
                self._close_activity_stage(task_id, "done")
                self.queue.update(
                    task_id,
                    status="done",
                    progress=1.0,
                    message=(
                        "完成（已完成独立审核与建议复核，核验范围见文档）；"
                        + receipt_message
                    ),
                    result={
                        "doc": merged,
                        "report": report,
                        "formal_output_receipt": receipt_result,
                    },
                )
        return merged

    def _download_progress(self, task_id, fraction):
        self._checkpoint()
        self._emit(
            task_id, "download", 0.05 + fraction * 0.35, f"下载音频 {fraction:.0%}"
        )
