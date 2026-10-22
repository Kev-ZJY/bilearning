"""进程内任务状态与 SSE 事件分发。

当前 API 直接启动线程并调用 register；没有独立的队列消费者。
状态、订阅和取消均属于单进程，没有 Redis 或多进程任务调度。
update 合并状态快照；publish 分发一次性的 step/token 事件。
"""
from __future__ import annotations

import threading
import time
import copy
import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Optional

# 订阅者回调：收到任务字典快照
Subscriber = Callable[[dict], Any]

# 一次性事件字段：只出现在事件帧里，**绝不写进 _status**。
# 若写进去，会被后续 update 覆盖（token 丢字）并被每次心跳重发（step 重复）。
# 靠「字段名在不在这个集合里」来区分状态与事件，比让每个调用点自己记得
# 哪些该落盘可靠得多。
_EVENT_ONLY_FIELDS = frozenset({"event", "step", "token", "segment_id", "phase"})

# 终态：SSE 收到就关流，消费方也据此判断「不用再等了」。
#
# 为什么 cancelled 与 done / failed 并列而不是复用 failed：
# 三者在**用户能采取的动作**上完全不同 —— done 可以读回复，failed 要看错误，
# cancelled 则什么都不该发生（回复不该落库、气泡不该出现）。若把中止混成
# failed，前端会弹一条红色报错，用户以为自己的操作失败了，而他刚刚做的
# 恰恰是主动取消。
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})
_RECOVERY_ID = re.compile(r"^[a-f0-9]{24}$")
_ACTIVITY_SCOPE_FIELDS = (
    "actor", "agent_id", "parent_task_id", "child_task_id", "stage", "round", "cycle", "execution",
)
_ACTIVITY_PERSISTENCE_ERROR = "活动记录暂时无法保存"


class TaskQueue:
    """进程内任务状态与事件订阅。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # task_id -> 任务状态字典（统一存放，作为权威状态源）
        self._status: dict[str, dict] = {}
        # 订阅者列表（任务变化时通知）
        self._subscribers: list[Subscriber] = []
        # 已被请求取消的 task_id。
        #
        # 刻意与 _status 分开存，而不是只在 _status 里放一个 cancelled 标志位：
        # **取消是一次性事件，不是状态**。执行方（agent / pipeline）要的是一个
        # 不会随 update 覆盖而丢失的查询入口 —— 心跳 update 会在取消之后继续
        # 跑很多次，若标志只存在于 merge 型快照里，写法稍有差池就会被覆盖。
        # 分开存之后，「是否被取消」这个问题有了唯一且不可被覆写的答案。
        self._cancelled: set[str] = set()
        self._activity_bindings: dict[str, tuple[Path, dict[str, Any]]] = {}
        self._activity_keys: dict[str, dict[str, list[tuple[dict[str, Any], int]]]] = {}
        self._recorded_activity_sequences: dict[str, set[int]] = {}
        self._legacy_activity_sequences: dict[str, dict[tuple[Any, ...], int]] = {}
        self._activity_seq_highwater: dict[str, int] = {}

    from contextlib import contextmanager

    @contextmanager
    def commit_guard(self, task_id):
        """取消与短事务提交线性化：先取消则禁止写入，先提交则取消不回滚。"""
        from app.core.cancel import TaskCancelled
        with self._lock:
            if task_id in self._cancelled:
                raise TaskCancelled()
            yield

    # ---- 取消 ----------------------------------------------------------
    def cancel(self, task_id: str) -> bool:
        """请求取消任务，返回该任务是否存在（不存在返回 False）。

        登记成功后立刻把状态推到 ``cancelled`` 并通知订阅者，这样 SSE 会
        立即关流、前端立刻停住——**不必等执行方真正停下来**。执行方
        （下一次检查点看到标志后抛 :class:`TaskCancelled`）再负责收尾。

        为什么取消要「两段」而不是一步到位
        ----------------------------------
        排队里正在跑的任务可能正卡在一个不可中断的阻塞调用上（本项目的
        主路径是非流式 LLM 请求，最长 900s）。一步到位要求那个调用立刻返回，
        那是做不到的。所以拆成「立即反馈」+「尽快收敛」：用户点下停止到
        界面响应是毫秒级，后台线程在下一个检查点退出，最坏情况等当前这次
        LLM 调用返回。

        **已经结束的任务不受影响**（状态保持原样，也不登记取消标志）。
        用户慢三秒按停止是很常见的：答案已经在库里并显示在界面上了，
        此时若把状态改成 cancelled，界面会跳成「已取消」，
        用户会以为刚才那段答案丢了 —— 停止一个已经完成的任务，
        正确的后果就是「什么都不发生」。

        返回值只表示「这个 task_id 存在」，不表示「状态被改了」。
        端点据此区分 404（真不存在）与 200（存在，可能是已完成的 no-op）。
        """
        with self._lock:
            task = self._status.get(task_id)
            if task is None:
                return False
            if task.get("status") in TERMINAL_STATUSES:
                return True
            activity = task.get("activity") or {}
            now = time.time() * 1000
            for row in activity.get("steps", []):
                if (
                    not isinstance(row, dict)
                    or row.get("step") != "child_task"
                    or row.get("status") != "running"
                    or row.get("execution") != "child_task"
                    or row.get("parent_task_id") != task_id
                    or not isinstance(row.get("child_task_id"), str)
                    or not row.get("child_task_id")
                ):
                    continue
                row["status"] = "cancelled"
                row["detail"] = "已请求取消，结束状态未确认"
                row["finished_at"] = now
                started_at = row.get("started_at")
                if isinstance(started_at, (int, float)) and not isinstance(started_at, bool):
                    row["elapsed_ms"] = max(
                        int(row.get("elapsed_ms") or 0),
                        max(0, int(now - started_at)),
                    )
            self._cancelled.add(task_id)
            task["status"] = "cancelled"
            task["message"] = "已取消"
            task["cancelled"] = True
            self._persist_activity_locked(task_id, task)
            snapshot = copy.deepcopy(task)
        self._notify(snapshot)
        return True

    def is_cancelled(self, task_id: str) -> bool:
        """该任务是否已被请求取消。执行方的检查点只该调这个。"""
        with self._lock:
            return task_id in self._cancelled

    def find_by_message(self, message_id: int) -> Optional[str]:
        """按 message_id 反查对应的 task_id，找不到返回 None。

        撤回功能需要它：前端只知道「我要撤回第 5 条消息」，而取消要的是
        task_id。两者之间唯一的联结就是 register 时写进 payload 的
        ``message_id``（见 ``app/api/messages.py``）。

        刻意做成 O(n) 全扫而不是建索引：任务数是用户一轮会话的量级（几十），
        而一个索引要维护、要考虑清理（任务本身没有 TTL）。**先量再优化**——
        现在全扫的成本是几十次字典取值，比一次数据库查询还便宜。
        """
        with self._lock:
            for tid, task in self._status.items():
                payload = task.get("payload") or {}
                if payload.get("message_id") == message_id:
                    return tid
        return None

    def _guard_status(self, task: dict, task_id: str, fields: dict) -> None:
        """取消之后不许被任何 update 改回非终态。

        为什么需要这个守卫：agent 的心跳线程在取消之后还会继续发
        ``update(status="running")``（:mod:`app.agent.runtime.heartbeat` 每 30 秒一次，
        而一次 LLM 调用可能跑几分钟）。没有这个守卫，界面会在用户点完停止
        之后又跳回「正在处理」，然后一直转到超时——**中止看起来完全没生效**，
        比不做中止更糟。
        """
        current_status = task.get("status")
        if task.get("cancelled") and fields.get("status") not in (None, "cancelled"):
            fields = dict(fields)
            fields["status"] = "cancelled"
            fields["message"] = "已取消"
        elif current_status in TERMINAL_STATUSES and fields.get("status") not in (None, current_status):
            fields = dict(fields)
            fields["status"] = current_status
        task.update(fields)
        task["task_id"] = task_id

    def register(self, task_id: str, payload: dict) -> str:
        """登记初始状态并通知订阅者；执行线程由调用方管理。"""
        task = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0.0,
            "message": "排队中",
            "payload": payload,
            "result": None,
            "error": None,
            "stage": None,
            "activity": {"steps": [], "segments": [], "started_at": time.time()*1000, "elapsed_ms": 0},
        }
        with self._lock:
            self._status[task_id] = task
            self._activity_bindings.pop(task_id, None)
            self._activity_keys[task_id] = {}
            self._recorded_activity_sequences[task_id] = set()
            self._legacy_activity_sequences[task_id] = {}
            self._activity_seq_highwater[task_id] = 0
        self._notify(dict(task))
        return task_id


    def update(self, task_id: str, **fields: Any) -> None:
        """更新任务状态并通知订阅者。

        常用字段：status, progress, message, result, error, stage。
        不存在的 task_id 静默忽略（Worker 可能在极端时序下更新已清理任务）。

        **这是「状态快照」，字段会累积并被后续 update 覆盖。**
        一次性事件（step / token）必须用 :meth:`publish`，理由见那里。
        """
        with self._lock:
            task = self._status.get(task_id)
            if task is None:
                return
            self._guard_status(task, task_id, fields)
            if fields.get("status") in TERMINAL_STATUSES:
                task["activity"]["elapsed_ms"] = time.time()*1000-task["activity"]["started_at"]
            self._persist_activity_locked(task_id, task)
            snapshot = copy.deepcopy(task)
        self._notify(snapshot)

    def publish(self, task_id: str, **event_fields: Any) -> None:
        """推一条**一次性事件**，不写入状态快照。

        为什么必须有这个方法（踩过的坑，勿合并回 update）
        ----------------------------------------------
        ``_status`` 是 merge 型快照字典：``update()`` 做的是 ``task.update(fields)``，
        同一个 key 会被下一次 update 覆盖。而 step / token 是**事件流**语义 ——
        「推过就是推过了」，不该被后续状态更新冲掉。

        早期我用 ``update(task_id, event="step", step=...)`` 发 step、
        ``update(task_id, event="token", token=...)`` 发 token，于是三件事同时坏掉：

        1. **token 被覆盖**：流式正文 4 个 chunk 里，前 3 个的 ``token`` 字段
           被第 4 个覆盖掉。只剩最后一个能推出去。端到端实测收到 2 条 token，
           而直连端点是 4 条、间隔 0.2-0.5s —— 差异全部出在这一层。
        2. **step 被反复重发**：``event``/``step`` 字段留在快照里，之后**每一次
           普通的 update**（进度推进、心跳）都会把上一条 step 再推一遍。实测抓到
           ``seq=11`` 被推了两次。
        3. **终态重复**：``_finish()`` 的 done 更新同样带着残留的 step 字段，
           于是「任务完成」这条事件里还挂着一个已完成的 step，前端会多渲染一行。

        修法是承认这两种数据模型不同，而不是硬塞进一个 dict：
        - :meth:`update`  = 状态（快照、累积、覆盖）
        - :meth:`publish` = 事件（一次性、不落盘、不被覆盖）

        ``publish`` 仍然从当前状态拼一份完整快照推给订阅者（前端依赖
        task_id / status / progress），只是**不把这些事件字段写回 _status**。

        **关于 status / progress / stage 这类状态字段**：它们既该落进
        _status 又该出现在事件帧里。如果调用方为了落盘先 ``update`` 再
        ``publish``，一次事件会触发**两次** ``_notify``，SSE 就会推两条帧 ——
        实测会看到同一条 step 重复出现（seq 相同）。所以这两个方法都接受
        ``write=True`` 来区分「这次要落盘」：

        - 只发一次性事件、不涉及状态 → 直接 publish（不落盘）
        - 事件 + 状态要一起落盘 → 用带 ``_write=True`` 的 publish（见实现）

        为避免和业务字段混淆，落盘开关是**下划线前缀的关键字** ``_write``，
        它不会混进事件负载。
        """
        write = bool(event_fields.pop("_write", False))
        recorded_seq = event_fields.pop("_activity_recorded_seq", None)
        with self._lock:
            task = self._status.get(task_id)
            if task is None:
                return
            task["task_id"] = task_id
            if write:
                # 状态字段落盘，但 event/step/token 之类的一次性字段不落盘。
                # 做法：先把非事件字段写进 _status，再基于它拼事件帧。
                self._guard_status(
                    task, task_id,
                    {k: v for k, v in event_fields.items()
                     if k not in _EVENT_ONLY_FIELDS},
                )
            if task.get("cancelled") or task.get("status") in TERMINAL_STATUSES:
                if write:
                    self._persist_activity_locked(task_id, task)
                snapshot = copy.deepcopy(task)
                snapshot.update({
                    k: v for k, v in event_fields.items()
                    if k not in _EVENT_ONLY_FIELDS
                })
                # Late tokens and steps must not add timeline entries or appear
                # as fresh work after cancellation/completion.
                event_only = any(k in event_fields for k in _EVENT_ONLY_FIELDS)
                if event_only:
                    return
                should_notify = True
            else:
                should_notify = False
            activity = task["activity"]
            kind = event_fields.get("event")
            if not should_notify and kind == "step" and event_fields.get("step"):
                from app.core.activity import sanitize_step

                step = dict(event_fields["step"])
                raw_seq = step.get("seq")
                recorded_sequences = self._recorded_activity_sequences.get(task_id, set())
                has_recorded_row = any(
                    isinstance(row, dict) and row.get("seq") == recorded_seq
                    for row in activity.get("steps", [])
                )
                if (
                    type(recorded_seq) is int and recorded_seq > 0
                    and recorded_seq in recorded_sequences and has_recorded_row
                ):
                    sequence = recorded_seq
                else:
                    sequence = self._legacy_activity_seq(
                        task_id, raw_seq, activity, step=step, stage=event_fields.get("stage"),
                        scope_fields=event_fields,
                    )
                previous = next((x for x in activity["steps"] if x["seq"] == sequence), None)
                safe_step = sanitize_step(
                    step, seq=sequence,
                    order=previous.get("order") if previous is not None else time.time_ns(),
                )
                if previous is not None:
                    previous.update(safe_step)
                else:
                    activity["steps"].append(safe_step)
                # Event frames are immutable snapshots: a later running->done
                # update must not mutate an already delivered earlier frame.
                event_fields["step"] = copy.deepcopy(safe_step)
            if not should_notify and kind in ("token", "segment"):
                sid = event_fields.get("segment_id", "answer")
                segment = next((x for x in activity["segments"] if x["id"] == sid), None)
                if segment is None:
                    segment = {"id": sid, "text": "", "phase": "pending", "order": time.time_ns()}
                    activity["segments"].append(segment)
                if kind == "token":
                    segment["text"] += event_fields.get("token", "")
                if event_fields.get("phase"):
                    segment["phase"] = event_fields["phase"]
            if not should_notify and (kind == "step" or write):
                self._persist_activity_locked(task_id, task)
            if not should_notify:
                snapshot = copy.deepcopy(task)
                # 事件字段只进这份临时快照，不进 _status。
                snapshot.update(event_fields)
        self._notify(snapshot)

    def _legacy_activity_seq(
        self, task_id: str, raw_seq: Any, activity: dict, *,
        step: dict[str, Any] | None = None, stage: Any = None,
        scope_fields: dict[str, Any] | None = None,
    ) -> int:
        """Map legacy per-component sequences into this task's unique sequence space."""
        mapping = self._legacy_activity_sequences.setdefault(task_id, {})
        step = step if isinstance(step, dict) else {}
        scope_fields = scope_fields if isinstance(scope_fields, dict) else {}
        correlation = None
        if type(raw_seq) is int and raw_seq > 0:
            label = step.get("label") if isinstance(step.get("label"), str) else ""
            name = step.get("step") if isinstance(step.get("step"), str) else ""
            correlation = (
                raw_seq, name, label,
                stage if isinstance(stage, str) else None,
                scope_fields.get("actor") if isinstance(scope_fields.get("actor"), str) else None,
                scope_fields.get("agent_id") if isinstance(scope_fields.get("agent_id"), str) else None,
                scope_fields.get("parent_task_id") if isinstance(scope_fields.get("parent_task_id"), str) else None,
                scope_fields.get("child_task_id") if isinstance(scope_fields.get("child_task_id"), str) else None,
                scope_fields.get("round") if type(scope_fields.get("round")) is int else None,
                scope_fields.get("cycle") if type(scope_fields.get("cycle")) is int else None,
            )
        if correlation is not None and correlation in mapping:
            return mapping[correlation]
        used = [
            row.get("seq") for row in activity.get("steps", [])
            if isinstance(row, dict) and type(row.get("seq")) is int
        ]
        highwater = max(self._activity_seq_highwater.get(task_id, 0), max(used, default=0))
        if type(raw_seq) is int and raw_seq > 0:
            existing = next((
                row for row in activity.get("steps", [])
                if isinstance(row, dict) and row.get("seq") == raw_seq
            ), None)
            if existing is not None and isinstance(step, dict):
                incoming_label = step.get("label")
                same_label = bool(incoming_label) and incoming_label == existing.get("label")
                same_stage = stage is None or existing.get("stage") == stage
                incoming_step = step.get("step")
                same_step = bool(incoming_step) and incoming_step == existing.get("step")
                if same_stage and (same_label or same_step):
                    if correlation is not None:
                        mapping[correlation] = raw_seq
                    self._activity_seq_highwater[task_id] = max(highwater, raw_seq)
                    return raw_seq
        if type(raw_seq) is int and raw_seq > highwater and raw_seq not in used:
            seq = raw_seq
        else:
            seq = highwater + 1
        self._activity_seq_highwater[task_id] = seq
        if correlation is not None:
            mapping[correlation] = seq
        return seq

    def _next_activity_seq(self, task_id: str, activity: dict) -> int:
        used = [
            row.get("seq") for row in activity.get("steps", [])
            if isinstance(row, dict) and type(row.get("seq")) is int
        ]
        seq = max(self._activity_seq_highwater.get(task_id, 0), max(used, default=0)) + 1
        self._activity_seq_highwater[task_id] = seq
        return seq

    def _persist_activity_locked(self, task_id: str, task: dict) -> None:
        binding = self._activity_bindings.get(task_id)
        if binding is None:
            return
        work, identity = binding
        task.pop("activity_persistence_error", None)
        try:
            from app.core.activity import write_activity_snapshot

            write_activity_snapshot(
                work,
                identity,
                {
                    "status": task.get("status"), "stage": task.get("stage"),
                    "message": task.get("message"), "error": task.get("error"),
                    "model_budget": task.get("model_budget"),
                },
                task.get("activity") or {},
            )
        except Exception:
            task["activity_persistence_error"] = _ACTIVITY_PERSISTENCE_ERROR

    def bind_activity(self, task_id: str, work: Path) -> None:
        """Bind a task to its own work directory and persist a sanitized snapshot."""
        with self._lock:
            task = self._status.get(task_id)
            if task is None:
                return None
            payload = task.get("payload") or {}
            project_id, video_id = payload.get("project_id"), payload.get("video_id")
            try:
                work = Path(work)
            except (TypeError, ValueError, OSError):
                task["activity_persistence_error"] = _ACTIVITY_PERSISTENCE_ERROR
                return None
            recovery_id = work.name
            try:
                expected_recovery_id = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:24]
            except (AttributeError, UnicodeEncodeError):
                task["activity_persistence_error"] = _ACTIVITY_PERSISTENCE_ERROR
                return None
            if (
                type(project_id) is not int or project_id <= 0
                or type(video_id) is not int or video_id <= 0
                or not isinstance(recovery_id, str) or _RECOVERY_ID.fullmatch(recovery_id) is None
                or not isinstance(task_id, str)
                or expected_recovery_id != recovery_id
            ):
                task["activity_persistence_error"] = _ACTIVITY_PERSISTENCE_ERROR
                return None
            identity = {
                "task_id": task_id, "project_id": project_id,
                "video_id": video_id, "recovery_id": recovery_id,
            }
            previous = self._activity_bindings.get(task_id)
            if previous is not None and previous != (work, identity):
                task["activity_persistence_error"] = _ACTIVITY_PERSISTENCE_ERROR
                return None
            self._activity_bindings[task_id] = (work, identity)
            self._persist_activity_locked(task_id, task)
            return None

    def record_activity(self, task_id: str, key: str, **fields: Any) -> int | None:
        """Record or update one scoped AgentStep; return its task-global sequence."""
        if not isinstance(key, str) or not key or len(key) > 256:
            return None
        try:
            from app.core.activity import sanitize_activity_value, sanitize_step

            scoped = {
                field: sanitize_activity_value(fields[field], field_name=field)
                for field in _ACTIVITY_SCOPE_FIELDS if fields.get(field) is not None
            }
            normalized = dict(fields)
            # The key itself is only an in-memory correlation handle and is never serialized.
            for field in fields:
                if field in {"arguments"}:
                    normalized[field] = sanitize_activity_value(fields[field], field_name=field)
                elif field in {"detail", "result_summary", "error", "model", "tool_name", "stage",
                               "agent_id", "parent_task_id", "child_task_id", "step", "label"}:
                    normalized[field] = sanitize_activity_value(fields[field], field_name=field)
            with self._lock:
                task = self._status.get(task_id)
                if task is None or task.get("cancelled") or task.get("status") in TERMINAL_STATUSES:
                    return None
                activity = task.setdefault("activity", {
                    "steps": [], "segments": [], "started_at": time.time() * 1000, "elapsed_ms": 0,
                })
                keyed = self._activity_keys.setdefault(task_id, {})
                candidates = [
                    (saved_scope, saved_seq)
                    for saved_scope, saved_seq in keyed.get(key, [])
                    if saved_scope == scoped
                ]
                selected = candidates[0] if len(candidates) == 1 else None
                existing = None
                if selected is not None:
                    existing = next((row for row in activity["steps"] if row.get("seq") == selected[1]), None)
                if existing is None:
                    seq = self._next_activity_seq(task_id, activity)
                    now = time.time() * 1000
                    normalized.setdefault("started_at", now)
                    normalized.setdefault("order", time.time_ns())
                    step = sanitize_step(normalized, seq=seq)
                    if step["status"] in {"done", "failed", "cancelled"}:
                        step.setdefault("finished_at", now)
                    activity["steps"].append(step)
                    keyed.setdefault(key, []).append((scoped, seq))
                    self._recorded_activity_sequences.setdefault(task_id, set()).add(seq)
                else:
                    seq = selected[1]
                    now = time.time() * 1000
                    normalized["order"] = existing.get("order", time.time_ns())
                    if "started_at" not in existing:
                        normalized.setdefault("started_at", now)
                    step = sanitize_step(normalized, seq=seq, order=existing.get("order"))
                    existing.update(step)
                    if step["status"] in {"done", "failed", "cancelled"}:
                        existing.setdefault("finished_at", now)
                activity["elapsed_ms"] = max(
                    int(activity.get("elapsed_ms") or 0),
                    max(0, int(time.time() * 1000 - activity.get("started_at", time.time() * 1000))),
                )
                self._persist_activity_locked(task_id, task)
                snapshot = copy.deepcopy(task)
        except Exception:
            # Activity is observational metadata and cannot fail the task's work.
            return None
        self._notify(snapshot)
        return seq

    def get_status(self, task_id: str) -> Optional[dict]:
        """查任务状态，不存在返回 None。返回快照避免外部修改内部状态。"""
        with self._lock:
            task = self._status.get(task_id)
            return copy.deepcopy(task) if task is not None else None

    def list_all(self) -> list[dict]:
        """列出所有任务（按提交顺序）。返回快照列表。"""
        with self._lock:
            return [dict(t) for t in self._status.values()]

    def subscribe(self, fn: Subscriber) -> None:
        """注册回调，任务变化时（submit/update）通知。fn 收到任务字典快照。"""
        with self._lock:
            self._subscribers.append(fn)

    def unsubscribe(self, fn: Subscriber) -> None:
        """注销回调。

        事件流（events.py）在客户端断开时用它清理订阅，避免回调泄漏累积。
        必须用公开的这个方法而不是直接碰 ``_subscribers``：
        (1) ``list.remove`` 用 ``==`` 比较对象，两个内容相同但身份不同的闭包会被
            误删；这里用 ``id()`` 精确匹配回调身份，杜绝误删。
        (2) 加锁保护，与并发的 submit/update 安全协作。
        """
        with self._lock:
            for i, sub in enumerate(self._subscribers):
                if id(sub) == id(fn):
                    self._subscribers.pop(i)
                    break

    def _notify(self, task: dict) -> None:
        """通知所有订阅者。单个订阅者抛异常不能影响其他订阅者。"""
        # 复制一份，避免回调里增删订阅者导致迭代异常
        with self._lock:
            subs = list(self._subscribers)
        for fn in subs:
            try:
                fn(task)
            except Exception:
                # 订阅者异常隔离：记录但不向上抛
                pass


# 模块级单例：编排层与 Worker 共享同一个队列实例
_queue_instance: Optional[TaskQueue] = None
_queue_lock = threading.Lock()


def get_queue() -> TaskQueue:
    """返回全局 TaskQueue 实例（首次调用时创建，线程安全）。"""
    global _queue_instance
    if _queue_instance is None:
        with _queue_lock:
            if _queue_instance is None:
                _queue_instance = TaskQueue()
    return _queue_instance
