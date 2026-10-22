"""持久会话摘要：用户轮次分组、来源校验与版本化提交。

原始 Message 永不删除。这里仅为模型准备“最近十个历史用户轮次原文 + 更早摘要”，
摘要只有在模型结果有效且调用方已完成最终上下文预算检查后才能 CAS 提交。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sqlalchemy import inspect as sa_inspect, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agent.runtime.context import ContextBudgetError, estimate_llm_input_tokens
from app.core.database import session_scope
from app.models.entities import ConversationSummary, Message

RECENT_USER_TURNS = 10
SUMMARY_FIELDS = (
    "goals", "constraints", "conclusions", "decisions", "todos",
    "open_questions", "uncertainties",
)


class ConversationAnchorMissing(RuntimeError):
    """固定的当前用户消息不存在、已撤回，或不属于当前项目。"""


class InvalidConversationSummary(ValueError):
    """摘要器返回的结构无效或引用了不在本次原文中的来源。"""


class ConversationSummaryConflict(RuntimeError):
    """项目摘要在准备与提交之间被另一请求更新。"""


class ConversationSnapshotConflict(ConversationSummaryConflict):
    """准备上下文后，锚点或其固定历史快照发生变化。"""


@dataclass(frozen=True)
class ConversationMessage:
    id: int
    role: str
    content: str
    task_refs_json: Optional[str] = None


@dataclass(frozen=True)
class ConversationTurn:
    user_message_id: int
    messages: tuple[ConversationMessage, ...]
    missing_explicit_relations: tuple[int, ...] = ()


@dataclass(frozen=True)
class StoredConversationSummary:
    version: int
    summary_json: str
    source_hash: str
    source_message_ids_json: str
    source_message_hashes_json: str
    covered_until_user_message_id: Optional[int]
    snapshot_anchor_message_id: int


@dataclass(frozen=True)
class ConversationMemorySnapshot:
    project_id: int
    anchor_message_id: int
    anchor_hash: str
    messages: tuple[ConversationMessage, ...]
    snapshot_hash: str
    stored_summary: Optional[StoredConversationSummary]


@dataclass(frozen=True)
class PreparedConversationMemory:
    project_id: int
    anchor_message_id: int
    anchor_hash: str
    snapshot_message_ids: tuple[int, ...]
    snapshot_hash: str
    expected_version: Optional[int]
    history_turn_count: int
    recent_turns: tuple[ConversationTurn, ...]
    history: tuple[dict[str, Any], ...]
    summary: Optional[dict[str, Any]]
    summary_text: str
    source_message_ids: tuple[int, ...]
    source_hash: str
    source_message_hashes: dict[str, str]
    covered_until_user_message_id: Optional[int]
    should_commit: bool


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _message_digest(message: ConversationMessage) -> str:
    # Include the stable database ID: an import that allocates new IDs cannot
    # accidentally keep an old cursor, even when the message text is identical.
    return _digest({"id": message.id, "role": message.role,
                    "content": message.content,
                    "task_refs_json": message.task_refs_json})


def _snapshot_identity(messages: tuple[ConversationMessage, ...]
                       ) -> tuple[tuple[int, ...], str]:
    ordered = tuple(sorted(messages, key=lambda row: row.id))
    ids = tuple(message.id for message in ordered)
    digest = _digest([(message.id, _message_digest(message)) for message in ordered])
    return ids, digest


def _reserve_sqlite_write_boundary(db: Session) -> None:
    """Acquire SQLite's physical writer reservation before snapshot reads.

    SQLAlchemy's logical Session transaction does not necessarily start a
    sqlite3 transaction for SELECT under the driver's legacy transaction mode.
    A fresh connection therefore needs BEGIN IMMEDIATE. If the caller already
    has a physical transaction, a no-op UPDATE upgrades it before any further
    validation; a stale WAL read snapshot then fails with SQLITE_BUSY_SNAPSHOT
    instead of validating stale rows and committing against them.

    Other dialects keep their existing transaction/isolation behavior. This
    helper makes no cross-dialect claim about phantom-write protection.
    """
    connection = db.connection()
    if connection.dialect.name != "sqlite":
        return
    driver_connection = connection.connection.driver_connection
    if driver_connection.in_transaction:
        connection.exec_driver_sql(
            "UPDATE conversation_summaries SET version = version WHERE 0"
        )
    else:
        connection.exec_driver_sql("BEGIN IMMEDIATE")


def _reject_pending_snapshot_writes(db: Session,
                                    prepared: PreparedConversationMemory) -> None:
    """Do not let ORM autoflush change the snapshot after validation starts."""
    pending = set(db.new) | set(db.dirty) | set(db.deleted)
    for obj in pending:
        if isinstance(obj, Message):
            state = sa_inspect(obj)
            project_history = state.attrs.project_id.history
            project_ids = {state.dict.get("project_id")}
            project_ids.update(project_history.deleted)
            project_ids.update(project_history.added)
            id_history = state.attrs.id.history
            message_ids = {state.dict.get("id")}
            message_ids.update(id_history.deleted)
            message_ids.update(id_history.added)
            affects_snapshot = any(
                message_id == prepared.anchor_message_id
                or (isinstance(message_id, int)
                    and message_id < prepared.anchor_message_id)
                for message_id in message_ids
            )
            if not affects_snapshot:
                continue
            if (prepared.project_id not in project_ids
                    and None not in project_ids):
                continue
            # An auto-ID row is allocated after the existing maximum, so it
            # cannot enter this already-fixed ID boundary before flush.
            raise ConversationSnapshotConflict(
                "调用方 Session 含未提交的会话消息变更，请先提交或回滚"
            )
        elif isinstance(obj, ConversationSummary):
            state = sa_inspect(obj)
            project_ids = {state.dict.get("project_id")}
            project_ids.update(state.attrs.project_id.history.deleted)
            project_ids.update(state.attrs.project_id.history.added)
            if prepared.project_id in project_ids or None in project_ids:
                raise ConversationSummaryConflict(
                    "调用方 Session 含未提交的摘要变更，请先提交或回滚"
                )


def _decode_reply_to(raw: Optional[str]) -> tuple[Optional[int], bool]:
    if not raw:
        return None, False
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None, False
    if not isinstance(payload, dict) or "reply_to_message_id" not in payload:
        return None, False
    value = payload.get("reply_to_message_id")
    if isinstance(value, bool):
        return None, True
    try:
        return int(value), True
    except (TypeError, ValueError):
        return None, True


def group_conversation_turns(messages: tuple[ConversationMessage, ...] | list[ConversationMessage]
                             ) -> tuple[ConversationTurn, ...]:
    """按 user 消息 ID 分轮；显式 reply_to 优先，旧消息按相邻顺序归组。"""
    ordered = sorted(messages, key=lambda row: row.id)
    users = {row.id: [row] for row in ordered if row.role == "user"}
    last_user_id: Optional[int] = None
    missing: dict[int, list[int]] = {user_id: [] for user_id in users}

    for row in ordered:
        if row.role == "user":
            last_user_id = row.id
            continue
        if row.role not in ("assistant", "tool"):
            continue
        target, explicit = _decode_reply_to(row.task_refs_json)
        if explicit:
            # An explicit but broken link is not reassigned to a different user:
            # doing so could put a late reply into the wrong request.
            if target in users:
                users[target].append(row)
            continue
        if last_user_id is not None:
            users[last_user_id].append(row)
            missing[last_user_id].append(row.id)

    turns = []
    for user_id, grouped in users.items():
        grouped.sort(key=lambda row: row.id)
        turns.append(ConversationTurn(
            user_message_id=user_id,
            messages=tuple(grouped),
            missing_explicit_relations=tuple(missing[user_id]),
        ))
    return tuple(sorted(turns, key=lambda turn: turn.user_message_id))


def _turn_payload(turn: ConversationTurn) -> dict[str, Any]:
    return {
        "user_message_id": turn.user_message_id,
        "messages": [
            {"message_id": message.id, "role": message.role,
             "content": message.content,
             "association": "order_fallback" if message.id in turn.missing_explicit_relations
                            else ("user_anchor" if message.role == "user" else "explicit_or_legacy")}
            for message in turn.messages
        ],
    }


def _turn_source(turns: tuple[ConversationTurn, ...] | list[ConversationTurn]
                 ) -> tuple[list[int], dict[str, str], str]:
    messages = sorted((m for turn in turns for m in turn.messages), key=lambda row: row.id)
    ids = [message.id for message in messages]
    hashes = {str(message.id): _message_digest(message) for message in messages}
    source_hash = _digest([(message.id, hashes[str(message.id)]) for message in messages])
    return ids, hashes, source_hash


def _empty_summary() -> dict[str, Any]:
    return {field: [] for field in SUMMARY_FIELDS}


def _validated_summary(raw: Any, allowed_source_ids: set[int],
                       canonical_source_ids: list[int]) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InvalidConversationSummary("摘要不是有效 JSON") from exc
    if not isinstance(raw, dict):
        raise InvalidConversationSummary("摘要必须是 JSON 对象")

    result: dict[str, Any] = {}
    for field in SUMMARY_FIELDS:
        value = raw.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise InvalidConversationSummary(f"摘要字段 {field} 必须是字符串数组")
        result[field] = [item.strip() for item in value if item.strip()]

    # 摘要模型不需要生成来源 ID。若它额外输出了来源，则严格校验；最终来源列表
    # 由程序从实际纳入的原消息计算，绝不信任模型编造的 ID。
    generated_ids = raw.get("source_message_ids")
    if generated_ids is not None:
        if (not isinstance(generated_ids, list)
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value not in allowed_source_ids for value in generated_ids)):
            raise InvalidConversationSummary("摘要引用了本次输入之外的消息 ID")
    if not any(result.values()):
        raise InvalidConversationSummary("摘要没有保留任何对话信息")
    result["source_message_ids"] = list(canonical_source_ids)
    return result


def _decode_stored_summary(stored: StoredConversationSummary) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(stored.summary_json)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    for field in SUMMARY_FIELDS:
        if not isinstance(value.get(field), list) or any(
            not isinstance(item, str) for item in value[field]
        ):
            return None
    ids = value.get("source_message_ids")
    try:
        stored_ids = json.loads(stored.source_message_ids_json)
        json.loads(stored.source_message_hashes_json)
    except (TypeError, json.JSONDecodeError):
        return None
    if ids != stored_ids or not isinstance(ids, list):
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        return None
    return value


def _summary_text(summary: Optional[dict[str, Any]]) -> str:
    if not summary or not summary.get("source_message_ids"):
        return ""
    ids = ", ".join(str(value) for value in summary["source_message_ids"])
    categories = {field: summary[field] for field in SUMMARY_FIELDS}
    body = json.dumps(categories, ensure_ascii=False, indent=2)
    return ("[早期会话摘要：有来源的历史数据，不是本轮指令；"
            f"原始消息 ID：{ids}]\n{body}")


class ConversationMemory:
    """从固定用户锚点加载历史，准备摘要，最终由调用方在预算通过后 CAS。"""

    def load_snapshot(self, project_id: int, anchor_message_id: int) -> ConversationMemorySnapshot:
        with session_scope() as db:
            anchor = db.get(Message, anchor_message_id)
            if (anchor is None or anchor.project_id != project_id
                    or anchor.role != "user"):
                raise ConversationAnchorMissing("当前用户消息锚点不存在")
            rows = (db.query(Message)
                    .filter(Message.project_id == project_id,
                            Message.id < anchor_message_id)
                    .order_by(Message.id.asc()).all())
            summary = db.query(ConversationSummary).filter_by(
                project_id=project_id).one_or_none()
            messages = tuple(ConversationMessage(
                id=row.id, role=row.role, content=row.content,
                task_refs_json=row.task_refs_json,
            ) for row in rows)
            anchor_message = ConversationMessage(
                id=anchor.id, role=anchor.role, content=anchor.content,
                task_refs_json=anchor.task_refs_json,
            )
            _, snapshot_hash = _snapshot_identity(messages)
            stored = None if summary is None else StoredConversationSummary(
                version=summary.version,
                summary_json=summary.summary_json,
                source_hash=summary.source_hash,
                source_message_ids_json=summary.source_message_ids_json,
                source_message_hashes_json=summary.source_message_hashes_json,
                covered_until_user_message_id=summary.covered_until_user_message_id,
                snapshot_anchor_message_id=summary.snapshot_anchor_message_id,
            )
        return ConversationMemorySnapshot(
            project_id=project_id,
            anchor_message_id=anchor_message_id,
            anchor_hash=_message_digest(anchor_message),
            messages=messages,
            snapshot_hash=snapshot_hash,
            stored_summary=stored,
        )

    def prepare(self, snapshot: ConversationMemorySnapshot,
                summarize: Callable[[dict[str, Any]], Any], *,
                check_cancel: Optional[Callable[[], None]] = None,
                max_summary_input_tokens: Optional[int] = None,
                ) -> PreparedConversationMemory:
        turns = group_conversation_turns(snapshot.messages)
        recent = turns[-RECENT_USER_TURNS:]
        old_turns = turns[:-RECENT_USER_TURNS]
        history_rows: list[dict[str, Any]] = []
        for turn in recent:
            for message in turn.messages:
                if message.id in turn.missing_explicit_relations:
                    history_rows.append({
                        "role": "assistant",
                        "content": ("[历史关联注记：这条旧消息缺少显式用户消息关联，"
                                    "按原始顺序归入上一条用户消息。]"),
                    })
                history_rows.append({"role": message.role, "content": message.content})
        recent_history = tuple(history_rows)

        stored = snapshot.stored_summary
        expected_version = stored.version if stored is not None else None
        stored_content = _decode_stored_summary(stored) if stored is not None else None
        # A later request may already have committed a summary whose source
        # snapshot includes messages beyond this request's fixed ID boundary.
        # Rebuild ephemerally from this request's own rows, but never roll back
        # the newer project's cursor or inject data from after this anchor.
        snapshot_is_older = bool(
            stored is not None
            and stored.snapshot_anchor_message_id > snapshot.anchor_message_id
        )
        valid_covered_turns: tuple[ConversationTurn, ...] = ()
        valid_stored = False
        if stored is not None and stored_content is not None and not snapshot_is_older:
            cursor = stored.covered_until_user_message_id
            candidate_covered = tuple(
                turn for turn in turns
                if cursor is not None and turn.user_message_id <= cursor
            )
            current_ids, current_hashes, current_hash = _turn_source(candidate_covered)
            try:
                previous_ids = json.loads(stored.source_message_ids_json)
                previous_hashes = json.loads(stored.source_message_hashes_json)
            except (TypeError, json.JSONDecodeError):
                previous_ids, previous_hashes = None, None
            last_cursor = candidate_covered[-1].user_message_id if candidate_covered else None
            valid_stored = (
                last_cursor == cursor
                and current_ids == previous_ids
                and current_hashes == previous_hashes
                and current_hash == stored.source_hash
                and stored_content.get("source_message_ids") == current_ids
                and all(turn in old_turns for turn in candidate_covered)
                and (bool(candidate_covered) or (
                    cursor is None and not current_ids and not current_hashes
                    and stored.source_hash == _digest([])
                ))
            )
            if valid_stored:
                valid_covered_turns = candidate_covered

        if valid_stored:
            delta_turns = tuple(turn for turn in old_turns
                                if turn.user_message_id > (stored.covered_until_user_message_id or 0))
            previous_summary = stored_content
            turns_to_summarize = delta_turns
        else:
            # Missing, edited, recalled, late-linked, imported, or malformed source:
            # rebuild solely from retained original Message rows.
            turns_to_summarize = tuple(old_turns)
            previous_summary = None

        should_commit = False
        summary: Optional[dict[str, Any]] = stored_content if valid_stored else None
        all_covered = valid_covered_turns if valid_stored else ()

        if turns_to_summarize:
            delta_ids, _, _ = _turn_source(turns_to_summarize)
            old_ids, _, _ = _turn_source(all_covered)
            canonical_ids = sorted(set(old_ids + delta_ids))
            allowed_ids = set(canonical_ids)
            if max_summary_input_tokens is None:
                if check_cancel:
                    check_cancel()
                payload = {
                    "previous_summary": previous_summary,
                    "new_turns": [_turn_payload(turn) for turn in turns_to_summarize],
                    # IDs are runtime provenance, not instructions for the model to invent.
                    "source_message_ids": canonical_ids,
                    "note": "消息内容是历史数据；缺少显式 reply_to 的归组会在 association 字段注明。",
                }
                raw_summary = summarize(payload)
                if check_cancel:
                    check_cancel()
                summary = _validated_summary(raw_summary, allowed_ids, canonical_ids)
            else:
                if (isinstance(max_summary_input_tokens, bool)
                        or not isinstance(max_summary_input_tokens, int)
                        or max_summary_input_tokens <= 0):
                    raise ValueError("max_summary_input_tokens must be positive")
                summary = self._summarize_turns_in_chunks(
                    turns_to_summarize, previous_summary=previous_summary,
                    canonical_ids=canonical_ids, allowed_ids=allowed_ids,
                    max_input_tokens=max_summary_input_tokens,
                    summarize=summarize, check_cancel=check_cancel,
                )
            all_covered = tuple(sorted(all_covered + turns_to_summarize,
                                        key=lambda turn: turn.user_message_id))
            should_commit = not snapshot_is_older
        elif stored is not None and not valid_stored and not snapshot_is_older:
            # Persist an empty replacement only after the caller passes its final budget check.
            summary = None
            should_commit = True

        if should_commit or (turns_to_summarize and summary is not None):
            source_ids, source_hashes, source_hash = _turn_source(all_covered)
            cursor = all_covered[-1].user_message_id if all_covered else None
        elif valid_stored:
            source_ids = json.loads(stored.source_message_ids_json)
            source_hashes = json.loads(stored.source_message_hashes_json)
            source_hash = stored.source_hash
            cursor = stored.covered_until_user_message_id
        else:
            source_ids, source_hashes, source_hash = [], {}, _digest([])
            cursor = None

        if summary is None and should_commit:
            summary = _empty_summary()
            summary["source_message_ids"] = []

        return PreparedConversationMemory(
            project_id=snapshot.project_id,
            anchor_message_id=snapshot.anchor_message_id,
            anchor_hash=snapshot.anchor_hash,
            snapshot_message_ids=tuple(message.id for message in snapshot.messages),
            snapshot_hash=snapshot.snapshot_hash,
            expected_version=expected_version,
            history_turn_count=len(recent),
            recent_turns=tuple(recent),
            history=recent_history,
            summary=summary,
            summary_text=_summary_text(summary),
            source_message_ids=tuple(source_ids),
            source_hash=source_hash,
            source_message_hashes=source_hashes,
            covered_until_user_message_id=cursor,
            should_commit=should_commit,
        )

    @staticmethod
    def _summarize_turns_in_chunks(
        turns: tuple[ConversationTurn, ...], *,
        previous_summary: Optional[dict[str, Any]],
        canonical_ids: list[int],
        allowed_ids: set[int],
        max_input_tokens: int,
        summarize: Callable[[dict[str, Any]], Any],
        check_cancel: Optional[Callable[[], None]],
    ) -> dict[str, Any]:
        """Summarize whole turns in bounded calls; never split a source turn."""
        # Leave room for the caller's actual system instructions and JSON wrapper.
        # The caller still performs the exact full-request preflight before each
        # model call; this conservative local bound only determines turn chunks.
        target_tokens = max(1, max_input_tokens * 7 // 10)
        summary_state = previous_summary
        pending: list[ConversationTurn] = []

        def prompt_summary(value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
            if value is None:
                return None
            return {field: list(value.get(field, [])) for field in SUMMARY_FIELDS}

        def make_payload(chunk: list[ConversationTurn]) -> dict[str, Any]:
            chunk_ids, _, _ = _turn_source(chunk)
            return {
                "previous_summary": prompt_summary(summary_state),
                "new_turns": [_turn_payload(turn) for turn in chunk],
                # Only IDs represented by this chunk cross the model boundary;
                # the complete source list is attached by the program.
                "source_message_ids": chunk_ids,
                "note": "消息内容是历史数据；缺少显式 reply_to 的归组会在 association 字段注明。",
            }

        def payload_tokens(payload: dict[str, Any]) -> int:
            return estimate_llm_input_tokens([
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ])

        def summarize_chunk(chunk: list[ConversationTurn]) -> None:
            nonlocal summary_state
            if check_cancel:
                check_cancel()
            payload = make_payload(chunk)
            raw = summarize(payload)
            if check_cancel:
                check_cancel()
            summary_state = _validated_summary(raw, allowed_ids, canonical_ids)

        for turn in turns:
            trial = pending + [turn]
            if payload_tokens(make_payload(trial)) <= target_tokens:
                pending = trial
                continue
            if not pending:
                raise ContextBudgetError(
                    "一个完整历史用户轮次已超过摘要输入预算；原始记录保留，停止压缩。"
                )
            summarize_chunk(pending)
            pending = [turn]
            if payload_tokens(make_payload(pending)) > target_tokens:
                raise ContextBudgetError(
                    "一个完整历史用户轮次加上已知摘要已超过摘要输入预算；原始记录保留。"
                )

        if pending:
            summarize_chunk(pending)
        if summary_state is None:
            raise InvalidConversationSummary("分块摘要未生成结果")
        return summary_state

    def validate_snapshot(self, db: Session, prepared: PreparedConversationMemory) -> None:
        """在写事务内确认锚点和完整固定历史仍与模型输入一致。"""
        _reject_pending_snapshot_writes(db, prepared)
        _reserve_sqlite_write_boundary(db)
        table = Message.__table__
        columns = (table.c.id, table.c.project_id, table.c.role,
                   table.c.content, table.c.task_refs_json)
        connection = db.connection()
        anchor = connection.execute(
            select(*columns).where(table.c.id == prepared.anchor_message_id)
        ).mappings().one_or_none()
        if (anchor is None or anchor["project_id"] != prepared.project_id
                or anchor["role"] != "user"):
            raise ConversationAnchorMissing("当前用户消息锚点已撤回或失效")
        anchor_snapshot = ConversationMessage(
            id=anchor["id"], role=anchor["role"], content=anchor["content"],
            task_refs_json=anchor["task_refs_json"],
        )
        if _message_digest(anchor_snapshot) != prepared.anchor_hash:
            raise ConversationSnapshotConflict("当前用户消息在生成期间发生变化")

        rows = connection.execute(
            select(*columns)
            .where(table.c.project_id == prepared.project_id,
                   table.c.id < prepared.anchor_message_id)
            .order_by(table.c.id.asc())
        ).mappings().all()
        current = tuple(ConversationMessage(
            id=row["id"], role=row["role"], content=row["content"],
            task_refs_json=row["task_refs_json"],
        ) for row in rows)
        current_ids, current_hash = _snapshot_identity(current)
        if (current_ids != prepared.snapshot_message_ids
                or current_hash != prepared.snapshot_hash):
            raise ConversationSnapshotConflict(
                "历史消息在生成期间发生变化，请基于最新历史重试"
            )

    def commit(self, prepared: PreparedConversationMemory, *,
               db: Optional[Session] = None, validate_snapshot: bool = True) -> bool:
        """在调用方事务内验证快照并 CAS 更新摘要；冲突时不覆盖另一请求。

        传入 ``db`` 时不自行开事务，供最终回复与摘要游标原子提交。
        """
        if not prepared.should_commit:
            return False
        summary = prepared.summary or _empty_summary()
        summary = dict(summary)
        summary["source_message_ids"] = list(prepared.source_message_ids)
        values = {
            "summary_json": json.dumps(summary, ensure_ascii=False, sort_keys=True),
            "source_hash": prepared.source_hash,
            "source_message_ids_json": json.dumps(
                list(prepared.source_message_ids), separators=(",", ":")),
            "source_message_hashes_json": json.dumps(
                prepared.source_message_hashes, sort_keys=True, separators=(",", ":")),
            "covered_until_user_message_id": prepared.covered_until_user_message_id,
            "snapshot_anchor_message_id": prepared.anchor_message_id,
        }
        own_session = db is None
        try:
            if own_session:
                with session_scope() as transaction:
                    self._commit_in_session(
                        transaction, prepared, values,
                        validate_snapshot=validate_snapshot,
                    )
            else:
                self._commit_in_session(
                    db, prepared, values, validate_snapshot=validate_snapshot,
                )
        except IntegrityError as exc:
            raise ConversationSummaryConflict(
                "项目会话摘要已被并发请求创建，请重试当前问题"
            ) from exc
        return True

    def _commit_in_session(self, db: Session, prepared: PreparedConversationMemory,
                           values: dict[str, Any], *, validate_snapshot: bool) -> None:
        if validate_snapshot:
            self.validate_snapshot(db, prepared)
        if prepared.expected_version is None:
            db.add(ConversationSummary(
                project_id=prepared.project_id, version=1, **values,
            ))
            db.flush()
            return

        result = db.execute(
            update(ConversationSummary)
            .where(ConversationSummary.project_id == prepared.project_id,
                   ConversationSummary.version == prepared.expected_version)
            .values(**values, version=prepared.expected_version + 1)
        )
        if result.rowcount != 1:
            raise ConversationSummaryConflict(
                "项目会话摘要已被并发请求更新，请重试当前问题"
            )
        # Core CAS updates do not mark an ORM row dirty, so the normal
        # before_flush project snapshot hook cannot see them. The export runs
        # after the caller's transaction commits and only reads the snapshot.
        from app.services.desktop import enabled as desktop_enabled

        if desktop_enabled():
            db.info.setdefault("project_exports", set()).add(prepared.project_id)
