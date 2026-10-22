"""Thread-safe finite budget for model HTTP attempts within one task.

The budget counts each attempted POST immediately before it is sent. It stores
counts and limit only; request bodies, credentials, model output and reasoning
are intentionally outside this diagnostic.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class ModelAttemptBudgetExceeded(RuntimeError):
    """Raised before sending an HTTP request once a task has spent its budget."""

    def __init__(self, snapshot: dict[str, Any]):
        self.snapshot = dict(snapshot)
        super().__init__(
            "任务模型 HTTP 尝试预算已耗尽 "
            f"({snapshot['used']}/{snapshot['limit']}); 已保留已完成阶段的证据与草稿（若已生成）"
        )


class TaskBudget:
    """A single synchronized counter shared by every LLM client in a task."""

    def __init__(
        self,
        max_attempts: int = 200,
        *,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ):
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        self.max_attempts = max_attempts
        self.on_change = on_change
        self._lock = threading.Lock()
        self._used = 0
        self._denied = 0
        self._observer_error: str | None = None

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "limit": self.max_attempts,
            "used": self._used,
            "remaining": max(0, self.max_attempts - self._used),
            "denied": self._denied,
            "observer_error": self._observer_error,
            "status": "exhausted" if self._used >= self.max_attempts else "active",
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def reserve_attempt(self) -> None:
        """Consume one slot or raise before any network operation is started."""
        with self._lock:
            if self._used >= self.max_attempts:
                self._denied += 1
                snapshot = self._snapshot_locked()
                denied = True
            else:
                self._used += 1
                snapshot = self._snapshot_locked()
                denied = False
        if self.on_change is not None:
            try:
                self.on_change(dict(snapshot))
            except Exception as exc:
                # Persistence/progress observers must not turn a reserved HTTP
                # attempt into an apparent network failure and trigger retries.
                # Keep only the exception type: callback messages may contain
                # arbitrary data and are not part of the task-budget artifact.
                with self._lock:
                    self._observer_error = type(exc).__name__
                    snapshot = self._snapshot_locked()
        if denied:
            raise ModelAttemptBudgetExceeded(snapshot)
