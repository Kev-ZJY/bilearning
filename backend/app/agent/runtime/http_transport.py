"""Synchronous requests-shaped adapter over an asyncio/httpx transport.

Each call owns its client and event loop. Streaming responses retain those
resources until the body finishes or the response is closed, so cancellation
and the absolute total deadline also cover body reads.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json as json_module
import math
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import requests

from app.core.cancel import TaskCancelled


_POLL_INTERVAL_SECONDS = 0.025


class RequestDeadlineExceeded(TimeoutError):
    """The absolute total request deadline elapsed during headers or body."""

    def __init__(self, message: str, *, transport_phase: str = "connect_or_response_headers",
                 http_status: int | None = None, received_bytes: int = 0,
                 elapsed_ms: int = 0, request_id: str | None = None):
        self.transport_phase = transport_phase
        self.http_status = http_status
        self.received_bytes = received_bytes
        self.elapsed_ms = elapsed_ms
        self.request_id = request_id
        super().__init__(message)


@dataclass(frozen=True)
class _RequestControl:
    should_cancel: Callable[[], bool] | None = None
    on_transport_event: Callable[[dict], None] | None = None
    call_kind: str = "unknown"


_REQUEST_CONTROL: contextvars.ContextVar[_RequestControl] = contextvars.ContextVar(
    "http_transport_request_control", default=_RequestControl(),
)


@contextmanager
def request_controls(*, should_cancel: Callable[[], bool] | None = None,
                     on_transport_event: Callable[[dict], None] | None = None,
                     call_kind: str = "unknown"):
    """Set request-local cancellation and best-effort transport diagnostics."""
    if should_cancel is not None and not callable(should_cancel):
        raise TypeError("should_cancel must be callable")
    if on_transport_event is not None and not callable(on_transport_event):
        raise TypeError("on_transport_event must be callable")
    allowed_call_kinds = {
        "chat", "chat_json", "chat_text", "chat_tools", "stream_text",
        "stream_tools",
    }
    normalized_call_kind = (
        call_kind if isinstance(call_kind, str) and call_kind in allowed_call_kinds
        else "unknown"
    )
    token = _REQUEST_CONTROL.set(_RequestControl(
        should_cancel=should_cancel,
        on_transport_event=on_transport_event,
        call_kind=normalized_call_kind,
    ))
    try:
        yield
    finally:
        _REQUEST_CONTROL.reset(token)


def _parse_timeout(timeout: Any) -> tuple[float | None, float | None]:
    if timeout is None:
        return None, None
    if isinstance(timeout, (tuple, list)):
        if len(timeout) != 2:
            raise ValueError("timeout tuple must contain (connect, total)")
        connect, total = timeout
    else:
        connect = total = timeout

    def normalize(value: Any, name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} timeout must be a positive number or None")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} timeout must be a positive finite number")
        return value

    return normalize(connect, "connect"), normalize(total, "total")


class _Runtime:
    def __init__(self, connect_timeout: float | None, total_timeout: float | None):
        self.runner = asyncio.Runner()
        self.loop = self.runner.get_loop()
        self.deadline = (
            self.loop.time() + total_timeout if total_timeout is not None else None
        )
        self.control = _REQUEST_CONTROL.get()
        self.request_id = str(uuid.uuid4())
        self.call_kind = self.control.call_kind
        self.started_at = time.monotonic()
        self.transport_phase = "connect_or_response_headers"
        self.http_status: int | None = None
        self.received_bytes = 0
        self.request_sha256: str | None = None
        self.request_bytes: int | None = None
        self.response_format_sha256: str | None = None
        self.stream = False
        self._started_emitted = False
        self._terminal_emitted = False
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=None,
            write=connect_timeout,
            pool=connect_timeout,
        )
        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        )
        self.response: httpx.Response | None = None
        self.closed = False

    def _cancel_requested(self) -> bool:
        callback = self.control.should_cancel
        if callback is None:
            return False
        try:
            return bool(callback())
        except TaskCancelled:
            raise

    async def _await_controlled(self, awaitable):
        task = asyncio.create_task(awaitable)
        try:
            while True:
                if self._cancel_requested():
                    raise TaskCancelled()
                remaining = (
                    None if self.deadline is None else self.deadline - self.loop.time()
                )
                if remaining is not None and remaining <= 0:
                    raise self._deadline_error()
                wait_for = _POLL_INTERVAL_SECONDS
                if remaining is not None:
                    wait_for = min(wait_for, remaining)
                done, _pending = await asyncio.wait({task}, timeout=wait_for)
                if task in done:
                    result = task.result()
                    if self.deadline is not None and self.loop.time() > self.deadline:
                        close = getattr(result, "aclose", None)
                        if callable(close):
                            try:
                                await close()
                            except BaseException:
                                pass
                        if isinstance(result, httpx.Response):
                            self.response = result
                            self.http_status = result.status_code
                        raise self._deadline_error()
                    return result
        except BaseException:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    def run(self, awaitable):
        if self.closed:
            raise requests.exceptions.StreamConsumedError(
                "response transport is already closed"
            )
        try:
            return self.runner.run(self._await_controlled(awaitable))
        except BaseException as exc:
            self._emit_failure(exc)
            raise

    def set_phase(self, phase: str) -> None:
        if phase in {"connect_or_response_headers", "response_headers", "body"}:
            self.transport_phase = phase

    def _elapsed_ms(self) -> int:
        return max(0, int((time.monotonic() - self.started_at) * 1000))

    def _emit(self, kind: str, event: str, *, failure_type: str | None = None) -> None:
        callback = self.control.on_transport_event
        if callback is None:
            return
        row = {
            "kind": kind,
            "event": event,
            "request_id": self.request_id,
            "method": self.call_kind,
            "http_method": "POST",
            "request_sha256": self.request_sha256,
            "request_bytes": self.request_bytes,
            "response_format_sha256": self.response_format_sha256,
            "stream": self.stream,
            "transport_phase": self.transport_phase,
            "http_status": self.http_status,
            "received_bytes": self.received_bytes,
            "elapsed_ms": self._elapsed_ms(),
        }
        if failure_type is not None:
            row["failure_type"] = failure_type
        try:
            callback(row)
        except Exception:
            # Diagnostic persistence never changes transport behavior.
            pass

    def start(self, request: httpx.Request, *, json_payload: Any,
              stream: bool) -> None:
        body = request.content
        self.request_sha256 = hashlib.sha256(body).hexdigest()
        self.request_bytes = len(body)
        self.stream = bool(stream)
        response_format = (
            json_payload.get("response_format")
            if isinstance(json_payload, dict) else None
        )
        if response_format is not None:
            try:
                canonical = json_module.dumps(
                    response_format, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                self.response_format_sha256 = hashlib.sha256(canonical).hexdigest()
            except (TypeError, ValueError, OverflowError):
                self.response_format_sha256 = None
        self._started_emitted = True
        self._emit("request", "request_started")

    def response_headers_received(self, response: httpx.Response) -> None:
        self.response = response
        self.http_status = response.status_code
        self.set_phase("response_headers")
        self._emit("transport", "response_headers")
        self.set_phase("body")

    def add_received_bytes(self, chunk: bytes) -> None:
        self.received_bytes += len(chunk)

    async def read_body(self, response: httpx.Response) -> bytes:
        """Read a buffered response in one controlled coroutine, counting chunks.

        The reported byte count is the number of complete decoded body chunks
        yielded by httpx, not TCP framing or a partial chunk still in transit.
        """
        chunks = []
        async for chunk in response.aiter_bytes():
            chunk = bytes(chunk)
            self.add_received_bytes(chunk)
            chunks.append(chunk)
        return b"".join(chunks)

    def body_complete(self) -> None:
        if self._terminal_emitted:
            return
        self.set_phase("body")
        self._terminal_emitted = True
        self._emit("transport", "response_body_complete")

    def request_closed(self) -> None:
        if not self._started_emitted or self._terminal_emitted:
            return
        self._terminal_emitted = True
        self._emit("transport", "response_closed")

    def _deadline_error(self) -> RequestDeadlineExceeded:
        return RequestDeadlineExceeded(
            "HTTP request exceeded total deadline",
            transport_phase=self.transport_phase,
            http_status=self.http_status,
            received_bytes=self.received_bytes,
            elapsed_ms=self._elapsed_ms(),
            request_id=self.request_id,
        )

    def _emit_failure(self, exc: BaseException) -> None:
        if not self._started_emitted or self._terminal_emitted:
            return
        if isinstance(exc, TaskCancelled):
            event = "request_cancelled"
            failure_type = "TaskCancelled"
        elif isinstance(exc, RequestDeadlineExceeded):
            event = "request_failed"
            failure_type = "RequestDeadlineExceeded"
            self.transport_phase = exc.transport_phase
            self.http_status = exc.http_status
            self.received_bytes = exc.received_bytes
        else:
            event = "request_failed"
            allowed = {
                "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
                "ConnectError", "ReadError", "WriteError", "RemoteProtocolError",
                "ProxyError", "NetworkError", "TransportError", "RequestError",
            }
            name = type(exc).__name__
            failure_type = name if name in allowed else "TransportError"
        self._terminal_emitted = True
        self._emit("transport", event, failure_type=failure_type)

    async def _close_async(self):
        if self.response is not None:
            try:
                await self.response.aclose()
            except BaseException:
                pass
        try:
            await self.client.aclose()
        except BaseException:
            pass

    def close(self):
        if self.closed:
            return
        self.request_closed()
        try:
            self.runner.run(self._close_async())
        except BaseException:
            # Cleanup must not replace a cancellation, deadline, or transport error.
            pass
        finally:
            self.closed = True
            try:
                self.runner.close()
            except BaseException:
                pass


def _encoding(headers) -> str:
    content_type = headers.get("content-type", "")
    for parameter in content_type.split(";")[1:]:
        key, separator, value = parameter.strip().partition("=")
        if separator and key.casefold() == "charset":
            candidate = value.strip().strip("\"'")
            if candidate:
                return candidate
    return "utf-8"


def _line_parts(buffer: bytes, *, final: bool, delimiter: bytes | None):
    if delimiter is not None:
        if not delimiter:
            raise ValueError("delimiter must not be empty")
        lines = []
        while True:
            position = buffer.find(delimiter)
            if position < 0:
                break
            lines.append(buffer[:position])
            buffer = buffer[position + len(delimiter):]
        if final and buffer:
            lines.append(buffer)
            buffer = b""
        return lines, buffer

    lines = []
    while True:
        positions = [position for position in (buffer.find(b"\n"), buffer.find(b"\r"))
                     if position >= 0]
        if not positions:
            break
        position = min(positions)
        marker = buffer[position:position + 1]
        if marker == b"\r" and position + 1 == len(buffer) and not final:
            break
        width = 2 if marker == b"\r" and buffer[position:position + 2] == b"\r\n" else 1
        lines.append(buffer[:position])
        buffer = buffer[position + width:]
    if final and buffer:
        lines.append(buffer)
        buffer = b""
    return lines, buffer


class _BufferedResponse:
    def __init__(self, status_code: int, headers: httpx.Headers, content: bytes,
                 request_id: str | None = None):
        self.status_code = status_code
        self.request_id = request_id
        self.headers = httpx.Headers(headers)
        self.content = bytes(content)
        self.encoding = _encoding(self.headers)

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding, errors="replace")

    def json(self, **kwargs):
        return json_module.loads(self.content, **kwargs)

    def bounded_error_json(self, max_bytes: int = 32_768):
        """Parse only small JSON error bodies; do not retain or expose raw text."""
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        limit = min(max_bytes, 32_768)
        if len(self.content) > limit:
            return None
        try:
            value = json_module.loads(self.content)
        except (ValueError, TypeError, UnicodeDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def iter_lines(self, chunk_size=512, decode_unicode=False, delimiter=None):
        delimiter_bytes = delimiter
        if isinstance(delimiter_bytes, str):
            delimiter_bytes = delimiter_bytes.encode(self.encoding)
        if delimiter_bytes is not None and not isinstance(delimiter_bytes, bytes):
            raise TypeError("delimiter must be bytes or str")
        lines, _remaining = _line_parts(
            self.content, final=True, delimiter=delimiter_bytes,
        )
        if decode_unicode:
            return iter(line.decode(self.encoding, errors="replace") for line in lines)
        return iter(lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code} response", response=self,
            )

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


class _StreamingResponse:
    def __init__(self, runtime: _Runtime, response: httpx.Response):
        self._runtime = runtime
        self._response = response
        self.status_code = response.status_code
        self.request_id = runtime.request_id
        self.headers = httpx.Headers(response.headers)
        self.encoding = _encoding(self.headers)
        self._body = bytearray()
        self._body_complete = False
        self._closed = False
        self._iteration_active = False
        self._body_iterator = runtime.run(self._make_body_iterator())

    async def _make_body_iterator(self):
        return self._response.aiter_bytes().__aiter__()

    async def _next_async_chunk(self):
        try:
            return await self._body_iterator.__anext__()
        except StopAsyncIteration:
            return None

    def _pull_chunk(self):
        if self._body_complete:
            return None
        if self._closed:
            raise requests.exceptions.StreamConsumedError(
                "response stream was closed before the body was complete"
            )
        try:
            chunk = self._runtime.run(self._next_async_chunk())
        except BaseException:
            self.close()
            raise
        if chunk is None:
            self._body_complete = True
            self._runtime.body_complete()
            self.close()
            return None
        chunk = bytes(chunk)
        self._runtime.add_received_bytes(chunk)
        return chunk

    def _next_chunk(self):
        chunk = self._pull_chunk()
        if chunk is not None:
            self._body.extend(chunk)
        return chunk

    def bounded_error_json(self, max_bytes: int = 32_768):
        """Read and parse a JSON error body without buffering beyond 32 KiB.

        The underlying read still passes through the request runtime, retaining
        its absolute deadline and cancellation checks. A chunk that would cross
        the bound is not appended to the retained body and the response closes.
        """
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        limit = min(max_bytes, 32_768)
        if self._iteration_active:
            return None
        if len(self._body) > limit:
            self.close()
            return None
        while not self._body_complete:
            chunk = self._pull_chunk()
            if chunk is None:
                break
            room = limit - len(self._body)
            if len(chunk) > room:
                if room > 0:
                    self._body.extend(chunk[:room])
                self.close()
                return None
            self._body.extend(chunk)
        try:
            value = json_module.loads(bytes(self._body))
        except (ValueError, TypeError, UnicodeDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _read_all(self) -> bytes:
        if self._iteration_active:
            raise requests.exceptions.StreamConsumedError(
                "response stream is currently being iterated"
            )
        while not self._body_complete:
            if self._next_chunk() is None:
                break
        return bytes(self._body)

    @property
    def content(self) -> bytes:
        return self._read_all()

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding, errors="replace")

    def json(self, **kwargs):
        return json_module.loads(self.content, **kwargs)

    def iter_lines(self, chunk_size=512, decode_unicode=False, delimiter=None):
        delimiter_bytes = delimiter
        if isinstance(delimiter_bytes, str):
            delimiter_bytes = delimiter_bytes.encode(self.encoding)
        if delimiter_bytes is not None and not isinstance(delimiter_bytes, bytes):
            raise TypeError("delimiter must be bytes or str")
        if delimiter_bytes == b"":
            raise ValueError("delimiter must not be empty")
        if self._body_complete:
            lines, _remaining = _line_parts(
                bytes(self._body), final=True, delimiter=delimiter_bytes,
            )
            if decode_unicode:
                return iter(line.decode(self.encoding, errors="replace") for line in lines)
            return iter(lines)
        if self._iteration_active:
            raise requests.exceptions.StreamConsumedError(
                "response stream is already being iterated"
            )
        return self._iter_lines(decode_unicode=decode_unicode, delimiter=delimiter_bytes)

    def _iter_lines(self, *, decode_unicode: bool, delimiter: bytes | None):
        self._iteration_active = True
        buffer = b""
        try:
            while True:
                chunk = self._next_chunk()
                if chunk is None:
                    lines, buffer = _line_parts(buffer, final=True, delimiter=delimiter)
                    for line in lines:
                        yield line.decode(self.encoding, errors="replace") if decode_unicode else line
                    return
                buffer += chunk
                lines, buffer = _line_parts(buffer, final=False, delimiter=delimiter)
                for line in lines:
                    yield line.decode(self.encoding, errors="replace") if decode_unicode else line
        except GeneratorExit:
            self.close()
            raise
        except BaseException:
            self.close()
            raise
        finally:
            self._iteration_active = False

    def raise_for_status(self):
        if self.status_code >= 400:
            self.close()
            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code} response", response=self,
            )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def _map_httpx_error(exc: BaseException) -> BaseException | None:
    if isinstance(exc, httpx.ConnectTimeout):
        return requests.exceptions.ConnectTimeout("HTTP connection timed out")
    if isinstance(exc, httpx.TimeoutException):
        return requests.exceptions.Timeout("HTTP transport timed out")
    if isinstance(exc, httpx.ProxyError):
        return requests.exceptions.ProxyError("HTTP proxy request failed")
    if isinstance(exc, httpx.NetworkError):
        return requests.exceptions.ConnectionError("HTTP connection failed")
    if isinstance(exc, httpx.TransportError):
        return requests.exceptions.ConnectionError("HTTP transport failed")
    if isinstance(exc, httpx.RequestError):
        return requests.exceptions.RequestException("HTTP transport request failed")
    if isinstance(exc, httpx.InvalidURL):
        return requests.exceptions.InvalidURL("Invalid HTTP request URL")
    return None


def post(
    url: str,
    headers: dict[str, str] | None = None,
    json: Any = None,
    stream: bool = False,
    timeout: Any = None,
):
    """POST a JSON request with requests-compatible response accessors."""
    connect_timeout, total_timeout = _parse_timeout(timeout)
    runtime = _Runtime(connect_timeout, total_timeout)
    try:
        request = runtime.client.build_request("POST", url, headers=headers, json=json)
        runtime.start(request, json_payload=json, stream=stream)
        response = runtime.run(runtime.client.send(request, stream=True))
        runtime.response_headers_received(response)
        if stream:
            return _StreamingResponse(runtime, response)
        runtime.set_phase("body")
        content = runtime.run(runtime.read_body(response))
        runtime.body_complete()
        buffered = _BufferedResponse(response.status_code, response.headers, content,
                                     runtime.request_id)
        runtime.close()
        return buffered
    except BaseException as exc:
        runtime.close()
        if isinstance(exc, (TaskCancelled, RequestDeadlineExceeded)):
            raise
        mapped = _map_httpx_error(exc)
        if mapped is not None:
            raise mapped from None
        raise
