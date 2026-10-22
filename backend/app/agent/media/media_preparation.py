"""Prepare same-source ASR and complete 1fps frames before joint analysis.

Pipeline integration contract (kept separate from the live pipeline during the
current recovery run)::

    prepared = prepare_media(
        request, terms=terms,
        asr_engine_factory=lambda: WhisperTranscriber(model_dir=..., ...),
        checkpoint=self._checkpoint,
        video_source_resolver=lambda checkpoint: MediaSource(
            source_key=request.source_key,
            path=download_video(..., progress_cb=lambda _: checkpoint()),
        ),
    )
    # Future visual_review signature:
    # review_frames(video, work, duration, sources, call, checkpoint,
    #               progress=lambda message: None, *,
    #               prepared_frames: FrameBundle | None = None,
    #               media_identity: MediaIdentity | None = None)
    # Pass prepared.frames and prepared.identity. It must verify source key,
    # video SHA, duration, manifest hashes and second coverage, then reuse these
    # paths instead of calling extract_second_frames again.

The only parallel work is the ordered base -> medium ASR branch and the video
resolve -> original-video 1fps extraction branch. This module does not promise
an end-to-end speedup. On failure or cancellation it signals both workers,
waits for both futures to exit, and only then removes its private attempt tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from app.core.cancel import TaskCancelled

DurationProbe = Callable[[Path, Callable[[], None]], float]
AsrRunner = Callable[[Path, Path, str, tuple[str, ...], Callable[[], None]], "TranscriptBundle"]
FrameRunner = Callable[[Path, Path, str, Callable[[], None]], "FrameBundle"]
VideoSourceResolver = Callable[[Callable[[], None]], "MediaSource | None"]


@dataclass(frozen=True)
class MediaSource:
    """Resolved source for one immutable origin (for example bvid + CID)."""

    source_key: str
    path: Path | None

    def __post_init__(self) -> None:
        if not self.source_key:
            raise ValueError("媒体 source_key 不能为空")
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class MediaPreparationRequest:
    source_key: str
    audio_path: Path
    video_source: MediaSource | VideoSourceResolver | None
    work_dir: Path
    expected_duration_seconds: float
    page_index: int | None = None
    range_start_seconds: float = 0.0
    range_end_seconds: float | None = None
    timeline_origin_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_key:
            raise ValueError("媒体 source_key 不能为空")
        object.__setattr__(self, "audio_path", Path(self.audio_path))
        object.__setattr__(self, "work_dir", Path(self.work_dir))
        values = (
            self.expected_duration_seconds,
            self.range_start_seconds,
            self.timeline_origin_seconds,
        )
        if any(not math.isfinite(x) for x in values):
            raise ValueError("媒体时长与时间轴必须为有限数")
        if self.expected_duration_seconds <= 0:
            raise ValueError("必须提供有效的完整素材时长")
        if (
            self.range_start_seconds != 0
            or self.timeline_origin_seconds != 0
            or (
                self.range_end_seconds is not None
                and not math.isclose(
                    self.range_end_seconds, self.expected_duration_seconds,
                    rel_tol=0.0, abs_tol=1e-6,
                )
            )
        ):
            raise ValueError("当前媒体准备接口只支持全片；片段范围尚未实现，拒绝截断或重定位时间轴")


@dataclass(frozen=True)
class FrameRecord:
    second: int
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class FrameBundle:
    source_key: str
    video_sha256: str
    duration_seconds: float
    frames: tuple[FrameRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))


@dataclass(frozen=True)
class TranscriptBundle:
    source_key: str
    audio_sha256: str
    base_path: Path
    medium_path: Path
    audio_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_path", Path(self.base_path))
        object.__setattr__(self, "medium_path", Path(self.medium_path))


@dataclass(frozen=True)
class MediaIdentity:
    source_key: str
    page_index: int | None
    audio_sha256: str
    video_sha256: str | None
    audio_duration_seconds: float
    video_duration_seconds: float | None
    range_start_seconds: float
    range_end_seconds: float
    timeline_origin_seconds: float


@dataclass(frozen=True)
class BranchResult:
    name: Literal["asr", "frames"]
    status: Literal["pending", "running", "ready", "failed", "cancelled", "skipped"]
    elapsed_seconds: float
    media_duration_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class MediaPreparationResult:
    status: Literal["ready", "audio_only", "failed", "cancelled"]
    branches: tuple[BranchResult, ...]
    identity: MediaIdentity | None
    transcript: TranscriptBundle | None
    frames: FrameBundle | None
    attempt_dir: Path | None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def ready_for_joint_analysis(self) -> bool:
        return (
            self.status == "ready"
            and self.identity is not None
            and self.transcript is not None
            and self.frames is not None
            and all(branch.status == "ready" for branch in self.branches)
        )

    def require_joint_ready(self) -> "MediaPreparationResult":
        """Guard the only state that is safe to pass to visual analysis."""
        if not self.ready_for_joint_analysis:
            raise MediaPreparationError(self)
        return self


class MediaPreparationError(RuntimeError):
    """One branch failed; ``result`` contains both branches after they exited."""

    def __init__(self, result: MediaPreparationResult):
        self.result = result
        super().__init__(result.error_message or "媒体准备失败，未进入音画联合分析")


class MediaPreparationCancelled(TaskCancelled):
    """Cancellation after both workers have exited and cleanup has completed."""

    def __init__(self, result: MediaPreparationResult):
        self.result = result
        super().__init__("媒体准备已取消；ASR 与 FFmpeg 分支均已退出")


class NoVideoStreamError(ValueError):
    """The supplied local media is valid audio but contains no visual stream."""


@dataclass
class _MutableBranch:
    name: Literal["asr", "frames"]
    status: str = "pending"
    started: float | None = None
    elapsed: float = 0.0
    duration: float | None = None
    error_type: str | None = None
    error_message: str | None = None


def _sha256(path: Path, checkpoint: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            checkpoint()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def probe_media_duration(path: Path, checkpoint: Callable[[], None]) -> float:
    """Measure actual audio duration using the repository's cancellable ffprobe runner."""
    from app.worker.evidence import run_command
    from app.worker.framer import FFPROBE

    raw = run_command(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        checkpoint,
    )
    payload = json.loads(raw)
    duration = float((payload.get("format") or {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("无法确定音频素材时长")
    return duration


def _default_asr_runner(
    audio: Path,
    out_dir: Path,
    source_key: str,
    terms: tuple[str, ...],
    checkpoint: Callable[[], None],
    *,
    engine_factory: Callable[[], object] | None,
) -> TranscriptBundle:
    from app.worker.transcriber import WhisperTranscriber

    engine = (engine_factory or WhisperTranscriber)()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for track in ("base", "medium"):
        checkpoint()
        target = out_dir / f"{track}.txt"
        engine.run(
            track,
            audio,
            target,
            list(terms),
            progress_cb=lambda _pct, _message: checkpoint(),
        )
        checkpoint()
        outputs.append(target)
    return TranscriptBundle(
        source_key=source_key,
        audio_sha256=_sha256(audio, checkpoint),
        base_path=outputs[0],
        medium_path=outputs[1],
    )


def _default_frame_runner(
    video: Path,
    out_dir: Path,
    source_key: str,
    checkpoint: Callable[[], None],
) -> FrameBundle:
    from app.agent.media.visual_review import extract_second_frames

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        duration, paths = extract_second_frames(video, out_dir, checkpoint)
    except ValueError as exc:
        if "素材不含视频流" in str(exc):
            raise NoVideoStreamError("音频素材不含视频流") from exc
        raise
    rows = tuple(
        FrameRecord(second=index, path=path, sha256=_sha256(path, checkpoint))
        for index, path in enumerate(paths)
    )
    return FrameBundle(
        source_key=source_key,
        video_sha256=_sha256(video, checkpoint),
        duration_seconds=float(duration),
        frames=rows,
    )


def _close_enough(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(2.0, expected * 0.01)


def _branch_snapshot(branch: _MutableBranch) -> BranchResult:
    elapsed = branch.elapsed
    if branch.started is not None and branch.status == "running":
        elapsed += max(0.0, time.monotonic() - branch.started)
    return BranchResult(
        name=branch.name,
        status=branch.status,  # type: ignore[arg-type]
        elapsed_seconds=elapsed,
        media_duration_seconds=branch.duration,
        error_type=branch.error_type,
        error_message=branch.error_message,
    )


def _validate_transcript(
    transcript: TranscriptBundle,
    *,
    request: MediaPreparationRequest,
    audio_sha: str,
    audio_duration: float,
    checkpoint: Callable[[], None],
) -> None:
    if transcript.source_key != request.source_key or transcript.audio_sha256 != audio_sha:
        raise ValueError("ASR 输出的 source_key 或音频 SHA 与任务身份不一致")
    if not transcript.base_path.is_file() or not transcript.medium_path.is_file():
        raise ValueError("dual ASR 未同时产出 base 与 medium 文件")
    if not transcript.base_path.read_text(encoding="utf-8").strip():
        raise ValueError("base 转写为空")
    if not transcript.medium_path.read_text(encoding="utf-8").strip():
        raise ValueError("medium 转写为空")
    if transcript.audio_duration_seconds is not None and not _close_enough(
        transcript.audio_duration_seconds, audio_duration
    ):
        raise ValueError("ASR 声称的音频时长与实际媒体时长不一致")
    checkpoint()


def _validate_frames(
    frames: FrameBundle,
    *,
    request: MediaPreparationRequest,
    video_sha: str,
    checkpoint: Callable[[], None],
) -> None:
    if frames.source_key != request.source_key or frames.video_sha256 != video_sha:
        raise ValueError("帧 bundle 的 source_key 或视频 SHA 与任务身份不一致")
    duration = float(frames.duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("无法确定视频流时长")
    if not _close_enough(duration, request.expected_duration_seconds):
        raise ValueError("视频流实际时长与来源时长不一致")
    expected_seconds = tuple(range(math.ceil(duration)))
    actual_seconds = tuple(frame.second for frame in frames.frames)
    if actual_seconds != expected_seconds:
        raise ValueError(
            f"逐秒帧覆盖不完整：预期 {len(expected_seconds)} 帧 0..{len(expected_seconds)-1}，"
            f"实际 {len(actual_seconds)} 帧"
        )
    for frame in frames.frames:
        checkpoint()
        if not frame.path.is_file() or not frame.path.stat().st_size:
            raise ValueError(f"第 {frame.second} 秒帧缺失或为空")
        if _sha256(frame.path, checkpoint) != frame.sha256:
            raise ValueError(f"第 {frame.second} 秒帧 SHA 校验失败")


def prepare_media(
    request: MediaPreparationRequest,
    *,
    terms: tuple[str, ...] | list[str] = (),
    checkpoint: Callable[[], None] = lambda: None,
    asr_runner: AsrRunner | None = None,
    frame_runner: FrameRunner | None = None,
    video_source_resolver: VideoSourceResolver | None = None,
    audio_duration_probe: DurationProbe = probe_media_duration,
    asr_engine_factory: Callable[[], object] | None = None,
) -> MediaPreparationResult:
    """Run ASR and video/frame preparation concurrently, then join by identity.

    Injectable runners are intended for local fixture tests. In production the
    helper uses the existing ``WhisperTranscriber`` and
    ``visual_review.extract_second_frames`` adapters. ``video_source_resolver``
    runs inside the frame branch, so video download can overlap ASR as well.
    """
    if not request.audio_path.is_file():
        raise ValueError("音频素材不存在")
    request.work_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = Path(tempfile.mkdtemp(prefix="media-preparation-", dir=request.work_dir))
    cancel_event = threading.Event()
    state_lock = threading.Lock()
    branch_state = {name: _MutableBranch(name) for name in ("asr", "frames")}
    worker_errors: dict[str, BaseException] = {}
    asr_dir = attempt_dir / "asr"
    frames_dir = attempt_dir / "frames"
    normalized_terms = tuple(terms)

    def branch_checkpoint() -> None:
        checkpoint()
        if cancel_event.is_set():
            raise TaskCancelled("另一媒体分支已失败或取消")

    def run_branch(name: Literal["asr", "frames"], fn):
        state = branch_state[name]
        with state_lock:
            state.status = "running"
            state.started = time.monotonic()
        try:
            value = fn()
            branch_checkpoint()
            with state_lock:
                if state.status != "skipped":
                    state.status = "ready"
            return value
        except TaskCancelled as exc:
            with state_lock:
                state.status = "cancelled"
                state.error_type = type(exc).__name__
                state.error_message = str(exc)
            raise
        except BaseException as exc:
            with state_lock:
                state.status = "failed"
                state.error_type = type(exc).__name__
                state.error_message = str(exc)
                worker_errors[name] = exc
            cancel_event.set()
            raise
        finally:
            with state_lock:
                if state.started is not None:
                    state.elapsed += max(0.0, time.monotonic() - state.started)
                    state.started = None

    def asr_work():
        branch_checkpoint()
        audio_sha = _sha256(request.audio_path, branch_checkpoint)
        duration = float(audio_duration_probe(request.audio_path, branch_checkpoint))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("无法确定音频素材时长")
        if not _close_enough(duration, request.expected_duration_seconds):
            raise ValueError("音频实际时长与来源时长不一致")
        asr_dir.mkdir(parents=True, exist_ok=True)
        runner = asr_runner
        if runner is None:
            def runner(audio, out_dir, source_key, selected_terms, check):
                return _default_asr_runner(
                    audio, out_dir, source_key, selected_terms, check,
                    engine_factory=asr_engine_factory,
                )
        transcript = runner(
            request.audio_path, asr_dir, request.source_key, normalized_terms, branch_checkpoint
        )
        _validate_transcript(
            transcript,
            request=request,
            audio_sha=audio_sha,
            audio_duration=duration,
            checkpoint=branch_checkpoint,
        )
        if _sha256(request.audio_path, branch_checkpoint) != audio_sha:
            raise ValueError("ASR 运行期间音频文件发生变化")
        state = branch_state["asr"]
        state.duration = duration
        return transcript, audio_sha, duration

    def frames_work():
        branch_checkpoint()
        source: MediaSource | None
        if video_source_resolver is not None:
            source = video_source_resolver(branch_checkpoint)
            if source is None:
                raise ValueError("视频来源解析失败，不能按音频无画面跳过")
        elif callable(request.video_source):
            source = request.video_source(branch_checkpoint)
            if source is None:
                raise ValueError("视频来源解析失败，不能按音频无画面跳过")
        else:
            source = request.video_source
        if source is None or source.path is None:
            with state_lock:
                branch_state["frames"].status = "skipped"
                branch_state["frames"].error_type = "NoVideoSource"
                branch_state["frames"].error_message = "只有音频素材；未生成逐秒帧 bundle"
            return None
        if source.source_key != request.source_key:
            raise ValueError("视频来源身份与音频来源身份不一致（BVID/CID 或本地来源不匹配）")
        video = Path(source.path)
        if not video.is_file():
            raise ValueError("视频素材不存在")
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_sha = _sha256(video, branch_checkpoint)
        runner = frame_runner or _default_frame_runner
        try:
            frames = runner(video, frames_dir, request.source_key, branch_checkpoint)
        except NoVideoStreamError as exc:
            with state_lock:
                branch_state["frames"].status = "skipped"
                branch_state["frames"].error_type = type(exc).__name__
                branch_state["frames"].error_message = str(exc)
            return None
        _validate_frames(
            frames,
            request=request,
            video_sha=video_sha,
            checkpoint=branch_checkpoint,
        )
        if _sha256(video, branch_checkpoint) != video_sha:
            raise ValueError("抽帧期间原视频文件发生变化")
        state = branch_state["frames"]
        state.duration = frames.duration_seconds
        return frames, video_sha, video

    futures: dict[Future, str] = {}
    external_cancel = False
    audio_result = video_result = None
    failures: dict[str, BaseException] = {}
    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="media-prepare") as pool:
            futures[pool.submit(run_branch, "asr", asr_work)] = "asr"
            futures[pool.submit(run_branch, "frames", frames_work)] = "frames"
            pending = set(futures)
            while pending:
                try:
                    checkpoint()
                except TaskCancelled as exc:
                    external_cancel = True
                    cancel_event.set()
                    failures["caller"] = exc
                    for future in pending:
                        future.cancel()
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    name = futures[future]
                    if future.cancelled():
                        with state_lock:
                            branch_state[name].status = "cancelled"
                        continue
                    try:
                        value = future.result()
                        if name == "asr":
                            audio_result = value
                        else:
                            video_result = value
                    except TaskCancelled as exc:
                        failures[name] = exc
                        if not external_cancel:
                            cancel_event.set()
                    except BaseException as exc:
                        failures[name] = exc
                        cancel_event.set()
            # ThreadPoolExecutor waits for all running tasks before this point.
    except TaskCancelled as exc:
        external_cancel = True
        cancel_event.set()
        failures["caller"] = exc

    if not external_cancel:
        try:
            checkpoint()
        except TaskCancelled as exc:
            external_cancel = True
            cancel_event.set()
            failures["caller"] = exc

    if external_cancel:
        cancel_event.set()

    if not external_cancel and not failures:
        try:
            if audio_result is not None:
                audio_sha = audio_result[1]
                if _sha256(request.audio_path, checkpoint) != audio_sha:
                    raise ValueError("汇合前音频文件发生变化")
            if video_result is not None:
                video_sha = video_result[1]
                video_path = video_result[2]
                if _sha256(video_path, checkpoint) != video_sha:
                    raise ValueError("汇合前原视频文件发生变化")
        except TaskCancelled as exc:
            external_cancel = True
            cancel_event.set()
            failures["caller"] = exc
        except BaseException as exc:
            failures["join"] = exc

    branches = tuple(_branch_snapshot(branch_state[name]) for name in ("asr", "frames"))

    # A resolved audio-only input is compatible as an ASR artifact, but it is
    # explicitly partial and cannot be passed to joint visual review.
    audio_only = branch_state["frames"].status == "skipped"
    if not external_cancel and not failures and audio_result is not None and audio_only:
        transcript, audio_sha, audio_duration = audio_result
        identity = MediaIdentity(
            source_key=request.source_key,
            page_index=request.page_index,
            audio_sha256=audio_sha,
            video_sha256=None,
            audio_duration_seconds=audio_duration,
            video_duration_seconds=None,
            range_start_seconds=0.0,
            range_end_seconds=audio_duration,
            timeline_origin_seconds=request.timeline_origin_seconds,
        )
        return MediaPreparationResult(
            status="audio_only",
            branches=branches,
            identity=identity,
            transcript=transcript,
            frames=None,
            attempt_dir=attempt_dir,
            error_type="NoVideoSource",
            error_message="ASR 已就绪；缺少视频帧，不能进入音画联合分析",
        )

    if not external_cancel and not failures and audio_result is not None and video_result is not None:
        transcript, audio_sha, audio_duration = audio_result
        frames, video_sha, _video_path = video_result
        if not _close_enough(audio_duration, frames.duration_seconds):
            failures["join"] = ValueError("音频与视频实际时长不一致，拒绝跨源或错时合并")
        else:
            identity = MediaIdentity(
                source_key=request.source_key,
                page_index=request.page_index,
                audio_sha256=audio_sha,
                video_sha256=video_sha,
                audio_duration_seconds=audio_duration,
                video_duration_seconds=frames.duration_seconds,
                range_start_seconds=0.0,
                range_end_seconds=frames.duration_seconds,
                timeline_origin_seconds=request.timeline_origin_seconds,
            )
            return MediaPreparationResult(
                status="ready",
                branches=branches,
                identity=identity,
                transcript=transcript,
                frames=frames,
                attempt_dir=attempt_dir,
            )

    # Record any future error that raced with the first failing branch.
    for name, exc in worker_errors.items():
        failures.setdefault(name, exc)
    primary_error = next(
        (exc for exc in failures.values() if not isinstance(exc, TaskCancelled)),
        next(iter(failures.values()), None),
    )
    status: Literal["failed", "cancelled"] = "cancelled" if external_cancel else "failed"
    if status == "failed" and primary_error is not None and isinstance(primary_error, TaskCancelled):
        status = "cancelled"
    result = MediaPreparationResult(
        status=status,
        branches=branches,
        identity=None,
        transcript=None,
        frames=None,
        attempt_dir=None,
        error_type=(type(primary_error).__name__ if primary_error else "MediaJoinError"),
        error_message=(str(primary_error) if primary_error else "两条媒体分支未成功汇合"),
    )
    # Safe only after executor join: neither ASR nor FFmpeg can recreate inputs.
    shutil.rmtree(attempt_dir, ignore_errors=True)
    if status == "cancelled":
        raise MediaPreparationCancelled(result)
    raise MediaPreparationError(result)
