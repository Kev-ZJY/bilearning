"""Validated continuation from persisted, complete video evidence.

Recovery starts at the first stage whose durable inputs can be revalidated:
grounding the saved coverage draft.  This module never calls an LLM, reads
application credentials, writes the project document, or modifies a source
task directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Any

from PIL import Image

from app.core.database import session_scope
from app.core.queue import get_queue
from app.models.entities import Project, Video

RECOVERY_ID_RE = re.compile(r"^[a-f0-9]{24}$")
CHECKPOINT_VERSION = 2
MIN_TRANSCRIPT_CHARS = 200
_VISUAL_BATCH_SECONDS = 6
_PIPELINE_STAGES = {"pending", "downloading", "asr", "frames", "draft", "verify", "illustrate", "review", "revise", "render"}
_STARTUP_INTERRUPTION_ERROR = "上次应用退出时任务中断，可重新整理；已有文档已保留"
_STAT_HASH_CACHE: dict[tuple, dict[str, Any]] = {}
_VALID_JPEG_CACHE: set[tuple] = set()
_CACHE_LOCK = threading.Lock()
_CACHE_LIMIT = 8192


class RecoveryError(ValueError):
    """The persisted evidence cannot safely be resumed."""


class RecoveryNotFound(RecoveryError):
    """The project or requested video does not exist in this project."""


@dataclass
class RecoverySnapshot:
    project_id: int
    video_id: int
    recovery_id: str
    source_work: Path
    legacy_bootstrap: bool
    source_status: str
    source_error: str | None
    source_task_id: str | None
    intent: str
    terms: list[str]
    options: dict[str, Any]
    baseline_doc: dict
    baseline_doc_sha256: str
    media_identity: dict[str, dict]
    local_manifest: dict
    base_text: str
    medium_text: str
    visual_report: dict
    coverage_draft: dict
    sources: list[tuple[str, str]]
    frame_files: list[tuple[Path, Path, tuple]]
    source_file_receipts: list[dict]
    candidate_generation_fingerprint: dict | None
    target_candidate_generation_intent: dict | None
    candidate_generation_status: str
    candidate_generation_reason: str
    regenerate_initial_draft: bool


def document_sha256(doc: dict) -> str:
    raw = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _bytes_receipt(raw: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def _file_identity(path: Path) -> tuple:
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return (
        str(path), stat.st_dev, stat.st_ino, stat.st_size,
        stat.st_mtime_ns, stat.st_ctime_ns,
    )


def _sha256(path: Path, *, stat_cache: bool = False) -> dict[str, Any]:
    path = Path(path)
    before = path.stat()
    identity = (
        str(path.resolve(strict=True)), before.st_dev, before.st_ino,
        before.st_size, before.st_mtime_ns, before.st_ctime_ns,
    )
    if stat_cache:
        with _CACHE_LOCK:
            cached = _STAT_HASH_CACHE.get(identity)
        if cached is not None:
            return dict(cached)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RecoveryError("读取期间来源文件发生变化，拒绝恢复")
    result = {"sha256": digest.hexdigest(), "size": after.st_size}
    if stat_cache:
        with _CACHE_LOCK:
            if len(_STAT_HASH_CACHE) >= _CACHE_LIMIT:
                _STAT_HASH_CACHE.clear()
                _VALID_JPEG_CACHE.clear()
            _STAT_HASH_CACHE[identity] = dict(result)
    return result


def _within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve(strict=True).relative_to(Path(root).resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def _has_symlink_component(path: Path, stop: Path) -> bool:
    path = Path(os.path.abspath(path))
    stop = Path(os.path.abspath(stop))
    try:
        relative = path.relative_to(stop)
    except ValueError:
        return True
    cursor = stop
    if cursor.is_symlink():
        return True
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return True
    return False


def _safe_file(path: Path, root: Path, label: str) -> Path:
    path, root = Path(path), Path(root)
    if _has_symlink_component(path, root) or not _within(path, root) or not path.is_file():
        raise RecoveryError(f"{label}缺失、越界或经过符号链接，拒绝恢复")
    return path.resolve(strict=True)


def _read_json(path: Path, root: Path, label: str) -> tuple[dict, bytes]:
    safe = _safe_file(path, root, label)
    if safe.stat().st_size > 16 * 1024 * 1024:
        raise RecoveryError(f"{label}超过恢复读取上限")
    raw = safe.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"{label}无法解析") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label}格式无效")
    return value, raw


def _transcript_file(work: Path, kind: str) -> Path:
    """Accept both recovery's canonical names and the pipeline's ASR names."""
    for name in (f"{kind}.txt", f"transcript_{kind}.txt"):
        candidate = Path(work) / name
        if candidate.exists() or candidate.is_symlink():
            return _safe_file(candidate, work, f"{kind} 转写")
    raise RecoveryError(f"缺少 {kind} 转写")


def _load_project_state(project_id: int, video_id: int) -> tuple[Path, Video, dict, str]:
    with session_scope() as db:
        project = db.get(Project, project_id)
        video = db.get(Video, video_id)
        if project is None:
            raise RecoveryNotFound("项目不存在")
        if video is None or video.project_id != project_id:
            raise RecoveryNotFound("视频不存在或不属于当前项目")
        root = Path(project.storage_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise RecoveryError("项目保存目录不可用")
        if project.doc and project.doc.doc_json:
            try:
                baseline = json.loads(project.doc.doc_json)
            except json.JSONDecodeError as exc:
                raise RecoveryError("当前项目文档无法解析，拒绝恢复") from exc
            if not isinstance(baseline, dict):
                raise RecoveryError("当前项目文档格式无效，拒绝恢复")
        else:
            baseline = {}
        return root, video, baseline, document_sha256(baseline)


def _formal_content_is_empty(doc: dict) -> bool:
    modules = doc.get("modules") or []
    if not isinstance(modules, list):
        return False
    if any(isinstance(module, dict) and module.get("points") for module in modules):
        return False
    appendix = doc.get("appendix") or {}
    if not isinstance(appendix, dict):
        return False
    tables = appendix.get("cheatsheets") or []
    return isinstance(tables, list) and not tables


def _local_media(root: Path, video: Video, *, stat_cache: bool = False) -> tuple[dict, dict[str, dict], dict[str, Path]]:
    if video.source != "local":
        raise RecoveryError("仅支持已导入的本地 MP4 恢复")
    video_root = root / f"video_{video.id}"
    if video_root.is_symlink() or not video_root.is_dir() or not _within(video_root, root):
        raise RecoveryError("本地视频目录缺失、越界或为符号链接")
    manifest_path = _safe_file(video_root / "local.json", video_root, "本地媒体清单")
    manifest, manifest_raw = _read_json(manifest_path, video_root, "本地媒体清单")
    manifest["_validated_receipt"] = _bytes_receipt(manifest_raw)
    identities: dict[str, dict] = {}
    paths: dict[str, Path] = {}
    for kind in ("video", "audio"):
        raw = manifest.get(f"{kind}_path")
        if not isinstance(raw, str) or not raw.strip():
            raise RecoveryError(f"本地媒体清单缺少 {kind}_path")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = video_root / candidate
        media_path = _safe_file(candidate, video_root, f"本地 {kind} 文件")
        if kind == "video" and media_path.suffix.lower() != ".mp4":
            raise RecoveryError("本地视频来源不是 MP4")
        identities[kind] = {
            "path": media_path.relative_to(root).as_posix(),
            **_sha256(media_path, stat_cache=stat_cache),
        }
        paths[kind] = media_path
    return manifest, identities, paths


def _visual_text(report: dict) -> str:
    rows: list[str] = []
    for batch in report.get("batches", []):
        observations = batch.get("observations") or []
        for item in sorted(observations, key=lambda x: x.get("second", -1)):
            uncertainty = "，不确定" if item.get("uncertain") else ""
            rows.append(f"[画面 {int(item['second'])} 秒{uncertainty}] {item['text']}")
        if observations:
            end = int(max(item["second"] for item in observations)) + 1
            rows.append(f"[音画联合分析 {int(batch['start'])}–{end} 秒] {batch['analysis']}")
    return "\n".join(rows)


def _transcript_window(sources: list[tuple[str, str]], start: int, end: int) -> str:
    from app.agent.media.visual_review import transcript_window

    return transcript_window(sources, start, end)


def _validate_visual(root: Path, work: Path, report: dict, base: str, medium: str, duration: float, *, stat_cache: bool = False) -> list[tuple[Path, Path, tuple]]:
    if report.get("status") != "full_1fps_joint_review" or report.get("interval") != 1:
        raise RecoveryError("逐秒画面分析未明确完成全程覆盖")
    measured = report.get("duration")
    expected = report.get("expected_frames")
    if type(measured) not in (int, float) or not math.isfinite(measured) or measured <= 0:
        raise RecoveryError("逐秒画面报告缺少有效媒体时长")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise RecoveryError("视频登记时长无效")
    if abs(float(measured) - float(duration)) > max(2.0, float(duration) * 0.01):
        raise RecoveryError("画面分析时长与本地视频登记时长不匹配")
    count = math.ceil(float(measured))
    if type(expected) is not int or expected != count or report.get("reviewed_frames") != count:
        raise RecoveryError("全程逐秒画面数量与视频时长不匹配")
    times = report.get("times")
    if not isinstance(times, list) or times != list(range(count)):
        raise RecoveryError("逐秒画面报告缺帧、重复或时间标签顺序错误")

    sources = [("base", base), ("medium", medium)]
    batches = report.get("batches")
    expected_starts = list(range(0, count, _VISUAL_BATCH_SECONDS))
    if not isinstance(batches, list) or [row.get("start") for row in batches if isinstance(row, dict)] != expected_starts or len(batches) != len(expected_starts):
        raise RecoveryError("逐秒画面报告批次不连续或不完整")

    frame_pairs: list[tuple[Path, Path, tuple]] = []
    frame_dir: Path | None = None
    covered: list[int] = []
    for start, batch in zip(expected_starts, batches):
        end_second = min(start + _VISUAL_BATCH_SECONDS, count)
        observations, paths = batch.get("observations"), batch.get("paths")
        if type(batch.get("start")) is not int or batch["start"] != start:
            raise RecoveryError("画面批次起点无效")
        if type(batch.get("end")) not in (int, float) or abs(float(batch["end"]) - min(end_second, float(measured))) > 0.001:
            raise RecoveryError("画面批次终点与实际分析时长不匹配")
        if not isinstance(observations, list) or not isinstance(paths, list) or len(observations) != end_second - start or len(paths) != len(observations):
            raise RecoveryError("画面批次缺少逐帧观察或帧文件")
        observed_seconds = [item.get("second") if isinstance(item, dict) else None for item in observations]
        if observed_seconds != list(range(start, end_second)):
            raise RecoveryError("画面观察项未按秒完整且唯一覆盖")
        if not isinstance(batch.get("transcript"), str) or batch["transcript"] != _transcript_window(sources, start, min(end_second, float(measured))):
            raise RecoveryError("ASR 时间窗与逐秒画面报告中的原始转写不一致")
        if not isinstance(batch.get("analysis"), str) or not batch["analysis"].strip():
            raise RecoveryError("画面批次缺少音画联合分析")
        for observation, raw_path in zip(observations, paths):
            if (not isinstance(observation.get("text"), str) or not observation["text"].strip()
                    or type(observation.get("uncertain")) is not bool):
                raise RecoveryError("逐帧观察项格式无效")
            source = Path(raw_path)
            if not source.is_absolute():
                source = work / source
            safe = _safe_file(source, work, "原始抽帧")
            if safe.name != f"{int(observation['second']):08d}.jpg":
                raise RecoveryError("逐秒帧文件名与观察时间不匹配")
            if frame_dir is None:
                frame_dir = safe.parent
            elif safe.parent != frame_dir:
                raise RecoveryError("逐秒帧分布在多个目录，拒绝恢复")
            identity = _file_identity(safe)
            with _CACHE_LOCK:
                already_verified = identity in _VALID_JPEG_CACHE
            if not already_verified:
                try:
                    with Image.open(safe) as image:
                        if image.format != "JPEG":
                            raise RecoveryError("抽帧不是 JPEG 图片")
                        image.verify()
                except (OSError, ValueError) as exc:
                    if isinstance(exc, RecoveryError):
                        raise
                    raise RecoveryError("原始抽帧损坏，拒绝恢复") from exc
                if _file_identity(safe) != identity:
                    raise RecoveryError("校验期间原始抽帧发生变化")
                with _CACHE_LOCK:
                    _VALID_JPEG_CACHE.add(identity)
            rel = safe.relative_to(work.resolve(strict=True))
            frame_pairs.append((safe, rel, identity))
            covered.append(int(observation["second"]))
    if covered != list(range(count)) or frame_dir is None:
        raise RecoveryError("实际抽帧没有完整覆盖视频全程")
    children = list(frame_dir.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise RecoveryError("实际抽帧目录包含非普通文件")
    on_disk = sorted(path.name for path in children)
    if on_disk != [f"{second:08d}.jpg" for second in range(count)]:
        raise RecoveryError("实际抽帧目录存在缺失、额外或非预期文件")
    return frame_pairs


def _sources_and_validate_coverage(work: Path, base: str, medium: str, report: dict, coverage_draft: dict, legacy: bool) -> list[tuple[str, str]]:
    if legacy and any((work / name).exists() for name in ("subtitles.srt", "subtitles.json", "hard_subtitles.txt", "visual_ocr.txt")):
        raise RecoveryError("旧任务含未持久化来源类型，无法证明覆盖账本对应的完整来源集合")
    sources = [("base", base), ("medium", medium), ("visual_model_observations", _visual_text(report))]
    doc = coverage_draft.get("doc")
    ledger = coverage_draft.get("coverage")
    if not isinstance(doc, dict) or not isinstance(ledger, list) or not ledger:
        raise RecoveryError("完整 grounding 输入或覆盖账本缺失")
    from app.agent.knowledge.document_contract import validate_document

    if validate_document(doc):
        raise RecoveryError("覆盖草稿未通过文档结构门禁")
    from app.agent.knowledge import grounding
    try:
        primary_sources = grounding._primary_grounding_sources(sources)
    except ValueError as exc:
        raise RecoveryError("恢复覆盖账本没有可用的原始转写来源") from exc

    expected_primary_lines = [
        line.strip()
        for _, text in primary_sources
        for line in text.splitlines()
        if line.strip()
    ]
    expected_legacy_lines = [
        line.strip()
        for _, text in sources
        for line in text.splitlines()
        if line.strip()
    ]
    actual_lines: list[str] = []
    known_point_ids = {
        point.get("id")
        for module in doc.get("modules", []) if isinstance(module, dict)
        for point in module.get("points", []) if isinstance(point, dict)
    }
    for report_row in ledger:
        evidence = report_row.get("evidence") if isinstance(report_row, dict) else None
        dispositions = report_row.get("coverage") if isinstance(report_row, dict) else None
        if not isinstance(evidence, dict) or not evidence or not isinstance(dispositions, list):
            raise RecoveryError("覆盖账本缺少原始证据或逐行处置")
        if list(evidence) != [f"e{i + 1}" for i in range(len(evidence))]:
            raise RecoveryError("覆盖账本证据行号不连续")
        if any(not isinstance(value, str) or not value.strip() for value in evidence.values()):
            raise RecoveryError("覆盖账本包含无效原始证据")
        actual_lines.extend(evidence.values())
        seen: list[str] = []
        for row in dispositions:
            if not isinstance(row, dict) or not isinstance(row.get("evidence_ids"), list):
                raise RecoveryError("覆盖账本处置缺少证据行引用")
            refs = row["evidence_ids"]
            if not refs or any(ref not in evidence for ref in refs):
                raise RecoveryError("覆盖账本处置引用未知证据行")
            action = row.get("action")
            if action not in ("covered", "added", "ignored"):
                raise RecoveryError("覆盖账本包含未完成的证据处置")
            if action == "ignored":
                if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                    raise RecoveryError("忽略证据缺少理由")
            elif (
                not isinstance(row.get("point_ids"), list)
                or not row["point_ids"]
                or any(point_id not in known_point_ids for point_id in row["point_ids"])
            ):
                raise RecoveryError("覆盖证据未关联有效正文知识点")
            seen.extend(refs)
        if sorted(seen) != sorted(evidence):
            raise RecoveryError("覆盖账本没有逐行处置全部证据")
    if actual_lines not in (expected_primary_lines, expected_legacy_lines):
        raise RecoveryError("覆盖账本与完整原始来源集合不完全一致")
    return sources


def _active_task(project_id: int, video_id: int) -> dict | None:
    for task in get_queue().list_all():
        payload = task.get("payload") or {}
        if payload.get("project_id") == project_id and payload.get("video_id") == video_id:
            if not task.get("worker_finished"):
                return task
    return None


def has_active_video_worker(project_id: int, video_id: int) -> bool:
    """Check queue liveness without waiting on the per-video pipeline lock."""
    return _active_task(project_id, video_id) is not None


def _checkpoint_pair(work: Path, root: Path, *, verify_inputs: bool = True, stat_cache: bool = False) -> tuple[dict, dict, bytes]:
    cp_path = _safe_file(work / "checkpoint.json", work, "恢复检查点")
    raw = cp_path.read_bytes()
    try:
        cp = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryError("恢复检查点无法解析") from exc
    if not isinstance(cp, dict) or cp.get("version") not in (1, CHECKPOINT_VERSION):
        raise RecoveryError("恢复检查点版本无效")
    checksum = _safe_file(work / "checkpoint.sha256", work, "检查点哈希回执").read_text(encoding="ascii").strip()
    actual = hashlib.sha256(raw).hexdigest()
    if not re.fullmatch(r"[a-f0-9]{64}", checksum) or checksum != actual:
        raise RecoveryError("恢复检查点哈希不匹配")
    if cp.get("version") == CHECKPOINT_VERSION:
        state_path = _safe_file(work / "recovery-status.json", work, "恢复任务状态")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryError("恢复任务状态无法解析") from exc
        if not isinstance(state, dict) or state.get("checkpoint_sha256") != actual:
            raise RecoveryError("恢复任务状态与检查点回执不一致")
        if state.get("status") not in ("running", "failed", "cancelled", "done") or type(state.get("worker_finished")) is not bool:
            raise RecoveryError("恢复任务状态格式无效")
        # Version 2 checkpoints are immutable; the atomic status file is the
        # sole authority for lifecycle fields.
        cp = dict(cp)
        cp["status"] = state["status"]
        state = dict(state)
    else:
        state_path = _safe_file(work / "recovery-status.json", work, "恢复任务状态")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryError("恢复任务状态无法解析") from exc
        if (not isinstance(state, dict) or state.get("checkpoint_sha256") != actual
                or state.get("status") != cp.get("status")):
            raise RecoveryError("恢复任务状态与检查点回执不一致")
    provenance_path = _safe_file(work / "resume.json", work, "恢复来源回执")
    if hashlib.sha256(provenance_path.read_bytes()).hexdigest() != cp.get("resume_sha256"):
        raise RecoveryError("恢复来源回执哈希不匹配")
    if not isinstance(cp.get("inputs"), list) or not cp["inputs"]:
        raise RecoveryError("恢复检查点缺少输入哈希清单")
    for row in cp["inputs"]:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RecoveryError("恢复输入哈希清单格式无效")
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RecoveryError("恢复输入哈希路径越界")
        if verify_inputs:
            path = _safe_file(work / rel, work, "恢复输入")
            identity = _sha256(path, stat_cache=stat_cache)
            if identity["sha256"] != row.get("sha256") or identity["size"] != row.get("size"):
                raise RecoveryError("恢复输入文件哈希不匹配")
    return cp, state, raw


def _candidate_snapshot(project_id: int, video_id: int, recovery_id: str, *, allow_legacy: bool, latest_id: str | None = None, require_eligible: bool = True, stat_cache: bool = False) -> tuple[RecoverySnapshot, dict]:
    if not RECOVERY_ID_RE.fullmatch(recovery_id):
        raise RecoveryError("恢复标识必须为24位十六进制目录标识")
    root, video, baseline, baseline_hash = _load_project_state(project_id, video_id)
    video_root = root / f"video_{video_id}"
    tasks_root = video_root / "tasks"
    if video_root.is_symlink() or tasks_root.is_symlink() or not _within(tasks_root, root):
        raise RecoveryError("视频任务目录越界或包含符号链接")
    work = tasks_root / recovery_id
    if work.is_symlink() or not work.is_dir() or not _within(work, tasks_root):
        raise RecoveryError("恢复工件目录不存在或不属于当前视频")
    if latest_id and recovery_id != latest_id:
        raise RecoveryError("该任务已被更新的处理任务取代，拒绝恢复过期草稿")
    active = _active_task(project_id, video_id)
    if active:
        raise RecoveryError("该视频仍有未结束 worker，等待任务彻底静止后再恢复")

    is_legacy = not (work / "checkpoint.json").exists()
    if is_legacy and any((work / name).exists() or (work / name).is_symlink() for name in ("checkpoint.json", "checkpoint.sha256", "recovery-status.json", "resume.json", "recovery-unavailable.json")):
        raise RecoveryError("新恢复检查点缺失或损坏，不能降级当成旧版工件")
    source_status = video.status
    source_error = video.error
    source_task_id = None
    intent = "整理视频知识点"
    terms: list[str] = []
    options: dict[str, Any]
    checkpoint = None
    status_record = None
    if is_legacy:
        if not allow_legacy:
            raise RecoveryError("旧版任务没有可验证的恢复检查点")
        if source_status not in ("failed", "cancelled"):
            raise RecoveryError("旧版任务只允许从当前失败或取消状态恢复")
        if not _formal_content_is_empty(baseline):
            raise RecoveryError("旧任务没有可信的原始 baseline；当前项目知识非空，拒绝猜测后合并")
        from app.services.options import ProcessingOptions
        options = ProcessingOptions().model_dump()
    else:
        checkpoint, status_record, _ = _checkpoint_pair(work, work, stat_cache=stat_cache)
        if checkpoint.get("project_id") != project_id or checkpoint.get("video_id") != video_id or checkpoint.get("recovery_id") != recovery_id:
            raise RecoveryError("恢复检查点的项目/视频归属不一致")
        source_task_id = checkpoint.get("task_id")
        source_status = checkpoint.get("status")
        source_error = status_record.get("error")
        if source_status not in ("failed", "cancelled", "running", "done"):
            raise RecoveryError("恢复检查点状态不支持继续")
        if source_status in ("failed", "cancelled") and video.status != source_status:
            raise RecoveryError("视频已有更新的处理状态，该失败记录已过期")
        if source_status == "running" and video.status not in _PIPELINE_STAGES:
            startup_interrupted = (
                video.status == "failed"
                and video.error == _STARTUP_INTERRUPTION_ERROR
                and status_record.get("worker_finished") is False
            )
            if not startup_interrupted:
                raise RecoveryError("中断任务的视频状态已变化，拒绝覆盖当前项目")
            source_error = source_error or video.error
        if source_status != "done" and checkpoint.get("baseline_doc_sha256") != baseline_hash:
            raise RecoveryError("项目知识文档已偏离原恢复 baseline，拒绝覆盖")
        intent = checkpoint.get("intent")
        terms = checkpoint.get("terms")
        options = checkpoint.get("options")
        if not isinstance(intent, str) or not intent.strip() or not isinstance(terms, list) or any(not isinstance(x, str) for x in terms) or not isinstance(options, dict):
            raise RecoveryError("恢复检查点缺少非敏感处理意图、术语或选项")

    if not is_legacy and checkpoint.get("media_identity") is not None:
        manifest, media_identity, _ = _local_media(root, video, stat_cache=stat_cache)
        if checkpoint["media_identity"] != media_identity:
            raise RecoveryError("当前本地媒体身份与原恢复检查点不一致")
    else:
        manifest, media_identity, _ = _local_media(root, video, stat_cache=stat_cache)

    pending_regeneration = bool(
        not is_legacy and checkpoint.get("regenerate_initial_draft") is True
    )
    base_path = _transcript_file(work, "base")
    medium_path = _transcript_file(work, "medium")
    base_raw, medium_raw = base_path.read_bytes(), medium_path.read_bytes()
    try:
        base, medium = base_raw.decode("utf-8"), medium_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecoveryError("base 或 medium 转写不是有效 UTF-8") from exc
    if len(base.strip()) < MIN_TRANSCRIPT_CHARS or len(medium.strip()) < MIN_TRANSCRIPT_CHARS:
        raise RecoveryError("base 或 medium 转写不完整")
    visual_report, visual_raw = _read_json(work / "visual_review.json", work, "逐秒画面报告")
    coverage_name = "source-coverage-draft.json" if pending_regeneration else "coverage-draft.json"
    coverage_path = work / coverage_name
    coverage_draft, coverage_raw = _read_json(coverage_path, work, "完整覆盖草稿")
    frame_files = _validate_visual(root, work, visual_report, base, medium, video.duration or 0, stat_cache=stat_cache)
    # Raw ASR and original-frame observations are the only reusable evidence
    # when the old candidate's generation identity is unknown or mismatched.
    # Do not treat its coverage ledger as authoritative on that path.
    raw_source_pairs = [
        ("base", base),
        ("medium", medium),
        ("visual_model_observations", _visual_text(visual_report)),
    ]
    sources = raw_source_pairs

    if not is_legacy:
        # A checkpointed recovery must also bind the copied source set exactly.
        sources_file, _ = _read_json(work / "sources.json", work, "恢复原始来源快照")
        raw_source_rows = sources_file.get("sources")
        if not isinstance(raw_source_rows, list) or raw_source_rows != [{"name": name, "text": text} for name, text in sources]:
            raise RecoveryError("恢复来源快照与经验证的 ASR/画面报告不一致")
        if status_record and status_record.get("worker_finished") is False:
            live = get_queue().get_status(source_task_id) if source_task_id else None
            if live is not None and not live.get("worker_finished"):
                raise RecoveryError("原恢复 worker 仍未退出")

    receipt_files = []
    for path, raw in (
        (base_path, base_raw), (medium_path, medium_raw),
        (work / "visual_review.json", visual_raw),
        (coverage_path, coverage_raw),
    ):
        receipt_files.append({"path": path.relative_to(work.resolve(strict=True)).as_posix(), **_bytes_receipt(raw)})
    if isinstance(checkpoint, dict):
        bound_paths = {row.get("path") for row in checkpoint.get("inputs", []) if isinstance(row, dict)}
        extraction_name = "source-extraction.json" if pending_regeneration else "extraction.json"
        if extraction_name in bound_paths:
            extraction_path = _safe_file(work / extraction_name, work, "来源初稿抽取工件")
            _, extraction_raw = _read_json(extraction_path, work, "来源初稿抽取工件")
            # This artifact is provenance only. On an unknown/mismatched
            # generation it is deliberately not a reusable pipeline input.
            receipt_files.append({
                "path": extraction_path.relative_to(work.resolve(strict=True)).as_posix(),
                **_bytes_receipt(extraction_raw),
            })
        from app.agent.media.illustration_replay import load_bound_replay_cache, optional_replay_receipt
        optional_cache = optional_replay_receipt(checkpoint.get("optional_inputs"))
        if optional_cache and load_bound_replay_cache(work, optional_cache) is not None:
            receipt_files.append(copy.deepcopy(optional_cache))
        # Capture optional raw-output identities in the same read-only snapshot
        # as the core evidence. These are local artifacts, not provider-signed
        # receipts; they never establish content acceptance.
        for name in ("source-reviewed.json", "illustration_review.json", "activity.json"):
            try:
                _value, optional_raw = _read_json(work / name, work, "可选图例原始工件")
            except (RecoveryError, OSError):
                continue
            receipt_files.append({"path": name, **_bytes_receipt(optional_raw)})
    for path, _, validated_identity in frame_files:
        frame_receipt = _sha256(path, stat_cache=stat_cache)
        if _file_identity(path) != validated_identity:
            raise RecoveryError("抽帧在完整性校验后发生变化，拒绝恢复")
        receipt_files.append({"path": path.relative_to(work.resolve(strict=True)).as_posix(), **frame_receipt})
    receipt_files.append({
        "path": (video_root / "local.json").relative_to(root).as_posix(),
        **manifest.get("_validated_receipt", _sha256(video_root / "local.json", stat_cache=stat_cache)),
    })
    for kind, row in media_identity.items():
        receipt_files.append({"path": row["path"], "sha256": row["sha256"], "size": row["size"], "media_kind": kind})

    from app.services.recovery_versions import (
        compare_candidate_generation,
        current_candidate_generation_intent,
    )

    source_generation = None
    if isinstance(checkpoint, dict):
        source_generation = (
            checkpoint.get("source_candidate_generation_fingerprint")
            if pending_regeneration
            else checkpoint.get("candidate_generation_fingerprint")
        )
    snapshot_for_version = RecoverySnapshot(
        project_id=project_id, video_id=video_id, recovery_id=recovery_id,
        source_work=work, legacy_bootstrap=is_legacy, source_status=source_status,
        source_error=source_error, source_task_id=source_task_id, intent=intent,
        terms=list(terms), options=dict(options), baseline_doc=baseline,
        baseline_doc_sha256=baseline_hash, media_identity=media_identity,
        local_manifest=manifest, base_text=base, medium_text=medium,
        visual_report=visual_report, coverage_draft=coverage_draft, sources=sources,
        frame_files=frame_files, source_file_receipts=receipt_files,
        candidate_generation_fingerprint=source_generation,
        target_candidate_generation_intent=None,
        candidate_generation_status="unknown",
        candidate_generation_reason="候选生成指纹缺失，旧候选不可复用",
        regenerate_initial_draft=True,
    )
    target_generation = current_candidate_generation_intent(snapshot_for_version)
    generation_status, generation_reason = compare_candidate_generation(
        source_generation, target_generation, candidate_artifact=coverage_raw,
    )
    if generation_status == "matched":
        # Reuse requires both a matching generator receipt and a structurally
        # valid full-coverage artifact tied to this exact raw source set.
        validated_sources = _sources_and_validate_coverage(
            work, base, medium, visual_report, coverage_draft, is_legacy,
        )
        if validated_sources != raw_source_pairs:
            raise RecoveryError("匹配候选的来源集合与经验证的原始证据不一致")

    snapshot = RecoverySnapshot(
        project_id=project_id, video_id=video_id, recovery_id=recovery_id, source_work=work,
        legacy_bootstrap=is_legacy, source_status=source_status, source_error=source_error,
        source_task_id=source_task_id, intent=intent, terms=list(terms), options=dict(options),
        baseline_doc=baseline, baseline_doc_sha256=baseline_hash, media_identity=media_identity,
        local_manifest=manifest, base_text=base, medium_text=medium, visual_report=visual_report,
        coverage_draft=coverage_draft, sources=sources, frame_files=frame_files,
        source_file_receipts=receipt_files,
        candidate_generation_fingerprint=source_generation,
        target_candidate_generation_intent=target_generation,
        candidate_generation_status=generation_status,
        candidate_generation_reason=generation_reason,
        regenerate_initial_draft=generation_status != "matched",
    )
    reason = (
        "已校验完整 base/medium、逐秒覆盖、转写时间窗与本地媒体；旧任务没有 persisted checkpoint，"
        "本次将明确建立 legacy bootstrap 回执并从 grounding 继续。"
        if is_legacy else "已校验原始输入哈希和项目 baseline；将从 grounding 继续。"
    )
    if require_eligible and source_status not in ("failed", "cancelled", "running"):
        raise RecoveryError("恢复来源不处于失败、取消或中断状态")
    return snapshot, {"status": source_status, "reason": reason, "checkpoint": checkpoint, "status_record": status_record}


def _receipted_artifact_source(
    snapshot: RecoverySnapshot,
    candidate_names: tuple[str, ...],
    label: str,
    *,
    required: bool,
) -> Path | None:
    """Select a saved artifact only when this snapshot validated its receipt."""
    matches = [
        row for row in snapshot.source_file_receipts
        if isinstance(row, dict) and row.get("path") in candidate_names
    ]
    if len(matches) > 1:
        raise RecoveryError(f"{label}存在多个本次验证哈希，拒绝恢复")
    if not matches:
        if required:
            raise RecoveryError(f"{label}没有对应的初始验证哈希")
        return None
    return snapshot.source_work / matches[0]["path"]


def _candidate_ids(root: Path, video_id: int) -> list[tuple[str, Path, bool]]:
    tasks_root = root / f"video_{video_id}" / "tasks"
    if tasks_root.is_symlink() or not tasks_root.is_dir():
        return []
    rows = []
    for item in tasks_root.iterdir():
        if not RECOVERY_ID_RE.fullmatch(item.name) or item.is_symlink() or not item.is_dir():
            continue
        has_checkpoint = (item / "checkpoint.json").exists()
        if has_checkpoint or (item / "coverage-draft.json").exists():
            rows.append((item.name, item, has_checkpoint))
    rows.sort(key=lambda row: row[1].stat().st_mtime_ns, reverse=True)
    return rows


def _legacy_resume_history(root: Path, project_id: int, video_id: int, legacy_id: str) -> dict | None:
    """Find a validated immutable receipt that records this legacy failure."""
    rows = []
    for recovery_id, work, has_checkpoint in _candidate_ids(root, video_id):
        if not has_checkpoint or recovery_id == legacy_id:
            continue
        try:
            checkpoint, _, _ = _checkpoint_pair(work, work, verify_inputs=False)
            if (checkpoint.get("project_id") != project_id
                    or checkpoint.get("video_id") != video_id
                    or checkpoint.get("recovery_id") != recovery_id):
                continue
            provenance, _ = _read_json(work / "resume.json", work, "恢复来源回执")
            if provenance.get("source_recovery_id") != legacy_id:
                continue
            if provenance.get("project_id") != checkpoint.get("project_id"):
                continue
            if provenance.get("video_id") != video_id:
                continue
            if provenance.get("source_status") not in ("failed", "cancelled"):
                continue
            if provenance.get("source_error") is not None and not isinstance(provenance.get("source_error"), str):
                continue
            rows.append((work.stat().st_mtime_ns, provenance))
        except (RecoveryError, OSError, ValueError, KeyError, TypeError):
            continue
    return max(rows, key=lambda row: row[0])[1] if rows else None


def _checkpoint_history(work: Path, project_id: int, video_id: int, recovery_id: str) -> tuple[dict, dict, str | None]:
    """Read trusted historical fields before freshness/current-video checks."""
    # Historical fields are useful even when an old input is now damaged; only
    # eligibility checks verify every input byte. This never authorizes resume.
    checkpoint, status_record, _ = _checkpoint_pair(work, work, verify_inputs=False)
    if (
        checkpoint.get("project_id") != project_id
        or checkpoint.get("video_id") != video_id
        or checkpoint.get("recovery_id") != recovery_id
    ):
        raise RecoveryError("恢复检查点的项目/视频归属不一致")
    created_at = checkpoint.get("created_at")
    try:
        parsed = datetime.fromisoformat(created_at) if isinstance(created_at, str) else None
        if parsed is None or parsed.tzinfo is None:
            created_at = None
    except ValueError:
        created_at = None
    return checkpoint, status_record, created_at


def _history_lineage(work: Path, checkpoint: dict, project_id: int, video_id: int) -> dict:
    """Expose only recorded identities; a disk record never becomes a live task.

    Parent linkage is presentation provenance, not permission to resume. Read it
    from the already hash-verified receipt and require a same-video parent.
    Invalid/missing links leave the historical failure visible as its own turn.
    """
    recovery_id = checkpoint.get("recovery_id")
    recorded_task = checkpoint.get("task_id")
    task_valid = (
        isinstance(recorded_task, str) and 0 < len(recorded_task) <= 200
        and recorded_task.isprintable()
        and hashlib.sha256(recorded_task.encode()).hexdigest()[:24] == recovery_id
    )
    result = {"source_task_id": recorded_task if task_valid else None,
              "source_recovery_id": None}
    parent = checkpoint.get("source_recovery_id")
    if not isinstance(parent, str) or not RECOVERY_ID_RE.fullmatch(parent) or parent == recovery_id:
        return result
    try:
        provenance, _ = _read_json(work / "resume.json", work, "恢复来源回执")
        if (
            provenance.get("project_id") != project_id
            or provenance.get("video_id") != video_id
            or provenance.get("recovery_id") != recovery_id
            or provenance.get("source_recovery_id") != parent
        ):
            return result
        parent_work = work.parent / parent
        if parent_work.is_symlink() or not parent_work.is_dir():
            return result
        if (parent_work / "checkpoint.json").exists():
            _checkpoint_history(parent_work, project_id, video_id, parent)
        elif not (parent_work / "coverage-draft.json").is_file():
            return result
        result["source_recovery_id"] = parent
    except (RecoveryError, OSError, ValueError, KeyError, TypeError):
        pass
    return result


def discover_project_recoveries(project_id: int) -> list[dict]:
    """Read persisted recovery records without loading settings or credentials."""
    with session_scope() as db:
        project = db.get(Project, project_id)
        if project is None:
            raise RecoveryNotFound("项目不存在")
        root = Path(project.storage_root).expanduser().resolve(strict=True)
        videos = [
            {"id": v.id, "source": v.source, "status": v.status, "error": v.error,
             "duration": v.duration, "project_id": v.project_id}
            for v in db.query(Video).filter_by(project_id=project_id).all()
        ]
    entries = []
    for video in videos:
        if video["source"] != "local":
            continue
        candidates = _candidate_ids(root, video["id"])
        latest_legacy = next((rid for rid, _, has_cp in candidates if not has_cp), None)
        for recovery_id, work, has_checkpoint in candidates:
            # Restoring/copying files changes directory mtime. It cannot prove
            # when an uncheckpointed historical attempt actually started.
            created_at = None
            entry = {
                "task_id": None, "status": video["status"], "progress": 0.0,
                "message": "检测到可核验的持久化恢复工件", "stage": video["status"],
                "error": video["error"], "kind": "pipeline", "draft_available": False,
                "video_id": video["id"], "recovery_id": recovery_id,
                "source_task_id": None, "source_recovery_id": None,
                "candidate_generation_status": "unknown",
                "regenerate_initial_draft": True,
                "created_at": created_at,
                "resumable": False, "recovery_reason": "恢复工件尚未通过校验",
            }
            try:
                if has_checkpoint:
                    checkpoint, status_record, cp_created_at = _checkpoint_history(
                        work, project_id, video["id"], recovery_id
                    )
                    entry.update(_history_lineage(work, checkpoint, project_id, video["id"]))
                    persisted_state = checkpoint.get("status")
                    if persisted_state in ("failed", "cancelled", "done", "running"):
                        live = _active_task(project_id, video["id"])
                        checkpoint_task_id = checkpoint.get("task_id")
                        live_task_id = (live or {}).get("task_id")
                        checkpoint_task_is_live = (
                            isinstance(checkpoint_task_id, str) and bool(checkpoint_task_id)
                            and checkpoint_task_id == live_task_id
                        )
                        history_error = status_record.get("error")
                        if persisted_state == "running" and not checkpoint_task_is_live and not history_error:
                            history_error = _STARTUP_INTERRUPTION_ERROR
                        entry.update(
                            status=(
                                "failed" if persisted_state == "running" and not checkpoint_task_is_live
                                else persisted_state
                            ),
                            error=history_error,
                            created_at=cp_created_at,
                        )
                if not has_checkpoint:
                    history = _legacy_resume_history(root, project_id, video["id"], recovery_id)
                    if history is not None:
                        old_status = history["source_status"]
                        entry.update(
                            status=old_status,
                            error=history.get("source_error"),
                            resumable=False,
                            recovery_reason="该旧版失败记录已保留；恢复尝试已建立新的独立目录。",
                            message="原失败记录（已恢复尝试，不可重复提交）",
                        )
                        entries.append(entry)
                        continue
                if not has_checkpoint and recovery_id != latest_legacy:
                    raise RecoveryError("存在更新的旧版任务工件，此目录无法证明为当前失败尝试")
                snapshot, result = _candidate_snapshot(
                    project_id, video["id"], recovery_id,
                    allow_legacy=not has_checkpoint,
                    latest_id=candidates[0][0] if candidates else None,
                    require_eligible=False,
                    stat_cache=True,
                )
                state = result["status"]
                entry.update(
                    candidate_generation_status=snapshot.candidate_generation_status,
                    regenerate_initial_draft=snapshot.regenerate_initial_draft,
                )
                if state == "done":
                    entry.update(status="done", message="恢复任务已完成", recovery_reason="任务已完成，不能再次恢复",
                                 rebuildable=bool(candidates and candidates[0][0] == recovery_id
                                                  and video["status"] == "done"
                                                  and not _active_task(project_id, video["id"])
                                                  and not snapshot.legacy_bootstrap
                                                  and snapshot.target_candidate_generation_intent))
                else:
                    live = _active_task(project_id, video["id"])
                    video_status_ok = (
                        state in ("failed", "cancelled") and video["status"] == state
                        or state == "running" and video["status"] in _PIPELINE_STAGES
                        or state == "running"
                        and video["status"] == "failed"
                        and video["error"] == _STARTUP_INTERRUPTION_ERROR
                        and result["status_record"].get("worker_finished") is False
                    ) if not snapshot.legacy_bootstrap else video["status"] in ("failed", "cancelled")
                    latest_ok = bool(candidates and candidates[0][0] == recovery_id)
                    empty_doc_ok = not snapshot.legacy_bootstrap or _formal_content_is_empty(snapshot.baseline_doc)
                    generation_compatible = snapshot.candidate_generation_status == "matched"
                    generation_regenerable = snapshot.target_candidate_generation_intent is not None
                    generation_ready = generation_compatible or generation_regenerable
                    eligible = not live and video_status_ok and latest_ok and empty_doc_ok and generation_ready
                    if snapshot.legacy_bootstrap:
                        reason = (
                            "旧任务没有 persisted checkpoint；已校验当前知识文档为空、ASR 时间窗与完整逐秒帧，"
                            "默认意图/术语/选项会记录为 legacy bootstrap。"
                        )
                    else:
                        reason = "检查点输入哈希和当前项目 baseline 已验证，可从完整 grounding 继续。"
                    if live:
                        reason = "该视频仍有未退出的 worker，不能并行恢复。"
                    elif not video_status_ok:
                        reason = "视频当前状态与该失败/中断记录不匹配，工件可能已过期。"
                    elif not latest_ok:
                        reason = "该任务不是当前最新工件，拒绝用旧草稿覆盖项目。"
                    elif not empty_doc_ok:
                        reason = "旧任务没有可信 baseline 且当前正式知识非空，拒绝猜测合并。"
                    elif generation_compatible:
                        reason = "检查点输入哈希、baseline 与候选生成实现/初稿提示均已验证，可复用覆盖候选。"
                    elif generation_regenerable:
                        reason = (
                            f"{snapshot.candidate_generation_reason}；需基于已验证原始证据重新生成候选，"
                            "将保留旧候选为来源历史，并重新生成初稿与完整覆盖账本。"
                        )
                    elif not generation_ready:
                        reason = f"{snapshot.candidate_generation_reason}；当前初稿生成依赖摘要不可用。"
                    entry.update(
                        status=("failed" if state == "running" and not live else state),
                        error=(
                            video["error"]
                            if state == "running" and not live
                            and video["error"] == _STARTUP_INTERRUPTION_ERROR
                            else "上次进程退出前任务未完成"
                            if state == "running" and not live and not result["status_record"].get("error")
                            else entry["error"]
                        ),
                        resumable=eligible,
                        recovery_reason=reason,
                        message=(
                            "上次进程中断，可从来源支持核对继续"
                            if state == "running" and not live and eligible
                            else "可基于已验证原始证据重新生成初稿" if eligible and not generation_compatible
                            else "可从来源支持核对继续" if eligible else "恢复校验未通过"
                        ),
                    )
            except Exception as exc:
                entry["recovery_reason"] = (
                    str(exc)[:400] if isinstance(exc, RecoveryError)
                    else f"恢复工件损坏或格式无效（{type(exc).__name__}）"
                )
            entries.append(entry)
    return entries


def _copy_stable(source: Path, target: Path, expected: dict[str, Any]) -> dict[str, Any]:
    source = Path(source)
    before = _sha256(source)
    if before != {key: expected[key] for key in ("sha256", "size")}:
        raise RecoveryError("复制前来源哈希发生变化，拒绝恢复")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    copied = _sha256(target)
    after = _sha256(source)
    if copied != before or after != before:
        raise RecoveryError("复制期间来源或副本哈希变化，拒绝恢复")
    return copied


def _write_checkpoint(work: Path, checkpoint: dict, *, worker_finished: bool = False, error: str | None = None) -> None:
    checkpoint = dict(checkpoint)
    checkpoint["version"] = CHECKPOINT_VERSION
    raw = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    pending = work / ".checkpoint.json.tmp"
    pending.write_bytes(raw)
    pending.replace(work / "checkpoint.json")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = work / ".checkpoint.sha256.tmp"
    sidecar.write_text(digest + "\n", encoding="ascii")
    sidecar.replace(work / "checkpoint.sha256")
    state = {
        "status": checkpoint["status"], "worker_finished": bool(worker_finished),
        "checkpoint_sha256": digest, "error": error,
    }
    _atomic_status_replace(work, state)


def _atomic_status_replace(work: Path, state: dict) -> None:
    """Commit lifecycle metadata as one atomic replacement beside immutable inputs."""
    state_path = Path(work) / "recovery-status.json"
    pending = Path(work) / ".recovery-status.json.tmp"
    pending.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    pending.replace(state_path)


def persist_recovery_status(work: Path, status: str, *, worker_finished: bool = False, error: str | None = None) -> None:
    """Persist task terminal state; corruption remains fail-closed."""
    if status not in ("running", "failed", "cancelled", "done"):
        raise ValueError("无效的恢复任务状态")
    cp, state, raw = _checkpoint_pair(Path(work), Path(work))
    if cp.get("version") == CHECKPOINT_VERSION:
        state.update(
            status=status,
            worker_finished=bool(worker_finished),
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            error=error,
        )
        _atomic_status_replace(Path(work), state)
    else:
        # Compatibility with pre-v2 development checkpoints; migrate through a
        # fresh atomic receipt after validating all bound inputs.
        cp["status"] = status
        _write_checkpoint(Path(work), cp, worker_finished=worker_finished, error=error)


def _write_json(path: Path, data: Any) -> bytes:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    path.write_bytes(raw)
    return raw


def mark_recovery_unavailable(work: Path, reason: str) -> None:
    """Record an explicit boundary instead of guessing at unsupported inputs."""
    _write_json(Path(work) / "recovery-unavailable.json", {
        "reason": str(reason)[:400],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


def initialize_local_checkpoint(
    project_id: int,
    video_id: int,
    task_id: str,
    work: Path,
    *,
    intent: str,
    terms: list[str],
    options: dict,
    baseline_doc: dict,
    sources: list[tuple[str, str]],
    visual_report: dict,
    candidate_generation_fingerprint: dict | None = None,
) -> bool:
    """Bind complete normal local-pipeline inputs before entering grounding.

    The working task directory already owns its ASR, frame, visual and coverage
    artifacts, so this records/hashes those files in place. It never rewrites
    them and keeps local media references bound by path and SHA-256.
    """
    work = Path(work)
    expected_id = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:24]
    try:
        if work.name != expected_id:
            raise RecoveryError("本地任务目录与任务标识不一致")
        if [name for name, _ in sources] != ["base", "medium", "visual_model_observations"]:
            raise RecoveryError("当前恢复仅支持 base/medium/完整音画观察三类来源")
        root, video, _current_baseline, current_baseline_hash = _load_project_state(project_id, video_id)
        if document_sha256(baseline_doc) != current_baseline_hash:
            raise RecoveryError("整理期间项目知识文档已变化，不能建立旧 baseline 的恢复检查点")
        video_root = root / f"video_{video_id}"
        manifest_path = _safe_file(video_root / "local.json", video_root, "本地媒体清单")
        manifest, media_identity, _ = _local_media(root, video)
        manifest_data = manifest["_validated_receipt"]
        base_path = _transcript_file(work, "base")
        medium_path = _transcript_file(work, "medium")
        base = base_path.read_text(encoding="utf-8")
        medium = medium_path.read_text(encoding="utf-8")
        coverage, coverage_raw = _read_json(work / "coverage-draft.json", work, "完整覆盖草稿")
        from app.services.recovery_versions import (
            bind_candidate_artifact,
            has_actual_initial_coverage_prompt,
        )
        if candidate_generation_fingerprint is None:
            bound_candidate_generation = None
        else:
            if not has_actual_initial_coverage_prompt(candidate_generation_fingerprint):
                raise RecoveryError("初稿与实际初始覆盖提示摘要格式无效")
            bound_candidate_generation = bind_candidate_artifact(
                candidate_generation_fingerprint,
                coverage_raw,
            )
        extraction_path = work / "extraction.json"
        extraction_raw = None
        if extraction_path.exists() or extraction_path.is_symlink():
            extraction, extraction_raw = _read_json(extraction_path, work, "初稿抽取工件")
            from app.agent.knowledge.document_contract import validate_document
            if validate_document(extraction):
                raise RecoveryError("初稿抽取工件未通过文档结构校验")
        elif candidate_generation_fingerprint is not None:
            raise RecoveryError("带候选指纹的检查点缺少原始初稿抽取工件")
        visual_path = work / "visual_review.json"
        if visual_path.exists() or visual_path.is_symlink():
            visual, _ = _read_json(visual_path, work, "逐秒画面报告")
            if visual != visual_report:
                raise RecoveryError("流水线内存中的逐秒报告与磁盘证据不一致")
        else:
            _write_json(visual_path, visual_report)
            visual = visual_report
        _validate_visual(root, work, visual, base, medium, video.duration or 0)
        _sources_and_validate_coverage(work, base, medium, visual, coverage, False)
        if sources != [("base", base), ("medium", medium), ("visual_model_observations", _visual_text(visual))]:
            raise RecoveryError("流水线来源与可恢复输入不完全一致")
        if manifest_data != _sha256(video_root / "local.json"):
            raise RecoveryError("建立恢复检查点时本地媒体清单发生变化")

        source_raw = _write_json(work / "sources.json", {
            "sources": [{"name": name, "text": text} for name, text in sources],
        })
        media_copy = work / "recovery-input" / "local.json"
        media_copy.parent.mkdir(parents=True, exist_ok=True)
        _copy_stable(manifest_path, media_copy, manifest_data)
        input_paths = [
            base_path, medium_path, work / "visual_review.json",
            work / "coverage-draft.json", work / "sources.json", media_copy,
        ]
        if extraction_raw is not None:
            input_paths.append(extraction_path)
        frame_dir = None
        for batch in visual["batches"]:
            for frame_path in batch["paths"]:
                source = Path(frame_path)
                if not source.is_absolute():
                    source = work / source
                safe = _safe_file(source, work, "原始抽帧")
                if frame_dir is None:
                    frame_dir = safe.parent
                elif safe.parent != frame_dir:
                    raise RecoveryError("逐秒帧分布在多个目录，拒绝建立检查点")
                input_paths.append(safe)
        inputs = [
            {"path": path.relative_to(work.resolve(strict=True)).as_posix(), **_sha256(path)}
            for path in input_paths
        ]
        recovery_id = expected_id
        created_at = datetime.now(timezone.utc).isoformat()
        provenance = {
            "version": 1,
            "created_at": created_at,
            "project_id": project_id,
            "video_id": video_id,
            "recovery_id": recovery_id,
            "source_recovery_id": None,
            "source_task_id": task_id,
            "source_status": "fresh_local",
            "source_error": None,
            "legacy_bootstrap": False,
            "validation_note": "正常本地任务在完整 ASR/逐秒画面/覆盖草稿后建立原始来源检查点。",
            "intent": intent,
            "terms": list(terms),
            "options": options,
            "baseline_doc_sha256": current_baseline_hash,
            "media_identity": media_identity,
            "candidate_generation_fingerprint": bound_candidate_generation,
            "candidate_generation_status_at_checkpoint": (
                "captured" if bound_candidate_generation is not None else "unknown"
            ),
            "source_files": [
                {"path": row["path"], "sha256": row["sha256"], "size": row["size"], "media_kind": kind}
                for kind, row in media_identity.items()
            ],
            "copied_inputs": inputs,
            "resumed_from_stage": None,
        }
        resume_raw = _write_json(work / "resume.json", provenance)
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "task_id": task_id,
            "project_id": project_id,
            "video_id": video_id,
            "recovery_id": recovery_id,
            "source_recovery_id": None,
            "checkpoint_kind": "fresh_local",
            "status": "running",
            "created_at": created_at,
            "legacy_bootstrap": False,
            "intent": intent,
            "terms": list(terms),
            "options": options,
            "baseline_doc_sha256": current_baseline_hash,
            "media_identity": media_identity,
            "candidate_generation_fingerprint": bound_candidate_generation,
            "candidate_generation_status_at_checkpoint": (
                "captured" if bound_candidate_generation is not None else "unknown"
            ),
            "resume_sha256": hashlib.sha256(resume_raw).hexdigest(),
            "inputs": inputs,
        }
        _write_checkpoint(work, checkpoint, worker_finished=False)
        return True
    except Exception as exc:
        try:
            mark_recovery_unavailable(work, str(exc))
        except OSError:
            pass
        return False


def record_regenerated_candidate(
    work: Path,
    *,
    generation_intent: dict,
) -> dict:
    """Bind a newly generated initial draft only after complete coverage succeeds."""
    from app.services.recovery_versions import (
        bind_candidate_artifact,
        candidate_generation_prefix_matches,
        has_actual_initial_coverage_prompt,
    )

    work = Path(work)
    checkpoint, _state, _raw = _checkpoint_pair(work, work)
    if checkpoint.get("status") != "running" or checkpoint.get("regenerate_initial_draft") is not True:
        raise RecoveryError("恢复检查点不处于待重新生成初稿状态")
    expected = checkpoint.get("target_candidate_generation_intent")
    if (not has_actual_initial_coverage_prompt(generation_intent)
            or not candidate_generation_prefix_matches(expected, generation_intent)):
        raise RecoveryError("本次实际初稿生成依赖与排队时目标不一致，候选未绑定")

    extraction_path = _safe_file(work / "extraction.json", work, "新生成初稿")
    coverage_path = _safe_file(work / "coverage-draft.json", work, "新生成完整覆盖草稿")
    extraction, _ = _read_json(extraction_path, work, "新生成初稿")
    coverage, coverage_raw = _read_json(coverage_path, work, "新生成完整覆盖草稿")
    from app.agent.knowledge.document_contract import validate_document
    if validate_document(extraction) or validate_document(coverage.get("doc")):
        raise RecoveryError("新生成初稿或完整覆盖候选未通过文档结构门禁")
    base = _safe_file(work / "base.txt", work, "恢复 base 转写").read_text(encoding="utf-8")
    medium = _safe_file(work / "medium.txt", work, "恢复 medium 转写").read_text(encoding="utf-8")
    visual, _ = _read_json(work / "visual_review.json", work, "恢复逐秒画面报告")
    _sources_and_validate_coverage(work, base, medium, visual, coverage, False)

    bound = bind_candidate_artifact(generation_intent, coverage_raw)
    old_inputs = list(checkpoint.get("inputs", []))
    paths = {row.get("path") for row in old_inputs if isinstance(row, dict)}
    if "extraction.json" in paths or "coverage-draft.json" in paths:
        raise RecoveryError("待重生成检查点意外包含当前候选输入，拒绝覆盖来源")
    additions = [
        {"path": "extraction.json", **_sha256(extraction_path)},
        {"path": "coverage-draft.json", **_sha256(coverage_path)},
    ]
    provenance, _ = _read_json(work / "resume.json", work, "恢复来源回执")
    provenance.update({
        "candidate_generation_fingerprint": bound,
        "candidate_generation_status_at_checkpoint": "captured_after_regeneration",
        "regenerate_initial_draft": False,
        "regenerated_candidate_artifact_path": "coverage-draft.json",
        "regeneration_completed_at": datetime.now(timezone.utc).isoformat(),
        "resumed_from_stage": "grounding",
        "reused_evidence": ["base", "medium", "full_1fps_joint_review"],
    })
    provenance_raw = json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    resume_tmp = work / ".resume.json.tmp"
    resume_tmp.write_bytes(provenance_raw)
    resume_tmp.replace(work / "resume.json")

    checkpoint.update({
        "candidate_generation_fingerprint": bound,
        "candidate_generation_status_at_checkpoint": "captured_after_regeneration",
        "regenerate_initial_draft": False,
        "regenerated_candidate_artifact_path": "coverage-draft.json",
        "inputs": old_inputs + additions,
        "resume_sha256": hashlib.sha256(provenance_raw).hexdigest(),
    })
    _write_checkpoint(work, checkpoint, worker_finished=False)
    return bound


def prepare_recovery_attempt(project_id: int, video_id: int, recovery_id: str, task_id: str,
                             *, rebuild_completed: bool = False) -> tuple[Path, dict]:
    """Revalidate, copy, and hash inputs into a new immutable task directory."""
    entries = discover_project_recoveries(project_id)
    match = next((row for row in entries if row["video_id"] == video_id and row["recovery_id"] == recovery_id), None)
    if match is None:
        raise RecoveryNotFound("没有找到该视频的恢复工件")
    if rebuild_completed and not match.get("rebuildable"):
        raise RecoveryError("只允许从当前已完成且输入完整的任务复用素材重新整理")
    if not rebuild_completed and not match["resumable"]:
        raise RecoveryError(match["recovery_reason"])

    latest_id = recovery_id if rebuild_completed else next((row["recovery_id"] for row in entries if row["video_id"] == video_id and row["resumable"]), None)
    snapshot, _ = _candidate_snapshot(
        project_id, video_id, recovery_id,
        allow_legacy=not rebuild_completed, latest_id=latest_id, require_eligible=not rebuild_completed,
    )
    if rebuild_completed:
        source_ids = _candidate_ids(snapshot.source_work.parents[2], video_id)
        _, current_video, _, _ = _load_project_state(project_id, video_id)
        if not source_ids or source_ids[0][0] != recovery_id or current_video.status != "done":
            raise RecoveryError("完成任务已被新任务取代，拒绝使用过期素材入口")
        if snapshot.source_status != "done" or snapshot.legacy_bootstrap:
            raise RecoveryError("重新整理来源必须为已完成检查点")
        snapshot.regenerate_initial_draft = True
        snapshot.options = {**snapshot.options, "max_model_attempts": 60}
        from app.services.recovery_versions import current_candidate_generation_intent
        snapshot.target_candidate_generation_intent = current_candidate_generation_intent(snapshot)
    root, video, baseline, baseline_hash = _load_project_state(project_id, video_id)
    if baseline_hash != snapshot.baseline_doc_sha256:
        raise RecoveryError("项目知识文档在恢复准备期间发生变化")
    video_root = root / f"video_{video_id}"
    recovery_id_new = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:24]
    new_work = video_root / "tasks" / recovery_id_new
    if new_work.exists() or new_work.is_symlink():
        raise RecoveryError("新恢复任务目录已存在")
    new_work.mkdir(parents=True, exist_ok=False)
    try:
        files: list[dict[str, Any]] = []
        optional_files: list[dict[str, Any]] = []
        receipt_by_source_path = {row["path"]: row for row in snapshot.source_file_receipts}
        regenerate = snapshot.regenerate_initial_draft

        def expected_source_receipt(source: Path) -> dict[str, Any]:
            source = Path(source).resolve(strict=True)
            if _within(source, snapshot.source_work):
                key = source.relative_to(snapshot.source_work.resolve(strict=True)).as_posix()
            else:
                key = source.relative_to(root.resolve(strict=True)).as_posix()
            receipt = receipt_by_source_path.get(key)
            if receipt is None:
                raise RecoveryError("恢复来源没有对应的初始验证哈希")
            return {"sha256": receipt["sha256"], "size": receipt["size"]}

        def copy_input(source: Path, relative: Path, expected: dict[str, Any] | None = None) -> Path:
            target = new_work / relative
            info = expected or expected_source_receipt(source)
            copied = _copy_stable(source, target, info)
            files.append({"path": relative.as_posix(), **copied})
            return target

        for source_kind, filename in (("base", "base.txt"), ("medium", "medium.txt")):
            original = (
                _transcript_file(snapshot.source_work, source_kind)
            )
            target = copy_input(original, Path(filename), expected_source_receipt(original))
        source_coverage_name = "source-coverage-draft.json" if regenerate else "coverage-draft.json"
        original_coverage = _receipted_artifact_source(
            snapshot,
            ("coverage-draft.json", "source-coverage-draft.json"),
            "覆盖候选", required=True,
        )
        assert original_coverage is not None
        copy_input(
            original_coverage, Path(source_coverage_name),
            expected_source_receipt(original_coverage),
        )
        if not regenerate:
            snapshot.coverage_draft = json.loads(
                (new_work / "coverage-draft.json").read_text(encoding="utf-8")
            )
        extraction_source = _receipted_artifact_source(
            snapshot,
            ("extraction.json", "source-extraction.json"),
            "抽取候选", required=False,
        )
        if extraction_source is not None:
            extraction_target_name = "source-extraction.json" if regenerate else "extraction.json"
            copy_input(extraction_source, Path(extraction_target_name), expected_source_receipt(extraction_source))
        original_visual = snapshot.source_work / "visual_review.json"
        expected_visual = expected_source_receipt(original_visual)
        if _sha256(original_visual) != expected_visual:
            raise RecoveryError("逐秒画面报告在验证后发生变化，拒绝恢复")
        visual = copy.deepcopy(snapshot.visual_report)
        for batch in visual["batches"]:
            updated = []
            for raw_path in batch["paths"]:
                original = Path(raw_path)
                if not original.is_absolute():
                    original = snapshot.source_work / original
                original = original.resolve(strict=True)
                relative = original.relative_to(snapshot.source_work.resolve(strict=True))
                expected = expected_source_receipt(original)
                copy_input(original, relative, expected)
                updated.append(str(new_work / relative))
            batch["paths"] = updated
        visual_path = new_work / "visual_review.json"
        visual_raw = _write_json(visual_path, visual)
        files.append({"path": "visual_review.json", "sha256": hashlib.sha256(visual_raw).hexdigest(), "size": len(visual_raw)})

        sources_payload = {"sources": [{"name": name, "text": text} for name, text in snapshot.sources]}
        source_raw = _write_json(new_work / "sources.json", sources_payload)
        files.append({"path": "sources.json", "sha256": hashlib.sha256(source_raw).hexdigest(), "size": len(source_raw)})

        manifest_source = video_root / "local.json"
        manifest_data = expected_source_receipt(manifest_source)
        copy_input(manifest_source, Path("recovery-input/local.json"), manifest_data)
        _, current_media_identity, media_paths = _local_media(root, video)
        if current_media_identity != snapshot.media_identity:
            raise RecoveryError("本地媒体身份在恢复准备期间发生变化")
        for kind, source in media_paths.items():
            identity = snapshot.media_identity[kind]
            suffix = source.suffix.lower() or ".bin"
            copy_input(source, Path("recovery-input") / f"source-{kind}{suffix}", identity)

        # Optional raw model decisions are additional hash-bound inputs. They
        # never replace a candidate or any illustration application gate.
        replay_count = 0
        report_path = snapshot.source_work / "illustration_review.json"
        reviewed_path = snapshot.source_work / "source-reviewed.json"
        activity_path = snapshot.source_work / "activity.json"
        if all(path.is_file() for path in (report_path, reviewed_path, activity_path)):
            from app.agent.media.illustration_replay import build_recovery_replay
            def optional_output(path, label):
                try:
                    expected = expected_source_receipt(path)
                    value, raw = _read_json(path, snapshot.source_work, label)
                    return value if _bytes_receipt(raw) == expected else None
                except (RecoveryError, OSError, ValueError):
                    return None

            source_doc = optional_output(reviewed_path, "图例输入")
            if source_doc == snapshot.coverage_draft.get("doc"):
                review_report = optional_output(report_path, "图例决策") or {}
                activity = optional_output(activity_path, "模型调用记录") or {}
                inherited = None
                cached_path = snapshot.source_work / "illustration-replay-cache.json"
                inherited_receipt = next((r for r in snapshot.source_file_receipts
                                          if r["path"] == cached_path.name), None)
                if inherited_receipt:
                    from app.agent.media.illustration_replay import load_bound_replay_cache
                    inherited = load_bound_replay_cache(snapshot.source_work, inherited_receipt)
                try:
                    replay = build_recovery_replay(
                        source_doc, review_report,
                        code_components=(snapshot.candidate_generation_fingerprint or {}).get("code_components", {}),
                        sources=snapshot.sources, visual_report=snapshot.visual_report,
                        media_identity=snapshot.media_identity, inputs=files,
                        activity=activity, inherited=inherited,
                    )
                except (ValueError, TypeError, KeyError, AttributeError):
                    # Malformed optional model output is a cache miss, never
                    # a reason to bypass or block the verified core inputs.
                    replay = None
                if replay:
                    replay["source_recovery_id"] = recovery_id
                    raw = _write_json(new_work / "illustration-replay-cache.json", replay)
                    optional_files.append({"path": "illustration-replay-cache.json", **_bytes_receipt(raw)})
                    replay_count = len(replay["entries"])

        original_files = list(snapshot.source_file_receipts)
        source_generation = snapshot.candidate_generation_fingerprint
        candidate_generation = None if regenerate else source_generation
        target_generation = snapshot.target_candidate_generation_intent
        provenance = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": project_id,
            "video_id": video_id,
            "recovery_id": recovery_id_new,
            "source_recovery_id": recovery_id,
            "source_task_id": snapshot.source_task_id,
            "source_status": snapshot.source_status,
            "operation": "rebuild_completed" if rebuild_completed else "resume",
            "source_error": snapshot.source_error,
            "legacy_bootstrap": snapshot.legacy_bootstrap,
            "validation_note": (
                "Legacy bootstrap: no persisted checkpoint existed; original intent, terms, options, source hashes, and baseline proof were unavailable. "
                "This attempt records hashes at recovery time, defaults to merge, and requires an empty formal project document."
                if snapshot.legacy_bootstrap else
                "The saved recovery checkpoint, input hashes, processing intent/options/terms, media identity, and baseline were validated."
            ),
            "bootstrap_assumptions": ({
                "intent": "整理视频知识点", "terms": [], "options": snapshot.options,
                "baseline_doc_sha256": snapshot.baseline_doc_sha256,
                "formal_project_content_was_empty": True,
            } if snapshot.legacy_bootstrap else {}),
            "intent": snapshot.intent,
            "terms": snapshot.terms,
            "options": snapshot.options,
            "baseline_doc_sha256": snapshot.baseline_doc_sha256,
            "media_identity": snapshot.media_identity,
            "source_files": original_files,
            "copied_inputs": copy.deepcopy(files),
            "optional_inputs": copy.deepcopy(optional_files),
            "source_candidate_generation_fingerprint": source_generation,
            "target_candidate_generation_intent": target_generation,
            "candidate_generation_fingerprint": candidate_generation,
            "candidate_generation_status_at_checkpoint": "pending" if regenerate else "captured",
            "regenerate_initial_draft": regenerate,
            "source_candidate_artifact_path": source_coverage_name,
            "resumed_from_stage": "initial_draft" if regenerate else "grounding",
            "reused_evidence": ["base", "medium", "full_1fps_joint_review"] + ([] if regenerate else ["coverage-draft"]),
            "replayable_illustration_decisions": replay_count,
            "new_stages": (["initial_draft", "initial_coverage"] if regenerate else []) + ["grounding", "coverage_reconciliation", "illustrations", "independent_review", "author_adjudication", "render", "guarded_commit"],
        }
        resume_raw = _write_json(new_work / "resume.json", provenance)
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "task_id": task_id,
            "project_id": project_id,
            "video_id": video_id,
            "recovery_id": recovery_id_new,
            "source_recovery_id": recovery_id,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "legacy_bootstrap": snapshot.legacy_bootstrap,
            "intent": snapshot.intent,
            "terms": snapshot.terms,
            "options": snapshot.options,
            "baseline_doc_sha256": snapshot.baseline_doc_sha256,
            "media_identity": snapshot.media_identity,
            "source_candidate_generation_fingerprint": source_generation,
            "target_candidate_generation_intent": target_generation,
            "candidate_generation_fingerprint": candidate_generation,
            "candidate_generation_status_at_checkpoint": "pending" if regenerate else "captured",
            "regenerate_initial_draft": regenerate,
            "resume_sha256": hashlib.sha256(resume_raw).hexdigest(),
            "inputs": files,
            "optional_inputs": optional_files,
        }
        _write_checkpoint(new_work, checkpoint, worker_finished=False)
        with session_scope() as db:
            current = db.get(Video, video_id)
            if current is None or current.project_id != project_id:
                raise RecoveryNotFound("视频不属于当前项目")
            current.status = "pending"
            current.error = None
        return new_work, {
            "intent": snapshot.intent, "terms": snapshot.terms, "options": snapshot.options,
            "baseline_doc": baseline, "baseline_doc_sha256": baseline_hash,
            "video": {"video_id": video_id, "bvid": "", "cid": ""},
            "recovery_id": recovery_id_new,
        }
    except BaseException:
        # Only remove the newly-created attempt directory; the source failure is immutable.
        shutil.rmtree(new_work, ignore_errors=True)
        raise


def load_recovery_inputs(work: Path, project_id: int, video_id: int) -> dict:
    """Verify the new checkpoint again immediately before model work begins."""
    work = Path(work)
    cp, _, _ = _checkpoint_pair(work, work)
    if cp.get("project_id") != project_id or cp.get("video_id") != video_id or cp.get("status") != "running":
        raise RecoveryError("新恢复任务检查点状态或归属发生变化")
    regenerate = cp.get("regenerate_initial_draft") is True
    root, video, baseline, baseline_hash = _load_project_state(project_id, video_id)
    if baseline_hash != cp.get("baseline_doc_sha256"):
        raise RecoveryError("项目知识文档已变化，恢复任务停止且不会提交")
    manifest, identities, _ = _local_media(root, video)
    if identities != cp.get("media_identity"):
        raise RecoveryError("本地媒体在恢复排队期间发生变化")
    from app.services.options import ProcessingOptions

    base = _safe_file(work / "base.txt", work, "恢复 base 转写").read_text(encoding="utf-8")
    medium = _safe_file(work / "medium.txt", work, "恢复 medium 转写").read_text(encoding="utf-8")
    visual, _ = _read_json(work / "visual_review.json", work, "恢复逐秒画面报告")
    coverage_name = "source-coverage-draft.json" if regenerate else "coverage-draft.json"
    coverage, coverage_raw = _read_json(work / coverage_name, work, "恢复来源覆盖草稿" if regenerate else "恢复覆盖草稿")
    sources_json, _ = _read_json(work / "sources.json", work, "恢复来源快照")
    sources = [(row["name"], row["text"]) for row in sources_json["sources"]]
    _validate_visual(root, work, visual, base, medium, video.duration or 0)
    raw_source_pairs = [("base", base), ("medium", medium), ("visual_model_observations", _visual_text(visual))]
    if sources != raw_source_pairs:
        raise RecoveryError("恢复来源快照内容与逐秒画面报告不一致")
    from types import SimpleNamespace
    from app.services.recovery_versions import (
        compare_candidate_generation,
        candidate_generation_prefix_matches,
        current_candidate_generation_intent,
    )
    source_generation = (
        cp.get("source_candidate_generation_fingerprint") if regenerate
        else cp.get("candidate_generation_fingerprint")
    )
    target_generation = current_candidate_generation_intent(SimpleNamespace(
        intent=cp.get("intent"), terms=cp.get("terms"), options=cp.get("options"),
        baseline_doc=baseline, sources=sources,
    ))
    expected_target = cp.get("target_candidate_generation_intent")
    if not candidate_generation_prefix_matches(expected_target, target_generation):
        raise RecoveryError("恢复排队后的初稿实现或提示依赖发生变化，任务停止并保留已校验原始输入")
    if regenerate:
        if source_generation is not None:
            source_candidate_path = work / "source-coverage-draft.json"
            if not _safe_file(source_candidate_path, work, "来源旧覆盖草稿").is_file():
                raise RecoveryError("待重生成恢复缺少单独保留的来源候选")
            # A self-invalid old candidate still forces regeneration; its bytes
            # remain available only as referenced history, never as a pipeline input.
            compare_candidate_generation(
                source_generation, target_generation, candidate_artifact=coverage_raw,
            )
        coverage_for_pipeline = None
    else:
        validated_sources = _sources_and_validate_coverage(
            work, base, medium, visual, coverage, False,
        )
        if validated_sources != raw_source_pairs:
            raise RecoveryError("匹配候选的来源集合与经验证的原始证据不一致")
        generation_status, generation_reason = compare_candidate_generation(
            source_generation, target_generation,
            candidate_artifact=coverage_raw,
        )
        if generation_status != "matched":
            raise RecoveryError(
                f"恢复初稿生成依赖在模型工作前未匹配（{generation_status}）：{generation_reason}"
            )
        coverage_for_pipeline = coverage
    input_paths = {row["path"] for row in cp["inputs"]}
    video_suffix = Path(cp["media_identity"]["video"]["path"]).suffix.lower() or ".bin"
    audio_suffix = Path(cp["media_identity"]["audio"]["path"]).suffix.lower() or ".bin"
    video_copy = Path("recovery-input") / f"source-video{video_suffix}"
    audio_copy = Path("recovery-input") / f"source-audio{audio_suffix}"
    if video_copy.as_posix() not in input_paths or audio_copy.as_posix() not in input_paths:
        raise RecoveryError("恢复检查点缺少本地媒体副本")
    options = ProcessingOptions(**cp["options"]).model_dump()
    if "max_model_attempts" in cp["options"]:
        limit = cp["options"]["max_model_attempts"]
        if type(limit) is not int or not 1 <= limit <= 200:
            raise RecoveryError("恢复任务模型预算无效")
        options["max_model_attempts"] = limit
    return {
        "coverage_draft": coverage_for_pipeline,
        "source_coverage_draft": coverage if regenerate else None,
        "source_candidate_generation_fingerprint": cp.get("source_candidate_generation_fingerprint"),
        "target_candidate_generation_intent": target_generation,
        "regenerate_initial_draft": regenerate,
        "visual_report": visual,
        "sources": sources,
        "baseline_doc": baseline,
        "baseline_doc_sha256": baseline_hash,
        "intent": cp["intent"],
        "terms": cp["terms"],
        "options": options,
        "video_path": _safe_file(work / video_copy, work, "恢复视频副本"),
        "audio_path": _safe_file(work / audio_copy, work, "恢复音频副本"),
    }
