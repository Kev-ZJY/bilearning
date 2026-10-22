"""Cancellable, local-only DASH muxing into a verified browser MP4."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from app.worker.evidence import run_command
from app.worker.framer import FFMPEG, FFPROBE


def inspect_media(path, checkpoint=lambda: None):
    # Force the MP4 demuxer and local protocols: media files cannot turn this
    # operation into an HLS/remote fetch or arbitrary playlist traversal.
    raw = run_command([FFPROBE, "-v", "error", "-protocol_whitelist", "file,pipe",
                       "-f", "mov", "-show_streams", "-show_format", "-of", "json", str(path)], checkpoint)
    report = json.loads(raw)
    if not report.get("streams"):
        raise ValueError("媒体没有有效音画轨道")
    return report


def _stream(report, kind):
    rows = [s for s in report.get("streams", []) if s.get("codec_type") == kind]
    if len(rows) != 1:
        raise ValueError(f"媒体缺少唯一的{kind}轨道")
    duration = float(rows[0].get("duration") or report.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("媒体轨道没有有效时长")
    return rows[0], duration


def verify_pair(video_report, audio_report, expected_duration=0):
    video, vduration = _stream(video_report, "video")
    audio, aduration = _stream(audio_report, "audio")
    if abs(vduration - aduration) > max(0.5, vduration * 0.01):
        raise ValueError("音频与视频时长不匹配，未合并")
    if expected_duration and abs(vduration - float(expected_duration)) > max(0.5, float(expected_duration) * 0.01):
        raise ValueError("音画时长与登记分P 不匹配，未合并")
    return video, audio, vduration


def verify_playable(report, expected_duration=0):
    video, audio, duration = verify_pair(report, report, expected_duration)
    if video.get("codec_name") != "h264" or video.get("pix_fmt") != "yuv420p" or audio.get("codec_name") != "aac":
        raise ValueError("缓存不是 H264/AAC 播放格式")
    return duration


def fingerprint(path, checkpoint=lambda: None):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            checkpoint()
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("合并期间源文件已变化，未发布缓存")
    return {"sha256": digest.hexdigest(), "size": after.st_size, "mtime_ns": after.st_mtime_ns}


def mux_streams(audio_path, video_path, output_path, *, expected_duration=0, checkpoint=lambda: None):
    audio_path, video_path, output_path = map(Path, (audio_path, video_path, output_path))
    checkpoint()
    video_report = inspect_media(video_path, checkpoint)
    audio_report = inspect_media(audio_path, checkpoint)
    video, audio, duration = verify_pair(video_report, audio_report, expected_duration)
    inputs = {"video": fingerprint(video_path, checkpoint), "audio": fingerprint(audio_path, checkpoint)}
    fd, name = tempfile.mkstemp(prefix=".mux-", suffix=".part.mp4", dir=output_path.parent)
    os.close(fd)
    pending = Path(name)
    try:
        video_codec = ["-c:v", "copy"] if video.get("codec_name") == "h264" and video.get("pix_fmt") == "yuv420p" else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        audio_codec = ["-c:a", "copy"] if audio.get("codec_name") == "aac" else ["-c:a", "aac", "-b:a", "128k"]
        run_command([FFMPEG, "-y", "-v", "error", "-nostdin",
                     "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", str(video_path),
                     "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", str(audio_path),
                     "-map", "0:v:0", "-map", "1:a:0", *video_codec, *audio_codec,
                     "-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4", str(pending)], checkpoint)
        report = inspect_media(pending, checkpoint)
        verify_playable(report, duration)
        for kind, path in (("video", video_path), ("audio", audio_path)):
            current = path.stat()
            if current.st_size != inputs[kind]["size"] or current.st_mtime_ns != inputs[kind]["mtime_ns"]:
                raise ValueError("合并期间源文件已变化，未发布缓存")
        output = fingerprint(pending, checkpoint)
        checkpoint()
        os.replace(pending, output_path)
        return {"inputs": inputs, "file_identity": output, "media": report}
    finally:
        pending.unlink(missing_ok=True)
