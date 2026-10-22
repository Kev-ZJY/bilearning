"""字幕与本机 OCR 证据；不上传原始图像，不把识别文本当唯一真值。"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from app.worker.framer import FFMPEG, FFPROBE, pick_interval
from app.worker.transcriber import fmt_srt_time, parse_srt_time


def run_command(cmd, checkpoint):
    """文件缓冲避免管道写满；检查取消时终止本次子进程。"""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        checkpoint()
        proc = subprocess.Popen(cmd, stdout=out, stderr=err)
        try:
            while True:
                checkpoint()
                try:
                    proc.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
            checkpoint()
            if proc.returncode:
                raise RuntimeError(f"{Path(cmd[0]).name} 执行失败（{proc.returncode}）")
            out.seek(0)
            return out.read().decode("utf-8", errors="replace")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def subtitle_text(body, duration=0):
    lines = []
    intervals = []
    if not isinstance(body, list):
        return "", {"status": "invalid"}
    for row in body:
        try:
            start, end = float(row["from"]), float(row["to"])
            content = row["content"].strip()
            if (
                not content
                or not math.isfinite(start + end)
                or start < 0
                or end <= start
            ):
                continue
            intervals.append((start, end, content))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    intervals.sort()
    for start, _, content in intervals:
        lines.append(f"[{fmt_srt_time(start)}] {content}")
    end = max((r[1] for r in intervals), default=0)
    text = "\n".join(lines)
    usable = len("".join(r[2] for r in intervals)) >= 200 and (
        not duration or end >= duration * 0.9
    )
    report = {
        "status": "usable" if usable else "insufficient",
        "segments": len(lines),
        "last_end": end,
        "quality_note": "长度与尾部检查仅为启发式，不证明字幕准确或全程完整",
    }
    return (text if usable else ""), report


def load_subtitles(
    client, bvid, cid, work, duration=0, local_path=None, checkpoint=lambda: None
):
    attempts = []
    if bvid:
        checkpoint()
        tracks = client.get_subtitles(bvid, cid)
        for track in tracks:
            try:
                checkpoint()
                url = track.get("subtitle_url", "")
                if url.startswith("//"):
                    url = "https:" + url
                parsed = urlparse(url)
                host = parsed.hostname or ""
                if parsed.scheme != "https" or not any(
                    host == d or host.endswith("." + d)
                    for d in ("hdslb.com", "bilibili.com")
                ):
                    continue
                # 不把媒体 URL 或鉴权参数写入产物/错误。
                response = requests.get(url, timeout=15, allow_redirects=False)
                response.raise_for_status()
                body = response.json().get("body", [])
                text, report = subtitle_text(body, duration)
                attempts.append(report)
                checkpoint()
                if text:
                    (work / "subtitles.json").write_text(
                        json.dumps(body, ensure_ascii=False), encoding="utf-8"
                    )
                    return text, {
                        **report,
                        "source": "bilibili",
                        "language": track.get("lan", "unknown"),
                    }
            except Exception as exc:
                attempts.append({"status": "failed", "error_type": type(exc).__name__})
    if local_path:
        try:
            streams = json.loads(
                run_command(
                    [
                        FFPROBE,
                        "-v",
                        "error",
                        "-show_streams",
                        "-of",
                        "json",
                        str(local_path),
                    ],
                    checkpoint,
                )
            )
            if any(
                s.get("codec_type") == "subtitle" for s in streams.get("streams", [])
            ):
                srt = run_command(
                    [
                        FFMPEG,
                        "-v",
                        "error",
                        "-i",
                        str(local_path),
                        "-map",
                        "0:s:0",
                        "-f",
                        "srt",
                        "-",
                    ],
                    checkpoint,
                )
                body = []
                for block in re.split(r"\n\s*\n", srt):
                    rows = block.strip().splitlines()
                    for i, line in enumerate(rows):
                        if "-->" in line:
                            a, b = line.split("-->")
                            body.append(
                                {
                                    "from": parse_srt_time(a),
                                    "to": parse_srt_time(b),
                                    "content": " ".join(rows[i + 1 :]),
                                }
                            )
                            break
                text, report = subtitle_text(body, duration)
                attempts.append(report)
                if text:
                    (work / "subtitles.srt").write_text(srt, encoding="utf-8")
                    return text, {**report, "source": "embedded"}
        except Exception as exc:
            attempts.append({"status": "failed", "error_type": type(exc).__name__})
    return "", {
        "status": "unavailable",
        "source": "subtitle",
        "attempts": attempts,
        "reason": "没有获取到质量检查通过的字幕，将回退本地转写；不代表原视频没有字幕",
    }


def collect_visual_evidence(
    video,
    work,
    duration,
    *,
    hard_subtitles=False,
    crop_bottom=0.3,
    checkpoint=lambda: None,
):
    checkpoint()
    if not shutil.which("tesseract"):
        return "", {
            "status": "unavailable",
            "reason": "未安装本机 tesseract，画面未核对",
        }
    try:
        languages = run_command(["tesseract", "--list-langs"], checkpoint).split()
    except Exception as exc:
        return "", {
            "status": "unavailable",
            "reason": f"OCR 能力探测失败：{type(exc).__name__}",
        }
    if not {"eng", "chi_sim"}.issubset(languages):
        return "", {
            "status": "unavailable",
            "reason": "OCR 需要 eng 与 chi_sim 语言包，画面未核对",
        }
    if not duration or duration <= 0:
        try:
            duration = float(
                run_command(
                    [
                        FFPROBE,
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(video),
                    ],
                    checkpoint,
                ).strip()
            )
        except Exception:
            return "", {
                "status": "unavailable",
                "reason": "无法确定视频时长，未进行画面采样",
            }
    if not math.isfinite(duration) or duration <= 0:
        return "", {"status": "unavailable", "reason": "视频时长无效"}
    interval = 1 if hard_subtitles else pick_interval(duration)
    samples = list(range(0, math.ceil(duration), interval))
    # Leptonica 在 macOS 下可能拒绝 /tmp 等符号链接路径；使用同一文件的规范路径。
    work = Path(work).resolve()
    video = Path(video).resolve()
    frames = work / ("subtitle_frames" if hard_subtitles else "visual_frames")
    frames.mkdir(parents=True, exist_ok=True)
    if not 0 < crop_bottom <= 1:
        raise ValueError("字幕裁剪比例必须在 (0,1] 内")
    output = []
    failures = []
    recognized = 0
    for second in samples:
        checkpoint()
        path = frames / f"{second:08d}.png"
        filters = (
            f"crop=iw:ih*{crop_bottom}:0:ih*(1-{crop_bottom}),negate,scale=iw*2:ih*2"
            if hard_subtitles
            else "scale=iw*2:ih*2"
        )
        try:
            # 时间来自明确 seek，不能从被筛选/分片后的枚举下标推算。
            run_command(
                [
                    FFMPEG,
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    str(float(second)),
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-vf",
                    filters,
                    str(path),
                ],
                checkpoint,
            )
            raw = run_command(
                ["tesseract", str(path), "stdout", "-l", "eng+chi_sim", "--psm", "6"],
                checkpoint,
            )
            text = re.sub(
                r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", raw
            ).strip()
            if text:
                recognized += 1
                output.append(f"[{fmt_srt_time(second)}] {text}")
        except Exception as exc:
            failures.append({"second": second, "error_type": type(exc).__name__})
    text = "\n".join(output)
    (work / ("hard_subtitles.txt" if hard_subtitles else "visual_ocr.txt")).write_text(
        text, encoding="utf-8"
    )
    return text, {
        "status": "partial" if text else "unavailable",
        "source": "hard_subtitle_ocr" if hard_subtitles else "frame_ocr",
        "interval": interval,
        "attempted": len(samples),
        "recognized": recognized,
        "failed": failures,
        "reason": (
            "已读取画面文字；OCR 可能误识别，采样可能漏内容，图表空间关系未经视觉模型核对"
            if text
            else "未取得可用画面文字，画面未核对；请检查媒体、字幕区域与 OCR 配置"
        ),
    }
