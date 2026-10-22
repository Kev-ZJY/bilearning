"""ffmpeg / ffprobe 抽帧与联系表生成。

保留联系表和独立抽帧工具；流水线实际 OCR 证据由 evidence.py 提供。
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

FFMPEG = os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.getenv("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"

# xstack / vstack 的单次多路输入上限。多数版本为 26，这里用保守值。
# make_contact_sheet 通过「逐行 xstack + 整体 vstack」规避该限制：
# 只要 cols <= XSTACK_MAX 且 行数 <= XSTACK_MAX 即可（即单表上限约 26*26=676 帧）。
XSTACK_MAX = 26


def pick_interval(duration_sec: float) -> int:
    """根据视频时长选择抽帧间隔（秒）。

    - <=1200 秒（20 分钟）：45（落在 30-60 区间）
    - <=3600 秒（60 分钟）：60
    - 更长：90
    """
    if duration_sec <= 1200:
        return 45
    if duration_sec <= 3600:
        return 60
    return 90


def estimate_frame_count(duration_sec: float, interval: int) -> int:
    """按间隔估算帧数，向上取整。interval 必须 >= 1。"""
    if interval < 1:
        raise ValueError("interval 必须 >= 1")
    return math.ceil(duration_sec / interval)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """执行命令，失败即抛异常（不静默吞掉）。"""
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def probe_duration(path: Path) -> float:
    """用 ffprobe 读取视频时长（秒）。"""
    out = _run([
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]).stdout.strip()
    return float(out) if out else 0.0


def _has_stream(path: Path, spec: str) -> bool:
    """检查是否存在某类流（spec 如 'v' 视频、's' 字幕）。"""
    out = _run([
        FFPROBE, "-v", "error",
        "-select_streams", spec,
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ]).stdout.strip()
    return bool(out)


def has_video_stream(path: Path) -> bool:
    """文件里是否存在视频流。"""
    return _has_stream(path, "v")


def has_subtitle_stream(path: Path) -> bool:
    """文件里是否存在字幕流。用于「有字幕就跳过 ASR」的优化判断。"""
    return _has_stream(path, "s")


def extract_frames(video: Path, out_dir: Path, interval: int | None = None) -> list[Path]:
    """按固定间隔抽帧为 JPG，返回排序后的文件列表。

    没有视频流（如纯音频文件）时直接返回空列表，不调用 ffmpeg，避免崩溃。
    命令：ffmpeg -i <video> -vf fps=1/<interval> -q:v 3 <out_dir>/frame_%05d.jpg
    """
    if not has_video_stream(video):
        return []

    if interval is None:
        interval = pick_interval(probe_duration(video))
    if interval < 1:
        raise ValueError("interval 必须 >= 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%05d.jpg"
    _run([
        FFMPEG, "-y", "-i", str(video),
        "-vf", f"fps=1/{interval}",
        "-q:v", "3",
        str(pattern),
    ])
    return sorted(out_dir.glob("frame_*.jpg"))


def make_contact_sheet(
    frames: list[Path],
    out: Path,
    cols: int = 6,
    cell: int = 320,
) -> Path:
    """把多张帧用 xstack 拼成一张联系表（out）。

    每个输入先等比 scale 并 pad 到 cell x cell（缩略图统一大小、网格对齐）。
    关键约束：xstack 单次输入上限为 XSTACK_MAX（常见 26）。本实现把帧按
    ``cols`` 切成若干行，每行用一次 xstack（grid=行宽x1）拼成，不足整行的末行
    pad 到整行宽度，最后用一次 vstack 把所有行纵向叠起来。

    因此限制为 cols <= XSTACK_MAX 且 行数(=ceil(N/cols)) <= XSTACK_MAX，
    即单表最多约 XSTACK_MAX*XSTACK_MAX 帧。超出则抛 ValueError。

    注意：本机 ffmpeg 9.0.1 的 xstack 手写 layout 坐标有偏移 bug，故统一使用
    ``grid`` 选项（按列 x 行网格自动布局）来保证拼图正确。
    """
    if not frames:
        raise ValueError("frames 为空，无法生成联系表")

    n = len(frames)
    if cols < 1:
        raise ValueError("cols 必须 >= 1")
    if cols > XSTACK_MAX:
        raise ValueError(f"cols 超过 xstack 输入上限 {XSTACK_MAX}")

    num_rows = math.ceil(n / cols)
    if num_rows > XSTACK_MAX:
        raise ValueError(
            f"帧数 {n} 需 {num_rows} 行，超过 vstack 上限 {XSTACK_MAX}（约 {XSTACK_MAX*cols} 帧）"
        )

    full_w = cols * cell
    inputs: list[str] = []
    parts: list[str] = []
    scale_labels: list[str] = []

    for i, f in enumerate(frames):
        inputs += ["-i", str(f)]
        # 等比缩放到 cell 内并居中 pad 到 cell x cell
        parts.append(
            f"[{i}]scale={cell}:{cell}:force_original_aspect_ratio=decrease,"
            f"pad={cell}:{cell}:(ow-iw)/2:(oh-ih)/2[s{i}]"
        )
        scale_labels.append(f"[s{i}]")

    # 逐行横向 xstack（grid=行宽x1），再把每行 pad 到 full_w 宽（保证 vstack 等宽）
    row_labels: list[str] = []
    for r in range(num_rows):
        start = r * cols
        end = min(start + cols, n)
        k = end - start
        if k == 1:
            # xstack 要求 inputs>=2，单行只剩一帧时直接用缩放后标签
            parts.append(f"{scale_labels[start]}pad={full_w}:{cell}:(ow-iw)/2:0[pr{r}]")
        else:
            parts.append(
                "".join(scale_labels[start:end])
                + f"xstack=inputs={k}:grid={k}x1[row{r}]"
            )
            parts.append(f"[row{r}]pad={full_w}:{cell}:(ow-iw)/2:0[pr{r}]")
        row_labels.append(f"[pr{r}]")

    # 叠合所有行
    if num_rows == 1:
        map_label = row_labels[0]
    else:
        parts.append(f"{''.join(row_labels)}vstack=inputs={num_rows}[v]")
        map_label = "[v]"

    filter_complex = ";".join(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        FFMPEG, "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", map_label,
        "-frames:v", "1",
        str(out),
    ])
    return out
