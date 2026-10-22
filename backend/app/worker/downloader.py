"""视频流下载器（m4s 音频/视频流）。

设计背景（均为真实请求验证结论，请勿"优化"掉任何一条）：
1. 默认只下音频：知识抽取只需听清内容，最高 116kbps 音频对 ASR 完全够用，
   只有视觉类内容（PPT/公式/代码演示）才需补下视频流。
2. CDN 下载必须带全五个请求头（见 build_headers），只带 UA 会 412。
3. 下载必须用 .part 临时文件 + os.replace 原子替换，否则中断会留下半截文件
   被后续流程当成完整音频使用。异常时必须清理 .part。
4. 流对象的 URL 字段在不同接口/版本下出现过 baseUrl 与 base_url 两种拼法，
   都遇到过，必须兼容。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import requests

from app.sources.bilibili_wbi import BilibiliClient, build_headers

# 256KB 分块，平衡内存占用与上下文切换开销
CHUNK_SIZE = 1 << 18


def pick_audio_stream(dash: dict) -> dict:
    """从 DASH 的 audio 列表里挑 bandwidth 最大的音频流。

    无流时抛 ValueError（调用方据此降级或报错，不该静默返回 None）。
    """
    streams = dash.get("audio", [])
    if not streams:
        raise ValueError("no audio stream available")
    return max(streams, key=lambda s: s.get("bandwidth", 0))


def pick_video_stream(dash: dict) -> dict:
    """优先 H264 视频流供页面播放；同编码取最大带宽，缺少时保留原回退。"""
    streams = dash.get("video", [])
    if not streams:
        raise ValueError("no video stream available")
    h264 = [stream for stream in streams if stream.get("codecid") == 7
            or str(stream.get("codecs", "")).lower().startswith(("avc", "h264"))]
    return max(h264 or streams, key=lambda s: s.get("bandwidth", 0))


def stream_url(s: dict) -> str:
    """取流地址，兼容 baseUrl 与 base_url 两种字段名（坑 4）。"""
    url = s.get("baseUrl") or s.get("base_url")
    if not url:
        raise ValueError("stream has no url (neither baseUrl nor base_url)")
    return url


def estimate_size(bitrate: int, duration_sec: float) -> float:
    """估算下载大小（MB）。bitrate 单位 bits/s。

    公式：bitrate(bits) * 秒 / 8(bits->bytes) / 1024 / 1024 -> MB。
    验算：116198 * 3600 / 8 / 1024 / 1024 ≈ 52 MB。
    """
    return bitrate * duration_sec / 8 / 1024 / 1024


def check_disk_space(path: Path, needed_mb: float) -> bool:
    """检查 path 所在文件系统是否有足够空闲空间（needed_mb）。用 shutil.disk_usage。"""
    usage = shutil.disk_usage(path)
    free_mb = usage.free / 1024 / 1024
    return free_mb >= needed_mb


def download_stream(
    url: str,
    dest: Path,
    bvid: str,
    progress_cb=None,
    timeout: int = 60,
) -> Path:
    """流式下载单个流到 dest，返回 dest。

    - chunk_size=256KB，边下边写 .part 临时文件。
    - 完成后 os.replace 原子替换，避免半截文件被当成完整音频。
    - 异常时清理 .part，不留垃圾。
    - progress_cb(fraction) 每写完一块回调一次，从 Content-Length 头算进度。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / (dest.name + ".part")
    headers = build_headers(bvid)

    try:
        with requests.get(
            url, headers=headers, stream=True, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0) or 0)
            downloaded = 0
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        progress_cb(downloaded / total if total else 0.0)
        os.replace(part, dest)
        return dest
    except BaseException:
        # 任何异常（含中断）都必须清理 .part，否则半截文件会污染后续流程
        if part.exists():
            part.unlink()
        raise


def download_audio(
    client: BilibiliClient,
    bvid: str,
    cid: int,
    out_dir: Path,
    quality: int = 64,
    progress_cb=None,
) -> dict:
    """完整音频下载流程，返回产物信息字典。

    steps: get_playurl -> pick_audio_stream -> 从 timelength 算 duration ->
    estimate_size -> check_disk_space（不足抛异常，信息含预计/可用 MB）-> download_stream。
    返回 {audio_path, audio_bitrate, duration_sec, size_mb, qualities}。

    注意 ``audio_path`` 是 **str** 不是 Path：这份 dict 会被 pipeline 原样
    ``json.dumps`` 写进 Video.artifacts_json，塞 Path 对象会在落库那一步抛
    "Object of type PosixPath is not JSON serializable"，而且是在流水线跑了
    几十秒、进度到了 96% 才炸。调用方需要 Path 时自己 Path(...) 包一层。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    playurl = client.get_playurl(bvid, cid, quality=quality)
    dash = playurl.get("dash", {})
    audio = pick_audio_stream(dash)

    bitrate = audio.get("bandwidth", 0)
    timelength_ms = playurl.get("timelength", 0)
    duration_sec = timelength_ms / 1000.0
    size_mb = estimate_size(bitrate, duration_sec)

    if not check_disk_space(out_dir, size_mb):
        usage = shutil.disk_usage(out_dir)
        free_mb = usage.free / 1024 / 1024
        raise OSError(
            f"磁盘空间不足：预计需要 {size_mb:.1f} MB，可用 {free_mb:.1f} MB"
        )

    url = stream_url(audio)
    audio_path = out_dir / f"{bvid}_{cid}.m4s"
    download_stream(url, audio_path, bvid, progress_cb=progress_cb)

    return {
        "audio_path": str(audio_path),
        "audio_bitrate": bitrate,
        "duration_sec": duration_sec,
        "size_mb": size_mb,
        "qualities": playurl.get("accept_description", []),
    }


def download_video(
    client: BilibiliClient,
    bvid: str,
    cid: int,
    out_dir: Path,
    quality: int = 64,
    progress_cb=None,
) -> Path:
    """视频流下载，供后续抽帧使用。同样先检查磁盘空间。返回视频文件 Path。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    playurl = client.get_playurl(bvid, cid, quality=quality)
    dash = playurl.get("dash", {})
    video = pick_video_stream(dash)

    bitrate = video.get("bandwidth", 0)
    duration_sec = playurl.get("timelength", 0) / 1000.0
    size_mb = estimate_size(bitrate, duration_sec)

    if not check_disk_space(out_dir, size_mb):
        usage = shutil.disk_usage(out_dir)
        free_mb = usage.free / 1024 / 1024
        raise OSError(
            f"磁盘空间不足：预计需要 {size_mb:.1f} MB，可用 {free_mb:.1f} MB"
        )

    url = stream_url(video)
    video_path = out_dir / f"{bvid}_{cid}.video.m4s"
    download_stream(url, video_path, bvid, progress_cb=progress_cb)
    return video_path
