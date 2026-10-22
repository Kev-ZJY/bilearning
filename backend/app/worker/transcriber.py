"""base 完成后发出就绪信号，并启动 medium；两个 ASR 模型按顺序运行。

视频流水线等待两份转写并完成逐秒画面分析后才起草，最终文档在独立审核后保存。
转写耗时与准确度取决于设备、音频和模型，不能由固定时长保证。
输出每行格式为 [HH:MM:SS,mmm] 文本；支持按时间读取片段。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

# 转写每行时间头前缀与后缀（用于 read_segment 解析）
WAIT_HEARTBEAT_SECONDS = 30

_LINE_PREFIX = "["
_LINE_SUFFIX = "]"


# ---------------------------------------------------------------------------
# 纯函数：术语清洗 + SRT 时间互转
# ---------------------------------------------------------------------------
def pick_terms(raw: str | list[str] | None) -> list[str]:
    """清理领域术语列表：去首尾空白、去重（保序）、去掉空项。

    输入支持逗号分隔字符串或 list。结果会喂给 Whisper 的 ``initial_prompt``，
    显著提升英文术语识别率。空输入返回空列表。
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [x for x in raw.split(",")]
    else:
        items = list(raw)

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def parse_srt_time(s: str) -> float:
    """解析 SRT 时间戳为秒（float）。

    支持毫秒分隔符逗号或点：``"00:00:01,500"`` -> 1.5，
    ``"01:02:03,250"`` -> 3723.25。格式 ``HH:MM:SS,mmm``。
    """
    s = s.strip()
    if "," in s:
        hms, frac = s.rsplit(",", 1)
    elif "." in s:
        hms, frac = s.rsplit(".", 1)
    else:
        hms, frac = s, "0"
    parts = hms.split(":")
    if len(parts) != 3:
        raise ValueError(f"无法解析 SRT 时间: {s!r}")
    h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    ms = int(frac.ljust(3, "0")[:3])
    return h * 3600 + m * 60 + sec + ms / 1000.0


def fmt_srt_time(t: float) -> str:
    """秒（float）格式化为 SRT 时间戳 ``HH:MM:SS,mmm``。t 被夹到非负。"""
    total_ms = int(round(max(0.0, float(t)) * 1000))
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    sec = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def format_srt(segments: list[tuple[float, str]]) -> str:
    """把 ``(start, text)`` 列表生成标准带编号的 SRT 文本。

    每段结束时间取下一句开始时间；末句给一个 2.5 秒的默认时长。
    块之间空行分隔。
    """
    n = len(segments)
    if n == 0:
        return ""
    blocks: list[str] = []
    for i, (start, text) in enumerate(segments):
        end = segments[i + 1][0] if i + 1 < n else start + 2.5
        blocks.append(
            f"{i + 1}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n{text}"
        )
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# 单模型转写
# ---------------------------------------------------------------------------
class WhisperTranscriber:
    """封装 faster-whisper 的单模型转写，模型实例按名缓存。

    - 模型缓存：同一实例内 ``_get_model(name)`` 复用已加载模型。base / medium
      各加载一次，后续 run 复用（用 dict 缓存，非 lru_cache 以便显式控制）。
    - 本地权重优先：``model_dir`` 存在时优先从 ``model_dir / name`` 加载
      （为将来「部署时打包模型不联网下载」留路）；否则回退到标准名称让
      faster-whisper 自己找 HF 缓存。
    """

    def __init__(
        self,
        model_dir: str | None = None,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self.model_dir = model_dir
        self.device = device
        # faster-whisper 的 compute_type 没有 "auto"，翻译为库默认 "default"
        self.compute_type = "default" if compute_type == "auto" else compute_type
        self._models: dict[str, object] = {}

    def _effective_compute_type(self) -> str:
        """算出真正该用的 compute_type，并说明为什么不擅自改成 int8。

        ctranslate2 的 "default" 会照搬权重文件里记录的精度。实测 medium 权重标的是
        float16，CPU 后端跑不了高效 float16，于是它打一条 warning 并回退成 float32。

        **不要把这里改成 int8。** 曾以为「CPU 上 int8 量化比 float32 快 2-4 倍」是
        常识，于是在 CPU 分支硬选了 int8。实测（Apple Silicon arm64，base 模型，
        32 秒中文音频，预热后各跑 3 次取中位）：

            float32  2.97s      int8  3.07s      int8/float32 = 1.04x

        两者在噪声范围内，int8 并不更快。ctranslate2 的 int8 算子在 x86 上有优势，
        在 Apple Silicon 上没有。所以这里保持 ctranslate2 自己的选择，只把那个
        无害的 warning 解释清楚，别让下一个人重犯同样的想当然。

        显式指定了具体值就照用，不擅自改。
        """
        return self.compute_type

    def _resolve_model_source(self, name: str) -> str:
        """决定模型从哪加载：本地目录还是 faster-whisper 自己的缓存。

        关键：``model_dir`` 只是「优先用本地权重」的约定，**目录不存在时必须回退到
        模型名**。

        踩过的坑：默认 ``whisper_model_dir="./models/whisper"``，而这个目录在开发机上
        根本不存在。此时若把拼接出来的 ``./models/whisper/base`` 当成 HF repo 名传给
        ``WhisperModel()``，faster-whisper 会走「联网下载」路径——即使 base/medium 权重
        早已躺在 HF 缓存里也用不上。表现是转写阶段静默卡住十几分钟、不报错也不产出
        文件（网络不通时下载请求会一直挂着）。

        所以这里显式判断目录是否存在：不存在就把模型名原样交回去，让
        faster-whisper 按自己的规则命中本地 HF 缓存。
        """
        if self.model_dir is None:
            return name
        path = Path(self.model_dir).expanduser() / name
        if path.is_dir():
            return str(path)
        return name

    def _get_model(self, name: str):
        """取（并缓存）指定名称的模型对象。"""
        if name in self._models:
            return self._models[name]

        # 延迟导入：避免在无需转写时加载 heavy 依赖
        from faster_whisper import WhisperModel

        source = self._resolve_model_source(name)
        model = WhisperModel(
            source, device=self.device, compute_type=self._effective_compute_type()
        )
        self._models[name] = model
        return model

    def run(
        self,
        name: str,
        audio: Path,
        out: Path,
        terms: list[str],
        progress_cb: Callable[[float | None, str], None] | None = None,
    ) -> Path:
        """转写音频，输出 ``[HH:MM:SS,mmm] 文本`` 行到 out，返回 out。

        - language="zh"：中文课程
        - vad_filter=True：去掉静音段省时间
        - initial_prompt：拼接术语，形如「以下是一段中文技术课程内容，涉及术语：
          HTTP、CORS、ETag」。terms 为空时不传该参数（避免干扰）。
        - progress_cb：可选进度回调 ``(pct, msg)``。``pct`` 为 0~100 的完成度，
          拿不到总时长时为 ``None``（降级模式）。长视频（几十小时）转写动辄数小时，
          靠它在 asr 阶段持续回传进度，避免前端看门狗误报「进度更新中断」。
        """
        audio = Path(audio)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        model = self._get_model(name)
        prompt = None
        if terms:
            prompt = "以下是一段中文技术课程内容，涉及术语：" + "、".join(terms)

        kwargs = {"language": "zh", "vad_filter": True}
        if prompt is not None:
            kwargs["initial_prompt"] = prompt

        # 关键：faster-whisper 这一版（>=1.x）的 ``transcribe()`` **不接受**
        # ``duration`` 入参——老文档/网上示例常写 ``model.transcribe(audio,
        # duration=...)``，但本机实际安装的版本签名里根本没有这个形参，强行传
        # 会 ``TypeError``。所以总时长只能从返回的 ``TranscriptionInfo`` 里取，
        # 不能提前喂进去。
        #
        # 进度分母必须用**原始音频总长** ``info.duration``，而不是
        # ``duration_after_vad``。原因（踩过的真 bug，极容易被下一个人「优化」回去）：
        # faster-whisper 在返回 segments 之前会调 ``restore_speech_timestamps()`` 把
        # ``seg.start`` / ``seg.end`` 还原到**原始音频时间轴**，所以分子 ``seg.end`` 是
        # 原始时间轴上的秒数。而 ``duration_after_vad`` 是 VAD 静音过滤后**有效人声**的
        # 秒数（比原音频短得多）。若拿它当分母，分子/分母不在一个量纲，进度被虚高——
        # 实测 93 秒音频里 23.8 秒人声，进度刚到 25% 就显示 98%、之后一路钉死 100%，
        # 比「没进度」更具误导性（41 小时视频会一开始就冲到 100% 然后不动）。
        # ``duration_after_vad`` 仍有价值（比如日志里记「有效人声占比」判断内容密度），
        # 但**绝不能当进度分母**。两字段都可能为 0/None（极短音频或读取失败）：优先
        # 原始 ``duration``，拿不到再退回 ``duration_after_vad``，再不行走「无百分比」
        # 降级分支，绝不除零。
        segments, info = model.transcribe(str(audio), **kwargs)

        # 总时长（进度分母）：优先原始音频时长 info.duration（与 seg.end 同轴），
        # 拿不到再退回 duration_after_vad，再到 0（触发降级分支）。
        total = getattr(info, "duration", None) or getattr(
            info, "duration_after_vad", None
        )
        if not total or total <= 0:
            total = 0.0

        lines: list[str] = []
        last_cb_pct = -100.0  # 上一次回调的百分比；初始化成不会命中 1% 节流的值
        last_cb_time = 0.0    # 上一次回调的单调时间（用于 2 秒节流）
        now = time.monotonic
        for seg in segments:
            ts = fmt_srt_time(seg.start)
            text = (seg.text or "").strip()
            if text:
                lines.append(f"[{ts}] {text}")

            # —— 进度回调（节流）——
            # 为什么要节流：Whisper 边消费边转写、每秒音频可能吐出多个 segment，
            # 41 小时视频轻松产出上万段，逐段回调会让 SSE 事件洪泛、前端看门狗
            # 反而被冲垮。用「百分比变化 >=1 个点」或「距上次回调 >=2 秒」双条件：
            # 前者保证长视频进度肉眼可见地推进，后者保证卡在某一段时仍有心跳。
            if progress_cb is not None:
                t = now()
                if total > 0:
                    # 已处理秒数取 segment 的结束时间（生成器按顺序吐，end 单调不降）。
                    processed = (
                        float(seg.end) if seg.end is not None else float(seg.start)
                    )
                    pct = min(100.0, max(0.0, processed / total * 100.0))
                    if (pct - last_cb_pct >= 1.0) or (t - last_cb_time >= 2.0):
                        progress_cb(pct, f"转写中 {pct:.0f}%")
                        last_cb_pct = pct
                        last_cb_time = t
                else:
                    # 降级：拿不到总时长就无法算百分比。仍按 2 秒节流汇报「已产出段数」，
                    # 让用户看到转写在推进（且不崩、不洪泛）。这个分支必须有测试覆盖。
                    if t - last_cb_time >= 2.0:
                        progress_cb(None, f"转写中（已产出 {len(lines)} 段）")
                        last_cb_time = t

        out.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        return out

    def outline(self, transcript: Path, samples: int = 30) -> str:
        """按时间采样输出，不读全文，让 Agent 快速摸清章节结构。

        文稿按时间有序，故按行号等间隔采样即可近似时间均匀。返回不超过
        ``samples`` 行；原文不足时全量返回。
        """
        transcript = Path(transcript)
        lines = [l for l in transcript.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) <= samples:
            return "\n".join(lines)
        step = len(lines) / samples
        picked = [lines[int(i * step)] for i in range(samples)]
        return "\n".join(picked)

    def read_segment(self, transcript: Path, start: float, end: float) -> str:
        """按时间区间 [start, end) 过滤出对应行，用于精读某段原始转写。

        解析行首 ``[HH:MM:SS,mmm]`` 取出时间戳进行比较。无时间头的行跳过。
        """
        transcript = Path(transcript)
        result: list[str] = []
        for line in transcript.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ts = self._line_time(line)
            if ts is None:
                continue
            if start <= ts < end:
                result.append(line)
        return "\n".join(result)

    @staticmethod
    def _line_time(line: str) -> float | None:
        """从 ``[时间] 文本`` 行解析出时间戳（秒）。"""
        i = line.find(_LINE_PREFIX)
        if i == -1:
            return None
        j = line.find(_LINE_SUFFIX, i + 1)
        if j == -1:
            return None
        try:
            return parse_srt_time(line[i + 1 : j])
        except (ValueError, Exception):
            return None


# ---------------------------------------------------------------------------
# 双轨调度：base 起草 + medium 修术语
# ---------------------------------------------------------------------------
class DualTrackTranscriber:
    """顺序执行 base / medium，base_ready 使上层无需等待 medium 即可起草。"""

    def __init__(self, engine=None, should_cancel=None):
        self.engine = engine or WhisperTranscriber()
        self.should_cancel = should_cancel or (lambda: False)
        self._stopped = threading.Event()
        self._started = False
        self.base_ready = threading.Event()
        self.base_error: Exception | None = None
        self.medium_error: Exception | None = None
        self._t2: threading.Thread | None = None
        # medium 线程**已创建**的信号。与 _t2 分开是因为二者是不同的状态：
        # _t2 是「线程对象」（可能还没 start），本事件是「线程已存在」。
        #
        # 为什么需要它（串行化引入的真实竞态，不是理论风险）：
        # base 线程里是「set(base_ready) → 创建 _t2 → start」。而上层拿到
        # base_ready 后一路走到 verify 阶段调 wait_medium()，**完全可能**
        # 在 base 线程执行到「创建 _t2」之前就调进来。此时 _t2 is None，
        # wait_medium 会静默直接返回——上层以为「medium 等好了」，
        # 实际 medium 一个字都没转，术语校验这一步被整段跳过。
        #
        # 旧并行实现没这个问题：两个线程在 start() 里同步创建完才返回，
        # 所以上层拿到控制权时 _t2 一定非 None。改成串行后才暴露出来。
        self._medium_started = threading.Event()
        # 锁保护 _t2 的读写（run_base 线程写、wait_medium 线程读）
        self._lock = threading.Lock()

    def start(
        self, audio: Path, out_dir: Path, terms: list[str], progress_cb=None
    ) -> tuple[Path, Path, threading.Event]:
        if self._started:
            raise RuntimeError("转写实例不可重复启动")
        self._started = True
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base_out = out_dir / "base.txt"
        medium_out = out_dir / "medium.txt"

        def progress(pct, message):
            self._check_cancel()
            if progress_cb:
                progress_cb(pct, message)

        def run_base():
            # 只把 progress_cb 接到 base：base 是用户等待的主线（base 一完成就进入
            # 起草），它的进度才对用户有意义。medium 是后台修术语的慢线，且它开始
            # 之前用户早就看到草稿阶段了，它的进度毫无意义——直接忽略回调，
            # 避免无意义的事件抖动。
            try:
                self.engine.run("base", audio, base_out, terms, progress_cb=progress)
                self._check_cancel()
            except BaseException as exc:
                self.base_error = exc
                self.base_ready.set()
                self._medium_started.set()
                return
            # base 跑完立即通知，绝不能等 medium
            self.base_ready.set()
            # **串行的关键**：base 跑完才开始 medium。
            #
            # 顺序很重要——先 set 事件再起 medium 线程。set 是瞬时的，
            # 而 Thread.start() 有微小开销（创建线程 + 调度）。
            # 若写成「先 start 再 set」，上层就可能被这几百微秒的调度延迟
            # 挡住，起草的起点被 medium 的启动开销推后。
            t2 = threading.Thread(target=self._run_medium, daemon=True,
                                  args=(audio, medium_out, terms))
            with self._lock:
                self._t2 = t2
            # 先 start 再 set _medium_started：wait_medium 等的是「线程已存在」，
            # 拿到句柄就该去 join。若顺序反了，wait_medium 可能拿到 None。
            try:
                t2.start()
            except BaseException as exc:
                self.medium_error = exc
            finally:
                self._medium_started.set()

        t1 = threading.Thread(target=run_base, daemon=True)
        t1.start()
        return base_out, medium_out, self.base_ready

    def _run_medium(self, audio: Path, medium_out: Path, terms: list[str]) -> None:
        """medium 转写。在 base 完成后由 base 线程启动（串行，见类 docstring）。"""
        try:
            self._check_cancel()
            self.engine.run("medium", audio, medium_out, terms,
                            progress_cb=lambda pct, msg: self._check_cancel())
            self._check_cancel()
        except BaseException as exc:
            self.medium_error = exc

    def _check_cancel(self):
        from app.core.cancel import TaskCancelled
        if self._stopped.is_set() or self.should_cancel():
            raise TaskCancelled()

    def stop(self):
        self._stopped.set()

    def wait_medium(self, timeout: float | None = None, progress_cb=None) -> None:
        """单一截止时间覆盖创建与 join；超时绝不作为完成。"""
        started = last_tick = time.monotonic()
        deadline = None if timeout is None else started + timeout
        def remaining():
            nonlocal last_tick
            self._check_cancel()
            now = time.monotonic()
            if progress_cb and now-last_tick >= WAIT_HEARTBEAT_SECONDS:
                progress_cb(now-started)
                last_tick=now
            if deadline is None: return .1
            left = deadline - time.monotonic()
            if left <= 0: raise TimeoutError("medium 转写未在截止时间内完成，未提交文档")
            return min(.1, left)
        while not self._medium_started.wait(remaining()):
            pass
        if self.base_error is not None: raise self.base_error
        with self._lock:
            t2 = self._t2
        if t2 is None: raise RuntimeError("medium 线程没有启动")
        while t2.is_alive(): t2.join(remaining())
        if self.medium_error is not None: raise self.medium_error
        self._check_cancel()
