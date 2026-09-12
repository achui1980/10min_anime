"""TTS。跟 v1 的 LLMProvider 完全同构：真实引擎藏在 Protocol 后面，测试用假引擎。"""

from __future__ import annotations

import asyncio
import random
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from tenmin.config import RenderConfig
from tenmin.models import Beat, Script, VoiceChunk, VoiceTrack
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.chunks import plan_chunks
from tenmin.render.ffmpeg import probe_duration
from tenmin.script.budget import SPEECH_RATE_CPS, narration_chars

TTS_MAX_ATTEMPTS = 3

# --- 合成结果的时长体检。依据全写在 _duration_bounds 的 docstring 里 ---
# 刻意不做成 RenderConfig 旋钮：它不是「这部番想要多长的解说」那类创作参数，而是
# 「Edge TTS 物理上不可能吐出这种音频」的合法性边界，标定自 115 个真实 chunk。
DURATION_TOLERANCE = 0.5
DURATION_ABSOLUTE_SLACK = 2.0

# --- 重试退避。参数与命名一律沿用 script/llm.py 的既有模式 ---
# 基数 1 秒：Edge TTS 的瞬时 403 / WebSocket 握手失败的恢复窗是秒级。
# 上限 30 秒：单集有上百个 chunk，单次等待再往上翻只会把一次注定失败的运行拖成十几分钟。
# 抖动是**乘性**的 [1, 1.25)：只会加不会减，同时打散批量模式下多集连着撞同一个限流窗。
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
BACKOFF_JITTER_RATIO = 0.25

# 「重试一定没用」的异常。edge_tts.Communicate.__init__ 与 data_classes.TTSConfig
# .__post_init__ 对 voice / rate / volume / pitch / proxy / 超时参数做的全是纯参数校验
# （communicate.py:346/355/360-363 抛 TypeError，data_classes.py:38 的
# validate_string_param 抛 ValueError，例如 rate 不匹配 ^[+-]\d+%$ 就是
# "Invalid rate '+abc%'."）。重试层每次都传同一组参数进去，所以它们必然以同样的方式再炸
# 一次 —— 背靠背发三遍只是把用户看到报错的时间推后几秒。
_PERMANENT_ERRORS = (TypeError, ValueError)


async def _sleep(seconds: float) -> None:
    """退避用的 sleep。**刻意做成模块级函数**，测试 monkeypatch 掉它就既不真睡、
    又能把整条退避序列的时长逐个断言出来。"""
    await asyncio.sleep(seconds)


def _rand() -> float:
    """[0, 1) 的抖动源。同样是模块级函数，为的是让退避序列在测试里可确定。"""
    return random.random()


def _backoff_delay(attempt: int) -> float:
    """第 attempt 次尝试失败后要等多久（attempt 从 1 开始）。"""
    delay = min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS)
    return delay * (1.0 + BACKOFF_JITTER_RATIO * _rand())


class TTSError(RuntimeError):
    """TTS 层已经自带一句人话的异常。进了 cli.PIPELINE_ERRORS，所以只印消息不印 traceback。"""


@runtime_checkable
class TTSEngine(Protocol):
    async def synthesize(self, text: str, out_path: Path) -> float:
        """合成一段语音，返回真实时长（秒）。"""
        ...


_RATE_PERCENT = re.compile(r"^([+-]\d+)%$")


def _speed_factor(rate: str) -> float:
    """把 edge-tts 的 rate 字符串换算成语速倍率。认不出就当 1.0（体检退化成最宽的 band）。

    edge-tts 自己会校验 `^[+-]\\d+%$`（data_classes.py 的 validate_string_param），
    所以这里认不出的形态在 Communicate 构造期就已经炸了，兜底只是不让体检自己抛。
    """
    match = _RATE_PERCENT.match(rate.strip())
    if match is None:
        return 1.0
    return max(0.1, 1.0 + int(match.group(1)) / 100.0)


def _expected_seconds(text: str, rate: str) -> float:
    """这段文本「应该」有多长。分母是 script/budget.py 那个 4.5 字/秒。"""
    return narration_chars(text) / SPEECH_RATE_CPS / _speed_factor(rate)


def _duration_bounds(text: str, rate: str) -> tuple[float, float]:
    """合成结果的合理时长区间。落在区间外一律当「这个文件不能用」。

    依据（实测，不是拍脑袋）：work/ 下 10 集共 115 个真实 chunk 的
    `实测时长 / (字数 / 4.5)` 落在 **0.809 – 1.179**（中位数 0.926）。所以
    ±50% 的相对带宽相当于给实测下界留了 1.6 倍、上界留了 1.27 倍的余量。

    再叠一个 2 秒的**绝对**余量，因为纯比例带宽会误杀短句：Edge TTS 每段音频首尾都带
    固定的静音开销，一个 2 字的 chunk 估算只有 0.44 秒、实测能到 1.2 秒（比值 2.7），
    纯 ±50% 会把它判死。实测最短的 chunk 是 9 字 / 2.16 秒，加了绝对余量后它距上界还
    有 2.8 秒；把这个 band 套回全部 115 个真实 chunk，**零误杀**，最紧的一侧余量
    2.16 秒。

    换句话说：这条体检只拦「零时长」「只出了几分之一就断流」「挂了几分钟吐出一堆静音」
    这类物理上不可能是正常语音的结果，正常创作与正常网络抖动一律放过。
    """
    expected = _expected_seconds(text, rate)
    lower = max(0.0, expected * (1.0 - DURATION_TOLERANCE) - DURATION_ABSOLUTE_SLACK)
    upper = expected * (1.0 + DURATION_TOLERANCE) + DURATION_ABSOLUTE_SLACK
    return lower, upper


class EdgeTTSEngine:
    """Edge TTS。合成先落 `.part` 再原子改名，所以正式文件永远是体检过的。"""

    def __init__(
        self,
        voice: str = "zh-CN-YunxiNeural",
        rate: str = "+0%",
        *,
        proxy: str | None = None,
        connect_timeout: int = 10,
        receive_timeout: int = 60,
        chunk_timeout_seconds: float = 300.0,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.proxy = proxy
        self.connect_timeout = connect_timeout
        self.receive_timeout = receive_timeout
        self.chunk_timeout_seconds = chunk_timeout_seconds

    async def synthesize(self, text: str, out_path: Path) -> float:
        import edge_tts

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # edge_tts.Communicate.save() 内部是 `open(audio_fname, "wb")` 的**流式**写
        # （communicate.py:616）。Ctrl-C 或断网留下的是一个「非空但截断」的 mp3，
        # 而复用判据只看 st_size > 0 —— 它偏短的时长会静默扭曲此后全部时间轴偏移。
        # 所以合成一律落临时文件，体检通过才 os.replace 到正式名字（同目录内的
        # rename 在 POSIX 上是原子的）。
        part_path = out_path.with_name(out_path.name + ".part")
        try:
            try:
                async with asyncio.timeout(self.chunk_timeout_seconds):
                    communicate = edge_tts.Communicate(
                        text,
                        self.voice,
                        rate=self.rate,
                        proxy=self.proxy,
                        connect_timeout=self.connect_timeout,
                        receive_timeout=self.receive_timeout,
                    )
                    await communicate.save(str(part_path))
            except TimeoutError as error:
                # asyncio.timeout 到点抛的是内置 TimeoutError。edge-tts 内部的
                # sock_read 超时是 aiohttp 的异常，两者不会互相误吞。
                raise TTSError(
                    f"单个 chunk 合成超时（超过总时长上限 {self.chunk_timeout_seconds} 秒）："
                    f"{text!r}。确实需要更久的话调高 render.tts_chunk_timeout_seconds。"
                ) from error
            # 阻塞的 subprocess，必须扔到线程里：直接 await 不了，直接调会把事件循环
            # 整个卡住（P2-B 的 TTS 并发化就完全白做）。
            duration = await asyncio.to_thread(probe_duration, part_path)
            lower, upper = _duration_bounds(text, self.rate)
            if duration <= 0 or not (lower <= duration <= upper):
                raise TTSError(
                    f"合成结果时长 {duration:.3f} 秒不在合理区间 "
                    f"[{lower:.3f}, {upper:.3f}] 内（{narration_chars(text)} 字，"
                    f"rate={self.rate}），大概率是流被截断或只出了静音：{text!r}"
                )
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
        part_path.replace(out_path)
        return duration


def build_tts_engine(cfg: RenderConfig) -> TTSEngine:
    return EdgeTTSEngine(
        voice=cfg.voice,
        rate=cfg.rate,
        proxy=cfg.tts_proxy,
        connect_timeout=cfg.tts_connect_timeout,
        receive_timeout=cfg.tts_receive_timeout,
        chunk_timeout_seconds=cfg.tts_chunk_timeout_seconds,
    )


async def synthesize_with_retry(
    engine: TTSEngine,
    text: str,
    out_path: Path,
    *,
    label: str,
    max_attempts: int = TTS_MAX_ATTEMPTS,
) -> float:
    """单个 chunk 重试 max_attempts 次，失败之间指数退避。

    失败时报清楚是哪个 chunk、原文是什么、试了几次、最后那次是什么异常。原来这里只留
    `str(error)`：aiohttp 的连接类异常 stringify 经常是空串，用户看到的是
    「连续 3 次合成失败：''」，既不知道是网络还是参数错，也没有 traceback 可查。
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await engine.synthesize(text, out_path)
        except _PERMANENT_ERRORS as error:
            raise _exhausted(label, text, error, attempt) from error
        except Exception as error:  # noqa: BLE001 - 网络层什么都可能抛
            if attempt >= max_attempts:
                raise _exhausted(label, text, error, attempt) from error
        await _sleep(_backoff_delay(attempt))


def _exhausted(label: str, text: str, error: Exception, attempt: int) -> TTSError:
    detail = str(error) or repr(error)
    return TTSError(
        f"{label} 合成失败（已尝试 {attempt} 次）：{text!r}\n"
        f"{type(error).__name__}: {detail}"
    )


async def synthesize_track(
    script: Script,
    episode: int,
    voice_dir: Path,
    engine: TTSEngine,
    *,
    reuse: bool = True,
    reporter: ProgressReporter | None = None,
    max_attempts: int = TTS_MAX_ATTEMPTS,
) -> tuple[VoiceTrack, list[str]]:
    """合成整集旁白。chunk 独立落盘，重跑只补缺的那几个。"""
    reporter = reporter or NullProgressReporter()
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    planned_by_beat: list[tuple[Beat, list[tuple[str, float]]]] = []
    for beat in script.beats:
        planned = plan_chunks(beat)
        if not planned:
            warnings.append(f"beat {beat.id} 没有旁白文本，已跳过配音")
            continue
        planned_by_beat.append((beat, planned))

    total_chunks = sum(len(planned) for _, planned in planned_by_beat)

    chunks: list[VoiceChunk] = []
    serial = 0
    for beat, planned in planned_by_beat:
        for index, (text, hold_after) in enumerate(planned, start=1):
            serial += 1
            reporter.substep("voice", serial, total_chunks, text[:20])
            filename = f"chunk_{serial:03d}.mp3"
            out_path = voice_dir / filename
            label = f"beat {beat.id} 的第 {index} 个 chunk"
            if reuse and out_path.is_file() and out_path.stat().st_size > 0:
                duration = await asyncio.to_thread(probe_duration, out_path)
            else:
                duration = await synthesize_with_retry(
                    engine, text, out_path, label=label, max_attempts=max_attempts
                )
            chunks.append(
                VoiceChunk(
                    beat_id=beat.id,
                    index=index,
                    text=text,
                    path=filename,
                    duration=duration,
                    hold_after=hold_after,
                )
            )
    total = sum(chunk.duration + chunk.hold_after for chunk in chunks)
    return VoiceTrack(episode=episode, chunks=chunks, total_seconds=total), warnings
