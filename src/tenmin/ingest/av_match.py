"""截图与音频匹配：媒体抽取、dHash 定位、音频包络与跨集 OP/ED 终点。

匹配结果不应用阈值，由调用方决定是否接受。
"""

import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from tenmin.render import ffmpeg

HASH_WIDTH = 9
HASH_HEIGHT = 8
HASH_BYTES = 72
HASH_BITS = 64
MAX_DISTANCE = 12
COARSE_FPS = 2
FINE_FPS = 10
FINE_RADIUS_SECONDS = 1
_SCALE_CHAIN = "scale=9:8:flags=area,format=gray"
_SHOWINFO = re.compile(r"\[[^\]\n]*showinfo[^\]]*\].*?\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:(\S+)")


def dhash(gray: bytes | Sequence[int] | np.ndarray) -> int:
    """将 9×8 灰度像素的左右比较结果按行优先编码为 64 位整数。"""
    if isinstance(gray, (bytes, bytearray)):
        pixels = np.frombuffer(gray, dtype=np.uint8)
    else:
        pixels = np.asarray(gray)
    if pixels.size != HASH_BYTES:
        raise ValueError(f"dHash 需要 {HASH_BYTES} 个灰度像素，收到 {pixels.size} 个")
    grid = pixels.reshape(HASH_HEIGHT, HASH_WIDTH).astype(np.int16)
    comparisons = grid[:, :-1] > grid[:, 1:]
    result = 0
    for bit in comparisons.ravel():
        result = (result << 1) | int(bit)
    return result


def hamming(a: int, b: int) -> int:
    """返回两个哈希之间不同位的数量。"""
    return (a ^ b).bit_count()


def best_frame(hashes: Sequence[int], ref: int) -> tuple[int, int]:
    """返回最小距离的帧索引及距离；同距取首帧，不应用匹配阈值。"""
    if len(hashes) == 0:
        raise ValueError("没有可比较的帧")
    return min(
        ((index, hamming(value, ref)) for index, value in enumerate(hashes)),
        key=lambda match: match[1],
    )


def _hash_frames(args: list[str], *, ffmpeg_path: str) -> tuple[list[int], str]:
    """共享的 9×8 灰度帧读取层，保留 stderr 供时间戳解析与诊断。"""
    hashes: list[int] = []
    stderr = ffmpeg.run_raw_frames(
        args,
        frame_size=HASH_BYTES,
        on_frame=lambda frame: hashes.append(dhash(frame)),
        ffmpeg=ffmpeg_path,
    )
    return hashes, stderr


def sample_hashes(
    video: Path, start: float, end: float, fps: float, *, ffmpeg_path: str = ffmpeg.FFMPEG
) -> list[tuple[float, int]]:
    """抽取有界片段的哈希；showinfo 的相对 PTS 加回片段起点。"""
    if end <= start:
        return []
    hashes, stderr = _hash_frames(
        [
            "-hide_banner",
            "-v",
            "info",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={fps:g},{_SCALE_CHAIN},showinfo",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        ffmpeg_path=ffmpeg_path,
    )
    try:
        times = [start + float(match.group(1)) for match in _SHOWINFO.finditer(stderr)]
    except ValueError as error:
        raise ffmpeg.FFmpegError(f"ffmpeg showinfo 时间戳无效：\n{ffmpeg.tail(stderr)}") from error
    if len(times) != len(hashes):
        raise ffmpeg.FFmpegError(
            f"ffmpeg 帧数 {len(hashes)} 与时间戳数 {len(times)} 不一致：\n{ffmpeg.tail(stderr)}"
        )
    # fps 会按采样网格舍入并补帧；先验证完整输出对齐，再剔除搜索窗口外的帧。
    return [(time, value) for time, value in zip(times, hashes, strict=True) if start <= time < end]


def image_hash(image: Path, *, ffmpeg_path: str = ffmpeg.FFMPEG) -> int:
    """用与视频相同的缩放链读取一张截图。"""
    hashes, stderr = _hash_frames(
        [
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(image),
            "-map",
            "0:v:0",
            "-vf",
            _SCALE_CHAIN,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        ffmpeg_path=ffmpeg_path,
    )
    if len(hashes) != 1:
        raise ffmpeg.FFmpegError(
            f"截图需要恰好一帧，收到 {len(hashes)} 帧：\n{ffmpeg.tail(stderr)}"
        )
    return hashes[0]


@dataclass(frozen=True)
class Match:
    time: float
    distance: int


def locate(
    video: Path, ref: int, *, start: float, end: float, ffmpeg_path: str = ffmpeg.FFMPEG
) -> Match | None:
    """先粗采样，再在最佳帧前后一秒细采样；同距取最早时间。"""
    coarse = sample_hashes(video, start, end, COARSE_FPS, ffmpeg_path=ffmpeg_path)
    if not coarse:
        return None
    index, _ = best_frame([value for _, value in coarse], ref)
    coarse_best = coarse[index]
    center = coarse_best[0]
    fine = sample_hashes(
        video,
        max(start, center - FINE_RADIUS_SECONDS),
        min(end, center + FINE_RADIUS_SECONDS),
        FINE_FPS,
        ffmpeg_path=ffmpeg_path,
    )
    candidates = sorted([*fine, coarse_best], key=lambda sample: sample[0])
    index, distance = best_frame([value for _, value in candidates], ref)
    return Match(time=candidates[index][0], distance=distance)


SAMPLE_RATE = 8000
ENVELOPE_RATE = 10
_HOP = SAMPLE_RATE // ENVELOPE_RATE
BANDS = 16
# 每个 FFT bin 为 10 Hz；忽略 40 Hz 以下的低频，保留至 Nyquist。
_BAND_EDGES = np.geomspace(4, _HOP // 2 + 1, BANDS + 1).astype(int)
REFERENCE_SECONDS = 150.0
SIMILARITY_WINDOW = ENVELOPE_RATE
SIMILARITY_THRESHOLD = 0.5
DROP_SECONDS = 2.0
MAX_LAG_FRAMES = ENVELOPE_RATE
ALIGN_FRAMES = 20 * ENVELOPE_RATE
OUTLIER_SECONDS = 10.0
ED_MATCH_SECONDS = 5.0


def read_pcm(
    video: Path, start: float, duration: float, *, ffmpeg_path: str = ffmpeg.FFMPEG
) -> bytes:
    """读取音频匹配所需的 8kHz 单声道 PCM。"""
    return ffmpeg.run_pcm(
        video, start=start, duration=duration, sample_rate=SAMPLE_RATE, ffmpeg=ffmpeg_path
    )


def audio_envelope(pcm: bytes) -> np.ndarray:
    """s16le 单声道 8 kHz PCM 转为每秒 10 帧的对数频带能量。

    丢弃末尾单个字节及不足 800 个采样的一帧。Hann 窗后按频带求功率谱之和，
    log10(能量 + 1) 让静音帧为零。
    """
    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float64)
    count = samples.size // _HOP
    if count == 0:
        return np.zeros((0, BANDS))
    frames = samples[: count * _HOP].reshape(count, _HOP) * np.hanning(_HOP)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    energies = np.stack(
        [power[:, lo:hi].sum(axis=1) for lo, hi in itertools.pairwise(_BAND_EDGES)], axis=1
    )
    return np.log10(energies + 1.0)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """拉平后的 Pearson 相关；任一段方差为零时返回零。"""
    x = a.ravel() - a.mean()
    y = b.ravel() - b.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator == 0:
        return 0.0
    return float(x @ y) / denominator


def _span(ref: np.ndarray, env: np.ndarray, lag: int) -> tuple[int, int]:
    """ref[i] 与 env[i + lag] 对齐时，参考包络可比较的区间。"""
    return max(0, -lag), min(len(ref), len(env) - lag)


def _best_lag(ref: np.ndarray, env: np.ndarray) -> int:
    """按前 200 个重叠帧在 ±10 帧内找最佳错位，同分优先绝对值小的。"""
    best_lag, best_score = 0, -math.inf
    for lag in sorted(range(-MAX_LAG_FRAMES, MAX_LAG_FRAMES + 1), key=abs):
        first, stop = _span(ref, env, lag)
        stop = min(stop, first + ALIGN_FRAMES)
        if stop - first < SIMILARITY_WINDOW:
            continue
        score = _correlation(ref[first:stop], env[first + lag : stop + lag])
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag


def op_length(ref_env: np.ndarray, env: np.ndarray) -> float | None:
    """返回 env 时间轴上的 OP 终点秒数；从未对齐或从未持续掉落则返回 None。

    先去掉各频带的时间均值并搜索错位，再逐帧滑动一秒窗。见过相关 ≥0.5 后，
    连续 20 个低相关窗口才判结束；终点取首个低相关窗的中心。
    """
    ref = ref_env - ref_env.mean(axis=0) if len(ref_env) else ref_env
    other = env - env.mean(axis=0) if len(env) else env
    lag = _best_lag(ref, other)
    first, stop = _span(ref, other, lag)
    hold = round(DROP_SECONDS * ENVELOPE_RATE)
    matched = False
    run = 0
    for i in range(first, stop - SIMILARITY_WINDOW + 1):
        value = _correlation(
            ref[i : i + SIMILARITY_WINDOW], other[i + lag : i + lag + SIMILARITY_WINDOW]
        )
        if value >= SIMILARITY_THRESHOLD:
            matched = True
            run = 0
            continue
        if not matched:
            continue
        run += 1
        if run == hold:
            end = i - hold + 1 + lag + SIMILARITY_WINDOW // 2
            return end / ENVELOPE_RATE
    return None


def ed_endpoints(
    ref_env: np.ndarray, env: np.ndarray, ref_seconds: float, seconds: float
) -> tuple[float | None, float | None]:
    """返回两条各自时间轴上的 ED 终点，None 表示该侧证据不足。

    沿用 OP 的去均值、错位与滑窗相关，但要求至少五秒连续匹配才能测尾。
    静音或频带能量没有有效时间变化的窗不提供匹配证据；已有歌曲证据后算掉落。
    持续掉落时分别返回歌曲终点；持续匹配耗尽时只确认确实覆盖到的文件 EOF，
    不能把参考读段边界或较短集 EOF 当成较长集歌曲终点。
    """
    ref = ref_env - ref_env.mean(axis=0) if len(ref_env) else ref_env
    other = env - env.mean(axis=0) if len(env) else env
    lag = _best_lag(ref, other)
    first, stop = _span(ref, other, lag)
    hold = round(DROP_SECONDS * ENVELOPE_RATE)
    sustained = round(ED_MATCH_SECONDS * ENVELOPE_RATE) - SIMILARITY_WINDOW + 1
    matched = False
    high_run = low_run = 0
    for i in range(first, stop - SIMILARITY_WINDOW + 1):
        ref_window = ref[i : i + SIMILARITY_WINDOW]
        other_window = other[i + lag : i + lag + SIMILARITY_WINDOW]
        # 全段去均值会把静音变成固定的非零频带向量。频带间的方差不算时间证据；
        # 1e-8 仅排除浮点去均值残差，两侧都须有实际的频带能量时间变化。
        varying = all(
            np.any(np.ptp(window, axis=0) > 1e-8) for window in (ref_window, other_window)
        )
        # 匹配本身也只比较时间变化，避免静音/固定音色改变全段均值后，
        # 把后续两段无关音频的共同频带偏置误认成持续歌曲匹配。
        value = (
            _correlation(
                ref_window - ref_window.mean(axis=0),
                other_window - other_window.mean(axis=0),
            )
            if varying else 0.0
        )
        if value >= SIMILARITY_THRESHOLD:
            high_run += 1
            matched = matched or high_run >= sustained
            low_run = 0
            continue
        high_run = 0
        if not matched:
            continue
        low_run += 1
        if low_run == hold:
            end = (i - hold + 1 + SIMILARITY_WINDOW // 2) / ENVELOPE_RATE
            return min(end, ref_seconds), min(end + lag / ENVELOPE_RATE, seconds)
    if not matched or low_run or high_run < sustained:
        return None, None
    # 包络舍弃不足一帧的尾部。只容许这个量化差，不容许参考读长截断。
    tolerance = 1 / ENVELOPE_RATE + 1e-6
    return (
        ref_seconds if abs(stop / ENVELOPE_RATE - ref_seconds) <= tolerance else None,
        seconds if abs((stop + lag) / ENVELOPE_RATE - seconds) <= tolerance else None,
    )


def median_length(lengths: Sequence[float]) -> float | None:
    """剔除距原始中位数超过十秒的值再取中位数；没有保留值则返回 None。"""
    if len(lengths) == 0:
        return None
    center = median(lengths)
    kept = [value for value in lengths if abs(value - center) <= OUTLIER_SECONDS]
    if not kept:
        return None
    return median(kept)
