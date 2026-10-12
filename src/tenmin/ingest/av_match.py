"""截图与音频匹配的纯函数：dHash、最佳帧、音频包络与跨集 OP 长度。

本模块不读取配置、不执行 IO；未来的图片读取与视频取帧层负责提供
9×8 灰度像素或 PCM 字节，并在匹配结果上应用阈值，与这里的计算逻辑分离。
"""

import itertools
import math
from collections.abc import Sequence
from statistics import median

import numpy as np

HASH_WIDTH = 9
HASH_HEIGHT = 8
HASH_BYTES = 72
HASH_BITS = 64
MAX_DISTANCE = 12


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
DEFAULT_OP_LENGTH = 90.0


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


def median_length(lengths: Sequence[float]) -> float | None:
    """剔除距原始中位数超过十秒的值再取中位数；没有保留值则返回 None。"""
    if len(lengths) == 0:
        return None
    center = median(lengths)
    kept = [value for value in lengths if abs(value - center) <= OUTLIER_SECONDS]
    if not kept:
        return None
    return median(kept)
