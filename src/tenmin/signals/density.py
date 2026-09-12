"""字密度信号。低字密度 = 情绪爆发；密度突变 = 叙事节奏换挡。"""

from __future__ import annotations

import re
import statistics

from tenmin.config import DEFAULT_SIGNALS, SignalsConfig
from tenmin.intervals import spoken_lines
from tenmin.models import DialogueLine, DialogueTrack, Signal

# 阈值的权威定义在 tenmin.config.SignalsConfig；这个别名只为老调用点/文档保留。
LOW_DENSITY_RATIO = DEFAULT_SIGNALS.low_density_ratio

_WHITESPACE = re.compile(r"\s+")


def _chars(text: str) -> int:
    return len(_WHITESPACE.sub("", text))


def char_rate(line: DialogueLine) -> float:
    duration = line.duration
    if duration <= 0:
        return 0.0
    return _chars(line.text) / duration


def _speech_lines(track: DialogueTrack) -> list[DialogueLine]:
    return spoken_lines(track.lines)


def median_char_rate(track: DialogueTrack) -> float:
    rates = [char_rate(ln) for ln in _speech_lines(track)]
    if not rates:
        return 0.0
    return statistics.median(rates)


def find_low_density(
    track: DialogueTrack, *, cfg: SignalsConfig = DEFAULT_SIGNALS
) -> list[Signal]:
    median = median_char_rate(track)
    if median <= 0:
        return []
    threshold = median * cfg.low_density_ratio
    signals: list[Signal] = []
    for line in _speech_lines(track):
        if line.duration < cfg.low_density_min_seconds:
            continue
        rate = char_rate(line)
        if rate >= threshold:
            continue
        signals.append(
            Signal(
                start=line.start,
                end=line.end,
                source="low_density",
                strength=cfg.low_density_strength,
                detail=f"density:{rate:.2f}",
                anchor_lines=[line.idx],
            )
        )
    return signals


def find_density_shifts(
    track: DialogueTrack, *, cfg: SignalsConfig = DEFAULT_SIGNALS
) -> list[Signal]:
    """不重叠 30s 桶 -> 每桶总字数 -> 一阶差分 -> 全局 z-score。"""
    if track.duration <= 0:
        return []
    window = cfg.shift_window_seconds
    z_threshold = cfg.shift_z_threshold
    bucket_count = int(track.duration // window) + 1
    buckets = [0] * bucket_count
    for line in _speech_lines(track):
        index = min(int(line.start // window), bucket_count - 1)
        buckets[index] += _chars(line.text)
    if bucket_count < 3:
        return []

    diffs = [buckets[i] - buckets[i - 1] for i in range(1, bucket_count)]
    if len(diffs) < 2:
        return []
    stdev = statistics.pstdev(diffs)
    if stdev == 0:
        return []
    mean = statistics.fmean(diffs)

    signals: list[Signal] = []
    for offset, diff in enumerate(diffs):
        z = (diff - mean) / stdev
        if abs(z) <= z_threshold:
            continue
        bucket_index = offset + 1
        start = bucket_index * window
        end = min(start + window, track.duration)
        if end <= start:
            continue
        signals.append(
            Signal(
                start=start,
                end=end,
                source="density_shift",
                strength=cfg.shift_strength,
                detail=f"shift:z={z:+.2f}",
                anchor_lines=[],
            )
        )
    return signals


def find_density_signals(
    track: DialogueTrack, *, cfg: SignalsConfig = DEFAULT_SIGNALS
) -> list[Signal]:
    return [*find_low_density(track, cfg=cfg), *find_density_shifts(track, cfg=cfg)]
