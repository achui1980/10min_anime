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

# 「桶字数差分的标准差算作 0」的判据。见 find_density_shifts 里的注释。
_STDEV_EPS = 1e-9


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


def _bucket_runs(track: DialogueTrack, window: float) -> list[list[int]]:
    """把 [0, duration) 切成不重叠的满桶，扣掉 OP/ED，返回连续存活桶下标的分段。

    两处刻意的取舍：

    1. **尾部残桶整个丢掉。** duration 几乎不可能是 window 的整数倍（实测 11 集全部
       不是），原先的 `int(duration // window) + 1` 会造出一个只覆盖 `duration % window`
       秒的残桶。而 duration 就是最后一条 cue 的终点，那条 cue 必然起于最后一个满桶里，
       所以残桶字数恒为 0 —— 桶内字数又没按桶的实际时长归一化，于是差分末项是个大负值，
       稳定造出一条片尾假「节奏骤降」。
       备选方案是改用 `字数 / 桶实际时长` 的速率做差分（满桶之间等价于线性缩放，
       z 分数不变），但那会把问题翻转：`duration % window` 可能只有 0.001 秒，
       一条落进 0.001 秒残桶的行会让速率炸上天，变成一条同样固定的假「节奏骤升」。
       残桶最多丢掉不到 window 秒的片尾内容（实测全部落在 ED 里），代价有界且已知。

    2. **与 OP/ED 有任何重叠的桶整桶丢掉，而不是按重叠比例折算。** 半个桶被 OP 盖住时
       字数天然减半，折算只是把伪影换个大小、消不掉它。丢桶后按「连续存活段」分组，
       差分只在段内做：跨过 OP 空洞去比较两侧的桶同样是伪影（OP 前是开场铺垫、OP 后是
       正片，本来就不该直接做一阶差分）。
       gaps.py 早就显式扣掉了 op_range/ed_range，density.py 没扣，于是 OP 段落里
       kind=="credits" 的行被 spoken_lines 过滤掉、桶字数骤降为 0，OP 进入与离开各产生
       一条假 density_shift，再在 aggregate 里给相邻真高光加强度、污染排序。
    """
    bucket_count = int(track.duration // window)
    blocks = [b for b in (track.op_range, track.ed_range) if b is not None]
    runs: list[list[int]] = []
    for index in range(bucket_count):
        start = index * window
        end = start + window
        masked = any(b_start < end and start < b_end for b_start, b_end in blocks)
        if masked:
            continue
        if runs and runs[-1][-1] == index - 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def find_density_shifts(
    track: DialogueTrack, *, cfg: SignalsConfig = DEFAULT_SIGNALS
) -> list[Signal]:
    """不重叠 30s 满桶（扣掉 OP/ED）-> 每桶总字数 -> 段内一阶差分 -> 全局 z-score。"""
    if track.duration <= 0:
        return []
    window = cfg.shift_window_seconds
    # 守卫前移：满桶不足 3 个时连 2 个差分都凑不出，没必要先把全部行遍历一遍再退出。
    if int(track.duration // window) < 3:
        return []

    runs = _bucket_runs(track, window)
    if not runs:
        return []
    highest = max(run[-1] for run in runs)
    buckets = [0] * (highest + 1)
    for line in _speech_lines(track):
        index = int(line.start // window)
        if index <= highest:
            buckets[index] += _chars(line.text)

    # (差分值, 差分归属的桶下标)。只在同一存活段内做差分，不跨屏蔽空洞。
    diffs = [
        (buckets[b] - buckets[a], b) for run in runs for a, b in zip(run, run[1:], strict=False)
    ]
    # 屏蔽区可能把桶切得很碎，所以这条守卫是真的会命中的（不像满桶数 >= 3 那条）。
    if len(diffs) < 2:
        return []
    values = [value for value, _ in diffs]
    stdev = statistics.pstdev(values)
    # 刻意不用 `stdev == 0`：浮点相等判断漏掉 1e-16 之后 z = diff / 1e-16 会爆成天文
    # 数字，于是每一个桶边界都变成「节奏突变」。
    if stdev < _STDEV_EPS:
        return []
    mean = statistics.fmean(values)

    signals: list[Signal] = []
    for value, bucket_index in diffs:
        z = (value - mean) / stdev
        if abs(z) <= cfg.shift_z_threshold:
            continue
        start = bucket_index * window
        signals.append(
            Signal(
                start=start,
                # 满桶，所以右界一定 <= duration，不需要再 clamp。
                end=start + window,
                source="density_shift",
                strength=cfg.shift_strength,
                detail=f"shift:z={z:+.2f}",
                anchor_lines=[],
            )
        )
    signals.sort(key=lambda s: s.start)
    return signals


def find_density_signals(
    track: DialogueTrack, *, cfg: SignalsConfig = DEFAULT_SIGNALS
) -> list[Signal]:
    return [*find_low_density(track, cfg=cfg), *find_density_shifts(track, cfg=cfg)]
