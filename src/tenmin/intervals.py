"""区间代数 + 「这一行算有人在说话」的单一定义。

本模块刻意只依赖 `tenmin.models`，不 import signals / ingest 里的任何东西，
所以它是一个中立的底层层：ingest 与 signals 都可以放心 import 它，不构成
反向依赖（`ingest/credits.py` 原先为了避开「ingest 不能依赖 signals」而
把静默间隙算法重抄了一遍，这个模块就是为了消掉那份重复而存在的）。

刻意**不**提供的东西：

- `silent_gaps` 没有 `min_seconds` 参数。`signals/gaps.py` 必须先扣掉 OP/ED
  再按时长筛（顺序反了会把「被 OP 切开后仍够长的碎片」误杀），`ingest/credits.py`
  筛的是 OP 静区时长窗而不是下限。放一个两边都不能用的参数只会变成陷阱。
- 不构造 `Signal` / 不做强度分档 / 不做窗口过滤。这些是上层语义，留给调用方。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

from tenmin.models import SPEECH_KINDS, DialogueLine

# 半开区间 [start, end)，秒。
Interval = tuple[float, float]


def is_spoken(line: DialogueLine) -> bool:
    """唯一的「这一行算有人在说话」谓词。三个条件缺一不可：

    1. `kind` 在 `SPEECH_KINDS` 里；
    2. 文本 strip 后非空；
    3. 时长严格为正 —— `srt_parser` 会把 `end < start` 的坏 cue 夹成零时长
       （`end = start`），这类行没有任何时间跨度，既不该推进静默游标，
       也不该进字密度统计。
    """
    return line.kind in SPEECH_KINDS and bool(line.text.strip()) and line.duration > 0


def spoken_lines(lines: Iterable[DialogueLine]) -> list[DialogueLine]:
    """筛出「有人说话」的行，按起点升序返回。原列表不受影响。"""
    return sorted((ln for ln in lines if is_spoken(ln)), key=lambda ln: ln.start)


class SilentGap(NamedTuple):
    """一段没人说话的区间，以及它两侧的说话行。

    `before` 是把游标推到 `start` 的那一行（不一定是时间上紧邻的前一行：
    字幕重叠时游标由跨度最长的那一行持有）。`after` 是间隙右侧的第一行，
    片尾间隙没有右侧行，为 `None`。
    """

    start: float
    end: float
    before: DialogueLine
    after: DialogueLine | None

    @property
    def duration(self) -> float:
        return self.end - self.start


def silent_gaps(
    lines: Iterable[DialogueLine], *, duration: float | None = None
) -> list[SilentGap]:
    """列出全部无人说话的间隙，不做任何时长筛选。

    首行之前的黑场/前奏不算间隙。给了 `duration` 时，末行之后到片长的那段
    也算一个间隙（`after is None`）；不给就只算行与行之间的。
    游标用 `max(cursor, line.end)` 单调推进，重叠字幕不会算出负长度间隙。
    """
    spoken = spoken_lines(lines)
    if not spoken:
        return []

    gaps: list[SilentGap] = []
    cursor = spoken[0].end
    cursor_line = spoken[0]
    for line in spoken[1:]:
        if line.start > cursor:
            gaps.append(SilentGap(cursor, line.start, cursor_line, line))
        if line.end > cursor:
            cursor = line.end
            cursor_line = line
    if duration is not None and duration > cursor:
        gaps.append(SilentGap(cursor, duration, cursor_line, None))
    return gaps


def group_adjacent[T](
    items: Iterable[T], *, bounds: Callable[[T], Interval], max_gap: float
) -> list[list[T]]:
    """把条目按时间邻接分组：与当前组右界的间隔 <= max_gap 就并进去。

    组的右界是**组内已见的最大 end**（`current_max_end`），不是上一条的 end ——
    否则一个被包住的短区间会让右界回退，把本该同组的下一条切成新组。
    该变量随 append 增量更新，全程 O(n log n)（只有排序），不重扫当前组。
    """
    ordered = sorted(items, key=bounds)
    groups: list[list[T]] = []
    current_max_end = 0.0
    for item in ordered:
        start, end = bounds(item)
        if groups and start - current_max_end <= max_gap:
            groups[-1].append(item)
            if end > current_max_end:
                current_max_end = end
        else:
            groups.append([item])
            current_max_end = end
    return groups


def merge_intervals(intervals: Iterable[Interval], *, max_gap: float = 0.0) -> list[Interval]:
    """排序后把间隔 <= max_gap 的相邻区间合并成一个跨度，按起点升序返回。

    默认 `max_gap=0.0`：只合并重叠或首尾相接的区间。
    刻意不修正、也不拒绝 `start > end` 的反向区间 —— 调用方有责任传合法区间，
    这里只保证同样的输入给出同样的输出。

    「加一道 `start > end` 就抛异常的校验」考虑过，结论是不加：那会把一个全函数变成
    偏函数，对所有调用方都是行为改动（原来返回、现在抛），属于判定改动而不是整洁改动。
    而且两个现存调用方喂进来的都是模型字段（`ingest/credits` 的 DialogueLine、
    `render/audio` 的字幕 cue），反向区间在这一层根本构造不出来：
    `srt_parser` 已经把 `end < start` 的坏 cue 夹成零时长并计入 `clamped_cues`。
    真要加，该加在能报出「哪条数据坏了」的那一层，不是这里。
    """
    return [
        (group[0][0], max(end for _, end in group))
        for group in group_adjacent(intervals, bounds=lambda iv: iv, max_gap=max_gap)
    ]


def subtract(interval: Interval, blocks: Sequence[Interval]) -> list[Interval]:
    """从一个区间里挖掉若干个屏蔽区间，返回剩下的碎片（零长度碎片会被丢掉）。"""
    pieces = [interval]
    for block_start, block_end in blocks:
        nxt: list[Interval] = []
        for start, end in pieces:
            if block_end <= start or block_start >= end:
                nxt.append((start, end))
                continue
            if start < block_start:
                nxt.append((start, min(block_start, end)))
            if end > block_end:
                nxt.append((max(block_end, start), end))
        pieces = nxt
    return [(s, e) for s, e in pieces if e > s]
