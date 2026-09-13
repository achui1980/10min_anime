"""把多路 Signal 聚成 Highlight，并组装 SignalReport。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from tenmin.config import DEFAULT_SIGNALS, SignalsConfig
from tenmin.intervals import group_adjacent
from tenmin.models import (
    STRENGTH_MAX,
    DialogueLine,
    DialogueTrack,
    Highlight,
    Signal,
    SignalReport,
)
from tenmin.signals.density import (
    char_rate,
    find_density_signals,
    line_rates,
    median_char_rate,
)
from tenmin.signals.gaps import find_silent_gaps

# 强度上限不是本模块自己的旋钮：它就是 models.Highlight.strength 的 Field(le=...) 上界。
# 两处必须一致，否则 min(base + extra, ...) 一超界就直接 ValidationError，
# 所以这里直接复用 models.STRENGTH_MAX，不再写第二份字面量。
MAX_STRENGTH = STRENGTH_MAX

# density_shift 是区域性证据：它的区间是 density.py 里死板的 30s 统计桶，
# 表达「这一片区域的叙事节奏换挡了」，不是一个精确的演出区间。
# 所以它只贡献强度加成与 trigger 标记，不参与聚簇的邻接链，也不决定 highlight 边界。
REGIONAL_SOURCE = "density_shift"


def _adjacent(
    a: tuple[float, float], b: tuple[float, float], min_separation: float
) -> bool:
    """两个区间重叠，或彼此间隔不超过 min_separation。

    刻意**不**复用 intervals.group_adjacent 的邻接判据，两者语义不同、不是重复：

    - `group_adjacent` 是单向的（`start - current_max_end <= max_gap`）。它按起点排序
      后单调向前扫，只问「下一条离已见右界够近吗」，用来把一串区间**切成组**。
    - 这里是双向的（`b[0] - a[1] <= sep and a[0] - b[1] <= sep`）。宿主簇的跨度 span
      与区域性信号的先后顺序不定 —— 一个 30s 统计桶可能整个落在某个精确簇**之前**，
      单向判据会漏掉这种情况，于是 density_shift 自成一簇，凭空多出一个
      「只有节奏换挡、没有任何精确证据」的 highlight。

    换成单向判据会静默改变 highlight 的数量与强度，别当成重复给合并掉。
    """
    return b[0] - a[1] <= min_separation and a[0] - b[1] <= min_separation


class _Span(NamedTuple):
    """一个簇的时间跨度。聚簇阶段算一次就固化，别再对同一批信号反复 min/max。"""

    start: float
    end: float


def _span_of(signals: Sequence[Signal]) -> _Span:
    return _Span(min(s.start for s in signals), max(s.end for s in signals))


class _Cluster(NamedTuple):
    """一个聚簇及其派生量。

    - `signals`：**整簇**，含挂上来的 density_shift。强度、trigger、anchor 都算它。
    - `bounds`：决定 highlight 时间码的子集。只要簇里有精确信号就只看它们；整簇都是
      density_shift 时才退回用桶区间本身 —— 那类 highlight 表达的就是「这一区域节奏
      变了」，没有更精确的时间码可用。
    - `span`：`bounds` 的跨度。对精确簇来说它就是挂载前算好的那个跨度（挂 regional
      不会动 bounds），所以整条流程只算一次。
    """

    signals: list[Signal]
    bounds: list[Signal]
    span: _Span


def _cluster_precise(signals: Sequence[Signal], min_separation: float) -> list[list[Signal]]:
    """只用精确（非 density_shift）信号连成邻接链。

    邻接链刻意不让 density_shift 参与：一条 30s 宽的信号只要首尾相接就能无限接力，
    会把相隔几十秒的真间隙串成一个不存在的长段。
    """
    return group_adjacent(
        (s for s in signals if s.source != REGIONAL_SOURCE),
        bounds=lambda s: (s.start, s.end),
        max_gap=min_separation,
    )


def _attach_regional(
    signals: Sequence[Signal], precise: list[list[Signal]], min_separation: float
) -> list[_Cluster]:
    """把区域性信号挂到它重叠/相邻的那个精确簇上，挂不上的自成一簇。

    区域性信号只能挂进精确簇，不能挂进别的区域性信号自成的簇 —— 否则两个首尾相接的
    30s 桶又会重新桥接起来。所以跨度快照 `spans` 取的是**挂载前**的精确簇。

    挂载用双指针而不是对 spans 全量线性扫（原实现是 O(R × C)）。两条单调性撑着它：

    1. `group_adjacent` 按起点排序后单调分组，新组的条件是 `start - 组内最大右界
       > max_gap`，所以 spans 按 start **与** end 都严格升序、互不相交。
    2. `regional` 按 start 升序遍历。

    于是「因为太靠左（`signal.start - span.end > sep`）而被排除」的 span 一旦被越过，
    后面 start 更大的 signal 更不可能用到它 —— 游标只进不退。

    而且只需要检查游标那一个 span：`_adjacent` 的左条件在 spans 上是「后缀成立」
    （span.end 递增），右条件是「前缀成立」（span.start 递增），两者的交是一段
    连续区间 [cursor, m]。原实现取的是下标最小的相邻簇，那就恰好是 cursor。
    """
    spans = [_span_of(group) for group in precise]
    members = [list(group) for group in precise]
    orphans: list[_Cluster] = []

    cursor = 0
    for signal in sorted(
        (s for s in signals if s.source == REGIONAL_SOURCE),
        key=lambda s: (s.start, s.end),
    ):
        probe = (signal.start, signal.end)
        while cursor < len(spans) and signal.start - spans[cursor].end > min_separation:
            cursor += 1
        if cursor < len(spans) and _adjacent(spans[cursor], probe, min_separation):
            members[cursor].append(signal)
        else:
            orphans.append(
                _Cluster([signal], [signal], _Span(signal.start, signal.end))
            )

    return [
        # bounds 就是挂载前的那份精确信号列表：整簇 = precise + 追加的 regional，
        # 按 source 过滤掉 regional 之后逐元素等于 precise，顺序也一致。
        _Cluster(members[index], group, spans[index])
        for index, group in enumerate(precise)
    ] + orphans


def _cluster(signals: Sequence[Signal], min_separation: float) -> list[_Cluster]:
    precise = _cluster_precise(signals, min_separation)
    return _attach_regional(signals, precise, min_separation)


class _TrackIndex(NamedTuple):
    """track.lines 的一次性索引：anchor 行号 -> 它在 track.lines 里的位置。

    值是 list 而不是单个位置：idx 是 **cue 级**的键，`split_dual_track` 从同一条 cue
    拆出的台词与内心独白共享同一个 idx。

    刻意**不**像 `script/validate.py` 的 `AnchorIndex` 那样把 `merged_from` 里的旧行号
    也指向该行 —— `_summary` 原来就只比 `ln.idx in anchors`，加进来会静默改变
    summary 选到的那条行。两边判据不同不是重复，别合并。
    """

    lines: list[DialogueLine]
    positions: dict[int, list[int]]


def _index_track(track: DialogueTrack) -> _TrackIndex:
    """O(L) 建索引，取代 `_summary` 里每簇一次的 O(L) 全量扫描。

    一集 L≈400 行、H≈40 个簇，原实现是 O(H×L)。实测 11 集真实素材：
    `for ln in track.lines` 一共比较 143752 次，只为找出 617 条候选行。
    """
    positions: dict[int, list[int]] = {}
    for position, line in enumerate(track.lines):
        positions.setdefault(line.idx, []).append(position)
    return _TrackIndex(track.lines, positions)


def _summary(
    bounds: list[Signal], span: _Span, index: _TrackIndex | None, cfg: SignalsConfig
) -> str:
    duration = span.end - span.start
    strongest = max(bounds, key=lambda s: s.strength)
    if strongest.source == "gap":
        return f"无台词演出段 {duration:.1f}s"

    anchors = {idx for s in bounds for idx in s.anchor_lines}
    candidates: list[DialogueLine] = []
    if index is not None:
        # 位置先去重再升序 —— 还原成原实现「按 track.lines 顺序过滤」的顺序，
        # 因为下面 min() 平手时取的是先出现的那条。
        candidates = [
            line
            for line in (
                index.lines[position]
                for position in sorted(
                    {p for idx in anchors for p in index.positions.get(idx, ())}
                )
            )
            if line.text and line.duration > 0
        ]
    if not candidates:
        return f"低语速片段 {duration:.1f}s"
    # char_rate 直接现算，不做速率查表：item 1 已经把它里面的正则换成零分配的
    # isspace 计数，而实测 11 集里 617 次候选行只有 31 次是重复的（586 条不同的行），
    # 为省这 31 次去挂一份 memo 是负收益。
    slowest = min(candidates, key=char_rate)
    return slowest.text[: cfg.summary_max_chars]


def aggregate(
    signals: list[Signal],
    track: DialogueTrack | None,
    *,
    cfg: SignalsConfig = DEFAULT_SIGNALS,
) -> list[Highlight]:
    highlights: list[Highlight] = []
    index = None if track is None else _index_track(track)
    for cluster in _cluster(signals, cfg.min_separation):
        # 强度、trigger、anchor 都算**整簇**（density_shift 照样贡献）；时间码与
        # summary 都只看 bounds。
        #
        # summary 跟着 bounds 走是刻意的，别按上面那半句「只有边界只看 bounds」去改：
        # summary 里的秒数就是 `span.end - span.start`，拿整簇算会让它自称 30 秒
        # 而 highlight 自己的时间窗只有 5.3 秒（实测 11 集 338 个簇里有 35 个如此），
        # 等于把自相矛盾的素材喂给 LLM。
        # anchor 那半句对 bounds/整簇其实无差别：density_shift 的 anchor_lines 恒为空
        # （见 density.find_density_shifts），所以两种取法给出同一个集合。
        base = max(s.strength for s in cluster.signals)
        extra = len({s.source for s in cluster.signals}) - 1
        strength = min(base + extra, MAX_STRENGTH)
        triggers = sorted({s.detail for s in cluster.signals if s.detail})
        anchor_lines = sorted({idx for s in cluster.signals for idx in s.anchor_lines})
        highlights.append(
            Highlight(
                start=cluster.span.start,
                end=cluster.span.end,
                strength=strength,
                triggers=triggers,
                summary=_summary(cluster.bounds, cluster.span, index, cfg),
                anchor_lines=anchor_lines,
            )
        )
    return sorted(highlights, key=lambda h: h.start)


def build_report(
    track: DialogueTrack, *, cfg: SignalsConfig = DEFAULT_SIGNALS
) -> SignalReport:
    gaps = find_silent_gaps(track, cfg=cfg)
    # 字数与语速只扫一遍，median 与两个 detector 共用这一份快照。见 density.line_rates。
    rates = line_rates(track)
    median = median_char_rate(track, rates=rates)
    signals = [*gaps, *find_density_signals(track, cfg=cfg, rates=rates, median=median)]
    return SignalReport(
        episode=track.episode,
        silent_gaps=gaps,
        median_char_rate=median,
        highlights=aggregate(signals, track, cfg=cfg),
    )
