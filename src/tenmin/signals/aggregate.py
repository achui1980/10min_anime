"""把多路 Signal 聚成 Highlight，并组装 SignalReport。"""

from __future__ import annotations

from tenmin.models import DialogueTrack, Highlight, Signal, SignalReport
from tenmin.signals.density import (
    char_rate,
    find_density_signals,
    median_char_rate,
)
from tenmin.signals.gaps import find_silent_gaps

MIN_SEPARATION = 2.0
MAX_STRENGTH = 5
SUMMARY_MAX_CHARS = 30

# density_shift 是区域性证据：它的区间是 density.py 里死板的 30s 统计桶，
# 表达「这一片区域的叙事节奏换挡了」，不是一个精确的演出区间。
# 所以它只贡献强度加成与 trigger 标记，不参与聚簇的邻接链，也不决定 highlight 边界。
REGIONAL_SOURCE = "density_shift"


def _adjacent(
    a: tuple[float, float], b: tuple[float, float], min_separation: float
) -> bool:
    """两个区间重叠，或彼此间隔不超过 min_separation。"""
    return b[0] - a[1] <= min_separation and a[0] - b[1] <= min_separation


def _cluster(signals: list[Signal], min_separation: float) -> list[list[Signal]]:
    """先用精确信号连成邻接链，再把区域性信号挂到它重叠/相邻的那个簇上。

    邻接链刻意只由非 density_shift 信号驱动：一条 30s 宽的信号只要首尾相接就能
    无限接力，会把相隔几十秒的真间隙串成一个不存在的长段。
    """
    precise = sorted(
        (s for s in signals if s.source != REGIONAL_SOURCE),
        key=lambda s: (s.start, s.end),
    )
    clusters: list[list[Signal]] = []
    for signal in precise:
        if clusters and signal.start - max(s.end for s in clusters[-1]) <= min_separation:
            clusters[-1].append(signal)
        else:
            clusters.append([signal])

    # 精确簇的跨度快照。区域性信号只能挂进这些簇，不能挂进别的区域性信号自成的簇
    # —— 否则两个首尾相接的 30s 桶又会重新桥接起来。
    spans = [(min(s.start for s in c), max(s.end for s in c)) for c in clusters]
    regional = sorted(
        (s for s in signals if s.source == REGIONAL_SOURCE),
        key=lambda s: (s.start, s.end),
    )
    for signal in regional:
        host = next(
            (
                index
                for index, span in enumerate(spans)
                if _adjacent(span, (signal.start, signal.end), min_separation)
            ),
            None,
        )
        if host is None:
            clusters.append([signal])
        else:
            clusters[host].append(signal)
    return clusters


def _boundary_signals(cluster: list[Signal]) -> list[Signal]:
    """决定 highlight 时间码的信号子集。

    只要簇里有精确信号就只看它们；整簇都是 density_shift 时才退回用桶区间本身 ——
    这类 highlight 表达的就是「这一区域节奏变了」，没有更精确的时间码可用。
    """
    precise = [s for s in cluster if s.source != REGIONAL_SOURCE]
    return precise or cluster


def _summary(cluster: list[Signal], track: DialogueTrack | None) -> str:
    start = min(s.start for s in cluster)
    end = max(s.end for s in cluster)
    duration = end - start
    strongest = max(cluster, key=lambda s: s.strength)
    if strongest.source == "gap":
        return f"无台词演出段 {duration:.1f}s"

    anchors = {idx for s in cluster for idx in s.anchor_lines}
    candidates = []
    if track is not None:
        candidates = [
            ln
            for ln in track.lines
            if ln.idx in anchors and ln.text and ln.duration > 0
        ]
    if not candidates:
        return f"低语速片段 {duration:.1f}s"
    slowest = min(candidates, key=char_rate)
    return slowest.text[:SUMMARY_MAX_CHARS]


def aggregate(
    signals: list[Signal],
    track: DialogueTrack | None,
    min_separation: float = MIN_SEPARATION,
) -> list[Highlight]:
    highlights: list[Highlight] = []
    for cluster in _cluster(signals, min_separation):
        bounds = _boundary_signals(cluster)
        # 强度、trigger、anchor 都算整簇（density_shift 照样贡献）；只有边界只看 bounds。
        base = max(s.strength for s in cluster)
        extra = len({s.source for s in cluster}) - 1
        strength = min(base + extra, MAX_STRENGTH)
        triggers = sorted({s.detail for s in cluster if s.detail})
        anchor_lines = sorted({idx for s in cluster for idx in s.anchor_lines})
        highlights.append(
            Highlight(
                start=min(s.start for s in bounds),
                end=max(s.end for s in bounds),
                strength=strength,
                triggers=triggers,
                summary=_summary(bounds, track),
                anchor_lines=anchor_lines,
            )
        )
    return sorted(highlights, key=lambda h: h.start)


def build_report(track: DialogueTrack) -> SignalReport:
    gaps = find_silent_gaps(track)
    signals = [*gaps, *find_density_signals(track)]
    return SignalReport(
        episode=track.episode,
        silent_gaps=gaps,
        median_char_rate=median_char_rate(track),
        highlights=aggregate(signals, track),
    )
