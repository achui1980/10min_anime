"""无字幕间隙检测。长间隙 = 动画演出高光，v1 最重要的一条零成本规则。"""

from __future__ import annotations

from tenmin.models import DialogueTrack, Signal

MIN_GAP_SECONDS = 3.0
SPEECH_KINDS = ("dialogue", "monologue")
_STRENGTH_TABLE = ((15.0, 4), (8.0, 3), (0.0, 2))


def _strength(duration: float) -> int:
    for threshold, strength in _STRENGTH_TABLE:
        if duration >= threshold:
            return strength
    return 2


def _subtract(
    interval: tuple[float, float], blocks: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """从一个区间里挖掉若干个屏蔽区间，返回剩下的碎片。"""
    pieces = [interval]
    for block_start, block_end in blocks:
        nxt: list[tuple[float, float]] = []
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


def find_silent_gaps(
    track: DialogueTrack, min_seconds: float = MIN_GAP_SECONDS
) -> list[Signal]:
    spoken = sorted(
        (ln for ln in track.lines if ln.kind in SPEECH_KINDS and ln.text),
        key=lambda ln: ln.start,
    )
    if not spoken:
        return []

    blocks = [b for b in (track.op_range, track.ed_range) if b is not None]

    # 原始间隙：首行之前的黑场不算，末行之后到片长算
    raw: list[tuple[float, float, list[int]]] = []
    cursor = spoken[0].end
    cursor_idx = spoken[0].idx
    for line in spoken[1:]:
        if line.start > cursor:
            raw.append((cursor, line.start, [cursor_idx, line.idx]))
        if line.end > cursor:
            cursor = line.end
            cursor_idx = line.idx
    if track.duration > cursor:
        raw.append((cursor, track.duration, [cursor_idx]))

    signals: list[Signal] = []
    for start, end, anchors in raw:
        for piece_start, piece_end in _subtract((start, end), blocks):
            duration = piece_end - piece_start
            if duration < min_seconds:
                continue
            signals.append(
                Signal(
                    start=piece_start,
                    end=piece_end,
                    source="gap",
                    strength=_strength(duration),
                    detail=f"gap:{duration:.1f}s",
                    anchor_lines=anchors,
                )
            )
    signals.sort(key=lambda s: s.start)
    return signals
