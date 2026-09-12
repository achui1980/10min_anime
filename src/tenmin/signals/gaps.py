"""无字幕间隙检测。长间隙 = 动画演出高光，v1 最重要的一条零成本规则。"""

from __future__ import annotations

from tenmin.intervals import silent_gaps, subtract
from tenmin.models import DialogueTrack, Signal

MIN_GAP_SECONDS = 3.0
_STRENGTH_TABLE = ((15.0, 4), (8.0, 3), (0.0, 2))


def _strength(duration: float) -> int:
    for threshold, strength in _STRENGTH_TABLE:
        if duration >= threshold:
            return strength
    return 2


def find_silent_gaps(
    track: DialogueTrack, min_seconds: float = MIN_GAP_SECONDS
) -> list[Signal]:
    blocks = [b for b in (track.op_range, track.ed_range) if b is not None]

    signals: list[Signal] = []
    # 时长筛选必须发生在扣掉 OP/ED 之后：先筛会把「被 OP 切开后仍够长的碎片」误杀。
    for gap in silent_gaps(track.lines, duration=track.duration):
        anchors = [gap.before.idx] if gap.after is None else [gap.before.idx, gap.after.idx]
        for piece_start, piece_end in subtract((gap.start, gap.end), blocks):
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
