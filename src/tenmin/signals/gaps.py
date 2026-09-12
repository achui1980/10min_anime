"""无字幕间隙检测。长间隙 = 动画演出高光，v1 最重要的一条零成本规则。"""

from __future__ import annotations

from tenmin.config import DEFAULT_SIGNALS, SignalsConfig
from tenmin.intervals import silent_gaps, subtract
from tenmin.models import DialogueTrack, Signal


def _strength(duration: float, cfg: SignalsConfig) -> int:
    """间隙时长分档。强度取值范围由 models.STRENGTH_MIN/MAX 约束。"""
    if duration >= cfg.gap_strong_seconds:
        return 4
    if duration >= cfg.gap_medium_seconds:
        return 3
    return 2


def find_silent_gaps(
    track: DialogueTrack, *, cfg: SignalsConfig = DEFAULT_SIGNALS
) -> list[Signal]:
    blocks = [b for b in (track.op_range, track.ed_range) if b is not None]

    signals: list[Signal] = []
    # 时长筛选必须发生在扣掉 OP/ED 之后：先筛会把「被 OP 切开后仍够长的碎片」误杀。
    for gap in silent_gaps(track.lines, duration=track.duration):
        anchors = [gap.before.idx] if gap.after is None else [gap.before.idx, gap.after.idx]
        for piece_start, piece_end in subtract((gap.start, gap.end), blocks):
            duration = piece_end - piece_start
            if duration < cfg.min_gap_seconds:
                continue
            signals.append(
                Signal(
                    start=piece_start,
                    end=piece_end,
                    source="gap",
                    strength=_strength(duration, cfg),
                    detail=f"gap:{duration:.1f}s",
                    anchor_lines=anchors,
                )
            )
    signals.sort(key=lambda s: s.start)
    return signals
