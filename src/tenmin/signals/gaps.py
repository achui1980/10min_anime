"""无字幕间隙检测。长间隙 = 动画演出高光，v1 最重要的一条零成本规则。"""

from __future__ import annotations

from tenmin.config import DEFAULT_SIGNALS, SignalsConfig
from tenmin.intervals import silent_gaps, subtract
from tenmin.models import DialogueTrack, Signal


def _strength(duration: float, cfg: SignalsConfig) -> int:
    """间隙时长分档。强度取值范围由 models.STRENGTH_MIN/MAX 约束。

    末档 `return 2` 不是死代码：任何 `duration < gap_medium_seconds` 都会走到它，
    而 find_silent_gaps 传进来的 duration 恒 `>= cfg.min_gap_seconds`（默认 3.0），
    所以它接的是 [min_gap_seconds, gap_medium_seconds) 这一档，不需要额外守卫。
    """
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
    # 这次排序**当前是空操作**：silent_gaps 按时间递增产出，subtract 保持碎片顺序，
    # 所以 signals 生成时就已经有序。留着是防御性的 —— 下游（aggregate 的双指针挂载）
    # 依赖「信号按起点有序」这条不变量，而它现在只是间接成立。真要去掉的话得先在
    # aggregate 那边把有序性显式化，不值当。
    signals.sort(key=lambda s: s.start)
    return signals
