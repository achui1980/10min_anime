"""无字幕间隙检测。长间隙 = 动画演出高光，v1 最重要的一条零成本规则。"""

from __future__ import annotations

from tenmin.config import DEFAULT_SIGNALS, SignalsConfig
from tenmin.intervals import SilentGap, silent_gaps, subtract
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


def _piece_anchors(gap: SilentGap, piece_start: float, piece_end: float) -> list[int]:
    """一个碎片该带哪些 anchor 行。判据是几何的：只认**紧邻这个碎片**的说话行。

    原实现把整条间隙的 `[before, after]` 原样发给每一个碎片。一条被 OP/ED 切成两片的
    间隙于是让左片带上位于右片之后的那行 —— 实测 saijo/E08 的左片
    （1327.8-1348.2）带着一个 1394.4s 的 anchor，隔着整段 ED、离自己右界 46 秒。
    下游两处都被这个坑到：`validate._check_anchor_coverage` 的「clip 时间窗必须装得下
    自己的 anchor_lines」收到自相矛盾的输入，喂给 LLM 的高能点也在说「这段无台词画面
    对应的是那句台词」，而那句其实在片尾之后。

    `gap.before` 结束于 `gap.start`，所以只有起点没被切掉的碎片紧邻它；
    `gap.after` 起始于 `gap.end`，所以只有终点没被切掉的碎片紧邻它。
    两端都被切掉的碎片一条都没有，anchor_lines 为空 —— 那是诚实的（「这是一段纯画面」），
    而 `density_shift` 的 anchor_lines 本来也恒为空，下游早就吃得下。

    浮点相等在这里是可靠的：`intervals.subtract` 对没被裁到的那一端返回的就是传进去的
    那个 float 本身，没有做过任何算术。
    """
    anchors: list[int] = []
    if piece_start == gap.start:
        anchors.append(gap.before.idx)
    if piece_end == gap.end and gap.after is not None:
        anchors.append(gap.after.idx)
    return anchors


def find_silent_gaps(
    track: DialogueTrack, *, cfg: SignalsConfig = DEFAULT_SIGNALS
) -> list[Signal]:
    blocks = [b for b in (track.op_range, track.ed_range) if b is not None]

    signals: list[Signal] = []
    # 时长筛选必须发生在扣掉 OP/ED 之后：先筛会把「被 OP 切开后仍够长的碎片」误杀。
    for gap in silent_gaps(track.lines, duration=track.duration):
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
                    anchor_lines=_piece_anchors(gap, piece_start, piece_end),
                )
            )
    # 这次排序**当前是空操作**：silent_gaps 按时间递增产出，subtract 保持碎片顺序，
    # 所以 signals 生成时就已经有序。留着是防御性的 —— 下游（aggregate 的双指针挂载）
    # 依赖「信号按起点有序」这条不变量，而它现在只是间接成立。真要去掉的话得先在
    # aggregate 那边把有序性显式化，不值当。
    signals.sort(key=lambda s: s.start)
    return signals
