"""LLM 输出的后处理校验。LLM 会编造时间戳，这里是唯一的拦网。

模块分成两半，别把它们混起来：

- `check_script`：**纯读**。只看、只返回 warning，一个字节都不改 script。
- `repair_script`：**显式修复**。深拷贝一份再改（丢 clip、按字幕覆写时间戳、回填
  is_silent_highlight），返回那个**新**对象。

`validate_script` 是两者的组合，对外形状与历史一致。拆开的动因是调用方
（script/single.py）需要「先校验、后比较」：预算重写轮会拿新稿替换旧稿，而原来
validate 就地改写并把同一个对象塞回 ValidationResult，上一版根本没被保留下来，
「两版择优」物理上做不到。
"""

from __future__ import annotations

from pydantic import BaseModel

from tenmin.models import Beat, Clip, DialogueLine, DialogueTrack, Script, SignalReport

ANCHOR_TOLERANCE_SECONDS = 5.0
SILENT_OVERLAP_SECONDS = 1.0
MIN_BEATS = 3
# 「落在窗**外**的 anchor 行占比」的上限。名字里的 OUTSIDE 是刻意的：它原来叫
# ANCHOR_COVERAGE_MIN_RATIO（「覆盖率下限」），而代码里比的是 outside/total，
# 语义正好反过来 —— 读代码的人会以为 0.5 是「至少一半要被覆盖」。
#
# 实测真实 LLM 输出里 14/18 个 clip 至少有一条 anchor 落在窗外，多数只差 1-3 秒无害，
# 所以只在「过半 anchor 都在窗外」时才报——那种情况说明旁白讲的内容整段没有画面。
# 边界语义：**严格超过**才报。恰好一半在窗外是平手，不报（有测试锁着）。
ANCHOR_OUTSIDE_MAX_RATIO = 0.5


class ScriptValidationError(RuntimeError):
    """校验后剧本不可用，调用方应重试一次 LLM。"""


class ValidationResult(BaseModel):
    script: Script
    warnings: list[str] = []


def _fully_inside(clip: Clip, window: tuple[float, float] | None) -> bool:
    if window is None:
        return False
    return clip.start >= window[0] and clip.end <= window[1]


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _anchor_matches(track: DialogueTrack, anchor_lines: list[int]) -> list[DialogueLine]:
    return [
        ln
        for ln in track.lines
        if ln.idx in anchor_lines or any(m in anchor_lines for m in ln.merged_from)
    ]


def _anchor_time(track: DialogueTrack, anchor_lines: list[int]) -> float | None:
    starts = [ln.start for ln in _anchor_matches(track, anchor_lines)]
    if not starts:
        return None
    return min(starts)


def _quote_matches(
    tracks: dict[int, DialogueTrack], episodes: list[int], quote: str
) -> list[DialogueLine]:
    """按整行完全相等反查金句出处。找不到是合法情况（跨 cue 拼接／双轨半句）。"""
    target = quote.strip()
    found: list[DialogueLine] = []
    for episode in episodes:
        track = tracks.get(episode)
        if track is None:
            continue
        found.extend(ln for ln in track.lines if ln.text.strip() == target)
    return found


def _check_hold_quotes(beat: Beat, tracks: dict[int, DialogueTrack]) -> list[str]:
    """留白金句的原声必须落在本 beat 某个 clip 的时间窗内，否则剪辑师放不出来。"""
    if not beat.clips:
        return []
    episodes = list(dict.fromkeys(clip.episode for clip in beat.clips))
    warnings: list[str] = []
    for hold in beat.audio.holds:
        matches = _quote_matches(tracks, episodes, hold.quote)
        if not matches:
            continue
        if any(
            _overlap(ln.start, ln.end, clip.start, clip.end) > 0
            for ln in matches
            for clip in beat.clips
        ):
            continue
        warnings.append(
            f"{beat.label}：留白金句「{hold.quote}」的原声在 "
            f"{matches[0].start:.1f} 秒，不在本节点任何 clip 的时间窗内，"
            f"剪辑时放不出这句原声"
        )
    return warnings


def _check_anchor_coverage(beat: Beat, tracks: dict[int, DialogueTrack]) -> list[str]:
    """clip 的时间窗必须装得下自己的 anchor_lines，否则旁白讲的内容没有画面。"""
    warnings: list[str] = []
    for clip in beat.clips:
        track = tracks.get(clip.episode)
        if track is None:
            continue
        lines = _anchor_matches(track, clip.anchor_lines)
        total = len(lines)
        if total == 0:
            continue
        outside = sum(
            1
            for ln in lines
            if _overlap(ln.start, ln.end, clip.start, clip.end) <= 0
        )
        if outside / total > ANCHOR_OUTSIDE_MAX_RATIO:
            warnings.append(
                f"{beat.label}：clip {clip.start:.1f}-{clip.end:.1f} 的 anchor 行有 "
                f"{outside}/{total} 条落在时间窗外，旁白讲的内容缺画面"
            )
    return warnings


def check_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
) -> list[str]:
    """纯读的语义校验。返回全部 warning，**不改** script、也不抛异常。

    reports 目前用不到，但保留在签名里：它与 repair_script 共用一套入参，
    调用方（single.py）拿同一组素材调两个函数，签名对齐比少一个参数更值。
    """
    warnings: list[str] = []
    for beat in script.beats:
        warnings.extend(_check_hold_quotes(beat, tracks))
        warnings.extend(_check_anchor_coverage(beat, tracks))
    return warnings


def repair_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
) -> tuple[Script, list[str]]:
    """丢掉不可用的 clip、按字幕覆写偏差过大的时间戳、回填 is_silent_highlight。

    返回 **新** Script（深拷贝后再改），入参一字不动。修不了的（节点数不足、
    某个节点的 clip 全灭）抛 ScriptValidationError，调用方重试一次 LLM。
    """
    if len(script.beats) < MIN_BEATS:
        raise ScriptValidationError(
            f"剧本节点数 {len(script.beats)} 少于下限 {MIN_BEATS}，重试"
        )

    repaired = script.model_copy(deep=True)
    warnings: list[str] = []

    for beat in repaired.beats:
        kept: list[Clip] = []
        for clip in beat.clips:
            track = tracks.get(clip.episode)
            if track is None:
                warnings.append(f"{beat.label}：clip 引用了不存在的集数 {clip.episode}，丢弃")
                continue
            if clip.end <= clip.start:
                warnings.append(
                    f"{beat.label}：clip {clip.start:.1f}-{clip.end:.1f} 时长非正，丢弃"
                )
                continue
            if clip.start < 0 or clip.end > track.duration:
                warnings.append(
                    f"{beat.label}：clip {clip.start:.1f}-{clip.end:.1f} 越界"
                    f"（正片 0-{track.duration:.1f}），丢弃"
                )
                continue
            if _fully_inside(clip, track.op_range):
                warnings.append(
                    f"{beat.label}：clip {clip.start:.1f}-{clip.end:.1f} 落在片头曲内，丢弃"
                )
                continue
            if _fully_inside(clip, track.ed_range):
                warnings.append(
                    f"{beat.label}：clip {clip.start:.1f}-{clip.end:.1f} 落在片尾曲内，丢弃"
                )
                continue

            anchor_start = _anchor_time(track, clip.anchor_lines)
            if (
                anchor_start is not None
                and abs(anchor_start - clip.start) > ANCHOR_TOLERANCE_SECONDS
            ):
                duration = clip.duration
                warnings.append(
                    f"{beat.label}：clip 起点 {clip.start:.1f} 与 anchor 行时间 "
                    f"{anchor_start:.1f} 偏差超过 {ANCHOR_TOLERANCE_SECONDS:.0f} 秒，"
                    f"以字幕时间为准"
                )
                clip.start = anchor_start
                clip.end = min(anchor_start + duration, track.duration)
                if clip.end <= clip.start:
                    warnings.append(f"{beat.label}：anchor 覆写后时长非正，丢弃")
                    continue

            report = reports.get(clip.episode)
            gaps = report.silent_gaps if report else []
            clip.is_silent_highlight = any(
                _overlap(clip.start, clip.end, gap.start, gap.end)
                >= SILENT_OVERLAP_SECONDS
                for gap in gaps
            )
            kept.append(clip)

        if not kept:
            raise ScriptValidationError(
                f"{beat.label} 的所有 clip 都未通过校验，剧本不可用，重试"
            )
        beat.clips = kept

    return repaired, warnings


def validate_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
) -> ValidationResult:
    """repair 一遍再 check 一遍。对外形状与历史一致，但**不再改动入参**。"""
    repaired, warnings = repair_script(script, tracks, reports)
    warnings.extend(check_script(repaired, tracks, reports))
    return ValidationResult(script=repaired, warnings=warnings)
