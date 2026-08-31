"""LLM 输出的后处理校验。LLM 会编造时间戳，这里是唯一的拦网。"""

from __future__ import annotations

from pydantic import BaseModel

from tenmin.models import Clip, DialogueTrack, Script, SignalReport

ANCHOR_TOLERANCE_SECONDS = 5.0
SILENT_OVERLAP_SECONDS = 1.0
MIN_BEATS = 3


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


def _anchor_time(track: DialogueTrack, anchor_lines: list[int]) -> float | None:
    starts = [
        ln.start
        for ln in track.lines
        if ln.idx in anchor_lines or any(m in anchor_lines for m in ln.merged_from)
    ]
    if not starts:
        return None
    return min(starts)


def validate_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
) -> ValidationResult:
    warnings: list[str] = []

    if len(script.beats) < MIN_BEATS:
        raise ScriptValidationError(
            f"剧本节点数 {len(script.beats)} 少于下限 {MIN_BEATS}，重试"
        )

    for beat in script.beats:
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

    return ValidationResult(script=script, warnings=warnings)
