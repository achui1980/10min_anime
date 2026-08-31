"""渲染「分段文案与剪辑时间轴对照表」。列顺序是用户确认过的模板，不许改。"""

from __future__ import annotations

from tenmin.models import AudioDirection, Clip, Script
from tenmin.script.budget import beat_seconds, narration_chars
from tenmin.timecode import format_timestamp

COLUMNS = ("节点", "原片截取时间戳", "建议画面特征", "分段解说文案", "剪辑与原声处理")
LEGEND = "★ = 该片段命中无字幕演出高光区间，纯字幕方案取不到"
EMPTY = "-"

_ORIGINAL_AUDIO_LABELS = {
    "duck": "原声压低垫底",
    "mute": "原声静音",
    "full": "原声全开",
}


def escape_cell(text: str) -> str:
    return text.strip().replace("|", "\\|").replace("\n", "<br>")


def render_timestamp_cell(clips: list[Clip]) -> str:
    if not clips:
        return EMPTY
    parts = []
    for clip in clips:
        span = f"{format_timestamp(clip.start)} - {format_timestamp(clip.end)}"
        if clip.is_silent_highlight:
            span += " ★"
        parts.append(span)
    return parts[0] + "".join(f"<br>接 {part}" for part in parts[1:])


def render_visual_cell(clips: list[Clip]) -> str:
    visuals = [clip.visual.strip() for clip in clips if clip.visual.strip()]
    if not visuals:
        return EMPTY
    return escape_cell(" ➔ ".join(visuals))


def render_audio_cell(audio: AudioDirection) -> str:
    parts = [_ORIGINAL_AUDIO_LABELS[audio.original_audio]]
    for hold in audio.holds:
        parts.append(f"留白 {hold.duration:.1f}s：「{hold.quote}」")
    for cue in audio.sfx:
        parts.append(f"音效 {cue.cue} @{cue.at:.1f}s")
    return "<br>".join(escape_cell(part) for part in parts)


def _readable_seconds(seconds: float) -> str:
    total = round(seconds)
    if total < 60:
        return f"{total} 秒"
    return f"{total // 60} 分 {total % 60} 秒"


def render_table(script: Script) -> str:
    if len(script.episodes) == 1:
        title = f"# {script.show} 第 {script.episodes[0]} 集 解说方案"
    else:
        title = f"# {script.show} 整季 解说方案"

    chars = sum(narration_chars(beat.narration) for beat in script.beats)
    total = sum(beat_seconds(beat) for beat in script.beats)
    meta = (
        f"旁白 {chars} 字 · 估算时长 {_readable_seconds(total)} · "
        f"{len(script.beats)} 个节点"
    )

    lines = [
        title,
        "",
        meta,
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for beat in script.beats:
        lines.append(
            "| "
            + " | ".join(
                (
                    escape_cell(beat.label),
                    render_timestamp_cell(beat.clips),
                    render_visual_cell(beat.clips),
                    escape_cell(beat.narration),
                    render_audio_cell(beat.audio),
                )
            )
            + " |"
        )
    lines.extend(["", LEGEND])
    return "\n".join(lines) + "\n"
