"""渲染「分段文案与剪辑时间轴对照表」。列顺序是用户确认过的模板，不许改。"""

from __future__ import annotations

from tenmin.models import AudioDirection, Beat, Clip, Script
from tenmin.script.budget import beat_seconds, narration_chars
from tenmin.timecode import format_timestamp, readable_seconds

COLUMNS = ("节点", "原片截取时间戳", "建议画面特征", "分段解说文案", "剪辑与原声处理")
LEGEND = "★ = 该片段命中无字幕演出高光区间，纯字幕方案取不到"
EMPTY = "-"

_ORIGINAL_AUDIO_LABELS = {
    "duck": "原声压低垫底",
    "mute": "原声静音",
    "full": "原声全开",
}


def escape_cell(text: str) -> str:
    """转义会被 markdown 表格／渲染器吃掉的字符。

    顺序要紧：
    1. `&` 必须最先转，否则原文里的 `&lt;` 会被渲染成 `<`；
    2. `<` `>` 必须在插入 `<br>` 之前转掉，否则连我们自己生成的 `<br>` 一起被转义。
    不转义 `<` 的话，narration 里一个 `<` 就会让渲染器把后面一截当 HTML 标签吞掉。
    """
    escaped = (
        text.strip()
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("`", "\\`")
    )
    return escaped.replace("\n", "<br>")


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
    # 用 .get 兜底：给 OriginalAudio Literal 加成员时忘了同步这张表，只该让这一格显示得
    # 不好看，不该让整篇文档渲染 KeyError。
    label = _ORIGINAL_AUDIO_LABELS.get(
        audio.original_audio, f"原声处理 {audio.original_audio}"
    )
    parts = [label]
    for hold in audio.holds:
        parts.append(f"留白 {hold.duration:.1f}s：「{hold.quote}」")
    for cue in audio.sfx:
        parts.append(f"音效 {cue.cue} @{cue.at:.1f}s")
    return "<br>".join(escape_cell(part) for part in parts)


def _beat_estimate(beat: Beat) -> float:
    """优先用 budget.apply_estimates 已经写进 script.json 的估算值。

    对照表原来自己 beat_seconds() 重算一遍，于是同一个数字有两个来源：用户手改
    script.json 的 est_seconds 之后，文档里的数字和产物里的数字会对不上。
    只有 est 还是默认的 0（budget 没跑过）时才回退到重算。
    """
    return beat.est_seconds if beat.est_seconds > 0 else beat_seconds(beat)


def _total_estimate(script: Script) -> float:
    if script.est_total_seconds > 0:
        return script.est_total_seconds
    return sum(_beat_estimate(beat) for beat in script.beats)


def render_table(script: Script) -> str:
    if len(script.episodes) == 1:
        title = f"# {script.show} 第 {script.episodes[0]} 集 解说方案"
    else:
        title = f"# {script.show} 整季 解说方案"

    chars = sum(narration_chars(beat.narration) for beat in script.beats)
    meta = (
        f"旁白 {chars} 字 · 估算时长 {readable_seconds(_total_estimate(script))} · "
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
