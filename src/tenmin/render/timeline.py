"""时间轴重算。全是纯函数。

对齐规则：clip.start 硬（它过了 v1 validate.py 的 anchor 校正，是有据可查的镜头起点），
clip 时长按 ratio 缩放（clip.end 是 LLM 猜的，只用来算比例）。
"""

from __future__ import annotations

from tenmin.models import (
    Beat,
    Script,
    SubtitleCue,
    Timeline,
    TimelineSegment,
    VoiceChunk,
    VoiceTrack,
)

# 画面与音频总时长的容忍差。超过就报 warning，不报错。
DRIFT_TOLERANCE = 0.5


def beat_audio_seconds(chunks: list[VoiceChunk]) -> float:
    """本 beat 的音频总时长。含 hold 静音——静音期间也得有画面。"""
    return sum(chunk.duration + chunk.hold_after for chunk in chunks)


def beat_clip_seconds(beat: Beat) -> float:
    return sum(clip.duration for clip in beat.clips)


def scale_ratio(audio_seconds: float, clip_seconds: float) -> float:
    if clip_seconds <= 0:
        return 0.0
    return audio_seconds / clip_seconds


def chunks_by_beat(track: VoiceTrack) -> dict[str, list[VoiceChunk]]:
    grouped: dict[str, list[VoiceChunk]] = {}
    for chunk in track.chunks:
        grouped.setdefault(chunk.beat_id, []).append(chunk)
    return grouped


def build_timeline(
    script: Script, track: VoiceTrack, source_duration: float
) -> tuple[Timeline, list[str]]:
    """按 beat 逐段重算画面时长，产出成片时间轴。"""
    grouped = chunks_by_beat(track)
    segments: list[TimelineSegment] = []
    subtitles: list[SubtitleCue] = []
    offsets: list[float] = []
    warnings: list[str] = []
    audio_cursor = 0.0
    picture_cursor = 0.0

    for beat in script.beats:
        chunks = grouped.get(beat.id, [])
        if not chunks:
            warnings.append(f"beat {beat.id} 没有配音 chunk，已跳过")
            continue

        # 音频游标：字幕与旁白落点都由它驱动
        for chunk in chunks:
            offsets.append(audio_cursor)
            subtitles.append(
                SubtitleCue(start=audio_cursor, end=audio_cursor + chunk.duration, text=chunk.text)
            )
            audio_cursor += chunk.duration + chunk.hold_after

        audio_seconds = beat_audio_seconds(chunks)
        clip_seconds = beat_clip_seconds(beat)
        if clip_seconds <= 0:
            warnings.append(f"beat {beat.id} 没有可用的 clip，该段将没有画面")
            continue

        ratio = scale_ratio(audio_seconds, clip_seconds)
        for clip in beat.clips:
            if clip.start >= source_duration:
                warnings.append(
                    f"beat {beat.id} 的 clip 起点 {clip.start:.1f}s 超出源片长 "
                    f"{source_duration:.1f}s，已丢弃该段"
                )
                continue
            source_end = clip.start + clip.duration * ratio
            if source_end > source_duration:
                warnings.append(
                    f"beat {beat.id} 的 clip {clip.start:.1f}s 延长到 {source_end:.1f}s "
                    f"超出源片长 {source_duration:.1f}s，已钳到片尾"
                )
                source_end = source_duration
            if source_end <= clip.start:
                warnings.append(
                    f"beat {beat.id} 的 clip {clip.start:.1f}s 缩放后时长为 0，已丢弃该段"
                )
                continue
            length = source_end - clip.start
            segments.append(
                TimelineSegment(
                    beat_id=beat.id,
                    source_start=clip.start,
                    source_end=source_end,
                    timeline_start=picture_cursor,
                    timeline_end=picture_cursor + length,
                )
            )
            picture_cursor += length

    if abs(picture_cursor - audio_cursor) > DRIFT_TOLERANCE:
        warnings.append(
            f"画面总时长 {picture_cursor:.1f}s 与音频总时长 {audio_cursor:.1f}s "
            f"相差超过 {DRIFT_TOLERANCE}s，成片尾部会有画面缺失或黑屏，请检查 timeline.json"
        )

    timeline = Timeline(
        episode=track.episode,
        segments=segments,
        subtitles=subtitles,
        narration_offsets=offsets,
        total_seconds=audio_cursor,
    )
    return timeline, warnings
