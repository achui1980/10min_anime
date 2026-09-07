"""黄金样本：真 script.json 的时间轴重算。断言结构不变量，不断言快照数字。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tenmin.models import Script, VoiceChunk, VoiceTrack
from tenmin.render.chunks import plan_chunks
from tenmin.render.subtitles import render_ass
from tenmin.render.timeline import (
    beat_audio_seconds,
    beat_clip_seconds,
    build_timeline,
    chunks_by_beat,
    scale_ratio,
)
from tenmin.script.budget import SPEECH_RATE_CPS, narration_chars

FIXTURES = Path(__file__).parent / "fixtures"

# 真实剧集约 24 分钟；样本里最后一个 clip 结束在 1314 秒，1440 足够宽松。
SOURCE_DURATION = 1440.0


@pytest.fixture(scope="module")
def golden_script() -> Script:
    path = FIXTURES / "akujo_e02.script.json"
    if not path.exists():
        pytest.skip("缺少 tests/fixtures/akujo_e02.script.json")
    return Script.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_voice(golden_script) -> VoiceTrack:
    """不联网：按 4.5 字/秒给每个 chunk 一个确定时长，模拟一次配音结果。"""
    chunks: list[VoiceChunk] = []
    serial = 0
    for beat in golden_script.beats:
        for index, (text, hold_after) in enumerate(plan_chunks(beat), start=1):
            serial += 1
            chunks.append(
                VoiceChunk(
                    beat_id=beat.id,
                    index=index,
                    text=text,
                    path=f"chunk_{serial:03d}.mp3",
                    duration=narration_chars(text) / SPEECH_RATE_CPS,
                    hold_after=hold_after,
                )
            )
    total = sum(chunk.duration + chunk.hold_after for chunk in chunks)
    return VoiceTrack(episode=2, chunks=chunks, total_seconds=total)


def test_golden_script_shape(golden_script):
    assert len(golden_script.beats) == 6
    assert sum(len(beat.clips) for beat in golden_script.beats) == 18


def test_golden_timeline_keeps_every_clip(golden_script, golden_voice):
    timeline, warnings = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    assert len(timeline.segments) == 18
    assert {segment.beat_id for segment in timeline.segments} == {
        beat.id for beat in golden_script.beats
    }
    assert warnings == []


def test_golden_every_beat_shrinks(golden_script, golden_voice):
    """实测结论：素材多于旁白，绝大多数 beat 的 ratio 落在 (0, 1)。

    这份剧本的 outro 是例外：它只有 1 个 23 秒的 clip，旁白却有 33.67 秒，
    ratio≈1.46（其余五个 beat 是 0.41 / 0.52 / 0.46 / 0.58 / 0.52）。
    那不是 bug，是这份剧本的真实形态，所以只断言 ratio > 0。
    """
    by_beat = chunks_by_beat(golden_voice)
    for beat in golden_script.beats:
        ratio = scale_ratio(
            beat_audio_seconds(by_beat[beat.id]), beat_clip_seconds(beat)
        )
        assert ratio > 0.0, f"{beat.id} 的 ratio={ratio}"


def test_golden_source_starts_untouched(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    expected = [clip.start for beat in golden_script.beats for clip in beat.clips]
    assert [segment.source_start for segment in timeline.segments] == pytest.approx(
        expected
    )


def test_golden_timeline_is_contiguous(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    assert timeline.segments[0].timeline_start == pytest.approx(0.0)
    for previous, current in zip(
        timeline.segments, timeline.segments[1:], strict=False
    ):
        assert current.timeline_start == pytest.approx(previous.timeline_end)
        assert current.timeline_end > current.timeline_start


def test_golden_picture_matches_audio(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    picture = sum(segment.duration for segment in timeline.segments)
    assert picture == pytest.approx(timeline.total_seconds, abs=0.01)
    assert timeline.total_seconds == pytest.approx(golden_voice.total_seconds, abs=0.01)


def test_golden_subtitles_cover_every_chunk(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    # 多句话的 chunk 现在会拆成多条按比例分配时间的字幕，条数只会 >= chunk 数
    assert len(timeline.subtitles) >= len(golden_voice.chunks)
    assert len(timeline.narration_offsets) == len(golden_voice.chunks)
    for cue in timeline.subtitles:
        assert cue.end > cue.start
        assert cue.text.strip() == cue.text
    # 字幕按时间顺序衔接，不重叠、不留缝
    for previous, current in zip(timeline.subtitles, timeline.subtitles[1:], strict=False):
        assert current.start >= previous.end - 1e-6


def test_golden_ass_renders(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    ass = render_ass(timeline.subtitles)
    assert "[Script Info]" in ass
    assert ass.count("\nDialogue: ") == len(timeline.subtitles)
