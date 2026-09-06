import pytest
from pydantic import ValidationError

from tenmin.models import (
    AudioDirection,
    Beat,
    Clip,
    DialogueLine,
    DialogueTrack,
    Highlight,
    Hold,
    LLMScript,
    Script,
    SfxCue,
    Signal,
    SignalReport,
    SubtitleCue,
    Timeline,
    TimelineSegment,
    VoiceChunk,
    VoiceTrack,
)


def test_dialogue_line_defaults():
    line = DialogueLine(idx=1, start=7.12, end=11.48, text="你好", raw="你好")
    assert line.speaker is None
    assert line.kind == "dialogue"
    assert line.suspect is False
    assert line.merged_from == []
    assert line.duration == pytest.approx(4.36)


def test_dialogue_line_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        DialogueLine(idx=1, start=0.0, end=1.0, text="x", raw="x", kind="singing")


def test_dialogue_track_defaults():
    track = DialogueTrack(episode=2, duration=1416.622)
    assert track.source == "srt"
    assert track.op_range is None
    assert track.ed_range is None
    assert track.lines == []


def test_signal_duration():
    sig = Signal(start=134.6, end=153.486, source="gap", strength=4, detail="gap:18.9s")
    assert sig.duration == pytest.approx(18.886, abs=1e-3)
    assert sig.anchor_lines == []


def test_signal_report_defaults():
    report = SignalReport(episode=2)
    assert report.silent_gaps == []
    assert report.highlights == []
    assert report.median_char_rate == 0.0


def test_highlight_strength_bounds():
    with pytest.raises(ValidationError):
        Highlight(start=0.0, end=1.0, strength=6)
    with pytest.raises(ValidationError):
        Highlight(start=0.0, end=1.0, strength=0)
    assert Highlight(start=0.0, end=1.0, strength=5).triggers == []


def test_clip_defaults():
    clip = Clip(episode=2, start=1100.0, end=1104.7)
    assert clip.visual == ""
    assert clip.is_silent_highlight is False
    assert clip.duration == pytest.approx(4.7)


def test_audio_direction_defaults():
    audio = AudioDirection()
    assert audio.original_audio == "duck"
    assert audio.sfx == []
    assert audio.holds == []


def test_beat_and_script_defaults():
    beat = Beat(
        id="b1",
        label="Hook 开场",
        role="hook",
        narration="开场文案",
        clips=[Clip(episode=2, start=7.12, end=11.48)],
        audio=AudioDirection(
            holds=[Hold(at=1.0, duration=3.0, quote="金句")],
            sfx=[SfxCue(at=0.5, cue="impact")],
        ),
    )
    assert beat.est_seconds == 0.0
    script = Script(show="才女的侍从", episodes=[2], beats=[beat])
    assert script.mode == "single_episode"
    assert script.target_seconds == pytest.approx(240.0)
    assert script.est_total_seconds == 0.0


def test_llm_script_schema_has_no_computed_fields():
    """喂给 LLM 的 schema 不能包含我们自己算的字段。"""
    schema = LLMScript.model_json_schema()
    dumped = str(schema)
    assert "est_seconds" not in dumped
    assert "est_total_seconds" not in dumped
    assert "is_silent_highlight" not in dumped
    assert "beats" in schema["properties"]


def test_voice_chunk_defaults():
    chunk = VoiceChunk(beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=3.0)
    assert chunk.hold_after == 0.0


def test_voice_track_holds_chunks():
    chunk = VoiceChunk(
        beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=3.0, hold_after=1.5
    )
    track = VoiceTrack(episode=2, chunks=[chunk], total_seconds=4.5)
    assert track.chunks[0].hold_after == 1.5
    assert track.total_seconds == 4.5


def test_timeline_segment_fields():
    seg = TimelineSegment(
        beat_id="b1", source_start=100.0, source_end=120.0, timeline_start=0.0, timeline_end=20.0
    )
    assert seg.source_end - seg.source_start == 20.0


def test_timeline_defaults_are_empty():
    timeline = Timeline(episode=2, total_seconds=0.0)
    assert timeline.segments == []
    assert timeline.subtitles == []
    assert timeline.narration_offsets == []


def test_timeline_roundtrips_json():
    timeline = Timeline(
        episode=2,
        segments=[
            TimelineSegment(
                beat_id="b1",
                source_start=100.0,
                source_end=120.0,
                timeline_start=0.0,
                timeline_end=20.0,
            )
        ],
        subtitles=[SubtitleCue(start=0.0, end=8.0, text="第一句。")],
        narration_offsets=[0.0],
        total_seconds=20.0,
    )
    restored = Timeline.model_validate_json(timeline.model_dump_json())
    assert restored == timeline
