import pytest

from tenmin.models import AudioDirection, Beat, Clip, Hold, Script, VoiceChunk, VoiceTrack
from tenmin.render.timeline import (
    beat_audio_seconds,
    beat_clip_seconds,
    build_timeline,
    chunks_by_beat,
    scale_ratio,
)


def one_beat_script(clips: list[Clip], holds: list[Hold] | None = None) -> Script:
    return Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(
                id="b1",
                label="Hook",
                role="hook",
                narration="第一句。第二句。",
                clips=clips,
                audio=AudioDirection(holds=holds or []),
            )
        ],
    )


def two_chunk_track() -> VoiceTrack:
    chunks = [
        VoiceChunk(
            beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3",
            duration=8.0, hold_after=2.0,
        ),
        VoiceChunk(
            beat_id="b1", index=2, text="第二句。", path="chunk_002.mp3", duration=10.0
        ),
    ]
    return VoiceTrack(episode=2, chunks=chunks, total_seconds=20.0)


def test_beat_audio_seconds_includes_holds():
    assert beat_audio_seconds(two_chunk_track().chunks) == pytest.approx(20.0)


def test_beat_clip_seconds_sums_clip_durations():
    beat = one_beat_script(
        [Clip(episode=2, start=100.0, end=140.0), Clip(episode=2, start=200.0, end=220.0)]
    ).beats[0]
    assert beat_clip_seconds(beat) == pytest.approx(60.0)


def test_scale_ratio_shrinks_when_material_is_longer():
    assert scale_ratio(20.0, 40.0) == pytest.approx(0.5)


def test_scale_ratio_grows_when_narration_is_longer():
    assert scale_ratio(30.0, 10.0) == pytest.approx(3.0)


def test_scale_ratio_on_zero_clip_seconds_is_zero():
    assert scale_ratio(20.0, 0.0) == 0.0


def test_chunks_by_beat_groups_in_order():
    track = VoiceTrack(
        episode=2,
        chunks=[
            VoiceChunk(beat_id="b1", index=1, text="a", path="chunk_001.mp3", duration=1.0),
            VoiceChunk(beat_id="b2", index=1, text="b", path="chunk_002.mp3", duration=2.0),
            VoiceChunk(beat_id="b1", index=2, text="c", path="chunk_003.mp3", duration=3.0),
        ],
    )
    grouped = chunks_by_beat(track)
    assert [c.text for c in grouped["b1"]] == ["a", "c"]
    assert [c.text for c in grouped["b2"]] == ["b"]


def test_build_timeline_scales_single_clip():
    script = one_beat_script([Clip(episode=2, start=100.0, end=140.0)])
    timeline, warnings = build_timeline(script, two_chunk_track(), source_duration=1400.0)
    assert warnings == []
    assert len(timeline.segments) == 1
    seg = timeline.segments[0]
    assert seg.beat_id == "b1"
    assert seg.source_start == pytest.approx(100.0)
    assert seg.source_end == pytest.approx(120.0)
    assert seg.timeline_start == pytest.approx(0.0)
    assert seg.timeline_end == pytest.approx(20.0)
    assert timeline.total_seconds == pytest.approx(20.0)
    assert timeline.episode == 2


def test_build_timeline_places_subtitles_and_offsets():
    script = one_beat_script([Clip(episode=2, start=100.0, end=140.0)])
    timeline, _ = build_timeline(script, two_chunk_track(), source_duration=1400.0)
    assert [(c.start, c.end, c.text) for c in timeline.subtitles] == [
        (0.0, 8.0, "第一句。"),
        (10.0, 20.0, "第二句。"),
    ]
    # hold 期间没有字幕，画面干净、只有原声
    assert timeline.narration_offsets == pytest.approx([0.0, 10.0])


def test_build_timeline_splits_multi_sentence_chunk_into_per_sentence_cues():
    """一个 chunk 里如果有多句话，字幕要按句切开、按字数比例分配时间，
    不能整段话一次性挂在屏幕上几十秒。"""
    script = one_beat_script([Clip(episode=2, start=100.0, end=140.0)])
    chunks = [
        VoiceChunk(
            beat_id="b1", index=1,
            text="第一句。第二句更长一些。",
            path="chunk_001.mp3", duration=12.0,
        )
    ]
    track = VoiceTrack(episode=2, chunks=chunks, total_seconds=12.0)
    timeline, warnings = build_timeline(script, track, source_duration=1400.0)
    assert warnings == []
    cues = timeline.subtitles
    assert [c.text for c in cues] == ["第一句。", "第二句更长一些。"]
    # 4 字 vs 8 字，12 秒按比例切成 4 秒 / 8 秒
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(4.0)
    assert cues[1].start == pytest.approx(4.0)
    assert cues[1].end == pytest.approx(12.0)
    # 音频游标只按整个 chunk 的时长推进一次，不受切句影响
    assert timeline.narration_offsets == pytest.approx([0.0])


def test_build_timeline_scales_two_clips_proportionally():
    script = one_beat_script(
        [Clip(episode=2, start=100.0, end=140.0), Clip(episode=2, start=200.0, end=220.0)]
    )
    chunks = [
        VoiceChunk(beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=30.0)
    ]
    track = VoiceTrack(episode=2, chunks=chunks, total_seconds=30.0)
    timeline, warnings = build_timeline(script, track, source_duration=1400.0)
    assert warnings == []
    first, second = timeline.segments
    # ratio = 30 / 60 = 0.5
    assert (first.source_start, first.source_end) == pytest.approx((100.0, 120.0))
    assert (first.timeline_start, first.timeline_end) == pytest.approx((0.0, 20.0))
    assert (second.source_start, second.source_end) == pytest.approx((200.0, 210.0))
    assert (second.timeline_start, second.timeline_end) == pytest.approx((20.0, 30.0))


def test_build_timeline_clamps_at_source_end_with_warning():
    script = one_beat_script([Clip(episode=2, start=100.0, end=110.0)])
    chunks = [
        VoiceChunk(beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=30.0)
    ]
    track = VoiceTrack(episode=2, chunks=chunks, total_seconds=30.0)
    timeline, warnings = build_timeline(script, track, source_duration=120.0)
    # ratio = 3.0，本该切到 130，源片只有 120
    assert timeline.segments[0].source_end == pytest.approx(120.0)
    assert timeline.segments[0].timeline_end == pytest.approx(20.0)
    assert any("钳到片尾" in w for w in warnings)
    # 画面 20s 与音频 30s 相差超过 0.5s，应额外报一条
    assert any("相差超过" in w for w in warnings)


def test_build_timeline_warns_on_beat_without_chunks():
    script = one_beat_script([Clip(episode=2, start=100.0, end=140.0)])
    timeline, warnings = build_timeline(
        script, VoiceTrack(episode=2, chunks=[]), source_duration=1400.0
    )
    assert timeline.segments == []
    assert any("没有配音 chunk" in w for w in warnings)


def test_build_timeline_warns_on_beat_without_clips():
    script = one_beat_script([])
    timeline, warnings = build_timeline(script, two_chunk_track(), source_duration=1400.0)
    assert timeline.segments == []
    # 音频照旧推进，字幕仍然产出
    assert len(timeline.subtitles) == 2
    assert timeline.total_seconds == pytest.approx(20.0)
    assert any("没有可用的 clip" in w for w in warnings)


def test_build_timeline_drops_clip_starting_past_source_end():
    script = one_beat_script([Clip(episode=2, start=200.0, end=240.0)])
    timeline, warnings = build_timeline(script, two_chunk_track(), source_duration=150.0)
    assert timeline.segments == []
    assert any("超出源片长" in w for w in warnings)


def test_build_timeline_segments_are_contiguous_and_monotonic():
    script = Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(
                id="b1",
                label="Hook",
                role="hook",
                narration="第一句。",
                clips=[Clip(episode=2, start=100.0, end=140.0)],
            ),
            Beat(
                id="b2",
                label="收尾",
                role="outro",
                narration="第二句。",
                clips=[Clip(episode=2, start=300.0, end=340.0)],
            ),
        ],
    )
    track = VoiceTrack(
        episode=2,
        chunks=[
            VoiceChunk(beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=10.0),
            VoiceChunk(beat_id="b2", index=1, text="第二句。", path="chunk_002.mp3", duration=20.0),
        ],
        total_seconds=30.0,
    )
    timeline, warnings = build_timeline(script, track, source_duration=1400.0)
    assert warnings == []
    starts = [s.timeline_start for s in timeline.segments]
    assert starts == sorted(starts)
    for previous, current in zip(timeline.segments, timeline.segments[1:], strict=False):
        assert current.timeline_start == pytest.approx(previous.timeline_end)
    assert timeline.segments[-1].timeline_end == pytest.approx(timeline.total_seconds)
