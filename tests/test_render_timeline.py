from itertools import pairwise

import pytest

from tenmin.models import AudioDirection, Beat, Clip, Hold, Script, VoiceChunk, VoiceTrack
from tenmin.render.timeline import (
    align_to_frame,
    beat_audio_seconds,
    beat_clip_seconds,
    build_timeline,
    chunks_by_beat,
    scale_ratio,
    sentence_cues,
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


def test_build_timeline_rejects_beat_without_clips():
    """有 chunk 却一个 clip 都没有 = 音频占了时间、画面没有输出 → 此后全片失同步。

    原来这里只报一条 warning 就继续：音频游标已经推进（字幕也照样产出），而画面
    游标没动，于是**后面每一段画面都相对旁白整体前移**。硬失败才是对的，理由见
    build_timeline 的 docstring。
    """
    script = one_beat_script([])
    with pytest.raises(ValueError) as exc:
        build_timeline(script, two_chunk_track(), source_duration=1400.0)
    assert "b1" in str(exc.value)
    assert "20.0" in str(exc.value)


def test_build_timeline_rejects_beat_whose_clips_all_fall_outside_the_source():
    """clip 被逐条丢弃到一个不剩，跟「压根没写 clip」是同一种失同步。"""
    script = one_beat_script([Clip(episode=2, start=200.0, end=240.0)])
    with pytest.raises(ValueError) as exc:
        build_timeline(script, two_chunk_track(), source_duration=150.0)
    assert "b1" in str(exc.value)


def test_build_timeline_keeps_going_when_the_beat_still_has_one_clip():
    """降级的边界：同一个 beat 里丢掉一条、留下一条时照旧只报 warning。"""
    script = one_beat_script(
        [Clip(episode=2, start=200.0, end=240.0), Clip(episode=2, start=10.0, end=50.0)]
    )
    timeline, warnings = build_timeline(script, two_chunk_track(), source_duration=150.0)
    assert len(timeline.segments) == 1
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


# --- 帧边界对齐（P2-A 第 6 项）---------------------------------------------


def test_align_to_frame_snaps_to_the_nearest_frame():
    fps = 24000 / 1001  # 23.976023976…
    assert align_to_frame(0.0, fps) == pytest.approx(0.0)
    # 一帧 = 0.0417 秒。0.05 秒最近的是第 1 帧（0.04171），不是第 0 帧。
    assert align_to_frame(0.05, fps) == pytest.approx(1 / fps)
    assert align_to_frame(0.02, fps) == pytest.approx(0.0)
    # 已经落在边界上的值必须原样返回（不能被浮点误差推走一帧）
    assert align_to_frame(2400 / fps, fps) == pytest.approx(2400 / fps)


def test_align_to_frame_is_a_noop_without_a_frame_rate():
    assert align_to_frame(1.234, None) == pytest.approx(1.234)


def test_build_timeline_aligns_segment_bounds_to_frame_boundaries():
    """记进 timeline.json 的数字必须就是 ffmpeg 会用的那个帧边界。

    不对齐时每段首尾各被 ffmpeg 按帧取整一次，段时长与声明值差最多一帧，
    23 段累起来就是成片时长与 total_seconds 对不上的那个零点几秒。
    """
    fps = 24000 / 1001
    script = one_beat_script([Clip(episode=2, start=100.0, end=140.0)])
    timeline, warnings = build_timeline(
        script, two_chunk_track(), source_duration=1400.0, frame_rate=fps
    )
    assert warnings == []
    seg = timeline.segments[0]
    for value in (seg.source_start, seg.source_end):
        # 每个边界都必须是整数个帧
        assert (value * fps) == pytest.approx(round(value * fps), abs=1e-6)
    assert seg.source_start == pytest.approx(align_to_frame(100.0, fps))
    # 段时长是整数个帧，所以画面坐标也跟着变成帧的整数倍
    assert (seg.duration * fps) == pytest.approx(round(seg.duration * fps), abs=1e-6)
    assert timeline.frame_rate == pytest.approx(fps)


def test_build_timeline_without_frame_rate_keeps_the_raw_bounds():
    """帧率未知时不猜：边界原样落盘（老行为）。"""
    script = one_beat_script([Clip(episode=2, start=100.0, end=140.0)])
    timeline, _ = build_timeline(script, two_chunk_track(), source_duration=1400.0)
    assert timeline.segments[0].source_start == pytest.approx(100.0)
    assert timeline.segments[0].source_end == pytest.approx(120.0)
    assert timeline.frame_rate is None


def test_build_timeline_drops_a_clip_with_a_negative_start():
    """`clip.start` 只被拦过上界。负数会原样变成 `trim=start=-3.000`。

    validate.py 拦得住 LLM 那条路（它按 DialogueTrack.duration 查 [0, 片长]），
    但 script.json 是文档里写明的人工编辑面，手改成负数就直接漏到这里。
    """
    script = one_beat_script(
        [Clip(episode=2, start=-3.0, end=20.0), Clip(episode=2, start=10.0, end=50.0)]
    )
    timeline, warnings = build_timeline(
        script, two_chunk_track(), source_duration=1400.0
    )
    assert [s.source_start for s in timeline.segments] == pytest.approx([10.0])
    assert any("-3.0" in w and "丢弃" in w for w in warnings)


def test_build_timeline_drops_a_clip_with_a_non_finite_start():
    """NaN 比谁都不大也不小，两道边界检查都拦不住它，最后会写出非法 JSON。"""
    script = one_beat_script(
        [
            Clip(episode=2, start=float("nan"), end=20.0),
            Clip(episode=2, start=10.0, end=50.0),
        ]
    )
    timeline, warnings = build_timeline(
        script, two_chunk_track(), source_duration=1400.0
    )
    assert [s.source_start for s in timeline.segments] == pytest.approx([10.0])
    assert any("nan" in w.lower() for w in warnings)


def test_build_timeline_drift_tolerance_comes_from_config():
    from tenmin.config import RenderConfig

    script = one_beat_script([Clip(episode=2, start=100.0, end=110.0)])
    chunks = [
        VoiceChunk(beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=30.0)
    ]
    track = VoiceTrack(episode=2, chunks=chunks, total_seconds=30.0)
    # 画面被钳到片尾只剩 20 秒，音频 30 秒 → 默认 0.5 秒容忍度下必报
    _, warnings = build_timeline(script, track, source_duration=120.0)
    assert any("相差超过" in w for w in warnings)
    _, loose = build_timeline(
        script, track, source_duration=120.0, cfg=RenderConfig(drift_tolerance=20.0)
    )
    assert not any("相差超过" in w for w in loose)


# --- 相邻 cue 共享精确边界，不插间隙（P2-E B3 的决定，锁住它）--------------
#
# 这不是一个 TDD 循环（没有改任何行为），是把 B3 的结论钉下来：实测 10 集已生成的
# .ass 共 362 条 Dialogue，`end == start` 0 条、`end < start` 0 条、相邻重叠 0 条，
# 298 对相邻 cue 在厘秒级恰好首尾相接。插 20–40ms 间隙会让这 298 处每一处都多一次
# 字幕闪断，换来的是一个测不到的问题。理由全文见 sentence_cues 的 docstring。


def test_sentence_cues_hand_off_at_exactly_the_same_instant():
    chunk = VoiceChunk(
        beat_id="b1", index=1, text="第一句。第二句。第三句。", path="c.mp3", duration=9.0
    )
    cues = sentence_cues(chunk, 100.0)
    assert len(cues) == 3
    for before, after in pairwise(cues):
        assert after.start == before.end
    assert cues[0].start == 100.0
    assert cues[-1].end == pytest.approx(109.0)


def test_sentence_cues_never_produce_a_zero_length_cue():
    """最短的一句（1 个字）在最短的 chunk 里也拿得到非零时长。

    结构上的下界：cue 时长 = 句字数 × (chunk 时长 / chunk 字数)，而 render/tts.py 的
    时长体检把 chunk 时长压在 ≈0.11 秒/字以上，所以厘秒精度下也不会舍成 0。
    """
    chunk = VoiceChunk(
        beat_id="b1", index=1, text="好。" + "一二三四五六七八九十" * 5, path="c.mp3", duration=11.6
    )
    cues = sentence_cues(chunk, 0.0)
    assert min(cue.end - cue.start for cue in cues) > 0.01
