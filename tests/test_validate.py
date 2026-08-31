import pytest

from tenmin.models import (
    Beat,
    Clip,
    DialogueLine,
    DialogueTrack,
    Script,
    Signal,
    SignalReport,
)
from tenmin.script.validate import (
    ANCHOR_TOLERANCE_SECONDS,
    ScriptValidationError,
    validate_script,
)


def dline(idx, start, end, text="台词"):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text)


def make_track(duration=1416.0, op=None, ed=None, lines=None):
    return DialogueTrack(
        episode=2,
        duration=duration,
        op_range=op,
        ed_range=ed,
        lines=lines if lines is not None else [dline(1, 10.0, 12.0)],
    )


def make_report(gaps=()):
    return SignalReport(
        episode=2,
        silent_gaps=[
            Signal(
                start=g[0],
                end=g[1],
                source="gap",
                strength=4,
                detail=f"gap:{g[1] - g[0]:.1f}s",
            )
            for g in gaps
        ],
    )


def clip(start, end, anchors=(), silent=False):
    return Clip(
        episode=2,
        start=start,
        end=end,
        visual="画面",
        anchor_lines=list(anchors),
        is_silent_highlight=silent,
    )


def make_script(clips_per_beat, *, pad=True):
    """pad=True 时补齐到 MIN_BEATS 个节点，避免触发节点数下限检查。
    填充 clip 落在 300s 附近，刻意避开测试里用到的 OP(153-225) / ED(1348+) 区间。"""
    rows = list(clips_per_beat)
    if pad:
        while len(rows) < 3:
            rows.append([clip(300.0 + 10 * len(rows), 305.0 + 10 * len(rows))])
    beats = []
    for i, clips in enumerate(rows):
        beats.append(
            Beat(
                id=f"b{i + 1}",
                label=f"节点{i + 1}",
                role="hook" if i == 0 else "act",
                narration="旁白" * 20,
                clips=list(clips),
            )
        )
    return Script(show="才女的侍从", episodes=[2], beats=beats)


def run(script, track=None, report=None):
    return validate_script(
        script,
        tracks={2: track or make_track()},
        reports={2: report or make_report()},
    )


def test_valid_script_passes_unchanged():
    s = make_script([[clip(10.0, 15.0)], [clip(100.0, 105.0)]])
    result = run(s)
    assert result.warnings == []
    assert len(result.script.beats[0].clips) == 1


def test_anchor_tolerance_constant():
    assert ANCHOR_TOLERANCE_SECONDS == pytest.approx(5.0)


def test_clip_past_episode_end_is_dropped():
    s = make_script([[clip(10.0, 15.0), clip(9000.0, 9005.0)]])
    result = run(s, track=make_track(duration=1416.0))
    assert len(result.script.beats[0].clips) == 1
    assert any("越界" in w for w in result.warnings)


def test_clip_with_negative_start_is_dropped():
    s = make_script([[clip(10.0, 15.0), clip(-5.0, 2.0)]])
    result = run(s)
    assert len(result.script.beats[0].clips) == 1


def test_clip_with_non_positive_duration_is_dropped():
    s = make_script([[clip(10.0, 15.0), clip(50.0, 50.0)]])
    result = run(s)
    assert len(result.script.beats[0].clips) == 1


def test_clip_inside_op_is_dropped():
    s = make_script([[clip(10.0, 15.0), clip(160.0, 200.0)]])
    result = run(s, track=make_track(op=(153.486, 224.681)))
    assert len(result.script.beats[0].clips) == 1
    assert any("片头" in w for w in result.warnings)


def test_clip_inside_ed_is_dropped():
    s = make_script([[clip(10.0, 15.0), clip(1360.0, 1380.0)]])
    result = run(s, track=make_track(ed=(1348.18, 1416.622)))
    assert len(result.script.beats[0].clips) == 1
    assert any("片尾" in w for w in result.warnings)


def test_clip_partially_overlapping_op_is_kept():
    s = make_script([[clip(148.0, 160.0)]])
    result = run(s, track=make_track(op=(153.486, 224.681)))
    assert len(result.script.beats[0].clips) == 1


def test_clip_for_unknown_episode_is_dropped():
    s = make_script([[Clip(episode=99, start=10.0, end=15.0, visual="画面")]])
    with pytest.raises(ScriptValidationError) as exc:
        run(s)
    assert "不存在的集数" not in str(exc.value)  # 报的是「clip 全丢」而非集数本身
    assert "节点1" in str(exc.value)


def test_anchor_mismatch_beyond_tolerance_overwrites_start():
    track = make_track(lines=[dline(42, 600.0, 604.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    kept = result.script.beats[0].clips[0]
    assert kept.start == pytest.approx(600.0)
    assert kept.end == pytest.approx(606.0)  # 保留原 6 秒时长
    assert any("anchor" in w for w in result.warnings)


def test_anchor_mismatch_within_tolerance_keeps_start():
    track = make_track(lines=[dline(42, 103.0, 107.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    assert result.script.beats[0].clips[0].start == pytest.approx(100.0)
    assert result.warnings == []


def test_anchor_overwrite_clamped_to_episode_end():
    track = make_track(duration=1000.0, lines=[dline(42, 996.0, 999.0)])
    s = make_script([[clip(100.0, 130.0, anchors=[42])]])
    result = run(s, track=track)
    kept = result.script.beats[0].clips[0]
    assert kept.end <= 1000.0
    assert kept.end > kept.start


def test_unknown_anchor_line_is_ignored():
    s = make_script([[clip(100.0, 106.0, anchors=[9999])]])
    result = run(s)
    assert result.script.beats[0].clips[0].start == pytest.approx(100.0)


def test_silent_highlight_recomputed_true():
    s = make_script([[clip(1330.0, 1340.0, silent=False)]])
    result = run(s, report=make_report(gaps=[(1328.367, 1348.18)]))
    assert result.script.beats[0].clips[0].is_silent_highlight is True


def test_silent_highlight_recomputed_false_when_llm_lied():
    s = make_script([[clip(10.0, 15.0, silent=True)]])
    result = run(s, report=make_report(gaps=[(1328.367, 1348.18)]))
    assert result.script.beats[0].clips[0].is_silent_highlight is False


def test_silent_highlight_needs_one_second_overlap():
    s = make_script([[clip(1328.0, 1328.9, silent=False)]])
    result = run(s, report=make_report(gaps=[(1328.367, 1348.18)]))
    assert result.script.beats[0].clips[0].is_silent_highlight is False


def test_beat_losing_all_clips_raises():
    s = make_script([[clip(10.0, 15.0)], [clip(9000.0, 9005.0)]])
    with pytest.raises(ScriptValidationError) as exc:
        run(s)
    assert "节点2" in str(exc.value)


def test_script_with_no_beats_raises():
    s = Script(show="才女的侍从", episodes=[2], beats=[])
    with pytest.raises(ScriptValidationError):
        run(s)


def test_script_with_too_few_beats_raises():
    s = make_script([[clip(10.0, 15.0)], [clip(20.0, 25.0)]], pad=False)
    with pytest.raises(ScriptValidationError) as exc:
        run(s)
    assert "节点数" in str(exc.value)


def test_three_beats_is_enough():
    s = make_script(
        [[clip(10.0, 15.0)], [clip(20.0, 25.0)], [clip(30.0, 35.0)]], pad=False
    )
    assert len(run(s).script.beats) == 3
