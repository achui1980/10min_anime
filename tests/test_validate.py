import pytest

from tenmin.models import (
    Beat,
    Clip,
    DialogueLine,
    DialogueTrack,
    Hold,
    Script,
    Signal,
    SignalReport,
)
from tenmin.script.validate import (
    ANCHOR_OUTSIDE_MAX_RATIO,
    ANCHOR_TOLERANCE_SECONDS,
    ScriptValidationError,
    check_script,
    repair_script,
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


# --- 校验 A：留白金句的原声必须落在本 beat 某个 clip 内 ---

QUOTE = "原来您对我的认知只有这种程度"


def hold(quote, at=5.0, duration=2.0):
    return Hold(at=at, duration=duration, quote=quote, note="留给原声")


def with_holds(script, holds, beat_index=0):
    script.beats[beat_index].audio.holds = list(holds)
    return script


def test_hold_quote_outside_every_clip_warns():
    """真实缺陷 A：金句在 41.1s，clip 却是 0.6-20.0 与 167.8-209.8。"""
    track = make_track(lines=[dline(1, 5.0, 8.0, "别的台词"), dline(17, 41.1, 43.0, QUOTE)])
    s = with_holds(
        make_script([[clip(0.6, 20.0), clip(167.8, 209.8)]]), [hold(QUOTE)]
    )
    result = run(s, track=track)
    assert len(result.warnings) == 1, result.warnings
    assert "留白金句" in result.warnings[0]
    assert "41.1" in result.warnings[0]


def test_hold_quote_inside_a_clip_does_not_warn():
    track = make_track(lines=[dline(17, 41.1, 43.0, QUOTE)])
    s = with_holds(make_script([[clip(40.0, 60.0)]]), [hold(QUOTE)])
    result = run(s, track=track)
    assert result.warnings == []


def test_hold_quote_touching_clip_edge_counts_as_inside():
    """金句尾巴伸进 clip 起点之后，仍算放得出来。"""
    track = make_track(lines=[dline(17, 38.0, 41.0, QUOTE)])
    s = with_holds(make_script([[clip(40.0, 60.0)]]), [hold(QUOTE)])
    assert run(s, track=track).warnings == []


def test_hold_quote_not_found_in_track_is_skipped_silently():
    """跨 cue 拼接／双轨半句在字幕里找不到完全相等的行，这是已知合法情况。"""
    track = make_track(lines=[dline(17, 41.1, 43.0, "字幕里真正的那行")])
    s = with_holds(make_script([[clip(0.6, 20.0)]]), [hold("LLM 自己缝出来的半句")])
    assert run(s, track=track).warnings == []


def test_hold_quote_matches_any_of_multiple_occurrences():
    """同一句台词出现多次，只要有一次落在 clip 里就不报。"""
    track = make_track(
        lines=[dline(17, 41.1, 43.0, QUOTE), dline(88, 300.0, 302.0, QUOTE)]
    )
    s = with_holds(make_script([[clip(295.0, 310.0)]]), [hold(QUOTE)])
    assert run(s, track=track).warnings == []


def test_hold_quote_is_compared_after_stripping():
    track = make_track(lines=[dline(17, 41.1, 43.0, QUOTE)])
    s = with_holds(make_script([[clip(0.6, 20.0)]]), [hold(f"  {QUOTE} ")])
    result = run(s, track=track)
    assert len(result.warnings) == 1, result.warnings
    assert "留白金句" in result.warnings[0]


# --- 校验 B：clip 时间窗必须覆盖自己的 anchor_lines ---


def test_anchor_outside_max_ratio_constant():
    assert ANCHOR_OUTSIDE_MAX_RATIO == pytest.approx(0.5)


def test_clip_window_missing_most_anchors_warns():
    """真实缺陷 B：clip 408.9-453.9 只有 45 秒，anchors 却铺到 542.9。"""
    track = make_track(
        lines=[
            dline(1, 410.0, 412.0),
            dline(2, 500.0, 502.0),
            dline(3, 520.0, 522.0),
            dline(4, 540.8, 542.9),
        ]
    )
    s = make_script([[clip(408.9, 453.9, anchors=[1, 2, 3, 4])]])
    result = run(s, track=track)
    assert len(result.warnings) == 1, result.warnings
    assert "anchor 行有" in result.warnings[0]
    assert "3/4" in result.warnings[0]
    kept = result.script.beats[0].clips[0]
    assert kept.start == pytest.approx(408.9)  # 只报警，不改时间戳
    assert kept.end == pytest.approx(453.9)


def test_clip_window_missing_few_anchors_does_not_warn():
    track = make_track(
        lines=[dline(1, 410.0, 412.0), dline(2, 420.0, 422.0), dline(3, 455.0, 457.0)]
    )
    s = make_script([[clip(408.9, 453.9, anchors=[1, 2, 3])]])
    assert run(s, track=track).warnings == []


def test_clip_window_missing_exactly_half_anchors_does_not_warn():
    """比例正好等于阈值不报，只有严格超过才报。"""
    track = make_track(
        lines=[
            dline(1, 410.0, 412.0),
            dline(2, 420.0, 422.0),
            dline(3, 500.0, 502.0),
            dline(4, 520.0, 522.0),
        ]
    )
    s = make_script([[clip(408.9, 453.9, anchors=[1, 2, 3, 4])]])
    assert run(s, track=track).warnings == []


def test_anchor_coverage_counts_merged_lines():
    """合并过的行按 merged_from 反查，与现有 anchor 校验口径一致。"""
    merged = DialogueLine(
        idx=90, start=600.0, end=602.0, text="合并行", raw="合并行", merged_from=[7, 8]
    )
    track = make_track(lines=[dline(1, 410.0, 412.0), merged, dline(3, 700.0, 702.0)])
    s = make_script([[clip(408.9, 453.9, anchors=[1, 7, 3])]])
    result = run(s, track=track)
    assert len(result.warnings) == 1, result.warnings
    assert "2/3" in result.warnings[0]


def test_anchor_coverage_ignores_clip_without_matching_anchors():
    s = make_script([[clip(408.9, 453.9, anchors=[9999])]])
    assert run(s).warnings == []


# --- C1：check（纯读）与 repair（返回新对象）的拆分 ---


def test_check_script_does_not_touch_the_input():
    """check 是纯读：连 is_silent_highlight 这种「本来会被回填」的字段都不许动。"""
    s = make_script([[clip(1330.0, 1340.0, silent=False)]])
    before = s.model_dump_json()
    check_script(s, tracks={2: make_track()}, reports={2: make_report(gaps=[(1328.4, 1348.2)])})
    assert s.model_dump_json() == before


def test_repair_script_returns_a_new_object_and_leaves_the_input_alone():
    track = make_track(lines=[dline(42, 600.0, 604.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    before = s.model_dump_json()
    repaired, _ = repair_script(s, tracks={2: track}, reports={2: make_report()})
    assert repaired is not s
    assert repaired.beats[0].clips[0].start == pytest.approx(600.0)
    assert s.model_dump_json() == before, "输入必须一字未改"


def test_validate_script_no_longer_mutates_the_input_script():
    """原来 validate_script 就地改写并把同一个对象塞回 ValidationResult，
    调用方拿不到「校验前」那一版，无法先校验后比较。"""
    s = make_script([[clip(1330.0, 1340.0, silent=False)]])
    before = s.model_dump_json()
    result = run(s, report=make_report(gaps=[(1328.4, 1348.2)]))
    assert result.script is not s
    assert result.script.beats[0].clips[0].is_silent_highlight is True
    assert s.model_dump_json() == before
