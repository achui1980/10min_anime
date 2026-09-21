import pytest

from tenmin.config import DEFAULT_VALIDATE
from tenmin.models import (
    Beat,
    Clip,
    DialogueLine,
    DialogueTrack,
    Hold,
    Script,
    SfxCue,
    Signal,
    SignalReport,
)
from tenmin.script.budget import beat_seconds
from tenmin.script.validate import (
    ANCHOR_OUTSIDE_MAX_RATIO,
    ANCHOR_OVERWRITE_MAX_SECONDS,
    CREDITS_OVERLAP_MAX_RATIO,
    SPEC_MAX_CLIPS_PER_BEAT,
    SPEC_MIN_BEATS,
    AnchorIndex,
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
    """pad=True 时补齐到 SPEC_MIN_BEATS 个节点、并把第二个节点标成 climax，
    避免触发节点数下限与「零个 climax」这两条结构 warning。
    填充 clip 落在 300s 附近，刻意避开测试里用到的 OP(153-225) / ED(1348+) 区间。"""
    rows = list(clips_per_beat)
    if pad:
        while len(rows) < SPEC_MIN_BEATS:
            rows.append([clip(300.0 + 10 * len(rows), 305.0 + 10 * len(rows))])
    beats = []
    for i, clips in enumerate(rows):
        # 末节点做成 outro + 「收尾：」label：B1 的结构校验要求首 hook、末 outro，
        # 不然每一条「warnings == []」的断言都会被结构 warning 污染。原来这里
        # 全是 hook/act，最后一个是 act。
        last = i == len(rows) - 1
        if i == 0:
            role = "hook"
        elif last:
            role = "outro"
        elif i == 1:
            role = "climax"
        else:
            role = "act"
        beats.append(
            Beat(
                id=f"b{i + 1}",
                label=(f"收尾：节点{i + 1}" if last else f"节点{i + 1}"),
                role=role,
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


def test_anchor_tolerance_default():
    """对着权威来源（config）断言，不再经过 validate 那个没人读的模块级别名。"""
    assert DEFAULT_VALIDATE.anchor_tolerance_seconds == pytest.approx(5.0)


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
    """重叠比例低于 CREDITS_OVERLAP_MAX_RATIO 才算「只是擦到片头曲」。
    原来这里写的是 clip(148.0, 160.0)——6.5/12 = 54% 是片头曲画面，旧的「完全落入」
    判据把它放过，正是 B5 要修的缺陷，所以那组数字不能留。"""
    s = make_script([[clip(140.0, 160.0)]])  # 6.5/20 = 33%
    result = run(s, track=make_track(op=(153.486, 224.681)))
    assert len(result.script.beats[0].clips) == 1


def test_clip_for_unknown_episode_is_dropped():
    s = make_script([[Clip(episode=99, start=10.0, end=15.0, visual="画面")]])
    with pytest.raises(ScriptValidationError) as exc:
        run(s)
    assert "不存在的集数" not in str(exc.value)  # 报的是「clip 全丢」而非集数本身
    assert "节点1" in str(exc.value)


def test_anchor_mismatch_beyond_tolerance_overwrites_start():
    # 原来这里 anchor 在 600、clip 在 100（幅度 500 秒），而 B4 之后那个量级会被
    # 拒绝覆写。改成 40 秒的幅度：仍然远超 5 秒容差，是这条规则的正常工作区间。
    track = make_track(lines=[dline(42, 140.0, 144.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    kept = result.script.beats[0].clips[0]
    assert kept.start == pytest.approx(140.0)
    assert kept.end == pytest.approx(146.0)  # 保留原 6 秒时长
    assert any("anchor" in w for w in result.warnings)


def test_anchor_mismatch_within_tolerance_keeps_start():
    track = make_track(lines=[dline(42, 103.0, 107.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    assert result.script.beats[0].clips[0].start == pytest.approx(100.0)
    assert result.warnings == []


def test_anchor_overwrite_clamped_to_episode_end():
    # 幅度 46 秒，落在 ANCHOR_OVERWRITE_MAX_SECONDS 之内，覆写照做。
    track = make_track(duration=1000.0, lines=[dline(42, 996.0, 999.0)])
    s = make_script([[clip(950.0, 980.0, anchors=[42])]])
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
    # 与间隙只重叠 0.633 秒。clip 本身给足 3 秒，避免撞上 min_clip_seconds。
    s = make_script([[clip(1326.0, 1329.0, silent=False)]])
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


# --- AnchorIndex（原先每个 clip 都全量扫一遍 track.lines） ---


def test_anchor_index_returns_lines_in_track_order():
    """原实现是「按 track.lines 顺序过滤」，索引查表后必须还原成同一个顺序。

    _anchor_time 取的是匹配行 start 的最小值，靠顺序的地方是 _check_anchor_coverage
    的告警文本与 min() 的平手语义。
    """
    lines = [dline(30, 300.0, 302.0), dline(10, 100.0, 102.0), dline(20, 200.0, 202.0)]
    index = AnchorIndex(make_track(lines=lines))
    assert [ln.idx for ln in index.matches([10, 20, 30])] == [30, 10, 20]


def test_anchor_index_resolves_merged_from_line_numbers():
    """anchor_lines 里可能写的是被 merge_continuations 并掉的旧行号。"""
    merged = DialogueLine(
        idx=90, start=600.0, end=602.0, text="合并行", raw="合并行", merged_from=[7, 8]
    )
    index = AnchorIndex(make_track(lines=[dline(1, 10.0, 12.0), merged]))
    assert [ln.idx for ln in index.matches([8])] == [90]
    assert [ln.idx for ln in index.matches([90])] == [90]


def test_anchor_index_counts_a_line_hit_twice_only_once():
    """idx 与 merged_from 同时命中时原实现也只把这行算一条（它是个 or 过滤）。"""
    merged = DialogueLine(
        idx=90, start=600.0, end=602.0, text="合并行", raw="合并行", merged_from=[7, 8]
    )
    index = AnchorIndex(make_track(lines=[merged]))
    assert [ln.idx for ln in index.matches([90, 7, 8])] == [90]


def test_anchor_index_keeps_every_segment_sharing_one_idx():
    """双轨拆行让同一个 idx 对应多条 line，索引的值必须是列表。"""
    lines = [dline(7, 10.0, 12.0, "台词"), dline(7, 10.0, 12.0, "独白")]
    index = AnchorIndex(make_track(lines=lines))
    assert [ln.text for ln in index.matches([7])] == ["台词", "独白"]


def test_anchor_index_unknown_line_number_matches_nothing():
    index = AnchorIndex(make_track(lines=[dline(1, 10.0, 12.0)]))
    assert index.matches([9999]) == []


def test_check_script_builds_the_anchor_index_once_per_track(monkeypatch):
    from tenmin.script import validate as validate_module

    built = []
    real = validate_module.AnchorIndex
    monkeypatch.setattr(
        validate_module,
        "AnchorIndex",
        lambda track: (built.append(track), real(track))[1],
    )
    track = make_track(
        lines=[dline(i + 1, i * 2.0, i * 2.0 + 1.5) for i in range(100)]
    )
    s = make_script(
        [[clip(408.9 + i, 453.9 + i, anchors=[i + 1]) for i in range(5)]]
    )
    check_script(s, {2: track}, {})
    assert len(built) == 1


# --- C1：check（纯读）与 repair（返回新对象）的拆分 ---


def test_check_script_does_not_touch_the_input():
    """check 是纯读：连 is_silent_highlight 这种「本来会被回填」的字段都不许动。"""
    s = make_script([[clip(1330.0, 1340.0, silent=False)]])
    before = s.model_dump_json()
    check_script(s, tracks={2: make_track()}, reports={2: make_report(gaps=[(1328.4, 1348.2)])})
    assert s.model_dump_json() == before


def test_repair_script_returns_a_new_object_and_leaves_the_input_alone():
    track = make_track(lines=[dline(42, 140.0, 144.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    before = s.model_dump_json()
    repaired, _ = repair_script(s, tracks={2: track}, reports={2: make_report()})
    assert repaired is not s
    assert repaired.beats[0].clips[0].start == pytest.approx(140.0)
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


# --- B4：anchor 覆写的两个漏洞 ---


def test_anchor_overwrite_reruns_the_op_window_check():
    """漏洞 (a)：覆写后原来只重查了 end <= start，被拽进片头曲的 clip 会保留。"""
    track = make_track(op=(153.486, 224.681), lines=[dline(42, 160.0, 164.0)])
    s = make_script([[clip(10.0, 15.0), clip(210.0, 220.0, anchors=[42])]])
    result = run(s, track=track)
    kept = list(result.script.beats[0].clips)
    assert len(kept) == 1, [(c.start, c.end) for c in kept]
    assert kept[0].start == pytest.approx(10.0)
    assert any("片头" in w for w in result.warnings), result.warnings


def test_anchor_overwrite_clamped_too_short_is_dropped():
    """覆写把起点推到离片尾只剩 0.5 秒时，钳完的 clip 短得没有画面可用。
    原来这里只重查 `end <= start`，0.5 秒的残段照样留到成片里。"""
    track = make_track(duration=1000.0, lines=[dline(42, 999.5, 999.9)])
    s = make_script([[clip(10.0, 15.0), clip(950.0, 960.0, anchors=[42])]])
    result = run(s, track=track)
    assert len(result.script.beats[0].clips) == 1
    assert any("过短" in w for w in result.warnings), result.warnings


def test_anchor_overwrite_beyond_the_cap_is_refused_with_a_warning():
    """漏洞 (b)：_anchor_time 取所有匹配行 start 的最小值，anchor_lines 跨长场景时
    起点会被拉到很早的位置，而这一步原来是**强制覆写**。实测 saijo E02 的 script.json
    因为行号过期，22 个 clip 的覆写幅度达 14.6–118.9 秒。"""
    track = make_track(lines=[dline(42, 700.0, 704.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    kept = result.script.beats[0].clips[0]
    assert kept.start == pytest.approx(100.0), "幅度超过上限就不该改写"
    assert kept.end == pytest.approx(106.0)
    assert any("没有覆写" in w for w in result.warnings), result.warnings


def test_anchor_overwrite_within_the_cap_still_happens():
    track = make_track(lines=[dline(42, 130.0, 134.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    assert result.script.beats[0].clips[0].start == pytest.approx(130.0)


def test_anchor_overwrite_cap_constant():
    assert ANCHOR_OVERWRITE_MAX_SECONDS == pytest.approx(60.0)


# --- B5：OP/ED 判据从「完全落入」改成重叠比例 ---


def test_clip_mostly_inside_op_is_dropped():
    """真实缺陷：saijo E06 stage1 的 clip 195.6-225.6 有 19.1 秒（63.5%）压在
    OP(104.5, 214.7) 上，只有 10.9 秒是正片；旧的「完全落入」判据原样放过它，
    成片里就出现片头曲画面。"""
    s = make_script([[clip(10.0, 15.0), clip(195.6, 225.6)]])
    result = run(s, track=make_track(op=(104.5, 214.7)))
    assert len(result.script.beats[0].clips) == 1
    assert any("片头" in w for w in result.warnings), result.warnings


def test_clip_grazing_op_tail_is_kept():
    """实测 saijo E07/E09 的擦边重叠只有 1.5 秒 / 0.3 秒（1.1%-12.2%），无害，必须放过。"""
    s = make_script([[clip(258.0, 270.0)]])
    result = run(s, track=make_track(op=(181.5, 259.5)))
    assert len(result.script.beats[0].clips) == 1
    assert result.warnings == []


def test_credits_overlap_max_ratio_constant():
    assert CREDITS_OVERLAP_MAX_RATIO == pytest.approx(0.5)


# --- B3：最小 clip 时长 ---


def test_flash_frame_clip_is_dropped():
    """0.2 秒的 clip 一路进渲染就是一帧闪屏。实测 263 个真实 clip 最短 3.09 秒。"""
    s = make_script([[clip(10.0, 15.0), clip(50.0, 50.2)]])
    result = run(s)
    assert len(result.script.beats[0].clips) == 1
    assert any("过短" in w for w in result.warnings), result.warnings


def test_clip_at_exactly_the_minimum_is_kept():
    s = make_script([[clip(50.0, 50.0 + DEFAULT_VALIDATE.min_clip_seconds)]])
    assert len(run(s).script.beats[0].clips) == 1


def test_min_clip_seconds_default():
    assert DEFAULT_VALIDATE.min_clip_seconds == pytest.approx(1.5)


# --- A1：时间轴单调性 ---


def timeline_script(starts, roles=None):
    """按给定起点造一串 beat。roles 不给时全是 act（都参与单调性检查）。"""
    beats = []
    for i, start in enumerate(starts):
        beats.append(
            Beat(
                id=f"t{i + 1}",
                label=f"阶段{i + 1}",
                role=(roles[i] if roles else "act"),
                narration="旁白" * 20,
                clips=[clip(start, start + 10.0)],
            )
        )
    return Script(show="才女的侍从", episodes=[2], beats=beats)


def test_beats_going_backwards_beyond_the_cap_warns():
    """提示词把「事件顺序必须与时间戳一致」列为废稿条件，但代码里原来没有任何
    跨 beat 的时序检查。实测 saijo E02 的 act4 起点比 act3 早 230 秒。"""
    s = timeline_script([100.0, 700.0, 470.0, 900.0])
    result = run(s)
    hits = [w for w in result.warnings if "倒退" in w]
    assert len(hits) == 1, result.warnings
    assert "阶段3" in hits[0]


def test_beats_going_backwards_within_the_cap_does_not_warn():
    """实测 13 份样本里用中位数口径会抓到一个只倒退 5.0 秒的抖动，纯噪声。"""
    s = timeline_script([100.0, 200.0, 190.0, 300.0])
    assert [w for w in run(s).warnings if "倒退" in w] == []


def test_hook_is_exempt_from_monotonicity():
    """hook 本来就允许抓全片任何位置的钩子。"""
    s = timeline_script([1200.0, 100.0, 300.0], roles=["hook", "act", "act"])
    assert [w for w in run(s).warnings if "倒退" in w] == []


def test_outro_is_exempt_from_monotonicity():
    s = timeline_script([100.0, 900.0, 200.0], roles=["hook", "act", "outro"])
    assert [w for w in run(s).warnings if "倒退" in w] == []


def test_timeline_regression_cap_constant():
    assert DEFAULT_VALIDATE.timeline_regression_max_seconds == pytest.approx(60.0)


# --- A2：hold / sfx 的落点不能超出本节点旁白 ---


def test_hold_at_beyond_the_beat_span_warns():
    """at 是相对本 beat 旁白起点的偏移。超过本 beat 的总跨度意味着留白落在旁白之外，
    而 render/chunks.py 的 assign_holds 会静默把它贴到最后一句。"""
    s = make_script([[clip(10.0, 30.0)]])
    s.beats[0].audio.holds = [Hold(at=300.0, duration=2.0, quote="金句")]
    hits = [w for w in run(s).warnings if "留白落点" in w]
    assert len(hits) == 1, run(s).warnings


def test_hold_at_at_the_very_end_of_the_narration_does_not_warn():
    """实测 76 个真实 hold 的 at/旁白秒数最大 1.004——「把留白放在这段旁白最后」
    是正常创作，不能报。"""
    s = make_script([[clip(10.0, 30.0)]])
    seconds = beat_seconds(s.beats[0])
    s.beats[0].audio.holds = [Hold(at=seconds, duration=2.0, quote="金句")]
    assert [w for w in run(s).warnings if "留白落点" in w] == []


def test_sfx_at_beyond_the_beat_span_warns():
    s = make_script([[clip(10.0, 30.0)]])
    s.beats[0].audio.sfx = [SfxCue(at=300.0, cue="impact")]
    hits = [w for w in run(s).warnings if "音效落点" in w]
    assert len(hits) == 1, run(s).warnings


# --- A3：画面总时长 vs 旁白时长 ---



def test_beat_with_far_too_little_footage_warns():
    """clip 太少时 render/timeline.py 会按 ratio = 旁白/画面 把每个 clip 往后延长，
    延到超出源片长再钳到片尾，成片画面错位。"""
    # make_script 的旁白是 40 字 ≈ 8.9 秒；1.6 秒画面 → 拉伸 5.6 倍
    s = make_script([[clip(10.0, 11.6)]])
    hits = [w for w in run(s).warnings if "画面只有" in w]
    assert len(hits) == 1, run(s).warnings


def test_beat_with_far_too_much_footage_warns():
    s = make_script([[clip(10.0, 800.0)]])  # 790 秒画面 vs 8.9 秒旁白 → 只用得上 1.1%
    hits = [w for w in run(s).warnings if "画面多达" in w]
    assert len(hits) == 1, run(s).warnings


def test_beat_within_the_measured_stretch_range_does_not_warn():
    """实测 85 个真实 beat 的 画面/旁白 比值落在 0.396–5.176（拉伸 0.19–2.53 倍），
    这整段区间都必须放过。"""
    s = make_script([[clip(10.0, 25.0)]])
    seconds = beat_seconds(s.beats[0])
    for ratio in (0.40, 1.0, 5.0):
        one = make_script([[clip(10.0, 10.0 + seconds * ratio)]])
        assert [w for w in run(one).warnings if "画面" in w] == [], ratio


def test_stretch_bounds_constants():
    assert DEFAULT_VALIDATE.stretch_max == pytest.approx(4.0)
    assert DEFAULT_VALIDATE.stretch_min == pytest.approx(0.125)


def test_footage_budget_uses_the_render_layer_clip_sum(monkeypatch):
    """A3 的分母必须走 render/timeline.py 的 beat_clip_seconds。

    那个文件的 docstring 写着「ratio 的分母只能有一处算法」，而 validate 这边原来有
    一份内联的 `sum(clip.duration for clip in beat.clips)` —— 两份实现，谁改都不会
    惊动另一份，而它们算的是同一个 ratio 的同一个分母。
    """
    from tenmin.script import validate as validate_module

    seen: list[str] = []
    real = validate_module.beat_clip_seconds

    def spy(beat):
        seen.append(beat.id)
        return real(beat)

    monkeypatch.setattr(validate_module, "beat_clip_seconds", spy)
    s = make_script([[clip(10.0, 25.0)]])
    check_script(s, {2: make_track()}, {2: make_report()})
    assert seen == [b.id for b in s.beats]


# --- render.rate 必须一路穿到 validate（M2）-------------------------------
#
# validate 的两处 `beat_seconds(beat)` 原来不传 rate，落到 budget.DEFAULT_RATE
# （"+0%"）；而 render/chunks.py 的 assign_holds 传了 rate，两边的注释都写着
# 「同口径」。rate != "+0%" 时同一个 hold.at 在 script 阶段与 voice 阶段拿到两个
# 不同上界（一个放过一个报警），stretch_max/min 那两个实测阈值的前提也被整体乘上
# 1/speed_factor(rate)。


@pytest.mark.parametrize("rate", ["+20%", "-20%", "+50%"])
def test_cue_offset_span_matches_the_voice_stage_span(rate):
    """validate 判 hold.at 的上界，必须跟 voice 阶段真正用的那个上界是同一个数。

    判据刻意做成「两边报不报是同一个布尔值」，而不是「validate 在某个算出来的数字上
    不报」：后者会把 voice 阶段的实现细节抄一遍进测试，而这条不变量本身就是
    「两个阶段不许分叉」。
    """
    from tenmin.render.chunks import assign_holds, split_sentences

    seen: list[bool] = []
    for offset in (-3.0, -0.001, 0.0, 5.0):
        s = make_script([[clip(10.0, 30.0)]])
        beat = s.beats[0]
        sentences = split_sentences(beat.narration)
        # 跨度 = 旁白 + 本节点全部留白，所以要把下面那个 hold 的 2.0 秒算进来。
        span = beat_seconds(beat, rate=rate) + 2.0
        beat.audio.holds = [Hold(at=max(0.0, span + offset), duration=2.0, quote="金句")]

        voice_warnings: list[str] = []
        assign_holds(sentences, beat.audio.holds, rate=rate, warnings=voice_warnings)
        voice_hit = any("超出本节点跨度" in w for w in voice_warnings)

        script_warnings = check_script(
            s, {2: make_track()}, {2: make_report()}, rate=rate
        )
        script_hit = any("留白落点" in w for w in script_warnings)

        assert voice_hit == script_hit, (offset, voice_warnings, script_warnings)
        seen.append(voice_hit)
    # 两种结论都必须出现过，否则这个测试什么都没证明（比如两边恒不报也会绿）。
    assert set(seen) == {True, False}, seen


def test_footage_budget_span_follows_the_configured_rate():
    """A3 的拉伸倍率分母是旁白秒数，语速一变它就变。"""
    # 40 字旁白 ≈ 8.889 秒（+0%）。画面给 2.3 秒 → 拉伸 3.87 倍，默认语速下不报。
    s = make_script([[clip(10.0, 12.3)]])
    assert [w for w in check_script(s, {2: make_track()}, {2: make_report()})
            if "画面只有" in w] == []
    # rate="-40%" 让旁白变成 14.8 秒 → 拉伸 6.4 倍，超过 stretch_max=4，必须报。
    hits = [
        w
        for w in check_script(s, {2: make_track()}, {2: make_report()}, rate="-40%")
        if "画面只有" in w
    ]
    assert len(hits) == 1, hits


def test_validate_script_and_repair_script_take_rate():
    """三个入口的签名必须一致 —— single.py 手上只有一个 rate，三处都要能收。"""
    s = make_script([[clip(10.0, 30.0)]])
    s.beats[0].audio.holds = [Hold(at=300.0, duration=2.0, quote="金句")]
    tracks, reports = {2: make_track()}, {2: make_report()}
    repair_script(s, tracks, reports, rate="+20%")
    result = validate_script(s, tracks, reports, rate="+20%")
    assert any("留白落点" in w for w in result.warnings)


# --- B1：节点结构约定（提示词 single_episode.md:34,36,37）---


def structure_script(roles, labels=None):
    beats = []
    for i, role in enumerate(roles):
        beats.append(
            Beat(
                id=f"s{i + 1}",
                label=(labels[i] if labels else f"阶段{i + 1}"),
                role=role,
                narration="旁白" * 20,
                clips=[clip(300.0 + i * 10, 305.0 + i * 10)],
            )
        )
    return Script(show="才女的侍从", episodes=[2], beats=beats)


def test_first_beat_must_be_hook():
    s = structure_script(["act", "act", "outro"])
    hits = [w for w in run(s).warnings if "首节点" in w]
    assert len(hits) == 1, run(s).warnings


def test_last_beat_must_be_outro():
    s = structure_script(["hook", "act", "act"])
    hits = [w for w in run(s).warnings if "role 是" in w and "末节点" in w]
    assert len(hits) == 1, run(s).warnings


def test_last_beat_label_must_start_with_the_outro_prefix():
    s = structure_script(["hook", "act", "outro"], labels=["Hook 开场", "阶段一", "大结局"])
    hits = [w for w in run(s).warnings if "收尾：" in w]
    assert len(hits) == 1, run(s).warnings


def test_too_many_beats_warns_but_does_not_raise():
    s = structure_script(["hook"] + ["act"] * 8 + ["outro"])
    hits = [w for w in run(s).warnings if "节点数" in w]
    assert len(hits) == 1, run(s).warnings


def test_more_than_one_climax_warns():
    s = structure_script(["hook", "climax", "climax", "outro"])
    hits = [w for w in run(s).warnings if "climax" in w]
    assert len(hits) == 1, run(s).warnings


def test_the_shape_real_products_use_warns_about_nothing():
    """实测 13 份真实 script.json 的 role 序列全是
    hook + act* + 一个 climax + outro、6–7 个节点、末节点 label 以「收尾：」开头。
    这个形状必须一条 warning 都不报。"""
    s = structure_script(
        ["hook", "act", "act", "act", "climax", "act", "outro"],
        labels=["Hook 开场", "阶段一", "阶段二", "阶段三", "阶段四", "阶段五", "收尾：完"],
    )
    assert run(s).warnings == []


# --- 结构：后补的三条 warning 与唯一一条判错 ---


def test_too_few_beats_warns_but_does_not_raise():
    """提示词要求 5–8 个节点，但 4 个节点照样能出片（见 ValidateConfig.min_beats
    的注释），所以只给 warning。判错会烧掉一次几百秒的调用并赌上整集掉件。"""
    s = structure_script(["hook", "act", "climax", "outro"])
    hits = [w for w in run(s).warnings if "低于提示词要求" in w]
    assert len(hits) == 1, run(s).warnings


def test_beat_count_at_the_spec_floor_does_not_warn():
    """边界：恰好 SPEC_MIN_BEATS 个不报。"""
    s = structure_script(["hook", "act", "climax", "act", "outro"])
    assert len(s.beats) == SPEC_MIN_BEATS
    assert [w for w in run(s).warnings if "低于提示词要求" in w] == []


def test_no_climax_warns_but_does_not_raise():
    """零个 climax 只给 warning 是刻意的：beat.role == "climax" 在 src/ 里
    **没有任何消费者**（全项目只有 validate.py 读 role，且只用 hook/outro 豁免
    时间线检查），render / timeline / audio / tts / docgen 一处都不读。为一个
    不影响输出的字段判错等于白烧一次调用。"""
    s = structure_script(["hook", "act", "act", "act", "outro"])
    hits = [w for w in run(s).warnings if "没有 role=climax" in w]
    assert len(hits) == 1, run(s).warnings


def test_too_many_clips_in_one_beat_warns_but_does_not_raise():
    """单节点 clip 超上限是量变不是质变：每个 clip 被 render/timeline.py 的
    per-beat ratio 摊得更短、切点更多，但照样出片。"""
    rows = [
        [
            clip(100.0 + 10 * i, 105.0 + 10 * i)
            for i in range(SPEC_MAX_CLIPS_PER_BEAT + 1)
        ]
    ]
    s = make_script(rows)
    hits = [w for w in run(s).warnings if "个 clip，超过提示词要求" in w]
    assert len(hits) == 1, run(s).warnings


def test_clip_count_at_the_spec_ceiling_does_not_warn():
    """边界：恰好 SPEC_MAX_CLIPS_PER_BEAT 个不报。"""
    rows = [
        [clip(100.0 + 10 * i, 105.0 + 10 * i) for i in range(SPEC_MAX_CLIPS_PER_BEAT)]
    ]
    assert run(make_script(rows)).warnings == []


def test_too_many_holds_raises():
    """全片留白数量是**唯一**升级成判错的创作约定：每处 hold 是 2–4 秒旁白静音，
    超上限那一版实测写了 11 处 = 22–44 秒，占 240 秒预算的 1/6 到 1/5，而且 hold
    时长还会从旁白字数预算里扣，直接挤掉解说内容。模型改一轮就能修。"""
    s = make_script([[clip(10.0, 15.0)]])
    over = DEFAULT_VALIDATE.max_holds + 1
    s.beats[0].audio.holds = [
        Hold(at=1.0 + i, duration=2.0, quote="金句") for i in range(over)
    ]
    with pytest.raises(ScriptValidationError) as exc:
        run(s)
    assert "留白" in str(exc.value)
    # 落盘给 pipeline 用的那一版必须带着（一次真实调用可达 561 秒）。
    assert exc.value.script is not None


def test_holds_at_the_cap_do_not_raise():
    """边界：恰好 max_holds 处不判错。上限取 8 而不是提示词原来写的 6，是因为
    tests/snapshots/saijo_e02.script.json（当质量基准用的样本）本身有 8 处 ——
    按 6 判错第一个被拦的就是自家黄金快照。提示词已同步改成 3–8。"""
    s = make_script([[clip(10.0, 15.0)]])
    s.beats[0].audio.holds = [
        Hold(at=1.0 + i, duration=2.0, quote="金句")
        for i in range(DEFAULT_VALIDATE.max_holds)
    ]
    assert len(run(s).script.beats) == SPEC_MIN_BEATS


def test_hold_count_is_summed_across_beats():
    """判据是**全片**留白数，不是单节点。提示词写的是「全集给 3–8 处」。"""
    s = make_script([[clip(10.0, 15.0)], [clip(100.0, 105.0)]])
    per_beat = [Hold(at=1.0 + i, duration=2.0, quote="金句") for i in range(5)]
    s.beats[0].audio.holds = list(per_beat)
    s.beats[1].audio.holds = list(per_beat)
    with pytest.raises(ScriptValidationError):
        run(s)


# --- B2：narration 非空 ---


def test_empty_narration_warns():
    """内部 Beat.narration 刻意保持宽松（人手清空是合法编辑），但 validate 该说一声。"""
    s = make_script([[clip(10.0, 15.0)]])
    s.beats[0].narration = "   "
    hits = [w for w in run(s).warnings if "旁白为空" in w]
    assert len(hits) == 1, run(s).warnings


# --- B8：缺 SignalReport 时不再静默当成「无静音间隙」 ---


def test_missing_signal_report_warns_instead_of_silently_clearing_the_flag():
    """reports.get(...) 为 None 时原来静默退化成「本集没有静音间隙」，
    is_silent_highlight 全 False，一点痕迹都不留。而 single.py 只放一集的 report，
    跨集 clip 会静默丢失静音标记。"""
    s = make_script([[clip(10.0, 15.0)]])
    result = validate_script(s, tracks={2: make_track()}, reports={})
    hits = [w for w in result.warnings if "静音间隙信号" in w]
    assert len(hits) == 1, result.warnings
    assert result.script.beats[0].clips[0].is_silent_highlight is False


def test_missing_signal_report_warns_once_per_episode_not_per_clip():
    s = make_script([[clip(10.0, 15.0), clip(20.0, 25.0), clip(30.0, 35.0)]])
    result = validate_script(s, tracks={2: make_track()}, reports={})
    assert len([w for w in result.warnings if "静音间隙信号" in w]) == 1


# --- B9：留白金句反查改成归一化匹配 ---


def test_quote_is_matched_after_dropping_punctuation():
    """原来用整行完全相等反查，标点差一个就找不到，于是 `if not matches: continue`
    让大量 hold 静默跳过检查。实测 61 个真实 hold 的定位率只有 67.2%。"""
    track = make_track(lines=[dline(17, 41.1, 43.0, "原来，您对我的认知只有这种程度……")])
    s = with_holds(make_script([[clip(0.6, 20.0)]]), [hold("原来您对我的认知只有这种程度")])
    result = run(s, track=track)
    assert len(result.warnings) == 1, result.warnings
    assert "留白金句" in result.warnings[0]


def test_quote_that_is_a_fragment_of_a_longer_line_is_matched():
    track = make_track(lines=[dline(17, 41.1, 43.0, "所以我说了，原来您对我的认知只有这种程度")])
    s = with_holds(make_script([[clip(0.6, 20.0)]]), [hold("原来您对我的认知只有这种程度")])
    assert len(run(s, track=track).warnings) == 1


def test_quote_stitched_from_two_cues_matches_the_longer_half():
    """LLM 常把相邻两条字幕缝成一句金句。要求「行是金句的一大半」才算命中。"""
    track = make_track(
        lines=[
            dline(17, 41.1, 43.0, "伊月肯听我的话"),
            dline(18, 43.0, 45.5, "只是因为那是他的工作吗"),
        ]
    )
    s = with_holds(
        make_script([[clip(0.6, 20.0)]]), [hold("伊月肯听我的话 只是因为那是他的工作吗")]
    )
    assert len(run(s, track=track).warnings) == 1


def test_a_one_character_line_does_not_match_every_quote():
    """反向包含必须带长度闸，否则「嗯」这种行会命中任何金句，把检查稀释成噪声。"""
    track = make_track(lines=[dline(17, 41.1, 43.0, "嗯")])
    s = with_holds(make_script([[clip(0.6, 20.0)]]), [hold("原来您对我的认知只有这种程度")])
    assert run(s, track=track).warnings == []
