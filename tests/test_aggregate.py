import random

import pytest

from tenmin.models import DialogueLine, DialogueTrack, Signal
from tenmin.signals import aggregate as aggregate_module
from tenmin.signals.aggregate import aggregate, build_report


def sig(start, end, source, strength, detail="x", anchors=None):
    return Signal(
        start=start,
        end=end,
        source=source,
        strength=strength,
        detail=detail,
        anchor_lines=anchors or [],
    )


def dline(idx, start, end, text, kind="dialogue"):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind=kind)


def test_aggregate_empty_returns_empty():
    assert aggregate([], track=None) == []


def test_single_signal_keeps_its_strength():
    highlights = aggregate([sig(10.0, 25.0, "gap", 4, "gap:15.0s")], track=None)
    assert len(highlights) == 1
    assert highlights[0].strength == 4
    assert highlights[0].triggers == ["gap:15.0s"]


def test_far_apart_signals_stay_separate():
    highlights = aggregate(
        [sig(0.0, 5.0, "gap", 2), sig(100.0, 105.0, "gap", 2)], track=None
    )
    assert len(highlights) == 2


def test_near_signals_from_same_source_do_not_stack():
    highlights = aggregate(
        [sig(0.0, 5.0, "gap", 2, "gap:5.0s"), sig(6.0, 9.0, "gap", 2, "gap:3.0s")],
        track=None,
    )
    assert len(highlights) == 1
    assert highlights[0].strength == 2
    assert highlights[0].start == pytest.approx(0.0)
    assert highlights[0].end == pytest.approx(9.0)


def test_two_distinct_sources_add_one():
    highlights = aggregate(
        [sig(0.0, 5.0, "gap", 4), sig(5.5, 8.0, "low_density", 3)], track=None
    )
    assert len(highlights) == 1
    assert highlights[0].strength == 5  # 4 + 1


def test_three_distinct_sources_cap_at_five():
    highlights = aggregate(
        [
            sig(0.0, 5.0, "gap", 4),
            sig(5.5, 8.0, "low_density", 3),
            sig(9.0, 12.0, "density_shift", 2),
        ],
        track=None,
    )
    assert len(highlights) == 1
    assert highlights[0].strength == 5  # 4 + 2 = 6 -> 封顶 5


def test_triggers_are_sorted_and_deduped():
    highlights = aggregate(
        [
            sig(0.0, 5.0, "gap", 4, "gap:5.0s"),
            sig(5.5, 8.0, "low_density", 3, "density:0.92"),
            sig(6.0, 8.0, "low_density", 3, "density:0.92"),
        ],
        track=None,
    )
    assert highlights[0].triggers == ["density:0.92", "gap:5.0s"]


def test_gap_dominant_summary_reports_duration():
    highlights = aggregate([sig(10.0, 29.8, "gap", 4, "gap:19.8s")], track=None)
    assert highlights[0].summary == "无台词演出段 19.8s"


def test_anchor_lines_merged_and_sorted():
    highlights = aggregate(
        [
            sig(0.0, 5.0, "gap", 4, anchors=[12, 13]),
            sig(5.5, 8.0, "low_density", 3, anchors=[13, 14]),
        ],
        track=None,
    )
    assert highlights[0].anchor_lines == [12, 13, 14]


def test_density_dominant_summary_uses_slowest_line_text():
    track = DialogueTrack(
        episode=1,
        duration=60.0,
        lines=[dline(7, 10.0, 16.0, "才没有染呢我")],
    )
    highlights = aggregate(
        [sig(10.0, 16.0, "low_density", 3, "density:0.91", anchors=[7])], track=track
    )
    assert highlights[0].summary == "才没有染呢我"


def test_density_summary_truncates_to_thirty_chars():
    long_text = "啊" * 50
    track = DialogueTrack(episode=1, duration=60.0, lines=[dline(7, 10.0, 16.0, long_text)])
    highlights = aggregate(
        [sig(10.0, 16.0, "low_density", 3, anchors=[7])], track=track
    )
    assert highlights[0].summary == "啊" * 30


def test_summary_falls_back_when_anchor_missing():
    track = DialogueTrack(episode=1, duration=60.0, lines=[])
    highlights = aggregate([sig(10.0, 16.0, "low_density", 3, anchors=[7])], track=track)
    assert highlights[0].summary == "低语速片段 6.0s"


def test_highlights_sorted_by_start():
    highlights = aggregate(
        [sig(100.0, 105.0, "gap", 2), sig(0.0, 5.0, "gap", 2)], track=None
    )
    assert highlights[0].start == pytest.approx(0.0)


# --- anchor 行索引（原先 _summary 每个簇都全量扫一遍 track.lines） ---


def test_summary_builds_the_line_index_once_for_all_clusters(monkeypatch):
    """一集 L≈400 行、H≈40 个簇，原实现是 O(H×L) 次比较。

    实测 11 集真实素材：`for ln in track.lines` 一共比较 143752 次，只为找出 617 条
    候选行。索引改成在 aggregate 入口建一次，之后逐簇 O(1) 查表。
    """
    track = DialogueTrack(
        episode=1,
        duration=1000.0,
        lines=[dline(i + 1, i * 2.0, i * 2.0 + 1.5, "啊啊啊") for i in range(200)],
    )
    calls = []
    real = aggregate_module._index_track
    monkeypatch.setattr(
        aggregate_module, "_index_track", lambda t: (calls.append(t), real(t))[1]
    )
    signals = [
        sig(i * 100.0, i * 100.0 + 4.0, "low_density", 3, anchors=[i * 10 + 1])
        for i in range(9)
    ]
    assert len(aggregate(signals, track)) == 9
    assert len(calls) == 1


def test_summary_picks_slowest_among_segments_sharing_one_idx():
    """双轨拆行让同一个 idx 对应多条 line，索引的值必须是列表而不是单个位置。"""
    track = DialogueTrack(
        episode=1,
        duration=60.0,
        lines=[
            dline(7, 10.0, 16.0, "快快快快快快快快快快快快"),  # 2.0 字/秒
            dline(7, 10.0, 16.0, "慢"),  # 0.167 字/秒
        ],
    )
    highlights = aggregate(
        [sig(10.0, 16.0, "low_density", 3, anchors=[7])], track=track
    )
    assert highlights[0].summary == "慢"


def test_summary_breaks_char_rate_ties_by_track_order():
    """语速相同时取 track.lines 里靠前的那条 —— 原实现 min() 的语义就是这样。"""
    track = DialogueTrack(
        episode=1,
        duration=60.0,
        lines=[dline(7, 10.0, 16.0, "先"), dline(8, 20.0, 26.0, "后")],
    )
    highlights = aggregate(
        [sig(10.0, 26.0, "low_density", 3, anchors=[7, 8])], track=track
    )
    assert highlights[0].summary == "先"


def test_summary_skips_zero_duration_and_empty_anchor_lines():
    track = DialogueTrack(
        episode=1,
        duration=60.0,
        lines=[
            dline(7, 10.0, 10.0, "零时长"),  # duration == 0，不算候选
            dline(8, 12.0, 18.0, ""),  # 空文本，不算候选
        ],
    )
    highlights = aggregate(
        [sig(10.0, 18.0, "low_density", 3, anchors=[7, 8])], track=track
    )
    assert highlights[0].summary == "低语速片段 8.0s"


# --- 区域信号挂载：双指针与朴素全扫等价 ---


def _naive_host(spans, probe, sep):
    """改造前的写法：对 spans 全量线性扫，取第一个相邻的。"""
    return next(
        (i for i, span in enumerate(spans) if aggregate_module._adjacent(span, probe, sep)),
        None,
    )


def test_regional_attachment_matches_the_naive_scan_on_random_inputs():
    """双指针必须与「全量扫、取下标最小的相邻簇」逐例相同。

    spans 由 group_adjacent 产生，因此按 start 升序且互不相交（分组间隔严格
    > max_gap）；regional 也按 start 升序。这两条单调性是双指针成立的全部依据，
    这里用随机输入把它钉住。
    """
    rng = random.Random(20260913)
    for _ in range(500):
        sep = rng.choice([0.0, 0.5, 2.0, 30.0])
        # 精确信号：随机撒点，交给 group_adjacent 去分组，保证 spans 的形状是真实的。
        precise = [
            sig(s, s + rng.uniform(0.5, 20.0), "gap", 2)
            for s in sorted(rng.uniform(0.0, 600.0) for _ in range(rng.randint(0, 12)))
        ]
        regional = [
            sig(b * 30.0, b * 30.0 + 30.0, "density_shift", 2)
            for b in sorted(rng.sample(range(20), rng.randint(0, 8)))
        ]
        groups = aggregate_module._cluster_precise(precise, sep)
        spans = [aggregate_module._span_of(g) for g in groups]
        expected = [
            _naive_host(spans, (s.start, s.end), sep)
            for s in sorted(regional, key=lambda s: (s.start, s.end))
        ]
        clusters = aggregate_module._cluster([*precise, *regional], sep)
        actual = []
        for signal in sorted(regional, key=lambda s: (s.start, s.end)):
            hosts = [
                i
                for i, c in enumerate(clusters[: len(groups)])
                if signal in c.signals
            ]
            actual.append(hosts[0] if hosts else None)
        assert actual == expected, (sep, spans, [(s.start, s.end) for s in regional])


def test_regional_signal_entirely_before_a_precise_cluster_still_attaches():
    """`_adjacent` 是双向的：30s 桶整个落在精确簇**之前**也要挂上去。

    单向判据（只问「下一条离已见右界够近吗」）会漏掉这种情况，于是 density_shift
    自成一簇，凭空多出一个「只有节奏换挡、没有任何精确证据」的 highlight。
    双指针不能把这条语义弄丢。
    """
    signals = [sig(31.0, 36.0, "gap", 4), sig(0.0, 30.0, "density_shift", 2)]
    highlights = aggregate(signals, track=None)
    assert len(highlights) == 1
    assert highlights[0].start == pytest.approx(31.0)
    assert highlights[0].strength == 5  # 4 + 1 个额外来源


def test_regional_signal_out_of_reach_forms_its_own_cluster():
    signals = [sig(500.0, 505.0, "gap", 4), sig(0.0, 30.0, "density_shift", 2)]
    highlights = aggregate(signals, track=None)
    assert [h.strength for h in highlights] == [2, 4]
    assert highlights[0].start == pytest.approx(0.0)
    assert highlights[0].end == pytest.approx(30.0)


def test_cluster_span_is_computed_once_per_cluster(monkeypatch):
    """同一批 min(start)/max(end) 原先在 _cluster、aggregate、_summary 里各算一次。"""
    calls = []
    real = aggregate_module._span_of
    monkeypatch.setattr(
        aggregate_module, "_span_of", lambda g: (calls.append(g), real(g))[1]
    )
    signals = [sig(i * 100.0, i * 100.0 + 5.0, "gap", 2) for i in range(6)]
    assert len(aggregate(signals, track=None)) == 6
    assert len(calls) == 6


def test_summary_duration_always_matches_the_highlight_time_window():
    """summary 里的秒数必须等于 highlight 自己的 end-start。

    `aggregate` 里那句注释原先写「强度、trigger、anchor 都算整簇；只有边界只看
    bounds」，读起来像是 summary 也该拿整簇。但传下去的一直是 bounds，而这才是对的：
    实测 11 集 338 个簇里有 35 个两种传参结果不同，全部形如
    `无台词演出段 5.3s`（bounds）vs `无台词演出段 30.0s`（整簇）—— 后者会让 summary
    自称 30 秒而 highlight 自己的时间窗只有 5.3 秒，直接把自相矛盾的素材喂给 LLM。
    """
    signals = [sig(31.0, 36.3, "gap", 4, "gap:5.3s"), sig(0.0, 30.0, "density_shift", 2)]
    highlight = aggregate(signals, track=None)[0]
    assert highlight.summary == f"无台词演出段 {highlight.end - highlight.start:.1f}s"
    assert highlight.summary == "无台词演出段 5.3s"


# --- build_report ---


def test_build_report_wires_everything(golden_track):
    report = build_report(golden_track)
    assert report.episode == golden_track.episode
    assert report.median_char_rate > 0
    # Task 10 实测：本集有 26 个 ≥3s 的无字幕间隙（这是候选池，不是高光集合）。
    # 真正的回归护栏是 test_gaps.py 的宽松区间测试，这里只验 build_report 的接线。
    assert len(report.silent_gaps) == 26
    assert report.highlights
    assert all(1 <= h.strength <= 5 for h in report.highlights)
    assert report.highlights == sorted(report.highlights, key=lambda h: h.start)


@pytest.mark.parametrize(
    ("start", "end", "label"),
    [
        (134.6, 153.5, "学院全景"),
        (737.0, 745.4, "天王寺登场"),
        (1233.6, 1247.0, "女仆凝视"),
        (1269.7, 1273.4, "递药丸反应"),
        (1317.1, 1322.7, "震惊定格"),
        (1328.4, 1348.2, "定格收尾"),
    ],
)
def test_build_report_golden_covers_known_visual_peaks(golden_track, start, end, label):
    """召回断言：人工挑出的每个视觉高光都要被某个 highlight 覆盖（重叠 ≥1s）。

    这里只断召回，不断强度。6 个高光对应的间隙时长横跨 3.7-19.8 秒，
    必然被 gaps.py 的三档强度表打散到 2/3/5 三档，任何统一的强度下限都兜不住。
    收敛与优先级排序是 LLM 在 script 阶段的职责。
    """
    report = build_report(golden_track)
    covered = [
        h for h in report.highlights if min(h.end, end) - max(h.start, start) >= 1.0
    ]
    assert covered, f"{label} {start}-{end} 没有被任何 highlight 覆盖"


def test_build_report_golden_long_gaps_reach_top_strength(golden_track):
    """两个 ≥15s 的无字幕间隙必须顶到强度 5（它们各自还叠加了别路信号）。"""
    report = build_report(golden_track)
    top = [h for h in report.highlights if h.strength == 5]
    assert len(top) == 2, [(h.start, h.end, h.strength, h.triggers) for h in top]
    for moment in (140.0, 1330.0):
        assert any(h.start <= moment <= h.end for h in top), moment
