import pytest

from tenmin.models import DialogueLine, DialogueTrack, Signal
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
