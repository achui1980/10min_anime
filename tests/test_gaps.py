import pytest

from tenmin.models import DialogueLine, DialogueTrack
from tenmin.signals.gaps import find_silent_gaps


def track(lines, duration, op=None, ed=None):
    return DialogueTrack(episode=1, duration=duration, op_range=op, ed_range=ed, lines=lines)


def dline(idx, start, end, text="台词", kind="dialogue"):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind=kind)


def test_no_lines_no_gaps():
    assert find_silent_gaps(track([], duration=100.0)) == []


def test_ignores_gap_before_first_line():
    """首行之前是黑场/OP前奏，不算演出高光。"""
    gaps = find_silent_gaps(track([dline(1, 50.0, 52.0)], duration=52.0))
    assert gaps == []


def test_detects_trailing_gap_to_end_of_episode():
    gaps = find_silent_gaps(track([dline(1, 0.0, 2.0)], duration=12.0))
    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(2.0)
    assert gaps[0].end == pytest.approx(12.0)


def test_detects_gap_between_lines():
    gaps = find_silent_gaps(track([dline(1, 0.0, 2.0), dline(2, 12.0, 14.0)], duration=14.0))
    assert len(gaps) == 1
    assert gaps[0].source == "gap"
    assert gaps[0].duration == pytest.approx(10.0)
    assert gaps[0].anchor_lines == [1, 2]
    assert gaps[0].detail == "gap:10.0s"


def test_below_threshold_is_dropped():
    gaps = find_silent_gaps(track([dline(1, 0.0, 2.0), dline(2, 4.5, 6.0)], duration=6.0))
    assert gaps == []


def test_subtracts_op_range_producing_two_subgaps():
    """一个 92s 的原始间隙被 OP 切成两段，只有 >=3s 的那段留下。"""
    lines = [dline(1, 130.0, 134.6), dline(2, 226.0, 230.0)]
    gaps = find_silent_gaps(track(lines, duration=230.0, op=(153.486, 224.681)))
    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(134.6)
    assert gaps[0].end == pytest.approx(153.486)
    assert gaps[0].duration == pytest.approx(18.886, abs=1e-3)
    assert gaps[0].strength == 4


def test_subtracts_ed_range():
    lines = [dline(1, 1320.0, 1328.367), dline(2, 1360.0, 1365.0)]
    gaps = find_silent_gaps(track(lines, duration=1416.622, ed=(1348.180, 1416.622)))
    assert len(gaps) == 1
    assert gaps[0].end == pytest.approx(1348.180)
    assert gaps[0].duration == pytest.approx(19.813, abs=1e-3)


def test_gap_fully_inside_op_is_dropped():
    lines = [dline(1, 150.0, 160.0), dline(2, 200.0, 210.0)]
    gaps = find_silent_gaps(track(lines, duration=210.0, op=(140.0, 205.0)))
    assert gaps == []


def test_overlapping_lines_do_not_create_negative_gaps():
    lines = [dline(1, 0.0, 20.0), dline(2, 5.0, 8.0), dline(3, 30.0, 32.0)]
    gaps = find_silent_gaps(track(lines, duration=32.0))
    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(20.0)
    assert gaps[0].end == pytest.approx(30.0)


def test_credits_and_noise_lines_are_not_speech():
    lines = [
        dline(1, 0.0, 2.0),
        dline(2, 5.0, 6.0, kind="credits"),
        dline(3, 6.5, 7.0, kind="noise"),
        dline(4, 12.0, 14.0),
    ]
    gaps = find_silent_gaps(track(lines, duration=14.0))
    assert len(gaps) == 1
    assert gaps[0].duration == pytest.approx(10.0)


def test_zero_duration_line_does_not_split_a_long_gap():
    """回归：srt_parser 把 end < start 的坏 cue 夹成零时长（end = start）。

    这类行原先会推进游标，把一整段 10s 静默切成 2.5s + 5.0s 两段，两段都低于
    MIN_GAP_SECONDS 里 5.0 这一档以外的阈值时会双双消失 —— 真高光直接丢掉。
    现在零时长行不算「有人说话」，长间隙保持完整。
    """
    lines = [dline(1, 0.0, 2.0), dline(2, 4.5, 4.5), dline(3, 12.0, 14.0)]
    gaps = find_silent_gaps(track(lines, duration=14.0))
    assert len(gaps) == 1
    assert gaps[0].start == pytest.approx(2.0)
    assert gaps[0].end == pytest.approx(12.0)
    assert gaps[0].anchor_lines == [1, 3]


def test_zero_duration_line_no_longer_erases_a_gap_entirely():
    """更狠的一版：坏 cue 落在中间，切出的两段都 <3s，间隙曾经整条消失。"""
    lines = [dline(1, 0.0, 2.0), dline(2, 4.0, 4.0), dline(3, 6.5, 8.0)]
    gaps = find_silent_gaps(track(lines, duration=8.0))
    assert len(gaps) == 1
    assert gaps[0].duration == pytest.approx(4.5)


@pytest.mark.parametrize(
    "gap_len,expected",
    [(3.5, 2), (7.9, 2), (8.0, 3), (14.9, 3), (15.0, 4), (19.8, 4)],
)
def test_strength_thresholds(gap_len, expected):
    lines = [dline(1, 0.0, 2.0), dline(2, 2.0 + gap_len, 2.0 + gap_len + 1.0)]
    gaps = find_silent_gaps(track(lines, duration=2.0 + gap_len + 1.0))
    assert gaps[0].strength == expected


# --- 黄金样本 ---


# spec §6.1 人工看片标定的视觉高光区间。这是「长无字幕间隙 = 演出高光」这条
# 核心假设的验收集：每个区间都必须被某个检出的间隙覆盖（recall）。
# 反过来不成立 —— 检出的间隙远多于视觉高光，收敛是 Task 12 aggregate.py 的事。
#
# 00:18:20→00:18:25（泳装定格）不在列表里：实测该区段是连续对白
# （334 '好宽敞' / 335 '伊月' / 336 '一起泡澡吧'），字幕轨里没有间隙。
# 所以本集 recall 是 6/7，不是 spec 原先写的 100%。
KNOWN_VISUAL_PEAKS = [
    (134.6, 153.5, "学院全景"),
    (737.0, 745.4, "天王寺登场"),
    (1233.6, 1247.0, "女仆凝视"),
    (1269.7, 1273.4, "递药丸反应"),
    (1317.1, 1322.7, "震惊定格"),
    (1328.4, 1348.2, "定格收尾"),
]


def test_golden_gap_count_is_a_candidate_pool(golden_track):
    """≥3s 间隙是候选池，不是高光集合。这里只守一个量级区间，防止算法失控。"""
    gaps = find_silent_gaps(golden_track)
    assert 20 <= len(gaps) <= 32, [
        (round(g.start, 2), round(g.end, 2), round(g.duration, 2)) for g in gaps
    ]


@pytest.mark.parametrize(("start", "end", "label"), KNOWN_VISUAL_PEAKS)
def test_golden_gaps_cover_known_visual_peaks(golden_track, start, end, label):
    """核心假设的 recall 测试。这条红了说明「无字幕间隙 = 演出高光」在本集不成立。"""
    gaps = find_silent_gaps(golden_track)
    covered = [g for g in gaps if min(g.end, end) - max(g.start, start) >= 1.0]
    assert covered, f"{label}（{start}-{end}）没有被任何间隙覆盖"


def test_golden_long_gaps_get_top_strength(golden_track):
    """两个最长间隙（18.9s 学院全景、19.8s 定格收尾）必须是最高强度。"""
    gaps = find_silent_gaps(golden_track)
    long_gaps = [g for g in gaps if g.duration >= 15.0]
    assert len(long_gaps) == 2
    assert all(g.strength == 4 for g in long_gaps)


def test_golden_strength_is_monotone_in_duration(golden_track):
    """强度必须随时长单调不减 —— 这是强度表的结构不变量。"""
    gaps = sorted(find_silent_gaps(golden_track), key=lambda g: g.duration)
    strengths = [g.strength for g in gaps]
    assert strengths == sorted(strengths), [
        (round(g.duration, 2), g.strength) for g in gaps
    ]


def test_golden_longest_gap(golden_track):
    gaps = find_silent_gaps(golden_track)
    longest = max(gaps, key=lambda g: g.duration)
    assert longest.end == pytest.approx(1348.180, abs=0.001)
    assert longest.duration == pytest.approx(19.813, abs=0.05)
    assert longest.strength == 4


def test_golden_gaps_never_overlap_op_or_ed(golden_track):
    op = golden_track.op_range
    ed = golden_track.ed_range
    for gap in find_silent_gaps(golden_track):
        for block in (op, ed):
            if block is None:
                continue
            assert gap.end <= block[0] + 1e-6 or gap.start >= block[1] - 1e-6, gap
