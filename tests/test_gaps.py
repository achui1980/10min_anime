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


# --- 被 OP/ED 切开的碎片只保留紧邻自己的 anchor ---


def test_unsplit_gap_keeps_both_neighbouring_anchors():
    lines = [dline(7, 10.0, 12.0), dline(9, 30.0, 32.0)]
    (gap,) = find_silent_gaps(track(lines, duration=32.0))
    assert gap.anchor_lines == [7, 9]


def test_trailing_gap_has_only_the_left_anchor():
    lines = [dline(7, 10.0, 12.0)]
    (gap,) = find_silent_gaps(track(lines, duration=40.0))
    assert gap.anchor_lines == [7]


def test_split_gap_fragments_do_not_share_each_others_anchors():
    """抄 saijo/E08 的真实形状：一条 66.6s 的间隙被 ED 切成两片。

    原实现给**两片**都挂上 `[before, after]`。于是左片（1327.8-1348.2）带着一个
    位于 1394.4s 的 anchor —— 隔着整段 ED、离自己右界 46 秒。validate 的
    「clip 时间窗必须装得下自己的 anchor_lines」会收到自相矛盾的输入，而喂给 LLM 的
    高能点也在说「这段无台词画面对应的是那句台词」，其实那句在片尾之后。

    正确的判据是几何的：`before` 结束于 gap.start，只有起点没被切掉的碎片才紧邻它；
    `after` 起始于 gap.end，只有终点没被切掉的碎片才紧邻它。
    """
    lines = [dline(380, 1320.0, 1327.825), dline(384, 1394.393, 1400.0)]
    gaps = find_silent_gaps(
        track(lines, duration=1400.0, ed=(1348.221, 1383.672))
    )
    assert [(round(g.start, 3), round(g.end, 3), g.anchor_lines) for g in gaps] == [
        (1327.825, 1348.221, [380]),
        (1383.672, 1394.393, [384]),
    ]


def test_fragment_touching_neither_neighbour_has_no_anchor():
    """两端都被切掉的碎片一条紧邻的说话行都没有，anchor_lines 就该是空的。

    实测 11 集里有 4 条这样的信号，全是「ED 之后到片长」那一小段（3 秒级）。
    空 anchor 是诚实的（「这是一段纯画面」）；挂一个几十秒外的行是主动误导。
    `density_shift` 的 anchor_lines 本来也恒为空，下游早就吃得下这种情况。
    """
    lines = [dline(7, 10.0, 12.0)]
    gaps = find_silent_gaps(track(lines, duration=100.0, ed=(20.0, 90.0)))
    assert [(round(g.start, 1), round(g.end, 1), g.anchor_lines) for g in gaps] == [
        (12.0, 20.0, [7]),
        (90.0, 100.0, []),
    ]


def test_silent_gaps_are_returned_in_start_order():
    """`find_silent_gaps` 末尾那次 sort 的**真实**理由是「产物顺序确定」。

    它原来的注释说是「下游（aggregate 的双指针挂载）依赖信号按起点有序」，而那个依赖
    不成立：`aggregate._attach_regional` 自己 `sorted(...)`、`_cluster_precise` 靠
    `intervals.group_adjacent` 内部排序。既然理由改成了「产物顺序」，就该由这条测试守着
    —— 否则那次 sort 又变成一个没人守的空操作。
    """
    t = track(
        [
            dline(1, 0.0, 1.0),
            dline(2, 10.0, 11.0),
            dline(3, 25.0, 26.0),
            dline(4, 45.0, 46.0),
        ],
        duration=60.0,
    )
    gaps = find_silent_gaps(t)
    assert len(gaps) >= 3
    assert [g.start for g in gaps] == sorted(g.start for g in gaps)
