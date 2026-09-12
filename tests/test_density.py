import pytest

from tenmin.models import DialogueLine, DialogueTrack
from tenmin.signals import density as density_module
from tenmin.signals.density import (
    LOW_DENSITY_RATIO,
    char_rate,
    find_density_shifts,
    find_density_signals,
    find_low_density,
    median_char_rate,
)


def dline(idx, start, end, text, kind="dialogue"):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind=kind)


def track(lines, duration):
    return DialogueTrack(episode=1, duration=duration, lines=lines)


def test_char_rate_ignores_whitespace():
    assert char_rate(dline(1, 0.0, 2.0, "你 好 呀 吗")) == pytest.approx(2.0)


def test_char_rate_zero_duration_is_zero():
    assert char_rate(dline(1, 5.0, 5.0, "你好")) == 0.0


def test_char_rate_matches_golden_269_arithmetic():
    assert char_rate(dline(269, 0.0, 6.589, "才没有染呢我")) == pytest.approx(0.911, abs=0.01)


def test_median_char_rate_uses_speech_lines_only():
    lines = [
        dline(1, 0.0, 1.0, "一二三四"),  # 4.0
        dline(2, 2.0, 3.0, "一二三四五六"),  # 6.0
        dline(3, 4.0, 5.0, "制作委员会", kind="credits"),
    ]
    assert median_char_rate(track(lines, 5.0)) == pytest.approx(5.0)


def test_median_char_rate_empty_track_is_zero():
    assert median_char_rate(track([], 10.0)) == 0.0


def test_find_low_density_flags_slow_line():
    lines = [
        dline(1, 0.0, 1.0, "一二三四五"),  # 5.0
        dline(2, 2.0, 3.0, "一二三四五"),  # 5.0
        dline(3, 4.0, 5.0, "一二三四五"),  # 5.0
        dline(4, 6.0, 9.0, "啊"),  # 0.333 < 0.4*5.0 = 2.0，时长 3s
    ]
    signals = find_low_density(track(lines, 9.0))
    assert len(signals) == 1
    assert signals[0].source == "low_density"
    assert signals[0].strength == 3
    assert signals[0].anchor_lines == [4]
    assert signals[0].start == pytest.approx(6.0)
    assert signals[0].end == pytest.approx(9.0)
    assert signals[0].detail.startswith("density:")


def test_find_low_density_requires_min_duration():
    lines = [
        dline(1, 0.0, 1.0, "一二三四五"),
        dline(2, 2.0, 3.0, "一二三四五"),
        dline(3, 4.0, 5.5, "啊"),  # 时长 1.5s < 2.0，不发信号
    ]
    assert find_low_density(track(lines, 5.5)) == []


def test_low_density_ratio_constant():
    assert LOW_DENSITY_RATIO == pytest.approx(0.4)


def _bucket_track(bucket_chars: list[int]) -> DialogueTrack:
    """每个 30s 桶塞一条行，字数由 bucket_chars 指定。"""
    lines = []
    for i, count in enumerate(bucket_chars):
        lines.append(dline(i + 1, i * 30.0 + 1.0, i * 30.0 + 3.0, "啊" * count))
    return track(lines, duration=len(bucket_chars) * 30.0)


def test_find_density_shifts_detects_jump():
    signals = find_density_shifts(_bucket_track([30, 30, 30, 30, 120, 30, 30, 30]))
    starts = sorted(s.start for s in signals)
    assert starts == pytest.approx([120.0, 150.0])
    assert all(s.source == "density_shift" for s in signals)
    assert all(s.strength == 2 for s in signals)
    assert all(s.detail.startswith("shift:z=") for s in signals)


def test_find_density_shifts_flat_track_has_none():
    assert find_density_shifts(_bucket_track([30, 30, 30, 30, 30])) == []


def test_find_density_shifts_needs_at_least_three_buckets():
    assert find_density_shifts(_bucket_track([30, 90])) == []


def test_find_density_shifts_clamps_last_bucket_to_duration():
    t = _bucket_track([30, 30, 30, 30, 120, 30, 30, 30])
    t.duration = 235.0
    for signal in find_density_shifts(t):
        assert signal.end <= 235.0


def test_find_density_signals_combines_both_rules():
    t = _bucket_track([30, 30, 30, 30, 120, 30, 30, 30])
    sources = {s.source for s in find_density_signals(t)}
    assert "density_shift" in sources


# --- 尾部不完整桶 / stdev 数值稳定性 / OP-ED 屏蔽 ---


def test_find_density_shifts_ignores_incomplete_tail_bucket():
    """duration 不是 window 整数倍时，尾部那个残桶必然造出一条固定的假「节奏骤降」。

    修复前 bucket_count = int(duration // window) + 1，window=30 / duration=241 时
    得到 9 个桶，第 9 个只覆盖 240→241 共 1 秒。而 duration 是「最后一条 cue 的终点」，
    那条 cue 必然起于 240 之前、落进第 8 桶，所以残桶字数恒为 0；桶内字数又没有按桶的
    实际时长归一化，于是 diffs 末项是个大负值，稳定触发一条片尾假信号。
    """
    t = _bucket_track([30] * 8)  # 0-240 八个满桶，完全平坦
    t.duration = 241.0
    assert find_density_shifts(t) == []


def test_find_density_shifts_tail_bucket_does_not_shrink_real_signal():
    """残桶被丢弃，但完整桶里的真信号一条都不能少。"""
    chars = [30, 30, 30, 30, 120, 30, 30, 30]
    t = _bucket_track(chars)
    t.duration = 240.0 + 7.046  # 抄 saijo E02 的 duration % 30
    starts = sorted(s.start for s in find_density_shifts(t))
    assert starts == pytest.approx([120.0, 150.0])


def test_find_density_shifts_all_signals_stay_inside_full_buckets():
    t = _bucket_track([30, 30, 30, 30, 120, 30, 30, 30])
    t.duration = 235.0
    for signal in find_density_shifts(t):
        assert signal.end <= 210.0  # 只剩 7 个完整桶，覆盖 0-210


def test_find_density_shifts_treats_near_zero_stdev_as_flat(monkeypatch):
    """`stdev == 0` 的浮点相等判断漏掉 1e-16，z = diff / 1e-16 会爆成天文数字。

    后果是**每一个**桶边界都变成「节奏突变」。当前 buckets 是整数字数累加，
    pstdev 对全等整数序列给的是精确 0.0，所以这个坑是潜伏的而不是已激活的；
    这里直接把 pstdev 打桩成 1e-16 来锁住 EPS 判据，避免以后桶统计一改成
    浮点（比如按桶实际时长归一化成速率）就立刻踩上去。
    """
    monkeypatch.setattr(density_module.statistics, "pstdev", lambda _values: 1e-16)
    t = _bucket_track([30, 30, 30, 31, 30, 30, 30, 30])
    assert find_density_shifts(t) == []


def test_find_density_shifts_masks_op_range():
    """OP 段落的 credits 行被 spoken_lines 过滤掉，桶字数骤降为 0。

    修复前 OP 进入与离开各产生一条假 density_shift（实测 work/ 下 10 集有 op_range
    的素材里，片头 60-270 秒区间的 shift 有 19 条，其中 13 条是这类边界伪影），
    随后在 aggregate 里给相邻真高光加强度、污染排序。
    """
    lines = [
        dline(1, 1.0, 3.0, "啊" * 60),
        dline(2, 31.0, 33.0, "啊" * 60),
        dline(3, 61.0, 63.0, "啊" * 60),
        # 90-180 是 OP：整段只有 credits 行，spoken_lines 一条都不留。
        dline(4, 95.0, 100.0, "监督 山田太郎", kind="credits"),
        dline(5, 150.0, 155.0, "制作委员会", kind="credits"),
        dline(6, 181.0, 183.0, "啊" * 60),
        dline(7, 211.0, 213.0, "啊" * 60),
        dline(8, 241.0, 243.0, "啊" * 60),
        dline(9, 271.0, 273.0, "啊" * 60),
    ]
    t = DialogueTrack(
        episode=1, duration=300.0, op_range=(90.0, 180.0), lines=lines
    )
    assert find_density_shifts(t) == []


def test_find_density_shifts_no_diff_bridges_a_masked_hole():
    """跨过 OP 空洞的差分同样是伪影：屏蔽区两侧的桶不能互相做差。"""
    lines = [
        dline(1, 1.0, 3.0, "啊" * 20),
        dline(2, 31.0, 33.0, "啊" * 20),
        dline(3, 61.0, 63.0, "啊" * 20),
        dline(4, 181.0, 183.0, "啊" * 200),
        dline(5, 211.0, 213.0, "啊" * 200),
        dline(6, 241.0, 243.0, "啊" * 200),
    ]
    t = DialogueTrack(
        episode=1, duration=270.0, op_range=(90.0, 180.0), lines=lines
    )
    # 桶 0-2（各 20 字）与桶 6-8（各 200 字）内部都完全平坦；
    # 唯一的「突变」是跨 OP 空洞的 20 -> 200，它不该被算出来。
    assert find_density_shifts(t) == []


def test_find_density_shifts_masks_ed_range():
    lines = [
        dline(1, 1.0, 3.0, "啊" * 60),
        dline(2, 31.0, 33.0, "啊" * 60),
        dline(3, 61.0, 63.0, "啊" * 60),
        dline(4, 91.0, 93.0, "啊" * 60),
        dline(5, 125.0, 130.0, "制作委员会", kind="credits"),
    ]
    t = DialogueTrack(
        episode=1, duration=150.0, ed_range=(120.0, 150.0), lines=lines
    )
    assert find_density_shifts(t) == []


def test_find_density_shifts_partially_masked_bucket_is_dropped():
    """跨屏蔽边界的桶整桶丢弃：半个桶被 OP 覆盖，字数天然减半，本身就是伪影。"""
    t = DialogueTrack(
        episode=1,
        duration=300.0,
        op_range=(95.0, 175.0),  # 与桶 [90,120) 和 [150,180) 各重叠一部分
        lines=[dline(i + 1, i * 30.0 + 1.0, i * 30.0 + 3.0, "啊" * 60) for i in range(10)],
    )
    for signal in find_density_shifts(t):
        assert signal.end <= 90.0 or signal.start >= 180.0


# --- 黄金样本 ---


def test_golden_median_char_rate_is_sane(golden_track):
    median = median_char_rate(golden_track)
    assert 2.0 < median < 12.0, median


def test_golden_line_269_is_among_slowest(golden_track):
    """269 行「才没有染呢我」6 字拖 6.589 秒，是本集的低语速标杆之一。

    它不是全集最慢：237 行「这」1 字拖 1.876 秒（0.533 字/秒）更慢，但被
    LOW_DENSITY_MIN_SECONDS 过滤掉；在 duration >= 2.0 的子集里 16 行「什么」
    （0.888 字/秒）也比它慢。所以这里断言「在最慢 3 行内」而不是「最慢」。
    """
    spoken = [
        ln
        for ln in golden_track.lines
        if ln.kind in ("dialogue", "monologue") and ln.text and ln.duration >= 2.0
    ]
    slowest = sorted(spoken, key=char_rate)[:3]
    assert 269 in [ln.idx for ln in slowest], [(ln.idx, ln.text) for ln in slowest]
    line_269 = next(ln for ln in golden_track.lines if ln.idx == 269)
    assert char_rate(line_269) == pytest.approx(0.911, abs=0.02)


def test_golden_line_269_produces_low_density_signal(golden_track):
    signals = find_low_density(golden_track)
    assert any(269 in s.anchor_lines for s in signals)
    assert all(s.strength == 3 for s in signals)


def test_golden_density_shifts_exist_and_are_strength_two(golden_track):
    signals = find_density_shifts(golden_track)
    assert signals
    assert all(s.strength == 2 for s in signals)
    assert all(0.0 <= s.start < s.end <= golden_track.duration for s in signals)
