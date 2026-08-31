import pytest

from tenmin.models import DialogueLine, DialogueTrack
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
