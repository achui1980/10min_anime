import pytest

from tenmin.models import AudioDirection, Beat, Hold, Script
from tenmin.script.budget import (
    SPEECH_RATE_CPS,
    apply_estimates,
    beat_seconds,
    budget_deviation,
    narration_chars,
    needs_rewrite,
    rewrite_instruction,
    script_seconds,
)


def beat(bid, chars, holds=(), label=None, role="act"):
    return Beat(
        id=bid,
        label=label or f"阶段：{bid}",
        role=role,
        narration="啊" * chars,
        audio=AudioDirection(holds=[Hold(at=h[0], duration=h[1], quote="金句") for h in holds]),
    )


def script(beats, target=240.0):
    return Script(show="测试番", episodes=[2], target_seconds=target, beats=beats)


def test_speech_rate_constant():
    assert SPEECH_RATE_CPS == pytest.approx(4.5)


def test_narration_chars_ignores_whitespace():
    assert narration_chars("你好 世界\n再见") == 6


def test_beat_seconds_without_holds():
    assert beat_seconds(beat("b1", 45)) == pytest.approx(10.0)


def test_beat_seconds_includes_holds():
    b = beat("b1", 45, holds=[(2.0, 3.0), (8.0, 2.5)])
    assert beat_seconds(b) == pytest.approx(15.5)


def test_beat_seconds_empty_narration_is_hold_only():
    b = beat("b1", 0, holds=[(0.0, 4.0)])
    assert beat_seconds(b) == pytest.approx(4.0)


def test_script_seconds_sums_beats():
    s = script([beat("b1", 45), beat("b2", 90, holds=[(1.0, 2.0)])])
    assert script_seconds(s) == pytest.approx(10.0 + 20.0 + 2.0)


def test_budget_deviation_over_target():
    s = script([beat("b1", 45 * 30)], target=240.0)  # 1350 字 = 300s
    assert budget_deviation(s) == pytest.approx(0.25)


def test_budget_deviation_under_target_is_negative():
    s = script([beat("b1", 45 * 20)], target=240.0)  # 900 字 = 200s
    assert budget_deviation(s) == pytest.approx(-1 / 6)


def test_budget_deviation_zero_target_is_zero():
    s = script([beat("b1", 45)], target=0.0)
    assert budget_deviation(s) == 0.0


def test_needs_rewrite_within_tolerance():
    s = script([beat("b1", 45 * 22)], target=240.0)  # 220s，偏差 -8.3%
    assert needs_rewrite(s) is False


def test_needs_rewrite_beyond_tolerance():
    s = script([beat("b1", 45 * 30)], target=240.0)  # 300s，偏差 +25%
    assert needs_rewrite(s) is True


def test_apply_estimates_overwrites_llm_numbers():
    s = script([beat("b1", 45), beat("b2", 90)])
    s.beats[0].est_seconds = 999.0
    s.est_total_seconds = 999.0
    apply_estimates(s)
    assert s.beats[0].est_seconds == pytest.approx(10.0)
    assert s.beats[1].est_seconds == pytest.approx(20.0)
    assert s.est_total_seconds == pytest.approx(30.0)


def test_apply_estimates_returns_same_object():
    s = script([beat("b1", 45)])
    assert apply_estimates(s) is s


def test_rewrite_instruction_says_trim_when_too_long():
    s = script([beat("b1", 45 * 30, label="阶段一：太长了")], target=240.0)
    text = rewrite_instruction(s)
    assert "精简" in text
    assert "阶段一：太长了" in text
    assert "1350" in text  # 当前字数
    assert "240" in text  # 目标秒数


def test_rewrite_instruction_says_expand_when_too_short():
    s = script([beat("b1", 45 * 10, label="阶段一：太短了")], target=240.0)
    text = rewrite_instruction(s)
    assert "扩写" in text


def test_rewrite_instruction_lists_every_beat():
    s = script([beat("b1", 45 * 30, label="节点甲"), beat("b2", 45 * 5, label="节点乙")])
    text = rewrite_instruction(s)
    assert "节点甲" in text
    assert "节点乙" in text
