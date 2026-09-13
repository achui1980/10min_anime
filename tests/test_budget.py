import pytest
from pydantic import ValidationError

from tenmin.models import AudioDirection, Beat, Hold, Script
from tenmin.script.budget import (
    SPEECH_RATE_CPS,
    apply_estimates,
    beat_seconds,
    budget_chars,
    budget_deviation,
    narration_chars,
    narration_chars_for_seconds,
    narration_seconds,
    narration_span_seconds,
    needs_rewrite,
    rewrite_instruction,
    script_seconds,
    speed_factor,
    total_estimate,
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


def test_budget_deviation_rejects_a_non_positive_target():
    """原来 target_seconds <= 0 直接返回 0.0，也就是「达标」——一个配错的目标时长
    被静默当成完美达标，预算这一整套机制悄悄失效。"""
    s = script([beat("b1", 45)])
    s.target_seconds = 0.0  # 构造期已被 gt=0 拦住，只能靠赋值绕进来
    with pytest.raises(ValueError, match="必须大于 0"):
        budget_deviation(s)


def test_script_model_rejects_a_non_positive_target_at_construction():
    with pytest.raises(ValidationError):
        script([beat("b1", 45)], target=0.0)


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
    updated = apply_estimates(s)
    assert updated.beats[0].est_seconds == pytest.approx(10.0)
    assert updated.beats[1].est_seconds == pytest.approx(20.0)
    assert updated.est_total_seconds == pytest.approx(30.0)


def test_apply_estimates_returns_a_new_object_and_leaves_the_input_alone():
    """原来是就地改并返回同一个对象（老测试把这件事固化成了断言）。single.py 的预算
    重写轮要同时持有两个版本才能择优，所以这里必须返回新对象。"""
    s = script([beat("b1", 45)])
    before = s.model_dump_json()
    updated = apply_estimates(s)
    assert updated is not s
    assert updated.est_total_seconds == pytest.approx(10.0)
    assert s.model_dump_json() == before


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


# --- D2：语速跟随 render.rate ---


def test_speed_factor_parses_the_edge_tts_rate_string():
    assert speed_factor("+0%") == pytest.approx(1.0)
    assert speed_factor("+20%") == pytest.approx(1.2)
    assert speed_factor("-25%") == pytest.approx(0.75)


def test_speed_factor_falls_back_to_one_on_garbage():
    assert speed_factor("很快") == pytest.approx(1.0)


def test_narration_seconds_follows_the_rate():
    """render.rate 改成 +20% 之后，同一段字数的估算时长必须跟着变短，
    否则预算估算与真实 TTS 时长直接脱钩。"""
    assert narration_seconds("啊" * 45) == pytest.approx(10.0)
    assert narration_seconds("啊" * 45, rate="+20%") == pytest.approx(10.0 / 1.2)


def test_beat_seconds_follows_the_rate_but_never_scales_holds():
    """留白是真静音，跟语速无关，不能被 speed_factor 缩放。"""
    b = beat("b1", 45, holds=[(2.0, 3.0)])
    assert beat_seconds(b, rate="+20%") == pytest.approx(10.0 / 1.2 + 3.0)


def test_script_seconds_follows_the_rate():
    s = script([beat("b1", 45), beat("b2", 45)])
    assert script_seconds(s, rate="+20%") == pytest.approx(2 * 10.0 / 1.2)


def test_tts_duration_band_and_budget_share_one_speed_factor():
    """render/tts.py 原来有一份自己的 _speed_factor，两份实现随时会分叉。"""
    from tenmin.render import tts

    assert tts.speed_factor is speed_factor


# --- D1：budget_chars 必须扣掉留白占走的秒数 ---


def test_budget_chars_deducts_the_hold_seconds():
    """原来是 int(target * 4.5)，而紧跟着的文案写「（含留白已占用的时间）」——
    数字与说明自相矛盾。实测 13 份真实 script.json 的全片留白 10.5–21.0 秒，
    折算 48–95 字：模型照旧公式写必然系统性超长，又触发下一轮重写。"""
    s = script([beat("b1", 45, holds=[(2.0, 3.0)]), beat("b2", 45, holds=[(4.0, 2.0)])])
    assert budget_chars(s) == int((240.0 - 5.0) * 4.5)


def test_budget_chars_without_holds_equals_the_old_formula():
    s = script([beat("b1", 45)])
    assert budget_chars(s) == int(240.0 * 4.5)


def test_budget_chars_follows_the_rate():
    s = script([beat("b1", 45)])
    assert budget_chars(s, rate="+20%") == int(240.0 * 4.5 * 1.2)


def test_budget_chars_clamps_to_zero_when_holds_eat_the_whole_budget():
    s = script([beat("b1", 45, holds=[(0.0, 15.0)] * 20)], target=240.0)
    assert budget_chars(s) == 0


def test_rewrite_instruction_says_the_number_is_net_of_holds():
    s = script([beat("b1", 45 * 30, holds=[(2.0, 3.0)])], target=240.0)
    text = rewrite_instruction(s)
    assert "已经扣掉" in text
    assert str(int((240.0 - 3.0) * 4.5)) in text


# --- D6：估算的单一入口 ---


def test_total_estimate_prefers_the_stored_number():
    s = script([beat("b1", 45)])
    s.est_total_seconds = 77.0
    assert total_estimate(s) == pytest.approx(77.0)


def test_total_estimate_recomputes_when_missing():
    s = script([beat("b1", 45)])
    assert total_estimate(s) == pytest.approx(10.0)


def test_budget_deviation_reads_the_stored_estimate():
    """原来 budget_deviation 每次重新遍历 script_seconds，而 apply_estimates 刚把
    同一个结果写进 est_total_seconds —— 双源迟早不一致。"""
    s = script([beat("b1", 45)], target=100.0)
    s.est_total_seconds = 120.0
    assert budget_deviation(s) == pytest.approx(0.2)


# --- D8：返工文案搬进 prompts/rewrite.md ---


def test_rewrite_instruction_comes_from_the_prompt_file():
    from tenmin.script.prompt import load_prompt

    assert "{{beat_rows}}" in load_prompt("rewrite.md")


def test_rewrite_instruction_says_trim_when_deviation_is_exactly_zero():
    """原来 `deviation > 0 → 精简`，恰好为 0 时会说「扩写 0 秒」。"""
    s = script([beat("b1", 45 * 24)], target=240.0)  # 1080 字 = 240s，偏差 0
    assert budget_deviation(s) == pytest.approx(0.0)
    assert "扩写" not in rewrite_instruction(s)


def test_rewrite_instruction_admits_when_expanding_is_impossible():
    """「要扩写、但旁白净额度已经用完」是个做不到的要求，得说清该动留白。

    重算口径下这种状态算术上不可能（deviation < 0 ⟺ 当前字数 < budget_chars）；
    它只在 total_estimate 读那份被人手改过的 est_total_seconds 时才到得到。
    """
    s = script([beat("b1", 45 * 20, holds=[(0.0, 15.0)] * 15)], target=240.0)
    s.est_total_seconds = 100.0  # 人手改成与重算不一致
    text = rewrite_instruction(s)
    assert "扩写" in text
    assert "不能" in text and "减少或缩短留白" in text


# --- 秒→字的反向换算只有一份实现（N2）--------------------------------------


@pytest.mark.parametrize("rate", ["+0%", "+20%", "-30%"])
@pytest.mark.parametrize("seconds", [0.0, 1.0, 15.0, 225.0, 240.0])
def test_narration_chars_for_seconds_is_the_inverse_of_narration_seconds(seconds, rate):
    """两个方向必须是同一条换算线上的两半。

    正向（字→秒）早就收敛到 narration_seconds，反向（秒→字）却有过三份内联拷贝
    （script/single.py 的首轮字数额度、budget_chars、rewrite_instruction 的
    delta_chars），全都手写 `* SPEECH_RATE_CPS * speed_factor(rate)`。
    """
    chars = narration_chars_for_seconds(seconds, rate=rate)
    assert chars == int(seconds * SPEECH_RATE_CPS * speed_factor(rate))
    # 换回去误差不超过一个字（int 向零截断）
    assert narration_seconds("字" * chars, rate=rate) <= seconds + 1e-9


def test_budget_chars_goes_through_the_single_entry_point():
    s = script([beat("b1", 100)], target=240.0)
    assert budget_chars(s) == narration_chars_for_seconds(240.0)


def test_narration_chars_for_seconds_does_not_clamp_negatives():
    """钳负数是**调用点**的语义（budget_chars 要钳），不该藏进换算里。

    藏进来会静默改掉 script/single.py 在 target_seconds < HOLD_RESERVE_SECONDS 时的行为。
    """
    assert narration_chars_for_seconds(-10.0) < 0


# --- beat_seconds 与 chunks 的跨度只有一份实现（N2）--------------------------


@pytest.mark.parametrize("rate", ["+0%", "+20%", "-20%"])
def test_beat_seconds_is_narration_span_seconds(rate):
    b = beat("b1", 60, holds=[(1.0, 2.0), (5.0, 3.0)])
    assert beat_seconds(b, rate=rate) == narration_span_seconds(
        b.narration, b.audio.holds, rate=rate
    )


@pytest.mark.parametrize("rate", ["+0%", "+20%", "-20%"])
def test_assign_holds_span_is_the_same_number_as_beat_seconds(rate):
    """render/chunks.py 的 assign_holds 原来有一份等价的第三实现
    （`offsets[-1] + sum(hold.duration ...)`），注释自己声明「同口径」。

    那正是 M2 那个「validate 不传 rate」的不一致能溜进来的结构原因。判据做成
    「恰好在 beat_seconds 上不报、多一点就报」——也就是两边用的是**同一个数**。
    """
    from tenmin.render.chunks import assign_holds, split_sentences

    b = beat("b1", 60, holds=[(1.0, 2.0)])
    span = beat_seconds(b, rate=rate)
    sentences = split_sentences(b.narration)

    def warns(at: float) -> bool:
        holds = [Hold(at=at, duration=2.0, quote="金句")]
        out: list[str] = []
        assign_holds(sentences, holds, rate=rate, warnings=out)
        return any("超出本节点跨度" in w for w in out)

    # b 自带的 hold 是 2.0 秒，warns() 里那个也是 2.0 秒，所以跨度相同。
    assert warns(span) is False
    assert warns(span + 0.5) is True
