"""时长预算。全是纯函数。

「4.5 字/秒」这个语速在本模块**只有一份**（SPEECH_RATE_CPS），render/tts.py 的时长
体检也从这里取，两边不会分叉。真实语速由 RenderConfig.rate 决定，所以所有把字数换算
成秒数的函数都收一个 `rate` 参数（默认 `"+0%"` = 1.0 倍，与历史行为逐点等价）。
"""

from __future__ import annotations

import re

from tenmin.config import DEFAULT_LLM, DEFAULT_RENDER
from tenmin.models import Beat, Script
from tenmin.script.prompt import load_prompt, render_prompt

# 中文旁白的基准语速（字/秒），对应 rate="+0%"。
#
# 这是全项目唯一一份：render/tts.py 的合成结果时长体检（_duration_bounds）、
# render/chunks.py 的句偏移估算、docgen/table.py 的估算列、本模块的预算全都从这里取。
# 标定依据见 render/tts.py 的 _duration_bounds docstring（115 个真实 chunk 的
# `实测时长 / (字数 / 4.5)` 落在 0.809–1.179，中位 0.926）。
SPEECH_RATE_CPS = 4.5

# 抄 config.LLMConfig.budget_tolerance 的默认值，只是它的模块级别名。
# 权威定义在 config，调用方（script/single.py）一律传 cfg.llm.budget_tolerance。
DEFAULT_TOLERANCE = DEFAULT_LLM.budget_tolerance

# 不传 rate 时用的默认语速字符串。等于 RenderConfig.rate 的默认值，所以老调用点
# （没有 cfg 可拿的单测、一次性脚本）的行为跟历史逐点一致。
DEFAULT_RATE = DEFAULT_RENDER.rate

_REWRITE_TEMPLATE = "rewrite.md"  # 同时当 render_prompt 的模板名（只进报错消息）

_WHITESPACE = re.compile(r"\s+")
_RATE_PERCENT = re.compile(r"^([+-]\d+)%$")


def speed_factor(rate: str) -> float:
    """把 edge-tts 的 rate 字符串换算成语速倍率。认不出就当 1.0。

    edge-tts 自己会校验 `^[+-]\\d+%$`（data_classes.py 的 validate_string_param），
    所以这里认不出的形态在 Communicate 构造期就已经炸了，兜底只是不让估算自己抛。

    原来这个函数住在 render/tts.py（叫 `_speed_factor`），只有时长体检在用；预算估算
    则完全无视 rate，用户把 rate 改成 "+20%" 之后 budget 与真实 TTS 时长直接脱钩。
    挪到这里是为了让两个消费者共用同一份实现（tts.py 反过来 import 它）。
    """
    match = _RATE_PERCENT.match(rate.strip())
    if match is None:
        return 1.0
    return max(0.1, 1.0 + int(match.group(1)) / 100.0)


def narration_chars(text: str) -> int:
    """旁白的计费字数：所有非空白字符，一律按 1 个字计。

    **中文标点按全额计费是刻意的，不是遗漏。** 直觉上「，。！？」几乎不占发音时长，
    应该打折；实测（work/saijo 10 集共 115 个真实 chunk，9135 个 CJK 字 + 1198 个标点
    + 5 个拉丁字符）说的是反话——`实测时长 / (计费字数 / 4.5)` 这个比值的离散度：

        全额计费（现行）      CV 0.0708   跨度 0.370   ← 最紧
        标点 ×0.5            CV 0.0789   跨度 0.461
        标点 ×0.3            CV 0.0840   跨度 0.558
        标点 ×0.2            CV 0.0871   跨度 0.613
        标点 ×0.0            CV 0.0947   跨度 0.744

    越给标点打折，分布越**散**，而且是单调的。原因很直白：Edge TTS 在句读处会插入真实
    的停顿，中文标点确实消耗发音时长。拉丁字母按词估算也测了，语料里只有 5 个拉丁字符
    （占 0.05%），对分布无任何影响。

    所以「按类别加权」这个改动**没有做**——数据不支持。已知的系统性偏差是别的：这个
    比值的中位数是 0.926 而不是 1.0，也就是全额计费整体高估约 8%；这一份高估被
    _duration_bounds 的 ±50% 带宽和 budget 的 ±12% 容差都吸收掉了，而且它是**均匀**的
    高估（不随标点密度漂移），不会像加权那样把某些 chunk 推出体检 band。
    """
    return len(_WHITESPACE.sub("", text))


def narration_seconds(text: str, *, rate: str = DEFAULT_RATE) -> float:
    """一段旁白文本的估算时长（秒）。字数换算成秒的**唯一**入口。"""
    return narration_chars(text) / SPEECH_RATE_CPS / speed_factor(rate)


def hold_seconds(beat: Beat) -> float:
    """本节点全部留白的总秒数。真静音，跟语速无关，永远不被 speed_factor 缩放。"""
    return sum(hold.duration for hold in beat.audio.holds)


def beat_seconds(beat: Beat, *, rate: str = DEFAULT_RATE) -> float:
    return narration_seconds(beat.narration, rate=rate) + hold_seconds(beat)


def script_seconds(script: Script, *, rate: str = DEFAULT_RATE) -> float:
    return sum(beat_seconds(beat, rate=rate) for beat in script.beats)


def script_hold_seconds(script: Script) -> float:
    """全片留白总秒数。实测 13 份真实 script.json 是 10.5–21.0 秒（中位 17）。"""
    return sum(hold_seconds(beat) for beat in script.beats)


def apply_estimates(script: Script, *, rate: str = DEFAULT_RATE) -> Script:
    """用本模块重算的数字覆盖 LLM 填的 est_*，返回**新** Script。

    原来是就地改并返回同一个对象。跟 script/validate.py 的 repair_script 一起改成
    返回新对象：调用方（single.py 的预算重写轮）需要同时持有两个版本才能择优。
    """
    updated = script.model_copy(deep=True)
    for beat in updated.beats:
        beat.est_seconds = beat_seconds(beat, rate=rate)
    updated.est_total_seconds = sum(beat.est_seconds for beat in updated.beats)
    return updated


def total_estimate(script: Script, *, rate: str = DEFAULT_RATE) -> float:
    """全片估算时长的**唯一**读取入口：优先用 apply_estimates 存好的数，缺了才重算。

    原来 budget_deviation 每次都重新遍历 script_seconds，而 apply_estimates 刚把同一个
    结果写进 est_total_seconds —— 两个来源迟早不一致（人手改了 est_total_seconds、
    或者用不同的 rate 算过一遍）。口径抄 docgen/table.py 的 _total_estimate。
    """
    if script.est_total_seconds > 0:
        return script.est_total_seconds
    return script_seconds(script, rate=rate)


def budget_deviation(script: Script, *, rate: str = DEFAULT_RATE) -> float:
    """(实际 - 目标) / 目标。正数偏长，负数偏短。

    target_seconds <= 0 时原来返回 0.0，也就是「达标」—— 一个配错的目标时长会被静默
    当成完美达标，预算这一整套机制悄悄失效。现在响亮报错（ValueError 在
    cli.PIPELINE_ERRORS 里，用户看到一行红字）。models.Script.target_seconds 已加
    gt=0，所以正常路径进不来这里；剩下的入口是「构造完再赋值」（pydantic 默认不校验
    赋值）与旧产物，这条就是给它们兜底的。
    """
    if script.target_seconds <= 0:
        raise ValueError(
            f"target_seconds 是 {script.target_seconds}，必须大于 0："
            f"时长预算全靠它做分母。请检查 project.yaml 的 target_seconds "
            f"与 script.json 里那一份"
        )
    return (total_estimate(script, rate=rate) - script.target_seconds) / script.target_seconds


def needs_rewrite(
    script: Script,
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    rate: str = DEFAULT_RATE,
) -> bool:
    return abs(budget_deviation(script, rate=rate)) > tolerance


def budget_chars(script: Script, *, rate: str = DEFAULT_RATE) -> int:
    """全片旁白的目标字数：**扣掉留白占走的秒数**之后再换算。

    原来是 `int(target * SPEECH_RATE_CPS)`，而紧跟着的文案写的是「（含留白已占用的
    时间）」—— 数字与说明自相矛盾。实测 13 份真实 script.json 的全片留白是 10.5–21.0
    秒，按 4.5 字/秒折算就是 48–95 字（中位 77 字）：模型照旧公式写必然系统性超长，
    又触发下一轮重写。旧公式对 target=240 恒为 1080 字，新公式给出 985–1032 字。

    留白吃满整个预算时会算出 <= 0，钳到 0：那种剧本已经没有任何旁白空间了，
    rewrite_instruction 会另给一条「留白已占满预算」的兜底文案。
    """
    speakable = script.target_seconds - script_hold_seconds(script)
    return max(0, int(speakable * SPEECH_RATE_CPS * speed_factor(rate)))


def rewrite_instruction(
    script: Script,
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    rate: str = DEFAULT_RATE,
) -> str:
    """时长返工要求。文案本体在 prompts/rewrite.md，这里只负责算数字。"""
    actual = total_estimate(script, rate=rate)
    target = script.target_seconds
    deviation = budget_deviation(script, rate=rate)
    # `>= 0 → 精简`：原来写的是 `> 0`，偏差恰好为 0 时会说「扩写 0 秒」。当前只在
    # needs_rewrite 为真时调用所以碰不到，但这是隐患，不是「不会发生」。
    verb = "精简" if deviation >= 0 else "扩写"
    delta = abs(actual - target)
    chars_budget = budget_chars(script, rate=rate)
    holds = script_hold_seconds(script)

    rows = "\n".join(
        f"- {beat.label}：{narration_chars(beat.narration)} 字，"
        f"留白 {hold_seconds(beat):.1f} 秒，估算 {beat_seconds(beat, rate=rate):.1f} 秒"
        for beat in script.beats
    )
    # 「让模型扩写，但旁白的净额度已经用完」这种自相矛盾的要求要说清楚该往哪儿动 ——
    # 旁白一个字都写不下了，能动的只有留白。
    #
    # 算术上这在**重算**口径下不可能出现：deviation < 0 ⟺ 旁白秒数 < target - 留白
    # ⟺ 当前字数 < budget_chars，所以要扩写就一定还有额度。它只在 total_estimate
    # 走「优先读存的 est_total_seconds」那条路、而那个数被人手改成与重算不一致时才
    # 到得了（D6 保留这个优先级就是为了尊重人工编辑）。既然到得了就得有兜底，
    # 不能让模型收到一条它物理上完不成的指令。
    current_chars = sum(narration_chars(beat.narration) for beat in script.beats)
    impossible = (
        f"注意：全片留白已经占掉 {holds:.1f} 秒，目标 {target:.0f} 秒里留给旁白的净额度"
        f"只剩 {chars_budget} 字，而当前旁白已经有 {current_chars} 字。"
        f"这一版**不能**靠加旁白来补时长，请改为减少或缩短留白。"
        if verb == "扩写" and chars_budget <= current_chars
        else ""
    )
    return render_prompt(
        load_prompt(_REWRITE_TEMPLATE),
        _REWRITE_TEMPLATE,
        actual=f"{actual:.1f}",
        target=f"{target:.0f}",
        deviation=f"{deviation:+.1%}",
        tolerance=f"{tolerance:.0%}",
        verb=verb,
        delta=f"{delta:.0f}",
        delta_chars=int(delta * SPEECH_RATE_CPS * speed_factor(rate)),
        budget_chars=chars_budget,
        hold_seconds=f"{holds:.1f}",
        beat_rows=rows,
        impossible_note=impossible,
    )
