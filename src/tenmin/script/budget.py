"""时长预算。全是纯函数。"""

from __future__ import annotations

import re

from tenmin.models import Beat, Script

SPEECH_RATE_CPS = 4.5
DEFAULT_TOLERANCE = 0.12

_WHITESPACE = re.compile(r"\s+")


def narration_chars(text: str) -> int:
    return len(_WHITESPACE.sub("", text))


def beat_seconds(beat: Beat) -> float:
    speech = narration_chars(beat.narration) / SPEECH_RATE_CPS
    holds = sum(hold.duration for hold in beat.audio.holds)
    return speech + holds


def script_seconds(script: Script) -> float:
    return sum(beat_seconds(beat) for beat in script.beats)


def apply_estimates(script: Script) -> Script:
    """用本模块重算的数字覆盖 LLM 填的 est_*。原地修改并返回同一对象。"""
    for beat in script.beats:
        beat.est_seconds = beat_seconds(beat)
    script.est_total_seconds = sum(beat.est_seconds for beat in script.beats)
    return script


def budget_deviation(script: Script) -> float:
    """(实际 - 目标) / 目标。正数偏长，负数偏短。"""
    if script.target_seconds <= 0:
        return 0.0
    return (script_seconds(script) - script.target_seconds) / script.target_seconds


def needs_rewrite(script: Script, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    return abs(budget_deviation(script)) > tolerance


def rewrite_instruction(script: Script, tolerance: float = DEFAULT_TOLERANCE) -> str:
    actual = script_seconds(script)
    target = script.target_seconds
    deviation = budget_deviation(script)
    verb = "精简" if deviation > 0 else "扩写"
    delta = abs(actual - target)
    budget_chars = int(target * SPEECH_RATE_CPS)

    lines = [
        f"上一版旁白估算时长 {actual:.1f} 秒，目标 {target:.0f} 秒，"
        f"偏差 {deviation:+.1%}，超出容差 ±{tolerance:.0%}。",
        f"请{verb}约 {delta:.0f} 秒（约 {int(delta * SPEECH_RATE_CPS)} 字）。"
        f"全片旁白总字数应控制在 {budget_chars} 字左右（含留白已占用的时间）。",
        "各节点当前字数：",
    ]
    for beat in script.beats:
        holds = sum(hold.duration for hold in beat.audio.holds)
        lines.append(
            f"- {beat.label}：{narration_chars(beat.narration)} 字，"
            f"留白 {holds:.1f} 秒，估算 {beat_seconds(beat):.1f} 秒"
        )
    lines.append(
        f"只调整字数，不要改动节点划分、clip 时间戳、留白金句。"
        f"优先{verb}信息密度最低的节点。"
    )
    return "\n".join(lines)
