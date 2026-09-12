"""单集剧本生成。一次 LLM 调用 + 最多一次重试 + 最多一轮预算重写。"""

from __future__ import annotations

from tenmin.config import ProjectConfig
from tenmin.models import (
    SPEECH_KINDS,
    AudioDirection,
    Beat,
    Clip,
    DialogueTrack,
    LLMScript,
    Script,
    SignalReport,
)
from tenmin.script.budget import apply_estimates, needs_rewrite, rewrite_instruction
from tenmin.script.llm import LLMProvider
from tenmin.script.prompt import load_prompt, render_prompt
from tenmin.script.validate import ScriptValidationError, validate_script
from tenmin.timecode import format_timestamp, readable_seconds

SYSTEM_PROMPT = "你是一名资深番剧解说号写手。严格按要求输出 JSON，不要输出任何解释文字。"


def build_dialogue_block(track: DialogueTrack) -> str:
    rows = []
    for line in track.lines:
        # 刻意不用 intervals.is_spoken：那个谓词还要求 duration > 0，是给区间运算用的。
        # 这里是「给 LLM 看的台词清单」，一条被坏时间码夹成零时长的行，正文照样是
        # 剧情内容，不能从上下文里抹掉。
        if line.kind not in SPEECH_KINDS or not line.text:
            continue
        mark = " ?" if line.suspect else ""
        rows.append(
            f"{line.idx} | {format_timestamp(line.start)} - {format_timestamp(line.end)}"
            f" | {line.speaker or '-'} | {line.kind}{mark} | {line.text}"
        )
    return "\n".join(rows)


def build_highlight_block(report: SignalReport) -> str:
    if not report.highlights:
        return "（本集未检出高能点，无）"
    rows = []
    for highlight in report.highlights:
        triggers = "、".join(highlight.triggers) or "-"
        rows.append(
            f"- {format_timestamp(highlight.start)} - {format_timestamp(highlight.end)}"
            f" | 强度 {highlight.strength} | {triggers} | {highlight.summary}"
        )
    return "\n".join(rows)


def build_glossary_block(glossary: dict[str, str]) -> str:
    if not glossary:
        return "（无术语表）"
    return "\n".join(f"- {key} → {value}" for key, value in glossary.items())


def build_user_prompt(
    cfg: ProjectConfig, track: DialogueTrack, report: SignalReport
) -> str:
    return render_prompt(
        load_prompt("single_episode.md"),
        show=cfg.show,
        episode_number=track.episode,
        target_seconds=f"{cfg.target_seconds:.0f}",
        duration_readable=readable_seconds(track.duration),
        glossary_block=build_glossary_block(cfg.glossary),
        highlight_block=build_highlight_block(report),
        dialogue_block=build_dialogue_block(track),
        example_block=load_prompt("examples/saijo_e02.md"),
    )


def to_script(llm_script: LLMScript, cfg: ProjectConfig, episode: int) -> Script:
    """把 LLM 输出转成内部 Script。episode 是**本次生成的那一集**。

    原来这里写的是 `[e.number for e in cfg.episodes]`，把 project.yaml 登记的全部集数都
    塞进单集的 Script.episodes；配上 docgen/table.py 的 `len(script.episodes) == 1` 判断，
    project 只要登记了 ≥2 集，每一集的对照表标题都会变成「整季 解说方案」。
    """
    beats = []
    for llm_beat in llm_script.beats:
        beats.append(
            Beat(
                id=llm_beat.id,
                label=llm_beat.label,
                role=llm_beat.role,
                narration=llm_beat.narration,
                clips=[
                    Clip(
                        episode=clip.episode,
                        start=clip.start,
                        end=clip.end,
                        visual=clip.visual,
                        anchor_lines=list(clip.anchor_lines),
                    )
                    for clip in llm_beat.clips
                ],
                audio=AudioDirection(
                    original_audio=llm_beat.original_audio,
                    sfx=list(llm_beat.sfx),
                    holds=list(llm_beat.holds),
                ),
            )
        )
    return Script(
        show=cfg.show,
        mode="single_episode",
        episodes=[episode],
        target_seconds=cfg.target_seconds,
        beats=beats,
    )


async def generate_script(
    cfg: ProjectConfig,
    track: DialogueTrack,
    report: SignalReport,
    provider: LLMProvider,
) -> tuple[Script, list[str]]:
    user_prompt = build_user_prompt(cfg, track, report)
    tracks = {track.episode: track}
    reports = {report.episode: report}
    warnings: list[str] = []

    async def draft(prompt: str) -> tuple[Script, list[str]]:
        llm_script = await provider.complete(SYSTEM_PROMPT, prompt, LLMScript)
        result = validate_script(to_script(llm_script, cfg, track.episode), tracks, reports)
        return apply_estimates(result.script), list(result.warnings)

    try:
        script, stage_warnings = await draft(user_prompt)
    except ScriptValidationError as first_error:
        warnings.append(f"首轮剧本校验失败，重试一次：{first_error}")
        script, stage_warnings = await draft(
            f"{user_prompt}\n\n## 上一轮的问题\n\n"
            f"{first_error}\n请重新输出，确保每个节点至少有一个有效 clip，"
            f"且所有时间戳都落在正片范围内。"
        )
    warnings.extend(stage_warnings)

    if needs_rewrite(script):
        instruction = rewrite_instruction(script)
        warnings.append(f"首版时长超出容差，触发一轮重写：{instruction.splitlines()[0]}")
        script, stage_warnings = await draft(
            f"{user_prompt}\n\n## 时长返工要求\n\n{instruction}"
        )
        warnings.extend(stage_warnings)
        if needs_rewrite(script):
            warnings.append(
                f"重写后估算时长 {script.est_total_seconds:.1f} 秒仍超出容差，"
                f"已接受该版本，请人工调整 script.json"
            )

    return script, warnings
