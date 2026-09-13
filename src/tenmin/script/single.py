"""单集剧本生成。一次 LLM 调用 + 最多一次重试 + 最多一轮预算重写。"""

from __future__ import annotations

from tenmin.config import ProjectConfig
from tenmin.models import (
    SPEECH_KINDS,
    AudioDirection,
    Beat,
    Clip,
    DialogueTrack,
    LLMBeat,
    LLMScript,
    Script,
    SignalReport,
)
from tenmin.script.budget import apply_estimates, needs_rewrite, rewrite_instruction
from tenmin.script.llm import LLMProvider
from tenmin.script.prompt import load_prompt, render_prompt
from tenmin.script.validate import ScriptValidationError, validate_script
from tenmin.timecode import format_timestamp, readable_seconds

# 提示词一律住在 prompts/*.md（全项目策略），这句原来硬编码在代码里。
SYSTEM_PROMPT = load_prompt("system.md").strip()

# LLMBeat 上这三个字段在内部模型里住在 Beat.audio 底下，是两个模型形状上唯一的差异。
# 从 AudioDirection 自己的字段表推导，不写死字面量：给 AudioDirection 加字段时只要
# LLMBeat 也加了同名字段，映射自动跟上；没加就自动留默认值。
_AUDIO_FIELDS = tuple(
    name for name in AudioDirection.model_fields if name in LLMBeat.model_fields
)


def build_dialogue_block(track: DialogueTrack) -> str:
    rows = []
    for line in track.lines:
        # 刻意不用 intervals.is_spoken：那个谓词还要求 duration > 0，是给区间运算用的。
        # 这里是「给 LLM 看的台词清单」，一条被坏时间码夹成零时长的行，正文照样是
        # 剧情内容，不能从上下文里抹掉。
        if line.kind not in SPEECH_KINDS or not line.text:
            continue
        mark = " ?" if line.suspect else ""
        # 合并来源必须写出来：validate.py 的 _anchor_matches **会认** merged_from 里的
        # 旧行号（anchor_lines=[8] 能匹配到 idx=7 这一行），而这个清单原来只印 idx，
        # 模型根本看不见 8 和 9 —— 双向信息不对称，模型没法主动引用那些号。
        # 格式 `7(+8,9)`。
        extra = ",".join(str(m) for m in line.merged_from if m != line.idx)
        number = f"{line.idx}(+{extra})" if extra else str(line.idx)
        rows.append(
            f"{number} | {format_timestamp(line.start)} - {format_timestamp(line.end)}"
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

    字段映射走 `model_dump` 而不是手工逐字段搬运：原来是后者，给 Clip / Beat 新增一个
    LLM 也该填的字段时会**静默丢失**（不报错、不传值，默认值一路漂到成片）。

    刻意**不**做「全字段对拷」：`LLM*` 镜像模型不含 est_seconds / est_total_seconds /
    is_silent_highlight（models.py:284 的注释说明这是设计意图，那三个由 budget.py 与
    validate.py 计算），所以这里只搬 LLMBeat / LLMClip **自己声明过**的字段，其余留默认值。
    """
    beats = []
    for llm_beat in llm_script.beats:
        data = llm_beat.model_dump()
        clips = [Clip.model_validate(clip) for clip in data.pop("clips")]
        audio = AudioDirection.model_validate(
            {name: data.pop(name) for name in _AUDIO_FIELDS}
        )
        beats.append(Beat(**data, clips=clips, audio=audio))
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
