"""单集剧本生成。一次 LLM 调用 + 最多一次重试 + 最多一轮预算重写。"""

from __future__ import annotations

from tenmin.config import LLMConfig, ProjectConfig
from tenmin.models import (
    SPEECH_KINDS,
    AudioDirection,
    Beat,
    Clip,
    DialogueTrack,
    LLMBeat,
    LLMClip,
    LLMScript,
    Script,
    SignalReport,
)
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.script.budget import (
    SPEECH_RATE_CPS,
    apply_estimates,
    budget_deviation,
    needs_rewrite,
    rewrite_instruction,
    speed_factor,
)
from tenmin.script.llm import LLMProvider
from tenmin.script.prompt import load_prompt, render_prompt
from tenmin.script.validate import ScriptValidationError, validate_script
from tenmin.timecode import format_timestamp, readable_seconds

# 提示词一律住在 prompts/*.md（全项目策略），这句原来硬编码在代码里。
SYSTEM_PROMPT = load_prompt("system.md").strip()

# single_episode.md 里 few-shot 范例那一节的标题。返工轮按它切掉整节（见 build_user_prompt）。
_EXAMPLE_SECTION_HEADING = "## 参考范例"

# 首轮 prompt 里给留白预留多少秒。首轮还没有稿子，所以算不出真实留白（budget.budget_chars
# 是拿成稿的 script_hold_seconds 去扣的），只能按提示词自己要求的「3–6 处 hold、一般
# 2–4 秒」预留一个规划值：6–24 秒，取中段。实测 13 份真实 script.json 的全片留白是
# 10.5–21.0 秒（中位 17），15 落在这个区间里。
# 刻意不做成 config 旋钮：它不是「这部番想要什么」，而是「提示词自己那条 3–6 处 × 2–4 秒
# 的要求折算成秒是多少」——改了 hold 的数量要求才该动它，而那是提示词模板的事。
HOLD_RESERVE_SECONDS = 15.0

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
    cfg: ProjectConfig,
    track: DialogueTrack,
    report: SignalReport,
    *,
    with_example: bool = True,
) -> str:
    """拼首轮的 user prompt。

    `with_example=False` 用于**返工轮**：把 few-shot 范例整段摘掉（连它上面那个
    `## 参考范例` 小节标题一起）。取舍依据见 _followup_prompt 的 docstring。
    """
    template = load_prompt("single_episode.md")
    if not with_example:
        template = template.split(_EXAMPLE_SECTION_HEADING)[0].rstrip() + "\n"
    return render_prompt(
        template,
        show=cfg.show,
        episode_number=track.episode,
        target_seconds=f"{cfg.target_seconds:.0f}",
        duration_readable=readable_seconds(track.duration),
        speech_rate=f"{SPEECH_RATE_CPS:g}",
        tolerance=f"{cfg.llm.budget_tolerance:.0%}",
        hold_reserve=f"{HOLD_RESERVE_SECONDS:.0f}",
        narration_char_budget=int(
            (cfg.target_seconds - HOLD_RESERVE_SECONDS)
            * SPEECH_RATE_CPS
            * speed_factor(cfg.render.rate)
        ),
        glossary_block=build_glossary_block(cfg.glossary),
        highlight_block=build_highlight_block(report),
        dialogue_block=build_dialogue_block(track),
        example_block=load_prompt("examples/saijo_e02.md") if with_example else "",
    )


def to_llm_script(script: Script) -> LLMScript:
    """`to_script` 的逆映射：把内部 Script 还原成模型自己那套 schema 的形状。

    返工轮要把上一版交回模型，交的必须是**它自己会输出的形状**（不能带 est_seconds /
    is_silent_highlight 这些内部字段，那只会教它去填不该填的东西）。而且交回去的是
    **repair 之后**的版本：坏 clip 已经丢掉、偏差过大的时间戳已按字幕校准，所以模型
    不会把上一轮已经修掉的错再写回来。
    """
    beats = []
    for beat in script.beats:
        data = {
            name: getattr(beat, name)
            for name in LLMBeat.model_fields
            if name not in _AUDIO_FIELDS and name != "clips"
        }
        data.update({name: getattr(beat.audio, name) for name in _AUDIO_FIELDS})
        data["clips"] = [
            {name: getattr(clip, name) for name in LLMClip.model_fields}
            for clip in beat.clips
        ]
        beats.append(LLMBeat.model_validate(data))
    return LLMScript(beats=beats)


def _followup_prompt(base: str, previous: Script, heading: str, instruction: str) -> str:
    """返工轮的 prompt：素材 + **上一版的 JSON** + 本轮要求。

    原来两个返工轮（校验重试、预算重写）都只是在整份 prompt 后面缀一行要求，**从不把
    上一版交回模型**——于是 budget.rewrite_instruction 里那句「不要改动节点划分、clip
    时间戳、留白金句」是个模型物理上做不到的要求（它手上根本没有上一版），所以重写轮
    几乎必然打乱时间戳、再走一遍 validate。

    哪些部分重发、哪些不重发（实测 saijo E02 的一份真实 prompt 共 34628 字符）：

    - 对白轨 27077 字（78.2%）：**必须重发**。重试要修的语义错误（时间戳越界、人物
      关系写反、事件顺序与时间戳不一致）全部只能对着对白原文才判得出来；预算重写要
      改的是旁白正文，而提示词里 5 条废稿条件有 4 条是「以对白原文为准」——把源材料
      撤掉再让它改写散文，正是制造幻觉的做法。所以这里**不能**照搬 P1-D 那种
      「schema 修复轮完全不重发正文」的省法：那一层的报错是纯格式问题，不需要上下文。
    - 高能点清单 2067 字（6.0%）：必须重发。「每个节点至少有一个 clip 落在 gap 区间里」
      这条要求靠它。
    - few-shot 范例 3049 字（8.8%）：**不重发**。它唯一的作用是教「格式、语气、节奏」，
      而走到返工轮时模型已经交出过一份合 schema 的稿子，格式显然学会了；范例自己还带
      着「只用来学格式，不要学它的内容，写出来的画面描述在范例里出现过就是抄错了」的
      警告，撤掉只会降低污染风险。
    - 上一版 JSON 11275 字：新增的成本。

    净效果：返工轮的 prompt 从 34628 涨到约 42854 字符（+23.8%）。这是让那条指令从
    「不可能完成」变成「可以完成」的代价，值得。
    """
    payload = to_llm_script(previous).model_dump_json(indent=2)
    return (
        f"{base}\n\n## 上一版输出\n\n"
        f"下面是你上一轮交的稿子，本轮请在它的基础上**局部修改**，不要从头重写：\n\n"
        f"```json\n{payload}\n```\n\n"
        f"## {heading}\n\n{instruction}"
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
    *,
    reporter: ProgressReporter | None = None,
) -> tuple[Script, list[str]]:
    """生成一集的剧本：首稿 + 最多 N 次校验重试 + 最多 M 轮预算返工。

    N / M / 预算容差全部从 `cfg.llm` 取（validation_retries / budget_rewrite_rounds /
    budget_tolerance），原来是写死在函数体里的 1 / 1 / DEFAULT_TOLERANCE。
    """
    reporter = reporter or NullProgressReporter()
    llm = cfg.llm
    rate = cfg.render.rate
    tracks = {track.episode: track}
    reports = {report.episode: report}
    first_prompt = build_user_prompt(cfg, track, report)
    followup_base = build_user_prompt(cfg, track, report, with_example=False)
    warnings: list[str] = []

    # 一轮 = 一次 LLM 调用。上界用于给进度条一个确定的 total（rich 的 substep 行有
    # TimeElapsedColumn，用户就能区分「在跑」与「卡死」——一次真实调用可达 561 秒）。
    total_rounds = 1 + llm.validation_retries + llm.budget_rewrite_rounds
    round_index = 0

    async def draft(prompt: str, label: str) -> tuple[Script, list[str]]:
        nonlocal round_index
        round_index += 1
        reporter.substep("script", round_index, total_rounds, label)
        llm_script = await provider.complete(SYSTEM_PROMPT, prompt, LLMScript)
        result = validate_script(
            to_script(llm_script, cfg, track.episode),
            tracks,
            reports,
            cfg=cfg.validate_script,
        )
        return (
            apply_estimates(result.script, rate=rate),
            list(result.warnings),
        )

    # while True + break 而不是 for + assert：src/ 里不许有 assert（python -O 会把它整句
    # 剥掉，见 tests/test_source_hygiene.py），而这个循环只有 break 与 raise 两个出口。
    # 退避重试的同款写法见 render/tts.py 的 synthesize_with_retry。
    prompt, label = first_prompt, "生成初稿"
    attempts = llm.validation_retries + 1
    attempt = 0
    while True:
        attempt += 1
        try:
            script, stage_warnings = await draft(prompt, label)
            break
        except ScriptValidationError as error:
            if attempt >= attempts:
                raise
            warnings.append(f"第 {attempt} 轮剧本校验失败，重试：{error}")
            label = f"校验失败，第 {attempt} 次重试"
            # 校验失败的现场（哪些 clip 被丢了）已经在 error.script 里，交回去让模型
            # 只改那几处，而不是整篇重生成。
            prompt = (
                _followup_prompt(
                    followup_base,
                    error.script,
                    "上一轮的问题",
                    f"{error}\n请在上一版基础上修掉这个问题，"
                    f"确保每个节点至少有一个有效 clip，且所有时间戳都落在正片范围内。",
                )
                if error.script is not None
                else f"{followup_base}\n\n## 上一轮的问题\n\n{error}\n请重新输出。"
            )

    # `stage_warnings`（这一版的 validate warning）**刻意先不进 warnings**：返工轮可能
    # 把这一版整个换掉，提前放进去就会变成「在描述一份不存在的稿子」。统一等到下面
    # 确定了采纳哪一版再一起 extend。

    for round_number in range(1, llm.budget_rewrite_rounds + 1):
        if not needs_rewrite(script, llm.budget_tolerance, rate=rate):
            break
        instruction = rewrite_instruction(script, llm.budget_tolerance, rate=rate)
        warnings.append(
            f"第 {round_number} 轮时长超出容差，触发返工：{instruction.splitlines()[0]}"
        )
        try:
            candidate, candidate_warnings = await draft(
                _followup_prompt(followup_base, script, "时长返工要求", instruction),
                f"时长返工第 {round_number} 轮",
            )
        except ScriptValidationError as error:
            # 返工轮把稿子写坏了：上一版是通过校验的，留着它比整次失败好。
            warnings.append(f"时长返工第 {round_number} 轮的稿子没通过校验，保留上一版：{error}")
            break
        script, stage_warnings = _pick_better(
            script, stage_warnings, candidate, candidate_warnings, llm, rate, warnings
        )

    # warnings 必须跟着**被采纳的那一版**：原来首版的 validate warning（比如
    # 「clip 落在片头曲内，丢弃」）在触发返工时就已经进了列表，而那份稿子随后被丢弃，
    # 用户看到的是在描述一份**不存在的稿子**。
    warnings.extend(stage_warnings)
    if needs_rewrite(script, llm.budget_tolerance, rate=rate):
        warnings.append(
            f"估算时长 {script.est_total_seconds:.1f} 秒仍超出容差 "
            f"±{llm.budget_tolerance:.0%}，已接受该版本，请人工调整 script.json"
        )
    return script, warnings


def _pick_better(
    current: Script,
    current_warnings: list[str],
    candidate: Script,
    candidate_warnings: list[str],
    llm: LLMConfig,
    rate: str,
    warnings: list[str],
) -> tuple[Script, list[str]]:
    """两版取偏差绝对值小的那个。

    原来是**无条件**用新稿替换旧稿，即使新稿偏差更大；第二轮 needs_rewrite 仍为真时也
    只加一条 warning 就接受了可能更差的版本。
    """
    current_deviation = abs(budget_deviation(current, rate=rate))
    candidate_deviation = abs(budget_deviation(candidate, rate=rate))
    if candidate_deviation < current_deviation:
        warnings.append(
            f"采纳重写版（偏差 {candidate_deviation:+.1%}），"
            f"首版偏差 {current_deviation:+.1%}"
        )
        return candidate, candidate_warnings
    warnings.append(
        f"重写版偏差 {candidate_deviation:+.1%} 不优于首版 {current_deviation:+.1%}，"
        f"采纳首版"
    )
    return current, current_warnings
