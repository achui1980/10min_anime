"""一集对白的翻译编排。

体量实测（`work/saijo/01_dialogue` 那份真实对白轨：408 行里 396 条 dialogue、8 条
credits、3 条 monologue、1 条 noise，日语原文量级 5~6k 字符）：一次调用完全塞得下，
所以刻意不分批。分批要付的代价是「术语跨批不一致」加「调用次数乘以批数」，换来的是
解决一个不存在的问题。

真正的风险不是成本而是**对齐**：进去多少条就必须出来多少条。模型漏一条、合并两条，
中文就跟时间戳整条错位，而那种错只有看成片字幕时才发现，且是从错位那句往后全错。所以
输出带 id，拿回来先核对 id 集合，不对就把缺的/多的报给模型重来。
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from functools import cache
from importlib.resources import files
from pathlib import Path

from tenmin.config import ProjectConfig
from tenmin.models import SPEECH_KINDS, DialogueLine, DialogueTrack, TranslatedTrack
from tenmin.script.llm import (
    LLMProvider,
    LLMResponseFormatError,
    RepairContext,
    complete_with_schema_repair,
)
from tenmin.script.prompt import render_prompt
from tenmin.script.usage import UsageRecord, track_call
from tenmin.translate.glossary import effective_glossary

# 走 importlib.resources 而不是 `Path(__file__).parent`，理由跟 script.prompt 的
# PROMPTS_DIR 一样：把「资源怎么定位」交给标准库。
_PROMPTS_DIR = Path(str(files("tenmin.translate") / "prompts"))

_TEMPLATE = "translate_lines.md"  # 同时当 render_prompt 的模板名（只进报错消息）
_SYSTEM = "system.md"

# 报给模型的缺失/多余 id 最多列这么多个。回灌的消息要是三百个数字，真正要修的内容
# 就被挤出模型的注意力了 —— 列几个够它明白「你漏了东西，重新数一遍」就行。而且
# complete_with_schema_repair 会把 str(exc) 截到 REPAIR_ERROR_MAX_CHARS 才进回灌消息，
# 硬拼全量清单的下场是被静默砍掉半截。
_MAX_REPORTED_IDS = 12


@cache
def _load(name: str) -> str:
    """读一份本包的提示词资产，带缓存。

    刻意**不**复用 script.prompt.load_prompt：它的 PROMPTS_DIR 硬绑在 tenmin.script
    上（实测它只收一个文件名、并校验解析后的路径没跑出那个目录），拿它读本包的模板
    会直接 FileNotFoundError。把 script 那份改成收目录参数是更大的动静，而这里只要
    两行。渲染那一半（`render_prompt` 的双向占位符校验）照旧共用。
    """
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def select_translatable(track: DialogueTrack) -> list[tuple[int, DialogueLine]]:
    """挑出要翻译的行，连同它们在轨里的位置（从 1 起）。

    只翻**说出口的话**（SPEECH_KINDS，即 dialogue 与 monologue）：片头片尾的 staff
    名单（credits）、整行括注（screen_text）、拟声符号（noise）不是台词，翻它们是白
    花钱；而主题曲歌词在手填了 OP/ED 区间时也会落进 credits。

    monologue 刻意**在**这一类里：它是 split_dual_track 从双轨字幕拆出来的内心独白，
    是真说过的台词，漏掉就少几条中文字幕（实测那份真实对白轨里有 3 条）。

    位置下标而不是 DialogueLine.idx：后者在双轨字幕被拆开时会重复（同一个 idx 配不同
    的 segment_index），拿它当键会撞。位置天然唯一，也让对齐校验变成一次集合相等判断，
    而且它正是 srt_writer.render_zh_srt 回查时间戳时用的那个键。
    """
    return [
        (position, line)
        for position, line in enumerate(track.lines, start=1)
        if line.kind in SPEECH_KINDS
    ]


def build_lines_block(selected: Sequence[tuple[int, DialogueLine]]) -> str:
    """对白轨渲染成 `id | 原文` 的清单。

    一行一条是安全的：ingest.normalize 把 cue 内部的换行折成空格（`_fold`）之后才建
    DialogueLine，所以 `line.text` 里不会有 \\n 把一条撑成两行。
    """
    return "\n".join(f"{position} | {line.text}" for position, line in selected)


def _glossary_block(glossary: Mapping[str, str]) -> str:
    """跟 script 阶段的术语表块同形，空表时给一句话而不是留白。"""
    if not glossary:
        return "（暂无已定术语，自己定并在 glossary 里报上来）"
    return "\n".join(f"- {term} → {zh}" for term, zh in sorted(glossary.items()))


def _sample(ids: Collection[int]) -> str:
    ordered = sorted(ids)
    head = ordered[:_MAX_REPORTED_IDS]
    shown = "、".join(str(i) for i in head)
    if len(ordered) > len(head):
        shown += f" 等共 {len(ordered)} 个"
    return shown


def check_alignment(expected_ids: Collection[int], translated: TranslatedTrack) -> None:
    """译文的 id 集合必须跟送进去的完全相等。

    按集合判而不是按条数判：条数对上但有重复的情况（模型把某条译了两遍、漏了另一条）
    单看数量查不出来。

    抛 LLMResponseFormatError 是硬契约：complete_with_schema_repair 的 except 只网
    ValidationError 与这一族，抛别的（ValueError / AssertionError）会直接冒出去、
    一次重试都没有。消息里带上具体缺了哪几个 id，模型照着就能改。
    """
    expected = set(expected_ids)
    got = [line.id for line in translated.lines]
    seen = set(got)

    missing = expected - seen
    unknown = seen - expected
    duplicated = {i for i in got if got.count(i) > 1}

    problems: list[str] = []
    if missing:
        problems.append(f"漏了这些 id：{_sample(missing)}")
    if unknown:
        problems.append(f"输出了输入里没有的 id：{_sample(unknown)}")
    if duplicated:
        problems.append(f"这些 id 出现了多次：{_sample(duplicated)}")
    if problems:
        raise LLMResponseFormatError(
            f"译文没有逐条对齐（输入 {len(expected)} 条，输出 {len(got)} 条）。"
            + "；".join(problems)
            + "。请重新输出全部条目，每条 id 恰好一次。"
        )


async def translate_track(
    cfg: ProjectConfig,
    track: DialogueTrack,
    provider: LLMProvider,
    *,
    accumulated: Mapping[str, str],
    usage: list[UsageRecord] | None = None,
) -> TranslatedTrack:
    """翻译一集对白。

    accumulated 是前面几集攒下的术语表；project.yaml 里手写的那份会盖在它上面（手写
    是纠错入口，必须赢），这条偏序由 glossary.effective_glossary 定。
    """
    selected = select_translatable(track)
    if not selected:
        # 一条台词都没有就别发请求。整集都是 credits 的对白轨是解析出错的征兆，但那不是
        # 这一层该判的事 —— 这里只负责别为空清单花一次调用。
        return TranslatedTrack(episode=track.episode)

    glossary = effective_glossary(accumulated, cfg.glossary)
    expected_ids = {position for position, _ in selected}
    template = _load(_TEMPLATE)
    system = _load(_SYSTEM).strip()

    body = render_prompt(
        template,
        _TEMPLATE,
        # ensure_ascii=False：schema 里的中文描述不能被转义成 \\uXXXX。
        schema=json.dumps(TranslatedTrack.model_json_schema(), ensure_ascii=False, indent=2),
        episode_number=track.episode,
        glossary_block=_glossary_block(glossary),
        lines_block=build_lines_block(selected),
    )

    async def send(repair: RepairContext | None) -> str:
        if repair is None:
            return await track_call(
                usage, provider, cfg.llm, "translate", provider.complete(system, body)
            )
        # 纠错轮**照旧重发整份正文**，刻意不学 llm.py 那条「修复轮完全不重发正文」的
        # 省法：那一层修的是纯格式问题，而这里要修的是「第 137、298 条漏了」，模型得
        # 对着原文才补得出那几条的译文。代价是重试一次就多发一份对白轨。
        return await track_call(
            usage,
            provider,
            cfg.llm,
            "translate_repair",
            provider.complete(
                system,
                f"{body}\n\n## 上一次的输出有问题\n\n"
                f"{repair.error}\n\n上一次的输出（可能被截断）：\n\n{repair.bad_output}\n",
            ),
        )

    result = await complete_with_schema_repair(
        send,
        TranslatedTrack,
        max_attempts=cfg.llm.max_attempts,
        label=f"E{track.episode:02d} 翻译",
        check=lambda parsed: check_alignment(expected_ids, parsed),
    )
    # 集号以对白轨为准。它在 schema 里是必填字段（去掉它模型就少一个照抄的锚点），但
    # 值是我们自己的事实，不是模型的选择 —— 而下游要拿它当产物文件名（zh/E{NN}.zh.json）。
    # 模型填错时静默换个集号比报错难查得多，所以这里直接盖掉而不是再加一条校验。
    return result.model_copy(update={"episode": track.episode})
