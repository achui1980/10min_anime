"""把 SRT 汇聚成标准化 DialogueTrack。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from tenmin.config import DEFAULT_CREDITS, DEFAULT_INGEST, CreditsConfig, IngestConfig
from tenmin.ingest.clean import (
    clean_text,
    extract_prefix,
    is_noise,
    is_suspect,
    split_dual_track,
)
from tenmin.ingest.credits import find_credit_ranges, in_credit_window, is_credits
from tenmin.ingest.srt_parser import load_srt_detailed
from tenmin.models import DialogueLine, DialogueTrack

# 整行都被一对圆括号包住 —— 屏幕注释/拟声，不是台词。
# 中间刻意禁止再出现闭括号（`[^）)]*` 而不是 `.*`）：贪婪的 `.*` 会把
# 「（超过 40 字的长注释）真台词（小声）」整行匹配成功 → kind="screen_text" →
# 被 SPEECH_KINDS 过滤掉，真台词从语音轨消失、两侧静默间隙还被虚假拉长。
# 括号字符类与 clean._PAREN_PREFIX 保持一致（同样只认圆括号的全角/半角两种，
# 不含【】《》—— 那些交给 credits._BRACKET_WRAPPED 与 is_credits 处理）。
_FULL_PAREN = re.compile(r"^[（(][^）)]*[）)]$")

_NEWLINE_RUN = re.compile(r"\s*\n\s*")

# 句末标点集合。刻意不进 config：它是「中文/日文怎么断句」的语言学事实，
# 不是调参旋钮，放进 project.yaml 只会被误改。数值阈值见 config.IngestConfig。
TERMINAL_PUNCT = frozenset("。！？…」』、，,!?.")


def _fold(text: str) -> str:
    return _NEWLINE_RUN.sub(" ", text).strip()


def _classify(
    body: str,
    *,
    is_second_segment: bool,
    speaker: str | None,
    show_title: str,
    window: bool,
    credits: CreditsConfig,
) -> str:
    if is_noise(body):
        return "noise"
    if is_credits(body, show_title=show_title, in_credit_window=window, cfg=credits):
        return "credits"
    if _FULL_PAREN.match(body):
        return "screen_text"
    if is_second_segment and speaker is not None:
        return "monologue"
    return "dialogue"


def _can_merge(prev: DialogueLine, nxt: DialogueLine, cfg: IngestConfig) -> bool:
    if prev.kind != "dialogue" or nxt.kind != "dialogue":
        return False
    # 同一条 cue 被 split_dual_track 拆出的两段不能合并回去（它们是叠在同一时间区间的
    # 台词与内心独白，不是被硬折断的一句）。idx 是 cue 级的键，所以这个判断就是
    # 「来自同一条 cue 吗」。idx 改用 position 之后这个守卫的语义才真正准确：原先用
    # 文件序号时，两条不相关的 cue 只要序号撞了也会被误判成同 cue 而拒绝合并。
    if prev.idx == nxt.idx:
        return False
    if prev.speaker != nxt.speaker:
        return False
    if prev.suspect or nxt.suspect:
        return False
    if not prev.text or not nxt.text:
        return False
    if nxt.start - prev.end >= cfg.merge_max_gap:
        return False
    # 被硬折断的续行都很短；单行 >=4s 说明它本身就是完整的一句（往往是拖长音），
    # 合并它会破坏 signals/density.py 的字密度最小值锚点。
    if (
        prev.duration >= cfg.merge_max_line_seconds
        or nxt.duration >= cfg.merge_max_line_seconds
    ):
        return False
    if prev.text[-1] in TERMINAL_PUNCT:
        return False
    if len(prev.text) + len(nxt.text) > cfg.merge_max_chars:
        return False
    return True


def merge_continuations(
    lines: list[DialogueLine], cfg: IngestConfig = DEFAULT_INGEST
) -> list[DialogueLine]:
    """把被硬折成两块的同一句话合回去。被并入的 SRT 序号记进 merged_from。"""
    out: list[DialogueLine] = []
    for line in lines:
        if out and _can_merge(out[-1], line, cfg):
            prev = out[-1]
            out[-1] = prev.model_copy(
                update={
                    "end": line.end,
                    "text": prev.text + line.text,
                    "raw": f"{prev.raw}\n{line.raw}",
                    "merged_from": [*prev.merged_from, line.idx],
                }
            )
        else:
            out.append(line)
    return out


def _resolve_manual_range(
    episode_range: tuple[float, float] | None,
    project_default: tuple[float, float | None] | None,
    duration: float,
) -> tuple[float, float] | None:
    """OP/ED 区间三级回退的前两级：逐集手填 → 项目级手填。

    都没填就返回 None，交给第三级（find_credit_ranges 的启发式推断）。

    项目级默认的终点允许是 None（= 到片尾），在这里解析成**这一集**的真实片长 ——
    片长逐集不同（实测 1315.94-1510.0 秒），所以这个解析只能发生在知道 duration
    的地方，不能在 config 加载期就定下来。

    解析结果不合法时（duration 未知或早于区间起点，例如空字幕轨上写了
    `default_ed_range: [1290, null]`）返回 None 而不是一个起点晚于终点的区间：
    写反的区间在下游一路不报错，intervals.subtract 会把它当空集静默忽略。
    """
    if episode_range is not None:
        return episode_range
    if project_default is None:
        return None
    start, end = project_default
    resolved_end = duration if end is None else end
    if resolved_end <= start:
        return None
    return (start, resolved_end)


def credit_range_source(
    episode_range: tuple[float, float] | None,
    project_default: tuple[float, float | None] | None,
    duration: float,
) -> Literal["episode", "project", "inferred"]:
    """三级回退里实际生效的是哪一级。给 `tenmin inspect` 的显示用。

    刻意复用 `_resolve_manual_range` 而不是自己再判一遍「字段填了没」：项目级默认
    可能填了却在这一集上解析不出合法区间（例如空字幕轨上的 `[1290, null]`），
    那时实际走的是推断，显示也必须说推断。两处各写一遍判据就会静默分叉。
    """
    if episode_range is not None:
        return "episode"
    if _resolve_manual_range(None, project_default, duration) is not None:
        return "project"
    return "inferred"


def build_track(
    srt_path: Path,
    *,
    episode: int,
    show_title: str = "",
    convert_traditional: bool = True,
    glossary: dict[str, str] | None = None,
    op_range: tuple[float, float] | None = None,
    ed_range: tuple[float, float] | None = None,
    merge_lines: bool = False,
    duration: float | None = None,
    ingest: IngestConfig | None = None,
    credits: CreditsConfig | None = None,
) -> DialogueTrack:
    """把 SRT 汇聚成标准化对白轨。

    merge_lines 默认关闭。番剧字幕的 cue 通常是背靠背排的（本项目黄金样本间隔中位数
    0.001 秒）且基本不打句读，跨 cue 合并会把不同说话人黏成一条。句内折行本来就在
    cue 内部用 \\n 表示，_fold 已经处理掉了。带可靠说话人标注的字幕源才适合开启。

    duration 是**这一集的真实片长**（秒），由调用方从源视频 probe 出来。传了就以它
    为准，没传（None）才退化到 `max(cue.end)`。这个值不是可选的装饰：ED 窗
    （in_credit_window 的 ed_keyword_window_seconds）、ED 聚簇（ed_cluster_tail_seconds）
    与静区兜底 OP 的判定全是「距片尾多少秒」，而字幕通常在 ED staff 名单跑完前就停了，
    用字幕末尾当片尾会把整个片尾窗系统性地往前挪。刻意**不**取两者较大值：视频比字幕
    短（片源被裁过、字幕对不上片源）时悄悄回退到字幕值只会把这个真问题永久隐藏。

    ingest / credits 收全部数值阈值。刻意传整个 config 对象而不是散装参数：
    两者加起来有近 20 个旋钮，摊平成关键字参数这个签名就没法看了。
    """
    ingest = ingest or DEFAULT_INGEST
    credits = credits or DEFAULT_CREDITS
    parsed = load_srt_detailed(srt_path)
    # merge_continuations 只比较相邻元素、in_credit_window 逐条按 duration 判断，
    # 两者都默认 cue 按时间有序。乱序 SRT（合并多个字幕源时常见）会导致错误合并与
    # 错误的 duration 归因，而且全程不报错。O(n log n) 相对整条流水线可以忽略。
    # 实测 work/ 下 11 集素材本来就有序，排序后产物逐字节不变。
    cues = sorted(parsed.cues, key=lambda cue: (cue.start, cue.end))
    if duration is None:
        duration = max((cue.end for cue in cues), default=0.0)
    # 区间必须在逐 cue 分类**之前**定下来：手填区间直接当 in_credit_window 的窗，
    # 而窗决定 is_credits 的规则 2b/4/5 是否开火、也就决定 kind=="credits"。
    # 手填那条路没有鸡生蛋问题（区间不需要先分类就知道）；只有落到第三级的推断
    # 才需要「先分类、再从 credits 行反推区间」的老顺序。
    manual_op = _resolve_manual_range(op_range, credits.default_op_range, duration)
    manual_ed = _resolve_manual_range(ed_range, credits.default_ed_range, duration)
    lines: list[DialogueLine] = []

    for cue in cues:
        cleaned = clean_text(cue.text, convert=convert_traditional, glossary=glossary)
        window = in_credit_window(
            cue.start, duration, cfg=credits, op=manual_op, ed=manual_ed
        )
        for position, segment in enumerate(split_dual_track(cleaned)):
            speaker, body = extract_prefix(segment)
            # 被夹成零时长的坏 cue 一律算可疑行：DialogueLine.suspect 本来就是为这类
            # 「只打标不删除、交给有全局上下文的 LLM 判断」的行准备的。
            suspect = is_suspect(segment) or cue.clamped
            body = _fold(body)
            kind = _classify(
                body,
                is_second_segment=position > 0,
                speaker=speaker,
                show_title=show_title,
                window=window,
                credits=credits,
            )
            lines.append(
                DialogueLine(
                    idx=cue.idx,
                    src_idx=cue.src_idx,
                    segment_index=position,
                    start=cue.start,
                    end=cue.end,
                    text=body,
                    raw=cue.text,
                    speaker=speaker,
                    kind=kind,
                    suspect=suspect,
                )
            )

    if merge_lines:
        lines = merge_continuations(lines, ingest)
    inferred_op, inferred_ed = find_credit_ranges(lines, duration, cfg=credits)
    return DialogueTrack(
        episode=episode,
        source="srt",
        duration=duration,
        op_range=manual_op or inferred_op,
        ed_range=manual_ed or inferred_ed,
        lines=lines,
        skipped_blocks=parsed.skipped_blocks,
        clamped_cues=parsed.clamped_cues,
    )
