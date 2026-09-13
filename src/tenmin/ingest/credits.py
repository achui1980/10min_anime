"""staff/credits 行识别与 OP/ED 区间推断。"""

from __future__ import annotations

import math
import re

# 所有阈值的权威定义在 tenmin.config.CreditsConfig；DEFAULT_CREDITS 只是它的
# 默认实例，让不关心配置的调用点（单测、一次性脚本）可以继续零参数调用。
from tenmin.config import DEFAULT_CREDITS, CreditsConfig
from tenmin.ingest.clean import _TITLE_CARD
from tenmin.intervals import merge_intervals, silent_gaps
from tenmin.models import DialogueLine

# 无条件生效：日文专有写法或高度特定的复合词，正常中文台词里不可能出现。
_KEYWORDS_ALWAYS = (
    "製作委員会",
    "製作委員會",
    "制作委员会",
    "作詞",
    "作词",
    "作曲",
    "編曲",
    "编曲",
    "フォント",
    "STAFF",
    "Studio",
    "主題歌",
    "主题歌",
)

# 仅片头片尾窗内生效：这些词同时是通用中文词汇，子串匹配会误伤台词。
# 实测「你现在的说话方式也是演出来的吧」被「演出」命中 → 整行被踢出台词，
# 两侧间隙被虚假合并成一个假高光直接喂给 LLM。
_KEYWORDS_IN_WINDOW = (
    "製作",
    "制作",
    "協力",
    "协力",
    "作画",
    "監督",
    "监督",
    "脚本",
    "演出",
    "原作",
)
# `_TITLE_CARD`（规则 6 用）不在这里定义，从 clean.py import：原先两处各写一份且不
# 一致（这边多了 `\s*`，能接 `第 3 集`，clean 那边不能），已收敛成宽的那份。
# import 方向安全 —— clean.py 只依赖 re / functools / typing，不 import ingest 里的
# 任何东西，所以不成环。
_NAME_LIST_EVEN = re.compile(r"^(?:[\u4e00-\u9fff]{2,4})(?:\s+[\u4e00-\u9fff]{2,4})+$")
_NAME_LIST_RAGGED = re.compile(r"^(?:[\u4e00-\u9fff]{1,5})(?:\s+[\u4e00-\u9fff]{1,5}){2,}$")
_BRACKET_WRAPPED = re.compile(r"^[《『「(（]\s*(?P<inner>.+?)\s*[》』」)）]$")
_LATIN = re.compile(r"[A-Za-z]")
_NON_SPACE = re.compile(r"\S")


def _title_overlap(text: str, show_title: str) -> float:
    title_chars = set(show_title) - set(" 　")
    if not title_chars:
        return 0.0
    return len(title_chars & set(text)) / len(title_chars)


def is_credits(
    text: str,
    *,
    show_title: str = "",
    in_credit_window: bool = False,
    cfg: CreditsConfig = DEFAULT_CREDITS,
) -> bool:
    """判断一行清洗后的文本是不是 staff / 版权 / 标题卡。

    规则 1、2a、3、6 无条件生效；规则 2b、4、5 只在片头片尾时间窗内生效，
    否则 `早安 早安` 这类空格分隔的正常台词会被纯人名正则误伤，
    `演出来` 这类正常台词也会被 `演出` 的子串匹配误伤。
    """
    stripped = text.strip()
    if not stripped:
        return False

    # 1. 版权标记
    if "©" in stripped or "(C)" in stripped.upper():
        return True

    # 2a. staff 关键词里无条件生效的那批（繁简与日文都列）
    upper = stripped.upper()
    for keyword in _KEYWORDS_ALWAYS:
        if keyword.upper() in upper:
            return True

    # 6. 标题卡
    if _TITLE_CARD.search(stripped) and len(stripped) <= cfg.title_card_max_len:
        return True

    # 3. 被书名号/引号包裹且与剧名字符高度重合
    wrapped = _BRACKET_WRAPPED.match(stripped)
    if wrapped and show_title:
        overlap = _title_overlap(wrapped.group("inner"), show_title)
        if overlap >= cfg.title_overlap_threshold:
            return True

    if not in_credit_window:
        return False

    # 2b. staff 关键词里有中文歧义的那批，只在片头片尾窗内才敢认
    for keyword in _KEYWORDS_IN_WINDOW:
        if keyword.upper() in upper:
            return True

    # 4. 纯人名罗列。EVEN 接「河原正信 有贺史英」这种齐整两段；
    # RAGGED 接「慧 诹访 豊 和田雄一郎」这种参差不齐但至少三段的 staff 罗列。
    # RAGGED 要求 3 段以上，否则「早安 早安」这类两段短台词会被误伤。
    # 段数 >= 4 时字数门槛放宽到 4，接「慧 诹 访 郎」这种被 OCR 拆碎的人名。
    if _NAME_LIST_EVEN.match(stripped) or _NAME_LIST_RAGGED.match(stripped):
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", stripped))
        segment_count = len(stripped.split())
        min_cjk = (
            cfg.name_list_many_min_cjk
            if segment_count >= cfg.name_list_many_segments
            else cfg.name_list_min_cjk
        )
        if cjk_count >= min_cjk:
            return True

    # 5. 拉丁字母为主
    non_space = _NON_SPACE.findall(stripped)
    if len(non_space) >= cfg.latin_min_len:
        latin_ratio = len(_LATIN.findall(stripped)) / len(non_space)
        if latin_ratio > cfg.latin_ratio_threshold:
            return True

    return False


def credit_window_bounds(
    duration: float, *, cfg: CreditsConfig = DEFAULT_CREDITS
) -> tuple[float, float]:
    """算出 (片头窗上界, 片尾窗下界)。片尾窗被关掉时下界是 `math.inf`。

    两个标称窗（credit_head_window / ed_keyword_window_seconds）加起来共享一份
    `duration * credit_window_max_ratio` 的覆盖预算，片头窗优先拿 —— OP staff 一定在
    片头，ED staff 在短片里未必存在。预算不够时片头窗先缩到预算大小，剩下多少给片尾窗，
    剩 0 就整体关掉片尾窗。

    这不是一个 on/off 开关而是连续退化：duration 600-760 之间片尾窗按剩余预算逐渐
    变窄，<=600 才彻底关闭，所以不存在「多一秒少一秒行为跳变」的悬崖。
    """
    budget = duration * cfg.credit_window_max_ratio
    head_end = min(cfg.credit_head_window, budget)
    tail_length = min(cfg.ed_keyword_window_seconds, budget - head_end)
    tail_start = duration - tail_length if tail_length > 0 else math.inf
    return head_end, tail_start


def in_credit_window(
    start: float, duration: float, *, cfg: CreditsConfig = DEFAULT_CREDITS
) -> bool:
    """片头 0-300s 或片尾最后 80s。给 is_credits 的规则 2b/4/5 开门。

    这里用的是 credit_head_window / ed_keyword_window_seconds 两个独立旋钮，
    跟聚簇用的 op_search_* / ed_cluster_tail_seconds 不共享数值：片尾窗口刻意比
    ed_cluster_tail_seconds 窄，理由见 CreditsConfig 里的注释。

    两个窗的实际大小由 credit_window_max_ratio 兜底收缩，保证并集永远覆盖不满整条
    时间轴 —— 否则短 track（以及 duration=0 的空字幕）上每一行都会拿到
    in_credit_window=True，本文件头部记录的「演出来」被「演出」命中那类事故就会从
    「只在片头片尾发生」升级成「全片发生」。
    """
    if duration <= 0:
        # 片长未知（空字幕 / 解析失败）。宁可漏判 credits，也不能把激进规则对全片放开。
        return False
    head_end, tail_start = credit_window_bounds(duration, cfg=cfg)
    return start <= head_end or start >= tail_start


def _silent_gaps_in_window(
    lines: list[DialogueLine], window: tuple[float, float]
) -> list[tuple[float, float]]:
    """列出起点落在 window 内的全部无字幕静默间隙，不做任何时长筛选。

    静默的定义（哪种行算「有人说话」）由 tenmin.intervals 独家持有，
    ingest 与 signals 两层共用同一份，不会再各自分叉。不传 duration，
    因为片尾到片长的那段静默跟 OP 识别无关。
    """
    return [
        (gap.start, gap.end)
        for gap in silent_gaps(lines)
        if window[0] <= gap.start <= window[1]
    ]


def _op_from_silence(
    lines: list[DialogueLine], cfg: CreditsConfig
) -> tuple[float, float] | None:
    """credits 聚簇算不出 OP 时的兜底。

    「没字幕的 OP」（字幕组一条 staff 行都没打）在聚簇法下必然漏判，
    但它整段就是一个 90 秒级的静默区，反过来比有字幕的 OP 更好认。

    必须「先按时长过滤、再取最长」，顺序不能反。若先取窗内最长再验时长，
    某集片头附近只要存在一个 >op_max_silent_span 的非-OP 静区
    （例如整段无对白的长过场），它就会挤掉真正的 90 秒 OP 候选，
    使兜底直接返回 None——本该救回来的 OP 反而丢了。
    """
    window = (cfg.op_search_start, cfg.op_search_end)
    candidates = [
        gap
        for gap in _silent_gaps_in_window(lines, window)
        if cfg.op_min_silent_span <= gap[1] - gap[0] <= cfg.op_max_silent_span
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda gap: gap[1] - gap[0])


def find_credit_ranges(
    lines: list[DialogueLine],
    duration: float,
    *,
    cfg: CreditsConfig = DEFAULT_CREDITS,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """从已标记 kind=="credits" 的行推断 OP / ED 区间。

    OP 走两条路：credits 聚簇为主路，算不出来时退到静区兜底。
    ED 只走聚簇，不做兜底——片尾前的长静场（定格收尾）是真高光，兜底会吃掉它。
    """
    credit_lines = [ln for ln in lines if ln.kind == "credits"]
    clusters = merge_intervals(
        ((ln.start, ln.end) for ln in credit_lines), max_gap=cfg.cluster_max_gap
    )

    op_candidates = [
        c
        for c in clusters
        if cfg.op_search_start <= c[0] <= cfg.op_search_end
        and cfg.op_span_min <= c[1] - c[0] <= cfg.op_span_max
    ]
    op = max(op_candidates, key=lambda c: c[1] - c[0]) if op_candidates else None
    if op is None:
        op = _op_from_silence(lines, cfg)

    ed_candidates = [
        c for c in clusters if c[0] >= duration - cfg.ed_cluster_tail_seconds
    ]
    ed = min(ed_candidates, key=lambda c: c[0]) if ed_candidates else None

    return op, ed
