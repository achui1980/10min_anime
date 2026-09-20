"""staff/credits 行识别与 OP/ED 区间推断。"""

from __future__ import annotations

import math
import re
from functools import lru_cache

# 所有阈值的权威定义在 tenmin.config.CreditsConfig；DEFAULT_CREDITS 只是它的
# 默认实例，让不关心配置的调用点（单测、一次性脚本）可以继续零参数调用。
from tenmin.config import DEFAULT_CREDITS, CreditsConfig
from tenmin.ingest.clean import _TITLE_CARD
from tenmin.intervals import merge_intervals, silent_gaps
from tenmin.models import DialogueLine

# 无条件生效：日文专有写法或高度特定的复合词，正常中文台词里不可能出现。
#
# 判据是「这个词在中文台词里不可能出现」，所以**普通英文单词不够格**：`STAFF` 与
# `Studio` 原先在这张表里，而它们是按 `.upper()` 子串无条件匹配的，
# 「去studio看看」整行就会被判成 credits、被踢出台词轨、两侧间隙虚假合并 ——
# 跟下面记录的「演出」事故完全同类。两者已挪进 _KEYWORDS_IN_WINDOW。
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
    "主題歌",
    "主题歌",
)

# 仅片头片尾窗内生效：这些词同时是通用中文词汇（或普通英文单词），
# 子串匹配会误伤台词。
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
    "STAFF",
    "Studio",
)


def _keyword_matcher(keywords: tuple[str, ...]) -> re.Pattern[str]:
    """把一张关键词表编译成一条 alternation 正则，在**大写化后的**文本上搜。

    与原来那句 `any(kw.upper() in upper for kw in keywords)` 逐字节等价（alternation
    命中 ⟺ 任一分支是子串），但把每行 22 次 `.upper()` + 22 次子串扫描收成一次扫描。
    `is_credits` 是每行都走的热路径，实测 600 行 × 200 轮：0.099s -> 0.011s（9 倍）。

    正则用**大写化后的关键词**编译、搜大写化后的文本，跟原来 `.upper()` 对 `.upper()`
    完全同源 —— 刻意不用 `re.IGNORECASE`，那条路跟 `str.upper()` 在 `ß` -> `SS`、
    `ﬅ` -> `ST` 这类折叠上并不一致。

    空表会编译成 `""`，而那条正则匹配任何文本 —— 等于把整条字幕全判成 credits。
    两张表都是本文件里的静态字面量，真空了一定是改错了，所以直接拒绝。
    """
    if not keywords:
        raise ValueError("关键词表不能为空：空的 alternation 正则会匹配任何文本")
    return re.compile("|".join(re.escape(keyword.upper()) for keyword in keywords))


_ALWAYS_RE = _keyword_matcher(_KEYWORDS_ALWAYS)
_IN_WINDOW_RE = _keyword_matcher(_KEYWORDS_IN_WINDOW)
# `_TITLE_CARD`（规则 6 用）不在这里定义，从 clean.py import：原先两处各写一份且不
# 一致（这边多了 `\s*`，能接 `第 3 集`，clean 那边不能），已收敛成宽的那份。
# import 方向安全 —— clean.py 只依赖 re / functools / typing，不 import ingest 里的
# 任何东西，所以不成环。
#
# 汉字字符类只写一份：规则 4 的两条人名正则与它的字数门槛用的是同一个范围。
# 刻意**不**跟 `clean._CJK` 合并 —— 那个范围多了假名（`\u3040-\u30ff`），
# 语义是「这段文本有没有 CJK 内容」；这里数的是「汉字有几个」（日文 staff 名的汉字），
# 把假名算进来会改变 cjk_count。两者不是重复。
_HAN_CLASS = r"[\u4e00-\u9fff]"
_HAN = re.compile(_HAN_CLASS)
_NAME_LIST_EVEN = re.compile(rf"^(?:{_HAN_CLASS}{{2,4}})(?:\s+{_HAN_CLASS}{{2,4}})+$")
_NAME_LIST_RAGGED = re.compile(
    rf"^(?:{_HAN_CLASS}{{1,5}})(?:\s+{_HAN_CLASS}{{1,5}}){{2,}}$"
)
_BRACKET_WRAPPED = re.compile(r"^[《『「(（]\s*(?P<inner>.+?)\s*[》』」)）]$")
_LATIN = re.compile(r"[A-Za-z]")
_NON_SPACE = re.compile(r"\S")


@lru_cache(maxsize=8)
def _title_chars(show_title: str) -> frozenset[str]:
    """剧名的字符集（去掉半角/全角空格）。按剧名缓存 —— 原先每次调用都重建一遍，
    而一次 build_track 会对每行都可能调到。实测 600 次 × 300 轮：0.051s -> 0.004s。
    """
    return frozenset(show_title) - frozenset(" 　")


def _title_overlap(text: str, show_title: str) -> float:
    title_chars = _title_chars(show_title)
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
    # 大写化一次就够。原先规则 1 与规则 2a 各调一次 `stripped.upper()`。
    upper = stripped.upper()

    # 1. 版权标记
    if "©" in stripped or "(C)" in upper:
        return True

    # 2a. staff 关键词里无条件生效的那批（繁简与日文都列）
    if _ALWAYS_RE.search(upper):
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

    # 2b. staff 关键词里有中文歧义（或本身就是普通英文单词）的那批，只在片头片尾窗内才敢认
    if _IN_WINDOW_RE.search(upper):
        return True

    # 4. 纯人名罗列。EVEN 接「河原正信 有贺史英」这种齐整两段；
    # RAGGED 接「慧 诹访 豊 和田雄一郎」这种参差不齐但至少三段的 staff 罗列。
    # RAGGED 要求 3 段以上，否则「早安 早安」这类两段短台词会被误伤。
    # 段数 >= 4 时字数门槛放宽到 4，接「慧 诹 访 郎」这种被 OCR 拆碎的人名。
    if _NAME_LIST_EVEN.match(stripped) or _NAME_LIST_RAGGED.match(stripped):
        cjk_count = len(_HAN.findall(stripped))
        segment_count = len(stripped.split())
        min_cjk = (
            cfg.name_list_many_min_cjk
            if segment_count >= cfg.name_list_many_segments
            else cfg.name_list_min_cjk
        )
        if cjk_count >= min_cjk:
            return True

    # 5. 拉丁字母为主。
    # 两个 findall 刻意保留 —— 「改成生成器求和省掉中间列表」实测是**负收益**：
    # 600 行 × 200 轮，findall 0.074s vs 生成器 0.110s（C 层扫描比 Python 层逐字符
    # 迭代快，即使前者要建一个小列表）。而且这段在规则 3 之后的 `not in_credit_window`
    # 早退之下，实测 11 集只有 18.3% 的行走到（4656 行里 852 行）。
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
    start: float,
    duration: float,
    *,
    cfg: CreditsConfig = DEFAULT_CREDITS,
    op: tuple[float, float] | None = None,
    ed: tuple[float, float] | None = None,
) -> bool:
    """给 is_credits 的规则 2b/4/5 开门的时间窗。

    两种模式，`op` / `ed` 任一非 None 就走第一种：

    1. **手填区间模式**：窗就是 op 与 ed 这两段（各留 cfg.manual_window_margin 秒
       余量），盲窗完全不参与。只填了一半时另一半是关闭的，**不**回退到盲窗 ——
       否则语义会变成「手填窗与盲窗的并集」，比两者单独用都糟（saijo/E01 就是
       这个形态：这集真的没有 OP，只有 ED 可填，此时片头不该再对激进规则开门）。
       这条路不依赖 duration：区间是绝对时间。

    2. **盲窗模式**（两个区间都没有）：片头 0-300s 或片尾最后 80s，与本函数
       历史行为逐点等价。用的是 credit_head_window / ed_keyword_window_seconds
       两个独立旋钮，跟聚簇用的 op_search_* / ed_cluster_tail_seconds 不共享
       数值：片尾窗口刻意比 ed_cluster_tail_seconds 窄，理由见 CreditsConfig。

       两个窗的实际大小由 credit_window_max_ratio 兜底收缩，保证并集永远覆盖不满
       整条时间轴 —— 否则短 track（以及 duration=0 的空字幕）上每一行都会拿到
       in_credit_window=True，本文件头部记录的「演出来」被「演出」命中那类事故就会
       从「只在片头片尾发生」升级成「全片发生」。

    手填模式**刻意不**受 credit_window_max_ratio 约束：那条约束是给盲窗兜底用的，
    而手填区间的覆盖范围是用户的显式声明，静默把它收缩成跟声明不一样的东西比
    覆盖过宽更难查。区间覆盖过宽这件事该由 ingest 的告警去说，不是在这里悄悄改。
    """
    if op is not None or ed is not None:
        margin = cfg.manual_window_margin
        return any(
            window[0] - margin <= start <= window[1] + margin
            for window in (op, ed)
            if window is not None
        )
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

    `lines` 走两遍，但两遍的判据不同、不是重复劳动：这里筛的是 `kind == "credits"`，
    `_op_from_silence` 那边筛的是 `is_spoken`（而且只在主路失败时才走）。
    """
    clusters = merge_intervals(
        ((ln.start, ln.end) for ln in lines if ln.kind == "credits"),
        max_gap=cfg.cluster_max_gap,
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
