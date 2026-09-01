"""staff/credits 行识别与 OP/ED 区间推断。"""

from __future__ import annotations

import re

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
_TITLE_CARD = re.compile(r"第\s*[一二三四五六七八九十百\d]+\s*[集話话]")
_NAME_LIST_EVEN = re.compile(r"^(?:[\u4e00-\u9fff]{2,4})(?:\s+[\u4e00-\u9fff]{2,4})+$")
_NAME_LIST_RAGGED = re.compile(r"^(?:[\u4e00-\u9fff]{1,5})(?:\s+[\u4e00-\u9fff]{1,5}){2,}$")
_BRACKET_WRAPPED = re.compile(r"^[《『「(（]\s*(?P<inner>.+?)\s*[》』」)）]$")
_LATIN = re.compile(r"[A-Za-z]")
_NON_SPACE = re.compile(r"\S")

# 下界 30 秒：实测最早的 OP 起点是《恶女》第 2 集的 53.554 秒，取 30 留足余量。
# 原来的 60 秒会把它挡在窗外，OP 就退化成「最长的演出高光」。
OP_START_WINDOW = (30.0, 300.0)
OP_SPAN_WINDOW = (40.0, 120.0)
ED_TAIL_SECONDS = 120.0
ED_WINDOW_SECONDS = 80.0
CLUSTER_MAX_GAP = 35.0

# 静区兜底的 OP 时长区间。实测 OP 静默长度 90.7-94.5 秒；
# 下界 60 是为了不把 20 秒级的演出静场误判成 OP，
# 上界 120 是为了不把整段无对白的过场误判成 OP。
OP_MIN_SILENT_SPAN = 60.0
OP_MAX_SILENT_SPAN = 120.0

_SPEECH_KINDS = ("dialogue", "monologue")
_TITLE_OVERLAP_THRESHOLD = 0.6
_NAME_LIST_MIN_CJK = 6
# 段数够多时放宽字数门槛：「慧 诹 访 郎」只有 4 个 CJK 字符，
# 但切成 4 段本身就是 staff 罗列的形态，不可能是台词。
_NAME_LIST_MANY_SEGMENTS = 4
_NAME_LIST_MANY_MIN_CJK = 4
_LATIN_RATIO_THRESHOLD = 0.6
_LATIN_MIN_LEN = 6


def _title_overlap(text: str, show_title: str) -> float:
    title_chars = set(show_title) - set(" 　")
    if not title_chars:
        return 0.0
    return len(title_chars & set(text)) / len(title_chars)


def is_credits(text: str, *, show_title: str = "", in_credit_window: bool = False) -> bool:
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
    if _TITLE_CARD.search(stripped) and len(stripped) <= 24:
        return True

    # 3. 被书名号/引号包裹且与剧名字符高度重合
    wrapped = _BRACKET_WRAPPED.match(stripped)
    if wrapped and show_title:
        if _title_overlap(wrapped.group("inner"), show_title) >= _TITLE_OVERLAP_THRESHOLD:
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
            _NAME_LIST_MANY_MIN_CJK
            if segment_count >= _NAME_LIST_MANY_SEGMENTS
            else _NAME_LIST_MIN_CJK
        )
        if cjk_count >= min_cjk:
            return True

    # 5. 拉丁字母为主
    non_space = _NON_SPACE.findall(stripped)
    if len(non_space) >= _LATIN_MIN_LEN:
        latin_ratio = len(_LATIN.findall(stripped)) / len(non_space)
        if latin_ratio > _LATIN_RATIO_THRESHOLD:
            return True

    return False


def in_credit_window(start: float, duration: float) -> bool:
    """片头 0-300s 或片尾最后 80s。给 is_credits 的规则 4-5 开门。

    片尾窗口刻意比 ED_TAIL_SECONDS 窄：黄金样本最后一句真台词在 1325.5s
    （片长 1416.6s，距片尾 91s），ED staff 第一行在 1348.2s。80s 的阈值
    落在两者之间的 19.8s 无字幕间隙里，两侧各留约 11s 余量。
    """
    return start <= OP_START_WINDOW[1] or start >= duration - ED_WINDOW_SECONDS


def _cluster(lines: list[DialogueLine], max_gap: float) -> list[tuple[float, float]]:
    ordered = sorted(lines, key=lambda ln: ln.start)
    clusters: list[tuple[float, float]] = []
    for ln in ordered:
        if clusters and ln.start - clusters[-1][1] <= max_gap:
            begin, finish = clusters[-1]
            clusters[-1] = (begin, max(finish, ln.end))
        else:
            clusters.append((ln.start, ln.end))
    return clusters


def _silent_gaps_in_window(
    lines: list[DialogueLine], window: tuple[float, float]
) -> list[tuple[float, float]]:
    """列出起点落在 window 内的全部无字幕静默间隙，不做任何时长筛选。

    静默的定义与 signals/gaps.py 严格对齐：只有 kind 为 dialogue / monologue
    且有正文的行算「有人说话」，credits / noise / screen_text 都不打断静默；
    游标用 max(cursor, line.end) 单调推进，避免重叠字幕算出负长度间隙。
    这里刻意重算而不复用 signals，因为 ingest 层不能反向依赖 signals 层。
    """
    spoken = sorted(
        (ln for ln in lines if ln.kind in _SPEECH_KINDS and ln.text),
        key=lambda ln: ln.start,
    )
    gaps: list[tuple[float, float]] = []
    if not spoken:
        return gaps

    cursor = spoken[0].end
    for line in spoken[1:]:
        if line.start > cursor and window[0] <= cursor <= window[1]:
            gaps.append((cursor, line.start))
        cursor = max(cursor, line.end)
    return gaps


def _op_from_silence(lines: list[DialogueLine]) -> tuple[float, float] | None:
    """credits 聚簇算不出 OP 时的兜底。

    「没字幕的 OP」（字幕组一条 staff 行都没打）在聚簇法下必然漏判，
    但它整段就是一个 90 秒级的静默区，反过来比有字幕的 OP 更好认。

    必须「先按时长过滤、再取最长」，顺序不能反。若先取窗内最长再验时长，
    某集片头附近只要存在一个 >OP_MAX_SILENT_SPAN 的非-OP 静区
    （例如整段无对白的长过场），它就会挤掉真正的 90 秒 OP 候选，
    使兜底直接返回 None——本该救回来的 OP 反而丢了。
    """
    candidates = [
        gap
        for gap in _silent_gaps_in_window(lines, OP_START_WINDOW)
        if OP_MIN_SILENT_SPAN <= gap[1] - gap[0] <= OP_MAX_SILENT_SPAN
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda gap: gap[1] - gap[0])


def find_credit_ranges(
    lines: list[DialogueLine], duration: float, max_gap: float = CLUSTER_MAX_GAP
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """从已标记 kind=="credits" 的行推断 OP / ED 区间。

    OP 走两条路：credits 聚簇为主路，算不出来时退到静区兜底。
    ED 只走聚簇，不做兜底——片尾前的长静场（定格收尾）是真高光，兜底会吃掉它。
    """
    credit_lines = [ln for ln in lines if ln.kind == "credits"]
    clusters = _cluster(credit_lines, max_gap)

    op_candidates = [
        c
        for c in clusters
        if OP_START_WINDOW[0] <= c[0] <= OP_START_WINDOW[1]
        and OP_SPAN_WINDOW[0] <= c[1] - c[0] <= OP_SPAN_WINDOW[1]
    ]
    op = max(op_candidates, key=lambda c: c[1] - c[0]) if op_candidates else None
    if op is None:
        op = _op_from_silence(lines)

    ed_candidates = [c for c in clusters if c[0] >= duration - ED_TAIL_SECONDS]
    ed = min(ed_candidates, key=lambda c: c[0]) if ed_candidates else None

    return op, ed
