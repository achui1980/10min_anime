"""staff/credits 行识别与 OP/ED 区间推断。"""

from __future__ import annotations

import re

from tenmin.models import DialogueLine

_KEYWORDS = (
    "製作",
    "制作",
    "製作委員会",
    "製作委員會",
    "制作委员会",
    "作詞",
    "作词",
    "作曲",
    "編曲",
    "编曲",
    "協力",
    "协力",
    "フォント",
    "STAFF",
    "Studio",
    "作画",
    "監督",
    "监督",
    "脚本",
    "演出",
    "原作",
    "主題歌",
    "主题歌",
)
_TITLE_CARD = re.compile(r"第\s*[一二三四五六七八九十百\d]+\s*[集話话]")
_NAME_LIST = re.compile(r"^(?:[\u4e00-\u9fff]{2,4})(?:\s+[\u4e00-\u9fff]{2,4})+$")
_BRACKET_WRAPPED = re.compile(r"^[《『「(（]\s*(?P<inner>.+?)\s*[》』」)）]$")
_LATIN = re.compile(r"[A-Za-z]")
_NON_SPACE = re.compile(r"\S")

OP_START_WINDOW = (60.0, 300.0)
OP_SPAN_WINDOW = (40.0, 120.0)
ED_TAIL_SECONDS = 120.0
CLUSTER_MAX_GAP = 30.0
_TITLE_OVERLAP_THRESHOLD = 0.6
_NAME_LIST_MIN_CJK = 6
_LATIN_RATIO_THRESHOLD = 0.6
_LATIN_MIN_LEN = 6


def _title_overlap(text: str, show_title: str) -> float:
    title_chars = set(show_title) - set(" 　")
    if not title_chars:
        return 0.0
    return len(title_chars & set(text)) / len(title_chars)


def is_credits(text: str, *, show_title: str = "", in_credit_window: bool = False) -> bool:
    """判断一行清洗后的文本是不是 staff / 版权 / 标题卡。

    规则 1-3 与规则 6 无条件生效；规则 4-5 只在片头片尾时间窗内生效，
    否则 `早安 早安` 这类空格分隔的正常台词会被纯人名正则误伤。
    """
    stripped = text.strip()
    if not stripped:
        return False

    # 1. 版权标记
    if "©" in stripped or "(C)" in stripped.upper():
        return True

    # 2. staff 关键词（繁简与日文都列）
    upper = stripped.upper()
    for keyword in _KEYWORDS:
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

    # 4. 纯人名罗列
    if _NAME_LIST.match(stripped):
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", stripped))
        if cjk_count >= _NAME_LIST_MIN_CJK:
            return True

    # 5. 拉丁字母为主
    non_space = _NON_SPACE.findall(stripped)
    if len(non_space) >= _LATIN_MIN_LEN:
        latin_ratio = len(_LATIN.findall(stripped)) / len(non_space)
        if latin_ratio > _LATIN_RATIO_THRESHOLD:
            return True

    return False


def in_credit_window(start: float, duration: float) -> bool:
    """片头 0-300s 或片尾最后 150s。给 is_credits 的规则 4-5 开门。"""
    return start <= OP_START_WINDOW[1] or start >= duration - 150.0


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


def find_credit_ranges(
    lines: list[DialogueLine], duration: float, max_gap: float = CLUSTER_MAX_GAP
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """从已标记 kind=="credits" 的行推断 OP / ED 区间。"""
    credit_lines = [ln for ln in lines if ln.kind == "credits"]
    clusters = _cluster(credit_lines, max_gap)

    op_candidates = [
        c
        for c in clusters
        if OP_START_WINDOW[0] <= c[0] <= OP_START_WINDOW[1]
        and OP_SPAN_WINDOW[0] <= c[1] - c[0] <= OP_SPAN_WINDOW[1]
    ]
    op = max(op_candidates, key=lambda c: c[1] - c[0]) if op_candidates else None

    ed_candidates = [c for c in clusters if c[0] >= duration - ED_TAIL_SECONDS]
    ed = min(ed_candidates, key=lambda c: c[0]) if ed_candidates else None

    return op, ed
