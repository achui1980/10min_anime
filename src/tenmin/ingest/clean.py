"""字幕清洗。文本层（本任务）+ 结构层（下一任务）。"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opencc import OpenCC

# 未闭合的花括号（`{\an8` 少了右括号）刻意不管：放宽成 `\{[^{}]*\}?` 会让一个孤零零的
# `{` 把它之后直到下一个花括号/文本末尾的内容全部吃掉，那比留下一段看得见的垃圾危险得多。
# 而且这种行必然含反斜杠，已经被 `_ODD_SYMBOLS` 判成 suspect、连着原文交给 LLM 判断。
# 实测 11 集真实素材里 21 个 `{` 全部成对闭合，0 处残留。
_ASS_OVERRIDE = re.compile(r"\{[^{}]*\}")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_INLINE_SPACE = re.compile(r"[ \t\u3000]+")


@lru_cache(maxsize=1)
def _converter() -> OpenCC:
    from opencc import OpenCC

    return OpenCC("t2s")


def strip_markup(text: str) -> str:
    """剥 ASS/SSA 覆盖块与 HTML 标签，逐行去首尾空白，行内连续空白折成单空格。"""
    text = _ASS_OVERRIDE.sub("", text)
    text = _HTML_TAG.sub("", text)
    lines = [_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines)


def to_simplified(text: str) -> str:
    if not text:
        return text
    return _converter().convert(text)


@lru_cache(maxsize=32)
def _glossary_keys(items: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """术语表的 key 按长度倒序（长词优先），空 key 剔掉。

    `normalize.build_track` 在 cue 循环里调 `apply_glossary`，原先每条 cue 都要重排
    一次。缓存键刻意是 `tuple(glossary.items())`（**保持插入顺序**）而不是
    `tuple(sorted(items))`：`sorted(key=len, reverse=True)` 是稳定排序，等长 key 的
    先后由插入顺序决定，而那个先后会改变结果（`{"BC":"y","AB":"x"}` 与
    `{"AB":"x","BC":"y"}` 作用在 `"ABC"` 上分别给出 `"Ay"` 与 `"xC"`）。
    按 sorted(items) 做键会把这两张表折成同一个缓存项，直接给出错误答案。
    """
    return tuple(key for key in sorted((k for k, _ in items), key=len, reverse=True) if key)


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    """按 key 长度倒序替换，长词优先，避免短词先吃掉长词的一部分。

    刻意**不**编译成 alternation 正则一次扫完。那个写法与「逐个 str.replace」并不等价，
    有两类反例（都有测试）：

    1. **连锁替换**：`{"A": "B", "B": "C"}` 作用在 `"A"` 上，顺序 replace 得到 `"C"`，
       而正则一次扫描只会给出 `"B"`。
    2. **等长 key 重叠**：`{"BC": "y", "AB": "x"}` 作用在 `"ABC"` 上，顺序 replace 按
       插入顺序先换 `"BC"` 得到 `"Ay"`，而正则的最左匹配先命中 `"AB"` 给出 `"xC"`。

    所以这里只把每条 cue 重复做的 key 排序缓存掉，替换本身照旧。
    """
    if not glossary:
        return text
    for key in _glossary_keys(tuple(glossary.items())):
        text = text.replace(key, glossary[key])
    return text


def clean_text(
    text: str, *, convert: bool = True, glossary: dict[str, str] | None = None
) -> str:
    """文本层清洗，顺序固定：剥标签 -> 繁转简 -> 术语表。保留行间换行。"""
    text = strip_markup(text)
    if convert:
        text = to_simplified(text)
    text = apply_glossary(text, glossary or {})
    return text.strip("\n").strip()


_PAREN_PREFIX = re.compile(r"^[（(]\s*(?P<inner>[^）)]{1,40})\s*[）)]\s*")
# 与 credits.py 共用同一份，见那边的 import。原先两处各写一份且不一致
# （credits 那份多了 `\s*`，能接 `第 3 集`），收敛成宽的那份 —— 它是严格超集，
# 而两个用法里「多认出一个标题卡」的方向都是安全的。
_TITLE_CARD = re.compile(r"第\s*[一二三四五六七八九十百\d]+\s*[集話话]")
_NOISE = re.compile(r"^[\s\-–—•·%0-9]*$")
_DIGIT_RUN = re.compile(r"\d{2,}")

# 「短到没有语义」的长度上限。两个用法共享它：孤立的拉丁字母串（OCR 把汉字识别成
# 几个字母的典型残留），以及整段不含 CJK 的短片段。
_FRAGMENT_MAX_LEN = 4
_LATIN_FRAGMENT = re.compile(
    rf"(?<![A-Za-z])[A-Za-z]{{1,{_FRAGMENT_MAX_LEN}}}(?![A-Za-z])"
)
_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
_ODD_SYMBOLS = frozenset("%#~`^*_|\\")
# 括号里的说话人名最长多少字。**不是**「多短算碎片」的同类项：它是名字的长度上界，
# 方向与 _FRAGMENT_MAX_LEN 相反，取值也不同，别合并。
_SPEAKER_MAX_LEN = 8


def extract_prefix(text: str) -> tuple[str | None, str]:
    """处理行首圆括号。

    返回 (speaker, 剩余文本)。三种情况：
    - `(伊月) 我知道了` -> ("伊月", "我知道了")            短前缀当说话人
    - `(第二集…) 早上好` -> (None, "早上好")               标题卡/长注释剥离，不设说话人
    - `(制作委员会)`      -> (None, "(制作委员会)")         整行都在括号里，原样返回给 credits 判定
    """
    match = _PAREN_PREFIX.match(text)
    if not match:
        return None, text
    inner = match.group("inner").strip()
    remainder = text[match.end() :].strip()
    if not remainder:
        return None, text
    if len(inner) <= _SPEAKER_MAX_LEN and not _TITLE_CARD.search(inner):
        return inner, remainder
    return None, remainder


def is_noise(text: str) -> bool:
    """纯符号/纯数字/空行 —— 没有任何语义内容。"""
    return bool(_NOISE.match(text))


def is_suspect(text: str) -> bool:
    """疑似 OCR 混入或残缺。刻意宽松：只打标不删除，交给有全局上下文的 LLM 判断。

    必须在换行折叠成空格之前调用。
    """
    if _DIGIT_RUN.search(text):
        return True
    if any(ch in _ODD_SYMBOLS for ch in text):
        return True
    if text.count('"') % 2 == 1:
        return True
    for segment in (part.strip() for part in text.split("\n")):
        if segment and len(segment) <= _FRAGMENT_MAX_LEN and not _CJK.search(segment):
            return True
    if _CJK.search(text) and _LATIN_FRAGMENT.search(text):
        return True
    return False


def split_dual_track(text: str) -> list[str]:
    """台词与内心独白叠加在同一时间区间时，按带 `(角色名)` 前缀的段落拆开。

    只在存在第二段及以后带说话人前缀时才拆；否则整块原样返回（普通折行台词）。
    """
    segments = [part.strip() for part in text.split("\n")]
    segments = [part for part in segments if part]
    if len(segments) < 2:
        return [text]
    # 只问 extract_prefix 一次。原先写的是
    # `_PAREN_PREFIX.match(part) and extract_prefix(part)[0] is not None`，
    # 左边那半是多余的（extract_prefix 匹配不上时必然返回 `(None, text)`），
    # 而它还让同一个正则对每段跑两遍（自己一遍，extract_prefix 里面又一遍）。
    has_later_prefix = any(
        extract_prefix(part)[0] is not None for part in segments[1:]
    )
    if not has_later_prefix:
        return [text]
    return segments
