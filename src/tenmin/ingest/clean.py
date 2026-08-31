"""字幕清洗。文本层（本任务）+ 结构层（下一任务）。"""

from __future__ import annotations

import re
from functools import lru_cache

_ASS_OVERRIDE = re.compile(r"\{[^{}]*\}")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_INLINE_SPACE = re.compile(r"[ \t\u3000]+")


@lru_cache(maxsize=1)
def _converter():
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


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    """按 key 长度倒序替换，长词优先，避免短词先吃掉长词的一部分。"""
    if not glossary:
        return text
    for key in sorted(glossary, key=len, reverse=True):
        if key:
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
    return "\n".join(line for line in text.split("\n")).strip("\n").strip()


_PAREN_PREFIX = re.compile(r"^[（(]\s*(?P<inner>[^）)]{1,40})\s*[）)]\s*")
_TITLE_CARD = re.compile(r"第[一二三四五六七八九十百\d]+[集話话]")
_NOISE = re.compile(r"^[\s\-–—•·%0-9]*$")
_DIGIT_RUN = re.compile(r"\d{2,}")
_LATIN_FRAGMENT = re.compile(r"(?<![A-Za-z])[A-Za-z]{1,4}(?![A-Za-z])")
_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
_ODD_SYMBOLS = frozenset("%#~`^*_|\\")
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
        if segment and len(segment) <= 4 and not _CJK.search(segment):
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
    has_later_prefix = any(
        _PAREN_PREFIX.match(part) and extract_prefix(part)[0] is not None
        for part in segments[1:]
    )
    if not has_later_prefix:
        return [text]
    return segments
