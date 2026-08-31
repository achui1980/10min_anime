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
