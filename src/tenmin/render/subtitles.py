"""生成 ASS 字幕。全是纯函数，输出逐字符可测。

ASS 的时间是 H:MM:SS.cc（厘秒、小时不补零），跟 timecode.format_timestamp 不同，
所以这里单独写格式化函数。
"""

from __future__ import annotations

from tenmin.models import SubtitleCue

DEFAULT_FONT_NAME = "Lantinghei SC"
DEFAULT_FONT_SIZE = 52
PLAY_RES_X = 1920
PLAY_RES_Y = 1080
MARGIN_LR = 60
# 全角字符的实际显示宽度近似等于字号本身；用 1.05 留一点余量，
# 免得断行算准了但描边（Outline）一挤又超出画面。
CJK_CHAR_WIDTH_RATIO = 1.05

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
# 黄字黑边、加粗、底部居中、四周留 60px。Alignment 2 = 底部居中。
# 参考 B 站/抖音吐槽解说类视频最常见的配色：黄色文字 + 黑色粗描边，
# 对比度强，任何背景下都清楚，比白字彩边更贴近"二次元解说"的既有印象。
# Spacing 加到 3，字号偏大时给字间留点缝，不然描边一粗字就糊成一片。
# 字号用 52（比 48 大一点，48/Outline3 已经清楚但想再对比一版更粗更大的），
# Outline 用 4：配合更大的字号，描边稍粗一点仍保持轮廓清晰。
_STYLE_TAIL = "&H0000FFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,3,0,1,4,1,2,60,60,60,1"
# 优先在这些字符之后断行（标点收尾，不会把标点甩到下一行开头）。
_BREAK_AFTER = "，、。！？；：—…”』」）,.!?;:"


def format_ass_time(seconds: float) -> str:
    """秒 → H:MM:SS.cc。负数按 0 处理。"""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    hours, rem = divmod(total_cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def escape_text(text: str) -> str:
    """ASS 正文转义：换行用 \\N，花括号是覆盖标签的定界符必须去掉。"""
    cleaned = text.replace("{", "").replace("}", "")
    return cleaned.replace("\r\n", "\n").replace("\n", "\\N").strip()


def max_chars_per_line(font_size: int) -> int:
    """给定字号，估算一行能塞下多少个全角字符。

    libass 的自动换行（WrapStyle）只在空格处断行，中文没有空格，
    长句会被当成一个不可断的“单词”直接冲出画面。所以断行必须自己算好、
    手动插 \\N，不能指望 libass 帮忙。
    """
    usable_width = PLAY_RES_X - 2 * MARGIN_LR
    if font_size <= 0:
        return usable_width
    chars = int(usable_width // (font_size * CJK_CHAR_WIDTH_RATIO))
    return max(chars, 1)


def _wrap_single_line(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    pieces: list[str] = []
    remaining = line
    while len(remaining) > max_chars:
        window_start = max(1, int(max_chars * 0.6))
        break_at = max_chars
        for i in range(max_chars, window_start, -1):
            if remaining[i - 1] in _BREAK_AFTER:
                break_at = i
                break
        pieces.append(remaining[:break_at])
        remaining = remaining[break_at:]
    if remaining:
        pieces.append(remaining)
    return pieces


def wrap_text(text: str, max_chars: int) -> str:
    """按字符数手动断行；已有的换行原样保留，只处理其中过长的单行。"""
    if max_chars <= 0:
        return text
    result: list[str] = []
    for raw_line in text.split("\n"):
        result.extend(_wrap_single_line(raw_line, max_chars))
    return "\n".join(result)


def render_ass(
    cues: list[SubtitleCue],
    *,
    font_size: int = DEFAULT_FONT_SIZE,
    font_name: str = DEFAULT_FONT_NAME,
) -> str:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {PLAY_RES_X}",
        f"PlayResY: {PLAY_RES_Y}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        _STYLE_FORMAT,
        f"Style: Narration,{font_name},{font_size},{_STYLE_TAIL}",
        "",
        "[Events]",
        _EVENT_FORMAT,
    ]
    max_chars = max_chars_per_line(font_size)
    for cue in cues:
        wrapped = wrap_text(cue.text, max_chars)
        lines.append(
            f"Dialogue: 0,{format_ass_time(cue.start)},{format_ass_time(cue.end)},"
            f"Narration,,0,0,0,,{escape_text(wrapped)}"
        )
    return "\n".join(lines) + "\n"
