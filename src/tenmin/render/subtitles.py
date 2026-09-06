"""生成 ASS 字幕。全是纯函数，输出逐字符可测。

ASS 的时间是 H:MM:SS.cc（厘秒、小时不补零），跟 timecode.format_timestamp 不同，
所以这里单独写格式化函数。
"""

from __future__ import annotations

from tenmin.models import SubtitleCue

DEFAULT_FONT_NAME = "Source Han Sans SC"
DEFAULT_FONT_SIZE = 48
PLAY_RES_X = 1920
PLAY_RES_Y = 1080

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
# 白字黑边、底部居中、四周留 60px。Alignment 2 = 底部居中。
_STYLE_TAIL = "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,60,1"


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
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        _STYLE_FORMAT,
        f"Style: Narration,{font_name},{font_size},{_STYLE_TAIL}",
        "",
        "[Events]",
        _EVENT_FORMAT,
    ]
    for cue in cues:
        lines.append(
            f"Dialogue: 0,{format_ass_time(cue.start)},{format_ass_time(cue.end)},"
            f"Narration,,0,0,0,,{escape_text(cue.text)}"
        )
    return "\n".join(lines) + "\n"
