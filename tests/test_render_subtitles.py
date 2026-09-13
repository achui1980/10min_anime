from tenmin.models import SubtitleCue
from tenmin.render.subtitles import (
    DEFAULT_FONT_SIZE,
    escape_text,
    format_ass_time,
    max_chars_per_line,
    render_ass,
    wrap_text,
)

EXPECTED = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, \
BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, \
BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Narration,Lantinghei SC,52,&H0000FFFF,&H000000FF,&H00000000,\
&H80000000,1,0,0,0,100,100,3,0,1,4,1,2,60,60,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:08.00,Narration,,0,0,0,,第一句。
Dialogue: 0,0:00:10.00,0:00:20.00,Narration,,0,0,0,,第二句。
"""


def test_format_ass_time_at_zero():
    assert format_ass_time(0.0) == "0:00:00.00"


def test_format_ass_time_centiseconds():
    assert format_ass_time(3.5) == "0:00:03.50"


def test_format_ass_time_rounds_to_centiseconds():
    assert format_ass_time(3661.239) == "1:01:01.24"


def test_format_ass_time_clamps_negative():
    assert format_ass_time(-1.0) == "0:00:00.00"


def test_escape_text_converts_newline():
    assert escape_text("上一行\n下一行") == "上一行\\N下一行"


def test_escape_text_strips_braces():
    # ASS 用花括号写覆盖标签，正文里的花括号必须去掉
    assert escape_text("这里{有}花括号") == "这里有花括号"


def test_render_ass_matches_expected_output():
    cues = [
        SubtitleCue(start=0.0, end=8.0, text="第一句。"),
        SubtitleCue(start=10.0, end=20.0, text="第二句。"),
    ]
    assert render_ass(cues) == EXPECTED


def test_render_ass_honours_font_size():
    out = render_ass([SubtitleCue(start=0.0, end=1.0, text="喂")], font_size=64)
    assert "Style: Narration,Lantinghei SC,64," in out


def test_render_ass_honours_font_name():
    out = render_ass([SubtitleCue(start=0.0, end=1.0, text="喂")], font_name="PingFang SC")
    assert "Style: Narration,PingFang SC,52," in out


def test_render_ass_style_is_bold_with_yellow_fill_black_outline():
    # 二次元解说风格：黄字、加粗、粗黑色描边（参考 B 站/抖音吐槽解说常见配色）。
    out = render_ass([SubtitleCue(start=0.0, end=1.0, text="喂")])
    style_line = next(line for line in out.splitlines() if line.startswith("Style:"))
    fields = style_line.split(",")
    assert fields[3] == "&H0000FFFF"  # PrimaryColour（文字填充色，黄）
    assert fields[5] == "&H00000000"  # OutlineColour（描边色，黑）
    assert fields[7] == "1"  # Bold
    assert fields[16] == "4"  # Outline width


def test_render_ass_without_cues_still_has_headers():
    out = render_ass([])
    assert "[Events]" in out
    assert "Dialogue:" not in out


def test_max_chars_per_line_for_default_font_size():
    # 1920 - 2*60 margins = 1800px 可用宽度；48号字按 1.05 倍宽度估算。
    assert max_chars_per_line(48) == 35


def test_max_chars_per_line_shrinks_with_bigger_font():
    assert max_chars_per_line(96) < max_chars_per_line(48)


def test_wrap_text_leaves_short_line_untouched():
    assert wrap_text("第一句。", 35) == "第一句。"


def test_wrap_text_breaks_long_line_without_spaces():
    # libass 只在空格处自动换行，中文没有空格，所以必须手动断行插入 \n。
    text = "一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十"  # 30 字
    wrapped = wrap_text(text, 10)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= 10 for line in lines)
    assert "".join(lines) == text


def test_wrap_text_prefers_breaking_after_punctuation():
    text = "前半句内容刚好十个字，后面还有一些字"
    wrapped = wrap_text(text, 12)
    first_line = wrapped.split("\n")[0]
    assert first_line.endswith("，")


def test_wrap_text_preserves_existing_newlines():
    assert wrap_text("第一行\n第二行", 35) == "第一行\n第二行"


def test_render_ass_wraps_long_cue_into_multiple_lines():
    long_text = "下午的任务是去女厕取回大小姐落下的钱包，定位来自缝在钱包里的追踪器——这家人对女儿的管理精度已经到这地步了。"
    cues = [SubtitleCue(start=0.0, end=10.0, text=long_text)]
    out = render_ass(cues)
    dialogue_line = next(line for line in out.splitlines() if line.startswith("Dialogue"))
    assert "\\N" in dialogue_line
    # 每个物理行都不应超过按字号估算出的单行字符上限。
    text_part = dialogue_line.split(",", 9)[-1]
    for segment in text_part.split("\\N"):
        assert len(segment) <= max_chars_per_line(DEFAULT_FONT_SIZE)


# --- 反斜杠 = ASS 控制字符（P2-E C3）---------------------------------------
#
# escape_text 剥掉了 `{}`（覆盖标签的定界符）却放过了 `\`，而 ASS 的 `\N` / `\n`
# / `\h` 在**花括号之外的正文里**同样生效：一段含反斜杠的旁白就能自己插硬换行、
# 插硬空格，把断行算好的版式打乱。
# 判据跟 `{}` 完全一样，所以处置也一样 —— **剥掉**。刻意不转义成 `\\`：ASS 压根没有
# 反斜杠转义机制（`\\` 不是「一个字面反斜杠」，libass 只会把它当一个未知标签吃掉），
# 也不换成全角 `＼`（那是往正文里塞一个作者没写的字）。
# 实测 11 份真实 script.json 的 73 段 narration：反斜杠出现 **0 次**，不改现有产物。


def test_escape_text_strips_backslash():
    assert escape_text("这里\\N不该换行") == "这里N不该换行"


def test_escape_text_strips_backslash_before_converting_real_newlines():
    """剥反斜杠必须发生在「真换行 → \\N」之前，否则自己插的 \\N 会被自己吃掉。"""
    assert escape_text("上一行\n下\\h一行") == "上一行\\N下h一行"
