from tenmin.models import SubtitleCue
from tenmin.render.subtitles import escape_text, format_ass_time, render_ass

EXPECTED = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, \
BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, \
BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Narration,Source Han Sans SC,48,&H00FFFFFF,&H000000FF,&H00000000,\
&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,60,1

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
    assert "Style: Narration,Source Han Sans SC,64," in out


def test_render_ass_honours_font_name():
    out = render_ass([SubtitleCue(start=0.0, end=1.0, text="喂")], font_name="PingFang SC")
    assert "Style: Narration,PingFang SC,48," in out


def test_render_ass_without_cues_still_has_headers():
    out = render_ass([])
    assert "[Events]" in out
    assert "Dialogue:" not in out
