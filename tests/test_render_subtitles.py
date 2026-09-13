from tenmin.models import SubtitleCue
from tenmin.render.subtitles import (
    DEFAULT_FONT_SIZE,
    display_cells,
    escape_text,
    format_ass_time,
    max_cells_per_line,
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


def test_max_cells_per_line_for_default_font_size():
    # 1920 - 2*60 margins = 1800px 可用宽度；48 号字一格按 1.05×48/2 = 25.2px 估算
    # → 71 格（口径从「字符数」换成「格数」之后，取整发生在更细的粒度上，
    # 所以是 71 而不是「35 个全角字 ×2」的 70）。
    assert max_cells_per_line(48) == 71


def test_max_cells_per_line_shrinks_with_bigger_font():
    assert max_cells_per_line(96) < max_cells_per_line(48)


def test_wrap_text_leaves_short_line_untouched():
    assert wrap_text("第一句。", 70) == "第一句。"


def test_wrap_text_breaks_long_line_without_spaces():
    # libass 只在空格处自动换行，中文没有空格，所以必须手动断行插入 \n。
    text = "一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十"  # 30 字
    wrapped = wrap_text(text, 20)  # 20 格 = 10 个全角字
    lines = wrapped.split("\n")
    assert len(lines) > 1
    assert all(display_cells(line) <= 20 for line in lines)
    assert "".join(lines) == text


def test_wrap_text_prefers_breaking_after_punctuation():
    text = "前半句内容刚好十个字，后面还有一些字"
    wrapped = wrap_text(text, 24)  # 24 格 = 12 个全角字
    first_line = wrapped.split("\n")[0]
    assert first_line.endswith("，")


def test_wrap_text_preserves_existing_newlines():
    assert wrap_text("第一行\n第二行", 70) == "第一行\n第二行"


def test_render_ass_wraps_long_cue_into_multiple_lines():
    long_text = (
        "下午的任务是去女厕取回大小姐落下的钱包，"
        "定位来自缝在钱包里的追踪器——这家人对女儿的管理精度已经到这地步了。"
    )
    cues = [SubtitleCue(start=0.0, end=10.0, text=long_text)]
    out = render_ass(cues)
    dialogue_line = next(line for line in out.splitlines() if line.startswith("Dialogue"))
    assert "\\N" in dialogue_line
    # 每个物理行都不应超过按字号估算出的单行格数上限。
    text_part = dialogue_line.split(",", 9)[-1]
    for segment in text_part.split("\\N"):
        assert display_cells(segment) <= max_cells_per_line(DEFAULT_FONT_SIZE)


# --- 反斜杠 = ASS 控制字符 -------------------------------------------------
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


# --- 边距只能有一份真相 ----------------------------------------------------
#
# MARGIN_LR = 60 决定断行宽度，而 Style 行原来硬编码着第二份 `60,60,60`
# （MarginL/MarginR/MarginV）决定 libass 实际留的边距。两份字面量分开写，改一份不改
# 另一份就会「按 60px 算断行、按别的值排版」——算出来的行宽从此对不上画面。


def test_style_line_margins_come_from_the_module_constants():
    from tenmin.render.subtitles import MARGIN_LR, MARGIN_V

    style = next(line for line in render_ass([]).splitlines() if line.startswith("Style:"))
    fields = style.split(",")
    assert fields[-4:-1] == [str(MARGIN_LR), str(MARGIN_LR), str(MARGIN_V)]


def test_changing_margin_lr_moves_both_the_wrap_width_and_the_style_line(monkeypatch):
    """改一个常量，断行宽度与 Style 行必须一起动。"""
    from tenmin.render import subtitles as module

    before = module.max_cells_per_line(DEFAULT_FONT_SIZE)
    monkeypatch.setattr(module, "MARGIN_LR", 300)
    style = next(line for line in module.render_ass([]).splitlines() if line.startswith("Style:"))
    assert style.split(",")[-4:-2] == ["300", "300"]
    assert module.max_cells_per_line(DEFAULT_FONT_SIZE) < before


# --- 行宽按「格」量，不按字符数 --------------------------------------------
#
# 原来断行宽度用 len() 度量，把半角字符当成跟汉字一样宽，于是混了拉丁字母/数字的
# 行被提前折断（保守，从不超宽，但白扔掉可用宽度）。
#
# 分档判据用 unicodedata.east_asian_width，**Ambiguous（A）算宽**。这一条是真
# libass 实测定下来的，不是照搬 wcwidth 的通俗版本（W/F 算 2、其余算 1）：
# Lantinghei SC / 字号 52 下逐帧量墨迹包围盒，`—`(A) 47.2px、`…`(A) 47.3px、
# `·`(A) 47.3px，跟 `字`(W) 的 47.3px 一模一样 —— 在 CJK 字体里 A 就是全角。
# 而语料（10 集 10338 字）里非 W/F 的字符只有 9 个：`—`×196、`'`×42、`…`×6、
# `·`×1、`B`/`I`/`T`/`O`/`K` 各 1。也就是说「A 算窄」会把 203/250 个非 W/F 字符
# 全部低估一半 —— 一行 64 个 `—` 会被判成刚好放得下，实测却是 3021px，超出可用
# 宽度 1800px 的 68%。


def test_display_cells_counts_cjk_as_two():
    from tenmin.render.subtitles import display_cells

    assert display_cells("汉字") == 4


def test_display_cells_counts_ascii_as_one():
    from tenmin.render.subtitles import display_cells

    assert display_cells("ab1") == 3


def test_display_cells_counts_ambiguous_width_as_wide():
    """`—` / `…` / `·` 的 EAW 是 A，但在 CJK 字体里实测就是全角（47.3px）。"""
    from tenmin.render.subtitles import display_cells

    assert display_cells("——") == 4
    assert display_cells("…·") == 4


def test_max_cells_per_line_keeps_the_pure_cjk_capacity_unchanged():
    """预算翻倍、度量也翻倍，所以纯 CJK 一行还是装同样多个字。

    取整在更细的粒度上做，预算不是整整两倍（52 号字 32 → 65 而不是 64），但
    `floor(2x) // 2 == floor(x)` 恒成立，所以**全角字的容量逐点不变** —— 这正是
    「10 集 .ass 只有 2 条 cue 变化」的原因。
    """
    from tenmin.render.subtitles import max_cells_per_line

    assert max_cells_per_line(52) == 65
    assert max_cells_per_line(52) // 2 == 32
    assert max_cells_per_line(48) // 2 == 35


def test_wrap_text_fits_more_ascii_on_one_line_than_before():
    """32 个汉字与 64 个半角字符都刚好占满一行。"""
    assert wrap_text("a" * 64, 64) == "a" * 64
    assert wrap_text("字" * 32, 65) == "字" * 32
    assert wrap_text("字" * 33, 65).count("\n") == 1


def test_render_ass_never_exceeds_the_usable_width():
    from tenmin.render.subtitles import (
        MARGIN_LR,
        PLAY_RES_X,
        display_cells,
        max_cells_per_line,
    )

    long_text = "他把伪装一个一个拆掉：不是IT企业的继承人，是此花家的佣人；每一天都是他自己的选择。"
    out = render_ass([SubtitleCue(start=0.0, end=10.0, text=long_text)])
    dialogue = next(line for line in out.splitlines() if line.startswith("Dialogue"))
    budget = max_cells_per_line(DEFAULT_FONT_SIZE)
    for segment in dialogue.split(",", 9)[-1].split("\\N"):
        assert display_cells(segment) <= budget
    # 预算换算回像素也必须落在可用宽度里
    assert budget * DEFAULT_FONT_SIZE * 1.05 / 2 <= PLAY_RES_X - 2 * MARGIN_LR


# --- 行首标点回收与回溯窗口的边界 ------------------------------------------
#
# 断行原来只从 max_cells 往回找到 60% 处，找不到标点就硬切。两个后果实测出来了
# （10 集 139 次断行）：
#
# 1. **行首标点**：5 行以标点开头，其中 4 行**整行只有一个 `。`** ——
#    `…工作内容是当一个高中女生的贴身侍从` ⏎ `。` 这种。CJK 排版的禁则里
#    「句读不可置于行首」是最基本的一条，而这里的成本是把它挤进上一行（追い込み）。
# 刻意**不**处理的两件事（都有实测依据，见 _wrap_single_line / 本次报告）：
#
# - 硬切切在 CJK 词中间（`完` ⏎ `全不是一个人`）。判准需要分词器 = 新依赖。
# - 把回溯窗口的下界改成**含** window_start 本身。它能把硬切从 11 次降到 8 次，
#   代价是多一条 3 行的 cue（8 → 9）与多一个物理行（498 → 499）—— 3 行字幕挡掉的
#   画面比切在词中间难看得多，这笔换不值。
# - 切在拉丁词内部或数字与单位之间：实测 10 集 139 次断行 **0 次**发生（语料 97.6%
#   是全角字符，非 W/F 的字符只有 9 个、其中 5 个各出现 1 次）。没有可修的东西。


def test_wrap_text_pulls_a_leading_punctuation_back_to_the_previous_line():
    """下一行开头是标点时把它挤回上一行，宁可超一格也不让 `。` 独占行首。"""
    text = "一二三四五六七八九十。后面还有内容"
    # 20 格 = 10 个全角字：硬切正好落在 `。` 之前
    assert wrap_text(text, 20).split("\n")[0] == "一二三四五六七八九十。"


def test_wrap_text_pulls_a_whole_dash_pair_back():
    """`——` 是两个字符，只挤一个会在下一行开头留一个孤立的 `—`。"""
    text = "一二三四五六七八九十——后面还有内容"
    assert wrap_text(text, 20).split("\n")[0] == "一二三四五六七八九十——"


def test_wrap_text_does_not_hang_more_than_the_measured_headroom():
    """挂出去的格数有上限：语料里最长的行首标点串是 `——`（2 个字符 / 4 格）。"""
    text = "一二三四五六七八九十。。。。。后面还有内容"
    first = wrap_text(text, 20).split("\n")[0]
    assert first == "一二三四五六七八九十。。"


def test_wrap_text_backtrack_window_still_excludes_its_lower_bound():
    """回溯窗口的下界仍然是开区间 —— 这是实测选择，不是漏掉的差一错。

    改成闭区间能少 3 次硬切，但会多出一条 3 行的 cue，得不偿失（数据见上面的注释）。
    """
    # 20 格 = 10 个全角字，60% → window_start = 6；标点正好在第 6 个字符之后，
    # 落在窗口之外，所以照旧硬切在第 10 个字符处。
    text = "一二三四五，六七八九十一二三四五"
    assert wrap_text(text, 20).split("\n")[0] == "一二三四五，六七八九"


# --- 可读性检查：行数上限与最短显示时长 ------------------------------------
#
# 每条 cue 原来既无行数上限也无最短显示时长。实测（10 集、修完 A1/C1/C5 之后的
# 360 条 cue）：
#
#   行数分布 {1 行: 233, 2 行: 119, 3 行: 8}
#   时长 min 0.80s / p50 5.77s / p90 9.60s / max 15.36s；<0.7s 的 **0 条**
#
# 那 8 条 3 行的 cue 是 54–79 字的**单句**（10.6–15.4 秒），也就是说它们跟 chunk 粒度
# 无关（A5 的「每句一 chunk」压根救不了它们），唯一的修法是把句子写短。
#
# **力度选 warning，不改产物**，依据：
# - 拆长 cue 需要句内的时间切点，而句内时间只能靠字数比例估（B1 实测这个估算在句
#   边界上的误差 p50 0.365 秒、max 1.615 秒）。拆出来的下半句有可能在念到之前就
#   出现或念完之后才出现 —— 用一个已知有 1.6 秒误差的估算去修「字幕太长」，换来的
#   是「字幕对不上口型」。而 warning 指向的动作（把这句写短）同时修好字幕与配音节奏。
# - 合并过短的 cue 在真实数据上 0 次触发，写了也是没被跑过的代码。
#
# 两个阈值都是「这部番想要什么」的创作旋钮（判据见 AGENTS.md），所以进 RenderConfig；
# 设成 0 就是关掉这一项检查。


def _cue(start: float, end: float, text: str) -> SubtitleCue:
    return SubtitleCue(start=start, end=end, text=text)


def test_check_cue_legibility_is_silent_on_a_normal_cue():
    from tenmin.render.subtitles import check_cue_legibility

    assert check_cue_legibility([_cue(0.0, 5.0, "第一句。")]) == []


def test_check_cue_legibility_warns_when_a_cue_needs_three_lines():
    from tenmin.render.subtitles import check_cue_legibility

    warnings = check_cue_legibility([_cue(0.0, 15.0, "一二三四五六七八九十" * 8)])
    assert len(warnings) == 1
    assert "3 行" in warnings[0]
    assert "0:00:00.00" in warnings[0]


def test_check_cue_legibility_warns_when_a_cue_flashes_by():
    from tenmin.render.subtitles import check_cue_legibility

    warnings = check_cue_legibility([_cue(1.0, 1.3, "好。")])
    assert len(warnings) == 1
    assert "0.30" in warnings[0]


def test_check_cue_legibility_lets_the_shortest_real_cue_pass():
    """work/saijo 全季最短的一条真实 cue 是 0.80 秒的「下集见。」，它是正常创作。"""
    from tenmin.render.subtitles import check_cue_legibility

    assert check_cue_legibility([_cue(0.0, 0.80, "下集见。")]) == []


def test_check_cue_legibility_thresholds_are_configurable_and_zero_disables():
    from tenmin.render.subtitles import check_cue_legibility

    cues = [_cue(0.0, 0.2, "好。"), _cue(1.0, 15.0, "一二三四五六七八九十" * 8)]
    assert len(check_cue_legibility(cues)) == 2
    assert check_cue_legibility(cues, max_lines=3, min_seconds=0.1) == []
    assert check_cue_legibility(cues, max_lines=0, min_seconds=0) == []


def test_check_cue_legibility_counts_lines_with_the_real_font_metrics():
    """行数是按 font_size / width 真算出来的，不是按字数猜的。"""
    from tenmin.render.subtitles import check_cue_legibility

    cues = [_cue(0.0, 15.0, "一二三四五六七八九十" * 5)]
    assert check_cue_legibility(cues, font_size=52, width=1920) == []
    assert len(check_cue_legibility(cues, font_size=52, width=640)) == 1
