"""生成 ASS 字幕。全是纯函数，输出逐字符可测。

ASS 的时间是 H:MM:SS.cc（厘秒、小时不补零），跟 timecode.format_timestamp 不同，
所以这里单独写格式化函数。
"""

from __future__ import annotations

import unicodedata

from tenmin.config import DEFAULT_RENDER
from tenmin.models import SubtitleCue

# 全部从 config 的默认值派生，不再写第二份字面量：PlayRes 必须跟 render/video.py 的
# scale=WxH 完全一致，否则 libass 会按 PlayRes 与实际画面的比例静默缩放字号，
# 「改了分辨率字幕突然变小」这种问题极难查。
DEFAULT_FONT_NAME = DEFAULT_RENDER.subtitle_font_name
DEFAULT_FONT_SIZE = DEFAULT_RENDER.font_size
PLAY_RES_X = DEFAULT_RENDER.width
PLAY_RES_Y = DEFAULT_RENDER.height
# 字幕四周留白（px，PlayRes 坐标系）。**这两个值同时喂断行计算与 Style 行的
# MarginL/MarginR/MarginV** —— 原来 Style 行里硬编码着第二份 `60,60,60`，改一份不改
# 另一份就会「按一个宽度算断行、按另一个宽度排版」，算出来的行宽从此对不上画面。
#
# 为什么留在模块级、不进 RenderConfig（判据见 AGENTS.md）：Style 行里每一个数字
# （颜色、Bold、Outline、Spacing、Alignment、边距）都同等地是「这部番想要什么」，
# 单独把边距拎出来做旋钮没有依据 —— 已经进 config 的那四个（font_size、
# subtitle_font_name、width、height）进去的理由是**实测出来的耦合 bug**（PlayRes 与
# video.py 的 scale 不一致会让 libass 静默缩放字号），边距没有这个问题。真要参数化
# 该是「字幕样式全面上 config」那一个专项，而不是这里再开一个特例。这次要修的只是
# 「同一个数字写了两遍」。
MARGIN_LR = 60
MARGIN_V = 60
# 全角字符的实际显示宽度近似等于字号本身；用 1.05 留一点余量，
# 免得断行算准了但描边（Outline）一挤又超出画面。
#
# 实测（真 libass 渲染 + 逐帧量墨迹包围盒，Lantinghei SC / 字号 52 / Spacing=3）：
# 一个全角字的真实前进宽度是 **47.3px**，而这里的模型值是 52×1.05 = 54.6px ——
# 模型比实测宽 15%，也就是这条余量实际留了 15% 而不是 5%。
CJK_CHAR_WIDTH_RATIO = 1.05

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
# 黄字黑边、加粗、底部居中、四周留 MARGIN_LR / MARGIN_V。Alignment 2 = 底部居中。
# 参考 B 站/抖音吐槽解说类视频最常见的配色：黄色文字 + 黑色粗描边，
# 对比度强，任何背景下都清楚，比白字彩边更贴近"二次元解说"的既有印象。
# Spacing 加到 3，字号偏大时给字间留点缝，不然描边一粗字就糊成一片。
# 字号用 52（比 48 大一点，48/Outline3 已经清楚但想再对比一版更粗更大的），
# Outline 用 4：配合更大的字号，描边稍粗一点仍保持轮廓清晰。
# 边距三个字段刻意从 MARGIN_LR / MARGIN_V 拼出来，不再写第二份字面量（见上面的注释）。
_STYLE_HEAD = "&H0000FFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,3,0,1,4,1,2"


def _style_tail() -> str:
    """Style 行里字号之后的全部字段。边距每次从常量取，所以只有一份真相。"""
    return f"{_STYLE_HEAD},{MARGIN_LR},{MARGIN_LR},{MARGIN_V},1"


# 优先在这些字符之后断行（标点收尾，不会把标点甩到下一行开头）。
_BREAK_AFTER = "，、。！？；：—…”』」）,.!?;:"
# 回溯窗口：从 max_chars 往回最多找到这个比例处，再往前就宁可硬切。
# 太小会切出很短的行（一行只剩几个字，读起来比切在词中间更难受），太大等于关掉
# 「优先在标点后断行」。实测 10 集 139 次断行：只有 12 次（8.6%）在这个窗口里找不到
# 标点、退化成硬切。
_BREAK_SEARCH_MIN_RATIO = 0.6
# 行首标点最多能往上一行挤多少格（见 _hang_leading_punctuation）。
# 4 格 = 2 个全角标点，标定自语料：10 集里出现过的行首标点串最长就是 `——`。
_HANG_MAX_CELLS = 4


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
    """ASS 正文转义：换行用 \\N，花括号与反斜杠是控制字符必须去掉。

    反斜杠跟花括号是**同一个判据**（P2-E C3）：`{}` 是覆盖标签的定界符，而 `\\N` /
    `\\n` / `\\h` 在花括号**之外**的正文里同样生效 —— 一段含反斜杠的旁白就能自己插
    硬换行、插硬空格，把这边算好的断行版式打乱。所以处置也一样：剥掉。

    刻意不转义成 `\\\\`：ASS 压根没有反斜杠转义机制（`\\\\` 不是「一个字面反斜杠」，
    libass 只会把它当一个未知标签吃掉），也不换成全角 `＼`（那是往正文里塞一个作者
    没写的字）。剥反斜杠必须排在「真换行 → `\\N`」**之前**，否则会把自己插的 `\\N`
    吃回去。

    实测 11 份真实 script.json 的 73 段 narration：反斜杠 0 次出现。
    """
    cleaned = text.replace("{", "").replace("}", "").replace("\\", "")
    return cleaned.replace("\r\n", "\n").replace("\n", "\\N").strip()


def display_cells(text: str) -> int:
    """这段文本占多少个「半角格」。一个全角字 = 2 格。

    分档判据是 unicodedata.east_asian_width，**Ambiguous（A）跟 W/F 一样算宽**。
    这一条是真 libass 实测定下来的，不是照搬 wcwidth 的通俗版本（W/F 算 2、其余算 1）：
    Lantinghei SC / 字号 52 / Spacing=3 下逐帧量墨迹包围盒，一个字符的真实前进宽度

        `字`(W) 47.3px   `，`(F) 47.3px   `—`(A) 47.2px   `…`(A) 47.3px   `·`(A) 47.3px
        `'`(Na) 12.8px   `i`(Na) 14.6px   `(`(Na) 20.4px  `0`(Na) 32.6px  `A`(Na) 37.8px

    也就是说在 CJK 字体里 A 就是全角。而语料（work/saijo 10 集 10338 字）里非 W/F 的
    字符只有 9 个：`—`×196、`'`×42、`…`×6、`·`×1、`B`/`I`/`T`/`O`/`K` 各 1 —— 「A 算窄」
    会把 203/250 个非 W/F 字符全部低估一半，一行 64 个 `—` 会被判成刚好放得下，实测却是
    3021px，超出可用宽度 1800px 的 68%。

    已知上界（刻意接受）：半角字符是**比例**宽度，不是恒定半格。实测 `W` 49.7px、
    `O` 42.2px、`0` 32.6px 都比 1 格的模型值（27.3px）宽。吸收它的是
    CJK_CHAR_WIDTH_RATIO 那 15% 余量：满一行 64 格纯 CJK 实测 1514px，离 1800px 还有
    286px，够 19 个大写字母各超出的 14.9px。一行**全是**大写拉丁字母才会真超宽，那属于
    「旁白从中文变成英文」的形态变化，届时该调的是 font_size 而不是这个度量。
    """
    return sum(2 if unicodedata.east_asian_width(char) in "WFA" else 1 for char in text)


def max_cells_per_line(font_size: int, width: int = PLAY_RES_X) -> int:
    """给定字号与画布宽度，估算一行能塞下多少个**半角格**（全角字 ×2）。

    libass 的自动换行（WrapStyle）只在空格处断行，中文没有空格，
    长句会被当成一个不可断的“单词”直接冲出画面。所以断行必须自己算好、
    手动插 \\N，不能指望 libass 帮忙。

    口径原来是「全角字符个数」，配 len() 度量 —— 半角字符被当成跟汉字一样宽，混了
    拉丁字母/数字的行提前折断（保守，从不超宽，但白扔可用宽度）。现在预算与度量都换成
    格，两边同时 ×2，所以**纯 CJK 的断行结果逐点不变**。
    """
    usable_width = width - 2 * MARGIN_LR
    if font_size <= 0:
        return usable_width
    cells = int(usable_width // (font_size * CJK_CHAR_WIDTH_RATIO / 2))
    return max(cells, 1)


def _wrap_single_line(line: str, max_cells: int) -> list[str]:
    if display_cells(line) <= max_cells:
        return [line]
    pieces: list[str] = []
    remaining = line
    while display_cells(remaining) > max_cells:
        # 先找「放得下的最长前缀」有多少个字符，再在这个上限内往回找标点。
        limit = _cells_prefix_length(remaining, max_cells)
        window_start = max(1, int(limit * _BREAK_SEARCH_MIN_RATIO))
        break_at = limit
        # 下界**不含** window_start 本身。看起来像差一错，实测是刻意保留的：把它改成
        # 含下界能把硬切从 11 次降到 8 次，代价是多出一条 3 行的 cue（8 → 9）与多一个
        # 物理行（498 → 499）—— 3 行字幕挡掉的画面比「切在词中间」难看得多，这笔换不值。
        for i in range(limit, window_start, -1):
            if remaining[i - 1] in _BREAK_AFTER:
                break_at = i
                break
        break_at = _hang_leading_punctuation(remaining, break_at)
        pieces.append(remaining[:break_at])
        remaining = remaining[break_at:]
    if remaining:
        pieces.append(remaining)
    return pieces


def _hang_leading_punctuation(text: str, break_at: int) -> int:
    """下一行开头是标点时把它挤回上一行（CJK 排版的「追い込み」），返回新的断点。

    「句读不可置于行首」是 CJK 禁则里最基本的一条，而原来的断行会违反它：实测 10 集
    139 次断行里 5 行以标点开头，其中 4 行**整行只有一个 `。`**
    （`…工作内容是当一个高中女生的贴身侍从` ⏎ `。`）。

    代价是这一行会超出 max_cells 最多 _HANG_MAX_CELLS 格。为什么这是安全的：
    predicted 宽度模型本身比 libass 实测宽 15%（见 CJK_CHAR_WIDTH_RATIO），实测满行
    1552px 离可用宽度 1800px 还有 248px，而 2 个全角标点只占 94.6px。
    """
    hung = 0
    while break_at < len(text) and text[break_at] in _BREAK_AFTER:
        width = display_cells(text[break_at])
        if hung + width > _HANG_MAX_CELLS:
            break
        hung += width
        break_at += 1
    return break_at


def _cells_prefix_length(text: str, max_cells: int) -> int:
    """最多占 max_cells 格的最长前缀有几个字符。至少 1，否则一格都放不下时会死循环。"""
    used = 0
    for index, char in enumerate(text):
        width = display_cells(char)
        if used + width > max_cells:
            return max(index, 1)
        used += width
    return len(text)


def wrap_text(text: str, max_cells: int) -> str:
    """按显示格数手动断行；已有的换行原样保留，只处理其中过长的单行。"""
    if max_cells <= 0:
        return text
    result: list[str] = []
    for raw_line in text.split("\n"):
        result.extend(_wrap_single_line(raw_line, max_cells))
    return "\n".join(result)


def render_ass(
    cues: list[SubtitleCue],
    *,
    font_size: int = DEFAULT_FONT_SIZE,
    font_name: str = DEFAULT_FONT_NAME,
    width: int = PLAY_RES_X,
    height: int = PLAY_RES_Y,
) -> str:
    """width/height 必须跟 render/video.py 的 scale 用同一对值（见 RenderConfig）。"""
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        _STYLE_FORMAT,
        f"Style: Narration,{font_name},{font_size},{_style_tail()}",
        "",
        "[Events]",
        _EVENT_FORMAT,
    ]
    max_cells = max_cells_per_line(font_size, width)
    for cue in cues:
        wrapped = wrap_text(cue.text, max_cells)
        lines.append(
            f"Dialogue: 0,{format_ass_time(cue.start)},{format_ass_time(cue.end)},"
            f"Narration,,0,0,0,,{escape_text(wrapped)}"
        )
    return "\n".join(lines) + "\n"
