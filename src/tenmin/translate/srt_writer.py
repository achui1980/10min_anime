"""中文字幕交付物。

译文轨里只有「第几条对白 → 中文」，时间戳要回到对白轨上取 —— 这是刻意的分工：翻译
不碰时间，时间轴的唯一真相在对白轨里。
"""

from __future__ import annotations

from tenmin.models import DialogueTrack, TranslatedTrack
from tenmin.timecode import format_timestamp


def _format_timestamp(seconds: float) -> str:
    """SRT 的时间戳。

    毫秒分隔符是逗号，而共用的 timecode.format_timestamp 产出点号版本（它的 docstring
    明令不要拿它写 SRT）。两者的差别只在**一个字符**上，所以这里不重抄一份实现、只换
    分隔符 —— 重抄会漏掉 nan/inf 守卫（手抄那份的 `int(round(x))` 对 inf 抛
    OverflowError，而它**不在** cli.PIPELINE_ERRORS 里，用户看到整页 traceback）与负数
    夹紧。实测 format_timestamp(1.5) == "00:00:01.500"（只有一个点，所以整串 replace 是
    安全的）、format_timestamp(-3.0) == "00:00:00.000"、inf/nan 都抛
    ValueError("秒数必须是有限数，收到 ...")。

    ingest/asr.py 的同名私有函数是同一个一行式。刻意不跨包 import 它（它是私有名，而且
    那个模块会连带拖进 render.ffmpeg），也刻意不为这一行在 timecode 里新开一个公开名 ——
    两处共 2 行，多一层间接反而更难读。第三处出现时再提。
    """
    return format_timestamp(seconds).replace(".", ",")


def render_zh_srt(track: DialogueTrack, translated: TranslatedTrack) -> str:
    """对白轨的时间 + 译文轨的中文 → 一份可直接拖进播放器的 SRT。

    id 是对白在 track.lines 里的位置（从 1 起）。没有译文的行直接跳过（片头片尾那些
    行压根不会被送去翻译），译文为空白的也跳过 —— 空字幕块会让播放器显示一个空行。

    带了轨里不存在的 id 则报错：那意味着上游的对齐校验漏了，静默丢掉会让「译文少了
    一句」这种事永远查不出来。0 与负数同样拦下，不然 `track.lines[id - 1]` 会按 Python
    的负数下标静默配到末尾某一行的时间。
    """
    total = len(track.lines)
    blocks: list[tuple[float, float, str]] = []
    for line in translated.lines:
        if line.id < 1 or line.id > total:
            raise ValueError(
                f"译文里的 id {line.id} 超出对白轨范围（共 {total} 行），上游的对齐校验没拦住"
            )
        text = line.zh.strip()
        if not text:
            continue
        source = track.lines[line.id - 1]
        blocks.append((source.start, source.end, text))

    # 按时间排序而不是按译文顺序：模型可能乱序回填，而播放器要求 cue 单调递增。
    blocks.sort(key=lambda item: (item[0], item[1]))
    rendered = [
        f"{index}\n{_format_timestamp(start)} --> {_format_timestamp(end)}\n{text}\n"
        for index, (start, end, text) in enumerate(blocks, start=1)
    ]
    return "\n".join(rendered)
