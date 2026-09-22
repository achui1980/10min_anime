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

    id 是对白在 track.lines 里的位置（从 1 起）。轨里没有对应译文的行直接跳过（片头片尾
    那些行压根不会被送去翻译），译文为空白的也跳过 —— 空字幕块会让播放器显示一个空行。
    这里只按「有没有内容」判，不追究空白是「从没送去翻译」还是「模型把一条译成了空串」：
    后者是内容级的漏译，该由翻译阶段自己的 id 对齐校验（进去多少条就该出来多少条）拦，
    计划落在 translate/lines.py，目前还没有这个模块。

    反过来，带了轨里不存在的 id 一律报错，跟上面的静默跳过刻意不对称：多出来的 id 是
    「上游对齐没对上」的硬证据，而且它必然指到一行**不是这句**的时间上 —— 静默处理等于
    往交付物里塞一条错时间的字幕。0 与近边界的负数同样拦下，不然 `track.lines[id - 1]`
    会按 Python 的负数下标静默配到末尾某一行的时间：id 为 0 时总是最后一行，负数则在
    |id| < len(lines) 时命中倒数第 |id| + 1 行。更负的 id 已经越出列表，抛的是
    IndexError —— 而它**不在** cli.PIPELINE_ERRORS 里，用户会看到整页 traceback，所以
    这条守卫的意义不只是「别配错时间」，也包括「别把越界暴露成 traceback」。
    """
    total = len(track.lines)
    blocks: list[tuple[float, float, int, str]] = []
    for line in translated.lines:
        if line.id < 1 or line.id > total:
            raise ValueError(
                f"译文里的 id {line.id} 超出对白轨范围（共 {total} 行），上游的对齐校验没拦住"
            )
        text = line.zh.strip()
        if not text:
            continue
        source = track.lines[line.id - 1]
        blocks.append((source.start, source.end, line.id, text))

    # 按时间排序而不是按译文顺序：模型可能乱序回填，而播放器要求 cue 单调递增。第三个
    # 键 line.id 是为「同起点同终点」那种排不开的情况兜底 —— 那不是边角料，normalize 的
    # split_dual_track 从一条 cue 拆出的各段沿用**同一对** start/end（见 normalize 里
    # 建 DialogueLine 的那段），只按时间排的话它们的先后完全由模型的回填顺序决定，同一
    # 份素材换一次调用就可能换一个字节。line.id 是对白轨里的位置，也就是源文件的阅读
    # 顺序，拿它收尾等于「时间相同就按原文顺序」。
    blocks.sort(key=lambda item: (item[0], item[1], item[2]))
    rendered = [
        f"{index}\n{_format_timestamp(start)} --> {_format_timestamp(end)}\n{text}\n"
        for index, (start, end, _id, text) in enumerate(blocks, start=1)
    ]
    return "\n".join(rendered)
