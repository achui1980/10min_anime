"""SRT 时间戳与秒数之间的互转，以及给人看的时长格式化。ingest、docgen、script 共用。"""

from __future__ import annotations

import math
import re

# 时间戳的正则**片段**（4 个捕获组：时、分、秒、毫秒），全项目唯一一份。
# `ingest/srt_parser.py` 的 `-->` 行正则拿它拼两次 —— 原先那边把同一个模式又抄了两遍，
# 任一处放宽另一处就失配。
#
# `(?!\d)` 是给非锚定使用场景准备的：本模块的 `_TS` 靠 `\s*$` 就能挡住 `,0000` 里多出来
# 的那一位，但 srt_parser 的 `search` 没有右锚，右侧时间戳的 `\d{1,3}` 会吃掉前 3 位就
# 收工，把 `00:00:02,1234` 静默当成 `00:00:02,123`。宁可整块拒绝（计入 skipped_blocks，
# 用户看得到）也不要给出一个看起来合法的错时间 —— 这跟分/秒溢出（`[0-5]?\d` 直接拒）
# 已有的处置一致。
TIMESTAMP_PATTERN = r"(\d{1,3}):([0-5]?\d):([0-5]?\d)[,.](\d{1,3})(?!\d)"

_TS = re.compile(rf"^\s*{TIMESTAMP_PATTERN}\s*$")


def timestamp_from_groups(hours: str, minutes: str, seconds: str, millis: str) -> float:
    """已经捕获好的 4 个分组 -> 秒。毫秒位不足 3 位按右侧补零。

    给已经用 `TIMESTAMP_PATTERN` 匹配过的调用方复用，省掉「拿到整串再对它重跑一遍
    完整正则」那一步（srt_parser 原先每条 cue 要跑 4 次正则操作）。
    """
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")) / 1000.0
    )


def parse_timestamp(text: str) -> float:
    """把 `HH:MM:SS,mmm` 或 `HH:MM:SS.mmm` 解析成秒。毫秒位不足 3 位按右侧补零。"""
    m = _TS.match(text)
    if not m:
        raise ValueError(f"无法解析的时间戳: {text!r}")
    return timestamp_from_groups(*m.groups())


def _finite(seconds: float) -> float:
    """挡住 nan / inf。

    原来靠下游的 `int(round(...))` 自己炸：nan 抛 ValueError、inf 抛 OverflowError，
    消息都是 CPython 的「cannot convert float ...」，看不出坏的是哪个数据。
    而 OverflowError **不在** `cli.PIPELINE_ERRORS` 里，于是 inf 会把 traceback 糊到
    用户脸上；统一成 ValueError 就走正常的红字报错 + exit 1。
    """
    if not math.isfinite(seconds):
        raise ValueError(f"秒数必须是有限数，收到 {seconds!r}")
    return seconds


def format_timestamp(seconds: float) -> str:
    """把秒格式化成 `HH:MM:SS.mmm`。负数按 0 处理，nan/inf 报错。

    注意毫秒分隔符是 `.` 而不是 SRT 的 `,` —— 这是给人看的显示格式（对照表、CLI 输出、
    喂 LLM 的台词清单），**不要**拿它去写 SRT 文件。ASS 的 `H:MM:SS.cc` 也不同，见
    render/subtitles.py。
    """
    seconds = _finite(seconds)
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def readable_seconds(seconds: float) -> str:
    """把秒格式化成人读的「N 分 N 秒」／不足一分钟时「N 秒」。负数按 0 处理，nan/inf 报错。

    docgen/table.py 的 `_readable_seconds` 与 script/single.py 的 `_readable_duration`
    原来各有一份实现，输出还不完全一样（后者对 <60 秒会吐「0 分 24 秒」）。这里合并成一份。

    先 round 到整秒再 divmod：旧的 single.py 实现是 `int(s // 60)` 分 + `f"{rest:.0f}"` 秒，
    rest=59.6 时会吐出「23 分 60 秒」这种进位越界的结果。
    """
    total = max(0, round(_finite(seconds)))
    minutes, secs = divmod(total, 60)
    if minutes == 0:
        return f"{secs} 秒"
    return f"{minutes} 分 {secs} 秒"
