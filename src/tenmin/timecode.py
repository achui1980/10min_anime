"""SRT 时间戳与秒数之间的互转，以及给人看的时长格式化。ingest、docgen、script 共用。"""

from __future__ import annotations

import re

_TS = re.compile(r"^\s*(\d{1,3}):([0-5]?\d):([0-5]?\d)[,.](\d{1,3})\s*$")


def parse_timestamp(text: str) -> float:
    """把 `HH:MM:SS,mmm` 或 `HH:MM:SS.mmm` 解析成秒。毫秒位不足 3 位按右侧补零。"""
    m = _TS.match(text)
    if not m:
        raise ValueError(f"无法解析的时间戳: {text!r}")
    hours, minutes, seconds, millis = m.groups()
    millis = millis.ljust(3, "0")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def format_timestamp(seconds: float) -> str:
    """把秒格式化成 `HH:MM:SS.mmm`。负数按 0 处理。

    注意毫秒分隔符是 `.` 而不是 SRT 的 `,` —— 这是给人看的显示格式（对照表、CLI 输出、
    喂 LLM 的台词清单），**不要**拿它去写 SRT 文件。ASS 的 `H:MM:SS.cc` 也不同，见
    render/subtitles.py。
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def readable_seconds(seconds: float) -> str:
    """把秒格式化成人读的「N 分 N 秒」／不足一分钟时「N 秒」。负数按 0 处理。

    docgen/table.py 的 `_readable_seconds` 与 script/single.py 的 `_readable_duration`
    原来各有一份实现，输出还不完全一样（后者对 <60 秒会吐「0 分 24 秒」）。这里合并成一份。

    先 round 到整秒再 divmod：旧的 single.py 实现是 `int(s // 60)` 分 + `f"{rest:.0f}"` 秒，
    rest=59.6 时会吐出「23 分 60 秒」这种进位越界的结果。
    """
    total = max(0, round(seconds))
    minutes, secs = divmod(total, 60)
    if minutes == 0:
        return f"{secs} 秒"
    return f"{minutes} 分 {secs} 秒"
