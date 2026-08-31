"""SRT 时间戳与秒数之间的互转。ingest 与 docgen 共用。"""

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
    """把秒格式化成 `HH:MM:SS.mmm`。负数按 0 处理。"""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
