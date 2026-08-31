"""SRT 解析。产物 RawCue 未经任何清洗，text 保留块内换行。"""

from __future__ import annotations

import codecs
import re
from pathlib import Path

from charset_normalizer import from_bytes

from tenmin.models import RawCue
from tenmin.timecode import parse_timestamp

_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n+")
_TS_LINE = re.compile(
    r"(\d{1,3}:[0-5]?\d:[0-5]?\d[,.]\d{1,3})\s*-->\s*(\d{1,3}:[0-5]?\d:[0-5]?\d[,.]\d{1,3})"
)


def decode_bytes(data: bytes) -> str:
    """先剥 UTF-8 BOM 硬试 UTF-8，失败再交给 charset-normalizer 嗅探。"""
    if data.startswith(codecs.BOM_UTF8):
        data = data[len(codecs.BOM_UTF8) :]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    best = from_bytes(data).best()
    if best is None:
        return data.decode("utf-8", errors="replace")
    return str(best)


def parse_srt(text: str) -> list[RawCue]:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[RawCue] = []
    position = 0
    for block in _BLOCK_SPLIT.split(text):
        if not block.strip():
            continue
        lines = block.split("\n")
        ts_at = next((i for i, line in enumerate(lines) if _TS_LINE.search(line)), None)
        if ts_at is None:
            continue
        position += 1
        match = _TS_LINE.search(lines[ts_at])
        assert match is not None
        start = parse_timestamp(match.group(1))
        end = parse_timestamp(match.group(2))
        if end < start:
            end = start
        idx = position
        if ts_at > 0:
            head = lines[ts_at - 1].strip()
            if head.isdigit():
                idx = int(head)
        body = "\n".join(lines[ts_at + 1 :]).strip("\n")
        cues.append(RawCue(idx=idx, start=start, end=end, text=body))
    return cues


def load_srt(path: Path) -> list[RawCue]:
    return parse_srt(decode_bytes(Path(path).read_bytes()))
