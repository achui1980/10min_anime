"""SRT 解析。产物 RawCue 未经任何清洗，text 保留块内换行。"""

from __future__ import annotations

import codecs
import re
from pathlib import Path
from typing import NamedTuple

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


class SrtParseResult(NamedTuple):
    """解析结果 + 两个「悄悄丢数据」的计数器。

    这两个数字必须能一路传到用户眼前（DialogueTrack 的同名字段 -> dialogue.json ->
    `tenmin inspect` / run_pipeline 的 warnings）：一个格式略歪的字幕文件可能丢掉大量
    对白，而原先解析器对此完全沉默 —— 无告警、不设 suspect、不计数。

    本项目刻意不引入 logging 体系，所以走的是「计数进产物 + 汇总进 warnings」这条路。
    """

    cues: list[RawCue]
    # 找不到时间戳行、被整块跳过的块数。
    skipped_blocks: int
    # `end < start` 被夹成零时长的 cue 数。这些 cue 的 `clamped` 为 True。
    clamped_cues: int


def parse_srt_detailed(text: str) -> SrtParseResult:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[RawCue] = []
    position = 0
    skipped_blocks = 0
    clamped_cues = 0
    for block in _BLOCK_SPLIT.split(text):
        if not block.strip():
            continue
        lines = block.split("\n")
        # 一次循环同时拿到行号与 match：原先第一遍用生成器找 ts_at、第二遍再对同一行
        # 重新 search 一次，并用 `assert match is not None` 兜住第二次的结果 ——
        # `python -O` 下 assert 被剥离后那句会退化成 AttributeError。
        ts_at = None
        match = None
        for index, line in enumerate(lines):
            if found := _TS_LINE.search(line):
                ts_at, match = index, found
                break
        if match is None or ts_at is None:
            skipped_blocks += 1
            continue
        position += 1
        start = parse_timestamp(match.group(1))
        end = parse_timestamp(match.group(2))
        clamped = end < start
        if clamped:
            end = start
            clamped_cues += 1
        # idx 一律用 position（单调递增，因此唯一）。文件里写的序号另存 src_idx：
        # 它可能重复、乱序、或干脆不存在，而 idx 是全项目的定位主键（aggregate 的
        # `ln.idx in anchors`、validate 的 anchor 匹配都按它查），两条时间完全不同的
        # cue 共享一个 idx 会让 _anchor_time 静默取到错误的时间点。
        src_idx = None
        if ts_at > 0:
            head = lines[ts_at - 1].strip()
            if head.isdigit():
                src_idx = int(head)
        body = "\n".join(lines[ts_at + 1 :]).strip("\n")
        cues.append(
            RawCue(
                idx=position,
                src_idx=src_idx,
                start=start,
                end=end,
                text=body,
                clamped=clamped,
            )
        )
    return SrtParseResult(cues, skipped_blocks, clamped_cues)


def parse_srt(text: str) -> list[RawCue]:
    """只要 cue 列表。要坏数据计数请用 parse_srt_detailed。"""
    return parse_srt_detailed(text).cues


def load_srt_detailed(path: Path) -> SrtParseResult:
    return parse_srt_detailed(decode_bytes(Path(path).read_bytes()))


def load_srt(path: Path) -> list[RawCue]:
    return load_srt_detailed(path).cues
