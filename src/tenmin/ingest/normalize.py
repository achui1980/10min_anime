"""把 SRT 汇聚成标准化 DialogueTrack。"""

from __future__ import annotations

import re
from pathlib import Path

from tenmin.ingest.clean import (
    clean_text,
    extract_prefix,
    is_noise,
    is_suspect,
    split_dual_track,
)
from tenmin.ingest.credits import find_credit_ranges, in_credit_window, is_credits
from tenmin.ingest.srt_parser import load_srt
from tenmin.models import DialogueLine, DialogueTrack

_FULL_PAREN = re.compile(r"^[（(].*[）)]$")
_NEWLINE_RUN = re.compile(r"\s*\n\s*")

TERMINAL_PUNCT = frozenset("。！？…」』、，,!?.")
MERGE_MAX_GAP = 0.3
MERGE_MAX_CHARS = 40
MERGE_MAX_LINE_SECONDS = 4.0


def _fold(text: str) -> str:
    return _NEWLINE_RUN.sub(" ", text).strip()


def _classify(
    body: str, *, is_second_segment: bool, speaker: str | None, show_title: str, window: bool
) -> str:
    if is_noise(body):
        return "noise"
    if is_credits(body, show_title=show_title, in_credit_window=window):
        return "credits"
    if _FULL_PAREN.match(body):
        return "screen_text"
    if is_second_segment and speaker is not None:
        return "monologue"
    return "dialogue"


def _can_merge(prev: DialogueLine, nxt: DialogueLine) -> bool:
    if prev.kind != "dialogue" or nxt.kind != "dialogue":
        return False
    if prev.idx == nxt.idx:
        return False
    if prev.speaker != nxt.speaker:
        return False
    if prev.suspect or nxt.suspect:
        return False
    if not prev.text or not nxt.text:
        return False
    if nxt.start - prev.end >= MERGE_MAX_GAP:
        return False
    # 被硬折断的续行都很短；单行 >=4s 说明它本身就是完整的一句（往往是拖长音），
    # 合并它会破坏 signals/density.py 的字密度最小值锚点。
    if prev.duration >= MERGE_MAX_LINE_SECONDS or nxt.duration >= MERGE_MAX_LINE_SECONDS:
        return False
    if prev.text[-1] in TERMINAL_PUNCT:
        return False
    if len(prev.text) + len(nxt.text) > MERGE_MAX_CHARS:
        return False
    return True


def merge_continuations(lines: list[DialogueLine]) -> list[DialogueLine]:
    """把被硬折成两块的同一句话合回去。被并入的 SRT 序号记进 merged_from。"""
    out: list[DialogueLine] = []
    for line in lines:
        if out and _can_merge(out[-1], line):
            prev = out[-1]
            out[-1] = prev.model_copy(
                update={
                    "end": line.end,
                    "text": prev.text + line.text,
                    "raw": f"{prev.raw}\n{line.raw}",
                    "merged_from": [*prev.merged_from, line.idx],
                }
            )
        else:
            out.append(line)
    return out


def build_track(
    srt_path: Path,
    *,
    episode: int,
    show_title: str = "",
    convert_traditional: bool = True,
    glossary: dict[str, str] | None = None,
    op_range: tuple[float, float] | None = None,
    ed_range: tuple[float, float] | None = None,
    merge_lines: bool = False,
) -> DialogueTrack:
    """把 SRT 汇聚成标准化对白轨。

    merge_lines 默认关闭。番剧字幕的 cue 通常是背靠背排的（本项目黄金样本间隔中位数
    0.001 秒）且基本不打句读，跨 cue 合并会把不同说话人黏成一条。句内折行本来就在
    cue 内部用 \\n 表示，_fold 已经处理掉了。带可靠说话人标注的字幕源才适合开启。
    """
    cues = load_srt(srt_path)
    duration = max((cue.end for cue in cues), default=0.0)
    lines: list[DialogueLine] = []

    for cue in cues:
        cleaned = clean_text(cue.text, convert=convert_traditional, glossary=glossary)
        window = in_credit_window(cue.start, duration)
        for position, segment in enumerate(split_dual_track(cleaned)):
            speaker, body = extract_prefix(segment)
            suspect = is_suspect(segment)
            body = _fold(body)
            kind = _classify(
                body,
                is_second_segment=position > 0,
                speaker=speaker,
                show_title=show_title,
                window=window,
            )
            lines.append(
                DialogueLine(
                    idx=cue.idx,
                    start=cue.start,
                    end=cue.end,
                    text=body,
                    raw=cue.text,
                    speaker=speaker,
                    kind=kind,
                    suspect=suspect,
                )
            )

    if merge_lines:
        lines = merge_continuations(lines)
    inferred_op, inferred_ed = find_credit_ranges(lines, duration)
    return DialogueTrack(
        episode=episode,
        source="srt",
        duration=duration,
        op_range=op_range or inferred_op,
        ed_range=ed_range or inferred_ed,
        lines=lines,
    )
