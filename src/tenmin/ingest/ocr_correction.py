"""整集一次校对 OCR 原稿；模型只返回要改的行，时间和条数始终由程序保留。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from importlib.resources import files
from pathlib import Path

import httpx
from pydantic import BaseModel

from tenmin import atomic
from tenmin.config import AsrConfig, LLMConfig
from tenmin.ingest.asr import render_srt
from tenmin.ingest.srt_parser import load_srt, load_srt_detailed
from tenmin.models import RawCue
from tenmin.script.llm import LLMProvider

_SYSTEM = (files("tenmin.ingest") / "prompts" / "ocr_correction.md").read_text(encoding="utf-8")
# original 与该 id 原文的相似度低于此值视为模型贴错了行。
_MIN_ORIGINAL_RATIO = 0.8
# 改动字数上限：max(_MAX_EDIT_CHARS, 原文字数 × _MAX_EDIT_RATIO)，超出视为改写说法。
_MAX_EDIT_CHARS = 2
_MAX_EDIT_RATIO = 0.3


class _Fix(BaseModel):
    id: int
    original: str
    text: str
    reason: str = ""
    uncertain: bool = False


class _Fixes(BaseModel):
    lines: list[_Fix]


@dataclass(frozen=True)
class ReviewItem:
    id: int
    start: float
    end: float
    text: str
    reason: str
    candidate: str = ""


def _edit_size(before: str, after: str) -> int:
    """被替换、删除或插入的字数（取两侧较大者）。"""
    size = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, before, after).get_opcodes():
        if tag != "equal":
            size += max(i2 - i1, j2 - j1)
    return size


def _too_large(before: str, after: str) -> bool:
    limit = max(_MAX_EDIT_CHARS, int(len(before) * _MAX_EDIT_RATIO))
    return _edit_size(before, after) > limit


def _user_message(
    cues: list[RawCue], asr: list[RawCue] | None, glossary: dict[str, str] | None = None
) -> str:
    lines = []
    if glossary:
        lines += ["# 人物名表（日文 → 中文）"]
        lines += [f"{ja} → {zh}" for ja, zh in glossary.items()]
        lines += [""]
    lines += ["# OCR 字幕原稿（id | 开始-结束秒 | 文本）"]
    lines += [
        f"{pos} | {cue.start:.2f}-{cue.end:.2f} | {cue.text}"
        for pos, cue in enumerate(cues, start=1)
    ]
    if asr:
        lines += ["", "# 日语语音转写（开始-结束秒 | 文本）"]
        lines += [f"{cue.start:.2f}-{cue.end:.2f} | {cue.text}" for cue in asr]
    return "\n".join(lines)


def apply_fixes(
    cues: list[RawCue], fixes: list[_Fix]
) -> tuple[list[RawCue], list[ReviewItem], list[str]]:
    """三道校验后套用修改；返回新 cue、review 与被丢弃的说明。

    非法（越界/重复/空/含字幕结构）或错位（original 对不上）的条目丢弃；
    改动过大或模型自己拿不准的条目保留原文并进 review。
    """
    result = list(cues)
    review: list[ReviewItem] = []
    dropped: list[str] = []
    seen: set[int] = set()
    for fix in fixes:
        text = " ".join(fix.text.split("\n")).strip()
        if not 1 <= fix.id <= len(cues):
            dropped.append(f"id {fix.id} 越界")
            continue
        if fix.id in seen:
            dropped.append(f"id {fix.id} 重复")
            continue
        seen.add(fix.id)
        cue = cues[fix.id - 1]
        if not text or re.search(r"-->", text):
            dropped.append(f"id {fix.id} 文本为空或含时间戳")
            continue
        if SequenceMatcher(None, fix.original, cue.text).ratio() < _MIN_ORIGINAL_RATIO:
            dropped.append(f"id {fix.id} 原文对不上（疑似错位）")
            continue
        if text == cue.text and not fix.uncertain:
            continue
        reason = fix.reason or "无法确认"
        if fix.uncertain:
            review.append(ReviewItem(fix.id, cue.start, cue.end, cue.text, reason, text))
            continue
        if _too_large(cue.text, text):
            review.append(ReviewItem(
                fix.id, cue.start, cue.end, cue.text, f"改动过大，未采用：{reason}", text
            ))
            continue
        result[fix.id - 1] = cue.model_copy(update={"text": text})
    return result, review, dropped


async def correct_cues(
    cues: list[RawCue],
    provider: LLMProvider,
    *,
    asr: list[RawCue] | None = None,
    glossary: dict[str, str] | None = None,
) -> tuple[list[RawCue], list[ReviewItem], list[str]]:
    """整集一次调用；模型只返回需要修改或拿不准的行。"""
    if not cues:
        raise ValueError("OCR 原稿没有可校对的字幕")
    response = await provider.complete(_SYSTEM, _user_message(cues, asr, glossary), _Fixes)
    return apply_fixes(cues, response.lines)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(
    cfg: LLMConfig, asr: Path | None = None, glossary: dict[str, str] | None = None
) -> str:
    fields = cfg.model_dump(
        include={"provider", "model", "base_url", "thinking", "temperature", "max_output_tokens"}
    )
    fields["prompt_sha256"] = hashlib.sha256(_SYSTEM.encode("utf-8")).hexdigest()
    fields["asr_sha256"] = _digest(asr) if asr is not None and asr.is_file() else None
    if glossary:
        fields["glossary"] = dict(sorted(glossary.items()))
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def _state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def correction_pending(
    raw: Path, corrected: Path, state: Path, *, cfg: LLMConfig, source_identity: str,
    asr: Path | None = None, glossary: dict[str, str] | None = None,
) -> bool:
    """失败降级后供全局 ingest 新鲜度判据识别待重试的校对。"""
    if not raw.is_file() or not corrected.is_file():
        return True
    saved = _state(state)
    if saved is None:
        return True
    same_source = (
        saved.get("raw_sha256") == _digest(raw)
        and saved.get("source_identity") == source_identity
    )
    if same_source and saved.get("machine_sha256") != _digest(corrected):
        return False
    current = (_digest(raw), source_identity, _identity(cfg, asr, glossary))
    expected = (saved.get("raw_sha256"), saved.get("source_identity"), saved.get("llm_identity"))
    return saved.get("status") != "success" or current != expected


def record_fallback(
    raw: Path, corrected: Path, state: Path, *, cfg: LLMConfig, source_identity: str,
    asr: Path | None = None, glossary: dict[str, str] | None = None,
) -> None:
    """记录交付物是机器回退稿，而不是人工稿；它仍须自动重试。"""
    _write_state(
        raw, corrected, state, cfg=cfg, source_identity=source_identity,
        status="fallback", asr=asr, glossary=glossary,
    )


def _write_state(
    raw: Path, corrected: Path, state: Path,
    *, cfg: LLMConfig, source_identity: str, status: str, asr: Path | None = None,
    glossary: dict[str, str] | None = None,
) -> None:
    atomic.write_text(
        state,
        json.dumps({
            "status": status,
            "raw_sha256": _digest(raw),
            "source_identity": source_identity,
            "llm_identity": _identity(cfg, asr, glossary),
            "machine_sha256": _digest(corrected),
        }, ensure_ascii=False, indent=2) + "\n",
    )


def is_human_edited(corrected: Path, state: Path) -> bool:
    """没有可信状态的既有文件也按人工稿保护；mtime 不是编辑身份。"""
    saved = _state(state)
    return corrected.is_file() and (
        saved is None or saved.get("machine_sha256") != _digest(corrected)
    )


async def select_corrected(
    raw: Path,
    corrected: Path,
    state: Path,
    review: Path,
    *,
    provider_factory: Callable[[], LLMProvider],
    cfg: LLMConfig,
    source_identity: str,
    overwrite_unmanaged: bool = False,
    asr: Path | None = None,
    glossary: dict[str, str] | None = None,
) -> tuple[Path, list[str], bool]:
    """选用成功/人工校对稿；失败用原稿并在下次自动重试。

    ``asr`` 是可选的日语语音转写 SRT，只作读音/形近字佐证放进提示词；
    ``glossary`` 是日文名→中文名的人物名表，用来统一人名写法。
    """
    raw_sha = _digest(raw)
    saved = _state(state)
    human_edit = corrected.is_file() and (
        (saved is None and _digest(corrected) != raw_sha)
        or (saved is not None and saved.get("machine_sha256") != _digest(corrected))
    )
    current = saved is not None and (
        saved.get("raw_sha256") == raw_sha
        and saved.get("source_identity") == source_identity
        and saved.get("llm_identity") == _identity(cfg, asr, glossary)
    )
    same_source = (
        saved is not None and saved.get("raw_sha256") == raw_sha
        and saved.get("source_identity") == source_identity
    )
    if human_edit and same_source:
        return corrected, [], False
    if current and corrected.is_file() and saved.get("status") == "success":
        return corrected, [], False
    if human_edit and not (overwrite_unmanaged and saved is None):
        message = (
            f"OCR AI 校对稿 {corrected} 曾被人工修改，来源已更新或无可信状态；"
            "保留旧稿，本次使用原稿"
        )
        return raw, [message], True

    provider = None
    try:
        parsed = load_srt_detailed(raw)
        if parsed.skipped_blocks or parsed.clamped_cues:
            raise ValueError("OCR 原稿含无效时间戳，跳过 AI 校对以保留原稿")
        asr_cues = load_srt(asr) if asr is not None and asr.is_file() else None
        try:
            provider = provider_factory()
            cues, items, dropped = await correct_cues(
                parsed.cues, provider, asr=asr_cues, glossary=glossary
            )
        finally:
            if provider is not None:
                aclose = getattr(provider, "aclose", None)
                if aclose is not None:
                    await aclose()
    except (RuntimeError, ValueError, httpx.HTTPError, TimeoutError) as exc:
        return raw, [f"OCR AI 校对失败：{exc or type(exc).__name__}"], True

    atomic.write_text_if_changed(corrected, render_srt(cues))
    atomic.write_text_if_changed(
        review, json.dumps([item.__dict__ for item in items], ensure_ascii=False, indent=2) + "\n"
    )
    messages = []
    if dropped:
        messages.append(f"OCR AI 校对丢弃了 {len(dropped)} 条不合格修改：{'；'.join(dropped)}")
    _write_state(
        raw, corrected, state, cfg=cfg, source_identity=source_identity,
        status="success", asr=asr, glossary=glossary,
    )
    return corrected, messages, False


def prepare_asr(
    video: Path, cache: Path, *, asr_config: AsrConfig, ffmpeg_path: str
) -> tuple[Path | None, str | None]:
    """取得给校对用的日语转写：可用缓存直接复用，否则转写；失败返回警告、不带日语。"""
    from tenmin.ingest import asr as asr_module
    from tenmin.ingest.resolve import _is_usable_cache

    if _is_usable_cache(cache, video):
        return cache, None
    try:
        asr_module.transcribe(video, cache, asr=asr_config, ffmpeg_path=ffmpeg_path)
    except asr_module.ASRError as exc:
        return None, f"日语转写失败，本次校对不带日语：{exc}"
    return cache, None
