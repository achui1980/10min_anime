"""对 OCR 原稿逐条文字校对；原始 cue 的时间和数量始终由程序保留。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import httpx
from pydantic import BaseModel

from tenmin import atomic
from tenmin.config import LLMConfig
from tenmin.ingest.asr import render_srt
from tenmin.ingest.srt_parser import load_srt_detailed
from tenmin.models import RawCue
from tenmin.script.llm import LLMProvider

_SYSTEM = (files("tenmin.ingest") / "prompts" / "ocr_correction.md").read_text(encoding="utf-8")
_BATCH_SIZE = 60
_CONTEXT_SIZE = 2


class _CorrectedLine(BaseModel):
    id: int
    text: str
    uncertain: bool
    reason: str = ""


class _Correction(BaseModel):
    lines: list[_CorrectedLine]


@dataclass(frozen=True)
class ReviewItem:
    id: int
    start: float
    end: float
    text: str
    reason: str
    candidate: str = ""


def _verify(expected: set[int], response: _Correction) -> dict[int, _CorrectedLine]:
    ids = [line.id for line in response.lines]
    if Counter(ids) != Counter(expected):
        raise ValueError("OCR 校对未逐条对齐：存在重复、遗漏或多余的 id")
    if any(not line.text.strip() for line in response.lines):
        raise ValueError("OCR 校对返回空文本")
    if any(re.search(r"\n[ \t\r]*\n|-->", line.text) for line in response.lines):
        raise ValueError("OCR 校对文本包含字幕块分隔或时间戳，不能改变 cue 结构")
    return {line.id: line for line in response.lines}


def alignment_hint(exc: ValueError, expected: set[int], response: _Correction) -> str:
    """把上一次没对齐的具体 id 回灌给模型，让重试有的放矢。"""
    actual = [line.id for line in response.lines]
    return (
        f"{exc}；缺少 {sorted(expected - set(actual))}；"
        f"多余 {sorted(set(actual) - expected)}；"
        f"重复 {sorted({i for i in actual if actual.count(i) > 1})}。"
        "重新返回 correct_these 的全部条目，每个 id 恰好一次。"
    )


async def verified_batch(
    provider: LLMProvider, system: str, payload: dict, expected: set[int], *, max_attempts: int
) -> dict[int, _CorrectedLine]:
    """同一批按 max_attempts 重试；仍不合格就抛出最后一次的 ValueError。"""
    for attempt in range(max_attempts):
        response = await provider.complete(
            system, json.dumps(payload, ensure_ascii=False), _Correction
        )
        try:
            return _verify(expected, response)
        except ValueError as exc:
            if attempt + 1 >= max_attempts:
                raise
            payload["alignment_error"] = alignment_hint(exc, expected, response)
    raise ValueError("OCR 校对重试次数必须为正数")


async def correct_cues(
    cues: list[RawCue], provider: LLMProvider, *, cfg: LLMConfig,
    failures: list[str] | None = None,
) -> tuple[list[RawCue], list[ReviewItem]]:
    """每批附前后两条的只读上下文；某批重试后仍不合格只回退该批。

    回退的那批保留原文并全部进 review；全部批次都失败才整体报错。
    失败说明追加进 ``failures``，调用方据此把状态记成待重试。
    """
    if not cues:
        raise ValueError("OCR 原稿没有可校对的字幕")
    result: list[RawCue] = []
    review: list[ReviewItem] = []
    failed: list[str] = []
    batches = 0
    for begin in range(0, len(cues), _BATCH_SIZE):
        batch = cues[begin : begin + _BATCH_SIZE]
        context = cues[
            max(0, begin - _CONTEXT_SIZE) : begin + _BATCH_SIZE + _CONTEXT_SIZE
        ]
        context_lines = [
            {"id": pos + 1, "text": cue.text}
            for pos, cue in enumerate(context, start=max(0, begin - _CONTEXT_SIZE))
            if pos < begin or pos >= begin + len(batch)
        ]
        payload = {
            "context_only": context_lines,
            "correct_these": [
                {"id": begin + pos + 1, "text": cue.text} for pos, cue in enumerate(batch)
            ],
        }
        batches += 1
        expected = set(range(begin + 1, begin + len(batch) + 1))
        try:
            lines = await verified_batch(
                provider, _SYSTEM, payload, expected, max_attempts=cfg.max_attempts
            )
        except ValueError as exc:
            note = f"第 {begin + 1}–{begin + len(batch)} 条校对失败，保留原文：{exc}"
            failed.append(note)
            for pos, cue in enumerate(batch, start=begin + 1):
                review.append(ReviewItem(pos, cue.start, cue.end, cue.text, note))
                result.append(cue)
            continue
        for pos, cue in enumerate(batch, start=begin + 1):
            line = lines[pos]
            if line.uncertain:
                review.append(
                    ReviewItem(pos, cue.start, cue.end, cue.text, line.reason or "无法确认")
                )
            result.append(
                cue.model_copy(update={"text": cue.text if line.uncertain else line.text})
            )
    if len(failed) == batches:
        raise ValueError(f"OCR 校对全部批次失败：{failed[-1]}")
    if failures is not None:
        failures.extend(failed)
    return result, review


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(cfg: LLMConfig) -> str:
    fields = cfg.model_dump(
        include={"provider", "model", "base_url", "thinking", "temperature", "max_output_tokens"}
    )
    fields["prompt_sha256"] = hashlib.sha256(_SYSTEM.encode("utf-8")).hexdigest()
    semantic = (files("tenmin.ingest") / "prompts" / "ocr_semantic.md").read_bytes()
    fields["semantic_sha256"] = hashlib.sha256(semantic).hexdigest()
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
    raw: Path, corrected: Path, state: Path, *, cfg: LLMConfig, source_identity: str
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
    current = (_digest(raw), source_identity, _identity(cfg))
    expected = (saved.get("raw_sha256"), saved.get("source_identity"), saved.get("llm_identity"))
    return saved.get("status") != "success" or current != expected


def record_fallback(
    raw: Path, corrected: Path, state: Path, *, cfg: LLMConfig, source_identity: str
) -> None:
    """记录交付物是机器回退稿，而不是人工稿；它仍须自动重试。"""
    _write_state(raw, corrected, state, cfg=cfg, source_identity=source_identity, status="fallback")


def _write_state(
    raw: Path, corrected: Path, state: Path,
    *, cfg: LLMConfig, source_identity: str, status: str,
) -> None:
    atomic.write_text(
        state,
        json.dumps({
            "status": status,
            "raw_sha256": _digest(raw),
            "source_identity": source_identity,
            "llm_identity": _identity(cfg),
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
) -> tuple[Path, list[str], bool]:
    """选用成功/人工校对稿；失败用原稿并在下次自动重试。"""
    raw_sha = _digest(raw)
    saved = _state(state)
    human_edit = corrected.is_file() and (
        (saved is None and _digest(corrected) != raw_sha)
        or (saved is not None and saved.get("machine_sha256") != _digest(corrected))
    )
    current = saved is not None and (
        saved.get("raw_sha256") == raw_sha
        and saved.get("source_identity") == source_identity
        and saved.get("llm_identity") == _identity(cfg)
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
    messages = []
    semantic_pending = False
    failures: list[str] = []
    try:
        parsed = load_srt_detailed(raw)
        if parsed.skipped_blocks or parsed.clamped_cues:
            raise ValueError("OCR 原稿含无效时间戳，跳过 AI 校对以保留原稿")
        try:
            provider = provider_factory()
            cues, items = await correct_cues(
                parsed.cues, provider, cfg=cfg, failures=failures
            )
            from tenmin.ingest.ocr_semantic import semantic_review

            try:
                cues, semantic_items = await semantic_review(
                    parsed.cues, cues, provider,
                    max_attempts=cfg.max_attempts, failures=failures,
                )
                kept = {item.id for item in semantic_items}
                items = semantic_items + [item for item in items if item.id not in kept]
                items.sort(key=lambda item: item.id)
            except (RuntimeError, ValueError, httpx.HTTPError, TimeoutError) as exc:
                messages.append(f"OCR 语义复核失败，使用第一轮校对稿：{exc}")
                semantic_pending = True
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
    if failures:
        messages.extend(f"OCR AI 校对部分失败：{note}" for note in failures)
        semantic_pending = True
    _write_state(
        raw, corrected, state, cfg=cfg, source_identity=source_identity,
        status="partial" if failures else ("semantic_pending" if semantic_pending else "success"),
    )
    return corrected, messages, semantic_pending
