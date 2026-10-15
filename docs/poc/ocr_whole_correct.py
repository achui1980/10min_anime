"""PROTOTYPE（可丢弃）：整集一次性 OCR AI 校对，只返回要改的行。

用法：
  uv run python docs/poc/ocr_whole_correct.py RAW.srt -o OUT_DIR [--asr JA.srt] [--name NAME]

产物（与 tenmin ocr-audit 兼容）：
  OUT_DIR/NAME.zh-Hans.srt         校对稿
  OUT_DIR/NAME.zh-Hans.raw.srt     原稿副本
  OUT_DIR/NAME.zh-Hans.review.json 不确定行
  OUT_DIR/NAME.missing.json        有日语语音、无中文字幕的时间段（需 --asr）
  OUT_DIR/NAME.timings.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel

from tenmin.config import LLMConfig, Settings
from tenmin.ingest.asr import render_srt
from tenmin.ingest.srt_parser import load_srt
from tenmin.models import RawCue
from tenmin.script.llm import build_provider

SYSTEM = """你是中文硬字幕 OCR 校对员。输入是一整集动画的 OCR 字幕原稿（每行 id、时间、文本），
可能附带同一集日语原声的语音转写（带时间，转写本身也可能有错）。
OCR 常见错误：形近字（而/后、巴/已、洽/治、成/戌、官/宫、络/终、佳/住）、漏字、多出画面文字、繁简混杂。
任务：通读整集理解剧情和人物，找出 OCR 识别错误并修正。
规则：
- 只修 OCR 识别错误，不润色、不改写、不翻译、不改语气，标点与空格尽量保持原样。
- 日语转写用于佐证：按时间对照台词含义、人名读音（如 じんし→壬氏）。
- 同一人名/术语在全集写法应一致，少数异写多半是 OCR 错。
- 有把握就直接改，uncertain=false；确实拿不准才 uncertain=true（此时 text 填你的最佳猜测）。
- reason 用一句话写证据。
- 只返回需要修改或拿不准的行，没问题的行不要返回。每个 id 最多出现一次。\n- original 必须逐字抄写该 id 的原稿文本，text 是修改后的该行（只改这一行自己的错字）。"""


class Fix(BaseModel):
    id: int
    original: str
    text: str
    reason: str
    uncertain: bool = False


class Fixes(BaseModel):
    lines: list[Fix]


def _ts(seconds: float) -> str:
    return f"{seconds:.2f}"


def _user(raw: list[RawCue], asr: list[RawCue] | None) -> str:
    parts = ["# OCR 字幕原稿（id | 开始-结束秒 | 文本）"]
    parts += [f"{i} | {_ts(c.start)}-{_ts(c.end)} | {c.text}" for i, c in enumerate(raw, 1)]
    if asr:
        parts.append("\n# 日语语音转写（开始-结束秒 | 文本）")
        parts += [f"{_ts(c.start)}-{_ts(c.end)} | {c.text}" for c in asr]
    return "\n".join(parts)


def _missing(raw: list[RawCue], asr: list[RawCue]) -> list[dict]:
    out = []
    for seg in asr:
        overlap = any(c.start < seg.end and seg.start < c.end for c in raw)
        if not overlap and seg.end - seg.start >= 0.8:
            out.append({"start": seg.start, "end": seg.end, "ja": seg.text})
    return out


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--asr", type=Path)
    p.add_argument("--name", default="whole")
    p.add_argument("--provider", default="minimax")
    p.add_argument("--model", default="MiniMax-M3")
    a = p.parse_args()

    raw = load_srt(a.raw)
    asr = load_srt(a.asr) if a.asr else None
    a.out.mkdir(parents=True, exist_ok=True)
    user = _user(raw, asr)
    provider = build_provider(LLMConfig(provider=a.provider, model=a.model), Settings())
    t0 = time.monotonic()
    try:
        result = await provider.complete(SYSTEM, user, Fixes)
    finally:
        close = getattr(provider, "aclose", None)
        if close:
            await close()
    elapsed = time.monotonic() - t0

    texts = [c.text for c in raw]
    review, seen, dropped = [], set(), []
    for fix in result.lines:
        text = " ".join(fix.text.split("\n")).strip()
        if not 1 <= fix.id <= len(raw) or fix.id in seen or not text or "-->" in text:
            dropped.append(fix.model_dump())
            continue
        cue = raw[fix.id - 1]
        if SequenceMatcher(None, fix.original, cue.text).ratio() < 0.8 or \
                SequenceMatcher(None, text, cue.text).ratio() < 0.3:
            dropped.append({**fix.model_dump(), "why": "original/text 与该 id 原文不符"})
            continue
        seen.add(fix.id)
        if fix.uncertain:
            review.append({"id": fix.id, "start": cue.start, "end": cue.end,
                           "text": cue.text, "reason": fix.reason, "candidate": text})
        else:
            texts[fix.id - 1] = text
    corrected = [c.model_copy(update={"text": t}) for c, t in zip(raw, texts, strict=True)]

    stem = a.out / f"{a.name}.zh-Hans"
    Path(f"{stem}.srt").write_text(render_srt(corrected), encoding="utf-8")
    shutil.copyfile(a.raw, f"{stem}.raw.srt")
    Path(f"{stem}.review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    if asr:
        (a.out / f"{a.name}.missing.json").write_text(
            json.dumps(_missing(raw, asr), ensure_ascii=False, indent=2), encoding="utf-8")
    usage = getattr(provider, "last_usage", None)
    stats = {"seconds": round(elapsed, 1), "returned": len(result.lines),
             "applied": sum(o != n for o, n in zip([c.text for c in raw], texts, strict=True)),
             "uncertain": len(review), "dropped": dropped, "prompt_chars": len(user),
             "usage": str(usage)}
    (a.out / f"{a.name}.timings.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped"}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
