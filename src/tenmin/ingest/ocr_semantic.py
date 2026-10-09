"""以原稿和第一轮采用文本共同约束第二轮语义复核。"""

import json
import re
from importlib.resources import files

from tenmin.config import DEFAULT_LLM
from tenmin.ingest.ocr_correction import (
    _BATCH_SIZE,
    _CONTEXT_SIZE,
    ReviewItem,
    _Correction,
    _verify,
)
from tenmin.models import RawCue
from tenmin.script.llm import LLMProvider

SYSTEM = (files("tenmin.ingest") / "prompts" / "ocr_semantic.md").read_text(encoding="utf-8")
_NUMBERS = re.compile(
    r"\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿]+(?=岁|年|月|天|分钟|秒|元|手|倍|%)"
)


async def semantic_review(
    original: list[RawCue], first: list[RawCue], provider: LLMProvider,
    *, max_attempts: int = DEFAULT_LLM.max_attempts,
) -> tuple[list[RawCue], list[ReviewItem]]:
    result = []
    review = []
    for begin in range(0, len(first), _BATCH_SIZE):
        end = min(begin + _BATCH_SIZE, len(first))
        rows = [
            {"id": i + 1, "original": original[i].text, "text": first[i].text}
            for i in range(max(0, begin - _CONTEXT_SIZE), min(len(first), end + _CONTEXT_SIZE))
        ]
        payload = {
            "context_only": [row for row in rows if not begin < row["id"] <= end],
            "correct_these": [row for row in rows if begin < row["id"] <= end],
        }
        expected = set(range(begin + 1, end + 1))
        for attempt in range(max_attempts):
            response = await provider.complete(
                SYSTEM, json.dumps(payload, ensure_ascii=False), _Correction
            )
            try:
                lines = _verify(expected, response)
                break
            except ValueError as exc:
                if attempt + 1 == max_attempts:
                    raise
                actual = [line.id for line in response.lines]
                payload["alignment_error"] = (
                    f"{exc}；缺少 {sorted(expected - set(actual))}；"
                    f"多余 {sorted(set(actual) - expected)}；"
                    f"重复 {sorted({i for i in actual if actual.count(i) > 1})}。"
                    "重新返回 correct_these 的全部条目。"
                )
        for i in range(begin, end):
            cue = first[i]
            line = lines[i + 1]
            number_changed = _NUMBERS.findall(cue.text) != _NUMBERS.findall(line.text)
            if line.uncertain or number_changed:
                reason = (
                    "明确数字发生变化，需核对原字幕"
                    if number_changed else line.reason or "无法确认"
                )
                review.append(ReviewItem(
                    i + 1, cue.start, cue.end, cue.text,
                    reason, line.text,
                ))
            result.append(cue.model_copy(update={
                "text": cue.text if line.uncertain or number_changed else line.text
            }))
    return result, review
