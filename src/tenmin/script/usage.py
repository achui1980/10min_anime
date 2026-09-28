"""每次 provider.complete() 的用量记录；只观察成本，不参与阶段新鲜度。"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from tenmin import atomic
from tenmin.config import LLMConfig

RoundKind = Literal["draft", "validation_retry", "budget_rewrite", "translate", "translate_repair"]


class UsageRecord(BaseModel):
    """一次 complete() 的用量；None 表示 provider 没给该字段。"""

    provider: str
    model: str
    round: RoundKind
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    requests: int | None = None
    elapsed_seconds: float
    ok: bool


async def track_call[T](
    sink: list[UsageRecord] | None,
    provider: Any,
    llm: LLMConfig,
    kind: RoundKind,
    call: Awaitable[T],
) -> T:
    """记录成功或失败的调用；await 后立即读 last_usage，避免并发覆盖。"""
    started = time.monotonic()
    ok = False
    try:
        result = await call
        ok = True
        return result
    finally:
        if sink is not None:
            usage = getattr(provider, "last_usage", None)
            sink.append(
                UsageRecord(
                    provider=llm.provider,
                    model=llm.model,
                    round=kind,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    cached_tokens=getattr(usage, "cached_tokens", None),
                    requests=getattr(usage, "requests", None),
                    elapsed_seconds=time.monotonic() - started,
                    ok=ok,
                )
            )


def write_usage(path: Path, episode: int, records: list[UsageRecord]) -> None:
    """原子写入 {episode, calls} 格式的用量文件。"""
    payload = {"episode": episode, "calls": [r.model_dump(mode="json") for r in records]}
    atomic.write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
