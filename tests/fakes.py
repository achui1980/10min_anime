"""测试用假 LLM provider。绝不联网。"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class FakeProvider:
    """按调用顺序返回预置的响应。"""

    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if not self.responses:
            raise AssertionError("FakeProvider 的预置响应已用尽")
        payload = self.responses.pop(0)
        if isinstance(payload, BaseModel):
            return payload
        if schema is not None:
            if isinstance(payload, str):
                return schema.model_validate_json(payload)
            return schema.model_validate(payload)
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)


class EpisodeAwareProvider:
    """按 prompt 里的集号返回响应，可选人为延迟。绝不联网。

    FakeProvider 是「按调用顺序 pop」的，多集并发下 complete 的调用顺序不确定，
    「哪一集拿到哪份稿子」就变成了随机数（而 script/validate.py 会拿 clip 的集号跟
    本集的对白轨核对，串错了直接判错）。所以并发相关的测试一律用这个。

    `events` 是一条 (事件, 集号) 的时间线，测试可以往同一个列表里塞别的阶段的事件，
    用来断言「谁在谁之前」。`peak` 是并发峰值。
    """

    _EPISODE_IN_PROMPT = re.compile(r"集数：第 (\d+) 集")

    def __init__(self, responder: Any, *, delay: float = 0.0):
        self.responder = responder
        self.delay = delay
        self.events: list[tuple[str, int]] = []
        self.in_flight = 0
        self.peak = 0

    def episode_of(self, user: str) -> int:
        match = self._EPISODE_IN_PROMPT.search(user)
        if match is None:
            raise AssertionError("prompt 里找不到集号，EpisodeAwareProvider 没法分派")
        return int(match.group(1))

    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        episode = self.episode_of(user)
        self.events.append(("llm-start", episode))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        self.events.append(("llm-done", episode))
        return self.responder(episode)

    @property
    def called_episodes(self) -> list[int]:
        return [episode for event, episode in self.events if event == "llm-start"]


class FakeTTSEngine:
    """测试用假 TTS engine。绝不联网，按调用顺序返回预置时长。"""

    def __init__(self, durations: list[float], fingerprint: str = "fake|voice|+0%"):
        self.durations = list(durations)
        self.fingerprint = fingerprint
        self.calls: list[dict[str, Any]] = []

    async def synthesize(self, text: str, out_path: Path) -> float:
        self.calls.append({"text": text, "out_path": out_path})
        if not self.durations:
            raise AssertionError("FakeTTSEngine 的预置时长已用尽")
        duration = self.durations.pop(0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake mp3")
        return duration


class FlakyTTSEngine:
    """前 fail_times 次抛错，之后返回 duration。测重试用。"""

    fingerprint = "fake|voice|+0%"

    def __init__(self, fail_times: int, duration: float = 3.0):
        self.fail_times = fail_times
        self.duration = duration
        self.attempts = 0

    async def synthesize(self, text: str, out_path: Path) -> float:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(f"网络抖动 {self.attempts}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake mp3")
        return self.duration


class FailingTTSEngine:
    """每次都抛同一个异常。测「什么错该重试、什么错该立刻放弃」用。"""

    fingerprint = "fake|voice|+0%"

    def __init__(self, error: Exception):
        self.error = error
        self.attempts = 0

    async def synthesize(self, text: str, out_path: Path) -> float:
        self.attempts += 1
        raise self.error


class FakeReporter:
    """测试用假 progress reporter。记录每次调用，方便断言顺序和参数。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def stage_start(self, stage: str) -> None:
        self.calls.append(("stage_start", stage))

    def stage_skip(self, stage: str) -> None:
        self.calls.append(("stage_skip", stage))

    def stage_done(self, stage: str) -> None:
        self.calls.append(("stage_done", stage))

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        self.calls.append(("substep", stage, current, total, label))

    def episode_start(self, number: int, index: int, total: int) -> None:
        self.calls.append(("episode_start", number, index, total))

    def episode_done(self, number: int, index: int, total: int) -> None:
        self.calls.append(("episode_done", number, index, total))
