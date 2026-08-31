"""测试用假 LLM provider。绝不联网。"""

from __future__ import annotations

import json
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
