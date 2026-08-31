"""LLM provider。v1 只有 Gemini 一家。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from tenmin.config import LLMConfig, Settings


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        """schema 非空时返回该 schema 的实例，否则返回纯文本。"""
        ...


class GeminiProvider:
    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key)

    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
        )
        response = await self._client.aio.models.generate_content(
            model=self.model, contents=user, config=config
        )
        if schema is None:
            return response.text
        if response.parsed is not None:
            return response.parsed
        return schema.model_validate_json(response.text)


def build_provider(cfg: LLMConfig, settings: Settings) -> LLMProvider:
    if cfg.provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "缺少 Gemini API key，请设置环境变量 TENMIN_GEMINI_API_KEY"
            )
        return GeminiProvider(api_key=settings.gemini_api_key, model=cfg.model)
    raise ValueError(f"不支持的 LLM provider：{cfg.provider}")
