"""LLM provider。目前支持 Gemini 与 MiniMax（OpenAI 兼容接口）。"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ValidationError

from tenmin.config import LLMConfig, Settings

MINIMAX_BASE_URL = "https://api.minimax.cn/v1"
MINIMAX_MAX_ATTEMPTS = 3
# read=None：实测 MiniMax-M3 处理 ~35k 字符 prompt 需要 561 秒，非流式模式下服务端在
# 这 561 秒里零字节返回，正好贴着旧的 600 秒读超时悬崖。流式下每个 SSE chunk 都会刷新
# 读活性，因此读超时交给 chunk 间隔而不是整体耗时（这里直接关掉固定读超时）。
MINIMAX_TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=120.0, pool=30.0)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:json)?[ \t]*\n?|\n?```[ \t]*$", re.M)


def _strip_reasoning(text: str) -> str:
    """MiniMax-M3 每次都在正文前吐一个 <think>…</think> 推理块，必须剥掉。"""
    return _THINK_BLOCK.sub("", text).strip()


def _extract_json(text: str) -> str:
    """剥掉推理块与 markdown 围栅，再取第一个 { 到最后一个 }。

    模型偶尔会在 JSON 前后加解释文字，所以不能直接 json.loads 整个 content。
    """
    cleaned = _FENCE.sub("", _strip_reasoning(text)).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"MiniMax 返回里找不到 JSON 对象：{cleaned[:200]!r}")
    return cleaned[start : end + 1]


def _sse_delta(line: str) -> str:
    """从一行 SSE 里取增量文本；不是可用的数据行就返回空串。

    容忍这些真实形态：空行/心跳行、`data: [DONE]`、只带 finish_reason 而 delta 为空的
    收尾 chunk、delta.content 为 null、以及个别实现插进来的非 JSON 行。
    """
    line = line.strip()
    if not line.startswith("data:"):
        return ""
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return ""
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return ""
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        """schema 非空时返回该 schema 的实例，否则返回纯文本。"""
        ...


class GeminiProvider:
    def __init__(self, api_key: str, model: str = "gemini-3.6-flash"):
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


class MiniMaxProvider:
    """MiniMax 的 OpenAI 兼容接口。

    与 GeminiProvider 的关键差别：MiniMax 接受 response_format=json_schema 却完全
    不执行它（实测传严格嵌套 schema 后，返回的字段名全是模型自己编的），所以 schema
    只能写进提示词正文，再靠 pydantic 校验 + 带着报错重试来兜住。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "MiniMax-M3",
        base_url: str = MINIMAX_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    def _schema_prompt(self, user: str, schema: type[BaseModel]) -> str:
        spec = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        return (
            f"{user}\n\n"
            "## 输出 JSON Schema（必须严格遵守）\n\n"
            f"```json\n{spec}\n```\n\n"
            "只输出符合上面 schema 的 JSON 对象本身。字段名一个字都不能改，"
            "不要在外面再套一层包裹对象，不要加解释文字，不要加 markdown 围栅。"
        )

    async def _stream_once(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> str:
        """发一次流式请求，把所有 delta.content 拼成完整文本。"""
        parts: list[str] = []
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        ) as response:
            if response.is_error:
                # 流式响应在读取 body 之前 raise_for_status 只能给出空错误体。
                await response.aread()
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = _sse_delta(line)
                if chunk:
                    parts.append(chunk)
        return "".join(parts)

    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        content = self._schema_prompt(user, schema) if schema is not None else user
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True}
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}

        last_error = ""
        async with httpx.AsyncClient(timeout=MINIMAX_TIMEOUT) as client:
            for _ in range(MINIMAX_MAX_ATTEMPTS):
                text = await self._stream_once(client, payload)
                if schema is None:
                    return _strip_reasoning(text)
                try:
                    return schema.model_validate_json(_extract_json(text))
                except (ValidationError, ValueError) as exc:
                    last_error = str(exc)[:1500]
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "上面的输出不符合 schema，校验报错如下。"
                                "请只输出修正后的完整 JSON 对象，不要解释。\n\n"
                                f"{last_error}"
                            ),
                        }
                    )
        raise RuntimeError(
            f"MiniMax 连续 {MINIMAX_MAX_ATTEMPTS} 次输出不符合 {schema.__name__}：{last_error}"
        )


def build_provider(cfg: LLMConfig, settings: Settings) -> LLMProvider:
    if cfg.provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "缺少 Gemini API key，请设置环境变量 TENMIN_GEMINI_API_KEY"
            )
        return GeminiProvider(api_key=settings.gemini_api_key, model=cfg.model)
    if cfg.provider == "minimax":
        if not settings.minimax_api_key:
            raise RuntimeError(
                "缺少 MiniMax API key，请设置环境变量 TENMIN_MINIMAX_API_KEY"
            )
        return MiniMaxProvider(
            api_key=settings.minimax_api_key,
            model=cfg.model,
            base_url=cfg.base_url or MINIMAX_BASE_URL,
        )
    raise ValueError(f"不支持的 LLM provider：{cfg.provider}")
