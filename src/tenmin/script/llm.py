"""LLM provider。目前支持 Gemini、MiniMax，以及任意 OpenAI 兼容接口
（如 DeepSeek，通过 provider: openai_compatible 配置）。"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ValidationError

from tenmin.config import LLMConfig, Settings

MINIMAX_BASE_URL = "https://api.minimax.cn/v1"
OPENAI_COMPATIBLE_MAX_ATTEMPTS = 3
# read=None：实测 MiniMax-M3 处理 ~35k 字符 prompt 需要 561 秒，非流式模式下服务端在
# 这 561 秒里零字节返回，正好贴着旧的 600 秒读超时悬崖。流式下每个 SSE chunk 都会刷新
# 读活性，因此读超时交给 chunk 间隔而不是整体耗时（这里直接关掉固定读超时）。
OPENAI_COMPATIBLE_TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=120.0, pool=30.0)

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S)
_THINK_OPEN = re.compile(r"<think\b[^>]*>")
_FENCE = re.compile(r"^```(?:json)?[ \t]*\n?|\n?```[ \t]*$", re.M)


class LLMError(RuntimeError):
    """provider 抛出、且已经自带一句人话的异常的基类。

    继承 RuntimeError 而不是 Exception 是为了跟 script.validate.ScriptValidationError
    保持同一形态。**这个基类必须在 cli.py 的 PIPELINE_ERRORS 里**，否则用户看到的是
    一整页 traceback 而不是一行红字。
    """


class LLMResponseFormatError(LLMError, ValueError):
    """模型输出的**形态**不对：找不到 JSON 对象、`<think>` 没闭合、一个字都没返回。

    刻意同时继承 ValueError 两个理由：
    1. `_extract_json` 抛 ValueError 是既有契约（有测试锁着），而 cli.py 的
       PIPELINE_ERRORS 里那条 ValueError 也一直兜着它。
    2. 让「校验失败 → 回灌报错重试」那一层可以写成
       `except (ValidationError, LLMResponseFormatError)` 这种窄网。原来写的是
       `except (ValidationError, ValueError)`，`_extract_json` 之外任何偶发的
       ValueError（比如 provider 自己的 bug）都会被误判成「模型输出不合 schema」，
       白白触发两轮 ~35k 字符的昂贵重试，最后报一个完全指错方向的错。
    """


def _strip_reasoning(text: str) -> str:
    """MiniMax-M3 每次都在正文前吐一个 <think>…</think> 推理块，必须剥掉。

    只有闭合的块会被剥掉；剥完还剩一个裸的 `<think>` 说明这一段是**被截断的**推理，
    此时整段推理内容都还在文本里，交给 _extract_json 只会让它从推理里抓到第一个
    `{`，最后给出一个指向完全错误方向的 schema 报错。所以这里单独报错。
    """
    stripped = _THINK_BLOCK.sub("", text)
    if _THINK_OPEN.search(stripped):
        raise LLMResponseFormatError(
            "模型输出里有 <think> 却没有闭合的 </think>，推理块很可能被截断了"
            f"（收到 {len(text)} 字符）。这通常是流提前断掉或撞到输出上限，"
            f"原始输出开头：{text[:200]!r}"
        )
    return stripped.strip()


def _extract_json(text: str) -> str:
    """剥掉推理块与 markdown 围栅，再取第一个 { 到最后一个 }。

    模型偶尔会在 JSON 前后加解释文字，所以不能直接 json.loads 整个 content。
    """
    cleaned = _FENCE.sub("", _strip_reasoning(text)).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMResponseFormatError(f"模型返回里找不到 JSON 对象：{cleaned[:200]!r}")
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


class OpenAICompatibleProvider:
    """通用 OpenAI 兼容 provider：走 chat/completions 流式接口，
    schema 统一写进 prompt 文本（不依赖 response_format=json_schema），
    用 pydantic 校验 + 报错重试。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    def _extra_payload_fields(self) -> dict[str, Any]:
        """子类可覆写，往请求体里加自己专属的字段（比如 MiniMax 的 thinking）。"""
        return {}

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self._extra_payload_fields(),
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}

        last_error = ""
        async with httpx.AsyncClient(timeout=OPENAI_COMPATIBLE_TIMEOUT) as client:
            for _ in range(OPENAI_COMPATIBLE_MAX_ATTEMPTS):
                text = await self._stream_once(client, payload)
                if schema is None:
                    return _strip_reasoning(text)
                try:
                    return schema.model_validate_json(_extract_json(text))
                except (ValidationError, LLMResponseFormatError) as exc:
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
            f"{type(self).__name__} 连续 {OPENAI_COMPATIBLE_MAX_ATTEMPTS} 次"
            f"输出不符合 {schema.__name__}：{last_error}"
        )


class MiniMaxProvider(OpenAICompatibleProvider):
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
        thinking: str = "disabled",
    ) -> None:
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self.thinking = thinking

    def _extra_payload_fields(self) -> dict[str, Any]:
        return {"thinking": {"type": self.thinking}}


def build_provider(cfg: LLMConfig, settings: Settings) -> LLMProvider:
    """Settings 里的 API key 是 SecretStr（防止误打进日志），
    交给各 provider 前在这里统一 get_secret_value() 取明文。
    """
    if cfg.provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "缺少 Gemini API key，请设置环境变量 TENMIN_GEMINI_API_KEY"
            )
        return GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(), model=cfg.model
        )
    if cfg.provider == "minimax":
        if not settings.minimax_api_key:
            raise RuntimeError(
                "缺少 MiniMax API key，请设置环境变量 TENMIN_MINIMAX_API_KEY"
            )
        return MiniMaxProvider(
            api_key=settings.minimax_api_key.get_secret_value(),
            model=cfg.model,
            base_url=cfg.base_url or MINIMAX_BASE_URL,
            thinking=cfg.thinking,
        )
    if cfg.provider == "openai_compatible":
        if not settings.openai_compatible_api_key:
            raise RuntimeError(
                "缺少 API key，请设置环境变量 TENMIN_OPENAI_COMPATIBLE_API_KEY"
            )
        if not cfg.base_url:
            raise ValueError(
                "openai_compatible provider 必须在 project.yaml 里配置 llm.base_url"
            )
        return OpenAICompatibleProvider(
            api_key=settings.openai_compatible_api_key.get_secret_value(),
            model=cfg.model,
            base_url=cfg.base_url,
        )

    raise ValueError(f"不支持的 LLM provider：{cfg.provider}")
