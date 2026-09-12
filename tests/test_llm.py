import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tenmin.config import LLMConfig, Settings
from tenmin.models import LLMScript
from tenmin.script import llm
from tenmin.script.llm import (
    CONNECT_TIMEOUT_SECONDS,
    MINIMAX_BASE_URL,
    POOL_TIMEOUT_SECONDS,
    GeminiProvider,
    LLMBusinessError,
    LLMError,
    LLMFinishReasonError,
    LLMHTTPError,
    LLMProvider,
    LLMResponseFormatError,
    LLMSchemaError,
    LLMTransportError,
    MiniMaxProvider,
    OpenAICompatibleProvider,
    _extract_json,
    _strip_reasoning,
    build_provider,
)

from .fakes import FakeProvider


class Toy(BaseModel):
    value: int


def test_fake_provider_satisfies_protocol():
    provider: LLMProvider = FakeProvider([])
    assert hasattr(provider, "complete")


@pytest.mark.asyncio
async def test_fake_provider_parses_into_schema():
    provider = FakeProvider([{"value": 7}])
    result = await provider.complete("sys", "usr", Toy)
    assert isinstance(result, Toy)
    assert result.value == 7


@pytest.mark.asyncio
async def test_fake_provider_records_calls():
    provider = FakeProvider([{"value": 1}])
    await provider.complete("SYS", "USR", Toy)
    assert provider.calls[0]["system"] == "SYS"
    assert provider.calls[0]["user"] == "USR"


@pytest.mark.asyncio
async def test_fake_provider_exhausted_raises():
    provider = FakeProvider([])
    with pytest.raises(AssertionError):
        await provider.complete("s", "u", Toy)


def test_build_provider_returns_gemini():
    settings = Settings(gemini_api_key="fake-key")
    provider = build_provider(LLMConfig(provider="gemini", model="gemini-3.6-flash"), settings)
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-3.6-flash"


def test_build_provider_without_key_raises():
    settings = Settings(gemini_api_key=None)
    with pytest.raises(RuntimeError) as exc:
        build_provider(LLMConfig(provider="gemini"), settings)
    assert "TENMIN_GEMINI_API_KEY" in str(exc.value)


def test_build_provider_unknown_raises():
    settings = Settings(gemini_api_key="fake-key")
    cfg = LLMConfig(provider="gemini")
    object.__setattr__(cfg, "provider", "openai")  # 绕过 Literal 校验模拟未知 provider
    with pytest.raises(ValueError) as exc:
        build_provider(cfg, settings)
    assert "openai" in str(exc.value)


@pytest.mark.asyncio
async def test_gemini_provider_passes_schema_through(monkeypatch):
    """不联网：替换掉 client，只验证参数拼装。"""
    captured = {}

    class FakeModels:
        async def generate_content(self, *, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config

            class Resp:
                parsed = Toy(value=42)
                text = '{"value": 42}'

            return Resp()

    class FakeAio:
        models = FakeModels()

    class FakeClient:
        aio = FakeAio()

    provider = GeminiProvider(api_key="fake-key", model="gemini-3.6-flash")
    monkeypatch.setattr(provider, "_client", FakeClient())

    result = await provider.complete("SYS", "USR", Toy)
    assert result == Toy(value=42)
    assert captured["model"] == "gemini-3.6-flash"
    assert captured["contents"] == "USR"
    assert captured["config"].system_instruction == "SYS"
    assert captured["config"].response_mime_type == "application/json"
    assert captured["config"].response_schema is Toy


@pytest.mark.asyncio
async def test_gemini_provider_without_schema_returns_text(monkeypatch):
    class FakeModels:
        async def generate_content(self, *, model, contents, config):
            assert config.response_schema is None
            assert config.response_mime_type is None

            class Resp:
                parsed = None
                text = "纯文本回答"

            return Resp()

    class FakeAio:
        models = FakeModels()

    class FakeClient:
        aio = FakeAio()

    provider = GeminiProvider(api_key="fake-key", model="gemini-3.6-flash")
    monkeypatch.setattr(provider, "_client", FakeClient())

    assert await provider.complete("SYS", "USR") == "纯文本回答"


# --- Gemini 与 OpenAI 兼容路径共用同一层 schema 修复重试 ---


class _FakeGeminiResponse:
    """够用的 Gemini 响应替身。默认形态是「正常收尾」。"""

    def __init__(
        self,
        text: str | None,
        *,
        parsed=None,
        finish_reason: str | None = "STOP",
        block_reason: str | None = None,
        usage=None,
    ):
        self.text = text
        self.parsed = parsed
        self.usage_metadata = usage
        if block_reason is not None:
            self.candidates = []
            self.prompt_feedback = SimpleNamespace(block_reason=block_reason)
        else:
            self.candidates = [SimpleNamespace(finish_reason=finish_reason)]


def _fake_gemini(monkeypatch, provider, responses: list) -> list[dict]:
    """按序回放 Gemini 响应，记录每次调用的 model/contents/config。"""
    calls: list[dict] = []
    queue = list(responses)

    class FakeModels:
        async def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents, "config": config})
            assert queue, "假 Gemini 的响应队列已用尽"
            item = queue.pop(0)
            return item if isinstance(item, _FakeGeminiResponse) else _FakeGeminiResponse(item)

    monkeypatch.setattr(
        provider, "_client", SimpleNamespace(aio=SimpleNamespace(models=FakeModels()))
    )
    return calls


@pytest.mark.asyncio
async def test_gemini_retries_on_schema_violation(monkeypatch):
    """两个 provider 的健壮性原先严重不对称：OpenAI 兼容那条有 3 次带报错回灌的
    自修复，Gemini 一次都没有，字段名一错就直接把 pydantic 报错冒到用户脸上。"""
    provider = GeminiProvider(api_key="fake-key")
    calls = _fake_gemini(monkeypatch, provider, ['{"val": 1}', '{"value": 7}'])

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=7)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_gemini_gives_up_after_max_attempts(monkeypatch):
    attempts = LLMConfig().max_attempts
    provider = GeminiProvider(api_key="fake-key")
    calls = _fake_gemini(monkeypatch, provider, ['{"val": 1}'] * attempts)

    with pytest.raises(LLMSchemaError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "Toy" in str(exc.value)
    assert len(calls) == attempts


@pytest.mark.asyncio
async def test_gemini_repair_round_omits_original_prompt(monkeypatch):
    marked = "12 | 00:01:02,000 - 00:01:04,000 | 佐伯 | dialogue | 独特字幕轨标记串"
    provider = GeminiProvider(api_key="fake-key")
    calls = _fake_gemini(monkeypatch, provider, ['{"val": 1}', '{"value": 1}'])

    assert await provider.complete("SYS", marked, Toy) == Toy(value=1)

    assert marked in calls[0]["contents"]
    assert marked not in calls[1]["contents"]
    assert "value" in calls[1]["contents"]
    assert '{"val": 1}' in calls[1]["contents"]


@pytest.mark.asyncio
async def test_gemini_schema_error_carries_raw_output(monkeypatch):
    provider = GeminiProvider(api_key="fake-key")
    _fake_gemini(monkeypatch, provider, ['{"val": 1}'] * LLMConfig().max_attempts)

    with pytest.raises(LLMSchemaError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert exc.value.raw_output == '{"val": 1}'


@pytest.mark.asyncio
async def test_openai_compatible_schema_error_carries_raw_output(monkeypatch):
    """失败时最需要现场的就是原始模型输出，不能只留 last_error 的前 1500 字符。"""
    _fake_httpx(monkeypatch, ['<think>算</think>\n{"val": 1}'] * LLMConfig().max_attempts)
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMSchemaError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert exc.value.raw_output == '<think>算</think>\n{"val": 1}'


def test_llm_schema_error_is_an_llm_error():
    assert issubclass(LLMSchemaError, LLMError)


@pytest.mark.asyncio
async def test_gemini_max_tokens_finish_reason_is_diagnosable(monkeypatch):
    """输出是一份完整剧本 JSON（很大），撞默认 max output tokens 被截断是现实风险。
    原实现只会让 model_validate_json(None) 抛一个看不懂的 TypeError。"""
    provider = GeminiProvider(api_key="fake-key")
    calls = _fake_gemini(
        monkeypatch, provider, [_FakeGeminiResponse(None, finish_reason="MAX_TOKENS")]
    )

    with pytest.raises(LLMFinishReasonError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "MAX_TOKENS" in str(exc.value)
    assert "max_output_tokens" in str(exc.value)
    # 截断不会靠重试自己好，绝不能进 schema 修复循环
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_gemini_safety_finish_reason_is_diagnosable(monkeypatch):
    provider = GeminiProvider(api_key="fake-key")
    calls = _fake_gemini(
        monkeypatch, provider, [_FakeGeminiResponse(None, finish_reason="SAFETY")]
    )

    with pytest.raises(LLMFinishReasonError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "安全策略" in str(exc.value)
    assert "SAFETY" in str(exc.value)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_gemini_enum_finish_reason_is_read_by_name(monkeypatch):
    """真实 SDK 给的是 types.FinishReason 枚举，不是字符串。"""
    provider = GeminiProvider(api_key="fake-key")
    reason = SimpleNamespace(name="MAX_TOKENS")
    _fake_gemini(monkeypatch, provider, [_FakeGeminiResponse(None, finish_reason=reason)])

    with pytest.raises(LLMFinishReasonError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "MAX_TOKENS" in str(exc.value)


@pytest.mark.asyncio
async def test_gemini_prompt_level_block_is_diagnosable(monkeypatch):
    """prompt 自己被拦下时连 candidates 都没有。"""
    provider = GeminiProvider(api_key="fake-key")
    _fake_gemini(monkeypatch, provider, [_FakeGeminiResponse(None, block_reason="SAFETY")])

    with pytest.raises(LLMFinishReasonError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "block_reason" in str(exc.value)


@pytest.mark.asyncio
async def test_gemini_stop_finish_reason_passes_through(monkeypatch):
    provider = GeminiProvider(api_key="fake-key")
    _fake_gemini(
        monkeypatch,
        provider,
        [_FakeGeminiResponse('{"value": 8}', finish_reason="STOP")],
    )
    assert await provider.complete("SYS", "USR", Toy) == Toy(value=8)


@pytest.mark.asyncio
async def test_gemini_finish_reason_is_checked_without_schema_too(monkeypatch):
    provider = GeminiProvider(api_key="fake-key")
    _fake_gemini(monkeypatch, provider, [_FakeGeminiResponse(None, finish_reason="SAFETY")])

    with pytest.raises(LLMFinishReasonError):
        await provider.complete("SYS", "USR")


@pytest.mark.asyncio
async def test_gemini_empty_text_without_finish_reason_retries(monkeypatch):
    """text is None 但 finish_reason 正常：形态问题，值得重试一次。"""
    provider = GeminiProvider(api_key="fake-key")
    calls = _fake_gemini(
        monkeypatch,
        provider,
        [_FakeGeminiResponse(None), _FakeGeminiResponse('{"value": 2}')],
    )

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=2)
    assert len(calls) == 2


def test_llm_finish_reason_error_is_an_llm_error():
    assert issubclass(LLMFinishReasonError, LLMError)


# --- MiniMax ---


def test_strip_reasoning_removes_think_block():
    text = "<think>The user is asking me to output {\"ok\": true}</think>\n\n{\"ok\": true}"
    assert _strip_reasoning(text) == '{"ok": true}'


def test_strip_reasoning_removes_multiline_think_block():
    text = "<think>\n第一行推理\n第二行推理\n</think>\n{\"value\": 1}"
    assert _strip_reasoning(text) == '{"value": 1}'


def test_strip_reasoning_passthrough_without_think():
    assert _strip_reasoning("  裸文本  ") == "裸文本"


def test_extract_json_bare():
    assert _extract_json('{"value": 1}') == '{"value": 1}'


def test_extract_json_fenced():
    assert _extract_json('```json\n{"value": 1}\n```') == '{"value": 1}'


def test_extract_json_think_plus_fence_plus_prose():
    text = (
        "<think>先想一下\n再想一下</think>\n\n"
        "好的，下面是结果：\n"
        '```json\n{"value": 1}\n```\n'
        "以上就是我的输出。"
    )
    assert _extract_json(text) == '{"value": 1}'


def test_extract_json_without_brace_raises():
    with pytest.raises(ValueError) as exc:
        _extract_json("<think>只有推理没有 JSON</think>\n抱歉我无法回答")
    assert "找不到 JSON" in str(exc.value)


def test_extract_json_without_brace_raises_dedicated_type():
    """专属异常类型：`except ValueError` 那种大网会把偶发的 ValueError 误判成
    「模型输出不合 schema」，进而触发一轮 ~35k 字符的昂贵重试。"""
    with pytest.raises(LLMResponseFormatError):
        _extract_json("没有大括号")


def test_llm_response_format_error_is_both_llm_error_and_value_error():
    """继承 ValueError 是为了保住 _extract_json 抛 ValueError 子类的既有契约；
    继承 LLMError 是为了让 cli.py 的 PIPELINE_ERRORS 一网打尽。"""
    assert issubclass(LLMResponseFormatError, LLMError)
    assert issubclass(LLMResponseFormatError, ValueError)
    assert issubclass(LLMError, RuntimeError)


def test_strip_reasoning_unclosed_think_raises():
    """流被截断时没有 </think>，整段推理会留在文本里，_extract_json 很可能从推理
    内容里抓到 `{`，最后给出一个指向完全错误方向的 schema 报错。"""
    with pytest.raises(LLMResponseFormatError) as exc:
        _strip_reasoning('<think>我先想想，大概是 {"value": 1} 这样')
    assert "</think>" in str(exc.value)


def test_extract_json_unclosed_think_raises_instead_of_grabbing_reasoning():
    with pytest.raises(LLMResponseFormatError) as exc:
        _extract_json('<think>大概是 {"val": 1} 吧')
    assert "</think>" in str(exc.value)


def test_strip_reasoning_closed_block_after_unclosed_prefix_is_still_an_error():
    """`</think>` 出现在 `<think>` 之前不算闭合。"""
    with pytest.raises(LLMResponseFormatError):
        _strip_reasoning('</think><think>{"value": 1}')


def test_minimax_provider_satisfies_protocol():
    provider = MiniMaxProvider(api_key="fake-key")
    assert isinstance(provider, LLMProvider)


def test_minimax_provider_defaults():
    provider = MiniMaxProvider(api_key="fake-key")
    assert provider.model == "MiniMax-M3"
    assert provider.base_url == MINIMAX_BASE_URL
    assert provider.thinking == "disabled"


def test_minimax_provider_honours_thinking_override():
    provider = MiniMaxProvider(api_key="fake-key", thinking="adaptive")
    assert provider.thinking == "adaptive"


def test_minimax_provider_strips_trailing_slash_from_base_url():
    provider = MiniMaxProvider(api_key="k", base_url="https://example.test/v1/")
    assert provider.base_url == "https://example.test/v1"


def test_minimax_schema_prompt_embeds_field_names():
    provider = MiniMaxProvider(api_key="fake-key")
    prompt = provider._schema_prompt("原始提示词", LLMScript)
    assert prompt.startswith("原始提示词")
    for field in ("beats", "narration", "anchor_lines", "original_audio"):
        assert field in prompt
    assert "JSON Schema" in prompt


def test_minimax_schema_prompt_keeps_cjk_readable():
    """ensure_ascii=False：schema 里的中文枚举/描述不应被转义成 \\uXXXX。"""
    provider = MiniMaxProvider(api_key="fake-key")
    prompt = provider._schema_prompt("u", LLMScript)
    assert "\\u" not in prompt


def test_build_provider_returns_minimax():
    settings = Settings(minimax_api_key="fake-key")
    cfg = LLMConfig(provider="minimax", model="MiniMax-M3")
    provider = build_provider(cfg, settings)
    assert isinstance(provider, MiniMaxProvider)
    assert provider.model == "MiniMax-M3"
    assert provider.base_url == MINIMAX_BASE_URL
    assert provider.thinking == "disabled"


def test_build_provider_minimax_honours_thinking_override():
    settings = Settings(minimax_api_key="fake-key")
    cfg = LLMConfig(provider="minimax", thinking="adaptive")
    provider = build_provider(cfg, settings)
    assert provider.thinking == "adaptive"


def test_build_provider_minimax_honours_base_url_override():
    settings = Settings(minimax_api_key="fake-key")
    cfg = LLMConfig(provider="minimax", model="m1", base_url="https://proxy.test/v1")
    provider = build_provider(cfg, settings)
    assert isinstance(provider, MiniMaxProvider)
    assert provider.model == "m1"
    assert provider.base_url == "https://proxy.test/v1"


def test_build_provider_minimax_without_key_raises():
    settings = Settings(minimax_api_key=None)
    with pytest.raises(RuntimeError) as exc:
        build_provider(LLMConfig(provider="minimax"), settings)
    assert "TENMIN_MINIMAX_API_KEY" in str(exc.value)


def test_build_provider_returns_openai_compatible():
    settings = Settings(openai_compatible_api_key="fake-key")
    cfg = LLMConfig(
        provider="openai_compatible",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )
    provider = build_provider(cfg, settings)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "deepseek-chat"
    assert provider.base_url == "https://api.deepseek.com/v1"


def test_build_provider_openai_compatible_without_key_raises():
    settings = Settings(openai_compatible_api_key=None)
    cfg = LLMConfig(
        provider="openai_compatible", base_url="https://api.deepseek.com/v1"
    )
    with pytest.raises(RuntimeError) as exc:
        build_provider(cfg, settings)
    assert "TENMIN_OPENAI_COMPATIBLE_API_KEY" in str(exc.value)


def test_build_provider_openai_compatible_without_base_url_raises():
    settings = Settings(openai_compatible_api_key="fake-key")
    cfg = LLMConfig(provider="openai_compatible", base_url=None)
    with pytest.raises(ValueError) as exc:
        build_provider(cfg, settings)
    assert "base_url" in str(exc.value)


def _sse(*payloads: dict, done: bool = True, extra_lines: tuple[str, ...] = ()) -> str:
    """按 MiniMax 的 SSE 线格式拼一个响应体。payloads 是每个 chunk 的原始 JSON。"""
    lines = [f"data: {json.dumps(p, ensure_ascii=False)}" for p in payloads]
    lines.extend(extra_lines)
    if done:
        lines.append("data: [DONE]")
    # SSE 事件之间是空行分隔，httpx 的 aiter_lines 会原样把空行交给我们。
    return "\n\n".join(lines) + "\n\n"


def _delta(content: str) -> dict:
    return {"choices": [{"delta": {"content": content}}]}


def _sse_from_chunks(*chunks: str) -> str:
    return _sse(*[_delta(c) for c in chunks])


def _mock_httpx(monkeypatch, bodies: list) -> list[dict]:
    """用 httpx.MockTransport 按序回放 SSE 响应体。

    刻意保留真实的 httpx.AsyncClient（只注入 transport），这样 SSE 分行/解码走的是
    httpx 自己的 aiter_lines 实现，测到的是生产路径而不是手写的假迭代器。

    队列里每一项可以是：
    - str：HTTP 200 + 这段正文
    - (status, text)：指定状态码
    - (status, text, headers)：再指定响应头（比如 Retry-After）
    - Exception 实例：这一次请求直接抛它（模拟连接失败/读超时）
    """
    import httpx

    real_client_cls = httpx.AsyncClient
    requests: list[dict] = []
    queue = list(bodies)

    def handler(request: httpx.Request) -> httpx.Response:
        # request.content 是当次序列化后的字节，天然是快照；provider 内部的 messages
        # 会被后续重试就地追加，所以只能从这里取"当次"请求内容。
        requests.append(
            {
                "url": str(request.url),
                "headers": request.headers,
                "json": json.loads(request.content),
            }
        )
        assert queue, "假 httpx 的响应队列已用尽"
        body = queue.pop(0)
        if isinstance(body, BaseException):
            raise body
        headers: dict[str, str] = {}
        if isinstance(body, tuple):
            if len(body) == 3:
                status, text, headers = body
            else:
                status, text = body
        else:
            status, text = 200, body
        return httpx.Response(status, content=text.encode("utf-8"), headers=headers)

    def factory(**kwargs):
        requests.append({"__init__": kwargs})
        return real_client_cls(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return requests


def _fake_httpx(monkeypatch, contents: list[str]) -> list[dict]:
    """老签名的便捷包装：每个 content 作为单 chunk 的流式响应回放。"""
    return _mock_httpx(monkeypatch, [_sse_from_chunks(c) for c in contents])


# --- OpenAICompatibleProvider ---


def test_openai_compatible_provider_defaults():
    provider = OpenAICompatibleProvider(
        api_key="fake-key",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )
    assert provider.model == "deepseek-chat"
    assert provider.base_url == "https://api.deepseek.com/v1"


def test_openai_compatible_provider_strips_trailing_slash_from_base_url():
    provider = OpenAICompatibleProvider(
        api_key="fake-key",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1/",
    )
    assert provider.base_url == "https://api.deepseek.com/v1"


def test_openai_compatible_provider_extra_payload_fields_defaults_empty():
    provider = OpenAICompatibleProvider(
        api_key="fake-key",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )
    assert provider._extra_payload_fields() == {}


@pytest.mark.asyncio
async def test_openai_compatible_complete_first_try_sends_no_thinking_field(monkeypatch):
    log = _fake_httpx(monkeypatch, ['<think>算一下</think>\n```json\n{"value": 42}\n```'])
    provider = OpenAICompatibleProvider(
        api_key="secret", model="deepseek-chat", base_url="https://api.deepseek.com/v1"
    )

    result = await provider.complete("SYS", "USR", Toy)

    assert result == Toy(value=42)
    posts = [r for r in log if "url" in r]
    assert len(posts) == 1
    assert posts[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert posts[0]["headers"]["Authorization"] == "Bearer secret"
    body = posts[0]["json"]
    assert body["model"] == "deepseek-chat"
    assert body["response_format"] == {"type": "json_object"}
    assert "thinking" not in body
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"].startswith("USR")


@pytest.mark.asyncio
async def test_minimax_complete_first_try(monkeypatch):
    log = _fake_httpx(monkeypatch, ['<think>算一下</think>\n```json\n{"value": 42}\n```'])
    provider = MiniMaxProvider(api_key="secret", model="MiniMax-M3")

    result = await provider.complete("SYS", "USR", Toy)

    assert result == Toy(value=42)
    posts = [r for r in log if "url" in r]
    assert len(posts) == 1
    assert posts[0]["url"] == f"{MINIMAX_BASE_URL}/chat/completions"
    assert posts[0]["headers"]["Authorization"] == "Bearer secret"
    body = posts[0]["json"]
    assert body["model"] == "MiniMax-M3"
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"].startswith("USR")


@pytest.mark.asyncio
async def test_minimax_complete_honours_adaptive_thinking_override(monkeypatch):
    log = _fake_httpx(monkeypatch, ['<think>算一下</think>\n```json\n{"value": 42}\n```'])
    provider = MiniMaxProvider(api_key="secret", model="MiniMax-M3", thinking="adaptive")

    await provider.complete("SYS", "USR", Toy)

    posts = [r for r in log if "url" in r]
    assert posts[0]["json"]["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_minimax_complete_retries_on_wrong_field_names(monkeypatch):
    """核心保护：第一次字段名全错，带着校验报错重试，第二次成功。"""
    log = _fake_httpx(
        monkeypatch,
        [
            # MiniMax 实测行为：外层多套一层包裹 + 字段名自己编
            '<think>我来输出</think>\n{"script": {"val": 42}}',
            '<think>修正一下</think>\n```json\n{"value": 42}\n```',
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    result = await provider.complete("SYS", "USR", Toy)

    assert isinstance(result, Toy)
    assert result.value == 42

    posts = [r for r in log if "url" in r]
    assert len(posts) == 2

    first = posts[0]["json"]["messages"]
    second = posts[1]["json"]["messages"]
    assert len(first) == 2
    # 重试请求带上了 system + 只含 schema 的纠错指令 + 助手的错误输出 + 携带报错的指令
    assert len(second) == 4
    assert second[0] == first[0]
    # 纠错轮**不重发**首轮那份正文（见 test_..._repair_round_omits_original_prompt）
    assert second[1] != first[1]
    assert second[2]["role"] == "assistant"
    assert '{"script": {"val": 42}}' in second[2]["content"]
    assert second[3]["role"] == "user"
    assert "不符合 schema" in second[3]["content"]
    assert "value" in second[3]["content"]


@pytest.mark.asyncio
async def test_minimax_complete_gives_up_after_max_attempts(monkeypatch):
    attempts = LLMConfig().max_attempts
    log = _fake_httpx(monkeypatch, ['{"val": 1}'] * attempts)
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMSchemaError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "Toy" in str(exc.value)
    assert str(attempts) in str(exc.value)
    assert len([r for r in log if "url" in r]) == attempts


@pytest.mark.asyncio
async def test_openai_compatible_honours_max_attempts_override(monkeypatch):
    """max_attempts 从 LLMConfig 接线进来，不再是模块级硬编码常量。"""
    log = _fake_httpx(monkeypatch, ['{"val": 1}'] * 5)
    provider = OpenAICompatibleProvider(
        api_key="k", model="m", base_url="https://x.test/v1", max_attempts=2
    )

    with pytest.raises(LLMSchemaError):
        await provider.complete("SYS", "USR", Toy)

    assert len([r for r in log if "url" in r]) == 2


@pytest.mark.asyncio
async def test_openai_compatible_repair_round_omits_original_prompt(monkeypatch):
    """纠错轮只发 schema + 报错 + 坏输出，**不重发字幕轨**。

    首轮的 user 消息是 ~35k 字符（对白轨 + 高能点 + few-shot 示例），原实现把它
    留在 messages[1] 里逐轮重发，第 3 次请求 ≈ 完整 prompt + 2 份坏输出。
    """
    marked = "12 | 00:01:02,000 - 00:01:04,000 | 佐伯 | dialogue | 独特字幕轨标记串"
    log = _fake_httpx(monkeypatch, ['{"val": 1}', '{"value": 1}'])
    provider = OpenAICompatibleProvider(
        api_key="k", model="m", base_url="https://x.test/v1"
    )

    assert await provider.complete("SYS", marked, Toy) == Toy(value=1)

    posts = [r for r in log if "url" in r]
    assert marked in json.dumps(posts[0]["json"], ensure_ascii=False)
    assert marked not in json.dumps(posts[1]["json"], ensure_ascii=False)
    # 但 schema 本身必须还在，否则模型无从修正
    assert "value" in json.dumps(posts[1]["json"], ensure_ascii=False)


@pytest.mark.asyncio
async def test_openai_compatible_repair_round_truncates_bad_output(monkeypatch):
    """坏输出也要截断：整份坏剧本回灌一遍照样是几万 token。"""
    from tenmin.script.llm import REPAIR_OUTPUT_MAX_CHARS

    junk = "x" * (REPAIR_OUTPUT_MAX_CHARS + 5000)
    log = _fake_httpx(monkeypatch, [f'{{"val": "{junk}"}}', '{"value": 1}'])
    provider = OpenAICompatibleProvider(
        api_key="k", model="m", base_url="https://x.test/v1"
    )

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=1)

    posts = [r for r in log if "url" in r]
    echoed = posts[1]["json"]["messages"][2]["content"]
    assert len(echoed) < len(junk)
    assert "已截断" in echoed or "只保留前" in echoed


@pytest.mark.asyncio
async def test_minimax_complete_missing_json_also_retries(monkeypatch):
    """_extract_json 抛的 ValueError 也要走重试，不能直接冒出去。"""
    log = _fake_httpx(
        monkeypatch,
        ["<think>想不出来</think>\n抱歉，我拒绝回答。", '{"value": 3}'],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=3)
    assert len([r for r in log if "url" in r]) == 2


@pytest.mark.asyncio
async def test_minimax_complete_without_schema_returns_stripped_text(monkeypatch):
    log = _fake_httpx(monkeypatch, ["<think>推理一下</think>\n\n纯文本回答"])
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR") == "纯文本回答"

    posts = [r for r in log if "url" in r]
    assert len(posts) == 1
    assert "response_format" not in posts[0]["json"]
    # 无 schema 时用户消息原样透传，不注入 schema 段
    assert posts[0]["json"]["messages"][1]["content"] == "USR"


# --- HTTP 200 + 业务错误 ---


@pytest.mark.asyncio
async def test_business_error_in_a_200_stream_fails_immediately(monkeypatch, sleeps):
    """MiniMax 的业务错误是 HTTP 200 + base_resp.status_code != 0（1002 限流 /
    1008 余额不足 / 2013 参数错）。

    原实现里 _sse_delta 对「没有 choices 的数据行」一律返回空串，于是整条流一个
    delta 都没有 → 累积文本为空 → _extract_json("") 抛「找不到 JSON」→ 白跑 3 次
    ~35k 字符的 prompt，最后给出一个指向完全错误方向的 schema 报错。
    """
    log = _mock_httpx(
        monkeypatch,
        [_sse({"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}})],
    )
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMBusinessError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert exc.value.code == 1008
    assert "1008" in str(exc.value)
    assert "insufficient balance" in str(exc.value)
    # 余额不足重试是纯浪费，也绝不能进 schema 修复循环
    assert len([r for r in log if "url" in r]) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_business_rate_limit_code_is_retryable(monkeypatch, sleeps):
    """1002 是 429 的业务码版本，跟 429 一样值得退避重试。"""
    log = _mock_httpx(
        monkeypatch,
        [
            _sse({"base_resp": {"status_code": 1002, "status_msg": "rate limit"}}),
            _sse_from_chunks('{"value": 6}'),
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=6)
    assert len([r for r in log if "url" in r]) == 2
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_business_rate_limit_gives_up_after_transport_max_attempts(monkeypatch, sleeps):
    attempts = LLMConfig().transport_max_attempts
    body = _sse({"base_resp": {"status_code": 1002, "status_msg": "rate limit"}})
    log = _mock_httpx(monkeypatch, [body] * attempts)
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMBusinessError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert str(attempts) in str(exc.value)
    assert len([r for r in log if "url" in r]) == attempts


@pytest.mark.asyncio
async def test_base_resp_status_code_zero_is_not_an_error(monkeypatch):
    """真实的成功 chunk 每一条都带 base_resp.status_code == 0，不能误判成错误。"""
    log = _mock_httpx(
        monkeypatch,
        [
            _sse(
                {
                    "choices": [{"delta": {"content": '{"value": 4}'}}],
                    "base_resp": {"status_code": 0, "status_msg": ""},
                }
            )
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=4)
    assert len([r for r in log if "url" in r]) == 1


@pytest.mark.asyncio
async def test_openai_style_error_field_in_a_200_stream_fails_immediately(monkeypatch, sleeps):
    """通用 OpenAI 兼容实现把错误塞进 data 行的 error 字段，同样是 HTTP 200。"""
    log = _mock_httpx(
        monkeypatch,
        [_sse({"error": {"code": "context_length_exceeded", "message": "prompt 太长"}})],
    )
    provider = OpenAICompatibleProvider(
        api_key="k", model="m", base_url="https://x.test/v1"
    )

    with pytest.raises(LLMBusinessError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "context_length_exceeded" in str(exc.value)
    assert "prompt 太长" in str(exc.value)
    assert len([r for r in log if "url" in r]) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_malformed_sse_lines_are_reported_in_the_final_error(monkeypatch):
    """整条流全畸形时原实现静默 return ""，一点痕迹都不留。"""
    body = "data: not-json\n\ndata: also{not}json\n\ndata: [DONE]\n\n"
    log = _mock_httpx(monkeypatch, [body] * LLMConfig().max_attempts)
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMSchemaError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "畸形" in str(exc.value)
    assert "6" in str(exc.value)  # 3 轮 x 2 条
    assert len([r for r in log if "url" in r]) == LLMConfig().max_attempts


@pytest.mark.asyncio
async def test_transport_retry_count_is_reported_in_the_final_error(monkeypatch, sleeps):
    log = _mock_httpx(
        monkeypatch,
        [(429, "slow down"), _sse_from_chunks('{"val": 1}')]
        + [_sse_from_chunks('{"val": 1}')] * 2,
    )
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMSchemaError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "传输层重试 1 次" in str(exc.value)
    assert len([r for r in log if "url" in r]) == 4


def test_llm_business_error_is_an_llm_error():
    assert issubclass(LLMBusinessError, LLMError)


# --- MiniMax 流式 ---


@pytest.mark.asyncio
async def test_minimax_stream_accumulates_chunks(monkeypatch):
    """多个 data: 行按序累加，且请求体里 stream=True。"""
    log = _mock_httpx(
        monkeypatch,
        [_sse_from_chunks('{"val', 'ue": ', "42}")],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=42)

    posts = [r for r in log if "url" in r]
    assert len(posts) == 1
    assert posts[0]["json"]["stream"] is True


@pytest.mark.asyncio
async def test_minimax_stream_think_block_split_across_chunks(monkeypatch):
    """</think> 落在 chunk 边界中间也必须能正确剥除。"""
    chunks = ["<thi", "nk>先推理一下\n再推理一下</thi", 'nk>\n{"value"', ": 7}"]
    log = _mock_httpx(monkeypatch, [_sse_from_chunks(*chunks)])
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=7)
    assert len([r for r in log if "url" in r]) == 1


@pytest.mark.asyncio
async def test_minimax_stream_without_schema_returns_stripped_text(monkeypatch):
    log = _mock_httpx(
        monkeypatch,
        [_sse_from_chunks("<think>推理", "一下</think>\n\n纯文本", "回答")],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR") == "纯文本回答"
    posts = [r for r in log if "url" in r]
    assert posts[0]["json"]["stream"] is True
    assert "response_format" not in posts[0]["json"]


@pytest.mark.asyncio
async def test_minimax_stream_skips_malformed_lines(monkeypatch):
    """畸形 data: 行与心跳空行都要跳过，不能崩。"""
    body = (
        'data: {"choices": [{"delta": {"content": "{\\"value\\": "}}]}\n\n'
        "data: not-json\n\n"
        "\n"
        ": ping\n\n"
        'data: {"choices": [{"delta": {"content": "9}"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    log = _mock_httpx(monkeypatch, [body])
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=9)
    assert len([r for r in log if "url" in r]) == 1


@pytest.mark.asyncio
async def test_minimax_stream_tolerates_delta_without_content(monkeypatch):
    """最后一个 chunk 常带 finish_reason 而 delta 为空；content 也可能是 null。"""
    log = _mock_httpx(
        monkeypatch,
        [
            _sse(
                {"choices": [{"delta": {"role": "assistant"}}]},
                _delta('{"value": 5}'),
                {"choices": [{"delta": {"content": None}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": []},
            )
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=5)
    assert len([r for r in log if "url" in r]) == 1


@pytest.mark.asyncio
async def test_minimax_stream_retries_on_wrong_field_names(monkeypatch):
    """流式下的重试：第一次真实失败形状，第二次修正；messages 是超集。"""
    log = _mock_httpx(
        monkeypatch,
        [
            _sse_from_chunks("<think>我来输出</think>\n", '{"script": ', '{"val": 42}}'),
            _sse_from_chunks("<think>修正一下</think>\n", '```json\n{"value": 42}\n```'),
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=42)

    posts = [r for r in log if "url" in r]
    assert len(posts) == 2
    first = posts[0]["json"]["messages"]
    second = posts[1]["json"]["messages"]
    assert len(first) == 2
    assert len(second) == 4
    assert second[0] == first[0]
    assert second[1] != first[1]
    assert second[2]["role"] == "assistant"
    assert '{"script": {"val": 42}}' in second[2]["content"]
    assert second[3]["role"] == "user"
    assert "不符合 schema" in second[3]["content"]
    assert posts[1]["json"]["stream"] is True


@pytest.mark.asyncio
async def test_openai_compatible_timeout_mapping(monkeypatch):
    """超时的四元组映射。

    read 从原来的 None 换成有限值是这次改动的核心：read=None 关掉了 chunk 间隔的
    活性检测，服务端吐了首字节之后 stall 就永久挂着。read 是「**两个 chunk 之间**
    最多等多久」，不是整段生成时长，所以 120 秒不会误杀长思考。
    """
    log = _mock_httpx(monkeypatch, [_sse_from_chunks('{"value": 1}')])
    provider = MiniMaxProvider(api_key="secret")

    await provider.complete("SYS", "USR", Toy)

    timeout = [r for r in log if "__init__" in r][0]["__init__"]["timeout"]
    assert timeout.read == pytest.approx(LLMConfig().read_timeout_seconds)
    assert timeout.write == pytest.approx(LLMConfig().timeout_seconds)
    assert timeout.connect == pytest.approx(CONNECT_TIMEOUT_SECONDS)
    assert timeout.pool == pytest.approx(POOL_TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_openai_compatible_timeout_fields_are_configurable(monkeypatch):
    log = _mock_httpx(monkeypatch, [_sse_from_chunks('{"value": 1}')])
    provider = OpenAICompatibleProvider(
        api_key="k",
        model="m",
        base_url="https://x.test/v1",
        timeout_seconds=7.0,
        read_timeout_seconds=11.0,
    )

    await provider.complete("SYS", "USR", Toy)

    timeout = [r for r in log if "__init__" in r][0]["__init__"]["timeout"]
    assert timeout.write == pytest.approx(7.0)
    assert timeout.read == pytest.approx(11.0)


@pytest.mark.asyncio
async def test_openai_compatible_total_timeout_aborts_a_stalled_request(monkeypatch):
    """read 只管 chunk 间隔，所以还要一个整次请求的总截止，否则「每 100 秒吐一个
    字节」能让一次请求挂到天荒地老。"""
    import httpx as _httpx

    real_client_cls = _httpx.AsyncClient

    async def handler(request):
        await asyncio.sleep(1.0)
        return _httpx.Response(200, content=b"")

    monkeypatch.setattr(
        _httpx,
        "AsyncClient",
        lambda **kw: real_client_cls(transport=_httpx.MockTransport(handler), **kw),
    )
    provider = OpenAICompatibleProvider(
        api_key="k", model="m", base_url="https://x.test/v1", total_timeout_seconds=0.05
    )

    with pytest.raises(LLMTransportError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "0.05" in str(exc.value)


# --- 传输层重试与退避 ---


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """把退避的 sleep 换成 no-op 并记录时长，抖动固定成 0 让序列可断言。

    绝不真睡：退避基数 1 秒、上限 30 秒，真睡一遍这组测试要跑几分钟。
    """
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(llm, "_sleep", fake_sleep)
    monkeypatch.setattr(llm, "_rand", lambda: 0.0)
    return recorded


def test_backoff_is_exponential_and_capped(monkeypatch):
    monkeypatch.setattr(llm, "_rand", lambda: 0.0)
    delays = [llm._backoff_delay(n) for n in range(1, 9)]
    assert delays[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert all(d <= llm.BACKOFF_MAX_SECONDS for d in delays)
    assert delays[-1] == pytest.approx(llm.BACKOFF_MAX_SECONDS)


def test_backoff_jitter_is_multiplicative_and_bounded(monkeypatch):
    """抖动是乘性的 [1, 1+ratio)，永远不缩短退避、也不会突破上限太多。"""
    monkeypatch.setattr(llm, "_rand", lambda: 1.0)
    assert llm._backoff_delay(1) == pytest.approx(1.0 * (1 + llm.BACKOFF_JITTER_RATIO))
    assert llm._backoff_delay(3) == pytest.approx(4.0 * (1 + llm.BACKOFF_JITTER_RATIO))


def test_backoff_honours_retry_after_without_jitter(monkeypatch):
    monkeypatch.setattr(llm, "_rand", lambda: 1.0)
    assert llm._backoff_delay(1, retry_after=17.0) == pytest.approx(17.0)
    # 恶意/离谱的 Retry-After 要夹住，不然一次 429 能把整条流水线钉死几小时
    assert llm._backoff_delay(1, retry_after=99999.0) == pytest.approx(
        llm.RETRY_AFTER_MAX_SECONDS
    )


def test_parse_retry_after_accepts_seconds_and_ignores_http_date():
    assert llm._parse_retry_after("12") == pytest.approx(12.0)
    assert llm._parse_retry_after(" 2.5 ") == pytest.approx(2.5)
    assert llm._parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") is None
    assert llm._parse_retry_after(None) is None
    assert llm._parse_retry_after("") is None


@pytest.mark.asyncio
async def test_openai_compatible_retries_429_then_succeeds(monkeypatch, sleeps):
    """原实现里 429 抛的 HTTPStatusError 在 try 之外，直接冲出重试循环——
    OPENAI_COMPATIBLE_MAX_ATTEMPTS 只覆盖 schema 校验失败，传输层零重试零退避。"""
    log = _mock_httpx(
        monkeypatch,
        [
            (429, '{"base_resp": {"status_msg": "rate limited"}}'),
            (503, "upstream busy"),
            _sse_from_chunks('{"value": 42}'),
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=42)
    assert len([r for r in log if "url" in r]) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_openai_compatible_honours_retry_after_header(monkeypatch, sleeps):
    log = _mock_httpx(
        monkeypatch,
        [
            (429, "slow down", {"Retry-After": "9"}),
            _sse_from_chunks('{"value": 1}'),
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=1)
    assert sleeps == [9.0]
    assert len([r for r in log if "url" in r]) == 2


@pytest.mark.asyncio
async def test_openai_compatible_gives_up_after_transport_max_attempts(monkeypatch, sleeps):
    attempts = LLMConfig().transport_max_attempts
    log = _mock_httpx(monkeypatch, [(429, "rate limited")] * attempts)
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMHTTPError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert exc.value.status_code == 429
    assert len([r for r in log if "url" in r]) == attempts
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_openai_compatible_does_not_retry_on_401(monkeypatch, sleeps):
    """401/403/400 重试是纯浪费：key 不会在 1 秒后自己变对。"""
    log = _mock_httpx(monkeypatch, [(401, '{"error": {"message": "invalid api key"}}')])
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMHTTPError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert exc.value.status_code == 401
    assert len([r for r in log if "url" in r]) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_http_error_message_carries_status_and_body(monkeypatch, sleeps):
    """原实现为了让错误体可读而 aread()，但 httpx 的 HTTPStatusError 消息不含 body，
    body 只挂在 exc.response 上，而上层没有任何地方读它——用户看到的仍然只是
    「429 Too Many Requests for url ...」。"""
    _mock_httpx(monkeypatch, [(402, '{"base_resp":{"status_code":1008,"msg":"余额不足"}}')])
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMHTTPError) as exc:
        await provider.complete("SYS", "USR", Toy)

    message = str(exc.value)
    assert "402" in message
    assert "余额不足" in message
    assert "chat/completions" in message


@pytest.mark.asyncio
async def test_http_error_message_truncates_a_huge_body(monkeypatch, sleeps):
    _mock_httpx(monkeypatch, [(400, "x" * 50000)])
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMHTTPError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert len(str(exc.value)) < 3000


@pytest.mark.asyncio
async def test_openai_compatible_retries_transport_errors(monkeypatch, sleeps):
    import httpx as _httpx

    log = _mock_httpx(
        monkeypatch,
        [
            _httpx.ConnectError("[Errno 61] Connection refused"),
            _httpx.ReadTimeout("chunk 间隔超时"),
            _httpx.RemoteProtocolError("服务端提前断流"),
            _sse_from_chunks('{"value": 3}'),
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=3)
    assert len([r for r in log if "url" in r]) == 4
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_transport_error_message_survives_retry_exhaustion(monkeypatch, sleeps):
    import httpx as _httpx

    attempts = LLMConfig().transport_max_attempts
    _mock_httpx(monkeypatch, [_httpx.ConnectError("Connection refused")] * attempts)
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(LLMTransportError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "Connection refused" in str(exc.value)
    assert str(attempts) in str(exc.value)
    assert isinstance(exc.value.__cause__, _httpx.ConnectError)


@pytest.mark.asyncio
async def test_transport_max_attempts_is_configurable(monkeypatch, sleeps):
    import httpx as _httpx

    log = _mock_httpx(monkeypatch, [_httpx.ConnectError("nope")] * 5)
    provider = OpenAICompatibleProvider(
        api_key="k", model="m", base_url="https://x.test/v1", transport_max_attempts=1
    )

    with pytest.raises(LLMTransportError):
        await provider.complete("SYS", "USR", Toy)

    assert len([r for r in log if "url" in r]) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_openai_compatible_reuses_one_client_across_rounds(monkeypatch, sleeps):
    """provider 持有 client：原实现每次 complete 新建一个 AsyncClient，
    重复 TLS 握手、连接池完全不复用。"""
    log = _mock_httpx(
        monkeypatch,
        [
            (429, "slow down"),
            _sse_from_chunks('{"val": 1}'),
            _sse_from_chunks('{"value": 1}'),
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=1)
    assert len([r for r in log if "url" in r]) == 3
    assert len([r for r in log if "__init__" in r]) == 1

    await provider.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_aclose_is_idempotent(monkeypatch):
    _mock_httpx(monkeypatch, [_sse_from_chunks('{"value": 1}')])
    provider = MiniMaxProvider(api_key="secret")

    await provider.aclose()  # 还没建过 client
    await provider.complete("SYS", "USR", Toy)
    await provider.aclose()
    await provider.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_rebuilds_client_after_aclose(monkeypatch):
    log = _mock_httpx(
        monkeypatch, [_sse_from_chunks('{"value": 1}'), _sse_from_chunks('{"value": 2}')]
    )
    provider = MiniMaxProvider(api_key="secret")

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=1)
    await provider.aclose()
    assert await provider.complete("SYS", "USR", Toy) == Toy(value=2)
    assert len([r for r in log if "__init__" in r]) == 2

