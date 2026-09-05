import copy

import pytest
from pydantic import BaseModel

from tenmin.config import LLMConfig, Settings
from tenmin.models import LLMScript
from tenmin.script.llm import (
    MINIMAX_BASE_URL,
    MINIMAX_MAX_ATTEMPTS,
    GeminiProvider,
    LLMProvider,
    MiniMaxProvider,
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


def test_minimax_provider_satisfies_protocol():
    provider = MiniMaxProvider(api_key="fake-key")
    assert isinstance(provider, LLMProvider)


def test_minimax_provider_defaults():
    provider = MiniMaxProvider(api_key="fake-key")
    assert provider.model == "MiniMax-M3"
    assert provider.base_url == MINIMAX_BASE_URL


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


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def _fake_httpx(monkeypatch, contents: list[str]) -> list[dict]:
    """把 httpx.AsyncClient 换成按序回放 contents 的假客户端；返回请求记录（深拷贝）。"""
    import httpx

    requests: list[dict] = []
    queue = list(contents)

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            requests.append({"__init__": kwargs})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def post(self, url, *, headers, json):
            # payload["messages"] 与 provider 内部的 messages 是同一对象，会被后续
            # 重试就地追加，所以必须深拷贝才能断言"当次"请求的内容。
            requests.append(
                {"url": url, "headers": dict(headers), "json": copy.deepcopy(json)}
            )
            assert queue, "假 httpx 的响应队列已用尽"
            return _FakeResponse(queue.pop(0))

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return requests


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
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"].startswith("USR")


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
    # 重试请求带上了原始两条 + 助手的错误输出 + 携带报错的纠正指令
    assert len(second) == 4
    assert second[:2] == first
    assert second[2]["role"] == "assistant"
    assert '{"script": {"val": 42}}' in second[2]["content"]
    assert second[3]["role"] == "user"
    assert "不符合 schema" in second[3]["content"]
    assert "value" in second[3]["content"]


@pytest.mark.asyncio
async def test_minimax_complete_gives_up_after_max_attempts(monkeypatch):
    log = _fake_httpx(monkeypatch, ['{"val": 1}'] * MINIMAX_MAX_ATTEMPTS)
    provider = MiniMaxProvider(api_key="secret")

    with pytest.raises(RuntimeError) as exc:
        await provider.complete("SYS", "USR", Toy)

    assert "Toy" in str(exc.value)
    assert str(MINIMAX_MAX_ATTEMPTS) in str(exc.value)
    assert len([r for r in log if "url" in r]) == MINIMAX_MAX_ATTEMPTS


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
