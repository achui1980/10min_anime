import pytest
from pydantic import BaseModel

from tenmin.config import LLMConfig, Settings
from tenmin.script.llm import GeminiProvider, LLMProvider, build_provider

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
    provider = build_provider(LLMConfig(provider="gemini", model="gemini-2.5-pro"), settings)
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-2.5-pro"


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

    provider = GeminiProvider(api_key="fake-key", model="gemini-2.5-pro")
    monkeypatch.setattr(provider, "_client", FakeClient())

    result = await provider.complete("SYS", "USR", Toy)
    assert result == Toy(value=42)
    assert captured["model"] == "gemini-2.5-pro"
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

    provider = GeminiProvider(api_key="fake-key", model="gemini-2.5-pro")
    monkeypatch.setattr(provider, "_client", FakeClient())

    assert await provider.complete("SYS", "USR") == "纯文本回答"
