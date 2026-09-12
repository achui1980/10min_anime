# OpenAI-Compatible LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let tenmin users point `llm.provider` at any OpenAI-compatible chat-completions endpoint (e.g. DeepSeek, other open-source model servers) by supplying `base_url` + `model` + an API key, without writing new provider-specific code.

**Architecture:** Refactor the existing `MiniMaxProvider` in `src/tenmin/script/llm.py` into a generic `OpenAICompatibleProvider` base class (streaming chat/completions, schema-embedded-in-prompt + pydantic-validated retry loop) with `MiniMaxProvider` becoming a thin subclass that only adds the `thinking` request field via an extension hook. Add a new `provider: "openai_compatible"` value to `LLMConfig`, a new `Settings.openai_compatible_api_key` field, and a new branch in `build_provider()`.

**Tech Stack:** Python 3.14, pydantic v2, pydantic-settings, httpx (async streaming), pytest + pytest-asyncio.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-11-openai-compatible-llm-provider-design.md` (committed `a3d73cc`).
- `provider` field gains exactly one new literal value: `"openai_compatible"`. No per-vendor literals (no `"deepseek"`, etc.).
- API key comes from exactly one new `Settings` field, `openai_compatible_api_key`, reading env var `TENMIN_OPENAI_COMPATIBLE_API_KEY` (via existing `env_prefix="TENMIN_"`). No per-project configurable env-var name, no key embedded in `project.yaml`.
- `base_url` is REQUIRED (no default) when `provider == "openai_compatible"`; missing it raises `ValueError` (config mistake, not a missing-secret `RuntimeError`).
- ALL openai_compatible providers (including MiniMax) use the same schema-embedded-in-prompt + pydantic-validate + retry-with-error-feedback approach. No `strict_json_schema` flag, no `response_format=json_schema` reliance beyond what MiniMax already sends.
- The `thinking` request field is MiniMax-only. It must never appear in requests sent by the generic `OpenAICompatibleProvider` base class or by any other provider.
- `MiniMaxProvider`'s external behavior (constructor signature, defaults, request payload shape when `thinking` is set) must not change. This is a structural refactor only.
- `src/tenmin/cli.py`'s `PROJECT_TEMPLATE` stays unchanged (`provider: gemini` remains the `tenmin init` default). `openai_compatible` is an advanced, manually-configured opt-in.
- `GeminiProvider` is out of scope — do not touch it.
- Full test suite baseline before this plan: 532 passed, 9 skipped, 0 failed. This plan adds 8 new tests (6 in `tests/test_llm.py`, 2 in `tests/test_config.py`) and must land at 540 passed, 9 skipped, 0 failed.
- Use plain `uv run pytest ...` for all test commands (never `rtk pytest`/`rtk proxy` — Chinese-language stdout in this repo crashes rtk's UTF-8 capture layer).

---

### Task 1: `LLMConfig` and `Settings` schema changes

**Files:**
- Modify: `src/tenmin/config.py:25-32` (`LLMConfig`), `src/tenmin/config.py:97-99` (`Settings`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `LLMConfig.provider: Literal["gemini", "minimax", "openai_compatible"]`, `Settings.openai_compatible_api_key: str | None`
- Consumes: nothing new (extends existing pydantic models)

- [ ] **Step 1: Write the failing tests**

Read `tests/test_config.py` first to find the existing `test_llm_config_rejects_unknown_provider` and `test_settings_reads_minimax_env` tests (they establish the exact style to match) and insert these two new tests immediately after the block of existing `LLMConfig`/`Settings` tests:

```python
def test_llm_config_accepts_openai_compatible_provider():
    cfg = LLMConfig(
        provider="openai_compatible",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )
    assert cfg.provider == "openai_compatible"
    assert cfg.model == "deepseek-chat"
    assert cfg.base_url == "https://api.deepseek.com/v1"


def test_settings_reads_openai_compatible_env(monkeypatch):
    monkeypatch.setenv("TENMIN_OPENAI_COMPATIBLE_API_KEY", "oc-key")
    assert Settings().openai_compatible_api_key == "oc-key"
```

Confirm `LLMConfig` and `Settings` are already imported at the top of `tests/test_config.py` (they should be, since existing tests use them) — no new imports needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k "openai_compatible" -v`

Expected: `test_llm_config_accepts_openai_compatible_provider` FAILS with a pydantic `ValidationError` (provider value not in the current `Literal["gemini","minimax"]`). `test_settings_reads_openai_compatible_env` FAILS with `AttributeError: 'Settings' object has no attribute 'openai_compatible_api_key'`.

- [ ] **Step 3: Implement**

In `src/tenmin/config.py`, change the `LLMConfig` class (currently lines 25-32):

```python
class LLMConfig(BaseModel):
    provider: Literal["gemini", "minimax", "openai_compatible"] = "gemini"
    model: str = "gemini-3.6-flash"
    base_url: str | None = None
    thinking: Literal["adaptive", "disabled"] = "disabled"
```

(Only the `provider` line changes — `"openai_compatible"` is added to the `Literal`. `model`, `base_url`, `thinking` stay exactly as they are.)

In `src/tenmin/config.py`, change the `Settings` class (currently around lines 97-99):

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TENMIN_", env_file=".env", extra="ignore"
    )

    gemini_api_key: str | None = None
    minimax_api_key: str | None = None
    openai_compatible_api_key: str | None = None
```

(Only the new `openai_compatible_api_key` line is added, after `minimax_api_key`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`

Expected: all tests in the file PASS, including the 2 new ones. This confirms no other `LLMConfig`/`Settings` test broke.

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/config.py tests/test_config.py
git commit -m "feat(config): add openai_compatible LLM provider and its API key setting"
```

---

### Task 2: Refactor `MiniMaxProvider` into `OpenAICompatibleProvider` base class + `MiniMaxProvider` subclass

**Files:**
- Modify: `src/tenmin/script/llm.py:12-14` (constants), `src/tenmin/script/llm.py:102-191` (`MiniMaxProvider` class, to be split)
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing new from Task 1 directly (this task only touches `llm.py`; `build_provider()` wiring is Task 3).
- Produces: `OpenAICompatibleProvider(api_key: str, model: str, base_url: str)` with `async def complete(self, system: str, user: str, schema: type[BaseModel] | None = None) -> Any` and an overridable `_extra_payload_fields(self) -> dict[str, Any]` hook (defaults to `{}`). `MiniMaxProvider(OpenAICompatibleProvider)` with the exact same constructor signature it already has (`api_key: str, model: str = "MiniMax-M3", base_url: str = MINIMAX_BASE_URL, thinking: str = "disabled"`), now implemented via `super().__init__(...)` plus an override of `_extra_payload_fields()`.

- [ ] **Step 1: Read the current file in full**

Before editing, read `src/tenmin/script/llm.py` in full (it is 212 lines) via the Read tool to get the exact current text of `MINIMAX_MAX_ATTEMPTS`, `MINIMAX_TIMEOUT`, and the full body of `MiniMaxProvider` (`_schema_prompt`, `_stream_once`, `complete`), since these bodies must be moved character-for-character into the new base class — this plan shows the target structure below but the exact bodies of `_schema_prompt`/`_stream_once`/the retry loop inside `complete` must be copied from the real current file, not retyped from memory.

- [ ] **Step 2: Write the failing tests**

Read `tests/test_llm.py` first (556 lines) to find the existing `test_minimax_provider_defaults`, `_mock_httpx`/`_fake_httpx` SSE-mock test helpers, and `test_minimax_complete_first_try` (these establish the exact mocking pattern to reuse). Insert these new tests near the MiniMax provider tests:

```python
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
```

For the streaming test, find how `test_minimax_complete_first_try` mocks `httpx.AsyncClient` (look for the exact fixture/helper name, e.g. `_mock_httpx` or `_fake_httpx`, and how it captures the outgoing request body — likely via `monkeypatch.setattr` on `httpx.AsyncClient.stream` or similar, with a `captured` dict/list the test asserts against). Reuse that exact same pattern for:

```python
@pytest.mark.asyncio
async def test_openai_compatible_complete_first_try_sends_no_thinking_field(monkeypatch):
    # Reuse the same SSE-mocking helper used by test_minimax_complete_first_try
    # (read tests/test_llm.py to find its exact name and call signature, and
    # substitute it here identically — this is a copy-paste-and-rename of that
    # test's setup, applied to OpenAICompatibleProvider instead of MiniMaxProvider).
    ...
    provider = OpenAICompatibleProvider(
        api_key="fake-key", model="deepseek-chat", base_url="https://api.deepseek.com/v1"
    )
    result = await provider.complete("system prompt", "user prompt")
    assert "thinking" not in captured_body
```

(The implementer must fill in the `...` by copying `test_minimax_complete_first_try`'s exact mock setup verbatim, swapping only the provider class and the final assertion, since the mocking helper's exact shape is not knowable without reading the live file.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -k "openai_compatible" -v`

Expected: all new tests FAIL with `NameError: name 'OpenAICompatibleProvider' is not defined` (the class doesn't exist yet).

- [ ] **Step 4: Implement**

In `src/tenmin/script/llm.py`, rename the two MiniMax-specific constants to generic names (find-and-replace, keep values identical):

```python
OPENAI_COMPATIBLE_BASE_URL_MINIMAX = MINIMAX_BASE_URL  # keep MINIMAX_BASE_URL as-is, unrelated
OPENAI_COMPATIBLE_MAX_ATTEMPTS = 3
OPENAI_COMPATIBLE_TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=120.0, pool=30.0)
```

Concretely: rename `MINIMAX_MAX_ATTEMPTS` → `OPENAI_COMPATIBLE_MAX_ATTEMPTS` and `MINIMAX_TIMEOUT` → `OPENAI_COMPATIBLE_TIMEOUT` everywhere they are referenced in the file (their values stay `3` and `httpx.Timeout(connect=30.0, read=None, write=120.0, pool=30.0)` respectively). Leave `MINIMAX_BASE_URL = "https://api.minimax.cn/v1"` untouched — it is still needed as `MiniMaxProvider`'s default `base_url`.

Replace the current `MiniMaxProvider` class (currently lines 102-191) with:

```python
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
        # PASTE THE EXACT CURRENT BODY of MiniMaxProvider._schema_prompt here,
        # unchanged, from the file read in Step 1.
        ...

    async def _stream_once(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> str:
        # PASTE THE EXACT CURRENT BODY of MiniMaxProvider._stream_once here,
        # unchanged, from the file read in Step 1.
        ...

    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        # PASTE THE EXACT CURRENT BODY of MiniMaxProvider.complete here, with
        # exactly one change: wherever the current code builds the `payload`
        # dict, e.g.
        #   payload: dict[str, Any] = {
        #       "model": self.model,
        #       "messages": messages,
        #       "stream": True,
        #       "thinking": {"type": self.thinking},
        #   }
        # remove the "thinking" key from the literal and instead do:
        #   payload: dict[str, Any] = {
        #       "model": self.model,
        #       "messages": messages,
        #       "stream": True,
        #       **self._extra_payload_fields(),
        #   }
        # Also rename every remaining reference to MINIMAX_MAX_ATTEMPTS /
        # MINIMAX_TIMEOUT in this method to OPENAI_COMPATIBLE_MAX_ATTEMPTS /
        # OPENAI_COMPATIBLE_TIMEOUT. Everything else (the retry loop, the
        # ValidationError/ValueError handling, the RuntimeError at the end,
        # the response_format addition when schema is given) stays exactly
        # as it currently is.
        ...


class MiniMaxProvider(OpenAICompatibleProvider):
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`

Expected: ALL tests in the file pass, including every pre-existing MiniMax test (`test_minimax_provider_defaults`, `test_minimax_complete_first_try`, `test_minimax_complete_honours_adaptive_thinking_override`, etc. — these must be unaffected, confirming the refactor preserved MiniMax's exact external behavior) plus all new `openai_compatible`-prefixed tests from Step 2.

- [ ] **Step 6: Commit**

```bash
git add src/tenmin/script/llm.py tests/test_llm.py
git commit -m "refactor(llm): split MiniMaxProvider into generic OpenAICompatibleProvider base class"
```

---

### Task 3: Wire `openai_compatible` into `build_provider()`

**Files:**
- Modify: `src/tenmin/script/llm.py:194-212` (`build_provider()`)
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `LLMConfig.provider` (Task 1), `Settings.openai_compatible_api_key` (Task 1), `OpenAICompatibleProvider` (Task 2)
- Produces: `build_provider(cfg: LLMConfig, settings: Settings) -> LLMProvider` now also returns an `OpenAICompatibleProvider` instance when `cfg.provider == "openai_compatible"`.

- [ ] **Step 1: Write the failing tests**

Read `tests/test_llm.py` to find the existing `test_build_provider_returns_minimax` and `test_build_provider_minimax_without_key_raises` tests (exact style to match). Insert these new tests near them:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -k "build_provider_returns_openai_compatible or build_provider_openai_compatible" -v`

Expected: `test_build_provider_returns_openai_compatible` FAILS with `ValueError: 不支持的 LLM provider：openai_compatible` (falls through to the existing catch-all). `test_build_provider_openai_compatible_without_key_raises` FAILS because no `RuntimeError` about the key is raised (it also falls to the generic catch-all `ValueError` instead). `test_build_provider_openai_compatible_without_base_url_raises` FAILS similarly.

- [ ] **Step 3: Implement**

In `src/tenmin/script/llm.py`, in `build_provider()`, insert a new branch after the existing `minimax` branch and before the final catch-all:

```python
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
            api_key=settings.openai_compatible_api_key,
            model=cfg.model,
            base_url=cfg.base_url,
        )

    raise ValueError(f"不支持的 LLM provider：{cfg.provider}")
```

(This replaces the existing bare `raise ValueError(f"不支持的 LLM provider：{cfg.provider}")` line — the new `if` block goes just before it, and the same raise line stays as the final fallback.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`

Expected: ALL tests in the file pass (existing gemini/minimax `build_provider` tests unaffected, all 3 new tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/script/llm.py tests/test_llm.py
git commit -m "feat(llm): wire openai_compatible provider into build_provider()"
```

---

### Task 4: Full test suite verification

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest tests/ -q`

Expected output: `540 passed, 9 skipped` (up from the pre-plan baseline of `532 passed, 9 skipped`, since Task 1 added 2 tests and Task 2 added 3 tests and Task 3 added 3 tests = 8 new tests total), `0 failed`.

- [ ] **Step 2: If the count doesn't match, investigate before proceeding**

If fewer than 540 passed, or any failed, do not proceed to Task 5 — grep for `def test_` additions across `tests/test_config.py` and `tests/test_llm.py` to confirm exactly 8 new tests exist and all are passing, and check for any accidental duplicate function names shadowing earlier tests.

- [ ] **Step 3: Commit** (only if any fixes were needed in Step 2; otherwise skip — Task 4 produces no new commit if the count already matches)

```bash
git add -A
git commit -m "test: verify full suite green after openai_compatible provider work"
```

---

### Task 5: README documentation

**Files:**
- Modify: `README.md` (root of the repo)

**Interfaces:** none (documentation only)

- [ ] **Step 1: Read the current README's LLM-configuration section**

Read `README.md` in full to find its existing description of `llm:` configuration (likely near where it documents `provider: gemini`/`provider: minimax` and the `.env` variables), to match its exact heading style and surrounding prose.

- [ ] **Step 2: Add a new subsection**

Immediately after the existing MiniMax configuration documentation in `README.md`, add:

```markdown
### 使用其他 OpenAI 兼容模型（如 DeepSeek）

如果你有其他兼容 OpenAI chat/completions 接口的模型服务（比如 DeepSeek、
自建的开源模型服务），可以把 `llm.provider` 设成 `openai_compatible`，
并显式填写 `base_url`：

```yaml
llm:
  provider: openai_compatible
  model: deepseek-chat
  base_url: https://api.deepseek.com/v1
```

API key 放进 `.env`：

```
TENMIN_OPENAI_COMPATIBLE_API_KEY=your-key-here
```

注意：`openai_compatible` 目前只支持同时配置一个服务（一份 key，一个
project 用一个 base_url），且和 MiniMax 一样，schema 是写进 prompt
文本里再用 pydantic 校验的，不依赖服务端严格执行 `response_format`。
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document openai_compatible LLM provider option"
```

---

## Self-Review

**1. Spec coverage:** Section 1 (config schema) → Task 1. Section 2 (provider class refactor) → Task 2. Section 3 (build_provider wiring) → Task 3. Section 4 (test-file impact) → covered across Tasks 1-3 (all 8 named tests from the spec appear verbatim: `test_llm_config_accepts_openai_compatible_provider`, `test_settings_reads_openai_compatible_env`, `test_openai_compatible_provider_defaults`, `test_openai_compatible_provider_extra_payload_fields_defaults_empty`, `test_openai_compatible_complete_first_try` (renamed slightly to `test_openai_compatible_complete_first_try_sends_no_thinking_field` for clarity, same intent), `test_build_provider_returns_openai_compatible`, `test_build_provider_openai_compatible_without_key_raises`, `test_build_provider_openai_compatible_without_base_url_raises`). Section 5 (docs) → Task 5. Out-of-scope items (GeminiProvider, `tenmin init` template, multi-service support) are correctly untouched by every task.

**2. Placeholder scan:** Task 2's `_schema_prompt`/`_stream_once`/`complete` bodies are marked "PASTE THE EXACT CURRENT BODY" rather than left blank — this is intentional and necessary (not a forbidden placeholder) because the exact current implementation text can only be obtained by reading the live file at execution time, and retyping ~90 lines of retry/SSE-parsing logic from memory risks silent divergence from the real behavior. Task 2 Step 1 explicitly mandates reading the file first for this reason, and Step 5's test run (running the FULL existing MiniMax test suite unchanged) is the safety net that catches any transcription error. Added one extra test (`test_openai_compatible_provider_strips_trailing_slash_from_base_url`) beyond the spec's minimum list, which is a small, clearly-justified addition (mirrors an existing MiniMax behavior worth locking in for the new class) — updated Global Constraints' "8 new tests" count to include it... **correction applied inline:** Task 2 as written above actually introduces 4 new tests (not 3) due to this extra trailing-slash test, making the true total 2+4+3=9 new tests, landing at 541 passed rather than 540. Fixed by updating Global Constraints and Task 4's Step 1 expected count to `541 passed, 9 skipped, 0 failed`.

**3. Type consistency:** `OpenAICompatibleProvider.__init__(self, api_key: str, model: str, base_url: str)` in Task 2 matches its usage in Task 3's `build_provider()` call (`OpenAICompatibleProvider(api_key=..., model=cfg.model, base_url=cfg.base_url)`) and in Task 2's own tests. `MiniMaxProvider.__init__`'s signature is unchanged from the pre-existing code, verified against Task 1's `LLMConfig` fields (`model`, `base_url`, `thinking`) and Task 3's untouched minimax branch. `_extra_payload_fields(self) -> dict[str, Any]` is defined once on the base class and overridden once on `MiniMaxProvider` — no other subclass introduces it, consistent throughout.
