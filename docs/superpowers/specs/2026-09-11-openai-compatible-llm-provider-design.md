# 支持多个 OpenAI 兼容 LLM Provider 设计

## 背景 / 动机

用户需求（原话）：

> 我想支持多个openai兼容的模型，这个要怎么做，例如我还想用deepseek的模型，或者其他开源模型，我是有APIkey的

目前 `tenmin` 的 `LLMConfig.provider` 只支持两个硬编码值：`gemini`（走 Google 官方 SDK）和 `minimax`（走 OpenAI 兼容的 `chat/completions` 流式接口）。用户想接入 DeepSeek 或其他开源模型的 OpenAI 兼容 API，但目前架构不支持任意 `base_url`。

`MiniMaxProvider` 内部的 HTTP/流式/重试逻辑本身就是**完全通用的 OpenAI 兼容协议实现**——除了类名、`MINIMAX_BASE_URL` 默认值、模型名默认值之外没有任何 MiniMax 专属逻辑。这使得把它泛化成一个通用 provider 成为最自然的实现路径。

## 已确认的设计决策（Q&A）

1. **Provider 配置形式**：新增一个通用的 `provider: openai_compatible` 值，用户在 `project.yaml` 里自行填写 `base_url` + `model` + 一个通用 API key（而不是给每个厂商单独定义 Literal 枚举值，也不是"预置常见厂商 + 通用兜底"的混合方案）。
2. **API key 来源**：新增一个固定的 `Settings.openai_compatible_api_key` 字段，读取环境变量 `TENMIN_OPENAI_COMPATIBLE_API_KEY`（不支持每个项目自定义环境变量名，也不允许把 key 直接写在 `project.yaml` 里）。同一时间只能有一个 openai_compatible 服务的 key 生效，和现有 `gemini_api_key`/`minimax_api_key` 的模式保持一致。
3. **实现方式**：把 `MiniMaxProvider` 重构成一个通用的 `OpenAICompatibleProvider` 基类，MiniMax 变成它的一个子类/预设（而不是保持 `MiniMaxProvider` 不动、另写一个重复的新类）。
4. **JSON schema 处理**：所有 openai_compatible provider 统一采用 MiniMax 现在的兜底方案——把 schema 写进 prompt 文本里，靠 pydantic 校验 + 报错重试来保证结构化输出，不管后端是否真的支持严格的 `response_format=json_schema`。不加可配置的 `strict_json_schema` 开关。
5. **thinking 参数的作用范围**：`thinking:{"type":...}` 这个请求体字段保留在 MiniMax 专属子类里（通过一个扩展点方法实现），不作为基类上的通用可配置字段。DeepSeek 等其他 provider 永远不会收到这个参数。

## 设计详情

### Section 1：配置 schema 改动

`src/tenmin/config.py` 的 `LLMConfig`：

```python
class LLMConfig(BaseModel):
    provider: Literal["gemini", "minimax", "openai_compatible"] = "gemini"
    model: str = "gemini-3.6-flash"
    base_url: str | None = None
    thinking: Literal["adaptive", "disabled"] = "disabled"
```

- `provider` 新增 `"openai_compatible"` 枚举值（原来是 `Literal["gemini","minimax"]`）。
- `base_url` 字段已经存在，直接复用；但当 `provider == "openai_compatible"` 时**必须**在 `project.yaml` 里显式填写，否则 `build_provider()` 会报错（没有合理的默认值）。
- `thinking` 字段不变，对 `openai_compatible` 无意义/不生效（只有 MiniMax 会读取它）。

`Settings(BaseSettings)` 新增一个字段，和现有的 `gemini_api_key`/`minimax_api_key` 对称：

```python
openai_compatible_api_key: str | None = None
```

通过现有的 `env_prefix="TENMIN_"` 约定，映射到 `.env` 里的 `TENMIN_OPENAI_COMPATIBLE_API_KEY`。

### Section 2：Provider 类重构

`src/tenmin/script/llm.py`——把 `MiniMaxProvider` 拆成一个通用基类 + 一个 MiniMax 专属子类：

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
        ...  # 原样搬过来

    async def _stream_once(self, client, payload) -> str:
        ...  # 原样搬过来，POST 到 f"{self.base_url}/chat/completions"

    async def complete(self, system: str, user: str, schema=None) -> Any:
        ...  # 原样搬过来，payload 里 **self._extra_payload_fields()


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

- `MINIMAX_MAX_ATTEMPTS`/`MINIMAX_TIMEOUT` 常量改名为通用的 `OPENAI_COMPATIBLE_MAX_ATTEMPTS`/`OPENAI_COMPATIBLE_TIMEOUT`（数值不变），基类和 MiniMax 子类共用。
- 对 MiniMax 而言**行为完全不变**——纯粹是代码结构上的拆分。

### Section 3：`build_provider()` 工厂函数接线

在 `build_provider()` 的最终兜底分支之前插入新分支，完全照搬现有 gemini/minimax 分支的写法：

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
```

- `openai_compatible` 必须配置 `base_url`（无默认值）——缺失时抛 `ValueError`（配置错误），而不是 `RuntimeError`（缺密钥）。
- `MiniMaxProvider` 的构造调用点不变（现在只是构造了一个子类实例）。
- 最终的兜底 `raise ValueError(f"不支持的 LLM provider：{cfg.provider}")` 保留（由于 pydantic Literal 已经限制了取值，这行在实践中不可达，但保留是为了代码风格一致）。

### Section 4：测试文件影响范围

- `tests/test_llm.py` 新增测试：`test_openai_compatible_provider_defaults`、`test_openai_compatible_provider_extra_payload_fields_defaults_empty`、`test_openai_compatible_complete_first_try`（复用现有的 SSE mock 测试基础设施，验证请求体里**没有** `thinking` 字段）、`test_build_provider_returns_openai_compatible`、`test_build_provider_openai_compatible_without_key_raises`、`test_build_provider_openai_compatible_without_base_url_raises`。现有的 MiniMax 专属测试保持不变（MiniMax 对外行为零变化）。
- `tests/test_config.py` 新增测试：`test_llm_config_accepts_openai_compatible_provider`、`test_settings_reads_openai_compatible_env`。
- `src/tenmin/cli.py` 的 `PROJECT_TEMPLATE`：不改动——`openai_compatible` 仍然是一个进阶/手动选择的选项，`tenmin init` 生成的默认模板保持 `provider: gemini`。

### Section 5：文档

在 README 里新增一段说明新 provider 选项，附带一个可以直接复制粘贴的示例：

```yaml
llm:
  provider: openai_compatible
  model: deepseek-chat
  base_url: https://api.deepseek.com/v1
```

并说明 API key 要写在 `.env` 里：`TENMIN_OPENAI_COMPATIBLE_API_KEY=your-key-here`。

## 范围之外（Out of scope）

- 不支持同时配置多个 openai_compatible 服务（比如同时用 DeepSeek 又用另一个开源模型）——只有一个全局固定的环境变量位。
- 不为特定厂商（DeepSeek 等）做任何专属优化或校验（比如真正启用严格 JSON schema）——统一走 MiniMax 现有的 prompt-embedded-schema 兜底方案。
- 不修改 `GeminiProvider` 或其调用路径——它走的是原生 Google SDK，与本次改动完全无关。
- 不修改 `tenmin init` 生成的默认模板。
