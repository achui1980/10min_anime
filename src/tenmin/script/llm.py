"""LLM provider。目前支持 Gemini、MiniMax，以及任意 OpenAI 兼容接口
（如 DeepSeek，通过 provider: openai_compatible 配置）。"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, overload, runtime_checkable

import httpx
from pydantic import BaseModel, ValidationError

from tenmin.config import DEFAULT_LLM, LLMConfig, Settings

MINIMAX_BASE_URL = "https://api.minimax.cn/v1"

# httpx.Timeout 里不由 LLMConfig 管的两格。它们是「建连」与「等连接池空位」的上限，
# 跟模型有多慢、prompt 有多长完全无关，没有调它们的场景。
CONNECT_TIMEOUT_SECONDS = 30.0
POOL_TIMEOUT_SECONDS = 30.0

# --- 传输层退避 ---
# 基数 1 秒：429 的正常恢复窗是秒级，第一次重试等太久纯属浪费。
# 上限 30 秒：单集 script 阶段本来就是分钟级，30 秒的单次等待还在「用户愿意等」的
# 范围内，而再往上翻只会把一次注定失败的运行拖成十几分钟。
# 抖动是**乘性**的 [1, 1.25)：只会加不会减（不缩短退避），同时打散批量模式下 10 集
# 连着撞同一个限流窗时的同相重试。
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
BACKOFF_JITTER_RATIO = 0.25
# Retry-After 说多久就等多久，但要夹住：服务端（或中间的代理）给一个离谱的值时，
# 一次 429 能把整条流水线钉死几小时。
RETRY_AFTER_MAX_SECONDS = 120.0
# 429 = 限流，5xx = 服务端/网关侧的瞬时故障。其余 4xx（401/403/400/404）重试是纯
# 浪费：key 不会在 1 秒后自己变对，请求体也不会自己变合法。
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# 连接/读取类的传输异常。刻意不用 httpx.TransportError 这个大父类：
# UnsupportedProtocol（base_url 写成了 ftp://）与 LocalProtocolError（我们自己拼错了
# 请求）也在它底下，重试它们只是把同一个必错的请求发四遍。
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)

# 错误响应体拼进异常消息时的上限。1000 字符足够看清 JSON 错误体里的 code/message，
# 又不会在终端里刷屏（有些网关的 4xx 会返回一整页 HTML）。
HTTP_ERROR_BODY_MAX_CHARS = 1000

# 回灌给模型的两段文本各自的上限。坏输出可能是一整份坏剧本（几万字符），报错也可能是
# 一长串 pydantic 逐字段清单，原样回灌等于把纠错轮的 token 成本推回到首轮量级。
REPAIR_ERROR_MAX_CHARS = 1500
REPAIR_OUTPUT_MAX_CHARS = 4000

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S)
_THINK_OPEN = re.compile(r"<think\b[^>]*>")
_FENCE = re.compile(r"^```(?:json)?[ \t]*\n?|\n?```[ \t]*$", re.M)


async def _sleep(seconds: float) -> None:
    """退避用的 sleep。**刻意做成模块级函数**，测试 monkeypatch 掉它就既不真睡、
    又能把整条退避序列的时长逐个断言出来。"""
    await asyncio.sleep(seconds)


def _rand() -> float:
    """[0, 1) 的抖动源。同样是模块级函数，为的是让退避序列在测试里可确定。"""
    return random.random()


def _backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """第 attempt 次尝试失败后要等多久（attempt 从 1 开始）。

    Retry-After 优先（服务端最清楚还要等多久），只做上限夹取、不叠抖动；
    否则指数退避 base * 2^(attempt-1)，夹到上限后再叠一个乘性抖动。
    """
    if retry_after is not None:
        return max(0.0, min(retry_after, RETRY_AFTER_MAX_SECONDS))
    delay = min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS)
    return delay * (1.0 + BACKOFF_JITTER_RATIO * _rand())


def _parse_retry_after(value: str | None) -> float | None:
    """只认秒数形式的 Retry-After。

    HTTP-date 形式（`Wed, 21 Oct 2015 07:28:00 GMT`）一律返回 None 退化成指数退避：
    解析它要处理时区与本机时钟漂移，而实测的 LLM 网关（MiniMax / DeepSeek / OpenAI）
    给的都是秒数，为一个没人发的形态引入时钟依赖不划算。
    """
    if not value:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


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


class LLMSchemaError(LLMError):
    """连续 max_attempts 轮都没拿到合 schema 的 JSON。

    raw_output 挂着**最后一次的原始模型输出**（未截断）。这类失败最需要现场，而原
    实现只把 last_error 的前 1500 字符塞进消息、原始输出全部丢弃。落盘的事交给
    pipeline.run_script（那一层才知道 Paths），见 03_script/E{NN}.raw.txt。
    """

    def __init__(self, message: str, *, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


class LLMHTTPError(LLMError):
    """HTTP 4xx/5xx。**响应体摘要直接进异常消息**。

    原实现只是 `await response.aread()` 之后 `raise_for_status()`：httpx 的
    HTTPStatusError 消息里不含 body（body 只挂在 exc.response 上），而 single.py /
    pipeline.py / cli.py 没有任何一处去读它，用户实际看到的仍然只是一句
    「429 Too Many Requests for url ...」——一个字的诊断信息都没有。
    """

    def __init__(
        self,
        *,
        status_code: int,
        body: str,
        url: str = "",
        retry_after: float | None = None,
        attempts: int = 1,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.url = url
        self.retry_after = retry_after
        self.attempts = attempts
        tries = f"（已尝试 {attempts} 次）" if attempts > 1 else ""
        super().__init__(
            f"LLM 接口返回 HTTP {status_code}{tries}：{url}\n"
            f"响应体：{_truncate(body, HTTP_ERROR_BODY_MAX_CHARS)}"
        )


class LLMTransportError(LLMError):
    """连不上 / 建连超时 / chunk 间隔超时 / 服务端提前断流 / 撞到总截止。"""


def _truncate(text: str, limit: int) -> str:
    """超长就截断并注明原长度，免得读报错的人以为模型只输出了这么点。"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…（已截断，原输出共 {len(text)} 字符，这里只保留前 {limit} 个）"


def _schema_spec(schema: type[BaseModel]) -> str:
    """schema 的 JSON 文本。ensure_ascii=False：中文枚举/描述不能被转义成 \\uXXXX。"""
    return json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class RepairContext:
    """纠错轮的输入：上一轮的坏输出与校验报错，**两者都已截断**。

    刻意不含首轮那份正文：模型要做的只是「把刚才那段 JSON 改成合 schema 的样子」，
    对白轨/高能点/few-shot 示例对这件事零增量信息。
    """

    bad_output: str
    error: str


async def _complete_with_schema_repair[T: BaseModel](
    send: Callable[[RepairContext | None], Awaitable[str]],
    schema: type[T],
    *,
    max_attempts: int,
    label: str,
    diagnostics: Callable[[], str] | None = None,
) -> T:
    """provider 无关的「校验失败 → 把报错回灌 → 重试」包装层。

    原先这段逻辑只长在 OpenAICompatibleProvider.complete 里，Gemini 一次自修复都
    没有（字段名一错就把 pydantic 报错直接冒给用户）。提到这里之后两条路径共用。

    send 收到 None 表示首轮（发完整 prompt），收到 RepairContext 表示纠错轮；
    「纠错轮的请求长什么样」是 provider 的事（messages 列表 vs 单个 contents 串），
    所以留给闭包决定，这一层只管计数、校验、回灌与耗尽后的报错。

    diagnostics 是可选的「补充现场」回调，耗尽时拼进错误消息（目前用来报告流里跳过
    了几条畸形 SSE 行、传输层重试了几次）。
    """
    repair: RepairContext | None = None
    last_error = ""
    last_raw = ""
    for _ in range(max_attempts):
        last_raw = await send(repair)
        try:
            return schema.model_validate_json(_extract_json(last_raw))
        except (ValidationError, LLMResponseFormatError) as exc:
            last_error = str(exc)[:REPAIR_ERROR_MAX_CHARS]
            repair = RepairContext(
                bad_output=_truncate(last_raw, REPAIR_OUTPUT_MAX_CHARS),
                error=last_error,
            )
    note = diagnostics() if diagnostics is not None else ""
    suffix = f"（{note}）" if note else ""
    raise LLMSchemaError(
        f"{label} 连续 {max_attempts} 次输出不符合 {schema.__name__}{suffix}：{last_error}",
        raw_output=last_raw,
    )


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
    """两条重载让「传 schema 时返回该 schema 的实例」成为静态可知的事实。

    原来只有一个返回 Any 的签名，于是 single.py 里的 llm_script 毫无类型，
    改错字段名要等运行时才炸。PEP 695 的方法级类型参数（`[T: BaseModel]`）在
    py312 起可用，也避开了 ruff UP047 对旧式 TypeVar 的抱怨。
    """

    @overload
    async def complete(self, system: str, user: str, schema: None = None) -> str: ...

    @overload
    async def complete[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T: ...

    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        """schema 非空时返回该 schema 的实例，否则返回纯文本。"""
        ...


_REPAIR_HEADER = (
    "你刚才为一个任务输出了 JSON，但不符合要求的 schema。"
    "任务正文这里**刻意不重复**，你只需要把输出改成合 schema 的样子。"
)


class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        *,
        max_attempts: int = DEFAULT_LLM.max_attempts,
    ):
        from google import genai

        self.model = model
        self.max_attempts = max_attempts
        self._client = genai.Client(api_key=api_key)

    async def _generate(
        self, system: str, contents: str, schema: type[BaseModel] | None
    ) -> Any:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
        )
        return await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config
        )

    def _repair_contents(self, schema: type[BaseModel], repair: RepairContext) -> str:
        return (
            f"{_REPAIR_HEADER}\n\n"
            "## 输出 JSON Schema（必须严格遵守）\n\n"
            f"```json\n{_schema_spec(schema)}\n```\n\n"
            "## 上一轮的输出\n\n"
            f"{repair.bad_output}\n\n"
            "## 校验报错\n\n"
            f"{repair.error}\n\n"
            "只输出修正后的完整 JSON 对象，不要解释，不要加 markdown 围栅。"
        )

    @overload
    async def complete(self, system: str, user: str, schema: None = None) -> str: ...

    @overload
    async def complete[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T: ...

    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        if schema is None:
            response = await self._generate(system, user, None)
            return response.text

        async def send(repair: RepairContext | None) -> str:
            contents = user if repair is None else self._repair_contents(schema, repair)
            response = await self._generate(system, contents, schema)
            text = response.text
            if text is None:
                raise LLMResponseFormatError(
                    "Gemini 一个字都没返回（response.text is None），没有可校验的内容。"
                )
            return text

        return await _complete_with_schema_repair(
            send, schema, max_attempts=self.max_attempts, label=type(self).__name__
        )


class OpenAICompatibleProvider:
    """通用 OpenAI 兼容 provider：走 chat/completions 流式接口，
    schema 统一写进 prompt 文本（不依赖 response_format=json_schema），
    用 pydantic 校验 + 报错重试。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        *,
        max_attempts: int = DEFAULT_LLM.max_attempts,
        transport_max_attempts: int = DEFAULT_LLM.transport_max_attempts,
        timeout_seconds: float = DEFAULT_LLM.timeout_seconds,
        read_timeout_seconds: float = DEFAULT_LLM.read_timeout_seconds,
        total_timeout_seconds: float = DEFAULT_LLM.total_timeout_seconds,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_attempts = max_attempts
        self.transport_max_attempts = transport_max_attempts
        self._api_key = api_key
        self._total_timeout_seconds = total_timeout_seconds
        # read 是**相邻两个 chunk 之间**最多等多久，不是整段生成时长——流式下每个
        # SSE chunk 都会刷新读活性，所以给它一个有限上限并不会误杀「思考了 9 分钟
        # 才吐完」的长请求，只会杀掉「吐了首字节之后 stall」的死流。原来这里是
        # read=None，等于把 chunk 间隔的活性检测整个关掉，一旦 stall 就永久挂着。
        self._timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS,
            read=read_timeout_seconds,
            write=timeout_seconds,
            pool=POOL_TIMEOUT_SECONDS,
        )
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """provider 持有 client，生命周期跟 provider 一致。

        原实现每次 complete 都 `async with httpx.AsyncClient(...)`，于是同一次
        script 阶段的每一轮重试都要重做一遍 TLS 握手，批量模式下 10 集就是 10 份。
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """关掉持有的连接池。cli.py 在 run_pipeline 之后调它。幂等。"""
        client, self._client = self._client, None
        if client is not None and not client.is_closed:
            await client.aclose()

    def _extra_payload_fields(self) -> dict[str, Any]:
        """子类可覆写，往请求体里加自己专属的字段（比如 MiniMax 的 thinking）。"""
        return {}

    def _schema_prompt(self, user: str, schema: type[BaseModel]) -> str:
        return (
            f"{user}\n\n"
            "## 输出 JSON Schema（必须严格遵守）\n\n"
            f"```json\n{_schema_spec(schema)}\n```\n\n"
            "只输出符合上面 schema 的 JSON 对象本身。字段名一个字都不能改，"
            "不要在外面再套一层包裹对象，不要加解释文字，不要加 markdown 围栅。"
        )

    def _repair_messages(
        self, system: str, schema: type[BaseModel], repair: RepairContext
    ) -> list[dict[str, str]]:
        """纠错轮的 messages。形状跟原来一样是「system + user + assistant + user」，
        差别只在 **messages[1] 不再是首轮那份 ~35k 字符的正文**，而是一段只含 schema
        的提醒。原实现把首轮 messages 原地 append，于是第 3 次请求 ≈ 完整 prompt +
        2 份坏输出，纠错轮的 token 成本接近翻倍。
        """
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"{_REPAIR_HEADER}\n\n"
                    "## 输出 JSON Schema（必须严格遵守）\n\n"
                    f"```json\n{_schema_spec(schema)}\n```"
                ),
            },
            {"role": "assistant", "content": repair.bad_output},
            {
                "role": "user",
                "content": (
                    "上面的输出不符合 schema，校验报错如下。"
                    "请只输出修正后的完整 JSON 对象，不要解释。\n\n"
                    f"{repair.error}"
                ),
            },
        ]

    def _payload(
        self, messages: list[dict[str, str]], *, json_mode: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self._extra_payload_fields(),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    async def _stream_once(self, payload: dict[str, Any]) -> str:
        """发一次流式请求，把所有 delta.content 拼成完整文本。

        整次请求外面套一个 asyncio.timeout：read 只管**相邻两个 chunk**的间隔，
        「每 100 秒吐一个字节」这种半死不活的流照样能挂到天荒地老，所以还要一个
        整体截止。
        """
        parts: list[str] = []
        try:
            async with asyncio.timeout(self._total_timeout_seconds):
                async with self._get_client().stream(
                    "POST",
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                ) as response:
                    if response.is_error:
                        # 流式响应在读 body 之前拿不到错误体，必须先 aread。
                        body = (await response.aread()).decode("utf-8", "replace")
                        raise LLMHTTPError(
                            status_code=response.status_code,
                            body=body,
                            url=self._endpoint,
                            retry_after=_parse_retry_after(
                                response.headers.get("retry-after")
                            ),
                        )
                    async for line in response.aiter_lines():
                        chunk = _sse_delta(line)
                        if chunk:
                            parts.append(chunk)
        except TimeoutError as exc:
            # asyncio.timeout 到点抛的是内置 TimeoutError（httpx 自己的超时是
            # httpx.TimeoutException，两者没有继承关系，不会互相误吞）。
            raise LLMTransportError(
                f"一次 LLM 请求超过了总时长上限 {self._total_timeout_seconds} 秒"
                f"（{self._endpoint}）。确实需要更久的话调高 llm.total_timeout_seconds。"
            ) from exc
        return "".join(parts)

    async def _stream_with_retries(self, payload: dict[str, Any]) -> str:
        """在 _stream_once 外面套传输层重试。

        **跟 schema 修复重试分开计数**：原实现只有一个 3 次的循环、且只覆盖 schema
        校验失败，_stream_once 抛的 HTTPStatusError(429/5xx) / ConnectError /
        ReadTimeout / RemoteProtocolError 全部在 try 之外，直接冲出循环，传输层等于
        零重试零退避——一个 429 就把整集的 script 阶段打死。
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._stream_once(payload)
            except LLMHTTPError as exc:
                retryable = exc.status_code in RETRYABLE_STATUS_CODES
                retry_after = exc.retry_after
                error: Exception = exc
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                retryable = True
                retry_after = None
                error = exc
            if not retryable or attempt >= self.transport_max_attempts:
                raise self._exhausted(error, attempt) from error
            await _sleep(_backoff_delay(attempt, retry_after))

    def _exhausted(self, error: Exception, attempt: int) -> LLMError:
        """把最后一次失败翻译成带现场的 LLMError。

        httpx 的传输类异常必须包一层：它们的 str() 经常是空串（ReadTimeout('')），
        而且不在 cli.py 那张「已经自带一句人话」的表所覆盖的语义里。
        """
        if isinstance(error, LLMHTTPError):
            return LLMHTTPError(
                status_code=error.status_code,
                body=error.body,
                url=error.url,
                retry_after=error.retry_after,
                attempts=attempt,
            )
        detail = str(error) or type(error).__name__
        return LLMTransportError(
            f"连接 LLM 接口失败，已尝试 {attempt} 次仍不通"
            f"（{self._endpoint}）：{type(error).__name__}: {detail}"
        )

    @overload
    async def complete(self, system: str, user: str, schema: None = None) -> str: ...

    @overload
    async def complete[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T: ...

    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        if schema is None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            text = await self._stream_with_retries(
                self._payload(messages, json_mode=False)
            )
            return _strip_reasoning(text)

        first = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._schema_prompt(user, schema)},
        ]

        async def send(repair: RepairContext | None) -> str:
            messages = (
                first if repair is None else self._repair_messages(system, schema, repair)
            )
            return await self._stream_with_retries(self._payload(messages, json_mode=True))

        return await _complete_with_schema_repair(
            send, schema, max_attempts=self.max_attempts, label=type(self).__name__
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
        **kwargs: Any,
    ) -> None:
        # kwargs 原样转给基类（max_attempts / transport_max_attempts / 三个超时）。
        # 这里刻意不逐个重列：本类跟基类的唯一差别就是 thinking 这一个字段，重列 5 个
        # 参数只会多出一处随基类演进而漂移的拷贝。
        super().__init__(api_key=api_key, model=model, base_url=base_url, **kwargs)
        self.thinking = thinking

    def _extra_payload_fields(self) -> dict[str, Any]:
        return {"thinking": {"type": self.thinking}}


def _transport_kwargs(cfg: LLMConfig) -> dict[str, Any]:
    """LLMConfig -> OpenAICompatibleProvider 的传输层参数。

    超时的最终映射（刻意**不**把 timeout_seconds 当成单一整体超时塞进
    httpx.Timeout(timeout_seconds)——那会让 read 从「无上限」一步跳到 120 秒，
    是一个没人声明过的行为变更）：

    - timeout_seconds        -> httpx.Timeout(write=...)   逐字对齐它原本的出处
    - read_timeout_seconds   -> httpx.Timeout(read=...)    chunk 间隔上限
    - total_timeout_seconds  -> asyncio.timeout(...)       一次请求的总截止
    - connect / pool 不可配，见 CONNECT_TIMEOUT_SECONDS / POOL_TIMEOUT_SECONDS
    """
    return {
        "max_attempts": cfg.max_attempts,
        "transport_max_attempts": cfg.transport_max_attempts,
        "timeout_seconds": cfg.timeout_seconds,
        "read_timeout_seconds": cfg.read_timeout_seconds,
        "total_timeout_seconds": cfg.total_timeout_seconds,
    }


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
            api_key=settings.gemini_api_key.get_secret_value(),
            model=cfg.model,
            max_attempts=cfg.max_attempts,
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
            **_transport_kwargs(cfg),
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
            **_transport_kwargs(cfg),
        )

    raise ValueError(f"不支持的 LLM provider：{cfg.provider}")
