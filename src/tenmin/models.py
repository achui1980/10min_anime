"""全部数据模型。阶段之间只通过这些模型的 JSON 序列化通信。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

LineKind = Literal["dialogue", "monologue", "screen_text", "credits", "noise"]

# 「这一行算有人在说话」的唯一 kind 白名单。screen_text / credits / noise 都不算，
# 所以它们不会打断静默间隙、也不进字密度统计。曾经在 signals 与 ingest 里各有一份
# 平行定义，导致过滤规则悄悄分叉；判定逻辑本身见 tenmin.intervals.is_spoken。
SPEECH_KINDS: frozenset[LineKind] = frozenset({"dialogue", "monologue"})

# 信号/高光强度的取值范围。signals/aggregate.py 的强度上限与 config.SignalsConfig
# 的强度字段约束都从这里取，三处不会再各写一份字面量（曾经 aggregate.MAX_STRENGTH=5
# 与下面两个 Field(le=...) 是隐式耦合，改一处不改另一处就直接 ValidationError）。
STRENGTH_MIN = 1
STRENGTH_MAX = 5

# Hold.duration 的硬上界（秒）。这是「物理合法性」边界，不是「写得好不好」的判断：
# 实测 work/ 下 11 份 + tests/ 下 1 份真实 script.json 共 69 个 hold，duration 全部落在
# 2.0–4.0 秒（直方图 2.0×7 / 2.5×19 / 3.0×34 / 3.5×4 / 4.0×5），prompt
# (script/prompts/single_episode.md:60) 要求的也是「一般 2–4 秒」。这里取 15 秒 ≈ 实测
# 上界的 3.75 倍，只拦「模型把 3 写成 30」这类既往成片里插一整段死寂（render/chunks.py
# 把它变成真静音）、又让时长预算彻底失真（script/budget.py 直接把它计入总时长）的离谱值，
# 正常创作空间一律放过。
# 「超过 N 秒就该给个 warning」是业务合理性判断，属于 script/validate.py 的职责，不在这里管。
HOLD_MAX_SECONDS = 15.0


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("不能是纯空白字符串")
    return value


def _reject_non_positive(value: float) -> float:
    if value <= 0:
        raise ValueError("必须大于 0")
    return value


# 非空、且不能只有空白的字符串。
# JSON schema 里只体现为 minLength: 1，纯空白的那一半由 AfterValidator 兜住 —— 后者不进
# schema，所以不会给 Gemini 的 response_schema 引入它不认识的关键字（见下面 _StageModel
# 的注释）。
NonBlankStr = Annotated[str, Field(min_length=1), AfterValidator(_reject_blank)]

# 「大于 0 的秒数」。刻意用 ge=0 + AfterValidator 而不是 pydantic 的 gt=0：
# gt 会在 JSON schema 里生成 exclusiveMinimum，而 google.genai 的 types.Schema 没有这个
# 字段（只有 minimum/maximum/min_length/additional_properties），于是 GeminiProvider 在
# 发请求时 t_schema(LLMScript) 会直接 ValidationError。Hold 是 LLM schema 的一部分，
# 所以这条约束必须用 Gemini 认识的关键字来表达。
PositiveSeconds = Annotated[float, Field(ge=0), AfterValidator(_reject_non_positive)]


def _reject_duplicate_beat_ids(beats: Sequence[Beat] | Sequence[LLMBeat]) -> None:
    """beat.id 是渲染阶段的连接键，重复了会静默串台，必须直接判错。

    render/timeline.py:73 把配音 chunk 按 chunk.beat_id 收进一个 dict，再在 :90 用
    grouped.get(beat.id) 取回 —— 两个同 id 的 beat 会各自拿到两段的**全部** chunk，
    第二段静默复读第一段的音频。下游没有任何办法察觉或恢复，而重复 id 也不可能是有意的
    人工编辑，所以这里不做降级。
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for beat in beats:
        if beat.id in seen and beat.id not in duplicates:
            duplicates.append(beat.id)
        seen.add(beat.id)
    if duplicates:
        raise ValueError(f"beat id 重复：{'、'.join(duplicates)}")


class _StageModel(BaseModel):
    """所有阶段间模型的公共基类，只做一件事：extra="forbid"。

    阶段产物 JSON 一半是 LLM 写的、一半是允许人手改的（script.json 与 timeline.json 都是
    文档里写明的人工编辑面）。pydantic 默认的 extra="ignore" 会把 `dur` / `duraton` 这类
    拼错的键静默丢掉，字段悄悄退回默认值，错误一路漂到成片里才以「人眼才能发现」的形式爆
    出来。forbid 让拼错的键在读入那一刻就报错；对 LLM 输出来说，这个 ValidationError 正好
    会触发 OpenAICompatibleProvider 已有的 schema 修复重试（script/llm.py:176）。

    实测 work/ 下 42 份真实产物（script/voice/timeline/dialogue/signals）与
    tests/fixtures、tests/snapshots 里的样本，键集与模型字段完全一致，不会误伤旧产物。

    刻意**不**设 frozen=True：script/validate.py 与 script/budget.py 当前依赖就地改写
    （clip.start 覆写、beat.est_seconds 回填），冻结会大面积炸。
    """

    model_config = ConfigDict(extra="forbid")


SignalSource = Literal["gap", "low_density", "density_shift"]
SfxKind = Literal["impact", "whoosh", "comedy", "suspense", "uplift"]
BeatRole = Literal["hook", "act", "climax", "outro"]
OriginalAudio = Literal["duck", "mute", "full"]


class RawCue(_StageModel):
    """SRT 解析器的直接产物，未做任何清洗。text 保留内部换行。"""

    idx: int
    start: float
    end: float
    text: str


class DialogueLine(_StageModel):
    """标准化后的一行对白。不删任何行，只打标，由下游按 kind 过滤。"""

    idx: int
    start: float
    end: float
    text: str
    raw: str
    speaker: str | None = None
    kind: LineKind = "dialogue"
    suspect: bool = False
    merged_from: list[int] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class DialogueTrack(_StageModel):
    episode: int
    source: Literal["srt", "asr"] = "srt"
    duration: float
    op_range: tuple[float, float] | None = None
    ed_range: tuple[float, float] | None = None
    lines: list[DialogueLine] = Field(default_factory=list)


class Signal(_StageModel):
    """单条检测规则的原始输出，聚类前的中间产物。"""

    start: float
    end: float
    source: SignalSource
    strength: int = Field(ge=STRENGTH_MIN, le=STRENGTH_MAX)
    detail: str
    anchor_lines: list[int] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class Highlight(_StageModel):
    """多路信号聚类后的高能点。"""

    start: float
    end: float
    strength: int = Field(ge=STRENGTH_MIN, le=STRENGTH_MAX)
    triggers: list[str] = Field(default_factory=list)
    summary: str = ""
    anchor_lines: list[int] = Field(default_factory=list)


class SignalReport(_StageModel):
    episode: int
    silent_gaps: list[Signal] = Field(default_factory=list)
    median_char_rate: float = 0.0
    highlights: list[Highlight] = Field(default_factory=list)


class Clip(_StageModel):
    """一段要截取的原片。visual 是「建议画面特征」列，v5 会当视觉检索 query。

    刻意**不**给 start/end 加 ge=0 或 end > start 的硬约束：坏 clip 的拦网在
    script/validate.py:132-142，那里是「丢弃这一条、保留其余、附一条 warning」的优雅降级。
    而 script/single.py:92 的 to_script() 是直接拿 LLM 的原始数字构造本模型的，跑在
    validate_script() 之前；一旦这里抛 ValidationError，single.py:135 只 catch
    ScriptValidationError，异常会直接逃出去把整次运行打死 —— 18 个 clip 里坏 1 个就全盘报废。
    """

    episode: int
    start: float
    end: float
    visual: str = ""
    anchor_lines: list[int] = Field(default_factory=list)
    is_silent_highlight: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


class Hold(_StageModel):
    """旁白让位、原声顶上的留白。at 相对本 beat 旁白起点。必须计入时长预算。"""

    at: float = Field(ge=0, description="相对本节点旁白起点的秒数，不能为负")
    duration: PositiveSeconds = Field(
        ge=0,
        le=HOLD_MAX_SECONDS,
        description=f"留白时长（秒），必须大于 0 且不超过 {HOLD_MAX_SECONDS:.0f}，一般 2–4 秒",
    )
    quote: str
    note: str = ""


class SfxCue(_StageModel):
    at: float = Field(ge=0, description="相对本节点旁白起点的秒数，不能为负")
    cue: SfxKind
    note: str = ""


class AudioDirection(_StageModel):
    original_audio: OriginalAudio = "duck"
    sfx: list[SfxCue] = Field(default_factory=list)
    holds: list[Hold] = Field(default_factory=list)


class Beat(_StageModel):
    """对照表的一行。clips 是列表，支持跨时间点拼接（对照表里用「接」连起来）。

    narration 刻意**不**加非空约束（与 LLMBeat 相反）：人手改 script.json 时清空某段旁白
    是合法编辑（只要画面不要解说），render/tts.py:75 会给出 warning 并跳过配音。
    """

    id: NonBlankStr
    label: str
    role: BeatRole
    narration: str
    clips: list[Clip] = Field(default_factory=list)
    audio: AudioDirection = Field(default_factory=AudioDirection)
    est_seconds: float = 0.0


class Script(_StageModel):
    show: str
    mode: Literal["single_episode", "season"] = "single_episode"
    episodes: list[int] = Field(default_factory=list)
    target_seconds: float = 240.0
    est_total_seconds: float = 0.0
    beats: list[Beat] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_beat_ids(self) -> Script:
        _reject_duplicate_beat_ids(self.beats)
        return self


# --- 以下三个模型只用于喂 LLM 的 response_schema ---
# 刻意不含 est_seconds / est_total_seconds / is_silent_highlight：
# 这三个字段由 budget.py 与 validate.py 计算，让 LLM 填只会引入噪声。
#
# 严格度分工：能被「丢弃单条、保留其余」优雅降级的错误（clip 时间窗越界／倒挂）留给
# validate.py，这里保持宽松；无法降级、只能重来的错误（空 id、空 narration、重复 id、
# 拼错的键）在这里就判死 —— OpenAICompatibleProvider 会 catch 这个 ValidationError 并把
# 报错文本回灌给模型重试（script/llm.py:174-180），比放过去更省一轮返工。


class LLMClip(_StageModel):
    episode: int
    start: float
    end: float
    visual: str
    anchor_lines: list[int] = Field(default_factory=list)


class LLMBeat(_StageModel):
    id: NonBlankStr
    label: str
    role: BeatRole
    narration: NonBlankStr
    clips: list[LLMClip] = Field(default_factory=list)
    original_audio: OriginalAudio = "duck"
    holds: list[Hold] = Field(default_factory=list)
    sfx: list[SfxCue] = Field(default_factory=list)


class LLMScript(_StageModel):
    beats: list[LLMBeat] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_beat_ids(self) -> LLMScript:
        _reject_duplicate_beat_ids(self.beats)
        return self


# --- v2 渲染阶段的模型 ---


class VoiceChunk(_StageModel):
    """一段独立合成的旁白。duration 是 TTS 出来后实测的，不是估算。"""

    beat_id: str
    index: int
    text: str
    path: str
    duration: float
    hold_after: float = 0.0


class VoiceTrack(_StageModel):
    """一集的全部旁白 chunk。total_seconds 含 hold 静音。"""

    episode: int
    chunks: list[VoiceChunk] = Field(default_factory=list)
    total_seconds: float = 0.0


class TimelineSegment(_StageModel):
    """一段画面。source_* 是原片坐标，timeline_* 是成片坐标。"""

    beat_id: str
    source_start: float
    source_end: float
    timeline_start: float
    timeline_end: float

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start


class SubtitleCue(_StageModel):
    """一条烧进画面的旁白字幕。坐标是成片坐标。"""

    start: float
    end: float
    text: str


class Timeline(_StageModel):
    """v2 的人工编辑面。改完它跑 --from audio 就能重出片。"""

    episode: int
    segments: list[TimelineSegment] = Field(default_factory=list)
    subtitles: list[SubtitleCue] = Field(default_factory=list)
    narration_offsets: list[float] = Field(default_factory=list)
    total_seconds: float = 0.0
