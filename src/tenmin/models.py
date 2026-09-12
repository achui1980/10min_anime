"""全部数据模型。阶段之间只通过这些模型的 JSON 序列化通信。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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

SignalSource = Literal["gap", "low_density", "density_shift"]
SfxKind = Literal["impact", "whoosh", "comedy", "suspense", "uplift"]
BeatRole = Literal["hook", "act", "climax", "outro"]
OriginalAudio = Literal["duck", "mute", "full"]


class RawCue(BaseModel):
    """SRT 解析器的直接产物，未做任何清洗。text 保留内部换行。"""

    idx: int
    start: float
    end: float
    text: str


class DialogueLine(BaseModel):
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


class DialogueTrack(BaseModel):
    episode: int
    source: Literal["srt", "asr"] = "srt"
    duration: float
    op_range: tuple[float, float] | None = None
    ed_range: tuple[float, float] | None = None
    lines: list[DialogueLine] = Field(default_factory=list)


class Signal(BaseModel):
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


class Highlight(BaseModel):
    """多路信号聚类后的高能点。"""

    start: float
    end: float
    strength: int = Field(ge=STRENGTH_MIN, le=STRENGTH_MAX)
    triggers: list[str] = Field(default_factory=list)
    summary: str = ""
    anchor_lines: list[int] = Field(default_factory=list)


class SignalReport(BaseModel):
    episode: int
    silent_gaps: list[Signal] = Field(default_factory=list)
    median_char_rate: float = 0.0
    highlights: list[Highlight] = Field(default_factory=list)


class Clip(BaseModel):
    """一段要截取的原片。visual 是「建议画面特征」列，v5 会当视觉检索 query。"""

    episode: int
    start: float
    end: float
    visual: str = ""
    anchor_lines: list[int] = Field(default_factory=list)
    is_silent_highlight: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


class Hold(BaseModel):
    """旁白让位、原声顶上的留白。at 相对本 beat 旁白起点。必须计入时长预算。"""

    at: float
    duration: float
    quote: str
    note: str = ""


class SfxCue(BaseModel):
    at: float
    cue: SfxKind
    note: str = ""


class AudioDirection(BaseModel):
    original_audio: OriginalAudio = "duck"
    sfx: list[SfxCue] = Field(default_factory=list)
    holds: list[Hold] = Field(default_factory=list)


class Beat(BaseModel):
    """对照表的一行。clips 是列表，支持跨时间点拼接（对照表里用「接」连起来）。"""

    id: str
    label: str
    role: BeatRole
    narration: str
    clips: list[Clip] = Field(default_factory=list)
    audio: AudioDirection = Field(default_factory=AudioDirection)
    est_seconds: float = 0.0


class Script(BaseModel):
    show: str
    mode: Literal["single_episode", "season"] = "single_episode"
    episodes: list[int] = Field(default_factory=list)
    target_seconds: float = 240.0
    est_total_seconds: float = 0.0
    beats: list[Beat] = Field(default_factory=list)


# --- 以下三个模型只用于喂 LLM 的 response_schema ---
# 刻意不含 est_seconds / est_total_seconds / is_silent_highlight：
# 这三个字段由 budget.py 与 validate.py 计算，让 LLM 填只会引入噪声。


class LLMClip(BaseModel):
    episode: int
    start: float
    end: float
    visual: str
    anchor_lines: list[int] = Field(default_factory=list)


class LLMBeat(BaseModel):
    id: str
    label: str
    role: BeatRole
    narration: str
    clips: list[LLMClip] = Field(default_factory=list)
    original_audio: OriginalAudio = "duck"
    holds: list[Hold] = Field(default_factory=list)
    sfx: list[SfxCue] = Field(default_factory=list)


class LLMScript(BaseModel):
    beats: list[LLMBeat] = Field(default_factory=list)


# --- v2 渲染阶段的模型 ---


class VoiceChunk(BaseModel):
    """一段独立合成的旁白。duration 是 TTS 出来后实测的，不是估算。"""

    beat_id: str
    index: int
    text: str
    path: str
    duration: float
    hold_after: float = 0.0


class VoiceTrack(BaseModel):
    """一集的全部旁白 chunk。total_seconds 含 hold 静音。"""

    episode: int
    chunks: list[VoiceChunk] = Field(default_factory=list)
    total_seconds: float = 0.0


class TimelineSegment(BaseModel):
    """一段画面。source_* 是原片坐标，timeline_* 是成片坐标。"""

    beat_id: str
    source_start: float
    source_end: float
    timeline_start: float
    timeline_end: float

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start


class SubtitleCue(BaseModel):
    """一条烧进画面的旁白字幕。坐标是成片坐标。"""

    start: float
    end: float
    text: str


class Timeline(BaseModel):
    """v2 的人工编辑面。改完它跑 --from audio 就能重出片。"""

    episode: int
    segments: list[TimelineSegment] = Field(default_factory=list)
    subtitles: list[SubtitleCue] = Field(default_factory=list)
    narration_offsets: list[float] = Field(default_factory=list)
    total_seconds: float = 0.0
