"""全部数据模型。阶段之间只通过这些模型的 JSON 序列化通信。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LineKind = Literal["dialogue", "monologue", "screen_text", "credits", "noise"]
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
    strength: int = Field(ge=1, le=5)
    detail: str
    anchor_lines: list[int] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class Highlight(BaseModel):
    """多路信号聚类后的高能点。"""

    start: float
    end: float
    strength: int = Field(ge=1, le=5)
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
