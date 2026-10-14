"""原始图片评级、缓存身份与检查结果的数据契约。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_LABELS = ("general", "sensitive", "questionable", "explicit")
_SCORE_SUM_TOLERANCE = 0.001
_IMAGE_PATH = re.compile(r"frames/frame_[0-9]{6,}\.jpg")


class ContentCheckError(RuntimeError):
    """内容检查无法完成的可预期执行错误。"""


class Scores(BaseModel):
    """模型的四类原始图片评级分数，不做重归一化。"""

    model_config = ConfigDict(extra="forbid")

    general: float = Field(ge=0, le=1, allow_inf_nan=False)
    sensitive: float = Field(ge=0, le=1, allow_inf_nan=False)
    questionable: float = Field(ge=0, le=1, allow_inf_nan=False)
    explicit: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_sum(self) -> Scores:
        total = sum(getattr(self, label) for label in _LABELS)
        if not 1 - _SCORE_SUM_TOLERANCE <= total <= 1 + _SCORE_SUM_TOLERANCE:
            raise ValueError("scores.general/sensitive/questionable/explicit 的和必须接近 1")
        return self


class FrameScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: float = Field(ge=0, allow_inf_nan=False)
    image: str
    label: Literal["general", "sensitive", "questionable", "explicit"]
    scores: Scores

    @field_validator("image")
    @classmethod
    def check_image(cls, value: str) -> str:
        if _IMAGE_PATH.fullmatch(value) is None:
            raise ValueError("image 必须为 frames/frame_000001.jpg 形式的相对路径")
        return value

    @model_validator(mode="after")
    def check_label(self) -> FrameScore:
        expected = max(_LABELS, key=lambda label: getattr(self.scores, label))
        if self.label != expected:
            raise ValueError(f"label 必须等于 scores 按固定类序取 argmax 的评级 {expected}")
        return self


class CandidateRange(BaseModel):
    """候选采样点的首尾时间，不表示精确片段边界。"""

    model_config = ConfigDict(extra="forbid")

    first: float = Field(ge=0, allow_inf_nan=False)
    last: float = Field(ge=0, allow_inf_nan=False)
    count: int = Field(ge=1)

    @model_validator(mode="after")
    def check_bounds(self) -> CandidateRange:
        if self.first > self.last:
            raise ValueError("first 必须小于或等于 last")
        return self


class CacheIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_path: str
    video_size: int = Field(ge=0)
    video_mtime_ns: int = Field(ge=0)
    repository: str
    revision: str
    model: str
    model_sha256: str
    preprocessing_version: str
    sampling_version: str
    sample_interval: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("video_path")
    @classmethod
    def check_video_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("video_path 必须为已解析的绝对路径")
        return value


class CheckManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    identity: CacheIdentity
    providers: list[str]
    extraction_seconds: float = Field(ge=0, allow_inf_nan=False)
    inference_seconds: float = Field(ge=0, allow_inf_nan=False)
    sensitive_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    adult_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    frames: list[FrameScore] = Field(min_length=1)
    candidate_count: int = Field(ge=0)
    candidate_ranges: list[CandidateRange]

    @model_validator(mode="after")
    def check_frames_and_candidates(self) -> CheckManifest:
        if any(a.time > b.time for a, b in zip(self.frames, self.frames[1:], strict=False)):
            raise ValueError("frames.time 必须非递减")
        if len({frame.image for frame in self.frames}) != len(self.frames):
            raise ValueError("frames.image 必须唯一")
        if self.candidate_count > len(self.frames):
            raise ValueError("candidate_count 不能超过 frames 数量")
        for candidate in self.candidate_ranges:
            if (candidate.first < self.frames[0].time
                    or candidate.last > self.frames[-1].time):
                raise ValueError("candidate_ranges.first/last 必须在 frames 时间范围内")
            available = sum(
                candidate.first <= frame.time <= candidate.last for frame in self.frames
            )
            if candidate.count > available:
                raise ValueError("candidate_ranges.count 不能超过范围内的 frames 数量")
        if sum(candidate.count for candidate in self.candidate_ranges) != self.candidate_count:
            raise ValueError("candidate_count 必须等于 candidate_ranges.count 的和")
        return self


@dataclass(frozen=True)
class SampledFrame:
    path: Path
    time: float


@dataclass(frozen=True)
class ModelFiles:
    weights: Path
    metadata: Path
    sha256: str


@dataclass(frozen=True)
class CheckOutcome:
    status: Literal["completed", "cached", "report_rebuilt"]
    candidate_count: int
    frame_count: int
    report_path: Path
    results_path: Path
