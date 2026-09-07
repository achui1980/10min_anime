"""project.yaml 的模型与加载，以及环境变量读取。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EpisodeConfig(BaseModel):
    number: int
    srt: Path
    video: Path | None = None
    op_range: tuple[float, float] | None = None
    ed_range: tuple[float, float] | None = None


class LocaleConfig(BaseModel):
    convert_traditional: bool = True


class LLMConfig(BaseModel):
    provider: Literal["gemini", "minimax"] = "gemini"
    model: str = "gemini-3.6-flash"
    base_url: str | None = None


class RenderConfig(BaseModel):
    """v2 渲染参数。voice 与 rate 直接喂 Edge-TTS。"""

    voice: str = "zh-CN-YunxiNeural"
    rate: str = "+0%"
    video_encoder: str = "libx264"
    duck_db: float = -12.0
    font_size: int = 52
    fade_out_seconds: float = 1.5
    outro_card_seconds: float = 3.0
    outro_message: str = "解说结束，谢谢观看"


class ProjectConfig(BaseModel):
    show: str
    slug: str
    mode: Literal["single_episode", "season"] = "single_episode"
    target_seconds: float = 240.0
    locale: LocaleConfig = Field(default_factory=LocaleConfig)
    episodes: list[EpisodeConfig] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    _root: Path = PrivateAttr(default=Path("."))

    @property
    def root(self) -> Path:
        """project.yaml 所在目录，也就是 work/<slug>/。"""
        return self._root

    def bind_root(self, root: Path) -> ProjectConfig:
        """绑定项目目录（work/<slug>/）。load_project 会自动调用，测试也可直接用。"""
        self._root = Path(root).resolve()
        return self

    def srt_path(self, episode: EpisodeConfig) -> Path:
        """episode.srt 是相对 project.yaml 的路径；绝对路径原样返回。"""
        if episode.srt.is_absolute():
            return episode.srt
        return self._root / episode.srt

    def video_path(self, episode: EpisodeConfig) -> Path:
        """源视频路径。相对路径按 project.yaml 所在目录解析。"""
        if episode.video is None:
            raise ValueError(
                f"第 {episode.number} 集没有配置 video，"
                "请在 project.yaml 的 episodes 里补上源视频路径"
            )
        if episode.video.is_absolute():
            return episode.video
        return self._root / episode.video


def load_project(path: Path) -> ProjectConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到项目配置: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ProjectConfig.model_validate(data).bind_root(path.parent)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TENMIN_", env_file=".env", extra="ignore"
    )

    gemini_api_key: str | None = None
    minimax_api_key: str | None = None
