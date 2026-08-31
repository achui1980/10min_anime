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
    op_range: tuple[float, float] | None = None
    ed_range: tuple[float, float] | None = None


class LocaleConfig(BaseModel):
    convert_traditional: bool = True


class LLMConfig(BaseModel):
    provider: Literal["gemini"] = "gemini"
    model: str = "gemini-2.5-pro"


class ProjectConfig(BaseModel):
    show: str
    slug: str
    mode: Literal["single_episode", "season"] = "single_episode"
    target_seconds: float = 240.0
    locale: LocaleConfig = Field(default_factory=LocaleConfig)
    episodes: list[EpisodeConfig] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)

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
