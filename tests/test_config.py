from pathlib import Path

import pytest
from pydantic import ValidationError

from tenmin.config import (
    EpisodeConfig,
    LLMConfig,
    ProjectConfig,
    RenderConfig,
    Settings,
    load_project,
)

SAMPLE = """\
show: 才女的侍从
slug: saijo
mode: single_episode
target_seconds: 240
locale:
  convert_traditional: true
episodes:
  - number: 2
    srt: srt/E02.srt
    op_range: [153.486, 224.681]
glossary:
  伊月: 伊月
llm:
  provider: gemini
  model: gemini-3.6-flash
"""

MINIMAL = """\
show: 某番
slug: demo
episodes:
  - number: 1
    srt: srt/E01.srt
"""


def test_load_project_full(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(SAMPLE, encoding="utf-8")
    cfg = load_project(path)
    assert cfg.show == "才女的侍从"
    assert cfg.slug == "saijo"
    assert cfg.mode == "single_episode"
    assert cfg.target_seconds == pytest.approx(240.0)
    assert cfg.locale.convert_traditional is True
    assert len(cfg.episodes) == 1
    assert cfg.episodes[0].number == 2
    assert cfg.episodes[0].op_range == (153.486, 224.681)
    assert cfg.episodes[0].ed_range is None
    assert cfg.glossary == {"伊月": "伊月"}
    assert cfg.llm.model == "gemini-3.6-flash"


def test_load_project_minimal_applies_defaults(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    cfg = load_project(path)
    assert cfg.mode == "single_episode"
    assert cfg.target_seconds == pytest.approx(240.0)
    assert cfg.locale.convert_traditional is True
    assert cfg.glossary == {}
    assert cfg.llm.provider == "gemini"


def test_load_project_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "nope.yaml")


def test_load_project_rejects_unknown_mode(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(MINIMAL.replace("slug: demo", "slug: demo\nmode: whatever"), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_project(path)


def test_episode_srt_path_resolves_against_project_dir(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    cfg = load_project(path)
    assert cfg.root == tmp_path
    assert cfg.srt_path(cfg.episodes[0]) == tmp_path / "srt" / "E01.srt"


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("TENMIN_GEMINI_API_KEY", "fake-key")
    assert Settings().gemini_api_key == "fake-key"


def test_settings_missing_key_is_none(monkeypatch):
    monkeypatch.delenv("TENMIN_GEMINI_API_KEY", raising=False)
    assert Settings(_env_file=None).gemini_api_key is None


def test_project_config_is_constructible_in_memory():
    cfg = ProjectConfig(show="X", slug="x", episodes=[{"number": 1, "srt": "a.srt"}])
    assert cfg.episodes[0].srt.name == "a.srt"


def test_llm_config_base_url_defaults_to_none():
    cfg = LLMConfig()
    assert cfg.provider == "gemini"
    assert cfg.model == "gemini-3.6-flash"
    assert cfg.base_url is None


def test_llm_config_accepts_minimax_provider():
    cfg = LLMConfig(provider="minimax", model="MiniMax-M3")
    assert cfg.provider == "minimax"
    assert cfg.model == "MiniMax-M3"
    assert cfg.base_url is None


def test_llm_config_accepts_base_url_override():
    cfg = LLMConfig(provider="minimax", base_url="https://proxy.test/v1")
    assert cfg.base_url == "https://proxy.test/v1"


def test_llm_config_rejects_unknown_provider():
    with pytest.raises(ValidationError):
        LLMConfig(provider="openai")


def test_llm_config_thinking_defaults_to_disabled():
    """MiniMax-M3 默认关闭深度思考，省掉 <think> 推理块的耗时（结果本来就被丢弃）。"""
    cfg = LLMConfig()
    assert cfg.thinking == "disabled"


def test_llm_config_accepts_adaptive_thinking():
    cfg = LLMConfig(provider="minimax", thinking="adaptive")
    assert cfg.thinking == "adaptive"


def test_llm_config_rejects_unknown_thinking():
    with pytest.raises(ValidationError):
        LLMConfig(thinking="always")


def test_load_project_with_minimax_llm(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(
        MINIMAL + "llm:\n  provider: minimax\n  model: MiniMax-M3\n"
        "  base_url: https://proxy.test/v1\n",
        encoding="utf-8",
    )
    cfg = load_project(path)
    assert cfg.llm.provider == "minimax"
    assert cfg.llm.model == "MiniMax-M3"
    assert cfg.llm.base_url == "https://proxy.test/v1"


def test_settings_reads_minimax_env(monkeypatch):
    monkeypatch.setenv("TENMIN_MINIMAX_API_KEY", "mm-key")
    assert Settings().minimax_api_key == "mm-key"


def test_settings_missing_minimax_key_is_none(monkeypatch):
    monkeypatch.delenv("TENMIN_MINIMAX_API_KEY", raising=False)
    assert Settings(_env_file=None).minimax_api_key is None


def test_settings_keys_are_independent(monkeypatch):
    monkeypatch.setenv("TENMIN_GEMINI_API_KEY", "g-key")
    monkeypatch.delenv("TENMIN_MINIMAX_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key == "g-key"
    assert settings.minimax_api_key is None


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


def test_render_config_defaults():
    cfg = RenderConfig()
    assert cfg.voice == "zh-CN-YunxiNeural"
    assert cfg.rate == "+0%"
    assert cfg.video_encoder == "libx264"
    assert cfg.duck_db == -12.0
    assert cfg.font_size == 52
    assert cfg.fade_out_seconds == 1.5
    assert cfg.outro_card_seconds == 3.0
    assert cfg.outro_message == "解说结束，谢谢观看"


def test_project_config_has_render_defaults():
    cfg = ProjectConfig(show="剧名", slug="slug")
    assert cfg.render.voice == "zh-CN-YunxiNeural"


def test_project_config_reads_render_block():
    cfg = ProjectConfig.model_validate(
        {
            "show": "剧名",
            "slug": "slug",
            "render": {"voice": "zh-CN-XiaoxiaoNeural", "duck_db": -9.0},
        }
    )
    assert cfg.render.voice == "zh-CN-XiaoxiaoNeural"
    assert cfg.render.duck_db == -9.0
    # 没写的字段仍取默认值
    assert cfg.render.video_encoder == "libx264"


def test_video_path_resolves_relative_to_root(tmp_path):
    cfg = ProjectConfig.model_validate(
        {
            "show": "剧名",
            "slug": "slug",
            "episodes": [{"number": 2, "srt": "srt/E02.srt", "video": "video/E02.mkv"}],
        }
    ).bind_root(tmp_path)
    assert cfg.video_path(cfg.episodes[0]) == tmp_path.resolve() / "video/E02.mkv"


def test_video_path_passes_absolute_through(tmp_path):
    absolute = tmp_path / "elsewhere" / "E02.mkv"
    cfg = ProjectConfig.model_validate(
        {
            "show": "剧名",
            "slug": "slug",
            "episodes": [{"number": 2, "srt": "srt/E02.srt", "video": str(absolute)}],
        }
    ).bind_root(tmp_path)
    assert cfg.video_path(cfg.episodes[0]) == absolute


def test_video_path_without_video_raises():
    cfg = ProjectConfig.model_validate(
        {"show": "剧名", "slug": "slug", "episodes": [{"number": 2, "srt": "srt/E02.srt"}]}
    )
    with pytest.raises(ValueError) as exc:
        cfg.video_path(cfg.episodes[0])
    assert "第 2 集没有配置 video" in str(exc.value)


def test_episode_config_video_defaults_to_none():
    episode = EpisodeConfig(number=2, srt=Path("srt/E02.srt"))
    assert episode.video is None
