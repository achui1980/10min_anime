from pathlib import Path

import pytest
from pydantic import ValidationError

from tenmin.config import (
    CreditsConfig,
    EpisodeConfig,
    IngestConfig,
    LLMConfig,
    ProjectConfig,
    RenderConfig,
    Settings,
    SignalsConfig,
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
    assert Settings().gemini_api_key.get_secret_value() == "fake-key"


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
    assert Settings().minimax_api_key.get_secret_value() == "mm-key"


def test_settings_missing_minimax_key_is_none(monkeypatch):
    monkeypatch.delenv("TENMIN_MINIMAX_API_KEY", raising=False)
    assert Settings(_env_file=None).minimax_api_key is None


def test_settings_keys_are_independent(monkeypatch):
    monkeypatch.setenv("TENMIN_GEMINI_API_KEY", "g-key")
    monkeypatch.delenv("TENMIN_MINIMAX_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key.get_secret_value() == "g-key"
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
    assert Settings().openai_compatible_api_key.get_secret_value() == "oc-key"


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


def test_config_path_points_at_project_yaml(tmp_path):
    cfg = ProjectConfig.model_validate(
        {"show": "剧名", "slug": "slug", "episodes": []}
    ).bind_root(tmp_path)
    assert cfg.config_path == tmp_path.resolve() / "project.yaml"


def test_config_path_matches_the_file_load_project_read(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    assert load_project(path).config_path == path.resolve()


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


# --- 约束与 validator ---


@pytest.mark.parametrize("bad", [0, 0.0, -1, -240.0])
def test_target_seconds_must_be_positive(bad):
    """target_seconds <= 0 会让 script 阶段的时长预算直接失去意义，必须挡在配置层。"""
    with pytest.raises(ValidationError):
        ProjectConfig(show="X", slug="x", target_seconds=bad)


def test_target_seconds_accepts_positive():
    assert ProjectConfig(show="X", slug="x", target_seconds=1.0).target_seconds == 1.0


@pytest.mark.parametrize("field", ["op_range", "ed_range"])
def test_credit_range_rejects_reversed_bounds(field):
    with pytest.raises(ValidationError) as exc:
        EpisodeConfig(number=1, srt=Path("a.srt"), **{field: (200.0, 100.0)})
    assert "起点" in str(exc.value)


@pytest.mark.parametrize("field", ["op_range", "ed_range"])
def test_credit_range_rejects_equal_bounds(field):
    with pytest.raises(ValidationError):
        EpisodeConfig(number=1, srt=Path("a.srt"), **{field: (100.0, 100.0)})


@pytest.mark.parametrize("field", ["op_range", "ed_range"])
def test_credit_range_rejects_negative_bounds(field):
    with pytest.raises(ValidationError) as exc:
        EpisodeConfig(number=1, srt=Path("a.srt"), **{field: (-1.0, 100.0)})
    assert "负数" in str(exc.value)


@pytest.mark.parametrize("field", ["op_range", "ed_range"])
def test_credit_range_accepts_valid(field):
    episode = EpisodeConfig(number=1, srt=Path("a.srt"), **{field: (0.0, 90.0)})
    assert getattr(episode, field) == (0.0, 90.0)


# --- Settings 的 API key 是 SecretStr ---


def test_api_keys_are_secret_str(monkeypatch):
    monkeypatch.delenv("TENMIN_GEMINI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        gemini_api_key="g-secret",
        minimax_api_key="m-secret",
        openai_compatible_api_key="o-secret",
    )
    assert settings.gemini_api_key.get_secret_value() == "g-secret"
    assert settings.minimax_api_key.get_secret_value() == "m-secret"
    assert settings.openai_compatible_api_key.get_secret_value() == "o-secret"


def test_api_keys_do_not_leak_into_repr():
    """误把 Settings 打进日志/异常时不能泄露 key。"""
    settings = Settings(_env_file=None, gemini_api_key="g-secret")
    assert "g-secret" not in repr(settings)
    assert "g-secret" not in str(settings.gemini_api_key)


# --- IngestConfig ---


def test_ingest_config_defaults():
    cfg = IngestConfig()
    assert cfg.merge_max_gap == pytest.approx(0.3)
    assert cfg.merge_max_chars == 40
    assert cfg.merge_max_line_seconds == pytest.approx(4.0)


def test_project_config_has_ingest_block():
    cfg = ProjectConfig.model_validate(
        {"show": "剧名", "slug": "slug", "ingest": {"merge_max_chars": 60}}
    )
    assert cfg.ingest.merge_max_chars == 60
    assert cfg.ingest.merge_max_gap == pytest.approx(0.3)


# --- CreditsConfig ---


def test_credits_config_defaults():
    cfg = CreditsConfig()
    assert cfg.op_search_start == pytest.approx(30.0)
    assert cfg.op_search_end == pytest.approx(300.0)
    assert cfg.credit_head_window == pytest.approx(300.0)
    assert cfg.op_span_min == pytest.approx(40.0)
    assert cfg.op_span_max == pytest.approx(120.0)
    assert cfg.ed_cluster_tail_seconds == pytest.approx(120.0)
    assert cfg.ed_keyword_window_seconds == pytest.approx(80.0)
    assert cfg.cluster_max_gap == pytest.approx(35.0)
    assert cfg.op_min_silent_span == pytest.approx(60.0)
    assert cfg.op_max_silent_span == pytest.approx(120.0)
    assert cfg.title_overlap_threshold == pytest.approx(0.6)
    assert cfg.title_card_max_len == 24
    assert cfg.name_list_min_cjk == 6
    assert cfg.name_list_many_segments == 4
    assert cfg.name_list_many_min_cjk == 4
    assert cfg.latin_ratio_threshold == pytest.approx(0.6)
    assert cfg.latin_min_len == 6


def test_project_config_has_credits_block():
    cfg = ProjectConfig.model_validate(
        {"show": "剧名", "slug": "slug", "credits": {"op_span_max": 200.0}}
    )
    assert cfg.credits.op_span_max == pytest.approx(200.0)
    assert cfg.credits.op_span_min == pytest.approx(40.0)


# --- SignalsConfig ---


def test_signals_config_defaults():
    cfg = SignalsConfig()
    assert cfg.min_gap_seconds == pytest.approx(3.0)
    assert cfg.gap_strong_seconds == pytest.approx(15.0)
    assert cfg.gap_medium_seconds == pytest.approx(8.0)
    assert cfg.low_density_ratio == pytest.approx(0.4)
    assert cfg.low_density_min_seconds == pytest.approx(2.0)
    assert cfg.low_density_strength == 3
    assert cfg.shift_window_seconds == pytest.approx(30.0)
    assert cfg.shift_z_threshold == pytest.approx(1.5)
    assert cfg.shift_strength == 2
    assert cfg.min_separation == pytest.approx(2.0)
    assert cfg.summary_max_chars == 30


def test_signals_strength_fields_respect_model_bounds():
    """强度字段的取值范围必须跟 models.Signal/Highlight 的 Field(ge=1, le=5) 一致。"""
    with pytest.raises(ValidationError):
        SignalsConfig(low_density_strength=6)
    with pytest.raises(ValidationError):
        SignalsConfig(shift_strength=0)


def test_project_config_has_signals_block():
    cfg = ProjectConfig.model_validate(
        {"show": "剧名", "slug": "slug", "signals": {"min_gap_seconds": 10.0}}
    )
    assert cfg.signals.min_gap_seconds == pytest.approx(10.0)
    assert cfg.signals.summary_max_chars == 30


# --- RenderConfig 新增字段 ---


def test_render_config_new_defaults():
    cfg = RenderConfig()
    assert cfg.crf == "20"
    assert cfg.preset == "medium"
    assert cfg.width == 1920
    assert cfg.height == 1080
    assert cfg.videotoolbox_bitrate == "6000k"
    assert cfg.audio_codec == "aac"
    assert cfg.audio_bitrate == "192k"
    assert cfg.subtitle_font_name == "Lantinghei SC"
    assert cfg.outro_font_name == "Lantinghei SC"
    assert cfg.tts_max_attempts == 3
    assert cfg.tts_concurrency == 1
    assert cfg.tts_proxy is None
    assert cfg.drift_tolerance == pytest.approx(0.5)
    assert cfg.ffmpeg_path == "ffmpeg"
    assert cfg.ffprobe_path == "ffprobe"


# --- LLMConfig 新增字段 ---


def test_llm_config_new_defaults():
    cfg = LLMConfig()
    assert cfg.temperature is None
    assert cfg.max_output_tokens is None
    assert cfg.timeout_seconds == pytest.approx(120.0)
    assert cfg.max_attempts == 3
    assert cfg.budget_tolerance == pytest.approx(0.12)
