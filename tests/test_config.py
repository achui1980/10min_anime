import pytest
from pydantic import ValidationError

from tenmin.config import ProjectConfig, Settings, load_project

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
  model: gemini-2.5-pro
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
    assert cfg.llm.model == "gemini-2.5-pro"


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
