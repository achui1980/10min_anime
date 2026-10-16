from pathlib import Path

import pytest

from tenmin import animeko
from tenmin.config import load_project
from tenmin.pipeline import auto_fetch_glossary

DATA = {
    "subject": {"name": "x"},
    "characters": [
        {"role": 1, "character": {"name": "猫猫", "nameCn": "猫猫"}},
        {"role": 2, "character": {"name": "高順", "nameCn": "高顺"}},
    ],
}


def _project(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "project.yaml"
    path.write_text(
        f"show: 测试\nslug: t\n# 注释\nglossary:\n  高順: 高順君\n{extra}", encoding="utf-8"
    )
    return path


def test_without_subject_does_nothing(tmp_path, monkeypatch):
    cfg = load_project(_project(tmp_path, ""))
    monkeypatch.setattr(animeko, "load", lambda *a, **k: pytest.fail("不应联网"))
    assert auto_fetch_glossary(cfg) == (cfg, [])


def test_with_subject_appends_without_overwrite(tmp_path, monkeypatch):
    path = _project(tmp_path, "animeko_subject: 568244\n")
    monkeypatch.setattr(animeko, "load", lambda root, subject, **k: DATA)
    cfg, warnings = auto_fetch_glossary(load_project(path))
    assert warnings == []
    assert cfg.glossary == {"高順": "高順君", "猫猫": "猫猫"}
    assert "# 注释" in path.read_text(encoding="utf-8")


def test_fetch_failure_only_warns(tmp_path, monkeypatch):
    path = _project(tmp_path, "animeko_subject: 568244\n")

    def boom(*a, **k):
        raise ValueError("访问 animeko 失败")

    monkeypatch.setattr(animeko, "load", boom)
    before = path.read_text(encoding="utf-8")
    cfg, warnings = auto_fetch_glossary(load_project(path))
    assert "访问 animeko 失败" in warnings[0]
    assert path.read_text(encoding="utf-8") == before
    assert cfg.glossary == {"高順": "高順君"}
