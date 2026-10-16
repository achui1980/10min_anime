"""管线 ingest 前按截图自动识别 OP/ED 并写回 project.yaml。"""

from pathlib import Path

import pytest

from tenmin import importer
from tenmin.config import load_project
from tenmin.pipeline import auto_match_credits


def _project(tmp_path: Path, episode: str, credits: str = "") -> Path:
    (tmp_path / "e1.mp4").write_bytes(b"v")
    (tmp_path / "op.png").write_bytes(b"i")
    (tmp_path / "ed.png").write_bytes(b"i")
    yaml_path = tmp_path / "project.yaml"
    yaml_path.write_text(
        "show: 测试\n"
        "slug: t\n"
        "# 注释保留\n"
        "episodes:\n"
        f"  - number: 1\n    video: e1.mp4\n{episode}"
        "credits:\n"
        "  op_reference: op.png\n"
        "  ed_reference: ed.png\n"
        f"{credits}",
        encoding="utf-8",
    )
    return yaml_path


@pytest.fixture
def matcher(monkeypatch):
    calls = []

    def fake_hashes(cfg, refs):
        return dict.fromkeys(refs, 1)

    def fake_match(cfg, candidates, refs):
        calls.append([number for number, _, _ in candidates])
        return {
            number: (
                importer.AvSide((63.5, 153.5), distance=5, estimated=True),
                importer.AvSide((1404.6, 1440.0), distance=3),
            )
            for number, _, _ in candidates
        }

    monkeypatch.setattr(importer, "reference_hashes", fake_hashes)
    monkeypatch.setattr(importer, "match_credits", fake_match)
    return calls


def test_writes_floored_ranges_and_reloads(tmp_path, matcher, capsys):
    yaml_path = _project(tmp_path, "")
    cfg, warnings = auto_match_credits(load_project(yaml_path), [1])
    assert warnings == []
    assert cfg.episodes[0].op_range == (63.0, 153.0)
    assert cfg.episodes[0].ed_range == (1404.0, 1440.0)
    text = yaml_path.read_text(encoding="utf-8")
    assert "# 注释保留" in text
    assert "OP 63–153 秒（估计）" in capsys.readouterr().out


def test_manual_range_is_not_matched(tmp_path, matcher):
    yaml_path = _project(tmp_path, "    op_range: [10, 100]\n")
    before = yaml_path.read_text(encoding="utf-8")
    auto_match_credits(load_project(yaml_path), [1])
    assert matcher == [[]] or matcher == []
    assert yaml_path.read_text(encoding="utf-8") == before


def test_project_default_skips_matching(tmp_path, matcher):
    yaml_path = _project(tmp_path, "", "  default_op_range: [0, 90]\n")
    auto_match_credits(load_project(yaml_path), [1])
    assert matcher == []


def test_one_side_failure_writes_nothing(tmp_path, monkeypatch):
    yaml_path = _project(tmp_path, "")
    before = yaml_path.read_text(encoding="utf-8")
    monkeypatch.setattr(importer, "reference_hashes", lambda cfg, refs: {"op": 1})
    monkeypatch.setattr(
        importer,
        "match_credits",
        lambda cfg, candidates, refs: {
            1: (importer.AvSide((0.5, 90.5)), importer.AvSide(None, reason="没找到"))
        },
    )
    cfg, warnings = auto_match_credits(load_project(yaml_path), [1])
    assert len(warnings) == 1 and "没找到" in warnings[0]
    assert cfg.episodes[0].op_range is None
    assert yaml_path.read_text(encoding="utf-8") == before
