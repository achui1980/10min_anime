import pytest
import yaml
from typer.testing import CliRunner

from tenmin.cli import app

runner = CliRunner()


def out(result) -> str:
    """click 8.2 起 result.output 只含 stdout，stderr 单独拿。这里合并，避免版本差异。"""
    text = result.output or ""
    try:
        text += result.stderr or ""
    except (ValueError, AttributeError):
        pass
    return text


@pytest.fixture
def work(tmp_path):
    return tmp_path / "work"


def test_init_creates_project_yaml(work):
    result = runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    assert result.exit_code == 0, out(result)
    path = work / "saijo" / "project.yaml"
    assert path.exists()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["slug"] == "saijo"
    assert data["mode"] == "single_episode"
    assert data["target_seconds"] == 240


def test_init_creates_srt_dir(work):
    runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    assert (work / "saijo" / "srt").is_dir()


def test_init_refuses_to_clobber(work):
    runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    result = runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    assert result.exit_code != 0
    assert "已存在" in out(result)


def test_run_missing_project_fails(work):
    result = runner.invoke(app, ["run", "nope", "--work-dir", str(work)])
    assert result.exit_code != 0
    assert "project.yaml" in out(result)


def test_run_unknown_stage_fails(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app, ["run", "saijo", "--work-dir", str(work), "--only", "nope"]
    )
    assert result.exit_code != 0


def _bootstrap(work, golden_srt_path):
    root = work / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "E02.srt").write_bytes(golden_srt_path.read_bytes())
    (root / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "show": "才女的侍从",
                "slug": "saijo",
                "mode": "single_episode",
                "target_seconds": 240,
                "episodes": [{"number": 2, "srt": "srt/E02.srt"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return root


def test_run_only_ingest_needs_no_api_key(work, golden_srt_path, monkeypatch):
    monkeypatch.delenv("TENMIN_GEMINI_API_KEY", raising=False)
    root = _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"]
    )
    assert result.exit_code == 0, out(result)
    assert (root / "01_dialogue" / "E02.dialogue.json").exists()


def test_run_script_stage_without_api_key_fails(work, golden_srt_path, monkeypatch):
    monkeypatch.delenv("TENMIN_GEMINI_API_KEY", raising=False)
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(app, ["run", "saijo", "--work-dir", str(work)])
    assert result.exit_code != 0
    assert "TENMIN_GEMINI_API_KEY" in out(result)


def test_inspect_prints_summary(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"])
    runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "signals"])
    result = runner.invoke(
        app, ["inspect", "saijo", "--work-dir", str(work), "--episode", "2"]
    )
    assert result.exit_code == 0, out(result)
    text = out(result)
    assert "对白轨" in text
    assert "无字幕间隙" in text
    assert "153.486" in text


def test_inspect_shows_suspect_lines(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"])
    runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "signals"])
    result = runner.invoke(
        app, ["inspect", "saijo", "--work-dir", str(work), "--episode", "2", "--suspect"]
    )
    assert result.exit_code == 0, out(result)
    assert "80-08" in out(result)


def test_inspect_without_ingest_fails(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app, ["inspect", "saijo", "--work-dir", str(work), "--episode", "2"]
    )
    assert result.exit_code != 0
