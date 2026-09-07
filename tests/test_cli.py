import pytest
import yaml
from typer.testing import CliRunner

from tenmin.cli import app
from tenmin.render.ffmpeg import FFmpegError

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


def _minimal_project(tmp_path):
    root = tmp_path / "akujo2"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: 我是不才恶女\n"
        "slug: akujo2\n"
        "episodes:\n"
        "- number: 2\n"
        "  srt: srt/E02.srt\n",
        encoding="utf-8",
    )
    return root


def test_init_template_has_video_and_render(tmp_path):
    result = runner.invoke(app, ["init", "akujo2", "--work-dir", str(tmp_path)])

    assert result.exit_code == 0
    data = yaml.safe_load(
        (tmp_path / "akujo2" / "project.yaml").read_text(encoding="utf-8")
    )
    assert data["episodes"][0]["video"] == "video/E02.mkv"
    assert data["render"]["voice"] == "zh-CN-YunxiNeural"
    assert data["render"]["rate"] == "+0%"
    assert data["render"]["video_encoder"] == "libx264"
    assert data["render"]["duck_db"] == -12.0
    assert data["render"]["font_size"] == 52
    assert data["render"]["fade_out_seconds"] == 1.5
    assert data["render"]["outro_card_seconds"] == 3.0
    assert data["render"]["outro_message"] == "解说结束，谢谢观看"
    assert (tmp_path / "akujo2" / "video").is_dir()


def test_run_passes_tts_engine_when_voice_wanted(tmp_path, monkeypatch):
    _minimal_project(tmp_path)
    sentinel = object()
    monkeypatch.setattr("tenmin.cli.build_tts_engine", lambda cfg: sentinel)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "voice"]
    )

    assert result.exit_code == 0
    assert captured["tts_engine"] is sentinel


def test_run_skips_tts_engine_when_voice_not_wanted(tmp_path, monkeypatch):
    _minimal_project(tmp_path)

    def boom(cfg):
        raise AssertionError("只跑 docgen 不该造 TTS engine")

    monkeypatch.setattr("tenmin.cli.build_tts_engine", boom)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "docgen"]
    )

    assert result.exit_code == 0
    assert captured["tts_engine"] is None


def test_run_reports_ffmpeg_error(tmp_path, monkeypatch):
    _minimal_project(tmp_path)

    async def boom(cfg, provider, **kwargs):
        raise FFmpegError("你的 ffmpeg 没编 libass")

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "render"]
    )

    assert result.exit_code == 1
    assert "没编 libass" in result.output


def test_run_prints_mp4_path(tmp_path, monkeypatch):
    root = _minimal_project(tmp_path)
    mp4 = root / "07_render" / "E02.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"")

    async def fake_pipeline(cfg, provider, **kwargs):
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "render"]
    )

    assert result.exit_code == 0
    assert "成品视频" in result.output
    assert str(mp4) in result.output
