"""两个检查入口共用服务，只有本次成功结果才能显示报告。"""

import asyncio

import pytest
import yaml
from typer.testing import CliRunner

from tenmin import cli, pipeline
from tenmin.config import DEFAULT_CONTENT_CHECK, DEFAULT_RENDER, ContentCheckConfig
from tenmin.content_check.models import CheckOutcome, ContentCheckError
from tenmin.pipeline import Paths

runner = CliRunner()


def out(result) -> str:
    """合并 stdout/stderr，沿用 test_cli.py 的 Click 版本兼容方式。"""
    text = result.output or ""
    try:
        text += result.stderr or ""
    except (ValueError, AttributeError):
        pass
    return text


def forbid_upstream(*args, **kwargs):
    pytest.fail("检查入口不应构造项目、LLM 或 TTS")


def test_independent_command_needs_no_project_or_provider(tmp_path, monkeypatch):
    video = tmp_path / "视频.mp4"
    video.write_bytes(b"video")
    seen = []

    def fake(video, output_dir, **kwargs):
        seen.append((video, output_dir, kwargs))
        return CheckOutcome("completed", 3, 10, output_dir / "report.html",
                            output_dir / "results.json")

    monkeypatch.setattr(cli, "check_video", fake, raising=False)
    monkeypatch.setattr(cli, "load_project", forbid_upstream)
    monkeypatch.setattr(cli, "Settings", forbid_upstream)
    monkeypatch.setattr(cli, "build_provider", forbid_upstream)
    monkeypatch.setattr(cli, "build_tts_engine", forbid_upstream)
    result = runner.invoke(cli.app, ["check-content", str(video), "--offline"])
    assert result.exit_code == 0, out(result)
    assert "已检查 10 帧；候选 3 帧" in out(result)
    destination = video.with_name("视频.content-check")
    assert str(destination / "report.html") in out(result)
    assert seen == [(video, destination, {
        "cfg": DEFAULT_CONTENT_CHECK,
        "ffmpeg": DEFAULT_RENDER.ffmpeg_path,
        "ffprobe": DEFAULT_RENDER.ffprobe_path,
        "force": False, "offline": True,
    })]


def test_independent_overrides_are_forwarded(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    destination = tmp_path / "报告"
    seen = []

    def fake(video_arg, directory, **kwargs):
        seen.append((video_arg, directory, kwargs))
        return CheckOutcome("completed", 1, 2, directory / "report.html",
                            directory / "results.json")

    monkeypatch.setattr(cli, "check_video", fake, raising=False)
    result = runner.invoke(cli.app, [
        "check-content", str(video), "--output", str(destination), "--interval", "2.5",
        "--sensitive-threshold", "0", "--adult-threshold", "1", "--force",
    ])
    assert result.exit_code == 0, out(result)
    assert seen == [(video, destination, {
        "cfg": ContentCheckConfig(sample_interval=2.5, sensitive_threshold=0, adult_threshold=1),
        "ffmpeg": DEFAULT_RENDER.ffmpeg_path, "ffprobe": DEFAULT_RENDER.ffprobe_path,
        "force": True, "offline": False,
    })]


@pytest.mark.parametrize("option,value", [
    ("--interval", "nan"), ("--interval", "inf"), ("--interval", "0"),
    ("--interval", "-1"), ("--sensitive-threshold", "nan"),
    ("--sensitive-threshold", "inf"), ("--sensitive-threshold", "-0.1"),
    ("--sensitive-threshold", "1.1"), ("--adult-threshold", "nan"),
    ("--adult-threshold", "inf"), ("--adult-threshold", "-0.1"),
    ("--adult-threshold", "1.1"),
])
def test_invalid_config_exits_one_before_service(tmp_path, monkeypatch, option, value):
    monkeypatch.setattr(cli, "check_video", forbid_upstream, raising=False)
    result = runner.invoke(cli.app, ["check-content", str(tmp_path / "v.mp4"), option, value])
    assert result.exit_code == 1, out(result)
    assert "validation error" in out(result)
    assert "检查报告：" not in out(result)


def test_parameter_format_error_keeps_typer_exit_two(tmp_path):
    result = runner.invoke(cli.app, [
        "check-content", str(tmp_path / "v.mp4"), "--interval", "not-a-number",
    ])
    assert result.exit_code == 2, out(result)
    assert "not-a-number" in out(result)


def test_expected_service_error_exits_one_without_success(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ContentCheckError("离线模型缺失；先联网运行检查以下载模型")

    monkeypatch.setattr(cli, "check_video", fail, raising=False)
    result = runner.invoke(cli.app, ["check-content", str(tmp_path / "v.mp4"), "--offline"])
    assert result.exit_code == 1, out(result)
    assert "离线模型缺失" in out(result)
    assert "已检查" not in out(result)
    assert "检查报告：" not in out(result)


@pytest.mark.parametrize("error", [ValueError("service bug"), RuntimeError("service bug")])
def test_unknown_service_error_is_not_swallowed(tmp_path, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "check_video", fail, raising=False)
    result = runner.invoke(cli.app, ["check-content", str(tmp_path / "v.mp4")])
    assert result.exception is error
    assert "service bug" not in out(result)
    assert "检查报告：" not in out(result)


@pytest.mark.parametrize("status", ["completed", "cached", "report_rebuilt"])
def test_zero_candidates_are_checked_not_skipped(tmp_path, monkeypatch, status):
    report = tmp_path / "report.html"
    monkeypatch.setattr(cli, "check_video", lambda *args, **kwargs: CheckOutcome(
        status, 0, 10, report, tmp_path / "results.json"), raising=False)
    result = runner.invoke(cli.app, ["check-content", str(tmp_path / "v.mp4")])
    assert result.exit_code == 0, out(result)
    assert "已检查 10 帧；候选 0 帧" in out(result)
    assert f"检查报告：{report}" in out(result)
    assert "未启用" not in out(result)


def project(tmp_path, *, check=None, numbers=(2,)):
    root = tmp_path / "show"
    root.mkdir()
    data = {"show": "test", "slug": "show", "episodes": [
        {"number": number, "video": f"missing-source-{number}.mp4"} for number in numbers
    ]}
    if check is not None:
        data["content_check"] = check
    (root / "project.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return Paths(root)


@pytest.mark.parametrize("selection", [["--only", "check"], ["--from", "check"]])
@pytest.mark.parametrize("status", ["completed", "cached", "report_rebuilt"])
def test_project_check_only_prints_collected_success(tmp_path, monkeypatch, selection, status):
    paths = project(tmp_path)
    video = paths.video(2)
    video.parent.mkdir()
    video.write_bytes(b"final")
    calls = []

    def fake(video_arg, directory, **kwargs):
        calls.append((video_arg, directory, kwargs))
        return CheckOutcome(status, 3, 10, directory / "report.html", directory / "results.json")

    monkeypatch.setattr(pipeline, "check_video", fake)
    monkeypatch.setattr(cli, "build_provider", forbid_upstream)
    monkeypatch.setattr(cli, "build_tts_engine", forbid_upstream)
    result = runner.invoke(cli.app, [
        "run", "show", "--work-dir", str(tmp_path), *selection, "--force",
    ])
    assert result.exit_code == 0, out(result)
    assert "已检查 10 帧；候选 3 帧" in out(result)
    assert f"检查报告：{paths.check_dir(2) / 'report.html'}" in out(result)
    assert calls[0][0] == video
    assert calls[0][2]["force"] is True
    assert "检查未启用" not in out(result)


def test_failed_project_check_does_not_print_old_report_and_continues(tmp_path, monkeypatch):
    paths = project(tmp_path, numbers=(2, 3))
    paths.video(2).parent.mkdir()
    for number in (2, 3):
        paths.video(number).write_bytes(b"final")
        paths.check_dir(number).mkdir(parents=True)
        (paths.check_dir(number) / "report.html").write_text("old report", encoding="utf-8")

    def fake(video_arg, directory, **kwargs):
        if video_arg == paths.video(2):
            raise ContentCheckError("依赖缺失")
        return CheckOutcome("cached", 0, 10, directory / "report.html", directory / "results.json")

    monkeypatch.setattr(pipeline, "check_video", fake)
    result = runner.invoke(cli.app, ["run", "show", "--work-dir", str(tmp_path), "--only", "check"])
    assert result.exit_code == 0, out(result)
    assert "E02：本次检查失败／报告未更新：依赖缺失" in out(result)
    assert str(paths.check_dir(2) / "report.html") not in out(result)
    assert f"检查报告：{paths.check_dir(3) / 'report.html'}" in out(result)
    assert "已检查 10 帧；候选 0 帧" in out(result)


@pytest.mark.parametrize("selection", [[], ["--from", "render"], ["--only", "render"]])
def test_disabled_notice_only_when_check_is_selected(tmp_path, monkeypatch, selection):
    project(tmp_path)

    async def fake(cfg, provider, **kwargs):
        assert kwargs["check_results"] == {}
        return []

    monkeypatch.setattr(cli, "run_pipeline", fake)
    monkeypatch.setattr(cli, "build_provider", lambda *args: None)
    monkeypatch.setattr(cli, "build_tts_engine", lambda *args: None)
    result = runner.invoke(cli.app, ["run", "show", "--work-dir", str(tmp_path), *selection])
    assert result.exit_code == 0, out(result)
    assert ("检查未启用" in out(result)) == (selection != ["--only", "render"])
    assert "检查报告：" not in out(result)


def test_two_entries_share_video_config_and_candidate_count(tmp_path, monkeypatch):
    check = {"sample_interval": 2.5, "sensitive_threshold": 0.3, "adult_threshold": 0.7}
    paths = project(tmp_path, check=check)
    video = paths.video(2)
    video.parent.mkdir()
    video.write_bytes(b"final")
    seen = []

    def fake(video_arg, directory, **kwargs):
        seen.append((video_arg, kwargs))
        return CheckOutcome("completed", 3, 10, directory / "report.html",
                            directory / "results.json")

    monkeypatch.setattr(cli, "check_video", fake, raising=False)
    monkeypatch.setattr(pipeline, "check_video", fake)
    independent = runner.invoke(cli.app, [
        "check-content", str(video), "--interval", "2.5", "--sensitive-threshold", "0.3",
        "--adult-threshold", "0.7",
    ])
    registered = runner.invoke(cli.app, [
        "run", "show", "--work-dir", str(tmp_path), "--episode", "2", "--only", "check",
    ])
    for result in (independent, registered):
        assert result.exit_code == 0, out(result)
        assert "已检查 10 帧；候选 3 帧" in out(result)
    assert len(seen) == 2
    assert seen[0][0] == seen[1][0] == video
    assert seen[0][1]["cfg"] == seen[1][1]["cfg"] == ContentCheckConfig(**check)
    for key in ("ffmpeg", "ffprobe", "force"):
        assert seen[0][1][key] == seen[1][1][key]


@pytest.mark.parametrize("selection", [["--only", "check"], ["--from", "check"],
                                        ["--from", "render"]])
@pytest.mark.parametrize("error", [ValueError("service bug"), RuntimeError("service bug"),
                                   asyncio.CancelledError()])
def test_project_unknown_check_error_propagates_unchanged(
    tmp_path, monkeypatch, selection, error,
):
    paths = project(tmp_path, check={"enabled": True}, numbers=(2, 3))
    paths.video(2).parent.mkdir()
    paths.video(2).write_bytes(b"final")
    calls = []

    def fail(video, directory, **kwargs):
        calls.append(video)
        raise error

    monkeypatch.setattr(pipeline, "check_video", fail)
    monkeypatch.setattr(pipeline, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_is_fresh", lambda *args: True)
    arguments = ["run", "show", "--work-dir", str(tmp_path), *selection]
    if isinstance(error, asyncio.CancelledError):
        # CliRunner 捕获 Exception，不捕获 CancelledError（BaseException）。
        with pytest.raises(asyncio.CancelledError) as caught:
            runner.invoke(cli.app, arguments)
        assert caught.value is error
    else:
        result = runner.invoke(cli.app, arguments)
        assert result.exception is error
        assert "service bug" not in out(result)
        assert "本次检查失败" not in out(result)
        assert "检查报告：" not in out(result)
        assert "已检查" not in out(result)
    assert calls == [paths.video(2)]


@pytest.mark.parametrize("error", [ValueError("上游数据无效"), FileNotFoundError("缺少上游产物")])
def test_other_stage_expected_errors_still_handled_with_check_selected(
    tmp_path, monkeypatch, error,
):
    project(tmp_path, check={"enabled": True})

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(pipeline, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "run_render", fail)
    result = runner.invoke(cli.app, [
        "run", "show", "--work-dir", str(tmp_path), "--from", "render",
    ])
    assert result.exit_code == 1, out(result)
    assert isinstance(result.exception, SystemExit)
    assert str(error) in out(result)
    assert "检查报告：" not in out(result)
