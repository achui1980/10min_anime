"""`tenmin mask-apply`：给任意视频按标注打上整屏高斯模糊。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tenmin.cli import app
from tenmin.mask import MaskRange, MaskSpec
from tenmin.mask_apply import (
    MaskApplyError,
    apply_mask,
    build_apply_args,
    default_output_path,
    load_spec,
)

runner = CliRunner()


def out(result) -> str:
    """click 8.2 起 result.output 只含 stdout，stderr 单独拿。这里合并，避免版本差异。"""
    text = result.output or ""
    try:
        text += result.stderr or ""
    except (ValueError, AttributeError):
        pass
    return text


def _mask_file(tmp_path: Path, payload: dict, name: str = "mask.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _spec_payload(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "sigma": 20.0,
        "ranges": [{"start": 10.0, "end": 12.5}],
    }
    payload.update(overrides)
    return payload


def _fake_video(tmp_path: Path) -> Path:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    return video


def _capture_args(monkeypatch, tmp_path):
    """把 run_with_progress 换成假实现，抓下真正传给 ffmpeg 的 argv。"""
    captured: dict = {}

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, ffmpeg=None):
        captured["args"] = list(args)
        captured["total_seconds"] = total_seconds
        Path(args[-1]).write_bytes(b"fake output")
        return ""

    monkeypatch.setattr("tenmin.mask_apply.run_with_progress", fake_run_with_progress)
    return captured


# --- 命令行拼装 ---------------------------------------------------------


def test_default_output_path_appends_masked_suffix():
    assert default_output_path(Path("/tmp/a/b.mp4")) == Path("/tmp/a/b.masked.mp4")
    assert default_output_path(Path("/tmp/a/b.mkv")) == Path("/tmp/a/b.masked.mkv")


def test_build_apply_args_shape(tmp_path):
    video = tmp_path / "in.mp4"
    out_path = tmp_path / "out.mp4"
    args = build_apply_args(
        video,
        out_path=out_path,
        ranges=[MaskRange(start=10.0, end=12.5)],
        sigma=20.0,
    )
    assert args[-1] == str(out_path)
    assert args[args.index("-vf") + 1] == (
        "gblur=sigma=20:steps=3:enable='between(t,10.000,12.500)'"
    )
    maps = [args[i + 1] for i, a in enumerate(args) if a == "-map"]
    assert maps == ["0:v:0", "0:a:0?"]
    assert args[args.index("-c:a") + 1] == "copy"
    assert "-pix_fmt" not in args


def test_build_apply_args_keeps_the_timeline_zero_based(tmp_path):
    """区间用的是这个视频自己的时间轴：普通 -i，不搬容器起点、不重置时间戳。"""
    args = build_apply_args(
        tmp_path / "in.mp4",
        out_path=tmp_path / "out.mp4",
        ranges=[MaskRange(start=10.0, end=12.5)],
    )
    joined = " ".join(args)
    assert "-copyts" not in args
    assert "setpts" not in joined
    assert args[args.index("-i") + 1] == str(tmp_path / "in.mp4")


def test_build_apply_args_adds_faststart_only_for_mp4_family(tmp_path):
    for suffix, expected in ((".mp4", True), (".mov", True), (".m4v", True), (".mkv", False)):
        args = build_apply_args(
            tmp_path / "in",
            out_path=tmp_path / f"out{suffix}",
            ranges=[MaskRange(start=1.0, end=2.0)],
        )
        assert ("-movflags" in args) is expected, suffix


def test_build_apply_args_rejects_empty_ranges(tmp_path):
    with pytest.raises(MaskApplyError):
        build_apply_args(tmp_path / "in.mp4", out_path=tmp_path / "o.mp4", ranges=[])


def test_load_spec_reports_missing_file(tmp_path):
    with pytest.raises(MaskApplyError) as excinfo:
        load_spec(tmp_path / "nope.json")
    assert "找不到遮罩标注文件" in str(excinfo.value)


# --- apply_mask 的行为 ---------------------------------------------------


def test_apply_mask_writes_to_a_part_file_then_renames(monkeypatch, tmp_path):
    """ffmpeg 必须写进 .part 临时文件，成功后才改名到正式路径。"""
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload())
    captured = _capture_args(monkeypatch, tmp_path)

    target = apply_mask(video, mask_file, ffmpeg="ffmpeg", ffprobe="ffprobe")

    assert target == tmp_path / "clip.masked.mp4"
    assert target.read_bytes() == b"fake output"
    written = Path(captured["args"][-1])
    assert written != target
    assert written.name == "clip.masked.part.mp4"
    assert not written.exists()


def test_apply_mask_removes_the_part_file_when_ffmpeg_fails(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload())

    def boom(args, *, total_seconds, on_progress=None, ffmpeg=None):
        Path(args[-1]).write_bytes(b"half written")
        raise RuntimeError("ffmpeg 挂了")

    monkeypatch.setattr("tenmin.mask_apply.run_with_progress", boom)

    with pytest.raises(RuntimeError):
        apply_mask(video, mask_file, ffmpeg="ffmpeg", ffprobe="ffprobe")

    assert not (tmp_path / "clip.masked.mp4").exists()
    assert not (tmp_path / "clip.masked.part.mp4").exists()


def test_apply_mask_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload())
    (tmp_path / "clip.masked.mp4").write_bytes(b"existing")
    _capture_args(monkeypatch, tmp_path)

    with pytest.raises(MaskApplyError) as excinfo:
        apply_mask(video, mask_file, ffmpeg="ffmpeg", ffprobe="ffprobe")

    assert "已存在" in str(excinfo.value)
    assert "--force" in str(excinfo.value)


def test_apply_mask_allows_overwrite_with_force(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload())
    (tmp_path / "clip.masked.mp4").write_bytes(b"existing")
    _capture_args(monkeypatch, tmp_path)

    target = apply_mask(video, mask_file, force=True, ffmpeg="ffmpeg", ffprobe="ffprobe")

    assert target.read_bytes() == b"fake output"


def test_apply_mask_refuses_output_equal_to_source(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload())
    _capture_args(monkeypatch, tmp_path)

    with pytest.raises(MaskApplyError) as excinfo:
        apply_mask(video, mask_file, out_path=video, ffmpeg="ffmpeg", ffprobe="ffprobe")

    assert "同一个文件" in str(excinfo.value)


def test_apply_mask_rejects_a_mask_file_without_ranges(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload(ranges=[]))
    _capture_args(monkeypatch, tmp_path)

    with pytest.raises(MaskApplyError) as excinfo:
        apply_mask(video, mask_file, ffmpeg="ffmpeg", ffprobe="ffprobe")

    assert "没有任何遮罩区间" in str(excinfo.value)


def test_apply_mask_reports_a_missing_source(monkeypatch, tmp_path):
    mask_file = _mask_file(tmp_path, _spec_payload())
    _capture_args(monkeypatch, tmp_path)

    with pytest.raises(MaskApplyError) as excinfo:
        apply_mask(tmp_path / "nope.mp4", mask_file, ffmpeg="ffmpeg", ffprobe="ffprobe")

    assert "找不到要处理的视频" in str(excinfo.value)


def test_apply_mask_merges_overlapping_ranges(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(
        tmp_path,
        _spec_payload(
            ranges=[
                {"start": 10.0, "end": 12.0},
                {"start": 11.0, "end": 14.0},
            ]
        ),
    )
    captured = _capture_args(monkeypatch, tmp_path)

    apply_mask(video, mask_file, ffmpeg="ffmpeg", ffprobe="ffprobe")

    vf = captured["args"][captured["args"].index("-vf") + 1]
    assert vf == "gblur=sigma=20:steps=3:enable='between(t,10.000,14.000)'"


def test_apply_mask_takes_sigma_from_the_file(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload(sigma=8.0))
    captured = _capture_args(monkeypatch, tmp_path)

    apply_mask(video, mask_file, ffmpeg="ffmpeg", ffprobe="ffprobe")

    vf = captured["args"][captured["args"].index("-vf") + 1]
    assert vf.startswith("gblur=sigma=8:")


def test_apply_mask_sigma_argument_overrides_the_file(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload(sigma=8.0))
    captured = _capture_args(monkeypatch, tmp_path)

    apply_mask(video, mask_file, sigma=30.0, ffmpeg="ffmpeg", ffprobe="ffprobe")

    vf = captured["args"][captured["args"].index("-vf") + 1]
    assert vf.startswith("gblur=sigma=30:")


def test_apply_mask_rejects_a_non_positive_sigma(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload())
    _capture_args(monkeypatch, tmp_path)

    with pytest.raises(MaskApplyError) as excinfo:
        apply_mask(video, mask_file, sigma=0, ffmpeg="ffmpeg", ffprobe="ffprobe")

    assert "sigma" in str(excinfo.value)


# --- CLI ----------------------------------------------------------------


def test_cli_mask_apply_runs_and_reports_the_output(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload())
    calls: list = []

    def fake_apply_mask(source, mask, **kwargs):
        calls.append((source, mask, kwargs))
        target = tmp_path / "clip.masked.mp4"
        target.write_bytes(b"fake output")
        return target

    monkeypatch.setattr("tenmin.cli.apply_mask", fake_apply_mask)

    result = runner.invoke(
        app,
        ["mask-apply", str(video), "--mask", str(mask_file)],
    )

    assert result.exit_code == 0, out(result)
    assert "已输出" in out(result)
    assert "clip.masked.mp4" in out(result)
    assert calls and calls[0][0] == video


def test_cli_mask_apply_warns_when_the_mask_carries_an_episode(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload(episode=2))
    monkeypatch.setattr(
        "tenmin.cli.apply_mask",
        lambda *a, **k: tmp_path / "clip.masked.mp4",
    )

    result = runner.invoke(
        app,
        ["mask-apply", str(video), "--mask", str(mask_file)],
    )

    assert result.exit_code == 0, out(result)
    assert "episode=2" in out(result)


def test_cli_mask_apply_fails_when_the_mask_file_is_missing(tmp_path):
    video = _fake_video(tmp_path)

    result = runner.invoke(
        app,
        ["mask-apply", str(video), "--mask", str(tmp_path / "nope.json")],
    )

    assert result.exit_code == 1
    assert "找不到遮罩标注文件" in out(result)


def test_cli_mask_apply_requires_the_mask_option(tmp_path):
    video = _fake_video(tmp_path)

    result = runner.invoke(app, ["mask-apply", str(video)])

    assert result.exit_code != 0


def test_cli_mask_apply_passes_the_output_and_sigma_through(monkeypatch, tmp_path):
    video = _fake_video(tmp_path)
    mask_file = _mask_file(tmp_path, _spec_payload())
    calls: list = []
    target = tmp_path / "custom.mp4"

    def fake_apply_mask(source, mask, **kwargs):
        calls.append(kwargs)
        target.write_bytes(b"fake output")
        return target

    monkeypatch.setattr("tenmin.cli.apply_mask", fake_apply_mask)

    result = runner.invoke(
        app,
        [
            "mask-apply",
            str(video),
            "--mask",
            str(mask_file),
            "--out",
            str(target),
            "--sigma",
            "12.5",
        ],
    )

    assert result.exit_code == 0, out(result)
    assert calls[0]["out_path"] == target
    assert calls[0]["sigma"] == 12.5


def test_mask_spec_round_trips_through_the_annotation_file(tmp_path):
    """导出的 mask.json 能被后端原样读回来（报告页导出 → 独立命令这条链的前提）。"""
    payload = _spec_payload(episode=None)
    payload["ranges"] = [{"start": 83.4, "end": 86.0}]
    path = _mask_file(tmp_path, payload)

    spec = load_spec(path)

    assert isinstance(spec, MaskSpec)
    assert spec.ranges == [MaskRange(start=83.4, end=86.0)]
    assert spec.sigma == 20.0
