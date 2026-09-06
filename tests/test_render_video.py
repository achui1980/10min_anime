from pathlib import Path

import pytest

from tenmin.models import Timeline, TimelineSegment
from tenmin.render.video import (
    build_render_args,
    escape_filter_path,
    quality_args,
    render_video,
)


def make_timeline() -> Timeline:
    return Timeline(
        episode=2,
        segments=[
            TimelineSegment(
                beat_id="b1",
                source_start=100.0,
                source_end=120.0,
                timeline_start=0.0,
                timeline_end=20.0,
            ),
            TimelineSegment(
                beat_id="b2",
                source_start=200.0,
                source_end=210.0,
                timeline_start=20.0,
                timeline_end=30.0,
            ),
        ],
        subtitles=[],
        narration_offsets=[],
        total_seconds=30.0,
    )


def build(tmp_path, **overrides):
    kwargs = {
        "video": tmp_path / "source.mkv",
        "timeline": make_timeline(),
        "audio": tmp_path / "06_audio" / "E02.mixed.m4a",
        "ass": tmp_path / "05_timeline" / "E02.ass",
        "out_path": tmp_path / "07_render" / "E02.mp4",
        "encoder": "libx264",
    }
    kwargs.update(overrides)
    return build_render_args(**kwargs)


def test_escape_filter_path_wraps_in_single_quotes():
    assert escape_filter_path(Path("/tmp/work/E02.ass")) == "'/tmp/work/E02.ass'"


def test_escape_filter_path_escapes_single_quote():
    assert escape_filter_path(Path("/tmp/it's/E02.ass")) == "'/tmp/it\\'s/E02.ass'"


def test_escape_filter_path_keeps_spaces_and_brackets():
    """源片名里带方括号和空格是常态，单引号包住就够了。"""
    assert escape_filter_path(Path("/a b/[LoliHouse] x.ass")) == "'/a b/[LoliHouse] x.ass'"


def test_quality_args_software_encoder_uses_crf():
    assert quality_args("libx264") == ["-crf", "20", "-preset", "medium"]


def test_quality_args_videotoolbox_uses_bitrate():
    """videotoolbox 不认 -crf/-preset，只能给码率。"""
    assert quality_args("h264_videotoolbox") == ["-b:v", "6000k"]


def test_build_render_args_filter_graph_matches_expected(tmp_path):
    args = build(tmp_path)
    graph = args[args.index("-filter_complex") + 1]
    ass = tmp_path / "05_timeline" / "E02.ass"
    assert graph == (
        "[0:v]trim=start=100.000:end=120.000,setpts=PTS-STARTPTS,"
        "scale=1920:1080,setsar=1[v0];"
        "[0:v]trim=start=200.000:end=210.000,setpts=PTS-STARTPTS,"
        "scale=1920:1080,setsar=1[v1];"
        "[v0][v1]concat=n=2:v=1:a=0[vcat];"
        f"[vcat]subtitles=filename='{ass}'[vout]"
    )


def test_build_render_args_inputs_video_then_audio(tmp_path):
    args = build(tmp_path)
    assert args[:5] == [
        "-y",
        "-i",
        str(tmp_path / "source.mkv"),
        "-i",
        str(tmp_path / "06_audio" / "E02.mixed.m4a"),
    ]


def test_build_render_args_maps_burned_video_and_mixed_audio(tmp_path):
    args = build(tmp_path)
    assert args[args.index("-map") : args.index("-map") + 4] == [
        "-map",
        "[vout]",
        "-map",
        "1:a",
    ]


def test_build_render_args_copies_audio_and_adds_faststart(tmp_path):
    args = build(tmp_path)
    out = tmp_path / "07_render" / "E02.mp4"
    assert args[-13:] == [
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(out),
    ]


def test_build_render_args_honours_custom_resolution(tmp_path):
    args = build(tmp_path, width=1280, height=720)
    graph = args[args.index("-filter_complex") + 1]
    assert "scale=1280:720" in graph
    assert "scale=1920:1080" not in graph


def test_build_render_args_rejects_empty_timeline(tmp_path):
    timeline = make_timeline()
    timeline.segments = []
    with pytest.raises(ValueError) as exc:
        build(tmp_path, timeline=timeline)
    assert "segment" in str(exc.value)


def test_render_video_runs_ffmpeg_and_returns_path(tmp_path, monkeypatch):
    from tenmin.render import video as video_module

    seen: list[list[str]] = []

    def fake_run(args):
        seen.append(list(args))
        return ""

    monkeypatch.setattr(video_module, "run", fake_run)
    out_path = tmp_path / "07_render" / "E02.mp4"
    result = render_video(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        audio=tmp_path / "06_audio" / "E02.mixed.m4a",
        ass=tmp_path / "05_timeline" / "E02.ass",
        out_path=out_path,
        encoder="libx264",
    )
    assert result == out_path
    assert out_path.parent.is_dir()
    assert seen[0][-1] == str(out_path)
