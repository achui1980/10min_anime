from pathlib import Path

import pytest

from tenmin.models import Timeline, TimelineSegment
from tenmin.render.video import (
    build_render_args,
    escape_drawtext,
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


def test_escape_filter_path_escapes_single_quote_for_both_layers():
    """字面单引号要「闭引号 + \\' + 重开引号」，而且 \\' 那一层是给 AVOption 分词器的。

    filtergraph 的 '...' 里放不进字面单引号（av_get_token 见到 ' 就收尾），所以必须
    先闭合、给一个转义的引号、再重开。但只做到这一步还不够：层 2（filter 自己的
    AVOption 分词器）拿到裸的 ' 会再吃掉一次，所以送进层 1 引号里的那份也得是 \\'。
    两层合起来，一个字面 ' 展开成 \\'\\''。实测（ffmpeg 9.0.1）：只做 '\\'' 会渲染出
    /tmp/its/E02.ass 而报 Unable to open。
    """
    assert escape_filter_path(Path("/tmp/it's/E02.ass")) == "'/tmp/it\\'\\''s/E02.ass'"


def test_escape_filter_path_escapes_colon_for_the_avoption_layer():
    """冒号是 AVOption 的分隔符，而层 1 的 '...' 原样透传，挡不住它。

    实测（ffmpeg 9.0.1）不转义会得到 `Error parsing a filter description`。
    """
    assert escape_filter_path(Path("/tmp/a:b/E02.ass")) == "'/tmp/a\\:b/E02.ass'"


def test_escape_filter_path_doubles_backslash_for_the_avoption_layer():
    """反斜杠翻倍是给层 2 的：层 1 的 '...' 里反斜杠原样透传，不会被提前吃掉。"""
    assert escape_filter_path(Path("/tmp/a\\b/E02.ass")) == "'/tmp/a\\\\b/E02.ass'"


def test_escape_filter_path_keeps_spaces_and_brackets():
    """源片名里带方括号和空格是常态，单引号包住就够了（实测这两类本来就 rc=0）。"""
    assert escape_filter_path(Path("/a b/[LoliHouse] x.ass")) == "'/a b/[LoliHouse] x.ass'"


def test_escape_filter_path_combines_quote_and_colon_and_backslash():
    got = escape_filter_path(Path("/o'b:c\\d/E02.ass"))
    assert got == "'/o\\'\\''b\\:c\\\\d/E02.ass'"


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


def test_escape_drawtext_escapes_backslash_and_quote():
    raw = "it's\\path"
    assert escape_drawtext(raw) == "it\\'s\\\\path"


def test_build_render_args_without_fade_or_outro_keeps_vout_label(tmp_path):
    """没要求淡出/片尾卡片时，-map 仍然是 [vout]，行为与老版本完全一致。"""
    args = build(tmp_path)
    assert args[args.index("-map") + 1] == "[vout]"


def test_build_render_args_applies_fade_out_before_video_ends(tmp_path):
    args = build(tmp_path, fade_out_seconds=5.0)
    graph = args[args.index("-filter_complex") + 1]
    ass = tmp_path / "05_timeline" / "E02.ass"
    assert graph.endswith(
        f"[vcat]subtitles=filename='{ass}'[vout];"
        "[vout]fade=t=out:st=25.000:d=5.000[vfaded]"
    )
    assert args[args.index("-map") + 1] == "[vfaded]"


def test_build_render_args_appends_outro_card(tmp_path):
    args = build(
        tmp_path,
        fade_out_seconds=5.0,
        outro_seconds=3.0,
        outro_title="才女的侍从 · EP02",
        outro_message="解说结束，谢谢观看",
    )
    graph = args[args.index("-filter_complex") + 1]
    assert "color=c=black:s=1920x1080:d=3.000[cardbg]" in graph
    assert "drawtext=font='Lantinghei SC':text='才女的侍从 · EP02'" in graph
    assert "drawtext=font='Lantinghei SC':text='解说结束，谢谢观看'" in graph
    assert graph.endswith("[vfaded][card]concat=n=2:v=1:a=0[vfinal]")
    assert args[args.index("-map") + 1] == "[vfinal]"


def test_render_video_runs_ffmpeg_and_returns_path(tmp_path, monkeypatch):
    from tenmin.atomic import part_path
    from tenmin.render import video as video_module

    seen: list[list[str]] = []

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        seen.append(list(args))
        # 真 ffmpeg 一定会把输出文件写出来；假的也得写，否则原子改名无从下手。
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(video_module, "run_with_progress", fake_run_with_progress)
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
    # ffmpeg 写的是同目录的 .part，跑完才原子改名到 out_path
    assert seen[0][-1] == str(part_path(out_path))
    assert out_path.is_file()


def test_render_video_reports_substep_progress(tmp_path, monkeypatch):
    from tenmin.render import video as video_module

    from .fakes import FakeReporter

    captured: dict[str, float] = {}

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        captured["total_seconds"] = total_seconds
        if on_progress is not None:
            on_progress(0.5)
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(video_module, "run_with_progress", fake_run_with_progress)
    reporter = FakeReporter()
    render_video(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        audio=tmp_path / "06_audio" / "E02.mixed.m4a",
        ass=tmp_path / "05_timeline" / "E02.ass",
        out_path=tmp_path / "07_render" / "E02.mp4",
        encoder="libx264",
        reporter=reporter,
    )
    assert captured["total_seconds"] == pytest.approx(30.0)
    assert reporter.calls == [("substep", "render", 50, 100, "")]


def test_render_video_passes_configured_ffmpeg_binary(tmp_path, monkeypatch):
    """RenderConfig.ffmpeg_path 必须真的传到 subprocess，不然那个旋钮是哑的。"""
    from tenmin.render import video as video_module

    seen: dict[str, str] = {}

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, ffmpeg="ffmpeg"):
        seen["ffmpeg"] = ffmpeg
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(video_module, "run_with_progress", fake_run_with_progress)
    render_video(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        audio=tmp_path / "06_audio" / "E02.mixed.m4a",
        ass=tmp_path / "05_timeline" / "E02.ass",
        out_path=tmp_path / "07_render" / "E02.mp4",
        encoder="libx264",
        ffmpeg="/opt/libass/bin/ffmpeg",
    )
    assert seen["ffmpeg"] == "/opt/libass/bin/ffmpeg"


# --- 产物原子写（P1-G 第 1 项）---------------------------------------------


def test_render_video_tells_ffmpeg_to_write_a_part_file(tmp_path, monkeypatch):
    from tenmin.atomic import part_path
    from tenmin.render import video as video_module

    seen: list[list[str]] = []

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        seen.append(list(args))
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(video_module, "run_with_progress", fake_run_with_progress)
    out_path = tmp_path / "07_render" / "E02.mp4"
    render_video(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        audio=tmp_path / "06_audio" / "E02.mixed.m4a",
        ass=tmp_path / "05_timeline" / "E02.ass",
        out_path=out_path,
        encoder="libx264",
    )
    assert seen[0][-1] == str(part_path(out_path))
    assert seen[0][-1].endswith(".mp4")
    assert out_path.is_file()
    assert not part_path(out_path).exists()


def test_render_video_keeps_the_previous_artifact_when_ffmpeg_fails(tmp_path, monkeypatch):
    from tenmin.atomic import part_path
    from tenmin.render import video as video_module
    from tenmin.render.ffmpeg import FFmpegError

    out_path = tmp_path / "07_render" / "E02.mp4"
    out_path.parent.mkdir(parents=True)
    out_path.write_bytes(b"good")
    before = out_path.stat().st_mtime_ns

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        Path(args[-1]).write_bytes(b"truncated")
        raise FFmpegError("boom")

    monkeypatch.setattr(video_module, "run_with_progress", fake_run_with_progress)
    with pytest.raises(FFmpegError):
        render_video(
            video=tmp_path / "source.mkv",
            timeline=make_timeline(),
            audio=tmp_path / "06_audio" / "E02.mixed.m4a",
            ass=tmp_path / "05_timeline" / "E02.ass",
            out_path=out_path,
            encoder="libx264",
        )
    assert out_path.read_bytes() == b"good"
    assert out_path.stat().st_mtime_ns == before
    assert not part_path(out_path).exists()


# --- filtergraph 转义的真 ffmpeg 回归锁（P1-I）-------------------------------
#
# 上面那些断言只钉住「我们拼出了什么字符串」，钉不住「ffmpeg 认不认」。这一段真跑
# ffmpeg，输入自己用 lavfi 造（不依赖 work/ 的素材），一次约 1 秒。

MINIMAL_ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, Alignment, MarginV, Encoding
Style: Default,Arial,48,&H0000FFFF,2,40,1

[Events]
Format: Layer, Start, End, Style, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,\u6d4b\u8bd5\u5b57\u5e55
"""


def _lavfi_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """造一段 2 秒的测试视频与一条测试音轨，返回 (video, audio)。"""
    import subprocess

    video = tmp_path / "src.mp4"
    audio = tmp_path / "src.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
         "testsrc=size=320x180:rate=10:duration=2", "-c:v", "libx264",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(video)],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=1", "-c:a", "aac", str(audio)],
        capture_output=True, check=True,
    )
    return video, audio


def _one_second_timeline() -> Timeline:
    return Timeline(
        episode=2,
        segments=[
            TimelineSegment(
                beat_id="b1",
                source_start=0.0,
                source_end=1.0,
                timeline_start=0.0,
                timeline_end=1.0,
            )
        ],
        subtitles=[],
        narration_offsets=[],
        total_seconds=1.0,
    )


@pytest.mark.render
@pytest.mark.parametrize(
    "dirname",
    [
        "plain",
        "it's ok",  # 字面单引号：/Users/.../O'Brien/... 这种目录很现实
        "a:b",  # 冒号是 AVOption 分隔符
        "a\\b",
        "a b",
        "[LoliHouse]",
        "a,b",
        "a;b",
        "\u624d\u5973\u7684\u4f8d\u4ece",
        "O'B:x\\y",  # 三种一起上
    ],
)
def test_real_ffmpeg_accepts_ass_paths_with_special_characters(tmp_path, dirname):
    """字幕路径带特殊字符时，真 ffmpeg 必须能打开它。

    subtitles 滤镜打不开文件会硬失败（实测 rc=254 Unable to open），所以「跑通」
    就等价于「打开的是我们想要的那个文件」—— 不需要另外验证路径解析对不对。
    """
    from tenmin.render.ffmpeg import has_filter
    from tenmin.render.video import render_video

    if not has_filter("subtitles"):
        pytest.skip("ffmpeg 没编 libass，跑不了 subtitles 滤镜")

    video, audio = _lavfi_inputs(tmp_path)
    ass_dir = tmp_path / dirname
    ass_dir.mkdir()
    ass = ass_dir / "E02.ass"
    ass.write_text(MINIMAL_ASS, encoding="utf-8")

    out = render_video(
        video=video,
        timeline=_one_second_timeline(),
        audio=audio,
        ass=ass,
        out_path=tmp_path / "out" / "E02.mp4",
        encoder="libx264",
    )
    assert out.is_file()
    assert out.stat().st_size > 0
