from pathlib import Path

import pytest

from tenmin.config import RenderConfig
from tenmin.models import Timeline, TimelineSegment
from tenmin.render.video import (
    build_render_args,
    escape_filter_arg,
    escape_filter_path,
    frame_rate_arg,
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
        "scale=1920:1080,setsar=1,format=yuv420p[v0];"
        "[1:v]trim=start=200.000:end=210.000,setpts=PTS-STARTPTS,"
        "scale=1920:1080,setsar=1,format=yuv420p[v1];"
        "[v0][v1]concat=n=2:v=1:a=0[vcat];"
        f"[vcat]subtitles=filename='{ass}'[vout]"
    )


def test_build_render_args_opens_one_input_per_segment_with_input_level_seek(tmp_path):
    """每段一个 `-ss/-t` 输入，只解码用到的那几分钟，而不是把整部片解一遍。

    `-copyts` 是关键：它让 filter 看到的仍然是**原片时间戳**，于是 trim 的
    start/end 一个字都不用改，选出来的帧集合与「满长度输入 + trim」完全相同。
    """
    args = build(tmp_path)
    source = str(tmp_path / "source.mkv")
    # 段 1：100.0-120.0；段 2：200.0-210.0。两端各留 0.5 秒余量。
    assert args[:16] == [
        "-y",
        "-copyts",
        "-ss",
        "99.500",
        "-t",
        "21.000",
        "-i",
        source,
        "-ss",
        "199.500",
        "-t",
        "11.000",
        "-i",
        source,
        "-i",
        str(tmp_path / "06_audio" / "E02.mixed.m4a"),
    ]


def test_build_render_args_seek_lead_is_clamped_at_zero(tmp_path):
    """段起点在片头 0.2 秒时不能算出负的 -ss（ffmpeg 会把负值当成「距片尾」）。

    钳到 0 之后 `-ss 0` 与不写 `-ss` 完全等价，所以干脆不写。
    """
    timeline = make_timeline()
    timeline.segments[0].source_start = 0.2
    timeline.segments[0].source_end = 1.2
    args = build(tmp_path, timeline=timeline)
    assert args[:6] == [
        "-y",
        "-copyts",
        "-t",
        "1.700",
        "-i",
        str(tmp_path / "source.mkv"),
    ]


def test_build_render_args_maps_the_audio_input_after_every_segment(tmp_path):
    """音轨的输入序号跟着段数走。写死 `1:a` 的话多段时会去映射段 1 的画面。"""
    args = build(tmp_path)
    assert args[args.index("-map") : args.index("-map") + 4] == [
        "-map",
        "[vout]",
        "-map",
        "2:a",
    ]


def test_build_render_args_inputs_video_then_audio(tmp_path):
    args = build(tmp_path)
    assert args.count("-i") == 3
    assert args[-1] == str(tmp_path / "07_render" / "E02.mp4")


def test_build_render_args_maps_burned_video_and_mixed_audio(tmp_path):
    args = build(tmp_path)
    assert args[args.index("-map") : args.index("-map") + 4] == [
        "-map",
        "[vout]",
        "-map",
        "2:a",
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


def test_escape_filter_arg_handles_drawtext_text_too(tmp_path):
    """drawtext 的 text 跟文件路径过的是同两层，转义规则完全一样，共用一个函数。

    drawtext 独有的 `%` strftime 展开不靠转义解决 —— 由 expansion=none 关掉（实测
    `%Y-%m-%d` 渲染出来跟 textfile= 的基准真值像素逐字节相同）。
    """
    assert escape_filter_arg("it's\\path") == "'it\\'\\''s\\\\path'"
    assert escape_filter_arg("a:b") == "'a\\:b'"
    assert escape_filter_arg("100%") == "'100%'"


def test_build_render_args_keeps_expansion_none_so_percent_is_literal(tmp_path):
    """没有 expansion=none 的话 drawtext 会把 % 当 strftime 展开，`100%` 就变了。"""
    args = build(tmp_path, outro_seconds=3.0, outro_title="100%", outro_message="50%")
    graph = args[args.index("-filter_complex") + 1]
    assert graph.count("expansion=none") == 2


def test_build_render_args_escapes_quotes_and_colons_in_outro_text(tmp_path):
    """片尾卡的文案来自 config（outro_title 由 cfg.show 拼出），带撇号的剧名很常见。

    走的是跟字幕路径同一套两层转义 —— 实测（ffmpeg 9.0.1）不转义时 ' 会让 drawtext
    的 Eval 报 `Invalid chars '[o]' at the end of expression`，: 会让层 1 就地报
    `Error parsing a filter description`。
    """
    args = build(
        tmp_path,
        outro_seconds=3.0,
        outro_title="It's 9:00",
        outro_message="a\\b",
    )
    graph = args[args.index("-filter_complex") + 1]
    assert "text='It\\'\\''s 9\\:00'" in graph
    assert "text='a\\\\b'" in graph


def test_build_render_args_escapes_the_outro_font_name(tmp_path):
    """font= 跟 text= 在同一个 AVOption 串里，字体名也来自 config，同样要转义。

    原来这条测试是 `monkeypatch.setattr(video_module, "OUTRO_FONT_NAME", ...)` —— 那时
    `build_render_args` 压根没有 `outro_font_name` 参数，只能去改模块级别名（也就是断言
    实现细节）。现在按参数传，断言的是行为。
    """
    args = build(
        tmp_path,
        outro_seconds=3.0,
        outro_title="x",
        outro_message="y",
        outro_font_name="It's:Font",
    )
    graph = args[args.index("-filter_complex") + 1]
    assert "font='It\\'\\''s\\:Font'" in graph


def test_build_render_args_outro_font_name_defaults_to_the_config_default(tmp_path):
    args = build(tmp_path, outro_seconds=3.0, outro_title="x", outro_message="y")
    graph = args[args.index("-filter_complex") + 1]
    assert f"font='{RenderConfig().outro_font_name}'" in graph


def test_render_video_passes_the_outro_font_name_through(tmp_path, monkeypatch):
    """render_video 也得有这个参数，否则 pipeline 递进来的值到不了 filtergraph。"""
    from tenmin.render import video as video_module

    seen: dict[str, object] = {}

    def fake_run(args, **_):
        seen["graph"] = args[args.index("-filter_complex") + 1]
        # atomic_path 要求块结束时临时文件还在，所以假 run 也得把它造出来。
        Path(args[-1]).write_bytes(b"x")
        return ""

    monkeypatch.setattr(video_module, "run_with_progress", fake_run)
    out = tmp_path / "07_render" / "E02.mp4"
    render_video(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        audio=tmp_path / "a.m4a",
        ass=tmp_path / "s.ass",
        out_path=out,
        encoder="libx264",
        outro_seconds=3.0,
        outro_title="t",
        outro_message="m",
        outro_font_name="MyCustomFont",
    )
    assert "font='MyCustomFont'" in seen["graph"]


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
        frame_rate=23.976023976023978,
    )
    graph = args[args.index("-filter_complex") + 1]
    assert (
        "color=c=black:s=1920x1080:r=24000/1001:d=3.000,setsar=1,format=yuv420p[cardbg]"
        in graph
    )
    assert "drawtext=font='Lantinghei SC':text='才女的侍从 · EP02'" in graph
    assert "drawtext=font='Lantinghei SC':text='解说结束，谢谢观看'" in graph
    assert graph.endswith("[vfaded][card]concat=n=2:v=1:a=0[vfinal]")
    assert args[args.index("-map") + 1] == "[vfinal]"


def test_outro_card_runs_at_the_source_frame_rate(tmp_path):
    """卡片不带 `r=` 时 color 源默认 25fps，跟 23.976 的正片 concat 出来是 VFR。

    实测（真实 E02 成片）修前 `avg_frame_rate=869000000/36223751`（23.990）而
    `r_frame_rate=24000/1001`（23.976）—— 总时长对得上，帧率是编出来的。
    """
    args = build(tmp_path, outro_seconds=3.0, frame_rate=25.0)
    graph = args[args.index("-filter_complex") + 1]
    assert "r=25:" in graph


def test_outro_card_omits_the_rate_when_the_frame_rate_is_unknown(tmp_path):
    """帧率是 None 时退回老行为（不写 r=），而不是猜一个数写上去。

    生产路径上 pipeline.run_render 一定探到了帧率（源片不存在的话渲染本来就跑不了），
    所以这条分支只服务于「库调用方/单测手上没有真源片」。
    """
    args = build(tmp_path, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert "color=c=black:s=1920x1080:d=3.000,setsar=1,format=yuv420p[cardbg]" in graph
    assert ":r=" not in graph


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


# --- 产物原子写 -------------------------------------------------------------


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


# --- filtergraph 转义的真 ffmpeg 回归锁 -------------------------------------
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


@pytest.mark.render
@pytest.mark.parametrize(
    "title",
    [
        "\u624d\u5973\u7684\u4f8d\u4ece \u00b7 EP02",  # 现状：正常文案
        "It's ok",
        "9:00",
        "a\\b",
        "100%",  # expansion=none 关掉了 strftime 展开
        "%Y-%m-%d",
        "a,b",
        "a=b",
        "[x];y",
        "O'B:x\\y%z,w=v",  # 全都来一遍
    ],
)
def test_real_ffmpeg_accepts_outro_text_with_special_characters(tmp_path, title):
    """片尾卡文案带特殊字符时，真 ffmpeg 必须能解析并渲染。"""
    from tenmin.render.ffmpeg import has_filter
    from tenmin.render.video import render_video

    if not has_filter("subtitles"):
        pytest.skip("ffmpeg 没编 libass，跑不了 subtitles 滤镜")
    if not has_filter("drawtext"):
        pytest.skip("ffmpeg 没编 drawtext 滤镜，画不了片尾黑卡")

    video, audio = _lavfi_inputs(tmp_path)
    ass = tmp_path / "E02.ass"
    ass.write_text(MINIMAL_ASS, encoding="utf-8")

    out = render_video(
        video=video,
        timeline=_one_second_timeline(),
        audio=audio,
        ass=ass,
        out_path=tmp_path / "out" / "E02.mp4",
        encoder="libx264",
        outro_seconds=0.5,
        outro_title=title,
        outro_message=title,
    )
    assert out.is_file()
    assert out.stat().st_size > 0


def test_frame_rate_arg_restores_the_exact_rational():
    """十进制近似会让成片仍然不是严格 CFR，见 frame_rate_arg 的 docstring。"""
    assert frame_rate_arg(23.976023976023978) == "24000/1001"
    assert frame_rate_arg(29.97002997002997) == "30000/1001"
    assert frame_rate_arg(25.0) == "25"


def test_render_video_progress_total_is_the_timeline_output_length(tmp_path, monkeypatch):
    """进度分母 = Timeline.output_seconds，跟混音那边同源（两份真相已经收敛）。"""
    from tenmin.render import video as video_module

    captured: dict[str, float] = {}

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        captured["total_seconds"] = total_seconds
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(video_module, "run_with_progress", fake_run_with_progress)
    timeline = make_timeline()
    render_video(
        video=tmp_path / "source.mkv",
        timeline=timeline,
        audio=tmp_path / "a.m4a",
        ass=tmp_path / "s.ass",
        out_path=tmp_path / "07_render" / "E02.mp4",
        encoder="libx264",
        outro_seconds=3.0,
    )
    assert captured["total_seconds"] == pytest.approx(timeline.output_seconds(3.0))


def test_render_video_reports_each_percent_only_once(tmp_path, monkeypatch):
    """同一个整数百分比不该回调两次。

    ffmpeg 每秒发好几个 progress 块，而 substep 收的是整数百分比 —— 一个 240 秒的
    成片会把同一个数字重复发出去几十遍（rich 那边就是几十次无用重绘）。
    """
    from tenmin.render import video as video_module

    from .fakes import FakeReporter

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        assert on_progress is not None
        for fraction in (0.0, 0.001, 0.004, 0.5, 0.502, 0.999, 1.0):
            on_progress(fraction)
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(video_module, "run_with_progress", fake_run_with_progress)
    reporter = FakeReporter()
    render_video(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        audio=tmp_path / "a.m4a",
        ass=tmp_path / "s.ass",
        out_path=tmp_path / "07_render" / "E02.mp4",
        encoder="libx264",
        reporter=reporter,
    )
    assert [call[2] for call in reporter.calls] == [0, 50, 99, 100]
