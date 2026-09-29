import subprocess
from pathlib import Path

import pytest

from tenmin.config import RenderConfig
from tenmin.models import (
    HoldWindow,
    SubtitleCue,
    Timeline,
    TimelineSegment,
    VoiceChunk,
    VoiceTrack,
)
from tenmin.render.audio import (
    SILENCE_CHANNEL_LAYOUT,
    SILENCE_SAMPLE_FMT,
    SILENCE_SAMPLE_RATE,
    build_mix_args,
    duck_gain,
    duck_volume_expr,
    mix_audio,
    mixed_total_seconds,
)


def make_timeline() -> Timeline:
    """两个 segment、三个旁白 chunk，画面总时长与音频总时长都是 30 秒。"""
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
        subtitles=[
            SubtitleCue(start=0.0, end=8.0, text="第一句"),
            SubtitleCue(start=10.0, end=20.0, text="第二句"),
            SubtitleCue(start=20.0, end=30.0, text="第三句"),
        ],
        narration_offsets=[0.0, 10.0, 20.0],
        total_seconds=30.0,
    )


def make_track() -> VoiceTrack:
    return VoiceTrack(
        episode=2,
        chunks=[
            VoiceChunk(
                beat_id="b1",
                index=1,
                text="第一句",
                path="chunk_001.mp3",
                duration=8.0,
                hold_after=2.0,
            ),
            VoiceChunk(
                beat_id="b1",
                index=2,
                text="第二句",
                path="chunk_002.mp3",
                duration=10.0,
            ),
            VoiceChunk(
                beat_id="b2",
                index=1,
                text="第三句",
                path="chunk_003.mp3",
                duration=10.0,
            ),
        ],
        total_seconds=30.0,
    )


EXPECTED_GRAPH = (
    "[0:a]atrim=start=100.000:end=120.000,asetpts=PTS-STARTPTS[o0];"
    "[0:a]atrim=start=200.000:end=210.000,asetpts=PTS-STARTPTS[o1];"
    "[o0][o1]concat=n=2:v=0:a=1[orig];"
    "[orig]volume='if(gt(between(t,0.000,8.000)+between(t,10.000,30.000)"
    ",0),0.2512,1.0000)':eval=frame[ducked];"
    "[1:a]adelay=delays=0:all=1[n0];"
    "[2:a]adelay=delays=10000:all=1[n1];"
    "[3:a]adelay=delays=20000:all=1[n2];"
    "[n0][n1][n2]amix=inputs=3:normalize=0[voice];"
    "[ducked][voice]amix=inputs=2:normalize=0[mix];"
    "[mix]apad=whole_dur=30.000,atrim=end=30.000[mixlen];"
    "[mixlen]alimiter=limit=1:level=false:latency=true[limited]"
)


def build(tmp_path, **overrides):
    kwargs = {
        "video": tmp_path / "source.mkv",
        "timeline": make_timeline(),
        "track": make_track(),
        "voice_dir": tmp_path / "04_voice" / "E02",
        "out_path": tmp_path / "06_audio" / "E02.mixed.m4a",
        "duck_db": -12.0,
    }
    kwargs.update(overrides)
    return build_mix_args(**kwargs)


def test_duck_gain_zero_db_is_unity():
    assert duck_gain(0.0) == pytest.approx(1.0)


def test_duck_gain_minus_twelve_db():
    assert duck_gain(-12.0) == pytest.approx(0.2512, abs=1e-4)


def test_duck_volume_expr_without_cues_stays_full():
    assert duck_volume_expr([], 0.2512) == "1.0000"


def test_duck_volume_expr_keeps_windows_separated_by_a_gap():
    cues = [SubtitleCue(start=0.0, end=8.0, text="a"), SubtitleCue(start=10.0, end=20.0, text="b")]
    assert duck_volume_expr(cues, 0.2512) == (
        "if(gt(between(t,0.000,8.000)+between(t,10.000,20.000),0),0.2512,1.0000)"
    )


# --- ducking 区间合并（第 4 项）---------------------------------------------
# 相邻 cue 大多首尾相接（真实素材实测约 87%），一个 cue 一个 between() 会拼出
# 几十项的表达式，而它们本可以是少数几个连续窗。between() 是闭区间，所以合并
# 相接/重叠的区间在数学上逐点等价。


def test_duck_volume_expr_merges_touching_cues():
    """前一条的 end 正好是后一条的 start：between 是闭区间，合并后逐点等价。"""
    cues = [SubtitleCue(start=0.0, end=8.0, text="a"), SubtitleCue(start=8.0, end=20.0, text="b")]
    assert duck_volume_expr(cues, 0.2512) == (
        "if(gt(between(t,0.000,20.000),0),0.2512,1.0000)"
    )


def test_duck_volume_expr_merges_overlapping_cues():
    cues = [
        SubtitleCue(start=0.0, end=8.0, text="a"),
        SubtitleCue(start=5.0, end=6.0, text="被包住的短句"),
        SubtitleCue(start=7.5, end=20.0, text="b"),
    ]
    assert duck_volume_expr(cues, 0.2512) == (
        "if(gt(between(t,0.000,20.000),0),0.2512,1.0000)"
    )


def test_duck_volume_expr_sorts_before_merging():
    """cue 顺序不该影响结果 —— 合并前先按起点排序。"""
    cues = [SubtitleCue(start=8.0, end=20.0, text="b"), SubtitleCue(start=0.0, end=8.0, text="a")]
    assert duck_volume_expr(cues, 0.2512) == (
        "if(gt(between(t,0.000,20.000),0),0.2512,1.0000)"
    )


def test_build_mix_args_inputs_video_then_every_chunk(tmp_path):
    args = build(tmp_path)
    voice_dir = tmp_path / "04_voice" / "E02"
    assert args[:3] == ["-y", "-i", str(tmp_path / "source.mkv")]
    assert args[3:5] == ["-i", str(voice_dir / "chunk_001.mp3")]
    assert args[5:7] == ["-i", str(voice_dir / "chunk_002.mp3")]
    assert args[7:9] == ["-i", str(voice_dir / "chunk_003.mp3")]


def test_build_mix_args_filter_graph_matches_expected(tmp_path):
    args = build(tmp_path)
    graph = args[args.index("-filter_complex") + 1]
    assert graph == EXPECTED_GRAPH


def test_build_mix_args_maps_mix_and_encodes_aac(tmp_path):
    args = build(tmp_path)
    out = tmp_path / "06_audio" / "E02.mixed.m4a"
    assert args[-7:] == ["-map", "[limited]", "-c:a", "aac", "-b:a", "192k", str(out)]


def test_build_mix_args_single_chunk_skips_voice_amix(tmp_path):
    timeline = make_timeline()
    timeline.segments = timeline.segments[:1]
    timeline.subtitles = timeline.subtitles[:1]
    timeline.narration_offsets = [0.0]
    track = make_track()
    track.chunks = track.chunks[:1]
    args = build(tmp_path, timeline=timeline, track=track)
    graph = args[args.index("-filter_complex") + 1]
    assert "amix=inputs=1" not in graph
    assert "[ducked][n0]amix=inputs=2:normalize=0[mix]" in graph


def test_build_mix_args_rejects_empty_timeline(tmp_path):
    timeline = make_timeline()
    timeline.segments = []
    with pytest.raises(ValueError) as exc:
        build(tmp_path, timeline=timeline)
    assert "segment" in str(exc.value)


def test_build_mix_args_rejects_empty_track(tmp_path):
    track = make_track()
    track.chunks = []
    with pytest.raises(ValueError) as exc:
        build(tmp_path, track=track)
    assert "chunk" in str(exc.value)


def test_build_mix_args_rejects_offset_count_mismatch(tmp_path):
    timeline = make_timeline()
    timeline.narration_offsets = [0.0]
    with pytest.raises(ValueError):
        build(tmp_path, timeline=timeline)


def test_build_mix_args_pins_length_even_without_fade_or_outro(tmp_path):
    """不淡出、不加片尾时也必须钉长度 —— 输出长度不该由「哪条输入最长」决定。"""
    args = build(tmp_path)
    graph = args[args.index("-filter_complex") + 1]
    assert "[mix]apad=whole_dur=30.000,atrim=end=30.000[mixlen]" in graph


def test_build_mix_args_applies_fade_out_before_mix_ends(tmp_path):
    args = build(tmp_path, fade_out_seconds=5.0)
    graph = args[args.index("-filter_complex") + 1]
    assert (
        "[mix]apad=whole_dur=30.000,atrim=end=30.000[mixlen];"
        "[mixlen]afade=t=out:st=25.000:d=5.000[mixfaded]"
    ) in graph


def test_build_mix_args_appends_silence_for_outro_card(tmp_path):
    args = build(tmp_path, fade_out_seconds=5.0, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert (
        "[mixlen]afade=t=out:st=25.000:d=5.000[mixfaded];"
        "[mixfaded]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[mixfmt];"
        "anullsrc=r=48000:cl=stereo:d=3.000[silence];"
        "[mixfmt][silence]concat=n=2:v=0:a=1,asetpts=N/SR/TB[mixfinal]"
    ) in graph


def test_build_mix_args_regenerates_pts_after_the_outro_concat(tmp_path):
    """concat 出来的 pts 有洞，mp4 muxer 会据此随机写出一个偏短的容器时长。

    真实素材实测（work/saijo E02，同一条 argv 跑 6 遍）：容器 duration 在
    217.404000 与 214.424229 之间乱跳，而两者的 AAC 帧数都是 10192、解码出的
    样本逐字节相同 —— 也就是说样本没丢，是 moov 里那个数字写错了。按样本数重建
    pts（asetpts=N/SR/TB）之后 6/6 都是 217.404000。
    """
    args = build(tmp_path, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert "concat=n=2:v=0:a=1,asetpts=N/SR/TB[mixfinal]" in graph


def test_build_mix_args_outro_without_fade_concats_mix_directly(tmp_path):
    args = build(tmp_path, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert (
        "[mix]apad=whole_dur=30.000,atrim=end=30.000[mixlen];"
        "[mixlen]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[mixfmt];"
        "anullsrc=r=48000:cl=stereo:d=3.000[silence];"
        "[mixfmt][silence]concat=n=2:v=0:a=1,asetpts=N/SR/TB[mixfinal]"
    ) in graph


# --- 输出长度受控（第 2 项）-------------------------------------------------
# 原来 build_mix_args 既没有 -t 也没有 apad，输出长度是 amix（默认 duration=longest）
# 算出来的 max(画面音频, 末条配音结束)，而不是 timeline 声明的长度。render 阶段
# `-c:a copy -map 1:a` 之后，mp4 的容器时长会被音频反过来决定。


def test_build_mix_args_pins_length_to_timeline_total(tmp_path):
    args = build(tmp_path)
    graph = args[args.index("-filter_complex") + 1]
    assert "[mix]apad=whole_dur=30.000,atrim=end=30.000[mixlen]" in graph


def test_build_mix_args_pins_length_before_fade_and_outro(tmp_path):
    """长度必须先钉死，再淡出、再接片尾静音：淡出的起点是 timeline 坐标，
    钉长度放在淡出之后的话，混音短了一截时淡出会落在不存在的样本上。"""
    args = build(tmp_path, fade_out_seconds=5.0, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert (
        "[mix]apad=whole_dur=30.000,atrim=end=30.000[mixlen];"
        "[mixlen]afade=t=out:st=25.000:d=5.000[mixfaded];"
        "[mixfaded]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[mixfmt];"
        "anullsrc=r=48000:cl=stereo:d=3.000[silence];"
        "[mixfmt][silence]concat=n=2:v=0:a=1,asetpts=N/SR/TB[mixfinal]"
    ) in graph


def test_build_mix_args_skips_length_pin_when_total_seconds_is_zero(tmp_path):
    """total_seconds 为 0 时钉长度只会产出一个空音轨，宁可退回不钉。"""
    timeline = make_timeline()
    timeline.total_seconds = 0.0
    args = build(tmp_path, timeline=timeline)
    graph = args[args.index("-filter_complex") + 1]
    assert "apad" not in graph
    assert "atrim=end=0.000" not in graph
    assert args[args.index("-map") + 1] == "[limited]"


# --- 限幅（第 3 项）---------------------------------------------------------
# ducked 原声（-12dB）+ 满量程旁白在响场景相加可能冲过满刻度，而 amix
# normalize=0 不做任何归一化。真实素材实测最响的一集（work/saijo E06）编码前峰值
# -0.90 dBFS —— 没削波，但只剩 0.9dB 余量。


def test_build_mix_args_limits_the_final_signal(tmp_path):
    args = build(tmp_path)
    graph = args[args.index("-filter_complex") + 1]
    assert graph.endswith("[mixlen]alimiter=limit=1:level=false:latency=true[limited]")
    assert args[args.index("-map") + 1] == "[limited]"


def test_build_mix_args_limits_after_every_absolute_time_filter(tmp_path):
    """限幅必须排在 atrim / afade / concat **之后**。

    latency=true 补偿前瞻延迟的做法是把输出 pts 往前挪一个 attack 窗，而那三个
    滤镜全部按绝对 timeline 时刻工作。真实素材实测把它插在 apad/atrim 之前，
    `atrim=end=214.404` 会少留 239 个样本（≈5ms@48kHz，正好一个 attack 窗）。
    """
    args = build(tmp_path, fade_out_seconds=5.0, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert graph.endswith("[mixfinal]alimiter=limit=1:level=false:latency=true[limited]")
    for earlier in ("apad=", "atrim=end=30.000", "afade=", "concat=n=2:v=0:a=1"):
        assert graph.index(earlier) < graph.index("alimiter"), earlier


@pytest.mark.render
def test_alimiter_options_are_transparent_below_the_ceiling():
    """limit=1:level=false:latency=true 对没超标的信号必须逐字节不变。

    三个参数缺一不可：level 默认 true，latency 默认 false（不补偿前瞻延迟，
    整条轨会平移几毫秒）。任何一个用默认值都会改听感，而用户对成片听感有既定期待。
    """
    plain = _lavfi_pcm("sine=f=440:d=2:r=48000,volume=0.5")
    limited = _lavfi_pcm(
        "sine=f=440:d=2:r=48000,volume=0.5,alimiter=limit=1:level=false:latency=true"
    )
    assert limited == plain


@pytest.mark.render
def test_alimiter_actually_caps_a_clipping_signal():
    """真超标时限幅必须把峰值压到 0 dBFS，否则这一层等于没加。"""
    assert _lavfi_peak_db("sine=f=440:d=2:r=48000,volume=24") > 1.0
    peak = _lavfi_peak_db(
        "sine=f=440:d=2:r=48000,volume=24,alimiter=limit=1:level=false:latency=true"
    )
    assert peak == pytest.approx(0.0, abs=0.01)


def _lavfi_pcm(graph: str) -> bytes:
    """跑一条 lavfi 滤镜链，取回 f32le 原始样本。"""
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", graph,
         "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _lavfi_peak_db(graph: str) -> float:
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-f", "lavfi", "-i", graph,
         "-af", "astats=measure_perchannel=Peak_level:measure_overall=Peak_level",
         "-f", "null", "-"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    peaks = [
        float(line.rsplit(":", 1)[1])
        for line in completed.stderr.splitlines()
        if "Peak level dB:" in line
    ]
    assert peaks, completed.stderr
    return peaks[-1]


# --- 片尾静音的格式（第 5 项）-----------------------------------------------
# 图里其余分支都继承源片格式，只有 anullsrc 写死 48kHz/stereo。ffmpeg 会自动协商
# （44.1k/mono + 48k/stereo 拼一起实测 rc=0，不是崩），但 5.1 源片会被**静默下混**。
# 在 concat 前显式 aformat，让「下混」这件事是写出来的而不是协商出来的。


def test_build_mix_args_forces_a_known_format_before_the_outro_concat(tmp_path):
    args = build(tmp_path, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert (
        "[mix]apad=whole_dur=30.000,atrim=end=30.000[mixlen];"
        "[mixlen]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[mixfmt];"
        "anullsrc=r=48000:cl=stereo:d=3.000[silence];"
        "[mixfmt][silence]concat=n=2:v=0:a=1,asetpts=N/SR/TB[mixfinal]"
    ) in graph


def test_build_mix_args_leaves_the_format_alone_without_an_outro_card(tmp_path):
    """没有片尾静音就没有两路格式相遇的点，这条路继续继承源片格式，行为不变。"""
    graph = build(tmp_path)[build(tmp_path).index("-filter_complex") + 1]
    assert "aformat" not in graph


def test_build_mix_args_keeps_silence_and_aformat_in_sync(tmp_path):
    """静音源与 aformat 的采样率/声道布局必须来自同一组常量，不能各写一份。"""
    args = build(tmp_path, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert f"anullsrc=r={SILENCE_SAMPLE_RATE}:cl={SILENCE_CHANNEL_LAYOUT}" in graph
    assert f"sample_fmts={SILENCE_SAMPLE_FMT}" in graph
    assert f"sample_rates={SILENCE_SAMPLE_RATE}" in graph
    assert f"channel_layouts={SILENCE_CHANNEL_LAYOUT}" in graph


# --- 进度反馈（第 6 项）-----------------------------------------------------
# mix_audio 原来用 ffmpeg.run()，一次几分钟的真实编码期间界面上什么都不动。
# render 阶段早就在用 run_with_progress 了。


def test_mixed_total_seconds_is_the_body_plus_the_outro_card():
    """产物长度的唯一真相：正片长度（build_mix_args 钉的那个）+ 片尾卡片。"""
    timeline = make_timeline()
    assert mixed_total_seconds(timeline, 0.0) == pytest.approx(30.0)
    assert mixed_total_seconds(timeline, 3.0) == pytest.approx(33.0)


def test_mix_audio_reports_progress_against_the_pinned_length(tmp_path, monkeypatch):
    """进度条的总长必须跟钉住的产物长度一致，否则百分比会停在别的地方。"""
    from tenmin.render import audio as audio_module

    seen: dict[str, float] = {}

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        seen["total_seconds"] = total_seconds
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(audio_module, "run_with_progress", fake_run_with_progress)
    audio_module.mix_audio(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        track=make_track(),
        voice_dir=tmp_path / "04_voice" / "E02",
        out_path=tmp_path / "06_audio" / "E02.mixed.m4a",
        duck_db=-12.0,
        fade_out_seconds=1.5,
        outro_seconds=3.0,
    )
    assert seen["total_seconds"] == pytest.approx(33.0)


def test_mix_audio_forwards_progress_to_the_reporter(tmp_path, monkeypatch):
    from tenmin.render import audio as audio_module

    from .fakes import FakeReporter

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        if on_progress is not None:
            on_progress(0.5)
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(audio_module, "run_with_progress", fake_run_with_progress)
    reporter = FakeReporter()
    audio_module.mix_audio(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        track=make_track(),
        voice_dir=tmp_path / "04_voice" / "E02",
        out_path=tmp_path / "06_audio" / "E02.mixed.m4a",
        duck_db=-12.0,
        reporter=reporter,
    )
    assert reporter.calls == [("substep", "audio", 50, 100, "")]


def test_mix_audio_runs_ffmpeg_and_returns_path(tmp_path, monkeypatch):
    from tenmin.atomic import part_path
    from tenmin.render import audio as audio_module

    seen: list[list[str]] = []

    def fake_run(args, **_):
        seen.append(list(args))
        # 真 ffmpeg 一定会把输出文件写出来；假的也得写，否则原子改名无从下手。
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(audio_module, "run_with_progress", fake_run)
    out_path = tmp_path / "06_audio" / "E02.mixed.m4a"
    result = audio_module.mix_audio(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        track=make_track(),
        voice_dir=tmp_path / "04_voice" / "E02",
        out_path=out_path,
        duck_db=-12.0,
    )
    assert result == out_path
    assert out_path.parent.is_dir()
    assert seen[0][0] == "-y"
    # ffmpeg 写的是同目录的 .part，跑完才原子改名到 out_path
    assert seen[0][-1] == str(part_path(out_path))
    assert out_path.is_file()


def test_mix_audio_is_exported():
    assert callable(mix_audio)


def test_mix_audio_passes_configured_ffmpeg_binary(tmp_path, monkeypatch):
    """RenderConfig.ffmpeg_path 必须真的传到 subprocess，不然那个旋钮是哑的。"""
    from tenmin.render import audio as audio_module

    seen: dict[str, str] = {}

    def fake_run(args, *, ffmpeg="ffmpeg", **_):
        seen["ffmpeg"] = ffmpeg
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(audio_module, "run_with_progress", fake_run)
    audio_module.mix_audio(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        track=make_track(),
        voice_dir=tmp_path / "04_voice" / "E02",
        out_path=tmp_path / "06_audio" / "E02.mixed.m4a",
        duck_db=-12.0,
        ffmpeg="/opt/libass/bin/ffmpeg",
    )
    assert seen["ffmpeg"] == "/opt/libass/bin/ffmpeg"


# --- 产物原子写 -------------------------------------------------------------
# mix_audio 原来直接 `-y` 写最终路径，被打断就留一个 mtime 最新的截断 m4a，
# 而 pipeline._is_fresh 只比 mtime，于是下一轮把它当最新产物跳过、坏音频进成片。


def test_mix_audio_tells_ffmpeg_to_write_a_part_file(tmp_path, monkeypatch):
    from tenmin.atomic import part_path
    from tenmin.render import audio as audio_module

    seen: list[list[str]] = []

    def fake_run(args, **_):
        seen.append(list(args))
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(audio_module, "run_with_progress", fake_run)
    out_path = tmp_path / "06_audio" / "E02.mixed.m4a"
    audio_module.mix_audio(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        track=make_track(),
        voice_dir=tmp_path / "04_voice" / "E02",
        out_path=out_path,
        duck_db=-12.0,
    )
    assert seen[0][-1] == str(part_path(out_path))
    # 扩展名必须留着：ffmpeg 靠它推断容器格式
    assert seen[0][-1].endswith(".m4a")
    assert out_path.is_file()
    assert not part_path(out_path).exists()


def test_mix_audio_keeps_the_previous_artifact_when_ffmpeg_fails(tmp_path, monkeypatch):
    from tenmin.atomic import part_path
    from tenmin.render import audio as audio_module
    from tenmin.render.ffmpeg import FFmpegError

    out_path = tmp_path / "06_audio" / "E02.mixed.m4a"
    out_path.parent.mkdir(parents=True)
    out_path.write_bytes(b"good")
    before = out_path.stat().st_mtime_ns

    def fake_run(args, **_):
        Path(args[-1]).write_bytes(b"truncated")
        raise FFmpegError("boom")

    monkeypatch.setattr(audio_module, "run_with_progress", fake_run)
    with pytest.raises(FFmpegError):
        audio_module.mix_audio(
            video=tmp_path / "source.mkv",
            timeline=make_timeline(),
            track=make_track(),
            voice_dir=tmp_path / "04_voice" / "E02",
            out_path=out_path,
            duck_db=-12.0,
        )
    assert out_path.read_bytes() == b"good"
    assert out_path.stat().st_mtime_ns == before
    assert not part_path(out_path).exists()


# --- 留白窗单独增益（第 6 项）------------------------------------------------
# timeline.hold_windows 是人能手改的产物。混音时必须照 render/timeline.py 那套
# 「静音窗合法性」规则再验一遍，而不是直接信任 timeline.json 里写的数字；没请求
# 增益（hold_gains 为空）时图必须逐字节不变。


def test_valid_hold_windows_rejects_non_chronological_source_reuse():
    """`_valid_hold_windows` 的连续性判断必须跟 render/timeline.py 的 `_assess_holds`
    同口径：**不排序**，只看给定顺序上是否首尾相接。

    这里构造一个「倒叙」剪辑：同一个 beat 的三段画面按时间轴顺序拼在一起，但它们
    在原片里的位置是 100→90→110（先跳回去、再跳过去），本身并不连续。如果验证时
    先排序（90,100,110）再看首尾相接，会误判为连续从而接受这个窗口；不排序的话，
    按 100→90→110 这个真实顺序看，第一段到第二段之间有一个 20 秒的跳跃，必须
    判定不连续、拒绝这个窗口。
    """
    from tenmin.render.audio import _valid_hold_windows

    segments = [
        TimelineSegment(beat_id="b1", source_start=100.0, source_end=110.0,
                         timeline_start=8.0, timeline_end=18.0),
        TimelineSegment(beat_id="b1", source_start=90.0, source_end=100.0,
                         timeline_start=18.0, timeline_end=28.0),
        TimelineSegment(beat_id="b1", source_start=110.0, source_end=120.0,
                         timeline_start=28.0, timeline_end=38.0),
    ]
    timeline = Timeline(
        episode=2,
        segments=segments,
        subtitles=[],
        narration_offsets=[0.0],
        total_seconds=38.0,
    )
    window = HoldWindow(beat_id="b1", hold_index=0, quote="a", episode=2,
                        source_start=90.0, source_end=120.0, start=8.0, end=38.0)
    timeline.hold_windows = [window]
    track = VoiceTrack(
        episode=2,
        chunks=[VoiceChunk(beat_id="b1", index=1, text="第一句",
                            path="chunk_001.mp3", duration=8.0)],
        total_seconds=38.0,
    )

    kept, warnings = _valid_hold_windows(timeline, track)

    assert kept == []
    assert len(warnings) == 1 and "留白窗" in warnings[0]


def test_valid_hold_windows_rejects_backward_source_overlap_without_forward_gap():
    """连续性判断必须是**对称**的：不仅要拒绝正向跳跃（gap），也要拒绝反向重叠
    （下一段的原片起点落在上一段原片终点之前）——跟 render/timeline.py 的
    `_assess_holds` 保持一致的对称判据（用 `abs(...)` 判差值，而不是只判其中一个
    方向）。

    这里构造一个只有反向重叠、没有任何补偿性正向跳跃的场景：同一个 beat 的两段
    画面按时间轴顺序拼在一起，第二段在原片里的起点比第一段的终点早 15 秒（
    110 → 95），此外序列里再没有别的跳跃能让「净差值」凑巧抵消。如果连续性判断
    只检查正向跳跃（`played[i+1][0] - played[i][1] > tolerance`），这种反向重叠
    不会触发任何一侧的判断，会被错误地当成连续、错误地保留这个留白窗。
    """
    from tenmin.render.audio import _valid_hold_windows

    segments = [
        TimelineSegment(beat_id="b1", source_start=100.0, source_end=110.0,
                         timeline_start=8.0, timeline_end=18.0),
        TimelineSegment(beat_id="b1", source_start=95.0, source_end=105.0,
                         timeline_start=18.0, timeline_end=28.0),
    ]
    timeline = Timeline(
        episode=2,
        segments=segments,
        subtitles=[],
        narration_offsets=[0.0],
        total_seconds=28.0,
    )
    window = HoldWindow(beat_id="b1", hold_index=0, quote="a", episode=2,
                        source_start=95.0, source_end=110.0, start=8.0, end=28.0)
    timeline.hold_windows = [window]
    track = VoiceTrack(
        episode=2,
        chunks=[VoiceChunk(beat_id="b1", index=1, text="第一句",
                            path="chunk_001.mp3", duration=8.0)],
        total_seconds=28.0,
    )

    kept, warnings = _valid_hold_windows(timeline, track)

    assert kept == []
    assert len(warnings) == 1 and "留白窗" in warnings[0]


def test_manually_moved_window_is_ignored_and_valid_second_window_keeps_identity():
    from tenmin.render.audio import _valid_hold_windows

    timeline = make_timeline()
    bad = HoldWindow(beat_id="b1", hold_index=0, quote="a", episode=2,
                     source_start=108, source_end=109, start=7, end=10)
    good = HoldWindow(beat_id="b1", hold_index=1, quote="b", episode=2,
                      source_start=108, source_end=109, start=8, end=10)
    timeline.hold_windows = [bad, good]
    kept, warnings = _valid_hold_windows(timeline, make_track())
    assert kept == [(1, good)]
    assert len(warnings) == 1 and "留白窗" in warnings[0]


def test_window_gain_is_only_inside_silence_and_before_ducking(tmp_path):
    timeline = make_timeline()
    timeline.hold_windows = [HoldWindow(beat_id="b1", hold_index=0, quote="a",
        episode=2, source_start=108, source_end=109, start=8, end=10)]
    args = build(tmp_path, timeline=timeline, hold_gains={0: -6}, hold_fade_seconds=0.1)
    graph = args[args.index("-filter_complex") + 1]
    assert "between(t,8.000,10.000)" in graph
    assert "0.501187" in graph  # -6 dB linear amplitude
    assert graph.index("[holdbalanced]") < graph.index("[ducked]")


def test_hold_gain_expr_keeps_both_amplitudes_for_two_distinct_windows():
    """两个不重叠窗口各自的增益必须都出现在表达式里，不能被 reversed() 的嵌套折叠
    只剩最后一个——这条链路原来只测过单窗口，没测过「两个窗口、两个不同增益」。"""
    from tenmin.render.audio import hold_gain_expr

    first = HoldWindow(beat_id="b1", hold_index=0, quote="a", episode=2,
                        source_start=0.0, source_end=1.0, start=0.0, end=2.0)
    second = HoldWindow(beat_id="b1", hold_index=1, quote="b", episode=2,
                         source_start=0.0, source_end=1.0, start=5.0, end=8.0)
    windows = [(0, first), (1, second)]
    gains = {0: -4.0, 1: 3.0}

    expr = hold_gain_expr(windows, gains, fade=0.0)

    amp_first = 10 ** (-4.0 / 20)
    amp_second = 10 ** (3.0 / 20)
    assert f"{amp_first:.6f}" in expr
    assert f"{amp_second:.6f}" in expr
    assert "between(t,0.000,2.000)" in expr
    assert "between(t,5.000,8.000)" in expr


def test_hold_gain_expr_ramp_boundaries_match_the_fade_window():
    """校验淡入/淡出坡道本身的子表达式，不只是常量目标增益——覆盖
    fade_in_end/fade_out_start 与线性插值公式的具体数值。"""
    from tenmin.render.audio import hold_gain_expr

    window = HoldWindow(beat_id="b1", hold_index=0, quote="a", episode=2,
                         source_start=0.0, source_end=1.0, start=10.0, end=20.0)
    gain_db = -6.0
    amp = 10 ** (gain_db / 20)
    fade = 2.0  # 窗口时长 10s 的一半是 5s，2s 不会被夹到更小，fade_eff == fade

    expr = hold_gain_expr([(0, window)], {0: gain_db}, fade)

    fade_in_end = 12.000
    fade_out_start = 18.000
    assert f"between(t,10.000,{fade_in_end:.3f})" in expr
    assert f"between(t,{fade_out_start:.3f},20.000)" in expr
    assert (
        f"1.000000+({amp:.6f}-1.000000)*(t-10.000)/{fade:.6f}"
    ) in expr
    assert (
        f"{amp:.6f}+(1.000000-{amp:.6f})*(t-{fade_out_start:.3f})/{fade:.6f}"
    ) in expr


def test_no_hold_gains_leaves_the_graph_byte_identical(tmp_path):
    """hold_gains 为 None/空 dict 时必须是同一份 EXPECTED_GRAPH——这是硬约束。"""
    args_default = build(tmp_path)
    args_none = build(tmp_path, hold_gains=None)
    args_empty = build(tmp_path, hold_gains={})
    graph = args_default[args_default.index("-filter_complex") + 1]
    assert graph == EXPECTED_GRAPH
    assert args_none == args_default
    assert args_empty == args_default


def test_audio_delays_follow_final_timeline_not_original_voice_hold_after(tmp_path):
    timeline = make_timeline()
    timeline.narration_offsets = [0, 8, 18]
    args = build(tmp_path, timeline=timeline)
    graph = args[args.index("-filter_complex") + 1]
    assert "adelay=delays=8000" in graph and "adelay=delays=18000" in graph


# --- 留白窗单独增益的响度决策 -----------------------------------------------
# _valid_hold_windows 只判「这个窗口可信不可信」，不判「该不该调音量、调多少」。
# gain_for_window 是后半段：拿一对实测响度（留白窗原声 / 旁白参考）决定调多少 dB，
# 边界见 RenderConfig 的 hold_relative_lu/hold_gain_max_db/hold_silence_floor_lufs。


@pytest.mark.parametrize(
    ("source", "voice", "gain", "warn"),
    [
        (-25, -32, -4, False),  # upper bound -29, source is 4 above -> reduce 4 dB
        (-37, -30, 4, False),  # lower bound -33, source is 4 below -> boost 4 dB
        (-50, -30, 0, True),  # below silence floor -> no boost, warn
        (None, -30, 0, True),  # unmeasurable source -> no boost, warn
        (-30, None, 0, True),  # unmeasurable voice reference -> no adjustment, warn
        (-10, -32, -6, False),  # would need -10 dB reduction, capped at -6
    ],
)
def test_window_gain_meets_nearest_relative_bound_without_amplifying_noise(
    source, voice, gain, warn
):
    from tenmin.render.audio import gain_for_window

    actual, message = gain_for_window(source, voice, cfg=RenderConfig())
    assert actual == gain
    assert (message is not None) is warn


def test_ebur128_parser_ignores_invalid_and_uses_last_integrated_report():
    from tenmin.render.audio import _parse_ebur128_i

    assert _parse_ebur128_i("I: -29.0 LUFS\nI: -23.5 LUFS") == -23.5
    assert _parse_ebur128_i("I: -inf LUFS") is None


# --- _meter_args：响度实测用的最小 ffmpeg 图 ---------------------------
# `_meter_args` 是纯图构造函数（跟 build_mix_args 同一个套路：不碰 subprocess，
# 只负责拼 argv），所以这里直接调用检查返回值，不需要 monkeypatch audio.run——
# 那个壳子在 `_measure_hold_gains` 的测试里才真正被调用。两个分支都必须复用
# `_origin_parts`/`_voice_parts` 产出的**同一份**图片段，不能另起一份实现。


def test_meter_args_window_branch_trims_post_concat_orig_not_raw_stream(tmp_path):
    from tenmin.render.audio import _meter_args

    timeline = make_timeline()
    track = make_track()
    window = HoldWindow(
        beat_id="b1", hold_index=0, quote="a", episode=2,
        source_start=108, source_end=109, start=8.0, end=10.0,
    )
    args = _meter_args(
        tmp_path / "source.mkv", timeline, track, tmp_path / "voice", window=window
    )
    graph = args[args.index("-filter_complex") + 1]
    assert "[orig]atrim=start=8.000:end=10.000" in graph
    # 不能是对 [0:a] 直接 trim —— 必须先经过 concat 落到 timeline 坐标。
    assert "[0:a]atrim=start=8.000" not in graph
    assert "ebur128=peak=true" in graph


def test_meter_args_voice_branch_selects_narration_intervals_via_aselect(tmp_path):
    from tenmin.render.audio import _meter_args

    timeline = make_timeline()
    track = make_track()
    args = _meter_args(
        tmp_path / "source.mkv", timeline, track, tmp_path / "voice", window=None
    )
    graph = args[args.index("-filter_complex") + 1]
    assert "aselect=" in graph
    assert "between(t," in graph
    assert "ebur128=peak=true" in graph


def test_meter_args_neither_branch_encodes_or_loudnorms(tmp_path):
    from tenmin.render.audio import _meter_args

    timeline = make_timeline()
    track = make_track()
    window = HoldWindow(
        beat_id="b1", hold_index=0, quote="a", episode=2,
        source_start=108, source_end=109, start=8.0, end=10.0,
    )
    for candidate_window in (window, None):
        args = _meter_args(
            tmp_path / "source.mkv", timeline, track, tmp_path / "voice",
            window=candidate_window,
        )
        graph = args[args.index("-filter_complex") + 1]
        assert "loudnorm" not in graph
        assert args[-3:] == ["-f", "null", "-"]


def test_meter_args_inputs_video_then_every_chunk_like_the_real_mix(tmp_path):
    """两条分支都要保持跟 build_mix_args 一致的输入顺位，即便某条分支不需要视频。"""
    from tenmin.render.audio import _meter_args

    timeline = make_timeline()
    track = make_track()
    voice_dir = tmp_path / "voice"
    args = _meter_args(tmp_path / "source.mkv", timeline, track, voice_dir, window=None)
    assert args[:3] == ["-i", str(tmp_path / "source.mkv"), "-i"]
    assert args[3] == str(voice_dir / "chunk_001.mp3")


# --- _measure_hold_gains：一窗失败不该拖累其它窗 ---------------------------


def test_measure_hold_gains_skips_the_window_whose_meter_fails_but_still_measures_the_rest(
    tmp_path, monkeypatch
):
    from tenmin.render.audio import _measure_hold_gains
    from tenmin.render.ffmpeg import FFmpegError

    timeline = make_timeline()
    track = make_track()
    first = HoldWindow(beat_id="b1", hold_index=0, quote="a", episode=2,
                        source_start=108, source_end=109, start=8.0, end=10.0)
    second = HoldWindow(beat_id="b2", hold_index=1, quote="b", episode=2,
                         source_start=205, source_end=206, start=20.0, end=22.0)
    windows = [(0, first), (1, second)]

    responses = iter([
        "I: -30.0 LUFS\n",  # 旁白参考响度，先成功
        FFmpegError("boom"),  # 第一个窗口的响度测量失败
        "I: -25.0 LUFS\n",  # 第二个窗口正常测出来
    ])
    calls: list[list[str]] = []

    def fake_run(args, *, ffmpeg="ffmpeg"):
        calls.append(list(args))
        outcome = next(responses)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("tenmin.render.audio.run", fake_run)

    gains, warnings = _measure_hold_gains(
        tmp_path / "source.mkv", timeline, track, tmp_path / "voice", windows,
        cfg=RenderConfig(), ffmpeg="ffmpeg",
    )

    assert len(calls) == 3
    assert 0 not in gains
    assert 1 in gains
    assert any("b1" in w and "0" in w for w in warnings)


def test_measure_hold_gains_short_circuits_when_voice_reference_is_unmeasurable(
    tmp_path, monkeypatch
):
    """旁白参考响度测不出时，不该为任何窗口发起测量——没有参照，测了也白测。"""
    from tenmin.render.audio import _measure_hold_gains

    timeline = make_timeline()
    track = make_track()
    windows = [
        (0, HoldWindow(beat_id="b1", hold_index=0, quote="a", episode=2,
                        source_start=108, source_end=109, start=8.0, end=10.0)),
    ]

    calls: list[list[str]] = []

    def fake_run(args, *, ffmpeg="ffmpeg"):
        calls.append(list(args))
        return "I: -inf LUFS\n"

    monkeypatch.setattr("tenmin.render.audio.run", fake_run)

    gains, warnings = _measure_hold_gains(
        tmp_path / "source.mkv", timeline, track, tmp_path / "voice", windows,
        cfg=RenderConfig(), ffmpeg="ffmpeg",
    )

    assert gains == {}
    assert len(warnings) == 1
    assert len(calls) == 1


def test_mix_audio_reports_each_percent_only_once(tmp_path, monkeypatch):
    """跟 render_video 同一个毛病、同一个修法：整数百分比没变就不回调。"""
    from tenmin.render import audio as audio_module

    from .fakes import FakeReporter

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        assert on_progress is not None
        for fraction in (0.0, 0.002, 0.5, 0.5009, 1.0):
            on_progress(fraction)
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(audio_module, "run_with_progress", fake_run_with_progress)
    reporter = FakeReporter()
    voice_dir = tmp_path / "04_voice" / "E02"
    voice_dir.mkdir(parents=True)
    mix_audio(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        track=make_track(),
        voice_dir=voice_dir,
        out_path=tmp_path / "06_audio" / "E02.mixed.m4a",
        duck_db=-12.0,
        reporter=reporter,
    )
    assert [call[2] for call in reporter.calls] == [0, 50, 100]
