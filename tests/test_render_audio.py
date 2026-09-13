import subprocess
from pathlib import Path

import pytest

from tenmin.models import SubtitleCue, Timeline, TimelineSegment, VoiceChunk, VoiceTrack
from tenmin.render.audio import (
    build_mix_args,
    duck_gain,
    duck_volume_expr,
    mix_audio,
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
        "anullsrc=r=48000:cl=stereo:d=3.000[silence];"
        "[mixfaded][silence]concat=n=2:v=0:a=1,asetpts=N/SR/TB[mixfinal]"
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
        "anullsrc=r=48000:cl=stereo:d=3.000[silence];"
        "[mixlen][silence]concat=n=2:v=0:a=1,asetpts=N/SR/TB[mixfinal]"
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
        "anullsrc=r=48000:cl=stereo:d=3.000[silence];"
        "[mixfaded][silence]concat=n=2:v=0:a=1,asetpts=N/SR/TB[mixfinal]"
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


def test_mix_audio_runs_ffmpeg_and_returns_path(tmp_path, monkeypatch):
    from tenmin.atomic import part_path
    from tenmin.render import audio as audio_module

    seen: list[list[str]] = []

    def fake_run(args, **_):
        seen.append(list(args))
        # 真 ffmpeg 一定会把输出文件写出来；假的也得写，否则原子改名无从下手。
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(audio_module, "run", fake_run)
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

    def fake_run(args, *, ffmpeg="ffmpeg"):
        seen["ffmpeg"] = ffmpeg
        Path(args[-1]).write_bytes(b"\x00")
        return ""

    monkeypatch.setattr(audio_module, "run", fake_run)
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


# --- 产物原子写（P1-G 第 1 项）---------------------------------------------
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

    monkeypatch.setattr(audio_module, "run", fake_run)
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

    monkeypatch.setattr(audio_module, "run", fake_run)
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
