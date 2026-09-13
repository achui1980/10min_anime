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
    "[orig]volume='if(gt(between(t,0.000,8.000)+between(t,10.000,20.000)"
    "+between(t,20.000,30.000),0),0.2512,1.0000)':eval=frame[ducked];"
    "[1:a]adelay=delays=0:all=1[n0];"
    "[2:a]adelay=delays=10000:all=1[n1];"
    "[3:a]adelay=delays=20000:all=1[n2];"
    "[n0][n1][n2]amix=inputs=3:normalize=0[voice];"
    "[ducked][voice]amix=inputs=2:normalize=0[mix]"
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


def test_duck_volume_expr_lists_every_cue_window():
    cues = [SubtitleCue(start=0.0, end=8.0, text="a"), SubtitleCue(start=10.0, end=20.0, text="b")]
    assert duck_volume_expr(cues, 0.2512) == (
        "if(gt(between(t,0.000,8.000)+between(t,10.000,20.000),0),0.2512,1.0000)"
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
    assert args[-7:] == ["-map", "[mix]", "-c:a", "aac", "-b:a", "192k", str(out)]


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
    assert graph.endswith("[ducked][n0]amix=inputs=2:normalize=0[mix]")


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


def test_build_mix_args_without_fade_or_outro_keeps_mix_label(tmp_path):
    """没要求淡出/片尾时，-map 仍然是 [mix]，行为与老版本完全一致。"""
    args = build(tmp_path)
    assert args[args.index("-map") + 1] == "[mix]"


def test_build_mix_args_applies_fade_out_before_mix_ends(tmp_path):
    args = build(tmp_path, fade_out_seconds=5.0)
    graph = args[args.index("-filter_complex") + 1]
    assert graph.endswith(
        "[ducked][voice]amix=inputs=2:normalize=0[mix];"
        "[mix]afade=t=out:st=25.000:d=5.000[mixfaded]"
    )
    assert args[args.index("-map") + 1] == "[mixfaded]"


def test_build_mix_args_appends_silence_for_outro_card(tmp_path):
    args = build(tmp_path, fade_out_seconds=5.0, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert graph.endswith(
        "[mix]afade=t=out:st=25.000:d=5.000[mixfaded];"
        "anullsrc=r=48000:cl=stereo:d=3.000[silence];"
        "[mixfaded][silence]concat=n=2:v=0:a=1[mixfinal]"
    )
    assert args[args.index("-map") + 1] == "[mixfinal]"


def test_build_mix_args_outro_without_fade_concats_mix_directly(tmp_path):
    args = build(tmp_path, outro_seconds=3.0)
    graph = args[args.index("-filter_complex") + 1]
    assert graph.endswith(
        "[ducked][voice]amix=inputs=2:normalize=0[mix];"
        "anullsrc=r=48000:cl=stereo:d=3.000[silence];"
        "[mix][silence]concat=n=2:v=0:a=1[mixfinal]"
    )
    assert args[args.index("-map") + 1] == "[mixfinal]"


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
