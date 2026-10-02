"""拿真 ffmpeg 验证 audio.py 的响度/时长相关逻辑，而不是只对着假的 stderr 文本测。

跟 test_render_audio.py 里那些纯图构造测试不同——这里有一部分刻意**不**
monkeypatch 任何东西，就是要拿真 ffmpeg 的真产物去验解析器与两遍响度归一。
没装 ffmpeg 的机器上跳过，不是失败：这条检查跟渲染成片（`render` marker）没
关系，不该被那个更重的标记盖住，也不该在没有 ffmpeg 时报错。

后半部分是 `mix_audio` 发布前几道守卫的失败注入测试——它们只关心「守卫触发时
`out_path` 有没有被碰」，不需要真的跑一遍编码，所以直接 monkeypatch `run` /
`run_with_progress` / `probe_duration`，不依赖本机是否装了 ffmpeg。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tenmin.models import Timeline, TimelineSegment, VoiceChunk, VoiceTrack
from tenmin.render.audio import _loudnorm_stats, _parse_ebur128_i, mix_audio
from tenmin.render.ffmpeg import FFmpegError, probe_duration, run


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("本机没有 ffmpeg，跳过（这组检查需要真实 ffmpeg 编码/测量）")


def test_ebur128_parser_reads_a_real_ffmpeg_report(tmp_path: Path) -> None:
    _require_ffmpeg()

    wav = tmp_path / "voice.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
            "-c:a", "pcm_s16le", str(wav),
        ],
        check=True,
    )

    stderr = run(
        ["-hide_banner", "-i", str(wav), "-af", "ebur128=peak=true", "-f", "null", "-"],
        ffmpeg="ffmpeg",
    )

    assert _parse_ebur128_i(stderr) is not None
    # 手造的静音场景不靠真 ffmpeg 跑——静音检测的深入覆盖留给后续接线，这里只是
    # 确认解析器本身认得 `-inf LUFS` 这种合法输出。
    assert _parse_ebur128_i("I: -inf LUFS") is None


# --- mix_audio 的双遍响度归一：真 ffmpeg 编码 + 真 ffmpeg 复测 -----------------


def _make_tone(path: Path, *, duration: float, frequency: float, volume: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency={frequency}:duration={duration}:sample_rate=48000",
            "-af", f"volume={volume}",
            "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
    )


def _make_silence(path: Path, *, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={duration}",
            "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
    )


def _single_segment_timeline(total_seconds: float) -> Timeline:
    return Timeline(
        episode=1,
        segments=[
            TimelineSegment(
                beat_id="b1",
                source_start=0.0,
                source_end=total_seconds,
                timeline_start=0.0,
                timeline_end=total_seconds,
            )
        ],
        subtitles=[],
        narration_offsets=[0.0],
        total_seconds=total_seconds,
    )


def _single_chunk_track(duration: float, chunk_path: str) -> VoiceTrack:
    return VoiceTrack(
        episode=1,
        chunks=[
            VoiceChunk(
                beat_id="b1", index=1, text="配音", path=chunk_path, duration=duration
            )
        ],
        total_seconds=duration,
    )


def test_mix_audio_normalizes_real_media_to_the_configured_loudness(tmp_path):
    """真跑一遍编码，再用一次独立的 loudnorm meter 复测产物：积分响度要落在
    配置目标附近，真峰值不能超过配置的天花板，时长要跟 timeline 声明的对上。
    """
    _require_ffmpeg()
    duration = 5.0
    video = tmp_path / "source.wav"
    _make_tone(video, duration=duration, frequency=220.0, volume=0.8)

    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    chunk_path = "chunk_001.wav"
    _make_tone(voice_dir / chunk_path, duration=duration, frequency=880.0, volume=0.9)

    timeline = _single_segment_timeline(duration)
    track = _single_chunk_track(duration, chunk_path)
    out_path = tmp_path / "out.m4a"
    warnings: list[str] = []

    mix_audio(
        video=video,
        timeline=timeline,
        track=track,
        voice_dir=voice_dir,
        out_path=out_path,
        duck_db=-12.0,
        fade_out_seconds=0.0,
        outro_seconds=0.0,
        warnings=warnings,
    )

    assert out_path.is_file()

    stderr = run(
        [
            "-hide_banner", "-i", str(out_path),
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        ffmpeg="ffmpeg",
    )
    stats = _loudnorm_stats(stderr)
    assert stats is not None
    # 合成正弦波达不到的目标响度就留不出 1 LUFS 的余量，所以给个比较宽的容差。
    assert stats["input_i"] == pytest.approx(-14.0, abs=1.0)
    # 合成正弦波动态范围很小，loudnorm 能把 TP 精确压到目标附近，这里比 mix_audio
    # 自己那道「TP 超限报 warning」的检查收得更紧（0.1dB）。
    assert stats["input_tp"] <= -1.4

    assert probe_duration(out_path) == pytest.approx(duration, abs=0.1)


def test_mix_audio_handles_fully_silent_material(tmp_path):
    """源声与旁白都是数字静音：首遍响度测不出但属于合法的「全静音」，不该拒绝
    发布，只该在 warnings 里留一条说明，照样把（静音的）产物编出来。"""
    _require_ffmpeg()
    duration = 3.0
    video = tmp_path / "source.wav"
    _make_silence(video, duration=duration)

    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    chunk_path = "chunk_001.wav"
    _make_silence(voice_dir / chunk_path, duration=duration)

    timeline = _single_segment_timeline(duration)
    track = _single_chunk_track(duration, chunk_path)
    out_path = tmp_path / "out.m4a"
    warnings: list[str] = []

    mix_audio(
        video=video,
        timeline=timeline,
        track=track,
        voice_dir=voice_dir,
        out_path=out_path,
        duck_db=-12.0,
        warnings=warnings,
    )

    assert out_path.is_file()
    assert any("静音" in w for w in warnings)
    assert probe_duration(out_path) == pytest.approx(duration, abs=0.1)


# --- 发布前守卫的失败注入：只关心 out_path 有没有被碰，不需要真的跑一遍编码 ---
# 这几条不需要本机装 ffmpeg——`run` / `run_with_progress` / `probe_duration`
# 全部被换成假的，mix_audio 自己不会再触到真正的 ffmpeg 二进制。

_FAKE_LOUDNORM_STDERR = (
    "[Parsed_loudnorm_0 @ 0x0]\n"
    "{\n"
    '"input_i" : "-20.00",\n'
    '"input_tp" : "-3.00",\n'
    '"input_lra" : "5.00",\n'
    '"input_thresh" : "-30.00",\n'
    '"output_i" : "-14.00",\n'
    '"output_tp" : "-1.50",\n'
    '"output_lra" : "5.00",\n'
    '"output_thresh" : "-24.00",\n'
    '"normalization_type" : "dynamic",\n'
    '"target_offset" : "0.10"\n'
    "}\n"
)


def _prepare_fake_inputs(tmp_path, *, duration: float = 1.0):
    video = tmp_path / "source.bin"
    video.write_bytes(b"fake source video")
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    chunk_path = "chunk_001.mp3"
    (voice_dir / chunk_path).write_bytes(b"fake chunk")
    timeline = _single_segment_timeline(duration)
    track = _single_chunk_track(duration, chunk_path)
    return video, voice_dir, timeline, track


def _seed_existing_output(tmp_path) -> Path:
    out_path = tmp_path / "06_audio" / "out.m4a"
    out_path.parent.mkdir(parents=True)
    out_path.write_bytes(b"pre-existing good audio")
    return out_path


def _fake_encode(args: list[str], **_) -> str:
    """假的 `run_with_progress`：不跑真编码，只把目标文件落地成非空占位符。"""
    Path(args[-1]).write_bytes(b"\x00")
    return ""


def _assert_output_untouched(out_path: Path, before_bytes: bytes, before_mtime: int) -> None:
    from tenmin.atomic import part_path

    assert out_path.read_bytes() == before_bytes
    assert out_path.stat().st_mtime_ns == before_mtime
    assert not part_path(out_path).exists()


def test_mix_audio_rejects_a_malformed_first_meter_report(tmp_path, monkeypatch):
    from tenmin.render import audio as audio_module

    video, voice_dir, timeline, track = _prepare_fake_inputs(tmp_path)
    out_path = _seed_existing_output(tmp_path)
    before_bytes = out_path.read_bytes()
    before_mtime = out_path.stat().st_mtime_ns

    monkeypatch.setattr(audio_module, "run", lambda *_a, **_k: "not loudnorm json")

    with pytest.raises(FFmpegError, match="首遍响度测量"):
        mix_audio(
            video=video, timeline=timeline, track=track, voice_dir=voice_dir,
            out_path=out_path, duck_db=-12.0,
        )
    _assert_output_untouched(out_path, before_bytes, before_mtime)


def test_mix_audio_rejects_when_input_mutates_mid_flight(tmp_path, monkeypatch):
    from tenmin.render import audio as audio_module

    video, voice_dir, timeline, track = _prepare_fake_inputs(tmp_path)
    out_path = _seed_existing_output(tmp_path)
    before_bytes = out_path.read_bytes()
    before_mtime = out_path.stat().st_mtime_ns

    def fake_run(args, **_):
        video.write_bytes(b"changed while we were measuring it")
        return _FAKE_LOUDNORM_STDERR

    monkeypatch.setattr(audio_module, "run", fake_run)

    with pytest.raises(FFmpegError, match=r"输入.*变化"):
        mix_audio(
            video=video, timeline=timeline, track=track, voice_dir=voice_dir,
            out_path=out_path, duck_db=-12.0,
        )
    _assert_output_untouched(out_path, before_bytes, before_mtime)


def test_mix_audio_rejects_a_malformed_final_report(tmp_path, monkeypatch):
    from tenmin.render import audio as audio_module

    video, voice_dir, timeline, track = _prepare_fake_inputs(tmp_path)
    out_path = _seed_existing_output(tmp_path)
    before_bytes = out_path.read_bytes()
    before_mtime = out_path.stat().st_mtime_ns

    reports = iter([_FAKE_LOUDNORM_STDERR, "not loudnorm json either"])
    monkeypatch.setattr(audio_module, "run", lambda *_a, **_k: next(reports))
    monkeypatch.setattr(
        audio_module, "run_with_progress",
        _fake_encode,
    )

    with pytest.raises(FFmpegError, match="编码后响度"):
        mix_audio(
            video=video, timeline=timeline, track=track, voice_dir=voice_dir,
            out_path=out_path, duck_db=-12.0,
        )
    _assert_output_untouched(out_path, before_bytes, before_mtime)


def test_mix_audio_warns_but_publishes_true_peak_over_the_configured_ceiling(
    tmp_path, monkeypatch
):
    """真峰值超过 loudness_tp + 容差只报 warning、照常发布：动态 loudnorm 在大动态
    范围的真实素材上压不住峰值，硬拒绝会让整集卡在 audio 阶段。warning 里要带实测值
    和上限，用户才知道超了多少、该不该去调 loudness_i。"""
    from tenmin.render import audio as audio_module

    video, voice_dir, timeline, track = _prepare_fake_inputs(tmp_path)
    out_path = _seed_existing_output(tmp_path)

    over_ceiling_report = _FAKE_LOUDNORM_STDERR.replace(
        '"input_tp" : "-3.00"', '"input_tp" : "-0.20"'
    )
    reports = iter([_FAKE_LOUDNORM_STDERR, over_ceiling_report])
    monkeypatch.setattr(audio_module, "run", lambda *_a, **_k: next(reports))
    monkeypatch.setattr(
        audio_module, "run_with_progress",
        _fake_encode,
    )
    monkeypatch.setattr(audio_module, "probe_duration", lambda *_a, **_k: 1.0)

    warnings: list[str] = []
    result = mix_audio(
        video=video, timeline=timeline, track=track, voice_dir=voice_dir,
        out_path=out_path, duck_db=-12.0, warnings=warnings,
    )

    assert result == out_path
    peak_notices = [w for w in warnings if "真峰值" in w]
    assert len(peak_notices) == 1
    assert "-0.2" in peak_notices[0]
    assert "-1.0" in peak_notices[0]


def test_mix_audio_accepts_real_world_dynamic_loudnorm_overshoot(tmp_path, monkeypatch):
    """真实番剧音轨（akujo E11，默认 I=-14/TP=-1.5）实测编码后真峰值 -1.21dBTP，
    比配置目标超出约 0.29dB——这是 ffmpeg loudnorm `linear=false` 动态模式在真实
    内容上的固有特性，改 loudness_tp/loudness_i 都压不下去（见
    `_ENCODED_TP_TOLERANCE` 的注释）。0.5dB 的容差必须放过这个真实超标幅度，
    不能把正常素材也当成坏产物拒绝。"""
    from tenmin.render import audio as audio_module

    video, voice_dir, timeline, track = _prepare_fake_inputs(tmp_path)
    out_path = _seed_existing_output(tmp_path)

    real_world_report = _FAKE_LOUDNORM_STDERR.replace(
        '"input_tp" : "-3.00"', '"input_tp" : "-1.21"'
    )
    reports = iter([_FAKE_LOUDNORM_STDERR, real_world_report])
    monkeypatch.setattr(audio_module, "run", lambda *_a, **_k: next(reports))
    monkeypatch.setattr(
        audio_module, "run_with_progress",
        _fake_encode,
    )
    monkeypatch.setattr(audio_module, "probe_duration", lambda *_a, **_k: 1.0)

    result = mix_audio(
        video=video, timeline=timeline, track=track, voice_dir=voice_dir,
        out_path=out_path, duck_db=-12.0,
    )
    assert result == out_path


def test_mix_audio_rejects_a_duration_mismatch_after_encoding(tmp_path, monkeypatch):
    from tenmin.render import audio as audio_module

    video, voice_dir, timeline, track = _prepare_fake_inputs(tmp_path, duration=1.0)
    out_path = _seed_existing_output(tmp_path)
    before_bytes = out_path.read_bytes()
    before_mtime = out_path.stat().st_mtime_ns

    monkeypatch.setattr(audio_module, "run", lambda *_a, **_k: _FAKE_LOUDNORM_STDERR)
    monkeypatch.setattr(
        audio_module, "run_with_progress",
        _fake_encode,
    )
    # timeline 声明的总长是 1.0 秒，探针却读出一个差得远的数字。
    monkeypatch.setattr(audio_module, "probe_duration", lambda *_a, **_k: 40.0)

    with pytest.raises(FFmpegError, match=r"音轨时长.*与时间轴不符"):
        mix_audio(
            video=video, timeline=timeline, track=track, voice_dir=voice_dir,
            out_path=out_path, duck_db=-12.0,
        )
    _assert_output_untouched(out_path, before_bytes, before_mtime)
