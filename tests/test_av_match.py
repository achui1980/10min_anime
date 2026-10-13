"""截图匹配的计算、IO 契约与显式开启的真实媒体测试。"""

import os
import shlex
import shutil
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from tenmin.ingest import av_match
from tenmin.ingest.av_match import (
    HASH_BITS,
    HASH_BYTES,
    HASH_HEIGHT,
    HASH_WIDTH,
    MAX_DISTANCE,
    best_frame,
    dhash,
    hamming,
)
from tenmin.render import ffmpeg


def test_dhash_falling_rows_are_all_ones():
    assert (HASH_WIDTH, HASH_HEIGHT, HASH_BYTES, HASH_BITS, MAX_DISTANCE) == (9, 8, 72, 64, 12)
    gray = np.tile(np.arange(9, 0, -1), (8, 1))
    assert dhash(gray) == (1 << 64) - 1


def test_dhash_rising_rows_are_zero():
    gray = np.tile(np.arange(9), (8, 1))
    assert dhash(gray) == 0


def test_dhash_bytes_match_array():
    gray = np.random.default_rng(1).integers(30, 201, size=(8, 9), dtype=np.uint8)
    expected = dhash(gray)
    assert dhash(gray.tobytes()) == expected
    assert dhash(bytearray(gray.tobytes())) == expected
    assert dhash(gray.ravel().tolist()) == expected


def test_dhash_rejects_wrong_size():
    with pytest.raises(ValueError, match="72"):
        dhash(bytes(71))


def test_dhash_tolerates_brightness_and_contrast_changes():
    gray = np.random.default_rng(1).integers(30, 201, size=(8, 9))
    reference = dhash(gray)
    assert dhash(gray + 40) == reference
    assert hamming(dhash(np.round(gray * 0.6 + 20)), reference) <= 2


def test_dhash_unrelated_frames_exceed_threshold():
    reference = dhash(np.random.default_rng(10).integers(30, 201, size=(8, 9)))
    for seed in range(11, 21):
        gray = np.random.default_rng(seed).integers(30, 201, size=(8, 9))
        assert hamming(dhash(gray), reference) > MAX_DISTANCE


def test_hamming_counts_differing_bits():
    assert hamming(0b1011, 0b0001) == 2
    assert hamming(0, (1 << 64) - 1) == 64


def test_best_frame_keeps_first_tie_and_reports_distance_above_threshold():
    assert best_frame([15, 1, 3, 1], 0) == (1, 1)
    assert best_frame([(1 << 13) - 1], 0) == (0, 13)


def test_best_frame_rejects_empty_sequence():
    with pytest.raises(ValueError, match="没有可比较的帧"):
        best_frame([], 0)


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def test_audio_envelope_has_ten_frames_per_second_and_drops_the_tail():
    pcm = _pcm(np.zeros(8000 * 3 + 123)) + b"\xff"

    envelope = av_match.audio_envelope(pcm)

    assert envelope.shape == (30, 16)
    assert np.all(envelope == 0)


def test_audio_envelope_puts_a_tone_in_the_right_band():
    t = np.arange(8000 * 2) / 8000
    pcm = _pcm(8000 * np.sin(2 * np.pi * 440 * t))

    envelope = av_match.audio_envelope(pcm)

    # 440 Hz 是第 44 个 bin，落在 [40, 53) 的频带。
    assert int(np.argmax(envelope.mean(axis=0))) == 8


def test_audio_envelope_of_empty_input_is_empty():
    assert av_match.audio_envelope(b"").shape == (0, 16)


def _noise_envelope(seconds: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(5.0, 1.0, size=(round(seconds * 10), 16))


def _episode(op: np.ndarray, seconds: float, seed: int, *, lead_frames=0, gain=0.0):
    """先放少量前导帧，再共享 OP，剩余部分换成本集独立的噪声。"""
    frames = round(seconds * 10)
    rest = _noise_envelope(150.0, seed)
    return np.concatenate([rest[:lead_frames], op[:frames] + gain, rest[lead_frames + frames :]])[
        : len(op)
    ]


def test_op_length_finds_where_the_shared_audio_ends():
    ref = _noise_envelope(150.0, 0)
    other = _episode(ref, 90.0, seed=1)

    assert av_match.op_length(ref, other) == pytest.approx(90.0, abs=0.2)


def test_op_length_ignores_a_loudness_difference():
    ref = _noise_envelope(150.0, 0)
    other = _episode(ref, 72.0, seed=2, gain=1.5)

    assert av_match.op_length(ref, other) == pytest.approx(72.0, abs=0.2)


def test_op_length_absorbs_a_small_start_offset():
    ref = _noise_envelope(150.0, 0)
    other = _episode(ref, 85.0, seed=3, lead_frames=4)

    assert av_match.op_length(ref, other) == pytest.approx(85.4, abs=0.2)


def test_op_length_ignores_a_short_dip_inside_the_op():
    ref = _noise_envelope(150.0, 0)
    other = _episode(ref, 90.0, seed=4)
    other[300:310] = _noise_envelope(1.0, 5)

    assert av_match.op_length(ref, other) == pytest.approx(90.0, abs=0.2)


def test_op_length_is_none_when_the_two_never_line_up():
    ref = _noise_envelope(150.0, 0)
    other = _noise_envelope(150.0, 6)

    assert av_match.op_length(ref, other) is None


def test_op_length_is_none_when_the_op_outlasts_the_reference_audio():
    ref = _noise_envelope(150.0, 0)

    assert av_match.op_length(ref, ref.copy()) is None


def test_ed_endpoints_confirm_matching_audio_through_eof_beyond_150_seconds():
    song = _noise_envelope(170, 10)
    assert av_match.ed_endpoints(song, song + 1.5, 170, 170) == (170, 170)


def test_ed_endpoints_exclude_episode_specific_postcredits_with_start_offset():
    song = _noise_envelope(90, 10)
    ref = np.concatenate([song, _noise_envelope(30, 11)])
    peer = np.concatenate([_noise_envelope(0.4, 12), song, _noise_envelope(60, 13)])
    ends = av_match.ed_endpoints(ref, peer, 120, 150.4)
    assert ends == pytest.approx((90, 90.4), abs=0.2)


@pytest.mark.parametrize("kind", ["unrelated", "silence", "brief-match"])
def test_ed_endpoints_need_sustained_shared_audio(kind):
    ref = _noise_envelope(90, 10)
    peer = _noise_envelope(90, 11)
    if kind == "silence":
        ref[:] = peer[:] = 0
    elif kind == "brief-match":
        peer[:10] = ref[:10]
    assert av_match.ed_endpoints(ref, peer, 90, 90) == (None, None)


@pytest.mark.parametrize("ref_seconds,peer_seconds", [(60, 100), (100, 60)])
def test_ed_endpoints_short_file_eof_does_not_confirm_longer_peer(ref_seconds, peer_seconds):
    song = _noise_envelope(100, 10)
    ends = av_match.ed_endpoints(
        song[:ref_seconds * 10], song[:peer_seconds * 10], ref_seconds, peer_seconds
    )
    assert ends == ((60, None) if ref_seconds == 60 else (None, 60))


def test_ed_endpoints_read_cutoff_is_not_file_eof():
    song = _noise_envelope(150, 10)
    assert av_match.ed_endpoints(song, song.copy(), 180, 180) == (None, None)


def test_ed_endpoints_partial_drop_at_eof_is_not_confirmed():
    ref = _noise_envelope(90, 10)
    peer = ref.copy()
    peer[-10:] = _noise_envelope(1, 11)
    assert av_match.ed_endpoints(ref, peer, 90, 90) == (None, None)


@pytest.mark.parametrize("tail_seconds", [0, 10], ids=["silence-to-eof", "audio-after-silence"])
@pytest.mark.parametrize("shared_music", [False, True], ids=["unrelated", "shared-music"])
def test_ed_pcm_shared_silence_requires_preceding_music_evidence(tail_seconds, shared_music):
    rate = av_match.SAMPLE_RATE
    music = np.random.default_rng(20).normal(0, 5000, 20 * rate)
    other_music = music if shared_music else np.random.default_rng(21).normal(
        0, 5000, 20 * rate
    )
    silence = np.zeros(10 * rate)
    ref = av_match.audio_envelope(_pcm(np.concatenate([
        music, silence, np.random.default_rng(22).normal(0, 5000, tail_seconds * rate)
    ])))
    peer = av_match.audio_envelope(_pcm(np.concatenate([
        other_music, silence, np.random.default_rng(23).normal(0, 5000, tail_seconds * rate)
    ])))
    assert np.all(ref[200:300] == 0)
    assert np.all(peer[200:300] == 0)
    ends = av_match.ed_endpoints(ref, peer, 30 + tail_seconds, 30 + tail_seconds)
    if shared_music:
        assert ends == pytest.approx((20, 20), abs=0.6)
    else:
        assert ends == (None, None)


def test_ed_pcm_stationary_band_profile_cannot_prove_shared_song():
    rate = av_match.SAMPLE_RATE
    # 整数周期重复的非静音 PCM：频带间有变化，时间轴却没有有效变化。
    frame = _pcm(8000 * np.sin(2 * np.pi * 400 * np.arange(800) / rate))
    stationary = av_match.audio_envelope(frame * 100)
    ref = np.concatenate([stationary, av_match.audio_envelope(_pcm(
        np.random.default_rng(24).normal(0, 5000, 10 * rate)
    ))])
    peer = np.concatenate([stationary, av_match.audio_envelope(_pcm(
        np.random.default_rng(25).normal(0, 5000, 10 * rate)
    ))])
    assert av_match.ed_endpoints(ref, peer, 20, 20) == (None, None)


def test_median_length_drops_outliers_before_taking_the_median():
    assert av_match.median_length([89.9, 90.1, 90.0, 60.0, 130.0]) == pytest.approx(90.0)


def test_median_length_keeps_values_within_ten_seconds():
    assert av_match.median_length([80.0, 90.0, 99.0]) == pytest.approx(90.0)


def test_median_length_of_nothing_is_none():
    assert av_match.median_length([]) is None


def test_median_length_is_none_when_everything_is_an_outlier():
    assert av_match.median_length([60.0, 90.0]) is None


def test_run_pcm_returns_binary_stdout_with_exact_command(tmp_path, monkeypatch):
    video = tmp_path / "源 视频.mkv"
    video.touch()
    pcm = b"\x00\xff\x80\x01"
    binaries = []

    def which(binary):
        binaries.append(binary)
        return binary

    def run(argv, **kwargs):
        assert binaries == ["customff"]
        assert argv == [
            "customff",
            "-nostdin",
            "-nostats",
            "-hide_banner",
            "-v",
            "error",
            "-ss",
            "1.500",
            "-t",
            "2.000",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "s16le",
            "-c:a",
            "pcm_s16le",
            "-",
        ]
        assert kwargs == {"capture_output": True, "stdin": subprocess.DEVNULL}
        return subprocess.CompletedProcess(argv, 0, stdout=pcm, stderr=b"")

    monkeypatch.setattr(ffmpeg.shutil, "which", which)
    monkeypatch.setattr(ffmpeg.subprocess, "run", run)

    result = ffmpeg.run_pcm(video, start=1.5, duration=2.0, sample_rate=8000, ffmpeg="customff")

    assert isinstance(result, bytes)
    assert result == pcm


def test_run_pcm_nonzero_decodes_chinese_stderr_and_reports_command(tmp_path, monkeypatch):
    video = tmp_path / "源 视频.mkv"
    video.touch()
    commands = []
    stderr = "\n".join(["旧日志"] * ffmpeg.STDERR_TAIL_LINES + ["无法解码音频"])

    def run(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 7, stdout=b"", stderr=stderr.encode("utf-8"))

    monkeypatch.setattr(ffmpeg.shutil, "which", lambda binary: binary)
    monkeypatch.setattr(ffmpeg.subprocess, "run", run)

    with pytest.raises(ffmpeg.FFmpegError) as error:
        ffmpeg.run_pcm(video, start=1.5, duration=2.0, sample_rate=8000)

    message = str(error.value)
    assert "退出码 7" in message
    assert shlex.join(commands[0]) in message
    assert f"stderr 末尾 {ffmpeg.STDERR_TAIL_LINES} 行" in message
    assert message.endswith(ffmpeg.tail(stderr))
    assert message.count("旧日志") == ffmpeg.STDERR_TAIL_LINES - 1
    assert "无法解码音频" in message


def test_run_pcm_missing_video_uses_chinese_helper_error(tmp_path, monkeypatch):
    video = tmp_path / "missing.mkv"
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda binary: binary)
    monkeypatch.setattr(ffmpeg.subprocess, "run", lambda *args, **kwargs: pytest.fail("spawned"))

    with pytest.raises(ffmpeg.FFmpegError, match="找不到源视频") as error:
        ffmpeg.run_pcm(video, start=0.0, duration=2.0, sample_rate=8000)

    assert str(error.value) == f"找不到源视频: {video}"


def test_run_pcm_missing_binary_fails_before_spawning(tmp_path, monkeypatch):
    video = tmp_path / "video.mkv"
    video.touch()
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda binary: None)
    monkeypatch.setattr(ffmpeg.subprocess, "run", lambda *args, **kwargs: pytest.fail("spawned"))

    with pytest.raises(ffmpeg.FFmpegBinaryError, match="找不到可执行文件 'customff'"):
        ffmpeg.run_pcm(video, start=0.0, duration=2.0, sample_rate=8000, ffmpeg="customff")


def test_sample_hashes_uses_showinfo_pts_and_exact_filter_args(monkeypatch):
    frames = [bytes(72), bytes(range(72)), bytes(reversed(range(72)))]

    def run(args, *, frame_size, on_frame, ffmpeg):
        assert ffmpeg == "customff"
        assert frame_size == HASH_BYTES
        assert args == [
            "-hide_banner",
            "-v",
            "info",
            "-ss",
            "10.000",
            "-t",
            "2.000",
            "-i",
            "video.mkv",
            "-map",
            "0:v:0",
            "-vf",
            "fps=2,scale=9:8:flags=area,format=gray,showinfo",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
        for frame in frames:
            on_frame(frame)
        return "unrelated pts_time:99\n" + "\n".join(
            f"[Parsed_showinfo_3 @ abc] n: {i:3d} pts: {i} pts_time:{pts} fmt:gray"
            for i, pts in enumerate(["0", "5e-1", "1"])
        )

    monkeypatch.setattr(ffmpeg, "run_raw_frames", run)
    assert av_match.sample_hashes(Path("video.mkv"), 10, 12, 2, ffmpeg_path="customff") == [
        (10 + i * 0.5, dhash(frame)) for i, frame in enumerate(frames)
    ]


@pytest.mark.parametrize("pts", [[], ["0", "0.5"], ["bad"]])
def test_sample_hashes_rejects_timestamp_mismatch_or_invalid_value(monkeypatch, pts):
    def run(args, *, frame_size, on_frame, ffmpeg):
        on_frame(bytes(HASH_BYTES))
        return "\n".join(
            f"[Parsed_showinfo_3 @ abc] n: {i} pts: {i} pts_time:{value} fmt:gray"
            for i, value in enumerate(pts)
        )

    monkeypatch.setattr(ffmpeg, "run_raw_frames", run)
    with pytest.raises(ffmpeg.FFmpegError):
        av_match.sample_hashes(Path("video.mkv"), 0, 2, 2)


@pytest.mark.parametrize("end", [10, 9])
def test_sample_hashes_empty_interval_does_no_io(monkeypatch, end):
    monkeypatch.setattr(ffmpeg, "run_raw_frames", lambda *a, **kw: pytest.fail("IO"))
    assert av_match.sample_hashes(Path("video.mkv"), 10, end, 2) == []


def test_sample_hashes_filters_fractional_window_without_losing_hash_alignment(monkeypatch):
    pts = [-0.1, 0, 0.05, 0.09, 0.1, 0.2]
    frames = [bytes([i]) * HASH_BYTES for i in range(len(pts))]
    frames[2] = bytes(reversed(range(HASH_BYTES)))

    def run(args, *, frame_size, on_frame, ffmpeg):
        for frame in frames:
            on_frame(frame)
        return "\n".join(
            f"[Parsed_showinfo_3 @ abc] n: {i} pts: {i} pts_time:{value} fmt:gray"
            for i, value in enumerate(pts)
        )

    monkeypatch.setattr(ffmpeg, "run_raw_frames", run)
    assert av_match.sample_hashes(Path("video.mkv"), 1.31, 1.40, 10) == [
        (1.31, dhash(frames[1])),
        (1.36, dhash(frames[2])),
    ]


def test_sample_hashes_validates_count_before_filtering_outside_window(monkeypatch):
    monkeypatch.setattr(
        ffmpeg,
        "run_raw_frames",
        lambda *a, **kw: "[Parsed_showinfo_3 @ abc] n: 0 pts: 1 pts_time:0.1 fmt:gray",
    )
    with pytest.raises(ffmpeg.FFmpegError, match="帧数 0 与时间戳数 1 不一致"):
        av_match.sample_hashes(Path("video.mkv"), 1.31, 1.40, 10)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_image_hash_requires_one_frame_and_shared_scale_chain(monkeypatch, count):
    frame = bytes(reversed(range(HASH_BYTES)))

    def run(args, *, frame_size, on_frame, ffmpeg):
        assert ffmpeg == "customff"
        assert frame_size == HASH_BYTES
        assert args[args.index("-i") + 1] == "image.png"
        assert args[args.index("-vf") + 1] == "scale=9:8:flags=area,format=gray"
        assert args[args.index("-frames:v") + 1] == "1"
        assert args[-5:] == ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
        for _ in range(count):
            on_frame(frame)
        return "image diagnostics"

    monkeypatch.setattr(ffmpeg, "run_raw_frames", run)
    if count == 1:
        assert av_match.image_hash(Path("image.png"), ffmpeg_path="customff") == dhash(frame)
    else:
        with pytest.raises(ffmpeg.FFmpegError, match="image diagnostics"):
            av_match.image_hash(Path("image.png"), ffmpeg_path="customff")


def test_read_pcm_requests_eight_khz_and_passes_through_bytes(monkeypatch):
    def run(video, *, start, duration, sample_rate, ffmpeg):
        assert video == Path("video.mkv")
        assert (start, duration, sample_rate, ffmpeg) == (1, 2, 8000, "customff")
        return b"\x00\xff"

    monkeypatch.setattr(ffmpeg, "run_pcm", run)
    assert av_match.read_pcm(Path("video.mkv"), 1, 2, ffmpeg_path="customff") == b"\x00\xff"


def test_locate_recovers_earliest_start_of_static_match(monkeypatch):
    calls = []

    def sample(video, start, end, fps, *, ffmpeg_path):
        assert (video, ffmpeg_path) == (Path("video.mkv"), "customff")
        calls.append((start, end, fps))
        times = np.round(np.arange(start, end, 1 / fps), 3)
        return [(float(t), 0 if 12.3 <= t < 15.3 else 65535) for t in times]

    monkeypatch.setattr(av_match, "sample_hashes", sample, raising=False)
    match = av_match.locate(Path("video.mkv"), 0, start=10, end=20, ffmpeg_path="customff")
    assert match.time == pytest.approx(12.3)
    assert match.distance == 0
    assert calls == [(10, 20, 2), (11.5, 13.5, 10)]
    with pytest.raises(FrozenInstanceError):
        match.time = 99


def test_locate_empty_coarse_sampling_returns_none(monkeypatch):
    calls = []

    def sample(*args, **kwargs):
        calls.append(args)
        return []

    monkeypatch.setattr(av_match, "sample_hashes", sample, raising=False)
    assert av_match.locate(Path("video.mkv"), 0, start=0, end=6) is None
    assert len(calls) == 1


@pytest.mark.parametrize("center,window", [(0, (0, 1)), (5.5, (4.5, 6))])
def test_locate_clamps_fine_window_and_retains_coarse_best(monkeypatch, center, window):
    def sample(video, start, end, fps, *, ffmpeg_path):
        if fps == 2:
            return [(center, (1 << 13) - 1)]
        assert (start, end, fps) == (*window, 10)
        return [(start, (1 << 14) - 1)]

    monkeypatch.setattr(av_match, "sample_hashes", sample, raising=False)
    match = av_match.locate(Path("video.mkv"), 0, start=0, end=6)
    assert (match.time, match.distance) == (center, 13)


@pytest.fixture
def av_video(tmp_path) -> Path:
    if shutil.which(ffmpeg.FFMPEG) is None:
        pytest.skip("没有 ffmpeg")
    video = tmp_path / "av.mkv"
    subprocess.run(
        [
            ffmpeg.FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=24:duration=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=6",
            "-vf",
            "rotate=a=PI*t/3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    return video


@pytest.mark.render
def test_real_sample_hashes_reports_absolute_timestamps(av_video):
    samples = av_match.sample_hashes(av_video, 1, 5, 2)
    assert len(samples) == 8
    assert [time for time, _ in samples] == pytest.approx([1 + 0.5 * i for i in range(8)], abs=0.05)


@pytest.mark.render
def test_real_sample_hashes_excludes_fps_rounding_past_fractional_window(av_video, tmp_path):
    video = tmp_path / "five-fps.mkv"
    subprocess.run(
        [
            ffmpeg.FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-i",
            str(av_video),
            "-vf",
            "fps=5",
            "-an",
            "-c:v",
            "libx264",
            str(video),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    samples = av_match.sample_hashes(video, 1.31, 1.40, 10)
    assert all(1.31 <= time < 1.40 for time, _ in samples)
    assert samples == []


@pytest.mark.render
def test_real_screenshot_locates_its_video_frame(av_video, tmp_path):
    image = tmp_path / "frame.png"
    subprocess.run(
        [
            ffmpeg.FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-ss",
            "3",
            "-i",
            str(av_video),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            str(image),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    match = av_match.locate(av_video, av_match.image_hash(image), start=0, end=6)
    assert match is not None
    assert match.time == pytest.approx(3, abs=0.1)
    assert match.distance <= MAX_DISTANCE


@pytest.mark.render
def test_real_corrupt_image_raises_ffmpeg_error(av_video, tmp_path):
    image = tmp_path / "broken.png"
    image.write_bytes(b"not an image")
    with pytest.raises(ffmpeg.FFmpegError):
        av_match.image_hash(image)


@pytest.mark.render
def test_real_read_pcm_has_expected_duration_and_tone_band(av_video):
    pcm = av_match.read_pcm(av_video, 1, 2)
    assert len(pcm) == pytest.approx(32000, abs=160)
    assert int(np.argmax(av_match.audio_envelope(pcm).mean(axis=0))) == 8


# --- 真实素材（av marker，默认跳过） -------------------------------------------


@pytest.mark.parametrize("defect", [None, "fallback", "ed_fallback", "missing_episode"])
async def test_real_season_contract_rejects_false_success(tmp_path, monkeypatch, defect):
    """直接走真实素材测试的断言；只替换解码结果，扫描与集号解析仍用真实文件。"""
    from tenmin import importer

    sample = tmp_path / "sample"
    sample.mkdir()
    for number in (2, 7, 12):
        (sample / f"Show E{number:02d}.mp4").touch()
        (sample / f"Show E{number:02d}.srt").touch()
    (sample / "op.png").touch()
    (sample / "ed.png").touch()
    pairs = importer.scan_directory(sample).pairs
    ranges = importer.RangePreview(
        duration=1420.0,
        op_range=(10.0, 100.0),
        ed_range=(1300.0, 1420.0),
        write=((10.0, 100.0), (1300.0, 1420.0)),
        op_source="av",
        ed_source="av",
        op_distance=0,
        ed_distance=0,
    )
    plan = importer.ImportPlan(
        items=[
            importer.ImportItem(number=number, pair=pair, prefilled=False, ranges=ranges)
            for number, pair in zip((2, 7, 12), pairs, strict=True)
        ]
    )
    if defect == "fallback":
        plan.items[0] = replace(plan.items[0], ranges=replace(ranges, op_length_estimated=True))
    elif defect == "ed_fallback":
        plan.items[0] = replace(plan.items[0], ranges=replace(ranges, ed_length_estimated=True))
    elif defect == "missing_episode":
        plan.items.pop()

    async def decoded_plan(cfg, video_dir, *, provider_factory):
        assert cfg.episodes == []
        assert video_dir == sample
        return plan

    monkeypatch.setenv("TENMIN_AV_SAMPLE_DIR", str(sample))
    monkeypatch.setattr(importer, "build_import_plan", decoded_plan)

    if defect is None:
        await test_real_season_locates_op_and_ed_by_screenshot_and_audio(tmp_path)
    else:
        message = {
            "fallback": "OP 长度不能使用估算兜底",
            "ed_fallback": "ED 终点不能使用估算兜底",
            "missing_episode": "导入集号未完整覆盖素材",
        }[defect]
        with pytest.raises(AssertionError, match=message):
            await test_real_season_locates_op_and_ed_by_screenshot_and_audio(tmp_path)


@pytest.mark.av
@pytest.mark.skipif(
    not os.environ.get("TENMIN_AV_SAMPLE_DIR"), reason="需要 TENMIN_AV_SAMPLE_DIR 指向素材目录"
)
async def test_real_season_locates_op_and_ed_by_screenshot_and_audio(tmp_path):
    from tenmin.config import load_project
    from tenmin.importer import (
        build_import_plan,
        is_extra,
        parse_episode_number,
        scan_directory,
    )

    sample = Path(os.environ["TENMIN_AV_SAMPLE_DIR"])
    scan = scan_directory(sample)
    assert not scan.unpaired_videos, scan.unpaired_videos
    assert not scan.unpaired_srts, scan.unpaired_srts
    assert set(scan.references) == {"op", "ed"}, "素材目录必须提供 OP 与 ED 截图"
    expected_numbers = []
    for pair in scan.pairs:
        assert not is_extra(pair.video.stem), f"素材目录不能包含非正片：{pair.video.name}"
        assert not pair.ignored_srts, pair.ignored_srts
        number = parse_episode_number(pair.video.stem)
        assert isinstance(number, int) and number > 0, f"素材集号必须可解析：{pair.video.name}"
        expected_numbers.append(number)
    assert len(expected_numbers) >= 2, "跨集音频匹配至少需要两集素材"
    assert len(set(expected_numbers)) == len(expected_numbers), "素材集号必须唯一"

    root = tmp_path / "work" / "sample"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: sample\nslug: sample\nepisodes: []\n", encoding="utf-8"
    )
    cfg = load_project(root / "project.yaml")

    def no_llm():
        raise RuntimeError("真实素材测试不调 LLM")

    plan = await build_import_plan(cfg, sample, provider_factory=no_llm)

    assert not plan.failed, plan.failed
    assert not plan.ambiguous, plan.ambiguous
    assert not plan.registered, plan.registered
    assert not plan.extras, plan.extras
    assert not plan.unpaired_videos, plan.unpaired_videos
    assert not plan.unpaired_srts, plan.unpaired_srts
    assert sorted(item.number for item in plan.items) == sorted(expected_numbers), (
        "导入集号未完整覆盖素材"
    )
    for item in plan.items:
        ranges = item.ranges
        assert ranges.op_length_estimated is False, "OP 长度不能使用估算兜底"
        assert ranges.ed_length_estimated is False, "ED 终点不能使用估算兜底"
        assert (ranges.op_source, ranges.ed_source) == ("av", "av"), item.number
        assert ranges.op_distance is not None and ranges.op_distance <= MAX_DISTANCE
        assert ranges.ed_distance is not None and ranges.ed_distance <= MAX_DISTANCE
        assert ranges.op_range is not None and ranges.ed_range is not None
        assert 60.0 <= ranges.op_range[1] - ranges.op_range[0] <= 120.0
        assert 0 < ranges.ed_range[0] < ranges.ed_range[1] <= ranges.duration
        # 本验收素材 saijo E02/E12 的 ED 歌曲确实唱到 EOF，必须有音频证据。
        if item.number in (2, 12):
            assert ranges.ed_range[1] == ranges.duration
        assert ranges.write is not None
