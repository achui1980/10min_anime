"""截图匹配的纯函数契约，不依赖视频或文件 IO。"""

import shlex
import subprocess

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
