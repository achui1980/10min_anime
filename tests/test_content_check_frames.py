from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tenmin.content_check import frames
from tenmin.content_check.models import ContentCheckError, SampledFrame
from tenmin.render.ffmpeg import FFmpegError


def pts_log(*values: str) -> str:
    return "\n".join(
        f"[Parsed_showinfo_3 @ a] n: {index} pts: {index} pts_time:{value} "
        "duration:1 duration_time:0.04 fmt:yuvj420p"
        for index, value in enumerate(values)
    )


def test_parse_real_pts():
    stderr = (
        "[Parsed_showinfo_2 @ a] n: 0 pts: 0 pts_time:0.0213333\n"
        "[Parsed_showinfo_2 @ a] n: 1 pts: 9 pts_time:1.187\n"
    )
    assert frames.parse_pts(stderr) == [.0213333, 1.187]


def test_filter_samples_first_frame_and_preserves_display_ratio():
    assert frames.frame_filter(1) == (
        "select='isnan(prev_selected_t)+gte(t-prev_selected_t,1)',"
        "scale=640:trunc(640/dar/2)*2,setsar=1,showinfo"
    )
    assert "gte(t-prev_selected_t,0.75)" in frames.frame_filter(.75)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_filter_rejects_invalid_interval(interval):
    with pytest.raises(ContentCheckError, match="interval"):
        frames.frame_filter(interval)


def test_parse_ignores_unrelated_logs_and_accepts_scientific_notation():
    assert frames.parse_pts(
        "input metadata pts_time:99\n"
        "[Parsed_showinfo_3 @ a] config in time_base: 1/1000\n"
        + pts_log("2.13333e-2", "1.187")
    ) == [.0213333, 1.187]
    assert frames.parse_pts("no frames") == []


@pytest.mark.parametrize("values", [
    ("1.1", "1.0"), ("-0.01",), ("nan",), ("inf",), ("-inf",),
    ("N/A",), ("1e999",), ("broken",),
])
def test_parse_rejects_invalid_pts(values):
    with pytest.raises(ContentCheckError, match="PTS"):
        frames.parse_pts(pts_log(*values))


def test_parse_rejects_frame_log_without_pts():
    with pytest.raises(ContentCheckError, match="PTS"):
        frames.parse_pts("[Parsed_showinfo_3 @ a] n: 0 pts: 0 fmt:yuvj420p")


def test_parse_keeps_equal_pts_in_order():
    assert frames.parse_pts(pts_log("0.02", "0.02")) == [.02, .02]


@pytest.fixture
def readable_video(monkeypatch, tmp_path):
    video = tmp_path / "input with spaces.mp4"
    video.write_bytes(b"fake video")
    calls = []

    def fake_probe(path, *, ffprobe):
        calls.append((path, ffprobe))
        return 2.5

    monkeypatch.setattr(frames, "probe_duration", fake_probe)
    return video, calls


def test_extract_pairs_sorted_files_with_real_pts(monkeypatch, tmp_path, readable_video):
    video, probes = readable_video
    destination = tmp_path / "temporary frames with spaces"
    calls = []

    def fake_run(args, *, ffmpeg):
        assert probes == [(video, "custom ffprobe")]
        assert destination.is_dir()
        calls.append((args, ffmpeg))
        # Directory iteration order must not determine timestamp assignment.
        (destination / "frame_000002.jpg").write_bytes(b"second jpeg")
        (destination / "frame_000001.jpg").write_bytes(b"first jpeg")
        return pts_log("0.0213333", "1.187")

    monkeypatch.setattr(frames, "run", fake_run)
    result = frames.extract_frames(
        video, destination, interval=1, ffmpeg="custom ffmpeg", ffprobe="custom ffprobe",
    )
    assert result == [
        SampledFrame(destination / "frame_000001.jpg", .0213333),
        SampledFrame(destination / "frame_000002.jpg", 1.187),
    ]
    assert calls == [([
        "-hide_banner", "-nostdin", "-y", "-i", str(video),
        "-map", "0:v:0", "-an", "-vf", frames.frame_filter(1),
        "-fps_mode", "vfr", "-q:v", "2", str(destination / "frame_%06d.jpg"),
    ], "custom ffmpeg")]


@pytest.mark.parametrize(("file_count", "values"), [
    (0, ()), (0, ("0",)), (2, ("0",)), (1, ("0", "1")),
])
def test_extract_rejects_zero_frames_and_pts_mismatch(
    monkeypatch, tmp_path, readable_video, file_count, values,
):
    video, _ = readable_video
    destination = tmp_path / "frames"

    def fake_run(args, *, ffmpeg):
        for index in range(1, file_count + 1):
            (destination / f"frame_{index:06d}.jpg").write_bytes(b"jpeg")
        return pts_log(*values)

    monkeypatch.setattr(frames, "run", fake_run)
    with pytest.raises(ContentCheckError, match="帧"):
        frames.extract_frames(video, destination, interval=1, ffmpeg="ff", ffprobe="fp")


@pytest.mark.parametrize("where", ["probe", "run"])
def test_extract_wraps_ffmpeg_errors_with_diagnostic_and_cause(
    monkeypatch, tmp_path, readable_video, where,
):
    video, _ = readable_video
    destination = tmp_path / "frames"
    error = FFmpegError("decoder failed: diagnostic stderr")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(frames, "probe_duration" if where == "probe" else "run", fail)
    with pytest.raises(ContentCheckError, match="diagnostic stderr") as caught:
        frames.extract_frames(video, destination, interval=1, ffmpeg="ff", ffprobe="fp")
    assert caught.value.__cause__ is error
    if where == "probe":
        assert not destination.exists()


def test_extract_wraps_missing_video(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing source.mp4")

    monkeypatch.setattr(frames, "probe_duration", missing)
    with pytest.raises(ContentCheckError, match=r"missing source\.mp4"):
        frames.extract_frames(
            tmp_path / "source.mp4", tmp_path / "frames",
            interval=1, ffmpeg="ff", ffprobe="fp",
        )


def test_extract_refuses_nonempty_destination_without_touching_files(
    monkeypatch, tmp_path, readable_video,
):
    video, _ = readable_video
    destination = tmp_path / "frames"
    destination.mkdir()
    old = destination / "frame_000001.jpg"
    old.write_bytes(b"user image")

    def unexpected(*args, **kwargs):
        pytest.fail("must not run ffmpeg in a nonempty destination")

    monkeypatch.setattr(frames, "run", unexpected)
    with pytest.raises(ContentCheckError, match="空"):
        frames.extract_frames(video, destination, interval=1, ffmpeg="ff", ffprobe="fp")
    assert old.read_bytes() == b"user image"


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), 0, -1])
def test_extract_rejects_unusable_probed_duration(monkeypatch, tmp_path, duration):
    monkeypatch.setattr(frames, "probe_duration", lambda *args, **kwargs: duration)
    with pytest.raises(ContentCheckError, match="时长"):
        frames.extract_frames(
            tmp_path / "video.mp4", tmp_path / "frames",
            interval=1, ffmpeg="ff", ffprobe="fp",
        )


def require_display_rotation(ffmpeg: str) -> None:
    """Check option declarations separately so media failures cannot become skips."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-h", "full"],
        capture_output=True, text=True, check=True,
    )
    options = {
        line.split()[0].split("[", 1)[0]
        for line in result.stdout.splitlines() if line.startswith("-")
    }
    assert {"-version", "-y", "-loglevel"} <= options, "Invalid FFmpeg full help output"
    if "-display_rotation" not in options:
        pytest.skip(f"{ffmpeg} lacks -display_rotation; rotation fixture only")


@pytest.mark.parametrize("supported", [False, True])
def test_rotation_support_probe_matches_option_not_mentions(monkeypatch, supported):
    help_text = (
        "-version            show version\n-y overwrite\n-loglevel level\n"
        "-display_rotation_extra angle  unrelated option\n"
        "  description mentions -display_rotation but does not declare it\n"
    )
    if supported:
        help_text += "-display_rotation[:<stream_spec>] angle  set rotation\n"
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=help_text, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    if supported:
        require_display_rotation("/tools/ffmpeg")
    else:
        with pytest.raises(pytest.skip.Exception, match="lacks -display_rotation"):
            require_display_rotation("/tools/ffmpeg")
    assert calls == [([
        "/tools/ffmpeg", "-hide_banner", "-h", "full",
    ], {"capture_output": True, "text": True, "check": True})]


def test_rotation_support_probe_propagates_command_failure(monkeypatch):
    error = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="broken executable")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        require_display_rotation("ffmpeg")
    assert caught.value is error


def test_rotation_support_probe_rejects_invalid_help(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, stdout="", stderr=""))
    with pytest.raises(AssertionError, match="Invalid FFmpeg full help"):
        require_display_rotation("ffmpeg")


def test_unsupported_rotation_skips_before_media_commands(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: f"/tools/{name}")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        assert args == ["/tools/ffmpeg", "-hide_banner", "-h", "full"]
        return subprocess.CompletedProcess(
            args, 0, stdout="-version show version\n-y overwrite\n-loglevel level\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(pytest.skip.Exception, match="rotation fixture only"):
        test_real_media_display_geometry_and_playback_clock(tmp_path, "rotation")
    assert len(calls) == 1
    assert not (tmp_path / "base.mp4").exists()


@pytest.mark.parametrize("failure_call", [2, 3], ids=["encode", "rotation-remux"])
def test_supported_rotation_media_failures_are_not_skipped(monkeypatch, tmp_path, failure_call):
    monkeypatch.setattr(shutil, "which", lambda name: f"/tools/{name}")
    calls = []
    error = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="broken media")

    def fake_run(args, **kwargs):
        calls.append(args)
        if len(calls) == failure_call:
            raise error
        return subprocess.CompletedProcess(
            args, 0, stdout="-version show version\n-y overwrite\n-loglevel level\n"
            "-display_rotation angle rotation\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        test_real_media_display_geometry_and_playback_clock(tmp_path, "rotation")
    assert caught.value is error
    assert len(calls) == failure_call


@pytest.mark.parametrize("variant", ["sar", "rotation", "start", "av-offset"])
def test_real_media_display_geometry_and_playback_clock(tmp_path, variant):
    """Small local encodes protect DAR/autorotation and real playback PTS together."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("local ffmpeg and ffprobe required; no model or network used")
    if variant == "rotation":
        require_display_rotation(ffmpeg)
    base = tmp_path / "base.mp4"
    args = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=320x240:rate=10:duration=3"]
    if variant == "av-offset":
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                 "-vf", "setpts=PTS+0.2/TB", "-c:a", "aac", "-t", "3.2"]
    else:
        args += ["-vf", "setsar=2"]
    args += ["-c:v", "mpeg4", "-q:v", "2", str(base)]
    subprocess.run(args, capture_output=True, text=True, check=True)
    video = base
    if variant in {"rotation", "start"}:
        video = tmp_path / f"{variant}.mp4"
        input_options = ["-display_rotation", "90"] if variant == "rotation" else []
        options = [] if variant == "rotation" else [
            "-output_ts_offset", "5",
        ]
        subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-y", *input_options,
                        "-i", str(base),
                        "-c", "copy", *options, str(video)],
                       capture_output=True, text=True, check=True)
    source_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    source = json.loads(source_probe.stdout)["streams"][0]
    if variant == "rotation":
        assert any(item.get("rotation") == 90 for item in source["side_data_list"])
    if variant == "start":
        assert float(source["start_time"]) == pytest.approx(5)
    sampled = frames.extract_frames(video, tmp_path / "frames", interval=.75,
                                    ffmpeg=ffmpeg, ffprobe="ffprobe")
    expected = [.2, 1, 1.8, 2.6] if variant == "av-offset" else [0, .8, 1.6, 2.4]
    assert [frame.time for frame in sampled] == pytest.approx(expected, abs=.001)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                            str(sampled[0].path)], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    # Rotation inverts the display ratio, not merely the coded dimensions.
    height = {"sar": 240, "rotation": 1706, "start": 240, "av-offset": 480}[variant]
    assert (stream["width"], stream["height"]) == (640, height)
    assert stream["sample_aspect_ratio"] == "1:1"
