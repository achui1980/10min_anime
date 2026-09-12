import shlex
from pathlib import Path

import pytest

from tenmin.render.ffmpeg import (
    FFmpegError,
    has_encoder,
    has_filter,
    parse_names,
    preflight,
    probe_duration,
    run,
    run_with_progress,
    tail,
)

# 真实 ffmpeg 9 的 -filters 输出：标志位只有 2 列（早期版本是 3 列）。
# 这份样本是从 `ffmpeg -hide_banner -filters` 直接抄的，包括开头那段图例，
# 千万别手写简化版——正是手写的 3 列样本让解析器的 bug 藏了整整一个版本。
FILTERS_SAMPLE = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
  ------
 TS aap               AA->A      Apply Affine Projection algorithm to first audio stream.
 .. ass               V->V       Render ASS subtitles onto input video using the libass library.
 .. concat            N->N       Concatenate audio and video streams.
 .. subtitles         V->V       Render text subtitles onto input video using the libass library.
 TS volume            A->A       Change input volume.
"""

ENCODERS_SAMPLE = """Encoders:
 V..... libx264              libx264 H.264 / AVC
 V..... h264_videotoolbox    VideoToolbox H.264 Encoder
 A..... aac                  AAC (Advanced Audio Coding)
"""


def test_parse_names_from_filters():
    names = parse_names(FILTERS_SAMPLE)
    assert "subtitles" in names
    assert "ass" in names
    assert "concat" in names
    assert "volume" in names
    assert "aap" in names
    # 表头那行不该被当成名字
    assert "Filters:" not in names


def test_parse_names_from_encoders():
    names = parse_names(ENCODERS_SAMPLE)
    assert names == {"libx264", "h264_videotoolbox", "aac"}


def test_parse_names_on_empty_text():
    assert parse_names("") == set()


def test_tail_keeps_last_lines():
    text = "\n".join(str(i) for i in range(100))
    assert tail(text, lines=3) == "97\n98\n99"


def test_tail_shorter_than_limit_returns_all():
    assert tail("a\nb", lines=30) == "a\nb"


def test_tail_strips_trailing_blank_lines():
    assert tail("a\nb\n\n\n", lines=2) == "a\nb"


def test_ffmpeg_error_is_runtime_error():
    assert issubclass(FFmpegError, RuntimeError)
    with pytest.raises(RuntimeError):
        raise FFmpegError("boom")


class FakeCompleted:
    """替代 subprocess.CompletedProcess，只带测试关心的三个字段。"""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_returns_stderr_on_success(monkeypatch):
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return FakeCompleted(stdout="", stderr="frame= 100")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    assert run(["-i", "a.mkv"]) == "frame= 100"
    assert seen["args"] == ["ffmpeg", "-i", "a.mkv"]


def test_run_raises_with_stderr_tail(monkeypatch):
    noise = "\n".join(f"line {i}" for i in range(50))

    def fake_run(args, **kwargs):
        return FakeCompleted(returncode=1, stderr=noise + "\nInvalid argument")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    with pytest.raises(FFmpegError) as exc:
        run(["-i", "a.mkv"])
    message = str(exc.value)
    assert "Invalid argument" in message
    assert "line 49" in message
    # 只留末尾 30 行，开头的噪声不该出现
    assert "line 0" not in message


def test_run_error_command_is_paste_safe(monkeypatch):
    """报出去的命令必须能直接粘回 shell 跑。

    `' '.join(args)` 拼出来的东西对本项目是必坏的：-filter_complex 的值里有 `;`
    （分隔 filter）和 `'`（包路径），源片名常年长成 `[LoliHouse] xxx.mkv`。粘回
    shell 里 `;` 会被当命令分隔符、`[...]` 会被当 glob，用户复现不了自己的报错。
    """
    args = [
        "-i",
        "/v/[LoliHouse] 番 01.mkv",
        "-filter_complex",
        "[0:v]trim=start=1;[v0]subtitles=filename='/x/E02.ass'[vout]",
    ]

    def fake_run(_args, **kwargs):
        return FakeCompleted(returncode=1, stderr="Invalid argument")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    with pytest.raises(FFmpegError) as exc:
        run(args)
    message = str(exc.value)
    assert shlex.join(["ffmpeg", *args]) in message


def test_run_with_progress_error_reports_the_argv_actually_executed(monkeypatch):
    """报出去的命令必须是**真正执行**的那条，一个 token 都不许漏。

    原来这里重建命令字符串时漏掉了自己注入的 -progress pipe:1，于是用户照着报错
    粘回去跑的是另一条命令 —— 最坏的情况是那条能跑通，把真正的锅藏起来。
    """
    executed: dict[str, list[str]] = {}

    def fake_popen(args, **kwargs):
        executed["args"] = list(args)
        return FakePopen(["progress=end\n"], returncode=1, stderr="Invalid argument")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.Popen", fake_popen)
    with pytest.raises(FFmpegError) as exc:
        run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=10.0)
    assert shlex.join(executed["args"]) in str(exc.value)


def test_probe_duration_parses_csv(monkeypatch):
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return FakeCompleted(stdout="1425.501000\n")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    assert probe_duration(Path("a.mkv")) == pytest.approx(1425.501)
    assert seen["args"][0] == "ffprobe"
    assert seen["args"][-1] == "a.mkv"


def test_probe_duration_raises_on_unparsable(monkeypatch):
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.run",
        lambda args, **kwargs: FakeCompleted(stdout="N/A\n"),
    )
    with pytest.raises(FFmpegError) as exc:
        probe_duration(Path("a.mkv"))
    assert "a.mkv" in str(exc.value)


def test_has_filter_and_has_encoder(monkeypatch):
    # available_filters / available_encoders 带 lru_cache，用例之间必须清缓存
    from tenmin.render import ffmpeg as ffmpeg_mod

    def fake_run(args, **kwargs):
        if "-filters" in args:
            return FakeCompleted(stdout=FILTERS_SAMPLE)
        return FakeCompleted(stdout=ENCODERS_SAMPLE)

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    ffmpeg_mod.available_filters.cache_clear()
    ffmpeg_mod.available_encoders.cache_clear()
    assert has_filter("subtitles") is True
    assert has_filter("nosuchfilter") is False
    assert has_encoder("libx264") is True
    assert has_encoder("libx265") is False
    ffmpeg_mod.available_filters.cache_clear()
    ffmpeg_mod.available_encoders.cache_clear()


def test_preflight_reports_missing_libass(monkeypatch, tmp_path):
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name: False)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", lambda path: 100.0)
    with pytest.raises(RuntimeError) as exc:
        preflight(video, "libx264")
    assert "libass" in str(exc.value)


def test_preflight_reports_missing_encoder(monkeypatch, tmp_path):
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name: False)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", lambda path: 100.0)
    with pytest.raises(RuntimeError) as exc:
        preflight(video, "h264_videotoolbox")
    assert "h264_videotoolbox" in str(exc.value)


def test_preflight_reports_missing_video(monkeypatch, tmp_path):
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name: True)
    with pytest.raises(FileNotFoundError) as exc:
        preflight(tmp_path / "missing.mkv", "libx264")
    assert "missing.mkv" in str(exc.value)


def test_preflight_returns_source_duration(monkeypatch, tmp_path):
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", lambda path: 1425.5)
    assert preflight(video, "libx264") == pytest.approx(1425.5)


class FakePopen:
    """假的 subprocess.Popen，逐行喂 stdout，不真的起进程。"""

    class _Stderr:
        def __init__(self, text: str):
            self._text = text

        def read(self) -> str:
            return self._text

    def __init__(self, lines: list[str], returncode: int = 0, stderr: str = ""):
        self.stdout = iter(lines)
        self.stderr = FakePopen._Stderr(stderr)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def test_run_with_progress_reports_fraction_from_out_time_ms(monkeypatch):
    lines = [
        "frame=1\n",
        "out_time_ms=5000000\n",
        "progress=continue\n",
        "out_time_ms=10000000\n",
        "progress=end\n",
    ]
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(lines),
    )
    seen: list[float] = []
    run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=20.0, on_progress=seen.append)
    assert seen == [0.25, 0.5, 1.0]


def test_run_with_progress_raises_with_stderr_tail(monkeypatch):
    noise = "\n".join(f"line {i}" for i in range(50))
    stderr = noise + "\nInvalid argument\n"
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(["progress=end\n"], returncode=1, stderr=stderr),
    )
    with pytest.raises(FFmpegError) as exc:
        run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=10.0)
    message = str(exc.value)
    assert "Invalid argument" in message
    assert "line 0" not in message


def test_run_with_progress_clamps_fraction_to_one(monkeypatch):
    lines = ["out_time_ms=999999999\n"]
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(lines),
    )
    seen: list[float] = []
    run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=5.0, on_progress=seen.append)
    assert seen == [1.0]


def test_run_with_progress_works_without_on_progress_callback(monkeypatch):
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(["out_time_ms=1000000\n", "progress=end\n"]),
    )
    result = run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=1.0)
    assert result == ""
