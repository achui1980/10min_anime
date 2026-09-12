from __future__ import annotations

import shlex
import subprocess
import threading
from pathlib import Path
from unittest import mock

import pytest

from tenmin.render.ffmpeg import (
    FFmpegError,
    has_encoder,
    has_filter,
    parse_names,
    preflight,
    probe_duration,
    progress_seconds,
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


def test_tail_does_not_split_on_carriage_returns():
    """ffmpeg 的统计行是 `\\r` 结尾的，而 str.splitlines() 也在 `\\r` 上切。

    实测（60 秒编码、真实源片）：stderr 共 6520 字节 / 70 个 `\\n`，但里面有 4 个
    `\\r`，splitlines() 于是把它数成 75 行。也就是说「末尾 30 行」里有 5 行是进度
    碎片，真正的错误被顶出去 5 行。行数越少的报错越容易被完全顶掉。
    """
    stats = "".join(f"\rframe={i} fps=300 q=16.0 time=00:00:0{i}" for i in range(5))
    # 真实形态（从实测 stderr 抄的）：最后一条统计之后同样是 `\r`，然后才是真内容
    text = "real error line 1\nreal error line 2\n" + stats + "\r[out#0/mp4] muxing overhead"
    result = tail(text, lines=3)
    assert "real error line 1" in result
    assert "real error line 2" in result
    assert "[out#0/mp4] muxing overhead" in result
    # `\r` 的语义是「回到行首重写」，被覆盖掉的统计不该算进可见内容
    assert "frame=" not in result


def test_tail_drops_physical_lines_that_are_only_overwritten_stats():
    """一整行只有被覆盖掉的统计时，它没有任何可见内容，不该占掉一行配额。"""
    assert tail("real error\n\rframe=1 fps=2 q=16.0\r\n", lines=30) == "real error"


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
    assert seen["args"] == ["ffmpeg", "-nostdin", "-i", "a.mkv"]


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

    executed: dict[str, list[str]] = {}

    def fake_run(argv, **kwargs):
        executed["argv"] = list(argv)
        return FakeCompleted(returncode=1, stderr="Invalid argument")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    with pytest.raises(FFmpegError) as exc:
        run(args)
    message = str(exc.value)
    assert shlex.join(executed["argv"]) in message
    # 具体到本项目会踩的那几个字符：粘回 shell 必须是安全的
    assert "'/v/[LoliHouse] 番 01.mkv'" in message


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


def test_probe_duration_parses_csv(monkeypatch, tmp_path):
    seen = {}
    media = tmp_path / "a.mkv"
    media.write_bytes(b"fake")

    def fake_run(args, **kwargs):
        seen["args"] = args
        return FakeCompleted(stdout="1425.501000\n")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    assert probe_duration(media) == pytest.approx(1425.501)
    assert seen["args"][0] == "ffprobe"
    assert seen["args"][-1] == str(media)


def test_probe_duration_raises_on_unparsable(monkeypatch, tmp_path):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"fake")
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.run",
        lambda args, **kwargs: FakeCompleted(stdout="N/A\n"),
    )
    with pytest.raises(FFmpegError) as exc:
        probe_duration(media)
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
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name, **_: False)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name, **_: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", lambda path, **_: 100.0)
    with pytest.raises(RuntimeError) as exc:
        preflight(video, "libx264")
    assert "libass" in str(exc.value)


def test_preflight_reports_missing_encoder(monkeypatch, tmp_path):
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name, **_: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name, **_: False)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", lambda path, **_: 100.0)
    with pytest.raises(RuntimeError) as exc:
        preflight(video, "h264_videotoolbox")
    assert "h264_videotoolbox" in str(exc.value)


def test_preflight_reports_missing_video(monkeypatch, tmp_path):
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name, **_: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name, **_: True)
    with pytest.raises(FileNotFoundError) as exc:
        preflight(tmp_path / "missing.mkv", "libx264")
    assert "missing.mkv" in str(exc.value)


def test_preflight_returns_source_duration(monkeypatch, tmp_path):
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name, **_: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name, **_: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", lambda path, **_: 1425.5)
    assert preflight(video, "libx264") == pytest.approx(1425.5)


class FakePopen:
    """假的 subprocess.Popen，逐行喂 stdout，不真的起进程。

    实现了 context manager 与 kill()：被测代码现在把 Popen 放进 with 并在异常路径上
    显式 kill（不然 on_progress 一抛就留下孤儿 ffmpeg），假对象必须支持同一套协议。
    """

    class _Stderr:
        def __init__(self, text: str):
            self._text = text

        def read(self) -> str:
            return self._text

    def __init__(self, lines: list[str], returncode: int = 0, stderr: str = ""):
        self.stdout = iter(lines)
        self.stderr = FakePopen._Stderr(stderr)
        self._returncode = returncode
        self.killed = False

    def __enter__(self) -> FakePopen:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def kill(self) -> None:
        self.killed = True

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


# --- 进度字段解析（纯函数，单独测） ---
# 实测 ffmpeg 9 的 `-progress pipe:1` 一个块里同时发 out_time_us / out_time_ms /
# out_time 三个字段，而且 out_time_us 排在 out_time_ms **前面**。所以三者必须是
# 真正的「回退链」而不是各自独立处理，否则一个块会回调三次。


def test_progress_seconds_prefers_out_time_ms():
    fields = {"out_time_us": "1920000", "out_time_ms": "1920000", "out_time": "00:00:01.920000"}
    assert progress_seconds(fields) == pytest.approx(1.92)


def test_progress_seconds_falls_back_to_out_time_us():
    """某个 build 停发 out_time_ms 时，进度条不该静默冻在 0。"""
    assert progress_seconds({"out_time_us": "2500000"}) == pytest.approx(2.5)


def test_progress_seconds_falls_back_to_out_time_timecode():
    assert progress_seconds({"out_time": "01:02:03.500000"}) == pytest.approx(3723.5)


def test_progress_seconds_skips_na_and_uses_next_field():
    """ffmpeg 开跑瞬间会发 out_time_ms=N/A。它不该把整条回退链打死。"""
    assert progress_seconds({"out_time_ms": "N/A", "out_time": "00:00:04.000000"}) == pytest.approx(
        4.0
    )


def test_progress_seconds_returns_none_without_any_usable_field():
    assert progress_seconds({"frame": "1", "fps": "0.00"}) is None
    assert progress_seconds({"out_time_ms": "N/A", "out_time": "N/A"}) is None


def test_run_with_progress_works_on_builds_that_only_send_out_time_us(monkeypatch):
    lines = ["out_time_us=5000000\n", "progress=continue\n", "progress=end\n"]
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(lines),
    )
    seen: list[float] = []
    run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=20.0, on_progress=seen.append)
    assert seen == [0.25, 1.0]


def test_run_with_progress_reports_each_block_once_not_once_per_time_field(monkeypatch):
    """真实 ffmpeg 一个块里三个时间字段都发，回调必须只响一次。"""
    lines = [
        "frame=50\n",
        "out_time_us=1920000\n",
        "out_time_ms=1920000\n",
        "out_time=00:00:01.920000\n",
        "progress=continue\n",
    ]
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(lines),
    )
    seen: list[float] = []
    run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=3.84, on_progress=seen.append)
    assert seen == [pytest.approx(0.5)]


def test_run_with_progress_puts_progress_flag_before_the_input(monkeypatch):
    """-progress 是全局选项，追加在输出文件名之后只是「碰巧能用」。

    放到 -i 之前才是它该在的位置，也让报错里那条命令能原样粘回去复现。
    """
    executed: dict[str, list[str]] = {}

    def fake_popen(args, **kwargs):
        executed["args"] = list(args)
        return FakePopen(["progress=end\n"])

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.Popen", fake_popen)
    run_with_progress(["-y", "-i", "in.mp4", "out.mp4"], total_seconds=1.0)
    args = executed["args"]
    assert args.index("-progress") < args.index("-i")
    assert args[args.index("-progress") + 1] == "pipe:1"


# --- 可执行文件路径可覆盖（RenderConfig.ffmpeg_path / ffprobe_path 的接线） ---
# 用户常有两个 ffmpeg 装（本项目自己就有「哪个编了 libass」的痛点），必须能指定用哪个。


def test_run_uses_configured_ffmpeg_path(monkeypatch):
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return FakeCompleted()

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    run(["-i", "a.mkv"], ffmpeg="/opt/custom/bin/ffmpeg")
    assert seen["args"][0] == "/opt/custom/bin/ffmpeg"


def test_run_with_progress_uses_configured_ffmpeg_path(monkeypatch):
    seen = {}

    def fake_popen(args, **kwargs):
        seen["args"] = list(args)
        return FakePopen(["progress=end\n"])

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.Popen", fake_popen)
    run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=1.0, ffmpeg="/opt/x/ffmpeg")
    assert seen["args"][0] == "/opt/x/ffmpeg"


def test_probe_duration_uses_configured_ffprobe_path(monkeypatch, tmp_path):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"fake")
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return FakeCompleted(stdout="12.5\n")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    probe_duration(media, ffprobe="/opt/x/ffprobe")
    assert seen["args"][0] == "/opt/x/ffprobe"


def test_available_filters_cache_is_keyed_by_binary(monkeypatch):
    """lru_cache 必须按可执行文件路径做 key，否则两个 ffmpeg 会互相冒充。"""
    from tenmin.render import ffmpeg as ffmpeg_mod

    calls: list[str] = []

    def fake_run(args, **kwargs):
        calls.append(args[0])
        return FakeCompleted(stdout=FILTERS_SAMPLE)

    monkeypatch.setattr("tenmin.render.ffmpeg.shutil.which", lambda name: name)
    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    ffmpeg_mod.available_filters.cache_clear()
    try:
        assert "subtitles" in ffmpeg_mod.available_filters("/opt/a/ffmpeg")
        assert "subtitles" in ffmpeg_mod.available_filters("/opt/a/ffmpeg")
        assert "subtitles" in ffmpeg_mod.available_filters("/opt/b/ffmpeg")
    finally:
        ffmpeg_mod.available_filters.cache_clear()
    # 同一个 binary 只探一次，不同 binary 各探一次
    assert calls == ["/opt/a/ffmpeg", "/opt/b/ffmpeg"]


def test_preflight_threads_configured_binaries_through(monkeypatch, tmp_path):
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    seen: dict[str, str] = {}

    def fake_has_filter(name, *, ffmpeg="ffmpeg"):
        seen["filter_ffmpeg"] = ffmpeg
        return True

    def fake_has_encoder(name, *, ffmpeg="ffmpeg"):
        seen["encoder_ffmpeg"] = ffmpeg
        return True

    def fake_probe(path, *, ffprobe="ffprobe"):
        seen["probe_ffprobe"] = ffprobe
        return 100.0

    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", fake_has_filter)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", fake_has_encoder)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", fake_probe)
    preflight(video, "libx264", ffmpeg="/opt/x/ffmpeg", ffprobe="/opt/x/ffprobe")
    assert seen["filter_ffmpeg"] == "/opt/x/ffmpeg"
    assert seen["encoder_ffmpeg"] == "/opt/x/ffmpeg"
    assert seen["probe_ffprobe"] == "/opt/x/ffprobe"


# --- stderr 管道死锁 + 子进程生命周期。这两条刻意用**真的** subprocess 与真的管道 ---
# mock 出来的 FakePopen 永远不会背压，也就永远测不出这个 bug。


def _write_fake_ffmpeg(tmp_path: Path, body: str) -> str:
    """造一个假 ffmpeg（sh 脚本）。它忽略全部参数，只按 body 的剧本读写管道。

    开头那个自杀看门狗是为了回归时不留垃圾：一旦被测代码真的死锁，这个脚本会卡在
    「往写满的 stderr 管道里写」上，除了它自己没人能把它弄死。
    """
    script = tmp_path / "fake_ffmpeg.sh"
    script.write_text(
        "#!/bin/sh\n( sleep 45; kill -9 $$ ) >/dev/null 2>&1 &\n" + body, encoding="utf-8"
    )
    script.chmod(0o755)
    return str(script)


def _run_with_deadline(fn, seconds: float = 30.0):
    """在 daemon 线程里跑，超时就当死锁。

    死锁的表现是永久挂住，绝不能让它挂住整个测试进程 —— 所以刻意不用
    ThreadPoolExecutor：它的 __exit__ 会 shutdown(wait=True)，在卡死的 worker 上
    一样永久阻塞（实测：整个 pytest 进程被挂住，30 秒的 deadline 根本轮不到生效）。
    daemon 线程配 join(timeout) 才真的能放弃。
    """
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as error:  # noqa: BLE001 - 原样搬回主线程
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():  # pragma: no cover - 只在回归时走到
        pytest.fail(f"run_with_progress 在 {seconds} 秒内没返回：stderr 管道写满之后死锁了")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]


def test_run_with_progress_does_not_deadlock_on_a_stderr_flood(tmp_path):
    """stderr 写爆 64KB 管道时不许死锁。

    实测：一次 60 秒编码就产生 6520 字节 stderr，240 秒成片约 26KB，距 64KB 上限只有
    2-3 倍余量。一段 per-frame warning（`Past duration ... too large`、HEVC 解码器
    抱怨）就能突破 —— 然后 ffmpeg 阻塞在写 stderr、Python 阻塞在读 stdout，永久挂死
    且没有任何输出。这里让假 ffmpeg **先**写 ~230KB stderr 再写 stdout，精确复现那个
    顺序（已单独验证：不排空 stderr 的话这段真的会永久卡住）。
    """
    flood_lines = 5000
    fake = _write_fake_ffmpeg(
        tmp_path,
        f"i=0\n"
        f'while [ $i -lt {flood_lines} ]; do\n'
        f'  echo "[hevc @ 0x1] Past duration 0.999992 too large" >&2\n'
        f"  i=$((i+1))\n"
        f"done\n"
        f'echo "Conversion failed somewhere near the end" >&2\n'
        f"echo out_time_ms=5000000\n"
        f"echo progress=end\n",
    )
    seen: list[float] = []
    stderr = _run_with_deadline(
        lambda: run_with_progress(
            ["-i", "in.mp4", "out.mp4"],
            total_seconds=10.0,
            on_progress=seen.append,
            ffmpeg=fake,
        )
    )
    # 进度照常解析
    assert seen == [0.5, 1.0]
    # stderr 一个字节都没丢，尾部的真错误还在
    assert stderr.count("Past duration") == flood_lines
    assert "Conversion failed somewhere near the end" in stderr


def test_run_with_progress_error_tail_survives_a_stderr_flood(tmp_path):
    """洪水般的 stderr + 非零退出：报错里必须还是那句真错误，而不是超时/空串。"""
    fake = _write_fake_ffmpeg(
        tmp_path,
        "i=0\n"
        'while [ $i -lt 5000 ]; do echo "[hevc @ 0x1] Past duration too large" >&2; '
        "i=$((i+1)); done\n"
        'echo "Error while filtering: Invalid argument" >&2\n'
        "echo progress=end\n"
        "exit 1\n",
    )

    def go():
        with pytest.raises(FFmpegError) as exc:
            run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=10.0, ffmpeg=fake)
        return str(exc.value)

    message = _run_with_deadline(go)
    assert "Error while filtering: Invalid argument" in message


def test_run_with_progress_passes_nostats(monkeypatch):
    """进度是从 stdout 的 -progress 解析的，stderr 上那份统计是纯噪音。

    实测它同时是两个问题的来源：白占管道配额（60 秒编码里 506 字节），以及带进 4 个
    `\\r` 把 tail 的「末尾 30 行」搅成进度碎片。
    """
    seen: dict[str, list[str]] = {}

    def fake_popen(args, **kwargs):
        seen["args"] = list(args)
        return FakePopen(["progress=end\n"])

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.Popen", fake_popen)
    run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=1.0)
    assert "-nostats" in seen["args"]


def test_run_with_progress_kills_ffmpeg_when_on_progress_raises(tmp_path):
    """on_progress 抛异常（reporter/rich 出错）时不许遗弃子进程。

    原来 Popen 既没进 with 也没 try/finally：回调一炸，ffmpeg 就变成孤儿继续烧 CPU，
    管道也跟着泄漏。用户 Ctrl-C 是同一条路径。
    """
    fake = _write_fake_ffmpeg(
        tmp_path,
        "echo out_time_ms=1000000\necho progress=continue\nsleep 20 >/dev/null 2>&1\n",
    )
    spawned: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def recording_popen(*a, **k):
        proc = real_popen(*a, **k)
        spawned.append(proc)
        return proc

    with mock.patch.object(subprocess, "Popen", recording_popen):

        def boom(_fraction: float) -> None:
            raise RuntimeError("reporter 炸了")

        with pytest.raises(RuntimeError, match="reporter 炸了"):
            run_with_progress(
                ["-i", "in.mp4", "out.mp4"],
                total_seconds=10.0,
                on_progress=boom,
                ffmpeg=fake,
            )

    assert len(spawned) == 1
    # 进程必须已经死了（不是还在 sleep）
    assert spawned[0].poll() is not None


# --- 不许抢 stdin，短命令要有超时 ---


def test_run_passes_nostdin_and_detaches_stdin(monkeypatch):
    """ffmpeg 会抢共享 TTY 的 stdin，然后静默挂死整条流水线。

    subprocess.run(capture_output=True) **不**设 stdin，子进程直接继承父进程的 TTY。
    -nostdin 让 ffmpeg 自己不读，stdin=DEVNULL 从 OS 层面兜底 —— 后者才是结构性的
    保证，因为它不依赖被调的二进制认不认那个 flag。
    """
    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["args"] = list(args)
        seen["stdin"] = kwargs.get("stdin")
        return FakeCompleted()

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    run(["-i", "a.mkv"])
    assert "-nostdin" in seen["args"]
    assert seen["stdin"] is subprocess.DEVNULL


def test_run_with_progress_passes_nostdin_and_detaches_stdin(monkeypatch):
    seen: dict[str, object] = {}

    def fake_popen(args, **kwargs):
        seen["args"] = list(args)
        seen["stdin"] = kwargs.get("stdin")
        return FakePopen(["progress=end\n"])

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.Popen", fake_popen)
    run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=1.0)
    assert "-nostdin" in seen["args"]
    assert seen["stdin"] is subprocess.DEVNULL


def test_probe_duration_must_not_pass_nostdin_to_ffprobe(monkeypatch, tmp_path):
    """ffprobe **不认** -nostdin，加上去会直接把它打死。

    实测 ffprobe 9.0.1：`ffprobe -nostdin -v error ...` →
    `Failed to set value '-v' for option 'nostdin': Option not found`，退出码 1。
    所以 ffprobe 那边只能靠 stdin=DEVNULL。这条测试锁死「别顺手给它也加上」。
    """
    media = tmp_path / "a.mkv"
    media.write_bytes(b"fake")
    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["args"] = list(args)
        seen["stdin"] = kwargs.get("stdin")
        return FakeCompleted(stdout="1.5\n")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    probe_duration(media)
    assert "-nostdin" not in seen["args"]
    assert seen["stdin"] is subprocess.DEVNULL


def test_probe_duration_has_a_timeout_and_reports_it_readably(monkeypatch, tmp_path):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"fake")
    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout") or 0)

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    with pytest.raises(FFmpegError) as exc:
        probe_duration(media)
    assert isinstance(seen["timeout"], (int, float))
    message = str(exc.value)
    assert "超时" in message
    # 出错消息里必须有可粘贴的命令，不然「超时了」等于没说
    assert "a.mkv" in message


def test_capability_probe_has_a_timeout(monkeypatch):
    from tenmin.render import ffmpeg as ffmpeg_mod

    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return FakeCompleted(stdout=FILTERS_SAMPLE)

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    ffmpeg_mod.available_filters.cache_clear()
    try:
        ffmpeg_mod.available_filters()
    finally:
        ffmpeg_mod.available_filters.cache_clear()
    assert isinstance(seen["timeout"], (int, float))


def test_run_accepts_an_explicit_timeout_and_raises_ffmpeg_error(monkeypatch):
    """短命令（preflight 的探测）要能给超时；长时间编码刻意不给。"""

    def fake_run(args, **kwargs):
        assert kwargs.get("timeout") == 5.0
        raise subprocess.TimeoutExpired(args, 5.0)

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    with pytest.raises(FFmpegError, match="超时"):
        run(["-i", "a.mkv"], timeout=5.0)


def test_run_defaults_to_no_timeout_because_encoding_takes_minutes(monkeypatch):
    """mix_audio 是几分钟的真编码，给它设超时只会在慢机器上误杀。"""
    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return FakeCompleted()

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    run(["-i", "a.mkv"])
    assert seen["timeout"] is None


# --- 能力探测的失败必须是响的，而且不许被缓存 ---


@pytest.fixture
def clear_capability_cache():
    from tenmin.render import ffmpeg as ffmpeg_mod

    ffmpeg_mod.available_filters.cache_clear()
    ffmpeg_mod.available_encoders.cache_clear()
    yield ffmpeg_mod
    ffmpeg_mod.available_filters.cache_clear()
    ffmpeg_mod.available_encoders.cache_clear()


def test_capability_probe_reports_a_missing_binary_by_name(monkeypatch, clear_capability_cache):
    """ffmpeg 根本不在时，用户原来看到的是裸 FileNotFoundError: 'ffmpeg'。"""
    monkeypatch.setattr("tenmin.render.ffmpeg.shutil.which", lambda _name: None)
    with pytest.raises(FFmpegError) as exc:
        clear_capability_cache.available_filters("/nope/ffmpeg")
    message = str(exc.value)
    assert "/nope/ffmpeg" in message
    # 得告诉用户有 render.ffmpeg_path 这个旋钮可以指路
    assert "ffmpeg_path" in message


def test_capability_probe_raises_on_nonzero_exit_instead_of_returning_empty(
    monkeypatch, clear_capability_cache
):
    """非零退出原来被完全忽略，返回空集合。

    空集合会让 preflight 自信地报「你的 ffmpeg 没编 libass」—— 对「这个 ffmpeg 坏了」
    来说这是**错误诊断**，用户会去重装 libass，而问题根本不在那儿。
    """
    monkeypatch.setattr("tenmin.render.ffmpeg.shutil.which", lambda name: name)
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.run",
        lambda args, **kwargs: FakeCompleted(returncode=1, stderr="dyld: Library not loaded"),
    )
    with pytest.raises(FFmpegError) as exc:
        clear_capability_cache.available_filters()
    assert "dyld: Library not loaded" in str(exc.value)


def test_capability_probe_raises_when_nothing_parses(monkeypatch, clear_capability_cache):
    """退出码 0 但一个名字都没解析出来，同样是「结论不可信」而不是「什么都没有」。

    历史上正是这种情况让 has_filter("subtitles") 恒为 False、preflight 谎报没编
    libass（解析器的标志列宽度写死成 3 起，见 _NAME_LINE 的注释）。
    """
    monkeypatch.setattr("tenmin.render.ffmpeg.shutil.which", lambda name: name)
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.run",
        lambda args, **kwargs: FakeCompleted(stdout="totally unparseable\n"),
    )
    with pytest.raises(FFmpegError):
        clear_capability_cache.available_filters()


def test_capability_probe_does_not_cache_a_failure(monkeypatch, clear_capability_cache):
    """一次瞬时失败的空集合原来会被 lru_cache 永久钉在进程里。"""
    monkeypatch.setattr("tenmin.render.ffmpeg.shutil.which", lambda name: name)
    attempts = {"n": 0}

    def flaky(args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeCompleted(returncode=1, stderr="transient boom")
        return FakeCompleted(stdout=FILTERS_SAMPLE)

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", flaky)
    with pytest.raises(FFmpegError):
        clear_capability_cache.available_filters()
    # 第二次必须真的重探，而不是拿到上一次那个（空的/坏的）结果
    assert "subtitles" in clear_capability_cache.available_filters()
    assert attempts["n"] == 2
    # 成功的结果照常缓存
    assert "subtitles" in clear_capability_cache.available_filters()
    assert attempts["n"] == 2


def test_preflight_does_not_blame_libass_when_ffmpeg_is_missing(
    monkeypatch, clear_capability_cache, tmp_path
):
    """preflight 的错误诊断必须指向真正的原因。"""
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tenmin.render.ffmpeg.shutil.which", lambda _name: None)
    with pytest.raises(FFmpegError) as exc:
        preflight(video, "libx264")
    message = str(exc.value)
    assert "没编 libass" not in message, "不许把「缺二进制」诊断成「没编 libass」"
    assert "找不到可执行文件" in message


# --- probe_duration 的合理性检查 ---


def test_probe_duration_reports_a_missing_file_as_such(monkeypatch, tmp_path):
    """文件不存在时该说「文件不存在」，不是「读不出时长」。

    后者会把用户送去查 ffprobe / 容器格式，而真正的原因是路径写错了或者外置盘没挂上。
    也顺带省掉一次注定失败的子进程。
    """

    def unreachable(args, **kwargs):
        raise AssertionError("文件不存在时不该 spawn ffprobe")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", unreachable)
    missing = tmp_path / "video" / "E02.mkv"
    with pytest.raises(FileNotFoundError) as exc:
        probe_duration(missing)
    assert str(missing) in str(exc.value)


def test_probe_duration_is_a_file_not_found_error_so_ingest_still_degrades(tmp_path):
    """pipeline._source_duration 靠 catch (FFmpegError, OSError) 做降级。

    FileNotFoundError 是 OSError 子类，所以「只有 SRT 没有视频」那条合法用法不受影响。
    """
    assert issubclass(FileNotFoundError, OSError)


@pytest.mark.parametrize("bad", ["0", "0.000000", "-1.5"])
def test_probe_duration_rejects_non_positive_durations(monkeypatch, tmp_path, bad):
    """<= 0 的时长看起来是个正常数字，然后毒化整条时间轴。

    截断/空的容器就会给出这种值。timeline 拿它去算偏移不会报错，只会静默出一个
    时间轴全错的成片 —— 必须在源头拦掉。
    """
    media = tmp_path / "a.mkv"
    media.write_bytes(b"fake")
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.run",
        lambda args, **kwargs: FakeCompleted(stdout=bad + "\n"),
    )
    with pytest.raises(FFmpegError) as exc:
        probe_duration(media)
    assert "a.mkv" in str(exc.value)


def test_probe_duration_accepts_a_normal_duration(monkeypatch, tmp_path):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"fake")
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.run",
        lambda args, **kwargs: FakeCompleted(stdout="1509.994667\n"),
    )
    assert probe_duration(media) == pytest.approx(1509.994667)
