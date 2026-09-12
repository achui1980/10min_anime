"""ffmpeg / ffprobe 子进程封装。解析部分是纯函数，单独测。"""

from __future__ import annotations

import re
import shlex
import subprocess
import threading
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import IO

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
STDERR_TAIL_LINES = 30
# stderr 排空线程的 join 上限。走到这个上限只可能是 ffmpeg 已经退出但线程还没收到 EOF，
# 属于「不该发生」；给个上限是为了宁可丢掉诊断信息也不把出片卡成永久挂死。
STDERR_DRAIN_TIMEOUT_SECONDS = 10.0

# ffmpeg -filters / -encoders 每行形如 " .. ass  V->V  描述"，
# 标志列只由大写字母和点组成，名字是紧跟其后的第一个 token。
# 宽度必须从 2 起：-encoders 的标志列是 6 列（"V....D"），但 -filters 只有 2 列
# （ffmpeg 9 实测 " .. subtitles"）。写死 3 起会让 -filters 一个都解析不出来，
# 于是 has_filter("subtitles") 恒为 False，preflight 谎报「没编 libass」。
_NAME_LINE = re.compile(r"^\s*[A-Z.]{2,6}\s+(\S+)\s")


class FFmpegError(RuntimeError):
    """ffmpeg 非零退出。消息里必须带 stderr 尾部，否则等于没报错。"""


def parse_names(text: str) -> set[str]:
    """从 -filters / -encoders 的输出里抽出可用名字。"""
    return {m.group(1) for m in (_NAME_LINE.match(line) for line in text.splitlines()) if m}


# `-progress` 一个块里的时间字段，按优先级排列。实测 ffmpeg 9 一个块同时发这三个，
# 而且 out_time_us 排在 out_time_ms **前面** —— 所以它们必须是回退链而不是各自
# 独立处理，否则一个块会把 on_progress 回调三次。
#
# out_time_ms 的名字带 "ms" 但单位是**微秒**（众所周知的 ffmpeg 老 bug/历史遗留），
# 跟 out_time_us 同值，所以两者共用同一个除数 1_000_000。
_MICROSECOND_KEYS = ("out_time_ms", "out_time_us")
_TIMECODE_KEY = "out_time"


def _parse_timecode(value: str) -> float | None:
    """`HH:MM:SS.ffffff` → 秒。认不出返回 None（ffmpeg 会发 N/A）。"""
    parts = value.strip().lstrip("-").split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (float(p) for p in parts)
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def progress_seconds(fields: Mapping[str, str]) -> float | None:
    """从一个 `-progress` 块里读出「已经编码到第几秒」。读不出返回 None。

    只认 out_time_ms 是个静默失效的隐患：某个 build 停发它，进度条就永远冻在 0，
    而且没有任何警告 —— 渲染看起来卡死了，其实在正常跑。所以走回退链。
    """
    for key in _MICROSECOND_KEYS:
        raw = fields.get(key)
        if raw is None:
            continue
        try:
            return int(raw) / 1_000_000
        except ValueError:
            continue  # N/A：换下一个字段，别把整条链打死
    raw = fields.get(_TIMECODE_KEY)
    if raw is None:
        return None
    return _parse_timecode(raw)


def tail(text: str, lines: int = STDERR_TAIL_LINES) -> str:
    """取末尾若干行。ffmpeg 的真实错误永远在 stderr 尾部。

    刻意**不**用 str.splitlines()：它除了 `\\n` 还在 `\\r` 上切，而 ffmpeg 的统计行
    正是 `\\r` 结尾的。实测一次 60 秒编码的 stderr 有 70 个 `\\n` 但 4 个 `\\r`，
    splitlines() 把它数成 75 行 —— 「末尾 30 行」里凭空混进 5 行进度碎片，真正的
    错误被顶出去。加了 -nostats 之后正常情况下不再有 `\\r`（实测降到 0 个），
    但显式按 `\\r` 的真实语义处理才不用依赖「上游一定记得加那个 flag」。

    `\\r` 的语义是「回到行首重写」，所以一个物理行（`\\n` 之间）的可见内容就是
    最后一个 `\\r` 之后的那段 —— 跟终端上看到的一致。
    """
    stripped = text.rstrip("\n")
    if not stripped:
        return ""
    visible: list[str] = []
    for line in stripped.split("\n"):
        collapsed = line.rpartition("\r")[2]
        if not collapsed and "\r" in line:
            # 整行都是被后续写覆盖掉的统计，没有任何可见内容，不该占掉一行配额
            continue
        visible.append(collapsed)
    return "\n".join(visible[-lines:])


def run(args: list[str], *, ffmpeg: str = FFMPEG) -> str:
    """跑 ffmpeg，返回 stderr（ffmpeg 的进度与日志都在 stderr）。

    source 视频的容器元数据（title/album/description 等标签）常年是老字幕组用非
    UTF-8 编码硬塞进去的脏数据，ffmpeg 会把这些原始字节原样打到 stderr 里。严格
    UTF-8 解码遇到这种输入必炸——所以这里用 errors="replace"，脏字节换成 U+FFFD，
    不让一段无关的元数据把整条渲染流水线搞挂。
    """
    argv = [ffmpeg, *args]
    completed = subprocess.run(
        argv, capture_output=True, text=True, errors="replace"
    )
    if completed.returncode != 0:
        raise FFmpegError(
            f"ffmpeg 退出码 {completed.returncode}，命令：\n"
            f"{shlex.join(argv)}\n\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(completed.stderr)}"
        )
    return completed.stderr


def probe_duration(path: Path, *, ffprobe: str = FFPROBE) -> float:
    """用 ffprobe 读时长（秒）。"""
    args = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(path),
    ]
    completed = subprocess.run(args, capture_output=True, text=True, errors="replace")
    if completed.returncode != 0:
        raise FFmpegError(
            f"ffprobe 读不出 {path} 的时长，退出码 {completed.returncode}：\n"
            f"{tail(completed.stderr)}"
        )
    try:
        return float(completed.stdout.strip())
    except ValueError as error:
        raise FFmpegError(
            f"ffprobe 读不出 {path} 的时长，输出是 {completed.stdout.strip()!r}"
        ) from error


# maxsize 从 1 提到 8：key 是可执行文件路径，而 preflight 在批量模式下每集都调。
# 写死 1 的话，只要有人在同一个进程里交替问两个 ffmpeg，缓存就退化成每次重探。
@lru_cache(maxsize=8)
def available_filters(ffmpeg: str = FFMPEG) -> frozenset[str]:
    """这个 ffmpeg 编进了哪些滤镜。**按可执行文件路径缓存**，两个 ffmpeg 不会互相冒充。"""
    return _probe_capabilities(ffmpeg, "-filters")


@lru_cache(maxsize=8)
def available_encoders(ffmpeg: str = FFMPEG) -> frozenset[str]:
    """这个 ffmpeg 编进了哪些编码器。同样按可执行文件路径缓存。"""
    return _probe_capabilities(ffmpeg, "-encoders")


def _probe_capabilities(ffmpeg: str, flag: str) -> frozenset[str]:
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", flag],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return frozenset(parse_names(completed.stdout))


def has_filter(name: str, *, ffmpeg: str = FFMPEG) -> bool:
    return name in available_filters(ffmpeg)


def has_encoder(name: str, *, ffmpeg: str = FFMPEG) -> bool:
    return name in available_encoders(ffmpeg)


def preflight(
    video: Path,
    video_encoder: str,
    *,
    ffmpeg: str = FFMPEG,
    ffprobe: str = FFPROBE,
) -> float:
    """开跑前一次性检查，返回源片时长。任一项不满足立刻抛错。

    渲染动辄几分钟，绝不能跑完 TTS 才在最后一步炸掉。
    """
    if not has_filter("subtitles", ffmpeg=ffmpeg):
        raise RuntimeError(
            "你的 ffmpeg 没编 libass，subtitles 滤镜不可用，烧不了字幕。\n"
            "请重装：brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass"
        )
    if not has_encoder(video_encoder, ffmpeg=ffmpeg):
        raise RuntimeError(
            f"你的 ffmpeg 没有编码器 {video_encoder}，请改 project.yaml 的 render.video_encoder，"
            "或重装 ffmpeg"
        )
    if not Path(video).is_file():
        raise FileNotFoundError(f"找不到源视频 {video}，请检查 project.yaml 的 episodes[].video")
    return probe_duration(Path(video), ffprobe=ffprobe)


def _drain(stream: IO[str], sink: list[str]) -> None:
    """把一条管道读到 EOF。给 run_with_progress 的 stderr 排空线程用。

    读到一半管道被关掉（异常路径上 Popen.__exit__ 会关）会抛 ValueError / OSError，
    那时我们已经不要这份 stderr 了，静默收工就行 —— 让线程里冒异常只会往 stderr 打一段
    与真正错因无关的 traceback，把用户的注意力引错方向。
    """
    try:
        sink.append(stream.read())
    except (ValueError, OSError):  # pragma: no cover - 只在异常清理路径上走到
        pass


def run_with_progress(
    args: list[str],
    *,
    total_seconds: float,
    on_progress: Callable[[float], None] | None = None,
    ffmpeg: str = FFMPEG,
) -> str:
    """跟 run() 一样跑 ffmpeg，但额外加 -progress pipe:1，流式解析进度，
    每读完一个 progress 块就换算成 0.0~1.0 的比例回调 on_progress。

    时间字段的回退链见 progress_seconds。按「块」而不是按「行」回调，是因为一个块里
    out_time_us / out_time_ms / out_time 三个字段都会发，逐行处理会把回调打三遍。

    stderr 必须**并发**排空，不能等 stdout 读到 EOF 再读（原来就是这么写的）：管道容量
    只有 64KB，实测一次 60 秒编码就产生 6520 字节 stderr、240 秒成片约 26KB，一段
    per-frame warning（`Past duration ... too large`、HEVC 解码器抱怨）就能突破上限
    —— 然后 ffmpeg 阻塞在写 stderr、Python 阻塞在读 stdout，永久挂死且没有任何输出。
    -nostats 只是把噪音（实测 506 字节 + 4 个搅乱 tail 的 `\\r`）拿掉，**不能**当成
    修复：真正兜住这件事的是那个排空线程。
    """
    # -progress 是全局选项，放到 -i 之前才是它该在的位置（原来追加在输出文件名之后，
    # 碰巧能用而已）。argv 只拼一次，Popen 与出错消息共用同一个变量：原来出错分支自己
    # 重建了一遍字符串、且漏了 -progress pipe:1，报出来的命令不是真正跑的那条。
    argv = [ffmpeg, "-nostats", "-progress", "pipe:1", *args]

    def report(fields: dict[str, str]) -> None:
        seconds = progress_seconds(fields)
        if seconds is None or on_progress is None or total_seconds <= 0:
            return
        on_progress(max(0.0, min(1.0, seconds / total_seconds)))

    # with + 异常路径上显式 kill：原来 Popen 既没进 with 也没 try/finally，on_progress
    # 一抛（reporter/rich 出错）或用户 Ctrl-C，ffmpeg 就变成孤儿继续烧 CPU、管道泄漏。
    # kill 必须在 __exit__ 之前：__exit__ 先关管道再 wait()，对一个还在跑的几分钟编码
    # 来说那个 wait() 自己就是一次挂死。
    with subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    ) as process:
        try:
            drained: list[str] = []
            drainer = threading.Thread(
                target=_drain, args=(process.stderr, drained), daemon=True
            )
            drainer.start()

            block: dict[str, str] = {}
            for line in process.stdout:
                line = line.strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                block[key] = value
                if key != "progress":
                    continue
                # `progress=continue` / `progress=end` 是块的结束标记
                report(block)
                if value == "end" and on_progress is not None:
                    on_progress(1.0)
                block.clear()
            # 没有以 progress= 收尾的残块也要报一次：ffmpeg 被 kill / 提前断流时最后
            # 那个块是不完整的，丢掉它等于把「实际跑到哪」这条信息扔了。
            report(block)

            returncode = process.wait()
            # stdout 已经 EOF、进程已经退出，stderr 必然也到 EOF，这个 join 立刻返回。
            # 仍然给上限：宁可丢掉诊断信息，也不要把「出片」卡成永久挂死。
            drainer.join(STDERR_DRAIN_TIMEOUT_SECONDS)
            stderr = "".join(drained)
        except BaseException:
            process.kill()
            raise
    if returncode != 0:
        raise FFmpegError(
            f"ffmpeg 执行失败（退出码 {returncode}）：{shlex.join(argv)}\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(stderr)}"
        )
    return stderr
