"""ffmpeg / ffprobe 子进程封装。解析部分是纯函数，单独测。"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
STDERR_TAIL_LINES = 30

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


def run(args: list[str]) -> str:
    """跑 ffmpeg，返回 stderr（ffmpeg 的进度与日志都在 stderr）。

    source 视频的容器元数据（title/album/description 等标签）常年是老字幕组用非
    UTF-8 编码硬塞进去的脏数据，ffmpeg 会把这些原始字节原样打到 stderr 里。严格
    UTF-8 解码遇到这种输入必炸——所以这里用 errors="replace"，脏字节换成 U+FFFD，
    不让一段无关的元数据把整条渲染流水线搞挂。
    """
    argv = [FFMPEG, *args]
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


def probe_duration(path: Path) -> float:
    """用 ffprobe 读时长（秒）。"""
    args = [
        FFPROBE,
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


@lru_cache(maxsize=1)
def available_filters() -> set[str]:
    completed = subprocess.run(
        [FFMPEG, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return parse_names(completed.stdout)


@lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    completed = subprocess.run(
        [FFMPEG, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return parse_names(completed.stdout)


def has_filter(name: str) -> bool:
    return name in available_filters()


def has_encoder(name: str) -> bool:
    return name in available_encoders()


def preflight(video: Path, video_encoder: str) -> float:
    """开跑前一次性检查，返回源片时长。任一项不满足立刻抛错。

    渲染动辄几分钟，绝不能跑完 TTS 才在最后一步炸掉。
    """
    if not has_filter("subtitles"):
        raise RuntimeError(
            "你的 ffmpeg 没编 libass，subtitles 滤镜不可用，烧不了字幕。\n"
            "请重装：brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass"
        )
    if not has_encoder(video_encoder):
        raise RuntimeError(
            f"你的 ffmpeg 没有编码器 {video_encoder}，请改 project.yaml 的 render.video_encoder，"
            "或重装 ffmpeg"
        )
    if not Path(video).is_file():
        raise FileNotFoundError(f"找不到源视频 {video}，请检查 project.yaml 的 episodes[].video")
    return probe_duration(Path(video))


def run_with_progress(
    args: list[str],
    *,
    total_seconds: float,
    on_progress: Callable[[float], None] | None = None,
) -> str:
    """跟 run() 一样跑 ffmpeg，但额外加 -progress pipe:1，流式解析进度，
    每读到一条 out_time_ms 就换算成 0.0~1.0 的比例回调 on_progress。

    坑：ffmpeg 的 out_time_ms 字段名字带 "ms"，但实际单位是微秒（众所周知的
    ffmpeg 老 bug/历史遗留），所以换算要除以 1_000_000 而不是 1_000。
    """
    # argv 只拼一次，Popen 与出错消息共用同一个变量。原来出错分支自己重建了一遍
    # 字符串、且漏了 -progress pipe:1，报出来的命令不是真正跑的那条。
    argv = [FFMPEG, *args, "-progress", "pipe:1"]
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    for line in process.stdout:
        line = line.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key == "out_time_ms":
            try:
                microseconds = int(value)
            except ValueError:
                continue
            if on_progress is not None and total_seconds > 0:
                fraction = microseconds / 1_000_000 / total_seconds
                on_progress(max(0.0, min(1.0, fraction)))
        elif key == "progress" and value == "end":
            if on_progress is not None:
                on_progress(1.0)

    stderr = process.stderr.read()
    returncode = process.wait()
    if returncode != 0:
        raise FFmpegError(
            f"ffmpeg 执行失败（退出码 {returncode}）：{shlex.join(argv)}\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(stderr)}"
        )
    return stderr
