"""ffmpeg / ffprobe 子进程封装。解析部分是纯函数，单独测。"""

from __future__ import annotations

import re

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
STDERR_TAIL_LINES = 30

# ffmpeg -filters / -encoders 每行形如 "  T.. ass  V->V  描述"，
# 标志列只由大写字母和点组成，名字是紧跟其后的第一个 token。
_NAME_LINE = re.compile(r"^\s*[A-Z.]{3,6}\s+(\S+)\s")


class FFmpegError(RuntimeError):
    """ffmpeg 非零退出。消息里必须带 stderr 尾部，否则等于没报错。"""


def parse_names(text: str) -> set[str]:
    """从 -filters / -encoders 的输出里抽出可用名字。"""
    return {m.group(1) for m in (_NAME_LINE.match(line) for line in text.splitlines()) if m}


def tail(text: str, lines: int = STDERR_TAIL_LINES) -> str:
    """取末尾若干行。ffmpeg 的真实错误永远在 stderr 尾部。"""
    stripped = text.rstrip("\n")
    if not stripped:
        return ""
    return "\n".join(stripped.splitlines()[-lines:])
