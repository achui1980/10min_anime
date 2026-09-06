"""ffmpeg / ffprobe 子进程封装。解析部分是纯函数，单独测。"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

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


def run(args: list[str]) -> str:
    """跑 ffmpeg，返回 stderr（ffmpeg 的进度与日志都在 stderr）。"""
    completed = subprocess.run([FFMPEG, *args], capture_output=True, text=True)
    if completed.returncode != 0:
        raise FFmpegError(
            f"ffmpeg 退出码 {completed.returncode}，命令：\n"
            f"{FFMPEG} {' '.join(args)}\n\n"
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
    completed = subprocess.run(args, capture_output=True, text=True)
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
    completed = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True)
    return parse_names(completed.stdout)


@lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    completed = subprocess.run(
        [FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True
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
