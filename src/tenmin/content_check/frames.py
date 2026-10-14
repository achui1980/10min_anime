"""按真实播放时间抽取保持显示比例的截图。"""

import math
import re
from pathlib import Path

from tenmin.render.ffmpeg import FFmpegError, probe_duration, run

from .models import ContentCheckError, SampledFrame

_FRAME_LOG = re.compile(r"\[Parsed_showinfo_\d+\s+@[^]]+\]\s+n:\s*\d+\s")
_PTS = re.compile(r"\bpts_time:\s*(\S+)")


def frame_filter(interval: float) -> str:
    """先采样，再按旋转后的 DAR 缩放为方形像素，最后记录真实 PTS。"""
    if not math.isfinite(interval) or interval <= 0:
        raise ContentCheckError("抽帧 interval 必须是有限正秒数")
    return (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',"
        "scale=640:trunc(640/dar/2)*2,setsar=1,showinfo"
    )


def parse_pts(stderr: str) -> list[float]:
    """只读 showinfo 的帧日志；坏值或时间倒退不能静默丢弃。"""
    times: list[float] = []
    for line in stderr.splitlines():
        if _FRAME_LOG.search(line) is None:
            continue
        match = _PTS.search(line)
        if match is None:
            raise ContentCheckError(f"抽帧日志缺少 PTS：{line}")
        raw = match.group(1)
        try:
            time = float(raw)
        except ValueError as error:
            raise ContentCheckError(f"抽帧 PTS 不是有效秒数：{raw!r}") from error
        if not math.isfinite(time) or time < 0:
            raise ContentCheckError(f"抽帧 PTS 必须是有限非负秒数：{raw!r}")
        if times and time < times[-1]:
            raise ContentCheckError(f"抽帧 PTS 倒退：{times[-1]} → {time}")
        times.append(time)
    return times


def extract_frames(
    video: Path, destination: Path, *, interval: float, ffmpeg: str, ffprobe: str,
) -> list[SampledFrame]:
    """写入调用方隔离的空目录，返回按文件序号对齐的播放相对时间。

    普通 -i（不传 -copyts）由 FFmpeg 去掉容器起点，并保留音视频起点差；
    不减首采样帧时间。默认 autorotate 在滤镜前处理显示方向与 SAR。
    中途失败的截图留在临时目录，由调用方管理，不能作为成功产物发布。
    """
    filters = frame_filter(interval)
    try:
        duration = probe_duration(video, ffprobe=ffprobe)
        if not math.isfinite(duration) or duration <= 0:
            raise ContentCheckError(f"视频时长必须是有限正秒数：{duration}")
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise ContentCheckError(f"抽帧需要隔离的空目录：{destination}")
        args = [
            "-hide_banner", "-nostdin", "-y", "-i", str(video),
            "-map", "0:v:0", "-an", "-vf", filters, "-fps_mode", "vfr",
            "-q:v", "2", str(destination / "frame_%06d.jpg"),
        ]
        stderr = run(args, ffmpeg=ffmpeg)
        paths = sorted(destination.glob("frame_*.jpg"))
    except (FFmpegError, OSError) as error:
        raise ContentCheckError(f"内容检查抽帧失败（{video}）：{error}") from error
    times = parse_pts(stderr)
    if not paths:
        raise ContentCheckError(f"内容检查未抽取到任何帧：{video}")
    if len(paths) != len(times):
        raise ContentCheckError(
            f"抽帧文件数与 PTS 数不一致：{len(paths)} 帧 / {len(times)} 条 PTS"
        )
    return [SampledFrame(path, time) for path, time in zip(paths, times, strict=True)]
