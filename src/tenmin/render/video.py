"""视频渲染：一次 ffmpeg 调用完成 trim + concat + 烧字幕 + 挂音轨。

视频只编码一次。编两次等于多掉一次画质、多等几分钟。
代价是烧字幕出错时要连带重跑切片拼接——接受。
"""

from __future__ import annotations

from pathlib import Path

from tenmin.models import Timeline
from tenmin.render.ffmpeg import run

WIDTH = 1920
HEIGHT = 1080
PIX_FMT = "yuv420p"
CRF = "20"
PRESET = "medium"
VIDEOTOOLBOX_BITRATE = "6000k"


def escape_filter_path(path: Path) -> str:
    """filtergraph 里的路径整体用单引号包住，反斜杠与单引号再转义一次。

    源片名常带方括号和空格（`[LoliHouse] ... .mkv`），单引号包住就不用逐字符转义。
    """
    text = str(path).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def quality_args(encoder: str) -> list[str]:
    """videotoolbox 不认 -crf/-preset，只能给码率。"""
    if encoder.endswith("videotoolbox"):
        return ["-b:v", VIDEOTOOLBOX_BITRATE]
    return ["-crf", CRF, "-preset", PRESET]


def build_render_args(
    *,
    video: Path,
    timeline: Timeline,
    audio: Path,
    ass: Path,
    out_path: Path,
    encoder: str,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> list[str]:
    """拼出渲染用的 ffmpeg 参数列表（不含 ffmpeg 本身）。"""
    if not timeline.segments:
        raise ValueError("timeline 里没有任何 segment，无法渲染")

    parts: list[str] = []
    for index, segment in enumerate(timeline.segments):
        parts.append(
            f"[0:v]trim=start={segment.source_start:.3f}:end={segment.source_end:.3f},"
            f"setpts=PTS-STARTPTS,scale={width}:{height},setsar=1[v{index}]"
        )
    labels = "".join(f"[v{i}]" for i in range(len(timeline.segments)))
    parts.append(f"{labels}concat=n={len(timeline.segments)}:v=1:a=0[vcat]")
    parts.append(f"[vcat]subtitles=filename={escape_filter_path(ass)}[vout]")

    return [
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[vout]",
        "-map",
        "1:a",
        "-c:v",
        encoder,
        *quality_args(encoder),
        "-pix_fmt",
        PIX_FMT,
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(out_path),
    ]


def render_video(
    *,
    video: Path,
    timeline: Timeline,
    audio: Path,
    ass: Path,
    out_path: Path,
    encoder: str,
) -> Path:
    """真跑 ffmpeg 渲染，返回成品路径。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        build_render_args(
            video=video,
            timeline=timeline,
            audio=audio,
            ass=ass,
            out_path=out_path,
            encoder=encoder,
        )
    )
    return out_path
