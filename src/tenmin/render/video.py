"""视频渲染：一次 ffmpeg 调用完成 trim + concat + 烧字幕 + 挂音轨。

视频只编码一次。编两次等于多掉一次画质、多等几分钟。
代价是烧字幕出错时要连带重跑切片拼接——接受。
"""

from __future__ import annotations

from pathlib import Path

from tenmin.config import DEFAULT_RENDER
from tenmin.models import Timeline
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.ffmpeg import run_with_progress

# 全部从 config 的默认值派生。WIDTH/HEIGHT 尤其重要：它必须跟
# render/subtitles.py 的 PlayResX/Y 同源，否则烧上去的字幕会被静默缩放。
WIDTH = DEFAULT_RENDER.width
HEIGHT = DEFAULT_RENDER.height
CRF = DEFAULT_RENDER.crf
PRESET = DEFAULT_RENDER.preset
VIDEOTOOLBOX_BITRATE = DEFAULT_RENDER.videotoolbox_bitrate
OUTRO_FONT_NAME = DEFAULT_RENDER.outro_font_name
PIX_FMT = "yuv420p"


def escape_filter_path(path: Path) -> str:
    """filtergraph 里的路径整体用单引号包住，反斜杠与单引号再转义一次。

    源片名常带方括号和空格（`[LoliHouse] ... .mkv`），单引号包住就不用逐字符转义。
    """
    text = str(path).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def escape_drawtext(text: str) -> str:
    """drawtext 的 text 参数整体用单引号包住，反斜杠与单引号需要转义。"""
    return text.replace("\\", "\\\\").replace("'", "\\'")


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
    fade_out_seconds: float = 0.0,
    outro_seconds: float = 0.0,
    outro_title: str = "",
    outro_message: str = "",
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

    final_label = "[vout]"
    if fade_out_seconds > 0:
        fade_start = max(timeline.total_seconds - fade_out_seconds, 0.0)
        parts.append(
            f"{final_label}fade=t=out:st={fade_start:.3f}:d={fade_out_seconds:.3f}[vfaded]"
        )
        final_label = "[vfaded]"
    if outro_seconds > 0:
        # 结尾黑卡：番剧名+集数在上，感谢语在下，样式跟正片字幕保持一致（黄字黑边）。
        parts.append(f"color=c=black:s={width}x{height}:d={outro_seconds:.3f}[cardbg]")
        title = escape_drawtext(outro_title)
        message = escape_drawtext(outro_message)
        parts.append(
            f"[cardbg]drawtext=font='{OUTRO_FONT_NAME}':text='{title}':fontcolor=yellow:"
            "bordercolor=black:borderw=4:fontsize=64:x=(w-text_w)/2:y=(h-text_h)/2-60:"
            "expansion=none[card1]"
        )
        parts.append(
            f"[card1]drawtext=font='{OUTRO_FONT_NAME}':text='{message}':fontcolor=yellow:"
            "bordercolor=black:borderw=4:fontsize=44:x=(w-text_w)/2:y=(h-text_h)/2+40:"
            "expansion=none[card]"
        )
        parts.append(f"{final_label}[card]concat=n=2:v=1:a=0[vfinal]")
        final_label = "[vfinal]"

    return [
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-filter_complex",
        ";".join(parts),
        "-map",
        final_label,
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
    width: int = WIDTH,
    height: int = HEIGHT,
    fade_out_seconds: float = 0.0,
    outro_seconds: float = 0.0,
    outro_title: str = "",
    outro_message: str = "",
    reporter: ProgressReporter | None = None,
    ffmpeg: str = DEFAULT_RENDER.ffmpeg_path,
) -> Path:
    """真跑 ffmpeg 渲染，返回成品路径。"""
    reporter = reporter or NullProgressReporter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = build_render_args(
        video=video,
        timeline=timeline,
        audio=audio,
        ass=ass,
        out_path=out_path,
        encoder=encoder,
        width=width,
        height=height,
        fade_out_seconds=fade_out_seconds,
        outro_seconds=outro_seconds,
        outro_title=outro_title,
        outro_message=outro_message,
    )
    total_seconds = timeline.total_seconds + outro_seconds

    def _on_progress(fraction: float) -> None:
        reporter.substep("render", int(fraction * 100), 100, "")

    run_with_progress(
        args, total_seconds=total_seconds, on_progress=_on_progress, ffmpeg=ffmpeg
    )
    return out_path
