"""视频渲染：一次 ffmpeg 调用完成 trim + concat + 烧字幕 + 挂音轨。

视频只编码一次。编两次等于多掉一次画质、多等几分钟。
代价是烧字幕出错时要连带重跑切片拼接——接受。
"""

from __future__ import annotations

from pathlib import Path

from tenmin.atomic import atomic_path
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


def escape_filter_arg(value: str) -> str:
    """把任意字符串包成 filtergraph 里安全的一个 AVOption 值（含外层单引号）。

    值要过**两层**反转义，两层的规则不一样，这是这个函数唯一的难点：

    - **层 1：filtergraph 描述解析器。** 用 av_get_token 找 filter 参数的结尾，终止符
      是 `[],;`。它对引号的处理是「见到 `'` 就一路原样拷贝到下一个 `'`」——注意是
      *原样*，引号里的反斜杠**不**在这一层被消耗。代价是引号里放不进字面单引号。
    - **层 2：filter 自己的 AVOption 分词器。** 拿到层 1 的输出，按 `:` 切 key=value，
      同样用 av_get_token，这一层才会把 `\\X` 还原成 `X`、把裸的 `'` 当引号吃掉。

    所以规则是「先按层 2 转义，再用层 1 的引号把结果整体裹住」：

    1. 给层 2：`\\` → `\\\\`、`'` → `\\'`、`:` → `\\:`
    2. 给层 1：把上一步结果里剩下的字面 `'` 写成 `'\\''`（闭引号 + 转义引号 + 重开引号）

    于是一个字面单引号最终展开成 `\\'\\''`：`\\'` 是给层 2 的，`'...'` 的开合是给层 1 的。

    空格、`[]`、`,`、`;`、中文都不用管：层 1 的引号已经护住它们，层 2 也不拿它们当分隔符
    （实测 ffmpeg 9.0.1 这几类本来就 rc=0）。留着引号还顺带护住路径首尾的空白。

    实测（ffmpeg 9.0.1，真跑 subtitles 与 drawtext）：
    - 只做第 1 步 → 冒号让层 1 就地报 `Error parsing a filter description`
    - 只做第 2 步 → `\\` 会被层 2 吃掉，`a\\b` 变成 `ab`
    - 两步都做 → `' : \\ [ ] , ; = % 中文` 全部 rc=0

    已知局限：层 2 会掐掉**值本身**首尾的空白（首尾要保住得在层 2 再套一层引号）。
    路径不受影响（以 `/` 开头、以 `.ass` 结尾），drawtext 文案首尾空白会丢，但居中
    渲染时看不出来，所以不为它把转义再复杂化。
    """
    inner = value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    return "'" + inner.replace("'", "'\\''") + "'"


def escape_filter_path(path: Path) -> str:
    """filtergraph 里的文件路径。源片名常带方括号和空格，撇号目录（`O'Brien/`）也不罕见。"""
    return escape_filter_arg(str(path))


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
        #
        # font / text 三个值全部来自 config（outro_title 由 cfg.show 拼出、
        # outro_message 是 render.outro_message、字体名是 render.outro_font_name），
        # 所以一律走 escape_filter_arg —— 它已经含外层单引号，别再自己补一对。
        # `%` 不靠转义：drawtext 默认按 strftime 展开 `%`，靠末尾的 expansion=none
        # 关掉（实测 `%Y-%m-%d` 与 textfile= 的基准真值像素逐字节相同）。
        parts.append(f"color=c=black:s={width}x{height}:d={outro_seconds:.3f}[cardbg]")
        font = escape_filter_arg(OUTRO_FONT_NAME)
        title = escape_filter_arg(outro_title)
        message = escape_filter_arg(outro_message)
        parts.append(
            f"[cardbg]drawtext=font={font}:text={title}:fontcolor=yellow:"
            "bordercolor=black:borderw=4:fontsize=64:x=(w-text_w)/2:y=(h-text_h)/2-60:"
            "expansion=none[card1]"
        )
        parts.append(
            f"[card1]drawtext=font={font}:text={message}:fontcolor=yellow:"
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
    """真跑 ffmpeg 渲染，返回成品路径。

    跟 mix_audio 一样先落同目录的 `.part` 再原子改名 —— 一次渲染要几分钟，中途
    Ctrl-C 留下的截断 mp4 mtime 最新，_is_fresh 会把它当成品跳过（见 atomic 模块）。
    """
    reporter = reporter or NullProgressReporter()
    total_seconds = timeline.total_seconds + outro_seconds

    def _on_progress(fraction: float) -> None:
        reporter.substep("render", int(fraction * 100), 100, "")

    with atomic_path(out_path) as part:
        args = build_render_args(
            video=video,
            timeline=timeline,
            audio=audio,
            ass=ass,
            out_path=part,
            encoder=encoder,
            width=width,
            height=height,
            fade_out_seconds=fade_out_seconds,
            outro_seconds=outro_seconds,
            outro_title=outro_title,
            outro_message=outro_message,
        )
        run_with_progress(
            args, total_seconds=total_seconds, on_progress=_on_progress, ffmpeg=ffmpeg
        )
    return out_path
