"""独立遮罩：给任意视频按标注打上整屏高斯模糊（不经过 tenmin 的成片管线）。

**为什么要有这条独立路径**：遮罩最初只是 render 阶段的一个环节 —— 只对 tenmin 自己
剪出来的成片有效（那条路上遮罩区间用的是「成片时间轴」，而且 render 硬要求该集已有
timeline.json / mixed.m4a / .ass）。但「给一段视频打码」本身是个跟 tenmin 毫无关系的
通用需求：手上有一段素材、知道哪几秒不能给别人看，就得能直接糊掉。

**时间轴**：这条路上的区间用的是**这个视频自己的时间轴**。实现上用普通 `-i`（不传
`-copyts`、不加 `setpts`），ffmpeg 会按容器的 start_time 把第一帧归零 —— 跟
`content_check/frames.py` 抽帧那条路（同样是普通 `-i`）完全一致。所以 `tenmin
check-content` 出的报告里每帧的 `time` 跟这里的秒数是同一条轴：「标哪一帧就糊哪一帧」。

**跟集成路径的关系**：滤镜长什么样只有一份定义（`mask.blur_filter`），集成路径
（`render/video.py`）和这里共用，两边的模糊效果不会走样。区别只在时间轴：集成路径的
区间是成片时间轴，这里是这个视频自己的时间轴 —— 所以从成片的报告里导出的标注拿到这里
用会对不上，CLI 会就这件事给一句警告。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from tenmin.atomic import atomic_path
from tenmin.mask import (
    DEFAULT_SIGMA,
    MaskError,
    MaskRange,
    MaskSpec,
    blur_filter,
    load_mask,
    merge_ranges,
)
from tenmin.render.ffmpeg import (
    FFMPEG,
    FFPROBE,
    FFmpegError,
    probe_duration,
    run_with_progress,
)
from tenmin.render.video import CRF, PRESET, quality_args

# 默认输出名后缀：同目录、同扩展名的 `<stem>.masked.<ext>`。
OUTPUT_SUFFIX = ".masked"

# 默认视频编码器。跟项目默认值一致（libx264），但**不**继承项目的 width/height/
# pix_fmt：这条路的定位是「一份打过码的副本」，画面参数跟着源片走。
DEFAULT_ENCODER = "libx264"

# 这些容器支持 `-movflags +faststart`（moov 前置，网页播放秒开）。
_FASTSTART_SUFFIXES = frozenset({".mp4", ".mov", ".m4v"})


class MaskApplyError(MaskError):
    """遮罩应用失败：源片不在、标注里没有区间、输出已存在、输出跟源片同路径。

    继承 MaskError（它又是 ValueError 的子类），所以 CLI 那张 PIPELINE_ERRORS 的网
    一网就住，不用再加一行。
    """


def default_output_path(video: Path) -> Path:
    """默认输出路径：同目录的 `<stem>.masked.<ext>`。"""
    video = Path(video)
    return video.with_name(f"{video.stem}{OUTPUT_SUFFIX}{video.suffix}")


def build_apply_args(
    video: Path,
    *,
    out_path: Path,
    ranges: Iterable[MaskRange],
    sigma: float = DEFAULT_SIGMA,
    encoder: str = DEFAULT_ENCODER,
    crf: str = CRF,
    preset: str = PRESET,
) -> list[str]:
    """拼 ffmpeg 命令行：给视频加上高斯模糊，音频原样带走。

    **只带一条视频轨和第一条音轨**：字幕轨不搬运 —— 硬字幕在画面里（跟着一起糊），
    软字幕轨在 mp4 里要么得转成 mov_text（换容器规则一堆）、要么原样复制（碰到
    ASS/SRT 轨直接失败），不值得在这个命令里冒这个险。

    **不传 `-pix_fmt`**：源片是什么像素格式就出什么，像素格式跟着源片走。硬塞
    yuv420p 会把 10-bit 源片悄悄降成 8-bit，那是「顺手改坏别人的东西」。
    """
    ranges = list(ranges)
    if not ranges:
        raise MaskApplyError("标注里没有任何遮罩区间，没什么可遮的")
    args = [
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        blur_filter(ranges, sigma),
        "-c:v",
        encoder,
        *quality_args(encoder, crf=crf, preset=preset),
        "-c:a",
        "copy",
    ]
    if Path(out_path).suffix.lower() in _FASTSTART_SUFFIXES:
        args += ["-movflags", "+faststart"]
    args.append(str(out_path))
    return args


def load_spec(mask_file: Path) -> MaskSpec:
    """读标注文件，缺文件/坏文件给一句人话。"""
    path = Path(mask_file)
    spec = load_mask(path)
    if spec is None:
        raise MaskApplyError(f"找不到遮罩标注文件：{path}")
    return spec


def apply_mask(
    video: Path,
    mask_file: Path,
    *,
    out_path: Path | None = None,
    sigma: float | None = None,
    encoder: str = DEFAULT_ENCODER,
    crf: str = CRF,
    preset: str = PRESET,
    force: bool = False,
    ffmpeg: str = FFMPEG,
    ffprobe: str = FFPROBE,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """按标注给视频打码，返回输出路径。

    `sigma` 为 None 时用标注文件里记的值；给了就以命令行为准。
    """
    video = Path(video)
    if not video.is_file():
        raise MaskApplyError(f"找不到要处理的视频：{video}")
    spec = load_spec(mask_file)
    ranges = merge_ranges(spec.ranges)
    if not ranges:
        raise MaskApplyError("标注里没有任何遮罩区间，没什么可遮的")
    effective_sigma = spec.sigma if sigma is None else sigma
    if effective_sigma <= 0:
        raise MaskApplyError(f"模糊强度 sigma 必须大于 0（收到 {effective_sigma}）")

    target = Path(out_path) if out_path is not None else default_output_path(video)
    if target.resolve() == video.resolve():
        raise MaskApplyError("输出不能跟源片是同一个文件")
    if target.exists() and not force:
        raise MaskApplyError(f"输出已存在：{target}；要覆盖请加 --force")

    try:
        total_seconds = probe_duration(video, ffprobe=ffprobe)
    except (FFmpegError, OSError):
        total_seconds = 0.0
    with atomic_path(target) as part:
        args = build_apply_args(
            video,
            out_path=part,
            ranges=ranges,
            sigma=effective_sigma,
            encoder=encoder,
            crf=crf,
            preset=preset,
        )
        run_with_progress(
            args,
            total_seconds=total_seconds,
            on_progress=on_progress,
            ffmpeg=ffmpeg,
        )
    return target
