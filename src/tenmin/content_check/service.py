"""Local checking with reusable scores and results-last publication.

Only the versioned ownership marker authorizes replacement of the fixed frames
directory. Cache data never selects paths to delete. Concurrent writers to the
same output directory are not supported.
"""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from pydantic import ValidationError

from tenmin.atomic import write_text
from tenmin.config import ContentCheckConfig

from .candidates import screen_frames
from .frames import extract_frames
from .model import (
    MODEL,
    PREPROCESSING_VERSION,
    REPO,
    REVISION,
    SAMPLING_VERSION,
    load_model_files,
    score_frames,
)
from .models import CacheIdentity, CheckManifest, CheckOutcome, ContentCheckError
from .report import render_report

_MARKER = ".tenmin-content-check"
_OWNERSHIP = "1\n"


def _regular(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _images_exist(output_dir: Path, manifest: CheckManifest) -> bool:
    frames = output_dir / "frames"
    if frames.is_symlink() or not frames.is_dir():
        return False
    root = frames.resolve()
    return all(
        _regular(output_dir / frame.image)
        and (output_dir / frame.image).resolve().is_relative_to(root)
        for frame in manifest.frames
    )


def load_cached(output_dir: Path, identity: CacheIdentity) -> CheckManifest | None:
    """Malformed/incomplete disk data is a miss; never read a FIFO."""
    try:
        if output_dir.is_symlink() or not _regular(output_dir / "results.json"):
            return None
        cached = CheckManifest.model_validate_json(
            (output_dir / "results.json").read_text(encoding="utf-8"),
        )
        if cached.identity != identity or not _images_exist(output_dir, cached):
            return None
        return cached
    except (OSError, UnicodeError, ValidationError):
        return None


def _prepare_output(output_dir: Path) -> None:
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise ContentCheckError(f"输出目录不是普通目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / _MARKER
    if any(output_dir.iterdir()):
        try:
            owned = _regular(marker) and marker.read_text(encoding="utf-8") == _OWNERSHIP
        except UnicodeDecodeError as error:
            raise ContentCheckError(
                f"所有权标记不是有效 UTF-8；本次检查失败／报告未更新：{marker}",
            ) from error
        if not owned:
            raise ContentCheckError(f"非本工具拥有的非空输出目录：{output_dir}")
    else:
        # Isolated staging also avoids overwriting a user's unrelated .part file.
        with TemporaryDirectory(prefix="tenmin-content-check-", dir=output_dir.parent) as temp:
            staged = Path(temp) / _MARKER
            write_text(staged, _OWNERSHIP, encoding="utf-8")
            staged.replace(marker)
    frames = output_dir / "frames"
    if frames.is_symlink() or (frames.exists() and not frames.is_dir()):
        raise ContentCheckError(f"截图目录不是普通目录：{frames}")


def _video_stat(video: Path) -> tuple[str, int, int]:
    if not video.is_file():
        raise ContentCheckError(f"视频文件不存在或不是普通文件：{video}")
    stat = video.stat()
    return str(video.resolve()), stat.st_size, stat.st_mtime_ns


def _publish(
    output_dir: Path, manifest: CheckManifest, html: str, staging: Path,
    *, video: Path, replace_frames: bool,
) -> None:
    # Prepare both payloads first. Until the final JSON rename commits, failures
    # after invalidation leave results absent; post-commit cleanup can still fail.
    report = staging / "report.html"
    results = staging / "results.json"
    write_text(report, html, encoding="utf-8")
    write_text(results, manifest.model_dump_json(indent=2), encoding="utf-8")
    _prepare_output(output_dir)  # Recheck ownership/symlinks before destructive I/O.
    identity = manifest.identity
    if _video_stat(video) != (
        identity.video_path, identity.video_size, identity.video_mtime_ns,
    ):
        raise ContentCheckError("视频身份在检查中变化；本次检查失败／报告未更新")
    (output_dir / "results.json").unlink(missing_ok=True)
    if replace_frames:
        frames = output_dir / "frames"
        if frames.exists():
            shutil.rmtree(frames)
        (staging / "frames").replace(frames)
    report.replace(output_dir / "report.html")
    results.replace(output_dir / "results.json")


def check_video(
    video: Path, output_dir: Path, *, cfg: ContentCheckConfig,
    ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", force: bool = False,
    offline: bool = False,
) -> CheckOutcome:
    """Check a video or reuse scores; expected I/O failures never return success."""
    try:
        video_path, size, mtime = _video_stat(video)
        _prepare_output(output_dir)
    except (OSError, UnicodeError) as error:
        raise ContentCheckError(f"本次检查失败／报告未更新：{error}") from error
    files = load_model_files(offline=offline)
    identity = CacheIdentity(
        video_path=video_path, video_size=size, video_mtime_ns=mtime,
        repository=REPO, revision=REVISION, model=MODEL, model_sha256=files.sha256,
        preprocessing_version=PREPROCESSING_VERSION, sampling_version=SAMPLING_VERSION,
        sample_interval=cfg.sample_interval,
    )
    cached = None if force else load_cached(output_dir, identity)
    try:
        if cached is not None:
            selected, ranges = screen_frames(cached.frames, cfg)
            same_thresholds = (
                cached.sensitive_threshold == cfg.sensitive_threshold
                and cached.adult_threshold == cfg.adult_threshold
            )
            same_candidates = (
                cached.candidate_count == len(selected) and cached.candidate_ranges == ranges
            )
            if same_thresholds and same_candidates and _regular(output_dir / "report.html"):
                if _video_stat(video) != (video_path, size, mtime):
                    raise ContentCheckError("视频身份在检查中变化；本次检查失败／报告未更新")
                return CheckOutcome("cached", len(selected), len(cached.frames),
                                    output_dir / "report.html", output_dir / "results.json")
        with TemporaryDirectory(prefix="tenmin-content-check-", dir=output_dir.parent) as temp:
            staging = Path(temp)
            if cached is None:
                start = perf_counter()
                sampled = extract_frames(video, staging / "frames", interval=cfg.sample_interval,
                                         ffmpeg=ffmpeg, ffprobe=ffprobe)
                extraction_seconds = perf_counter() - start
                if not sampled:
                    raise ContentCheckError("未抽取到任何帧；本次检查失败／报告未更新")
                start = perf_counter()
                frames = score_frames(sampled, files)
                inference_seconds = perf_counter() - start
                selected, ranges = screen_frames(frames, cfg)
                manifest = CheckManifest(
                    identity=identity, providers=["CPUExecutionProvider"],
                    extraction_seconds=extraction_seconds, inference_seconds=inference_seconds,
                    sensitive_threshold=cfg.sensitive_threshold,
                    adult_threshold=cfg.adult_threshold,
                    frames=frames, candidate_count=len(selected), candidate_ranges=ranges,
                )
                if not _images_exist(staging, manifest):
                    raise ContentCheckError("抽帧图片缺失或路径无效；本次检查失败／报告未更新")
            else:
                manifest = CheckManifest.model_validate({
                    **cached.model_dump(), "sensitive_threshold": cfg.sensitive_threshold,
                    "adult_threshold": cfg.adult_threshold,
                    "candidate_count": len(selected), "candidate_ranges": ranges,
                })
            html = render_report(manifest)
            if _video_stat(video) != (video_path, size, mtime):
                raise ContentCheckError("视频身份在检查中变化；本次检查失败／报告未更新")
            _publish(output_dir, manifest, html, staging,
                     video=video, replace_frames=cached is None)
        return CheckOutcome("completed" if cached is None else "report_rebuilt",
                            manifest.candidate_count, len(manifest.frames),
                            output_dir / "report.html", output_dir / "results.json")
    except OSError as error:
        raise ContentCheckError(f"本次检查失败／报告未更新：{error}") from error
