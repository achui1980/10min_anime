"""`tenmin ocr`：脱离项目、批量把视频的画面硬字幕认成 SRT。

跟管线里的 OCR 那一岔共用识别本身（`ocr.recognize_cues`），其余全不沾：不读
project.yaml、不剔 OP/ED staff 字、不进 01_dialogue。默认繁转简（OpenCC t2s，跟
ingest 同一个转换器），`--traditional` 保留原文。

输出名带语言后缀（`<stem>.zh-Hans.srt` / `<stem>.zh-Hant.srt`），两种都要的话分两次跑
也不会互相覆盖。复用判据跟管线的 `.ocr.srt` 一样：非空且不比视频旧就跳过，`--force` 重跑。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from tenmin import atomic
from tenmin.config import DEFAULT_RENDER, AsrConfig, LLMConfig, OcrConfig
from tenmin.ingest import ocr
from tenmin.ingest.asr import render_srt
from tenmin.ingest.clean import to_simplified
from tenmin.ingest.ocr_correction import (
    correction_pending,
    is_human_edited,
    prepare_asr,
    record_fallback,
    select_corrected,
)
from tenmin.ingest.resolve import _is_usable_cache
from tenmin.render.ffmpeg import FFmpegError
from tenmin.script.llm import LLMProvider

VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi", ".ts"})

# 单个视频失败时记下来、接着跑下一个的异常族。OCRUnavailableError 是 OCRError 的子类，
# 但它意味着整台机器跑不了（没装 extra / 不是 macOS），在 run_batch 里先单独放行。
_PER_VIDEO_ERRORS = (ocr.OCRError, FFmpegError, FileNotFoundError, ValueError)


@dataclass
class BatchSummary:
    done: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def collect_videos(inputs: Iterable[Path]) -> list[Path]:
    """把命令行给的文件/目录展开成视频列表。

    目录只看一层、按扩展名（不分大小写）挑、按文件名排序；显式给的文件不看扩展名照收。
    同一个文件出现多次只留第一次。路径不存在抛 FileNotFoundError，最后一个都没有抛 ValueError。
    """
    videos: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        if item.is_dir():
            found = sorted(
                (p for p in item.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
                key=lambda p: p.name,
            )
        elif item.is_file():
            found = [item]
        else:
            raise FileNotFoundError(f"找不到 {item}")
        for video in found:
            key = video.resolve()
            if key not in seen:
                seen.add(key)
                videos.append(video)
    if not videos:
        known = " ".join(sorted(VIDEO_SUFFIXES))
        raise ValueError(f"没有找到要识别的视频（目录里只认 {known}）")
    return videos


def output_path(video: Path, out_dir: Path | None, *, simplified: bool) -> Path:
    """`<stem>.zh-Hans.srt`（简体）或 `<stem>.zh-Hant.srt`（繁体），放 out_dir 或视频旁边。"""
    suffix = ".zh-Hans.srt" if simplified else ".zh-Hant.srt"
    return (out_dir if out_dir is not None else video.parent) / f"{video.stem}{suffix}"


def run_batch(
    videos: Iterable[Path],
    *,
    out_dir: Path | None,
    simplified: bool,
    force: bool,
    ocr_config: OcrConfig,
    ffmpeg_path: str = DEFAULT_RENDER.ffmpeg_path,
    ffprobe_path: str = DEFAULT_RENDER.ffprobe_path,
) -> BatchSummary:
    """逐个识别并写 SRT。单个视频失败只记账、接着跑；OCRUnavailableError 直接抛出。"""
    summary = BatchSummary()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        dest = output_path(video, out_dir, simplified=simplified)
        if not force and _is_usable_cache(dest, video):
            print(f"  {dest.name} 已存在，跳过")
            summary.skipped.append(video)
            continue
        try:
            cues = ocr.recognize_cues(
                video, ocr=ocr_config, ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path
            )
        except ocr.OCRUnavailableError:
            raise
        except _PER_VIDEO_ERRORS as exc:
            print(f"  {video.name} 识别失败：{exc}")
            summary.failed.append((video, str(exc)))
            continue
        if simplified:
            cues = [cue.model_copy(update={"text": to_simplified(cue.text)}) for cue in cues]
        atomic.write_text(dest, render_srt(cues))
        print(f"  画面字幕识别完成，{len(cues)} 条 → {dest}")
        summary.done.append(video)
    return summary


async def run_batch_corrected(
    videos: Iterable[Path],
    *,
    out_dir: Path | None,
    simplified: bool,
    force: bool,
    ocr_config: OcrConfig,
    provider_factory: Callable[[], LLMProvider],
    llm_config: LLMConfig,
    asr_config: AsrConfig | None = None,
    ffmpeg_path: str = DEFAULT_RENDER.ffmpeg_path,
    ffprobe_path: str = DEFAULT_RENDER.ffprobe_path,
) -> BatchSummary:
    """独立 OCR 的可选校对：原稿缓存与交付物分开，失败使用原稿。

    给了 asr_config 就带日语转写校对，转写缓存写在 `<视频名>.ja.asr.srt`（与字幕同目录）。
    """
    summary = BatchSummary()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        dest = output_path(video, out_dir, simplified=simplified)
        raw = dest.with_name(f"{dest.stem}.raw.srt")
        state = dest.with_suffix(".ai-state.json")
        review = dest.with_suffix(".review.json")
        # 历史交付物缺少原稿和可信状态，必须先保护人工字幕。
        if dest.is_file() and not raw.is_file() and not force:
            summary.skipped.append(video)
            summary.warnings.append(
                f"{video.name}：现有 {dest.name} 无校对状态，已保留；"
                "备份后加 --force 重新识别"
            )
            continue
        try:
            if force or not _is_usable_cache(raw, video):
                cues = ocr.recognize_cues(
                    video, ocr=ocr_config, ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path
                )
                if simplified:
                    cues = [
                        cue.model_copy(update={"text": to_simplified(cue.text)}) for cue in cues
                    ]
                atomic.write_text(raw, render_srt(cues))
        except ocr.OCRUnavailableError:
            raise
        except _PER_VIDEO_ERRORS as exc:
            summary.failed.append((video, str(exc)))
            continue

        if force and dest.is_file() and not state.is_file():
            backup = dest.with_name(f"{dest.stem}.{dest.stat().st_mtime_ns}.bak.srt")
            atomic.copy_file(dest, backup)
            summary.warnings.append(f"{video.name}：现有字幕已备份至 {backup}")

        stat = video.stat()
        identity = f"{video.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        asr_path = None
        if asr_config is not None:
            asr_path, asr_warning = prepare_asr(
                video, dest.with_name(f"{video.stem}.ja.asr.srt"),
                asr_config=asr_config, ffmpeg_path=ffmpeg_path,
            )
            if asr_warning:
                summary.warnings.append(f"{video.name}：{asr_warning}")
        pending = correction_pending(
            raw, dest, state, cfg=llm_config, source_identity=identity, asr=asr_path
        )
        if not force and not pending:
            summary.skipped.append(video)
            continue
        chosen, warnings, _ = await select_corrected(
            raw,
            dest,
            state,
            review,
            provider_factory=provider_factory,
            cfg=llm_config,
            source_identity=identity,
            overwrite_unmanaged=force and not state.is_file(),
            asr=asr_path,
        )
        summary.warnings.extend(f"{video.name}：{message}" for message in warnings)
        if chosen == raw:
            if is_human_edited(dest, state):
                # 经人工修改且来源更新：先备份，再让交付路径指向本次真实原稿。
                backup = dest.with_name(f"{dest.stem}.{int(dest.stat().st_mtime_ns)}.bak.srt")
                atomic.copy_file(dest, backup)
                summary.warnings.append(f"{video.name}：旧人工字幕已备份至 {backup}")
            atomic.copy_file(raw, dest)
            record_fallback(
                raw, dest, state, cfg=llm_config, source_identity=identity, asr=asr_path
            )
        summary.done.append(video)
    return summary
