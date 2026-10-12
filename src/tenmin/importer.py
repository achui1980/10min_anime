"""`tenmin import`：扫描一个目录，把视频与同名 SRT 配对、识别集号、预览 OP/ED 区间。

只负责「算出要导入什么」和「按用户确认过的计划登记」，不启动流水线 —— 那是 cli 的事。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tenmin.ingest.ocr_batch import VIDEO_SUFFIXES

# 视频主体后面允许跟的语言后缀。顺序就是同一视频命中多个候选 SRT 时的优先级
# （主体完全相等的排在它们全部之前）。必须放开：`tenmin ocr` 自己的产物就是
# `<stem>.zh-Hans.srt` / `<stem>.zh-Hant.srt`，下载来的字幕也普遍带这类后缀。
LANGUAGE_SUFFIXES = ("zh-Hans", "zh-Hant", "sc", "tc", "chs", "cht", "zh", "ja", "jp", "jpn")
_LANGUAGE_SEPARATORS = (".", "_")


@dataclass(frozen=True)
class Pair:
    """一个视频与选中的那份 SRT；ignored_srts 是同样配得上、但优先级更低的候选。"""

    video: Path
    srt: Path
    ignored_srts: tuple[Path, ...] = ()


@dataclass
class Scan:
    pairs: list[Pair] = field(default_factory=list)
    unpaired_videos: list[Path] = field(default_factory=list)
    unpaired_srts: list[Path] = field(default_factory=list)


def _require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"找不到目录：{path}")
    if not path.is_dir():
        raise ValueError(f"不是目录：{path}")


def _srt_rank(video_stem: str, srt: Path) -> int | None:
    """这份 SRT 能不能配给这个视频；能就返回优先级（越小越优先），不能返回 None。

    大小写不敏感。只认「主体完全相等」与「主体 + 分隔符 + 一个语言后缀」两种形态，
    刻意不做模糊匹配：配错一集比少配一集难查得多。
    """
    stem = srt.name[: -len(".srt")].casefold()
    base = video_stem.casefold()
    if stem == base:
        return 0
    for rank, suffix in enumerate(LANGUAGE_SUFFIXES, start=1):
        for separator in _LANGUAGE_SEPARATORS:
            if stem == f"{base}{separator}{suffix.casefold()}":
                return rank
    return None


def scan_directory(video_dir: Path, srt_dir: Path | None = None) -> Scan:
    """单层枚举视频与 SRT 并配对。srt_dir 省略时从视频目录里找字幕。

    视频后缀复用 `ocr_batch.VIDEO_SUFFIXES`，字幕只认 `.srt`，两者都大小写不敏感。
    结果按文件名排序，保证预览与 LLM 兜底的输入顺序稳定。
    """
    _require_dir(video_dir)
    if srt_dir is not None:
        _require_dir(srt_dir)
    srt_root = srt_dir if srt_dir is not None else video_dir
    videos = sorted(
        (p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda p: p.name,
    )
    srts = sorted(
        (p for p in srt_root.iterdir() if p.is_file() and p.suffix.lower() == ".srt"),
        key=lambda p: p.name,
    )
    scan = Scan()
    claimed: set[Path] = set()
    for video in videos:
        ranked: list[tuple[int, Path]] = []
        for srt in srts:
            rank = _srt_rank(video.stem, srt)
            if rank is not None:
                ranked.append((rank, srt))
        if not ranked:
            scan.unpaired_videos.append(video)
            continue
        ranked.sort(key=lambda item: (item[0], item[1].name))
        scan.pairs.append(
            Pair(
                video=video,
                srt=ranked[0][1],
                ignored_srts=tuple(srt for _, srt in ranked[1:]),
            )
        )
        claimed.update(srt for _, srt in ranked)
    scan.unpaired_srts = [srt for srt in srts if srt not in claimed]
    return scan
