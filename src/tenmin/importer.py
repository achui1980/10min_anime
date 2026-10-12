"""`tenmin import`：扫描一个目录，把视频与同名 SRT 配对、识别集号、预览 OP/ED 区间。

只负责「算出要导入什么」和「按用户确认过的计划登记」，不启动流水线 —— 那是 cli 的事。
"""

from __future__ import annotations

import re
import unicodedata
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


@dataclass(frozen=True)
class Ambiguous:
    """文件名里剩下不止一个候选集号（升序去重），交给 LLM 兜底。"""

    candidates: tuple[int, ...]


# 第一级强模式。全部在 NFKC 归一（全角数字/字母 → 半角）之后匹配。
# `E` 前面不许是字母，挡掉 `WEB`、`THE01` 这类误命中；S01E01 也会被第二条命中，
# 两条给出同一个数字，去重后不构成歧义。
_STRONG_PATTERNS = (
    re.compile(r"S[0-9]{1,2}E([0-9]{1,3})(?![0-9])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])EP?([0-9]{1,3})(?![0-9])", re.IGNORECASE),
    re.compile(r"第\s*([0-9]{1,3})\s*[集話话]"),
)

# 第二级之前先剔掉的噪声 token：分辨率、编码、位深、音频编码（含声道写法 AAC2.0）、
# 四位年份。两侧不许紧挨字母数字，免得啃掉标题的一部分。
_NOISE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[0-9]{3,4}p|4k|[xh]\.?26[45]|avc|hevc|(?:aac|flac|opus)(?:[0-9]\.[0-9])?"
    r"|[0-9]{1,2}bit|(?:19|20)[0-9]{2}"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# 第二级的分隔符：spec 列的 `-`、空白、`[`、`]`、`.`、`_`，外加中日文命名常见的
# 全角/半角圆括号与 `【】`（只会让更多「被分隔符包住的数字」露出来，不会制造新的歧义判定）。
_WEAK_SEPARATORS = re.compile(r"[-\s\[\]._()（）【】]+")
_WEAK_NUMBER = re.compile(r"[0-9]{1,3}")

# 非正片素材。ASCII 关键词两侧不许是字母（`Spy` 不算 SP、`Nova` 不算 OVA），
# 后面跟数字照样算（NCED1、SP01）。
_EXTRA = re.compile(
    r"(?<![A-Za-z])(?:NCOP|NCED|SP|OVA|PV|Menu)(?![A-Za-z])|预告|預告",
    re.IGNORECASE,
)


def _normalize(stem: str) -> str:
    return unicodedata.normalize("NFKC", stem)


def is_extra(stem: str) -> bool:
    """文件名主体是不是 NCOP / NCED / SP / OVA / PV / 预告 / Menu 这类非正片素材。"""
    return _EXTRA.search(_normalize(stem)) is not None


def _verdict(numbers: list[int]) -> int | Ambiguous | None:
    unique = sorted(set(numbers))
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return Ambiguous(tuple(unique))


def parse_episode_number(stem: str) -> int | Ambiguous | None:
    """从文件名主体里认集号。纯函数，完全离线。

    强模式（E01 / EP01 / S01E01 / 第01集 / 第01話）任一命中就只看强模式；全部未命中
    才走弱模式：剔掉噪声 token 后，被分隔符包围的 1–3 位数字。剩一个 → 集号；
    一个都没有 → None；不止一个 → Ambiguous。**绝不**按排序顺序补集号。
    """
    text = _normalize(stem)
    strong = [int(m.group(1)) for pattern in _STRONG_PATTERNS for m in pattern.finditer(text)]
    if strong:
        return _verdict(strong)
    cleaned = _NOISE.sub(" ", text)
    weak = [
        int(token) for token in _WEAK_SEPARATORS.split(cleaned) if _WEAK_NUMBER.fullmatch(token)
    ]
    return _verdict(weak)
