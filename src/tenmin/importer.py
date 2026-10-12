"""`tenmin import`：扫描一个目录，把视频与同名 SRT 配对、识别集号、预览 OP/ED 区间。

只负责「算出要导入什么」和「按用户确认过的计划登记」，不启动流水线 —— 那是 cli 的事。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.ingest import av_match
from tenmin.ingest.normalize import build_track, credit_range_source
from tenmin.ingest.ocr_batch import VIDEO_SUFFIXES
from tenmin.pipeline import _source_duration, register_episode
from tenmin.render.ffmpeg import FFmpegBinaryError, FFmpegError
from tenmin.script.llm import LLMProvider

# 视频主体后面允许跟的语言后缀。顺序就是同一视频命中多个候选 SRT 时的优先级
# （主体完全相等的排在它们全部之前）。必须放开：`tenmin ocr` 自己的产物就是
# `<stem>.zh-Hans.srt` / `<stem>.zh-Hant.srt`，下载来的字幕也普遍带这类后缀。
LANGUAGE_SUFFIXES = ("zh-Hans", "zh-Hant", "sc", "tc", "chs", "cht", "zh", "ja", "jp", "jpn")
_LANGUAGE_SEPARATORS = (".", "_")
REFERENCE_SIDES = ("op", "ed")
REFERENCE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
SOURCE_LABELS = {"av": "画面+音频", "subtitle": "字幕推断"}
OP_SEARCH_SECONDS = 300
ED_SEARCH_SECONDS = 180


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
    references: dict[str, Path] = field(default_factory=dict)


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


def _find_references(video_dir: Path) -> dict[str, Path]:
    """只从视频目录顶层找 OP/ED 参考截图，每侧最多一张。"""
    candidates: dict[str, list[Path]] = {side: [] for side in REFERENCE_SIDES}
    for path in video_dir.iterdir():
        side = path.stem.casefold()
        if path.is_file() and side in candidates and path.suffix.lower() in REFERENCE_SUFFIXES:
            candidates[side].append(path)
    references: dict[str, Path] = {}
    for side, paths in candidates.items():
        if len(paths) > 1:
            names = "、".join(sorted(path.name for path in paths))
            raise ValueError(f"{side.upper()} 参考截图不止一张（{names}），只留一张再导入")
        if paths:
            references[side] = paths[0]
    return references


def scan_directory(video_dir: Path, srt_dir: Path | None = None) -> Scan:
    """单层枚举视频与 SRT 并配对。srt_dir 省略时从视频目录里找字幕。

    视频后缀复用 `ocr_batch.VIDEO_SUFFIXES`，字幕只认 `.srt`，两者都大小写不敏感。
    结果按文件名排序，保证预览与 LLM 兜底的输入顺序稳定。
    OP/ED 参考截图只从 video_dir 顶层查找，不查 srt_dir，也不进入未配对列表。
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
    scan.references = _find_references(video_dir)
    return scan


@dataclass(frozen=True)
class Ambiguous:
    """文件名里剩下不止一个候选集号（升序去重），交给 LLM 兜底。"""

    candidates: tuple[int, ...]


# 匹配任何模式之前先剔掉的方括号 CRC32 校验码（`[E3A1B2C4]`）：它以 `E`+数字开头时
# 会被强模式当成集号，纯数字时又会在弱模式里多出一个候选。
_CRC_TAG = re.compile(r"\[[0-9A-Fa-f]{8}\]")

# 集号数字后面允许跟的版本后缀（`03v2` = 第 3 集的第 2 版）。
_VERSION = r"(?:v[0-9]+)?"

# 第一级强模式。全部在 NFKC 归一（全角数字/字母 → 半角）之后匹配。
# `E` 前面不许是字母，挡掉 `WEB`、`THE01` 这类误命中；数字（含版本后缀）后面不许紧跟
# 字母数字，挡掉没加方括号的十六进制串（`E3A1`）。S01E01 也会被第二条命中，
# 两条给出同一个数字，去重后不构成歧义。
_STRONG_PATTERNS = (
    re.compile(rf"S[0-9]{{1,2}}E([0-9]{{1,3}}){_VERSION}(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(rf"(?<![A-Za-z])EP?([0-9]{{1,3}}){_VERSION}(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"第\s*([0-9]{1,3})\s*[集話话]"),
)

# 第二级之前先剔掉的噪声 token：分辨率、编码、位深、音频编码（含声道写法 AAC2.0、
# DDP5.1）、独立的声道写法（5.1ch）、四位年份。两侧不许紧挨字母数字，免得啃掉标题的一部分。
_NOISE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[0-9]{3,4}p|4k|[xh]\.?26[45]|avc|hevc|(?:aac|flac|opus|ddp?)(?:[0-9]\.[0-9])?"
    r"|[0-9]\.[0-9]ch"
    r"|[0-9]{1,2}bit|(?:19|20)[0-9]{2}"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# 第二级的分隔符：spec 列的 `-`、空白、`[`、`]`、`.`、`_`，外加中日文命名常见的
# 全角/半角圆括号与 `【】`（只会让更多「被分隔符包住的数字」露出来，不会制造新的歧义判定）。
_WEAK_SEPARATORS = re.compile(r"[-\s\[\]._()（）【】]+")
_WEAK_NUMBER = re.compile(rf"([0-9]{{1,3}}){_VERSION}", re.IGNORECASE)

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

    先剔掉方括号 CRC32 校验码。强模式（E01 / EP01 / S01E01 / 第01集 / 第01話，
    E/S..E 两种允许 `v2` 版本后缀）任一命中就只看强模式；全部未命中才走弱模式：
    剔掉噪声 token 后，被分隔符包围的 1–3 位数字（可带 `v2` 后缀）。剩一个 → 集号；
    一个都没有 → None；不止一个 → Ambiguous。**绝不**按排序顺序补集号。
    """
    text = _CRC_TAG.sub(" ", _normalize(stem))
    strong = [int(m.group(1)) for pattern in _STRONG_PATTERNS for m in pattern.finditer(text)]
    if strong:
        return _verdict(strong)
    cleaned = _NOISE.sub(" ", text)
    weak = [
        int(match.group(1))
        for token in _WEAK_SEPARATORS.split(cleaned)
        if (match := _WEAK_NUMBER.fullmatch(token))
    ]
    return _verdict(weak)


EXTRA: Literal["extra"] = "extra"
"""LLM 兜底判为「非正片」时 resolve_ambiguous 返回的标记。"""

_RESOLVE_SYSTEM = """\
你在整理同一部番剧同一季的视频文件。请为下面每个文件名判断它是正片第几集：
- 是正片：episode 填集号（正整数），extra 填 false。
- 不是正片（NCOP、NCED、SP、OVA、PV、预告、菜单等）：episode 填 null，extra 填 true。
- 无法确定：episode 填 null，extra 填 false。不要按文件的排列顺序猜集号。
按输入里的 id 逐条返回，不要遗漏，也不要新增。"""


class _EpisodeGuess(BaseModel):
    id: int
    episode: int | None = None
    extra: bool = False


class _EpisodeGuesses(BaseModel):
    files: list[_EpisodeGuess]


async def resolve_ambiguous(
    stems: Sequence[str], provider: LLMProvider
) -> list[int | Literal["extra"] | None]:
    """把正则认不出/认出多个候选的文件名**打包成一次调用**交给 LLM。

    返回值与 stems 逐位对齐：集号、`EXTRA`（非正片）或 None（仍无法确定）。
    模型漏答、答了不在输入里的 id、给出非正整数集号，一律当「无法确定」；同一个 id
    答了两次取第一次。provider 抛的异常原样外抛，由调用方决定怎么降级。
    """
    if not stems:
        return []
    user = json.dumps(
        {"files": [{"id": i, "filename": stem} for i, stem in enumerate(stems, start=1)]},
        ensure_ascii=False,
    )
    response = await provider.complete(_RESOLVE_SYSTEM, user, _EpisodeGuesses)
    by_id: dict[int, _EpisodeGuess] = {}
    for guess in response.files:
        by_id.setdefault(guess.id, guess)
    verdicts: list[int | Literal["extra"] | None] = []
    for i in range(1, len(stems) + 1):
        guess = by_id.get(i)
        if guess is None:
            verdicts.append(None)
        elif guess.extra:
            verdicts.append(EXTRA)
        elif guess.episode is not None and guess.episode >= 1:
            verdicts.append(guess.episode)
        else:
            verdicts.append(None)
    return verdicts


_TENTH = Decimal("0.1")


def round_range(value: tuple[float, float]) -> tuple[float, float] | None:
    """四舍五入到 0.1 秒（半入，经 str 走 Decimal，避免二进制浮点的 0.25 → 0.2）。

    取整后 start >= end 返回 None：调用方把它当推断失败，不写入。
    """
    start, end = (float(Decimal(str(x)).quantize(_TENTH, rounding=ROUND_HALF_UP)) for x in value)
    if start >= end:
        return None
    return (start, end)


@dataclass(frozen=True)
class AvSide:
    """一侧的画面+音频识别结果；range 为空表示识别失败，不能回落到字幕。"""

    range: tuple[float, float] | None
    distance: int | None = None
    reason: str | None = None
    estimated: bool = False


def reference_hashes(cfg: ProjectConfig, references: dict[str, Path]) -> dict[str, int]:
    """先验证全部截图，坏图必须在 LLM 兜底与导入写入之前报错。"""
    hashes = {}
    for side, path in references.items():
        try:
            hashes[side] = av_match.image_hash(path, ffmpeg_path=cfg.render.ffmpeg_path)
        except FFmpegBinaryError:
            raise
        except FFmpegError as error:
            raise ValueError(
                f"参考截图读不出来：{path}（{_first_line(error)}）。换一张或删掉它再导入"
            ) from error
    return hashes


def _locate_side(
    cfg: ProjectConfig, video: Path, ref: int, *, label: str, start: float, end: float
) -> av_match.Match | AvSide:
    try:
        match = av_match.locate(
            video, ref, start=start, end=end, ffmpeg_path=cfg.render.ffmpeg_path
        )
    except FFmpegBinaryError:
        raise
    except FFmpegError as error:
        return AvSide(None, reason=f"{label} 抽帧失败：{_first_line(error)}")
    if match is None:
        return AvSide(None, reason=f"{label} 搜索窗内没有画面")
    if match.distance > av_match.MAX_DISTANCE:
        return AvSide(
            None,
            distance=match.distance,
            reason=f"{label} 画面对不上（最佳距离 {match.distance}/64，超过 12/64）",
        )
    return match


def _envelope(cfg: ProjectConfig, video: Path, start: float):
    return av_match.audio_envelope(
        av_match.read_pcm(
            video, start, av_match.REFERENCE_SECONDS, ffmpeg_path=cfg.render.ffmpeg_path
        )
    )


def _op_sides(
    cfg: ProjectConfig, starts: dict[int, tuple[Path, float, av_match.Match]]
) -> dict[int, AvSide]:
    """选画面距离最小的集作音频参考，每集只读一次并共用稳健长度。"""
    if not starts:
        return {}
    sides: dict[int, AvSide] = {}
    usable = dict(starts)
    length = av_match.DEFAULT_OP_LENGTH
    estimated = True
    if len(starts) > 1:
        reference = min(starts, key=lambda n: (starts[n][2].distance, n))
        video, _, match = starts[reference]
        try:
            ref_env = _envelope(cfg, video, match.time)
        except FFmpegBinaryError:
            raise
        except FFmpegError as error:
            reason = f"OP 参考集 E{reference:02d} 的音频读不出来：{_first_line(error)}"
            return {
                n: AvSide(None, distance=m.distance, reason=reason)
                for n, (_, _, m) in starts.items()
            }
        lengths = []
        for number, (video, _, match) in starts.items():
            if number == reference:
                continue
            try:
                env = _envelope(cfg, video, match.time)
            except FFmpegBinaryError:
                raise
            except FFmpegError as error:
                sides[number] = AvSide(
                    None, distance=match.distance, reason=f"OP 音频读不出来：{_first_line(error)}"
                )
                del usable[number]
                continue
            measured = av_match.op_length(ref_env, env)
            if measured is not None:
                lengths.append(measured)
        if lengths:
            length = av_match.median_length(lengths)
            if length is None:
                values = "、".join(f"{value:g}" for value in sorted(lengths))
                reason = f"各集 OP 长度相差太大（{values} 秒），定不了终点"
                sides.update(
                    {
                        n: AvSide(None, distance=m.distance, reason=reason)
                        for n, (_, _, m) in usable.items()
                    }
                )
                return sides
            estimated = False
    sides.update(
        {
            number: AvSide(
                (match.time, min(match.time + length, duration)),
                distance=match.distance,
                estimated=estimated,
            )
            for number, (_, duration, match) in usable.items()
        }
    )
    return sides


def match_credits(
    cfg: ProjectConfig,
    candidates: Sequence[tuple[int, Pair, EpisodeConfig | None]],
    references: dict[str, int],
) -> dict[int, tuple[AvSide | None, AvSide | None]]:
    """对无手填区间的候选集做有界画面搜索，再批量确定 OP 长度。"""
    if not references:
        return {}
    eligible = [
        (number, pair)
        for number, pair, existing in candidates
        if existing is None or (existing.op_range is None and existing.ed_range is None)
    ]
    results: dict[int, tuple[AvSide | None, AvSide | None]] = {}
    starts: dict[int, tuple[Path, float, av_match.Match]] = {}
    for index, (number, pair) in enumerate(eligible, start=1):
        print(f"正在匹配画面 {index}/{len(eligible)}")
        video = pair.video.resolve()
        duration = _source_duration(cfg, EpisodeConfig(number=number, video=video))
        if duration is None:
            failed = AvSide(None, reason="片长未知，没做画面匹配")
            results[number] = (
                failed if references.get("op") is not None else None,
                failed if references.get("ed") is not None else None,
            )
            continue
        op = ed = None
        if (ref := references.get("op")) is not None:
            found = _locate_side(
                cfg, video, ref, label="OP", start=0, end=min(OP_SEARCH_SECONDS, duration)
            )
            if isinstance(found, av_match.Match):
                starts[number] = (video, duration, found)
            else:
                op = found
        if (ref := references.get("ed")) is not None:
            found = _locate_side(
                cfg,
                video,
                ref,
                label="ED",
                start=max(0, duration - ED_SEARCH_SECONDS),
                end=duration,
            )
            ed = (
                AvSide((found.time, duration), distance=found.distance)
                if isinstance(found, av_match.Match)
                else found
            )
        results[number] = (op, ed)
    for number, op in _op_sides(cfg, starts).items():
        results[number] = (op, results[number][1])
    return results


@dataclass(frozen=True)
class RangePreview:
    """一集的区间预览。

    op_range / ed_range 是按侧选择的识别值（已有手填/项目默认时用 build_track 原值）；
    write 是确认后要写进 project.yaml 的那一对（已取整），None 表示一个都不写，
    note 说明为什么不写。
    """

    duration: float | None
    op_range: tuple[float, float] | None
    ed_range: tuple[float, float] | None
    write: tuple[tuple[float, float], tuple[float, float]] | None
    note: str | None = None
    op_source: str | None = None
    ed_source: str | None = None
    op_distance: int | None = None
    ed_distance: int | None = None
    op_length_estimated: bool = False


def infer_ranges(
    cfg: ProjectConfig,
    *,
    number: int,
    srt: Path,
    video: Path,
    existing: EpisodeConfig | None = None,
    op_av: AvSide | None = None,
    ed_av: AvSide | None = None,
) -> RangePreview:
    """用 ingest 同一套 build_track 推断这一集的 OP/ED，并决定写不写。

    片长走 pipeline._source_duration：`FFmpegBinaryError`（ffprobe 配错）照旧外抛，
    其余探测失败降级为 None —— 此时 build_track 会退化到字幕末尾，ED 口径不可信，
    所以不写。existing 是预填条目：把它已有的区间原样喂给 build_track，预览才跟
    之后真正的 ingest 一致。
    提供 AvSide 的侧使用其结果，识别失败也不回落到字幕；其余侧用字幕推断。

    写入规则（OP/ED 两个字段直接驱动 credits 识别窗，只写一半会把另一半的窗关掉）：
    已有任一手填值 → 不写；片长未知 → 不写；项目级默认区间在这一集生效 → 不写
    （那不是推断值，钉成逐集值只会让项目默认以后改不动这一集）；OP 或 ED 任一推不出
    → 都不写；取整后任一区间无效 → 都不写。
    """
    duration = _source_duration(cfg, EpisodeConfig(number=number, video=Path(video).resolve()))
    manual_op = existing.op_range if existing is not None else None
    manual_ed = existing.ed_range if existing is not None else None
    track = build_track(
        srt,
        episode=number,
        source="srt",
        glossary=cfg.glossary,
        convert_traditional=cfg.locale.convert_traditional,
        show_title=cfg.show,
        op_range=manual_op,
        ed_range=manual_ed,
        duration=duration,
        ingest=cfg.ingest,
        credits=cfg.credits,
    )

    op_range, ed_range = track.op_range, track.ed_range
    op_source = ed_source = None
    op_distance = ed_distance = None
    op_length_estimated = False

    def preview(write=None, note=None) -> RangePreview:
        return RangePreview(
            duration=duration,
            op_range=op_range,
            ed_range=ed_range,
            write=write,
            note=note,
            op_source=op_source,
            ed_source=ed_source,
            op_distance=op_distance,
            ed_distance=ed_distance,
            op_length_estimated=op_length_estimated,
        )

    if manual_op is not None or manual_ed is not None:
        return preview(note="已有手填区间，原样保留")
    sources = {
        credit_range_source(None, cfg.credits.default_op_range, track.duration),
        credit_range_source(None, cfg.credits.default_ed_range, track.duration),
    }
    if sources != {"inferred"}:
        return preview(note="项目级默认区间生效，不写逐集区间")
    op_range = op_av.range if op_av is not None else track.op_range
    ed_range = ed_av.range if ed_av is not None else track.ed_range
    op_source = "av" if op_av is not None else "subtitle"
    ed_source = "av" if ed_av is not None else "subtitle"
    op_distance = op_av.distance if op_av is not None else None
    ed_distance = ed_av.distance if ed_av is not None else None
    op_length_estimated = op_av.estimated if op_av is not None else False
    if duration is None:
        return preview(note="片长未知，不写区间")
    if op_range is None or ed_range is None:
        note = "OP 与 ED 未能都识别，两个都不写"
        reasons = [
            side.reason
            for side in (op_av, ed_av)
            if side is not None and side.range is None and side.reason
        ]
        if reasons:
            note += "（" + "；".join(reasons) + "）"
        return preview(note=note)
    op = round_range(op_range)
    ed = round_range(ed_range)
    if op is None or ed is None:
        return preview(note="取整到 0.1 秒后区间无效，两个都不写")
    return preview(write=(op, ed))


@dataclass(frozen=True)
class Skipped:
    """跳过的一个视频及原因（给预览用）。"""

    path: Path
    reason: str


@dataclass(frozen=True)
class ImportItem:
    number: int
    pair: Pair
    prefilled: bool
    ranges: RangePreview

    @property
    def invalidates_outputs(self) -> bool:
        """补齐来源的预填集要写区间：EPISODE 配置切片含 op_range/ed_range，
        这一集已有的 ingest/timeline/audio/render 产物会失效重跑，预览要提示。"""
        return self.prefilled and self.ranges.write is not None


@dataclass
class ImportPlan:
    items: list[ImportItem] = field(default_factory=list)
    registered: list[Skipped] = field(default_factory=list)
    ambiguous: list[Skipped] = field(default_factory=list)
    failed: list[Skipped] = field(default_factory=list)
    extras: list[Path] = field(default_factory=list)
    unpaired_videos: list[Path] = field(default_factory=list)
    unpaired_srts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# LLM 兜底调用失败的形态：LLMError（RuntimeError 子类）、没被包装的 httpx 传输异常、
# 以及 schema 校验失败的 pydantic ValidationError（ValueError 子类）。
_LLM_CALL_FAILURES = (RuntimeError, ValueError, httpx.HTTPError)


def _describe(error: BaseException) -> str:
    return str(error) or type(error).__name__


def _first_line(error: BaseException) -> str:
    return _describe(error).splitlines()[0]


async def _llm_numbers(
    pending: list[Pair],
    provider_factory: Callable[[], LLMProvider],
    plan: ImportPlan,
) -> list[tuple[int, Pair]]:
    """正则没认出来的那批文件交给 LLM。provider 只在这里、且确实有待判文件时才构造。

    构造失败（缺 key）或调用失败都不中断命令：这批文件按歧义跳过并给一条 warning，
    正则已判定的集照常导入。
    """
    if not pending:
        return []

    def skip_all(reason: str) -> list[tuple[int, Pair]]:
        plan.ambiguous.extend(Skipped(pair.video, reason) for pair in pending)
        return []

    try:
        provider = provider_factory()
    except (RuntimeError, ValueError) as error:
        plan.warnings.append(
            f"LLM 兜底不可用（{_describe(error)}），{len(pending)} 个文件按集号无法确定跳过"
        )
        return skip_all("文件名里认不出集号，LLM 兜底不可用")
    try:
        verdicts = await resolve_ambiguous([pair.video.stem for pair in pending], provider)
    except _LLM_CALL_FAILURES as error:
        plan.warnings.append(
            f"LLM 兜底调用失败（{_describe(error)}），{len(pending)} 个文件按集号无法确定跳过"
        )
        return skip_all("文件名里认不出集号，LLM 兜底调用失败")
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()

    numbered: list[tuple[int, Pair]] = []
    for pair, verdict in zip(pending, verdicts, strict=True):
        if verdict == EXTRA:
            plan.extras.append(pair.video)
        elif verdict is None:
            plan.ambiguous.append(Skipped(pair.video, "文件名里认不出集号，LLM 也无法确定"))
        else:
            numbered.append((verdict, pair))
    return numbered


async def build_import_plan(
    cfg: ProjectConfig,
    video_dir: Path,
    srt_dir: Path | None = None,
    *,
    provider_factory: Callable[[], LLMProvider],
) -> ImportPlan:
    """扫描、识别集号、分拣，并对可导入集做区间推断。全程只读，不写任何文件。

    分拣顺序：非正片 → 集号（正则，再 LLM 兜底）→ 重复集号（同号文件全部跳过，
    不靠排序挑一个）→ 已登记且有来源（跳过，不替换来源、不覆盖区间）→ 其余可导入
    （预填条目补齐来源、保留已有区间）。`FFmpegBinaryError` 原样外抛。
    """
    scan = scan_directory(video_dir, srt_dir)
    references = reference_hashes(cfg, scan.references)
    plan = ImportPlan(
        unpaired_videos=list(scan.unpaired_videos), unpaired_srts=list(scan.unpaired_srts)
    )
    numbered: list[tuple[int, Pair]] = []
    pending: list[Pair] = []
    for pair in scan.pairs:
        if is_extra(pair.video.stem):
            plan.extras.append(pair.video)
            continue
        verdict = parse_episode_number(pair.video.stem)
        if isinstance(verdict, int):
            numbered.append((verdict, pair))
        else:
            pending.append(pair)
    numbered.extend(await _llm_numbers(pending, provider_factory, plan))

    counts = Counter(number for number, _ in numbered)
    by_number = {episode.number: episode for episode in cfg.episodes}
    candidates: list[tuple[int, Pair, EpisodeConfig | None]] = []
    for number, pair in sorted(numbered, key=lambda item: (item[0], item[1].video.name)):
        if counts[number] > 1:
            plan.ambiguous.append(Skipped(pair.video, f"集号 {number} 重复，同号文件都跳过"))
            continue
        existing = by_number.get(number)
        if existing is not None and existing.has_source:
            plan.registered.append(
                Skipped(pair.video, f"第 {number} 集已登记，跳过（不替换来源、不覆盖区间）")
            )
            continue
        candidates.append((number, pair, existing))
    matched = match_credits(cfg, candidates, references)
    for number, pair, existing in candidates:
        op_av, ed_av = matched.get(number, (None, None))
        try:
            ranges = infer_ranges(
                cfg,
                number=number,
                srt=pair.srt,
                video=pair.video,
                existing=existing,
                op_av=op_av,
                ed_av=ed_av,
            )
        except (ValueError, OSError) as error:
            plan.failed.append(Skipped(pair.video, f"字幕读不出来：{_describe(error)}"))
            continue
        plan.items.append(
            ImportItem(number=number, pair=pair, prefilled=existing is not None, ranges=ranges)
        )
    return plan


def apply_import_plan(cfg: ProjectConfig, plan: ImportPlan) -> ProjectConfig:
    """按确认过的计划逐集登记：拷 SRT 到 srt/E{NN}.srt、记视频绝对路径、写区间建议。

    每一集各自原子写回 project.yaml，不做跨集事务：中途失败时已登记的集保留。
    """
    for item in plan.items:
        op_range, ed_range = item.ranges.write if item.ranges.write is not None else (None, None)
        cfg = register_episode(
            cfg,
            episode=item.number,
            srt=item.pair.srt,
            video=item.pair.video,
            op_range=op_range,
            ed_range=ed_range,
        )
    return cfg


def _format_range(value: tuple[float, float] | None) -> str:
    if value is None:
        return "未识别"
    return f"{value[0]:.1f}–{value[1]:.1f} 秒"


def _format_side(
    value: tuple[float, float] | None, source: str | None, distance: int | None
) -> str:
    text = _format_range(value)
    if source is None:
        return text
    label = SOURCE_LABELS[source]
    if distance is not None:
        label += f"，距离 {distance}/{av_match.HASH_BITS}"
    return f"{text}（{label}）"


def render_preview(plan: ImportPlan) -> list[str]:
    """逐项预览：集数、视频、SRT、片长、OP/ED、是否写入区间，以及各类跳过项。"""
    lines = [f"可导入 {len(plan.items)} 集："]
    for item in plan.items:
        ranges = item.ranges
        label = "（预填条目，补齐来源）" if item.prefilled else ""
        lines.append(
            f"  E{item.number:02d}  视频：{item.pair.video.name}  字幕：{item.pair.srt.name}{label}"
        )
        duration = "片长未知" if ranges.duration is None else f"片长 {ranges.duration:.1f} 秒"
        lines.append(
            f"       {duration}  "
            f"OP：{_format_side(ranges.op_range, ranges.op_source, ranges.op_distance)}  "
            f"ED：{_format_side(ranges.ed_range, ranges.ed_source, ranges.ed_distance)}"
        )
        if ranges.op_length_estimated:
            lines.append(f"       OP 长度按 {av_match.DEFAULT_OP_LENGTH:g} 秒估计")
        if ranges.write is not None:
            op, ed = ranges.write
            lines.append(f"       将写入区间：op_range {list(op)}，ed_range {list(ed)}")
        else:
            lines.append(f"       不写区间：{ranges.note}")
        if item.pair.ignored_srts:
            lines.append(
                "       忽略的候选字幕：" + "、".join(p.name for p in item.pair.ignored_srts)
            )
        if item.invalidates_outputs:
            lines.append(
                f"       提示：写入区间会改变 E{item.number:02d} 的配置切片，"
                "它已有的 ingest/timeline/audio/render 产物会失效重跑"
            )
    for title, skipped in (
        ("已登记跳过", plan.registered),
        ("集号无法确定跳过", plan.ambiguous),
        ("字幕读取失败跳过", plan.failed),
    ):
        if skipped:
            lines.append(f"{title}：")
            lines.extend(f"  {entry.path.name}：{entry.reason}" for entry in skipped)
    for title, paths in (
        ("非正片跳过", plan.extras),
        ("未配对视频", plan.unpaired_videos),
        ("未配对字幕", plan.unpaired_srts),
    ):
        if paths:
            lines.append(f"{title}：")
            lines.extend(f"  {path.name}" for path in paths)
    return lines
