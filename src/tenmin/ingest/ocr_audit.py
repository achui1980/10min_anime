"""OCR AI 校对改动抽查报告。

只读三份产物（原稿、校对稿、review），列出被 AI 改过或标了不确定的行，
附字级差异与画面截图，供人工逐条判定改对/改错。不调 LLM、不改字幕、不联网。
"""

from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

from tenmin import atomic
from tenmin.ingest.srt_parser import load_srt
from tenmin.models import RawCue
from tenmin.render import ffmpeg as ff
from tenmin.render.ffmpeg import FFMPEG

_TIME_TOLERANCE = 0.0015
_FRAME_WIDTH = 960
_STANDALONE_SUFFIXES = (".zh-Hans.srt", ".zh-Hant.srt")
_PIPELINE_SUFFIX = ".ocr.corrected.srt"


@dataclass(frozen=True)
class Sources:
    raw: Path
    review: Path


@dataclass
class AuditItem:
    id: int
    start: float
    end: float
    original: str
    corrected: str
    changed: bool
    uncertain: bool
    reason: str
    candidate: str
    diff: list[list[str]]
    image: str = field(default="")

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class AuditResult:
    report: Path
    items: int
    total: int


def char_diff(a: str, b: str) -> list[list[str]]:
    """字级差异；replace 拆成 delete + insert。"""
    segments: list[list[str]] = []
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            segments.append(["equal", a[i1:i2]])
            continue
        if op in ("delete", "replace"):
            segments.append(["delete", a[i1:i2]])
        if op in ("insert", "replace"):
            segments.append(["insert", b[j1:j2]])
    return segments or [["equal", ""]]


def build_items(
    raw_cues: list[RawCue], corrected_cues: list[RawCue], review: list[dict]
) -> list[AuditItem]:
    if len(raw_cues) != len(corrected_cues):
        raise ValueError(
            f"原稿与校对稿条数不一致（{len(raw_cues)} ≠ {len(corrected_cues)}），无法逐条对照"
        )
    reviewed = {int(entry["id"]): entry for entry in review}
    items: list[AuditItem] = []
    for pos, (raw, fixed) in enumerate(zip(raw_cues, corrected_cues, strict=True), start=1):
        if (
            abs(raw.start - fixed.start) > _TIME_TOLERANCE
            or abs(raw.end - fixed.end) > _TIME_TOLERANCE
        ):
            raise ValueError(f"第 {pos} 条原稿与校对稿时间不一致，校对稿可能被手工改过时间轴")
        changed = raw.text != fixed.text
        entry = reviewed.get(pos)
        if not changed and entry is None:
            continue
        items.append(
            AuditItem(
                id=pos,
                start=raw.start,
                end=raw.end,
                original=raw.text,
                corrected=fixed.text,
                changed=changed,
                uncertain=entry is not None,
                reason=str(entry.get("reason", "")) if entry else "",
                candidate=str(entry.get("candidate", "")) if entry else "",
                diff=char_diff(raw.text, fixed.text),
            )
        )
    return items


def sources_for(corrected: Path) -> Sources:
    name = corrected.name
    for suffix in _STANDALONE_SUFFIXES:
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            base = suffix[: -len(".srt")]
            return Sources(
                raw=corrected.with_name(f"{stem}{base}.raw.srt"),
                review=corrected.with_name(f"{stem}{base}.review.json"),
            )
    if name.endswith(_PIPELINE_SUFFIX):
        stem = name[: -len(_PIPELINE_SUFFIX)]
        return Sources(
            raw=corrected.with_name(f"{stem}.ocr.srt"),
            review=corrected.with_name(f"{stem}.ocr.review.json"),
        )
    raise ValueError(
        f"认不出校对稿 {corrected.name}：应为 <名>.zh-Hans.srt / .zh-Hant.srt"
        " 或 E{NN}.ocr.corrected.srt"
    )


def find_corrected(video: Path, search_dir: Path) -> Path:
    for suffix in _STANDALONE_SUFFIXES:
        candidate = search_dir / f"{video.stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"在 {search_dir} 找不到 {video.stem} 的校对稿；先用 tenmin ocr --ai-correct 生成，"
        "或用 --corrected 指定"
    )


def render_report(items: list[AuditItem], *, title: str, total: int) -> str:
    template = resources.files("tenmin.ingest").joinpath("ocr_audit.html").read_text(
        encoding="utf-8"
    )
    payload = {
        "title": title,
        "total": total,
        "items": [{**asdict(item), "midpoint": item.midpoint} for item in items],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return template.replace("__TITLE__", _escape(title)).replace("__DATA__", data)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _extract_frame(video: Path, seconds: float, dest: Path, *, ffmpeg: str) -> None:
    ff.run(
        [
            "-v", "error", "-y",
            "-ss", f"{seconds:.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-vf", f"scale='min({_FRAME_WIDTH},iw)':-2",
            "-q:v", "3",
            str(dest),
        ],
        ffmpeg=ffmpeg,
    )


def _load_review(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是 review 列表")
    return [entry for entry in data if isinstance(entry, dict) and "id" in entry]


def audit(
    video: Path, corrected: Path, out_dir: Path, *, ffmpeg: str = FFMPEG
) -> AuditResult:
    sources = sources_for(corrected)
    if not corrected.is_file():
        raise FileNotFoundError(f"校对稿不存在：{corrected}")
    if not sources.raw.is_file():
        raise FileNotFoundError(f"原稿不存在：{sources.raw}（需要 --ai-correct 运行留下的原稿）")
    raw_cues = load_srt(sources.raw)
    corrected_cues = load_srt(corrected)
    items = build_items(raw_cues, corrected_cues, _load_review(sources.review))

    frames = out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for stale in frames.glob("*.jpg"):
        stale.unlink()
    for item in items:
        dest = frames / f"{item.id:04d}.jpg"
        _extract_frame(video, item.midpoint, dest, ffmpeg=ffmpeg)
        item.image = f"frames/{dest.name}"

    report = out_dir / "report.html"
    atomic.write_text(report, render_report(items, title=video.name, total=len(raw_cues)))
    return AuditResult(report=report, items=len(items), total=len(raw_cues))
