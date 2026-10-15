import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tenmin.ingest import ocr_audit
from tenmin.models import RawCue


def _cue(index: int, start: float, end: float, text: str) -> RawCue:
    return RawCue(idx=index, start=start, end=end, text=text)


def _srt(path: Path, cues: list[tuple[float, float, str]]) -> Path:
    def stamp(value: float) -> str:
        ms = round(value * 1000)
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    blocks = [
        f"{pos}\n{stamp(start)} --> {stamp(end)}\n{text}\n"
        for pos, (start, end, text) in enumerate(cues, start=1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def test_char_diff_marks_deleted_and_inserted_text():
    assert ocr_audit.char_diff("猫猫说话", "猫猫说活") == [
        ["equal", "猫猫说"],
        ["delete", "话"],
        ["insert", "活"],
    ]


def test_char_diff_of_identical_text_is_single_equal_segment():
    assert ocr_audit.char_diff("一样", "一样") == [["equal", "一样"]]


def test_build_items_keeps_only_changed_or_reviewed_lines():
    raw = [_cue(1, 0, 1, "甲"), _cue(2, 1, 2, "乙"), _cue(3, 2, 4, "丙")]
    corrected = [_cue(1, 0, 1, "甲"), _cue(2, 1, 2, "已"), _cue(3, 2, 4, "丙")]
    review = [{"id": 3, "start": 2, "end": 4, "text": "丙", "reason": "看不清", "candidate": "两"}]

    items = ocr_audit.build_items(raw, corrected, review)

    assert [item.id for item in items] == [2, 3]
    assert items[0].changed and not items[0].uncertain
    assert items[0].original == "乙" and items[0].corrected == "已"
    assert items[1].uncertain and not items[1].changed
    assert items[1].reason == "看不清" and items[1].candidate == "两"
    assert items[1].midpoint == 3.0


def test_build_items_rejects_misaligned_cue_counts():
    with pytest.raises(ValueError, match="条数"):
        ocr_audit.build_items([_cue(1, 0, 1, "甲")], [], [])


def test_build_items_rejects_changed_timing():
    with pytest.raises(ValueError, match="时间"):
        ocr_audit.build_items([_cue(1, 0, 1, "甲")], [_cue(1, 0, 2, "甲")], [])


def test_sources_for_standalone_and_pipeline_names(tmp_path):
    standalone = ocr_audit.sources_for(tmp_path / "片.zh-Hans.srt")
    assert standalone.raw == tmp_path / "片.zh-Hans.raw.srt"
    assert standalone.review == tmp_path / "片.zh-Hans.review.json"

    pipeline = ocr_audit.sources_for(tmp_path / "E02.ocr.corrected.srt")
    assert pipeline.raw == tmp_path / "E02.ocr.srt"
    assert pipeline.review == tmp_path / "E02.ocr.review.json"

    with pytest.raises(ValueError, match="认不出"):
        ocr_audit.sources_for(tmp_path / "随便.srt")


def test_find_corrected_prefers_simplified_next_to_video(tmp_path):
    video = tmp_path / "片.mp4"
    (tmp_path / "片.zh-Hant.srt").write_text("", encoding="utf-8")
    (tmp_path / "片.zh-Hans.srt").write_text("", encoding="utf-8")
    assert ocr_audit.find_corrected(video, tmp_path) == tmp_path / "片.zh-Hans.srt"
    with pytest.raises(FileNotFoundError, match="--ai-correct"):
        ocr_audit.find_corrected(tmp_path / "没有.mp4", tmp_path)


def test_render_report_escapes_script_end_tag():
    item = ocr_audit.AuditItem(
        id=1, start=0, end=1, original="</script>", corrected="<b>",
        changed=True, uncertain=False, reason="", candidate="",
        diff=ocr_audit.char_diff("</script>", "<b>"), image="frames/0001.jpg",
    )
    html = ocr_audit.render_report([item], title="片", total=5)
    assert "</script>\"" not in html
    payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    data = json.loads(payload)
    assert data["items"][0]["original"] == "</script>"
    assert data["total"] == 5


def test_audit_without_review_file_writes_report_and_frames(tmp_path, monkeypatch):
    raw = _srt(tmp_path / "片.zh-Hans.raw.srt", [(0, 1, "甲"), (1, 2, "乙")])
    corrected = _srt(tmp_path / "片.zh-Hans.srt", [(0, 1, "甲"), (1, 2, "已")])
    assert raw.is_file()
    calls = []

    def fake_extract(video, seconds, dest, *, ffmpeg):
        calls.append(seconds)
        dest.write_bytes(b"jpg")

    monkeypatch.setattr(ocr_audit, "_extract_frame", fake_extract)
    stale = tmp_path / "out" / "frames" / "9999.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    result = ocr_audit.audit(tmp_path / "片.mp4", corrected, tmp_path / "out")

    assert calls == [1.5]
    assert result.items == 1 and result.total == 2
    assert (tmp_path / "out" / "frames" / "0002.jpg").is_file()
    assert not stale.exists()
    assert "已" in result.report.read_text(encoding="utf-8")


def test_audit_requires_raw_draft(tmp_path):
    corrected = _srt(tmp_path / "片.zh-Hans.srt", [(0, 1, "甲")])
    with pytest.raises(FileNotFoundError, match="原稿"):
        ocr_audit.audit(tmp_path / "片.mp4", corrected, tmp_path / "out")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg")
def test_extract_frame_from_real_video(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
         "-t", "2", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    dest = tmp_path / "f.jpg"
    ocr_audit._extract_frame(video, 1.0, dest, ffmpeg="ffmpeg")
    assert dest.read_bytes()[:2] == b"\xff\xd8"


def _write_pair(directory: Path, stem: str) -> Path:
    raw = "1\n00:00:01,000 --> 00:00:02,000\n猫猫\n\n2\n00:00:03,000 --> 00:00:04,000\n不变\n"
    fixed = "1\n00:00:01,000 --> 00:00:02,000\n猫猫们\n\n2\n00:00:03,000 --> 00:00:04,000\n不变\n"
    (directory / f"{stem}.zh-Hans.raw.srt").write_text(raw, encoding="utf-8")
    corrected = directory / f"{stem}.zh-Hans.srt"
    corrected.write_text(fixed, encoding="utf-8")
    return corrected


def test_cli_ocr_audit_writes_report(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tenmin.cli import app

    video = tmp_path / "ep.mp4"
    video.write_bytes(b"x")
    _write_pair(tmp_path, "ep")
    monkeypatch.setattr(
        ocr_audit, "_extract_frame", lambda v, s, dest, *, ffmpeg: dest.write_bytes(b"\xff\xd8")
    )
    result = CliRunner().invoke(app, ["ocr-audit", str(video)])
    assert result.exit_code == 0, result.output
    assert "待抽查 1 条" in result.output
    assert (tmp_path / "ep.ocr-audit" / "report.html").is_file()


def test_cli_ocr_audit_missing_corrected_exits_1(tmp_path):
    from typer.testing import CliRunner

    from tenmin.cli import app

    video = tmp_path / "ep.mp4"
    video.write_bytes(b"x")
    result = CliRunner().invoke(app, ["ocr-audit", str(video)])
    assert result.exit_code == 1
    assert "--ai-correct" in result.output
