"""`tenmin ocr` 批量命令：展开输入、命名输出、复用/跳过、繁转简、失败不拖累其余。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tenmin.cli import app
from tenmin.config import OcrConfig
from tenmin.ingest import ocr, ocr_batch
from tenmin.ingest.srt_parser import load_srt_detailed
from tenmin.models import RawCue

runner = CliRunner()


def _touch(path: Path, *, mtime: float | None = None, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _fake_recognize(monkeypatch, *, fail: dict[str, Exception] | None = None) -> list[dict]:
    calls: list[dict] = []
    fail = fail or {}

    def fake(video, *, ocr, ffmpeg_path, ffprobe_path):
        calls.append({"video": Path(video), "ocr": ocr})
        if Path(video).name in fail:
            raise fail[Path(video).name]
        return [
            RawCue(idx=1, start=1.0, end=2.0, text="我們這裡見"),
            RawCue(idx=2, start=3.0, end=4.0, text="-說話\n-聽見了"),
        ]

    monkeypatch.setattr(ocr_batch.ocr, "recognize_cues", fake)
    return calls


# --- collect_videos -----------------------------------------------------------


def test_a_directory_yields_its_videos_sorted_and_not_recursively(tmp_path):
    _touch(tmp_path / "b.MKV")
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "a.zh-Hans.srt")
    _touch(tmp_path / "sub" / "c.mp4")

    assert [p.name for p in ocr_batch.collect_videos([tmp_path])] == ["a.mp4", "b.MKV"]


def test_an_explicit_file_is_taken_whatever_its_suffix(tmp_path):
    clip = _touch(tmp_path / "clip.bin")
    assert ocr_batch.collect_videos([clip]) == [clip]


def test_the_same_video_given_twice_is_recognized_once(tmp_path):
    video = _touch(tmp_path / "a.mp4")
    assert ocr_batch.collect_videos([video, tmp_path]) == [video]


def test_a_missing_path_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        ocr_batch.collect_videos([tmp_path / "nope.mp4"])


def test_no_video_at_all_is_an_error(tmp_path):
    _touch(tmp_path / "notes.txt")
    with pytest.raises(ValueError, match="没有找到"):
        ocr_batch.collect_videos([tmp_path])


# --- output_path --------------------------------------------------------------


def test_outputs_sit_next_to_the_video_with_a_language_suffix(tmp_path):
    video = tmp_path / "ep 01.mkv"
    assert ocr_batch.output_path(video, None, simplified=True) == tmp_path / "ep 01.zh-Hans.srt"
    assert ocr_batch.output_path(video, None, simplified=False) == tmp_path / "ep 01.zh-Hant.srt"


def test_an_output_dir_overrides_the_video_folder(tmp_path):
    out = tmp_path / "out"
    assert ocr_batch.output_path(tmp_path / "a.mp4", out, simplified=True) == out / "a.zh-Hans.srt"


# --- run_batch ----------------------------------------------------------------


def test_the_default_output_is_simplified(tmp_path, monkeypatch):
    _fake_recognize(monkeypatch)
    video = _touch(tmp_path / "a.mp4")

    summary = ocr_batch.run_batch(
        [video], out_dir=None, simplified=True, force=False, ocr_config=OcrConfig()
    )

    assert summary.done == [video]
    cues = load_srt_detailed(tmp_path / "a.zh-Hans.srt").cues
    assert [c.text for c in cues] == ["我们这里见", "-说话\n-听见了"]
    assert cues[0].start == pytest.approx(1.0)


def test_traditional_keeps_the_original_text(tmp_path, monkeypatch):
    _fake_recognize(monkeypatch)
    video = _touch(tmp_path / "a.mp4")

    ocr_batch.run_batch(
        [video], out_dir=None, simplified=False, force=False, ocr_config=OcrConfig()
    )

    texts = [c.text for c in load_srt_detailed(tmp_path / "a.zh-Hant.srt").cues]
    assert texts == ["我們這裡見", "-說話\n-聽見了"]


def test_the_output_dir_is_created(tmp_path, monkeypatch):
    _fake_recognize(monkeypatch)
    video = _touch(tmp_path / "a.mp4")
    out = tmp_path / "deep" / "out"

    ocr_batch.run_batch([video], out_dir=out, simplified=True, force=False, ocr_config=OcrConfig())

    assert (out / "a.zh-Hans.srt").is_file()


def test_a_fresh_output_is_skipped_and_force_redoes_it(tmp_path, monkeypatch, capsys):
    calls = _fake_recognize(monkeypatch)
    video = _touch(tmp_path / "a.mp4", mtime=1_000)
    _touch(tmp_path / "a.zh-Hans.srt", mtime=2_000, data=b"hand edited")

    summary = ocr_batch.run_batch(
        [video], out_dir=None, simplified=True, force=False, ocr_config=OcrConfig()
    )
    assert summary.skipped == [video]
    assert calls == []
    assert "已存在，跳过" in capsys.readouterr().out
    assert (tmp_path / "a.zh-Hans.srt").read_bytes() == b"hand edited"

    ocr_batch.run_batch([video], out_dir=None, simplified=True, force=True, ocr_config=OcrConfig())
    assert len(calls) == 1


def test_an_output_older_than_the_video_is_redone(tmp_path, monkeypatch):
    calls = _fake_recognize(monkeypatch)
    video = _touch(tmp_path / "a.mp4", mtime=2_000)
    _touch(tmp_path / "a.zh-Hans.srt", mtime=1_000)

    ocr_batch.run_batch([video], out_dir=None, simplified=True, force=False, ocr_config=OcrConfig())

    assert len(calls) == 1


def test_one_failure_does_not_stop_the_rest(tmp_path, monkeypatch):
    _fake_recognize(monkeypatch, fail={"a.mp4": ocr.OCRError("一条字幕都没认出来")})
    a = _touch(tmp_path / "a.mp4")
    b = _touch(tmp_path / "b.mp4")

    summary = ocr_batch.run_batch(
        [a, b], out_dir=None, simplified=True, force=False, ocr_config=OcrConfig()
    )

    assert summary.done == [b]
    assert summary.failed == [(a, "一条字幕都没认出来")]
    assert not (tmp_path / "a.zh-Hans.srt").exists()


def test_a_missing_extra_aborts_the_whole_batch(tmp_path, monkeypatch):
    calls = _fake_recognize(monkeypatch, fail={"a.mp4": ocr.OCRUnavailableError("装 extra")})
    a = _touch(tmp_path / "a.mp4")
    b = _touch(tmp_path / "b.mp4")

    with pytest.raises(ocr.OCRUnavailableError):
        ocr_batch.run_batch(
            [a, b], out_dir=None, simplified=True, force=False, ocr_config=OcrConfig()
        )
    assert len(calls) == 1


def test_the_config_reaches_recognition(tmp_path, monkeypatch):
    calls = _fake_recognize(monkeypatch)
    video = _touch(tmp_path / "a.mp4")
    config = OcrConfig(crop_top=0.6)

    ocr_batch.run_batch([video], out_dir=None, simplified=True, force=False, ocr_config=config)

    assert calls[0]["ocr"] is config


# --- CLI ----------------------------------------------------------------------


def test_cli_recognizes_a_directory_into_an_output_dir(tmp_path, monkeypatch):
    calls = _fake_recognize(monkeypatch)
    _touch(tmp_path / "v" / "a.mp4")
    _touch(tmp_path / "v" / "b.mkv")

    result = runner.invoke(
        app, ["ocr", str(tmp_path / "v"), "-o", str(tmp_path / "out"), "--crop-top", "0.65"]
    )

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "a.zh-Hans.srt",
        "b.zh-Hans.srt",
    ]
    assert calls[0]["ocr"].crop_top == 0.65
    assert "成功 2，跳过 0，失败 0" in result.output


def test_cli_traditional_flag(tmp_path, monkeypatch):
    _fake_recognize(monkeypatch)
    video = _touch(tmp_path / "a.mp4")

    result = runner.invoke(app, ["ocr", str(video), "--traditional"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "a.zh-Hant.srt").is_file()


def test_cli_exits_nonzero_when_any_video_failed(tmp_path, monkeypatch):
    _fake_recognize(monkeypatch, fail={"a.mp4": ocr.OCRError("没认出来")})
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "b.mp4")

    result = runner.invoke(app, ["ocr", str(tmp_path)])

    assert result.exit_code == 1
    assert "成功 1，跳过 0，失败 1" in result.output


def test_cli_rejects_a_missing_path(tmp_path, monkeypatch):
    calls = _fake_recognize(monkeypatch)

    result = runner.invoke(app, ["ocr", str(tmp_path / "nope.mp4")])

    assert result.exit_code == 1
    assert calls == []


def test_cli_rejects_an_out_of_range_crop_top(tmp_path, monkeypatch):
    calls = _fake_recognize(monkeypatch)
    video = _touch(tmp_path / "a.mp4")

    result = runner.invoke(app, ["ocr", str(video), "--crop-top", "1.5"])

    assert result.exit_code == 1
    assert calls == []


def test_cli_stops_when_the_extra_is_missing(tmp_path, monkeypatch):
    missing = ocr.OCRUnavailableError("跑一次 uv sync --extra ocr")
    _fake_recognize(monkeypatch, fail={"a.mp4": missing})
    _touch(tmp_path / "a.mp4")

    result = runner.invoke(app, ["ocr", str(tmp_path)])

    assert result.exit_code == 1
    assert "uv sync --extra ocr" in result.output
