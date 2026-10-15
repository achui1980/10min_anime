"""OCR AI 校对成功进入下游；失败不会被全局 ingest 完成戳永久跳过。"""

import asyncio
import json

from tenmin.config import ProjectConfig
from tenmin.ingest.resolve import SubtitleSource
from tenmin.pipeline import Paths, run_ingest_corrected, run_pipeline


def test_correction_failure_retries_then_updates_dialogue_and_translation(tmp_path, monkeypatch):
    cfg = ProjectConfig.model_validate({
        "show": "测试", "slug": "test", "ocr": {"enabled": True, "ai_correct": True},
        "episodes": [{"number": 1, "video": "ep.mp4"}],
    })
    cfg.bind_root(tmp_path)
    (tmp_path / "ep.mp4").write_bytes(b"video")
    (tmp_path / "srt").mkdir()
    raw = tmp_path / "srt" / "E01.ocr.srt"
    raw.write_text("1\n00:05:00,000 --> 00:05:02,000\n垃圾也杀丢\n", encoding="utf-8")
    monkeypatch.setattr(
        "tenmin.pipeline.resolve_subtitle_source",
        lambda *args, **kwargs: SubtitleSource(raw, "ocr"),
    )
    monkeypatch.setattr("tenmin.pipeline._source_duration", lambda *args: 1000)
    calls = []

    class Provider:
        async def complete(self, system, user, schema):
            calls.append(user)
            if len(calls) == 1:
                raise RuntimeError("连接超时")
            return schema.model_validate({"lines": [
                {"id": 1, "original": "垃圾也杀丢", "text": "垃圾也没丢"}
            ]})

    kwargs = {"only": ["ingest", "translate"], "correction_provider_factory": Provider}
    warnings = asyncio.run(run_pipeline(cfg, None, **kwargs))
    assert any("连接超时" in warning for warning in warnings)
    paths = Paths(tmp_path)
    assert "垃圾也杀丢" in paths.dialogue(1).read_text()
    asyncio.run(run_pipeline(cfg, None, **kwargs))
    assert len(calls) == 2
    assert "垃圾也没丢" in paths.dialogue(1).read_text()
    assert "垃圾也没丢" in paths.zh_subtitles(1).read_text()
    old_mtime = paths.dialogue(1).stat().st_mtime_ns
    asyncio.run(run_pipeline(cfg, None, **kwargs))
    assert len(calls) == 2
    assert paths.dialogue(1).stat().st_mtime_ns == old_mtime
    assert json.loads(paths.dialogue(1).read_text())["source"] == "ocr"


def test_non_ocr_source_never_constructs_correction_provider(tmp_path, monkeypatch):
    cfg = ProjectConfig.model_validate({
        "show": "测试", "slug": "test", "ocr": {"ai_correct": True},
        "episodes": [{"number": 1, "srt": "E01.srt"}],
    })
    cfg.bind_root(tmp_path)
    raw = tmp_path / "E01.srt"
    raw.write_text("1\n00:05:00,000 --> 00:05:02,000\n垃圾也没丢\n")
    monkeypatch.setattr(
        "tenmin.pipeline.resolve_subtitle_source",
        lambda *args, **kwargs: SubtitleSource(raw, "srt"),
    )
    monkeypatch.setattr("tenmin.pipeline._source_duration", lambda *args: 1000)

    def forbidden():
        raise AssertionError("非 OCR 来源不应创建 provider")

    tracks, warnings, pending = asyncio.run(run_ingest_corrected(cfg, provider_factory=forbidden))
    assert tracks[0].source == "srt" and not warnings and not pending
