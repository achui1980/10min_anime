import pytest

from tenmin.config import RenderConfig
from tenmin.models import AudioDirection, Beat, Hold, Script
from tenmin.render.tts import (
    TTS_MAX_ATTEMPTS,
    EdgeTTSEngine,
    TTSEngine,
    build_tts_engine,
    synthesize_track,
    synthesize_with_retry,
)

from .fakes import FakeTTSEngine, FlakyTTSEngine


def sample_script() -> Script:
    return Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(
                id="b1",
                label="Hook",
                role="hook",
                narration="第一句。第二句。",
                audio=AudioDirection(holds=[Hold(at=1.0, duration=1.5, quote="金句")]),
            ),
            Beat(id="b2", label="收尾", role="outro", narration="第三句。"),
        ],
    )


def test_fake_engine_satisfies_protocol():
    assert isinstance(FakeTTSEngine([1.0]), TTSEngine)


def test_edge_engine_satisfies_protocol():
    assert isinstance(EdgeTTSEngine(voice="zh-CN-YunxiNeural", rate="+0%"), TTSEngine)


def test_build_tts_engine_uses_render_config():
    engine = build_tts_engine(RenderConfig(voice="zh-CN-XiaoxiaoNeural", rate="+10%"))
    assert isinstance(engine, EdgeTTSEngine)
    assert engine.voice == "zh-CN-XiaoxiaoNeural"
    assert engine.rate == "+10%"


async def test_synthesize_with_retry_recovers(tmp_path):
    engine = FlakyTTSEngine(fail_times=2, duration=3.0)
    duration = await synthesize_with_retry(
        engine, "第一句。", tmp_path / "chunk_001.mp3", label="beat b1 的第 1 个 chunk"
    )
    assert duration == 3.0
    assert engine.attempts == 3


async def test_synthesize_with_retry_gives_up_after_max_attempts(tmp_path):
    engine = FlakyTTSEngine(fail_times=TTS_MAX_ATTEMPTS)
    with pytest.raises(RuntimeError) as exc:
        await synthesize_with_retry(
            engine, "第一句。", tmp_path / "chunk_001.mp3", label="beat b1 的第 1 个 chunk"
        )
    message = str(exc.value)
    assert "beat b1 的第 1 个 chunk" in message
    assert "第一句。" in message
    assert engine.attempts == TTS_MAX_ATTEMPTS


async def test_synthesize_track_builds_chunks(tmp_path):
    engine = FakeTTSEngine([3.0, 4.0, 5.0])
    track, warnings = await synthesize_track(sample_script(), 2, tmp_path, engine)
    assert warnings == []
    assert [c.path for c in track.chunks] == [
        "chunk_001.mp3",
        "chunk_002.mp3",
        "chunk_003.mp3",
    ]
    assert [c.beat_id for c in track.chunks] == ["b1", "b1", "b2"]
    assert [c.index for c in track.chunks] == [1, 2, 1]
    assert [c.text for c in track.chunks] == ["第一句。", "第二句。", "第三句。"]
    assert [c.duration for c in track.chunks] == [3.0, 4.0, 5.0]
    assert [c.hold_after for c in track.chunks] == [1.5, 0.0, 0.0]
    # 3 + 1.5 + 4 + 5
    assert track.total_seconds == pytest.approx(13.5)
    assert track.episode == 2


async def test_synthesize_track_writes_files(tmp_path):
    engine = FakeTTSEngine([3.0, 4.0, 5.0])
    await synthesize_track(sample_script(), 2, tmp_path, engine)
    assert (tmp_path / "chunk_001.mp3").exists()
    assert (tmp_path / "chunk_003.mp3").exists()


async def test_synthesize_track_reuses_existing_chunk(tmp_path, monkeypatch):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "chunk_001.mp3").write_bytes(b"already there")
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 9.0)
    engine = FakeTTSEngine([4.0, 5.0])
    track, _ = await synthesize_track(sample_script(), 2, tmp_path, engine)
    # 第一个 chunk 复用磁盘上的文件，engine 只被调了 2 次
    assert len(engine.calls) == 2
    assert track.chunks[0].duration == 9.0


async def test_synthesize_track_reuse_false_resynthesizes(tmp_path, monkeypatch):
    (tmp_path / "chunk_001.mp3").write_bytes(b"already there")
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 9.0)
    engine = FakeTTSEngine([3.0, 4.0, 5.0])
    track, _ = await synthesize_track(sample_script(), 2, tmp_path, engine, reuse=False)
    assert len(engine.calls) == 3
    assert track.chunks[0].duration == 3.0


async def test_synthesize_track_warns_on_empty_narration(tmp_path):
    script = Script(
        show="剧名",
        episodes=[2],
        beats=[Beat(id="b1", label="Hook", role="hook", narration="  ")],
    )
    track, warnings = await synthesize_track(script, 2, tmp_path, FakeTTSEngine([]))
    assert track.chunks == []
    assert len(warnings) == 1
    assert "b1" in warnings[0]
