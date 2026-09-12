import asyncio
import sys
import threading
import types
from pathlib import Path

import pytest

from tenmin.config import RenderConfig
from tenmin.models import AudioDirection, Beat, Hold, Script
from tenmin.render import tts as tts_module
from tenmin.render.tts import (
    TTS_MAX_ATTEMPTS,
    EdgeTTSEngine,
    TTSEngine,
    TTSError,
    build_tts_engine,
    synthesize_track,
    synthesize_with_retry,
)

from .fakes import FailingTTSEngine, FakeReporter, FakeTTSEngine, FlakyTTSEngine


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """抄 tests/test_llm.py 的同名 fixture：退避的 sleep 换成 no-op 并记录时长。

    绝不真睡：退避基数 1 秒、三次尝试就是 1+2 秒，一组测试真睡一遍要几十秒。
    """
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(tts_module, "_sleep", fake_sleep)
    monkeypatch.setattr(tts_module, "_rand", lambda: 0.0)
    return recorded


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


async def test_synthesize_with_retry_recovers(tmp_path, sleeps):
    engine = FlakyTTSEngine(fail_times=2, duration=3.0)
    duration = await synthesize_with_retry(
        engine, "第一句。", tmp_path / "chunk_001.mp3", label="beat b1 的第 1 个 chunk"
    )
    assert duration == 3.0
    assert engine.attempts == 3


async def test_synthesize_with_retry_gives_up_after_max_attempts(tmp_path, sleeps):
    engine = FlakyTTSEngine(fail_times=TTS_MAX_ATTEMPTS)
    with pytest.raises(RuntimeError) as exc:
        await synthesize_with_retry(
            engine, "第一句。", tmp_path / "chunk_001.mp3", label="beat b1 的第 1 个 chunk"
        )
    message = str(exc.value)
    assert "beat b1 的第 1 个 chunk" in message
    assert "第一句。" in message
    assert engine.attempts == TTS_MAX_ATTEMPTS


# --- 重试退避与异常保真 ---


def test_tts_backoff_is_exponential_and_capped(monkeypatch):
    monkeypatch.setattr(tts_module, "_rand", lambda: 0.0)
    assert tts_module._backoff_delay(1) == 1.0
    assert tts_module._backoff_delay(2) == 2.0
    assert tts_module._backoff_delay(3) == 4.0
    assert tts_module._backoff_delay(99) == tts_module.BACKOFF_MAX_SECONDS


def test_tts_backoff_jitter_only_adds(monkeypatch):
    monkeypatch.setattr(tts_module, "_rand", lambda: 1.0)
    assert tts_module._backoff_delay(1) == pytest.approx(1.25)


async def test_synthesize_with_retry_backs_off_between_attempts(tmp_path, sleeps):
    engine = FlakyTTSEngine(fail_times=2, duration=3.0)
    await synthesize_with_retry(
        engine, "第一句。", tmp_path / "chunk_001.mp3", label="chunk"
    )
    assert sleeps == [1.0, 2.0]


async def test_synthesize_with_retry_does_not_sleep_after_the_last_attempt(tmp_path, sleeps):
    engine = FlakyTTSEngine(fail_times=99)
    with pytest.raises(TTSError):
        await synthesize_with_retry(
            engine, "第一句。", tmp_path / "chunk_001.mp3", label="chunk"
        )
    assert sleeps == [1.0, 2.0]


async def test_synthesize_with_retry_honours_max_attempts(tmp_path, sleeps):
    engine = FlakyTTSEngine(fail_times=99)
    with pytest.raises(TTSError):
        await synthesize_with_retry(
            engine, "第一句。", tmp_path / "chunk_001.mp3", label="chunk", max_attempts=1
        )
    assert engine.attempts == 1
    assert sleeps == []


async def test_synthesize_with_retry_keeps_the_exception_type(tmp_path, sleeps):
    """aiohttp 的连接类异常 stringify 常常是空串，光印 str(error) 等于什么都没报。"""
    engine = FailingTTSEngine(ConnectionResetError())
    with pytest.raises(TTSError) as exc:
        await synthesize_with_retry(
            engine, "第一句。", tmp_path / "chunk_001.mp3", label="chunk"
        )
    assert "ConnectionResetError" in str(exc.value)


async def test_synthesize_with_retry_chains_the_original_error(tmp_path, sleeps):
    original = ConnectionResetError("断了")
    engine = FailingTTSEngine(original)
    with pytest.raises(TTSError) as exc:
        await synthesize_with_retry(
            engine, "第一句。", tmp_path / "chunk_001.mp3", label="chunk"
        )
    assert exc.value.__cause__ is original


async def test_synthesize_with_retry_reports_attempt_count(tmp_path, sleeps):
    engine = FailingTTSEngine(ConnectionResetError())
    with pytest.raises(TTSError) as exc:
        await synthesize_with_retry(
            engine, "第一句。", tmp_path / "chunk_001.mp3", label="chunk"
        )
    assert f"{TTS_MAX_ATTEMPTS} 次" in str(exc.value)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Invalid rate '+abc%'."),
        TypeError("voice must be str"),
    ],
)
async def test_synthesize_with_retry_fails_fast_on_permanent_errors(tmp_path, sleeps, error):
    """未知 voice 名 / 畸形 rate 字符串是 edge-tts 在构造期就抛的确定性错误，重试纯浪费。"""
    engine = FailingTTSEngine(error)
    with pytest.raises(TTSError) as exc:
        await synthesize_with_retry(
            engine, "第一句。", tmp_path / "chunk_001.mp3", label="chunk"
        )
    assert engine.attempts == 1
    assert sleeps == []
    assert exc.value.__cause__ is error


def test_tts_error_is_a_pipeline_error():
    """否则 CLI 只会吐一整页 traceback，而不是一行人话。"""
    from tenmin.cli import PIPELINE_ERRORS

    assert issubclass(TTSError, PIPELINE_ERRORS)


async def test_synthesize_track_passes_max_attempts_through(tmp_path, sleeps):
    engine = FlakyTTSEngine(fail_times=1)
    with pytest.raises(TTSError):
        await synthesize_track(sample_script(), 2, tmp_path, engine, max_attempts=1)
    assert engine.attempts == 1


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


@pytest.mark.asyncio
async def test_synthesize_track_reports_substep_progress(tmp_path):
    engine = FakeTTSEngine([3.0, 4.0, 5.0])
    reporter = FakeReporter()
    await synthesize_track(sample_script(), 2, tmp_path, engine, reporter=reporter)
    assert reporter.calls == [
        ("substep", "voice", 1, 3, "第一句。"),
        ("substep", "voice", 2, 3, "第二句。"),
        ("substep", "voice", 3, 3, "第三句。"),
    ]


# --- EdgeTTSEngine：原子落盘、时长体检、代理/超时 ---


class _StubCommunicate:
    """假的 edge_tts.Communicate。绝不联网，只记录构造参数并按脚本行为落盘。"""

    calls: list[dict] = []

    def __init__(
        self,
        text,
        voice,
        *,
        rate="+0%",
        proxy=None,
        connect_timeout=10,
        receive_timeout=60,
    ):
        self.text = text
        type(self).calls.append(
            {
                "text": text,
                "voice": voice,
                "rate": rate,
                "proxy": proxy,
                "connect_timeout": connect_timeout,
                "receive_timeout": receive_timeout,
            }
        )

    # 由 _stub_edge_tts 注入
    behaviour = "ok"

    async def save(self, audio_fname):
        path = Path(audio_fname)
        type(self).calls[-1]["saved_to"] = path
        if type(self).behaviour == "stall":
            path.write_bytes(b"partial")
            await asyncio.sleep(30)
        if type(self).behaviour == "boom":
            # 跟真 edge-tts 一样：流式写了一半才炸，磁盘上留一个非空但截断的文件。
            path.write_bytes(b"partial")
            raise ConnectionResetError("断流")
        path.write_bytes(b"fake mp3")


def _stub_edge_tts(monkeypatch, *, behaviour="ok"):
    module = types.ModuleType("edge_tts")
    _StubCommunicate.calls = []
    _StubCommunicate.behaviour = behaviour
    module.Communicate = _StubCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    return _StubCommunicate


async def test_edge_engine_synthesizes_into_a_part_file_then_renames(tmp_path, monkeypatch):
    """edge_tts 的 save() 是流式 open(fname,"wb")，中断就留一个截断的 mp3。
    所以正式路径上永远只能出现「已经体检过」的文件。"""
    stub = _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: 2.0)
    out = tmp_path / "chunk_001.mp3"

    duration = await EdgeTTSEngine().synthesize("第一句。", out)

    assert duration == 2.0
    assert stub.calls[0]["saved_to"].name.endswith(".part")
    assert out.is_file()
    assert list(tmp_path.glob("*.part")) == []


async def test_edge_engine_leaves_nothing_behind_when_save_fails(tmp_path, monkeypatch):
    _stub_edge_tts(monkeypatch, behaviour="boom")
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: 2.0)
    out = tmp_path / "chunk_001.mp3"

    with pytest.raises(ConnectionResetError):
        await EdgeTTSEngine().synthesize("第一句。", out)

    assert not out.exists()
    assert list(tmp_path.glob("*.part")) == []


async def test_edge_engine_rejects_zero_duration_audio(tmp_path, monkeypatch):
    _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: 0.0)
    out = tmp_path / "chunk_001.mp3"

    with pytest.raises(TTSError):
        await EdgeTTSEngine().synthesize("第一句。", out)

    assert not out.exists()
    assert list(tmp_path.glob("*.part")) == []


async def test_edge_engine_rejects_truncated_audio(tmp_path, monkeypatch):
    """90 字的旁白按 4.5 字/秒该有 20 秒；只出 1 秒说明流被截断了。"""
    _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: 1.0)
    with pytest.raises(TTSError) as exc:
        await EdgeTTSEngine().synthesize("一" * 90, tmp_path / "chunk_001.mp3")
    assert "时长" in str(exc.value)


async def test_edge_engine_rejects_absurdly_long_audio(tmp_path, monkeypatch):
    _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: 300.0)
    with pytest.raises(TTSError):
        await EdgeTTSEngine().synthesize("一" * 90, tmp_path / "chunk_001.mp3")


@pytest.mark.parametrize(
    ("text", "duration"),
    [
        # 实测 115 个真实 chunk 的 实测/估算 比落在 0.809–1.179，band 是 ±50% 再加
        # 2 秒固定余量，所以整个实测区间都在里面。
        ("一" * 88, 15.816),
        ("一" * 135, 25.584),
        ("一" * 14, 3.648),
        # 短句的固定开销（首尾静音）会把比值顶得很高，靠 2 秒的绝对余量兜住。
        ("好。", 1.2),
    ],
)
async def test_edge_engine_accepts_real_world_durations(tmp_path, monkeypatch, text, duration):
    _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: duration)
    assert await EdgeTTSEngine().synthesize(text, tmp_path / "c.mp3") == duration


async def test_edge_engine_duration_band_follows_the_rate(tmp_path, monkeypatch):
    """rate=+100% 是两倍速，同样的字数只该出一半时长。"""
    _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: 10.0)
    # 90 字 / 4.5 = 20 秒；+100% 后期望 10 秒，10.0 正中靶心。
    assert await EdgeTTSEngine(rate="+100%").synthesize("一" * 90, tmp_path / "a.mp3") == 10.0
    # 同样 10 秒在 +0% 下也仍在 band 内（下界 20*0.5-2 = 8），所以要用一个更极端的值
    # 才能证明 rate 真的进了公式：0% 时 40 秒偏高（上界 32），100% 时 40 秒更偏高。
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: 5.0)
    with pytest.raises(TTSError):
        # +0% 下 5 秒远低于下界 8 秒。
        await EdgeTTSEngine(rate="+0%").synthesize("一" * 90, tmp_path / "b.mp3")
    # 而 +100% 下期望 10 秒，下界 10*0.5-2 = 3，5 秒合法。
    assert await EdgeTTSEngine(rate="+100%").synthesize("一" * 90, tmp_path / "c.mp3") == 5.0


async def test_edge_engine_passes_proxy_and_timeouts_to_communicate(tmp_path, monkeypatch):
    stub = _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: 2.0)
    engine = EdgeTTSEngine(
        voice="zh-CN-XiaoxiaoNeural",
        rate="+10%",
        proxy="http://127.0.0.1:8080",
        connect_timeout=3,
        receive_timeout=17,
    )
    await engine.synthesize("第一句。", tmp_path / "c.mp3")
    call = stub.calls[0]
    assert call["voice"] == "zh-CN-XiaoxiaoNeural"
    assert call["rate"] == "+10%"
    assert call["proxy"] == "http://127.0.0.1:8080"
    assert call["connect_timeout"] == 3
    assert call["receive_timeout"] == 17


async def test_edge_engine_gives_up_on_a_stalled_chunk(tmp_path, monkeypatch):
    """edge-tts 内部的 sock_read=60 只管单次读，涓涓细流能挂远超 60 秒。"""
    _stub_edge_tts(monkeypatch, behaviour="stall")
    monkeypatch.setattr(tts_module, "probe_duration", lambda path: 2.0)
    out = tmp_path / "chunk_001.mp3"

    with pytest.raises(TTSError) as exc:
        await EdgeTTSEngine(chunk_timeout_seconds=0.01).synthesize("第一句。", out)

    assert "超时" in str(exc.value)
    assert not out.exists()
    assert list(tmp_path.glob("*.part")) == []


async def test_edge_engine_probes_duration_off_the_event_loop(tmp_path, monkeypatch):
    """probe_duration 是阻塞 subprocess，直接在 async 里调会卡住事件循环，
    P2-B 的 TTS 并发就白做了。"""
    _stub_edge_tts(monkeypatch)
    threads: list[int] = []

    def spy(path):
        threads.append(threading.get_ident())
        return 2.0

    monkeypatch.setattr(tts_module, "probe_duration", spy)
    await EdgeTTSEngine().synthesize("第一句。", tmp_path / "c.mp3")
    assert threads and threads[0] != threading.get_ident()
