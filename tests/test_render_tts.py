import asyncio
import re
import sys
import threading
import types
from pathlib import Path

import pytest

from tenmin.config import RenderConfig
from tenmin.models import AudioDirection, Beat, Hold, Script
from tenmin.render import tts as tts_module
from tenmin.render.chunks import is_pronounceable
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
    """concurrency=1 是刻意钉住的：FlakyTTSEngine 的 attempts 是**全局**计数，默认并发
    度下另外两个 chunk 也会各自贡献一次尝试，断言就不再是在测 max_attempts 转发了。
    并发下的同一条性质由 test_a_failure_stops_handing_out_new_chunks 覆盖。"""
    engine = FlakyTTSEngine(fail_times=1)
    with pytest.raises(TTSError):
        await synthesize_track(
            sample_script(), 2, tmp_path, engine, max_attempts=1, concurrency=1
        )
    assert engine.attempts == 1


async def test_synthesize_track_builds_chunks(tmp_path):
    engine = FakeTTSEngine([3.0, 4.0, 5.0])
    track, warnings = await synthesize_track(sample_script(), 2, tmp_path, engine)
    assert warnings == []
    # 序号仍然按顺序递增（人工试听要靠它），身份则由后面那段内容哈希管。
    assert [c.path.split(".")[0] for c in track.chunks] == [
        "chunk_001",
        "chunk_002",
        "chunk_003",
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
    track, _ = await synthesize_track(sample_script(), 2, tmp_path, engine)
    for chunk in track.chunks:
        assert (tmp_path / chunk.path).is_file()


async def test_synthesize_track_reuses_existing_chunk(tmp_path, monkeypatch):
    """第一轮落盘的 chunk，第二轮原样复用，engine 一次都不该被调。

    原来这个测试是手写一个 `chunk_001.mp3` 再断言 engine 只被调 2 次 —— 它锁死的正是
    「chunk 文件按位置序号命名、复用只看文件存在」这个 bug，所以必须改成先真跑一轮。
    """
    await synthesize_track(sample_script(), 2, tmp_path, FakeTTSEngine([3.0, 4.0, 5.0]))
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 9.0)

    engine = FakeTTSEngine([])
    track, _ = await synthesize_track(sample_script(), 2, tmp_path, engine)

    assert engine.calls == []
    assert [c.duration for c in track.chunks] == [9.0, 9.0, 9.0]


async def test_synthesize_track_reuse_false_resynthesizes(tmp_path, monkeypatch):
    await synthesize_track(sample_script(), 2, tmp_path, FakeTTSEngine([9.0, 9.0, 9.0]))
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
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: 2.0)
    out = tmp_path / "chunk_001.mp3"

    duration = await EdgeTTSEngine().synthesize("第一句。", out)

    assert duration == 2.0
    assert stub.calls[0]["saved_to"].name.endswith(".part")
    assert out.is_file()
    assert list(tmp_path.glob("*.part")) == []


async def test_edge_engine_leaves_nothing_behind_when_save_fails(tmp_path, monkeypatch):
    _stub_edge_tts(monkeypatch, behaviour="boom")
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: 2.0)
    out = tmp_path / "chunk_001.mp3"

    with pytest.raises(ConnectionResetError):
        await EdgeTTSEngine().synthesize("第一句。", out)

    assert not out.exists()
    assert list(tmp_path.glob("*.part")) == []


async def test_edge_engine_rejects_zero_duration_audio(tmp_path, monkeypatch):
    _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: 0.0)
    out = tmp_path / "chunk_001.mp3"

    with pytest.raises(TTSError):
        await EdgeTTSEngine().synthesize("第一句。", out)

    assert not out.exists()
    assert list(tmp_path.glob("*.part")) == []


async def test_edge_engine_rejects_truncated_audio(tmp_path, monkeypatch):
    """90 字的旁白按 4.5 字/秒该有 20 秒；只出 1 秒说明流被截断了。"""
    _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: 1.0)
    with pytest.raises(TTSError) as exc:
        await EdgeTTSEngine().synthesize("一" * 90, tmp_path / "chunk_001.mp3")
    assert "时长" in str(exc.value)


async def test_edge_engine_rejects_absurdly_long_audio(tmp_path, monkeypatch):
    _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: 300.0)
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
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: duration)
    assert await EdgeTTSEngine().synthesize(text, tmp_path / "c.mp3") == duration


async def test_edge_engine_duration_band_follows_the_rate(tmp_path, monkeypatch):
    """rate=+100% 是两倍速，同样的字数只该出一半时长。"""
    _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: 10.0)
    # 90 字 / 4.5 = 20 秒；+100% 后期望 10 秒，10.0 正中靶心。
    assert await EdgeTTSEngine(rate="+100%").synthesize("一" * 90, tmp_path / "a.mp3") == 10.0
    # 同样 10 秒在 +0% 下也仍在 band 内（下界 20*0.5-2 = 8），所以要用一个更极端的值
    # 才能证明 rate 真的进了公式：0% 时 40 秒偏高（上界 32），100% 时 40 秒更偏高。
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: 5.0)
    with pytest.raises(TTSError):
        # +0% 下 5 秒远低于下界 8 秒。
        await EdgeTTSEngine(rate="+0%").synthesize("一" * 90, tmp_path / "b.mp3")
    # 而 +100% 下期望 10 秒，下界 10*0.5-2 = 3，5 秒合法。
    assert await EdgeTTSEngine(rate="+100%").synthesize("一" * 90, tmp_path / "c.mp3") == 5.0


async def test_edge_engine_passes_proxy_and_timeouts_to_communicate(tmp_path, monkeypatch):
    stub = _stub_edge_tts(monkeypatch)
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: 2.0)
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
    monkeypatch.setattr(tts_module, "probe_duration", lambda path, **_: 2.0)
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

    def spy(path, **_):
        threads.append(threading.get_ident())
        return 2.0

    monkeypatch.setattr(tts_module, "probe_duration", spy)
    await EdgeTTSEngine().synthesize("第一句。", tmp_path / "c.mp3")
    assert threads and threads[0] != threading.get_ident()


# --- chunk 缓存必须绑内容，不能只绑位置 ---


def _rewritten_script() -> Script:
    """跟 sample_script() 的 chunk 数量与顺序完全一致，只改了第二个 chunk 的字。"""
    script = sample_script()
    script.beats[0].narration = "第一句。改写过的第二句。"
    return script


def test_chunk_filenames_keep_the_serial_and_carry_a_content_hash():
    assert re.fullmatch(r"chunk_007\.[0-9a-f]{8}\.mp3", tts_module.chunk_filename(7, "abc", "fp"))


async def test_rewritten_narration_is_not_silently_reused(tmp_path):
    """剧本改写后旧音频被原样复用 = 成片旁白跟它自己的字幕不符，且零警告。"""
    previous, _ = await synthesize_track(
        sample_script(), 2, tmp_path, FakeTTSEngine([3.0, 4.0, 5.0])
    )

    engine = FakeTTSEngine([4.5])
    track, _ = await synthesize_track(
        _rewritten_script(), 2, tmp_path, engine, previous=previous
    )

    assert [c["text"] for c in engine.calls] == ["改写过的第二句。"]
    assert [c.duration for c in track.chunks] == [3.0, 4.5, 5.0]


async def test_changing_the_voice_invalidates_the_cache(tmp_path):
    """voice / rate 必须进哈希，否则改了 RenderConfig.voice 也会复用旧音色。"""
    await synthesize_track(sample_script(), 2, tmp_path, FakeTTSEngine([3.0, 4.0, 5.0]))

    other = FakeTTSEngine([1.0, 2.0, 3.0], fingerprint="fake|另一个音色|+10%")
    await synthesize_track(sample_script(), 2, tmp_path, other)

    assert len(other.calls) == 3


async def test_reuse_matches_content_even_after_the_serial_shifts(tmp_path):
    """chunk 数量变了会让后面所有序号平移，但内容没变的那些仍该复用。"""
    previous, _ = await synthesize_track(
        sample_script(), 2, tmp_path, FakeTTSEngine([3.0, 4.0, 5.0])
    )

    shifted = sample_script()
    shifted.beats.insert(
        0, Beat(id="b0", label="新开场", role="hook", narration="插进来的新第一句。")
    )
    engine = FakeTTSEngine([1.0])
    track, _ = await synthesize_track(shifted, 2, tmp_path, engine, previous=previous)

    assert [c["text"] for c in engine.calls] == ["插进来的新第一句。"]
    assert [c.duration for c in track.chunks] == [1.0, 3.0, 4.0, 5.0]


async def test_reuse_prefers_the_duration_recorded_in_voice_json(tmp_path, monkeypatch):
    """复用路径原来每个 chunk 都要 spawn 一次 ffprobe，而时长早就记在 voice.json 里了。"""
    previous, _ = await synthesize_track(
        sample_script(), 2, tmp_path, FakeTTSEngine([3.0, 4.0, 5.0])
    )

    def boom(path, **_):
        raise AssertionError("时长已经记在 voice.json 里了，不该再 spawn ffprobe")

    monkeypatch.setattr("tenmin.render.tts.probe_duration", boom)
    track, _ = await synthesize_track(
        sample_script(), 2, tmp_path, FakeTTSEngine([]), previous=previous
    )
    assert [c.duration for c in track.chunks] == [3.0, 4.0, 5.0]


async def test_reuse_probes_when_the_duration_is_not_recorded(tmp_path, monkeypatch):
    await synthesize_track(sample_script(), 2, tmp_path, FakeTTSEngine([3.0, 4.0, 5.0]))
    probed: list[Path] = []

    def spy(path, **_):
        probed.append(path)
        return 9.0

    monkeypatch.setattr("tenmin.render.tts.probe_duration", spy)
    track, _ = await synthesize_track(sample_script(), 2, tmp_path, FakeTTSEngine([]))
    assert len(probed) == 3
    assert [c.duration for c in track.chunks] == [9.0, 9.0, 9.0]


async def test_recorded_duration_is_only_trusted_for_the_matching_file(tmp_path, monkeypatch):
    """voice.json 里记的是「哪个文件多长」；哈希对不上的条目一律不能用。"""
    previous, _ = await synthesize_track(
        sample_script(), 2, tmp_path, FakeTTSEngine([3.0, 4.0, 5.0])
    )
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 9.0)
    track, _ = await synthesize_track(
        _rewritten_script(),
        2,
        tmp_path,
        FakeTTSEngine([4.5]),
        previous=previous,
    )
    assert [c.duration for c in track.chunks] == [3.0, 4.5, 5.0]


async def test_duplicate_text_in_one_run_reuses_without_probing(tmp_path, monkeypatch):
    """同一段文字在剧本里出现两次：第二个 chunk 命中第一个的文件，不该再 spawn ffprobe。"""
    script = Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(id="b1", label="A", role="hook", narration="一模一样的一句。"),
            Beat(id="b2", label="B", role="outro", narration="一模一样的一句。"),
        ],
    )

    def boom(path, **_):
        raise AssertionError("这一轮刚合成过它，时长在内存里，不该 spawn ffprobe")

    monkeypatch.setattr("tenmin.render.tts.probe_duration", boom)
    engine = FakeTTSEngine([6.0])
    track, _ = await synthesize_track(script, 2, tmp_path, engine)

    assert len(engine.calls) == 1
    assert [c.duration for c in track.chunks] == [6.0, 6.0]
    assert track.chunks[0].path == track.chunks[1].path


# --- 不可发音的 chunk ---


def _quote_only_script(*, holds: list[Hold]) -> Script:
    """复刻实测事故的**原始输入**：work/saijo 的 E05 旁白用 `'…'` 当引号。

    P2-E A1 之前，chunks.split_sentences 在 `。` 之后就断开，把收尾的 `'` 留成一个独立
    片段；hold 一旦落在它上面，它就自己成为一个 chunk，Edge TTS 抛 NoAudioReceived，
    重试三次后整次运行中止。A1 之后这个输入在**上游**就不会再切出孤立引号了
    （见 test_the_incident_input_no_longer_produces_a_lone_quote_chunk）。
    """
    return Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(
                id="b1",
                label="A",
                role="hook",
                narration="第一句。第二句。'",
                audio=AudioDirection(holds=holds),
            )
        ],
    )


def test_the_incident_input_no_longer_produces_a_lone_quote_chunk():
    """P2-E A1 的回归测试：这条原来锁的是「事故输入真的会切出孤立 `'` chunk」。

    A1 把收尾符号吸收进前一句之后，同一个输入切出来的每个 chunk 都有内容可读，
    孤立引号从根上没有了 —— 所以这条从「证明症状存在」翻成「证明症状消失」。
    tts 层那道跳过不可发音 chunk 的闸门仍然保留（防御纵深），由下面几条用
    monkeypatch 直接喂输入来测。
    """
    from tenmin.render.chunks import plan_chunks

    script = _quote_only_script(
        holds=[Hold(at=1.8, duration=2.0, quote="q"), Hold(at=2.0, duration=3.0, quote="q")]
    )
    planned = plan_chunks(script.beats[0])
    assert planned == [("第一句。第二句。'", 5.0)]
    assert all(is_pronounceable(text) for text, _ in planned)


@pytest.fixture
def forced_plan(monkeypatch):
    """把 tts.py 看到的切句结果**直接**换掉，用来测它自己那道闸。

    为什么要 monkeypatch 而不是找一段真旁白：A1 之后 split_sentences 保证「切出来的
    每一句都有可发音内容」（唯一例外是整段旁白一个字都读不出来，那时只有一个 chunk），
    所以「多 chunk 里夹一个不可发音的」这种输入**上游已经产不出来了**。而 tts 层这道闸
    要顶住的正是「上游哪天又变了」，它保护的东西（被跳过的 chunk 带的留白不能凭空消失，
    否则此后整条时间轴前移）值得单独锁住。
    """

    def install(planned: list[tuple[str, float]]) -> None:
        monkeypatch.setattr(tts_module, "plan_chunks", lambda beat, **_: list(planned))

    return install


async def test_unpronounceable_chunk_is_skipped_and_its_hold_folded_back(
    tmp_path, forced_plan
):
    forced_plan([("第一句。第二句。", 2.0), ("'", 3.0)])
    engine = FakeTTSEngine([6.0])
    script = _quote_only_script(holds=[])

    track, warnings = await synthesize_track(script, 2, tmp_path, engine)

    assert [c["text"] for c in engine.calls] == ["第一句。第二句。"]
    assert [c.text for c in track.chunks] == ["第一句。第二句。"]
    # 被跳过的那个 chunk 带的 3 秒留白不能凭空消失，否则时间轴整体前移。
    assert [c.hold_after for c in track.chunks] == [5.0]
    assert track.total_seconds == pytest.approx(11.0)
    assert any("b1" in w and "'" in w for w in warnings)


async def test_beat_with_only_unpronounceable_text_is_skipped(tmp_path):
    engine = FakeTTSEngine([])
    script = Script(
        show="剧名",
        episodes=[2],
        beats=[Beat(id="b1", label="A", role="hook", narration="——")],
    )

    track, warnings = await synthesize_track(script, 2, tmp_path, engine)

    assert track.chunks == []
    assert engine.calls == []
    assert any("b1" in w for w in warnings)


async def test_hold_on_a_leading_unpronounceable_chunk_is_reported_not_swallowed(
    tmp_path, forced_plan
):
    """前面没有任何 chunk 可以挂的留白只能丢，但必须吭一声。"""
    forced_plan([("'", 4.0), ("第二句。", 0.0)])
    script = Script(
        show="剧名",
        episodes=[2],
        beats=[Beat(id="b1", label="A", role="hook", narration="'。第二句。")],
    )
    track, warnings = await synthesize_track(script, 2, tmp_path, FakeTTSEngine([5.0]))
    assert [c.text for c in track.chunks] == ["第二句。"]
    assert [c.hold_after for c in track.chunks] == [0.0]
    assert any("留白" in w for w in warnings)


async def test_progress_total_excludes_skipped_chunks(tmp_path, forced_plan):
    forced_plan([("第一句。第二句。", 2.0), ("'", 3.0)])
    reporter = FakeReporter()
    script = _quote_only_script(holds=[])
    await synthesize_track(script, 2, tmp_path, FakeTTSEngine([6.0]), reporter=reporter)
    assert reporter.calls == [("substep", "voice", 1, 1, "第一句。第二句。")]


# --- rate 一路传到切句（P2-E A2）-------------------------------------------


def _rate_sensitive_script() -> Script:
    """hold.at=2.4 落在两个 rate 的「句边界中点」之间，所以 chunk 划分会随 rate 翻面。

    +0% 的句偏移是 [2.222, 2.889] → 留白落在第 0 句后面，切成两个 chunk；
    +20% 是 [1.852, 2.407] → 落在第 1 句（最后一句）后面，整段只有一个 chunk。
    """
    return Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(
                id="b1",
                label="A",
                role="hook",
                narration="一二三四五六七八九。甲乙。",
                audio=AudioDirection(holds=[Hold(at=2.4, duration=2.0, quote="q")]),
            )
        ],
    )


async def test_synthesize_track_default_rate_keeps_the_historical_split(tmp_path):
    engine = FakeTTSEngine([4.0, 2.0])
    track, _ = await synthesize_track(_rate_sensitive_script(), 2, tmp_path, engine)
    assert [c.text for c in track.chunks] == ["一二三四五六七八九。", "甲乙。"]


async def test_synthesize_track_passes_rate_down_to_chunk_planning(tmp_path):
    """rate 不传下去的话，用户把 render.rate 调成 +20% 之后留白就插错句子。"""
    engine = FakeTTSEngine([5.0])
    track, _ = await synthesize_track(
        _rate_sensitive_script(), 2, tmp_path, engine, rate="+20%"
    )
    assert [c.text for c in track.chunks] == ["一二三四五六七八九。甲乙。"]
    assert [c.hold_after for c in track.chunks] == [2.0]


# --- 并发合成（tts_concurrency）---


class _ConcurrencySpyEngine:
    """按文本查表返回时长的假引擎，同时记录并发峰值与真实的合成顺序。

    刻意不复用 FakeTTSEngine：它按调用顺序 pop 时长，而并发下调用顺序本来就不确定，
    「哪个 chunk 拿到哪个时长」会变成一个随机数。这里按文本查表，结果与顺序无关。
    """

    fingerprint = "fake|voice|+0%"

    def __init__(self, durations: dict[str, float], delays: dict[str, float] | None = None):
        self.durations = durations
        self.delays = delays or {}
        self.in_flight = 0
        self.peak = 0
        self.started: list[str] = []
        self.finished: list[str] = []

    async def synthesize(self, text: str, out_path: Path) -> float:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.started.append(text)
        try:
            await asyncio.sleep(self.delays.get(text, 0.0))
        finally:
            self.in_flight -= 1
        self.finished.append(text)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake mp3")
        return self.durations[text]


def _wide_script(count: int) -> Script:
    """count 个 beat，每个正好一句，用来把并发度撑开。"""
    return Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(id=f"b{i}", label="A", role="hook", narration=f"第{i}句。")
            for i in range(count)
        ],
    )


def _wide_durations(count: int) -> dict[str, float]:
    return {f"第{i}句。": 1.0 + i for i in range(count)}


async def test_synthesize_track_is_serial_at_concurrency_one(tmp_path):
    """concurrency=1：任何时刻只有一个 chunk 在合成，且合成顺序 = 计划顺序
    （零行为变更的底线）。"""
    engine = _ConcurrencySpyEngine(
        _wide_durations(6), {f"第{i}句。": 0.01 for i in range(6)}
    )
    await synthesize_track(_wide_script(6), 2, tmp_path, engine, concurrency=1)
    assert engine.peak == 1
    assert engine.started == [f"第{i}句。" for i in range(6)]


async def test_synthesize_track_default_concurrency_follows_render_config(tmp_path):
    """不传 concurrency 时用 RenderConfig 的默认值，别在这里写第二份字面量。"""
    count = RenderConfig().tts_concurrency + 4
    engine = _ConcurrencySpyEngine(
        _wide_durations(count), {f"第{i}句。": 0.02 for i in range(count)}
    )
    await synthesize_track(_wide_script(count), 2, tmp_path, engine)
    assert engine.peak == RenderConfig().tts_concurrency


async def test_synthesize_track_honours_the_concurrency_limit(tmp_path):
    engine = _ConcurrencySpyEngine(
        _wide_durations(9), {f"第{i}句。": 0.02 for i in range(9)}
    )
    await synthesize_track(_wide_script(9), 2, tmp_path, engine, concurrency=3)
    assert engine.peak == 3


async def test_synthesize_track_actually_overlaps_the_network_waits(tmp_path):
    """并发的全部意义：墙钟时间接近 max 而不是 sum。

    每个 chunk 睡 0.05 秒，8 个串行至少 0.4 秒；并发 8 应该 ~0.05 秒。
    """
    engine = _ConcurrencySpyEngine(
        _wide_durations(8), {f"第{i}句。": 0.05 for i in range(8)}
    )
    started = asyncio.get_running_loop().time()
    await synthesize_track(_wide_script(8), 2, tmp_path, engine, concurrency=8)
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.2


async def test_synthesize_track_keeps_chunk_order_under_concurrency(tmp_path):
    """chunks 的顺序决定 render/audio.py 的 adelay 偏移与字幕顺序，绝不能跟着完成顺序走。

    延迟刻意反着排（最后一句最快），所以完成顺序必然是倒序。
    """
    count = 5
    delays = {f"第{i}句。": 0.01 * (count - i) for i in range(count)}
    engine = _ConcurrencySpyEngine(_wide_durations(count), delays)
    track, _ = await synthesize_track(
        _wide_script(count), 2, tmp_path, engine, concurrency=count
    )
    assert engine.finished == [f"第{i}句。" for i in reversed(range(count))]
    assert [c.text for c in track.chunks] == [f"第{i}句。" for i in range(count)]
    assert [c.beat_id for c in track.chunks] == [f"b{i}" for i in range(count)]
    assert [c.duration for c in track.chunks] == [1.0 + i for i in range(count)]
    assert [c.path.split(".")[0] for c in track.chunks] == [
        f"chunk_{i + 1:03d}" for i in range(count)
    ]


async def test_substep_is_a_completion_counter_under_concurrency(tmp_path):
    """原来报的是循环下标，并发下它会乱序。改成完成计数器后 current 必须严格 1..N 递增，
    label 是刚刚完成的那一句。"""
    count = 5
    delays = {f"第{i}句。": 0.01 * (count - i) for i in range(count)}
    engine = _ConcurrencySpyEngine(_wide_durations(count), delays)
    reporter = FakeReporter()
    await synthesize_track(
        _wide_script(count), 2, tmp_path, engine, concurrency=count, reporter=reporter
    )
    substeps = [c for c in reporter.calls if c[0] == "substep"]
    assert [c[2] for c in substeps] == [1, 2, 3, 4, 5]
    assert all(c[3] == count for c in substeps)
    # 完成顺序是倒序，所以 label 也必须是倒序 —— 它描述的是「刚完成的那一句」。
    assert [c[4] for c in substeps] == [f"第{i}句。" for i in reversed(range(count))]


async def test_duplicate_text_is_synthesized_once_even_under_concurrency(tmp_path, monkeypatch):
    """并发下两个同文本 chunk 会同时错过缓存 → 各自去 Edge TTS 合成一遍。

    序号不同所以 `.part` 其实不会互撞（用户报告里猜的那个失败模式不成立），但
    voice.json 会指向两个内容相同的文件，而串行下它们指向同一个 —— 一次白花的网络
    往返 + 与串行不一致的产物。
    """
    script = Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(id="b1", label="A", role="hook", narration="一模一样的一句。"),
            Beat(id="b2", label="B", role="outro", narration="一模一样的一句。"),
        ],
    )

    def boom(path, **_):
        raise AssertionError("这一轮刚合成过它，时长在内存里，不该 spawn ffprobe")

    monkeypatch.setattr("tenmin.render.tts.probe_duration", boom)
    engine = _ConcurrencySpyEngine(
        {"一模一样的一句。": 6.0}, {"一模一样的一句。": 0.02}
    )
    track, _ = await synthesize_track(script, 2, tmp_path, engine, concurrency=4)

    assert engine.started == ["一模一样的一句。"]
    assert [c.duration for c in track.chunks] == [6.0, 6.0]
    assert track.chunks[0].path == track.chunks[1].path


async def test_a_failing_chunk_raises_tts_error_not_an_exception_group(tmp_path, sleeps):
    """TaskGroup 把异常包成 ExceptionGroup，而 cli.PIPELINE_ERRORS 认的是 TTSError 本身
    —— 不解包的话用户拿到的是一整页 traceback。"""
    engine = FailingTTSEngine(ConnectionResetError("断了"))
    with pytest.raises(TTSError) as exc:
        await synthesize_track(_wide_script(4), 2, tmp_path, engine, concurrency=4)
    assert not isinstance(exc.value, BaseExceptionGroup)
    assert "ConnectionResetError" in str(exc.value)


async def test_a_failing_chunk_keeps_the_chunks_that_already_landed(tmp_path, sleeps):
    """一个 chunk 彻底失败时 TaskGroup 会取消其余 task。已经原子落盘的要留着（下次靠
    哈希复用），被取消的 task 留下的 `.part` 要清掉。"""

    class _OneBadApple:
        fingerprint = "fake|voice|+0%"

        async def synthesize(self, text: str, out_path: Path) -> float:
            if text == "第2句。":
                raise ConnectionResetError("断了")
            if text == "第3句。":
                # 慢到必然还在飞的时候就被取消，模拟「留下 .part 的那个 task」。
                out_path.with_name(out_path.name + ".part").write_bytes(b"partial")
                try:
                    await asyncio.sleep(10)
                except BaseException:
                    out_path.with_name(out_path.name + ".part").unlink(missing_ok=True)
                    raise
            out_path.write_bytes(b"fake mp3")
            return 2.0

    with pytest.raises(TTSError):
        await synthesize_track(
            _wide_script(4), 2, tmp_path, _OneBadApple(), concurrency=4, max_attempts=1
        )
    assert list(tmp_path.glob("*.part")) == []
    assert (tmp_path / tts_module.chunk_filename(1, "第0句。", "fake|voice|+0%")).is_file()


async def test_a_failure_stops_handing_out_new_chunks(tmp_path, sleeps):
    """并发实现如果是「每个 chunk 一个 task + Semaphore 限流」，N 个 task 会在同一轮
    事件循环里全被建出来，首个 chunk 失败时它们照样各发一次 Edge TTS 请求。

    worker 池 + 共享游标才有「游标推不动就没有新活」这个性质：8 个 chunk、并发 2、
    第一个就失败 —— 最多只该有 2 个 chunk 被碰过（那两个已经在飞的）。
    """

    class _FirstOneFails:
        fingerprint = "fake|voice|+0%"

        def __init__(self):
            self.seen: list[str] = []

        async def synthesize(self, text: str, out_path: Path) -> float:
            self.seen.append(text)
            if text == "第0句。":
                raise ConnectionResetError("断了")
            await asyncio.sleep(0.02)
            out_path.write_bytes(b"fake mp3")
            return 2.0

    engine = _FirstOneFails()
    with pytest.raises(TTSError):
        await synthesize_track(
            _wide_script(8), 2, tmp_path, engine, concurrency=2, max_attempts=1
        )
    assert len(engine.seen) <= 2


async def test_a_failing_chunk_keeps_the_original_error_in_the_chain(tmp_path, sleeps):
    """解包 ExceptionGroup 时不能把叶子异常自己的 __cause__ 弄丢。

    `raise leaf from group.__cause__` 看着对，但 TaskGroup 抛的组 `__cause__` 是 None，
    于是等价于 `raise leaf from None` —— 它会把 synthesize_with_retry 好不容易挂上去的
    那个 ConnectionResetError 从 __cause__ 里抹掉，traceback 里再也看不到真正的网络错误
    （而 aiohttp 那些异常 stringify 常常是空串，这条链是唯一的线索）。
    """
    original = ConnectionResetError("断了")
    engine = FailingTTSEngine(original)
    with pytest.raises(TTSError) as exc:
        await synthesize_track(
            _wide_script(2), 2, tmp_path, engine, concurrency=2, max_attempts=1
        )
    assert exc.value.__cause__ is original
    # ExceptionGroup 本身是噪音，不该再作为 context 印一遍。
    assert exc.value.__suppress_context__


@pytest.mark.parametrize("text", ["Q3 财报。", "第一句。", "ABC。", "２０２５。"])
def test_normal_text_stays_pronounceable(text):
    assert tts_module._is_pronounceable(text)


@pytest.mark.parametrize("text", ["'", "——", "，、。", "「」", "  ", "…"])
def test_punctuation_only_text_is_not_pronounceable(text):
    assert not tts_module._is_pronounceable(text)


def test_edge_engine_uses_configured_ffprobe(monkeypatch, tmp_path):
    """voice 阶段的时长体检也要用 render.ffprobe_path。

    不接线的话「PATH 上没有 ffprobe、只配了 ffprobe_path」的用户会看到 render 能跑
    但 voice 阶段炸 —— 半接线比不接线更难查。
    """
    from tenmin.render import tts as tts_module

    seen: dict[str, str] = {}

    def fake_probe(path, *, ffprobe="ffprobe"):
        seen["ffprobe"] = ffprobe
        return 2.0

    monkeypatch.setattr(tts_module, "probe_duration", fake_probe)

    class _FakeCommunicate:
        def __init__(self, *a, **k):
            pass

        async def save(self, path):
            Path(path).write_bytes(b"mp3")

    monkeypatch.setitem(
        __import__("sys").modules, "edge_tts", type("m", (), {"Communicate": _FakeCommunicate})
    )
    engine = tts_module.EdgeTTSEngine(ffprobe="/opt/x/ffprobe")
    asyncio.run(engine.synthesize("九个字的一句话", tmp_path / "c.mp3"))
    assert seen["ffprobe"] == "/opt/x/ffprobe"


def test_build_tts_engine_wires_ffprobe_path():
    from tenmin.config import RenderConfig
    from tenmin.render.tts import build_tts_engine

    engine = build_tts_engine(RenderConfig(ffprobe_path="/opt/x/ffprobe"))
    assert engine.ffprobe == "/opt/x/ffprobe"


async def test_synthesize_track_surfaces_bad_hold_warnings(tmp_path):
    """chunks.assign_holds 报的坏 hold 要一路冒到 voice 阶段的 warnings 里（P2-E A3）。

    voice 是这条流水线上第一个真正**消费** hold.at 的阶段，也是人工改完
    03_script/*.json 之后第一个跑到的阶段（validate_script() 只在 script 阶段跑）。
    """
    script = Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(
                id="b1",
                label="Hook",
                role="hook",
                narration="第一句。第二句。",
                audio=AudioDirection(holds=[Hold(at=900.0, duration=2.0, quote="金句")]),
            )
        ],
    )
    _, warnings = await synthesize_track(script, 2, tmp_path, FakeTTSEngine([5.0]))
    assert any("900.0" in w and "最后一句" in w for w in warnings)
