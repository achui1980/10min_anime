"""对白轨来源的三岔判据。ffprobe/ffmpeg/转写全部 fake —— 这一层的职责只是「选哪条路」。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tenmin.config import AsrConfig
from tenmin.ingest import resolve


@pytest.fixture
def stub(monkeypatch):
    """把三条路的出口都换成记账用的假货。

    `calls["kwargs"]` 单独存一份最后一次的关键字参数：可执行文件路径与 AsrConfig 的
    转发只能在这里当场查（调用完就没了），而把它们混进上面那三个列表会让
    「calls["probe"] == []」这类形状断言跟着一起变糊。
    """
    calls: dict[str, list | dict] = {"probe": [], "extract": [], "transcribe": [], "kwargs": {}}
    state = {"has_subtitle": False}

    def fake_has_subtitle(path, **kwargs):
        calls["probe"].append(Path(path))
        calls["kwargs"]["probe"] = kwargs
        return state["has_subtitle"]

    def fake_extract(video, dest, **kwargs):
        calls["extract"].append((Path(video), Path(dest)))
        calls["kwargs"]["extract"] = kwargs
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text("1\n00:00:01,000 --> 00:00:02,000\n抽出来的\n", encoding="utf-8")

    def fake_transcribe(video, dest, **kwargs):
        calls["transcribe"].append((Path(video), Path(dest)))
        calls["kwargs"]["transcribe"] = kwargs
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text("1\n00:00:01,000 --> 00:00:02,000\n転写した\n", encoding="utf-8")

    monkeypatch.setattr(resolve.ffmpeg, "has_subtitle_stream", fake_has_subtitle)
    monkeypatch.setattr(resolve.ffmpeg, "extract_subtitle_track", fake_extract)
    monkeypatch.setattr(resolve.asr, "transcribe", fake_transcribe)
    return calls, state


def _video(tmp_path: Path) -> Path:
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    return video


def _cache(tmp_path: Path) -> Path:
    return tmp_path / "srt" / "E11.asr.srt"


def _touch_relative_to(target: Path, video: Path, delta: float) -> None:
    """把 target 的 mtime 拨到 video 的 mtime ± delta。"""
    when = video.stat().st_mtime + delta
    os.utime(target, (when, when))


def test_a_handed_srt_wins_and_nothing_is_probed(tmp_path, stub):
    calls, _ = stub
    srt = tmp_path / "hand.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n手传\n", encoding="utf-8")

    source = resolve.resolve_subtitle_source(
        srt, _video(tmp_path), cache=_cache(tmp_path), asr_config=AsrConfig()
    )

    assert source.path == srt
    assert source.kind == "srt"
    assert calls["probe"] == []
    assert calls["transcribe"] == []


def test_a_handed_srt_works_without_any_video(tmp_path, stub):
    """视频对 render 阶段是硬需求，但对「对白轨从哪来」不是。"""
    srt = tmp_path / "hand.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n手传\n", encoding="utf-8")

    source = resolve.resolve_subtitle_source(
        srt, None, cache=_cache(tmp_path), asr_config=AsrConfig()
    )
    assert source == resolve.SubtitleSource(srt, "srt")


def test_an_embedded_subtitle_track_is_extracted_instead_of_transcribed(tmp_path, stub):
    """软字幕轨零成本零误差，比转写好得多 —— 这条分枝漏掉就是白付三分钟还掉质量。"""
    calls, state = stub
    state["has_subtitle"] = True
    cache = _cache(tmp_path)

    source = resolve.resolve_subtitle_source(
        None, _video(tmp_path), cache=cache, asr_config=AsrConfig()
    )

    assert source.kind == "srt"
    assert source.path.is_file()
    assert "抽出来的" in source.path.read_text(encoding="utf-8")
    assert calls["extract"]
    assert calls["transcribe"] == []


def test_an_existing_extracted_subtitle_is_re_extracted_anyway(tmp_path, stub):
    """软字幕那条路**不许**拿「文件在 + mtime 够新」当复用判据。

    抽取是 ffmpeg 直接流式写 dest，中途被打断留下的半份文件**语法合法**（SRT 没有文件尾
    结构），srt_parser 会零警告地把它解析成半份对白轨。所以这条路每次重抽，代价只是一次
    demux —— 跟下面转写那条路（原子落盘、失败不留文件，所以能安全复用）刻意不同。
    """
    calls, state = stub
    state["has_subtitle"] = True
    video = _video(tmp_path)
    cache = _cache(tmp_path)
    stale = resolve._embedded_dest(cache)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("1\n00:00:01,000 --> 00:00:02,000\n截断残骸\n", encoding="utf-8")
    _touch_relative_to(stale, video, +10)

    source = resolve.resolve_subtitle_source(None, video, cache=cache, asr_config=AsrConfig())

    assert len(calls["extract"]) == 1
    body = source.path.read_text(encoding="utf-8")
    assert "抽出来的" in body
    assert "截断残骸" not in body


def test_transcription_is_the_last_resort(tmp_path, stub):
    calls, state = stub
    state["has_subtitle"] = False
    cache = _cache(tmp_path)

    source = resolve.resolve_subtitle_source(
        None, _video(tmp_path), cache=cache, asr_config=AsrConfig()
    )

    assert source == resolve.SubtitleSource(cache, "asr")
    assert calls["transcribe"] == [(tmp_path / "e11.mp4", cache)]


def test_a_fresh_cache_is_reused_instead_of_retranscribing(tmp_path, stub):
    """一集转写要几分钟，--force 重跑 ingest 不该重付。"""
    calls, _ = stub
    video = _video(tmp_path)
    cache = _cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("1\n00:00:01,000 --> 00:00:02,000\n既存\n", encoding="utf-8")
    _touch_relative_to(cache, video, +10)

    source = resolve.resolve_subtitle_source(None, video, cache=cache, asr_config=AsrConfig())

    assert source.kind == "asr"
    assert "既存" in source.path.read_text(encoding="utf-8")
    assert calls["transcribe"] == []


def test_a_stale_cache_is_retranscribed(tmp_path, stub):
    """换了片源（重新压制、换了个版本）就得重转。"""
    calls, _ = stub
    cache = _cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("旧的\n", encoding="utf-8")
    video = _video(tmp_path)
    _touch_relative_to(cache, video, -10)

    source = resolve.resolve_subtitle_source(None, video, cache=cache, asr_config=AsrConfig())

    assert "転写した" in source.path.read_text(encoding="utf-8")
    assert calls["transcribe"]


def test_an_empty_cache_is_retranscribed(tmp_path, stub):
    """0 字节的缓存是上一次被打断留下的，不是产物。"""
    calls, _ = stub
    video = _video(tmp_path)
    cache = _cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("", encoding="utf-8")
    _touch_relative_to(cache, video, +10)

    resolve.resolve_subtitle_source(None, video, cache=cache, asr_config=AsrConfig())
    assert calls["transcribe"]


def test_neither_srt_nor_video_is_an_error(tmp_path, stub):
    with pytest.raises(ValueError) as excinfo:
        resolve.resolve_subtitle_source(
            None, None, cache=_cache(tmp_path), asr_config=AsrConfig()
        )
    assert "srt" in str(excinfo.value).lower() or "视频" in str(excinfo.value)


def test_a_missing_video_file_is_an_error(tmp_path, stub):
    with pytest.raises(FileNotFoundError):
        resolve.resolve_subtitle_source(
            None, tmp_path / "gone.mp4", cache=_cache(tmp_path), asr_config=AsrConfig()
        )


def test_a_missing_handed_srt_is_an_error(tmp_path, stub):
    with pytest.raises(FileNotFoundError):
        resolve.resolve_subtitle_source(
            tmp_path / "gone.srt",
            _video(tmp_path),
            cache=_cache(tmp_path),
            asr_config=AsrConfig(),
        )


def test_every_failure_mode_is_caught_by_the_cli_error_net():
    """逃出 PIPELINE_ERRORS 就意味着用户看到一整页 traceback。"""
    from tenmin.cli import PIPELINE_ERRORS

    assert issubclass(ValueError, PIPELINE_ERRORS)
    assert issubclass(FileNotFoundError, PIPELINE_ERRORS)


def test_the_extracted_subtitle_does_not_squat_the_asr_cache_name(tmp_path, stub):
    """从软字幕轨抽出来的东西不该占 ASR 缓存那个名字 —— 两者的处置**不一样**（那份能
    按 mtime 复用、这份每次重抽），混用同一个文件名会让人分不清手上这份 SRT 是抽的
    还是转的，也会让「复用」那半边的判据落到一份不该被信任的文件上。"""
    _, state = stub
    state["has_subtitle"] = True
    cache = _cache(tmp_path)

    source = resolve.resolve_subtitle_source(
        None, _video(tmp_path), cache=cache, asr_config=AsrConfig()
    )
    assert source.path != cache
    assert source.path.name.endswith(".embedded.srt")
    assert source.path.parent == cache.parent
    assert not cache.exists()


def test_the_configured_binaries_reach_the_embedded_path(tmp_path, stub):
    """`render.ffmpeg_path` 配成绝对路径的人（PATH 里没有 ffmpeg）全靠这一路转发。

    默认值等于 `ffmpeg`/`ffprobe` 时转发漏没漏是**看不出来的**，所以必须传不一样的值。
    """
    calls, state = stub
    state["has_subtitle"] = True

    resolve.resolve_subtitle_source(
        None,
        _video(tmp_path),
        cache=_cache(tmp_path),
        asr_config=AsrConfig(),
        ffmpeg_path="/opt/homebrew/bin/ffmpeg",
        ffprobe_path="/opt/homebrew/bin/ffprobe",
    )

    assert calls["kwargs"]["probe"] == {"ffprobe": "/opt/homebrew/bin/ffprobe"}
    assert calls["kwargs"]["extract"] == {"ffmpeg": "/opt/homebrew/bin/ffmpeg"}


def test_the_configured_ffmpeg_and_asr_config_reach_transcribe(tmp_path, stub):
    """转写那一路收的关键字是 `asr=`（它自己的参数名），而这边的形参叫 `asr_config`
    （模块里 `asr` 这个名字已经被导入的模块占了）。转错了配置会静默退回默认模型。"""
    calls, _ = stub
    config = AsrConfig(model="mlx-community/whisper-tiny", language="en")

    resolve.resolve_subtitle_source(
        None,
        _video(tmp_path),
        cache=_cache(tmp_path),
        asr_config=config,
        ffmpeg_path="/opt/homebrew/bin/ffmpeg",
    )

    assert calls["kwargs"]["transcribe"] == {
        "asr": config,
        "ffmpeg_path": "/opt/homebrew/bin/ffmpeg",
    }


def test_the_transcription_is_announced_before_it_starts(tmp_path, stub, capsys):
    """三条路里唯一一条既费时间又有损的。几分钟的静默会让人以为卡死了。"""
    resolve.resolve_subtitle_source(
        None, _video(tmp_path), cache=_cache(tmp_path), asr_config=AsrConfig()
    )
    assert "e11.mp4" in capsys.readouterr().out


def test_the_source_kind_is_only_srt_or_asr():
    """kind 是下游判「要不要繁转简」的唯一依据（日语过 OpenCC 会被改字），所以它的
    取值集合必须是封闭的。

    走 get_type_hints 而不是 `__annotations__`：resolve 有 `from __future__ import
    annotations`，直接读 `__annotations__` 拿到的是**字符串**，`== Literal[...]`
    永远为假。
    """
    from typing import Literal, get_args, get_type_hints

    field = get_type_hints(resolve.SubtitleSource)["kind"]
    assert field == Literal["srt", "asr"]
    assert set(get_args(field)) == {"srt", "asr"}
