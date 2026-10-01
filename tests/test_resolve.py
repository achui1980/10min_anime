"""对白轨来源的四岔判据。ffprobe/ffmpeg/OCR/转写全部 fake —— 这一层的职责只是「选哪条路」。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tenmin.config import AsrConfig, OcrConfig
from tenmin.ingest import resolve


@pytest.fixture
def stub(monkeypatch, capsys):
    """把三条路的出口都换成记账用的假货。

    `calls["kwargs"]` 单独存一份最后一次的关键字参数：可执行文件路径与 AsrConfig 的
    转发只能在这里当场查（调用完就没了），而把它们混进上面那三个列表会让
    「calls["probe"] == []」这类形状断言跟着一起变糊。

    `calls["announced_before"]` 存的是「转写开始那一刻 stdout 上已经有什么」。它也只能
    在这里当场取：测完再看 capsys 分不出那句告知是打在调用之前还是之后，而「之前」正是
    那句告知的**全部**价值（打在之后等于那几分钟静默照旧）。
    """
    calls: dict[str, list | dict] = {
        "probe": [],
        "extract": [],
        "transcribe": [],
        "recognize": [],
        "kwargs": {},
        "announced_before": [],
    }
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
        calls["announced_before"].append(capsys.readouterr().out)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text("1\n00:00:01,000 --> 00:00:02,000\n転写した\n", encoding="utf-8")

    def fake_recognize(video, dest, **kwargs):
        calls["recognize"].append((Path(video), Path(dest)))
        calls["kwargs"]["recognize"] = kwargs
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text("1\n00:00:01,000 --> 00:00:02,000\n認出來的\n", encoding="utf-8")

    monkeypatch.setattr(resolve.ffmpeg, "has_subtitle_stream", fake_has_subtitle)
    monkeypatch.setattr(resolve.ffmpeg, "extract_subtitle_track", fake_extract)
    monkeypatch.setattr(resolve.asr, "transcribe", fake_transcribe)
    monkeypatch.setattr(resolve.ocr, "recognize", fake_recognize)
    return calls, state


def _video(tmp_path: Path) -> Path:
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    return video


def _cache(tmp_path: Path) -> Path:
    return tmp_path / "srt" / "E11.asr.srt"


def _ocr_cache(tmp_path: Path) -> Path:
    return tmp_path / "srt" / "E11.ocr.srt"


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
    # 探的必须是**源视频**。探成 cache（那个文件此刻还不存在）会让这条分枝在真实运行里
    # 永远判 False，而 fake 照旧记一笔，形状断言看不出来。
    assert calls["probe"] == [tmp_path / "e11.mp4"]


def test_an_existing_extracted_subtitle_is_re_extracted_anyway(tmp_path, stub):
    """软字幕那条路**不许**拿「文件在 + mtime 够新」当复用判据。

    抽取是 ffmpeg 直接流式写 dest，中途被打断留下的半份文件**解析器不会报错**（SRT 没有
    文件尾结构，前缀即合法），只是对白少了一半。关键不是「一条警告都没有」—— 实测拿一份
    5 条 cue 的 SRT（209 字符）穷举全部 208 个截断点，140 个（67%）会多出至少一条
    skipped_blocks，只有 68 个（33%，截断落在正文里或正好在块边界）是真的零警告 —— 而
    「零警告」**不等于「只丢一点」**：那 68 个里有 14 个只解析出 1 条 cue，零警告截断点的
    cue 数是 1..5 全谱，损失最惨的那一档恰好落在一声不响的那一边（完整机制见
    resolve.resolve_subtitle_source 里的注释）。真正的问题是**那条警告指不出真因**：
    skipped_blocks 跟「片源字幕格式略歪」完全同形，而它恰好是报给用户看片源质量的那个数字。

    所以这条路每次重抽，代价只是一次 demux —— 跟转写那条路（原子落盘、失败不留文件，
    所以能安全复用）刻意不同。
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


# --- 位图字幕轨（PGS / VobSub）------------------------------------------------
#
# has_subtitle_stream 只回答「有没有字幕轨」，答不了「是不是文本」，所以位图轨必然走进
# 抽取那条分枝并在 ffmpeg 的跨族转码守卫上失败。不需要真的位图样本：要钉的是「认出那条
# 报错并换成能照着做的话」，fake 掉 extract_subtitle_track 抛一个带那句原话的
# FFmpegError 就够。


def _raise_bitmap_error(video, dest, **kwargs):
    raise resolve.ffmpeg.FFmpegError(
        "ffmpeg 输出尾部...\n"
        "Subtitle encoding currently only possible from text to text or bitmap to bitmap\n"
    )


def test_a_bitmap_subtitle_track_gets_an_actionable_chinese_error(monkeypatch, tmp_path, stub):
    _, state = stub
    state["has_subtitle"] = True
    monkeypatch.setattr(resolve.ffmpeg, "extract_subtitle_track", _raise_bitmap_error)

    with pytest.raises(ValueError) as caught:
        resolve.resolve_subtitle_source(
            None, _video(tmp_path), cache=_cache(tmp_path), asr_config=AsrConfig()
        )

    message = str(caught.value)
    assert "e11.mp4" in message
    assert "位图" in message
    # 两条出路都必须在消息里：位图轨是能 OCR 的（信息还在），所以这一层刻意不自动回落到
    # 听写 —— 那就有责任告诉人手上有哪些选择。
    assert "--srt" in message
    assert "语音转写" in message
    # ValueError 而不是 FFmpegError 也无妨（两者都在 cli.PIPELINE_ERRORS 里），但英文原话
    # 不许再出现在给人看的那行里，否则这个分枝只是在一屏报错上多贴了一句中文。
    assert "bitmap to bitmap" not in message


def test_an_unrelated_ffmpeg_failure_is_not_rewritten(monkeypatch, tmp_path, stub):
    """只认位图那一条。别的 ffmpeg 失败原样抛出去 —— 换掉消息等于把真因抹了。"""
    _, state = stub
    state["has_subtitle"] = True

    def fake_extract(video, dest, **kwargs):
        raise resolve.ffmpeg.FFmpegError("No space left on device")

    monkeypatch.setattr(resolve.ffmpeg, "extract_subtitle_track", fake_extract)

    with pytest.raises(resolve.ffmpeg.FFmpegError, match="No space left on device"):
        resolve.resolve_subtitle_source(
            None, _video(tmp_path), cache=_cache(tmp_path), asr_config=AsrConfig()
        )


def test_a_bitmap_track_does_not_silently_fall_back_to_transcription(monkeypatch, tmp_path, stub):
    """自动回落是个有诱惑力的错误：它把一次几分钟的有损操作藏进一个抽字幕的分枝里。"""
    calls, state = stub
    state["has_subtitle"] = True
    monkeypatch.setattr(resolve.ffmpeg, "extract_subtitle_track", _raise_bitmap_error)

    with pytest.raises(ValueError):
        resolve.resolve_subtitle_source(
            None, _video(tmp_path), cache=_cache(tmp_path), asr_config=AsrConfig()
        )

    assert calls["transcribe"] == []


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


@pytest.mark.parametrize(
    "name",
    ["E11.asr.srt", "E11.srt", "E11.asr.SRT", "dialogue.srt", "E11", "E11.embedded.srt"],
)
def test_the_embedded_dest_never_collides_with_the_cache(tmp_path, name):
    """抽出来那份**绝不能**落在转写缓存那个文件上，对任何 cache 名字都成立。

    只测 `E11.asr.srt` 这个规范名字等于没测：它恰好是
    `cache.name.replace(".asr.srt", ".embedded.srt")` 这种 naive 写法唯一能工作的形状。
    实测这 6 个名字里有 5 个会让 naive 写法**静默原样返回**、于是两条路撞在同一个文件上
    （只有 `E11.asr.srt` 例外）—— 而两条路的处置刻意相反（转写那份按 mtime 复用、抽出来
    那份每次重写），撞在一起意味着「复用」那半边的判据落到一份不该被信任的文件上。
    """
    cache = tmp_path / "srt" / name

    dest = resolve._embedded_dest(cache)

    assert dest != cache
    assert dest.parent == cache.parent
    assert dest.name.endswith(".embedded.srt")


def test_the_canonical_cache_name_yields_a_clean_embedded_name(tmp_path):
    """`E11.asr.srt` → `E11.embedded.srt`，而不是 `E11.asr.embedded.srt`。

    这一条**纯观感**（防撞车靠的是 `Path.stem`，上面那条参数化测试才是守不变量的那个），
    但还是值得钉：这个名字是 `srt/` 目录里人会看到、也可能被 `--srt` 手传回来的文件名，
    悄悄变形会留下两代命名的文件混在一个目录里。上面那条参数化测试**杀不掉**这个改动 ——
    `E11.asr.embedded.srt` 照样不撞车。
    """
    assert resolve._embedded_dest(tmp_path / "E11.asr.srt").name == "E11.embedded.srt"


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


def test_the_transcription_is_announced_before_it_starts(tmp_path, stub):
    """三条路里唯一一条既费时间又有损的。几分钟的静默会让人以为卡死了。

    断言的是「转写**开始**那一刻 stdout 上已经有那句话」，不是「跑完之后 stdout 上有」——
    后者对「把 print 挪到 transcribe 调用之后」这个改动完全不敏感，而那个改动恰好把这句
    告知的全部价值抹掉了。
    """
    calls, _ = stub

    resolve.resolve_subtitle_source(
        None, _video(tmp_path), cache=_cache(tmp_path), asr_config=AsrConfig()
    )

    announced = calls["announced_before"][0]
    assert "e11.mp4" in announced
    # 这一行的增量信息只有「为什么走到了这一步」；耗时提示是 transcribe 自己那句话的活儿。
    assert "没有字幕轨" in announced


def test_the_source_kind_is_only_srt_asr_or_ocr():
    """kind 是下游判「要不要繁转简」「translate 怎么处置」的唯一依据（日语过 OpenCC 会被
    改字），所以它的取值集合必须是封闭的。

    走 get_type_hints 而不是 `__annotations__`：resolve 有 `from __future__ import
    annotations`，直接读 `__annotations__` 拿到的是一个**未求值的 `ForwardRef`**
    （实测 Python 3.14.6：`ForwardRef("Literal['srt', 'asr']")`，`annotationlib.ForwardRef`
    类型，连 `isinstance(x, str)` 都是 False），`== Literal[...]` 永远为假 —— 也就是说
    那样写会得到一条永远红的测试。
    """
    from typing import Literal, get_args, get_type_hints

    field = get_type_hints(resolve.SubtitleSource)["kind"]
    assert field == Literal["srt", "asr", "ocr"]
    assert set(get_args(field)) == {"srt", "asr", "ocr"}


# --- 声明了硬字幕：画面 OCR ----------------------------------------------------


def _resolve_hardsub(tmp_path: Path, video: Path, **overrides):
    kwargs = {
        "cache": _cache(tmp_path),
        "asr_config": AsrConfig(),
        "hardsub": True,
        "ocr_cache": _ocr_cache(tmp_path),
        "ocr_config": OcrConfig(),
        **overrides,
    }
    return resolve.resolve_subtitle_source(None, video, **kwargs)


def test_a_declared_hardsub_is_recognized_instead_of_transcribed(tmp_path, stub):
    calls, _ = stub
    video = _video(tmp_path)

    source = _resolve_hardsub(tmp_path, video)

    assert source == resolve.SubtitleSource(_ocr_cache(tmp_path), "ocr")
    assert calls["recognize"] == [(video, _ocr_cache(tmp_path))]
    assert calls["transcribe"] == []
    assert "認出來的" in source.path.read_text(encoding="utf-8")


def test_a_declared_hardsub_still_prefers_an_embedded_subtitle_track(tmp_path, stub):
    """文本字幕轨是最准的素材：声明了硬字幕也照样先抽它。"""
    calls, state = stub
    state["has_subtitle"] = True

    source = _resolve_hardsub(tmp_path, _video(tmp_path))

    assert source.kind == "srt"
    assert source.path.name.endswith(".embedded.srt")
    assert calls["recognize"] == []


def test_a_handed_srt_still_wins_over_a_declared_hardsub(tmp_path, stub):
    calls, _ = stub
    srt = tmp_path / "hand.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n手传\n", encoding="utf-8")

    source = resolve.resolve_subtitle_source(
        srt,
        _video(tmp_path),
        cache=_cache(tmp_path),
        asr_config=AsrConfig(),
        hardsub=True,
        ocr_cache=_ocr_cache(tmp_path),
    )

    assert source == resolve.SubtitleSource(srt, "srt")
    assert calls["probe"] == []
    assert calls["recognize"] == []


def test_without_a_hardsub_declaration_nothing_changes(tmp_path, stub):
    """没声明就维持原来的行为：哪怕磁盘上恰好躺着一份新鲜的 .ocr.srt 也不看它。"""
    calls, _ = stub
    video = _video(tmp_path)
    leftover = _ocr_cache(tmp_path)
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_text("1\n00:00:01,000 --> 00:00:02,000\n舊的\n", encoding="utf-8")
    _touch_relative_to(leftover, video, +10)

    source = resolve.resolve_subtitle_source(
        None, video, cache=_cache(tmp_path), asr_config=AsrConfig(), ocr_cache=leftover
    )

    assert source == resolve.SubtitleSource(_cache(tmp_path), "asr")
    assert calls["recognize"] == []
    assert calls["transcribe"]


def test_a_fresh_ocr_cache_is_reused_instead_of_recognizing_again(tmp_path, stub):
    """整集识别要几分钟，--force 重跑 ingest 不该重付。"""
    calls, _ = stub
    video = _video(tmp_path)
    cache = _ocr_cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("1\n00:00:01,000 --> 00:00:02,000\n既存\n", encoding="utf-8")
    _touch_relative_to(cache, video, +10)

    source = _resolve_hardsub(tmp_path, video)

    assert source == resolve.SubtitleSource(cache, "ocr")
    assert calls["recognize"] == []


def test_a_hand_edited_ocr_cache_survives_an_ocr_config_change(tmp_path, stub):
    """新鲜度刻意不看 OcrConfig：手改过的 .ocr.srt 不许因为改了一个阈值就被静默冲掉。"""
    calls, _ = stub
    video = _video(tmp_path)
    cache = _ocr_cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("1\n00:00:01,000 --> 00:00:02,000\n手改過的\n", encoding="utf-8")
    _touch_relative_to(cache, video, +10)

    source = _resolve_hardsub(tmp_path, video, ocr_config=OcrConfig(crop_top=0.6, similarity=0.8))

    assert calls["recognize"] == []
    assert "手改過的" in source.path.read_text(encoding="utf-8")


def test_a_stale_ocr_cache_is_recognized_again(tmp_path, stub):
    """换了片源（重新压制、换了个版本）就得重认。"""
    calls, _ = stub
    cache = _ocr_cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("舊的\n", encoding="utf-8")
    video = _video(tmp_path)
    _touch_relative_to(cache, video, -10)

    source = _resolve_hardsub(tmp_path, video)

    assert calls["recognize"]
    assert "認出來的" in source.path.read_text(encoding="utf-8")


def test_an_empty_ocr_cache_is_recognized_again(tmp_path, stub):
    """0 字节的缓存是上一次被打断留下的，不是产物。"""
    calls, _ = stub
    video = _video(tmp_path)
    cache = _ocr_cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("", encoding="utf-8")
    _touch_relative_to(cache, video, +10)

    _resolve_hardsub(tmp_path, video)

    assert calls["recognize"]


def test_an_ocr_failure_does_not_fall_back_to_transcription(monkeypatch, tmp_path, stub):
    """用户明确说了要用画面上那份更好的素材，静默换成听写等于把它丢了。"""
    calls, _ = stub

    def unavailable(video, dest, **kwargs):
        raise resolve.ocr.OCRUnavailableError("跑一次 `uv sync --extra ocr` 再试。")

    monkeypatch.setattr(resolve.ocr, "recognize", unavailable)

    with pytest.raises(resolve.ocr.OCRUnavailableError):
        _resolve_hardsub(tmp_path, _video(tmp_path))

    assert calls["transcribe"] == []


def test_a_declared_hardsub_needs_an_ocr_cache_path(tmp_path, stub):
    with pytest.raises(ValueError, match="ocr_cache"):
        _resolve_hardsub(tmp_path, _video(tmp_path), ocr_cache=None)


def test_the_configured_binaries_and_ocr_config_reach_recognize(tmp_path, stub):
    """跟转写那一路同理：形参叫 `ocr_config`，下层收的是 `ocr=`。转错了会静默退回默认值。"""
    calls, _ = stub
    config = OcrConfig(crop_top=0.65, language="zh-Hans")

    _resolve_hardsub(
        tmp_path,
        _video(tmp_path),
        ocr_config=config,
        ffmpeg_path="/opt/homebrew/bin/ffmpeg",
        ffprobe_path="/opt/homebrew/bin/ffprobe",
    )

    assert calls["kwargs"]["recognize"] == {
        "ocr": config,
        "ffmpeg_path": "/opt/homebrew/bin/ffmpeg",
        "ffprobe_path": "/opt/homebrew/bin/ffprobe",
    }
