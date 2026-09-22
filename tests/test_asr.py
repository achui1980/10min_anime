"""语音转写层。刻意不测 whisper 的转写准确率（那是模型的事，且要几 G 模型 + 真视频），
只测我们自己那几层：段落转 cue 的过滤规则、SRT 渲染、缺依赖时的报错。"""

from pathlib import Path

import pytest

from tenmin.ingest import asr


def test_segments_to_cues_keeps_normal_segments():
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.5, "text": "こんにちは"},
            {"start": 3.0, "end": 4.0, "text": "元気ですか"},
        ]
    )
    assert [(c.idx, c.start, c.end, c.text) for c in cues] == [
        (1, 1.0, 2.5, "こんにちは"),
        (2, 3.0, 4.0, "元気ですか"),
    ]


def test_segments_to_cues_drops_zero_length_tail_hallucinations():
    """实测 whisper 在音频末尾会吐一串 start == end 的空段落，是边界 artifact。

    在这一层就丢掉，不指望下游清洗层接住 —— 让脏数据流进去会污染 ingest 的
    「跳过了几个块」统计，那个数字是给人判断片源质量的。
    """
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.0, "text": "本物"},
            {"start": 119.96, "end": 119.96, "text": ""},
            {"start": 119.96, "end": 119.96, "text": " "},
            {"start": 119.96, "end": 119.96, "text": ""},
        ]
    )
    assert len(cues) == 1
    assert cues[0].text == "本物"


def test_segments_to_cues_drops_reversed_and_zero_length_spans():
    """两条都要带**非空文本**，否则空文本那条规则会替时间那条规则把它们挡下来。

    上面那个尾部幻觉的用例里三条坏段落恰好既零长又空文本，所以它单独锁不住判据是
    `end <= start` 还是 `end < start` —— 把判据放宽成后者它照样全绿。
    """
    cues = asr.segments_to_cues(
        [
            {"start": 5.0, "end": 4.0, "text": "壊れてる"},
            {"start": 7.0, "end": 7.0, "text": "潰れてる"},
        ]
    )
    assert cues == []


def test_segments_to_cues_drops_pronunciation_free_text():
    """纯标点/纯符号的段落没有内容，留着只会在对白轨里占一行。"""
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.0, "text": "……"},
            {"start": 3.0, "end": 4.0, "text": "♪"},
            {"start": 5.0, "end": 6.0, "text": "、。"},
            {"start": 7.0, "end": 8.0, "text": "ありがとう"},
        ]
    )
    assert [c.text for c in cues] == ["ありがとう"]


def test_segments_to_cues_keeps_latin_and_digits():
    """歌名/型号这类内容是有效对白，不能被「没假名就丢」的规则误杀。"""
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.0, "text": "OK"},
            {"start": 3.0, "end": 4.0, "text": "2026"},
        ]
    )
    assert [c.text for c in cues] == ["OK", "2026"]


def test_segments_to_cues_strips_surrounding_whitespace():
    cues = asr.segments_to_cues([{"start": 1.0, "end": 2.0, "text": "  はい  "}])
    assert cues[0].text == "はい"


def test_segments_to_cues_renumbers_after_dropping():
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.0, "text": "一"},
            {"start": 2.0, "end": 2.0, "text": ""},
            {"start": 3.0, "end": 4.0, "text": "二"},
        ]
    )
    assert [c.idx for c in cues] == [1, 2]


def test_render_srt_uses_comma_millisecond_separator():
    """SRT 的毫秒分隔符是逗号。timecode.format_timestamp 产出的是点号版本，
    它的 docstring 明令不要拿它写 SRT，所以这里自己拼。"""
    from tenmin.models import RawCue

    text = asr.render_srt(
        [
            RawCue(idx=1, start=1.5, end=2.25, text="はい"),
            RawCue(idx=2, start=3661.007, end=3662.0, text="いいえ"),
        ]
    )
    assert "00:00:01,500 --> 00:00:02,250" in text
    assert "01:01:01,007 --> 01:01:02,000" in text
    assert text.startswith("1\n")
    assert "はい" in text and "いいえ" in text


def test_render_srt_round_trips_through_the_parser():
    """最硬的不变量：渲染出来的东西必须能被本项目自己的 SRT 解析器吃回去。"""
    from tenmin.ingest.srt_parser import parse_srt
    from tenmin.models import RawCue

    original = [
        RawCue(idx=1, start=1.5, end=2.25, text="はい"),
        RawCue(idx=2, start=10.0, end=12.125, text="そうですね"),
    ]
    reparsed = parse_srt(asr.render_srt(original))
    assert [(c.start, c.end, c.text) for c in reparsed] == [
        (1.5, 2.25, "はい"),
        (10.0, 12.125, "そうですね"),
    ]


def test_render_srt_of_nothing_is_empty():
    assert asr.render_srt([]) == ""


def test_transcribe_without_the_extra_raises_an_actionable_error(tmp_path, monkeypatch):
    """没装 asr extra 时要给出能照着做的一句话，而不是一个裸 ImportError。"""
    import builtins

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_whisper":
            raise ImportError("No module named 'mlx_whisper'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", lambda *a, **k: None)

    from tenmin.config import AsrConfig

    with pytest.raises(asr.ASRUnavailableError) as excinfo:
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())
    assert "uv sync --extra asr" in str(excinfo.value)


def test_asr_unavailable_is_an_asr_error():
    """上层想一把网住两者时只写一个 except 就够（cli 的 PIPELINE_ERRORS 只登记父类）。"""
    assert issubclass(asr.ASRUnavailableError, asr.ASRError)


def test_asr_errors_are_caught_by_the_cli_error_net():
    """逃出 PIPELINE_ERRORS 就意味着用户看到一整页 traceback。"""
    from tenmin.cli import PIPELINE_ERRORS

    assert issubclass(asr.ASRError, PIPELINE_ERRORS)
    assert issubclass(asr.ASRUnavailableError, PIPELINE_ERRORS)


def test_transcribe_writes_an_srt_from_the_model_output(tmp_path, monkeypatch):
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    dest = tmp_path / "sub" / "out.srt"
    seen: dict[str, object] = {}

    def fake_extract(src, wav, **kwargs):
        Path(wav).write_bytes(b"fake wav")

    def fake_load(**kwargs):
        seen.update(kwargs)
        return {
            "segments": [
                {"start": 0.5, "end": 1.5, "text": "おはよう"},
                {"start": 2.0, "end": 2.0, "text": ""},
            ]
        }

    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", fake_extract)
    monkeypatch.setattr(asr, "_run_model", fake_load)

    asr.transcribe(video, dest, asr=AsrConfig(model="m", language="ja"))

    assert dest.is_file()
    body = dest.read_text(encoding="utf-8")
    assert "おはよう" in body
    assert body.count("-->") == 1
    assert seen["path_or_hf_repo"] == "m"
    assert seen["language"] == "ja"


def test_transcribe_forwards_the_configured_ffmpeg_path(tmp_path, monkeypatch):
    """`render.ffmpeg_path` 配成绝对路径的人（PATH 里没有 ffmpeg）全靠这一路转发。

    默认值等于 `ffmpeg` 时转发漏没漏是**看不出来的**，所以必须传一个不一样的值。
    """
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    seen: dict[str, object] = {}

    def fake_extract(src, wav, **kwargs):
        seen.update(kwargs)
        Path(wav).write_bytes(b"x")

    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", fake_extract)
    monkeypatch.setattr(
        asr, "_run_model", lambda **k: {"segments": [{"start": 0.0, "end": 1.0, "text": "あ"}]}
    )

    asr.transcribe(
        video, tmp_path / "out.srt", asr=AsrConfig(), ffmpeg_path="/opt/homebrew/bin/ffmpeg"
    )

    assert seen["ffmpeg"] == "/opt/homebrew/bin/ffmpeg"


def test_transcribe_raises_when_the_model_finds_nothing(tmp_path, monkeypatch):
    """一条 cue 都没有意味着这一集彻底没对白轨，让它静默写个空文件会让后面所有阶段
    在莫名其妙的地方炸。在这里就报错。"""
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        asr.ffmpeg, "extract_audio_track", lambda src, wav, **k: Path(wav).write_bytes(b"x")
    )
    monkeypatch.setattr(asr, "_run_model", lambda **k: {"segments": []})

    with pytest.raises(asr.ASRError):
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())


def test_transcribe_leaves_no_file_behind_when_it_fails(tmp_path, monkeypatch):
    """报错那条路上不许留下任何东西 —— 半截/空的 SRT 会被 resolve 的「文件在就复用」
    判据当成好东西，于是一集彻底没对白的片源静默走完全程。"""
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    dest = tmp_path / "sub" / "out.srt"
    monkeypatch.setattr(
        asr.ffmpeg, "extract_audio_track", lambda src, wav, **k: Path(wav).write_bytes(b"x")
    )
    monkeypatch.setattr(asr, "_run_model", lambda **k: {"segments": []})

    with pytest.raises(asr.ASRError):
        asr.transcribe(video, dest, asr=AsrConfig())
    assert not dest.exists()
    assert list(dest.parent.glob("*")) == []


def test_transcribe_raises_when_segments_is_not_a_list(tmp_path, monkeypatch):
    """mlx-whisper 换了返回形状时要在这里响亮地停，而不是让一个 int 漂进 for 循环。"""
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        asr.ffmpeg, "extract_audio_track", lambda src, wav, **k: Path(wav).write_bytes(b"x")
    )
    monkeypatch.setattr(asr, "_run_model", lambda **k: {"segments": 42})

    with pytest.raises(asr.ASRError, match="segments"):
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())


def test_transcribe_cleans_up_the_temporary_wav(tmp_path, monkeypatch):
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    wavs: list[Path] = []

    def fake_extract(src, wav, **kwargs):
        wavs.append(Path(wav))
        Path(wav).write_bytes(b"x")

    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", fake_extract)
    monkeypatch.setattr(
        asr, "_run_model", lambda **k: {"segments": [{"start": 0.0, "end": 1.0, "text": "あ"}]}
    )

    asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())

    assert wavs
    assert not wavs[0].exists()


def test_transcribe_keeps_the_wav_out_of_the_destination_directory(tmp_path, monkeypatch):
    """那份 wav 是喂模型的入参、不是产物（一集 16k 单声道约 45 MB）。落在 work/ 下
    会让人以为它该被保留，也会在中途失败时留一个没人清的大文件。"""
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    dest = tmp_path / "sub" / "out.srt"
    wavs: list[Path] = []

    def fake_extract(src, wav, **kwargs):
        wavs.append(Path(wav))
        Path(wav).write_bytes(b"x")

    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", fake_extract)
    monkeypatch.setattr(
        asr, "_run_model", lambda **k: {"segments": [{"start": 0.0, "end": 1.0, "text": "あ"}]}
    )

    asr.transcribe(video, dest, asr=AsrConfig())

    assert list(dest.parent.iterdir()) == [dest]
    assert dest.parent not in wavs[0].parents
