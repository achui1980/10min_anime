"""语音转写层。刻意不测 whisper 的转写准确率（那是模型的事，且要几 G 模型 + 真视频），
只测我们自己那几层：段落转 cue 的过滤规则、SRT 渲染、缺依赖时的报错。"""

from pathlib import Path
from types import MappingProxyType

import pytest

from tenmin.ingest import asr


def test_segments_to_cues_keeps_normal_segments():
    """第二条刻意是 `MappingProxyType` 而不是 dict：元素级守卫声明的是 `Mapping`，
    而把它收窄成 `isinstance(segment, dict)` 在只喂 dict 的测试集上**全绿** —— 然后一个
    返回不可变映射（或任何 Mapping 实现）的模型版本会被判成「形状变了」而整批报错。
    """
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.5, "text": "こんにちは"},
            MappingProxyType({"start": 3.0, "end": 4.0, "text": "元気ですか"}),
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


def test_segments_to_cues_drops_non_finite_timestamps():
    """nan 必须在**这一层**挡住：`nan <= nan` 是 `False`，所以它溜过零长度那条判据，
    然后在 render_srt 里变成一个 ValueError —— 报错点离真因隔了两层。"""
    nan = float("nan")
    cues = asr.segments_to_cues(
        [
            {"start": nan, "end": nan, "text": "幻"},
            {"start": 1.0, "end": nan, "text": "幻"},
            # start 那一半必须单独有一条坏数据，否则 `isfinite(start) and isfinite(end)`
            # 收窄成只判 end 也全绿：上面三条的 end 侧都已经非有限。而这一条在只判 end 的
            # 版本下会被**留下来** —— end=2.0 是有限的，而兜底那句 `end <= start` 也救不了
            # （`2.0 <= nan` 是 False）。
            {"start": nan, "end": 2.0, "text": "幻"},
            {"start": float("-inf"), "end": float("inf"), "text": "幻"},
            {"start": 1.0, "end": 2.0, "text": "本物"},
        ]
    )
    assert [c.text for c in cues] == ["本物"]


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


def test_segments_to_cues_raises_when_a_segment_is_not_a_mapping():
    """整个形状守卫的存在理由就是「模型换了返回形状要响亮」，所以这里 raise 而不是
    continue —— 静默跳过等于把守卫拆了。没有它，一个 `[1, 2]` 会漏成裸 AttributeError，
    而 AttributeError 不在 cli.PIPELINE_ERRORS 里。"""
    with pytest.raises(asr.ASRError, match="segment"):
        asr.segments_to_cues([{"start": 1.0, "end": 2.0, "text": "本物"}, 42])


def test_segments_to_cues_accepts_a_generator():
    """声明的入参契约是 Iterable，收窄成 Sequence 会把生成器挡在外面（mlx-whisper 的
    某些版本就是流式吐 segment 的）。"""
    cues = asr.segments_to_cues({"start": 1.0, "end": 2.0, "text": "本物"} for _ in range(2))
    assert [c.idx for c in cues] == [1, 2]


def test_format_timestamp_clamps_negative_seconds():
    """负数不能原样拼出去。`-1:59:59,000` 看着像坏数据，但 srt_parser 的 `-->` 行正则用的
    是 search 不是 match，会在里面找到 `1:59:59,000` 当成 1 小时 59 分 —— 静默错时间比
    报错坏得多。"""
    assert asr._format_timestamp(-1.0) == "00:00:00,000"


def test_format_timestamp_rounds_millis_rather_than_truncating():
    """`2.4569` 是能区分 round 与截断的最小例子（截断得 ,456）。其余用例里的值都是
    浮点精确的千分数，对这个差别完全不敏感。"""
    assert asr._format_timestamp(2.4569) == "00:00:02,457"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_format_timestamp_rejects_non_finite_with_a_readable_message(value):
    """必须是 ValueError（在 cli.PIPELINE_ERRORS 里）且带上那个坏数据。

    手写一份 `int(round(...))` 的话 nan 得到 CPython 的「cannot convert float NaN to
    integer」，inf 更糟：OverflowError **不在** PIPELINE_ERRORS 里，整页 traceback。
    """
    from tenmin.cli import PIPELINE_ERRORS

    with pytest.raises(ValueError, match="秒数必须是有限数") as excinfo:
        asr._format_timestamp(value)
    assert isinstance(excinfo.value, PIPELINE_ERRORS)


def test_format_timestamp_only_swaps_the_one_separator():
    """钉住「只把点换成逗号」这个做法的前提：共用格式化器的输出里只有一个点。

    它今天成立（`HH:MM:SS.mmm`），但那是隐式耦合 —— 哪天 format_timestamp 改成带别的
    点（比如小时位溢出写成 `1.02:03:04.005`），replace 会静默把它一起换掉。
    """
    from tenmin.timecode import format_timestamp

    assert format_timestamp(1.5).count(".") == 1


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


def test_render_srt_renumbers_instead_of_reusing_cue_idx():
    """序号按列表位置重数。纯外观（srt_parser 只把它存进不参与定位的 src_idx），但
    docstring 明说了这件事，所以钉一下 —— 未经验证的 docstring 声明跟没写一样。"""
    from tenmin.models import RawCue

    text = asr.render_srt(
        [
            RawCue(idx=7, start=1.0, end=2.0, text="はい"),
            RawCue(idx=9, start=3.0, end=4.0, text="いいえ"),
        ]
    )
    assert [line for line in text.split("\n") if line.isdigit()] == ["1", "2"]


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


@pytest.mark.parametrize(
    "segments",
    [42, "abc", b"abc", {"a": "b"}, None],
    ids=["int", "str", "bytes", "dict", "none"],
)
def test_transcribe_raises_when_segments_is_not_a_list(tmp_path, monkeypatch, segments):
    """容器本身的形状不对。

    光判 `isinstance(x, Iterable)` 是不够的：`str`/`bytes`/`dict` 全是 Iterable 直接
    放行。它们**不会**漏成裸 AttributeError（逐元素那道守卫会接住），但接住之后报的是
    「有个 segment 不是 dict: <class 'str'>」—— 对着一个 `segments="abc"` 或
    `segments={"a": "b"}` 说「列表里有个元素不对」是在指错方向。所以这里断言的是
    **容器级**那条消息，不是随便一个 ASRError。
    """
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        asr.ffmpeg, "extract_audio_track", lambda src, wav, **k: Path(wav).write_bytes(b"x")
    )
    monkeypatch.setattr(asr, "_run_model", lambda **k: {"segments": segments})

    with pytest.raises(asr.ASRError, match="segments 不是个列表"):
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())


def test_transcribe_raises_when_a_segment_is_not_a_mapping(tmp_path, monkeypatch):
    """容器没问题，是**元素**的类型变了。没有这道守卫就是裸 AttributeError
    （`'int' object has no attribute 'get'`），而它不在 cli.PIPELINE_ERRORS 里。"""
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        asr.ffmpeg, "extract_audio_track", lambda src, wav, **k: Path(wav).write_bytes(b"x")
    )
    monkeypatch.setattr(asr, "_run_model", lambda **k: {"segments": [1, 2]})

    with pytest.raises(asr.ASRError, match="segment 不是 dict"):
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())


def test_transcribe_accepts_a_generator_of_segments(tmp_path, monkeypatch):
    """守卫不许把 Iterable 契约偷偷收窄成 Sequence。

    必须走 transcribe 而不是直接调 segments_to_cues —— 那道容器级守卫住在 transcribe 里，
    只测 segments_to_cues 的话把它改成 `isinstance(x, Sequence)` 照样全绿。
    """
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    dest = tmp_path / "out.srt"
    monkeypatch.setattr(
        asr.ffmpeg, "extract_audio_track", lambda src, wav, **k: Path(wav).write_bytes(b"x")
    )
    monkeypatch.setattr(
        asr,
        "_run_model",
        lambda **k: {"segments": ({"start": 0.0, "end": 1.0, "text": "あ"} for _ in range(1))},
    )

    asr.transcribe(video, dest, asr=AsrConfig())
    assert "あ" in dest.read_text(encoding="utf-8")


def test_transcribe_tells_a_missing_segments_key_apart_from_an_empty_transcription(
    tmp_path, monkeypatch
):
    """「返回里压根没有 segments 这个键」是形状变化，跟「转写出来是空的」处置不同：
    前者要去看 mlx-whisper 的 API，后者要去看片源有没有人声轨。原来 `result.get(...) or []`
    把两者并成同一句话，等于把形状变化伪装成片源问题。"""
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        asr.ffmpeg, "extract_audio_track", lambda src, wav, **k: Path(wav).write_bytes(b"x")
    )
    monkeypatch.setattr(asr, "_run_model", lambda **k: {"text": "おはよう"})

    with pytest.raises(asr.ASRError) as excinfo:
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())
    assert "segments" in str(excinfo.value)
    assert "没得到任何对白" not in str(excinfo.value)


def test_a_missing_segments_key_reports_cleanly_even_when_the_result_is_a_list(
    tmp_path, monkeypatch
):
    """这条守卫报错时**自己不许炸**。

    `"segments" not in result` 对 list 也成立（成员判断），所以一个直接返回 segment 列表
    的模型版本会走进这一支 —— 而裸 `sorted(result)` 在那里抛
    `TypeError: '<' not supported between instances of 'dict' and 'dict'`（实测），
    TypeError 不在 cli.PIPELINE_ERRORS 里，于是本该是一行红字的形状变化变成一整页
    traceback，报错语句比它要报的那件事更难查。

    刻意用 dict 列表而不是 `[1, 2]`：后者排得动，钉不住这个洞。
    """
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        asr.ffmpeg, "extract_audio_track", lambda src, wav, **k: Path(wav).write_bytes(b"x")
    )
    monkeypatch.setattr(asr, "_run_model", lambda **k: [{"start": 0.0}, {"end": 1.0}])

    with pytest.raises(asr.ASRError, match="没有 segments 这个键"):
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())


def test_transcribe_feeds_the_model_the_extracted_wav(tmp_path, monkeypatch):
    """模型吃的必须是抽出来的 wav、而且那一刻它还在。

    两件事都只能在 fake 里当场查：转写完 wav 就被删了，事后再看什么都看不到。
    直接喂 video 在真实运行里**不会报错**（mlx-whisper 自己能解 mp4），只是白丢掉
    extract_audio_track 存在的全部理由（16k 单声道、不依赖模型库自己找 ffmpeg）。

    顺手把 `word_timestamps` 一起钉住：它是唯一一个会让一集多花几十秒的开关，而它
    改的恰好是我们唯一消费的 segment 边界。
    """
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    checked: dict[str, object] = {}

    def fake_extract(src, wav, **kwargs):
        Path(wav).write_bytes(b"fake wav")

    def fake_model(**kwargs):
        audio = Path(str(kwargs["audio"]))
        checked["suffix"] = audio.suffix
        checked["existed"] = audio.is_file()
        checked["is_the_video"] = audio == video
        checked["word_timestamps"] = kwargs.get("word_timestamps")
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "あ"}]}

    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", fake_extract)
    monkeypatch.setattr(asr, "_run_model", fake_model)

    asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())

    assert checked == {
        "suffix": ".wav",
        "existed": True,
        "is_the_video": False,
        "word_timestamps": True,
    }


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


def test_transcribe_cleans_up_the_temporary_wav_when_the_model_blows_up(tmp_path, monkeypatch):
    """失败路径也得清 —— 那份 wav 约 46 MB，而「转写中途被打断」正是最常见的走法。"""
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    wavs: list[Path] = []

    def fake_extract(src, wav, **kwargs):
        wavs.append(Path(wav))
        Path(wav).write_bytes(b"x")

    def boom(**kwargs):
        raise asr.ASRError("模型炸了")

    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", fake_extract)
    monkeypatch.setattr(asr, "_run_model", boom)

    with pytest.raises(asr.ASRError):
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())

    assert wavs
    assert not wavs[0].exists()
    assert not wavs[0].parent.exists()


def test_transcribe_keeps_the_wav_out_of_the_destination_directory(tmp_path, monkeypatch):
    """那份 wav 是喂模型的入参、不是产物（一集 16k 单声道约 46 MB）。落在 work/ 下
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
