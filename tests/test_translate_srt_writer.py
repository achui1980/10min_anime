"""中文字幕交付物的拼装。纯函数。"""

import math

import pytest

from tenmin.models import DialogueLine, DialogueTrack, TranslatedLine, TranslatedTrack
from tenmin.translate.srt_writer import render_zh_srt


def _line(idx: int, start: float, end: float, text: str, kind: str = "dialogue"):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind=kind)


def _track(*lines: DialogueLine, duration: float = 100.0) -> DialogueTrack:
    return DialogueTrack(episode=11, source="asr", duration=duration, lines=list(lines))


def test_renders_timestamps_from_the_dialogue_track():
    """译文里只有 id 和中文，时间戳必须回到对白轨上取。"""
    track = _track(_line(1, 1.5, 2.25, "はい"), _line(2, 10.0, 12.125, "そうですね"))
    translated = TranslatedTrack(
        episode=11,
        lines=[TranslatedLine(id=1, zh="是的"), TranslatedLine(id=2, zh="说得对")],
    )

    body = render_zh_srt(track, translated)

    assert "00:00:01,500 --> 00:00:02,250" in body
    assert "是的" in body
    assert "00:00:10,000 --> 00:00:12,125" in body
    assert "说得对" in body


def test_the_millisecond_separator_is_a_comma_not_a_dot():
    """SRT 用逗号。共用的 timecode.format_timestamp 产出点号版本，只换分隔符不能换错。

    单独钉一条是因为「时间戳里出现点号」在上面那条测试里表现为「整串对不上」，看不出
    坏在分隔符上；而点号版本恰好能被本项目自己的 srt_parser 吃回去（它的正则收 `[,.]`），
    于是 round-trip 测试也拦不住 —— 只有播放器会拒。
    """
    track = _track(_line(1, 1.5, 2.25, "はい"))
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=1, zh="是的")])
    timestamp_line = render_zh_srt(track, translated).splitlines()[1]
    assert timestamp_line == "00:00:01,500 --> 00:00:02,250"
    assert "." not in timestamp_line


def test_blocks_are_numbered_from_one_consecutively():
    track = _track(_line(1, 1.0, 2.0, "あ"), _line(2, 3.0, 4.0, "い"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="啊"), TranslatedLine(id=2, zh="咦")]
    )
    body = render_zh_srt(track, translated)
    assert body.startswith("1\n")
    assert "\n2\n" in body


def test_untranslated_lines_are_skipped():
    """片头片尾那些行压根不会被送去翻译，字幕里也就不该出现空块。"""
    track = _track(
        _line(1, 1.0, 2.0, "作曲：某人", kind="credits"),
        _line(2, 3.0, 4.0, "本編のセリフ"),
    )
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=2, zh="正片台词")])

    body = render_zh_srt(track, translated)

    assert body.count("-->") == 1
    assert "正片台词" in body
    assert "作曲" not in body


def test_an_id_outside_the_track_is_an_error():
    """译文带了一个对白轨里不存在的 id 意味着上游对齐校验漏了 —— 别静默丢掉。"""
    track = _track(_line(1, 1.0, 2.0, "あ"))
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=99, zh="啊")])
    with pytest.raises(ValueError) as excinfo:
        render_zh_srt(track, translated)
    assert "99" in str(excinfo.value)


def test_a_non_positive_id_is_an_error():
    """0 与负数同样越界。它们不报错就会拿 track.lines[-1] 静默配到最后一行的时间。"""
    track = _track(_line(1, 1.0, 2.0, "あ"), _line(2, 3.0, 4.0, "い"))
    with pytest.raises(ValueError) as excinfo:
        render_zh_srt(track, TranslatedTrack(episode=11, lines=[TranslatedLine(id=0, zh="啊")]))
    assert "0" in str(excinfo.value)


def test_output_is_ordered_by_time_not_by_translation_order():
    track = _track(_line(1, 5.0, 6.0, "後"), _line(2, 1.0, 2.0, "先"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="后"), TranslatedLine(id=2, zh="先")]
    )
    body = render_zh_srt(track, translated)
    assert body.index("先") < body.index("后")


def test_lines_sharing_a_start_are_ordered_by_end():
    """同起点时按终点排。

    对白轨里同起点是常态（normalize 的 split_dual_track 会把一条 cue 拆成多段），排序键
    只取 start 时这些行的先后由「译文轨的回填顺序」决定 —— 那是模型定的，换一次调用就
    可能换一个顺序，交付物的字节也就跟着变。多一个终点键把它钉死。
    """
    track = _track(_line(1, 1.0, 9.0, "長い"), _line(2, 1.0, 2.0, "短い"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="长"), TranslatedLine(id=2, zh="短")]
    )
    body = render_zh_srt(track, translated)
    assert body.index("短") < body.index("长")


def test_multiline_translation_is_kept_as_is():
    track = _track(_line(1, 1.0, 2.0, "あ"))
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=1, zh="第一行\n第二行")])
    body = render_zh_srt(track, translated)
    assert "第一行\n第二行" in body


def test_blank_translations_are_skipped():
    track = _track(_line(1, 1.0, 2.0, "あ"), _line(2, 3.0, 4.0, "い"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="   "), TranslatedLine(id=2, zh="咦")]
    )
    body = render_zh_srt(track, translated)
    assert body.count("-->") == 1
    assert "咦" in body


def test_an_empty_translation_track_gives_an_empty_file():
    track = _track(_line(1, 1.0, 2.0, "あ"))
    assert render_zh_srt(track, TranslatedTrack(episode=11, lines=[])) == ""


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_a_non_finite_timestamp_raises_value_error(bad):
    """坏时间戳必须走 ValueError，不能是 OverflowError。

    ASR 出来的 duration 是模型算的，理论上能给出 inf。OverflowError 不在
    cli.PIPELINE_ERRORS 里，会把整页 traceback 糊到用户脸上 —— 这条同时钉住「时间戳
    格式化不要另抄一份、要复用带 nan/inf 守卫的那份」这个决定。
    """
    track = _track(_line(1, bad, 2.0, "あ"), duration=100.0)
    with pytest.raises(ValueError):
        render_zh_srt(track, TranslatedTrack(episode=11, lines=[TranslatedLine(id=1, zh="啊")]))


def test_round_trips_through_the_srt_parser():
    """最硬的不变量：产出的东西得能被本项目自己的解析器吃回去。"""
    from tenmin.ingest.srt_parser import parse_srt

    track = _track(_line(1, 1.5, 2.25, "はい"), _line(2, 10.0, 12.0, "いいえ"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="是"), TranslatedLine(id=2, zh="不是")]
    )
    cues = parse_srt(render_zh_srt(track, translated))
    assert [(c.start, c.end, c.text) for c in cues] == [(1.5, 2.25, "是"), (10.0, 12.0, "不是")]
