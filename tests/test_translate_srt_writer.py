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


@pytest.mark.parametrize("bad_id", [2, 99])
def test_an_id_outside_the_track_is_an_error(bad_id):
    """译文带了一个对白轨里不存在的 id 意味着上游对齐校验漏了 —— 别静默丢掉。

    轨只有 1 行，所以 bad_id=2 恰好是 total + 1 那一格，也就是上界守卫唯一的边界。只测
    99 是不够的：把 `> total` 放宽成 `> total + 1` 时 99 照样被拦住，而 2 会走到
    `track.lines[1]` 抛 IndexError —— IndexError 不在 cli.PIPELINE_ERRORS 里，用户拿到的
    是整页 traceback，正是本文件另一条测试（非有限时间戳）立的那条标准。
    """
    track = _track(_line(1, 1.0, 2.0, "あ"))
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=bad_id, zh="啊")])
    with pytest.raises(ValueError) as excinfo:
        render_zh_srt(track, translated)
    assert str(bad_id) in str(excinfo.value)


def test_a_non_positive_id_is_an_error():
    """0 与近边界的负数同样越界。不报错就会拿 track.lines[-1] 静默配到最后一行的时间。

    轨给 2 行是为了让 id=0 那一格真的能静默命中：更负的 id（|id| >= len(lines)）已经越出
    列表，抛的是 IndexError 而不是本函数的 ValueError，那条路由上界守卫的同一句判断兜住。
    """
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
    """起点相同、终点不同（重叠的两条 cue）时按终点排，短的先出。

    只取 start 当排序键时这两行的先后由「译文轨的回填顺序」决定 —— 那是模型定的，换一次
    调用就可能换一个顺序，交付物的字节也就跟着变。多一个终点键把它钉死。

    刻意**不**拿 split_dual_track 举例：它从一条 cue 拆出的各段沿用同一对 start/end，
    终点键对那种情况一点用都没有（实测同 start 同 end 两行、译文顺序对调，输出顺序跟着
    对调）。那一格由下一条测试的 line.id 键收口。
    """
    track = _track(_line(1, 1.0, 9.0, "長い"), _line(2, 1.0, 2.0, "短い"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="长"), TranslatedLine(id=2, zh="短")]
    )
    body = render_zh_srt(track, translated)
    assert body.index("短") < body.index("长")


def test_lines_sharing_both_ends_are_ordered_by_id():
    """同起点**同终点**时按 line.id 排，也就是按对白轨里的原文顺序。

    这一格是 normalize 的 split_dual_track 造出来的常态：它从一条 cue 拆出的各段沿用同一
    对 start/end，所以 (start, end) 两个键都排不开它们。少了 id 键，稳定排序就把先后交回
    给「译文轨的回填顺序」—— 这里刻意把译文按 [2, 1] 倒着给，模型乱序回填时交付物的字节
    不该跟着变。
    """
    track = _track(_line(1, 1.0, 2.0, "A段"), _line(1, 1.0, 2.0, "B段"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=2, zh="乙"), TranslatedLine(id=1, zh="甲")]
    )
    body = render_zh_srt(track, translated)
    assert body.index("甲") < body.index("乙")


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

    ASR 出来的时间戳是模型算的，理论上能给出 inf。OverflowError 不在
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
