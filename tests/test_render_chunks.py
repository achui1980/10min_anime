import pytest

from tenmin.models import AudioDirection, Beat, Hold
from tenmin.render.chunks import (
    assign_holds,
    is_pronounceable,
    plan_chunks,
    sentence_offsets,
    split_sentences,
)


def test_split_sentences_keeps_punctuation():
    assert split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]


def test_split_sentences_keeps_tail_without_punctuation():
    assert split_sentences("第一句。没有句号的尾巴") == ["第一句。", "没有句号的尾巴"]


def test_split_sentences_merges_repeated_punctuation():
    assert split_sentences("真的吗？！接着说。") == ["真的吗？！", "接着说。"]


def test_split_sentences_ignores_blank_input():
    assert split_sentences("   \n  ") == []


# --- 句末闭合符（P2-E A1）-------------------------------------------------
#
# 生产实证：work/saijo 的 E05 旁白用 ASCII `'` 当引号，`。` 之后的收引号被甩成了
# 一个独立片段，最后变成两条整条内容就是 `'` 的字幕 cue（0.19s / 0.18s）。


def test_split_sentences_absorbs_a_trailing_closing_quote():
    """`。'` 收尾时那个 `'` 必须留在前一句里，不能自成一句。"""
    assert split_sentences("维护家族名誉就是她应尽的义务。'") == [
        "维护家族名誉就是她应尽的义务。'"
    ]


def test_split_sentences_keeps_an_opening_quote_with_its_own_sentence():
    """同一个 `'` 既当开引号又当收引号，靠奇偶区分：开引号必须留给它引的那句。

    朴素的「终止符之后一律吸收闭合符」在这句上是错的：它会把 `跳。` 后面那个**开**
    引号粘到前一句尾巴上，引号跟它引的话被拆开（实测这还会改掉 E05 两个 chunk 的
    文本、让已经合成好的 mp3 失效）。
    """
    assert split_sentences("雏子从二楼往下跳。'来接住我。'下一秒她笑了。") == [
        "雏子从二楼往下跳。",
        "'来接住我。'",
        "下一秒她笑了。",
    ]


def test_split_sentences_absorbs_unambiguous_closers_regardless_of_parity():
    """`」』”）` 这些只可能收尾，不需要数奇偶。"""
    assert split_sentences("他喊：「快跑。」然后就跑了。") == [
        "他喊：「快跑。」",
        "然后就跑了。",
    ]
    assert split_sentences("（他没说话。）现场很安静。") == [
        "（他没说话。）",
        "现场很安静。",
    ]


def test_split_sentences_merges_an_unpronounceable_fragment_into_the_previous():
    """引号数量不成对时奇偶会判错，兜底那一层保证碎片不会自成一句。

    这里 `'` 出现 3 次，第三个按奇偶被当成开引号 → 先被切成独立片段，
    再由「没有任何可发音字符的句子折进前一句」这条兜底收掉。切完每一句都有内容可读，
    而且一个字都没丢。
    """
    text = "甲说：'开始。'结束。'"
    sentences = split_sentences(text)
    assert sentences == ["甲说：'开始。'", "结束。'"]
    assert all(is_pronounceable(s) for s in sentences)
    assert "".join(sentences) == text


def test_split_sentences_merges_a_leading_unpronounceable_fragment_into_the_next():
    """碎片出现在最前面时没有「前一句」可挂，折到下一句头上。"""
    assert split_sentences("——。第一句。") == ["——。第一句。"]


def test_split_sentences_never_loses_a_character():
    """切句是纯重新分组：拼回去必须跟去掉首尾空白的原文逐字节相同。"""
    text = "雏子从二楼往下跳。'来接住我。'下一秒她笑着宣布：'我有个提议'——'我当人质。'"
    assert "".join(split_sentences(text)) == text


def test_is_pronounceable_needs_a_letter_or_digit():
    assert is_pronounceable("第一句。")
    assert is_pronounceable("OK")
    assert not is_pronounceable("'")
    assert not is_pronounceable("——。")


def test_sentence_offsets_accumulate_estimated_duration():
    # 每句 4 个字，4.5 字/秒 → 每句 0.888… 秒，offsets 是「该句结束时刻」
    offsets = sentence_offsets(["第一句。", "第二句。", "第三句。"])
    assert offsets == pytest.approx([0.8889, 1.7778, 2.6667], abs=1e-4)


def test_assign_holds_picks_nearest_boundary():
    sentences = ["第一句。", "第二句。", "第三句。"]
    holds = [Hold(at=2.0, duration=2.0, quote="金句")]
    # offsets 是 [0.889, 1.778, 2.667]；2.0 离 1.778 最近 → 索引 1
    assert assign_holds(sentences, holds) == {1: 2.0}


def test_assign_holds_sums_holds_on_same_boundary():
    sentences = ["第一句。", "第二句。"]
    holds = [Hold(at=1.7, duration=2.0, quote="甲"), Hold(at=1.9, duration=1.0, quote="乙")]
    assert assign_holds(sentences, holds) == {1: 3.0}


def test_assign_holds_on_empty_sentences_returns_empty():
    assert assign_holds([], [Hold(at=1.0, duration=1.0, quote="金句")]) == {}


def test_plan_chunks_merges_sentences_between_holds():
    beat = Beat(
        id="b1",
        label="Hook",
        role="hook",
        narration="第一句。第二句。第三句。",
        audio=AudioDirection(holds=[Hold(at=2.0, duration=2.0, quote="金句")]),
    )
    assert plan_chunks(beat) == [("第一句。第二句。", 2.0), ("第三句。", 0.0)]


def test_plan_chunks_without_holds_is_one_chunk():
    beat = Beat(id="b1", label="Hook", role="hook", narration="第一句。第二句。")
    assert plan_chunks(beat) == [("第一句。第二句。", 0.0)]


def test_plan_chunks_on_empty_narration_returns_empty():
    beat = Beat(id="b1", label="Hook", role="hook", narration="")
    assert plan_chunks(beat) == []
