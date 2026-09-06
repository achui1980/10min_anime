import pytest

from tenmin.models import AudioDirection, Beat, Hold
from tenmin.render.chunks import (
    assign_holds,
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
