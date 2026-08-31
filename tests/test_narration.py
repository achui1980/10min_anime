import re

from tenmin.docgen.narration import render_narration
from tenmin.models import Beat, Script


def beat(bid, text, role="act"):
    return Beat(id=bid, label=f"节点{bid}", role=role, narration=text)


def script(beats):
    return Script(show="才女的侍从", episodes=[2], beats=list(beats))


def test_single_beat():
    assert render_narration(script([beat("b1", "第一段", role="hook")])) == "第一段\n"


def test_beats_separated_by_blank_line():
    out = render_narration(script([beat("b1", "甲", role="hook"), beat("b2", "乙")]))
    assert out == "甲\n\n乙\n"


def test_separator_count_matches_beat_count():
    beats = [beat(f"b{i}", f"段{i}") for i in range(5)]
    assert render_narration(script(beats)).count("\n\n") == 4


def test_strips_per_beat_whitespace():
    out = render_narration(script([beat("b1", "  甲  \n", role="hook"), beat("b2", "\n乙")]))
    assert out == "甲\n\n乙\n"


def test_char_count_equals_sum_of_beats():
    beats = [beat("b1", "啊" * 30, role="hook"), beat("b2", "啊" * 70)]
    out = render_narration(script(beats))
    assert len(re.sub(r"\s+", "", out)) == 100


def test_no_markers_leak_into_output():
    beats = [beat("b1", "甲", role="hook"), beat("b2", "乙")]
    out = render_narration(script(beats))
    for marker in ("★", "|", "#", "节点", "留白", "<br>"):
        assert marker not in out


def test_empty_beats_produce_empty_file():
    assert render_narration(script([])) == ""


def test_beat_with_blank_narration_is_skipped():
    beats = [beat("b1", "甲", role="hook"), beat("b2", "   "), beat("b3", "乙")]
    assert render_narration(script(beats)) == "甲\n\n乙\n"
