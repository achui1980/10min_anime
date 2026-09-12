import pytest

from tenmin.intervals import (
    SilentGap,
    group_adjacent,
    is_spoken,
    merge_intervals,
    silent_gaps,
    spoken_lines,
    subtract,
)
from tenmin.models import SPEECH_KINDS, DialogueLine


def dline(idx, start, end, text="台词", kind="dialogue"):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind=kind)


# --- SPEECH_KINDS ---


def test_speech_kinds_is_a_frozenset_of_two_kinds():
    assert SPEECH_KINDS == frozenset({"dialogue", "monologue"})
    assert isinstance(SPEECH_KINDS, frozenset)


# --- is_spoken / spoken_lines ---


@pytest.mark.parametrize("kind", ["dialogue", "monologue"])
def test_is_spoken_accepts_speech_kinds(kind):
    assert is_spoken(dline(1, 0.0, 1.0, kind=kind)) is True


@pytest.mark.parametrize("kind", ["screen_text", "credits", "noise"])
def test_is_spoken_rejects_non_speech_kinds(kind):
    assert is_spoken(dline(1, 0.0, 1.0, kind=kind)) is False


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_is_spoken_rejects_blank_text(text):
    assert is_spoken(dline(1, 0.0, 1.0, text=text)) is False


def test_is_spoken_rejects_zero_duration():
    """srt_parser 会把 end < start 的坏 cue 夹成零时长。这类行不算有人说话。"""
    assert is_spoken(dline(1, 12.0, 12.0)) is False


def test_is_spoken_rejects_negative_duration():
    assert is_spoken(dline(1, 12.0, 10.0)) is False


def test_spoken_lines_empty_input():
    assert spoken_lines([]) == []


def test_spoken_lines_filters_and_sorts_by_start():
    lines = [
        dline(3, 20.0, 22.0),
        dline(1, 0.0, 2.0),
        dline(2, 10.0, 10.0),  # 零时长
        dline(4, 5.0, 6.0, kind="credits"),
        dline(5, 7.0, 8.0, text="  "),
        dline(6, 9.0, 11.0, kind="monologue"),
    ]
    assert [ln.idx for ln in spoken_lines(lines)] == [1, 6, 3]


def test_spoken_lines_does_not_mutate_input_order():
    lines = [dline(2, 10.0, 12.0), dline(1, 0.0, 2.0)]
    spoken_lines(lines)
    assert [ln.idx for ln in lines] == [2, 1]


# --- silent_gaps ---


def test_silent_gaps_empty_input():
    assert silent_gaps([], duration=100.0) == []


def test_silent_gaps_single_line_without_duration_has_no_gap():
    assert silent_gaps([dline(1, 0.0, 2.0)]) == []


def test_silent_gaps_ignores_gap_before_first_line():
    """首行之前是黑场/OP 前奏，不是「静默间隙」。"""
    gaps = silent_gaps([dline(1, 50.0, 52.0)], duration=52.0)
    assert gaps == []


def test_silent_gaps_between_two_lines():
    gaps = silent_gaps([dline(1, 0.0, 2.0), dline(2, 12.0, 14.0)])
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.start == pytest.approx(2.0)
    assert gap.end == pytest.approx(12.0)
    assert gap.duration == pytest.approx(10.0)
    assert gap.before.idx == 1
    assert gap.after is not None
    assert gap.after.idx == 2


def test_silent_gaps_trailing_gap_has_no_after_line():
    gaps = silent_gaps([dline(1, 0.0, 2.0)], duration=12.0)
    assert len(gaps) == 1
    assert gaps[0] == SilentGap(start=2.0, end=12.0, before=gaps[0].before, after=None)
    assert gaps[0].before.idx == 1
    assert gaps[0].after is None


def test_silent_gaps_no_trailing_gap_when_duration_not_beyond_cursor():
    assert silent_gaps([dline(1, 0.0, 12.0)], duration=12.0) == []
    assert silent_gaps([dline(1, 0.0, 12.0)], duration=8.0) == []


def test_silent_gaps_overlapping_lines_never_produce_negative_gaps():
    lines = [dline(1, 0.0, 20.0), dline(2, 5.0, 8.0), dline(3, 30.0, 32.0)]
    gaps = silent_gaps(lines)
    assert [(g.start, g.end) for g in gaps] == [(20.0, 30.0)]


def test_silent_gaps_before_line_is_the_cursor_owner_not_the_previous_line():
    """游标由最长的那一行持有，anchor 必须指向它，而不是紧邻的短行。"""
    lines = [dline(1, 0.0, 20.0), dline(2, 5.0, 8.0), dline(3, 30.0, 32.0)]
    gaps = silent_gaps(lines)
    assert gaps[0].before.idx == 1


def test_silent_gaps_non_speech_lines_do_not_break_silence():
    lines = [
        dline(1, 0.0, 2.0),
        dline(2, 5.0, 6.0, kind="credits"),
        dline(3, 6.5, 7.0, kind="noise"),
        dline(4, 8.0, 8.5, kind="screen_text"),
        dline(5, 12.0, 14.0),
    ]
    gaps = silent_gaps(lines)
    assert [(g.start, g.end) for g in gaps] == [(2.0, 12.0)]


def test_silent_gaps_zero_duration_line_does_not_split_a_gap():
    """回归：零时长坏 cue 曾经推进游标，把一段长静默切成两段短的（各自被阈值丢掉）。"""
    lines = [dline(1, 0.0, 2.0), dline(2, 7.0, 7.0), dline(3, 12.0, 14.0)]
    gaps = silent_gaps(lines)
    assert [(g.start, g.end) for g in gaps] == [(2.0, 12.0)]


def test_silent_gaps_unsorted_input_is_handled():
    lines = [dline(3, 30.0, 32.0), dline(1, 0.0, 2.0), dline(2, 12.0, 14.0)]
    gaps = silent_gaps(lines)
    assert [(g.start, g.end) for g in gaps] == [(2.0, 12.0), (14.0, 30.0)]


# --- group_adjacent ---


def test_group_adjacent_empty_input():
    assert group_adjacent([], bounds=lambda iv: iv, max_gap=1.0) == []


def test_group_adjacent_single_item():
    assert group_adjacent([(0.0, 1.0)], bounds=lambda iv: iv, max_gap=1.0) == [[(0.0, 1.0)]]


def test_group_adjacent_splits_on_gap_beyond_threshold():
    items = [(0.0, 1.0), (2.0, 3.0), (10.0, 11.0)]
    groups = group_adjacent(items, bounds=lambda iv: iv, max_gap=1.0)
    assert groups == [[(0.0, 1.0), (2.0, 3.0)], [(10.0, 11.0)]]


def test_group_adjacent_uses_running_max_end_not_last_end():
    """被包住的短区间不能让组的右界回退，否则下一个区间会被错判成新组。"""
    items = [(0.0, 20.0), (1.0, 2.0), (21.0, 22.0)]
    groups = group_adjacent(items, bounds=lambda iv: iv, max_gap=2.0)
    assert groups == [[(0.0, 20.0), (1.0, 2.0), (21.0, 22.0)]]


def test_group_adjacent_sorts_by_start_then_end():
    items = [(5.0, 6.0), (0.0, 1.0), (0.0, 0.5)]
    groups = group_adjacent(items, bounds=lambda iv: iv, max_gap=0.0)
    assert groups == [[(0.0, 0.5), (0.0, 1.0)], [(5.0, 6.0)]]


def test_group_adjacent_works_on_arbitrary_objects():
    lines = [dline(1, 0.0, 1.0), dline(2, 100.0, 101.0)]
    groups = group_adjacent(lines, bounds=lambda ln: (ln.start, ln.end), max_gap=5.0)
    assert [[ln.idx for ln in g] for g in groups] == [[1], [2]]


# --- merge_intervals ---


def test_merge_intervals_empty_input():
    assert merge_intervals([]) == []


def test_merge_intervals_single_interval():
    assert merge_intervals([(1.0, 2.0)]) == [(1.0, 2.0)]


def test_merge_intervals_merges_touching_intervals_by_default():
    assert merge_intervals([(0.0, 1.0), (1.0, 2.0)]) == [(0.0, 2.0)]


def test_merge_intervals_keeps_disjoint_intervals_apart_by_default():
    assert merge_intervals([(0.0, 1.0), (1.5, 2.0)]) == [(0.0, 1.0), (1.5, 2.0)]


def test_merge_intervals_honours_max_gap():
    assert merge_intervals([(0.0, 1.0), (5.0, 6.0)], max_gap=4.0) == [(0.0, 6.0)]
    assert merge_intervals([(0.0, 1.0), (5.0, 6.0)], max_gap=3.9) == [
        (0.0, 1.0),
        (5.0, 6.0),
    ]


def test_merge_intervals_fully_overlapping_intervals_collapse():
    assert merge_intervals([(0.0, 10.0), (2.0, 4.0), (0.0, 10.0)]) == [(0.0, 10.0)]


def test_merge_intervals_nested_interval_does_not_shrink_the_end():
    assert merge_intervals([(0.0, 20.0), (1.0, 2.0)]) == [(0.0, 20.0)]


def test_merge_intervals_sorts_unsorted_input():
    assert merge_intervals([(10.0, 11.0), (0.0, 1.0)]) == [(0.0, 1.0), (10.0, 11.0)]


def test_merge_intervals_does_not_normalise_reversed_intervals():
    """刻意不修正 start > end：调用方有责任传合法区间，这里只保证结果确定。"""
    assert merge_intervals([(5.0, 1.0)]) == [(5.0, 1.0)]


# --- subtract ---


def test_subtract_no_blocks_returns_interval_unchanged():
    assert subtract((0.0, 10.0), []) == [(0.0, 10.0)]


def test_subtract_block_in_the_middle_splits_into_two():
    assert subtract((0.0, 10.0), [(4.0, 6.0)]) == [(0.0, 4.0), (6.0, 10.0)]


def test_subtract_block_covering_everything_returns_empty():
    assert subtract((2.0, 8.0), [(0.0, 10.0)]) == []


def test_subtract_block_clipping_the_head():
    assert subtract((0.0, 10.0), [(0.0, 4.0)]) == [(4.0, 10.0)]


def test_subtract_block_clipping_the_tail():
    assert subtract((0.0, 10.0), [(6.0, 12.0)]) == [(0.0, 6.0)]


def test_subtract_disjoint_block_is_a_noop():
    assert subtract((0.0, 10.0), [(20.0, 30.0)]) == [(0.0, 10.0)]


def test_subtract_touching_block_is_a_noop():
    assert subtract((0.0, 10.0), [(10.0, 20.0)]) == [(0.0, 10.0)]
    assert subtract((10.0, 20.0), [(0.0, 10.0)]) == [(10.0, 20.0)]


def test_subtract_multiple_blocks_produce_multiple_pieces():
    assert subtract((0.0, 20.0), [(4.0, 6.0), (10.0, 12.0)]) == [
        (0.0, 4.0),
        (6.0, 10.0),
        (12.0, 20.0),
    ]


def test_subtract_drops_zero_length_pieces():
    assert subtract((0.0, 0.0), []) == []


def test_subtract_reversed_interval_returns_empty():
    assert subtract((10.0, 5.0), []) == []
