import pytest

from tenmin.ingest.credits import find_credit_ranges, is_credits
from tenmin.models import DialogueLine


def line(idx: int, start: float, end: float, text: str = "x") -> DialogueLine:
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind="credits")


def test_is_credits_copyright():
    assert is_credits("©2024 才女的侍从制作委员会") is True
    assert is_credits("(C) SOME STUDIO") is True


def test_is_credits_keywords_both_scripts():
    assert is_credits("制作委员会") is True
    assert is_credits("製作委員會") is True
    assert is_credits("作词 作曲 编曲") is True
    assert is_credits("フォント协力") is True
    assert is_credits("监督 山田太郎") is True


def test_is_credits_title_card():
    assert is_credits("(第二集当侍从的第一天)") is True
    assert is_credits("第2话 决战") is True


def test_is_credits_title_wrapped_in_brackets():
    assert is_credits("《才女的侍从》", show_title="才女的侍从") is True
    # 字符集重合率不够，不算
    assert is_credits("《完全无关的作品》", show_title="才女的侍从") is False


def test_is_credits_name_list_only_inside_window():
    text = "河原正信 有贺史英"
    assert is_credits(text, in_credit_window=True) is True
    assert is_credits(text, in_credit_window=False) is False


def test_is_credits_latin_heavy_only_inside_window():
    text = "HE IYA Synergy SP"
    assert is_credits(text, in_credit_window=True) is True
    assert is_credits(text, in_credit_window=False) is False


def test_is_credits_does_not_flag_normal_dialogue():
    for text in ["我今天要去学院上课", "早安 早安", "这药就是会让人硬不起来的药", "才没有染呢我"]:
        assert is_credits(text, in_credit_window=True) is False, text


def test_find_credit_ranges_golden_shape():
    lines = [
        line(51, 153.486, 158.0),
        line(52, 175.0, 180.0),
        line(53, 200.0, 205.0),
        line(54, 220.0, 224.681),
        line(401, 1348.180, 1355.0),
        line(403, 1370.0, 1380.0),
        line(404, 1395.0, 1400.0),
        line(405, 1410.0, 1416.622),
    ]
    op, ed = find_credit_ranges(lines, duration=1416.622)
    assert op == pytest.approx((153.486, 224.681))
    assert ed == pytest.approx((1348.180, 1416.622))


def test_find_credit_ranges_ignores_too_short_op_cluster():
    lines = [line(10, 100.0, 105.0), line(11, 106.0, 110.0)]
    op, ed = find_credit_ranges(lines, duration=1400.0)
    assert op is None
    assert ed is None


def test_find_credit_ranges_no_credits_lines():
    assert find_credit_ranges([], duration=1400.0) == (None, None)


def test_find_credit_ranges_picks_longest_op_candidate():
    lines = [
        line(1, 70.0, 75.0),
        line(2, 100.0, 115.0),  # 簇 A: 70 -> 115，跨度 45
        line(3, 200.0, 205.0),
        line(4, 230.0, 235.0),
        line(5, 260.0, 290.0),  # 簇 B: 200 -> 290，跨度 90
    ]
    op, _ = find_credit_ranges(lines, duration=1400.0)
    assert op == pytest.approx((200.0, 290.0))


def test_golden_sample_credit_lines(golden_track):
    kinds = {}
    for ln in golden_track.lines:
        kinds.setdefault(ln.idx, set()).add(ln.kind)
    for idx in (51, 52, 53, 54, 401, 403, 404, 405):
        assert "credits" in kinds[idx], f"第 {idx} 行没被识别成 credits"


def test_golden_sample_op_ed_ranges(golden_track):
    assert golden_track.op_range is not None
    assert golden_track.ed_range is not None
    assert golden_track.op_range[0] == pytest.approx(153.486, abs=0.001)
    span = golden_track.op_range[1] - golden_track.op_range[0]
    assert 60.0 <= span <= 100.0, f"OP 跨度 {span}s 不在预期区间"
    assert golden_track.ed_range[0] == pytest.approx(1348.180, abs=0.001)
