import pytest

from tenmin.ingest.credits import find_credit_ranges, in_credit_window, is_credits
from tenmin.models import DialogueLine


def line(idx: int, start: float, end: float, text: str = "x") -> DialogueLine:
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind="credits")


def speech(idx: int, start: float, end: float, text: str = "台词") -> DialogueLine:
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind="dialogue")


def test_is_credits_copyright():
    assert is_credits("©2024 才女的侍从制作委员会") is True
    assert is_credits("(C) SOME STUDIO") is True


def test_is_credits_keywords_both_scripts():
    assert is_credits("制作委员会") is True
    assert is_credits("製作委員會") is True
    assert is_credits("作词 作曲 编曲") is True
    assert is_credits("フォント协力") is True
    assert is_credits("监督 山田太郎", in_credit_window=True) is True
    assert is_credits("监督 山田太郎", in_credit_window=False) is False


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


def test_is_credits_name_list_accepts_ragged_segments():
    # 黄金样本第 52 行，段长 1/2/1/5。齐整的 {2,4} 正则接不到它，OP 区间就算不出来。
    assert is_credits("慧 诹访 豊 和田雄一郎", in_credit_window=True) is True


def test_in_credit_window_excludes_tail_dialogue():
    duration = 1416.622
    assert in_credit_window(153.486, duration) is True  # OP staff
    assert in_credit_window(1348.180, duration) is True  # ED staff 第一行
    # 黄金样本 398/400 行是真台词。片尾窗若放到 150s，会被纯人名规则误判成 credits，
    # ED 区间起点就会错成 1314.396。
    assert in_credit_window(1325.532, duration) is False
    assert in_credit_window(1314.396, duration) is False
    assert in_credit_window(700.0, duration) is False


def test_in_credit_window_leaves_a_free_middle_on_short_tracks():
    """短片长时两个窗口的并集不能覆盖整条时间轴。

    修复前的判据是 `start <= 300 or start >= duration - 80`，duration <= 380 时
    这两个条件的并集就是全时间轴 → 每一行都 in_credit_window → is_credits 的
    激进规则（通用中文词「演出」「制作」、纯人名罗列、拉丁占比）对全片生效 →
    真台词被踢出语音轨 → 两侧静默间隙虚假合并成假高光。
    """
    for duration in (100.0, 380.0):
        outside = [
            t
            for t in (d * duration / 20 for d in range(20))
            if not in_credit_window(t, duration)
        ]
        assert outside, f"duration={duration} 时整条时间轴都落在 credit 窗内"


def test_in_credit_window_disables_tail_window_when_it_would_touch_head():
    """片长不足以容纳「片头窗 + 中段 + 片尾窗」时，片尾窗必须整体关闭。"""
    # 380 秒：片头窗 300 + 片尾窗 80 刚好首尾相接，没有中段可留。
    assert in_credit_window(370.0, 380.0) is False
    # 381 秒同理（预算 190.5 仍远不够 300+80），也不该开尾窗。
    assert in_credit_window(370.0, 381.0) is False
    # 800 秒：预算 400 >= 300+80，两个窗都按标称值生效，中段 300-720 自由。
    assert in_credit_window(250.0, 800.0) is True
    assert in_credit_window(500.0, 800.0) is False
    assert in_credit_window(750.0, 800.0) is True


def test_in_credit_window_unknown_duration_opens_nothing():
    """duration=0（空字幕/解析失败）时不能把激进规则对全片放开。"""
    assert in_credit_window(0.0, 0.0) is False
    assert in_credit_window(10.0, 0.0) is False


def test_in_credit_window_unchanged_for_full_length_episodes():
    """实测最短的真实素材 1315.94 秒，预算 657.97 >= 300+80，行为与修复前逐点一致。"""
    duration = 1315.94
    assert in_credit_window(299.9, duration) is True
    assert in_credit_window(300.1, duration) is False
    assert in_credit_window(duration - 80.1, duration) is False
    assert in_credit_window(duration - 79.9, duration) is True


def test_op_from_silence_prefers_gap_inside_span_window():
    # 一条 credits 行都没有 → 聚簇路径算不出 OP → 走 _op_from_silence 兜底。
    # 静区一 40.0-190.0（150s，超出 OP_MAX_SILENT_SPAN，干扰项且比真 OP 更长）
    # 静区二 200.0-290.0（90s，落在 [60,120] 内，真 OP）
    # 两个静区起点 40.0 / 200.0 都落在 OP_START_WINDOW=(30,300) 内。
    lines = [
        speech(1, 20.0, 40.0, "台词一"),
        speech(2, 190.0, 200.0, "台词二"),
        speech(3, 290.0, 300.0, "台词三"),
    ]
    op, ed = find_credit_ranges(lines, duration=1400.0)
    assert op == pytest.approx((200.0, 290.0))
    assert ed is None


def test_op_from_silence_ignores_gap_outside_span_window():
    # 只留 150s 的干扰静区，兜底必须放弃而不是硬认一个过长静区当 OP。
    lines = [
        speech(1, 20.0, 40.0, "台词一"),
        speech(2, 190.0, 200.0, "台词二"),
    ]
    op, _ = find_credit_ranges(lines, duration=1400.0)
    assert op is None


def test_golden_sample_credit_lines(golden_track):
    kinds = {}
    for ln in golden_track.lines:
        kinds.setdefault(ln.idx, set()).add(ln.kind)
    for idx in (51, 52, 53, 54, 401, 402, 403, 405):
        assert "credits" in kinds[idx], f"第 {idx} 行没被识别成 credits"


def test_golden_sample_op_ed_ranges(golden_track):
    assert golden_track.op_range is not None
    assert golden_track.ed_range is not None
    assert golden_track.op_range[0] == pytest.approx(153.486, abs=0.001)
    span = golden_track.op_range[1] - golden_track.op_range[0]
    assert 60.0 <= span <= 100.0, f"OP 跨度 {span}s 不在预期区间"
    assert golden_track.ed_range[0] == pytest.approx(1348.180, abs=0.001)


# --- 关键词匹配的热路径 ---


def test_keyword_matcher_matches_the_same_lines_as_the_old_upper_loop():
    """预编译 alternation 正则必须与 `any(kw.upper() in upper for kw in keywords)`
    逐例等价 —— alternation 命中 ⟺ 任一分支是子串。

    正则是用**大写化后**的关键词编译、在大写化后的文本上搜的，所以跟原来那句
    `.upper()` 对 `.upper()` 完全同源，不涉及 re.IGNORECASE 与 str.upper 的差异
    （`ß` -> `SS`、`ﬅ` -> `ST` 这类）。
    """
    from tenmin.ingest.credits import (
        _KEYWORDS_ALWAYS,
        _KEYWORDS_IN_WINDOW,
        _keyword_matcher,
    )

    samples = [
        "制作委员会",
        "製作委員會",
        "作词 作曲 编曲",
        "フォント协力",
        "HE IYA J. C. STAFF 作画部 ディーロク",
        "去studio看看吧",
        "监督 山田太郎",
        "我今天要去学院上课",
        "",
        "STUDIO",
        "主题歌 演唱",
    ]
    for keywords in (_KEYWORDS_ALWAYS, _KEYWORDS_IN_WINDOW):
        matcher = _keyword_matcher(keywords)
        for text in samples:
            upper = text.upper()
            expected = any(kw.upper() in upper for kw in keywords)
            assert bool(matcher.search(upper)) is expected, (text, keywords)


def test_keyword_matcher_rejects_an_empty_keyword_tuple():
    """`"|".join([])` 是空串，编译出来的正则匹配任何文本 —— 那会把整条字幕全判成
    credits。空表一定是改错了，直接拒绝而不是静默生成一个吃掉一切的正则。
    """
    from tenmin.ingest.credits import _keyword_matcher

    with pytest.raises(ValueError):
        _keyword_matcher(())


def test_ascii_studio_no_longer_fires_outside_the_credit_window():
    """`Studio` / `STAFF` 是普通英文单词，`.upper()` 后按子串无条件匹配，
    「去studio看看」这类台词会被整行判成 credits。

    这跟本文件头部记录的「演出」事故是同一类风险，而修法也用同一套既有机制：
    挪进 `_KEYWORDS_IN_WINDOW`，只在片头片尾窗内才敢认。实测 11 集真实素材里
    `Studio` 命中 0 行、`STAFF` 命中 1 行（E04 @1348.1s 的真 ED staff 行，
    而它落在 ED 窗内 [1340.0, 1420.0]），所以产物逐字节不变。
    """
    assert is_credits("去studio看看吧", in_credit_window=False) is False
    assert is_credits("这家staff很热情", in_credit_window=False) is False


def test_real_ed_staff_line_is_still_credits_inside_the_window():
    text = "HE IYA J. C. STAFF 作画部 ディーロク"
    assert is_credits(text, in_credit_window=True) is True


def test_title_overlap_cache_keeps_different_show_titles_apart():
    """`_title_overlap` 的字符集改成按 show_title 缓存，别把两部番的标题串味。"""
    assert is_credits("《才女的侍从》", show_title="才女的侍从") is True
    assert is_credits("《才女的侍从》", show_title="完全无关的作品") is False
    assert is_credits("《完全无关的作品》", show_title="完全无关的作品") is True
    assert is_credits("《才女的侍从》", show_title="才女的侍从") is True


def test_title_overlap_empty_title_never_matches():
    assert is_credits("《随便什么》", show_title="") is False
    assert is_credits("《随便什么》", show_title="　 ") is False
