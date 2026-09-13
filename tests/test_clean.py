from tenmin.ingest.clean import (
    apply_glossary,
    clean_text,
    extract_prefix,
    is_noise,
    is_suspect,
    split_dual_track,
    strip_markup,
    to_simplified,
)


def test_strip_markup_removes_ass_override_blocks():
    assert strip_markup(r"{\an8}{\fs20}你好") == "你好"


def test_strip_markup_removes_html_tags():
    assert strip_markup("<i>台词</i>") == "台词"
    assert strip_markup('<font color="#ffffff">台词</font>') == "台词"


def test_strip_markup_keeps_newlines_and_trims_each_line():
    assert strip_markup("  上句  \n  下句  ") == "上句\n下句"


def test_strip_markup_collapses_inner_spaces_but_keeps_cjk():
    assert strip_markup("你好\u3000世界") == "你好 世界"


def test_to_simplified():
    assert to_simplified("才女的侍從") == "才女的侍从"
    assert to_simplified("製作委員會") == "制作委员会"


def test_apply_glossary_runs_on_simplified_text():
    assert apply_glossary("伊月与大小姐", {"大小姐": "千金"}) == "伊月与千金"


def test_apply_glossary_longest_key_wins():
    glossary = {"天王寺": "TENNOJI", "天王寺璃子": "RIKO"}
    assert apply_glossary("天王寺璃子来了", glossary) == "RIKO来了"


def test_clean_text_pipeline_order():
    # 繁体 -> 简体 -> 术语表覆盖，术语表用简体 key 就能命中
    assert clean_text("才女的侍從", glossary={"侍从": "管家"}) == "才女的管家"


def test_clean_text_can_skip_conversion():
    assert clean_text("才女的侍從", convert=False) == "才女的侍從"


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text("   \n  ") == ""


# --- 结构层清洗 ---


def test_extract_prefix_speaker():
    assert extract_prefix("(伊月) 我知道了") == ("伊月", "我知道了")
    assert extract_prefix("（天王寺）走了") == ("天王寺", "走了")


def test_extract_prefix_long_prefix_is_annotation_not_speaker():
    speaker, rest = extract_prefix("(第二集当侍从的第一天) 早上好")
    assert speaker is None
    assert rest == "早上好"


def test_extract_prefix_title_card_short_but_matches_episode_pattern():
    speaker, rest = extract_prefix("(第二集) 早上好")
    assert speaker is None
    assert rest == "早上好"


def test_extract_prefix_whole_line_is_parenthesized_returns_unchanged():
    # 整行都在括号里，交给 credits / screen_text 判定，这里不动它
    assert extract_prefix("(制作委员会)") == (None, "(制作委员会)")


def test_extract_prefix_no_prefix():
    assert extract_prefix("普通台词") == (None, "普通台词")


def test_is_noise():
    assert is_noise("-") is True
    assert is_noise("- – —") is True
    assert is_noise("00") is True
    assert is_noise("%") is True
    assert is_noise("正常台词") is False
    assert is_noise("") is True


def test_is_suspect_digit_run():
    assert is_suspect("80-08 浙谷339") is True


def test_is_suspect_short_latin_segment():
    assert is_suspect("boo") is True


def test_is_suspect_unbalanced_quote():
    assert is_suspect('"杰斯电器\n00') is True


def test_is_suspect_odd_symbol():
    assert is_suspect("じやがいも\n%") is True


def test_is_suspect_clean_line():
    assert is_suspect("我今天要去学院上课") is False
    assert is_suspect("这药就是会让人硬不起来的药") is False


def test_split_dual_track():
    parts = split_dual_track("刚才的回答相当精彩\n(伊月) 她随时都被旁人包围")
    assert parts == ["刚才的回答相当精彩", "(伊月) 她随时都被旁人包围"]


def test_split_dual_track_no_second_speaker_returns_single():
    assert split_dual_track("第二句上\n第二句下") == ["第二句上\n第二句下"]


def test_split_dual_track_three_segments():
    parts = split_dual_track("(甲) 一\n(乙) 二\n(丙) 三")
    assert parts == ["(甲) 一", "(乙) 二", "(丙) 三"]


# --- 术语表：为什么**不**改成 alternation 正则 ---


def test_apply_glossary_cascades_through_replacement_values():
    """`{"A": "B", "B": "C"}` 作用在 `"A"` 上会连锁两次，得到 `"C"`。

    这条（以及下面两条）不是「顺便测一下」，而是**否决 alternation 正则方案的依据**：
    `re.compile("A|B")` 一次扫描只会把 `"A"` 换成 `"B"` 就收工，给出 `"B"`。
    要求是「必须与按长度倒序逐个 str.replace 逐字节等价」，
    alternation 做不到，所以这次只把每条 cue 重复做的 key 排序缓存下来，
    替换本身照旧走 str.replace。
    """
    assert apply_glossary("A", {"A": "B", "B": "C"}) == "C"


def test_apply_glossary_equal_length_overlap_follows_insertion_order():
    """等长 key 重叠时，胜负由 dict 的插入顺序决定，而不是位置最靠左者优先。

    `sorted(key=len, reverse=True)` 是稳定排序，等长 key 保持插入顺序，于是先登记的
    `"BC"` 先吃掉 `"ABC"` 的后两个字符。alternation 的最左匹配会先命中 `"AB"`，
    给出 `"xC"` —— 两者不等价。
    """
    assert apply_glossary("ABC", {"BC": "y", "AB": "x"}) == "Ay"
    assert apply_glossary("ABC", {"AB": "x", "BC": "y"}) == "xC"


def test_apply_glossary_longer_key_beats_shorter_prefix():
    assert apply_glossary("ABC", {"AB": "x", "ABC": "y"}) == "y"


def test_apply_glossary_skips_empty_key():
    assert apply_glossary("你好", {"": "X", "你": "我"}) == "我好"


def test_apply_glossary_sorts_the_keys_only_once_across_cues(monkeypatch):
    """normalize 在 cue 循环里调 apply_glossary，原先每条 cue 都重排一次 key。"""
    from tenmin.ingest import clean as clean_module

    clean_module._glossary_keys.cache_clear()
    glossary = {"甲": "1", "乙乙": "2", "丙丙丙": "3"}
    before = clean_module._glossary_keys.cache_info()
    for _ in range(50):
        apply_glossary("甲乙乙丙丙丙", glossary)
    after = clean_module._glossary_keys.cache_info()
    assert after.misses - before.misses == 1
    assert after.hits - before.hits == 49


def test_apply_glossary_cache_key_keeps_insertion_order_apart():
    """两个内容相同但插入顺序不同的术语表结果不同，所以缓存必须把它们分开。

    缓存键刻意用 `tuple(glossary.items())`（保持插入顺序）而**不是**
    `tuple(sorted(items))` —— 后者会把这两个表折成同一个键，直接给出错误答案。
    """
    assert apply_glossary("ABC", {"BC": "y", "AB": "x"}) != apply_glossary(
        "ABC", {"AB": "x", "BC": "y"}
    )


# --- 标题卡正则收敛 ---


def test_extract_prefix_title_card_with_spaces_is_not_a_speaker():
    """`_TITLE_CARD` 原先在 clean.py 与 credits.py 各有一份且不一致：credits 那份多了
    `\\s*`，能接 `第 3 集`，clean 这份不能。收敛成宽的那份（严格超集），
    因为在 extract_prefix 里「多认出一个标题卡」= 少把标题卡误当说话人，方向是安全的。
    """
    assert extract_prefix("（第 3 集）早上好") == (None, "早上好")
    assert extract_prefix("（第3集）早上好") == (None, "早上好")


def test_extract_prefix_still_accepts_a_normal_short_speaker():
    assert extract_prefix("（伊月）我知道了") == ("伊月", "我知道了")
