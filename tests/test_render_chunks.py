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


# --- 句末闭合符 -----------------------------------------------------------
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


# --- 句偏移跟着 render.rate 走 ---------------------------------------------
#
# budget.narration_seconds 与 tts 的时长体检都按 render.rate 缩放，
# 只有这里还写死 4.5 字/秒。后果：rate != "+0%" 时 hold 被按到偏移不对的句边界上。


def _rate_sensitive_sentences() -> list[str]:
    """10 字 + 3 字。offsets 在 +0% 下是 [2.222, 2.889]，+20% 下是 [1.852, 2.407]。

    hold.at=2.4 恰好夹在两个 rate 的「中点」之间（2.4 / 2.13），所以最近的边界会翻面：
    +0% 选第 0 句，+20% 选第 1 句。
    """
    return ["一二三四五六七八九。", "甲乙。"]


def test_sentence_offsets_default_rate_matches_the_historical_numbers():
    """默认 rate 必须逐点等于改动前的行为（speed_factor("+0%") 恒为 1.0）。"""
    assert sentence_offsets(_rate_sensitive_sentences()) == pytest.approx(
        [2.2222, 2.8889], abs=1e-4
    )


def test_sentence_offsets_scale_with_rate():
    """语速快 20% → 同样的字数少占 1/1.2 的时间。"""
    assert sentence_offsets(_rate_sensitive_sentences(), rate="+20%") == pytest.approx(
        [2.2222 / 1.2, 2.8889 / 1.2], abs=1e-4
    )


def test_assign_holds_follows_rate():
    holds = [Hold(at=2.4, duration=2.0, quote="金句")]
    sentences = _rate_sensitive_sentences()
    assert assign_holds(sentences, holds) == {0: 2.0}
    assert assign_holds(sentences, holds, rate="+20%") == {1: 2.0}


def test_plan_chunks_follows_rate():
    beat = Beat(
        id="b1",
        label="Hook",
        role="hook",
        narration="一二三四五六七八九。甲乙。",
        audio=AudioDirection(holds=[Hold(at=2.4, duration=2.0, quote="金句")]),
    )
    assert plan_chunks(beat) == [("一二三四五六七八九。", 2.0), ("甲乙。", 0.0)]
    assert plan_chunks(beat, rate="+20%") == [("一二三四五六七八九。甲乙。", 2.0)]


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


# --- 两种被静默掩盖的坏 hold -----------------------------------------------
#
# 原来 assign_holds 对两种坏输入一声不响：多个 hold 落到同一句边界时直接相加，
# hold.at 超出这段旁白的总跨度时直接贴到最后一句。两者都不失同步，所以走
# warnings 通道（跟 render/tts.py 的 _plan_pronounceable 同一种传法），不判错。


def test_assign_holds_warns_when_two_holds_land_on_the_same_sentence():
    """两个 hold 落到同一句 → 静音时长照旧相加，但要说出来。

    相加本身不算错（成片里就是一段更长的静音），错的是「其中一个 hold 的落点
    被吞掉了」：它本来想停在别的地方。实测 11 份真实 script.json 的 58 个带 hold
    的 beat：0 次发生，所以这条 warning 在健康数据上完全不出声。
    """
    sentences = ["第一句。", "第二句。"]
    holds = [Hold(at=1.7, duration=2.0, quote="甲"), Hold(at=1.9, duration=1.0, quote="乙")]
    warnings: list[str] = []
    assert assign_holds(sentences, holds, warnings=warnings) == {1: 3.0}
    assert len(warnings) == 1
    assert "同一句" in warnings[0]
    assert "乙" in warnings[0]


def test_assign_holds_warns_when_hold_is_beyond_the_whole_beat():
    """hold.at 超出「旁白 + 本节点全部留白」→ 贴到最后一句，同时报一条。

    上界刻意跟 script/validate.py 的 _check_hold_quotes 那道闸（`beat_seconds`）
    完全一致，所以「把留白放在这段旁白的最后」这种正常创作不会触发：实测 11 份真实
    script.json 的 61 个 hold，一条 warning 都不出。真正会踩到的是**人工改过
    03_script/*.json 之后直接 `--from voice`** —— validate_script() 只在 script
    阶段跑，那条路上没有任何人再查一遍 at。
    """
    sentences = ["第一句。", "第二句。"]  # 估算总长 1.778 秒
    holds = [Hold(at=300.0, duration=2.0, quote="金句")]
    warnings: list[str] = []
    assert assign_holds(sentences, holds, warnings=warnings) == {1: 2.0}
    assert len(warnings) == 1
    assert "300.0" in warnings[0]
    assert "最后一句" in warnings[0]


def test_assign_holds_stays_quiet_within_the_beat_span():
    """at 落在「旁白估算总长」之后、但仍在「旁白 + 留白」之内 → 不出声。

    这正是真实数据里唯一一处越过纯旁白总长的形态（work/saijo E09 的 climax：
    at=27.0，纯旁白估算 26.89 秒），它是正常创作，不该报。
    """
    sentences = ["第一句。", "第二句。"]  # 1.778 秒
    holds = [Hold(at=3.0, duration=2.0, quote="金句")]
    warnings: list[str] = []
    assert assign_holds(sentences, holds, warnings=warnings) == {1: 2.0}
    assert warnings == []


def test_plan_chunks_passes_hold_warnings_up():
    beat = Beat(
        id="b1",
        label="Hook",
        role="hook",
        narration="第一句。第二句。",
        audio=AudioDirection(holds=[Hold(at=900.0, duration=2.0, quote="金句")]),
    )
    warnings: list[str] = []
    plan_chunks(beat, warnings=warnings)
    assert any("900.0" in w for w in warnings)


def test_plan_chunks_without_a_warnings_list_still_works():
    """warnings 是可选的：既有调用点（含测试）不传也照旧。"""
    beat = Beat(
        id="b1",
        label="Hook",
        role="hook",
        narration="第一句。第二句。",
        audio=AudioDirection(holds=[Hold(at=900.0, duration=2.0, quote="金句")]),
    )
    assert plan_chunks(beat) == [("第一句。第二句。", 2.0)]


# --- 换行符不再把两个词粘在一起 -------------------------------------------
#
# 原来 split_sentences 见到 `\n` 直接 `continue`，一个分隔符都不留：
# "hello\nworld" → "helloworld"。CJK 无碍（本来就不靠空格分词），旁白里嵌拉丁文时
# 就粘成一个词 —— TTS 会把它当一个生词读，字幕上也少一个词界。
# 实测 11 份真实 script.json 的 73 段 narration：`\n` 出现 **0 次**，所以这是纯
# 防御性修复，不改任何现有产物。


def test_split_sentences_turns_a_newline_into_a_separator():
    assert split_sentences("hello\nworld") == ["hello world"]


def test_split_sentences_collapses_a_run_of_line_breaks():
    assert split_sentences("hello\r\n\n  world") == ["hello world"]


def test_split_sentences_does_not_leak_a_separator_at_a_sentence_edge():
    """换行紧贴句末标点时不能在下一句开头留空格（strip 兜住），也不能改字数。"""
    assert split_sentences("第一句。\n第二句。") == ["第一句。", "第二句。"]


def test_split_sentences_keeps_pure_cjk_byte_for_byte():
    """没有换行的输入必须逐字节不变 —— 全部 115 个存量 chunk 的内容哈希靠这条。"""
    text = "他被塞进面包车后座的那一刻，口袋里只剩两百块日圆——他全部的财产。'走吧。'"
    assert "".join(split_sentences(text)) == text
