import pytest

from tenmin.ingest.normalize import build_track, merge_continuations
from tenmin.models import DialogueLine


def dline(idx, start, end, text, **kwargs):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, **kwargs)


def test_merge_continuations_merges_split_sentence():
    lines = [dline(1, 0.0, 1.0, "我觉得这件事"), dline(2, 1.1, 2.5, "应该再想想")]
    merged = merge_continuations(lines)
    assert len(merged) == 1
    assert merged[0].idx == 1
    assert merged[0].text == "我觉得这件事应该再想想"
    assert merged[0].start == pytest.approx(0.0)
    assert merged[0].end == pytest.approx(2.5)
    assert merged[0].merged_from == [2]


def test_merge_continuations_respects_terminal_punctuation():
    lines = [dline(1, 0.0, 1.0, "我知道了。"), dline(2, 1.1, 2.0, "走吧")]
    assert len(merge_continuations(lines)) == 2


def test_merge_continuations_respects_gap():
    lines = [dline(1, 0.0, 1.0, "我觉得这件事"), dline(2, 3.0, 4.0, "应该再想想")]
    assert len(merge_continuations(lines)) == 2


def test_merge_continuations_skips_suspect_lines():
    lines = [dline(1, 0.0, 1.0, "我觉得这件事", suspect=True), dline(2, 1.1, 2.0, "应该再想想")]
    assert len(merge_continuations(lines)) == 2


def test_merge_continuations_skips_different_speaker():
    lines = [
        dline(1, 0.0, 1.0, "我觉得这件事", speaker="甲"),
        dline(2, 1.1, 2.0, "应该再想想", speaker="乙"),
    ]
    assert len(merge_continuations(lines)) == 2


def test_merge_continuations_skips_same_idx_dual_track():
    lines = [dline(7, 0.0, 2.0, "台词"), dline(7, 0.0, 2.0, "独白", kind="monologue")]
    assert len(merge_continuations(lines)) == 2


def test_merge_continuations_respects_length_cap():
    long_a = "啊" * 30
    long_b = "哦" * 30
    assert len(merge_continuations([dline(1, 0.0, 1.0, long_a), dline(2, 1.1, 2.0, long_b)])) == 2


def test_merge_continuations_only_merges_dialogue_kind():
    lines = [
        dline(1, 0.0, 1.0, "制作委员会", kind="credits"),
        dline(2, 1.1, 2.0, "山田", kind="credits"),
    ]
    assert len(merge_continuations(lines)) == 2


def test_merge_continuations_skips_long_lines():
    """6.589s 的「才没有染呢我」是拖长音的完整一句，不是被折断的续行；
    合并它会破坏 Task 11 的字密度最小值锚点。"""
    lines = [dline(1, 0.0, 6.589, "才没有染呢我"), dline(2, 6.7, 8.0, "真的")]
    assert len(merge_continuations(lines)) == 2


def test_build_track_basic(tmp_path):
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n(伊月) 我知道了\n\n"
        "2\n00:00:05,000 --> 00:00:07,000\n<i>才女的侍從</i>\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1, show_title="才女的侍从")
    assert track.episode == 1
    assert track.source == "srt"
    assert track.duration == pytest.approx(7.0)
    assert track.lines[0].speaker == "伊月"
    assert track.lines[0].text == "我知道了"
    assert track.lines[1].text == "才女的侍从"


def test_build_track_honours_explicit_ranges(tmp_path):
    srt = tmp_path / "e01.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n台词\n", encoding="utf-8")
    track = build_track(srt, episode=1, op_range=(1.0, 2.0), ed_range=(2.5, 3.0))
    assert track.op_range == pytest.approx((1.0, 2.0))
    assert track.ed_range == pytest.approx((2.5, 3.0))


def test_build_track_can_disable_conversion(tmp_path):
    srt = tmp_path / "e01.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n才女的侍從\n", encoding="utf-8")
    track = build_track(srt, episode=1, convert_traditional=False)
    assert track.lines[0].text == "才女的侍從"


def test_build_track_does_not_merge_across_cues_by_default(tmp_path):
    """cue 背靠背排且不打句读时，跨 cue 合并会把不同说话人黏成一条。"""
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n是\n\n"
        "2\n00:00:02,001 --> 00:00:03,000\n您有什么吩咐\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1)
    assert len(track.lines) == 2
    assert [ln.text for ln in track.lines] == ["是", "您有什么吩咐"]


def test_build_track_can_opt_into_merging(tmp_path):
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n是\n\n"
        "2\n00:00:02,001 --> 00:00:03,000\n您有什么吩咐\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1, merge_lines=True)
    assert len(track.lines) == 1
    assert track.lines[0].text == "是您有什么吩咐"
    assert track.lines[0].merged_from == [2]


def test_build_track_short_episode_keeps_ambiguous_dialogue(tmp_path):
    """短片长不能让 is_credits 的激进规则对全片生效。

    修复前 in_credit_window 等价于 `start <= 300 or start >= duration - 80`，
    duration=100 时两个条件的并集就是全时间轴，于是「你现在的说话方式也是演出来的吧」
    被 _KEYWORDS_IN_WINDOW 的「演出」子串命中 → kind="credits" → 被 SPEECH_KINDS
    过滤掉 → 语音轨里凭空少一句，两侧静默间隙虚假合并成一个假高光。
    """
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n开场\n\n"
        "2\n00:00:55,000 --> 00:00:58,000\n你现在的说话方式也是演出来的吧\n\n"
        "3\n00:01:38,000 --> 00:01:40,000\n收场\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1)
    assert track.duration == pytest.approx(100.0)
    ambiguous = next(ln for ln in track.lines if "演出来" in ln.text)
    assert ambiguous.kind == "dialogue"
    assert [ln.kind for ln in track.lines] == ["dialogue", "dialogue", "dialogue"]


def test_build_track_empty_srt_opens_no_credit_window(tmp_path):
    """duration=0 时 in_credit_window 曾对每一行恒真（0 <= 300 且 0 >= 0-80）。"""
    srt = tmp_path / "e01.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,000\n监督 山田太郎\n", encoding="utf-8")
    track = build_track(srt, episode=1)
    assert track.duration == pytest.approx(0.0)
    # 「监督」只在 credit 窗内才敢认；片长未知时不开窗，所以它仍是 dialogue。
    assert track.lines[0].kind == "dialogue"


def test_build_track_full_paren_line_is_screen_text(tmp_path):
    """整行都在圆括号里的注释仍然判 screen_text。"""
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n(远处传来钟声)\n\n"
        "2\n00:00:05,000 --> 00:00:07,000\n"
        "（这是一段超过四十个字的长注释所以不会被当成说话人前缀剥掉）\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1)
    assert [ln.kind for ln in track.lines] == ["screen_text", "screen_text"]


def test_build_track_paren_note_plus_dialogue_is_not_swallowed(tmp_path):
    """贪婪 `^[（(].*[）)]$` 会把「（长注释）真台词（结尾）」整行吞成 screen_text。

    后果：真台词被 SPEECH_KINDS 过滤掉，语音轨凭空少一句，两侧静默间隙虚假拉长
    （合并成一个不存在的长「无台词演出段」高光）。禁止中间出现闭括号即可。

    注释刻意超过 40 字：clean._PAREN_PREFIX 的 inner 上限是 40，超了它就整体不匹配，
    行首括号组不会被 extract_prefix 剥掉，body 仍以「（」开头，这才踩得到 _FULL_PAREN。
    """
    srt = tmp_path / "e01.srt"
    note = "这是一段刻意超过四十个中文字符的长注释用来让说话人前缀正则整体失配从而保留行首括号"
    assert len(note) > 40
    srt.write_text(
        f"1\n00:00:01,000 --> 00:00:03,000\n（{note}）我绝对不会放手的（小声）\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1)
    assert len(track.lines) == 1
    assert track.lines[0].kind == "dialogue"
    assert "我绝对不会放手的" in track.lines[0].text


def test_build_track_sorts_cues_by_time(tmp_path):
    """乱序 SRT（合并多个字幕源时常见）必须先按时间排好。

    merge_continuations 只比较相邻元素、in_credit_window 逐条按 duration 判断，
    两者都默认时间有序；乱序输入会导致错误合并与错误的 duration 归因。
    """
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:20,000 --> 00:00:22,000\n第三句\n\n"
        "2\n00:00:01,000 --> 00:00:03,000\n第一句\n\n"
        "3\n00:00:10,000 --> 00:00:12,000\n第二句\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1)
    assert [ln.text for ln in track.lines] == ["第一句", "第二句", "第三句"]
    assert [ln.start for ln in track.lines] == pytest.approx([1.0, 10.0, 20.0])


def test_build_track_sorting_fixes_wrong_merge_on_unsorted_input(tmp_path):
    """乱序时 merge_continuations 会把时间上不相邻的两条黏成一条。"""
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n我觉得这件事\n\n"
        "2\n00:00:30,000 --> 00:00:31,000\n完全无关的一句\n\n"
        "3\n00:00:02,100 --> 00:00:03,000\n应该再想想\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1, merge_lines=True)
    texts = [ln.text for ln in track.lines]
    assert "我觉得这件事应该再想想" in texts
    assert "完全无关的一句" in texts


# --- 黄金样本：12 类实测脏数据 ---


def test_golden_noise_line_375(golden_track, lines_for_idx):
    found = lines_for_idx(golden_track, 375)
    assert found, "第 375 行不见了"
    assert any(ln.kind == "noise" for ln in found)


@pytest.mark.parametrize("idx", [41, 67, 105, 187, 291])
def test_golden_suspect_lines(golden_track, lines_for_idx, idx):
    found = lines_for_idx(golden_track, idx)
    assert found, f"第 {idx} 行不见了"
    assert any(ln.suspect for ln in found), f"第 {idx} 行没被标 suspect"


def test_golden_line_41_ocr_text_survives_as_dialogue(golden_track, lines_for_idx):
    """车牌 OCR 混入的行不删除，只打标 —— 删除由 LLM 提示词负责。"""
    found = lines_for_idx(golden_track, 41)
    assert any("339" in ln.text for ln in found)


def test_golden_dual_track_line_200(golden_track, lines_for_idx):
    found = lines_for_idx(golden_track, 200)
    assert len(found) == 2, f"第 200 行应拆成 2 条，实际 {len(found)} 条"
    first, second = found
    assert first.kind == "dialogue"
    assert second.kind == "monologue"
    assert second.speaker == "伊月"
    assert first.start == pytest.approx(second.start)
    assert first.end == pytest.approx(second.end)


@pytest.mark.parametrize("idx", [201, 202])
def test_golden_dual_track_lines_201_202(golden_track, lines_for_idx, idx):
    found = lines_for_idx(golden_track, idx)
    assert len(found) == 2, f"第 {idx} 行应拆成 2 条，实际 {len(found)} 条"
    assert found[1].kind == "monologue"


@pytest.mark.parametrize("idx", [56, 57])
def test_golden_title_card_prefix_stripped(golden_track, lines_for_idx, idx):
    found = lines_for_idx(golden_track, idx)
    assert found, f"第 {idx} 行不见了"
    for ln in found:
        assert not ln.text.startswith("(")
        assert not ln.text.startswith("（")
        assert "第二集" not in ln.text
        assert ln.speaker is None


def test_golden_track_shape(golden_track):
    assert golden_track.episode == 2
    assert golden_track.duration == pytest.approx(1416.622, abs=0.001)
    # 拆行会让总行数略多于 405，合并会让它略少；不应该差太多
    assert 400 <= len(golden_track.lines) <= 420
    dialogue = [ln for ln in golden_track.lines if ln.kind in ("dialogue", "monologue")]
    assert len(dialogue) >= 350
