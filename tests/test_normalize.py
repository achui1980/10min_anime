import pytest

from tenmin.config import CreditsConfig
from tenmin.ingest.normalize import (
    build_track,
    credit_range_source,
    merge_continuations,
)
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


def test_build_track_duration_falls_back_to_last_cue_end(tmp_path):
    """没传视频时长就退化到字幕末尾——历史行为，必须保住。"""
    srt = tmp_path / "e01.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n台词\n", encoding="utf-8")
    assert build_track(srt, episode=1).duration == pytest.approx(3.0)
    assert build_track(srt, episode=1, duration=None).duration == pytest.approx(3.0)


def test_build_track_prefers_explicit_duration(tmp_path):
    """字幕通常在片尾前就结束了，视频真实时长才是集长的权威值。

    ED 窗（in_credit_window）、ED 聚簇（ed_cluster_tail_seconds）、尾部无字幕间隙
    三个判定全挂在 duration 上，用字幕末尾会让「距片尾多少秒」整体前移。
    """
    srt = tmp_path / "e01.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n台词\n", encoding="utf-8")
    assert build_track(srt, episode=1, duration=1416.6).duration == pytest.approx(1416.6)


def test_build_track_explicit_duration_shorter_than_cues_is_still_honoured(tmp_path):
    """视频比字幕短（片源被裁过/字幕对不上）时也以视频为准，不悄悄取 max。

    悄悄取两者较大值会让「字幕越界」这种真问题永远看不见；以视频为准至少让
    越界的 cue 在下游表现为「落在片尾之后」，可查。
    """
    srt = tmp_path / "e01.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:30,000\n台词\n", encoding="utf-8")
    assert build_track(srt, episode=1, duration=10.0).duration == pytest.approx(10.0)


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


def test_build_track_records_bad_cue_counts(tmp_path):
    """坏 cue 的数量必须进产物 JSON，否则「丢了一半对白」这件事没人看得见。"""
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:09,000 --> 00:00:02,000\n倒挂的一句\n\n"
        "没有时间戳的垃圾块\n\n"
        "3\n00:00:10,000 --> 00:00:12,000\n正常的一句\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1)
    assert track.skipped_blocks == 1
    assert track.clamped_cues == 1


def test_build_track_marks_clamped_cue_as_suspect(tmp_path):
    """被夹成零时长的 cue 要打 suspect —— 这个字段本来就是为可疑行准备的。"""
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:09,000 --> 00:00:02,000\n倒挂的一句\n\n"
        "2\n00:00:10,000 --> 00:00:12,000\n正常的一句\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1)
    assert track.lines[0].suspect is True
    assert track.lines[1].suspect is False


def test_build_track_clean_srt_has_zero_bad_counts(tmp_path):
    srt = tmp_path / "e01.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n台词\n", encoding="utf-8")
    track = build_track(srt, episode=1)
    assert (track.skipped_blocks, track.clamped_cues) == (0, 0)


def test_build_track_idx_plus_segment_index_is_unique(tmp_path):
    """`(idx, segment_index)` 才是行级唯一键；idx 单独是 cue 级的。

    split_dual_track 把一条 cue 拆成多段时，每段都沿用 cue 的 idx（拆出来的段时间码
    完全相同，所以按 idx 反查时间窗的下游拿到的结果一致），行级区分靠 segment_index。
    """
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n(天王寺)出身高贵\n(伊月)这个人超在意雏子\n\n"
        "2\n00:00:04,000 --> 00:00:05,000\n普通台词\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1)
    assert [ln.idx for ln in track.lines] == [1, 1, 2]
    assert [ln.segment_index for ln in track.lines] == [0, 1, 0]
    keys = [(ln.idx, ln.segment_index) for ln in track.lines]
    assert len(set(keys)) == len(keys)
    # 同 cue 拆出的两段时间码完全相同 —— 这就是「idx 重复无害」的前提。
    assert track.lines[0].start == track.lines[1].start
    assert track.lines[0].end == track.lines[1].end


def test_build_track_keeps_file_serial_in_src_idx(tmp_path):
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "7\n00:00:01,000 --> 00:00:02,000\nA\n\n7\n00:00:03,000 --> 00:00:04,000\nB\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1)
    assert [ln.idx for ln in track.lines] == [1, 2]
    assert [ln.src_idx for ln in track.lines] == [7, 7]


def test_build_track_does_not_merge_two_segments_of_one_cue(tmp_path):
    """_can_merge 的 `prev.idx == nxt.idx` 守卫依赖同 cue 段共享 idx。

    idx 改用 position 之后这个守卫的语义反而更准了：原先两条不相关的 cue 只要文件序号
    撞了就会被误判成「同一个 cue 的两段」而拒绝合并。
    """
    srt = tmp_path / "e01.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n(天王寺)我觉得这件事\n(伊月)应该再想想\n",
        encoding="utf-8",
    )
    track = build_track(srt, episode=1, merge_lines=True)
    assert len(track.lines) == 2


def test_build_track_golden_idx_unchanged_by_position_switch(golden_track):
    """黄金样本的文件序号本来就等于位置，所以 idx 取值一字未变、存量 anchor 不受影响。"""
    assert all(ln.src_idx == ln.idx for ln in golden_track.lines)
    counts: dict[int, int] = {}
    for ln in golden_track.lines:
        counts[ln.idx] = counts.get(ln.idx, 0) + 1
    # idx 只在「一条 cue 被拆成多段」时重复；重复的段时间码相同，按 idx 反查时间窗无害。
    for idx, count in counts.items():
        if count > 1:
            same = [ln for ln in golden_track.lines if ln.idx == idx]
            assert len({(ln.start, ln.end) for ln in same}) == 1, idx
    keys = [(ln.idx, ln.segment_index) for ln in golden_track.lines]
    assert len(set(keys)) == len(keys)


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


# --- OP/ED 区间的三级回退 --------------------------------------------------


def _tc(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round(s % 1 * 1000)):03d}"


def _write_srt(path, cues):
    """cues: [(start, end, text)]"""
    blocks = [
        f"{i}\n{_tc(start)} --> {_tc(end)}\n{text}\n"
        for i, (start, end, text) in enumerate(cues, start=1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


# 片头 31.1s 的 `此花同学 早安` 是 saijo/E01 的真台词：6 个 CJK 字、两段齐整，
# 盲窗（0-300）下会被 is_credits 规则 4（纯人名罗列）误杀。
# 355.0s 的 `副监督 野吕纯恵` 是 saijo2/E01 的真 staff 行：盲窗下 >300 漏成台词。
_MIXED_CUES = [
    (31.1, 37.5, "此花同学 早安"),
    (300.5, 303.0, "正片对白一句"),
    (355.0, 358.0, "副监督 野吕纯恵"),
    (700.0, 703.0, "正片中段对白"),
]


def _kind_at(track, start):
    return next(ln.kind for ln in track.lines if ln.start == pytest.approx(start))


def test_build_track_prefers_episode_range_over_project_default(tmp_path):
    srt = _write_srt(tmp_path / "e.srt", _MIXED_CUES)
    track = build_track(
        srt,
        episode=1,
        duration=1430.0,
        op_range=(181.0, 260.0),
        credits=CreditsConfig(default_op_range=(300.0, 390.0)),
    )
    assert track.op_range == (181.0, 260.0)


def test_build_track_uses_project_default_when_episode_range_missing(tmp_path):
    srt = _write_srt(tmp_path / "e.srt", _MIXED_CUES)
    track = build_track(
        srt,
        episode=1,
        duration=1430.0,
        credits=CreditsConfig(
            default_op_range=(300.0, 390.0), default_ed_range=(1290.0, None)
        ),
    )
    assert track.op_range == (300.0, 390.0)
    # null 终点解析成这一集的真实片长
    assert track.ed_range == (1290.0, 1430.0)


def test_build_track_falls_back_to_inference_when_nothing_filled(tmp_path):
    """两级手填都空时走 find_credit_ranges。这是三个现有 project 的形态。"""
    srt = _write_srt(
        tmp_path / "e.srt",
        [
            (60.0, 63.0, "© 某制作委员会"),
            (90.0, 93.0, "© 作画监督 某人"),
            (105.0, 108.0, "© 某某某"),
            (700.0, 703.0, "正片中段对白"),
        ],
    )
    track = build_track(srt, episode=1, duration=1430.0)
    assert track.op_range == (60.0, 108.0)


def test_build_track_manual_op_range_drives_the_credit_window(tmp_path):
    """手填区间同时修两个方向：接住窗外的 staff 行、放过窗内的真台词。

    盲窗是 [0,300] ∪ [尾-80,尾]，于是 31.1s 的真台词落在窗内被人名正则误杀、
    355.0s 的 staff 行落在窗外漏成台词。手填 OP=[281,368] 之后两者同时纠正。
    """
    srt = _write_srt(tmp_path / "e.srt", _MIXED_CUES)

    blind = build_track(srt, episode=1, duration=1430.0)
    assert _kind_at(blind, 31.1) == "credits"  # 误杀
    assert _kind_at(blind, 355.0) == "dialogue"  # 漏判

    manual = build_track(srt, episode=1, duration=1430.0, op_range=(281.0, 368.0))
    assert _kind_at(manual, 31.1) == "dialogue"  # 救回
    assert _kind_at(manual, 355.0) == "credits"  # 接住
    assert _kind_at(manual, 700.0) == "dialogue"  # 中段不受影响


def test_build_track_without_manual_ranges_is_unchanged(tmp_path):
    """没填任何区间时逐行 kind 必须与改动前一致（产物逐字节不变的单元级版本）。"""
    srt = _write_srt(tmp_path / "e.srt", _MIXED_CUES)
    track = build_track(srt, episode=1, duration=1430.0)
    assert [ln.kind for ln in track.lines] == [
        "credits",
        "dialogue",
        "dialogue",
        "dialogue",
    ]


def test_credit_range_source_reports_which_level_won():
    """来源标注必须与 _resolve_manual_range 同源，否则显示与行为会静默分叉。"""
    assert credit_range_source((181.0, 260.0), (300.0, 390.0), 1430.0) == "episode"
    assert credit_range_source(None, (300.0, 390.0), 1430.0) == "project"
    assert credit_range_source(None, None, 1430.0) == "inferred"
    # 项目级默认写了但在这一集上解析不出合法区间（duration 早于起点）→ 实际走推断，
    # 显示也必须说推断，不能因为「字段填了」就报 project。
    assert credit_range_source(None, (1290.0, None), 100.0) == "inferred"
