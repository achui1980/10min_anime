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
