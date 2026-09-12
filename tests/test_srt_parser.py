import pytest

from tenmin.ingest.srt_parser import (
    decode_bytes,
    load_srt,
    load_srt_detailed,
    parse_srt,
    parse_srt_detailed,
)

SIMPLE = """\
1
00:00:07,120 --> 00:00:11,480
第一句

2
00:00:12,000 --> 00:00:14,500
第二句上
第二句下
"""


def test_parse_simple():
    cues = parse_srt(SIMPLE)
    assert len(cues) == 2
    assert cues[0].idx == 1
    assert cues[0].start == pytest.approx(7.12)
    assert cues[0].end == pytest.approx(11.48)
    assert cues[0].text == "第一句"
    assert cues[1].idx == 2
    assert cues[1].text == "第二句上\n第二句下"


def test_parse_strips_bom_and_crlf():
    cues = parse_srt("\ufeff1\r\n00:00:01,000 --> 00:00:02,000\r\n台词\r\n\r\n")
    assert len(cues) == 1
    assert cues[0].text == "台词"


def test_parse_falls_back_to_positional_index_when_number_missing():
    text = "00:00:01,000 --> 00:00:02,000\nA\n\n00:00:03,000 --> 00:00:04,000\nB\n"
    cues = parse_srt(text)
    assert [c.idx for c in cues] == [1, 2]


def test_parse_skips_blocks_without_timestamp():
    text = (
        "1\n00:00:01,000 --> 00:00:02,000\nA\n\n"
        "垃圾块没有时间戳\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nC\n"
    )
    cues = parse_srt(text)
    assert [c.idx for c in cues] == [1, 3]
    assert [c.text for c in cues] == ["A", "C"]


def test_parse_tolerates_extra_blank_lines_and_dot_millis():
    text = "1\n00:00:01.000 --> 00:00:02.000\nA\n\n\n\n2\n00:00:03,000 --> 00:00:04,000\nB\n"
    assert len(parse_srt(text)) == 2


def test_parse_clamps_reversed_timestamps():
    cues = parse_srt("1\n00:00:09,000 --> 00:00:02,000\nA\n")
    assert cues[0].end == pytest.approx(cues[0].start)


def test_parse_keeps_empty_text_block():
    cues = parse_srt("1\n00:00:01,000 --> 00:00:02,000\n\n")
    assert len(cues) == 1
    assert cues[0].text == ""


# --- 坏数据的可见性 ---


def test_parse_detailed_counts_skipped_blocks():
    """无时间戳的块被静默跳过，全程没有任何计数。

    一个格式略歪的字幕文件可能丢掉大量对白而流水线毫无提示。
    """
    text = (
        "1\n00:00:01,000 --> 00:00:02,000\nA\n\n"
        "垃圾块没有时间戳\n\n"
        "另一个垃圾块\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nC\n"
    )
    result = parse_srt_detailed(text)
    assert [c.text for c in result.cues] == ["A", "C"]
    assert result.skipped_blocks == 2
    assert result.clamped_cues == 0


def test_parse_detailed_counts_clamped_cues():
    """`end < start` 的坏 cue 被夹成零时长，原先无告警、不设 suspect、不计数。"""
    result = parse_srt_detailed(
        "1\n00:00:09,000 --> 00:00:02,000\nA\n\n2\n00:00:10,000 --> 00:00:11,000\nB\n"
    )
    assert result.clamped_cues == 1
    assert result.skipped_blocks == 0
    assert result.cues[0].clamped is True
    assert result.cues[1].clamped is False


def test_parse_srt_stays_a_plain_list():
    """老调用点与老测试继续拿到纯 list，不受详细版影响。"""
    cues = parse_srt("1\n00:00:01,000 --> 00:00:02,000\nA\n")
    assert isinstance(cues, list)
    assert len(cues) == 1


def test_load_srt_detailed_reads_from_disk(tmp_path):
    path = tmp_path / "e01.srt"
    path.write_text(
        "1\n00:00:09,000 --> 00:00:02,000\nA\n\n没有时间戳\n", encoding="utf-8"
    )
    result = load_srt_detailed(path)
    assert (result.skipped_blocks, result.clamped_cues) == (1, 1)


def test_golden_sample_has_no_skipped_or_clamped(golden_srt_path):
    """黄金样本是干净的：两个计数都必须是 0，否则这两个计数器自己就有问题。"""
    result = load_srt_detailed(golden_srt_path)
    assert (result.skipped_blocks, result.clamped_cues) == (0, 0)


def test_decode_bytes_utf8_bom():    assert decode_bytes("\ufeff你好".encode()) == "你好"


def test_decode_bytes_gbk():
    # 样本必须够长：8 字节时 cp949 与 gb18030 的 chaos/coherence 全为 0.0 打平，
    # charset-normalizer 会误选 cp949。真实字幕是 25KB 级别，嗅探有充足语言信号。
    text = "你好世界，这是一段中文字幕文本。他说得没错。"
    assert "你好世界" in decode_bytes(text.encode("gbk"))


def test_golden_sample_shape(golden_srt_path):
    cues = load_srt(golden_srt_path)
    assert len(cues) == 405
    assert cues[0].idx == 1
    assert cues[-1].idx == 405
    assert 7.0 <= cues[0].start < 8.0
    assert cues[-1].end == pytest.approx(1416.622, abs=0.001)
    assert all(c.end >= c.start for c in cues)
    assert all(b.start >= a.start for a, b in zip(cues, cues[1:], strict=False))


def test_golden_sample_known_anchors(golden_srt_path):
    cues = {c.idx: c for c in load_srt(golden_srt_path)}
    # 第 51 行是 OP staff 名单第一条，spec 记录其起点为 153.486
    assert cues[51].start == pytest.approx(153.486, abs=0.001)
    # 第 269 行「才沒有染呢我」时长 6.589s，是全集字密度最低的一行
    assert cues[269].end - cues[269].start == pytest.approx(6.589, abs=0.001)
