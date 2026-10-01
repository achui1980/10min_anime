"""硬字幕 OCR 层。刻意不测 Vision 的识别准确率（那要 macOS + 真片源，见 -m ocr 那条），
只测我们自己那几层：步长、滤镜几何、pts 解析、单帧清洗、多帧归并、编排与报错。"""

from __future__ import annotations

import pytest

from tenmin.ingest import ocr
from tenmin.ingest.ocr import FrameText, TextBox

# --- 步长 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src_fps", "expected"),
    [(24000 / 1001, 6), (24.0, 6), (30.0, 8), (60.0, 15), (120.0, 30), (1.0, 1), (0.5, 1)],
)
def test_sample_step_rounds_to_a_real_frame(src_fps, expected):
    """每次取到的都是一帧真实存在的帧；源帧率比目标密度还低时每帧都取（步长不会是 0）。"""
    assert ocr.sample_step(src_fps, 4.0) == expected


# --- 滤镜几何 -------------------------------------------------------------------


def test_frame_filter_for_a_1080p_source():
    chain, out_height = ocr.frame_filter(width=1920, height=1080, step=6, crop_top=0.72)
    # 1080 × 0.28 = 302.4 → 302；起点 1080 - 302 = 778；1280 × 302 / 1920 = 201.3 → 202
    assert out_height == 202
    assert chain == (
        "select='not(mod(n\\,6))',crop=1920:302:0:778,scale=1280:202,format=gray,showinfo"
    )


@pytest.mark.parametrize(("width", "height"), [(1280, 720), (3840, 2160), (1440, 1080)])
def test_frame_filter_always_scales_to_the_fixed_width_and_an_even_height(width, height):
    """宽度固定、高度取偶数：rawvideo 一帧的字节数必须在开跑前就是确定的整数。"""
    chain, out_height = ocr.frame_filter(width=width, height=height, step=6, crop_top=0.72)
    assert f"scale={ocr.OCR_WIDTH}:{out_height}," in chain
    assert out_height % 2 == 0
    assert chain.endswith(",showinfo")


def test_frame_filter_crops_all_the_way_to_the_bottom():
    chain, _ = ocr.frame_filter(width=1920, height=1080, step=6, crop_top=0.5)
    assert "crop=1920:540:0:540," in chain


# --- pts 解析 -------------------------------------------------------------------

# 真实 ffmpeg 9.0.1 的 showinfo 输出（`-f lavfi -i testsrc ... ,showinfo`），含它在同一个
# 滤镜实例上打的 config / color_range 行与 Output 段落 —— 那些行不能被当成帧。
SHOWINFO_SAMPLE = """\
[Parsed_showinfo_4 @ 0xad8c34fc0] config in time_base: 1001/24000, frame_rate: 24000/1001
[Parsed_showinfo_4 @ 0xad8c34fc0] config out time_base: 0/0, frame_rate: 0/0
[Parsed_showinfo_4 @ 0xad8c34fc0] n:   0 pts:      0 pts_time:0       duration:      1 \
duration_time:0.0417083 fmt:gray cl:unspecified sar:1/1 s:1280x264 i:P iskey:1 type:I
[Parsed_showinfo_4 @ 0xad8c34fc0] color_range:pc color_space:unknown color_primaries:unknown
Output #0, rawvideo, to 'pipe:':
[Parsed_showinfo_4 @ 0xad8c34fc0] n:   1 pts:      6 pts_time:0.25025 duration:      1 \
duration_time:0.0417083 fmt:gray cl:unspecified sar:1/1 s:1280x264 i:P iskey:1 type:I
[Parsed_showinfo_4 @ 0xad8c34fc0] n:   2 pts:     12 pts_time:0.5005  duration:      1 \
duration_time:0.0417083 fmt:gray cl:unspecified sar:1/1 s:1280x264 i:P iskey:1 type:I
"""


def test_parse_showinfo_pts_reads_each_frame_in_order():
    assert ocr.parse_showinfo_pts(SHOWINFO_SAMPLE) == [0.0, 0.25025, 0.5005]


def test_parse_showinfo_pts_keeps_uneven_spacing_as_is():
    """可变帧率片源上间隔不均匀，原样采用 —— 不许按序号 × 步长重算。"""
    stderr = "".join(
        f"[Parsed_showinfo_4 @ 0x1] n: {i} pts: {i} pts_time:{t} duration: 1\n"
        for i, t in enumerate(["0", "0.2", "0.55", "0.6"])
    )
    assert ocr.parse_showinfo_pts(stderr) == [0.0, 0.2, 0.55, 0.6]


def test_parse_showinfo_pts_refuses_an_unreadable_timestamp():
    with pytest.raises(ocr.OCRError):
        ocr.parse_showinfo_pts("[Parsed_showinfo_4 @ 0x1] n: 0 pts: NOPTS pts_time:NOPTS\n")


def test_parse_showinfo_pts_of_nothing_is_empty():
    assert ocr.parse_showinfo_pts("") == []


# --- 单帧清洗 -------------------------------------------------------------------


def _box(text: str, *, center_x: float = 0.5, y: float = 0.2, width: float = 0.3) -> TextBox:
    return TextBox(text=text, x=center_x - width / 2, y=y, width=width, height=0.15)


def test_frame_text_keeps_a_centered_subtitle():
    assert ocr.frame_text([_box("我們走吧")], center_tolerance=0.08) == "我們走吧"


def test_frame_text_drops_staff_credits_on_the_sides():
    boxes = [_box("監督 山田", center_x=0.15), _box("我們走吧"), _box("原作 鈴木", center_x=0.85)]
    assert ocr.frame_text(boxes, center_tolerance=0.08) == "我們走吧"


def test_frame_text_center_tolerance_is_measured_from_the_box_center():
    """判的是框**中心**离中线多远，不是框的左边沿（宽框的左边沿离中线很远）。"""
    assert ocr.frame_text([_box("偏了", center_x=0.59)], center_tolerance=0.08) == ""
    assert ocr.frame_text([_box("差一點", center_x=0.57)], center_tolerance=0.08) == "差一點"
    wide = _box("一整行很長的字幕", center_x=0.5, width=0.8)
    assert ocr.frame_text([wide], center_tolerance=0.08) == "一整行很長的字幕"


def test_frame_text_requires_a_chinese_character():
    boxes = [_box("MIMM", y=0.6), _box("/1/7", y=0.3)]
    assert ocr.frame_text(boxes, center_tolerance=0.08) == ""


def test_frame_text_joins_lines_top_to_bottom_with_vision_y_pointing_up():
    """Vision 的 y 轴从下往上：y 大的那一行在画面上方，应该排在前面。"""
    boxes = [_box("-我才不要", y=0.1), _box("-快過來", y=0.5)]
    assert ocr.frame_text(boxes, center_tolerance=0.08) == "-快過來\n-我才不要"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("不要•••", "不要…"),
        ("不要⋯", "不要…"),
        ("不要...", "不要…"),
        ("不要..", "不要…"),
        ("不要。。。", "不要…"),
        ("不要・・・", "不要…"),
        ("不要……", "不要…"),
        ("不要…。", "不要…"),
        ("不要.", "不要…"),
        ("不要。", "不要…"),
        ("我…我不知道", "我…我不知道"),
        ("伊莉莎白·克洛", "伊莉莎白·克洛"),
        ("第3.5話", "第3.5話"),
    ],
)
def test_normalize_ellipsis(raw, expected):
    assert ocr.normalize_ellipsis(raw) == expected


def test_frame_text_normalizes_the_ellipsis_on_every_line():
    boxes = [_box("等等。", y=0.5), _box("不要•••", y=0.1)]
    assert ocr.frame_text(boxes, center_tolerance=0.08) == "等等…\n不要…"


def test_frame_text_of_an_empty_frame_is_empty():
    assert ocr.frame_text([], center_tolerance=0.08) == ""


def test_frame_text_strips_surrounding_whitespace():
    assert ocr.frame_text([_box("  我們走吧 ")], center_tolerance=0.08) == "我們走吧"


# --- 多帧归并 -------------------------------------------------------------------

INTERVAL = 0.25


def _frames(*texts: str) -> list[FrameText]:
    return [FrameText(time=i * INTERVAL, text=text) for i, text in enumerate(texts)]


def _merge(frames: list[FrameText], **overrides) -> list[tuple[float, float, str]]:
    kwargs = {"interval": INTERVAL, "similarity": 0.6, "min_frames": 2, **overrides}
    return [(c.start, c.end, c.text) for c in ocr.merge_frames(frames, **kwargs)]


def test_similar_frames_merge_into_one_cue():
    assert _merge(_frames("我們走吧", "我們走吧", "我們走吧")) == [(0.0, 0.75, "我們走吧")]


def test_the_cue_end_is_the_last_frame_plus_one_interval():
    cues = _merge(_frames("", "我們走吧", "我們走吧"))
    assert cues == [(0.25, 0.75, "我們走吧")]


def test_a_single_blank_frame_inside_a_subtitle_is_tolerated():
    assert _merge(_frames("我們走吧", "", "我們走吧")) == [(0.0, 0.75, "我們走吧")]


def test_two_blank_frames_end_the_cue():
    cues = _merge(_frames("我們走吧", "我們走吧", "", "", "我們走吧", "我們走吧"))
    assert cues == [(0.0, 0.5, "我們走吧"), (1.0, 1.5, "我們走吧")]


def test_a_dissimilar_frame_starts_a_new_cue():
    cues = _merge(_frames("我們走吧", "我們走吧", "明天見", "明天見"))
    assert cues == [(0.0, 0.5, "我們走吧"), (0.5, 1.0, "明天見")]


def test_similarity_compares_against_the_last_non_blank_frame():
    """夹了空帧之后，比的仍然是上一个**有字**的帧，不是那个空帧。"""
    cues = _merge(_frames("我們走吧", "", "我們走吧", "我們走吧"))
    assert cues == [(0.0, 1.0, "我們走吧")]


def test_the_most_frequent_spelling_wins_the_vote():
    """多帧投票修掉单帧错字（未/末）与前缀杂字。"""
    cues = _merge(_frames("還未結束", "還末結束", "還未結束", "C還未結束"))
    assert cues == [(0.0, 1.0, "還未結束")]


def test_a_tied_vote_goes_to_the_earliest_spelling():
    """平票取最早出现的那个，结果确定（同一份素材跑两次得同一个字节）。"""
    cues = _merge(_frames("還末結束", "還未結束", "還未結束", "還末結束"))
    assert cues == [(0.0, 1.0, "還末結束")]


def test_cues_seen_on_fewer_than_min_frames_are_dropped():
    """实测只出现 1 帧的全是噪声。"""
    cues = _merge(_frames("找", "", "", "我們走吧", "我們走吧"))
    assert cues == [(0.75, 1.25, "我們走吧")]


def test_min_frames_is_configurable():
    assert _merge(_frames("找"), min_frames=1) == [(0.0, 0.25, "找")]


def test_an_overlapping_end_is_clamped_to_the_next_start():
    """interval 比实际帧距长时（可变帧率），上一条的终点会越过下一条的起点。"""
    frames = [
        FrameText(time=0.0, text="我們走吧"),
        FrameText(time=0.25, text="我們走吧"),
        FrameText(time=0.4, text="明天見"),
        FrameText(time=0.65, text="明天見"),
    ]
    assert _merge(frames) == [(0.0, 0.4, "我們走吧"), (0.4, 0.9, "明天見")]


def test_merged_cues_are_numbered_from_one():
    cues = ocr.merge_frames(
        _frames("我們走吧", "我們走吧", "", "", "明天見", "明天見"),
        interval=INTERVAL,
        similarity=0.6,
        min_frames=2,
    )
    assert [c.idx for c in cues] == [1, 2]


def test_merging_nothing_gives_nothing():
    assert _merge([]) == []
    assert _merge(_frames("", "", "")) == []
