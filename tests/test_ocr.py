"""硬字幕 OCR 层。刻意不测 Vision 的识别准确率（那要 macOS + 真片源，见 -m ocr 那条），
只测我们自己那几层：步长、滤镜几何、pts 解析、单帧清洗、多帧归并、编排与报错。"""

from __future__ import annotations

import builtins
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tenmin.config import OcrConfig
from tenmin.ingest import ocr
from tenmin.ingest.ocr import FrameText, TextBox
from tenmin.ingest.srt_parser import load_srt_detailed

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


# --- recognize：编排 ------------------------------------------------------------

# 在任何 monkeypatch 之前留一份真的 _recognize，给「缺依赖」那条用。
_REAL_RECOGNIZE = ocr._recognize

FPS = 24000 / 1001
STEP = 6
SAMPLE_INTERVAL = STEP / FPS
FRAME_SIZE = ocr.OCR_WIDTH * 202  # 1080p 片源、crop_top=0.72 时缩放后的帧


def _showinfo(times: list[float]) -> str:
    return "".join(
        f"[Parsed_showinfo_4 @ 0x1] n: {i} pts: {i * STEP} pts_time:{t} duration: 1\n"
        for i, t in enumerate(times)
    )


def _video(tmp_path: Path) -> Path:
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    return video


def _fake_ffmpeg(
    monkeypatch, texts: list[str], *, times: list[float] | None = None, duration: float = 60.0
) -> dict:
    """把 ffprobe 三连与抽帧换成假货。第 i 帧的首字节是 i，_recognize 的假货按它取 texts[i]。

    calls 里记下转发进来的参数：可执行文件路径与帧大小只能在调用当场查。
    """
    calls: dict = {"probes": [], "recognize": []}
    if times is None:
        times = [i * SAMPLE_INTERVAL for i in range(len(texts))]

    def probe(name, value):
        def fake(path, **kwargs):
            calls["probes"].append((name, Path(path), kwargs))
            return value

        return fake

    monkeypatch.setattr(ocr.ffmpeg, "probe_frame_rate", probe("fps", FPS))
    monkeypatch.setattr(ocr.ffmpeg, "probe_duration", probe("duration", duration))
    monkeypatch.setattr(ocr.ffmpeg, "probe_video_size", probe("size", (1920, 1080)))

    def fake_run_raw_frames(args, *, frame_size, on_frame, ffmpeg):
        calls["args"] = list(args)
        calls["frame_size"] = frame_size
        calls["ffmpeg"] = ffmpeg
        for index in range(len(texts)):
            on_frame(bytes([index]) + b"\0" * (frame_size - 1))
        return _showinfo(times)

    def fake_recognize(frame, *, width, height, language):
        calls["recognize"].append((width, height, language))
        text = texts[frame[0]]
        return [TextBox(text=text, x=0.35, y=0.2, width=0.3, height=0.15)] if text else []

    monkeypatch.setattr(ocr.ffmpeg, "run_raw_frames", fake_run_raw_frames)
    monkeypatch.setattr(ocr, "_recognize", fake_recognize)
    return calls


def test_recognize_writes_an_srt_of_the_merged_cues(tmp_path, monkeypatch):
    texts = ["", "我們走吧", "我們走吧", "我們走吧", "", "", "不要•••", "不要…"]
    _fake_ffmpeg(monkeypatch, texts)
    dest = tmp_path / "srt" / "E11.ocr.srt"

    ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    parsed = load_srt_detailed(dest)
    assert [cue.text for cue in parsed.cues] == ["我們走吧", "不要…"]
    first = parsed.cues[0]
    assert first.start == pytest.approx(SAMPLE_INTERVAL, abs=1e-3)
    assert first.end == pytest.approx(4 * SAMPLE_INTERVAL, abs=1e-3)


def test_recognize_uses_the_real_pts_not_the_frame_index(tmp_path, monkeypatch):
    """时间戳读 showinfo 的 pts，不按「序号 × 步长 / 帧率」推算。"""
    _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧"], times=[100.0, 100.25])
    dest = tmp_path / "E11.ocr.srt"

    ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    cue = load_srt_detailed(dest).cues[0]
    assert cue.start == pytest.approx(100.0)


def test_recognize_asks_ffmpeg_for_cropped_gray_frames(tmp_path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧"])
    video = _video(tmp_path)

    ocr.recognize(
        video,
        tmp_path / "E11.ocr.srt",
        ocr=OcrConfig(language="zh-Hans"),
        ffmpeg_path="/opt/x/ffmpeg",
        ffprobe_path="/opt/x/ffprobe",
    )

    args = calls["args"]
    assert args[args.index("-i") + 1] == str(video)
    assert args[args.index("-vf") + 1] == (
        "select='not(mod(n\\,6))',crop=1920:302:0:778,scale=1280:202,format=gray,showinfo"
    )
    assert args[args.index("-fps_mode") + 1] == "passthrough"
    assert args[args.index("-f") + 1] == "rawvideo"
    assert args[-1] == "-"
    assert calls["frame_size"] == FRAME_SIZE
    assert calls["ffmpeg"] == "/opt/x/ffmpeg"
    assert {kwargs["ffprobe"] for _, _, kwargs in calls["probes"]} == {"/opt/x/ffprobe"}
    assert {path for _, path, _ in calls["probes"]} == {video}
    assert calls["recognize"][0] == (ocr.OCR_WIDTH, 202, "zh-Hans")


def test_recognize_applies_the_configured_min_frames(tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, ["找", "", "", "我們走吧", "我們走吧"])
    dest = tmp_path / "E11.ocr.srt"

    ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig(min_frames=1))

    assert [cue.text for cue in load_srt_detailed(dest).cues] == ["找", "我們走吧"]


def test_recognize_applies_the_configured_center_tolerance(tmp_path, monkeypatch):
    """框中心在 0.65：默认容差 0.08 挡掉它（一条都没有 → 报错），放宽到 0.2 就收下。"""
    _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧"])
    off_center = [TextBox(text="我們走吧", x=0.5, y=0.2, width=0.3, height=0.15)]
    monkeypatch.setattr(ocr, "_recognize", lambda frame, **_: off_center)

    with pytest.raises(ocr.OCRError):
        ocr.recognize(_video(tmp_path), tmp_path / "E11.ocr.srt", ocr=OcrConfig())

    dest = tmp_path / "E12.ocr.srt"
    ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig(center_tolerance=0.2))
    assert [cue.text for cue in load_srt_detailed(dest).cues] == ["我們走吧"]


def test_recognize_announces_itself_before_the_first_frame(tmp_path, monkeypatch, capsys):
    """整集要跑几分钟。那句告知必须打在开工**之前**，否则几分钟静默会让人以为卡死了。"""
    calls = _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧"], duration=1430.0)
    seen_before: list[str] = []
    real_run = ocr.ffmpeg.run_raw_frames

    def spying_run(args, **kwargs):
        seen_before.append(capsys.readouterr().out)
        return real_run(args, **kwargs)

    monkeypatch.setattr(ocr.ffmpeg, "run_raw_frames", spying_run)

    ocr.recognize(_video(tmp_path), tmp_path / "E11.ocr.srt", ocr=OcrConfig())

    assert calls["recognize"]
    assert "e11.mp4 声明了硬字幕" in seen_before[0]
    # 1430 秒 / 7 倍实时 ≈ 204 秒 ≈ 3 分钟
    assert "约 3 分钟" in seen_before[0]


def test_recognize_reports_progress_every_ten_percent(tmp_path, monkeypatch, capsys):
    # 60 秒 × 23.976 / 6 ≈ 239 帧是进度的分母；假货喂满 239 帧
    texts = ["我們走吧"] * 239
    _fake_ffmpeg(monkeypatch, texts, duration=60.0)

    ocr.recognize(_video(tmp_path), tmp_path / "E11.ocr.srt", ocr=OcrConfig())

    out = capsys.readouterr().out
    marks = [f"{p}%" for p in range(10, 101, 10)]
    assert all(mark in out for mark in marks)
    assert out.count("画面字幕识别 ") == 10


def test_recognize_refuses_to_write_an_empty_srt(tmp_path, monkeypatch):
    """一条都没认出来多半是裁剪区没框住字幕；空文件会被当成合法缓存一直复用下去。"""
    _fake_ffmpeg(monkeypatch, ["", "", ""])
    dest = tmp_path / "E11.ocr.srt"

    with pytest.raises(ocr.OCRError) as excinfo:
        ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    assert "crop_top" in str(excinfo.value)
    assert not dest.exists()


def test_recognize_refuses_when_frames_and_timestamps_disagree(tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧", "我們走吧"], times=[0.0, 0.25])
    dest = tmp_path / "E11.ocr.srt"

    with pytest.raises(ocr.OCRError) as excinfo:
        ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    assert "3 帧" in str(excinfo.value)
    assert not dest.exists()


def _block_vision_imports(monkeypatch) -> None:
    real_import = builtins.__import__
    blocked = {"objc", "Quartz", "Vision", "Foundation"}

    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_recognize_without_the_extra_raises_an_actionable_error(tmp_path, monkeypatch):
    """没装 ocr extra（或者不是 macOS）时给一句能照着做的话，而不是裸 ImportError。"""
    _fake_ffmpeg(monkeypatch, ["我們走吧"])
    # _fake_ffmpeg 把 _recognize 换成了假货；这条要的恰恰是真的那个（它自己做 import）。
    monkeypatch.setattr(ocr, "_recognize", _REAL_RECOGNIZE)
    _block_vision_imports(monkeypatch)
    dest = tmp_path / "E11.ocr.srt"

    with pytest.raises(ocr.OCRUnavailableError) as excinfo:
        ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    assert "uv sync --extra ocr" in str(excinfo.value)
    assert not dest.exists()


def test_ocr_unavailable_is_an_ocr_error():
    assert issubclass(ocr.OCRUnavailableError, ocr.OCRError)


def test_ocr_errors_are_caught_by_the_cli_error_net():
    """逃出 PIPELINE_ERRORS 就意味着用户看到一整页 traceback。"""
    from tenmin.cli import PIPELINE_ERRORS

    assert issubclass(ocr.OCRError, PIPELINE_ERRORS)
    assert issubclass(ocr.OCRUnavailableError, PIPELINE_ERRORS)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg")
def test_recognize_drives_a_real_ffmpeg(tmp_path, monkeypatch):
    """用真 ffmpeg 跑一遍抽帧（lavfi 现造一段 2 秒的画面），只把 Vision 换成假货。

    锁的是假货锁不住的那几件事：滤镜链 ffmpeg 真的认、算出来的帧大小跟管道里的字节
    对得上、showinfo 的 pts 行数跟帧数一致。
    """
    video = tmp_path / "e11.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", "testsrc=size=1280x720:rate=24000/1001:duration=2",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    seen: list[tuple[int, int, int]] = []

    def fake_recognize(frame, *, width, height, language):
        seen.append((len(frame), width, height))
        return [TextBox(text="測試畫面", x=0.35, y=0.2, width=0.3, height=0.15)]

    monkeypatch.setattr(ocr, "_recognize", fake_recognize)
    dest = tmp_path / "E11.ocr.srt"

    ocr.recognize(video, dest, ocr=OcrConfig())

    # 2 秒 × 23.976 fps = 48 帧，每 6 帧取 1 帧 → 8 帧。720 × 0.28 = 201.6 → 向下取偶 200；
    # 源宽本来就是 1280，所以缩放后还是 1280×200。
    assert len(seen) == 8
    assert seen[0] == (ocr.OCR_WIDTH * 200, ocr.OCR_WIDTH, 200)
    cues = load_srt_detailed(dest).cues
    assert [cue.text for cue in cues] == ["測試畫面"]
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(8 * SAMPLE_INTERVAL, abs=1e-3)


@pytest.mark.ocr
def test_recognize_a_real_hardsub_episode(tmp_path):
    """真片源冒烟：macOS + `uv sync --extra ocr` + 一个带硬字幕的片源。

    跑法：TENMIN_OCR_SAMPLE_VIDEO=/path/to/[ANi]...[CHT].mp4 uv run pytest -m ocr
    只验「整条链路在真 Vision 上跑得通、认得出一批含汉字的 cue」，不验准确率。
    """
    sample = os.environ.get("TENMIN_OCR_SAMPLE_VIDEO")
    if not sample or not Path(sample).is_file():
        pytest.skip("设置 TENMIN_OCR_SAMPLE_VIDEO 指向一个带硬字幕的真实片源")
    dest = tmp_path / "E01.ocr.srt"

    ocr.recognize(Path(sample), dest, ocr=OcrConfig())

    cues = load_srt_detailed(dest).cues
    assert len(cues) > 50
    assert all(ocr._CJK.search(cue.text) for cue in cues)
