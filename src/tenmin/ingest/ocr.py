"""硬字幕 OCR：把烧在画面底部的字幕认出来，落成一份 SRT。

只在「这部番声明了硬字幕（ocr.enabled / episodes[].hardsub），而这一集既没有手传 SRT、
视频里也没有软字幕轨」时才走到这里。跟 asr.py 并排、结构仿照它：产物落成普通 SRT
（`srt/E{NN}.ocr.srt`），落盘即缓存、人能直接手改，手改后照样按 `kind="ocr"` 复用
（判据见 resolve._is_usable_cache）。

管线分三段，前两段是纯函数：

1. 抽帧：ffmpeg 每 step 帧取 1 帧、裁出画面底部、缩到固定宽度的灰度图，rawvideo 走管道，
   每帧的时间戳读 showinfo 打在 stderr 上的真实 pts。
2. 单帧清洗（frame_text）：只留水平居中、含汉字的文字框，从上到下拼起来，统一省略号。
3. 合并成 cue（merge_frames）：相邻帧相似就归入同一句，多帧投票定文本。

默认参数与这些规则全部出自一次实测，数据见 docs/superpowers/specs/2026-10-01-hardsub-ocr-design.md
与 config.OcrConfig 的注释。
"""

from __future__ import annotations

import difflib
import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tenmin import atomic
from tenmin.config import DEFAULT_RENDER, OcrConfig
from tenmin.ingest.asr import render_srt
from tenmin.models import RawCue
from tenmin.render import ffmpeg

# 交给 Vision 的图固定这么宽，高度按裁剪区的比例取偶数。720p 与 4K 片源交给 Vision 的
# 是同一个尺度的字；实测用的就是这个宽度（1080p 片源缩到 1280 宽）。
OCR_WIDTH = 1280

# 同一句字幕中间允许夹几个「没认出字」的空帧。1 = 容忍单帧漏认；再多就会把两句之间
# 正常的空档也吞掉。
_MAX_GAP_FRAMES = 1

# 认作「有字幕」的必要条件：至少含一个汉字（CJK 基本区 + 扩展 A）。实测的噪声帧
# （`/1/7`、`MIMM`）一个汉字都没有；含汉字的那类（`7000\n找`）靠 min_frames 清掉。
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

# Vision 对 `…` 的几种认法：`•••`、`⋯`、`...`、`・・・`、句尾一个 `.` 或 `。`。不先统一，
# 「不要…」这种短句会因为写法不同在相似度上被拆成好几条（实测拆成 5 条）。
#
# 第一支：一串点号类字符里只要含 `⋯` / `…` 就整串换成一个 `…`（顺带把 `……` 归一成 `…`，
# 投票才不会在两种写法之间分票）。第二支：没有 `⋯` / `…` 时，连续两个以上才算。单个 `·`
# 不动 —— 它在繁中字幕里是外文人名的分隔符（`伊莉莎白·克洛`）。
_ELLIPSIS_RUN = re.compile(r"[•·・.。⋯…]*[⋯…][•·・.。⋯…]*|[•·・.。]{2,}")
# 行尾单个 `.` / `。`。前置验证里 Vision 会把 `…` 认成单个 `.` 或句尾的 `。`，而繁中字幕
# 的行尾通常不打句号，所以按省略号处理。
_TRAILING_DOT = re.compile(r"[.。]$", re.MULTILINE)

# showinfo 给每个输出帧打的那一行：`n:   3 pts:     18 pts_time:0.75075 duration: ...`。
# 同一个滤镜实例还会打 `config in time_base` 与 `color_range` 这类行，它们不含这个形状。
_SHOWINFO_FRAME = re.compile(r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:(\S+)")

# 开工前那句耗时估算用的倍率：实测整集墙钟 201 秒 / 片长 1429.99 秒，约 7 倍实时。
_REALTIME_FACTOR = 7.0
# 进度每完成这么多分之一打一行。
_PROGRESS_STEPS = 10


class OCRError(RuntimeError):
    """识别本身失败（帧与时间戳对不上、一条字幕都没认出来、Vision 报错）。"""


class OCRUnavailableError(OCRError):
    """需要 OCR，但 OCR 依赖没装（或者不是 macOS）。

    单独一个类型的理由同 asr.ASRUnavailableError：这是「环境缺东西」，消息里要直接给出
    补齐的命令。继承 OCRError，cli.PIPELINE_ERRORS 只登记父类。
    """


@dataclass(frozen=True)
class TextBox:
    """Vision 认出的一行字。坐标按图的宽高归一化到 0–1，**y 轴从下往上**（Vision 的约定）。"""

    text: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class FrameText:
    """一个采样帧：真实时间戳（秒）与清洗后的文本。空串 = 这一帧没有字幕。"""

    time: float
    text: str


def sample_step(src_fps: float, sample_fps: float) -> int:
    """每隔几帧取一帧。`max(1, round(源帧率 / 目标密度))`。

    取整而不是按秒抽：每次取到的都是一帧真实存在的帧，不会有插值或重复帧。23.976 / 24
    fps 得 6，30 得 8，60 得 15，120 得 30；源帧率比目标还低时每帧都取。
    """
    return max(1, round(src_fps / sample_fps))


def frame_filter(*, width: int, height: int, step: int, crop_top: float) -> tuple[str, int]:
    """抽帧滤镜链，以及缩放后一帧的高度（像素）。

    按源片的真实像素算出确定的整数，而不是写 `crop=iw:ih*0.28:...,scale=1280:-2` 让
    ffmpeg 自己算：rawvideo 管道里一帧有多少字节必须在 ffmpeg 开跑之前就知道，而 ffmpeg
    对表达式的取整（crop 按色度采样向下取偶、scale -2 的四舍五入）是它的实现细节。
    裁剪高度向下取偶数，起点 = 画面高度 - 裁剪高度（一直裁到底）；缩放后高度按比例取偶数。

    `select` 按帧序号取帧，所以可变帧率片源上每秒取到的帧数会随源帧率波动 —— 投票和归并
    只看帧的先后，不受影响。末尾的 showinfo 为每个输出帧在 stderr 打一行 pts_time。
    """
    crop_height = max(2, int(height * (1 - crop_top)) // 2 * 2)
    crop_y = height - crop_height
    out_height = max(2, round(OCR_WIDTH * crop_height / width / 2) * 2)
    chain = (
        f"select='not(mod(n\\,{step}))',"
        f"crop={width}:{crop_height}:0:{crop_y},"
        f"scale={OCR_WIDTH}:{out_height},"
        "format=gray,"
        "showinfo"
    )
    return chain, out_height


def parse_showinfo_pts(stderr: str) -> list[float]:
    """按出现顺序取出 showinfo 每一帧的 pts_time（秒）。

    读真实 pts 而不是按 `序号 × step / 源帧率` 推算：那个公式只对恒定帧率成立，可变帧率
    片源上时间轴会越跑越偏。读不出数（比如 `NOPTS`）就抛，不猜。
    """
    times: list[float] = []
    for match in _SHOWINFO_FRAME.finditer(stderr):
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError as error:
            raise OCRError(f"showinfo 给出的帧时间戳读不出来：{raw!r}") from error
        if not math.isfinite(value):
            raise OCRError(f"showinfo 给出的帧时间戳不是有限数：{raw!r}")
        times.append(value)
    return times


def normalize_ellipsis(text: str) -> str:
    """把 Vision 对省略号的各种认法统一成一个 `…`。规则见 _ELLIPSIS_RUN 的注释。"""
    return _TRAILING_DOT.sub("…", _ELLIPSIS_RUN.sub("…", text))


def frame_text(observations: Sequence[TextBox], *, center_tolerance: float) -> str:
    """一帧的文字框 → 这一帧的字幕文本。空串表示这一帧没有字幕。

    1. 只留文字框中心离水平中线不到 center_tolerance 的行（去掉左右两侧的 staff 字）。
    2. 去掉不含汉字的行。
    3. 从上到下排序后用换行拼接。Vision 的 y 轴从下往上，所以按框中心的 y **降序**排。
    4. 统一省略号，去掉首尾空白。
    """
    kept = [
        box
        for box in observations
        if abs(box.x + box.width / 2 - 0.5) < center_tolerance and _CJK.search(box.text)
    ]
    kept.sort(key=lambda box: box.y + box.height / 2, reverse=True)
    joined = "\n".join(box.text.strip() for box in kept)
    return normalize_ellipsis(joined).strip()


def _finalize(frames: Sequence[FrameText], *, idx: int, interval: float) -> RawCue:
    """一组同一句的帧 → 一条 cue。文本取众数，平票取最早出现的那个（结果确定）。"""
    texts = [frame.text for frame in frames]
    counts = Counter(texts)
    best = max(counts.values())
    text = next(candidate for candidate in texts if counts[candidate] == best)
    return RawCue(idx=idx, start=frames[0].time, end=frames[-1].time + interval, text=text)


def merge_frames(
    frames: Sequence[FrameText], *, interval: float, similarity: float, min_frames: int
) -> list[RawCue]:
    """按时间顺序把采样帧归并成 cue。

    - 当前帧跟当前 cue 最后一个**非空帧**的 `SequenceMatcher.ratio()` 达到 similarity 就
      归入同一句；中间最多允许夹 _MAX_GAP_FRAMES 个空帧（容忍单帧漏认）。
    - 不相似，或者连续空帧超过上限，当前 cue 结束。
    - 定稿：文本取多帧众数（修掉单帧错字与前缀杂字），起点 = 首帧时间，终点 = 末帧时间 +
      interval（一个采样间隔）；非空帧少于 min_frames 的丢弃（实测只出现 1 帧的都是噪声）。
    - 去重叠：上一条的终点晚于下一条的起点时截到下一条的起点（实测有毫秒级重叠）。
    """
    groups: list[list[FrameText]] = []
    current: list[FrameText] = []
    gap = 0

    def close() -> None:
        if len(current) >= min_frames:
            groups.append(list(current))
        current.clear()

    for frame in frames:
        if not frame.text:
            if current:
                gap += 1
                if gap > _MAX_GAP_FRAMES:
                    close()
            continue
        if current and (
            difflib.SequenceMatcher(None, current[-1].text, frame.text).ratio() >= similarity
        ):
            current.append(frame)
        else:
            close()
            current.append(frame)
        gap = 0
    close()

    cues = [
        _finalize(group, idx=index, interval=interval)
        for index, group in enumerate(groups, start=1)
    ]
    for previous, following in itertools.pairwise(cues):
        if previous.end > following.start:
            previous.end = following.start
    return cues


def _recognize(frame: bytes, *, width: int, height: int, language: str) -> list[TextBox]:
    """真正调用 Apple Vision 的那一下：一帧 8 位灰度 rawvideo → 认出来的文字框。

    import 刻意写在函数体内：pyobjc 在 pyproject 的 ocr extra 里、只装得上 macOS，没装它
    的人必须能正常跑 `tenmin --help` 和所有不碰 OCR 的用法。单独提成一个函数是为了让
    测试能换掉它。

    参数：accurate 识别级别、zh-Hant、开语言校正 —— 实测 40 条抽样 39 条逐字正确就是这组。
    Vision 的 confidence 只有 0.3 / 0.5 / 1.0 几档，太粗，不取。每帧包一层 autorelease
    pool：这是一个没有 run loop 的长循环，不包的话 Objective-C 的临时对象要到进程结束才释放。
    """
    try:
        import objc
        import Quartz
        import Vision
        from Foundation import NSData
    except ImportError as exc:
        raise OCRUnavailableError(
            "这一集声明了硬字幕（ocr.enabled 或 episodes[].hardsub），需要识别画面字幕，"
            "但 OCR 依赖没装（只支持 macOS）。跑一次 `uv sync --extra ocr` 再试。"
        ) from exc

    with objc.autorelease_pool():
        data = NSData.dataWithBytes_length_(frame, len(frame))
        provider = Quartz.CGDataProviderCreateWithCFData(data)
        image = Quartz.CGImageCreate(
            width,
            height,
            8,
            8,
            width,
            Quartz.CGColorSpaceCreateDeviceGray(),
            Quartz.kCGImageAlphaNone,
            provider,
            None,
            False,
            Quartz.kCGRenderingIntentDefault,
        )
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_([language])
        request.setUsesLanguageCorrection_(True)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise OCRError(f"Vision 识别失败：{error}")
        boxes: list[TextBox] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            rect = observation.boundingBox()
            boxes.append(
                TextBox(
                    text=str(candidates[0].string()),
                    x=float(rect.origin.x),
                    y=float(rect.origin.y),
                    width=float(rect.size.width),
                    height=float(rect.size.height),
                )
            )
        return boxes


def _progress_printer(total: int) -> Callable[[int], None]:
    """返回一个「第 done 帧做完了」的回调，每跨过一个 10% 打一行。total 不可靠时不打。"""
    printed = {"step": 0}

    def report(done: int) -> None:
        if total <= 0:
            return
        step = min(_PROGRESS_STEPS, done * _PROGRESS_STEPS // total)
        if step > printed["step"]:
            printed["step"] = step
            print(f"  画面字幕识别 {step * 100 // _PROGRESS_STEPS}%（{done}/{total} 帧）")

    return report


def recognize(
    video: Path,
    dest: Path,
    *,
    ocr: OcrConfig,
    ffmpeg_path: str = DEFAULT_RENDER.ffmpeg_path,
    ffprobe_path: str = DEFAULT_RENDER.ffprobe_path,
) -> None:
    """识别 video 画面底部的硬字幕，把结果写成 dest 这份 SRT（繁体原文）。

    同步阻塞调用，理由同 asr.transcribe：ingest 是全局阶段，跑它时没有别的任务在飞。
    可执行文件参数叫 `ffmpeg_path` / `ffprobe_path`，理由同 asr.transcribe（模块顶层的
    `ffmpeg` 名字已被导入的模块占了）。

    一条字幕都没认出来时抛 OCRError、不写出空文件：那多半是裁剪区没框住字幕
    （ocr.crop_top 不对），空文件会被当成一份「这一集没有对白」的合法缓存复用下去。
    """
    src_fps = ffmpeg.probe_frame_rate(video, ffprobe=ffprobe_path)
    duration = ffmpeg.probe_duration(video, ffprobe=ffprobe_path)
    width, height = ffmpeg.probe_video_size(video, ffprobe=ffprobe_path)
    step = sample_step(src_fps, ocr.sample_fps)
    interval = step / src_fps
    chain, out_height = frame_filter(width=width, height=height, step=step, crop_top=ocr.crop_top)

    minutes = max(1, round(duration / _REALTIME_FACTOR / 60))
    print(f"  {video.name} 声明了硬字幕，开始识别画面字幕（约 {minutes} 分钟）")

    texts: list[str] = []
    report = _progress_printer(math.floor(duration * src_fps / step))

    def on_frame(frame: bytes) -> None:
        boxes = _recognize(frame, width=OCR_WIDTH, height=out_height, language=ocr.language)
        texts.append(frame_text(boxes, center_tolerance=ocr.center_tolerance))
        report(len(texts))

    stderr = ffmpeg.run_raw_frames(
        [
            "-hide_banner",
            "-v",
            "info",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vf",
            chain,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        frame_size=OCR_WIDTH * out_height,
        on_frame=on_frame,
        ffmpeg=ffmpeg_path,
    )

    times = parse_showinfo_pts(stderr)
    if len(times) != len(texts):
        raise OCRError(
            f"{video.name} 抽出 {len(texts)} 帧，showinfo 却报了 {len(times)} 个时间戳，"
            "帧与时间对不上，不猜。"
        )
    cues = merge_frames(
        [FrameText(time=t, text=x) for t, x in zip(times, texts, strict=True)],
        interval=interval,
        similarity=ocr.similarity,
        min_frames=ocr.min_frames,
    )
    if not cues:
        raise OCRError(
            f"{video.name} 的画面上一条字幕都没认出来。检查 ocr.crop_top（现在是 "
            f"{ocr.crop_top}，裁剪区要框住字幕所在的画面底部），或者这个片源其实没有硬字幕。"
        )

    # 走 atomic.write_text，理由同 asr.transcribe（源码卫生审计认不出 atomic_path 的暂存名）。
    atomic.write_text(dest, render_srt(cues))
    print(f"  画面字幕识别完成，{len(cues)} 条 → {dest.name}")
