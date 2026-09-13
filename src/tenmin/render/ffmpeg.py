"""ffmpeg / ffprobe 子进程封装。解析部分是纯函数，单独测。"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import IO

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
# 字体可用性只能靠 fontconfig 的 fc-list 查（ffmpeg 自己测不出来，见 font_available）。
# 它不一定装，所以那个函数的返回值是三态的。
FC_LIST = "fc-list"
STDERR_TAIL_LINES = 30
# stderr 排空线程的 join 上限。走到这个上限只可能是 ffmpeg 已经退出但线程还没收到 EOF，
# 属于「不该发生」；给个上限是为了宁可丢掉诊断信息也不把出片卡成永久挂死。
STDERR_DRAIN_TIMEOUT_SECONDS = 10.0

# --- 超时。只给「短命令」设，长时间编码刻意不设（见 run 的 docstring） ---
# ffprobe 读容器 duration 实测 0.045 秒（346MB mkv，本地 SSD）。60 秒是它的 1300 倍，
# 留给「源片在转速降下来的外置盘/网络挂载上」这类慢 I/O —— 本项目自己就跑在 /Volumes 上。
PROBE_TIMEOUT_SECONDS = 60.0
# `ffmpeg -filters` / `-encoders` 是纯进程内枚举，不碰任何媒体文件，正常是毫秒级。
CAPABILITY_TIMEOUT_SECONDS = 30.0

# ffmpeg -filters / -encoders 每行形如 " .. ass  V->V  描述"，
# 标志列只由大写字母和点组成，名字是紧跟其后的第一个 token。
# 宽度必须从 2 起：-encoders 的标志列是 6 列（"V....D"），但 -filters 只有 2 列
# （ffmpeg 9 实测 " .. subtitles"）。写死 3 起会让 -filters 一个都解析不出来，
# 于是 has_filter("subtitles") 恒为 False，preflight 谎报「没编 libass」。
_NAME_LINE = re.compile(r"^\s*[A-Z.]{2,6}\s+(\S+)\s")


class FFmpegError(RuntimeError):
    """ffmpeg 非零退出。消息里必须带 stderr 尾部，否则等于没报错。"""


def parse_names(text: str) -> set[str]:
    """从 -filters / -encoders 的输出里抽出可用名字。"""
    return {m.group(1) for m in (_NAME_LINE.match(line) for line in text.splitlines()) if m}


# `-progress` 一个块里的时间字段，按优先级排列。实测 ffmpeg 9 一个块同时发这三个，
# 而且 out_time_us 排在 out_time_ms **前面** —— 所以它们必须是回退链而不是各自
# 独立处理，否则一个块会把 on_progress 回调三次。
#
# out_time_ms 的名字带 "ms" 但单位是**微秒**（众所周知的 ffmpeg 老 bug/历史遗留），
# 跟 out_time_us 同值，所以两者共用同一个除数 1_000_000。
_MICROSECOND_KEYS = ("out_time_ms", "out_time_us")
_TIMECODE_KEY = "out_time"


def _parse_timecode(value: str) -> float | None:
    """`HH:MM:SS.ffffff` → 秒。认不出返回 None（ffmpeg 会发 N/A）。"""
    parts = value.strip().lstrip("-").split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (float(p) for p in parts)
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def progress_seconds(fields: Mapping[str, str]) -> float | None:
    """从一个 `-progress` 块里读出「已经编码到第几秒」。读不出返回 None。

    只认 out_time_ms 是个静默失效的隐患：某个 build 停发它，进度条就永远冻在 0，
    而且没有任何警告 —— 渲染看起来卡死了，其实在正常跑。所以走回退链。
    """
    for key in _MICROSECOND_KEYS:
        raw = fields.get(key)
        if raw is None:
            continue
        try:
            return int(raw) / 1_000_000
        except ValueError:
            continue  # N/A：换下一个字段，别把整条链打死
    raw = fields.get(_TIMECODE_KEY)
    if raw is None:
        return None
    return _parse_timecode(raw)


def _decode(raw: str | bytes | None) -> str:
    """TimeoutExpired.stderr 的类型随 text= 与是否真读到东西而变，统一成 str。"""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def tail(text: str, lines: int = STDERR_TAIL_LINES) -> str:
    """取末尾若干行。ffmpeg 的真实错误永远在 stderr 尾部。

    刻意**不**用 str.splitlines()：它除了 `\\n` 还在 `\\r` 上切，而 ffmpeg 的统计行
    正是 `\\r` 结尾的。实测一次 60 秒编码的 stderr 有 70 个 `\\n` 但 4 个 `\\r`，
    splitlines() 把它数成 75 行 —— 「末尾 30 行」里凭空混进 5 行进度碎片，真正的
    错误被顶出去。加了 -nostats 之后正常情况下不再有 `\\r`（实测降到 0 个），
    但显式按 `\\r` 的真实语义处理才不用依赖「上游一定记得加那个 flag」。

    `\\r` 的语义是「回到行首重写」，所以一个物理行（`\\n` 之间）的可见内容就是
    最后一个 `\\r` 之后的那段 —— 跟终端上看到的一致。
    """
    stripped = text.rstrip("\n")
    if not stripped:
        return ""
    visible: list[str] = []
    for line in stripped.split("\n"):
        collapsed = line.rpartition("\r")[2]
        if not collapsed and "\r" in line:
            # 整行都是被后续写覆盖掉的统计，没有任何可见内容，不该占掉一行配额
            continue
        visible.append(collapsed)
    return "\n".join(visible[-lines:])


def run(args: list[str], *, ffmpeg: str = FFMPEG, timeout: float | None = None) -> str:
    """跑 ffmpeg，返回 stderr（ffmpeg 的进度与日志都在 stderr）。

    source 视频的容器元数据（title/album/description 等标签）常年是老字幕组用非
    UTF-8 编码硬塞进去的脏数据，ffmpeg 会把这些原始字节原样打到 stderr 里。严格
    UTF-8 解码遇到这种输入必炸——所以这里用 errors="replace"，脏字节换成 U+FFFD，
    不让一段无关的元数据把整条渲染流水线搞挂。

    timeout 默认 None = 不设上限，这是**刻意的**：这个函数在生产里的调用方是
    mix_audio，一次几分钟的真实编码，设上限只会在慢机器上误杀一次已经跑了一半的活。
    长跑任务的中止交给用户 Ctrl-C（现在会正确杀掉子进程）。只有 preflight 里那些
    「本该毫秒级返回」的探测才传具体值进来。
    """
    argv = [ffmpeg, "-nostdin", *args]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            # ffmpeg 会抢共享 TTY 的 stdin 然后静默挂死整条流水线。-nostdin 让它自己
            # 不读，DEVNULL 从 OS 层面兜底 —— 后者才是结构性保证，不依赖 flag 支持。
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        # subprocess.run 超时会自己 kill 子进程再抛，这里只负责把它翻译成人话。
        raise FFmpegError(
            f"ffmpeg 超时（超过 {timeout} 秒）：\n{shlex.join(argv)}\n\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(_decode(error.stderr))}"
        ) from error
    if completed.returncode != 0:
        raise FFmpegError(
            f"ffmpeg 退出码 {completed.returncode}，命令：\n"
            f"{shlex.join(argv)}\n\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(completed.stderr)}"
        )
    return completed.stderr


def _probe_field(
    path: Path,
    entries: str,
    what: str,
    *,
    ffprobe: str,
    stream: str | None = None,
) -> str:
    """跑一次 ffprobe 读一个字段，返回 stdout（已 strip）。失败一律 FFmpegError。

    从 probe_duration 里抽出来的公共壳子：两处的失败处理、超时、以及那条「刻意不加
    -nostdin」的约束必须一模一样，各写一份迟早会漂。`what` 只进错误消息。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到媒体文件 {path}（读不了{what}）")
    args = [ffprobe, "-v", "error"]
    if stream is not None:
        args += ["-select_streams", stream]
    args += ["-show_entries", entries, "-of", "csv=p=0", str(path)]
    # 刻意**不**加 -nostdin：ffprobe 不认这个选项。实测 ffprobe 9.0.1 会直接报
    # `Failed to set value '-v' for option 'nostdin': Option not found` 并退出 1。
    # 它这边只能靠 stdin=DEVNULL 防抢 TTY。
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise FFmpegError(
            f"ffprobe 读 {path} 的{what}超时（超过 {PROBE_TIMEOUT_SECONDS} 秒）：\n"
            f"{shlex.join(args)}\n\n"
            f"源片是不是在一个很慢/已经掉线的盘上？\n{tail(_decode(error.stderr))}"
        ) from error
    if completed.returncode != 0:
        raise FFmpegError(
            f"ffprobe 读不出 {path} 的{what}，退出码 {completed.returncode}：\n"
            f"{tail(completed.stderr)}"
        )
    return completed.stdout.strip()


def probe_duration(path: Path, *, ffprobe: str = FFPROBE) -> float:
    """用 ffprobe 读时长（秒）。读不出、或读出来不是个正数，一律抛错。

    容器的 `format=duration` 是**容器声明**的时长，不是解码出来的样本数。对 edge-tts
    出的 mp3 来说它含编码器延迟与末尾 padding，所以比真实语音略长几十毫秒。这条**刻意
    不改**：render/tts.py 的时长体检 band 是按当前这个行为标定的（115 个真实 chunk），
    换成 `-count_frames` 之类的精确测法会让那套上下界整个失准。这里只是把这件事写明。

    两条合理性检查都是「宁可响亮失败」：
    - 文件不存在 → FileNotFoundError，说「文件不存在」而不是「读不出时长」。后者会把
      用户送去查 ffprobe / 容器格式，而真因往往是路径写错或外置盘没挂上。顺带省掉一次
      注定失败的子进程。刻意用 FileNotFoundError（OSError 子类），因为
      pipeline._source_duration 靠 catch (FFmpegError, OSError) 让「只有 SRT」这条
      合法用法降级，换成别的类型会把那条路打死。
    - 时长 <= 0 → FFmpegError。截断/空的容器会给出一个**看起来正常**的数字，timeline
      拿它去算偏移一路不报错，只会静默出一个时间轴全错的成片。
    """
    raw = _probe_field(path, "format=duration", "时长", ffprobe=ffprobe)
    try:
        duration = float(raw)
    except ValueError as error:
        raise FFmpegError(f"ffprobe 读不出 {path} 的时长，输出是 {raw!r}") from error
    if duration <= 0:
        raise FFmpegError(
            f"ffprobe 报 {path} 的时长是 {duration} 秒，这不可能是个能用的媒体文件"
            "（截断的下载？0 字节壳子？）。它会静默毒化整条时间轴，所以这里直接拦掉。"
        )
    return duration


def probe_frame_rate(path: Path, *, ffprobe: str = FFPROBE) -> float:
    """第一条视频轨的帧率（fps）。读不出、或读出来不是个正数，一律抛错。

    取 **r_frame_rate** 而不是 avg_frame_rate：前者是容器声明的「基准帧率」
    （实测真实源片 `24000/1001`），后者是「解出来的帧数 / 时长」的事后平均，
    在 VFR 或有丢帧的文件上是个不整齐的数（实测同一个文件 `1678552091/70009610`）。
    我们要的是「帧边界落在哪」，那是 r_frame_rate 的语义。

    值是**有理数字符串**，必须按分数解析：`float("24000/1001")` 直接 ValueError，
    而写成 `24000/1001 = 23.976…` 的十进制近似再去算帧号，长片尾部会累积到差一帧。

    `-select_streams v:0` 是必须的：真实源片里常有第二条「视频」轨（附图/封面，
    实测 work/saijo 的 mp4 就带一条 mjpeg 1200x800），它的 r_frame_rate 是
    `90000/1`。不选流的话 ffprobe 会把两条都打出来，取到哪一条纯看运气。

    `0/0` → 抛错：那是 ffprobe 对「这条流没有帧率」的常规回答（附图流就是它）。
    帧率 0 会让 render/timeline.py 的帧对齐把每一段都算成 0 秒，是典型的静默毒化。
    """
    raw = _probe_field(
        path, "stream=r_frame_rate", "帧率", ffprobe=ffprobe, stream="v:0"
    )
    numerator, _, denominator = raw.partition("/")
    try:
        rate = float(numerator) / float(denominator) if denominator else float(numerator)
    except (ValueError, ZeroDivisionError) as error:
        raise FFmpegError(
            f"ffprobe 读不出 {path} 的帧率，输出是 {raw!r}"
            "（`0/0` = 这条流没有帧率声明）"
        ) from error
    if rate <= 0:
        raise FFmpegError(
            f"ffprobe 报 {path} 的帧率是 {raw!r}（{rate}），这不可能是条能用的视频轨。"
            "帧率会被用来把 timeline 的每一段对齐到帧边界，0 会让每段都变成零长。"
        )
    return rate


# maxsize 从 1 提到 8：key 是可执行文件路径，而 preflight 在批量模式下每集都调。
# 写死 1 的话，只要有人在同一个进程里交替问两个 ffmpeg，缓存就退化成每次重探。
@lru_cache(maxsize=8)
def available_filters(ffmpeg: str = FFMPEG) -> frozenset[str]:
    """这个 ffmpeg 编进了哪些滤镜。**按可执行文件路径缓存**，两个 ffmpeg 不会互相冒充。"""
    return _probe_capabilities(ffmpeg, "-filters")


@lru_cache(maxsize=8)
def available_encoders(ffmpeg: str = FFMPEG) -> frozenset[str]:
    """这个 ffmpeg 编进了哪些编码器。同样按可执行文件路径缓存。"""
    return _probe_capabilities(ffmpeg, "-encoders")


def _require_binary(binary: str) -> None:
    """二进制不存在就给一句专门的人话，而不是让 subprocess 抛裸 FileNotFoundError。

    shutil.which 对裸名字走 PATH 查找、对绝对路径检查存在性与可执行位，两种形态都覆盖。
    """
    if shutil.which(binary) is None:
        raise FFmpegError(
            f"找不到可执行文件 {binary!r}。\n"
            "装一个（brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass），"
            "或用 project.yaml 的 render.ffmpeg_path / render.ffprobe_path 指定完整路径。"
        )


def _probe_capabilities(ffmpeg: str, flag: str) -> frozenset[str]:
    """问这个 ffmpeg 编了哪些滤镜/编码器。**任何一种读不到都必须抛**，不能返回空集合。

    原来这里完全不看 returncode，失败一律返回空集合。后果是错误诊断：preflight 看到
    空集合就自信地报「你的 ffmpeg 没编 libass」，而真正的原因可能是这个 ffmpeg 根本
    跑不起来（dyld 缺库）或者压根不存在 —— 用户于是去重装 libass，方向全错。

    失败走异常而不是返回空值，顺带把 lru_cache 的问题一并解决：lru_cache **不缓存
    异常**，所以一次瞬时失败不会被永久钉在进程里，只有成功的结果才进缓存。
    """
    _require_binary(ffmpeg)
    argv = [ffmpeg, "-nostdin", "-hide_banner", flag]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=CAPABILITY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise FFmpegError(
            f"探测 ffmpeg 能力超时（超过 {CAPABILITY_TIMEOUT_SECONDS} 秒）："
            f"\n{shlex.join(argv)}\n\n"
            f"{flag} 只是进程内枚举，正常是毫秒级 —— 这个 ffmpeg 大概率有问题。"
        ) from error
    if completed.returncode != 0:
        raise FFmpegError(
            f"探测 ffmpeg 能力失败（退出码 {completed.returncode}）：\n"
            f"{shlex.join(argv)}\n\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(completed.stderr)}"
        )
    names = parse_names(completed.stdout)
    if not names:
        # 退出码 0 却一个名字都没解析出来 = 结论不可信，绝不能当成「什么都没有」。
        # 历史事故：_NAME_LINE 的标志列宽度写死成 3 起，-filters 一个都没解析出来，
        # has_filter("subtitles") 恒为 False，preflight 谎报没编 libass 整整一个版本。
        raise FFmpegError(
            f"{shlex.join(argv)} 退出码 0，但一个名字都没解析出来。\n"
            f"输出末尾 {STDERR_TAIL_LINES} 行：\n{tail(completed.stdout)}"
        )
    return frozenset(names)


def has_filter(name: str, *, ffmpeg: str = FFMPEG) -> bool:
    return name in available_filters(ffmpeg)


def has_encoder(name: str, *, ffmpeg: str = FFMPEG) -> bool:
    return name in available_encoders(ffmpeg)


def has_audio_stream(path: Path, *, ffprobe: str = FFPROBE) -> bool:
    """源片里有没有音轨。

    render/audio.py 的 filtergraph 用 `[0:a]`，一个视频-only 的源（remux 出来的、
    或者只拿了视频轨的下载）会让混音直接失败 —— 而那是在跑完几分钟 TTS **之后**。

    实测：`-select_streams a` 在没有音轨时退出码仍然是 **0**、stdout 是空的。所以判据
    只能看 stdout 有没有内容，看 returncode 会永远判成「有音轨」。
    """
    path = Path(path)
    args = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    return bool(completed.stdout.strip())


def font_available(name: str) -> bool | None:
    """fontconfig 认不认这个字体家族。**None = 查不了**（不是「不可用」）。

    为什么只能靠 fc-list，不能靠 ffmpeg 自己试一遍：实测 ffmpeg 9.0.1 拿一个根本不存在
    的 font family 跑 drawtext 依然**退出码 0** —— fontconfig 静默替换成 PingFang 并
    正常渲染完。也就是说 ffmpeg 这条路在原理上就测不出「字体没装」。

    `fc-list "<family>" family` 命中时打出家族名（含本地化别名），没命中时 stdout 是空的，
    两种情况退出码都是 0，所以同样只能看 stdout。fc-list 不一定装，那时返回 None，让调用方
    如实说「跳过了这项检查」而不是谎报可用。
    """
    if shutil.which(FC_LIST) is None:
        return None
    try:
        completed = subprocess.run(
            [FC_LIST, name, "family"],
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=CAPABILITY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        # 查不了就说查不了。字体只影响外观，绝不值得为它把渲染打死。
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _warn_once(warnings: list[str] | None, message: str) -> None:
    """批量模式下 preflight 每集跑一次，而字体是全项目共享的，别刷 10 遍同一句。"""
    if warnings is None or message in warnings:
        return
    warnings.append(message)


def preflight(
    video: Path,
    video_encoder: str,
    *,
    ffmpeg: str = FFMPEG,
    ffprobe: str = FFPROBE,
    needs_drawtext: bool = False,
    font_names: Sequence[str] = (),
    warnings: list[str] | None = None,
) -> float:
    """开跑前一次性检查，返回源片时长。任一**致命**项不满足立刻抛错。

    渲染动辄几分钟，而 voice 阶段之前还有一轮 LLM，绝不能跑完 TTS 才在最后一步炸掉。
    所以这里要覆盖真实渲染会用到的**全部**外部依赖，尤其是最后一公里那几样：

    - subtitles 滤镜（libass）：烧字幕，必需。
    - 视频编码器：必需。
    - drawtext 滤镜（+libfreetype/fontconfig）：**只在片尾黑卡开着时**才用到
      （render/video.py 的 outro 分支），所以由 needs_drawtext 控制，关掉卡片的用户
      不该被它挡住。
    - 源片有音轨：render/audio.py 用 `[0:a]`，视频-only 的源会在跑完 TTS + 混音之后
      才炸 —— 正是 preflight 最该拦的那一类。
    - 字体：只发 warning，见下。

    抛的一律是 FFmpegError 而不是裸 RuntimeError：裸 RuntimeError **不在**
    cli.PIPELINE_ERRORS 里，于是「你的 ffmpeg 没编 libass」这句本来写得很清楚的人话，
    用户实际看到的是一整页 traceback。FFmpegError 是 RuntimeError 子类且已在表里。

    字体为什么是 warning 而不是 error：实测 fontconfig 会**静默替换**成别的字体并正常
    渲染完（ffmpeg 9.0.1 拿不存在的 family 跑 drawtext 退出码 0）。所以字体不对的后果
    是「字体长得不一样」这种纯外观问题，不是失败；为它拦住整次渲染是本末倒置。而且检测
    本身依赖 fc-list（不一定装），查不了的时候如实说查不了 —— 不假装检查过。
    """
    if not has_filter("subtitles", ffmpeg=ffmpeg):
        raise FFmpegError(
            "你的 ffmpeg 没编 libass，subtitles 滤镜不可用，烧不了字幕。\n"
            "请重装：brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass"
        )
    if not has_encoder(video_encoder, ffmpeg=ffmpeg):
        raise FFmpegError(
            f"你的 ffmpeg 没有编码器 {video_encoder}，请改 project.yaml 的 render.video_encoder，"
            "或重装 ffmpeg"
        )
    if needs_drawtext and not has_filter("drawtext", ffmpeg=ffmpeg):
        raise FFmpegError(
            "你的 ffmpeg 没编 drawtext 滤镜（需要 libfreetype + fontconfig），"
            "画不了片尾黑卡。\n"
            "请重装 ffmpeg，或把 project.yaml 的 render.outro_card_seconds 设成 0 关掉卡片。"
        )
    if not Path(video).is_file():
        raise FileNotFoundError(f"找不到源视频 {video}，请检查 project.yaml 的 episodes[].video")
    if not has_audio_stream(Path(video), ffprobe=ffprobe):
        raise FFmpegError(
            f"源视频 {video} 里没有音轨，混音这一步（原声 ducking）没法做。\n"
            "这个源大概率是只拿了视频轨的 remux，请换一个带音轨的源片。"
        )
    for name in font_names:
        state = font_available(name)
        if state is True:
            continue
        if state is None:
            _warn_once(
                warnings,
                f"没装 fc-list，跳过了字体可用性检查（配的是 {name!r}）。"
                "字体真的缺了的话 fontconfig 会静默换一个，成片字体会跟预期不一样。",
            )
        else:
            _warn_once(
                warnings,
                f"fontconfig 里找不到字体 {name!r}，fontconfig 会静默替换成别的字体，"
                "成片的字幕/片尾卡字体会跟预期不一样（不影响出片）。",
            )
    return probe_duration(Path(video), ffprobe=ffprobe)


def _drain(stream: IO[str], sink: list[str]) -> None:
    """把一条管道读到 EOF。给 run_with_progress 的 stderr 排空线程用。

    读到一半管道被关掉（异常路径上 Popen.__exit__ 会关）会抛 ValueError / OSError，
    那时我们已经不要这份 stderr 了，静默收工就行 —— 让线程里冒异常只会往 stderr 打一段
    与真正错因无关的 traceback，把用户的注意力引错方向。
    """
    try:
        sink.append(stream.read())
    except (ValueError, OSError):  # pragma: no cover - 只在异常清理路径上走到
        pass


def run_with_progress(
    args: list[str],
    *,
    total_seconds: float,
    on_progress: Callable[[float], None] | None = None,
    ffmpeg: str = FFMPEG,
) -> str:
    """跟 run() 一样跑 ffmpeg，但额外加 -progress pipe:1，流式解析进度，
    每读完一个 progress 块就换算成 0.0~1.0 的比例回调 on_progress。

    时间字段的回退链见 progress_seconds。按「块」而不是按「行」回调，是因为一个块里
    out_time_us / out_time_ms / out_time 三个字段都会发，逐行处理会把回调打三遍。

    stderr 必须**并发**排空，不能等 stdout 读到 EOF 再读（原来就是这么写的）：管道容量
    只有 64KB，实测一次 60 秒编码就产生 6520 字节 stderr、240 秒成片约 26KB，一段
    per-frame warning（`Past duration ... too large`、HEVC 解码器抱怨）就能突破上限
    —— 然后 ffmpeg 阻塞在写 stderr、Python 阻塞在读 stdout，永久挂死且没有任何输出。
    -nostats 只是把噪音（实测 506 字节 + 4 个搅乱 tail 的 `\\r`）拿掉，**不能**当成
    修复：真正兜住这件事的是那个排空线程。
    """
    # -progress 是全局选项，放到 -i 之前才是它该在的位置（原来追加在输出文件名之后，
    # 碰巧能用而已）。argv 只拼一次，Popen 与出错消息共用同一个变量：原来出错分支自己
    # 重建了一遍字符串、且漏了 -progress pipe:1，报出来的命令不是真正跑的那条。
    argv = [ffmpeg, "-nostdin", "-nostats", "-progress", "pipe:1", *args]

    def report(fields: dict[str, str]) -> None:
        seconds = progress_seconds(fields)
        if seconds is None or on_progress is None or total_seconds <= 0:
            return
        on_progress(max(0.0, min(1.0, seconds / total_seconds)))

    # with + 异常路径上显式 kill：原来 Popen 既没进 with 也没 try/finally，on_progress
    # 一抛（reporter/rich 出错）或用户 Ctrl-C，ffmpeg 就变成孤儿继续烧 CPU、管道泄漏。
    # kill 必须在 __exit__ 之前：__exit__ 先关管道再 wait()，对一个还在跑的几分钟编码
    # 来说那个 wait() 自己就是一次挂死。
    with subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    ) as process:
        try:
            drained: list[str] = []
            drainer = threading.Thread(
                target=_drain, args=(process.stderr, drained), daemon=True
            )
            drainer.start()

            block: dict[str, str] = {}
            for line in process.stdout:
                line = line.strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                block[key] = value
                if key != "progress":
                    continue
                # `progress=continue` / `progress=end` 是块的结束标记
                report(block)
                if value == "end" and on_progress is not None:
                    on_progress(1.0)
                block.clear()
            # 没有以 progress= 收尾的残块也要报一次：ffmpeg 被 kill / 提前断流时最后
            # 那个块是不完整的，丢掉它等于把「实际跑到哪」这条信息扔了。
            report(block)

            returncode = process.wait()
            # stdout 已经 EOF、进程已经退出，stderr 必然也到 EOF，这个 join 立刻返回。
            # 仍然给上限：宁可丢掉诊断信息，也不要把「出片」卡成永久挂死。
            drainer.join(STDERR_DRAIN_TIMEOUT_SECONDS)
            stderr = "".join(drained)
        except BaseException:
            process.kill()
            raise
    if returncode != 0:
        raise FFmpegError(
            f"ffmpeg 执行失败（退出码 {returncode}）：{shlex.join(argv)}\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(stderr)}"
        )
    return stderr
