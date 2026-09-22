"""语音转写：把一段视频的人声轨变成一份 SRT。

只在「这一集既没有手传 SRT、视频里也没有软字幕轨」时才走到这里。产物落成一份普通
SRT 而不是直接给出 cue 对象，有两个刻意的好处：转写一集要几分钟，落盘就等于缓存，
重跑 ingest 不用重付；而且 SRT 是人能直接改的格式，转差了可以手动修，改完下次就走
「手传 SRT」那条路。

跟 srt_parser 并排：它们是同一层的两个 cue 来源，上面由 resolve 决定走哪个。
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from tenmin import atomic
from tenmin.config import DEFAULT_RENDER, AsrConfig
from tenmin.models import RawCue
from tenmin.render import ffmpeg

# 「这段文本念得出来吗」。含任意假名、汉字、拉丁字母或数字就算有内容。
# 用途是丢掉 whisper 在音频边界吐出的空段落与纯标点段落。
# 判据刻意宽松：宁可留一条噪声让下游清洗层处理，也不要误杀「OK」「2026」这种短台词。
_PRONOUNCEABLE = re.compile(r"[0-9A-Za-z\u3040-\u30ff\u4e00-\u9fff]")


class ASRError(RuntimeError):
    """转写本身失败（模型报错、或者一条对白都没转出来）。"""


class ASRUnavailableError(ASRError):
    """需要转写，但转写依赖没装。

    单独一个类型是因为处置方式不同：这不是「数据坏了」，是「环境缺东西」，消息里要
    直接给出补齐的命令。继承 ASRError 让上层想一把网住时也能网住（cli.PIPELINE_ERRORS
    只登记父类）。
    """


def _format_timestamp(seconds: float) -> str:
    """SRT 的时间戳。

    毫秒分隔符是逗号。timecode.format_timestamp 产出的是点号版本（给人读的日志和
    文档用），它的 docstring 明令不要拿它写 SRT，所以这里自己拼一份。
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_cues(segments: Iterable[Mapping[str, object]]) -> list[RawCue]:
    """模型输出的段落列表 → RawCue 列表，顺带丢掉不能用的那些。

    三条过滤规则：

    - 文本剥掉空白后为空，以及 `end <= start`。这两条是**实测**的：设计期拿真实片源
      跑过一次转写，音频末尾吐出 5 条 `[119.96 -> 119.96]` 的零长度空段落，是解码
      边界的 artifact，不是内容。
    - 文本里没有任何可发音字符（纯标点、纯音符符号）。这条是**预防性**的，那次 spike
      没撞到；whisper 系模型用 `♪` 标记音乐段是公开行为，而一条只有符号的 cue 在
      对白轨里除了占一行没有别的作用。

    刻意在这一层就丢，而不是让它们流到下游的清洗层：清洗层丢掉的东西会计进
    「跳过了几个块」，而那个数字是给人判断**片源质量**的，混进模型的 artifact 就废了。

    idx 在过滤之后重新连续编号，跟 srt_parser 的语义一致（第几个**可用**的 cue）。
    """
    cues: list[RawCue] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text or not _PRONOUNCEABLE.search(text):
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        if end <= start:
            continue
        cues.append(RawCue(idx=len(cues) + 1, start=start, end=end, text=text))
    return cues


def render_srt(cues: Sequence[RawCue]) -> str:
    """RawCue 列表 → SRT 文本。

    序号按**在这个列表里的位置**重新数，不沿用 cue.idx：这份文本会被 srt_parser 吃
    回去，而那边只把文件里写的序号存进 src_idx（可重复、可乱序），所以两者不一致
    不会错位 —— 重新数纯粹是为了让人打开文件时看到的是 1、2、3。

    产出的东西必须能被本项目自己的 srt_parser 吃回去（有测试守着这条往返）。
    """
    blocks = [
        f"{index}\n{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n{cue.text}\n"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def _run_model(**kwargs: object) -> Mapping[str, object]:
    """真正调用 mlx-whisper 的那一下。

    import 刻意写在函数体内、而不是模块顶层：转写依赖在 pyproject 的 asr extra 里，
    没装它的人（绝大多数用法）必须能正常跑 `tenmin --help` 和所有非转写阶段。放到
    顶层会让整个 ingest 包 import 不动。

    单独提成一个函数是为了让测试能换掉它而不必装几个 G 的模型。
    """
    try:
        import mlx_whisper
    except ImportError as exc:
        raise ASRUnavailableError(
            "这一集没有字幕（既没传 --srt，视频里也没有软字幕轨），需要语音转写，"
            "但转写依赖没装。跑一次 `uv sync --extra asr` 再试。"
        ) from exc
    return mlx_whisper.transcribe(**kwargs)


def transcribe(
    video: Path,
    dest: Path,
    *,
    asr: AsrConfig,
    ffmpeg_path: str = DEFAULT_RENDER.ffmpeg_path,
) -> None:
    """转写 video 的人声轨，把结果写成 dest 这份 SRT。

    同步阻塞调用，不走 to_thread：ingest 是全局阶段，跑它的时候没有别的任务在飞，
    包一层 async 只会多一层看不出好处的管道。

    可执行文件参数叫 `ffmpeg_path` 而不是 render 层那套 `ffmpeg=` —— 本模块顶层
    `ffmpeg` 这个名字已经被导入的模块占了，同名参数会把它在函数体内遮掉。

    中间那份 wav 走系统临时目录、用完就删：它只是喂模型的入参，留在 work/ 下只会
    让人以为它是个产物（一集 24 分钟的 16k 单声道 16-bit wav 约 46 MB，推算，不是实测）。
    """
    print(f"  正在转写音轨（约需数分钟）: {video.name}")
    with tempfile.TemporaryDirectory(prefix="tenmin-asr-") as workdir:
        wav = Path(workdir) / "audio.wav"
        ffmpeg.extract_audio_track(video, wav, ffmpeg=ffmpeg_path)
        result = _run_model(
            audio=str(wav),
            path_or_hf_repo=asr.model,
            language=asr.language,
            word_timestamps=True,
        )

    raw_segments = result.get("segments") or []
    if not isinstance(raw_segments, Iterable):
        raise ASRError(f"转写结果里的 segments 不是个列表: {type(raw_segments)!r}")
    cues = segments_to_cues(raw_segments)
    if not cues:
        raise ASRError(
            f"转写 {video.name} 没得到任何对白。确认这个文件有人声轨，或者手动传一份 --srt。"
        )

    # 走 atomic.write_text 而不是 `with atomic_path(dest) as staged: staged.write_text(...)`：
    # 两者语义相同，但后者形态上是个裸 `X.write_text()`，会被
    # tests/test_source_hygiene.py 的 test_artifact_writes_go_through_atomic 判成绕过原子写
    # （那条审计只按 AST 看接收者名字，认不出 staged 来自 atomic_path）。
    atomic.write_text(dest, render_srt(cues))
    print(f"  转写完成，{len(cues)} 条对白 → {dest.name}")
