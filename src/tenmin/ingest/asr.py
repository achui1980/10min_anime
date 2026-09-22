"""语音转写：把一段视频的人声轨变成一份 SRT。

只在「这一集既没有手传 SRT、视频里也没有软字幕轨」时才走到这里。产物落成一份普通
SRT 而不是直接给出 cue 对象，有两个刻意的好处：转写一集要几分钟，落盘就等于缓存，
重跑 ingest 不用重付；而且 SRT 是人能直接改的格式，转差了可以手动修。手改**不会**把
这一集翻成「手传 SRT」那条路（那条只看 project.yaml 的 `episodes[].srt`），仍然按
`kind="asr"` 复用 —— 判据与理由见 resolve._is_usable_asr_cache 的 docstring。

跟 srt_parser 并排：它们是同一层的两个 cue 来源，上面由 resolve 决定走哪个。
"""

from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from tenmin import atomic
from tenmin.config import DEFAULT_RENDER, AsrConfig
from tenmin.models import RawCue
from tenmin.render import ffmpeg
from tenmin.timecode import format_timestamp

# 「这段文本念得出来吗」。含**全角**假名、汉字（CJK 基本区）、半角拉丁字母或数字就算有内容。
# 用途是丢掉 whisper 在音频边界吐出的空段落与纯标点段落。
# 判据刻意宽松：宁可留一条噪声让下游清洗层处理，也不要误杀「OK」「2026」这种短台词。
#
# 范围刻意**不**扩到半角片假名、全角字母数字、叠字符 `々`、CJK 扩展区 —— 实测这四类
# （`ﾊｲ` / `ＯＫ` / `々` / `㐀`）现在都会被判成「念不出来」而丢掉。不扩的理由是
# whisper 的日语输出用全角假名 + 半角数字（设计文档里那两行 spike 输出就是这个形态），
# 每扩一段范围都是一条没有素材支撑的放行规则；真撞到了再按实物扩。
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

    毫秒分隔符是逗号，而共用的 timecode.format_timestamp 产出的是点号版本（它的
    docstring 明令不要拿它写 SRT）。但两者的差别只在**一个字符**上，所以这里不重抄一份
    实现、只换分隔符。重抄的代价实测过，是三件事：

    - 漏掉 nan/inf 守卫。手抄那份的 `int(round(x))` 对 nan 抛
      `ValueError: cannot convert float NaN to integer`（看不出坏的是哪个数据），对 inf
      抛 `OverflowError` —— 而 OverflowError **不在** cli.PIPELINE_ERRORS 里，整页
      traceback。共用那份先过 `_finite`，统一成一句「秒数必须是有限数，收到 ...」。
    - 漏掉负数夹紧。这一条不报错，而是**静默错时间**：实测把 `-1:59:59,000` 喂给
      srt_parser 拼出的 `-->` 行正则（它用 search 不是 match），它匹配到的是
      `1:59:59,000`，于是一个负时间变成了 1 小时 59 分。
    - `int(round(...))` 手抄成 `int(...)`，毫秒位少一。

    隐式依赖「format_timestamp 的输出里只有一个点」（今天是 `HH:MM:SS.mmm`）。这条前提
    由 test_asr.py 的 test_format_timestamp_only_swaps_the_one_separator 钉住。
    """
    return format_timestamp(seconds).replace(".", ",")


def segments_to_cues(segments: Iterable[Mapping[str, object]]) -> list[RawCue]:
    """模型输出的段落列表 → RawCue 列表，顺带丢掉不能用的那些。

    过滤规则按「凭什么在这里」分三组：

    - **实测**的：文本剥掉空白后为空，以及 `end <= start`。设计期拿真实片源跑过一次
      转写，音频末尾吐出 5 条 `[119.96 -> 119.96]` 的零长度空段落，是解码边界的
      artifact，不是内容。
    - **预防性**的：文本里没有任何可发音字符（纯标点、纯音符符号）。那次 spike 没撞到；
      whisper 系模型用 `♪` 标记音乐段是公开行为，而一条只有符号的 cue 在对白轨里除了
      占一行没有别的作用。注意它在**行为上完全盖住**了上面那条空文本判据（空串里当然
      没有可发音字符），所以 `not text` 那半句删掉也全绿 —— 留着是因为它才是实测踩到的
      那条，读代码的人应该在这里直接看见它。
    - **推理**出来的：时间非有限。必须在这一层挡，因为 `nan <= nan` 是 `False`（实测），
      所以一个 nan 段落能溜过零长度那条判据，然后在 render_srt 里变成一个 ValueError ——
      报错点离真因隔了两层。

    三者都是**静默**丢弃，跟这个函数已有的角色一致（纯函数拿不到 warnings 列表，而本仓
    刻意不引入 logging）。代价是一个吐出全 nan 时间戳的模型最后只会得到一句「没得到任何
    对白」，得自己去想那是片源问题还是模型问题。

    刻意在这一层就丢，而不是让它们流到下游的清洗层：清洗层丢掉的东西会计进
    「跳过了几个块」，而那个数字是给人判断**片源质量**的，混进模型的 artifact 就废了。

    段落**不是 Mapping** 则 raise 而不是 continue：这个守卫的存在理由就是「模型换了返回
    形状要响亮」，静默跳过等于把守卫拆了（而且没有它，一个 `[1, 2]` 会漏成裸
    AttributeError，那个类型不在 cli.PIPELINE_ERRORS 里）。

    idx 在过滤之后重新连续编号（第几个**留下来**的段落）。它实际上没有任何消费者 ——
    render_srt 明确不用它（那边按列表位置重数），下游拿到的 cue 是 srt_parser 重新解析
    这份 SRT 得到的；留着只因为 RawCue 要求这个字段。刻意**不**跟 srt_parser 对坏时间的
    处置对齐：那边遇到 `end < start` 是夹紧并保留（计进 clamped_cues，因为坏字幕里那是
    人打错的时间、台词本身是真的），这里是丢弃（模型吐的零长度段落连文本都是空的，
    没有可救的内容）。
    """
    cues: list[RawCue] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ASRError(f"转写结果里有个 segment 不是 dict: {type(segment)!r}")
        text = str(segment.get("text", "")).strip()
        if not text or not _PRONOUNCEABLE.search(text):
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        cues.append(RawCue(idx=len(cues) + 1, start=start, end=end, text=text))
    return cues


def render_srt(cues: Sequence[RawCue]) -> str:
    """RawCue 列表 → SRT 文本。

    序号按**在这个列表里的位置**重新数，不沿用 cue.idx：这份文本会被 srt_parser 吃
    回去，而那边只把文件里写的序号存进 src_idx（可重复、可乱序），所以两者不一致
    不会错位 —— 重新数纯粹是为了让人打开文件时看到的是 1、2、3。

    产出的东西必须能被本项目自己的 srt_parser 吃回去（有测试守着这条往返），但那条往返
    只对**不含空行的 text** 成立：SRT 用空行分块，所以 `"はい\\n\\nいいえ"` 会被解析成
    「一条 cue + 一个 skipped_block」，而 skipped_blocks 正是报给用户的片源质量指标。
    `segments_to_cues` 的 `.strip()` 不动内部换行，所以理论上能流过来 —— 只是 whisper
    不产带空行的 segment，所以按最便宜的方式收：缩小声明范围，不加一层转义。
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
            # 这是唯一一个会让一集多花几十秒的开关，所以值得解释为什么开着：它跑一遍
            # DTW 词级对齐，并**据此微调 segment 的 start/end**（外加标点的前/后归并）。
            # 也就是说它不是「我们不消费的附加输出」—— segment 边界恰好是这一层唯一消费
            # 的两个数。关掉能省时间，但换来的是另一套没人标定过的时间轴。
            # 性能上它也不是相对实测基线的回退：设计期那次 spike（120 秒音频 15 秒、
            # 约 8 倍实时）就是带着它跑的。
            word_timestamps=True,
        )

    # 「压根没有 segments 这个键」跟「转写出来是空的」刻意分成两条消息：前者要去看
    # mlx-whisper 的 API 变了什么，后者要去看片源有没有人声轨。并成一句（原来的
    # `result.get("segments") or []`）等于把形状变化伪装成片源问题。
    if "segments" not in result:
        # `sorted(map(str, ...))` 而不是裸 `sorted(result)`：这条守卫存在的理由就是
        # 「模型返回的形状变了」，而 result 不是 dict 时它自己会炸。实测裸版本拿
        # `[{"a": 1}, {"b": 2}]`（最像的那种形状变化：直接返回 segment 列表）抛
        # `TypeError: '<' not supported between instances of 'dict' and 'dict'`，
        # 而 TypeError **不在** cli.PIPELINE_ERRORS 里 —— 报错语句自己漏出一整页
        # traceback，比它要报的那件事更难查。先 str() 再排序对任何可迭代对象都成立。
        # （注意不是「凡 list 必炸」：`[1, 2]` 排得动。炸的是元素之间不可比的那些，
        # 也就是真实场景里的绝大多数。）
        raise ASRError(f"转写结果里没有 segments 这个键，只有 {sorted(map(str, result))}")
    raw_segments = result["segments"]
    # str / bytes / Mapping 都是 Iterable，光判 Iterable 会把它们直接放行，然后在
    # segments_to_cues 的循环里漏成裸 AttributeError（`'str' object has no attribute
    # 'get'`）—— 而 AttributeError **不在** cli.PIPELINE_ERRORS 里。生成器仍然放行，
    # 所以声明的 Iterable 契约没有被偷偷收窄成 Sequence。元素级的类型检查在
    # segments_to_cues 里（那里才逐个看得到）。
    if isinstance(raw_segments, str | bytes | Mapping) or not isinstance(raw_segments, Iterable):
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
