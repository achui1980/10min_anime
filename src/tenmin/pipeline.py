"""阶段编排。每个阶段读上游文件、写自己的文件，靠 mtime 决定是否跳过。"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Sequence
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.util import load_yaml_guess_indent

from tenmin import atomic, config_slices
from tenmin.config import EpisodeConfig, ProjectConfig, _parse_project_yaml
from tenmin.docgen.narration import render_narration
from tenmin.docgen.table import render_table
from tenmin.ingest.normalize import build_track
from tenmin.ingest.resolve import resolve_subtitle_source
from tenmin.models import (
    DialogueTrack,
    Script,
    SignalReport,
    Timeline,
    TranslatedTrack,
    VoiceTrack,
)
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.audio import mix_audio
from tenmin.render.ffmpeg import (
    FFmpegBinaryError,
    FFmpegError,
    preflight,
    probe_duration,
    probe_frame_rate,
)
from tenmin.render.subtitles import check_cue_legibility, render_ass
from tenmin.render.timeline import build_timeline
from tenmin.render.tts import TTSEngine, synthesize_track
from tenmin.render.video import render_video
from tenmin.script.llm import LLMProvider, LLMSchemaError
from tenmin.script.single import generate_script
from tenmin.script.usage import UsageRecord, write_usage
from tenmin.script.validate import ScriptValidationError
from tenmin.signals.aggregate import build_report
from tenmin.translate.glossary import (
    effective_glossary,
    load_glossary,
    merge_glossary,
    save_glossary,
)
from tenmin.translate.lines import translate_track
from tenmin.translate.srt_writer import render_zh_srt

# translate 紧跟 ingest：它吃对白轨，而下游的解说稿要用它回写的累积术语表。
#
# **已知的顺序错位，已知且无害**：signals 是全局阶段（run_pipeline 在按集纵向循环
# **之前**就把所有集跑完），而 translate 住在纵向循环里 —— 所以真实执行顺序是 signals
# 先于 translate。这不违反任何依赖：signals 不读对白文本，只看时间戳与字数，跟语言无关。
# 这张表里的位置表达的是「逻辑上它紧跟 ingest」以及 --only / --from 的语义顺序。
STAGES = [
    "ingest",
    "translate",
    "signals",
    "script",
    "docgen",
    "voice",
    "timeline",
    "audio",
    "render",
]


def episode_stem(episode: int) -> str:
    """全项目所有按集产物的统一文件名前缀。至少两位，三位集号不截断。"""
    return f"E{episode:02d}"


# 产物布局的唯一权威表：方法名 -> (子目录, 文件名后缀)。后缀为空串表示产物是目录本身。
#
# 这些字符串是**磁盘上的存量契约**：work/ 下有 100 多个已生成的产物，改一个字符就等于
# 全部存量失效（流水线会认为什么都没跑过，重新调 LLM、重新 TTS、重新渲染几十分钟）。
# tests/test_pipeline.py 的 FROZEN_LAYOUT 按字面量逐条锁死这张表拼出来的结果。
#
# 目录名带序号前缀（01_/02_/…）是刻意的：`ls work/<slug>` 就能按流水线顺序读出来。
# 三个例外：out/ 不带序号，因为它是给人看的交付物目录，不是中间产物；srt/ 是输入目录，
# 理由见 asr_cache 那条；zh/ 是翻译阶段的中间产物目录，理由见 zh_lines 那条。
_ARTIFACTS: dict[str, tuple[str, str]] = {
    "dialogue": ("01_dialogue", ".dialogue.json"),
    "signals": ("02_signals", ".signals.json"),
    "script": ("03_script", ".script.json"),
    # 只在 LLM 连续 N 轮都没输出合 schema 的 JSON 时才写：存最后一次的**原始**模型
    # 输出。刻意跟 script.json 同目录：出事时用户看的就是 03_script/，把现场放在别处
    # 只会让人找不到。它不是任何阶段的输入或输出，不参与 _is_fresh。
    "script_raw": ("03_script", ".raw.txt"),
    # 只在语义校验（script/validate.py）连续 N 轮都没过时才写：存最后一次那份
    # **schema 合法但语义没过**的 Script。刻意跟 script_raw 分开一个槽位而不是复用它：
    # raw.txt 存的是「连 schema 都不合法的原始文本」，两者形态完全不同（一个是 JSON、
    # 一个可能是任意垃圾），共用一个文件名会让用户打开它时不知道该期待什么。
    # 它不是任何阶段的输入或输出，不参与 _is_fresh。
    "script_rejected": ("03_script", ".rejected.json"),
    # script 阶段**每次真正跑完都写**（哪怕一条 warning 都没有，写空列表）：
    # 原来 warnings 只在内存里攒着、运行末尾由 cli.py 打一次黄字，既不落盘也不进
    # 对照表；而阶段一旦 fresh 就被跳过，重跑连那次黄字都不再出现。于是 validate.py
    # 里那批「只给 warning」的检查（锚点落窗外、画面拉伸超界、留白引不到原声、时间线
    # 倒退）在「没人盯终端」的用法下等于空转。
    # 空列表与文件缺失刻意区分开：空 = 查过了没发现问题，缺失 = 从没跑过。每次都写
    # 也顺带避免上一跑的警告文件变成过期的谎言。
    # 它不是任何阶段的输入或输出，不参与 _is_fresh。
    "script_warnings": ("03_script", ".warnings.json"),
    # 诊断产物，不参与 _is_fresh；失败也写下本次已耗的用量。
    "script_usage": ("03_script", ".usage.json"),
    "table": ("out", ".解说方案.md"),
    "narration": ("out", ".narration.txt"),
    # 翻译阶段。目录刻意不占 0N 编号：现有的 01_dialogue → 07_render 是连续的，
    # 真按执行顺序插进去要把后面六个目录全部改名，而这些字符串是磁盘上的存量契约
    # （改一个字符，work/ 下已有的全部产物路径失配），同时全仓大量注释按名字引用它们。
    # docgen 把给人看的东西写进 out/ 已经立了「编号不等于执行顺序」这个先例。
    # 项目级的累积术语表也住这个目录，但它不带集号，所以不在本表里（见 Paths.glossary）。
    "zh_lines": ("zh", ".zh.json"),
    # 中文字幕是交付物，跟解说方案、配音文本并排放 out/：那个目录的约定是
    # 「给人看的东西都在这」，标准 SRT 可以直接拖进播放器。
    "zh_subtitles": ("out", ".zh.srt"),
    # 只有 source=asr 且真正翻译时才写；不是新鲜度输入或输出。
    "zh_usage": ("zh", ".usage.json"),
    "voice_dir": ("04_voice", ""),
    "voice": ("04_voice", ".voice.json"),
    "timeline": ("05_timeline", ".timeline.json"),
    "subtitles": ("05_timeline", ".ass"),
    "mixed_audio": ("06_audio", ".mixed.m4a"),
    "video": ("07_render", ".mp4"),
    # 语音转写的落点。它是本表里唯一一个落在 srt/（输入目录）里的条目，也是唯一一个
    # 既不在带序号的阶段目录、也不在 out/ 交付目录里的，刻意的：这份 SRT 是双重身份的
    # —— 它是 ingest 自己产出的缓存，同时也是下一次运行的**输入**（转差了就手改，
    # _is_usable_asr_cache 会按 mtime 认它、照样按 kind="asr" 复用，所以手改的结果仍然
    # 被当成机器听写，下游该繁转简还是该翻译不因手改而变，见 resolve 的 SubtitleSource
    # docstring；它**不会**走 resolve 的「手传 SRT」那条分枝 —— 手改一份缓存文件不会
    # 往 project.yaml 的 episodes[].srt 里放任何东西）。放进 srt/ 才让「手传的字幕和
    # 机器听写的字幕在同一个抽屉里、名字上一眼分得开」这件事成立。
    #
    # 后缀 `.asr.srt` 应当跟 ingest/resolve.py 的 _ASR_SUFFIX 保持一致，而这里是那个
    # 名字的**唯一权威**：resolve 只收一个现成的 cache 路径、对它的名字形状不作任何
    # 要求。所以不一致只丢观感（改成 `.transcribed.srt` 会得到 `E11.transcribed.embedded.srt`
    # 这种名字，而规范名得到的是干净的 `E11.embedded.srt`），不会撞车 —— 防撞车全靠
    # _embedded_dest 的 `Path(name).stem`。
    "asr_cache": ("srt", ".asr.srt"),
}


class Paths:
    """一个 project 的全部阶段产物路径。

    每个方法都是 _ARTIFACTS 表的一行薄包装。刻意保留显式方法而不是 __getattr__
    动态派发：调用点（pipeline / cli / 一堆测试）到处在用 paths.script(2)，
    动态派发会让拼错的名字变成运行时 AttributeError、IDE 跳转与补全全失效。
    这里要的是「布局知识只有一份」，不是「代码行数最少」。
    （这段原来写着方法个数，而那个数字在加第 14 个条目时就已经过期了 ——
    真正锁住「表与方法一一对应」的是 test_paths_exposes_exactly_the_frozen_artifacts。）

    表外还挂着一个 property（glossary）：项目级产物不带集号，_artifact() 拼不出来，
    理由见它自己的 docstring。
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def _artifact(self, kind: str, episode: int) -> Path:
        subdir, suffix = _ARTIFACTS[kind]
        return self.root / subdir / f"{episode_stem(episode)}{suffix}"

    def dialogue(self, episode: int) -> Path:
        return self._artifact("dialogue", episode)

    def signals(self, episode: int) -> Path:
        return self._artifact("signals", episode)

    def script(self, episode: int) -> Path:
        return self._artifact("script", episode)

    def script_raw(self, episode: int) -> Path:
        return self._artifact("script_raw", episode)

    def script_rejected(self, episode: int) -> Path:
        return self._artifact("script_rejected", episode)

    def script_warnings(self, episode: int) -> Path:
        return self._artifact("script_warnings", episode)

    def script_usage(self, episode: int) -> Path:
        return self._artifact("script_usage", episode)

    def table(self, episode: int) -> Path:
        return self._artifact("table", episode)

    def narration(self, episode: int) -> Path:
        return self._artifact("narration", episode)

    def zh_lines(self, episode: int) -> Path:
        return self._artifact("zh_lines", episode)

    def zh_subtitles(self, episode: int) -> Path:
        return self._artifact("zh_subtitles", episode)

    def zh_usage(self, episode: int) -> Path:
        return self._artifact("zh_usage", episode)

    @property
    def glossary(self) -> Path:
        """跨集累积的专有名词表。

        项目级、不带集号 —— 它的全部意义就是让第 2 集知道第 1 集把人名译成了什么。
        所以它进不了 _ARTIFACTS：那张表的每一行都要靠集号才拼得出文件名。
        刻意是 property 而不是方法：按集产物那张表有个测试断言 Paths 上可调用的公开
        名字集合正好等于表里的键，property 不是 callable，自动落在那个集合之外。

        目录名刻意从 _ARTIFACTS["zh_lines"] 上取而不是再写一个 "zh" 字面量：术语表跟
        逐集译文必须同目录（`ls work/<slug>/zh` 要能一眼看全翻译阶段的中间产物），
        而两份字面量之间没有任何机制能防止它们分叉。
        """
        return self.root / _ARTIFACTS["zh_lines"][0] / "glossary.json"

    def voice_dir(self, episode: int) -> Path:
        return self._artifact("voice_dir", episode)

    def voice(self, episode: int) -> Path:
        return self._artifact("voice", episode)

    def timeline(self, episode: int) -> Path:
        return self._artifact("timeline", episode)

    def subtitles(self, episode: int) -> Path:
        return self._artifact("subtitles", episode)

    def mixed_audio(self, episode: int) -> Path:
        return self._artifact("mixed_audio", episode)

    def video(self, episode: int) -> Path:
        return self._artifact("video", episode)

    def asr_cache(self, episode: int) -> Path:
        """语音转写结果的落点，也是 resolve_subtitle_source 的 cache 参数。"""
        return self._artifact("asr_cache", episode)


def stages_from(stage: str) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"未知阶段 {stage!r}，可选：{', '.join(STAGES)}")
    return STAGES[STAGES.index(stage) :]


def resolve_stages(
    *, from_stage: str = "ingest", only: Sequence[str] | None = None
) -> list[str]:
    """把 --from / --only 解析成实际要跑的阶段列表，永远按 STAGES 的顺序返回。

    CLI 与 run_pipeline 共用这一份实现。原先两边各算一遍（cli.py 算出来只为校验，
    再把 only/from_stage 原样传给 run_pipeline 让它重算），两份逻辑随时会分叉。

    only 给了就忽略 from_stage。只是过滤 STAGES，所以传进来的顺序无所谓、重复也无所谓；
    含未知阶段则抛 ValueError。only 传空序列（不是 None）表示「什么都不跑」，
    这个语义是刻意保留的既有行为。
    """
    if only is not None:
        unknown = [stage for stage in only if stage not in STAGES]
        if unknown:
            raise ValueError(f"未知阶段 {unknown[0]!r}，可选：{', '.join(STAGES)}")
        return [stage for stage in STAGES if stage in only]
    return stages_from(from_stage)


# 两个都走 atomic.write_text：产物半途被打断时，正式路径上要么是完整的旧内容、
# 要么根本不存在，绝不会留下一个 mtime 最新的半截文件让 _is_fresh 判成「已最新」。
def _write_json(path: Path, payload: str) -> None:
    atomic.write_text(path, payload)


def _write_text(path: Path, text: str) -> None:
    atomic.write_text(path, text)


def _is_fresh(outputs: list[Path], inputs: list[Path]) -> bool:
    """产物是否已经比输入新，可以整段跳过。

    语义与已知局限（改这个函数前先读完）：

    1. **前提是「产物要么不存在、要么完整」，而这个前提现在是被兜住的**。判据只有
       mtime，而被 Ctrl-C 打断的 ffmpeg / TTS 曾经会留下一个 mtime 恰好最新的半截文件，
       纯 mtime 比较必然把它当成最新产物直接跳过，坏产物一路进成片。**现在全部产物写入
       都走 `tenmin.atomic`**（临时 `.part` + `os.replace`，正式路径上永远只有完整
       内容；tests/test_source_hygiene.py 有一条审计守着「不许直接 write_text」），
       所以「写了一半」这一类已经到不了这里。
       这里另外还留着一条最低成本的兜底：**0 字节产物一律视为不新鲜**。本流水线没有
       任何一个阶段会合法地产出空文件（json/md/txt/m4a/mp4 都有内容），所以这条规则
       不会误伤；它挡的是「刚 open 就被打断」这一类，跟原子写是两层独立的保险。
    2. **inputs 必须包含这个阶段的配置切片**（run_pipeline 里的 is_fresh 包装自动加上，
       见 tenmin.config_slices）。漏了它就等于这个阶段的配置旋钮改了都不生效。原来这里
       放的是整个 project.yaml，结果是改任何一个旋钮、登记任何一集，全部阶段一起过期。
    3. **inputs 一个都不存在时返回 True（跳过）**，见下面的注释。
    """
    if not outputs:
        return False
    for path in outputs:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        if stat.st_size == 0:
            return False
    existing_inputs = [p for p in inputs if p.exists()]
    if not existing_inputs:
        # 刻意跳过而不是重跑：一个输入都不存在时重跑只可能崩（run_ingest 读不到 SRT）
        # 或产出垃圾，而跳过至少保住磁盘上已有的产物。配置切片进了 inputs 之后这个分支在
        # run_pipeline 里已经走不到（切片在开跑前必然已落盘），留着只为不给库调用方/单测
        # 埋 FileNotFoundError。
        return True
    newest_input = max(p.stat().st_mtime_ns for p in existing_inputs)
    oldest_output = min(p.stat().st_mtime_ns for p in outputs)
    return oldest_output >= newest_input


def _is_fresh_stamped(outputs: list[Path], inputs: list[Path], stamp: Path) -> bool:
    """给「内容不变就不写」的全局阶段（ingest / signals）用的新鲜度判据。

    这两个阶段在产出跟盘上逐字节相同时不改写产物、保住旧 mtime，下游才不会因为「内容
    没变、mtime 变了」被连带判过期（登记一集新番时 ingest 会把全部集重跑一遍，这条就是
    让已有的集纹丝不动的关键）。代价是产物的 mtime 不再代表「这个阶段上次跑完的时刻」：
    改一个不影响产出的阈值，重跑之后产物照样比切片旧，纯按产物判就会**每次**都重跑。
    所以这里按「上次跑完」的戳子判输入，同时核对产物的字节身份。失败的运行留下空戳子，
    不能退回按可能只写了一部分的产物判新鲜。

    戳子不存在（升级前的项目）或是旧版纯文本时退回按产物判，避免升级后白跑一遍。
    """
    if not _is_fresh(outputs, []):
        return False
    if not stamp.is_file():
        return _is_fresh(outputs, inputs)
    try:
        payload = stamp.read_text(encoding="utf-8")
        if not payload:
            return False
        recorded = json.loads(payload)
    except (OSError, UnicodeError):
        return False
    except ValueError:
        # 旧版戳子只有阶段名；仅它们可以按产物 mtime 退回判定。
        if payload == f"{stamp.stem}\n":
            return _is_fresh(outputs, inputs)
        return False
    if not isinstance(recorded, dict) or recorded != config_slices.output_digests(outputs):
        return False
    return _is_fresh([stamp], inputs)


def _active_episodes(cfg: ProjectConfig) -> list[EpisodeConfig]:
    """登记过来源（srt 或 video）的集。预填条目（只写了 op/ed）不参与任何阶段。"""
    return [episode for episode in cfg.episodes if episode.has_source]


def _source_duration(cfg: ProjectConfig, episode: EpisodeConfig) -> float | None:
    """尽力拿这一集源视频的真实片长；拿不到就返回 None（让调用方退化到字幕末尾）。

    刻意把**探测**类失败都吞成 None：ingest 是唯一不需要视频的阶段，「只有 SRT」是
    文档里写明的合法用法（`tenmin run <slug> --only ingest` 就能出对白轨与信号），
    绝不能因为视频缺失就把它打死。三类被吞掉的失败：

    - ValueError：project.yaml 的这一集没写 video（video_path 自己抛的）。
    - 文件不在：写了 video 但文件还没到位（下载中、换过外置盘）。
    - FFmpegError / OSError：文件在但 ffprobe 读不出（0 字节壳子、不是视频、
      容器元数据坏掉）。

    **`FFmpegBinaryError` 例外，它必须响亮地抛出去。** 「`render.ffprobe_path` 配错了」
    跟「这个文件探不出时长」是两件完全不同的事：前者一路降级下去的后果是 ingest 阶段
    **完全静默地**用字幕末尾当片长，而那会系统性挪动整个 ED 窗（实测真实片长比字幕
    末尾长 1.8~24.3 秒）—— 用户既没有报错也没有 warning，只有一个 OP/ED 判得不对的
    成片。它是 FFmpegError 子类，所以 except 的顺序（窄的在前）就是全部机制。

    剩下的降级是静默的：退化后的 duration 会照常写进 dialogue.json，`tenmin inspect`
    第一行就打它，对着片长一眼能看出是不是字幕末尾。
    """
    try:
        path = cfg.video_path(episode)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        return probe_duration(path, ffprobe=cfg.render.ffprobe_path)
    except FFmpegBinaryError:
        raise
    except (FFmpegError, OSError):
        return None


def _ingest_inputs(cfg: ProjectConfig) -> list[Path]:
    """ingest 阶段的新鲜度输入。

    每集取「手传字幕」与「源视频」里**配置了**的那些 —— 判据是字段不是 None，这一层
    压根不查文件在不在（存在性过滤在 _is_fresh 里做：`[p for p in inputs if p.exists()]`）。
    原来只取字幕，对一个只有视频的集会得到空列表 —— 而空输入在新鲜度判据里等于
    「跳过」，于是换了片源也不会重跑。
    反过来也别顺手把整份列表滤空：那样 ingest 就只盯配置切片，「改了字幕再重跑」
    会被静默 stage_skip，下游各阶段因为 dialogue.json 没变而跟着一起跳过。
    两条不变量各有一条测试守着。

    刻意**不**把 ingest 自己落在 srt/ 里的那两份产物算进来 —— 语音转写的缓存
    （Paths.asr_cache，`E{NN}.asr.srt`）与软字幕轨抽出来的那份（`E{NN}.embedded.srt`，
    见 ingest.resolve 的 _embedded_dest）：算进输入会让「解析完写出这份 SRT」这个动作
    立刻使 ingest 变得不新鲜，每次都重跑。两者各自的失效判据都在 ingest.resolve 里对着
    源视频判（缓存按 mtime 复用，抽出来那份每次重写）。当前实现两份都进不来（输入只来自
    episodes[].srt），所以这一段不是在描述一层真实过滤，而是给「顺手 glob 一下 srt/
    目录」这个改法留的警告 —— 两份都得排除，不是只排除缓存那一份。

    也别把 None 留在返回值里：唯一的消费者是 run_pipeline 的 ingest 分枝，而
    _is_fresh 对每个输入调 Path.exists，一个 None 会把它崩成 AttributeError ——
    指不到「这一集是生肉」这个根因。
    """
    inputs: list[Path] = []
    for episode in _active_episodes(cfg):
        srt = cfg.srt_path(episode)
        if srt is not None:
            inputs.append(srt)
        if episode.video is not None:
            inputs.append(cfg.video_path(episode))
    return inputs


def _translate_inputs(paths: Paths, episode: int) -> list[Path]:
    """翻译阶段的新鲜度输入。

    只看对白轨。累积术语表刻意**不**在里面：它既是这个阶段的输入（喂给模型的「已定
    术语」）又是它的输出（回写新认出来的词），算进输入集会让这个阶段永远不新鲜 ——
    每一次运行都重跑，每一次都重新付一集的翻译费。

    代价是：手改了术语表不会让已经翻好的集自动重翻（要 --force）。这是有意的取舍 ——
    改译名的主要目的是让**后面**几集用对写法，而那条路是通的（下一集的 translate 会读到
    改后的表）。解说稿那边不一样：那张表是它 prompt 的一部分，所以它把表算进了输入，
    见 _script_inputs。
    """
    return [paths.dialogue(episode)]


def _script_inputs(paths: Paths, episode: int) -> list[Path]:
    """解说稿阶段的新鲜度输入。

    累积术语表（zh/glossary.json）在里面，因为它是 prompt 的一部分：run_script 把它叠上
    project.yaml 手写的那份喂给模型。表变了旧解说稿里的译名就对不上了，必须重跑。

    「表变了」得是真的变了：save_glossary 在内容与盘上逐字节相同时不碰文件，所以一集
    translate 跑完并不必然刷这张表的 mtime。少了那一条，批处理里最后几集的 translate 会
    把前面几集刚写好的解说稿全部判旧、下次运行白重跑一遍。

    文件不存在时会被 _is_fresh 自己过滤掉（它只看存在的那些输入），所以无条件列进来是
    安全的 —— 现有的繁中片源压根不会有这个文件。
    """
    return [paths.dialogue(episode), paths.signals(episode), paths.glossary]


def run_ingest(cfg: ProjectConfig) -> list[DialogueTrack]:
    paths = Paths(cfg.root)
    tracks = []
    for episode in _active_episodes(cfg):
        # 三条来源路径（手传 SRT / 视频内嵌软字幕轨 / 语音转写）都归一成一份 SRT，
        # 所以 build_track 拿到的东西形态不变。这里每集**只解析一次**：软字幕轨那条
        # 分枝每次调用都会重抽一遍（见 ingest/resolve.py 里那段注释），多调一次就多
        # 一次 demux。
        source = resolve_subtitle_source(
            cfg.srt_path(episode),
            cfg.video_path(episode) if episode.video is not None else None,
            cache=paths.asr_cache(episode.number),
            asr_config=cfg.asr,
            ffmpeg_path=cfg.render.ffmpeg_path,
            ffprobe_path=cfg.render.ffprobe_path,
        )
        track = build_track(
            source.path,
            episode=episode.number,
            source=source.kind,
            glossary=cfg.glossary,
            convert_traditional=cfg.locale.convert_traditional,
            show_title=cfg.show,
            op_range=episode.op_range,
            ed_range=episode.ed_range,
            duration=_source_duration(cfg, episode),
            ingest=cfg.ingest,
            credits=cfg.credits,
        )
        # 内容不变就不写：保住旧 mtime，下游 signals/script 才不会被连带判过期。
        atomic.write_text_if_changed(
            paths.dialogue(episode.number), track.model_dump_json(indent=2)
        )
        tracks.append(track)
    return tracks


def ingest_warnings(tracks: Sequence[DialogueTrack]) -> list[str]:
    """把解析期悄悄丢掉/修补过的数据翻译成 warnings。

    本项目刻意不引入 logging，所以坏数据走的是已有的 warnings 通道（与 script /
    voice / timeline 三个阶段一致），另外这两个数字也常驻 dialogue.json，
    `tenmin inspect` 会打出来。run_ingest 的返回类型刻意不改成 tuple ——
    它有 4 个既有调用点把返回值直接当 list 用。
    """
    out: list[str] = []
    for track in tracks:
        if track.skipped_blocks:
            out.append(
                f"E{track.episode:02d}：SRT 里有 {track.skipped_blocks} 个块找不到时间戳行，"
                "已整块跳过（对白可能缺失，建议检查字幕源格式）"
            )
        if track.clamped_cues:
            out.append(
                f"E{track.episode:02d}：SRT 里有 {track.clamped_cues} 条 cue 的终点早于起点，"
                "已夹成零时长并标记 suspect"
            )
    return out


def _load_tracks(cfg: ProjectConfig) -> list[DialogueTrack]:
    paths = Paths(cfg.root)
    tracks = []
    for episode in _active_episodes(cfg):
        path = paths.dialogue(episode.number)
        if not path.exists():
            raise FileNotFoundError(f"缺少对白轨产物 {path}，请先跑 ingest 阶段")
        tracks.append(DialogueTrack.model_validate_json(path.read_text(encoding="utf-8")))
    return tracks


async def run_translate(
    cfg: ProjectConfig, provider: LLMProvider, episode: int
) -> TranslatedTrack:
    """翻译一集：落译文轨、中文字幕，并把新认出的术语并回累积表。

    只对听写来的对白动手。手传的字幕与从视频里抽出来的软字幕轨都是片源自带的，本来
    就是观众能读的语言。判据用对白轨的 source 字段，刻意不做语言自动检测 —— 那是个
    会错的猜测，而 source 是一个确定的事实。

    跳过时连 `zh/` 目录都不建（早退发生在任何写盘之前），所以现有的繁中片源在磁盘上
    看不出这个阶段存在过。代价是这个函数每次运行都会被叫一遍：那一集的产物永远不会出现，
    而新鲜度判据（_translate_inputs + _is_fresh）只看文件 mtime、拿不到 source 字段，
    于是它每次都判「不新鲜」。一次读盘换「判据不必认识对白轨的内容」，划得来。
    读的量是 _load_tracks 的全量（cfg.episodes 里每一集的对白轨），跟 run_script 同一个
    口径 —— 十几集的 json 相对一次 LLM 调用可以忽略。

    已知边界：手传一份日语 SRT（或者软字幕轨恰好是日语）时，这一阶段不会跑，而且那份
    对白还会被繁转简改字。目前的片源都不是这种情况，真碰上了再说。
    """
    track = next(t for t in _load_tracks(cfg) if t.episode == episode)
    if track.source != "asr":
        return TranslatedTrack(episode=episode)

    paths = Paths(cfg.root)
    accumulated = load_glossary(paths.glossary)
    usage: list[UsageRecord] = []
    try:
        translated = await translate_track(
            cfg, track, provider, accumulated=accumulated, usage=usage
        )
    finally:
        write_usage(paths.zh_usage(episode), episode, usage)

    _write_json(paths.zh_lines(episode), translated.model_dump_json(indent=2))
    _write_text(paths.zh_subtitles(episode), render_zh_srt(track, translated))
    # merge 的方向是「累积的赢」：已经定下的译名不许被后面某一集改掉。喂给模型的那份表
    # 另有一个方向（手写的赢），那一步在 translate_track 内部做。
    save_glossary(paths.glossary, merge_glossary(accumulated, translated.glossary))
    return translated


def run_signals(cfg: ProjectConfig) -> list[SignalReport]:
    paths = Paths(cfg.root)
    reports = []
    for track in _load_tracks(cfg):
        report = build_report(track, cfg=cfg.signals)
        # 同 run_ingest：内容不变就不写，别让 script 因为 mtime 刷新白跑。
        atomic.write_text_if_changed(
            paths.signals(track.episode), report.model_dump_json(indent=2)
        )
        reports.append(report)
    return reports


def _load_reports(cfg: ProjectConfig) -> list[SignalReport]:
    paths = Paths(cfg.root)
    reports = []
    for episode in _active_episodes(cfg):
        path = paths.signals(episode.number)
        if not path.exists():
            raise FileNotFoundError(f"缺少信号产物 {path}，请先跑 signals 阶段")
        reports.append(SignalReport.model_validate_json(path.read_text(encoding="utf-8")))
    return reports


async def run_script(
    cfg: ProjectConfig,
    provider: LLMProvider,
    episode: int,
    *,
    reporter: ProgressReporter | None = None,
) -> tuple[Script, list[str]]:
    paths = Paths(cfg.root)
    tracks = _load_tracks(cfg)
    reports = _load_reports(cfg)
    track = next(t for t in tracks if t.episode == episode)
    report = next(r for r in reports if r.episode == episode)
    # 累积表叠手写表，手写的赢：手写表是纠错入口，机器译错了人得能盖掉它。两边都空时
    # 结果是空字典，build_glossary_block 会印「（无术语表）」—— 现有的繁中片源走的就是
    # 这条路（它们压根没有 zh/glossary.json）。
    glossary = effective_glossary(load_glossary(paths.glossary), cfg.glossary)
    usage: list[UsageRecord] = []
    try:
        script, warnings = await generate_script(
            cfg, track, report, provider, reporter=reporter, glossary=glossary, usage=usage
        )
    except ScriptValidationError as error:
        # 跟下面 LLMSchemaError 的落盘同理：pipeline 是唯一知道产物往哪写的一层。
        # 一次真实调用可达 561 秒，重试耗尽后原来什么都不留。
        if error.script is None:
            raise
        rejected_path = paths.script_rejected(episode)
        _write_json(rejected_path, error.script.model_dump_json(indent=2))
        raise ScriptValidationError(
            f"{error}\n最后一版没通过校验的剧本已存到 {rejected_path}",
            script=error.script,
        ) from error
    except LLMSchemaError as error:
        # 落盘选在这一层：llm.py 不该知道 Paths（它是纯 provider 层，被单测直接实例化），
        # 而 single.py 只是拼 prompt 的无状态函数、同样拿不到项目根目录。pipeline 是
        # 「知道产物往哪写」的唯一一层，所以现场也在这里落。
        if not error.raw_output:
            raise
        raw_path = paths.script_raw(episode)
        _write_text(raw_path, error.raw_output)
        raise LLMSchemaError(
            f"{error}\n最后一次的原始模型输出已存到 {raw_path}",
            raw_output=error.raw_output,
        ) from error
    finally:
        write_usage(paths.script_usage(episode), episode, usage)
    _write_json(paths.script(episode), script.model_dump_json(indent=2))
    # 每次跑完都写，哪怕 warnings 是空的（理由见 _ARTIFACTS["script_warnings"]）。
    # 写在 script 落盘之后：warnings 描述的是刚写下去那一份剧本，顺序反了会出现
    # 「警告文件指向一份还没落盘的剧本」这种中间态。
    _write_json(
        paths.script_warnings(episode),
        json.dumps(
            {"episode": episode, "warnings": list(warnings)},
            ensure_ascii=False,
            indent=2,
        ),
    )
    return script, warnings


def run_docgen(cfg: ProjectConfig, episode: int) -> Script:
    paths = Paths(cfg.root)
    script_path = paths.script(episode)
    if not script_path.exists():
        raise FileNotFoundError(f"缺少剧本 {script_path}，请先跑 script 阶段")
    script = Script.model_validate_json(script_path.read_text(encoding="utf-8"))
    _write_text(paths.table(episode), render_table(script))
    _write_text(paths.narration(episode), render_narration(script))
    return script


def _find_episode(cfg: ProjectConfig, episode_number: int) -> EpisodeConfig:
    """按集数查找已注册的 episode 配置。

    找不到时报错并提示用户先用 --srt/--video/--episode 注册。
    """
    for episode in cfg.episodes:
        if episode.number == episode_number:
            if not episode.has_source:
                raise ValueError(
                    f"第 {episode_number} 集在 project.yaml 里只是预填条目（还没有 video）。"
                    f"请用 `tenmin run {cfg.slug} --episode {episode_number} "
                    "--video <视频路径>` 登记源片，已填的 op_range/ed_range 会保留。"
                )
            return episode
    raise ValueError(
        f"第 {episode_number} 集还没有注册。"
        f"请先用 `tenmin run <slug> --episode {episode_number} "
        "--srt <srt路径> --video <视频路径>` 注册这一集。"
    )


def _write_back_episode(yaml_path: Path, entry: EpisodeConfig) -> None:
    """只更新这一集的来源字段，保留其它 YAML 节点的注释、顺序与格式。

    从原文件猜序列缩进，兼容 init 的顶格列表和手写的缩进列表；
    长路径不折行。整份结果经原子写替换，避免中断损坏已登记的集数。
    """
    text = yaml_path.read_text(encoding="utf-8")
    round_trip = YAML()
    round_trip.preserve_quotes = True
    round_trip.width = 4096
    data, sequence_indent, sequence_offset = load_yaml_guess_indent(text, yaml=round_trip)
    # The helper stops at the first block sequence, so its indent describes
    # the position after `- ` rather than earlier nested maps. The parser's
    # key columns also handle `render: # settings` and intervening comments.
    mapping_indent = sequence_indent
    if isinstance(data, CommentedMap):
        mapping_indent = next(
            (
                value.lc.key(next(iter(value)))[1] - data.lc.key(key)[1]
                for key, value in data.items()
                if isinstance(value, CommentedMap) and value and not value.fa.flow_style()
            ),
            (sequence_indent - 2) if sequence_indent and sequence_offset else sequence_indent,
        )
    round_trip.indent(
        mapping=mapping_indent or 2,
        sequence=sequence_indent or 2,
        offset=sequence_offset or 0,
    )
    if data is None:
        data = CommentedMap()
    episodes = data.get("episodes")
    if episodes is None:
        episodes = CommentedSeq()
        data["episodes"] = episodes
    target = next(
        (
            item
            for item in episodes
            if isinstance(item, dict) and item.get("number") == entry.number
        ),
        None,
    )
    if target is None:
        # init 的 episodes: [] 是 flow style；新增映射时必须改回块式。
        if not episodes:
            episodes.fa.set_block_style()
        target = CommentedMap()
        target["number"] = entry.number
        episodes.append(target)
    if entry.srt is None:
        target.pop("srt", None)
    else:
        target["srt"] = entry.srt.as_posix()
    target["video"] = str(entry.video)
    buffer = io.StringIO()
    round_trip.dump(data, buffer)
    atomic.write_text(yaml_path, buffer.getvalue())


def register_episode(
    cfg: ProjectConfig, *, episode: int, srt: Path | None, video: Path
) -> ProjectConfig:
    """把这一集登记进 project.yaml：SRT 拷进项目目录，视频只记路径不拷。

    srt 传 None 就是生肉入口：yaml 里这一集只有 video，对白轨留给 ingest 那边的
    resolve_subtitle_source 去解（软字幕轨抽取或语音转写）。重登记一集时传 None
    会把已有的 srt 字段**清掉** —— 那正是「这一集改走生肉路线」的意思，留着旧值
    会让解析继续走「手传 SRT」那条路、对着一份用户已经不想用的字幕出片。

    如果这一集已经注册过，就覆盖 srt/video 路径（保留其它字段）；
    否则追加一条新的 episode 记录。返回更新后的 ProjectConfig（root 已绑定）。

    为什么 SRT 拷、视频不拷：
    - SRT 是几十 KB，而且是主要的人工编辑面（清洗规则调不好时要就地改字幕），
      拷进 srt/E{NN}.srt 让项目自洽、路径统一，成本可以忽略。
    - 源片实测 300MB~1.4GB。原实现 shutil.copyfile 整份拷进 video/，10 集就是
      3~14GB 的纯冗余 + 一次全量读写，而 config.video_path 本来就支持绝对路径。
      所以这里只把源片的绝对路径记进 yaml。

    为什么记绝对路径而不是 hardlink/symlink 到 video/E{NN}.mkv：
    - 零 I/O、零磁盘、零文件系统能力假设。hardlink 跨卷/跨网络挂载直接失败，
      symlink 在 Windows 上要额外权限，而本项目的源片常年住在外置盘与网络共享上。
    - 失败模式是响的：源片被移走/删掉时 render.ffmpeg.preflight 会抛
      「找不到源视频 …，请检查 project.yaml 的 episodes[].video」，
      而 symlink 只会留下一个悬空链接、错误信息指向 work/ 里那个假身份。
    - 用户在 project.yaml 里直接看到源片真实位置，可查可改。

    向后兼容：只有**本次登记的这一集**的 srt/video 两个字段会被改写（video 写成绝对路径）。
    其余集、以及 yaml 里别的一切（注释、键序、引号）都由 _write_back_episode 原样保留，
    存量的相对路径（work/saijo/ 下 10 个已经拷好的 mp4）逐字节不变，video_path() 照旧按
    project.yaml 所在目录解析。
    """
    # cfg 可能在载入后被人手改了文件；再次校验磁盘上的 YAML，避免拷好字幕
    # 才由 round-trip parser 报错，留下没有登记成功的 SRT。
    _parse_project_yaml(cfg.config_path)
    # 在拷贝字幕或改写配置之前检查输入，给出可读的错误并避免部分登记。
    if srt is not None and not Path(srt).is_file():
        raise FileNotFoundError(f"找不到要登记的字幕文件：{srt}")

    relative_srt: Path | None = None
    if srt is not None:
        srt_dest = cfg.root / "srt" / f"E{episode:02d}.srt"
        # 原子拷：这份 SRT 是 ingest 阶段的输入，半截字幕会静默产出一条缺对白的对白轨
        # （srt_parser 对截断输入不报错），而它的 mtime 是最新的，_is_fresh 不会重跑。
        atomic.copy_file(srt, srt_dest)
        relative_srt = srt_dest.relative_to(cfg.root)

    # 绝对化：CLI 传进来的 --video 通常是相对当前工作目录的，而 project.yaml 里的
    # 相对路径是相对 work/<slug>/ 解析的，原样存进去会指向完全不同的位置。
    video_source = Path(video).resolve()

    existing = next((e for e in cfg.episodes if e.number == episode), None)
    if existing is not None:
        # 预填条目也走这条：只改 srt/video，op_range/ed_range 原样留着。
        existing.srt = relative_srt
        existing.video = video_source
        entry = existing
    else:
        entry = EpisodeConfig(number=episode, srt=relative_srt, video=video_source)
        cfg.episodes.append(entry)

    _write_back_episode(cfg.config_path, entry)

    return cfg


def _load_script(cfg: ProjectConfig, episode: int) -> Script:
    path = Paths(cfg.root).script(episode)
    if not path.exists():
        raise FileNotFoundError(f"缺少剧本 {path}，请先跑 script 阶段")
    return Script.model_validate_json(path.read_text(encoding="utf-8"))


def _load_voice(cfg: ProjectConfig, episode: int) -> VoiceTrack:
    path = Paths(cfg.root).voice(episode)
    if not path.exists():
        raise FileNotFoundError(f"缺少配音产物 {path}，请先跑 voice 阶段")
    return VoiceTrack.model_validate_json(path.read_text(encoding="utf-8"))


def _load_timeline(cfg: ProjectConfig, episode: int) -> Timeline:
    path = Paths(cfg.root).timeline(episode)
    if not path.exists():
        raise FileNotFoundError(f"缺少时间轴产物 {path}，请先跑 timeline 阶段")
    return Timeline.model_validate_json(path.read_text(encoding="utf-8"))


def _load_previous_voice(cfg: ProjectConfig, episode: int) -> tuple[VoiceTrack | None, str | None]:
    """读上一轮的 voice.json，纯粹为了拿里面记下的 chunk 时长（省 ffprobe 子进程）。

    读不动就返回 None：voice 阶段马上就要整份覆写它，一份坏的旧产物不该挡住重新合成。
    但也不能吞掉不吭声 —— 它是文档里写明的人工编辑面，所以带一条 warning 出去。
    """
    path = Paths(cfg.root).voice(episode)
    if not path.is_file():
        return None, None
    try:
        return VoiceTrack.model_validate_json(path.read_text(encoding="utf-8")), None
    except (ValueError, UnicodeDecodeError) as error:
        return None, (
            f"读不动上一轮的 {path.name}（{type(error).__name__}），"
            "本轮复用的 chunk 会重新用 ffprobe 量时长"
        )


async def run_voice(
    cfg: ProjectConfig,
    engine: TTSEngine,
    episode: int,
    reporter: ProgressReporter | None = None,
) -> tuple[VoiceTrack, list[str]]:
    paths = Paths(cfg.root)
    script = _load_script(cfg, episode)
    previous, previous_warning = _load_previous_voice(cfg, episode)
    track, warnings = await synthesize_track(
        script,
        episode,
        paths.voice_dir(episode),
        engine,
        reporter=reporter,
        max_attempts=cfg.render.tts_max_attempts,
        previous=previous,
        concurrency=cfg.render.tts_concurrency,
        rate=cfg.render.rate,
        ffprobe=cfg.render.ffprobe_path,
    )
    if previous_warning is not None:
        warnings.insert(0, previous_warning)
    _write_json(paths.voice(episode), track.model_dump_json(indent=2))
    return track, warnings


def run_timeline(
    cfg: ProjectConfig,
    episode: int,
    *,
    source_duration: float | None = None,
    frame_rate: float | None = None,
) -> tuple[Timeline, list[str]]:
    """重算时间轴并落盘 timeline.json + ASS 字幕。

    source_duration 与 frame_rate 是**同一次源片探测的两半**，所以它们共享一个开关：
    传了 source_duration 就说明「调用方自己在管源片探测」（只有 SRT 没有视频的降级
    路径、单测手上只有一个空壳文件），那时不再去碰 video_path —— 那条路上压根没有可探
    的视频。

    **`run_pipeline` 刻意什么都不传，让这个函数自己探。** 它上面那轮 preflight 确实刚
    读过同一个源片的时长（而且返回值被整个丢弃），但复用它买不到什么：

    - 实测一次真实单集运行（work/saijo E02，346MB mp4，`--only
      ingest,timeline,audio,render --force`）对源片一共发 **5 次 ffprobe**：ingest 的
      `_source_duration` 1 次、preflight 的 has_audio_stream + probe_duration 2 次、
      本函数的 probe_duration + probe_frame_rate 2 次。复用 preflight 的结果只省掉
      其中 1 次，而单次 ffprobe 实测 31–33ms —— 对照 script 阶段一次 LLM 调用实测
      561 秒，收益是 0.006%。
    - 代价是一个真实的正确性陷阱：`frame_rate` 跟着同一个开关走，而 preflight **不探
      帧率**。只把 source_duration 递进来会让帧对齐**静默关掉**（段边界退回「ffmpeg
      自己按帧取整」），要保住它就得让 run_pipeline 自己去探帧率 —— 也就是把探测从这里
      搬到那里，帧率那一次一次都没省。而「忘了递 frame_rate 那一半」是个不会报错、
      只在成片首尾差不到一帧的失效模式。

    preflight 里那次 probe_duration 也不是白跑的：它同时是一项**检查**（时长 <= 0 就
    抛错，挡住截断/0 字节的容器），所以它该留在 preflight 里，只是返回值在生产路径上
    没有消费者。

    frame_rate 是 None 的后果是段边界不做帧对齐（见 render/timeline.py），成片退回
    「ffmpeg 自己按帧取整」的老行为：能出片，只是首尾各差不到一帧。
    """
    paths = Paths(cfg.root)
    episode_cfg = _find_episode(cfg, episode)
    script = _load_script(cfg, episode)
    track = _load_voice(cfg, episode)
    if source_duration is None:
        video = cfg.video_path(episode_cfg)
        source_duration = probe_duration(video, ffprobe=cfg.render.ffprobe_path)
        if frame_rate is None:
            frame_rate = probe_frame_rate(video, ffprobe=cfg.render.ffprobe_path)
    timeline, warnings = build_timeline(
        script, track, source_duration, frame_rate=frame_rate, cfg=cfg.render
    )
    # 可读性检查住在 render/subtitles.py 的 check_cue_legibility —— 只有它知道字号与画布宽度算出来
    # 的行数。接在这里而不是 build_timeline 里：render/timeline.py 压根不认识字体，而
    # 这个函数手上同时有 cfg.render 与 warnings。
    warnings.extend(
        check_cue_legibility(
            timeline.subtitles,
            font_size=cfg.render.font_size,
            width=cfg.render.width,
            max_lines=cfg.render.subtitle_max_lines,
            min_seconds=cfg.render.subtitle_min_seconds,
            max_chars=cfg.render.subtitle_soft_max_chars,
        )
    )
    _write_json(paths.timeline(episode), timeline.model_dump_json(indent=2))
    _write_text(
        paths.subtitles(episode),
        render_ass(
            timeline.subtitles,
            font_size=cfg.render.font_size,
            font_name=cfg.render.subtitle_font_name,
            width=cfg.render.width,
            height=cfg.render.height,
        ),
    )
    return timeline, warnings


def run_audio(
    cfg: ProjectConfig, episode: int, reporter: ProgressReporter | None = None
) -> Path:
    paths = Paths(cfg.root)
    episode_cfg = _find_episode(cfg, episode)
    timeline = _load_timeline(cfg, episode)
    track = _load_voice(cfg, episode)
    return mix_audio(
        video=cfg.video_path(episode_cfg),
        timeline=timeline,
        track=track,
        voice_dir=paths.voice_dir(episode),
        out_path=paths.mixed_audio(episode),
        duck_db=cfg.render.duck_db,
        fade_out_seconds=cfg.render.fade_out_seconds,
        outro_seconds=cfg.render.outro_card_seconds,
        audio_codec=cfg.render.audio_codec,
        audio_bitrate=cfg.render.audio_bitrate,
        limiter_ceiling=cfg.render.limiter_ceiling,
        reporter=reporter,
        ffmpeg=cfg.render.ffmpeg_path,
    )


def run_render(
    cfg: ProjectConfig, episode: int, reporter: ProgressReporter | None = None
) -> Path:
    paths = Paths(cfg.root)
    episode_cfg = _find_episode(cfg, episode)
    timeline = _load_timeline(cfg, episode)
    audio = paths.mixed_audio(episode)
    if not audio.exists():
        raise FileNotFoundError(f"缺少混音产物 {audio}，请先跑 audio 阶段")
    ass = paths.subtitles(episode)
    if not ass.exists():
        raise FileNotFoundError(f"缺少字幕产物 {ass}，请先跑 timeline 阶段")
    return render_video(
        video=cfg.video_path(episode_cfg),
        timeline=timeline,
        audio=audio,
        ass=ass,
        out_path=paths.video(episode),
        encoder=cfg.render.video_encoder,
        width=cfg.render.width,
        height=cfg.render.height,
        crf=cfg.render.crf,
        preset=cfg.render.preset,
        tune=cfg.render.tune,
        videotoolbox_bitrate=cfg.render.videotoolbox_bitrate,
        # 片尾黑卡要跟正片同帧率，否则成片是 VFR（见 render/video.py 那段注释）。
        # 帧率取 timeline 产物里记的那个 —— segments 的边界就是按它对齐的，
        # 这里再探一次只会引入「万一中途换了源片」的错位空间。
        frame_rate=timeline.frame_rate,
        fade_out_seconds=cfg.render.fade_out_seconds,
        outro_seconds=cfg.render.outro_card_seconds,
        outro_title=f"{cfg.show} · EP{episode:02d}",
        outro_message=cfg.render.outro_message,
        # 必须跟上面 preflight 递给 font_names 的是**同一个值**：preflight 查的字体
        # 与 drawtext 用的字体分叉过一次（前者读 config、后者读模块级默认值），
        # 结果是「查了一个用不到的字体、用了一个没查过的字体」。
        outro_font_name=cfg.render.outro_font_name,
        reporter=reporter,
        ffmpeg=cfg.render.ffmpeg_path,
    )


class _EpisodeLabelledReporter:
    """把 substep 的 label 打上集号前缀，其余 5 个方法原样转发。

    批量并发跑 script 时，N 集的 substep 全打在同一个 stage 名（"script"）上，而
    rich_progress 的 `_substep_tasks` 是按 stage 做 key 的 —— 不带集号的话那一行会在
    几集之间来回跳，却不告诉你现在跳的是谁。

    刻意**不**改成「每集一个 stage 名」（`substep("script E02", …)`）：那样每集会多出
    一个 rich 任务行，而 `stage_done("script")` 只 pop 得掉 `_substep_tasks["script"]`，
    其余几行会一直堆在终端里（`_add_row` 只有在 `_in_episode` 为真时才登记进
    `_episode_rows`，而预取的 task 可能在第一次 episode_start 之前就开始上报了）。
    一行共享、靠 label 说清是谁，是这里唯一不引入行泄漏的选择。
    """

    def __init__(self, inner: ProgressReporter, episode: int) -> None:
        self._inner = inner
        self._prefix = episode_stem(episode)

    def stage_start(self, stage: str) -> None:
        self._inner.stage_start(stage)

    def stage_skip(self, stage: str) -> None:
        self._inner.stage_skip(stage)

    def stage_done(self, stage: str) -> None:
        self._inner.stage_done(stage)

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        self._inner.substep(stage, current, total, f"{self._prefix} {label}".rstrip())

    def episode_start(self, number: int, index: int, total: int) -> None:
        self._inner.episode_start(number, index, total)

    def episode_done(self, number: int, index: int, total: int) -> None:
        self._inner.episode_done(number, index, total)


async def _drain_script_tasks(
    tasks: dict[int, asyncio.Task[tuple[Script, list[str]]] | None],
) -> None:
    """收摊：取消还在飞的 script task，并把每个 task 的结果/异常都取回来。

    两件事都必须做，漏一件都会在用户终端上留下噪音：

    - **取消未完成的**。run_pipeline 抛出去之后它们还在后台烧 token，事件循环关闭时
      asyncio 会印一串 "Task was destroyed but it is pending"。
    - **把已完成但没人 await 的异常取回来**。预取窗口里某一集先炸了、而循环还没走到
      它就因为别的原因退出时，那个异常会一直挂在 task 上，GC 时 asyncio 印
      "Task exception was never retrieved" —— 用户以为又炸了第二次。

    gather(return_exceptions=True) 一次把两类都吃掉。正常跑完的情况下每个 task 都已经
    被 await 过，gather 立刻返回缓存好的结果。
    """
    live = [task for task in tasks.values() if task is not None]
    for task in live:
        if not task.done():
            task.cancel()
    if live:
        await asyncio.gather(*live, return_exceptions=True)


async def run_pipeline(
    cfg: ProjectConfig,
    provider: LLMProvider,
    *,
    from_stage: str = "ingest",
    only: Sequence[str] | None = None,
    force: bool = False,
    tts_engine: TTSEngine | None = None,
    episode: int | None = None,
    reporter: ProgressReporter | None = None,
) -> list[str]:
    """返回本次运行累积的 warnings。

    only 传阶段名序列，只跑这些阶段（CLI 的 --only 传单元素列表，
    端到端测试会传 ["ingest", "signals"] 这样的多元素列表）。

    episode 不传（None）时是批量模式：script/docgen/voice/timeline/audio/render
    都会对 cfg.episodes 里注册的每一集分别跑一遍。传了具体集数时是单集模式：
    只处理这一集（ingest/signals 仍然是全局阶段，一直处理所有已注册的集）。

    **script 阶段的多集并发**（`llm.script_concurrency`）：编排保持「按集纵向」，
    只给 script 加一个**有界预取窗口** —— 走到第 i 集时确保前 `i + concurrency` 集的
    script task 都已经起了，然后 await 第 i 集那个。本次运行含 translate 时窗口被钳到 1
    （术语表是 script 的输入却由循环里的 translate 写）。取舍见 `_launch_scripts` 的注释。
    """
    if cfg.mode == "season":
        raise NotImplementedError("整季模式尚未实现，请使用 mode: single_episode")

    reporter = reporter or NullProgressReporter()

    wanted = resolve_stages(from_stage=from_stage, only=only)

    paths = Paths(cfg.root)
    active = _active_episodes(cfg)
    numbers = [ep.number for ep in active]
    ingest_inputs = _ingest_inputs(cfg)
    warnings: list[str] = []

    if episode is None:
        # 预填条目在批处理里跳过，但不能静默：用户会以为那一集跑过了。只在批处理模式下
        # 提示 —— 单集模式跑的是别的集，报一句无关的集号只是噪音。
        warnings.extend(
            f"第 {ep.number} 集还没有 video，已跳过"
            for ep in cfg.episodes
            if not ep.has_source
        )
        target_numbers = numbers
    else:
        _find_episode(cfg, episode)  # 找不到会抛 ValueError（"没有注册"）
        target_numbers = [episode]

    # 开跑前把每个阶段读到的那部分配置落成切片（内容没变不碰文件）。放在集号解析之后：
    # --episode 指到一个没注册 / 预填的集时应该先报错，而不是先往磁盘上写东西。
    config_slices.write_slices(cfg, active)

    def is_fresh(
        stage: str, number: int | None, outputs: list[Path], inputs: list[Path]
    ) -> bool:
        """每个阶段的判据都自动带上它自己的配置切片（替换原来的整个 project.yaml）。

        漏掉切片的话「改配置再重跑」会被 stage_skip，用户拿到的产物跟改动前一模一样且
        没有任何提示；反过来把整个 project.yaml 放进来，则是改任何一个旋钮都整季重跑。
        """
        return _is_fresh(
            outputs, [config_slices.slice_path(cfg.root, stage, number), *inputs]
        )

    number_to_index = {number: idx for idx, number in enumerate(target_numbers, start=1)}

    if "ingest" in wanted:
        outputs = [paths.dialogue(n) for n in numbers]
        # ingest 是全局阶段，但切片按集：每一集的切片都是它的输入。
        inputs = [
            *ingest_inputs,
            *(config_slices.slice_path(cfg.root, "ingest", n) for n in numbers),
        ]
        stamp = config_slices.stamp_path(cfg.root, "ingest")
        if force or not _is_fresh_stamped(outputs, inputs, stamp):
            reporter.stage_start("ingest")
            config_slices.invalidate_stamp(stamp)
            warnings.extend(ingest_warnings(run_ingest(cfg)))
            config_slices.touch_stamp(stamp, outputs)
            reporter.stage_done("ingest")
        else:
            reporter.stage_skip("ingest")

    if "signals" in wanted:
        outputs = [paths.signals(n) for n in numbers]
        inputs = [
            config_slices.slice_path(cfg.root, "signals", None),
            *(paths.dialogue(n) for n in numbers),
        ]
        stamp = config_slices.stamp_path(cfg.root, "signals")
        if force or not _is_fresh_stamped(outputs, inputs, stamp):
            reporter.stage_start("signals")
            config_slices.invalidate_stamp(stamp)
            run_signals(cfg)
            config_slices.touch_stamp(stamp, outputs)
            reporter.stage_done("signals")
        else:
            reporter.stage_skip("signals")

    # 前置检查放在**所有**按集阶段之前：绝不能跑完几分钟 TTS（更别说一轮 LLM），最后
    # 一步才发现 ffmpeg 没编 libass 或某一集的源视频缺失。
    # 只在真的要跑 audio/render 时才做（跟原逻辑一致：单独跑 voice 不该触发 ffmpeg 检查）。
    # 批量模式下要给每一集都做前置检查，不能只查第一集——否则第二集视频缺失/坏掉
    # 要等它自己的 voice 阶段（几分钟 TTS）跑完才会在 audio/render 阶段炸出来，
    # 失去 preflight 本来该有的「快速失败」意义。
    #
    # 位置从「script/docgen 循环之后、voice 循环之前」提到了循环之前：原先那两个循环
    # 各自调 reporter.episode_start，批量模式下每集被报两次，总进度条先 0→N 再跳回
    # 0→N。合并成单循环就必须把 preflight 挪出去，而挪到前面严格更好——检查本身很便宜
    # （ffmpeg -filters/-encoders 有 lru_cache，加一次 ffprobe），却能在花掉任何 LLM
    # token 之前就把「ffmpeg 不行」喊出来。
    if {"audio", "render"} & set(wanted):
        # drawtext 与片尾卡字体都只在卡片开着时才用到，关掉卡片的用户不该被它们挡住。
        needs_outro = cfg.render.outro_card_seconds > 0
        font_names = [cfg.render.subtitle_font_name]
        if needs_outro:
            font_names.append(cfg.render.outro_font_name)
        for number in target_numbers:
            preflight(
                cfg.video_path(_find_episode(cfg, number)),
                cfg.render.video_encoder,
                ffmpeg=cfg.render.ffmpeg_path,
                ffprobe=cfg.render.ffprobe_path,
                needs_drawtext=needs_outro,
                font_names=font_names,
                # 字体缺失只是 warning（fontconfig 会静默替换，纯外观问题）。本项目不引入
                # logging，warnings 是这类诊断唯一的出口，所以必须把列表递进去。
                warnings=warnings,
            )
    elif "timeline" in wanted:
        # timeline 也读源片（时长和帧率）；只跑 timeline 时不会进上面的 preflight。
        # 必须在 script/voice 之前检查，并且即使已有新鲜产物也要检查，不能让
        # _is_fresh 对缺失输入的过滤把坏片源变成 stage_skip。只探一次时长确认可读，
        # 不重复做音轨/编码器/字体等仅 audio/render 需要的前置检查。
        for number in target_numbers:
            episode_cfg = _find_episode(cfg, number)
            if episode_cfg.video is not None:
                video = cfg.video_path(episode_cfg)
                if not video.is_file():
                    raise FileNotFoundError(
                        f"找不到源视频 {video}，请检查 project.yaml 的 episodes[].video"
                    )
                probe_duration(video, ffprobe=cfg.render.ffprobe_path)

    # --- script 阶段的有界预取 ---
    # 值为 None 表示「这一集的 script 已经最新，不用跑」（跟「还没决定」区分开，
    # 后者是 key 压根不在 dict 里）。
    script_tasks: dict[int, asyncio.Task[tuple[Script, list[str]]] | None] = {}

    def _launch_scripts(through: int) -> None:
        """给 target_numbers 的前 through 集把 script task 起起来（已起过的跳过）。

        为什么是「有界预取」而不是「把 script 整个抽成横向阶段」：

        批量模式刻意是「按集纵向」而不是「按阶段横向」，理由是**中途失败要留下
        完整交付物、而不是一堆半成品**。把 script 抽成横向的并发阶段会直接
        推翻它：全部集的 script 跑完之前一集成片都不会有，而 script 恰好是最容易失败、
        也最慢的那一个阶段（实测单次调用 561 秒）—— 十集批量跑到第九集炸掉，用户手上
        是 8 份 script.json 和 0 个 mp4。

        有界预取两头都要：**纵向循环的顺序一个字没改**（第 i 集的 docgen/voice/…/render
        仍然紧跟着它自己的 script，所以第 k 集失败时前 k-1 集都是完整成片），同时前面几集
        跑 ffmpeg 的时候后面几集的 LLM 会话已经在飞了。窗口宽度就是并发度，所以
        concurrency=1 时窗口只有「当前这一集」，等价于原来那句直接 await。

        代价（只在 concurrency > 1 时付）：失败时预取窗口里在飞的那几集会被取消，
        那几次 LLM 调用的 token 白花了。串行下它们压根不会发出去。这也是默认值取 1 的
        三条依据之一，另两条见 config.LLMConfig.script_concurrency。

        新鲜度在**起 task 的时刻**判，比原来早了几集。这只在「这一集的判据与产物没有
        任何别的集能动」时才等价：dialogue/signals 都由循环之前的全局阶段写完了，
        配置切片在开跑前就写完了，paths.script(n) 也只会被第 n 集自己的 task 写。

        累积术语表是这条等价性唯一的破口 —— 它在 _script_inputs 里，却是**纵向循环里**
        的 translate 写的。所以本次运行含 translate 时窗口被钳到 1（见调用处）：不钳的话
        后面几集的 script 会在自己的 translate 之前既判新鲜度又执行，可能被判 fresh 而
        跳过（解说稿用旧译名），也可能 load_glossary 读到还缺本集术语的表 —— 非确定性。
        """
        for number in target_numbers[:through]:
            if number in script_tasks:
                continue
            inputs = _script_inputs(paths, number)
            if force or not is_fresh("script", number, [paths.script(number)], inputs):
                script_tasks[number] = asyncio.create_task(
                    run_script(
                        cfg,
                        provider,
                        episode=number,
                        reporter=_EpisodeLabelledReporter(reporter, number),
                    )
                )
            else:
                script_tasks[number] = None

    # 含 translate 的运行里预取窗口只能是「当前这一集」：术语表由纵向循环里的 translate
    # 写，而它是 script 的新鲜度输入之一，跨集预取会让判据取决于「循环走到哪了」。
    # 钳成 1 是保守的（繁中片源上 translate 是零动作，术语表压根不会出现），但换来的是
    # 确定性；想在批量重跑解说稿时拿回并发就用 --from script / --only script。
    script_window = 1 if "translate" in wanted else max(1, cfg.llm.script_concurrency)

    # 预取的 task 必须被这个 try/finally 完整包住：循环里**任何**阶段抛异常时
    # （不只是 script 自己），还在飞的那几集都得取消掉。
    try:
        for position, number in enumerate(target_numbers):
            if episode is None:
                reporter.episode_start(number, number_to_index[number], len(target_numbers))

            # 原片可以缺席于只跑 SRT 的旧项目；配置了视频时，各消费它的阶段都要
            # 独立依赖它（--only audio/render 不一定会先跑 timeline）。
            episode_cfg = _find_episode(cfg, number)
            video_inputs = (
                [cfg.video_path(episode_cfg)] if episode_cfg.video is not None else []
            )

            if "translate" in wanted:
                # **刻意不做多集并发**（script 上面那个有界预取窗口不往这里搬）：第 1 集
                # 写完累积术语表、第 2 集才读到含第 1 集的版本，并发会让累积失去意义，
                # 而且两个 task 会同时回写同一个文件。
                outputs = [paths.zh_lines(number), paths.zh_subtitles(number)]
                if force or not is_fresh(
                    "translate", number, outputs, _translate_inputs(paths, number)
                ):
                    reporter.stage_start("translate")
                    await run_translate(cfg, provider, number)
                    reporter.stage_done("translate")
                else:
                    reporter.stage_skip("translate")

            if "script" in wanted:
                _launch_scripts(position + script_window)
                task = script_tasks[number]
                if task is None:
                    reporter.stage_skip("script")
                else:
                    reporter.stage_start("script")
                    _, stage_warnings = await task
                    warnings.extend(stage_warnings)
                    reporter.stage_done("script")

            if "docgen" in wanted:
                outputs = [paths.table(number), paths.narration(number)]
                if force or not is_fresh("docgen", number, outputs, [paths.script(number)]):
                    reporter.stage_start("docgen")
                    run_docgen(cfg, episode=number)
                    reporter.stage_done("docgen")
                else:
                    reporter.stage_skip("docgen")

            if "voice" in wanted:
                outputs = [paths.voice(number)]
                if force or not is_fresh("voice", number, outputs, [paths.script(number)]):
                    reporter.stage_start("voice")
                    if tts_engine is None:
                        # 刻意不用 assert：python -O 下 assert 整句被剥离，None 会一路漂
                        # 进 synthesize_track，最后炸成 render/tts.py 里的 AttributeError，
                        # 报错指不到真正的原因。ValueError 在 cli.py 的捕获列表里，用户看到
                        # 的是一行红字而不是一整页 traceback。
                        raise ValueError(
                            "要跑 voice 阶段必须传 tts_engine。"
                            "CLI 会在 --only/--from 覆盖到 voice 时自动构造，"
                            "库调用方请自己传 render.tts.build_tts_engine(cfg.render)。"
                        )
                    _, stage_warnings = await run_voice(
                        cfg, tts_engine, episode=number, reporter=reporter
                    )
                    warnings.extend(stage_warnings)
                    reporter.stage_done("voice")
                else:
                    reporter.stage_skip("voice")

            if "timeline" in wanted:
                outputs = [paths.timeline(number), paths.subtitles(number)]
                inputs = [paths.script(number), paths.voice(number), *video_inputs]
                if force or not is_fresh("timeline", number, outputs, inputs):
                    reporter.stage_start("timeline")
                    _, stage_warnings = run_timeline(cfg, episode=number)
                    warnings.extend(stage_warnings)
                    reporter.stage_done("timeline")
                else:
                    reporter.stage_skip("timeline")

            if "audio" in wanted:
                outputs = [paths.mixed_audio(number)]
                inputs = [paths.timeline(number), paths.voice(number), *video_inputs]
                if force or not is_fresh("audio", number, outputs, inputs):
                    reporter.stage_start("audio")
                    run_audio(cfg, episode=number, reporter=reporter)
                    reporter.stage_done("audio")
                else:
                    reporter.stage_skip("audio")

            if "render" in wanted:
                outputs = [paths.video(number)]
                inputs = [
                    paths.mixed_audio(number),
                    paths.subtitles(number),
                    paths.timeline(number),
                    *video_inputs,
                ]
                if force or not is_fresh("render", number, outputs, inputs):
                    reporter.stage_start("render")
                    run_render(cfg, episode=number, reporter=reporter)
                    reporter.stage_done("render")
                else:
                    reporter.stage_skip("render")

            if episode is None:
                reporter.episode_done(number, number_to_index[number], len(target_numbers))
    finally:
        await _drain_script_tasks(script_tasks)

    return warnings
