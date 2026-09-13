"""阶段编排。每个阶段读上游文件、写自己的文件，靠 mtime 决定是否跳过。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import yaml

from tenmin import atomic
from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.docgen.narration import render_narration
from tenmin.docgen.table import render_table
from tenmin.ingest.normalize import build_track
from tenmin.models import DialogueTrack, Script, SignalReport, Timeline, VoiceTrack
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.audio import mix_audio
from tenmin.render.ffmpeg import (
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
from tenmin.script.validate import ScriptValidationError
from tenmin.signals.aggregate import build_report

STAGES = [
    "ingest",
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
# out/ 不带序号，因为它是给人看的交付物目录，不是中间产物。
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
    "table": ("out", ".解说方案.md"),
    "narration": ("out", ".narration.txt"),
    "voice_dir": ("04_voice", ""),
    "voice": ("04_voice", ".voice.json"),
    "timeline": ("05_timeline", ".timeline.json"),
    "subtitles": ("05_timeline", ".ass"),
    "mixed_audio": ("06_audio", ".mixed.m4a"),
    "video": ("07_render", ".mp4"),
}


class Paths:
    """一个 project 的全部阶段产物路径。

    13 个方法都是 _ARTIFACTS 表的一行薄包装。刻意保留显式方法而不是 __getattr__
    动态派发：调用点（pipeline / cli / 一堆测试）到处在用 paths.script(2)，
    动态派发会让拼错的名字变成运行时 AttributeError、IDE 跳转与补全全失效。
    这里要的是「布局知识只有一份」，不是「代码行数最少」。
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

    def table(self, episode: int) -> Path:
        return self._artifact("table", episode)

    def narration(self, episode: int) -> Path:
        return self._artifact("narration", episode)

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

    1. **前提是「产物要么不存在、要么完整」**。判据只有 mtime，而被 Ctrl-C 打断的
       ffmpeg / TTS 会留下一个 mtime 恰好最新的半截文件，纯 mtime 比较必然把它当成
       最新产物直接跳过，坏产物一路进成片。正解是产物原子写（临时文件 + os.replace），
       那是 P1-G 的范围，不在这里做。这里只加一条最低成本的兜底：**0 字节产物一律
       视为不新鲜**。本流水线没有任何一个阶段会合法地产出空文件（json/md/txt/m4a/mp4
       都有内容），所以这条规则不会误伤；它挡得住「刚 open 就被打断」这一类，挡不住
       「写了一半」——后者只能靠 P1-G。
    2. **inputs 必须包含 project.yaml**（调用点用 cfg.config_path 传进来）。所有阶段
       的行为都由它决定，漏了它就等于所有配置旋钮改了都不生效。
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
        # 或产出垃圾，而跳过至少保住磁盘上已有的产物。加了 project.yaml 进 inputs 之后
        # 这个分支在真实项目里已经走不到（project.yaml 必然存在，否则 load_project 就
        # 报错了），留着只为不给库调用方/单测埋 FileNotFoundError。
        return True
    newest_input = max(p.stat().st_mtime_ns for p in existing_inputs)
    oldest_output = min(p.stat().st_mtime_ns for p in outputs)
    return oldest_output >= newest_input


def _source_duration(cfg: ProjectConfig, episode: EpisodeConfig) -> float | None:
    """尽力拿这一集源视频的真实片长；拿不到就返回 None（让调用方退化到字幕末尾）。

    刻意把所有失败都吞成 None：ingest 是唯一不需要视频的阶段，「只有 SRT」是文档里
    写明的合法用法（`tenmin run <slug> --only ingest` 就能出对白轨与信号），绝不能
    因为视频缺失/没装 ffprobe 就把它打死。三类失败：

    - ValueError：project.yaml 的这一集没写 video（video_path 自己抛的）。
    - 文件不在：写了 video 但文件还没到位（下载中、换过外置盘）。
    - FFmpegError / OSError：文件在但 ffprobe 读不出（0 字节壳子、不是视频、
      ffprobe 不在 PATH 上）。

    降级是静默的：退化后的 duration 会照常写进 dialogue.json，`tenmin inspect`
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
    except (FFmpegError, OSError):
        return None


def run_ingest(cfg: ProjectConfig) -> list[DialogueTrack]:
    paths = Paths(cfg.root)
    tracks = []
    for episode in cfg.episodes:
        track = build_track(
            cfg.srt_path(episode),
            episode=episode.number,
            glossary=cfg.glossary,
            convert_traditional=cfg.locale.convert_traditional,
            show_title=cfg.show,
            op_range=episode.op_range,
            ed_range=episode.ed_range,
            duration=_source_duration(cfg, episode),
            ingest=cfg.ingest,
            credits=cfg.credits,
        )
        _write_json(paths.dialogue(episode.number), track.model_dump_json(indent=2))
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
    for episode in cfg.episodes:
        path = paths.dialogue(episode.number)
        if not path.exists():
            raise FileNotFoundError(f"缺少对白轨产物 {path}，请先跑 ingest 阶段")
        tracks.append(DialogueTrack.model_validate_json(path.read_text(encoding="utf-8")))
    return tracks


def run_signals(cfg: ProjectConfig) -> list[SignalReport]:
    paths = Paths(cfg.root)
    reports = []
    for track in _load_tracks(cfg):
        report = build_report(track, cfg=cfg.signals)
        _write_json(paths.signals(track.episode), report.model_dump_json(indent=2))
        reports.append(report)
    return reports


def _load_reports(cfg: ProjectConfig) -> list[SignalReport]:
    paths = Paths(cfg.root)
    reports = []
    for episode in cfg.episodes:
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
    try:
        script, warnings = await generate_script(
            cfg, track, report, provider, reporter=reporter
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
    _write_json(paths.script(episode), script.model_dump_json(indent=2))
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
            return episode
    raise ValueError(
        f"第 {episode_number} 集还没有注册。"
        f"请先用 `tenmin run <slug> --episode {episode_number} "
        "--srt <srt路径> --video <视频路径>` 注册这一集。"
    )


def register_episode(
    cfg: ProjectConfig, *, episode: int, srt: Path, video: Path
) -> ProjectConfig:
    """把这一集登记进 project.yaml：SRT 拷进项目目录，视频只记路径不拷。

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

    向后兼容：只有**本次登记的这一集**会被写成绝对路径。其余集的 srt/video 原样
    走各自 EpisodeConfig 的 model_dump 落盘，存量的相对路径（work/saijo/ 下 10 个
    已经拷好的 mp4）逐字节不变，video_path() 照旧按 project.yaml 所在目录解析。
    """
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
        existing.srt = relative_srt
        existing.video = video_source
    else:
        cfg.episodes.append(
            EpisodeConfig(number=episode, srt=relative_srt, video=video_source)
        )

    yaml_path = cfg.config_path
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    # 用每个 EpisodeConfig 自己的 model_dump 序列化，而不是手挑 number/srt/video，
    # 这样 op_range/ed_range 等字段（现有的和未来新增的）都不会在改写 yaml 时被静默丢掉。
    data["episodes"] = [
        e.model_dump(exclude_none=True, mode="json") for e in cfg.episodes
    ]
    # 原子写：project.yaml 是**每个阶段**的隐式输入（_is_fresh 把它加进 inputs），
    # 而这里是**改写**一个已有文件 —— 中途被打断会把用户已注册的全部集数毁掉。
    atomic.write_text(
        yaml_path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    )

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
    传了 source_duration 就说明「调用方自己在管源片探测」（run_pipeline 复用刚跑过的
    preflight 结果、只有 SRT 没有视频的降级路径、单测手上只有一个空壳文件），那时不再
    去碰 video_path —— 那条路上压根没有可探的视频。

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
    # 可读性检查（P2-E C2）住在 render/subtitles.py —— 只有它知道字号与画布宽度算出来
    # 的行数。接在这里而不是 build_timeline 里：render/timeline.py 压根不认识字体，而
    # 这个函数手上同时有 cfg.render 与 warnings。
    warnings.extend(
        check_cue_legibility(
            timeline.subtitles,
            font_size=cfg.render.font_size,
            width=cfg.render.width,
            max_lines=cfg.render.subtitle_max_lines,
            min_seconds=cfg.render.subtitle_min_seconds,
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

    **script 阶段的多集并发**（`llm.script_concurrency`）：编排保持 P0-C 的「按集纵向」，
    只给 script 加一个**有界预取窗口** —— 走到第 i 集时确保前 `i + concurrency` 集的
    script task 都已经起了，然后 await 第 i 集那个。取舍见 `_launch_scripts` 的注释。
    """
    if cfg.mode == "season":
        raise NotImplementedError("整季模式尚未实现，请使用 mode: single_episode")

    reporter = reporter or NullProgressReporter()

    wanted = resolve_stages(from_stage=from_stage, only=only)

    paths = Paths(cfg.root)
    numbers = [ep.number for ep in cfg.episodes]
    srt_inputs = [cfg.srt_path(ep) for ep in cfg.episodes]
    warnings: list[str] = []

    def is_fresh(outputs: list[Path], inputs: list[Path]) -> bool:
        """每个阶段的判据都自动带上 project.yaml。

        它是所有阶段的隐式输入：ingest/credits/signals 的全部阈值、glossary、
        render 的字号与编码器都住在那里。漏掉它的话「改配置再重跑」会被全部
        stage_skip，用户拿到的产物跟改动前一模一样且没有任何提示。
        """
        return _is_fresh(outputs, [cfg.config_path, *inputs])

    if episode is None:
        target_numbers = numbers
    else:
        _find_episode(cfg, episode)  # 找不到会抛 ValueError（"没有注册"）
        target_numbers = [episode]

    number_to_index = {number: idx for idx, number in enumerate(target_numbers, start=1)}

    if "ingest" in wanted:
        outputs = [paths.dialogue(n) for n in numbers]
        if force or not is_fresh(outputs, srt_inputs):
            reporter.stage_start("ingest")
            warnings.extend(ingest_warnings(run_ingest(cfg)))
            reporter.stage_done("ingest")
        else:
            reporter.stage_skip("ingest")

    if "signals" in wanted:
        outputs = [paths.signals(n) for n in numbers]
        inputs = [paths.dialogue(n) for n in numbers]
        if force or not is_fresh(outputs, inputs):
            reporter.stage_start("signals")
            run_signals(cfg)
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

    # --- script 阶段的有界预取 ---
    # 值为 None 表示「这一集的 script 已经最新，不用跑」（跟「还没决定」区分开，
    # 后者是 key 压根不在 dict 里）。
    script_tasks: dict[int, asyncio.Task[tuple[Script, list[str]]] | None] = {}

    def _launch_scripts(through: int) -> None:
        """给 target_numbers 的前 through 集把 script task 起起来（已起过的跳过）。

        为什么是「有界预取」而不是「把 script 整个抽成横向阶段」：

        P0-C 刻意把批量模式从「按阶段横向」改成「按集纵向」，理由是**中途失败要留下
        完整交付物、而不是一堆半成品**。把 script 抽成横向的并发阶段（选项 a）会直接
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

        新鲜度在**起 task 的时刻**判，比原来早了几集。等价性：script 阶段的输入是
        dialogue/signals（都由循环之前的全局阶段写完了）加 project.yaml，而
        paths.script(n) 只会被第 n 集自己的 task 写 —— 没有任何一集能改变另一集的判据。
        """
        for number in target_numbers[:through]:
            if number in script_tasks:
                continue
            inputs = [paths.dialogue(number), paths.signals(number)]
            if force or not is_fresh([paths.script(number)], inputs):
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

    # 预取的 task 必须被这个 try/finally 完整包住：循环里**任何**阶段抛异常时
    # （不只是 script 自己），还在飞的那几集都得取消掉。
    try:
        for position, number in enumerate(target_numbers):
            if episode is None:
                reporter.episode_start(number, number_to_index[number], len(target_numbers))

            if "script" in wanted:
                _launch_scripts(position + max(1, cfg.llm.script_concurrency))
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
                if force or not is_fresh(outputs, [paths.script(number)]):
                    reporter.stage_start("docgen")
                    run_docgen(cfg, episode=number)
                    reporter.stage_done("docgen")
                else:
                    reporter.stage_skip("docgen")

            if "voice" in wanted:
                outputs = [paths.voice(number)]
                if force or not is_fresh(outputs, [paths.script(number)]):
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
                inputs = [paths.script(number), paths.voice(number)]
                if force or not is_fresh(outputs, inputs):
                    reporter.stage_start("timeline")
                    _, stage_warnings = run_timeline(cfg, episode=number)
                    warnings.extend(stage_warnings)
                    reporter.stage_done("timeline")
                else:
                    reporter.stage_skip("timeline")

            if "audio" in wanted:
                outputs = [paths.mixed_audio(number)]
                inputs = [paths.timeline(number), paths.voice(number)]
                if force or not is_fresh(outputs, inputs):
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
                ]
                if force or not is_fresh(outputs, inputs):
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
