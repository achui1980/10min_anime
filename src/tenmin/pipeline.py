"""阶段编排。每个阶段读上游文件、写自己的文件，靠 mtime 决定是否跳过。"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import yaml

from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.docgen.narration import render_narration
from tenmin.docgen.table import render_table
from tenmin.ingest.normalize import build_track
from tenmin.models import DialogueTrack, Script, SignalReport, Timeline, VoiceTrack
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.audio import mix_audio
from tenmin.render.ffmpeg import FFmpegError, preflight, probe_duration
from tenmin.render.subtitles import render_ass
from tenmin.render.timeline import build_timeline
from tenmin.render.tts import TTSEngine, synthesize_track
from tenmin.render.video import render_video
from tenmin.script.llm import LLMProvider
from tenmin.script.single import generate_script
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


class Paths:
    def __init__(self, root: Path):
        self.root = Path(root)

    def dialogue(self, episode: int) -> Path:
        return self.root / "01_dialogue" / f"E{episode:02d}.dialogue.json"

    def signals(self, episode: int) -> Path:
        return self.root / "02_signals" / f"E{episode:02d}.signals.json"

    def script(self, episode: int) -> Path:
        return self.root / "03_script" / f"E{episode:02d}.script.json"

    def table(self, episode: int) -> Path:
        return self.root / "out" / f"E{episode:02d}.解说方案.md"

    def narration(self, episode: int) -> Path:
        return self.root / "out" / f"E{episode:02d}.narration.txt"

    def voice_dir(self, episode: int) -> Path:
        return self.root / "04_voice" / f"E{episode:02d}"

    def voice(self, episode: int) -> Path:
        return self.root / "04_voice" / f"E{episode:02d}.voice.json"

    def timeline(self, episode: int) -> Path:
        return self.root / "05_timeline" / f"E{episode:02d}.timeline.json"

    def subtitles(self, episode: int) -> Path:
        return self.root / "05_timeline" / f"E{episode:02d}.ass"

    def mixed_audio(self, episode: int) -> Path:
        return self.root / "06_audio" / f"E{episode:02d}.mixed.m4a"

    def video(self, episode: int) -> Path:
        return self.root / "07_render" / f"E{episode:02d}.mp4"


def stages_from(stage: str) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"未知阶段 {stage!r}，可选：{', '.join(STAGES)}")
    return STAGES[STAGES.index(stage) :]


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        return probe_duration(path)
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
    cfg: ProjectConfig, provider: LLMProvider, episode: int
) -> tuple[Script, list[str]]:
    tracks = _load_tracks(cfg)
    reports = _load_reports(cfg)
    track = next(t for t in tracks if t.episode == episode)
    report = next(r for r in reports if r.episode == episode)
    script, warnings = await generate_script(cfg, track, report, provider)
    _write_json(Paths(cfg.root).script(episode), script.model_dump_json(indent=2))
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
    """把外部传入的 srt/video 拷进项目目录，并把这一集写进 project.yaml。

    如果这一集已经注册过，就覆盖 srt/video 路径（保留其它字段）；
    否则追加一条新的 episode 记录。返回更新后的 ProjectConfig（root 已绑定）。
    """
    srt_dest = cfg.root / "srt" / f"E{episode:02d}.srt"
    video_dest = cfg.root / "video" / f"E{episode:02d}.mp4"
    srt_dest.parent.mkdir(parents=True, exist_ok=True)
    video_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(srt, srt_dest)
    shutil.copyfile(video, video_dest)

    relative_srt = srt_dest.relative_to(cfg.root)
    relative_video = video_dest.relative_to(cfg.root)

    existing = next((e for e in cfg.episodes if e.number == episode), None)
    if existing is not None:
        existing.srt = relative_srt
        existing.video = relative_video
    else:
        cfg.episodes.append(
            EpisodeConfig(number=episode, srt=relative_srt, video=relative_video)
        )

    yaml_path = cfg.root / "project.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    # 用每个 EpisodeConfig 自己的 model_dump 序列化，而不是手挑 number/srt/video，
    # 这样 op_range/ed_range 等字段（现有的和未来新增的）都不会在改写 yaml 时被静默丢掉。
    data["episodes"] = [
        e.model_dump(exclude_none=True, mode="json") for e in cfg.episodes
    ]
    yaml_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
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


async def run_voice(
    cfg: ProjectConfig,
    engine: TTSEngine,
    episode: int,
    reporter: ProgressReporter | None = None,
) -> tuple[VoiceTrack, list[str]]:
    paths = Paths(cfg.root)
    script = _load_script(cfg, episode)
    track, warnings = await synthesize_track(
        script, episode, paths.voice_dir(episode), engine, reporter=reporter
    )
    _write_json(paths.voice(episode), track.model_dump_json(indent=2))
    return track, warnings


def run_timeline(
    cfg: ProjectConfig, episode: int, *, source_duration: float | None = None
) -> tuple[Timeline, list[str]]:
    paths = Paths(cfg.root)
    episode_cfg = _find_episode(cfg, episode)
    script = _load_script(cfg, episode)
    track = _load_voice(cfg, episode)
    if source_duration is None:
        source_duration = probe_duration(cfg.video_path(episode_cfg))
    timeline, warnings = build_timeline(script, track, source_duration)
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


def run_audio(cfg: ProjectConfig, episode: int) -> Path:
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
        fade_out_seconds=cfg.render.fade_out_seconds,
        outro_seconds=cfg.render.outro_card_seconds,
        outro_title=f"{cfg.show} · EP{episode:02d}",
        outro_message=cfg.render.outro_message,
        reporter=reporter,
    )


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
    """
    if cfg.mode == "season":
        raise NotImplementedError("整季模式尚未实现，请使用 mode: single_episode")

    reporter = reporter or NullProgressReporter()

    if only is not None:
        unknown = [stage for stage in only if stage not in STAGES]
        if unknown:
            raise ValueError(f"未知阶段 {unknown[0]!r}，可选：{', '.join(STAGES)}")
        wanted = [stage for stage in STAGES if stage in only]
    else:
        wanted = stages_from(from_stage)

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

    for number in target_numbers:
        if episode is None:
            reporter.episode_start(number, number_to_index[number], len(target_numbers))

        if "script" in wanted:
            inputs = [paths.dialogue(number), paths.signals(number)]
            if force or not is_fresh([paths.script(number)], inputs):
                reporter.stage_start("script")
                _, stage_warnings = await run_script(cfg, provider, episode=number)
                warnings.extend(stage_warnings)
                reporter.stage_done("script")
            else:
                reporter.stage_skip("script")

        if "docgen" in wanted:
            outputs = [paths.table(number), paths.narration(number)]
            if force or not is_fresh(outputs, [paths.script(number)]):
                reporter.stage_start("docgen")
                run_docgen(cfg, episode=number)
                reporter.stage_done("docgen")
            else:
                reporter.stage_skip("docgen")

    # 前置检查放在 voice 之前：绝不能跑完几分钟 TTS，最后一步才发现 ffmpeg 没编 libass。
    # 只在真的要跑 audio/render 时才做（跟原逻辑一致：单独跑 voice 不该触发 ffmpeg 检查）。
    # 批量模式下要给每一集都做前置检查，不能只查第一集——否则第二集视频缺失/坏掉
    # 要等它自己的 voice 阶段（几分钟 TTS）跑完才会在 audio/render 阶段炸出来，
    # 失去 preflight 本来该有的「快速失败」意义。
    if {"audio", "render"} & set(wanted):
        for number in target_numbers:
            preflight(cfg.video_path(_find_episode(cfg, number)), cfg.render.video_encoder)

    for number in target_numbers:
        if episode is None:
            reporter.episode_start(number, number_to_index[number], len(target_numbers))

        if "voice" in wanted:
            outputs = [paths.voice(number)]
            if force or not is_fresh(outputs, [paths.script(number)]):
                reporter.stage_start("voice")
                assert tts_engine is not None
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
                run_audio(cfg, episode=number)
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

    return warnings
