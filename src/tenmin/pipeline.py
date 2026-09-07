"""阶段编排。每个阶段读上游文件、写自己的文件，靠 mtime 决定是否跳过。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.docgen.narration import render_narration
from tenmin.docgen.table import render_table
from tenmin.ingest.normalize import build_track
from tenmin.models import DialogueTrack, Script, SignalReport, Timeline, VoiceTrack
from tenmin.render.audio import mix_audio
from tenmin.render.ffmpeg import preflight, probe_duration
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

    @property
    def script(self) -> Path:
        return self.root / "03_script" / "script.json"

    @property
    def table(self) -> Path:
        return self.root / "out" / "解说方案.md"

    @property
    def narration(self) -> Path:
        return self.root / "out" / "narration.txt"

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
    if not outputs or any(not p.exists() for p in outputs):
        return False
    existing_inputs = [p for p in inputs if p.exists()]
    if not existing_inputs:
        return True
    newest_input = max(p.stat().st_mtime_ns for p in existing_inputs)
    oldest_output = min(p.stat().st_mtime_ns for p in outputs)
    return oldest_output >= newest_input


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
        )
        _write_json(paths.dialogue(episode.number), track.model_dump_json(indent=2))
        tracks.append(track)
    return tracks


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
        report = build_report(track)
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


async def run_script(cfg: ProjectConfig, provider: LLMProvider) -> tuple[Script, list[str]]:
    tracks = _load_tracks(cfg)
    reports = _load_reports(cfg)
    script, warnings = await generate_script(cfg, tracks[0], reports[0], provider)
    _write_json(Paths(cfg.root).script, script.model_dump_json(indent=2))
    return script, warnings


def run_docgen(cfg: ProjectConfig) -> Script:
    paths = Paths(cfg.root)
    if not paths.script.exists():
        raise FileNotFoundError(f"缺少剧本产物 {paths.script}，请先跑 script 阶段")
    script = Script.model_validate_json(paths.script.read_text(encoding="utf-8"))
    _write_text(paths.table, render_table(script))
    _write_text(paths.narration, render_narration(script))
    return script


def _only_episode(cfg: ProjectConfig) -> EpisodeConfig:
    """v2 只做单集。season 模式在 run_pipeline 入口就被拦掉了。"""
    if not cfg.episodes:
        raise ValueError("project.yaml 的 episodes 是空的，至少要配一集")
    return cfg.episodes[0]


def _load_script(cfg: ProjectConfig) -> Script:
    path = Paths(cfg.root).script
    if not path.exists():
        raise FileNotFoundError(f"缺少剧本产物 {path}，请先跑 script 阶段")
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
    cfg: ProjectConfig, engine: TTSEngine | None
) -> tuple[VoiceTrack, list[str]]:
    if engine is None:
        raise ValueError(
            "voice 阶段需要 TTS engine，请检查 project.yaml 的 render.voice 配置"
        )
    paths = Paths(cfg.root)
    episode = _only_episode(cfg).number
    script = _load_script(cfg)
    track, warnings = await synthesize_track(
        script, episode, paths.voice_dir(episode), engine
    )
    _write_json(paths.voice(episode), track.model_dump_json(indent=2))
    return track, warnings


def run_timeline(
    cfg: ProjectConfig, *, source_duration: float | None = None
) -> tuple[Timeline, list[str]]:
    paths = Paths(cfg.root)
    episode_cfg = _only_episode(cfg)
    episode = episode_cfg.number
    script = _load_script(cfg)
    track = _load_voice(cfg, episode)
    if source_duration is None:
        source_duration = probe_duration(cfg.video_path(episode_cfg))
    timeline, warnings = build_timeline(script, track, source_duration)
    _write_json(paths.timeline(episode), timeline.model_dump_json(indent=2))
    _write_text(
        paths.subtitles(episode),
        render_ass(timeline.subtitles, font_size=cfg.render.font_size),
    )
    return timeline, warnings


def run_audio(cfg: ProjectConfig) -> Path:
    paths = Paths(cfg.root)
    episode_cfg = _only_episode(cfg)
    episode = episode_cfg.number
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


def run_render(cfg: ProjectConfig) -> Path:
    paths = Paths(cfg.root)
    episode_cfg = _only_episode(cfg)
    episode = episode_cfg.number
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
        fade_out_seconds=cfg.render.fade_out_seconds,
        outro_seconds=cfg.render.outro_card_seconds,
        outro_title=f"{cfg.show} · EP{episode:02d}",
        outro_message=cfg.render.outro_message,
    )


async def run_pipeline(
    cfg: ProjectConfig,
    provider: LLMProvider,
    *,
    from_stage: str = "ingest",
    only: Sequence[str] | None = None,
    force: bool = False,
    tts_engine: TTSEngine | None = None,
) -> list[str]:
    """返回本次运行累积的 warnings。

    only 传阶段名序列，只跑这些阶段（CLI 的 --only 传单元素列表，
    端到端测试会传 ["ingest", "signals"] 这样的多元素列表）。
    """
    if cfg.mode == "season":
        raise NotImplementedError("整季模式尚未实现，请使用 mode: single_episode")

    if only is not None:
        unknown = [stage for stage in only if stage not in STAGES]
        if unknown:
            raise ValueError(f"未知阶段 {unknown[0]!r}，可选：{', '.join(STAGES)}")
        wanted = [stage for stage in STAGES if stage in only]
    else:
        wanted = stages_from(from_stage)

    paths = Paths(cfg.root)
    numbers = [episode.number for episode in cfg.episodes]
    srt_inputs = [cfg.srt_path(episode) for episode in cfg.episodes]
    warnings: list[str] = []

    if "ingest" in wanted:
        outputs = [paths.dialogue(n) for n in numbers]
        if force or not _is_fresh(outputs, srt_inputs):
            run_ingest(cfg)

    if "signals" in wanted:
        outputs = [paths.signals(n) for n in numbers]
        inputs = [paths.dialogue(n) for n in numbers]
        if force or not _is_fresh(outputs, inputs):
            run_signals(cfg)

    if "script" in wanted:
        inputs = [paths.dialogue(n) for n in numbers] + [paths.signals(n) for n in numbers]
        if force or not _is_fresh([paths.script], inputs):
            _, stage_warnings = await run_script(cfg, provider)
            warnings.extend(stage_warnings)

    if "docgen" in wanted:
        if force or not _is_fresh([paths.table, paths.narration], [paths.script]):
            run_docgen(cfg)

    # 前置检查放在 voice 之前：绝不能跑完几分钟 TTS，最后一步才发现 ffmpeg 没编 libass。
    if {"audio", "render"} & set(wanted):
        preflight(cfg.video_path(_only_episode(cfg)), cfg.render.video_encoder)

    if "voice" in wanted:
        outputs = [paths.voice(n) for n in numbers]
        if force or not _is_fresh(outputs, [paths.script]):
            _, stage_warnings = await run_voice(cfg, tts_engine)
            warnings.extend(stage_warnings)

    if "timeline" in wanted:
        outputs = [paths.timeline(n) for n in numbers] + [
            paths.subtitles(n) for n in numbers
        ]
        inputs = [paths.script] + [paths.voice(n) for n in numbers]
        if force or not _is_fresh(outputs, inputs):
            _, stage_warnings = run_timeline(cfg)
            warnings.extend(stage_warnings)

    if "audio" in wanted:
        outputs = [paths.mixed_audio(n) for n in numbers]
        inputs = [paths.timeline(n) for n in numbers] + [
            paths.voice(n) for n in numbers
        ]
        if force or not _is_fresh(outputs, inputs):
            run_audio(cfg)

    if "render" in wanted:
        outputs = [paths.video(n) for n in numbers]
        inputs = (
            [paths.mixed_audio(n) for n in numbers]
            + [paths.subtitles(n) for n in numbers]
            + [paths.timeline(n) for n in numbers]
        )
        if force or not _is_fresh(outputs, inputs):
            run_render(cfg)

    return warnings
