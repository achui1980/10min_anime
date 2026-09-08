"""tenmin 命令行入口。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import yaml

from tenmin.config import Settings, load_project
from tenmin.models import DialogueTrack, SignalReport
from tenmin.pipeline import STAGES, Paths, _find_episode, register_episode, run_pipeline
from tenmin.render.ffmpeg import FFmpegError
from tenmin.render.tts import build_tts_engine
from tenmin.script.llm import build_provider
from tenmin.timecode import format_timestamp

app = typer.Typer(add_completion=False, help="把番剧压成解说方案的流水线。")

WORK_DIR_OPTION = typer.Option(Path("work"), "--work-dir", help="项目根目录")

PROJECT_TEMPLATE = {
    "show": "剧名",
    "slug": "slug",
    "mode": "single_episode",
    "target_seconds": 240,
    "locale": {"convert_traditional": True},
    "episodes": [{"number": 2, "srt": "srt/E02.srt", "video": "video/E02.mkv"}],
    "glossary": {},
    "llm": {"provider": "gemini", "model": "gemini-3.6-flash"},
    "render": {
        "voice": "zh-CN-YunxiNeural",
        "rate": "+0%",
        "video_encoder": "libx264",
        "duck_db": -12.0,
        "font_size": 52,
        "fade_out_seconds": 1.5,
        "outro_card_seconds": 3.0,
        "outro_message": "解说结束，谢谢观看",
    },
}


def _project_file(work_dir: Path, slug: str) -> Path:
    path = work_dir / slug / "project.yaml"
    if not path.exists():
        typer.secho(f"找不到 {path}，先跑 tenmin init {slug}", fg="red", err=True)
        raise typer.Exit(code=1)
    return path


@app.command()
def init(slug: str, work_dir: Path = WORK_DIR_OPTION) -> None:
    """创建 work/<slug>/project.yaml 与 srt/ 目录。"""
    root = work_dir / slug
    project_file = root / "project.yaml"
    if project_file.exists():
        typer.secho(f"{project_file} 已存在，不覆盖", fg="red", err=True)
        raise typer.Exit(code=1)

    (root / "srt").mkdir(parents=True, exist_ok=True)
    (root / "video").mkdir(parents=True, exist_ok=True)
    payload = dict(PROJECT_TEMPLATE)
    payload["slug"] = slug
    payload["show"] = slug
    project_file.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    typer.echo(f"已创建 {project_file}")
    typer.echo(f"把字幕放进 {root / 'srt'}、源视频放进 {root / 'video'}，")
    typer.echo(f"改好 project.yaml 后跑 tenmin run {slug}")


@app.command()
def run(
    slug: str,
    work_dir: Path = WORK_DIR_OPTION,
    from_stage: str = typer.Option(
        "ingest", "--from", help=f"从哪个阶段开始，可选：{', '.join(STAGES)}"
    ),
    only: str | None = typer.Option(None, "--only", help="只跑某一个阶段"),
    force: bool = typer.Option(False, "--force", help="忽略 mtime 强制重跑"),
    episode: int | None = typer.Option(
        None, "--episode", help="要处理的集数；配合 --srt/--video 可注册新的一集"
    ),
    srt: Path | None = typer.Option(
        None, "--srt", help="要注册的字幕文件路径，需配合 --episode 和 --video"
    ),
    video: Path | None = typer.Option(
        None, "--video", help="要注册的视频文件路径，需配合 --episode 和 --srt"
    ),
) -> None:
    """跑流水线：ingest -> signals -> script -> docgen -> voice -> timeline -> audio -> render。"""
    cfg = load_project(_project_file(work_dir, slug))

    if (srt is None) != (video is None):
        typer.secho("--srt 和 --video 必须一起传", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if srt is not None and episode is None:
        typer.secho("传 --srt/--video 时必须同时传 --episode", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if srt is not None and video is not None:
        cfg = register_episode(cfg, episode=episode, srt=srt, video=video)

    if episode is not None:
        try:
            _find_episode(cfg, episode)
        except ValueError as error:
            typer.secho(str(error), fg=typer.colors.RED)
            raise typer.Exit(code=1) from error

    stages: list[str] | None
    if only:
        stages = [only]
    elif from_stage in STAGES:
        stages = STAGES[STAGES.index(from_stage) :]
    else:
        stages = None
    if stages is None or any(stage not in STAGES for stage in stages):
        typer.secho(f"未知阶段，可选：{', '.join(STAGES)}", fg="red", err=True)
        raise typer.Exit(code=2)

    provider = None
    if "script" in stages:
        try:
            provider = build_provider(cfg.llm, Settings())
        except RuntimeError as error:
            typer.secho(str(error), fg="red", err=True)
            raise typer.Exit(code=1) from error

    tts_engine = None
    if "voice" in stages:
        tts_engine = build_tts_engine(cfg.render)

    try:
        warnings = asyncio.run(
            run_pipeline(
                cfg,
                provider,
                from_stage=from_stage,
                only=[only] if only else None,
                force=force,
                tts_engine=tts_engine,
                episode=episode,
            )
        )
    except (NotImplementedError, FileNotFoundError, ValueError, FFmpegError) as error:
        typer.secho(str(error), fg="red", err=True)
        raise typer.Exit(code=1) from error

    for warning in warnings:
        typer.secho(f"[warn] {warning}", fg="yellow")

    paths = Paths(cfg.root)
    for episode_cfg in cfg.episodes:
        table_path = paths.table(episode_cfg.number)
        narration_path = paths.narration(episode_cfg.number)
        if table_path.exists():
            typer.echo(f"对照表：{table_path}")
        if narration_path.exists():
            typer.echo(f"配音文本：{narration_path}")
        video_path = paths.video(episode_cfg.number)
        if video_path.exists():
            typer.echo(f"成品视频：{video_path}")


@app.command()
def inspect(
    slug: str,
    work_dir: Path = WORK_DIR_OPTION,
    episode: int = typer.Option(..., "--episode", help="集数"),
    suspect: bool = typer.Option(False, "--suspect", help="只列疑似 OCR 噪声的行"),
) -> None:
    """打印对白轨与信号摘要，调试清洗规则用。"""
    cfg = load_project(_project_file(work_dir, slug))
    paths = Paths(cfg.root)

    if not paths.dialogue(episode).exists():
        typer.secho(
            f"缺少 {paths.dialogue(episode)}，先跑 tenmin run {slug} --only ingest",
            fg="red",
            err=True,
        )
        raise typer.Exit(code=1)

    track = DialogueTrack.model_validate_json(
        paths.dialogue(episode).read_text(encoding="utf-8")
    )
    counts: dict[str, int] = {}
    for line in track.lines:
        counts[line.kind] = counts.get(line.kind, 0) + 1

    typer.echo(f"对白轨 E{episode:02d}：{len(track.lines)} 行，时长 {track.duration:.3f}s")
    typer.echo("  分类：" + "、".join(f"{k}={v}" for k, v in sorted(counts.items())))
    typer.echo(f"  片头曲：{track.op_range}")
    typer.echo(f"  片尾曲：{track.ed_range}")

    if suspect:
        typer.echo("疑似噪声行：")
        for line in track.lines:
            if line.suspect:
                typer.echo(f"  {line.idx} | {format_timestamp(line.start)} | {line.raw!r}")
        return

    if not paths.signals(episode).exists():
        typer.secho(
            f"缺少 {paths.signals(episode)}，先跑 tenmin run {slug} --only signals",
            fg="red",
            err=True,
        )
        raise typer.Exit(code=1)

    report = SignalReport.model_validate_json(
        paths.signals(episode).read_text(encoding="utf-8")
    )
    typer.echo(f"字密度中位数：{report.median_char_rate:.2f} 字/秒")
    typer.echo(f"无字幕间隙 {len(report.silent_gaps)} 处：")
    for gap in sorted(report.silent_gaps, key=lambda g: -g.duration):
        typer.echo(
            f"  {format_timestamp(gap.start)} - {format_timestamp(gap.end)} "
            f"| {gap.duration:.3f}s | 强度 {gap.strength}"
        )
    typer.echo(f"高能点 {len(report.highlights)} 处：")
    for highlight in report.highlights:
        typer.echo(
            f"  {format_timestamp(highlight.start)} | 强度 {highlight.strength} "
            f"| {'、'.join(highlight.triggers)} | {highlight.summary}"
        )


def main() -> None:
    app()
