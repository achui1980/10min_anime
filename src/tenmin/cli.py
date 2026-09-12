"""tenmin 命令行入口。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import typer
import yaml

from tenmin.config import Settings, load_project
from tenmin.models import DialogueTrack, SignalReport
from tenmin.pipeline import (
    STAGES,
    Paths,
    _find_episode,
    register_episode,
    resolve_stages,
    run_pipeline,
)
from tenmin.render.ffmpeg import FFmpegError
from tenmin.render.tts import build_tts_engine
from tenmin.rich_progress import RichProgressReporter
from tenmin.script.llm import build_provider
from tenmin.script.validate import ScriptValidationError
from tenmin.timecode import format_timestamp

app = typer.Typer(add_completion=False, help="把番剧压成解说方案的流水线。")

# run_pipeline 会抛、且已经自带一句人话的异常。逃出这张表就意味着用户看到一整页
# traceback，所以新增会向上冒的异常类型时必须同步补这里。
#
# 逐条为什么在表里：
# - NotImplementedError：mode: season 还没实现（run_pipeline 第一行就抛）。
# - FileNotFoundError：上游产物缺失（_load_* 系列）、源片被移走。
# - ValueError：阶段名非法、这一集没注册、没给 tts_engine，以及 **pydantic 的
#   ValidationError**（它是 ValueError 子类，产物 json 被手改坏时走这条）。
#   所以这里不许把 ValueError 换成更窄的类型。
# - FFmpegError：ffmpeg 没编 libass / 编码器不存在 / 转码失败。
# - ScriptValidationError：它是 RuntimeError 子类而不是 ValueError 子类，
#   历史上漏在表外——LLM 出的剧本过不了 validate 时用户看的是裸 traceback。
# - httpx.HTTPError：provider 的 raise_for_status()（429/5xx）抛的 HTTPStatusError，
#   以及连不上/读超时的 TransportError。取它们的公共父类，免得再漏一个子类。
PIPELINE_ERRORS = (
    NotImplementedError,
    FileNotFoundError,
    ValueError,
    FFmpegError,
    ScriptValidationError,
    httpx.HTTPError,
)


def _error_message(error: BaseException) -> str:
    """httpx 的传输类异常经常 str() 为空（ReadTimeout('')），光印 str 会是一行空红字。"""
    return str(error) or type(error).__name__

WORK_DIR_OPTION = typer.Option(Path("work"), "--work-dir", help="项目根目录")

# 提到模块级是 B008 的标准解法（typer.Option 是函数调用，写在参数默认值里会被
# flake8-bugbear 判 B008）。WORK_DIR_OPTION 已经是这个形状，这里跟它保持一致。
ONLY_OPTION = typer.Option(
    None,
    "--only",
    help=(
        "只跑这些阶段，可重复传或用逗号分隔"
        "（--only ingest --only signals / --only ingest,signals）"
    ),
)
SRT_OPTION = typer.Option(
    None, "--srt", help="要注册的字幕文件路径，需配合 --episode 和 --video"
)
VIDEO_OPTION = typer.Option(
    None, "--video", help="要注册的视频文件路径，需配合 --episode 和 --srt"
)

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


def _parse_only(values: list[str] | None) -> list[str] | None:
    """把 --only 的原始取值摊平成阶段名列表；一个都没传则返回 None。

    两种写法都收：重复传（--only ingest --only signals）与逗号分隔
    （--only ingest,signals）。阶段名是否合法交给 pipeline.resolve_stages 判。
    返回 None 而不是空列表：run_pipeline 把 only=[] 当作「什么都不跑」，
    把「没传 --only」误传成空列表会让整条流水线静默空转。
    """
    if not values:
        return None
    stages = [part.strip() for value in values for part in value.split(",")]
    return [stage for stage in stages if stage]


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
    only: list[str] | None = ONLY_OPTION,
    force: bool = typer.Option(False, "--force", help="忽略 mtime 强制重跑"),
    episode: int | None = typer.Option(
        None, "--episode", help="要处理的集数；配合 --srt/--video 可注册新的一集"
    ),
    srt: Path | None = SRT_OPTION,
    video: Path | None = VIDEO_OPTION,
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

    only_stages = _parse_only(only)
    try:
        stages = resolve_stages(from_stage=from_stage, only=only_stages)
    except ValueError as error:
        typer.secho(str(error), fg="red", err=True)
        raise typer.Exit(code=2) from error

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
        with RichProgressReporter() as reporter:
            warnings = asyncio.run(
                run_pipeline(
                    cfg,
                    provider,
                    from_stage=from_stage,
                    only=only_stages,
                    force=force,
                    tts_engine=tts_engine,
                    episode=episode,
                    reporter=reporter,
                )
            )
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error

    for warning in warnings:
        typer.secho(f"[warn] {warning}", fg="yellow")

    paths = Paths(cfg.root)
    if episode is not None:
        printed_numbers = [episode]
    else:
        printed_numbers = [e.number for e in cfg.episodes]

    for number in printed_numbers:
        table_path = paths.table(number)
        narration_path = paths.narration(number)
        if table_path.exists():
            typer.echo(f"对照表：{table_path}")
        if narration_path.exists():
            typer.echo(f"配音文本：{narration_path}")
        video_path = paths.video(number)
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
    if track.skipped_blocks or track.clamped_cues:
        typer.secho(
            f"  解析期坏数据：跳过 {track.skipped_blocks} 个无时间戳块、"
            f"夹平 {track.clamped_cues} 条终点早于起点的 cue",
            fg="yellow",
        )

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
