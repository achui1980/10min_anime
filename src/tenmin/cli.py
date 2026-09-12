"""tenmin 命令行入口。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import typer
import yaml

from tenmin.config import ProjectConfig, Settings, load_project
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
from tenmin.render.tts import TTSError, build_tts_engine
from tenmin.rich_progress import RichProgressReporter
from tenmin.script.llm import LLMError, build_provider
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
# - httpx.HTTPError：provider 里没被包成 LLMError 的传输类异常（比如 base_url 写成
#   了不支持的 scheme）。取它们的公共父类，免得再漏一个子类。
# - LLMError：provider 自己抛的、已经带好人话的那一族（HTTP 4xx/5xx 带响应体摘要、
#   HTTP 200 + 业务错误码、传输层重试耗尽、连续 N 次不合 schema、Gemini 的
#   finish_reason 异常）。它是 RuntimeError 子类，**不在** ValueError 那条网里。
# - TTSError：voice 阶段某个 chunk 重试耗尽 / 合成出来的音频时长离谱。同样是
#   RuntimeError 子类，同样不在 ValueError 那条网里。
PIPELINE_ERRORS = (
    NotImplementedError,
    FileNotFoundError,
    ValueError,
    FFmpegError,
    ScriptValidationError,
    httpx.HTTPError,
    LLMError,
    TTSError,
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

PROJECT_TEMPLATE_FIELDS: dict[str, Any] = {
    "show": True,
    "slug": True,
    "mode": True,
    "target_seconds": True,
    "locale": True,
    "episodes": True,
    "glossary": True,
    "llm": {"provider", "model"},
    "render": {
        "voice",
        "rate",
        "video_encoder",
        "duck_db",
        "font_size",
        "fade_out_seconds",
        "outro_card_seconds",
        "outro_message",
    },
}
"""tenmin init 生成的 project.yaml 里登场哪些字段。

这里只挑**字段名**，值一律由 ProjectConfig 现场算（见 build_project_template），
所以「改了 config.py 的默认值、忘了改 cli.py 的模板」这类漂移不可能再发生——
原实现是把 show/mode/target_seconds/llm/render 的默认值全部手抄一遍的字面量 dict。

为什么不整份 model_dump（那样连字段名都不用挑）：ProjectConfig 底下现在有
ingest(3) / credits(21) / signals(11) / render(24) / llm(9) 近 70 个调参旋钮，
全吐出来的 project.yaml 没人能读，而这个文件是用户的主要编辑面。

为什么挑漏了不要紧：漏掉的字段照样走模型默认值，行为完全不变，只是「没在模板里
被推荐」而已，用户想调时补一行就生效。也就是说这张表过期是良性的（少一句广告），
而原来那份手抄值过期是有害的（磁盘上落一个错的默认值）。全部旋钮见 config.py。
"""

# 放在 yaml 正文前面的注释块。episodes 刻意是空的（见 build_project_template），
# 所以「怎么加一集」必须就地说清楚。
PROJECT_TEMPLATE_HEADER = """\
# tenmin 项目配置。这里只列了最常改的旋钮，值全部取自 tenmin 的默认配置。
# ingest / credits / signals / render 底下还有几十个阈值可以在这里覆盖，
# 完整清单见 src/tenmin/config.py（那是全项目经验阈值的唯一权威来源）。
#
# 登记一集（推荐，路径会自动填好；源片不会被拷进 work/，只记它的绝对路径）：
#   tenmin run {slug} --episode 2 --srt <字幕路径> --video <源片路径>
#
# 也可以手写。srt 相对本文件所在目录解析，video 可以是相对路径或绝对路径：
#   episodes:
#   - number: 2
#     srt: srt/E02.srt
#     video: /abs/path/to/E02.mkv
"""


def build_project_template(slug: str) -> dict[str, Any]:
    """算出 tenmin init 要写进 project.yaml 的内容。

    每次调用都新建一个 ProjectConfig 再 dump，返回的嵌套结构没有任何一层跟模块级
    常量共享对象——原实现 `dict(PROJECT_TEMPLATE)` 是浅拷贝，payload["episodes"]、
    payload["render"] 与那份全局 dict 是同一个对象，当时只改顶层的 slug/show 所以
    没暴露，但任何对嵌套字段的改写都会污染后续所有 init。

    episodes 刻意留空，不预置示例条目：register_episode 是「读回 cfg.episodes 再整份
    写回 yaml」的，示例条目会被当成一集真的番留下来，于是登记完第一集之后 episodes
    变成 [示例, 真的那集]，不带 --episode 的批处理模式就会去跑那个指向不存在的
    srt/E02.srt 的幽灵条目。手写 episodes 的形状见 PROJECT_TEMPLATE_HEADER。
    """
    return ProjectConfig(show=slug, slug=slug).model_dump(
        mode="json", include=PROJECT_TEMPLATE_FIELDS
    )


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

    # 只建 srt/：register_episode 会把字幕拷进去（它自己也会 mkdir，这里预先建出来
    # 是为了让「手动放字幕」的人一眼看到位置）。刻意不建 video/——P0-C 之后源片不再
    # 被拷进 work/，project.yaml 只记它的绝对路径，一个空的 video/ 夹在
    # 01_dialogue/…07_render/ 中间只会让人以为源片该放那儿。
    (root / "srt").mkdir(parents=True, exist_ok=True)
    payload = build_project_template(slug)
    project_file.write_text(
        PROJECT_TEMPLATE_HEADER.format(slug=slug)
        + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    typer.echo(f"已创建 {project_file}")
    typer.echo("登记一集（源片留在原地，只把绝对路径记进 project.yaml）：")
    typer.echo(f"  tenmin run {slug} --episode 2 --srt <字幕路径> --video <源片路径>")
    typer.echo(f"字幕会被拷进 {root / 'srt'}，清洗规则调不好时可以就地改它。")


async def _run_pipeline_and_close(
    cfg: ProjectConfig, provider: Any, **kwargs: Any
) -> list[str]:
    """跑流水线，然后关掉 provider 持有的连接池。

    OpenAICompatibleProvider 现在持有一个 httpx.AsyncClient（为的是让同一次运行的
    多轮重试、批量模式的多集共用连接池），所以它的生命周期必须有人收尾。用 getattr
    探测而不是写死类型：GeminiProvider 没有 aclose（google-genai 自己管连接），
    而库调用方/测试传进来的假 provider 更不会有。
    """
    try:
        return await run_pipeline(cfg, provider, **kwargs)
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()


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
                _run_pipeline_and_close(
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
