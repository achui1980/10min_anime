"""tenmin 命令行入口。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import typer
import yaml
from pydantic import ValidationError

from tenmin import animeko, atomic, importer
from tenmin.config import (
    DEFAULT_CONTENT_CHECK,
    DEFAULT_LLM,
    DEFAULT_OCR,
    DEFAULT_RENDER,
    AsrConfig,
    ContentCheckConfig,
    LLMConfig,
    OcrConfig,
    ProjectConfig,
    Settings,
    load_project,
)
from tenmin.content_check.models import CheckOutcome, ContentCheckError
from tenmin.content_check.service import check_video
from tenmin.ingest import ocr_audit, ocr_batch
from tenmin.ingest.asr import ASRError
from tenmin.ingest.normalize import credit_range_source
from tenmin.ingest.ocr import OCRError, OCRUnavailableError
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
# - ASRError：生肉片源（没传 --srt、视频里也没有软字幕轨）要走语音转写，而转写依赖
#   没装（子类 ASRUnavailableError，消息里带 `uv sync --extra asr`）或者一条对白都没
#   转出来。同样是 RuntimeError 子类，不在 ValueError 那条网里。只登记父类就够 ——
#   ASRUnavailableError 继承它。
# - OCRError：声明了硬字幕的片源要走画面 OCR，而 OCR 依赖没装（子类
#   OCRUnavailableError，消息里带 `uv sync --extra ocr`）、帧与时间戳对不上、或者一条
#   字幕都没认出来。理由与登记方式同 ASRError。
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
    ASRError,
    OCRError,
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
    None,
    "--video",
    help="要注册的视频文件路径，需配合 --episode；不传 --srt 就是生肉入口",
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
    "content_check": True,
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
ingest(3) / credits(21) / signals(11) / validate_script(8) / render(30) / llm(15)
/ asr(2) / ocr(7) 九十来个调参旋钮，全吐出来的 project.yaml 没人能读，而这个文件是用户的
主要编辑面。

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
# 没有字幕的生肉片源省掉 --srt 就行（对白轨靠视频里的软字幕轨、画面 OCR（声明了
# 硬字幕时，要先 `uv sync --extra ocr`）或语音转写拿，后者要先 `uv sync --extra asr`）：
#   tenmin run {slug} --episode 2 --video <源片路径>
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
    # 是为了让「手动放字幕」的人一眼看到位置）。刻意不建 video/——源片不会被拷进
    # work/（见 pipeline.register_episode），project.yaml 只记它的绝对路径，一个空的 video/ 夹在
    # 01_dialogue/…07_render/ 中间只会让人以为源片该放那儿。
    (root / "srt").mkdir(parents=True, exist_ok=True)
    payload = build_project_template(slug)
    # 原子写，跟 register_episode 改写同一个文件时用的是同一层：project.yaml 是全部
    # 配置切片的来源、也是已登记集数的唯一记录，所以它属于「产物」。
    # 后果本身有限（上面刚查过 exists()，半截 YAML 会让 load_project 响亮报 parse
    # error），但它是「原子写覆盖全部产物写入点」这条不变量的唯一缺口，而那条不变量
    # 现在由 tests/test_source_hygiene.py 的 test_artifact_writes_go_through_atomic 守着。
    atomic.write_text(
        project_file,
        PROJECT_TEMPLATE_HEADER.format(slug=slug)
        + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
    )
    typer.echo(f"已创建 {project_file}")
    typer.echo("登记一集（源片留在原地，只把绝对路径记进 project.yaml）：")
    typer.echo(f"  tenmin run {slug} --episode 2 --srt <字幕路径> --video <源片路径>")
    typer.echo(f"字幕会被拷进 {root / 'srt'}，清洗规则调不好时可以就地改它。")


def _print_artifacts(
    cfg: ProjectConfig,
    numbers: list[int],
    check_results: dict[int, CheckOutcome] | None = None,
) -> None:
    """跑完之后给出交付物路径（不存在的不打）。run 与 import 共用。

    检查结果只来自本次运行收集到的成功结果，不按旧报告是否存在推断。
    """
    paths = Paths(cfg.root)
    for number in numbers:
        table_path = paths.table(number)
        narration_path = paths.narration(number)
        if table_path.exists():
            typer.echo(f"对照表：{table_path}")
        if narration_path.exists():
            typer.echo(f"配音文本：{narration_path}")
        video_path = paths.video(number)
        if video_path.exists():
            typer.echo(f"成品视频：{video_path}")
        if check_results and number in check_results:
            typer.echo(f"E{number:02d}：")
            _print_check_outcome(check_results[number])


async def _run_pipeline_and_close(
    cfg: ProjectConfig, provider: Any, **kwargs: Any
) -> list[str]:
    """跑流水线，然后关掉 provider 持有的连接池。

    OpenAICompatibleProvider 现在持有一个 httpx.AsyncClient（为的是让同一次运行的
    多轮重试、批量模式的多集共用连接池），所以它的生命周期必须有人收尾。用 getattr
    探测而不是写死类型：GeminiProvider 也会关闭它自持有的 httpx 客户端，
    而库调用方/测试传进来的假 provider 可能没有 aclose。
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
    force: bool = typer.Option(False, "--force", help="忽略 mtime 和检查缓存，强制重跑"),
    episode: int | None = typer.Option(
        None, "--episode", help="要处理的集数；配合 --srt/--video 可注册新的一集"
    ),
    srt: Path | None = SRT_OPTION,
    video: Path | None = VIDEO_OPTION,
) -> None:
    """跑流水线：ingest→translate→signals→script→docgen→voice→timeline→audio→render→check。"""
    project_file = _project_file(work_dir, slug)

    # 三条规则（原来是「--srt 与 --video 必须一起传」那一条对称的规则）：
    # - 传 --srt 必须配 --video：视频是 render 阶段的硬需求，只有字幕出不了片。
    # - 只传 --video 合法，这就是生肉入口（对白轨靠软字幕轨抽取、画面 OCR（声明了
    #   硬字幕时）或语音转写拿）。
    # - 传了 --video 就必须说这是第几集，否则没法登记进 project.yaml。
    if srt is not None and video is None:
        typer.secho("传 --srt 时必须同时传 --video", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 消息只点 --video：上面那条已经把「只有 --srt」拦掉了，所以能走到这里必然带着
    # --video（--srt 可有可无）。原文案「传 --srt/--video 时」会在 --video 单飞的场合
    # 点一个用户压根没用的 flag。
    if video is not None and episode is None:
        typer.secho("传 --video 时必须同时传 --episode", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 配置读取、登记与集号查询同样可能抛用户可处理的错误。
    try:
        cfg = load_project(project_file)
        if video is not None:
            cfg = register_episode(cfg, episode=episode, srt=srt, video=video)
        if episode is not None:
            _find_episode(cfg, episode)
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error

    only_stages = _parse_only(only)
    try:
        stages = resolve_stages(from_stage=from_stage, only=only_stages)
    except ValueError as error:
        typer.secho(str(error), fg="red", err=True)
        raise typer.Exit(code=2) from error

    provider = None
    # translate 也吃 provider（它是第二个要 LLM 的阶段）。漏掉它的话
    # `--only translate` 会带着 provider=None 进 run_pipeline，炸在翻译那一层。
    if {"script", "translate"} & set(stages) and (
        episode is not None or any(ep.has_source for ep in cfg.episodes)
    ):
        try:
            provider = build_provider(cfg.llm, Settings())
        except RuntimeError as error:
            typer.secho(str(error), fg="red", err=True)
            raise typer.Exit(code=1) from error

    tts_engine = None
    if "voice" in stages:
        tts_engine = build_tts_engine(cfg.render)

    check_results: dict[int, CheckOutcome] = {}
    check_errors: list[Exception] = []
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
                    check_results=check_results,
                    check_errors=check_errors,
                )
            )
    except PIPELINE_ERRORS as error:
        if any(error is check_error for check_error in check_errors):
            raise
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error

    for warning in warnings:
        typer.secho(f"[warn] {warning}", fg="yellow")

    explicit_check = (only_stages is not None and "check" in only_stages) or (
        only_stages is None and from_stage == "check"
    )
    if "check" in stages and not cfg.content_check.enabled and not explicit_check:
        typer.echo("检查未启用")

    if episode is not None:
        printed_numbers = [episode]
    else:
        printed_numbers = [e.number for e in cfg.episodes if e.has_source]
    _print_artifacts(cfg, printed_numbers, check_results)


def _print_check_outcome(outcome: CheckOutcome) -> None:
    """两个入口使用相同的成功文案，缓存命中也显示当前候选数。"""
    typer.echo(f"已检查 {outcome.frame_count} 帧；候选 {outcome.candidate_count} 帧")
    typer.echo(f"检查报告：{outcome.report_path}")


CHECK_OUTPUT_OPTION = typer.Option(
    None, "--output", help="报告目录；默认在视频旁的 <stem>.content-check",
)
CHECK_INTERVAL_OPTION = typer.Option(
    DEFAULT_CONTENT_CHECK.sample_interval, "--interval", help="采样间隔（有限正秒数）",
)
CHECK_SENSITIVE_OPTION = typer.Option(
    DEFAULT_CONTENT_CHECK.sensitive_threshold, "--sensitive-threshold",
    help="sensitive 候选阈值（0~1）",
)
CHECK_ADULT_OPTION = typer.Option(
    DEFAULT_CONTENT_CHECK.adult_threshold, "--adult-threshold",
    help="questionable + explicit 阈值（0~1）",
)
CHECK_FORCE_OPTION = typer.Option(False, "--force", help="忽略检查缓存，重新抽帧和推理")
CHECK_OFFLINE_OPTION = typer.Option(False, "--offline", help="模型只读本地缓存")


@app.command("check-content")
def check_content(
    video: Path,
    output: Path | None = CHECK_OUTPUT_OPTION,
    interval: float = CHECK_INTERVAL_OPTION,
    sensitive_threshold: float = CHECK_SENSITIVE_OPTION,
    adult_threshold: float = CHECK_ADULT_OPTION,
    force: bool = CHECK_FORCE_OPTION,
    offline: bool = CHECK_OFFLINE_OPTION,
) -> None:
    """独立检查一个视频（需 `uv sync --extra content-check`），候选不影响退出码。"""
    destination = output if output is not None else video.with_name(f"{video.stem}.content-check")
    try:
        cfg = ContentCheckConfig(
            sample_interval=interval,
            sensitive_threshold=sensitive_threshold,
            adult_threshold=adult_threshold,
        )
    except ValidationError as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    try:
        outcome = check_video(
            video, destination, cfg=cfg,
            ffmpeg=DEFAULT_RENDER.ffmpeg_path, ffprobe=DEFAULT_RENDER.ffprobe_path,
            force=force, offline=offline,
        )
    except ContentCheckError as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    _print_check_outcome(outcome)


# 对白轨来源（DialogueTrack.source）给 inspect 显示用。三种来源下游的处置各不相同
# （繁转简、translate 怎么跑、OP/ED 推断靠不靠得住），所以值得一眼看见。
_SOURCE_LABEL = {
    "srt": "原生字幕",
    "ocr": "画面 OCR",
    "asr": "语音转写",
}

# OP/ED 区间三级回退里实际生效的那一级，给 inspect 显示用。标注的是「这个值
# 从哪来」：自动推断那一级不可靠（实测 13 集里 2 集 op=None、2 集 ed=None），
# 而手填那两级还会同时驱动 in_credit_window，两者的后果完全不同。
_RANGE_SOURCE_LABEL = {
    "episode": "逐集手填",
    "project": "项目默认",
    "inferred": "自动推断",
}


@app.command()
def inspect(
    slug: str,
    work_dir: Path = WORK_DIR_OPTION,
    episode: int = typer.Option(..., "--episode", help="集数"),
    suspect: bool = typer.Option(False, "--suspect", help="只列疑似 OCR 噪声的行"),
) -> None:
    """打印对白轨与信号摘要，调试清洗规则用。"""
    project_file = _project_file(work_dir, slug)
    try:
        cfg = load_project(project_file)
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    paths = Paths(cfg.root)

    ep_cfg = next((ep for ep in cfg.episodes if ep.number == episode), None)
    prefilled = [ep for ep in cfg.episodes if not ep.has_source]
    if prefilled:
        typer.echo(
            "未登记视频的预填条目：" + "、".join(f"E{ep.number:02d}" for ep in prefilled)
        )
    if ep_cfg is not None and not ep_cfg.has_source:
        typer.echo(
            f"E{episode:02d}：未登记视频（op_range={ep_cfg.op_range}，ed_range={ep_cfg.ed_range}）"
        )
        typer.echo(f"登记源片：tenmin run {slug} --episode {episode} --video <源片路径>")
        return

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
    # 「是否声明了硬字幕」读的是**现在**的 project.yaml，来源读的是上次 ingest 的产物；
    # 两者对不上（刚改了声明还没重跑 ingest、或者有软字幕轨优先）时并排摆着最好查。
    hardsub = ep_cfg is not None and cfg.hardsub_enabled(ep_cfg)
    typer.echo(
        f"  来源：{track.source}（{_SOURCE_LABEL[track.source]}），"
        f"硬字幕：{'已声明' if hardsub else '未声明'}"
    )
    typer.echo("  分类：" + "、".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for label, value, filled, default in (
        ("片头曲", track.op_range, ep_cfg and ep_cfg.op_range, cfg.credits.default_op_range),
        ("片尾曲", track.ed_range, ep_cfg and ep_cfg.ed_range, cfg.credits.default_ed_range),
    ):
        source = credit_range_source(filled or None, default, track.duration)
        typer.echo(f"  {label}：{value}（{_RANGE_SOURCE_LABEL[source]}）")
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


IMPORT_DIRECTORY_ARGUMENT = typer.Argument(..., help="视频所在目录（只看第一层，不递归）")
IMPORT_SRT_DIR_OPTION = typer.Option(None, "--srt-dir", help="字幕不在视频目录时，从这个目录配对")
IMPORT_YES_OPTION = typer.Option(False, "--yes", help="跳过确认，直接登记并运行（脚本 / CI 用）")
IMPORT_DRY_RUN_OPTION = typer.Option(False, "--dry-run", help="只预览：不写任何文件、不运行管线")


def _stdin_is_interactive() -> bool:
    """单独成函数，测试才能把 CliRunner 的非 TTY stdin 伪装成终端。"""
    return sys.stdin.isatty()


async def _run_imported_and_close(
    cfg: ProjectConfig,
    plan: importer.ImportPlan,
    provider: Any,
    *,
    check_results: dict[int, CheckOutcome] | None = None,
    check_errors: list[Exception] | None = None,
    **kwargs: Any,
) -> list[str]:
    """登记本次导入的集，再逐集跑单集模式管线，最后关掉 provider 的连接池。

    刻意逐集调用而不是一次批处理：批处理会把存量集也过一遍，`zh/glossary.json` 这类
    项目级输入一变动，存量集就会连带重跑。run_pipeline 每次调用都会清空收集器，
    所以逐集用新的收集器，再并进调用方传入的那份。
    """
    try:
        cfg = importer.apply_import_plan(cfg, plan)
        warnings: list[str] = []
        for item in plan.items:
            episode_results: dict[int, CheckOutcome] = {}
            episode_errors: list[Exception] = []
            try:
                warnings.extend(
                    await run_pipeline(
                        cfg,
                        provider,
                        episode=item.number,
                        check_results=episode_results,
                        check_errors=episode_errors,
                        **kwargs,
                    )
                )
            finally:
                if check_results is not None:
                    check_results.update(episode_results)
                if check_errors is not None:
                    check_errors.extend(episode_errors)
        return warnings
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()


@app.command("import")
def import_(
    slug: str,
    directory: Path = IMPORT_DIRECTORY_ARGUMENT,
    work_dir: Path = WORK_DIR_OPTION,
    srt_dir: Path | None = IMPORT_SRT_DIR_OPTION,
    yes: bool = IMPORT_YES_OPTION,
    dry_run: bool = IMPORT_DRY_RUN_OPTION,
) -> None:
    """扫描目录批量导入：配对视频与 SRT、识别集号、预览 OP/ED，确认后登记并逐集运行。"""
    project_file = _project_file(work_dir, slug)
    # 扫描、集号解析（含 LLM 兜底）、ffprobe 与区间推断都是只读的，--dry-run 也照跑：
    # 预览不完整就没有意义。
    try:
        cfg = load_project(project_file)
        plan = asyncio.run(
            importer.build_import_plan(
                cfg,
                directory,
                srt_dir,
                provider_factory=lambda: build_provider(cfg.llm, Settings()),
            )
        )
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error

    for line in importer.render_preview(plan, op_length=cfg.credits.op_length):
        typer.echo(line)
    for warning in plan.warnings:
        typer.secho(f"[warn] {warning}", fg="yellow")

    if not plan.items:
        typer.echo("没有可导入的集，未做任何改动。")
        return
    if dry_run:
        typer.echo("--dry-run：只预览，未写入任何文件，也未运行管线。")
        return
    if not yes:
        if not _stdin_is_interactive():
            typer.secho("当前不是交互终端，无法确认；确认要导入请加 --yes", fg="red", err=True)
            raise typer.Exit(code=1)
        if not typer.confirm(f"导入以上 {len(plan.items)} 集并运行管线？", default=False):
            typer.echo("已取消，未做任何改动。")
            return

    # 先把 provider 建好再登记：缺 key 时一个文件都不动，而不是登记完才在管线里炸。
    try:
        provider = build_provider(cfg.llm, Settings())
    except (RuntimeError, ValueError) as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    tts_engine = build_tts_engine(cfg.render)

    check_results: dict[int, CheckOutcome] = {}
    check_errors: list[Exception] = []
    try:
        with RichProgressReporter() as reporter:
            warnings = asyncio.run(
                _run_imported_and_close(
                    cfg,
                    plan,
                    provider,
                    tts_engine=tts_engine,
                    reporter=reporter,
                    check_results=check_results,
                    check_errors=check_errors,
                )
            )
    except PIPELINE_ERRORS as error:
        if any(error is check_error for check_error in check_errors):
            raise
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error

    for warning in warnings:
        typer.secho(f"[warn] {warning}", fg="yellow")
    _print_artifacts(cfg, [item.number for item in plan.items], check_results)


OCR_INPUTS_ARGUMENT = typer.Argument(..., help="视频文件或目录（目录只看一层），可传多个")
OCR_OUTPUT_OPTION = typer.Option(
    None, "--output", "-o", help="SRT 输出目录（不存在就建）；不传就写在视频旁边"
)
OCR_SIMPLIFIED_OPTION = typer.Option(
    True,
    "--simplified/--traditional",
    help="繁转简（默认）；--traditional 保留画面上的繁体原文",
)
OCR_FORCE_OPTION = typer.Option(False, "--force", help="输出已存在也重新识别")
OCR_CROP_TOP_OPTION = typer.Option(
    DEFAULT_OCR.crop_top,
    "--crop-top",
    help="从画面高度的这个比例往下裁出来识别（0~1，越小框得越高）",
)
OCR_CROP_BOTTOM_OPTION = typer.Option(
    DEFAULT_OCR.crop_bottom,
    "--crop-bottom",
    help="字幕带下沿占画面高度的比例（按文字框中心过滤；双语字幕可排除底部日文）",
)
OCR_AI_CORRECT_OPTION = typer.Option(
    False, "--ai-correct", help="用 LLM 校对 OCR 错字；失败时保留原稿"
)
OCR_ASR_ASSIST_OPTION = typer.Option(
    False,
    "--asr-assist",
    help="校对时附上日语语音转写确认读音（需 --ai-correct 与 asr extra）",
)
OCR_LLM_PROVIDER_OPTION = typer.Option(
    DEFAULT_LLM.provider, "--llm-provider", help="校对模型的 provider"
)
OCR_LLM_MODEL_OPTION = typer.Option(DEFAULT_LLM.model, "--llm-model", help="校对模型名称")
OCR_LLM_BASE_URL_OPTION = typer.Option(None, "--llm-base-url", help="OpenAI 兼容服务地址")


@app.command()
def ocr(
    inputs: list[Path] = OCR_INPUTS_ARGUMENT,
    output: Path | None = OCR_OUTPUT_OPTION,
    simplified: bool = OCR_SIMPLIFIED_OPTION,
    force: bool = OCR_FORCE_OPTION,
    crop_top: float = OCR_CROP_TOP_OPTION,
    crop_bottom: float | None = OCR_CROP_BOTTOM_OPTION,
    ai_correct: bool = OCR_AI_CORRECT_OPTION,
    asr_assist: bool = OCR_ASR_ASSIST_OPTION,
    llm_provider: str = OCR_LLM_PROVIDER_OPTION,
    llm_model: str = OCR_LLM_MODEL_OPTION,
    llm_base_url: str | None = OCR_LLM_BASE_URL_OPTION,
) -> None:
    """不建项目，批量把视频画面上的硬字幕认成 SRT（需 macOS + `uv sync --extra ocr`）。"""
    if asr_assist and not ai_correct:
        typer.secho("--asr-assist 需要同时指定 --ai-correct", fg="red", err=True)
        raise typer.Exit(code=1)
    try:
        ocr_config = OcrConfig(crop_top=crop_top, crop_bottom=crop_bottom)
        videos = ocr_batch.collect_videos(inputs)
        llm = LLMConfig(provider=llm_provider, model=llm_model, base_url=llm_base_url)
    except (FileNotFoundError, ValueError) as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"共 {len(videos)} 个视频")
    try:
        if ai_correct:
            summary = asyncio.run(
                ocr_batch.run_batch_corrected(
                    videos,
                    out_dir=output,
                    simplified=simplified,
                    force=force,
                    ocr_config=ocr_config,
                    llm_config=llm,
                    provider_factory=lambda: build_provider(llm, Settings()),
                    asr_config=AsrConfig() if asr_assist else None,
                )
            )
        else:
            summary = ocr_batch.run_batch(
                videos,
                out_dir=output,
                simplified=simplified,
                force=force,
                ocr_config=ocr_config,
            )
    except OCRUnavailableError as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"成功 {len(summary.done)}，跳过 {len(summary.skipped)}，失败 {len(summary.failed)}"
    )
    for video, reason in summary.failed:
        typer.secho(f"  失败：{video.name}：{reason}", fg="red", err=True)
    for warning in summary.warnings:
        typer.secho(f"[warn] {warning}", fg="yellow")
    if summary.failed:
        raise typer.Exit(code=1)


AUDIT_VIDEO_ARGUMENT = typer.Argument(..., help="做过 AI 校对的视频")
AUDIT_CORRECTED_OPTION = typer.Option(
    None, "--corrected", help="校对稿 SRT；不传就按视频名在 --srt-dir 里找"
)
AUDIT_SRT_DIR_OPTION = typer.Option(
    None, "--srt-dir", help="找校对稿的目录；默认视频所在目录"
)
AUDIT_OUTPUT_OPTION = typer.Option(
    None, "--output", "-o", help="报告目录；默认校对稿旁边的 <视频名>.ocr-audit/"
)


@app.command("ocr-audit")
def ocr_audit_command(
    video: Path = AUDIT_VIDEO_ARGUMENT,
    corrected: Path | None = AUDIT_CORRECTED_OPTION,
    srt_dir: Path | None = AUDIT_SRT_DIR_OPTION,
    output: Path | None = AUDIT_OUTPUT_OPTION,
) -> None:
    """生成 OCR AI 校对改动的抽查报告（只读，不调 LLM、不改字幕）。"""
    try:
        if not video.is_file():
            raise FileNotFoundError(f"视频不存在：{video}")
        if corrected is None:
            corrected = ocr_audit.find_corrected(video, srt_dir or video.parent)
        out_dir = output or corrected.parent / f"{video.stem}.ocr-audit"
        result = ocr_audit.audit(video, corrected, out_dir)
    except (FileNotFoundError, ValueError, FFmpegError) as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"共 {result.total} 条字幕，待抽查 {result.items} 条")
    typer.echo(f"报告：{result.report}")


GLOSSARY_SUBJECT_OPTION = typer.Option(..., "--subject", help="animeko 条目 ID（数字）")
GLOSSARY_YES_OPTION = typer.Option(False, "--yes", help="跳过确认直接写入 project.yaml")
GLOSSARY_DRY_RUN_OPTION = typer.Option(False, "--dry-run", help="只预览，不写入")
GLOSSARY_REFRESH_OPTION = typer.Option(False, "--refresh", help="忽略本地缓存重新拉取")


@app.command("glossary")
def glossary_command(
    slug: str,
    subject: int = GLOSSARY_SUBJECT_OPTION,
    work_dir: Path = WORK_DIR_OPTION,
    yes: bool = GLOSSARY_YES_OPTION,
    dry_run: bool = GLOSSARY_DRY_RUN_OPTION,
    refresh: bool = GLOSSARY_REFRESH_OPTION,
) -> None:
    """按 animeko 条目 ID 拉取角色名，生成人物名表并写入 project.yaml 的 glossary。"""
    project_file = _project_file(work_dir, slug)
    try:
        cfg = load_project(project_file)
        data = animeko.load(cfg.root, subject, refresh=refresh)
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error

    names = animeko.build_names(data["characters"])
    result = animeko.plan(cfg.glossary, names)
    info = data["subject"]
    title = info.get("nameCn") or info.get("name") or str(subject)
    typer.echo(f"{title}：角色名 {len(names)} 条")
    for name, name_cn in result.added.items():
        typer.echo(f"  新增  {name} → {name_cn}")
    for name, name_cn in result.same.items():
        typer.echo(f"  已有  {name} → {name_cn}")
    for name, (current, proposed) in result.conflicts.items():
        typer.secho(f"  保留  {name} → {current}（animeko：{proposed}）", fg="yellow")

    if not result.added:
        typer.echo("没有需要新增的条目。")
        return
    if dry_run:
        typer.echo("--dry-run：只预览，未写入。")
        return
    if not yes:
        if not _stdin_is_interactive():
            typer.secho("当前不是交互终端，无法确认；确认要写入请加 --yes", fg="red", err=True)
            raise typer.Exit(code=1)
        if not typer.confirm(f"写入 {len(result.added)} 条到 {project_file}？", default=False):
            typer.echo("已取消，未做任何改动。")
            return
    try:
        animeko.write_glossary(project_file, result.added)
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"已写入 {len(result.added)} 条：{project_file}")
