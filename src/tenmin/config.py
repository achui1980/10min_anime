"""project.yaml 的模型与加载，以及环境变量读取。

这里是全项目所有"经验阈值"的唯一权威来源。各阶段模块只保留从这里派生的
模块级别名（给老调用点与文档用），不再自己写第二份字面量——历史上 ingest 的
片头窗、subtitles 的 PlayRes 与 video 的 scale 都各有一份拷贝，改一处另一处
静默失配。新增旋钮请一律加到本文件的对应子模型上。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tenmin.models import STRENGTH_MAX, STRENGTH_MIN


def _validate_credit_range(value: tuple[float, float] | None) -> tuple[float, float] | None:
    """OP/ED 区间必须是 [start, end) 且 start < end、都不为负。

    写反的区间（end < start）在下游一路不报错：intervals.subtract 会把它当空集
    静默忽略，于是"我明明标了 OP"却完全不生效，只能靠肉眼看成片才发现。
    """
    if value is None:
        return None
    start, end = value
    if start < 0 or end < 0:
        raise ValueError(f"区间 {value} 不能含负数，时间轴从 0 秒开始")
    if start >= end:
        raise ValueError(f"区间 {value} 的起点必须小于终点")
    return value


class EpisodeConfig(BaseModel):
    number: int
    srt: Path
    video: Path | None = None
    op_range: tuple[float, float] | None = None
    ed_range: tuple[float, float] | None = None

    @field_validator("op_range", "ed_range")
    @classmethod
    def _check_ranges(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        return _validate_credit_range(value)


class LocaleConfig(BaseModel):
    convert_traditional: bool = True


class LLMConfig(BaseModel):
    provider: Literal["gemini", "minimax", "openai_compatible"] = "gemini"
    model: str = "gemini-3.6-flash"
    base_url: str | None = None
    # MiniMax-M3 默认打开"深度思考"，会先吐一大段 <think>…</think> 推理块再出正文，
    # 而这段推理内容目前直接被丢弃（见 script/llm.py 的 _strip_reasoning），纯粹是
    # 浪费掉的耗时（实测 ~35k 字符的 prompt 光推理阶段就能占大头）。默认关掉它。
    thinking: Literal["adaptive", "disabled"] = "disabled"

    # 以下五个字段目前"只定义不消费"，等 provider 层专项任务接线。
    # None = 不往请求里塞这个字段，用服务端自己的默认值。
    temperature: float | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    # 抄 script/llm.py 的 OPENAI_COMPATIBLE_TIMEOUT 的 write=120.0。
    # 那个 httpx.Timeout 是四元组（connect=30 / read=None / write=120 / pool=30），
    # read 故意不设上限（长 prompt 的流式生成可以几分钟不吐第一个 token）。
    timeout_seconds: float = Field(default=120.0, gt=0)
    # 抄 script/llm.py 的 OPENAI_COMPATIBLE_MAX_ATTEMPTS。
    max_attempts: int = Field(default=3, ge=1)
    # 抄 script/budget.py 的 DEFAULT_TOLERANCE。
    budget_tolerance: float = Field(default=0.12, ge=0)


class IngestConfig(BaseModel):
    """ingest 阶段的续行合并阈值（原 ingest/normalize.py 的模块常量）。

    只收数值旋钮。TERMINAL_PUNCT 那类字符集合留在 normalize.py：它是"这门语言
    怎么断句"的语言学事实，不是调参旋钮，放进 yaml 只会被误改。
    """

    merge_max_gap: float = Field(default=0.3, ge=0)
    merge_max_chars: int = Field(default=40, gt=0)
    # 被硬折断的续行都很短；单行 >=4s 说明它本身就是完整的一句（往往是拖长音）。
    merge_max_line_seconds: float = Field(default=4.0, gt=0)


class CreditsConfig(BaseModel):
    """OP/ED 与 staff 行识别的阈值（原 ingest/credits.py 的模块常量）。"""

    # 聚簇法找 OP 时，簇起点必须落在 [op_search_start, op_search_end] 内。
    # 下界 30 秒：实测最早的 OP 起点是《恶女》第 2 集的 53.554 秒，取 30 留足余量。
    # 原来的 60 秒会把它挡在窗外，OP 就退化成「最长的演出高光」。
    op_search_start: float = Field(default=30.0, ge=0)
    op_search_end: float = Field(default=300.0, gt=0)
    # in_credit_window 的片头窗上界。历史上跟 op_search_end 共用同一个 tuple 的 [1]，
    # 但两者语义无关（一个是"聚簇起点候选区间"，一个是"给 is_credits 规则 4-5 开门的
    # 片头窗"），拆开后可以各自独立调。
    credit_head_window: float = Field(default=300.0, gt=0)
    # 合格 OP 簇的跨度区间。
    op_span_min: float = Field(default=40.0, ge=0)
    op_span_max: float = Field(default=120.0, gt=0)
    # ED 聚簇：簇起点距片尾不超过这么多秒才算 ED 候选（原 ED_TAIL_SECONDS）。
    ed_cluster_tail_seconds: float = Field(default=120.0, gt=0)
    # in_credit_window 的片尾窗（原 ED_WINDOW_SECONDS）。刻意比 ed_cluster_tail_seconds
    # 窄：黄金样本最后一句真台词在 1325.5s（片长 1416.6s，距片尾 91s），ED staff 第一行
    # 在 1348.2s。80s 落在两者之间的 19.8s 无字幕间隙里，两侧各留约 11s 余量。
    ed_keyword_window_seconds: float = Field(default=80.0, gt=0)
    # in_credit_window 的「片头窗 + 片尾窗」总覆盖率上限（占片长的比例）。
    # 没有这条约束时两个标称窗（300 + 80）在 duration <= 380 的短 track 上会把整条
    # 时间轴覆盖满，is_credits 的激进规则（通用中文词「演出」「制作」、纯人名罗列、
    # 拉丁占比）就对每一行生效，真台词被踢出语音轨、静默间隙虚假合并成假高光。
    # 0.5 = 至少一半时间轴必须留在窗外。duration >= 760 时预算 >= 380 >= 300+80，
    # 两个窗都按标称值生效，与这条约束加入前逐点等价（实测最短素材 1315.94 秒）。
    credit_window_max_ratio: float = Field(default=0.5, gt=0, le=1)
    # 相邻 credits 行间隔不超过这么多秒就并进同一个簇。
    cluster_max_gap: float = Field(default=35.0, ge=0)
    # 静区兜底的 OP 时长区间。实测 OP 静默长度 90.7-94.5 秒；
    # 下界 60 是为了不把 20 秒级的演出静场误判成 OP，
    # 上界 120 是为了不把整段无对白的过场误判成 OP。
    op_min_silent_span: float = Field(default=60.0, ge=0)
    op_max_silent_span: float = Field(default=120.0, gt=0)
    # is_credits 规则 3：书名号内文本与剧名的字符重合率门槛。
    title_overlap_threshold: float = Field(default=0.6, ge=0, le=1)
    # is_credits 规则 6：标题卡的长度上限。
    title_card_max_len: int = Field(default=24, gt=0)
    # is_credits 规则 4：纯人名罗列的 CJK 字数门槛。
    name_list_min_cjk: int = Field(default=6, gt=0)
    # 段数够多时放宽字数门槛：「慧 诹 访 郎」只有 4 个 CJK 字符，
    # 但切成 4 段本身就是 staff 罗列的形态，不可能是台词。
    name_list_many_segments: int = Field(default=4, gt=0)
    name_list_many_min_cjk: int = Field(default=4, gt=0)
    # is_credits 规则 5：拉丁字母占比门槛与生效所需的最小非空白长度。
    latin_ratio_threshold: float = Field(default=0.6, ge=0, le=1)
    latin_min_len: int = Field(default=6, gt=0)


class SignalsConfig(BaseModel):
    """signals 阶段的阈值（原 signals/{gaps,density,aggregate}.py 的模块常量）。

    强度字段的上下界跟 models.Signal.strength / models.Highlight.strength 的
    Field(ge=1, le=5) 绑在同一对常量上，不会再出现"配了 6 结果 pydantic 炸"。
    """

    # gaps
    min_gap_seconds: float = Field(default=3.0, ge=0)
    # 间隙强度分档：>= gap_strong_seconds → 4，>= gap_medium_seconds → 3，否则 2。
    gap_strong_seconds: float = Field(default=15.0, ge=0)
    gap_medium_seconds: float = Field(default=8.0, ge=0)
    # density
    low_density_ratio: float = Field(default=0.4, ge=0)
    low_density_min_seconds: float = Field(default=2.0, ge=0)
    low_density_strength: int = Field(default=3, ge=STRENGTH_MIN, le=STRENGTH_MAX)
    shift_window_seconds: float = Field(default=30.0, gt=0)
    shift_z_threshold: float = Field(default=1.5, ge=0)
    shift_strength: int = Field(default=2, ge=STRENGTH_MIN, le=STRENGTH_MAX)
    # aggregate
    min_separation: float = Field(default=2.0, ge=0)
    summary_max_chars: int = Field(default=30, gt=0)


class RenderConfig(BaseModel):
    """v2 渲染参数。voice 与 rate 直接喂 Edge-TTS。"""

    voice: str = "zh-CN-YunxiNeural"
    rate: str = "+0%"
    video_encoder: str = "libx264"
    duck_db: float = -12.0
    font_size: int = Field(default=52, gt=0)
    fade_out_seconds: float = Field(default=1.5, ge=0)
    outro_card_seconds: float = Field(default=3.0, ge=0)
    outro_message: str = "解说结束，谢谢观看"

    # --- 已接线：字幕的 PlayRes 与视频的 scale 必须同源，否则字幕会被静默缩放 ---
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    subtitle_font_name: str = "Lantinghei SC"

    # --- 以下"只定义不消费"，等各自的 render 专项任务接线 ---
    # 抄 render/video.py 的 CRF / PRESET / VIDEOTOOLBOX_BITRATE。
    crf: str = "20"
    preset: str = "medium"
    videotoolbox_bitrate: str = "6000k"
    # 抄 render/audio.py 的 AUDIO_CODEC / AUDIO_BITRATE。
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    # 片尾黑卡 drawtext 用的字体。跟 subtitle_font_name 是两个独立旋钮：
    # 字幕字体换了不代表片尾卡也要换（卡片是纯 ASCII+中文标题，选择面更宽）。
    outro_font_name: str = "Lantinghei SC"
    # 抄 render/tts.py 的 TTS_MAX_ATTEMPTS。
    tts_max_attempts: int = Field(default=3, ge=1)
    # 新旋钮。默认 1 = 保持当前的串行合成行为，并发化是后续任务的事。
    tts_concurrency: int = Field(default=1, ge=1)
    # 新旋钮。edge-tts 走公司代理时需要；None = 不设代理。
    tts_proxy: str | None = None
    # 抄 render/timeline.py 的 DRIFT_TOLERANCE。
    drift_tolerance: float = Field(default=0.5, ge=0)
    # 抄 render/ffmpeg.py 的 FFMPEG / FFPROBE。
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"


class ProjectConfig(BaseModel):
    show: str
    slug: str
    mode: Literal["single_episode", "season"] = "single_episode"
    target_seconds: float = Field(default=240.0, gt=0)
    locale: LocaleConfig = Field(default_factory=LocaleConfig)
    episodes: list[EpisodeConfig] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    credits: CreditsConfig = Field(default_factory=CreditsConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    _root: Path = PrivateAttr(default=Path("."))

    @property
    def root(self) -> Path:
        """project.yaml 所在目录，也就是 work/<slug>/。"""
        return self._root

    def bind_root(self, root: Path) -> ProjectConfig:
        """绑定项目目录（work/<slug>/）。load_project 会自动调用，测试也可直接用。"""
        self._root = Path(root).resolve()
        return self

    def srt_path(self, episode: EpisodeConfig) -> Path:
        """episode.srt 是相对 project.yaml 的路径；绝对路径原样返回。"""
        if episode.srt.is_absolute():
            return episode.srt
        return self._root / episode.srt

    def video_path(self, episode: EpisodeConfig) -> Path:
        """源视频路径。相对路径按 project.yaml 所在目录解析。"""
        if episode.video is None:
            raise ValueError(
                f"第 {episode.number} 集没有配置 video，"
                "请在 project.yaml 的 episodes 里补上源视频路径"
            )
        if episode.video.is_absolute():
            return episode.video
        return self._root / episode.video


# 各阶段模块共用的默认实例。给不关心配置的调用点（单测、一次性脚本）当默认参数用，
# 免得每个模块各自 XxxConfig() 一份。注意它们是可变的 pydantic 模型：只读，别改。
DEFAULT_INGEST = IngestConfig()
DEFAULT_CREDITS = CreditsConfig()
DEFAULT_SIGNALS = SignalsConfig()
DEFAULT_RENDER = RenderConfig()


def load_project(path: Path) -> ProjectConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到项目配置: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ProjectConfig.model_validate(data).bind_root(path.parent)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TENMIN_", env_file=".env", extra="ignore"
    )

    gemini_api_key: SecretStr | None = None
    minimax_api_key: SecretStr | None = None
    openai_compatible_api_key: SecretStr | None = None
