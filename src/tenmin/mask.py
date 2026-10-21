"""人工标注的遮罩区间：数据模型、读写、区间合并、ffmpeg 表达式。

**这是什么**：内容检查（content_check）把最终成片按帧打分并生成报告；人工在报告里
标出「这几秒确实有问题」；本模块把那些标注存成一份小 JSON，render 阶段照着它在
成片上做高斯模糊。

**为什么是独立文件而不是回写 results.json**：content_check 有一条已经被验收过的不
变量 —— 报告页面的交互不得改写 results.json（docs/content-check-validation.md 实测
过交互前后 SHA-256 相同）。遮罩标注是**另一个动作**的产物，用另一个文件，两边都不破。
落点是 `08_check/E{NN}.mask.json`，刻意做成 `08_check/E{NN}/` 的**兄弟文件**而不是
里面的一个文件：那个目录归 content_check 所有（目录非空时它要求 `.tenmin-content-check`
标记），标注要是住在里面，content_check 下次跑起来会把自己的目录判成「非本工具拥有」。

**时间轴**：区间是**成片时间轴**（秒），跟 content_check 报告里每帧的 `time` 是同一
条时间轴 —— 报告本来就是拿最终成片抽的帧，所以「第几分钟有问题」直接就是「第几秒
模糊」。这一点是整个功能能成立的前提：一旦分岔，标注就会打在错误的地方，而且不报错。

**为什么文件里不带视频身份**：遮罩之后成片必然重新渲染、mtime 必变。带身份的话第二
道校验就会把「刚标好的标注」判成过期，越用越糊涂。集号是唯一的身份信息，它由 CLI 的
`--episode` 决定（导入时还会跟文件里记的对一遍）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tenmin.atomic import write_text_if_changed

# 标注文件的格式版本。往后改结构时 +1，旧文件读不出来时给出明确报错而不是
# 「字段对不上」的一堆 pydantic 噪音。
SCHEMA_VERSION = 1

# 默认模糊强度（gblur 的 sigma，单位：像素）。20 是拿真实 E02 试出来的档位：
# 16 还看得出轮廓、24 已经糊成一片，20 是「看得出是画面、看不清细节」的那一档。
DEFAULT_SIGMA = 20.0

# 报告页上「标记需遮罩」按钮给新区间的默认长度（秒）。采样间隔默认 1s，2s 能盖住
# 一帧到它前后各一秒，不至于只糊住一帧的宽度。
DEFAULT_RANGE_SECONDS = 2.0


class MaskError(ValueError):
    """遮罩标注有问题：文件不是合法 JSON、字段不对、区间非法、导入了别的集。

    继承 ValueError：CLI 的 PIPELINE_ERRORS 里已经有一张 ValueError 的网（而且那儿
    明写着「不许把它换成更窄的类型」），所以这样不用再往那张表里加一行。
    """


class MaskRange(BaseModel):
    """一段需要模糊的区间（成片时间轴，秒）。"""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self) -> MaskRange:
        if self.end <= self.start:
            raise ValueError(
                f"遮罩区间的结束时间必须大于开始时间（{self.start} → {self.end}）"
            )
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class MaskSpec(BaseModel):
    """一份遮罩标注。

    `episode` 可以是 None：独立命令 `tenmin check-content` 出的报告不知道自己是第几集，
    导出时就不带这个字段，导入时由 CLI 的 `--episode` 补上。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    episode: int | None = Field(default=None, ge=1)
    sigma: float = Field(default=DEFAULT_SIGMA, gt=0)
    ranges: list[MaskRange] = Field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(item.duration for item in self.ranges)


def merge_ranges(ranges: Iterable[MaskRange]) -> list[MaskRange]:
    """合并重叠或首尾相接的区间，结果按开始时间排序。

    合并是必要的，不只是整洁：报告页上标两帧相邻的候选时会得到两个重叠区间，而
    ffmpeg 的 enable 表达式里 `between(t,a,b)+between(t,c,d)` 两条都命中同一段时，
    那一段的每一帧要算两遍条件（结果一样，但表达式会越长越难读）。
    """
    ordered = sorted(ranges, key=lambda item: (item.start, item.end))
    merged: list[MaskRange] = []
    for item in ordered:
        if merged and item.start <= merged[-1].end:
            last = merged[-1]
            if item.end > last.end:
                merged[-1] = MaskRange(start=last.start, end=item.end)
            continue
        merged.append(item)
    return merged


def enable_expression(ranges: Iterable[MaskRange]) -> str:
    """把区间翻成 ffmpeg `gblur` 的 enable 时间线表达式。

    形如 `between(t,156.000,162.500)+between(t,170.000,175.000)`：多个区间用 `+` 相加，
    在 ffmpeg 的时间线表达式里「相加」就是「或」——落在任意一个区间内，这一帧就模糊。

    时间轴是**拼接后**的时间轴（concat 的输出 PTS 从 0 开始），也就是成片时间轴，
    跟标注文件里的秒数一一对应。这条对应关系是实测过的：concat 前后各做一遍同样的
    enable，模糊的帧集合逐帧一致。
    """
    return "+".join(
        f"between(t,{item.start:.3f},{item.end:.3f})" for item in ranges
    )


def parse_mask(payload: Any) -> MaskSpec:
    """把已经 parse 出来的 JSON 结构校验成 MaskSpec，坏消息统一成 MaskError。"""
    try:
        return MaskSpec.model_validate(payload)
    except ValidationError as error:
        raise MaskError(f"遮罩标注字段不合法：{error}") from error


def load_mask(path: Path) -> MaskSpec | None:
    """读标注文件；文件不存在返回 None（这一集没标过），文件坏了抛 MaskError。"""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise MaskError(f"读取遮罩标注 {path} 失败：{error}") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise MaskError(f"遮罩标注 {path} 不是合法 JSON：{error}") from error
    return parse_mask(payload)


def save_mask(path: Path, spec: MaskSpec) -> bool:
    """写标注文件，返回是否真的写了。

    内容跟盘上逐字节相同时**不碰文件**：这份文件是 render 阶段的新鲜度输入（mtime
    比成片新就重渲染），无谓地刷新 mtime 会白跑一次几分钟的渲染。
    """
    payload = json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    return write_text_if_changed(Path(path), payload)


def normalize(spec: MaskSpec, *, episode: int | None = None) -> MaskSpec:
    """对齐集号、合并区间。

    集号对不上直接报错：把 E02 的标注导进 E12 不会报错、只会把两集的时间轴搅乱
    （每集的成片时长和剪辑点都不一样），而那种错在成片里根本看不出来。
    """
    if spec.episode is not None and episode is not None and spec.episode != episode:
        raise MaskError(
            f"这份标注是 E{spec.episode:02d} 的，不能导入 E{episode:02d}；"
            "请在对应的集的报告里重新导出"
        )
    return MaskSpec(
        schema_version=SCHEMA_VERSION,
        episode=episode if episode is not None else spec.episode,
        sigma=spec.sigma,
        ranges=merge_ranges(spec.ranges),
    )
