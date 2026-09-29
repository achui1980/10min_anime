"""按阶段的配置切片：每个阶段只对「它真正读到的那部分配置」敏感。

切片是解析后的配置（model_dump(mode="json")，sort_keys），所以修改 YAML 注释、
键的顺序或显式写出默认值，不会触发重跑。每集只放本集的 EpisodeConfig。

改 STAGE_FIELDS 时：拿不准就多挂；LLM 阶段逐字段核对以免多付调用费用；
EXCLUDED 只收不影响产物内容的运维旋钮。此模块不 import pipeline，以免形成循环。
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tenmin import atomic
from tenmin.config import EpisodeConfig, ProjectConfig

SLICE_DIR = ".config"

# 本集的完整 EpisodeConfig（number / srt / video / op_range / ed_range）。
EPISODE = "episode"

# signals 全局执行且配置不含集号；ingest 虽也全局执行，却依赖每集的输入。
PROJECT_LEVEL_STAGES = frozenset({"signals"})

# 键顺序跟 pipeline.STAGES 一致。整段子配置会自动扣掉 EXCLUDED。
STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "ingest": ("locale", "ingest", "credits", "asr", "glossary", "show", EPISODE),
    # 翻译只读取术语表、provider 构造参数；对白轨是它的上游数据输入。
    "translate": (
        "glossary",
        "llm.provider",
        "llm.model",
        "llm.base_url",
        "llm.thinking",
        "llm.temperature",
        "llm.max_output_tokens",
    ),
    "signals": ("signals",),
    "script": (
        "llm",
        "validate_script",
        "target_seconds",
        "mode",
        "glossary",
        "show",
        "render.rate",
    ),
    "docgen": ("render.rate",),
    "voice": ("render.voice", "render.rate"),
    "timeline": (
        "render.font_size",
        "render.subtitle_font_name",
        "render.width",
        "render.height",
        "render.subtitle_max_lines",
        "render.subtitle_min_seconds",
        "render.subtitle_soft_max_chars",
        "render.drift_tolerance",
        EPISODE,
    ),
    "audio": (
        "render.duck_db",
        "render.fade_out_seconds",
        "render.outro_card_seconds",
        "render.audio_codec",
        "render.audio_bitrate",
        "render.limiter_ceiling",
        "render.hold_relative_lu",
        "render.hold_gain_max_db",
        "render.hold_silence_floor_lufs",
        "render.hold_fade_seconds",
        "render.loudness_i",
        "render.loudness_tp",
        "render.loudness_lra",
        EPISODE,
    ),
    "render": (
        "render.video_encoder",
        "render.width",
        "render.height",
        "render.crf",
        "render.preset",
        "render.tune",
        "render.videotoolbox_bitrate",
        "render.fade_out_seconds",
        "render.outro_card_seconds",
        "render.outro_message",
        "render.outro_font_name",
        "render.font_size",
        "render.subtitle_font_name",
        "show",
        EPISODE,
    ),
}

# 不影响产物内容的运维旋钮；语义重试与预算返工次数影响采纳的稿子，不在这里。
EXCLUDED: frozenset[str] = frozenset(
    {
        "slug",
        "llm.timeout_seconds",
        "llm.read_timeout_seconds",
        "llm.total_timeout_seconds",
        "llm.transport_max_attempts",
        "llm.max_attempts",
        "llm.script_concurrency",
        "render.tts_max_attempts",
        "render.tts_concurrency",
        "render.tts_proxy",
        "render.tts_connect_timeout",
        "render.tts_receive_timeout",
        "render.tts_chunk_timeout_seconds",
        "render.ffmpeg_path",
        "render.ffprobe_path",
    }
)

_SUBCONFIGS = frozenset(
    name
    for name, field in ProjectConfig.model_fields.items()
    if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)
)


def slice_path(root: Path, stage: str, episode: int | None) -> Path:
    """项目级阶段写 `.config/<stage>.json`，其余写 `.config/E{NN}.<stage>.json`。"""
    if stage in PROJECT_LEVEL_STAGES:
        return Path(root) / SLICE_DIR / f"{stage}.json"
    if episode is None:
        raise ValueError(f"{stage} 阶段的配置切片是按集的，必须给集号")
    return Path(root) / SLICE_DIR / f"E{episode:02d}.{stage}.json"


def slice_payload(cfg: ProjectConfig, stage: str, episode: EpisodeConfig | None) -> str:
    """序列化这个阶段读到的解析后配置，格式稳定且保留 Unicode。"""
    dumped = cfg.model_dump(mode="json", exclude={"episodes"})
    data: dict[str, Any] = {}
    for entry in STAGE_FIELDS[stage]:
        if entry == EPISODE:
            if episode is None:
                raise ValueError(f"{stage} 阶段的切片含本集配置，必须给 episode")
            data[EPISODE] = episode.model_dump(mode="json")
        elif "." in entry:
            top, leaf = entry.split(".", 1)
            data.setdefault(top, {})[leaf] = dumped[top][leaf]
        elif entry in _SUBCONFIGS:
            data[entry] = {
                key: value
                for key, value in dumped[entry].items()
                if f"{entry}.{key}" not in EXCLUDED
            }
        else:
            data[entry] = dumped[entry]
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_slice(path: Path, payload: str) -> None:
    """逐字节不变就保住 mtime；新建切片回填 epoch 0，避免升级重跑旧产物。

    升级前已改但尚未运行的配置在首次生成切片时不会失效旧产物，需 --force。
    删除 .config 后重建切片同样回填 epoch 0。
    """
    existed = path.exists()
    if atomic.write_text_if_changed(path, payload) and not existed:
        os.utime(path, ns=(0, 0))


def write_slices(cfg: ProjectConfig, episodes: Sequence[EpisodeConfig]) -> None:
    """项目级阶段写一份，其余阶段按传入的集各写一份。"""
    for stage in STAGE_FIELDS:
        if stage in PROJECT_LEVEL_STAGES:
            write_slice(slice_path(cfg.root, stage, None), slice_payload(cfg, stage, None))
            continue
        for episode in episodes:
            write_slice(
                slice_path(cfg.root, stage, episode.number),
                slice_payload(cfg, stage, episode),
            )


def stamp_path(root: Path, stage: str) -> Path:
    """全局阶段（ingest / signals）「上次跑完」的戳子。判据见 pipeline._is_fresh_stamped。"""
    return Path(root) / SLICE_DIR / f"{stage}.done"


def output_digests(outputs: Sequence[Path]) -> dict[str, str]:
    """按路径记录产物的字节身份，避免恢复旧备份时旧戳子掩盖变更。"""
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in outputs}


def invalidate_stamp(path: Path) -> None:
    """开跑前留下不完整标记；失败后不能退回按产物 mtime 判新鲜。"""
    atomic.write_text(path, "")


def touch_stamp(path: Path, outputs: Sequence[Path]) -> None:
    """阶段成功跑完后原子落盘产物身份；mtime 记录这次运行完成的时间。"""
    atomic.write_text(path, json.dumps(output_digests(outputs), sort_keys=True) + "\n")
