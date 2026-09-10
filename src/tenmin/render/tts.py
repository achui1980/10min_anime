"""TTS。跟 v1 的 LLMProvider 完全同构：真实引擎藏在 Protocol 后面，测试用假引擎。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from tenmin.config import RenderConfig
from tenmin.models import Beat, Script, VoiceChunk, VoiceTrack
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.chunks import plan_chunks
from tenmin.render.ffmpeg import probe_duration

TTS_MAX_ATTEMPTS = 3


@runtime_checkable
class TTSEngine(Protocol):
    async def synthesize(self, text: str, out_path: Path) -> float:
        """合成一段语音，返回真实时长（秒）。"""
        ...


class EdgeTTSEngine:
    def __init__(self, voice: str = "zh-CN-YunxiNeural", rate: str = "+0%") -> None:
        self.voice = voice
        self.rate = rate

    async def synthesize(self, text: str, out_path: Path) -> float:
        import edge_tts

        out_path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        await communicate.save(str(out_path))
        return probe_duration(out_path)


def build_tts_engine(cfg: RenderConfig) -> TTSEngine:
    return EdgeTTSEngine(voice=cfg.voice, rate=cfg.rate)


async def synthesize_with_retry(
    engine: TTSEngine, text: str, out_path: Path, *, label: str
) -> float:
    """单个 chunk 重试 TTS_MAX_ATTEMPTS 次。失败时报清楚是哪个 chunk、原文是什么。"""
    last_error = ""
    for _ in range(TTS_MAX_ATTEMPTS):
        try:
            return await engine.synthesize(text, out_path)
        except Exception as error:  # noqa: BLE001 - 网络层什么都可能抛
            last_error = str(error)
    raise RuntimeError(
        f"{label} 连续 {TTS_MAX_ATTEMPTS} 次合成失败：{text!r}\n{last_error}"
    )


async def synthesize_track(
    script: Script,
    episode: int,
    voice_dir: Path,
    engine: TTSEngine,
    *,
    reuse: bool = True,
    reporter: ProgressReporter | None = None,
) -> tuple[VoiceTrack, list[str]]:
    """合成整集旁白。chunk 独立落盘，重跑只补缺的那几个。"""
    reporter = reporter or NullProgressReporter()
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    planned_by_beat: list[tuple[Beat, list[tuple[str, float]]]] = []
    for beat in script.beats:
        planned = plan_chunks(beat)
        if not planned:
            warnings.append(f"beat {beat.id} 没有旁白文本，已跳过配音")
            continue
        planned_by_beat.append((beat, planned))

    total_chunks = sum(len(planned) for _, planned in planned_by_beat)

    chunks: list[VoiceChunk] = []
    serial = 0
    for beat, planned in planned_by_beat:
        for index, (text, hold_after) in enumerate(planned, start=1):
            serial += 1
            reporter.substep("voice", serial, total_chunks, text[:20])
            filename = f"chunk_{serial:03d}.mp3"
            out_path = voice_dir / filename
            label = f"beat {beat.id} 的第 {index} 个 chunk"
            if reuse and out_path.is_file() and out_path.stat().st_size > 0:
                duration = probe_duration(out_path)
            else:
                duration = await synthesize_with_retry(engine, text, out_path, label=label)
            chunks.append(
                VoiceChunk(
                    beat_id=beat.id,
                    index=index,
                    text=text,
                    path=filename,
                    duration=duration,
                    hold_after=hold_after,
                )
            )
    total = sum(chunk.duration + chunk.hold_after for chunk in chunks)
    return VoiceTrack(episode=episode, chunks=chunks, total_seconds=total), warnings
