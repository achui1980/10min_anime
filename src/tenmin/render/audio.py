"""混音：原声按 timeline 切片拼接后压低，旁白按 offset 延迟叠上去。

只负责拼 ffmpeg 命令行。命令行对不对由测试逐参数断言，ffmpeg 干得对不对是它自己的事。
"""

from __future__ import annotations

from pathlib import Path

from tenmin.config import DEFAULT_RENDER
from tenmin.models import SubtitleCue, Timeline, VoiceTrack
from tenmin.render.ffmpeg import run

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
FULL_VOLUME = 1.0


def duck_gain(duck_db: float) -> float:
    """把 dB 换成线性增益。-12dB ≈ 0.2512。"""
    return 10 ** (duck_db / 20)


def duck_volume_expr(cues: list[SubtitleCue], gain: float) -> str:
    """有旁白的区间压到 gain，其余（留白）回到原声全开。

    没有任何旁白时返回常量 1，让 volume 滤镜变成空操作。
    """
    if not cues:
        return f"{FULL_VOLUME:.4f}"
    windows = "+".join(f"between(t,{cue.start:.3f},{cue.end:.3f})" for cue in cues)
    return f"if(gt({windows},0),{gain:.4f},{FULL_VOLUME:.4f})"


def build_mix_args(
    *,
    video: Path,
    timeline: Timeline,
    track: VoiceTrack,
    voice_dir: Path,
    out_path: Path,
    duck_db: float,
    fade_out_seconds: float = 0.0,
    outro_seconds: float = 0.0,
) -> list[str]:
    """拼出混音用的 ffmpeg 参数列表（不含 ffmpeg 本身）。"""
    if not timeline.segments:
        raise ValueError("timeline 里没有任何 segment，无法混音")
    if not track.chunks:
        raise ValueError("voice track 里没有任何 chunk，无法混音")
    if len(track.chunks) != len(timeline.narration_offsets):
        raise ValueError(
            f"chunk 数 {len(track.chunks)} 与 narration_offsets 数 "
            f"{len(timeline.narration_offsets)} 不一致，timeline 与 voice 产物不匹配"
        )

    parts: list[str] = []
    for index, segment in enumerate(timeline.segments):
        parts.append(
            f"[0:a]atrim=start={segment.source_start:.3f}:end={segment.source_end:.3f},"
            f"asetpts=PTS-STARTPTS[o{index}]"
        )
    origin_labels = "".join(f"[o{i}]" for i in range(len(timeline.segments)))
    parts.append(f"{origin_labels}concat=n={len(timeline.segments)}:v=0:a=1[orig]")

    expr = duck_volume_expr(timeline.subtitles, duck_gain(duck_db))
    parts.append(f"[orig]volume='{expr}':eval=frame[ducked]")

    chunk_paths: list[str] = []
    for index, (chunk, offset) in enumerate(
        zip(track.chunks, timeline.narration_offsets, strict=True)
    ):
        chunk_paths.append(str(voice_dir / chunk.path))
        parts.append(
            f"[{index + 1}:a]adelay=delays={int(round(offset * 1000))}:all=1[n{index}]"
        )

    if len(track.chunks) == 1:
        voice_label = "[n0]"
    else:
        voice_labels = "".join(f"[n{i}]" for i in range(len(track.chunks)))
        parts.append(f"{voice_labels}amix=inputs={len(track.chunks)}:normalize=0[voice]")
        voice_label = "[voice]"
    parts.append(f"[ducked]{voice_label}amix=inputs=2:normalize=0[mix]")

    final_label = "[mix]"
    if fade_out_seconds > 0:
        fade_start = max(timeline.total_seconds - fade_out_seconds, 0.0)
        parts.append(
            f"{final_label}afade=t=out:st={fade_start:.3f}:d={fade_out_seconds:.3f}[mixfaded]"
        )
        final_label = "[mixfaded]"
    if outro_seconds > 0:
        # 片尾卡片没有声音，垫一段静音跟视频那边的黑卡对齐。
        parts.append(f"anullsrc=r=48000:cl=stereo:d={outro_seconds:.3f}[silence]")
        parts.append(f"{final_label}[silence]concat=n=2:v=0:a=1[mixfinal]")
        final_label = "[mixfinal]"

    args = ["-y", "-i", str(video)]
    for path in chunk_paths:
        args.extend(["-i", path])
    args.extend(
        [
            "-filter_complex",
            ";".join(parts),
            "-map",
            final_label,
            "-c:a",
            AUDIO_CODEC,
            "-b:a",
            AUDIO_BITRATE,
            str(out_path),
        ]
    )
    return args


def mix_audio(
    *,
    video: Path,
    timeline: Timeline,
    track: VoiceTrack,
    voice_dir: Path,
    out_path: Path,
    duck_db: float,
    fade_out_seconds: float = 0.0,
    outro_seconds: float = 0.0,
    ffmpeg: str = DEFAULT_RENDER.ffmpeg_path,
) -> Path:
    """真跑 ffmpeg 混音，返回产物路径。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        build_mix_args(
            video=video,
            timeline=timeline,
            track=track,
            voice_dir=voice_dir,
            out_path=out_path,
            duck_db=duck_db,
            fade_out_seconds=fade_out_seconds,
            outro_seconds=outro_seconds,
        ),
        ffmpeg=ffmpeg,
    )
    return out_path
