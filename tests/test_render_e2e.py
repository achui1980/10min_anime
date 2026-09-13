"""端到端：真 Edge-TTS + 真 ffmpeg 跑一遍。默认 skip，跑法 uv run pytest -m render。

自己去 work/ 底下找一个「能真跑」的项目：有注册好的源片、源片文件在、而且那一集
已经有 script.json。找不到就 skip 并说清缺什么。

原来这里写死 work/akujo2 并要求 episodes[].video 是绝对路径，于是在一个 video 写
相对路径（或压根没配 video）的仓库里永远 skip —— 加上带 render marker 默认不跑，
这个文件坏了很久没人发现。绝对路径那条限制是可以自己消掉的：把根换到 tmp 之前，
先按**原**项目根把源片路径解析成绝对路径再塞回去。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tenmin.config import EpisodeConfig, ProjectConfig, load_project
from tenmin.models import Script, Timeline
from tenmin.pipeline import Paths, run_audio, run_render, run_timeline, run_voice
from tenmin.render.audio import mixed_total_seconds
from tenmin.render.ffmpeg import has_encoder, has_filter, probe_duration
from tenmin.render.tts import build_tts_engine

pytestmark = pytest.mark.render

WORK_DIR = Path("work")


def _find_runnable() -> tuple[ProjectConfig, EpisodeConfig, Path]:
    """找一个有源片 + 有 script.json 的 (项目, 集, 源片绝对路径)。"""
    candidates = sorted(WORK_DIR.glob("*/project.yaml"))
    if not candidates:
        pytest.skip(f"{WORK_DIR} 底下没有任何 project.yaml")
    missing: list[str] = []
    for config_path in candidates:
        cfg = load_project(config_path)
        for episode in cfg.episodes:
            if episode.video is None:
                missing.append(f"{config_path}: E{episode.number:02d} 没配 video")
                continue
            video = cfg.video_path(episode).resolve()
            if not video.is_file():
                missing.append(f"{config_path}: 源片不存在 {video}")
                continue
            script = Paths(cfg.root).script(episode.number)
            if not script.is_file():
                missing.append(f"{config_path}: 缺少 {script}，请先跑 script 阶段")
                continue
            return cfg, episode, video
    pytest.skip("work/ 底下没有能端到端跑的项目：\n" + "\n".join(missing))


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    original, episode, video = _find_runnable()
    number = episode.number
    if not has_filter("subtitles"):
        pytest.skip(
            "ffmpeg 没编 libass，装法："
            "brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass"
        )
    if not has_encoder(original.render.video_encoder):
        pytest.skip(f"ffmpeg 没有 {original.render.video_encoder} 编码器")
    if original.render.outro_card_seconds > 0 and not has_filter("drawtext"):
        pytest.skip("ffmpeg 没编 drawtext 滤镜，画不了片尾黑卡")

    # 换到临时根目录，只留第一个 beat：绝不污染真实产物，也别为了一次测试跑满 4 分钟
    # 的 TTS + 编码。
    root = tmp_path_factory.mktemp("render_e2e")
    cfg = load_project(original.config_path).bind_root(root)
    # 源片路径已经按**原**项目根解析成绝对路径了，塞回去，换根才不会把它指到 tmp 里。
    for candidate in cfg.episodes:
        if candidate.number == number:
            candidate.video = video

    script = Script.model_validate_json(
        Paths(original.root).script(number).read_text(encoding="utf-8")
    )
    script.beats = script.beats[:1]
    trimmed = Paths(root).script(number)
    trimmed.parent.mkdir(parents=True, exist_ok=True)
    trimmed.write_text(script.model_dump_json(indent=2), encoding="utf-8")

    asyncio.run(run_voice(cfg, build_tts_engine(cfg.render), episode=number))
    run_timeline(cfg, episode=number)
    run_audio(cfg, episode=number)
    return cfg, number, run_render(cfg, episode=number)


def _timeline(cfg: ProjectConfig, number: int) -> Timeline:
    return Timeline.model_validate_json(
        Paths(cfg.root).timeline(number).read_text(encoding="utf-8")
    )


def test_e2e_produces_mp4(rendered):
    _, _, mp4 = rendered
    assert mp4.exists()
    assert mp4.stat().st_size > 0


def test_e2e_keeps_every_intermediate_artifact(rendered):
    cfg, number, _ = rendered
    paths = Paths(cfg.root)
    assert paths.voice(number).exists()
    assert paths.timeline(number).exists()
    assert paths.subtitles(number).exists()
    assert paths.mixed_audio(number).exists()


def test_e2e_mixed_audio_length_is_pinned_to_the_timeline(rendered):
    """混音产物的长度必须精确等于「timeline 声明的长度 + 片尾卡片」。

    这是 build_mix_args 那道 apad+atrim 的端到端验收：原来输出长度是 amix
    duration=longest 的副产物，而 render 阶段 `-c:a copy` 会让 mp4 的容器时长
    反过来被它决定。容差 50ms 是留给 AAC 的编码器 padding（容器声明的时长含它）。
    """
    cfg, number, _ = rendered
    timeline = _timeline(cfg, number)
    expected = mixed_total_seconds(timeline, cfg.render.outro_card_seconds)
    actual = probe_duration(
        Paths(cfg.root).mixed_audio(number), ffprobe=cfg.render.ffprobe_path
    )
    assert actual == pytest.approx(expected, abs=0.05)


def test_e2e_duration_matches_timeline(rendered):
    """成片时长要含片尾卡片 —— 原来这里只比 total_seconds，卡片开着就必然差 3 秒。"""
    cfg, number, mp4 = rendered
    timeline = _timeline(cfg, number)
    expected = timeline.total_seconds + cfg.render.outro_card_seconds
    assert probe_duration(mp4, ffprobe=cfg.render.ffprobe_path) == pytest.approx(
        expected, abs=1.0
    )
