"""端到端：真 Edge-TTS + 真 ffmpeg 跑一遍。默认 skip，跑法 uv run pytest -m render。

要求 project.yaml 里的 episodes[].video 写绝对路径——测试会把项目根换到 tmp 目录，
相对路径会失效。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tenmin.config import load_project
from tenmin.models import Script, Timeline
from tenmin.pipeline import Paths, run_audio, run_render, run_timeline, run_voice
from tenmin.render.ffmpeg import has_encoder, has_filter, probe_duration
from tenmin.render.tts import build_tts_engine

pytestmark = pytest.mark.render

PROJECT_FILE = Path("work") / "akujo2" / "project.yaml"


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    if not PROJECT_FILE.exists():
        pytest.skip(f"缺少 {PROJECT_FILE}")

    original = load_project(PROJECT_FILE)
    if not original.episodes or original.episodes[0].video is None:
        pytest.skip("project.yaml 没配 episodes[].video")
    video = original.video_path(original.episodes[0])
    if not video.is_absolute():
        pytest.skip("端到端测试要求 video 写绝对路径")
    if not video.exists():
        pytest.skip(f"源视频不存在：{video}")
    if not has_filter("subtitles"):
        pytest.skip(
            "ffmpeg 没编 libass，装法："
            "brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass"
        )
    if not has_encoder(original.render.video_encoder):
        pytest.skip(f"ffmpeg 没有 {original.render.video_encoder} 编码器")

    source_script = Paths(original.root).script
    if not source_script.exists():
        pytest.skip(f"缺少 {source_script}，请先跑 script 阶段")

    # 换到临时根目录，只留第一个 beat，绝不污染真实产物。
    root = tmp_path_factory.mktemp("render_e2e")
    cfg = load_project(PROJECT_FILE).bind_root(root)
    script = Script.model_validate_json(source_script.read_text(encoding="utf-8"))
    script.beats = script.beats[:1]
    trimmed = Paths(root).script
    trimmed.parent.mkdir(parents=True, exist_ok=True)
    trimmed.write_text(script.model_dump_json(indent=2), encoding="utf-8")

    asyncio.run(run_voice(cfg, build_tts_engine(cfg.render)))
    run_timeline(cfg)
    run_audio(cfg)
    return cfg, run_render(cfg)


def test_e2e_produces_mp4(rendered):
    _, mp4 = rendered
    assert mp4.exists()
    assert mp4.stat().st_size > 0


def test_e2e_keeps_every_intermediate_artifact(rendered):
    cfg, _ = rendered
    paths = Paths(cfg.root)
    episode = cfg.episodes[0].number
    assert paths.voice(episode).exists()
    assert paths.timeline(episode).exists()
    assert paths.subtitles(episode).exists()
    assert paths.mixed_audio(episode).exists()


def test_e2e_duration_matches_timeline(rendered):
    cfg, mp4 = rendered
    episode = cfg.episodes[0].number
    timeline = Timeline.model_validate_json(
        Paths(cfg.root).timeline(episode).read_text(encoding="utf-8")
    )
    assert probe_duration(mp4) == pytest.approx(timeline.total_seconds, abs=1.0)
