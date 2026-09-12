"""证明 config 层的阈值真的被下游消费了，不只是"定义了个字段"。

只有真正接线的字段才在这里测：ingest / credits / signals 全量，
render 里的 width / height / subtitle_font_name（"两份真相"修正）。
其余 RenderConfig / LLMConfig 新字段是留给后续任务接的，这里只测 default。
"""

from __future__ import annotations

import pytest

from tenmin.config import (
    CreditsConfig,
    IngestConfig,
    ProjectConfig,
    RenderConfig,
    SignalsConfig,
)
from tenmin.ingest.credits import find_credit_ranges, in_credit_window
from tenmin.ingest.normalize import build_track
from tenmin.models import DialogueLine, DialogueTrack, SubtitleCue
from tenmin.pipeline import Paths, run_ingest, run_signals, run_timeline, run_voice
from tenmin.render.subtitles import max_chars_per_line, render_ass
from tenmin.render.tts import build_tts_engine
from tenmin.render.video import build_render_args
from tenmin.signals.aggregate import build_report
from tenmin.signals.density import find_density_shifts, find_low_density
from tenmin.signals.gaps import find_silent_gaps

SRT_TWO_HALVES = """1
00:00:00,000 --> 00:00:01,000
前半句

2
00:00:01,100 --> 00:00:02,000
后半句
"""


def dline(idx: int, start: float, end: float, text: str = "喂", kind: str = "dialogue"):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind=kind)


def track(lines: list[DialogueLine], duration: float) -> DialogueTrack:
    return DialogueTrack(episode=1, duration=duration, lines=lines)


# --- IngestConfig 真的被 build_track 消费 ---


def test_ingest_config_merge_max_gap_is_honoured(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(SRT_TWO_HALVES, encoding="utf-8")

    # 两条 cue 间隔 0.1s < 默认 merge_max_gap=0.3 → 合并成一条。
    merged = build_track(srt, episode=1, merge_lines=True)
    assert len(merged.lines) == 1

    # 把阈值压到 0.05 就不该再合并。
    split = build_track(
        srt, episode=1, merge_lines=True, ingest=IngestConfig(merge_max_gap=0.05)
    )
    assert len(split.lines) == 2


def test_ingest_config_merge_max_chars_is_honoured(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(SRT_TWO_HALVES, encoding="utf-8")
    split = build_track(
        srt, episode=1, merge_lines=True, ingest=IngestConfig(merge_max_chars=5)
    )
    assert len(split.lines) == 2


def test_ingest_config_merge_max_line_seconds_is_honoured(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(SRT_TWO_HALVES, encoding="utf-8")
    split = build_track(
        srt, episode=1, merge_lines=True, ingest=IngestConfig(merge_max_line_seconds=0.5)
    )
    assert len(split.lines) == 2


# --- CreditsConfig 真的被 credits.py 消费 ---


def test_credits_config_head_window_is_honoured():
    duration = 1000.0
    assert in_credit_window(250.0, duration) is True
    narrow = CreditsConfig(credit_head_window=100.0)
    assert in_credit_window(250.0, duration, cfg=narrow) is False


def test_credits_config_ed_keyword_window_is_honoured():
    duration = 1000.0
    # 距片尾 90s，默认 ed_keyword_window_seconds=80 → 窗外。
    assert in_credit_window(910.0, duration) is False
    wide = CreditsConfig(ed_keyword_window_seconds=100.0)
    assert in_credit_window(910.0, duration, cfg=wide) is True


def test_credits_config_op_span_window_is_honoured():
    lines = [dline(1, 100.0, 130.0, "監督 山田太郎", kind="credits")]
    # 30s 跨度落在默认 op_span_min=40 之外 → 认不出 OP。
    assert find_credit_ranges(lines, duration=1000.0)[0] is None
    loose = CreditsConfig(op_span_min=10.0)
    assert find_credit_ranges(lines, duration=1000.0, cfg=loose)[0] == (100.0, 130.0)


def test_credits_config_ed_cluster_tail_is_honoured():
    lines = [dline(1, 800.0, 860.0, "作詞 某某", kind="credits")]
    # 起点距片尾 200s，默认 ed_cluster_tail_seconds=120 → 不算 ED。
    assert find_credit_ranges(lines, duration=1000.0)[1] is None
    loose = CreditsConfig(ed_cluster_tail_seconds=300.0)
    assert find_credit_ranges(lines, duration=1000.0, cfg=loose)[1] == (800.0, 860.0)


def test_credits_config_reaches_credits_through_build_track(tmp_path):
    """CreditsConfig 必须一路从 build_track 传到 credits.py，不能半路丢掉。"""
    srt = tmp_path / "a.srt"
    srt.write_text(
        "1\n00:01:40,000 --> 00:02:10,000\n監督 山田太郎\n\n"
        "2\n00:16:30,000 --> 00:16:40,000\n结束\n",
        encoding="utf-8",
    )
    assert build_track(srt, episode=1).op_range is None
    loose = build_track(srt, episode=1, credits=CreditsConfig(op_span_min=10.0))
    assert loose.op_range == (100.0, 130.0)


# --- SignalsConfig 真的被 signals/ 消费 ---


def test_signals_config_min_gap_seconds_is_honoured():
    t = track([dline(1, 0.0, 2.0), dline(2, 7.0, 9.0)], duration=9.0)
    assert len(find_silent_gaps(t)) == 1
    assert find_silent_gaps(t, cfg=SignalsConfig(min_gap_seconds=10.0)) == []


def test_signals_config_gap_strength_thresholds_are_honoured():
    t = track([dline(1, 0.0, 2.0), dline(2, 12.0, 14.0)], duration=14.0)
    # 10s 间隙：默认 (15, 8) 分档 → strength 3。
    assert find_silent_gaps(t)[0].strength == 3
    strong = SignalsConfig(gap_strong_seconds=9.0, gap_medium_seconds=5.0)
    assert find_silent_gaps(t, cfg=strong)[0].strength == 4


def test_signals_config_low_density_ratio_is_honoured():
    t = track(
        [
            dline(1, 0.0, 2.0, "十个字十个字"),
            dline(2, 2.0, 4.0, "十个字十个字"),
            dline(3, 4.0, 9.0, "少"),
        ],
        duration=9.0,
    )
    assert [s.anchor_lines for s in find_low_density(t)] == [[3]]
    # ratio 压到 0 就没有任何一行低于阈值。
    assert find_low_density(t, cfg=SignalsConfig(low_density_ratio=0.0)) == []


def test_signals_config_low_density_strength_is_honoured():
    t = track(
        [
            dline(1, 0.0, 2.0, "十个字十个字"),
            dline(2, 2.0, 4.0, "十个字十个字"),
            dline(3, 4.0, 9.0, "少"),
        ],
        duration=9.0,
    )
    assert find_low_density(t)[0].strength == 3
    tuned = SignalsConfig(low_density_strength=5)
    assert find_low_density(t, cfg=tuned)[0].strength == 5


def test_signals_config_shift_z_threshold_is_honoured():
    lines = [
        dline(idx, idx * 30.0, idx * 30.0 + 1.0, "字" * chars)
        for idx, chars in enumerate([3, 3, 3, 3, 12, 3, 3, 3])
    ]
    t = track(lines, duration=240.0)
    assert find_density_shifts(t) != []
    calm = SignalsConfig(shift_z_threshold=99.0)
    assert find_density_shifts(t, cfg=calm) == []


def test_signals_config_summary_max_chars_is_honoured():
    long_text = "一二三四五六七八九十" * 5
    t = track(
        [
            dline(1, 0.0, 1.0, "字" * 20),
            dline(2, 1.0, 2.0, "字" * 20),
            dline(3, 4.0, 30.0, long_text),
        ],
        duration=30.0,
    )
    default = build_report(t)
    short = build_report(t, cfg=SignalsConfig(summary_max_chars=4))
    assert max(len(h.summary) for h in default.highlights) == 30
    assert max(len(h.summary) for h in short.highlights) == 4


def test_signals_config_min_separation_is_honoured():
    t = track(
        [dline(1, 0.0, 2.0), dline(2, 6.0, 8.0), dline(3, 12.0, 14.0)], duration=14.0
    )
    # 两个 4s 间隙相距 2s，默认 min_separation=2.0 恰好把它们聚成一个 highlight。
    assert len(build_report(t).highlights) == 1
    assert len(build_report(t, cfg=SignalsConfig(min_separation=0.5)).highlights) == 2


# --- RenderConfig 的 width/height/subtitle_font_name 真的被消费 ---


def test_render_ass_honours_width_and_height():
    out = render_ass([SubtitleCue(start=0.0, end=1.0, text="喂")], width=1280, height=720)
    assert "PlayResX: 1280" in out
    assert "PlayResY: 720" in out


def test_max_chars_per_line_honours_width():
    assert max_chars_per_line(52, width=1280) < max_chars_per_line(52, width=1920)


def test_render_ass_wraps_against_configured_width():
    """断行必须跟着分辨率走，否则窄画布下字幕会冲出画面。"""
    cue = SubtitleCue(start=0.0, end=5.0, text="一二三四五六七八九十" * 5)
    narrow = render_ass([cue], width=960)
    wide = render_ass([cue], width=1920)
    assert narrow.count("\\N") > wide.count("\\N")


def test_pipeline_timeline_passes_render_config_into_subtitles(tmp_path, monkeypatch):
    from tenmin.render import subtitles as subtitles_module

    seen: dict[str, object] = {}
    real = subtitles_module.render_ass

    def spy(cues, **kwargs):
        seen.update(kwargs)
        return real(cues, **kwargs)

    monkeypatch.setattr("tenmin.pipeline.render_ass", spy)

    cfg = _minimal_project(tmp_path)
    cfg.render.width = 1280
    cfg.render.height = 720
    cfg.render.subtitle_font_name = "PingFang SC"
    _write_voice_and_script(cfg)

    run_timeline(cfg, episode=1, source_duration=100.0)
    assert seen["width"] == 1280
    assert seen["height"] == 720
    assert seen["font_name"] == "PingFang SC"
    assert seen["font_size"] == cfg.render.font_size


def test_build_render_args_scale_defaults_match_render_config():
    """字幕的 PlayRes 与视频的 scale 必须来自同一个 width/height，否则字幕会静默缩放。"""
    from pathlib import Path

    defaults = RenderConfig()
    args = build_render_args(
        video=Path("in.mkv"),
        timeline=_stub_timeline(),
        audio=Path("a.m4a"),
        ass=Path("s.ass"),
        out_path=Path("out.mp4"),
        encoder="libx264",
    )
    assert f"scale={defaults.width}:{defaults.height}" in " ".join(args)
    assert f"PlayResX: {defaults.width}" in render_ass([])
    assert f"PlayResY: {defaults.height}" in render_ass([])


# --- pipeline 层把 cfg 的子块传下去 ---


def test_run_ingest_and_signals_use_project_config(tmp_path):
    cfg = _minimal_project(tmp_path)
    cfg.signals.min_gap_seconds = 999.0
    run_ingest(cfg)
    reports = run_signals(cfg)
    assert reports[0].silent_gaps == []
    assert Paths(cfg.root).signals(1).is_file()


# --- helpers ---


def _stub_timeline():
    from tenmin.models import Timeline, TimelineSegment

    return Timeline(
        episode=1,
        total_seconds=10.0,
        segments=[
            TimelineSegment(
                beat_id="b1",
                source_start=0.0,
                source_end=10.0,
                timeline_start=0.0,
                timeline_end=10.0,
            )
        ],
        subtitles=[],
        narration_offsets=[],
    )


def _minimal_project(tmp_path) -> ProjectConfig:
    srt = tmp_path / "srt" / "E01.srt"
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n喂\n\n2\n00:00:20,000 --> 00:00:22,000\n嗯\n",
        encoding="utf-8",
    )
    video = tmp_path / "video" / "E01.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"")
    return ProjectConfig.model_validate(
        {
            "show": "剧名",
            "slug": "slug",
            "episodes": [{"number": 1, "srt": "srt/E01.srt", "video": "video/E01.mp4"}],
        }
    ).bind_root(tmp_path)


def _write_voice_and_script(cfg: ProjectConfig) -> None:
    from tenmin.models import Beat, Clip, Script, VoiceChunk, VoiceTrack

    paths = Paths(cfg.root)
    narration = "一二三四五六七八九十" * 5
    script = Script(
        # 原来这里写的是 episode=1 / title="标题"，Script 上并没有这两个字段，
        # extra="ignore" 时被静默丢掉（episodes 实际是 []）。改成真实字段名。
        show="剧名",
        episodes=[1],
        beats=[
            Beat(
                id="b1",
                label="开场",
                role="hook",
                narration=narration,
                clips=[Clip(episode=1, start=0.0, end=10.0)],
                est_seconds=5.0,
            )
        ],
    )
    paths.script(1).parent.mkdir(parents=True, exist_ok=True)
    paths.script(1).write_text(script.model_dump_json(), encoding="utf-8")

    voice = VoiceTrack(
        episode=1,
        total_seconds=5.0,
        chunks=[
            VoiceChunk(
                beat_id="b1",
                index=1,
                text=narration,
                path="chunk_001.mp3",
                duration=5.0,
                hold_after=0.0,
            )
        ],
    )
    paths.voice(1).parent.mkdir(parents=True, exist_ok=True)
    paths.voice(1).write_text(voice.model_dump_json(), encoding="utf-8")


# --- render.tts_* 接线 ---


def test_build_tts_engine_wires_proxy_and_timeouts():
    engine = build_tts_engine(
        RenderConfig(
            tts_proxy="http://127.0.0.1:8080",
            tts_connect_timeout=3,
            tts_receive_timeout=17,
            tts_chunk_timeout_seconds=44.0,
        )
    )
    assert engine.proxy == "http://127.0.0.1:8080"
    assert engine.connect_timeout == 3
    assert engine.receive_timeout == 17
    assert engine.chunk_timeout_seconds == 44.0


async def test_run_voice_wires_tts_max_attempts(tmp_path, monkeypatch):
    """render.tts_max_attempts 必须真的到达 synthesize_with_retry 的重试循环。"""
    from tenmin.render import tts as tts_module

    from .fakes import FlakyTTSEngine

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(tts_module, "_sleep", no_sleep)

    cfg = _minimal_project(tmp_path)
    cfg.render.tts_max_attempts = 2
    _write_voice_and_script(cfg)
    Paths(cfg.root).voice(1).unlink()

    engine = FlakyTTSEngine(fail_times=99)
    with pytest.raises(tts_module.TTSError):
        await run_voice(cfg, engine, episode=1)
    assert engine.attempts == 2
