import asyncio
import json
import os
from pathlib import Path

import pytest

from tenmin.config import EpisodeConfig, ProjectConfig, load_project
from tenmin.models import (
    AudioDirection,
    Beat,
    Clip,
    DialogueTrack,
    Hold,
    LLMBeat,
    LLMClip,
    LLMScript,
    Script,
)
from tenmin.pipeline import (
    STAGES,
    Paths,
    _find_episode,
    _is_fresh,
    ingest_warnings,
    register_episode,
    run_audio,
    run_docgen,
    run_ingest,
    run_pipeline,
    run_render,
    run_signals,
    run_timeline,
    run_voice,
    stages_from,
)
from tenmin.render.ffmpeg import FFmpegError

from .fakes import FakeProvider, FakeReporter, FakeTTSEngine

# STAGES 在 v2 里扩到 8 个，voice 之后的阶段需要 TTS engine 与源视频。
# 下面这些只关心 v1 链路的用例显式限定阶段范围。
V1_STAGES = ["ingest", "signals", "script", "docgen"]


@pytest.fixture
def project(tmp_path, golden_srt_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "E02.srt").write_bytes(golden_srt_path.read_bytes())
    cfg = ProjectConfig.model_validate(
        {
            "show": "才女的侍从",
            "slug": "saijo",
            "target_seconds": 240,
            "episodes": [{"number": 2, "srt": "srt/E02.srt"}],
        }
    )
    cfg.bind_root(root)
    (root / "project.yaml").write_text("show: 才女的侍从\n", encoding="utf-8")
    return cfg


def fake_script_response(episode: int = 2):
    labels = ["Hook 开场", "阶段一：入职即地狱", "收尾：修罗场引爆"]
    roles = ["hook", "act", "outro"]
    return LLMScript(
        beats=[
            LLMBeat(
                id=f"b{i + 1}",
                label=labels[i],
                role=roles[i],
                narration="啊" * 360,
                clips=[
                    LLMClip(
                        episode=episode,
                        start=300.0 + i * 100,
                        end=305.0 + i * 100,
                        visual="画面 ➔ 特写",
                    )
                ],
            )
            for i in range(3)
        ]
    )


def test_stage_names():
    assert STAGES == [
        "ingest",
        "signals",
        "script",
        "docgen",
        "voice",
        "timeline",
        "audio",
        "render",
    ]


def test_stages_from_middle():
    assert stages_from("signals") == [
        "signals",
        "script",
        "docgen",
        "voice",
        "timeline",
        "audio",
        "render",
    ]


def test_stages_from_unknown_raises():
    with pytest.raises(ValueError):
        stages_from("nope")


# --- _is_fresh ---


def _shift_mtime(path: Path, seconds: float) -> None:
    stamp = path.stat().st_mtime + seconds
    os.utime(path, (stamp, stamp))


def _file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_is_fresh_true_when_output_is_newer(tmp_path):
    src = _file(tmp_path / "in.txt")
    out = _file(tmp_path / "out.txt")
    assert _is_fresh([out], [src]) is True


def test_is_fresh_false_when_input_is_newer(tmp_path):
    src = _file(tmp_path / "in.txt")
    out = _file(tmp_path / "out.txt")
    _shift_mtime(src, 10.0)
    assert _is_fresh([out], [src]) is False


def test_is_fresh_false_when_an_output_is_missing(tmp_path):
    src = _file(tmp_path / "in.txt")
    out = _file(tmp_path / "out.txt")
    assert _is_fresh([out, tmp_path / "gone.txt"], [src]) is False


def test_is_fresh_false_when_outputs_empty(tmp_path):
    assert _is_fresh([], [_file(tmp_path / "in.txt")]) is False


def test_is_fresh_false_on_zero_byte_output(tmp_path):
    """Ctrl-C 打断 ffmpeg 留下的空壳 .m4a/.mp4 不能被当成最新产物。

    这是纯 mtime 比较最伤的失效模式：半截产物的 mtime 恰恰是最新的，于是下一次
    跑直接 stage_skip，坏产物一路进成片。真正的解法是产物原子写（P1-G），这里
    只做最低成本的兜底。
    """
    src = _file(tmp_path / "in.txt")
    out = tmp_path / "out.m4a"
    out.write_bytes(b"")
    _shift_mtime(out, 10.0)
    assert _is_fresh([out], [src]) is False


def test_is_fresh_true_when_no_input_exists(tmp_path):
    """输入一个都不存在时视为最新（跳过）。

    这是刻意的选择，不是漏判：重跑一个没有任何输入的阶段只可能崩（build_track
    读不到 SRT）或产出垃圾，而跳过至少保住了磁盘上已有的产物。加了 project.yaml
    进 inputs 之后，真实项目里这个分支已经走不到（project.yaml 一定存在）。
    """
    out = _file(tmp_path / "out.txt")
    assert _is_fresh([out], [tmp_path / "never.txt"]) is True


def test_paths_layout(tmp_path):
    paths = Paths(tmp_path)
    assert paths.script(2).name == "E02.script.json"
    assert paths.script(2).parent.name == "03_script"
    assert paths.table(2).name == "E02.解说方案.md"
    assert paths.table(2).parent.name == "out"
    assert paths.narration(2).name == "E02.narration.txt"
    assert paths.narration(2).parent.name == "out"


def test_paths_pads_episode_number(tmp_path):
    assert Paths(tmp_path).dialogue(12).name == "E12.dialogue.json"


def test_run_ingest_writes_dialogue_json(project):
    tracks = run_ingest(project)
    assert len(tracks) == 1
    path = Paths(project.root).dialogue(2)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["episode"] == 2
    assert len(data["lines"]) > 380


def test_run_ingest_detects_op_range(project):
    track = run_ingest(project)[0]
    assert track.op_range is not None
    assert track.op_range[0] == pytest.approx(153.486, abs=0.01)


# --- run_ingest 的集长来源：视频真实时长优先，拿不到才退化到字幕末尾 ---


def test_run_ingest_uses_real_video_duration(project, monkeypatch):
    """字幕末尾早于片尾是常态，而 ED 窗/ED 聚簇/尾部间隙三个判定全挂在 duration 上。

    run_timeline 早就在用 probe_duration 了，ingest 却一直在用 max(cue.end)。
    """
    video = _prepare_video(project)
    calls: list[Path] = []

    def fake_probe(path):
        calls.append(Path(path))
        return 1416.6

    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)

    track = run_ingest(project)[0]

    assert calls == [video]
    assert track.duration == pytest.approx(1416.6)


def test_run_ingest_falls_back_when_no_video_configured(project, monkeypatch):
    """只登记了 SRT、没登记视频是合法状态，ingest 不能因此崩。"""

    def boom(path):
        raise AssertionError("没配 video 时不该去 probe")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", boom)

    assert project.episodes[0].video is None
    track = run_ingest(project)[0]
    assert track.duration == pytest.approx(1416.6, abs=1.0)


def test_run_ingest_falls_back_when_video_file_missing(project, monkeypatch):
    """project.yaml 里写了 video 但文件还没到位（下载中/换过盘）也不能崩。"""
    project.episodes[0].video = Path("video/E02.mkv")

    def boom(path):
        raise AssertionError("文件不存在时不该去 probe")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", boom)

    track = run_ingest(project)[0]
    assert track.duration == pytest.approx(1416.6, abs=1.0)


def test_run_ingest_falls_back_when_ffprobe_fails(project, monkeypatch):
    """文件在但 ffprobe 读不出（0 字节壳子、非视频文件、没装 ffprobe）也降级。"""
    _prepare_video(project)

    def boom(path):
        raise FFmpegError("ffprobe 读不出时长")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", boom)

    track = run_ingest(project)[0]
    assert track.duration == pytest.approx(1416.6, abs=1.0)


def test_run_ingest_falls_back_when_ffprobe_is_not_installed(project, monkeypatch):
    """ffprobe 根本不在 PATH 上时 subprocess 抛 FileNotFoundError，同样降级。"""
    _prepare_video(project)

    def boom(path):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", boom)

    assert run_ingest(project)[0].duration == pytest.approx(1416.6, abs=1.0)


def test_run_signals_writes_signals_json(project):
    run_ingest(project)
    reports = run_signals(project)
    # Task 10 实测：本集有 26 个 ≥3s 的无字幕间隙（候选池，不是高光集合）。
    # tests/test_aggregate.py 对同一黄金样本断言的也是 26。
    assert len(reports[0].silent_gaps) == 26
    assert Paths(project.root).signals(2).exists()


def test_ingest_warnings_reports_bad_cue_counts():
    tracks = [
        DialogueTrack(episode=2, duration=100.0, skipped_blocks=3, clamped_cues=1),
        DialogueTrack(episode=3, duration=100.0),
    ]
    messages = ingest_warnings(tracks)
    assert len(messages) == 2
    assert "E02" in messages[0] and "3 个块" in messages[0]
    assert "E02" in messages[1] and "1 条 cue" in messages[1]


def test_ingest_warnings_silent_on_clean_tracks():
    assert ingest_warnings([DialogueTrack(episode=2, duration=100.0)]) == []


def test_run_signals_without_ingest_raises(project):
    with pytest.raises(FileNotFoundError):
        run_signals(project)


@pytest.mark.asyncio
async def test_run_pipeline_end_to_end(project):
    provider = FakeProvider([fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)
    paths = Paths(project.root)
    assert paths.script(2).exists()
    assert paths.table(2).exists()
    assert paths.narration(2).exists()
    assert "解说方案" in paths.table(2).read_text(encoding="utf-8")
    assert paths.narration(2).read_text(encoding="utf-8").startswith("啊")


@pytest.mark.asyncio
async def test_run_pipeline_skips_when_fresh(project):
    provider = FakeProvider([fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)
    # 第二次跑：provider 没有剩余响应，若真的再调 LLM 就会 AssertionError
    await run_pipeline(project, provider, only=V1_STAGES)
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_run_pipeline_force_reruns_llm(project):
    provider = FakeProvider([fake_script_response(), fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)
    await run_pipeline(project, provider, only=V1_STAGES, force=True)
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_run_pipeline_only_docgen_reuses_edited_script(project):
    provider = FakeProvider([fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)
    paths = Paths(project.root)

    script = Script.model_validate_json(paths.script(2).read_text(encoding="utf-8"))
    script.beats[0].narration = "人工改过的开场"
    paths.script(2).write_text(
        script.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
    )

    await run_pipeline(project, provider, only=["docgen"], force=True)
    assert "人工改过的开场" in paths.narration(2).read_text(encoding="utf-8")
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_run_pipeline_only_unknown_stage_raises(project):
    with pytest.raises(ValueError):
        await run_pipeline(project, FakeProvider([]), only=["nope"])


@pytest.mark.asyncio
async def test_run_pipeline_only_accepts_multiple_stages(project):
    """--only 只传一个阶段，但库调用方（端到端测试）会传多个。"""
    provider = FakeProvider([])
    await run_pipeline(project, provider, only=["ingest", "signals"])
    paths = Paths(project.root)
    assert paths.dialogue(2).exists()
    assert paths.signals(2).exists()
    assert not paths.script(2).exists()
    assert provider.calls == []


@pytest.mark.asyncio
async def test_run_pipeline_batch_mode_processes_all_episodes(project, golden_srt_path):
    # register a second episode by copying the same golden SRT under a new number
    second_srt = project.root / "srt" / "E01.srt"
    second_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    project.episodes.append(EpisodeConfig(number=1, srt=Path("srt/E01.srt")))

    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    await run_pipeline(project, provider, only=V1_STAGES)

    paths = Paths(project.root)
    assert paths.script(1).exists()
    assert paths.script(2).exists()
    assert paths.table(1).exists()
    assert paths.table(2).exists()


@pytest.mark.asyncio
async def test_run_pipeline_single_episode_mode_processes_only_that_episode(
    project, golden_srt_path
):
    second_srt = project.root / "srt" / "E01.srt"
    second_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    project.episodes.append(EpisodeConfig(number=1, srt=Path("srt/E01.srt")))

    provider = FakeProvider([fake_script_response(episode=1)])
    await run_pipeline(project, provider, only=V1_STAGES, episode=1)

    paths = Paths(project.root)
    assert paths.script(1).exists()
    assert not paths.script(2).exists()


@pytest.mark.asyncio
async def test_run_pipeline_season_mode_raises(project):
    project.mode = "season"
    with pytest.raises(NotImplementedError) as exc:
        await run_pipeline(project, FakeProvider([]))
    assert "整季模式尚未实现" in str(exc.value)


@pytest.mark.asyncio
async def test_run_pipeline_from_signals_keeps_dialogue(project):
    provider = FakeProvider([fake_script_response(), fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)
    dialogue = Paths(project.root).dialogue(2)
    before = dialogue.stat().st_mtime_ns
    await run_pipeline(project, provider, only=V1_STAGES[1:], force=True)
    assert dialogue.stat().st_mtime_ns == before
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_run_pipeline_reruns_every_stage_when_project_yaml_changes(project):
    """project.yaml 是每个阶段的隐式输入：改了阈值/glossary/render 必须让产物失效。

    P1-B 把大量经验阈值搬进了 project.yaml，而 _is_fresh 的 inputs 里根本没有它，
    于是「改 credits.op_span_min 再重跑」会被全部 stage_skip，用户看到的产物跟
    改动前一模一样，且没有任何提示。
    """
    provider = FakeProvider([fake_script_response(), fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)

    _shift_mtime(project.config_path, 10.0)

    reporter = FakeReporter()
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    for stage in V1_STAGES:
        assert ("stage_start", stage) in reporter.calls, stage
        assert ("stage_skip", stage) not in reporter.calls, stage


@pytest.mark.asyncio
async def test_run_pipeline_reruns_render_stages_when_project_yaml_changes(
    project, monkeypatch
):
    """voice 之后的阶段同样吃 project.yaml（render.font_size / duck_db / 编码器…）。"""
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder: 1400.0)
    # voice 重跑时 synthesize_track 会复用上一轮落盘的 chunk 并用真 ffprobe 量时长，
    # 而 FakeTTSEngine 写的是假 mp3 字节。
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 8.0)
    monkeypatch.setattr("tenmin.render.audio.run", _touch_output)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    stages = ["voice", "timeline", "audio", "render"]
    await run_pipeline(
        project, FakeProvider([]), only=stages, tts_engine=FakeTTSEngine([8.0] * 3)
    )

    _shift_mtime(project.config_path, 10.0)

    reporter = FakeReporter()
    await run_pipeline(
        project,
        FakeProvider([]),
        only=stages,
        tts_engine=FakeTTSEngine([8.0] * 3),
        reporter=reporter,
    )
    for stage in stages:
        assert ("stage_start", stage) in reporter.calls, stage
        assert ("stage_skip", stage) not in reporter.calls, stage


def test_run_docgen_without_script_raises(project):
    with pytest.raises(FileNotFoundError):
        run_docgen(project, episode=2)


def _write_script(path: Path, script: Script) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script.model_dump_json(indent=2), encoding="utf-8")


def _touch_output(args: list[str]) -> str:
    """假的 ffmpeg：不跑编码，只把输出文件创建出来。

    刻意写非空字节：_is_fresh 现在把 0 字节产物当「被打断的半截产物」判成不新鲜，
    写 b"" 会让所有「产物已是最新所以跳过」的断言被这条兜底规则掩盖掉。
    """
    out = Path(args[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\x00")
    return ""


def _touch_output_with_progress(
    args: list[str], *, total_seconds: float, on_progress=None
) -> str:
    """假的 ffmpeg（run_with_progress 版）：不跑编码，只把输出文件创建出来。"""
    if on_progress is not None:
        on_progress(1.0)
    return _touch_output(args)


def render_script() -> Script:
    """两个 beat、两个 clip 的最小剧本，配 FakeTTSEngine([8.0, 10.0, 10.0]) 用。"""
    return Script(
        show="才女的侍从",
        episodes=[2],
        target_seconds=30.0,
        beats=[
            Beat(
                id="b1",
                label="Hook",
                role="hook",
                narration="第一句。第二句。",
                clips=[Clip(episode=2, start=100.0, end=140.0)],
                audio=AudioDirection(
                    holds=[Hold(at=1.0, duration=2.0, quote="第一句")]
                ),
            ),
            Beat(
                id="b2",
                label="收尾",
                role="outro",
                narration="第三句。",
                clips=[Clip(episode=2, start=200.0, end=220.0)],
            ),
        ],
    )


def _prepare_video(cfg: ProjectConfig) -> Path:
    """造一个空壳视频文件并写进 config，供需要 video_path 的阶段用。"""
    video = cfg.root / "E02.mkv"
    video.write_bytes(b"")
    cfg.episodes[0].video = Path("E02.mkv")
    return video


def test_paths_render_layout(tmp_path):
    paths = Paths(tmp_path / "akujo2")
    assert paths.voice_dir(2).name == "E02"
    assert paths.voice_dir(2).parent.name == "04_voice"
    assert paths.voice(2).name == "E02.voice.json"
    assert paths.voice(2).parent.name == "04_voice"
    assert paths.timeline(2).name == "E02.timeline.json"
    assert paths.timeline(2).parent.name == "05_timeline"
    assert paths.subtitles(2).name == "E02.ass"
    assert paths.subtitles(2).parent.name == "05_timeline"
    assert paths.mixed_audio(2).name == "E02.mixed.m4a"
    assert paths.mixed_audio(2).parent.name == "06_audio"
    assert paths.video(2).name == "E02.mp4"
    assert paths.video(2).parent.name == "07_render"


@pytest.mark.asyncio
async def test_run_voice_writes_voice_json(project):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    engine = FakeTTSEngine([8.0, 10.0, 10.0])

    track, warnings = await run_voice(project, engine, episode=2)

    assert warnings == []
    assert [chunk.path for chunk in track.chunks] == [
        "chunk_001.mp3",
        "chunk_002.mp3",
        "chunk_003.mp3",
    ]
    assert track.total_seconds == pytest.approx(30.0)
    assert paths.voice(2).exists()
    assert (paths.voice_dir(2) / "chunk_001.mp3").exists()


@pytest.mark.asyncio
async def test_run_voice_without_script_raises(project):
    with pytest.raises(FileNotFoundError):
        await run_voice(project, FakeTTSEngine([]), episode=2)


@pytest.mark.asyncio
async def test_run_timeline_writes_timeline_and_ass(project):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    await run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2)

    timeline, warnings = run_timeline(project, episode=2, source_duration=1400.0)

    assert warnings == []
    assert len(timeline.segments) == 2
    assert timeline.segments[0].source_end == pytest.approx(120.0)
    assert timeline.segments[1].source_end == pytest.approx(210.0)
    assert timeline.segments[1].timeline_end == pytest.approx(30.0)
    assert timeline.narration_offsets == pytest.approx([0.0, 10.0, 20.0])
    assert paths.timeline(2).exists()
    assert paths.subtitles(2).read_text(encoding="utf-8").startswith("[Script Info]")


def test_run_timeline_without_voice_raises(project):
    _write_script(Paths(project.root).script(2), render_script())
    with pytest.raises(FileNotFoundError):
        run_timeline(project, episode=2, source_duration=1400.0)


def test_run_timeline_probes_source_when_duration_missing(project, monkeypatch):
    _write_script(Paths(project.root).script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    video = _prepare_video(project)
    calls: list[Path] = []

    def fake_probe(path):
        calls.append(Path(path))
        return 1400.0

    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)

    timeline, warnings = run_timeline(project, episode=2)

    assert calls == [video]
    assert warnings == []
    assert timeline.total_seconds == pytest.approx(30.0)


def test_run_timeline_uses_config_font_size(project):
    _write_script(Paths(project.root).script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    project.render.font_size = 72

    run_timeline(project, episode=2, source_duration=1400.0)

    ass = Paths(project.root).subtitles(2).read_text(encoding="utf-8")
    assert "Lantinghei SC,72," in ass


def test_run_audio_invokes_ffmpeg(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    captured: list[list[str]] = []

    def fake_run(args):
        captured.append(list(args))
        return _touch_output(args)

    monkeypatch.setattr("tenmin.render.audio.run", fake_run)

    out = run_audio(project, episode=2)

    assert out == paths.mixed_audio(2)
    assert out.exists()
    assert captured[0][:2] == ["-y", "-i"]
    assert captured[0][-1] == str(paths.mixed_audio(2))
    assert "amix=inputs=2:normalize=0[mix]" in captured[0][captured[0].index("-filter_complex") + 1]


def test_run_audio_without_timeline_raises(project):
    _write_script(Paths(project.root).script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _prepare_video(project)
    with pytest.raises(FileNotFoundError):
        run_audio(project, episode=2)


def test_run_render_invokes_ffmpeg(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    paths.mixed_audio(2).parent.mkdir(parents=True, exist_ok=True)
    paths.mixed_audio(2).write_bytes(b"")
    captured: list[list[str]] = []

    def fake_run_with_progress(args, *, total_seconds, on_progress=None):
        captured.append(list(args))
        return _touch_output(args)

    monkeypatch.setattr("tenmin.render.video.run_with_progress", fake_run_with_progress)

    out = run_render(project, episode=2)

    assert out == paths.video(2)
    assert out.exists()
    assert "-movflags" in captured[0]
    assert captured[0][-1] == str(paths.video(2))


def test_run_render_without_audio_raises(project):
    _write_script(Paths(project.root).script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    with pytest.raises(FileNotFoundError) as exc:
        run_render(project, episode=2)
    assert "audio 阶段" in str(exc.value)


@pytest.mark.asyncio
async def test_run_pipeline_from_voice_runs_render_stages(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder: 1400.0)
    monkeypatch.setattr("tenmin.render.audio.run", _touch_output)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    warnings = await run_pipeline(
        project,
        FakeProvider([]),
        from_stage="voice",
        tts_engine=FakeTTSEngine([8.0, 10.0, 10.0]),
    )

    assert warnings == []
    assert paths.voice(2).exists()
    assert paths.timeline(2).exists()
    assert paths.subtitles(2).exists()
    assert paths.mixed_audio(2).exists()
    assert paths.video(2).exists()


@pytest.mark.asyncio
async def test_run_pipeline_batch_mode_runs_full_pipeline_for_all_episodes(
    project, golden_srt_path, monkeypatch
):
    """批量模式（不传 episode，也不限制 only/from_stage）是这个 feature 的核心承诺：
    对每一集都要跑完整 8 个阶段，一直到渲出成片，不能只覆盖到 docgen。"""
    second_srt = project.root / "srt" / "E01.srt"
    second_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    project.episodes.append(EpisodeConfig(number=1, srt=Path("srt/E01.srt")))

    for episode_cfg in project.episodes:
        video_name = f"E{episode_cfg.number:02d}.mkv"
        (project.root / video_name).write_bytes(b"")
        episode_cfg.video = Path(video_name)

    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder: 1400.0)
    monkeypatch.setattr("tenmin.render.audio.run", _touch_output)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    # 处理顺序跟 cfg.episodes 一致：project 先注册了第 2 集，再 append 第 1 集。
    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    tts_engine = FakeTTSEngine([8.0] * 20)

    warnings = await run_pipeline(project, provider, tts_engine=tts_engine)

    assert warnings == []
    paths = Paths(project.root)
    for number in (1, 2):
        assert paths.dialogue(number).exists()
        assert paths.signals(number).exists()
        assert paths.script(number).exists()
        assert paths.table(number).exists()
        assert paths.narration(number).exists()
        assert paths.voice(number).exists()
        assert paths.timeline(number).exists()
        assert paths.subtitles(number).exists()
        assert paths.mixed_audio(number).exists()
        assert paths.video(number).exists()


@pytest.mark.asyncio
async def test_run_pipeline_skips_voice_when_fresh(project):
    _write_script(Paths(project.root).script(2), render_script())
    engine = FakeTTSEngine([8.0, 10.0, 10.0])

    await run_pipeline(project, FakeProvider([]), only=["voice"], tts_engine=engine)
    assert len(engine.calls) == 3

    # 预置时长已用尽：真的再合成一次就会 AssertionError
    await run_pipeline(project, FakeProvider([]), only=["voice"], tts_engine=engine)
    assert len(engine.calls) == 3


@pytest.mark.asyncio
async def test_run_pipeline_voice_only_skips_preflight(project, monkeypatch):
    _write_script(Paths(project.root).script(2), render_script())

    def boom(video, encoder):
        raise AssertionError("只跑 voice 不该做 ffmpeg 前置检查")

    monkeypatch.setattr("tenmin.pipeline.preflight", boom)

    await run_pipeline(
        project,
        FakeProvider([]),
        only=["voice"],
        tts_engine=FakeTTSEngine([8.0, 10.0, 10.0]),
    )

    assert Paths(project.root).voice(2).exists()


def test_find_episode_returns_matching_config(project):
    episode_cfg = _find_episode(project, 2)
    assert episode_cfg.number == 2


def test_find_episode_raises_when_not_registered(project):
    with pytest.raises(ValueError, match="没有注册"):
        _find_episode(project, 99)


def test_register_episode_copies_files_and_appends_yaml_entry(tmp_path, golden_srt_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "video").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nmode: single_episode\n"
        "target_seconds: 240\nepisodes:\n- number: 2\n  srt: srt/E02.srt\n"
        "  video: video/E02.mp4\n",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)

    source_srt = tmp_path / "incoming_E01.srt"
    source_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_video = tmp_path / "incoming_E01.mp4"
    source_video.write_bytes(b"fake video bytes")

    updated_cfg = register_episode(
        cfg, episode=1, srt=source_srt, video=source_video
    )

    assert (root / "srt" / "E01.srt").exists()
    assert (root / "video" / "E01.mp4").exists()
    assert len(updated_cfg.episodes) == 2
    new_entry = next(e for e in updated_cfg.episodes if e.number == 1)
    assert new_entry.srt == Path("srt/E01.srt")
    assert new_entry.video == Path("video/E01.mp4")

    # reload from disk to confirm the yaml file itself was updated
    reloaded = load_project(yaml_path)
    assert len(reloaded.episodes) == 2
    assert any(e.number == 1 for e in reloaded.episodes)
    assert any(e.number == 2 for e in reloaded.episodes)


def test_register_episode_updates_existing_entry_in_place(tmp_path, golden_srt_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "video").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nmode: single_episode\n"
        "target_seconds: 240\nepisodes:\n- number: 2\n  srt: srt/E02.srt\n"
        "  video: video/E02.mp4\n",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)

    source_srt = tmp_path / "replacement_E02.srt"
    source_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_video = tmp_path / "replacement_E02.mp4"
    source_video.write_bytes(b"replacement video bytes")

    updated_cfg = register_episode(
        cfg, episode=2, srt=source_srt, video=source_video
    )

    assert len(updated_cfg.episodes) == 1
    assert (root / "video" / "E02.mp4").read_bytes() == b"replacement video bytes"


def test_register_episode_preserves_other_episodes_op_range(tmp_path, golden_srt_path):
    """register_episode() 注册一个不相关的新集时，不能把其它已注册集的
    op_range/ed_range 从 project.yaml 里静默抹掉（Task 12 code review 发现的 bug）。
    """
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "video").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nmode: single_episode\n"
        "target_seconds: 240\nepisodes:\n- number: 2\n  srt: srt/E02.srt\n"
        "  video: video/E02.mp4\n"
        "  op_range: [153.486, 224.681]\n"
        "  ed_range: [1300.0, 1350.5]\n",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)

    source_srt = tmp_path / "incoming_E01.srt"
    source_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_video = tmp_path / "incoming_E01.mp4"
    source_video.write_bytes(b"fake video bytes")

    register_episode(cfg, episode=1, srt=source_srt, video=source_video)

    # reload from disk: episode 2's op_range/ed_range must survive the rewrite
    reloaded = load_project(yaml_path)
    episode_2 = next(e for e in reloaded.episodes if e.number == 2)
    assert episode_2.op_range == (153.486, 224.681)
    assert episode_2.ed_range == (1300.0, 1350.5)


@pytest.mark.asyncio
async def test_run_pipeline_reports_stage_start_and_done(project):
    reporter = FakeReporter()
    provider = FakeProvider([fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    calls = reporter.calls
    assert ("stage_start", "ingest") in calls
    assert ("stage_done", "ingest") in calls
    assert ("stage_start", "signals") in calls
    assert ("stage_done", "signals") in calls
    assert ("episode_start", 2, 1, 1) in calls
    assert ("stage_start", "script") in calls
    assert ("stage_done", "script") in calls
    assert ("stage_start", "docgen") in calls
    assert ("stage_done", "docgen") in calls
    # ingest 必须先于 signals，signals 必须先于 script
    assert calls.index(("stage_done", "ingest")) < calls.index(("stage_start", "signals"))
    assert calls.index(("stage_done", "signals")) < calls.index(("stage_start", "script"))


@pytest.mark.asyncio
async def test_run_pipeline_reports_stage_skip_on_second_run(project):
    provider = FakeProvider([fake_script_response(), fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)
    reporter = FakeReporter()
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    calls = reporter.calls
    assert ("stage_skip", "ingest") in calls
    assert ("stage_skip", "signals") in calls
    assert ("stage_skip", "script") in calls
    assert ("stage_skip", "docgen") in calls
    assert ("stage_start", "ingest") not in calls
    assert ("stage_start", "script") not in calls


@pytest.mark.asyncio
async def test_run_pipeline_batch_mode_reports_episode_start_for_each_episode(project):
    project.episodes.append(EpisodeConfig(number=1, srt=project.episodes[0].srt))
    reporter = FakeReporter()
    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    calls = reporter.calls
    assert ("episode_start", 2, 1, 2) in calls
    assert ("episode_start", 1, 2, 2) in calls


@pytest.mark.asyncio
async def test_run_voice_reports_substep_progress(project):
    _write_script(Paths(project.root).script(2), render_script())
    engine = FakeTTSEngine([8.0, 10.0, 10.0])
    reporter = FakeReporter()
    await run_voice(project, engine, episode=2, reporter=reporter)
    substeps = [call for call in reporter.calls if call[0] == "substep"]
    assert len(substeps) == 3
    assert substeps[-1] == ("substep", "voice", 3, 3, "第三句。")


def test_run_render_reports_substep_progress(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    engine = FakeTTSEngine([8.0, 10.0, 10.0])
    asyncio.run(run_voice(project, engine, episode=2))
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    paths.mixed_audio(2).parent.mkdir(parents=True, exist_ok=True)
    paths.mixed_audio(2).write_bytes(b"")

    def fake_run_with_progress(args, *, total_seconds, on_progress=None):
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        if on_progress is not None:
            on_progress(1.0)
        return ""

    monkeypatch.setattr("tenmin.render.video.run_with_progress", fake_run_with_progress)
    reporter = FakeReporter()
    run_render(project, episode=2, reporter=reporter)
    assert ("substep", "render", 100, 100, "") in reporter.calls
