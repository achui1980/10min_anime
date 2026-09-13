import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from tenmin.atomic import part_path
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
    run_script,
    run_signals,
    run_timeline,
    run_voice,
    stages_from,
)
from tenmin.render.ffmpeg import FFmpegError
from tenmin.script.llm import LLMSchemaError

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
                        # 360 字 ≈ 80 秒旁白，画面就给 80 秒。原来这里只给 5 秒，
                        # 拉伸倍率 16 倍——validate 的 A3「画面/旁白预算」会报 warning，
                        # 而这个 fixture 的 clip 长度本来就是随手写的无关变量。
                        end=300.0 + i * 100 + 80.0,
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


# work/ 下有 11 集的存量产物，产物路径改一个字符就等于全部存量产物失效（流水线会
# 认为什么都没跑过，重新调 LLM、重新 TTS、重新渲染）。所以这里把每一条路径按字面量
# 锁死：Paths 的实现怎么重构都行，拼出来的字符串必须逐字节不变。
FROZEN_LAYOUT = {
    "dialogue": "01_dialogue/E02.dialogue.json",
    "signals": "02_signals/E02.signals.json",
    "script": "03_script/E02.script.json",
    "script_raw": "03_script/E02.raw.txt",
    "table": "out/E02.解说方案.md",
    "narration": "out/E02.narration.txt",
    "voice_dir": "04_voice/E02",
    "voice": "04_voice/E02.voice.json",
    "timeline": "05_timeline/E02.timeline.json",
    "subtitles": "05_timeline/E02.ass",
    "mixed_audio": "06_audio/E02.mixed.m4a",
    "video": "07_render/E02.mp4",
}


def _artifact_methods() -> set[str]:
    return {
        name
        for name, obj in vars(Paths).items()
        if callable(obj) and not name.startswith("_")
    }


@pytest.mark.parametrize("kind", sorted(FROZEN_LAYOUT))
def test_paths_layout_is_frozen(tmp_path, kind):
    expected = tmp_path.joinpath(*FROZEN_LAYOUT[kind].split("/"))
    assert getattr(Paths(tmp_path), kind)(2) == expected


def test_paths_exposes_exactly_the_frozen_artifacts(tmp_path):
    """新增一种产物就必须同时进 FROZEN_LAYOUT，否则它的路径没人锁。"""
    assert _artifact_methods() == set(FROZEN_LAYOUT)


@pytest.mark.parametrize("episode,stem", [(1, "E01"), (12, "E12"), (123, "E123")])
def test_paths_pad_episode_number_consistently(tmp_path, episode, stem):
    """集号前缀是 E + 至少两位。三位集号不截断（f"E{123:02d}" == "E123"）。"""
    paths = Paths(tmp_path)
    for kind in FROZEN_LAYOUT:
        assert getattr(paths, kind)(episode).name.startswith(f"{stem}.") or getattr(
            paths, kind
        )(episode).name == stem, kind


def test_paths_are_rooted_at_the_given_root(tmp_path):
    paths = Paths(tmp_path / "work" / "saijo")
    for kind in FROZEN_LAYOUT:
        assert getattr(paths, kind)(2).is_relative_to(tmp_path / "work" / "saijo"), kind


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

    def fake_probe(path, **_):
        calls.append(Path(path))
        return 1416.6

    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)

    track = run_ingest(project)[0]

    assert calls == [video]
    assert track.duration == pytest.approx(1416.6)


def test_run_ingest_falls_back_when_no_video_configured(project, monkeypatch):
    """只登记了 SRT、没登记视频是合法状态，ingest 不能因此崩。"""

    def boom(path, **_):
        raise AssertionError("没配 video 时不该去 probe")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", boom)

    assert project.episodes[0].video is None
    track = run_ingest(project)[0]
    assert track.duration == pytest.approx(1416.6, abs=1.0)


def test_run_ingest_falls_back_when_video_file_missing(project, monkeypatch):
    """project.yaml 里写了 video 但文件还没到位（下载中/换过盘）也不能崩。"""
    project.episodes[0].video = Path("video/E02.mkv")

    def boom(path, **_):
        raise AssertionError("文件不存在时不该去 probe")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", boom)

    track = run_ingest(project)[0]
    assert track.duration == pytest.approx(1416.6, abs=1.0)


def test_run_ingest_falls_back_when_ffprobe_fails(project, monkeypatch):
    """文件在但 ffprobe 读不出（0 字节壳子、非视频文件、没装 ffprobe）也降级。"""
    _prepare_video(project)

    def boom(path, **_):
        raise FFmpegError("ffprobe 读不出时长")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", boom)

    track = run_ingest(project)[0]
    assert track.duration == pytest.approx(1416.6, abs=1.0)


def test_run_ingest_falls_back_when_ffprobe_is_not_installed(project, monkeypatch):
    """ffprobe 根本不在 PATH 上时 subprocess 抛 FileNotFoundError，同样降级。"""
    _prepare_video(project)

    def boom(path, **_):
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
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    # voice 重跑时 synthesize_track 会复用上一轮落盘的 chunk 并用真 ffprobe 量时长，
    # 而 FakeTTSEngine 写的是假 mp3 字节。
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 8.0)
    monkeypatch.setattr("tenmin.render.audio.run_with_progress", _touch_output)
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


# --- LLM 连续失败时把最后一次原始输出落盘 ---


class _SchemaBlowupProvider:
    """连续失败的 provider。raw_output 走 LLMSchemaError 送出来。"""

    def __init__(self, raw: str):
        self.raw = raw

    async def complete(self, system, user, schema=None):
        raise LLMSchemaError(
            f"FakeProvider 连续 3 次输出不符合 {schema.__name__}：val 不是 value",
            raw_output=self.raw,
        )


@pytest.mark.asyncio
async def test_run_script_dumps_the_raw_model_output_on_schema_failure(project):
    """这类失败最需要现场，而原实现只带 last_error 的前 1500 字符、原始输出全丢。"""
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])
    raw = '<think>我想想</think>\n{"script": {"val": 42}}'

    with pytest.raises(LLMSchemaError) as exc:
        await run_script(project, _SchemaBlowupProvider(raw), episode=2)

    dumped = Paths(project.root).script_raw(2)
    assert dumped.exists()
    assert dumped.read_text(encoding="utf-8") == raw
    # 报错里得指出文件在哪，否则落了盘也没人知道
    assert str(dumped) in str(exc.value)
    assert "val 不是 value" in str(exc.value)
    assert exc.value.raw_output == raw


@pytest.mark.asyncio
async def test_run_script_without_raw_output_does_not_write_an_empty_file(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])

    with pytest.raises(LLMSchemaError):
        await run_script(project, _SchemaBlowupProvider(""), episode=2)

    assert not Paths(project.root).script_raw(2).exists()


@pytest.mark.asyncio
async def test_run_script_success_leaves_no_raw_dump(project):
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)
    assert not Paths(project.root).script_raw(2).exists()


def _write_script(path: Path, script: Script) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script.model_dump_json(indent=2), encoding="utf-8")


def _touch_output(args: list[str], **_) -> str:
    """假的 ffmpeg：不跑编码，只把输出文件创建出来。

    刻意写非空字节：_is_fresh 现在把 0 字节产物当「被打断的半截产物」判成不新鲜，
    写 b"" 会让所有「产物已是最新所以跳过」的断言被这条兜底规则掩盖掉。
    """
    out = Path(args[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\x00")
    return ""


def _touch_output_with_progress(
    args: list[str], *, total_seconds: float, on_progress=None, **_
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
    # `chunk_003.<hash8>.mp3`：序号在前保留可读性，后面那段内容哈希管缓存身份。
    assert [chunk.path.split(".")[0] for chunk in track.chunks] == [
        "chunk_001",
        "chunk_002",
        "chunk_003",
    ]
    assert track.total_seconds == pytest.approx(30.0)
    assert paths.voice(2).exists()
    for chunk in track.chunks:
        assert (paths.voice_dir(2) / chunk.path).is_file()


@pytest.mark.asyncio
async def test_run_voice_reuses_recorded_durations_without_probing(project, monkeypatch):
    """复用路径原来每个 chunk 都要 spawn 一次 ffprobe，而时长早就写进 voice.json 了。"""
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    await run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2)

    def boom(path, **_):
        raise AssertionError("run_voice 该从 voice.json 里读时长，不该再 spawn ffprobe")

    monkeypatch.setattr("tenmin.render.tts.probe_duration", boom)
    engine = FakeTTSEngine([])
    track, _ = await run_voice(project, engine, episode=2)

    assert engine.calls == []
    assert [chunk.duration for chunk in track.chunks] == [8.0, 10.0, 10.0]


@pytest.mark.asyncio
async def test_run_voice_warns_when_the_old_voice_json_is_unreadable(project):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    paths.voice(2).parent.mkdir(parents=True, exist_ok=True)
    paths.voice(2).write_text("{ 这不是 json", encoding="utf-8")

    _, warnings = await run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2)

    assert any("voice.json" in w or "E02.voice.json" in w for w in warnings)


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

    def fake_probe(path, **_):
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

    def fake_run(args, **_):
        captured.append(list(args))
        return _touch_output(args)

    monkeypatch.setattr("tenmin.render.audio.run_with_progress", fake_run)

    out = run_audio(project, episode=2)

    assert out == paths.mixed_audio(2)
    assert out.exists()
    assert captured[0][:2] == ["-y", "-i"]
    # ffmpeg 写的是同目录的 .part，run_audio 返回前才原子改名到正式产物路径
    assert captured[0][-1] == str(part_path(paths.mixed_audio(2)))
    assert not part_path(paths.mixed_audio(2)).exists()
    assert "amix=inputs=2:normalize=0[mix]" in captured[0][captured[0].index("-filter_complex") + 1]


def test_run_audio_forwards_the_reporter(project, monkeypatch):
    """混音要几分钟，进度必须能上报出来 —— reporter 得真的传到 mix_audio。"""
    from .fakes import FakeReporter

    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)

    def fake_run(args, *, on_progress=None, **_):
        if on_progress is not None:
            on_progress(0.25)
        return _touch_output(args)

    monkeypatch.setattr("tenmin.render.audio.run_with_progress", fake_run)
    reporter = FakeReporter()

    run_audio(project, episode=2, reporter=reporter)

    assert ("substep", "audio", 25, 100, "") in reporter.calls


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

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
        captured.append(list(args))
        return _touch_output(args)

    monkeypatch.setattr("tenmin.render.video.run_with_progress", fake_run_with_progress)

    out = run_render(project, episode=2)

    assert out == paths.video(2)
    assert out.exists()
    assert "-movflags" in captured[0]
    # 同上：ffmpeg 写 .part，run_render 返回前才原子改名到 07_render/E02.mp4
    assert captured[0][-1] == str(part_path(paths.video(2)))
    assert not part_path(paths.video(2)).exists()


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
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    monkeypatch.setattr("tenmin.render.audio.run_with_progress", _touch_output)
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

    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    monkeypatch.setattr("tenmin.render.audio.run_with_progress", _touch_output)
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
async def test_run_pipeline_voice_without_tts_engine_raises_value_error(project):
    """原来这里是裸 assert：python -O 下被剥离，接着 None 一路漂进 synthesize_track，
    炸在 render/tts.py 里的 AttributeError，报错完全指不到「你忘了传 tts_engine」。

    选 ValueError 是因为 cli.py 的异常捕获列表里有它，会变成一行红字 + exit 1，
    而不是一整页 traceback。
    """
    _write_script(Paths(project.root).script(2), render_script())
    with pytest.raises(ValueError, match="tts_engine"):
        await run_pipeline(project, FakeProvider([]), only=["voice"], tts_engine=None)


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


def test_register_episode_copies_srt_and_appends_yaml_entry(tmp_path, golden_srt_path):
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

    # SRT 是几十 KB 的主要人工编辑面，照旧拷进项目目录。
    assert (root / "srt" / "E01.srt").exists()
    assert len(updated_cfg.episodes) == 2
    new_entry = next(e for e in updated_cfg.episodes if e.number == 1)
    assert new_entry.srt == Path("srt/E01.srt")
    assert new_entry.video == source_video.resolve()

    # reload from disk to confirm the yaml file itself was updated
    reloaded = load_project(yaml_path)
    assert len(reloaded.episodes) == 2
    assert any(e.number == 1 for e in reloaded.episodes)
    assert any(e.number == 2 for e in reloaded.episodes)


def test_register_episode_does_not_copy_the_video(tmp_path, golden_srt_path):
    """源片实测 300MB~1.4GB，整份拷进 work/ 等于磁盘占用翻倍 + 一次全量 I/O。

    config.video_path 本来就支持绝对路径，所以直接记源片位置。
    """
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: 才女的侍从\nslug: saijo\nepisodes: []\n", encoding="utf-8"
    )
    cfg = load_project(root / "project.yaml")

    source_srt = tmp_path / "incoming_E01.srt"
    source_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_video = tmp_path / "incoming_E01.mp4"
    source_video.write_bytes(b"fake video bytes")

    updated_cfg = register_episode(cfg, episode=1, srt=source_srt, video=source_video)

    assert not (root / "video" / "E01.mp4").exists()
    assert not (root / "video" / "E01.mkv").exists()
    # 记的路径必须真的能定位到源片
    assert updated_cfg.video_path(updated_cfg.episodes[0]) == source_video.resolve()


def test_register_episode_keeps_the_video_suffix(tmp_path, golden_srt_path):
    """传 .mkv 不能被改名成 .mp4。

    原实现无条件写 f"E{episode:02d}.mp4"，而 cli.PROJECT_TEMPLATE 的默认值恰恰是
    video/E02.mkv —— 自相矛盾，且 ffmpeg 会按容器实际内容而不是扩展名工作，所以
    这个错名一路不报错。
    """
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: 才女的侍从\nslug: saijo\nepisodes: []\n", encoding="utf-8"
    )
    cfg = load_project(root / "project.yaml")

    source_srt = tmp_path / "incoming_E01.srt"
    source_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_video = tmp_path / "incoming_E01.mkv"
    source_video.write_bytes(b"fake video bytes")

    register_episode(cfg, episode=1, srt=source_srt, video=source_video)

    recorded = yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8"))
    assert recorded["episodes"][0]["video"].endswith(".mkv")


def test_register_episode_leaves_existing_relative_video_paths_alone(
    tmp_path, golden_srt_path
):
    """向后兼容：work/saijo/ 下已经有 10 个拷好的 mp4，yaml 里记的是相对路径。

    注册新的一集会整份改写 episodes 列表，绝不能把存量集的相对路径改成别的形状
    ——那些文件真的在 work/saijo/video/ 下，改了就读不到了。
    """
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "video").mkdir(parents=True)
    (root / "video" / "E02.mp4").write_bytes(b"legacy copied video")
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nepisodes:\n- number: 2\n  srt: srt/E02.srt\n"
        "  video: video/E02.mp4\n",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)

    source_srt = tmp_path / "incoming_E01.srt"
    source_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_video = tmp_path / "incoming_E01.mkv"
    source_video.write_bytes(b"fake video bytes")

    register_episode(cfg, episode=1, srt=source_srt, video=source_video)

    reloaded = load_project(yaml_path)
    legacy = next(e for e in reloaded.episodes if e.number == 2)
    assert legacy.video == Path("video/E02.mp4")
    assert reloaded.video_path(legacy) == (root / "video" / "E02.mp4").resolve()
    assert reloaded.video_path(legacy).read_bytes() == b"legacy copied video"


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
    # 重新注册会把这一集指向新的源片，存量的 video/E02.mp4 不再被引用（但也不删）。
    assert updated_cfg.video_path(updated_cfg.episodes[0]) == source_video.resolve()
    assert (root / "srt" / "E02.srt").read_text(encoding="utf-8") == source_srt.read_text(
        encoding="utf-8"
    )


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
async def test_run_pipeline_reports_episode_start_exactly_once_per_episode(
    project, golden_srt_path, monkeypatch
):
    """批量模式下总进度条不许倒退。

    原实现有两个独立的 `for number in target_numbers` 循环（script+docgen 一个、
    voice..render 一个），各自调 episode_start，于是每集被报两次：总进度先 0→N
    再跳回 0→N。
    """
    second_srt = project.root / "srt" / "E01.srt"
    second_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    project.episodes.append(EpisodeConfig(number=1, srt=Path("srt/E01.srt")))
    for episode_cfg in project.episodes:
        video_name = f"E{episode_cfg.number:02d}.mkv"
        (project.root / video_name).write_bytes(b"\x00")
        episode_cfg.video = Path(video_name)

    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    monkeypatch.setattr("tenmin.render.audio.run_with_progress", _touch_output)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    reporter = FakeReporter()
    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    await run_pipeline(
        project,
        provider,
        tts_engine=FakeTTSEngine([8.0] * 20),
        reporter=reporter,
    )

    assert [c for c in reporter.calls if c[0] == "episode_start"] == [
        ("episode_start", 2, 1, 2),
        ("episode_start", 1, 2, 2),
    ]


@pytest.mark.asyncio
async def test_run_pipeline_batch_mode_reports_episode_done_for_each_episode(project):
    """episode_start 有始无终：没有 episode_done，总进度条永远差最后一格。"""
    project.episodes.append(EpisodeConfig(number=1, srt=project.episodes[0].srt))
    reporter = FakeReporter()
    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    calls = reporter.calls
    assert ("episode_done", 2, 1, 2) in calls
    assert ("episode_done", 1, 2, 2) in calls
    # 每集的 start/done 必须严格配对、按集包住这一集的阶段
    assert [c for c in calls if c[0] in ("episode_start", "episode_done")] == [
        ("episode_start", 2, 1, 2),
        ("episode_done", 2, 1, 2),
        ("episode_start", 1, 2, 2),
        ("episode_done", 1, 2, 2),
    ]


@pytest.mark.asyncio
async def test_run_pipeline_reports_episode_done_after_that_episodes_stages(project):
    project.episodes.append(EpisodeConfig(number=1, srt=project.episodes[0].srt))
    reporter = FakeReporter()
    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    calls = reporter.calls
    first_done = calls.index(("episode_done", 2, 1, 2))
    # 第一集的 docgen 必须在它自己的 episode_done 之前
    assert calls.index(("stage_done", "docgen")) < first_done
    # 而第二集的 episode_start 必须在第一集的 episode_done 之后
    assert first_done < calls.index(("episode_start", 1, 2, 2))


@pytest.mark.asyncio
async def test_run_pipeline_single_episode_mode_reports_neither_episode_hook(project):
    """单集模式没有「第几集/共几集」可言，episode_start 本来就不报，done 也不该报。"""
    reporter = FakeReporter()
    provider = FakeProvider([fake_script_response(episode=2)])
    await run_pipeline(project, provider, only=V1_STAGES, episode=2, reporter=reporter)
    assert [c for c in reporter.calls if c[0].startswith("episode_")] == []


@pytest.mark.asyncio
async def test_run_pipeline_preflights_all_episodes_before_any_tts(
    project, golden_srt_path, monkeypatch
):
    """preflight 的全部意义就是「绝不能跑完几分钟 TTS 才发现 ffmpeg 不行」。

    合并那两个循环时最容易顺手把 preflight 挪进循环体，于是第二集的视频缺失要等
    第一集渲完才炸。这条锁死：所有集的 preflight 都在第一次 TTS 之前。
    """
    second_srt = project.root / "srt" / "E01.srt"
    second_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    project.episodes.append(EpisodeConfig(number=1, srt=Path("srt/E01.srt")))
    for episode_cfg in project.episodes:
        video_name = f"E{episode_cfg.number:02d}.mkv"
        (project.root / video_name).write_bytes(b"\x00")
        episode_cfg.video = Path(video_name)

    events: list[str] = []

    class RecordingTTS(FakeTTSEngine):
        async def synthesize(self, text, out_path):
            events.append("tts")
            return await super().synthesize(text, out_path)

    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr(
        "tenmin.pipeline.preflight",
        lambda video, encoder, **_: (events.append(f"preflight:{Path(video).name}"), 1400.0)[1],
    )
    monkeypatch.setattr("tenmin.render.audio.run_with_progress", _touch_output)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    await run_pipeline(project, provider, tts_engine=RecordingTTS([8.0] * 20))

    preflights = [i for i, e in enumerate(events) if e.startswith("preflight:")]
    first_tts = events.index("tts")
    assert len(preflights) == 2, events
    assert max(preflights) < first_tts, events


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

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, **_):
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


def test_run_timeline_uses_configured_ffprobe_path(project, monkeypatch):
    """render.ffprobe_path / ffmpeg_path 必须从 project.yaml 一路传到 subprocess。"""
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _prepare_video(project)
    project.render.ffprobe_path = "/opt/x/ffprobe"
    seen: dict[str, str] = {}

    def fake_probe(path, *, ffprobe="ffprobe"):
        seen["ffprobe"] = ffprobe
        return 1400.0

    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)
    run_timeline(project, episode=2)
    assert seen["ffprobe"] == "/opt/x/ffprobe"


def test_run_audio_uses_configured_ffmpeg_path(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    project.render.ffmpeg_path = "/opt/x/ffmpeg"
    seen: dict[str, str] = {}

    def fake_run(args, *, ffmpeg="ffmpeg", **_):
        seen["ffmpeg"] = ffmpeg
        return _touch_output(args)

    monkeypatch.setattr("tenmin.render.audio.run_with_progress", fake_run)
    run_audio(project, episode=2)
    assert seen["ffmpeg"] == "/opt/x/ffmpeg"


def test_run_render_uses_configured_ffmpeg_path(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    paths.mixed_audio(2).parent.mkdir(parents=True, exist_ok=True)
    paths.mixed_audio(2).write_bytes(b"\x00")
    project.render.ffmpeg_path = "/opt/x/ffmpeg"
    seen: dict[str, str] = {}

    def fake_run_with_progress(args, *, total_seconds, on_progress=None, ffmpeg="ffmpeg"):
        seen["ffmpeg"] = ffmpeg
        return _touch_output(args)

    monkeypatch.setattr("tenmin.render.video.run_with_progress", fake_run_with_progress)
    run_render(project, episode=2)
    assert seen["ffmpeg"] == "/opt/x/ffmpeg"


def test_preflight_receives_configured_binaries(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    project.render.ffmpeg_path = "/opt/x/ffmpeg"
    project.render.ffprobe_path = "/opt/x/ffprobe"
    seen: dict[str, str] = {}

    def fake_preflight(video, encoder, *, ffmpeg="ffmpeg", ffprobe="ffprobe", **_):
        seen["ffmpeg"] = ffmpeg
        seen["ffprobe"] = ffprobe
        return 1400.0

    monkeypatch.setattr("tenmin.pipeline.preflight", fake_preflight)
    monkeypatch.setattr("tenmin.render.audio.run_with_progress", _touch_output)
    asyncio.run(
        run_pipeline(
            project,
            FakeProvider(render_script()),
            only=["audio"],
            episode=2,
            tts_engine=FakeTTSEngine([8.0, 10.0, 10.0]),
        )
    )
    assert seen == {"ffmpeg": "/opt/x/ffmpeg", "ffprobe": "/opt/x/ffprobe"}


async def _run_audio_only(project, monkeypatch):
    """跑到 audio 阶段（会触发 preflight），其余都用假的。"""
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    await run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2)
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    monkeypatch.setattr("tenmin.render.audio.run_with_progress", _touch_output)
    return await run_pipeline(
        project,
        FakeProvider(render_script()),
        only=["audio"],
        episode=2,
        tts_engine=FakeTTSEngine([8.0, 10.0, 10.0]),
    )


def test_preflight_gets_drawtext_requirement_from_outro_card_seconds(project, monkeypatch):
    """drawtext 只在片尾卡开着时才用到，preflight 的要求必须跟着配置走。"""
    seen: list[bool] = []

    def fake_preflight(video, encoder, *, needs_drawtext=False, **_):
        seen.append(needs_drawtext)
        return 1400.0

    monkeypatch.setattr("tenmin.pipeline.preflight", fake_preflight)
    project.render.outro_card_seconds = 3.0
    asyncio.run(_run_audio_only(project, monkeypatch))
    assert seen == [True]

    seen.clear()
    project.render.outro_card_seconds = 0.0
    asyncio.run(_run_audio_only(project, monkeypatch))
    assert seen == [False]


def test_preflight_gets_both_configured_font_names(project, monkeypatch):
    """字幕字体与片尾卡字体是两个独立旋钮，两个都要查。"""
    seen: dict[str, object] = {}

    def fake_preflight(video, encoder, *, font_names=(), **_):
        seen["fonts"] = list(font_names)
        return 1400.0

    monkeypatch.setattr("tenmin.pipeline.preflight", fake_preflight)
    project.render.subtitle_font_name = "Lantinghei SC"
    project.render.outro_font_name = "Hiragino Sans"
    project.render.outro_card_seconds = 3.0
    asyncio.run(_run_audio_only(project, monkeypatch))
    assert set(seen["fonts"]) == {"Lantinghei SC", "Hiragino Sans"}


def test_preflight_skips_the_outro_font_when_there_is_no_outro_card(project, monkeypatch):
    seen: dict[str, object] = {}

    def fake_preflight(video, encoder, *, font_names=(), **_):
        seen["fonts"] = list(font_names)
        return 1400.0

    monkeypatch.setattr("tenmin.pipeline.preflight", fake_preflight)
    project.render.subtitle_font_name = "Lantinghei SC"
    project.render.outro_font_name = "Hiragino Sans"
    project.render.outro_card_seconds = 0.0
    asyncio.run(_run_audio_only(project, monkeypatch))
    assert seen["fonts"] == ["Lantinghei SC"]


def test_preflight_warnings_reach_the_user(project, monkeypatch):
    """字体缺失是 warning，必须真的冒到 run_pipeline 的 warnings 里去。

    本项目刻意不引入 logging，warnings 通道是这类诊断唯一的出口。
    """

    def fake_preflight(video, encoder, *, warnings=None, **_):
        if warnings is not None:
            warnings.append("字体 'Lantinghei SC' 没找到")
        return 1400.0

    monkeypatch.setattr("tenmin.pipeline.preflight", fake_preflight)
    result = asyncio.run(_run_audio_only(project, monkeypatch))
    assert any("Lantinghei SC" in w for w in result)


def _project_config(tmp_path: Path) -> ProjectConfig:
    """一个已经落好 project.yaml 的最小项目，给原子写用例当底座。"""
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nmode: single_episode\n"
        "target_seconds: 240\nepisodes:\n- number: 2\n  srt: srt/E02.srt\n",
        encoding="utf-8",
    )
    return load_project(yaml_path)


# --- 产物原子写（P1-G 第 1 项）---------------------------------------------
# _is_fresh 只比 mtime，所以每一个「会被当成输入或产物」的文件都必须原子落盘，
# 否则半截文件的 mtime 恰好最新，下一轮直接跳过、坏产物一路进成片。


def test_write_text_never_leaves_a_partial_file_at_the_target(tmp_path, monkeypatch):
    """写文本的中途炸掉时，目标路径上必须还是旧内容（或干脆不存在）。"""
    from tenmin import pipeline as pipeline_module
    from tenmin.atomic import part_path

    target = tmp_path / "out" / "E02.narration.txt"
    target.parent.mkdir(parents=True)
    target.write_text("上一轮的完整旁白", encoding="utf-8")

    real_replace = Path.replace

    def boom(self, other):
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        pipeline_module._write_text(target, "半截")
    monkeypatch.setattr(Path, "replace", real_replace)

    assert target.read_text(encoding="utf-8") == "上一轮的完整旁白"
    assert not part_path(target).exists()


def test_write_json_goes_through_the_atomic_helper(tmp_path, monkeypatch):
    from tenmin import pipeline as pipeline_module

    seen: list[Path] = []

    def spy(path, text, **kwargs):
        seen.append(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")

    monkeypatch.setattr(pipeline_module.atomic, "write_text", spy)
    target = tmp_path / "05_timeline" / "E02.timeline.json"
    pipeline_module._write_json(target, "{}")
    assert seen == [target]
    assert target.read_text(encoding="utf-8") == "{}"


def test_register_episode_copies_the_srt_atomically(tmp_path, monkeypatch):
    """拷进来的 SRT 是 ingest 的输入。半截字幕会静默产出缺对白的对白轨。"""
    from tenmin import atomic as atomic_module
    from tenmin.atomic import part_path

    cfg = _project_config(tmp_path)
    srt = tmp_path / "source.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n台词\n", encoding="utf-8")
    dest = cfg.root / "srt" / "E03.srt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("上一轮的完整字幕", encoding="utf-8")

    def boom(src, target):
        Path(target).write_text("半截", encoding="utf-8")
        raise OSError("盘满了")

    # 打在 tenmin.atomic 上而不是 tenmin.pipeline 上：拷贝现在由 atomic.copy_file 做，
    # pipeline 自己已经不再直接 import shutil。
    monkeypatch.setattr(atomic_module.shutil, "copyfile", boom)
    with pytest.raises(OSError):
        register_episode(cfg, episode=3, srt=srt, video=tmp_path / "E03.mkv")
    assert dest.read_text(encoding="utf-8") == "上一轮的完整字幕"
    assert not part_path(dest).exists()


def test_register_episode_rewrites_project_yaml_atomically(tmp_path, monkeypatch):
    """project.yaml 是**每个阶段**的输入。改写它被打断不能把已注册的集数全毁掉。"""
    from tenmin.atomic import part_path

    cfg = _project_config(tmp_path)
    before = cfg.config_path.read_text(encoding="utf-8")
    srt = tmp_path / "source.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n台词\n", encoding="utf-8")

    real_replace = Path.replace
    calls: list[Path] = []

    def boom(self, other):
        if Path(other) == cfg.config_path:
            raise KeyboardInterrupt
        calls.append(Path(other))
        return real_replace(self, other)

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        register_episode(cfg, episode=3, srt=srt, video=tmp_path / "E03.mkv")
    monkeypatch.setattr(Path, "replace", real_replace)

    assert cfg.config_path.read_text(encoding="utf-8") == before
    assert not part_path(cfg.config_path).exists()
