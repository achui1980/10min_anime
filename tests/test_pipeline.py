import asyncio
import difflib
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from tenmin import config_slices
from tenmin.atomic import part_path
from tenmin.config import EpisodeConfig, ProjectConfig, ProjectConfigError, load_project
from tenmin.ingest.resolve import SubtitleSource
from tenmin.models import (
    AudioDirection,
    Beat,
    Clip,
    DialogueLine,
    DialogueTrack,
    Hold,
    LLMBeat,
    LLMClip,
    LLMScript,
    Script,
    TranslatedTrack,
)
from tenmin.pipeline import (
    _ARTIFACTS,
    STAGES,
    Paths,
    _find_episode,
    _ingest_inputs,
    _is_fresh,
    _is_fresh_stamped,
    _script_inputs,
    _timeline_inputs,
    _translate_inputs,
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
    run_translate,
    run_voice,
    stages_from,
)
from tenmin.progress import NullProgressReporter
from tenmin.render.ffmpeg import FFmpegError
from tenmin.script.llm import LLMSchemaError
from tenmin.script.validate import ScriptValidationError

from .fakes import (
    EpisodeAwareProvider,
    FakeProvider,
    FakeReporter,
    FakeTTSEngine,
)

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
    # 5 个节点、其中一个 role=climax：提示词「节点结构」一节要求的最小合规形状
    # （5–8 个节点、恰好 1 个 climax）。原来是 3 个节点、零 climax，validate 的结构
    # 检查补上下限之后会照实报两条 warning —— 这份 fixture 表达的是「一次健康的
    # 批处理」，该修的是 fixture 不是判据。
    labels = [
        "Hook 开场",
        "阶段一：入职即地狱",
        "阶段二：修罗场",
        "阶段三：反击",
        "收尾：修罗场引爆",
    ]
    roles = ["hook", "act", "climax", "act", "outro"]
    return LLMScript(
        beats=[
            LLMBeat(
                id=f"b{i + 1}",
                label=labels[i],
                role=roles[i],
                # 216 字 = 48 秒旁白（4.5 字/秒）。5 个节点合计仍是 240 秒，跟原来
                # 3 × 360 字完全相同，时长预算与返工轮的行为一字不变。
                # **带句读**：12 句 ×（17 字 + 「。」）。光秃秃的一个长单句会让字幕
                # 折成十几行 —— 字幕可读性检查照实报 warning。
                narration=("啊" * 17 + "。") * 12,
                clips=[
                    LLMClip(
                        episode=episode,
                        start=300.0 + i * 100,
                        # 画面跟旁白 1:1 给足 48 秒。给少了 validate 的 A3
                        # 「画面/旁白预算」会报拉伸倍率 warning。
                        end=300.0 + i * 100 + 48.0,
                        visual="画面 ➔ 特写",
                    )
                ],
            )
            for i in range(5)
        ]
    )


def test_stage_names():
    assert STAGES == [
        "ingest",
        "translate",
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
    跑直接 stage_skip，坏产物一路进成片。真正的解法是产物原子写（tenmin.atomic），这里
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
    读不到 SRT）或产出垃圾，而跳过至少保住了磁盘上已有的产物。加了配置切片
    进 inputs 之后，真实项目里这个分支已经走不到（切片开跑前就已落盘）。
    """
    out = _file(tmp_path / "out.txt")
    assert _is_fresh([out], [tmp_path / "never.txt"]) is True


def test_timeline_freshness_tracks_cached_voice_audio_files(tmp_path):
    from tenmin.models import VoiceChunk, VoiceTrack

    paths = Paths(tmp_path)
    audio = paths.voice_dir(2) / "chunk_001.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"voice")
    track = VoiceTrack(episode=2, chunks=[
        VoiceChunk(beat_id="b1", index=1, text="长句，继续说。", path=audio.name, duration=3.0)
    ])
    for path in (paths.script(2), paths.voice(2)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input", encoding="utf-8")
    timeline = paths.timeline(2)
    timeline.parent.mkdir(parents=True, exist_ok=True)
    timeline.write_text("output", encoding="utf-8")
    _shift_mtime(timeline, 10.0)
    assert _is_fresh([timeline], _timeline_inputs(paths, 2, track, []))
    _shift_mtime(audio, 20.0)
    assert not _is_fresh([timeline], _timeline_inputs(paths, 2, track, []))


@pytest.mark.asyncio
@pytest.mark.parametrize("stages", ["only", "from_timeline"])
async def test_missing_voice_chunk_cannot_skip_fresh_timeline(project, monkeypatch, stages):
    from tenmin.models import VoiceTrack

    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    await run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2)
    _write_dialogue_for_render_script(paths)
    run_timeline(project, episode=2, source_duration=1400.0)
    track = json.loads(paths.voice(2).read_text(encoding="utf-8"))
    missing = paths.voice_dir(2) / track["chunks"][0]["path"]
    missing.unlink()

    # 通用 _is_fresh 刻意忽略缺失输入；只有 timeline 对 voice.json 引用的 MP3 加硬闸。
    voice = VoiceTrack.model_validate_json(paths.voice(2).read_text(encoding="utf-8"))
    assert _is_fresh(
        [paths.timeline(2), paths.subtitles(2)],
        _timeline_inputs(paths, 2, voice, []),
    )
    old_timeline = paths.timeline(2).read_bytes()
    old_ass = paths.subtitles(2).read_bytes()
    provider = FakeProvider([])
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda *_args, **_kwargs: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda *_args, **_kwargs: 25.0)
    if stages == "from_timeline":
        monkeypatch.setattr("tenmin.pipeline.preflight", lambda *_args, **_kwargs: 1400.0)
    options = {"only": ["timeline"]} if stages == "only" else {"from_stage": "timeline"}

    with pytest.raises(FileNotFoundError) as exc:
        await run_pipeline(project, provider, episode=2, **options)
    assert str(missing) in str(exc.value)
    assert "voice" in str(exc.value)
    assert provider.calls == []
    assert paths.timeline(2).read_bytes() == old_timeline
    assert paths.subtitles(2).read_bytes() == old_ass


# work/ 下有 11 集的存量产物，产物路径改一个字符就等于全部存量产物失效（流水线会
# 认为什么都没跑过，重新调 LLM、重新 TTS、重新渲染）。所以这里把每一条路径按字面量
# 锁死：Paths 的实现怎么重构都行，拼出来的字符串必须逐字节不变。
FROZEN_LAYOUT = {
    "dialogue": "01_dialogue/E02.dialogue.json",
    "signals": "02_signals/E02.signals.json",
    "script": "03_script/E02.script.json",
    "script_raw": "03_script/E02.raw.txt",
    "script_rejected": "03_script/E02.rejected.json",
    "script_warnings": "03_script/E02.warnings.json",
    "script_usage": "03_script/E02.usage.json",
    "table": "out/E02.解说方案.md",
    "narration": "out/E02.narration.txt",
    "zh_lines": "zh/E02.zh.json",
    "zh_subtitles": "out/E02.zh.srt",
    "zh_usage": "zh/E02.usage.json",
    "voice_dir": "04_voice/E02",
    "voice": "04_voice/E02.voice.json",
    "timeline": "05_timeline/E02.timeline.json",
    "subtitles": "05_timeline/E02.ass",
    "mixed_audio": "06_audio/E02.mixed.m4a",
    "video": "07_render/E02.mp4",
    "asr_cache": "srt/E02.asr.srt",
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
    """新增一种产物就必须同时进 FROZEN_LAYOUT，否则它的路径没人锁。

    两个方向各一条断言，缺任何一条都留个洞：
    - 方法 ↔ FROZEN_LAYOUT：加了 Paths 方法却没进 FROZEN_LAYOUT，路径没人钉。
    - _ARTIFACTS ↔ FROZEN_LAYOUT：往表里加一行却不加方法（实测这个变异此前**存活**）。
      那样的条目只是无害的死数据，但 Paths 的 docstring 声称「表与方法一一对应」
      由本条测试锁住，所以那句话得真的成立。
    """
    assert _artifact_methods() == set(FROZEN_LAYOUT)
    assert set(_ARTIFACTS) == set(FROZEN_LAYOUT)


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


def test_glossary_path_is_project_level(tmp_path):
    """累积术语表不带集号：它的意义就是跨集共享。

    做成 property 而不是方法是刻意的 —— test_paths_exposes_exactly_the_frozen_artifacts
    断言 Paths 上可调用的公开名字集合正好等于按集产物那张表，property 不是 callable，
    自动落在那个集合之外。
    """
    paths = Paths(tmp_path)
    assert paths.glossary == tmp_path / "zh" / "glossary.json"


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


def test_run_ingest_still_degrades_on_a_plain_os_error(project, monkeypatch):
    """探测这个文件时撞上 OSError（换盘时的竞态、权限）仍然降级。

    这条原来叫 `..._when_ffprobe_is_not_installed`，用 `FileNotFoundError("ffprobe")`
    模拟「ffprobe 不在 PATH 上」。那个断言锁死的正是 M3 要修的 bug：二进制配错跟
    「这个文件探不出时长」被混成同一类，于是 `project.yaml` 里 ffprobe_path 打错一个
    字符，ingest 会完全静默地退化到「字幕末尾当片长」。现在二进制配错走
    FFmpegBinaryError（响亮报错，见下一条），这条只保留它原本合理的那一半语义。
    """
    _prepare_video(project)

    def boom(path, **_):
        raise OSError("Input/output error")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", boom)
    assert run_ingest(project)[0].duration == pytest.approx(1416.6, abs=1.0)


def test_run_ingest_refuses_to_degrade_on_a_misconfigured_ffprobe(project, monkeypatch):
    """`render.ffprobe_path` 配错必须**响亮**报错，绝不能静默退化到字幕末尾。

    降级本身是刻意的（只有 SRT 也能跑 ingest），但「配置写错」跟「探测失败」是两件
    不同的事：前者静默降级的后果是整个 ED 窗被系统性挪动（实测真实片长比字幕末尾长
    1.8~24.3 秒），而用户既没有报错也没有 warning。
    """
    from tenmin.render.ffmpeg import FFmpegBinaryError

    _prepare_video(project)

    def boom(path, **_):
        raise FFmpegBinaryError("找不到可执行文件 '/opt/typo/ffprobe'")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", boom)
    with pytest.raises(FFmpegBinaryError):
        run_ingest(project)


def test_misconfigured_ffprobe_reaches_ingest_for_real(project):
    """不打桩，走真实的 `probe_duration`：ffprobe_path 配错时 ingest 必须抛。"""
    from tenmin.render.ffmpeg import FFmpegBinaryError

    _prepare_video(project)
    project.render.ffprobe_path = "/definitely/not/a/real/ffprobe"
    with pytest.raises(FFmpegBinaryError):
        run_ingest(project)


def test_run_signals_writes_signals_json(project):
    run_ingest(project)
    reports = run_signals(project)
    # 实测：本集有 26 个 ≥3s 的无字幕间隙（候选池，不是高光集合）。
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
async def test_run_pipeline_reruns_the_stage_whose_config_changed(project):
    """改了 validate_script 只有读它的 script 重跑（docgen 因为 script.json 变了跟着
    重跑）；ingest / signals 一个字都没读它，必须跳过。"""
    provider = FakeProvider([fake_script_response(), fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)

    project.validate_script.max_beats = 7
    reporter = FakeReporter()
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    assert ("stage_skip", "ingest") in reporter.calls
    assert ("stage_skip", "signals") in reporter.calls
    assert ("stage_start", "script") in reporter.calls
    assert ("stage_start", "docgen") in reporter.calls
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_run_pipeline_reruns_ingest_when_its_config_changes(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest"])

    project.credits.op_span_min = 30.0
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=["ingest"], reporter=reporter)

    assert ("stage_start", "ingest") in reporter.calls


@pytest.mark.asyncio
async def test_touching_project_yaml_no_longer_reruns_anything(project):
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)

    _shift_mtime(project.config_path, 10.0)
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=V1_STAGES, reporter=reporter)

    for stage in V1_STAGES:
        assert ("stage_skip", stage) in reporter.calls, stage


@pytest.mark.asyncio
async def test_an_operational_knob_reruns_nothing(project):
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)

    project.llm.timeout_seconds = 999.0
    project.llm.script_concurrency = 3
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=V1_STAGES, reporter=reporter)

    for stage in V1_STAGES:
        assert ("stage_skip", stage) in reporter.calls, stage


@pytest.mark.asyncio
async def test_comment_and_default_value_edits_in_project_yaml_rerun_nothing(
    tmp_path, golden_srt_path
):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "E02.srt").write_bytes(golden_srt_path.read_bytes())
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\ntarget_seconds: 240\n"
        "episodes:\n- number: 2\n  srt: srt/E02.srt\n",
        encoding="utf-8",
    )
    await run_pipeline(
        load_project(yaml_path), FakeProvider([fake_script_response()]), only=V1_STAGES
    )

    edited = "# 加一行注释\n" + yaml_path.read_text(encoding="utf-8").replace(
        "target_seconds: 240", "target_seconds: 240.0  # 显式写一遍\nmode: single_episode"
    )
    yaml_path.write_text(edited, encoding="utf-8")
    _shift_mtime(yaml_path, 10.0)
    reporter = FakeReporter()
    await run_pipeline(
        load_project(yaml_path), FakeProvider([]), only=V1_STAGES, reporter=reporter
    )

    for stage in V1_STAGES:
        assert ("stage_skip", stage) in reporter.calls, stage


@pytest.mark.asyncio
async def test_run_pipeline_reruns_the_render_stages_when_the_voice_changes(
    project, monkeypatch
):
    """voice 之后的阶段吃的是各自的切片与上游产物：换配音音色 → voice 重跑 →
    voice.json 刷新 → timeline / audio / render 跟着重跑。"""
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    # voice 重跑时 synthesize_track 会复用上一轮落盘的 chunk 并用 ffprobe 量时长，
    # 而 FakeTTSEngine 写的是假 mp3 字节。
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 8.0)
    _stub_mix_audio_ffmpeg(monkeypatch)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)
    _write_dialogue_for_render_script(paths)

    stages = ["voice", "timeline", "audio", "render"]
    await run_pipeline(
        project, FakeProvider([]), only=stages, tts_engine=FakeTTSEngine([8.0] * 3)
    )

    project.render.voice = "zh-CN-XiaoxiaoNeural"
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


@pytest.mark.asyncio
async def test_replacing_video_reruns_only_video_dependent_stages(project, monkeypatch):
    video = _prepare_video(project)
    video.write_bytes(b"first source")
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    _stub_mix_audio_ffmpeg(monkeypatch)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    await run_pipeline(
        project,
        FakeProvider([fake_script_response()]),
        tts_engine=FakeTTSEngine([80.0] * 20),
    )
    paths = Paths(project.root)
    before = {
        stage: getattr(paths, stage)(2).stat().st_mtime_ns
        for stage in ("script", "voice", "timeline", "mixed_audio", "video")
    }
    video.write_bytes(b"replacement source")
    os.utime(video, ns=(max(before.values()) + 2_000_000_000,) * 2)

    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), tts_engine=FakeTTSEngine([]), reporter=reporter)

    for stage in ("script", "docgen", "voice"):
        assert ("stage_skip", stage) in reporter.calls
    for stage in ("timeline", "audio", "render"):
        assert ("stage_start", stage) in reporter.calls
    assert paths.script(2).stat().st_mtime_ns == before["script"]
    assert paths.voice(2).stat().st_mtime_ns == before["voice"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage", ["timeline", "audio", "render"], ids=["timeline", "audio", "movie"]
)
async def test_video_replacement_invalidates_each_stage_run_alone(project, monkeypatch, stage):
    video = _prepare_video(project)
    video.write_bytes(b"first source")
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    await run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2)
    _write_dialogue_for_render_script(paths)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    _stub_mix_audio_ffmpeg(monkeypatch)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)
    await run_pipeline(project, FakeProvider([]), only=["timeline", "audio", "render"])

    video.write_bytes(b"replacement source")
    last_output = max(
        paths.timeline(2).stat().st_mtime_ns,
        paths.mixed_audio(2).stat().st_mtime_ns,
        paths.video(2).stat().st_mtime_ns,
    )
    os.utime(video, ns=(last_output + 2_000_000_000,) * 2)
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=[stage], reporter=reporter)

    assert ("stage_start", stage) in reporter.calls


@pytest.mark.asyncio
async def test_srt_only_episode_still_runs_v1_stages_with_a_prefilled_neighbor(project):
    project.episodes.append(EpisodeConfig(number=3, op_range=(10.0, 100.0)))
    reporter = FakeReporter()

    warnings = await run_pipeline(
        project, FakeProvider([fake_script_response()]), only=V1_STAGES, reporter=reporter
    )

    assert Paths(project.root).script(2).exists()
    assert not Paths(project.root).script(3).exists()
    assert "第 3 集还没有 video，已跳过" in warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["audio", "render"], ids=["audio", "movie"])
async def test_missing_video_still_fails_preflight_for_media_stages(project, stage):
    project.episodes[0].video = Path("missing.mkv")

    with pytest.raises(FileNotFoundError, match=r"missing\.mkv"):
        await run_pipeline(project, FakeProvider([]), only=[stage])


@pytest.mark.asyncio
async def test_timeline_only_rejects_missing_source_after_successful_run(project, monkeypatch):
    video = _prepare_video(project)
    video.write_bytes(b"source")
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    await run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2)
    _write_dialogue_for_render_script(paths)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    await run_pipeline(project, FakeProvider([]), only=["timeline"])
    assert paths.timeline(2).exists()
    video.unlink()

    reporter = FakeReporter()
    with pytest.raises(FileNotFoundError, match=r"找不到源视频.*E02\.mkv.*episodes\[\]\.video"):
        await run_pipeline(project, FakeProvider([]), only=["timeline"], reporter=reporter)
    assert ("stage_skip", "timeline") not in reporter.calls


@pytest.mark.asyncio
async def test_timeline_only_rejects_unreadable_source_instead_of_skipping(project, monkeypatch):
    video = _prepare_video(project)
    video.write_bytes(b"source")
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    await run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2)
    _write_dialogue_for_render_script(paths)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    await run_pipeline(project, FakeProvider([]), only=["timeline"])
    assert paths.timeline(2).exists()

    def unreadable(path, **_):
        raise FFmpegError(f"无法读取源视频 {path}")

    monkeypatch.setattr("tenmin.pipeline.probe_duration", unreadable)
    reporter = FakeReporter()
    with pytest.raises(FFmpegError, match=r"无法读取源视频.*E02\.mkv"):
        await run_pipeline(project, FakeProvider([]), only=["timeline"], reporter=reporter)
    assert ("stage_skip", "timeline") not in reporter.calls


@pytest.mark.asyncio
async def test_missing_timeline_source_fails_before_script_api_call(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])
    project.episodes[0].video = Path("missing.mkv")
    provider = FakeProvider([])

    with pytest.raises(FileNotFoundError, match=r"missing\.mkv"):
        await run_pipeline(project, provider, only=["script", "timeline"])
    assert provider.calls == []


@pytest.mark.asyncio
async def test_run_pipeline_backdates_freshly_created_slices(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest"])

    slices = project.root / ".config"
    assert (slices / "signals.json").stat().st_mtime_ns == 0
    assert (slices / "E02.ingest.json").stat().st_mtime_ns == 0
    assert (slices / "E02.script.json").stat().st_mtime_ns == 0


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


# mix_audio 现在会真的调用 `run()` 做两遍响度体检（外加留白窗单独增益时的
# ebur128 测量）与编码后的 `probe_duration` 时长体检——这些全流水线测试只关心
# 「阶段接线对不对」，不该被这几道新守卫绊倒，所以统一假掉。按 filter graph 里
# 有没有 "loudnorm" 分支返回对应格式：两遍响度归一走 loudnorm 的 JSON 报告，
# 留白窗单独增益走 ebur128 的 `I: ... LUFS` 文本行。
#
# `input_i` 刻意等于 DEFAULT_RENDER.loudness_i（-14.0）：run_audio 会把
# mix_audio 的 warnings 原样透传进 run_pipeline 的聚合列表，这份假 stderr 同一份
# 文本会被当成「编码后体检」的实测结果去跟 cfg.loudness_i 比——差值超过 1 LU
# 就会真的产生一条「偏离目标」的 warning。这些全流水线测试要的是「一次健康的
# 批处理」，该让假素材看起来已经打到目标，不该往 `assert warnings == []` 里加
# 一条跟阶段接线无关的响度噪音。
_FAKE_LOUDNORM_STDERR = (
    "[Parsed_loudnorm_0 @ 0x0]\n"
    "{\n"
    '"input_i" : "-14.00",\n'
    '"input_tp" : "-3.00",\n'
    '"input_lra" : "5.00",\n'
    '"input_thresh" : "-30.00",\n'
    '"output_i" : "-14.00",\n'
    '"output_tp" : "-1.50",\n'
    '"output_lra" : "5.00",\n'
    '"output_thresh" : "-24.00",\n'
    '"normalization_type" : "dynamic",\n'
    '"target_offset" : "0.10"\n'
    "}\n"
)


def _fake_ffmpeg_meter(args: list[str], **_) -> str:
    joined = " ".join(args)
    if "loudnorm" in joined:
        return _FAKE_LOUDNORM_STDERR
    return "I: -23.0 LUFS\n"


def _stub_mix_audio_ffmpeg(
    monkeypatch,
    *,
    on_encode: Callable[[list[str]], None] | None = None,
    progress_fraction: float = 1.0,
    ffmpeg_seen: dict[str, str] | None = None,
) -> None:
    """把 `mix_audio` 接触到的三类 ffmpeg 调用（响度测量两遍/单独增益测量 + 真实
    编码）都换成假的。`probe_duration`（编码后时长体检）回落到这次编码实际声明的
    `total_seconds`——从 `run_with_progress` 的调用里现场捞，不需要每个测试各自
    算一遍期望时长。
    """
    state: dict[str, float] = {}

    def fake_run_with_progress(
        args, *, total_seconds, on_progress=None, ffmpeg="ffmpeg", **_
    ):
        state["total_seconds"] = total_seconds
        if ffmpeg_seen is not None:
            ffmpeg_seen["ffmpeg"] = ffmpeg
        if on_progress is not None:
            on_progress(progress_fraction)
        if on_encode is not None:
            on_encode(list(args))
        return _touch_output(args)

    def fake_probe_duration(path, **_):
        return state.get("total_seconds", 0.0)

    monkeypatch.setattr("tenmin.render.audio.run_with_progress", fake_run_with_progress)
    monkeypatch.setattr("tenmin.render.audio.run", _fake_ffmpeg_meter)
    monkeypatch.setattr("tenmin.render.audio.probe_duration", fake_probe_duration)


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


def _dialogue_for_render_script() -> DialogueTrack:
    """配 render_script() 用的最小对白轨：定位它唯一的 hold「第一句」。

    render_script() 的 b1 用 FakeTTSEngine([8.0, 10.0, 10.0]) 时第一个 chunk 的播放
    窗口是 [0, 8]，hold_after=2 落在时间轴 [8, 10]；clip 是 (100, 140)，所以对应源片
    位置是 [108, 110]。这里给一条落在这段区间内、kind 默认 dialogue 的行，让
    build_timeline 的留白核验能找到唯一出处、不撤销这个 hold —— 否则所有沿用
    render_script() 的既有测试断言的 `warnings == []` 会被一条撤销提示打破。
    """
    return DialogueTrack(
        episode=2,
        duration=1400.0,
        lines=[
            DialogueLine(idx=1, start=108.0, end=109.0, text="第一句", raw="第一句"),
        ],
    )


def _write_dialogue_for_render_script(paths: Paths, episode: int = 2) -> None:
    _file(paths.dialogue(episode), _dialogue_for_render_script().model_dump_json())


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
    _write_dialogue_for_render_script(paths)

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


def test_run_timeline_requires_current_dialogue(project):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    with pytest.raises(FileNotFoundError, match="对白轨"):
        run_timeline(project, episode=2, source_duration=1400.0)


def test_run_timeline_reads_current_dialogue_and_writes_compat_windows(project):
    """空对白轨定位不到 hold 出处：留白被撤销，但 timeline 仍要正常落盘。"""
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _file(paths.dialogue(2), DialogueTrack(episode=2, duration=1400.0).model_dump_json())

    timeline, _ = run_timeline(project, episode=2, source_duration=1400.0)

    assert "hold_windows" in paths.timeline(2).read_text(encoding="utf-8")
    assert timeline.hold_windows == []


def test_run_timeline_rejects_dialogue_from_another_episode(project):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _file(paths.dialogue(2), DialogueTrack(episode=3, duration=1400.0).model_dump_json())

    with pytest.raises(ValueError, match="集号"):
        run_timeline(project, episode=2, source_duration=1400.0)

    assert not paths.timeline(2).exists()


def test_run_timeline_probes_source_when_duration_missing(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _write_dialogue_for_render_script(paths)
    video = _prepare_video(project)
    calls: list[Path] = []

    def fake_probe(path, **_):
        calls.append(Path(path))
        return 1400.0

    def fake_frame_rate(path, **_):
        calls.append(Path(path))
        return 25.0

    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", fake_frame_rate)

    timeline, warnings = run_timeline(project, episode=2)

    # 片长与帧率是同一次源片探测的两半，都必须落在**同一个**文件上
    assert calls == [video, video]
    assert timeline.frame_rate == pytest.approx(25.0)
    assert warnings == []
    assert timeline.total_seconds == pytest.approx(30.0)


def test_run_timeline_uses_config_font_size(project):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _write_dialogue_for_render_script(paths)
    project.render.font_size = 72

    run_timeline(project, episode=2, source_duration=1400.0)

    ass = Paths(project.root).subtitles(2).read_text(encoding="utf-8")
    assert "Lantinghei SC,72," in ass


def test_run_audio_invokes_ffmpeg(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _write_dialogue_for_render_script(paths)
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    captured: list[list[str]] = []

    _stub_mix_audio_ffmpeg(monkeypatch, on_encode=captured.append)

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
    _write_dialogue_for_render_script(paths)
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)

    _stub_mix_audio_ffmpeg(monkeypatch, progress_fraction=0.25)
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
    _write_dialogue_for_render_script(paths)
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
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _write_dialogue_for_render_script(paths)
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    with pytest.raises(FileNotFoundError) as exc:
        run_render(project, episode=2)
    assert "audio 阶段" in str(exc.value)


@pytest.mark.asyncio
async def test_run_pipeline_from_voice_runs_render_stages(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    _write_dialogue_for_render_script(paths)
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    _stub_mix_audio_ffmpeg(monkeypatch)
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
async def test_run_pipeline_never_disables_frame_alignment(project, monkeypatch):
    """run_pipeline 走 timeline 阶段时，帧率**必须**被探到并记进产物。

    这条锁的是 M6 的结论：`run_timeline` 的 `source_duration` 与 `frame_rate` 共享一个
    开关，而 preflight 只探时长、不探帧率。「让 run_timeline 复用 preflight 刚探到的
    时长」这个看着很省的改法，只递 source_duration 就会把帧对齐**静默关掉** —— 成片
    退回「ffmpeg 自己按帧取整」，没有任何报错，只有首尾各差不到一帧。
    """
    from tenmin.models import Timeline

    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    _write_dialogue_for_render_script(paths)
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 23.976)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    _stub_mix_audio_ffmpeg(monkeypatch)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    await run_pipeline(
        project,
        FakeProvider([]),
        from_stage="voice",
        tts_engine=FakeTTSEngine([8.0, 10.0, 10.0]),
    )
    timeline = Timeline.model_validate_json(paths.timeline(2).read_text(encoding="utf-8"))
    assert timeline.frame_rate == pytest.approx(23.976)


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
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    _stub_mix_audio_ffmpeg(monkeypatch)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    # 处理顺序跟 cfg.episodes 一致：project 先注册了第 2 集，再 append 第 1 集。
    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    # 80 秒 = 360 字 / 4.5 字每秒，也就是这份 fixture 的旁白**真实**会有的长度。原来写
    # 的 8.0 秒相当于每秒念 45 个字，物理上不可能；后果是每条字幕只显示 0.67 秒，
    # 字幕可读性检查照实报了 72 条 warning，而这个 fixture 想表达的是「一次健康的
    # 批处理」。这里不该靠调阈值绕过去，该修的是假时长。
    tts_engine = FakeTTSEngine([80.0] * 20)

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
    op_range/ed_range 从 project.yaml 里静默抹掉（曾经真出过这个 bug）。
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
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    _stub_mix_audio_ffmpeg(monkeypatch)
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
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr(
        "tenmin.pipeline.preflight",
        lambda video, encoder, **_: (events.append(f"preflight:{Path(video).name}"), 1400.0)[1],
    )
    _stub_mix_audio_ffmpeg(monkeypatch)
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
    _write_dialogue_for_render_script(paths)
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
    _write_dialogue_for_render_script(paths)
    _prepare_video(project)
    project.render.ffprobe_path = "/opt/x/ffprobe"
    seen: dict[str, str] = {}

    def fake_probe(path, *, ffprobe="ffprobe"):
        seen["duration"] = ffprobe
        return 1400.0

    def fake_frame_rate(path, *, ffprobe="ffprobe"):
        seen["frame_rate"] = ffprobe
        return 25.0

    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", fake_frame_rate)
    run_timeline(project, episode=2)
    assert seen == {"duration": "/opt/x/ffprobe", "frame_rate": "/opt/x/ffprobe"}


def test_run_audio_uses_configured_ffmpeg_path(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _write_dialogue_for_render_script(paths)
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    project.render.ffmpeg_path = "/opt/x/ffmpeg"
    seen: dict[str, str] = {}

    _stub_mix_audio_ffmpeg(monkeypatch, ffmpeg_seen=seen)
    run_audio(project, episode=2)
    assert seen["ffmpeg"] == "/opt/x/ffmpeg"


def test_run_render_uses_configured_ffmpeg_path(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _write_dialogue_for_render_script(paths)
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
    _write_dialogue_for_render_script(paths)
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
    _stub_mix_audio_ffmpeg(monkeypatch)
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
    _write_dialogue_for_render_script(paths)
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    _stub_mix_audio_ffmpeg(monkeypatch)
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


def test_run_render_wires_the_outro_font_name_into_drawtext(project, monkeypatch):
    """preflight 查的那个字体名，必须**就是** drawtext 真正用上的那个。

    历史 bug：pipeline 把 `cfg.render.outro_font_name` 递给 preflight 去 fc-list 查，
    而 render/video.py 的 drawtext 读的是模块级 `OUTRO_FONT_NAME`（= config 默认值）。
    于是配了自定义字体的用户会收到一条**假 warning**（查一个永远不会被用到的字体），
    而真正会被烧进片尾卡的那个字体反倒没人查过。
    """
    from tenmin.config import RenderConfig
    from tenmin.render.video import build_render_args

    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]), episode=2))
    _write_dialogue_for_render_script(paths)
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    paths.mixed_audio(2).parent.mkdir(parents=True, exist_ok=True)
    paths.mixed_audio(2).write_bytes(b"\x00")
    project.render.outro_font_name = "Hiragino Sans"
    project.render.outro_card_seconds = 3.0

    seen: dict[str, object] = {}

    def fake_render_video(**kwargs):
        seen.update(kwargs)
        return kwargs["out_path"]

    monkeypatch.setattr("tenmin.pipeline.render_video", fake_render_video)
    run_render(project, episode=2)
    assert seen["outro_font_name"] == "Hiragino Sans"

    # 再往下一层：这个值真的进了 filtergraph（不是被 render_video 吃掉了）。
    graph_args = build_render_args(
        video=Path("in.mkv"),
        timeline=seen["timeline"],
        audio=Path("a.m4a"),
        ass=Path("s.ass"),
        out_path=Path("out.mp4"),
        encoder="libx264",
        outro_seconds=3.0,
        outro_title="t",
        outro_message="m",
        outro_font_name=seen["outro_font_name"],
    )
    graph = graph_args[graph_args.index("-filter_complex") + 1]
    assert "font='Hiragino Sans'" in graph
    assert RenderConfig().outro_font_name not in graph


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


def test_register_episode_rejects_a_missing_srt_before_touching_anything(tmp_path):
    cfg = _project_config(tmp_path)
    before = cfg.config_path.read_text(encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="找不到要登记的字幕文件"):
        register_episode(
            cfg, episode=3, srt=tmp_path / "nope.srt", video=tmp_path / "e03.mkv"
        )

    assert cfg.config_path.read_text(encoding="utf-8") == before
    assert [e.number for e in cfg.episodes] == [2]


def test_register_episode_rejects_invalidated_yaml_before_copying_srt(tmp_path):
    cfg = _project_config(tmp_path)
    before = (
        "show: 才女的侍从\nslug: saijo\nrender:\n"
        "  <<: &base {font_size: 50}\n  <<: {width: 1280}\n"
    )
    cfg.config_path.write_text(before, encoding="utf-8")
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,000\n台词\n", encoding="utf-8")

    with pytest.raises(ProjectConfigError, match="<<"):
        register_episode(cfg, episode=3, srt=source, video=tmp_path / "e03.mkv")

    assert cfg.config_path.read_text(encoding="utf-8") == before
    assert not (cfg.root / "srt" / "E03.srt").exists()
    assert [episode.number for episode in cfg.episodes] == [2]


def test_register_episode_preserves_a_single_merge_key(tmp_path):
    root = tmp_path / "saijo"
    root.mkdir()
    yaml_path = root / "project.yaml"
    before = (
        "show: 才女的侍从\nslug: saijo\nrender:\n"
        "  <<: {font_size: 50, width: 1280}\n  font_size: 60\n"
        "episodes: []\n"
    )
    yaml_path.write_text(before, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e03.mkv"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=3, srt=None, video=video)

    after = yaml_path.read_text(encoding="utf-8")
    assert "  <<: {font_size: 50, width: 1280}\n  font_size: 60\n" in after
    reloaded = load_project(yaml_path)
    assert reloaded.render.font_size == 60
    assert reloaded.render.width == 1280
    assert reloaded.episodes[0].video == video.resolve()


# --- 产物原子写 -------------------------------------------------------------
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


# --- E1：语义校验耗尽重试时把最后一版稿子落盘 ---


@pytest.mark.asyncio
async def test_run_script_persists_the_rejected_draft(project):
    """一次真实调用可达 561 秒，原来重试耗尽后什么都不留：用户既看不到模型写了什么，
    也无从判断是判据太严还是模型真写错了。"""
    run_ingest(project)
    run_signals(project)
    bad = LLMScript(
        beats=[
            LLMBeat(
                id=f"b{i + 1}",
                label=["Hook 开场", "阶段一", "收尾：完"][i],
                role=["hook", "act", "outro"][i],
                narration="啊" * 360,
                clips=[LLMClip(episode=2, start=9000.0, end=9080.0, visual="假的")],
            )
            for i in range(3)
        ]
    )
    # 首发 + LLMConfig.validation_retries 次重试，全喂同一份坏稿把重试耗尽。
    provider = FakeProvider([bad] * (1 + project.llm.validation_retries))
    paths = Paths(project.root)
    with pytest.raises(ScriptValidationError) as exc:
        await run_script(project, provider, episode=2)
    rejected = paths.script_rejected(2)
    assert rejected.exists()
    assert Script.model_validate_json(rejected.read_text(encoding="utf-8")).beats
    assert str(rejected) in str(exc.value)


@pytest.mark.asyncio
async def test_run_script_persists_an_empty_warnings_file(project):
    """warnings 原来只在内存里攒着、运行末尾由 cli.py 打一次黄字：不落盘、不进对照表，
    而阶段一旦 fresh 就被跳过，重跑连黄字都不再出现。于是 validate.py 里那批「只给
    warning」的检查在「没人盯终端」的用法下等于空转。

    没有 warning 也要写文件：空列表 = 查过了没发现问题，文件缺失 = 从没跑过，
    这两件事必须能分开。"""
    run_ingest(project)
    run_signals(project)
    paths = Paths(project.root)
    _, warnings = await run_script(
        project, FakeProvider([fake_script_response()]), episode=2
    )
    assert warnings == []
    path = paths.script_warnings(2)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "episode": 2,
        "warnings": [],
    }


@pytest.mark.asyncio
async def test_run_script_persists_the_warnings_it_found(project):
    """3 个节点、零 climax：两条结构 warning，都不判错（min_beats 还是 3）。
    这就是「只落盘不判错」那一档要留下的证据。"""
    off_spec = LLMScript(
        beats=[
            LLMBeat(
                id=f"b{i + 1}",
                label=["Hook 开场", "阶段一", "收尾：完"][i],
                role=["hook", "act", "outro"][i],
                narration=("啊" * 29 + "。") * 12,
                clips=[
                    LLMClip(
                        episode=2,
                        start=300.0 + i * 100,
                        end=300.0 + i * 100 + 80.0,
                        visual="画面",
                    )
                ],
            )
            for i in range(3)
        ]
    )
    run_ingest(project)
    run_signals(project)
    paths = Paths(project.root)
    _, warnings = await run_script(
        project, FakeProvider([off_spec, off_spec]), episode=2
    )
    assert warnings, "3 个节点 + 零 climax 该报 warning"
    payload = json.loads(paths.script_warnings(2).read_text(encoding="utf-8"))
    assert payload["episode"] == 2
    assert payload["warnings"] == warnings
    # 中文必须是原样的字，不是 \\uXXXX 转义 —— 这份文件是给人打开看的。
    assert "节点数" in paths.script_warnings(2).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_run_script_reports_substeps(project, monkeypatch):
    run_ingest(project)
    run_signals(project)
    seen: list[tuple[str, int, int, str]] = []

    class Recorder(NullProgressReporter):
        def substep(self, stage, current, total, label):
            seen.append((stage, current, total, label))

    await run_script(
        project, FakeProvider([fake_script_response()]), episode=2, reporter=Recorder()
    )
    assert seen and seen[0][0] == "script"


# --- script 的术语表来源：累积表叠手写表 ---


@pytest.mark.asyncio
async def test_run_script_feeds_the_accumulated_glossary(project):
    """盘上那份跨集累积表真的进了 prompt。

    project fixture 的 yaml 没写 glossary，所以这一条只可能来自 zh/glossary.json。
    """
    run_ingest(project)
    run_signals(project)
    paths = Paths(project.root)
    _file(paths.glossary, json.dumps({"リディア": "莉迪亚"}, ensure_ascii=False))
    provider = FakeProvider([fake_script_response()])

    await run_script(project, provider, episode=2)

    assert "リディア → 莉迪亚" in provider.calls[0]["user"]


@pytest.mark.asyncio
async def test_run_script_lets_the_manual_glossary_override_the_accumulated_one(project):
    """手写表是纠错入口：改了 project.yaml 却不生效会是个很难查的问题。"""
    project.glossary = {"リディア": "莉蒂亚"}
    run_ingest(project)
    run_signals(project)
    paths = Paths(project.root)
    _file(paths.glossary, json.dumps({"リディア": "莉迪亚"}, ensure_ascii=False))
    provider = FakeProvider([fake_script_response()])

    await run_script(project, provider, episode=2)

    prompt = provider.calls[0]["user"]
    assert "リディア → 莉蒂亚" in prompt
    assert "莉迪亚" not in prompt


@pytest.mark.asyncio
async def test_run_script_survives_a_missing_glossary(project):
    """现有的繁中片源压根不会有 zh/glossary.json，那不是错误。"""
    run_ingest(project)
    run_signals(project)
    assert not Paths(project.root).glossary.exists()
    provider = FakeProvider([fake_script_response()])

    await run_script(project, provider, episode=2)

    assert "（无术语表）" in provider.calls[0]["user"]


# --- script 阶段的多集并发（llm.script_concurrency）---


def _three_episode_project(project, golden_srt_path) -> ProjectConfig:
    """在 project fixture 上再挂两集（都用同一份黄金 SRT）。"""
    for number in (1, 3):
        srt = project.root / "srt" / f"E{number:02d}.srt"
        srt.write_bytes(golden_srt_path.read_bytes())
        project.episodes.append(
            EpisodeConfig(number=number, srt=Path(f"srt/E{number:02d}.srt"))
        )
    return project


@pytest.mark.asyncio
async def test_run_pipeline_runs_scripts_concurrently(project, golden_srt_path):
    """并发的全部意义：墙钟 ≈ max 而不是 sum。

    单集 script 实测可达 561 秒，3 集串行就是半小时起。这里用 0.15 秒的假延迟代替。
    """
    _three_episode_project(project, golden_srt_path)
    project.llm.script_concurrency = 3
    provider = EpisodeAwareProvider(fake_script_response, delay=0.15)

    started = asyncio.get_running_loop().time()
    await run_pipeline(project, provider, only=V1_STAGES)
    elapsed = asyncio.get_running_loop().time() - started

    assert sorted(provider.called_episodes) == [1, 2, 3]
    assert provider.peak == 3
    assert elapsed < 0.4  # 串行至少 0.45 秒


@pytest.mark.asyncio
async def test_run_pipeline_caps_script_concurrency(project, golden_srt_path):
    _three_episode_project(project, golden_srt_path)
    project.llm.script_concurrency = 2
    provider = EpisodeAwareProvider(fake_script_response, delay=0.05)
    await run_pipeline(project, provider, only=V1_STAGES)
    assert provider.peak == 2


@pytest.mark.asyncio
async def test_run_pipeline_does_not_prefetch_at_concurrency_one(
    project, golden_srt_path, monkeypatch
):
    """默认 script_concurrency=1 必须**一点都不预取**：下一集的 LLM 请求不能在这一集的
    后续阶段之前发出去。否则「中途失败留下完整交付物」的代价（白花的 token、被取消的
    请求）会在没人打开并发的情况下也照样付。"""
    _three_episode_project(project, golden_srt_path)
    provider = EpisodeAwareProvider(fake_script_response)

    real_docgen = run_docgen

    def spy_docgen(cfg, episode):
        provider.events.append(("docgen", episode))
        return real_docgen(cfg, episode)

    monkeypatch.setattr("tenmin.pipeline.run_docgen", spy_docgen)
    await run_pipeline(project, provider, only=V1_STAGES)

    assert provider.peak == 1
    # 每一集都必须是「LLM 起 → LLM 完 → docgen」三连，中间不许插进别的集。
    assert provider.events == [
        item
        for episode in (2, 1, 3)
        for item in (("llm-start", episode), ("llm-done", episode), ("docgen", episode))
    ]


@pytest.mark.asyncio
async def test_run_pipeline_writes_each_episodes_own_script(project, golden_srt_path):
    """并发下最容易错的一件事：把 A 集的稿子写进 B 集的产物。"""
    _three_episode_project(project, golden_srt_path)
    project.llm.script_concurrency = 3
    provider = EpisodeAwareProvider(fake_script_response, delay=0.05)
    await run_pipeline(project, provider, only=V1_STAGES)

    paths = Paths(project.root)
    for number in (1, 2, 3):
        script = Script.model_validate_json(paths.script(number).read_text("utf-8"))
        assert script.episodes == [number]
        assert {clip.episode for beat in script.beats for clip in beat.clips} == {number}


@pytest.mark.asyncio
async def test_run_pipeline_keeps_earlier_deliverables_when_a_later_script_fails(
    project, golden_srt_path
):
    """批量模式的不变量：中途失败要留下**完整**交付物，而不是一堆半成品。

    并发预取不许破坏它 —— 按集纵向的循环顺序没变，所以第 3 集的 script 炸掉时前两集
    的 docgen 产物必须都在。
    """
    _three_episode_project(project, golden_srt_path)
    project.llm.script_concurrency = 3
    project.llm.max_attempts = 1
    project.llm.validation_retries = 0

    def responder(episode):
        if episode == 3:
            raise LLMSchemaError("第 3 集的模型输出不合 schema", raw_output="{坏的")
        return fake_script_response(episode)

    provider = EpisodeAwareProvider(responder, delay=0.05)
    with pytest.raises(LLMSchemaError):
        await run_pipeline(project, provider, only=V1_STAGES)

    paths = Paths(project.root)
    # 第 3 集是 target_numbers 的最后一个，所以前两集应该已经整套跑完。
    for number in (2, 1):
        assert paths.script(number).exists()
        assert paths.table(number).exists()
        assert paths.narration(number).exists()
    assert not paths.script(3).exists()


@pytest.mark.asyncio
async def test_run_pipeline_cancels_pending_script_tasks_when_a_stage_fails(
    project, golden_srt_path, monkeypatch
):
    """一集的后续阶段炸了之后，还在飞的 script task 必须被取消。

    留着的话它们会在 run_pipeline 抛出去之后继续烧 token，事件循环关闭时 asyncio 还会
    印一串 "Task was destroyed but it is pending"。
    """
    _three_episode_project(project, golden_srt_path)
    project.llm.script_concurrency = 3

    def boom(cfg, episode):
        raise FFmpegError("docgen 炸了")

    monkeypatch.setattr("tenmin.pipeline.run_docgen", boom)
    provider = EpisodeAwareProvider(fake_script_response, delay=0.2)

    before = asyncio.all_tasks()
    with pytest.raises(FFmpegError):
        await run_pipeline(project, provider, only=V1_STAGES)
    leaked = asyncio.all_tasks() - before
    assert leaked == set()
    # 第一集之外的两集要么没起、要么被取消，绝不该跑完。
    assert ("llm-done", 1) not in provider.events
    assert ("llm-done", 3) not in provider.events


@pytest.mark.asyncio
async def test_run_pipeline_labels_script_substeps_with_the_episode(
    project, golden_srt_path
):
    """并发下 N 集的 substep 全打在同一个 stage（"script"）上，rich_progress 按 stage
    做 key —— 不带集号的话那一行会在几集之间来回跳而不告诉你现在跳的是谁。"""
    _three_episode_project(project, golden_srt_path)
    project.llm.script_concurrency = 3
    reporter = FakeReporter()
    provider = EpisodeAwareProvider(fake_script_response, delay=0.05)
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)

    labels = [c[4] for c in reporter.calls if c[0] == "substep" and c[1] == "script"]
    assert sorted(labels) == sorted(["E01 生成初稿", "E02 生成初稿", "E03 生成初稿"])


@pytest.mark.asyncio
async def test_run_pipeline_keeps_episode_hooks_paired_under_concurrency(
    project, golden_srt_path
):
    """总进度条是按 index 推进的，episode_start/done 必须仍然严格成对、按集顺序。"""
    _three_episode_project(project, golden_srt_path)
    project.llm.script_concurrency = 3
    reporter = FakeReporter()
    provider = EpisodeAwareProvider(fake_script_response, delay=0.05)
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)

    assert [c for c in reporter.calls if c[0].startswith("episode_")] == [
        ("episode_start", 2, 1, 3),
        ("episode_done", 2, 1, 3),
        ("episode_start", 1, 2, 3),
        ("episode_done", 1, 2, 3),
        ("episode_start", 3, 3, 3),
        ("episode_done", 3, 3, 3),
    ]


@pytest.mark.asyncio
async def test_run_pipeline_does_not_launch_scripts_for_fresh_episodes(
    project, golden_srt_path
):
    """已经最新的集不该起 task（也就不该发请求），并且仍然报 stage_skip。"""
    _three_episode_project(project, golden_srt_path)
    project.llm.script_concurrency = 3
    await run_pipeline(
        project, EpisodeAwareProvider(fake_script_response), only=V1_STAGES
    )

    reporter = FakeReporter()
    provider = EpisodeAwareProvider(fake_script_response)
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    assert provider.called_episodes == []
    assert [c for c in reporter.calls if c == ("stage_skip", "script")] == [
        ("stage_skip", "script")
    ] * 3


# --- 生肉入口：一集可以只有视频 -------------------------------------------


def test_the_transcription_cache_name_ends_with_asr_srt(tmp_path):
    """转写缓存的名字形状由 Paths 一处钉死。

    ingest/resolve.py 的 _embedded_dest 刻意对 cache 的名字**不作形状要求**（它靠
    Path(name).stem 对任何输入都能换出一个不撞车的 .embedded.srt 名），所以「转写
    缓存到底叫什么」这件事在那一侧完全没有守卫。如果调用方把它拼成 srt/E11.srt，
    抽出来的软字幕仍然叫 E11.embedded.srt，而这份机器听写的东西就跟手传字幕同名
    同形，从文件名上再也分不出来 —— 那正是 resolve 里两个后缀必须不同的理由。
    所以这条不变量必须由命名权威（Paths）这一侧来锁。
    """
    assert Paths(tmp_path).asr_cache(11).name.endswith(".asr.srt")


def test_the_transcription_cache_sits_next_to_the_hand_written_srt(tmp_path):
    """它落在 srt/ 而不是 01_dialogue/：转差了要能当手传字幕直接改、直接复用。"""
    assert Paths(tmp_path).asr_cache(11).parent == tmp_path / "srt"


def test_register_episode_without_an_srt_leaves_the_field_empty(tmp_path):
    cfg = _project_config(tmp_path)
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")

    updated = register_episode(cfg, episode=11, srt=None, video=video)

    episode = next(e for e in updated.episodes if e.number == 11)
    assert episode.srt is None
    assert episode.video == video.resolve()
    assert not (cfg.root / "srt" / "E11.srt").exists()


def test_register_episode_without_an_srt_omits_it_from_the_yaml(tmp_path):
    cfg = _project_config(tmp_path)
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=11, srt=None, video=video)

    data = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
    entry = next(e for e in data["episodes"] if e["number"] == 11)
    assert "srt" not in entry


def test_register_episode_without_an_srt_clears_a_previous_one(tmp_path, golden_srt_path):
    """重登记同一集时不传 srt，就等于「这一集改走生肉路线」，旧字幕字段必须清掉。

    留着旧值会让 resolve 继续走「手传 SRT」那条路、对着一份用户已经不想用的字幕
    出片，而 project.yaml 上看不出任何异常。
    """
    cfg = _project_config(tmp_path)
    srt = tmp_path / "incoming.srt"
    srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    register_episode(cfg, episode=11, srt=srt, video=video)

    register_episode(cfg, episode=11, srt=None, video=video)

    episode = next(e for e in cfg.episodes if e.number == 11)
    assert episode.srt is None


@pytest.mark.asyncio
async def test_ingest_freshness_survives_a_video_only_episode(project):
    """ingest 的新鲜度判据扫**全部**集的字幕输入，一集没有 srt 不能把它打死。

    只跑 ingest、且刻意让它判成「已最新」：这条要测的是判据本身，不是 run_ingest。
    判据里没滤掉 None 的话崩点是 _is_fresh 里的 `p.exists()`，报一个指不到根因的
    AttributeError。

    别把 only 换成 []（什么都不跑）：那样 srt 输入压根不会被送进 _is_fresh，
    测试对「有没有滤掉 None」完全无感（实测把滤除去掉照样全绿）。
    """
    video = project.root / "e11.mp4"
    video.write_bytes(b"fake")
    project.episodes.append(EpisodeConfig(number=11, video=video))

    paths = Paths(project.root)
    for number in (2, 11):
        _file(paths.dialogue(number), '{"episode": 0, "lines": []}')
        _shift_mtime(paths.dialogue(number), 60)

    reporter = FakeReporter()
    await run_pipeline(
        project, FakeProvider([]), only=["ingest"], reporter=reporter
    )

    assert ("stage_skip", "ingest") in reporter.calls


@pytest.mark.asyncio
async def test_a_newer_srt_makes_ingest_rerun(project):
    """改了字幕就必须重跑 ingest —— 那份 SRT 是这个阶段的输入，不只是个摆设。

    跟上面那条是一对：滤掉生肉集的 None 时很容易顺手把整份字幕输入都丢掉
    （让 _ingest_inputs 返回空列表），而那样一来 ingest 就只盯 project.yaml，
    「改字幕再重跑」会被静默 stage_skip、下游各阶段又因为 dialogue.json 没变而跟着
    一起跳过，用户拿到跟改动前一模一样的产物。实测把那份输入换成空列表时全仓测试
    **一条都不红**，所以这条不变量此前压根没人守。
    """
    paths = Paths(project.root)
    _file(paths.dialogue(2), '{"episode": 0, "lines": []}')
    _shift_mtime(paths.dialogue(2), 60)
    _shift_mtime(project.srt_path(project.episodes[0]), 120)

    reporter = FakeReporter()
    await run_pipeline(
        project, FakeProvider([]), only=["ingest"], reporter=reporter
    )

    assert ("stage_start", "ingest") in reporter.calls
    assert ("stage_skip", "ingest") not in reporter.calls


def test_ingest_inputs_include_a_video_only_episodes_video(tmp_path):
    """只有 video 的集也得能算出「该不该重跑」。原来的输入集合只取 srt，
    对这种集会得到一个空输入列表 —— 而空输入在新鲜度判据里等于「跳过」，
    于是换了片源也不会重跑。"""
    cfg = _project_config(tmp_path)
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=11, srt=None, video=video)

    assert video.resolve() in _ingest_inputs(cfg)


def test_ingest_inputs_include_both_when_both_exist(tmp_path, golden_srt_path):
    cfg = _project_config(tmp_path)
    srt = tmp_path / "hand.srt"
    srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=11, srt=srt, video=video)

    inputs = _ingest_inputs(cfg)
    assert video.resolve() in inputs
    assert cfg.root / "srt" / "E11.srt" in inputs


def test_ingest_inputs_exclude_the_asr_cache(tmp_path):
    """转写缓存是 ingest 自己的产物。把它算进 ingest 的输入会让
    「转写完写出缓存」这个动作立刻使 ingest 变得不新鲜 —— 每次都重跑。"""
    cfg = _project_config(tmp_path)
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=11, srt=None, video=video)
    _file(Paths(cfg.root).asr_cache(11), "1\n00:00:01,000 --> 00:00:02,000\nはい\n")

    assert all(".asr.srt" not in str(p) for p in _ingest_inputs(cfg))


def _fake_resolve(recorded: list[dict]):
    """替掉 run_ingest 里的来源解析：记下每次调用，按「有没有手传 srt」分两条路。

    刻意不整份写死成一个固定返回值：run_ingest 遍历全部集，而这些用例的项目里既有
    生肉集也有字幕齐全的第 2 集，写死会把后者一起改掉、看不出是哪一条路在起作用。
    """

    def fake(srt, video, **kwargs):
        recorded.append({"srt": srt, "video": video, **kwargs})
        if srt is not None:
            return SubtitleSource(srt, "srt")
        return SubtitleSource(kwargs["cache"], "asr")

    return fake


def _register_raw_episode(cfg: ProjectConfig, tmp_path: Path, text: str) -> ProjectConfig:
    """登记一个只有视频的第 11 集，并把它的转写缓存预先摆好（内容由调用方给）。"""
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=11, srt=None, video=video)
    _file(Paths(cfg.root).asr_cache(11), text)
    return cfg


def test_ingest_resolves_a_video_only_episode_through_the_source_layer(
    project, tmp_path, monkeypatch
):
    """生肉集不再撞守卫，而是走来源解析拿到一份 SRT。

    顺带钉住转写缓存的落点跟 Paths.asr_cache 一致：实现里内联一条与布局表不符的路径
    （目录或后缀写错）这条会红 —— 实测把 cache 换成 `01_dialogue/E11.asr.srt` 或
    `srt/E11.transcript.srt` 两种内联写法，这条都失败。

    但它**不**保护「路径必须经由 Paths 取」：断言两边读的是同一张表，所以换成一条与
    当前布局一致的内联字面量（`srt/E11.asr.srt`）全仓照样全绿（实测）。布局表自己改
    后缀由 test_paths_layout_is_frozen 与
    test_the_transcription_cache_name_ends_with_asr_srt 守，不由这条守。
    """
    cfg = _register_raw_episode(
        project, tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nはい\n"
    )
    recorded: list[dict] = []
    monkeypatch.setattr(
        "tenmin.pipeline.resolve_subtitle_source", _fake_resolve(recorded)
    )

    tracks = run_ingest(cfg)

    raw = next(call for call in recorded if call["srt"] is None)
    assert raw["cache"] == Paths(cfg.root).asr_cache(11)
    # 软字幕轨那条路每次调用都会重抽，所以每集只许解析一次。
    assert len(recorded) == len(cfg.episodes)
    assert next(t for t in tracks if t.episode == 11).source == "asr"


def test_ingest_never_runs_opencc_on_a_transcribed_track(project, tmp_path, monkeypatch):
    """项目开着繁转简（默认就开）也不能碰听写来的日语对白 —— OpenCC 会改字。

    这一条要在 run_ingest 这一层锁：normalize 那边的单元测试管「build_track 自己
    压得住 convert」，这里管「run_ingest 真的把 source 递了下去」。
    """
    assert project.locale.convert_traditional is True
    cfg = _register_raw_episode(
        project, tmp_path, "1\n00:00:01,000 --> 00:00:02,000\n製作の話\n"
    )
    monkeypatch.setattr("tenmin.pipeline.resolve_subtitle_source", _fake_resolve([]))

    tracks = run_ingest(cfg)

    track = next(t for t in tracks if t.episode == 11)
    assert "製作" in "".join(line.text for line in track.lines)


# --- translate 阶段 ---------------------------------------------------------


def _project_with_dialogue(
    tmp_path: Path, *, episode: int, source: str
) -> tuple[ProjectConfig, Paths]:
    """一个只登记了这一集的 project，外加一份已落盘的对白轨产物。

    刻意只登记一集：run_translate 走 _load_tracks，而它要求 cfg.episodes 里的**每一
    集**都有对白轨产物，多登记一集就得多摆一份产物，而这些用例只关心一集。

    yaml 里写不写 srt 跟着 source 走（听写来的那一集只有视频），纯粹为了让
    project.yaml 看着像真的：run_translate 的判据只有对白轨里的 source 字段。
    """
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True, exist_ok=True)
    video = tmp_path / f"e{episode}.mp4"
    video.write_bytes(b"fake")
    entry = f"- number: {episode}\n  video: {video}\n"
    if source == "srt":
        srt_name = f"E{episode:02d}.srt"
        (root / "srt" / srt_name).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n", encoding="utf-8"
        )
        entry += f"  srt: srt/{srt_name}\n"
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nmode: single_episode\n"
        f"target_seconds: 240\nepisodes:\n{entry}",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)
    paths = Paths(cfg.root)
    track = DialogueTrack(
        episode=episode,
        source=source,
        duration=100.0,
        lines=[
            DialogueLine(idx=1, start=1.0, end=2.0, text="はい", raw="はい"),
            DialogueLine(idx=2, start=3.0, end=4.0, text="そうですね", raw="そうですね"),
        ],
    )
    _file(paths.dialogue(episode), track.model_dump_json(indent=2))
    return cfg, paths


def _translation_response(glossary: dict[str, str] | None = None) -> dict:
    return {
        "episode": 11,
        "lines": [{"id": 1, "zh": "是的"}, {"id": 2, "zh": "说得对"}],
        "glossary": glossary or {},
    }


def test_translate_sits_between_ingest_and_signals():
    assert STAGES.index("ingest") < STAGES.index("translate") < STAGES.index("signals")


async def test_run_translate_writes_all_three_artifacts(tmp_path):
    """译文轨（中间产物）、中文字幕（交付物）、累积术语表（回写）。"""
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    provider = FakeProvider([_translation_response({"リディア": "莉迪亚"})])

    await run_translate(cfg, provider, 11)

    assert paths.zh_lines(11).is_file()
    assert paths.zh_subtitles(11).is_file()
    assert "是的" in paths.zh_subtitles(11).read_text(encoding="utf-8")
    assert json.loads(paths.glossary.read_text(encoding="utf-8")) == {"リディア": "莉迪亚"}


async def test_run_translate_accumulates_into_an_existing_glossary(tmp_path):
    """第 2 集要看得见第 1 集定下的译名，且不能把它改掉。"""
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    _file(paths.glossary, json.dumps({"リディア": "莉迪亚"}, ensure_ascii=False))
    provider = FakeProvider(
        [_translation_response({"リディア": "莉蒂亚", "ルーファス": "鲁弗斯"})]
    )

    await run_translate(cfg, provider, 11)

    stored = json.loads(paths.glossary.read_text(encoding="utf-8"))
    assert stored["リディア"] == "莉迪亚"
    assert stored["ルーファス"] == "鲁弗斯"


async def test_run_translate_leaves_an_unchanged_glossary_untouched(tmp_path):
    """内容没变就不许刷 mtime —— 累积表现在是 script 的真输入。

    不然一次 E01…E10 的批处理跑完，E10 的 translate 会把前 9 集的解说稿全部判旧，
    下一次运行白付 9 次 LLM 费（一次调用实测可达 561 秒）。
    """
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    provider = FakeProvider([_translation_response({"リディア": "莉迪亚"})] * 2)

    await run_translate(cfg, provider, 11)
    # 往回挪，免得「两次写盘落在同一个时间戳」把回归悄悄盖掉。
    _shift_mtime(paths.glossary, -600.0)
    stamp = paths.glossary.stat().st_mtime

    await run_translate(cfg, provider, 11)

    assert paths.glossary.stat().st_mtime == stamp


async def test_run_translate_skips_a_native_subtitle_episode(tmp_path):
    """现有片源自带中文字幕，整个阶段不该跑，zh/ 目录都不该建。

    判据是对白轨的 source 字段，刻意不做语言自动检测：那是个会错的猜测，而 source
    是一个确定的事实。FakeProvider 的预置响应给空列表，它被调用就抛。
    """
    cfg, paths = _project_with_dialogue(tmp_path, episode=2, source="srt")

    result = await run_translate(cfg, FakeProvider([]), 2)

    assert not paths.zh_lines(2).exists()
    assert not paths.zh_subtitles(2).exists()
    assert not paths.glossary.exists()
    # 返回值仍然得是这一集的空轨：库调用方拿它当「这一集翻了什么」的答案。
    assert result.episode == 2
    assert result.lines == []


async def test_run_pipeline_runs_translate_for_a_transcribed_episode(tmp_path):
    """接线：--only translate 真的会把这一集翻出来，而且进度上报是「跑了」而不是「跳了」。"""
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    provider = FakeProvider([_translation_response()])
    reporter = FakeReporter()

    await run_pipeline(cfg, provider, only=["translate"], reporter=reporter)

    assert paths.zh_lines(11).is_file()
    assert provider.calls
    assert ("stage_start", "translate") in reporter.calls
    assert ("stage_done", "translate") in reporter.calls


async def test_run_pipeline_skips_a_fresh_translate(tmp_path):
    """产物比对白轨新就别再付一次翻译费。"""
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    _file(paths.zh_lines(11), "{}")
    _file(paths.zh_subtitles(11), "x")
    _shift_mtime(paths.zh_lines(11), 60.0)
    _shift_mtime(paths.zh_subtitles(11), 60.0)
    provider = FakeProvider([])
    reporter = FakeReporter()

    await run_pipeline(cfg, provider, only=["translate"], reporter=reporter)

    assert provider.calls == []
    assert ("stage_skip", "translate") in reporter.calls


async def test_run_pipeline_rebuilds_a_deleted_zh_subtitle(tmp_path):
    """上面那条的姊妹用例：中文字幕是交付物，缺了就不许跳过。

    译文轨住在 zh/（中间产物）、中文字幕住在 out/（交付物），用户会单独去删后者。
    所以两者都得算进 translate 的产物集 —— 只看译文轨的话，这里会静默跳过，那份
    被删掉的字幕再也补不回来。
    """
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    _file(paths.zh_lines(11), "{}")
    _shift_mtime(paths.zh_lines(11), 60.0)
    assert not paths.zh_subtitles(11).exists()
    provider = FakeProvider([_translation_response()])

    await run_pipeline(cfg, provider, only=["translate"])

    assert provider.calls
    assert paths.zh_subtitles(11).is_file()
    assert "是的" in paths.zh_subtitles(11).read_text(encoding="utf-8")


def test_script_freshness_depends_on_the_glossary(tmp_path):
    """累积术语表是 script 的真输入：run_script 把它喂进 prompt 的术语表那一节，
    所以表变了旧解说稿里的译名就对不上，必须重跑。"""
    paths = Paths(tmp_path)
    _file(paths.dialogue(11), "{}")
    _file(paths.signals(11), "{}")
    script = _file(paths.script(11), "{}")

    assert paths.glossary in _script_inputs(paths, 11)
    assert _is_fresh([script], _script_inputs(paths, 11)) is True
    _file(paths.glossary, '{"リディア": "莉迪亚"}')
    _shift_mtime(paths.glossary, 60.0)
    assert _is_fresh([script], _script_inputs(paths, 11)) is False


def test_translate_freshness_does_not_include_the_glossary(tmp_path):
    """术语表既是 translate 的输入又是它的输出。把它算进输入集，这个阶段就永远不
    新鲜 —— 每次都重跑，每次都重新付翻译费。"""
    paths = Paths(tmp_path)
    _file(paths.dialogue(11), "{}")
    outputs = [_file(paths.zh_lines(11), "{}"), _file(paths.zh_subtitles(11), "x")]

    inputs = _translate_inputs(paths, 11)
    assert paths.dialogue(11) in inputs
    assert paths.glossary not in inputs

    _file(paths.glossary, '{"リディア": "莉迪亚"}')
    _shift_mtime(paths.glossary, 60.0)
    assert _is_fresh(outputs, _translate_inputs(paths, 11)) is True


# --- script 的预取窗口与 translate 的交互 -----------------------------------


def _project_with_asr_dialogues(
    tmp_path: Path, numbers: tuple[int, ...]
) -> tuple[ProjectConfig, Paths]:
    """多集全是听写来的（source="asr"）project，对白轨已落盘。

    只够 translate/script 做新鲜度判定用：这些用例把两个阶段本体都换成替身，所以
    不需要真 SRT，也不需要 signals 产物（_is_fresh 自己会滤掉不存在的输入）。
    """
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True, exist_ok=True)
    entries = ""
    for number in numbers:
        video = tmp_path / f"e{number}.mp4"
        video.write_bytes(b"fake")
        entries += f"- number: {number}\n  video: {video}\n"
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nmode: single_episode\n"
        f"target_seconds: 240\nepisodes:\n{entries}",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)
    paths = Paths(cfg.root)
    for number in numbers:
        track = DialogueTrack(
            episode=number,
            source="asr",
            duration=100.0,
            lines=[DialogueLine(idx=1, start=1.0, end=2.0, text="はい", raw="はい")],
        )
        _file(paths.dialogue(number), track.model_dump_json(indent=2))
    return cfg, paths


def _spy_translate_and_script(monkeypatch) -> list[str]:
    """把 translate/script 本体换成只记顺序的替身，返回那份共享的顺序清单。

    script 的替身记 start/done 两笔、中间让出一次事件循环：少了这个让出点，替身会在
    create_task 之后的第一次调度里一口气跑完，预取过的窗口跟没预取的长得一模一样，
    这些用例就分辨不出窗口宽度了（实测：不让出时，把窗口钳到 1 的变异照样绿）。
    """
    order: list[str] = []

    async def spy_translate(cfg, provider, episode):
        order.append(f"translate{episode}")
        return TranslatedTrack(episode=episode, lines=[])

    async def spy_script(cfg, provider, episode, *, reporter=None):
        order.append(f"script{episode}-start")
        await asyncio.sleep(0.05)
        order.append(f"script{episode}-done")
        return None, []

    monkeypatch.setattr("tenmin.pipeline.run_translate", spy_translate)
    monkeypatch.setattr("tenmin.pipeline.run_script", spy_script)
    return order


@pytest.mark.asyncio
async def test_run_pipeline_never_starts_a_script_before_its_own_translate(
    tmp_path, monkeypatch
):
    """累积术语表是 script 的输入，而写它的是**纵向循环里**的 translate。

    所以预取窗口一旦跨过当前这一集，后面几集的 script 就会在自己的 translate 之前既
    判新鲜度又执行 —— 可能被判 fresh 而跳过（解说稿用旧译名），也可能 load_glossary
    读到还缺本集术语的表。非确定性，两种都不该发生。
    """
    cfg, _ = _project_with_asr_dialogues(tmp_path, (1, 2, 3))
    cfg.llm.script_concurrency = 3
    order = _spy_translate_and_script(monkeypatch)

    await run_pipeline(cfg, FakeProvider([]), only=["translate", "script"])

    assert order == [
        item
        for number in (1, 2, 3)
        for item in (
            f"translate{number}",
            f"script{number}-start",
            f"script{number}-done",
        )
    ]


@pytest.mark.asyncio
async def test_run_pipeline_still_prefetches_scripts_without_translate(
    tmp_path, monkeypatch
):
    """钳窗口只许发生在 translate 真的在这次运行里的时候。

    预取窗口存在的理由是 script 最慢也最容易失败（单次调用实测 561 秒），
    `--only script` / `--from script` 这类不含 translate 的运行必须照旧预取。
    """
    cfg, _ = _project_with_asr_dialogues(tmp_path, (1, 2, 3))
    cfg.llm.script_concurrency = 3
    order = _spy_translate_and_script(monkeypatch)

    await run_pipeline(cfg, FakeProvider([]), only=["script"])

    # 三集的 script 全在第一集跑完之前就起了。只钉「三笔 start 排在最前」，不钉 done 的
    # 先后：那取决于三个同长 sleep 的醒来顺序，不是这条用例要保护的性质。
    assert order[:3] == ["script1-start", "script2-start", "script3-start"]
    assert sorted(order[3:]) == ["script1-done", "script2-done", "script3-done"]


# --- LLM 用量落盘（只用来观察成本，不是任何阶段的新鲜度输入） ---


@pytest.mark.asyncio
async def test_run_script_writes_a_usage_file(project):
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)
    data = json.loads(Paths(project.root).script_usage(2).read_text(encoding="utf-8"))
    assert data["episode"] == 2
    assert [call["round"] for call in data["calls"]] == ["draft"]
    call = data["calls"][0]
    assert call["provider"] == "gemini"
    assert call["prompt_tokens"] is None
    assert call["completion_tokens"] is None
    assert call["cached_tokens"] is None
    assert call["ok"] is True


@pytest.mark.asyncio
async def test_run_script_writes_the_usage_file_even_when_it_fails(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])
    with pytest.raises(LLMSchemaError):
        await run_script(project, _SchemaBlowupProvider("x"), episode=2)
    data = json.loads(Paths(project.root).script_usage(2).read_text(encoding="utf-8"))
    assert [call["ok"] for call in data["calls"]] == [False]


@pytest.mark.asyncio
async def test_run_translate_writes_a_usage_file(tmp_path):
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    await run_translate(cfg, FakeProvider([_translation_response()]), 11)
    data = json.loads(paths.zh_usage(11).read_text(encoding="utf-8"))
    assert data["episode"] == 11
    assert [call["round"] for call in data["calls"]] == ["translate"]


@pytest.mark.asyncio
async def test_a_skipped_translate_writes_no_usage_file(tmp_path):
    cfg, paths = _project_with_dialogue(tmp_path, episode=2, source="srt")
    await run_translate(cfg, FakeProvider([]), 2)
    assert not paths.zh_usage(2).exists()
    assert not (cfg.root / "zh").exists()


# --- 预填集条目：只有 op/ed、还没有 srt/video --------------------------------


@pytest.mark.asyncio
async def test_batch_mode_skips_a_prefilled_episode_with_a_notice(project):
    project.episodes.append(EpisodeConfig(number=3, ed_range=(1300.0, 1420.0)))

    warnings = await run_pipeline(
        project, FakeProvider([fake_script_response()]), only=V1_STAGES
    )

    paths = Paths(project.root)
    assert "第 3 集还没有 video，已跳过" in warnings
    assert paths.script(2).exists()
    assert not paths.dialogue(3).exists()


@pytest.mark.asyncio
async def test_single_episode_mode_on_a_prefilled_episode_asks_for_a_video(project):
    project.episodes.append(EpisodeConfig(number=3, ed_range=(1300.0, 1420.0)))

    with pytest.raises(ValueError, match="--video"):
        await run_pipeline(project, FakeProvider([]), only=V1_STAGES, episode=3)


def test_registering_a_prefilled_episode_keeps_its_credit_ranges(tmp_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nepisodes:\n"
        "- number: 3\n  op_range: [10.0, 100.0]\n  ed_range: [1300.0, 1420.0]\n",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)
    video = tmp_path / "e03.mkv"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=3, srt=None, video=video)

    reloaded = load_project(yaml_path)
    (episode,) = reloaded.episodes
    assert episode.op_range == (10.0, 100.0)
    assert episode.ed_range == (1300.0, 1420.0)
    assert episode.video == video.resolve()
    assert episode.has_source is True


# --- 登记写回 project.yaml：只动 episodes 里对应的那一条 --------------------

_COMMENTED_YAML = """\
# 我的番，这行注释要原样留着
show: "才女的侍从"   # 行尾注释，引号风格也得留着
slug: saijo
render:
  font_size: 60  # 字号
episodes:
  - number: 2
    srt: srt/E02.srt
    op_range: [153.5, 224.7]
  # 第 3 集先把片尾填上，视频还没下好
  - number: 3
    ed_range: [1300, 1420]
glossary:
  伊月: 伊月
"""


def _removed_lines(before: str, after: str) -> list[str]:
    return [
        line
        for line in difflib.ndiff(before.splitlines(), after.splitlines())
        if line.startswith("- ")
    ]


def test_register_episode_only_adds_lines_to_a_commented_yaml(tmp_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(_COMMENTED_YAML, encoding="utf-8")
    cfg = load_project(yaml_path)
    video3 = tmp_path / "e03.mkv"
    video3.write_bytes(b"fake")
    video4 = tmp_path / "e04.mkv"
    video4.write_bytes(b"fake")

    register_episode(cfg, episode=3, srt=None, video=video3)
    register_episode(cfg, episode=4, srt=None, video=video4)

    after = yaml_path.read_text(encoding="utf-8")
    assert _removed_lines(_COMMENTED_YAML, after) == []
    assert f"    video: {video3.resolve()}" in after.splitlines()
    reloaded = load_project(yaml_path)
    assert [e.number for e in reloaded.episodes] == [2, 3, 4]
    episode3 = next(e for e in reloaded.episodes if e.number == 3)
    assert episode3.ed_range == (1300.0, 1420.0)
    assert episode3.video == video3.resolve()


def test_reregistering_changes_only_that_entrys_source_lines(tmp_path, golden_srt_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(_COMMENTED_YAML, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e02.mkv"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=2, srt=None, video=video)

    after = yaml_path.read_text(encoding="utf-8")
    assert _removed_lines(_COMMENTED_YAML, after) == ["-     srt: srt/E02.srt"]
    assert "    op_range: [153.5, 224.7]" in after.splitlines()


def test_register_episode_preserves_four_space_mapping_indent(tmp_path):
    root = tmp_path / "saijo"
    root.mkdir()
    yaml_path = root / "project.yaml"
    before = """\
show: 才女的侍从
slug: saijo
render:
    font_size: 60  # 字号
    width: 1920
episodes:
    - number: 2
      ed_range: [1300, 1420]
glossary:
    伊月: 伊月
"""
    yaml_path.write_text(before, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e02.mkv"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=2, srt=None, video=video)

    after = yaml_path.read_text(encoding="utf-8")
    assert _removed_lines(before, after) == []
    assert "      video: " + str(video.resolve()) in after.splitlines()
    assert load_project(yaml_path).episodes[0].ed_range == (1300.0, 1420.0)


def test_register_episode_preserves_mapping_indent_after_inline_comment(tmp_path):
    root = tmp_path / "saijo"
    root.mkdir()
    yaml_path = root / "project.yaml"
    before = """\
show: 才女的侍从
slug: saijo
render: # settings
    font_size: 60
episodes:
- number: 2
  ed_range: [1300, 1420]
"""
    yaml_path.write_text(before, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e02.mkv"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=2, srt=None, video=video)

    assert yaml_path.read_text(encoding="utf-8") == before.replace(
        "  ed_range: [1300, 1420]\n",
        f"  ed_range: [1300, 1420]\n  video: {video.resolve()}\n",
    )


def test_register_episode_uses_block_mapping_indent_after_flow_map(tmp_path):
    root = tmp_path / "saijo"
    root.mkdir()
    yaml_path = root / "project.yaml"
    before = """\
show: 才女的侍从
slug: saijo
glossary: {伊月: 伊月}
render: # settings
    font_size: 60
episodes:
- number: 2
  ed_range: [1300, 1420]
"""
    yaml_path.write_text(before, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e02.mkv"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=2, srt=None, video=video)

    assert yaml_path.read_text(encoding="utf-8") == before.replace(
        "  ed_range: [1300, 1420]\n",
        f"  ed_range: [1300, 1420]\n  video: {video.resolve()}\n",
    )


def test_reregister_episode_preserves_nonempty_flow_sequence(tmp_path):
    root = tmp_path / "saijo"
    root.mkdir()
    yaml_path = root / "project.yaml"
    before = """\
show: 才女的侍从
slug: saijo
episodes: [{number: 2, srt: srt/E02.srt}, {number: 3, ed_range: [1300, 1420]}]
glossary: {伊月: 伊月}
"""
    yaml_path.write_text(before, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e02.mkv"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=2, srt=None, video=video)

    after = yaml_path.read_text(encoding="utf-8")
    assert after == before.replace(
        "{number: 2, srt: srt/E02.srt}",
        f"{{number: 2, video: {video.resolve()}}}",
    )
    assert load_project(yaml_path).episodes[1].ed_range == (1300.0, 1420.0)


def test_the_write_back_parser_rejects_duplicate_keys_like_load_project(tmp_path):
    """两处对重复键的态度必须一致。"""
    from ruamel.yaml import YAML
    from ruamel.yaml.constructor import DuplicateKeyError as RuamelDuplicateKeyError

    text = "show: 某番\nslug: demo\nrender:\n  crf: '18'\nrender:\n  crf: '20'\n"
    path = tmp_path / "project.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ProjectConfigError):
        load_project(path)
    with pytest.raises(RuamelDuplicateKeyError):
        YAML().load(text)


# --- 全局阶段：内容不变不写 + 按「上次跑完」判新鲜度 -------------------------


def test_stamped_freshness_falls_back_to_the_outputs_without_a_stamp(tmp_path):
    """升级前的项目没有戳子：退回按产物判，不因为缺戳子白跑一遍。"""
    src = _file(tmp_path / "in.txt")
    out = _file(tmp_path / "out.txt")
    stamp = tmp_path / "stage.done"
    assert _is_fresh_stamped([out], [src], stamp) is True
    _shift_mtime(src, 10.0)
    assert _is_fresh_stamped([out], [src], stamp) is False


def test_stamped_freshness_judges_by_the_stamp_when_present(tmp_path):
    """产物内容没变就没重写、mtime 停在旧值，但这个阶段确实刚跑过。"""
    src = _file(tmp_path / "in.txt")
    out = _file(tmp_path / "out.txt")
    stamp = tmp_path / "stage.done"
    _shift_mtime(src, 10.0)
    config_slices.touch_stamp(stamp, [out])
    _shift_mtime(stamp, 20.0)
    assert _is_fresh_stamped([out], [src], stamp) is True
    _shift_mtime(src, 20.0)
    assert _is_fresh_stamped([out], [src], stamp) is False


def test_stamped_freshness_still_requires_every_output(tmp_path):
    src = _file(tmp_path / "in.txt")
    stamp = tmp_path / "stage.done"
    config_slices.touch_stamp(stamp, [src])
    _shift_mtime(stamp, 20.0)
    assert _is_fresh_stamped([tmp_path / "gone.json"], [src], stamp) is False
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    assert _is_fresh_stamped([empty], [src], stamp) is False


@pytest.mark.asyncio
async def test_rerunning_ingest_and_signals_on_unchanged_input_keeps_their_mtimes(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])
    paths = Paths(project.root)
    before = (paths.dialogue(2).stat().st_mtime_ns, paths.signals(2).stat().st_mtime_ns)

    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"], force=True)

    after = (paths.dialogue(2).stat().st_mtime_ns, paths.signals(2).stat().st_mtime_ns)
    assert after == before


@pytest.mark.asyncio
async def test_a_no_op_ingest_config_change_invalidates_nothing_downstream(project):
    """没有手填区间时 manual_window_margin 不参与任何判定：ingest 该重跑（它的切片变了），
    但产出一字不变，下游一个都不许被连带判过期；而且下一次运行 ingest 自己也得是新鲜的
    （不能因为产物 mtime 停在旧值就每次都重跑）。"""
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)

    project.credits.manual_window_margin = 6.0
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=V1_STAGES, reporter=reporter)
    assert ("stage_start", "ingest") in reporter.calls
    for stage in ("signals", "script", "docgen"):
        assert ("stage_skip", stage) in reporter.calls, stage

    again = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=V1_STAGES, reporter=again)
    for stage in V1_STAGES:
        assert ("stage_skip", stage) in again.calls, stage


@pytest.mark.asyncio
async def test_run_pipeline_writes_the_global_stage_stamps(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])
    assert (project.root / ".config" / "ingest.done").is_file()
    assert (project.root / ".config" / "signals.done").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["ingest", "signals"])
async def test_failed_forced_global_stage_invalidates_its_stamp(project, monkeypatch, stage):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])
    stamp = project.root / ".config" / f"{stage}.done"
    assert stamp.is_file()

    def interrupted(cfg):
        # A stage can write some outputs before a later episode fails.
        output = getattr(Paths(cfg.root), "dialogue" if stage == "ingest" else "signals")(2)
        _file(output, "bad")
        raise RuntimeError("stage interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(f"tenmin.pipeline.run_{stage}", interrupted)
        with pytest.raises(RuntimeError, match="stage interrupted"):
            await run_pipeline(project, FakeProvider([]), only=[stage], force=True)

    assert stamp.read_bytes() == b""  # incomplete marker, not a completed stamp
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=[stage], reporter=reporter)
    assert ("stage_start", stage) in reporter.calls
    assert stamp.is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["ingest", "signals"])
async def test_stamped_stage_reruns_when_output_is_replaced_with_same_size_and_mtime(
    project, stage
):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])
    output = getattr(Paths(project.root), "dialogue" if stage == "ingest" else "signals")(2)
    before = output.stat()
    original = output.read_bytes()
    # Restoring an old backup can preserve both size and mtime: metadata alone is insufficient.
    output.write_bytes(b"!" + original[1:])
    os.utime(output, ns=(before.st_atime_ns, before.st_mtime_ns))

    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=[stage], reporter=reporter)

    assert ("stage_start", stage) in reporter.calls
    assert output.read_bytes() == original


def test_legacy_text_stamp_falls_back_to_output_mtime(tmp_path):
    src = _file(tmp_path / "in.txt")
    out = _file(tmp_path / "out.txt")
    stamp = _file(tmp_path / "stage.done", "stage\n")
    _shift_mtime(src, 10.0)
    _shift_mtime(stamp, 20.0)
    assert _is_fresh_stamped([out], [src], stamp) is False


def test_invalid_nonlegacy_stamp_is_not_treated_as_completed(tmp_path):
    src = _file(tmp_path / "in.txt")
    out = _file(tmp_path / "out.txt")
    stamp = _file(tmp_path / "stage.done", '{"partial":')
    _shift_mtime(stamp, 20.0)
    assert _is_fresh_stamped([out], [src], stamp) is False


# --- 端到端：每次改动只有预期的阶段重跑 ---------------------------------------


def _full_run_project(project, monkeypatch) -> None:
    """让 voice → render 在测试里跑得起来：空壳视频、假 ffprobe、假 ffmpeg。"""
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 80.0)
    _stub_mix_audio_ffmpeg(monkeypatch)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)


async def _first_full_run(project) -> None:
    # 80 秒一个 chunk：这份 fixture 的旁白真实会有的长度（理由见
    # test_run_pipeline_batch_mode_runs_full_pipeline_for_all_episodes）。
    await run_pipeline(
        project, FakeProvider([fake_script_response()]), tts_engine=FakeTTSEngine([80.0] * 20)
    )


async def _stages_rerun(project) -> set[str]:
    """再跑一遍全流程，返回真正重跑了的阶段。

    FakeProvider([]) 与 FakeTTSEngine([])：谁要是真去调 LLM 或合成语音，当场
    AssertionError。translate 刻意扣掉：繁中片源上它的产物永远不存在，所以每次都「开跑」
    然后在 run_translate 里零动作早退（见 run_translate 的 docstring），它出现在这里不代表
    任何配置被判过期。
    """
    reporter = FakeReporter()
    await run_pipeline(
        project, FakeProvider([]), tts_engine=FakeTTSEngine([]), reporter=reporter
    )
    started = {call[1] for call in reporter.calls if call[0] == "stage_start"}
    return started - {"translate"}


@pytest.mark.asyncio
async def test_an_untouched_project_reruns_nothing(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)
    assert await _stages_rerun(project) == set()


@pytest.mark.asyncio
async def test_editing_dialogue_invalidates_timeline_but_not_llm_or_tts(project, monkeypatch):
    """对白轨是 timeline 留白核验的输入，改它不该连带判 script/voice 过期。"""
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)
    path = Paths(project.root).dialogue(2)
    _shift_mtime(path, 100)
    reporter = FakeReporter()
    await run_pipeline(
        project,
        FakeProvider([]),
        only=["timeline", "audio", "render"],
        tts_engine=FakeTTSEngine([]),
        reporter=reporter,
    )
    starts = {call[1] for call in reporter.calls if call[0] == "stage_start"}
    assert starts == {"timeline", "audio", "render"}
    assert all(call[1] != "voice" for call in reporter.calls)


@pytest.mark.asyncio
async def test_changing_crf_reruns_only_render(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)

    project.render.crf = "18"
    render_commands: list[list[str]] = []

    def capture_render(args, **kwargs):
        render_commands.append(list(args))
        return _touch_output_with_progress(args, **kwargs)

    monkeypatch.setattr("tenmin.render.video.run_with_progress", capture_render)

    assert await _stages_rerun(project) == {"render"}
    assert len(render_commands) == 1
    assert render_commands[0][render_commands[0].index("-crf") + 1] == "18"


@pytest.mark.asyncio
async def test_changing_font_size_reruns_only_the_subtitle_consumers(project, monkeypatch):
    """当前 mtime 契约：字号让 timeline 重写 .ass 和 timeline.json，audio 读更新后的
    timeline.json 所以跟着重跑，render 最后重跑；LLM / TTS 都不重跑。"""
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)

    project.render.font_size = 60

    assert await _stages_rerun(project) == {"timeline", "audio", "render"}
    ass = Paths(project.root).subtitles(2).read_text(encoding="utf-8")
    assert f"Style: Narration,{project.render.subtitle_font_name},60," in ass


@pytest.mark.asyncio
async def test_operational_knobs_rerun_nothing_end_to_end(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)

    project.llm.timeout_seconds = 5.0
    project.llm.transport_max_attempts = 2
    project.render.tts_concurrency = 8
    project.render.ffprobe_path = "/opt/custom/bin/ffprobe"

    assert await _stages_rerun(project) == set()


@pytest.mark.asyncio
async def test_subtitle_limit_reruns_timeline_audio_render_only(project, monkeypatch):
    """字幕软上限只是 timeline 的输入；audio/render 跟着重跑是因为它们的产物
    时间戳落在 timeline 之后，不是因为它们自己的配置切片变了。"""
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)

    project.render.subtitle_soft_max_chars = 30

    assert await _stages_rerun(project) == {"timeline", "audio", "render"}


@pytest.mark.asyncio
async def test_loudness_change_reruns_audio_render_only(project, monkeypatch):
    """响度归一目标只进 audio 的配置切片；timeline 的产物内容与它无关，不该
    被连带重跑（不同于上一个字幕上限的例子）。"""
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)

    project.render.loudness_i = -16.0

    assert await _stages_rerun(project) == {"audio", "render"}


@pytest.mark.asyncio
async def test_clip_gap_change_reruns_script_and_dependent_stages(project, monkeypatch):
    """留白/clip 间隙的合法阈值住在 validate_script，是 script 阶段读的旋钮；
    改它必须让 script 真重跑一遍语义校验，并带动下游全部阶段。"""
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)

    project.validate_script.hold_clip_max_gap_seconds = 4.0

    reporter = FakeReporter()
    await run_pipeline(
        project,
        FakeProvider([fake_script_response()]),
        tts_engine=FakeTTSEngine([80.0] * 20),
        reporter=reporter,
    )
    started = {call[1] for call in reporter.calls if call[0] == "stage_start"}
    started -= {"translate"}
    assert started == {"script", "docgen", "voice", "timeline", "audio", "render"}


@pytest.mark.asyncio
async def test_batch_audio_warnings_do_not_leak_across_episodes(
    project, golden_srt_path, monkeypatch
):
    """每一集自己的混音警告只应该出现一次，且不会被记到别的集头上——mix_audio
    自己的 notices 是每次调用的局部变量，这里从 run_pipeline 的公开接口验证
    这条不变量真的成立，而不是只看 mix_audio 内部实现。"""
    second_srt = project.root / "srt" / "E01.srt"
    second_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    project.episodes.append(EpisodeConfig(number=1, srt=Path("srt/E01.srt")))

    for episode_cfg in project.episodes:
        video_name = f"E{episode_cfg.number:02d}.mkv"
        (project.root / video_name).write_bytes(b"")
        episode_cfg.video = Path(video_name)

    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    def fake_mix_audio(*, timeline, out_path, warnings=None, **_):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00")
        if warnings is not None:
            warnings.append(f"E{timeline.episode:02d}：留白原声近乎无声")
        return out_path

    monkeypatch.setattr("tenmin.pipeline.mix_audio", fake_mix_audio)

    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    tts_engine = FakeTTSEngine([80.0] * 20)

    warnings = await run_pipeline(project, provider, tts_engine=tts_engine)

    audio_notices = [w for w in warnings if "留白原声近乎无声" in w]
    assert audio_notices == ["E02：留白原声近乎无声", "E01：留白原声近乎无声"]


@pytest.mark.asyncio
async def test_registering_a_new_episode_leaves_the_existing_one_untouched(
    project, golden_srt_path, tmp_path, monkeypatch
):
    """登记第 1 集：ingest / signals 会把全部集重跑一遍，但第 2 集的产物不能
    重写（即使字节或 mtime 恰巧没变），它的 LLM 一次都不许再调。"""
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)
    paths = Paths(project.root)
    watched = [
        paths.dialogue(2),
        paths.signals(2),
        paths.script(2),
        paths.table(2),
        paths.narration(2),
    ]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in watched]
    slice_dir = project.root / ".config"
    slices_before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in slice_dir.glob("E02.*.json")
    }

    video = tmp_path / "e01.mkv"
    video.write_bytes(b"fake")
    # 新集的源片是个假文件：别让 ingest 去真跑 ffprobe。
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1500.0)
    register_episode(project, episode=1, srt=golden_srt_path, video=video)

    # 观察真正的原子写入口，而不是只比文件元数据：同字节/同 mtime 的覆盖也会被抓到。
    # 其他集的正常写盘照旧执行；_write_json 和 write_text_if_changed 最终都走这里。
    from tenmin import atomic

    original_write = atomic.write_text

    def record_write(path, text, *, encoding="utf-8"):
        if Path(path) in watched or (
            Path(path).parent == slice_dir and Path(path).name.startswith("E02.")
        ):
            pytest.fail(f"登记 E01 时重写了 E02 产物：{path}")
        return original_write(path, text, encoding=encoding)

    monkeypatch.setattr(atomic, "write_text", record_write)
    provider = FakeProvider([fake_script_response(episode=1)])
    await run_pipeline(project, provider, only=V1_STAGES)

    assert len(provider.calls) == 1
    assert "集数：第 1 集" in provider.calls[0]["user"]
    assert paths.script(1).exists()
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in watched] == before
    slices_after = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in slice_dir.glob("E02.*.json")
    }
    assert slices_after == slices_before
