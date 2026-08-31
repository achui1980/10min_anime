import json

import pytest

from tenmin.config import ProjectConfig
from tenmin.models import LLMBeat, LLMClip, LLMScript, Script
from tenmin.pipeline import (
    STAGES,
    Paths,
    run_docgen,
    run_ingest,
    run_pipeline,
    run_signals,
    stages_from,
)

from .fakes import FakeProvider


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


def fake_script_response():
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
                        episode=2,
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
    assert STAGES == ["ingest", "signals", "script", "docgen"]


def test_stages_from_middle():
    assert stages_from("signals") == ["signals", "script", "docgen"]


def test_stages_from_unknown_raises():
    with pytest.raises(ValueError):
        stages_from("nope")


def test_paths_layout(tmp_path):
    paths = Paths(tmp_path / "saijo")
    assert paths.dialogue(2).name == "E02.dialogue.json"
    assert paths.dialogue(2).parent.name == "01_dialogue"
    assert paths.signals(2).name == "E02.signals.json"
    assert paths.signals(2).parent.name == "02_signals"
    assert paths.script.name == "script.json"
    assert paths.script.parent.name == "03_script"
    assert paths.table.name == "解说方案.md"
    assert paths.narration.name == "narration.txt"
    assert paths.table.parent.name == "out"


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


def test_run_signals_writes_signals_json(project):
    run_ingest(project)
    reports = run_signals(project)
    # Task 10 实测：本集有 26 个 ≥3s 的无字幕间隙（候选池，不是高光集合）。
    # tests/test_aggregate.py 对同一黄金样本断言的也是 26。
    assert len(reports[0].silent_gaps) == 26
    assert Paths(project.root).signals(2).exists()


def test_run_signals_without_ingest_raises(project):
    with pytest.raises(FileNotFoundError):
        run_signals(project)


@pytest.mark.asyncio
async def test_run_pipeline_end_to_end(project):
    provider = FakeProvider([fake_script_response()])
    await run_pipeline(project, provider)
    paths = Paths(project.root)
    assert paths.script.exists()
    assert paths.table.exists()
    assert paths.narration.exists()
    assert "解说方案" in paths.table.read_text(encoding="utf-8")
    assert paths.narration.read_text(encoding="utf-8").startswith("啊")


@pytest.mark.asyncio
async def test_run_pipeline_skips_when_fresh(project):
    provider = FakeProvider([fake_script_response()])
    await run_pipeline(project, provider)
    # 第二次跑：provider 没有剩余响应，若真的再调 LLM 就会 AssertionError
    await run_pipeline(project, provider)
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_run_pipeline_force_reruns_llm(project):
    provider = FakeProvider([fake_script_response(), fake_script_response()])
    await run_pipeline(project, provider)
    await run_pipeline(project, provider, force=True)
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_run_pipeline_only_docgen_reuses_edited_script(project):
    provider = FakeProvider([fake_script_response()])
    await run_pipeline(project, provider)
    paths = Paths(project.root)

    script = Script.model_validate_json(paths.script.read_text(encoding="utf-8"))
    script.beats[0].narration = "人工改过的开场"
    paths.script.write_text(
        script.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
    )

    await run_pipeline(project, provider, only=["docgen"], force=True)
    assert "人工改过的开场" in paths.narration.read_text(encoding="utf-8")
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
    assert not paths.script.exists()
    assert provider.calls == []


@pytest.mark.asyncio
async def test_run_pipeline_season_mode_raises(project):
    project.mode = "season"
    with pytest.raises(NotImplementedError) as exc:
        await run_pipeline(project, FakeProvider([]))
    assert "整季模式尚未实现" in str(exc.value)


@pytest.mark.asyncio
async def test_run_pipeline_from_signals_keeps_dialogue(project):
    provider = FakeProvider([fake_script_response(), fake_script_response()])
    await run_pipeline(project, provider)
    dialogue = Paths(project.root).dialogue(2)
    before = dialogue.stat().st_mtime_ns
    await run_pipeline(project, provider, from_stage="signals", force=True)
    assert dialogue.stat().st_mtime_ns == before
    assert len(provider.calls) == 2


def test_run_docgen_without_script_raises(project):
    with pytest.raises(FileNotFoundError):
        run_docgen(project)
