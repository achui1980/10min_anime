"""端到端：SRT -> 对照表 + 配音文本。不联网。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from tenmin.config import load_project
from tenmin.models import Script
from tenmin.pipeline import Paths, run_pipeline
from tests.fakes import FakeProvider

OCR_GARBAGE = "80-08"
OCR_GARBAGE_CITY = "浙谷"

# STAGES 在 v2 里扩到 8 个，voice 之后的阶段需要 TTS engine 与源视频。
# 本文件只验证 v1 的「SRT -> 对照表 + 配音文本」链路，显式限定阶段范围。
V1_STAGES = ["ingest", "signals", "script", "docgen"]


def build_llm_payload(highlight_starts: list[float]) -> dict:
    """构造一份合法的 LLMScript JSON，clip 落在真实无字幕高光上。

    旁白字数刻意凑到时长预算内：
    (13 + 300) + (11 + 500) + (7 + 224) = 1055 字，1055 / 4.5 = 234.44s，
    加上 3.0 + 2.5 = 5.5s 留白 = 239.94s，与 target 240s 偏差 0.02%，
    因此 needs_rewrite 为假，LLM 只会被调用一次。改动字数会破坏
    test_offline_pipeline_calls_llm_exactly_once。
    """
    a, b, c = highlight_starts[0], highlight_starts[1], highlight_starts[2]
    return {
        "beats": [
            {
                "id": "b1",
                "label": "Hook 开场",
                "role": "hook",
                "narration": "开场一句话把人钉在屏幕前，" + "钩" * 300,
                "clips": [
                    {
                        "episode": 1,
                        "start": a,
                        "end": a + 4.0,
                        "visual": "抱倒 ➔ 愣住",
                        "anchor_lines": [],
                    },
                    {
                        "episode": 1,
                        "start": b,
                        "end": b + 3.0,
                        "visual": "浴室蒸气",
                        "anchor_lines": [],
                    },
                ],
                "original_audio": "duck",
                "holds": [{"at": 5.0, "duration": 3.0, "quote": "两个叛徒", "note": "留给原声"}],
                "sfx": [{"at": 1.0, "cue": "impact", "note": "砸下来"}],
            },
            {
                "id": "b2",
                "label": "阶段一：入职即地狱",
                "role": "act",
                "narration": "第一天就被丢进修罗场，" + "叙" * 500,
                "clips": [
                    {
                        "episode": 1,
                        "start": c,
                        "end": c + 6.0,
                        "visual": "学院全景 ➔ 走廊",
                        "anchor_lines": [],
                    }
                ],
                "original_audio": "duck",
                "holds": [{"at": 10.0, "duration": 2.5, "quote": "给我吃", "note": "投喂"}],
                "sfx": [],
            },
            {
                "id": "b3",
                "label": "收尾：修罗场引爆",
                "role": "outro",
                "narration": "而这只是开始，" + "尾" * 224,
                "clips": [
                    {
                        "episode": 1,
                        "start": a + 10.0,
                        "end": a + 14.0,
                        "visual": "定格",
                        "anchor_lines": [],
                    }
                ],
                "original_audio": "duck",
                "holds": [],
                "sfx": [],
            },
        ]
    }


@pytest.fixture
def project(tmp_path: Path, golden_srt_path: Path) -> Path:
    root = tmp_path / "work" / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "E02.srt").write_bytes(golden_srt_path.read_bytes())
    config = {
        "show": "才女的侍从",
        "slug": "saijo",
        "mode": "single_episode",
        "target_seconds": 240,
        "locale": {"convert_traditional": True},
        "episodes": [{"number": 1, "srt": "srt/E02.srt"}],
        "glossary": {},
        "llm": {"provider": "gemini", "model": "gemini-3.6-flash"},
    }
    (root / "project.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
    )
    return root


async def run_offline(root: Path) -> tuple[list[str], FakeProvider]:
    """先跑 ingest+signals 拿真实高光，再用它构造 LLM 回包跑完剩下两段。"""
    cfg = load_project(root / "project.yaml")
    dummy = FakeProvider([])
    await run_pipeline(cfg, dummy, only=["ingest", "signals"])

    report = json.loads(Paths(root).signals(1).read_text(encoding="utf-8"))
    starts = [
        gap["start"]
        for gap in sorted(report["silent_gaps"], key=lambda g: -(g["end"] - g["start"]))
    ]
    assert len(starts) >= 3, starts

    provider = FakeProvider([build_llm_payload(starts[:3])])
    warnings = await run_pipeline(cfg, provider, only=V1_STAGES[2:])
    return warnings, provider


async def test_offline_pipeline_produces_both_artifacts(project: Path):
    await run_offline(project)
    paths = Paths(project)
    assert paths.table(1).exists()
    assert paths.narration(1).exists()
    assert paths.script(1).exists()
    assert paths.dialogue(1).exists()
    assert paths.signals(1).exists()


async def test_offline_pipeline_calls_llm_exactly_once(project: Path):
    _, provider = await run_offline(project)
    assert len(provider.calls) == 1


async def test_table_has_five_columns_and_three_beats(project: Path):
    await run_offline(project)
    text = Paths(project).table(1).read_text(encoding="utf-8")
    assert "| 节点 | 原片截取时间戳 | 建议画面特征 | 分段解说文案 | 剪辑与原声处理 |" in text
    body = [ln for ln in text.splitlines() if ln.startswith("| ") and "---" not in ln]
    assert len(body) == 4, body  # 表头 + 3 个节点


async def test_table_marks_silent_highlights(project: Path):
    await run_offline(project)
    text = Paths(project).table(1).read_text(encoding="utf-8")
    assert "★" in text
    assert "★ = 该片段命中无字幕演出高光区间，纯字幕方案取不到" in text


async def test_table_records_holds_and_audio_direction(project: Path):
    await run_offline(project)
    text = Paths(project).table(1).read_text(encoding="utf-8")
    assert "原声压低垫底" in text
    assert "留白 3.0s：「两个叛徒」" in text
    assert "音效 impact @1.0s" in text


async def test_narration_is_plain_text(project: Path):
    await run_offline(project)
    text = Paths(project).narration(1).read_text(encoding="utf-8")
    for marker in ("★", "|", "#", "<br>", "节点", "留白", "音效"):
        assert marker not in text, marker
    assert text.count("\n\n") == 2


async def test_script_json_is_reloadable(project: Path):
    """script.json 是唯一人工编辑面，必须能原样读回。"""
    await run_offline(project)
    script = Script.model_validate_json(Paths(project).script(1).read_text(encoding="utf-8"))
    assert len(script.beats) == 3
    assert script.est_total_seconds > 0
    assert all(beat.clips for beat in script.beats)


async def test_docgen_rerun_does_not_call_llm(project: Path):
    """spec 要求 docgen 纯派生：重跑不得触发 LLM。"""
    await run_offline(project)
    cfg = load_project(project / "project.yaml")
    provider = FakeProvider([])
    await run_pipeline(cfg, provider, only=["docgen"], force=True)
    assert provider.calls == []


async def test_editing_script_json_changes_output(project: Path):
    """人工编辑 script.json 后重跑 docgen，产物随之变化。"""
    await run_offline(project)
    paths = Paths(project)
    script = Script.model_validate_json(paths.script(1).read_text(encoding="utf-8"))
    script.beats[0].narration = "人工改写过的开场"
    paths.script(1).write_text(script.model_dump_json(indent=2), encoding="utf-8")

    cfg = load_project(project / "project.yaml")
    await run_pipeline(cfg, FakeProvider([]), only=["docgen"], force=True)
    assert "人工改写过的开场" in paths.narration(1).read_text(encoding="utf-8")


async def test_ocr_garbage_never_reaches_output(project: Path):
    """回归护栏：第 41 行的车牌 OCR 噪声不得出现在任何交付物里。"""
    await run_offline(project)
    paths = Paths(project)
    for path in (paths.table(1), paths.narration(1)):
        text = path.read_text(encoding="utf-8")
        assert OCR_GARBAGE not in text, path
        assert OCR_GARBAGE_CITY not in text, path


async def test_credits_never_reach_output(project: Path):
    """OP/ED staff 名单不得进旁白。"""
    await run_offline(project)
    text = Paths(project).narration(1).read_text(encoding="utf-8")
    for name in ("J.C.STAFF", "制作委员会", "Synergy"):
        assert name not in text, name


async def test_estimated_duration_within_tolerance(project: Path):
    """build_llm_payload 的字数是照 4.5 字/秒凑到 240s 的，必须落在 ±12% 内。"""
    await run_offline(project)
    script = Script.model_validate_json(Paths(project).script(1).read_text(encoding="utf-8"))
    deviation = abs(script.est_total_seconds - script.target_seconds) / script.target_seconds
    assert deviation <= 0.12, script.est_total_seconds
    assert script.est_total_seconds == pytest.approx(239.94, abs=0.1)


async def test_second_run_skips_completed_stages(project: Path):
    """mtime 跳过：第二次全量跑不应重新调 LLM。"""
    await run_offline(project)
    cfg = load_project(project / "project.yaml")
    provider = FakeProvider([])
    await run_pipeline(cfg, provider, only=V1_STAGES)
    assert provider.calls == []


async def test_season_mode_is_rejected(project: Path):
    raw = yaml.safe_load((project / "project.yaml").read_text(encoding="utf-8"))
    raw["mode"] = "season"
    (project / "project.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )
    cfg = load_project(project / "project.yaml")
    with pytest.raises(NotImplementedError, match="整季模式尚未实现"):
        await run_pipeline(cfg, FakeProvider([]))


# --- 真 LLM 快照，默认跳过 ---

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


@pytest.mark.llm
@pytest.mark.skipif(
    not os.environ.get("TENMIN_GEMINI_API_KEY"), reason="需要 TENMIN_GEMINI_API_KEY"
)
async def test_real_llm_snapshot(project: Path):
    """真调 LLM 跑黄金样本，产物写进 tests/snapshots/ 供人工 diff。

    本测试不断言文案内容（LLM 输出不确定），只断言结构不变量。
    质量把关靠 git diff 人工看快照。
    """
    from tenmin.config import Settings
    from tenmin.script.llm import build_provider

    cfg = load_project(project / "project.yaml")
    provider = build_provider(cfg.llm, Settings())
    await run_pipeline(cfg, provider, only=V1_STAGES)

    paths = Paths(project)
    table = paths.table(1).read_text(encoding="utf-8")
    narration = paths.narration(1).read_text(encoding="utf-8")
    script = Script.model_validate_json(paths.script(1).read_text(encoding="utf-8"))

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    (SNAPSHOT_DIR / "saijo_e02.解说方案.md").write_text(table, encoding="utf-8")
    (SNAPSHOT_DIR / "saijo_e02.narration.txt").write_text(narration, encoding="utf-8")
    (SNAPSHOT_DIR / "saijo_e02.script.json").write_text(
        script.model_dump_json(indent=2), encoding="utf-8"
    )

    assert len(script.beats) >= 3
    assert all(beat.clips for beat in script.beats)
    assert any(clip.is_silent_highlight for beat in script.beats for clip in beat.clips)
    assert OCR_GARBAGE not in narration
    deviation = abs(script.est_total_seconds - script.target_seconds) / script.target_seconds
    assert deviation <= 0.12, script.est_total_seconds
