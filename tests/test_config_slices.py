"""按阶段的配置切片：映射完整性、排除表、切片内容与落盘。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.config_slices import (
    EPISODE,
    EXCLUDED,
    PROJECT_LEVEL_STAGES,
    STAGE_FIELDS,
    slice_path,
    slice_payload,
    write_slice,
    write_slices,
)
from tenmin.pipeline import STAGES


def _subconfig(name: str) -> type[BaseModel] | None:
    annotation = ProjectConfig.model_fields[name].annotation
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _leaves() -> set[str]:
    """ProjectConfig 的全部叶子字段（episodes 单独由 EPISODE 记号负责）。"""
    out: set[str] = set()
    for name in ProjectConfig.model_fields:
        if name == "episodes":
            continue
        sub = _subconfig(name)
        if sub is None:
            out.add(name)
        else:
            out |= {f"{name}.{field}" for field in sub.model_fields}
    return out


def _covered(entry: str) -> set[str]:
    if entry == EPISODE:
        return set()
    if "." in entry:
        return {entry}
    sub = _subconfig(entry)
    if sub is None:
        return {entry}
    return {f"{entry}.{field}" for field in sub.model_fields} - EXCLUDED


def test_the_stage_table_covers_exactly_the_pipeline_stages():
    assert list(STAGE_FIELDS) == STAGES


def test_every_config_leaf_is_mapped_or_excluded():
    mapped = set().union(*(_covered(e) for fields in STAGE_FIELDS.values() for e in fields))
    assert _leaves() - mapped - EXCLUDED == set()


def test_the_tables_only_name_real_fields():
    leaves = _leaves()
    assert EXCLUDED <= leaves
    for stage, fields in STAGE_FIELDS.items():
        for entry in fields:
            if entry == EPISODE:
                continue
            assert entry in ProjectConfig.model_fields or entry in leaves, (stage, entry)


def test_no_stage_names_an_excluded_field_explicitly():
    explicit = {e for fields in STAGE_FIELDS.values() for e in fields if "." in e}
    assert explicit & EXCLUDED == set()


def test_the_output_changing_retry_knobs_are_not_excluded():
    assert "llm.validation_retries" not in EXCLUDED
    assert "llm.budget_rewrite_rounds" not in EXCLUDED


def test_every_episode_field_reaches_ingest():
    assert EPISODE in STAGE_FIELDS["ingest"]


def _cfg(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "show": "才女的侍从",
            "slug": "saijo",
            "episodes": [{"number": 2, "srt": "srt/E02.srt", "op_range": [150, 220]}],
        }
    ).bind_root(tmp_path)


def _with(cfg: ProjectConfig, field: str, value: object) -> ProjectConfig:
    top, _, leaf = field.partition(".")
    if not leaf:
        return cfg.model_copy(update={top: value})
    sub = getattr(cfg, top)
    return cfg.model_copy(update={top: sub.model_copy(update={leaf: value})})


def _bump(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "x"
    if value is None:
        return "http://proxy.test:8080"
    raise TypeError(value)


def _payloads(cfg: ProjectConfig) -> dict[str, str]:
    episode = cfg.episodes[0]
    return {
        stage: slice_payload(cfg, stage, None if stage in PROJECT_LEVEL_STAGES else episode)
        for stage in STAGE_FIELDS
    }


def _changed(before: ProjectConfig, after: ProjectConfig) -> set[str]:
    old, new = _payloads(before), _payloads(after)
    return {stage for stage in old if old[stage] != new[stage]}


def test_a_payload_is_sorted_json_of_resolved_values(tmp_path):
    cfg = _cfg(tmp_path)
    payload = slice_payload(cfg, "voice", cfg.episodes[0])
    assert json.loads(payload) == {"render": {"rate": "+0%", "voice": "zh-CN-YunxiNeural"}}
    assert payload == (
        json.dumps(json.loads(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


def test_a_whole_subconfig_entry_drops_the_excluded_knobs(tmp_path):
    cfg = _cfg(tmp_path)
    data = json.loads(slice_payload(cfg, "script", cfg.episodes[0]))
    assert "timeout_seconds" not in data["llm"]
    assert "script_concurrency" not in data["llm"]
    assert data["llm"]["validation_retries"] == 2
    assert data["llm"]["budget_rewrite_rounds"] == 1


def test_the_episode_token_embeds_this_episodes_config(tmp_path):
    cfg = _cfg(tmp_path)
    data = json.loads(slice_payload(cfg, "ingest", cfg.episodes[0]))
    assert data[EPISODE] == {
        "number": 2,
        "srt": "srt/E02.srt",
        "video": None,
        "op_range": [150.0, 220.0],
        "ed_range": None,
    }


def test_a_per_episode_stage_needs_an_episode(tmp_path):
    with pytest.raises(ValueError):
        slice_payload(_cfg(tmp_path), "ingest", None)


@pytest.mark.parametrize("field", sorted(EXCLUDED))
def test_changing_an_excluded_knob_changes_no_slice(tmp_path, field):
    cfg = _cfg(tmp_path)
    top, _, leaf = field.partition(".")
    current = getattr(getattr(cfg, top), leaf) if leaf else getattr(cfg, top)
    assert _changed(cfg, _with(cfg, field, _bump(current))) == set()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("render.crf", "18", {"render"}),
        ("render.font_size", 60, {"timeline", "render"}),
        ("render.voice", "zh-CN-XiaoxiaoNeural", {"voice"}),
        ("render.rate", "+10%", {"script", "docgen", "voice"}),
        ("render.duck_db", -6.0, {"audio"}),
        ("render.fade_out_seconds", 2.0, {"audio", "render"}),
        ("llm.validation_retries", 5, {"script"}),
        ("llm.budget_tolerance", 0.2, {"script"}),
        ("llm.model", "gemini-x", {"translate", "script"}),
        ("credits.op_span_min", 30.0, {"ingest"}),
        ("asr.language", "en", {"ingest"}),
        ("signals.min_gap_seconds", 5.0, {"signals"}),
        ("validate_script.max_beats", 7, {"script"}),
        ("target_seconds", 200.0, {"script"}),
        ("glossary", {"伊月": "伊月君"}, {"ingest", "translate", "script"}),
        ("show", "别的番", {"ingest", "script", "render"}),
    ],
)
def test_a_config_change_reaches_exactly_the_stages_that_read_it(
    tmp_path, field, value, expected
):
    cfg = _cfg(tmp_path)
    assert _changed(cfg, _with(cfg, field, value)) == expected


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("validate_script.hold_clip_max_gap_seconds", 4.0, {"script"}),
        ("render.subtitle_soft_max_chars", 30, {"timeline"}),
        ("render.hold_relative_lu", 2, {"audio"}),
        ("render.hold_gain_max_db", 5, {"audio"}),
        ("render.hold_silence_floor_lufs", -42, {"audio"}),
        ("render.hold_fade_seconds", 0.2, {"audio"}),
        ("render.loudness_i", -16, {"audio"}),
        ("render.loudness_tp", -2, {"audio"}),
        ("render.loudness_lra", 9, {"audio"}),
    ],
)
def test_quality_field_affects_its_first_consumer(tmp_path, field, value, expected):
    cfg = _cfg(tmp_path)
    assert _changed(cfg, _with(cfg, field, value)) == expected


def test_an_episode_change_reaches_ingest_and_the_source_video_stages(tmp_path):
    cfg = _cfg(tmp_path)
    moved = cfg.episodes[0].model_copy(update={"op_range": (140.0, 220.0)})
    assert _changed(cfg, cfg.model_copy(update={"episodes": [moved]})) == {
        "ingest",
        "timeline",
        "audio",
        "render",
    }


def test_slice_paths(tmp_path):
    assert slice_path(tmp_path, "script", 2) == tmp_path / ".config" / "E02.script.json"
    assert slice_path(tmp_path, "signals", None) == tmp_path / ".config" / "signals.json"
    with pytest.raises(ValueError):
        slice_path(tmp_path, "script", None)


def test_a_new_slice_is_backdated_to_epoch_zero(tmp_path):
    path = tmp_path / ".config" / "E02.voice.json"
    write_slice(path, "{}\n")
    assert path.stat().st_mtime_ns == 0


def test_an_unchanged_slice_is_not_touched(tmp_path):
    path = tmp_path / ".config" / "E02.voice.json"
    write_slice(path, "{}\n")
    os.utime(path, ns=(10**18, 10**18))

    write_slice(path, "{}\n")

    assert path.stat().st_mtime_ns == 10**18


def test_a_changed_slice_gets_a_normal_mtime(tmp_path):
    path = tmp_path / ".config" / "E02.voice.json"
    write_slice(path, "{}\n")

    write_slice(path, '{"a": 1}\n')

    assert path.stat().st_mtime_ns > 0
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_write_slices_writes_one_file_per_stage_and_episode(tmp_path):
    cfg = _cfg(tmp_path)
    write_slices(cfg, cfg.episodes)
    names = sorted(p.name for p in (tmp_path / ".config").iterdir())
    expected = ["signals.json", *(f"E02.{s}.json" for s in STAGES if s != "signals")]
    assert names == sorted(expected)


def test_registering_another_episode_leaves_this_episodes_slices_alone(tmp_path):
    cfg = _cfg(tmp_path)
    write_slices(cfg, cfg.episodes)
    before = {p.name: p.read_bytes() for p in (tmp_path / ".config").iterdir()}

    cfg.episodes.append(EpisodeConfig(number=11, video=Path("/v/e11.mkv")))
    write_slices(cfg, cfg.episodes)

    after = {p.name: p.read_bytes() for p in (tmp_path / ".config").iterdir()}
    assert {name: after[name] for name in before} == before
    assert "E11.ingest.json" in after
