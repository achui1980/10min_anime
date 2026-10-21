"""遮罩标注：数据模型、区间合并、enable 表达式、读写与集号对齐。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tenmin.mask import (
    DEFAULT_RANGE_SECONDS,
    DEFAULT_SIGMA,
    SCHEMA_VERSION,
    MaskError,
    MaskRange,
    MaskSpec,
    enable_expression,
    load_mask,
    merge_ranges,
    normalize,
    parse_mask,
    save_mask,
)


def _range(start: float, end: float) -> MaskRange:
    return MaskRange(start=start, end=end)


# --- 模型 ---


def test_mask_error_is_a_value_error():
    """MaskError 继承 ValueError：CLI 的 PIPELINE_ERRORS 里已经有一张 ValueError 的网。"""
    assert issubclass(MaskError, ValueError)


def test_range_rejects_end_before_start():
    with pytest.raises(Exception) as excinfo:
        _range(10.0, 10.0)
    assert "结束时间必须大于开始时间" in str(excinfo.value)


def test_range_rejects_negative_start():
    with pytest.raises(Exception):
        _range(-1.0, 5.0)


def test_range_duration():
    assert _range(3.0, 8.5).duration == pytest.approx(5.5)


def test_spec_defaults():
    spec = MaskSpec()
    assert spec.schema_version == SCHEMA_VERSION
    assert spec.episode is None
    assert spec.sigma == DEFAULT_SIGMA
    assert spec.ranges == []
    assert spec.total_seconds == 0.0


def test_spec_total_seconds_sums_ranges():
    spec = MaskSpec(ranges=[_range(0.0, 2.0), _range(10.0, 12.5)])
    assert spec.total_seconds == pytest.approx(4.5)


def test_spec_rejects_unknown_fields():
    with pytest.raises(Exception):
        MaskSpec.model_validate({"schema_version": 1, "ranges": [], "surprise": 1})


def test_parse_mask_wraps_validation_error():
    with pytest.raises(MaskError):
        parse_mask({"schema_version": 1, "ranges": [{"start": 5.0, "end": 1.0}]})


def test_parse_mask_accepts_payload_without_episode():
    spec = parse_mask({"schema_version": 1, "ranges": [{"start": 0.0, "end": 2.0}]})
    assert spec.episode is None
    assert len(spec.ranges) == 1


# --- 区间合并 ---


def test_merge_ranges_keeps_disjoint_ranges():
    merged = merge_ranges([_range(10.0, 12.0), _range(0.0, 2.0)])
    assert [(item.start, item.end) for item in merged] == [(0.0, 2.0), (10.0, 12.0)]


def test_merge_ranges_merges_overlapping():
    merged = merge_ranges([_range(0.0, 5.0), _range(3.0, 8.0)])
    assert [(item.start, item.end) for item in merged] == [(0.0, 8.0)]


def test_merge_ranges_merges_touching_ranges():
    merged = merge_ranges([_range(0.0, 5.0), _range(5.0, 8.0)])
    assert [(item.start, item.end) for item in merged] == [(0.0, 8.0)]


def test_merge_ranges_keeps_container_range_covering_others():
    merged = merge_ranges([_range(0.0, 10.0), _range(2.0, 4.0)])
    assert [(item.start, item.end) for item in merged] == [(0.0, 10.0)]


def test_merge_ranges_of_empty_is_empty():
    assert merge_ranges([]) == []


# --- enable 表达式 ---


def test_enable_expression_single_range():
    assert enable_expression([_range(156.0, 162.5)]) == "between(t,156.000,162.500)"


def test_enable_expression_joins_ranges_with_plus():
    expression = enable_expression([_range(0.0, 2.0), _range(10.0, 12.0)])
    assert expression == "between(t,0.000,2.000)+between(t,10.000,12.000)"


# --- 读写 ---


def test_load_mask_returns_none_when_missing(tmp_path: Path):
    assert load_mask(tmp_path / "E02.mask.json") is None


def test_load_mask_reads_saved_file(tmp_path: Path):
    path = tmp_path / "E02.mask.json"
    save_mask(path, MaskSpec(episode=2, ranges=[_range(1.0, 2.0)]))
    spec = load_mask(path)
    assert spec is not None
    assert spec.episode == 2
    assert spec.ranges[0].end == 2.0


def test_load_mask_rejects_broken_json(tmp_path: Path):
    path = tmp_path / "E02.mask.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MaskError):
        load_mask(path)


def test_load_mask_rejects_bad_fields(tmp_path: Path):
    path = tmp_path / "E02.mask.json"
    path.write_text(json.dumps({"schema_version": 1, "sigma": -3}), encoding="utf-8")
    with pytest.raises(MaskError):
        load_mask(path)


def test_save_mask_writes_utf8_json(tmp_path: Path):
    path = tmp_path / "E02.mask.json"
    assert save_mask(path, MaskSpec(episode=2, sigma=12.5, ranges=[_range(1.0, 2.0)])) is True
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["episode"] == 2
    assert payload["sigma"] == 12.5
    assert payload["ranges"] == [{"start": 1.0, "end": 2.0}]


def test_save_mask_keeps_mtime_when_unchanged(tmp_path: Path):
    """内容没变就不碰文件：这份文件是 render 的新鲜度输入，白刷新 mtime 会白跑一次渲染。"""
    path = tmp_path / "E02.mask.json"
    spec = MaskSpec(episode=2, ranges=[_range(1.0, 2.0)])
    assert save_mask(path, spec) is True
    before = path.stat().st_mtime_ns
    assert save_mask(path, spec) is False
    assert path.stat().st_mtime_ns == before


# --- 集号对齐 ---


def test_normalize_fills_episode_when_missing():
    spec = normalize(MaskSpec(ranges=[_range(1.0, 2.0)]), episode=2)
    assert spec.episode == 2


def test_normalize_rejects_other_episode():
    with pytest.raises(MaskError) as excinfo:
        normalize(MaskSpec(episode=2), episode=12)
    assert "E02" in str(excinfo.value)
    assert "E12" in str(excinfo.value)


def test_normalize_merges_ranges():
    spec = normalize(MaskSpec(ranges=[_range(0.0, 5.0), _range(3.0, 8.0)]), episode=2)
    assert [(item.start, item.end) for item in spec.ranges] == [(0.0, 8.0)]


def test_normalize_keeps_sigma():
    spec = normalize(MaskSpec(sigma=8.0), episode=2)
    assert spec.sigma == 8.0


def test_default_range_seconds_covers_the_sampling_gap():
    """采样间隔默认 1s，默认区间要盖得住一帧前后各一秒。"""
    assert DEFAULT_RANGE_SECONDS >= 2.0
