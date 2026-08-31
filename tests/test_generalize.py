"""泛化复验：「长无字幕间隙 = 演出高光」这条规则在别的番上还成立吗？

spec §9 风险表明确要求：该规则只在《才女的侍从》第 2 集上验证过。
把 2-3 部不同类型番（日常 / 战斗 / 悬疑）的 SRT 放进 tests/fixtures/generalize/
后本文件自动生效。文件名任意，扩展名 .srt。

若某部番上无字幕间隙检出数为 0 或全部落在 OP/ED，说明规则不泛化，
按 spec §9 的应对：把音频能量曲线提前到 v1.5。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tenmin.ingest.normalize import build_track
from tenmin.signals.aggregate import build_report

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "generalize"
SRT_FILES = sorted(FIXTURE_DIR.glob("*.srt")) if FIXTURE_DIR.exists() else []

pytestmark = pytest.mark.generalize


@pytest.mark.skipif(not SRT_FILES, reason="tests/fixtures/generalize/ 里没有 SRT")
@pytest.mark.parametrize("srt_path", SRT_FILES, ids=lambda p: p.stem)
def test_rule_finds_silent_gaps(srt_path: Path):
    track = build_track(srt_path, episode=1)
    report = build_report(track)
    assert report.silent_gaps, f"{srt_path.name} 检出 0 个无字幕间隙，规则可能不泛化"


@pytest.mark.skipif(not SRT_FILES, reason="tests/fixtures/generalize/ 里没有 SRT")
@pytest.mark.parametrize("srt_path", SRT_FILES, ids=lambda p: p.stem)
def test_gap_count_is_in_usable_range(srt_path: Path):
    """一集里 3-30 个高光是可用区间。太少说明规则失效，太多说明阈值太松。"""
    track = build_track(srt_path, episode=1)
    report = build_report(track)
    assert 3 <= len(report.silent_gaps) <= 30, (
        f"{srt_path.name} 检出 {len(report.silent_gaps)} 个间隙，超出可用区间"
    )


@pytest.mark.skipif(not SRT_FILES, reason="tests/fixtures/generalize/ 里没有 SRT")
@pytest.mark.parametrize("srt_path", SRT_FILES, ids=lambda p: p.stem)
def test_has_at_least_one_long_gap(srt_path: Path):
    """至少一个 >= 8s 的间隙（强度 3 以上），否则拿不到真正的演出高光。"""
    track = build_track(srt_path, episode=1)
    report = build_report(track)
    longest = max((g.end - g.start) for g in report.silent_gaps)
    assert longest >= 8.0, f"{srt_path.name} 最长间隙只有 {longest:.1f}s"


@pytest.mark.skipif(not SRT_FILES, reason="tests/fixtures/generalize/ 里没有 SRT")
@pytest.mark.parametrize("srt_path", SRT_FILES, ids=lambda p: p.stem)
def test_gaps_are_outside_credits(srt_path: Path):
    """无字幕间隙不能全是 OP/ED——那说明 credits 识别失效了。"""
    track = build_track(srt_path, episode=1)
    report = build_report(track)
    for gap in report.silent_gaps:
        for window in (track.op_range, track.ed_range):
            if window is None:
                continue
            assert not (gap.start >= window[0] and gap.end <= window[1]), (
                f"{srt_path.name} 的间隙 {gap.start:.1f}-{gap.end:.1f} 落在 credits 区间内"
            )


@pytest.mark.skipif(not SRT_FILES, reason="tests/fixtures/generalize/ 里没有 SRT")
@pytest.mark.parametrize("srt_path", SRT_FILES, ids=lambda p: p.stem)
def test_dialogue_coverage_is_partial(srt_path: Path):
    """spec 发现 2：字幕只覆盖 40-60% 时长。这是整套方案成立的前提。"""
    track = build_track(srt_path, episode=1)
    spoken = sum(
        ln.duration for ln in track.lines if ln.kind in ("dialogue", "monologue")
    )
    coverage = spoken / track.duration
    assert 0.2 <= coverage <= 0.8, f"{srt_path.name} 字幕覆盖率 {coverage:.0%}，超出预期区间"
