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
    """一集里 3-30 个高光是可用区间。太少说明规则失效，太多说明阈值太松。

    只数 strength >= 3（即时长 >= 8s）的间隙。gaps.MIN_GAP_SECONDS = 3.0 让
    3-8s 的间隙也进了结果，但按 _STRENGTH_TABLE 这一档只有 strength 2，而它
    绝大多数是换气和镜头切换，不是演出高光：akujo_e01 的 34 个间隙里 21 个是
    strength 2，saijo_e05 的 33 个里 27 个是。把这一档算进可用区间，上界就只
    是在数换气次数，两个样本都会毫无余量地顶破 30。
    """
    track = build_track(srt_path, episode=1)
    report = build_report(track)
    strong = [gap for gap in report.silent_gaps if gap.strength >= 3]
    assert 3 <= len(strong) <= 30, (
        f"{srt_path.name} 检出 {len(strong)} 个 strength>=3 的间隙"
        f"（全部间隙 {len(report.silent_gaps)} 个），超出可用区间"
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
    # op_range 算不出来时必须报错，不能静默跳过。OP 本身就是一集里最长的无字幕
    # 区间，一旦识别不出来，整段 OP 就会被当成演出高光混进 silent_gaps；而这里
    # 一 continue，下面的断言体一次都不执行，测试反倒是绿的——假绿比红更危险。
    assert track.op_range is not None, (
        f"{srt_path.name} 的 OP 区间算不出来，说明 credits 识别或静区兜底失效，"
        f"间隙里很可能混着整段 OP"
    )
    # ed_range 仍允许为 None：ED 推断依赖片尾的 staff 字幕，而 akujo_e01 /
    # akujo_e02 这两集片尾一条 staff 字幕都没有，跟 OP 一样硬断言会直接打挂。
    windows = [w for w in (track.op_range, track.ed_range) if w is not None]
    for gap in report.silent_gaps:
        for window in windows:
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
