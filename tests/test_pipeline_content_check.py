"""Final-video checks preserve vertical execution and service-owned caching."""

import asyncio
from types import SimpleNamespace

import pytest

from tenmin import pipeline
from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.content_check.models import CheckOutcome, ContentCheckError
from tenmin.pipeline import Paths, run_pipeline
from tests.fakes import FakeReporter


@pytest.fixture
def cfg(tmp_path):
    return ProjectConfig(show="test", slug="test", episodes=[
        EpisodeConfig(number=2, video="missing-source.mp4"),
        EpisodeConfig(number=3, video="missing-source-3.mp4"),
    ]).bind_root(tmp_path)


def outcome(cfg, number, status="completed"):
    directory = cfg.root / "08_check" / f"E{number:02d}"
    return CheckOutcome(status, 1, 2, directory / "report.html", directory / "results.json")


def stub_check(monkeypatch, calls, *, error=None, status="completed"):
    def fake(cfg, number, *, force=False):
        calls.append((number, force))
        if error is not None and number == 2:
            raise error
        return outcome(cfg, number, status)
    monkeypatch.setattr(pipeline, "run_check", fake, raising=False)


@pytest.mark.parametrize("options", [{"only": ["check"]}, {"from_stage": "check"}])
async def test_explicit_check_runs_disabled_without_upstream(cfg, monkeypatch, options):
    calls = []
    stub_check(monkeypatch, calls)
    def forbidden(*args, **kwargs):
        pytest.fail("check-only touched source video or upstream")
    monkeypatch.setattr(ProjectConfig, "video_path", forbidden)
    monkeypatch.setattr(pipeline, "preflight", forbidden)
    monkeypatch.setattr(pipeline, "probe_duration", forbidden)
    results = {99: outcome(cfg, 99)}
    reporter = FakeReporter()
    warnings = await run_pipeline(cfg, None, reporter=reporter,
                                  check_results=results, force=True, **options)
    assert calls == [(2, True), (3, True)]
    assert results == {2: outcome(cfg, 2), 3: outcome(cfg, 3)}
    assert warnings == []
    assert reporter.calls == [
        event for index, number in enumerate([2, 3], start=1) for event in [
            ("episode_start", number, index, 2), ("stage_start", "check"),
            ("substep", "check", 2, 2,
             f"候选 1 帧；报告 {outcome(cfg, number).report_path}"),
            ("stage_done", "check"), ("episode_done", number, index, 2),
        ]
    ]


async def test_expected_failure_continues_and_collects_only_success(cfg, monkeypatch):
    calls = []
    stub_check(monkeypatch, calls, error=ContentCheckError("missing dependency"))
    results = {2: outcome(cfg, 2), 99: outcome(cfg, 99)}
    reporter = FakeReporter()
    warnings = await run_pipeline(cfg, None, only=["check"],
                                  check_results=results, reporter=reporter)
    assert calls == [(2, False), (3, False)]
    assert results == {3: outcome(cfg, 3)}
    assert warnings == ["E02：本次检查失败／报告未更新：missing dependency"]
    assert reporter.calls.count(("stage_done", "check")) == 1
    assert ("episode_done", 2, 1, 2) in reporter.calls
    assert reporter.calls[-1] == ("episode_done", 3, 2, 2)


@pytest.mark.parametrize("error", [RuntimeError("bug"), asyncio.CancelledError(),
                                  KeyboardInterrupt()])
async def test_unknown_errors_and_interrupts_propagate(cfg, monkeypatch, error):
    calls = []
    stub_check(monkeypatch, calls, error=error)
    results = {99: outcome(cfg, 99)}
    reporter = FakeReporter()
    with pytest.raises(type(error)):
        await run_pipeline(cfg, None, only=["check"], check_results=results,
                           reporter=reporter)
    assert calls == [(2, False)]
    assert results == {}
    assert not any(event[0] in {"stage_done", "episode_done"} for event in reporter.calls)


@pytest.mark.parametrize("status", ["completed", "cached", "report_rebuilt"])
async def test_service_called_each_run_and_all_success_statuses_collected(cfg, monkeypatch, status):
    calls = []
    stub_check(monkeypatch, calls, status=status)
    results = {}
    for _ in range(2):
        await run_pipeline(cfg, None, only=["check"], episode=2, check_results=results)
        assert results[2].status == status
    assert calls == [(2, False), (2, False)]


@pytest.mark.parametrize("enabled", [False, True])
async def test_full_pipeline_default_switch(cfg, monkeypatch, enabled):
    cfg.content_check.enabled = enabled
    calls = []
    stub_check(monkeypatch, calls)
    monkeypatch.setattr(pipeline, "_is_fresh", lambda *args: True)
    monkeypatch.setattr(pipeline, "_is_fresh_stamped", lambda *args: True)
    monkeypatch.setattr(pipeline, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_load_voice", lambda *args: SimpleNamespace(chunks=[]))
    reporter = FakeReporter()
    assert await run_pipeline(cfg, None, reporter=reporter) == []
    assert calls == ([(2, False), (3, False)] if enabled else [])
    for index, number in enumerate([2, 3], start=1):
        start = reporter.calls.index(("episode_start", number, index, 2))
        end = reporter.calls.index(("episode_done", number, index, 2))
        episode_events = reporter.calls[start + 1:end]
        assert ("stage_skip", "render") in episode_events
        if enabled:
            assert episode_events[-1] == ("stage_done", "check")
        else:
            assert episode_events[-1] == ("stage_skip", "check")
            assert ("substep", "check", 0, 0, "检查未启用") in episode_events


@pytest.mark.parametrize("fresh", [False, True])
async def test_from_render_checks_after_render_even_when_fresh(cfg, monkeypatch, fresh):
    cfg.content_check.enabled = True
    calls = []
    stub_check(monkeypatch, calls)
    monkeypatch.setattr(pipeline, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_is_fresh", lambda *args: fresh)
    monkeypatch.setattr(pipeline, "run_render", lambda *args, **kwargs: None)
    reporter = FakeReporter()
    await run_pipeline(cfg, None, from_stage="render", reporter=reporter)
    assert calls == [(2, False), (3, False)]
    render_event = ("stage_skip" if fresh else "stage_done", "render")
    for index, number in enumerate([2, 3], start=1):
        start = reporter.calls.index(("episode_start", number, index, 2))
        end = reporter.calls.index(("episode_done", number, index, 2))
        events = reporter.calls[start + 1:end]
        assert events.index(render_event) < events.index(("stage_start", "check"))
        assert events[-1] == ("stage_done", "check")


@pytest.mark.parametrize("only", [[], ["render"]])
async def test_only_selection_never_autoextends(cfg, monkeypatch, only):
    cfg.content_check.enabled = True
    calls = []
    stub_check(monkeypatch, calls)
    monkeypatch.setattr(pipeline, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_is_fresh", lambda *args: True)
    reporter = FakeReporter()
    results = {99: outcome(cfg, 99)}
    assert await run_pipeline(cfg, None, only=only, check_results=results,
                              reporter=reporter) == []
    assert calls == []
    assert results == {}
    assert not any(event[1] == "check" for event in reporter.calls)
    if not only:
        assert reporter.calls == []
        assert not (cfg.root / ".config").exists()


def test_run_check_missing_final_video_prompts_render(cfg, monkeypatch):
    with pytest.raises(ContentCheckError, match="render"):
        pipeline.run_check(cfg, 2)
    assert not (cfg.root / "08_check").exists()


def test_run_check_passes_final_video_and_exact_service_contract(cfg, monkeypatch):
    video = Paths(cfg.root).video(2)
    video.parent.mkdir()
    video.write_bytes(b"final video")
    cfg.render.ffmpeg_path = "/custom/ffmpeg"
    cfg.render.ffprobe_path = "/custom/ffprobe"
    def fake(video_arg, directory, **kwargs):
        assert video_arg == video
        assert directory == cfg.root / "08_check" / "E02"
        # episode 传下去是为了让报告页知道自己属于第几集（导出遮罩标注时要用）。
        assert kwargs == {"cfg": cfg.content_check, "ffmpeg": "/custom/ffmpeg",
                          "ffprobe": "/custom/ffprobe", "force": True, "episode": 2}
        return outcome(cfg, 2)
    monkeypatch.setattr(pipeline, "check_video", fake, raising=False)
    assert pipeline.run_check(cfg, 2, force=True) == outcome(cfg, 2)


async def test_missing_final_video_warns_without_upstream(cfg):
    results = {}
    warnings = await run_pipeline(cfg, None, only=["check"], check_results=results)
    assert len(warnings) == 2
    assert all("render" in warning for warning in warnings)
    assert results == {}
