from tenmin.progress import NullProgressReporter, ProgressReporter

from .fakes import FakeReporter


def test_null_reporter_is_protocol_conformant():
    assert isinstance(NullProgressReporter(), ProgressReporter)


def test_null_reporter_accepts_all_calls_without_error():
    reporter = NullProgressReporter()
    reporter.stage_start("ingest")
    reporter.stage_skip("signals")
    reporter.stage_done("script")
    reporter.substep("voice", 1, 3, "第一句")
    reporter.episode_start(2, 1, 3)


def test_fake_reporter_satisfies_protocol():
    assert isinstance(FakeReporter(), ProgressReporter)


def test_fake_reporter_records_every_call():
    reporter = FakeReporter()
    reporter.stage_start("ingest")
    reporter.stage_skip("signals")
    reporter.stage_done("script")
    reporter.substep("voice", 1, 3, "第一句")
    reporter.episode_start(2, 1, 3)
    assert reporter.calls == [
        ("stage_start", "ingest"),
        ("stage_skip", "signals"),
        ("stage_done", "script"),
        ("substep", "voice", 1, 3, "第一句"),
        ("episode_start", 2, 1, 3),
    ]
