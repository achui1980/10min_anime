from tenmin.progress import NullProgressReporter, ProgressReporter


def test_null_reporter_is_protocol_conformant():
    assert isinstance(NullProgressReporter(), ProgressReporter)


def test_null_reporter_accepts_all_calls_without_error():
    reporter = NullProgressReporter()
    reporter.stage_start("ingest")
    reporter.stage_skip("signals")
    reporter.stage_done("script")
    reporter.substep("voice", 1, 3, "第一句")
    reporter.episode_start(2, 1, 3)
