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
    reporter.episode_done(2, 1, 3)


def test_fake_reporter_satisfies_protocol():
    assert isinstance(FakeReporter(), ProgressReporter)


def test_fake_reporter_records_every_call():
    reporter = FakeReporter()
    reporter.stage_start("ingest")
    reporter.stage_skip("signals")
    reporter.stage_done("script")
    reporter.substep("voice", 1, 3, "第一句")
    reporter.episode_start(2, 1, 3)
    reporter.episode_done(2, 1, 3)
    assert reporter.calls == [
        ("stage_start", "ingest"),
        ("stage_skip", "signals"),
        ("stage_done", "script"),
        ("substep", "voice", 1, 3, "第一句"),
        ("episode_start", 2, 1, 3),
        ("episode_done", 2, 1, 3),
    ]


def test_protocol_requires_episode_done():
    """episode_start 有始无终的话，总进度条永远停在 N-1/N。

    runtime_checkable 的 Protocol 只查方法名，所以这条同时守着
    「Protocol 里声明了」和「三个实现都补齐了」。
    """

    class MissingEpisodeDone:
        def stage_start(self, stage: str) -> None: ...
        def stage_skip(self, stage: str) -> None: ...
        def stage_done(self, stage: str) -> None: ...
        def substep(self, stage: str, current: int, total: int, label: str) -> None: ...
        def episode_start(self, number: int, index: int, total: int) -> None: ...

    assert not isinstance(MissingEpisodeDone(), ProgressReporter)
