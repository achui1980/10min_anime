from tenmin.progress import ProgressReporter
from tenmin.rich_progress import RichProgressReporter


def test_rich_progress_reporter_satisfies_protocol():
    assert isinstance(RichProgressReporter(), ProgressReporter)


def test_rich_progress_reporter_handles_full_call_sequence():
    """纯粹的烟雾测试：只断言不抛异常，绝不断言终端渲染出的字符
    （rich 内部怎么画是它自己的事，我们只测「调用对不对」）。"""
    with RichProgressReporter() as reporter:
        reporter.episode_start(1, 1, 2)
        reporter.stage_start("ingest")
        reporter.stage_done("ingest")
        reporter.stage_skip("signals")
        reporter.stage_start("voice")
        reporter.substep("voice", 1, 3, "第一句")
        reporter.substep("voice", 2, 3, "第二句")
        reporter.substep("voice", 3, 3, "第三句")
        reporter.stage_done("voice")
        reporter.stage_start("render")
        reporter.substep("render", 50, 100, "")
        reporter.substep("render", 100, 100, "")
        reporter.stage_done("render")
        reporter.episode_start(2, 2, 2)
        reporter.stage_start("ingest")
        reporter.stage_done("ingest")
