from rich.progress import TimeElapsedColumn

from tenmin.progress import ProgressReporter
from tenmin.rich_progress import RichProgressReporter


def descriptions(reporter: RichProgressReporter) -> list[str]:
    """当前还挂在 Progress 上的行的描述。

    白盒读 _progress.tasks 是刻意的：这个类的**全部**行为就是「往 rich 上加/改/删
    任务行」，除此之外没有任何返回值可断言。这里断言的是任务行本身（有几行、描述是
    什么、进度到几），不是 rich 把它们画成了什么字符——后者是 rich 的事。
    """
    return [task.description for task in reporter._progress.tasks]


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


def test_row_descriptions_are_the_bare_stage_names():
    """阶段行的描述就是「符号 + 阶段名」，没有任何翻译层。

    原实现绕了一个 8 个 key 全部映射到自身的 STAGE_LABELS 字典，
    STAGE_LABELS.get(stage, stage) 恒等于 stage。这条把「拆掉那层之后描述一个字
    都没变」钉住。
    """
    reporter = RichProgressReporter()
    reporter.stage_start("ingest")
    reporter.stage_skip("signals")
    reporter.stage_done("ingest")
    reporter.stage_done("script")
    reporter.substep("voice", 1, 3, "第一句")

    assert descriptions(reporter) == [
        "✓ ingest",
        "⏭ signals（已是最新，跳过）",
        "✓ script",
        "  voice 第一句",
    ]


def test_episode_row_description_and_bar():
    reporter = RichProgressReporter()
    reporter.episode_start(7, 3, 10)
    task = reporter._progress.tasks[0]
    assert task.description == "总进度：第 3/10 集 (E07)"
    assert (task.completed, task.total) == (2, 10)


def test_time_elapsed_column_is_present():
    """已跑时长是唯一对 total=None 的阶段行有意义的时间列，别再被顺手删掉。"""
    columns = RichProgressReporter()._progress.columns
    assert any(isinstance(column, TimeElapsedColumn) for column in columns)


def test_total_progress_bar_reaches_full_after_the_last_episode():
    """原实现只有 episode_start（永远 completed=index-1）、没有 episode_done，
    于是最后一集跑完总进度条封顶在 N-1/N，从来到不了 100%。"""
    with RichProgressReporter() as reporter:
        for index in (1, 2, 3):
            reporter.episode_start(index, index, 3)
            reporter.stage_start("script")
            reporter.stage_done("script")
            reporter.episode_done(index, index, 3)

        total = reporter._progress.tasks[0]
        assert (total.completed, total.total) == (3, 3)
        assert total.finished


def test_episode_done_advances_the_bar_step_by_step():
    with RichProgressReporter() as reporter:
        reporter.episode_start(2, 1, 2)
        assert reporter._progress.tasks[0].completed == 0
        reporter.episode_done(2, 1, 2)
        assert reporter._progress.tasks[0].completed == 1
        reporter.episode_start(1, 2, 2)
        assert reporter._progress.tasks[0].completed == 1
        reporter.episode_done(1, 2, 2)
        assert reporter._progress.tasks[0].completed == 2


def test_episode_done_without_episode_start_still_shows_a_total_bar():
    """库调用方/未来的阶段顺序变化不该让这里炸成 None 上调 update。"""
    with RichProgressReporter() as reporter:
        reporter.episode_done(5, 2, 4)
        total = reporter._progress.tasks[0]
        assert (total.completed, total.total) == (2, 4)
