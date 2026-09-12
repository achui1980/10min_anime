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


def test_finished_stage_rows_stay_visible_until_the_episode_ends():
    """本集已完成的阶段是有用信息（这集干到哪儿了），跑这一集期间不许提前清掉。"""
    with RichProgressReporter() as reporter:
        reporter.episode_start(2, 1, 2)
        reporter.stage_start("script")
        reporter.stage_done("script")
        reporter.stage_skip("docgen")
        assert descriptions(reporter) == [
            "总进度：第 1/2 集 (E02)",
            "✓ script",
            "⏭ docgen（已是最新，跳过）",
        ]


def test_stage_rows_do_not_pile_up_across_episodes():
    """原实现每次 stage_start/stage_skip 都 add_task、从不 remove_task，
    批量 10 集 × 6 阶段就是 60 多行永久堆在终端里。"""
    with RichProgressReporter() as reporter:
        for index in (1, 2, 3, 4, 5):
            reporter.episode_start(index, index, 5)
            for stage in ("script", "docgen", "voice", "timeline", "audio", "render"):
                reporter.stage_start(stage)
                reporter.stage_done(stage)
            reporter.episode_done(index, index, 5)

        assert descriptions(reporter) == ["总进度：第 5/5 集 (E05)"]


def test_skipped_stage_rows_are_swept_too():
    """stage_skip 的行原先连 TaskID 都没记下来，谁也删不掉它。"""
    with RichProgressReporter() as reporter:
        for index in (1, 2, 3):
            reporter.episode_start(index, index, 3)
            for stage in ("script", "docgen", "voice"):
                reporter.stage_skip(stage)
            reporter.episode_done(index, index, 3)

        assert descriptions(reporter) == ["总进度：第 3/3 集 (E03)"]


def test_global_stage_rows_survive_the_episode_sweep():
    """ingest/signals 在按集循环之前跑完，不属于任何一集，它们的 ✓ 行该留着。"""
    with RichProgressReporter() as reporter:
        reporter.stage_start("ingest")
        reporter.stage_done("ingest")
        reporter.stage_skip("signals")
        reporter.episode_start(2, 1, 1)
        reporter.stage_start("script")
        reporter.stage_done("script")
        reporter.episode_done(2, 1, 1)

        assert descriptions(reporter) == [
            "✓ ingest",
            "⏭ signals（已是最新，跳过）",
            "总进度：第 1/1 集 (E02)",
        ]


def test_substep_row_is_removed_when_its_stage_finishes():
    with RichProgressReporter() as reporter:
        reporter.episode_start(2, 1, 1)
        reporter.stage_start("voice")
        reporter.substep("voice", 1, 3, "第一句")
        assert "  voice 第一句" in descriptions(reporter)

        reporter.stage_done("voice")
        assert descriptions(reporter) == ["总进度：第 1/1 集 (E02)", "✓ voice"]


def test_substep_task_is_not_reused_across_episodes():
    """原实现 _substep_tasks 按 stage 做 key 且跨集不清理，E01 的 voice 子任务行会被
    E02 直接复用（description 与 total 被 update 覆盖），语义混乱。

    这里刻意不调 stage_done：那是「阶段中途抛异常」的形状，兜底必须落在集切换上。
    """
    with RichProgressReporter() as reporter:
        reporter.episode_start(1, 1, 2)
        reporter.stage_start("voice")
        reporter.substep("voice", 1, 3, "第一句")
        first = reporter._substep_tasks["voice"]

        reporter.episode_done(1, 1, 2)
        reporter.episode_start(2, 2, 2)
        reporter.stage_start("voice")
        reporter.substep("voice", 1, 5, "另一句")

        assert reporter._substep_tasks["voice"] != first
        assert descriptions(reporter) == [
            "总进度：第 2/2 集 (E02)",
            "▶ voice",
            "  voice 另一句",
        ]


def test_stage_done_after_a_sweep_does_not_touch_a_removed_row():
    """清理必须连 _stage_tasks 里的悬空 TaskID 一起丢。

    留着的话下一集直接 stage_done（上一集只 start 没 done）会去 update 一个已经被
    remove_task 删掉的任务，rich 直接 KeyError。
    """
    with RichProgressReporter() as reporter:
        reporter.episode_start(1, 1, 2)
        reporter.stage_start("voice")
        reporter.episode_done(1, 1, 2)

        reporter.episode_start(2, 2, 2)
        reporter.stage_done("voice")

        assert descriptions(reporter) == ["总进度：第 2/2 集 (E02)", "✓ voice"]


def test_episode_start_sweeps_leftovers_when_episode_done_never_came():
    """episode_done 是新加的钩子，别让「只调 episode_start 的老调用方」堆行。"""
    with RichProgressReporter() as reporter:
        reporter.episode_start(1, 1, 2)
        reporter.stage_start("script")
        reporter.stage_done("script")
        reporter.episode_start(2, 2, 2)

        assert descriptions(reporter) == ["总进度：第 2/2 集 (E02)"]
