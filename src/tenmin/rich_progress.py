"""基于 rich 的进度条实现。全项目唯一一个直接依赖 rich 的模块——
pipeline.py 和 render/*.py 都只依赖 progress.py 的 Protocol，方便离线测试。"""

from __future__ import annotations

from typing import Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class RichProgressReporter:
    """把 ProgressReporter 的 6 个方法映射到 rich.progress.Progress 的任务上。

    行的生命周期（原实现只有「加」，从不「删」，批量 10 集 × 6 阶段就是 60 多行
    永久堆在终端里）：

    - 总进度行：第一次 episode_start 时建，一直留到最后。
    - 全局阶段行（ingest/signals，在按集循环之前跑完）：不属于任何一集，留着。
    - 按集阶段行：跟着这一集活，episode_done 时整批删。刻意不在 stage_done 时逐行
      删——「本集干到哪儿了」是有用信息，跑这一集期间该看得见；到了下一集才没用。
    - substep 行：stage_done 时就删（它是那个阶段内部的细粒度进度，阶段结束即无意义），
      集切换的整批清理只是兜底（阶段中途抛异常时 stage_done 不会来）。

    为什么不是 Progress(transient=True)：transient 只让所有行在退出 Live 时一起消失，
    跑的过程中该堆的还是堆着（rich 每帧重画那 60 行），而「总进度行」这种用户想留着
    看的东西反而会被一起抹掉。remove_task 是逐行清理，能只清该清的。
    """

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            # 绝大多数阶段行是 total=None（indeterminate），这一列对它们只显示
            # -:--:--。刻意保留原样：
            # - rich 对 indeterminate 任务显示占位符是它的既定行为，不算错；
            # - 真正需要「还要多久」的两种行都有确定的 total——总进度行（按集）与
            #   voice/render 的 substep 行（按句/按 ffmpeg 百分比），对它们是准的；
            # - 想让占位符消失就得把阶段行和总进度行拆成两个 Progress 实例各配一套
            #   column，而一个 console 同时只能有一个 Live，得再套 Group + Live。
            #   为了一列占位符做这个重构不值当。
            # 真嫌它吵的话，删掉这一行（连带上面的 TimeElapsedColumn 一起）比拆两个
            # Progress 便宜得多。
            TimeRemainingColumn(),
        )
        self._episode_task: TaskID | None = None
        self._stage_tasks: dict[str, TaskID] = {}
        self._substep_tasks: dict[str, TaskID] = {}
        # 属于「当前这一集」的行。第一次 episode_start 之前加的行（ingest/signals）
        # 不进这个列表，所以不会被集切换的清理带走。
        self._episode_rows: list[TaskID] = []
        self._in_episode = False

    def __enter__(self) -> RichProgressReporter:
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._progress.__exit__(exc_type, exc, tb)

    def episode_start(self, number: int, index: int, total: int) -> None:
        # 先扫一遍：episode_done 是后加的钩子，只调 episode_start 的老调用方
        # （以及 episode_done 之前就抛异常的情况）不该让行堆起来。
        self._sweep_episode_rows()
        self._in_episode = True
        self._update_episode_task(number, index, total, completed=index - 1)

    def episode_done(self, number: int, index: int, total: int) -> None:
        """把总进度推到 index，并清掉这一集的阶段行。

        原实现只有 episode_start（永远 completed=index-1），最后一集跑完总进度条就
        封顶在 N-1/N，从来到不了 100%。下一集的 episode_start 会再把它设回
        index-1，跟这里推上去的值相同，所以进度不会倒退。
        """
        self._update_episode_task(number, index, total, completed=index)
        self._sweep_episode_rows()

    def _sweep_episode_rows(self) -> None:
        if not self._episode_rows:
            return
        removed = set(self._episode_rows)
        for task_id in self._episode_rows:
            self._progress.remove_task(task_id)
        self._episode_rows.clear()
        # 两个字典里指向刚删掉的行的条目必须一起丢：留着的话下一集的
        # stage_done / substep 会去 update 一个已经不存在的任务（rich 直接 KeyError）。
        self._stage_tasks = {
            stage: task_id
            for stage, task_id in self._stage_tasks.items()
            if task_id not in removed
        }
        self._substep_tasks = {
            stage: task_id
            for stage, task_id in self._substep_tasks.items()
            if task_id not in removed
        }

    def _add_row(self, description: str, **kwargs: Any) -> TaskID:
        task_id = self._progress.add_task(description, **kwargs)
        if self._in_episode:
            self._episode_rows.append(task_id)
        return task_id

    def _remove_row(self, task_id: TaskID) -> None:
        self._progress.remove_task(task_id)
        if task_id in self._episode_rows:
            self._episode_rows.remove(task_id)

    def _update_episode_task(
        self, number: int, index: int, total: int, *, completed: int
    ) -> None:
        label = f"总进度：第 {index}/{total} 集 (E{number:02d})"
        if self._episode_task is None:
            self._episode_task = self._progress.add_task(
                label, total=total, completed=completed
            )
        else:
            self._progress.update(
                self._episode_task, description=label, completed=completed
            )

    def stage_start(self, stage: str) -> None:
        self._stage_tasks[stage] = self._add_row(f"▶ {stage}", total=None)

    def stage_skip(self, stage: str) -> None:
        self._add_row(f"⏭ {stage}（已是最新，跳过）", total=1, completed=1)

    def stage_done(self, stage: str) -> None:
        task_id = self._stage_tasks.pop(stage, None)
        if task_id is None:
            self._add_row(f"✓ {stage}", total=1, completed=1)
        else:
            self._progress.update(
                task_id, description=f"✓ {stage}", total=1, completed=1
            )
        substep_id = self._substep_tasks.pop(stage, None)
        if substep_id is not None:
            self._remove_row(substep_id)

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        description = f"  {stage} {label}".rstrip()
        task_id = self._substep_tasks.get(stage)
        if task_id is None:
            self._substep_tasks[stage] = self._add_row(
                description, total=total, completed=current
            )
        else:
            self._progress.update(
                task_id, description=description, total=total, completed=current
            )
