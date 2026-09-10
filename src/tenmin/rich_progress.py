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
    TimeRemainingColumn,
)

STAGE_LABELS: dict[str, str] = {
    "ingest": "ingest",
    "signals": "signals",
    "script": "script",
    "docgen": "docgen",
    "voice": "voice",
    "timeline": "timeline",
    "audio": "audio",
    "render": "render",
}


class RichProgressReporter:
    """把 ProgressReporter 的 5 个方法映射到 rich.progress.Progress 的任务上。"""

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        self._episode_task: TaskID | None = None
        self._stage_tasks: dict[str, TaskID] = {}
        self._substep_tasks: dict[str, TaskID] = {}

    def __enter__(self) -> RichProgressReporter:
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._progress.__exit__(exc_type, exc, tb)

    def episode_start(self, number: int, index: int, total: int) -> None:
        label = f"总进度：第 {index}/{total} 集 (E{number:02d})"
        if self._episode_task is None:
            self._episode_task = self._progress.add_task(
                label, total=total, completed=index - 1
            )
        else:
            self._progress.update(
                self._episode_task, description=label, completed=index - 1
            )

    def stage_start(self, stage: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        task_id = self._progress.add_task(f"▶ {label}", total=None)
        self._stage_tasks[stage] = task_id

    def stage_skip(self, stage: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        self._progress.add_task(f"⏭ {label}（已是最新，跳过）", total=1, completed=1)

    def stage_done(self, stage: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        task_id = self._stage_tasks.pop(stage, None)
        if task_id is None:
            task_id = self._progress.add_task(f"✓ {label}", total=1, completed=1)
        else:
            self._progress.update(
                task_id, description=f"✓ {label}", total=1, completed=1
            )

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        description = f"  {stage} {label}".rstrip()
        task_id = self._substep_tasks.get(stage)
        if task_id is None:
            task_id = self._progress.add_task(description, total=total, completed=current)
            self._substep_tasks[stage] = task_id
        else:
            self._progress.update(
                task_id, description=description, total=total, completed=current
            )
