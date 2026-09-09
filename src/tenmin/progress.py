"""进度上报的抽象接口。pipeline.py 和 render/*.py 只依赖这个 Protocol，
从不直接依赖 rich，这样它们离线测试时用一个 FakeReporter 就够了。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressReporter(Protocol):
    def stage_start(self, stage: str) -> None:
        """某阶段真的开始执行了（不是被 mtime 跳过）。"""
        ...

    def stage_skip(self, stage: str) -> None:
        """某阶段因为产物已是最新，被跳过。"""
        ...

    def stage_done(self, stage: str) -> None:
        """某阶段执行完毕。"""
        ...

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        """阶段内部的细粒度进度。目前只有 voice（按句）和 render（按 ffmpeg 百分比）会调用。"""
        ...

    def episode_start(self, number: int, index: int, total: int) -> None:
        """批量模式下，开始处理第几集（共几集）。单集模式不会调用这个方法。"""
        ...


class NullProgressReporter:
    """默认的空实现。所有方法都不做事，保证不传 reporter 时行为完全不变。"""

    def stage_start(self, stage: str) -> None:
        pass

    def stage_skip(self, stage: str) -> None:
        pass

    def stage_done(self, stage: str) -> None:
        pass

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        pass

    def episode_start(self, number: int, index: int, total: int) -> None:
        pass
