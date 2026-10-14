"""按原始评级分数筛选候选，并按采样点时间归组。"""

from collections.abc import Sequence

from tenmin.config import ContentCheckConfig
from tenmin.content_check.models import CandidateRange, FrameScore


def screen_frames(
    frames: Sequence[FrameScore], cfg: ContentCheckConfig,
) -> tuple[list[FrameScore], list[CandidateRange]]:
    selected = sorted(
        (f for f in frames if f.scores.sensitive >= cfg.sensitive_threshold
         or f.scores.questionable + f.scores.explicit >= cfg.adult_threshold),
        key=lambda f: f.time,
    )
    ranges = []
    for frame in selected:
        if ranges and frame.time - ranges[-1].last <= 1.5 * cfg.sample_interval:
            ranges[-1] = CandidateRange(first=ranges[-1].first, last=frame.time,
                                       count=ranges[-1].count + 1)
        else:
            ranges.append(CandidateRange(first=frame.time, last=frame.time, count=1))
    return selected, ranges
