"""LLM 输出的后处理校验。LLM 会编造时间戳，这里是唯一的拦网。

模块分成两半，别把它们混起来：

- `check_script`：**纯读**。只看、只返回 warning，一个字节都不改 script。
- `repair_script`：**显式修复**。深拷贝一份再改（丢 clip、按字幕覆写时间戳、回填
  is_silent_highlight），返回那个**新**对象。

`validate_script` 是两者的组合，对外形状与历史一致。拆开的动因是调用方
（script/single.py）需要「先校验、后比较」：预算重写轮会拿新稿替换旧稿，而原来
validate 就地改写并把同一个对象塞回 ValidationResult，上一版根本没被保留下来，
「两版择优」物理上做不到。
"""

from __future__ import annotations

from pydantic import BaseModel

from tenmin.models import Beat, Clip, DialogueLine, DialogueTrack, Script, SignalReport

ANCHOR_TOLERANCE_SECONDS = 5.0
SILENT_OVERLAP_SECONDS = 1.0
MIN_BEATS = 3

# anchor 覆写的幅度上限（秒）。超过它就**不改写**，只留一条 warning。
#
# 为什么需要这道闸：_anchor_time 取所有匹配行 start 的最小值，而实测 223 个真实 clip
# 里有 108 个的 anchor 行起点跨度（max-min）大于 clip 自身时长（中位 12.1 秒，最大
# 207 秒）—— anchor_lines 常态性地比 clip 覆盖得宽。多数情况下偏差落在 5 秒容差内不
# 触发覆写，但一旦对不上（saijo E02 的 script.json 因为 ingest 行号变过，22 个 clip
# 的覆写幅度达 14.6–118.9 秒），原来是**无条件强制覆写**，把 LLM 那份自洽的时间戳
# （start/end/visual 三者互相对得上）换成一个完全无关场景的起点。
#
# 取 60 秒 = 容差的 12 倍：这个量级已经超过一整个 beat 的常见画面跨度，两个来源里
# 必有一个是系统性错的，而我们分不清是哪一个 —— 分不清的时候就不该静默改写。
# 刻意**不**做成 config 旋钮：它不是「这部番想要什么」，而是「超过这个幅度我们已经
# 无法判断谁对」的可信度边界。
# 实测语料里 E02 之外的 10 个样本一次覆写都不触发，所以这个值没有真实的「合法覆写」
# 样本可标定，它是从推理来的，不是从数据来的。
ANCHOR_OVERWRITE_MAX_SECONDS = 60.0

# clip 的最短可用时长（秒）。比它短的一律丢弃 + warning。
#
# 实测 263 个真实 clip 的时长分布：min 3.09 / p01 3.34 / p05 4.00 / p50 15.81 /
# max 92.73，`< 3.0 秒` 的一个都没有。取 1.5 = 实测下界的一半，留足两倍余量，只拦
# 「0.2 秒」这种一路进渲染就是一帧闪屏的值。
#
# 处理策略选「丢弃」而不是「延长到最小值」：延长会把 clip 推进 OP/ED 或推出片长
# （两者都得再走一遍窗口检查），而一条 1.5 秒以下的 clip 本来就没有可用画面。丢弃
# 也跟本模块其余全部 clip 判据（越界／时长非正／片头片尾）的降级口径一致，
# 「某个节点的 clip 全灭就重试」那道网照样兜着。
MIN_CLIP_SECONDS = 1.5

# clip 与 OP/ED 的重叠比例上限。达到它就丢弃。
#
# 原来的判据是 `_fully_inside`——只在 clip **完全**落在 OP/ED 内时才丢。实测
# saijo E06 stage1 的 clip 195.6-225.6 有 19.1 秒（63.5%）压在 OP(104.5, 214.7) 上，
# 只有 10.9 秒是正片，旧判据原样放过，成片里就出现片头曲画面。
#
# 实测 228 个真实 clip 里与 OP/ED 有任何重叠的只有 5 个，比例是
# 1.000 / 1.000 / 0.635 / 0.122 / 0.011 —— 中间是空的，0.5 正好落在「整段是片头曲」
# 与「擦到片头曲尾巴 1 秒」这两簇之间。
# 边界语义：**达到**就丢（>=），不是严格超过。恰好一半是片头曲画面的 clip 没有保留价值。
# 跟 ANCHOR_OUTSIDE_MAX_RATIO 一样是「过半」这个定义，不是可调偏好，所以留在模块级。
CREDITS_OVERLAP_MAX_RATIO = 0.5
# 「落在窗**外**的 anchor 行占比」的上限。名字里的 OUTSIDE 是刻意的：它原来叫
# ANCHOR_COVERAGE_MIN_RATIO（「覆盖率下限」），而代码里比的是 outside/total，
# 语义正好反过来 —— 读代码的人会以为 0.5 是「至少一半要被覆盖」。
#
# 实测真实 LLM 输出里 14/18 个 clip 至少有一条 anchor 落在窗外，多数只差 1-3 秒无害，
# 所以只在「过半 anchor 都在窗外」时才报——那种情况说明旁白讲的内容整段没有画面。
# 边界语义：**严格超过**才报。恰好一半在窗外是平手，不报（有测试锁着）。
ANCHOR_OUTSIDE_MAX_RATIO = 0.5


class ScriptValidationError(RuntimeError):
    """校验后剧本不可用，调用方应重试一次 LLM。"""


class ValidationResult(BaseModel):
    script: Script
    warnings: list[str] = []


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _credits_overlap_ratio(clip: Clip, window: tuple[float, float] | None) -> float:
    """clip 有多大比例压在这个片头/片尾窗上。窗为 None 或 clip 时长非正都算 0。"""
    if window is None or clip.duration <= 0:
        return 0.0
    return _overlap(clip.start, clip.end, window[0], window[1]) / clip.duration


def _reject_reason(clip: Clip, track: DialogueTrack) -> str | None:
    """这条 clip 为什么不能用。返回 None 表示能用。

    刻意做成一个纯函数：anchor 覆写会改 start/end，改完必须把**全部**窗口检查再跑
    一遍（原来只重查了 `end <= start`，被 anchor 拽进片头曲或推出片长的 clip 会原样
    保留）。只有一份实现，就不会有「第一遍查了、第二遍漏了」这种分叉。
    """
    if clip.end <= clip.start:
        return f"clip {clip.start:.1f}-{clip.end:.1f} 时长非正，丢弃"
    if clip.start < 0 or clip.end > track.duration:
        return (
            f"clip {clip.start:.1f}-{clip.end:.1f} 越界"
            f"（正片 0-{track.duration:.1f}），丢弃"
        )
    if clip.duration < MIN_CLIP_SECONDS:
        return (
            f"clip {clip.start:.1f}-{clip.end:.1f} 只有 {clip.duration:.2f} 秒，"
            f"短于下限 {MIN_CLIP_SECONDS} 秒（进渲染就是一帧闪屏），过短丢弃"
        )
    for name, window in (("片头曲", track.op_range), ("片尾曲", track.ed_range)):
        ratio = _credits_overlap_ratio(clip, window)
        if ratio >= CREDITS_OVERLAP_MAX_RATIO:
            return (
                f"clip {clip.start:.1f}-{clip.end:.1f} 有 {ratio:.0%} 落在{name}内，丢弃"
            )
    return None


def _anchor_matches(track: DialogueTrack, anchor_lines: list[int]) -> list[DialogueLine]:
    return [
        ln
        for ln in track.lines
        if ln.idx in anchor_lines or any(m in anchor_lines for m in ln.merged_from)
    ]


def _anchor_time(track: DialogueTrack, anchor_lines: list[int]) -> float | None:
    starts = [ln.start for ln in _anchor_matches(track, anchor_lines)]
    if not starts:
        return None
    return min(starts)


def _quote_matches(
    tracks: dict[int, DialogueTrack], episodes: list[int], quote: str
) -> list[DialogueLine]:
    """按整行完全相等反查金句出处。找不到是合法情况（跨 cue 拼接／双轨半句）。"""
    target = quote.strip()
    found: list[DialogueLine] = []
    for episode in episodes:
        track = tracks.get(episode)
        if track is None:
            continue
        found.extend(ln for ln in track.lines if ln.text.strip() == target)
    return found


def _check_hold_quotes(beat: Beat, tracks: dict[int, DialogueTrack]) -> list[str]:
    """留白金句的原声必须落在本 beat 某个 clip 的时间窗内，否则剪辑师放不出来。"""
    if not beat.clips:
        return []
    episodes = list(dict.fromkeys(clip.episode for clip in beat.clips))
    warnings: list[str] = []
    for hold in beat.audio.holds:
        matches = _quote_matches(tracks, episodes, hold.quote)
        if not matches:
            continue
        if any(
            _overlap(ln.start, ln.end, clip.start, clip.end) > 0
            for ln in matches
            for clip in beat.clips
        ):
            continue
        warnings.append(
            f"{beat.label}：留白金句「{hold.quote}」的原声在 "
            f"{matches[0].start:.1f} 秒，不在本节点任何 clip 的时间窗内，"
            f"剪辑时放不出这句原声"
        )
    return warnings


def _check_anchor_coverage(beat: Beat, tracks: dict[int, DialogueTrack]) -> list[str]:
    """clip 的时间窗必须装得下自己的 anchor_lines，否则旁白讲的内容没有画面。"""
    warnings: list[str] = []
    for clip in beat.clips:
        track = tracks.get(clip.episode)
        if track is None:
            continue
        lines = _anchor_matches(track, clip.anchor_lines)
        total = len(lines)
        if total == 0:
            continue
        outside = sum(
            1
            for ln in lines
            if _overlap(ln.start, ln.end, clip.start, clip.end) <= 0
        )
        if outside / total > ANCHOR_OUTSIDE_MAX_RATIO:
            warnings.append(
                f"{beat.label}：clip {clip.start:.1f}-{clip.end:.1f} 的 anchor 行有 "
                f"{outside}/{total} 条落在时间窗外，旁白讲的内容缺画面"
            )
    return warnings


def check_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
) -> list[str]:
    """纯读的语义校验。返回全部 warning，**不改** script、也不抛异常。

    reports 目前用不到，但保留在签名里：它与 repair_script 共用一套入参，
    调用方（single.py）拿同一组素材调两个函数，签名对齐比少一个参数更值。
    """
    warnings: list[str] = []
    for beat in script.beats:
        warnings.extend(_check_hold_quotes(beat, tracks))
        warnings.extend(_check_anchor_coverage(beat, tracks))
    return warnings


def _apply_anchor(clip: Clip, track: DialogueTrack, label: str) -> list[str]:
    """按 anchor 行的字幕时间校准 clip 起点（就地改 clip，clip 已是深拷贝）。

    偏差超过 ANCHOR_OVERWRITE_MAX_SECONDS 时**不改**，只留 warning —— 那个量级说明
    两个来源必有一个系统性错了，而我们分不清是哪一个。
    """
    anchor_start = _anchor_time(track, clip.anchor_lines)
    if anchor_start is None:
        return []
    drift = abs(anchor_start - clip.start)
    if drift <= ANCHOR_TOLERANCE_SECONDS:
        return []
    if drift > ANCHOR_OVERWRITE_MAX_SECONDS:
        return [
            f"{label}：clip 起点 {clip.start:.1f} 与 anchor 行时间 {anchor_start:.1f} "
            f"差了 {drift:.1f} 秒，超过覆写上限 {ANCHOR_OVERWRITE_MAX_SECONDS:.0f} 秒，"
            f"没有覆写（anchor_lines 或 clip 时间戳之一是系统性错的，请人工核对）"
        ]
    duration = clip.duration
    clip.start = anchor_start
    clip.end = min(anchor_start + duration, track.duration)
    return [
        f"{label}：clip 起点 与 anchor 行时间 {anchor_start:.1f} 偏差超过 "
        f"{ANCHOR_TOLERANCE_SECONDS:.0f} 秒，以字幕时间为准"
    ]


def repair_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
) -> tuple[Script, list[str]]:
    """丢掉不可用的 clip、按字幕覆写偏差过大的时间戳、回填 is_silent_highlight。

    返回 **新** Script（深拷贝后再改），入参一字不动。修不了的（节点数不足、
    某个节点的 clip 全灭）抛 ScriptValidationError，调用方重试一次 LLM。

    顺序是刻意的：**先**按 anchor 校准时间戳，**再**拿校准后的最终值跑窗口检查。
    原来是反的（先查、再覆写、只重查 end<=start），于是被 anchor 拽进片头曲或推出
    片长的 clip 会原样留到成片里。
    """
    if len(script.beats) < MIN_BEATS:
        raise ScriptValidationError(
            f"剧本节点数 {len(script.beats)} 少于下限 {MIN_BEATS}，重试"
        )

    repaired = script.model_copy(deep=True)
    warnings: list[str] = []

    for beat in repaired.beats:
        kept: list[Clip] = []
        for clip in beat.clips:
            track = tracks.get(clip.episode)
            if track is None:
                warnings.append(f"{beat.label}：clip 引用了不存在的集数 {clip.episode}，丢弃")
                continue

            anchor_warnings = _apply_anchor(clip, track, beat.label)
            reason = _reject_reason(clip, track)
            if reason is not None:
                warnings.extend(anchor_warnings)
                warnings.append(f"{beat.label}：{reason}")
                continue
            warnings.extend(anchor_warnings)

            report = reports.get(clip.episode)
            gaps = report.silent_gaps if report else []
            clip.is_silent_highlight = any(
                _overlap(clip.start, clip.end, gap.start, gap.end)
                >= SILENT_OVERLAP_SECONDS
                for gap in gaps
            )
            kept.append(clip)

        if not kept:
            raise ScriptValidationError(
                f"{beat.label} 的所有 clip 都未通过校验，剧本不可用，重试"
            )
        beat.clips = kept

    return repaired, warnings


def validate_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
) -> ValidationResult:
    """repair 一遍再 check 一遍。对外形状与历史一致，但**不再改动入参**。"""
    repaired, warnings = repair_script(script, tracks, reports)
    warnings.extend(check_script(repaired, tracks, reports))
    return ValidationResult(script=repaired, warnings=warnings)
