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

import unicodedata
from collections.abc import Iterable

from pydantic import BaseModel

from tenmin.config import DEFAULT_VALIDATE, ValidateConfig
from tenmin.models import Beat, Clip, DialogueLine, DialogueTrack, Script, SignalReport

# import 方向说明：script/ → render/ 这个方向本来就有（render/timeline.py 反过来
# import script/budget.py），而 render/timeline.py 只依赖 render/chunks.py 与
# script/budget.py，两者都不认识本模块，所以不成环。
from tenmin.render.timeline import beat_clip_seconds
from tenmin.script.budget import DEFAULT_RATE, beat_seconds

# 这里**刻意没有** anchor_tolerance_seconds / min_beats / min_clip_seconds 的模块级
# 别名。config.py 的模块 docstring 把别名的用途写成「给老调用点、文档与测试用」，而这
# 三个（加 min_beats 共四个）在 src/ 里一个读取点都没有 —— 本模块的函数全部走 `cfg.xxx`，
# 别名只被三条「别名 == config 默认值」的同义反复测试引用。留着它们的唯一效果是给同一个
# 值造第二个名字、并让人误以为存在一个模块级旋钮。要断言默认值请直接对着权威来源写
# （`DEFAULT_VALIDATE.min_clip_seconds`），本文件的 timeline_regression / stretch 两组
# 测试本来就是这个写法。

SILENT_OVERLAP_SECONDS = 1.0

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

# 「某行是金句的一大半」这条反向包含的长度闸：行的归一化长度既要 >= 4 个字符、
# 又要 >= 金句长度的 60%。两条都是「一个字的行不能命中任何金句」这个必要条件的
# 两种表达（短金句靠绝对下限兜、长金句靠比例兜），不是可调偏好，所以留在模块级。
QUOTE_FRAGMENT_MIN_CHARS = 4
QUOTE_FRAGMENT_MIN_RATIO = 0.6

# 提示词 single_episode.md 的「节点结构」一节要求末节点 label 以这个前缀开头。
OUTRO_LABEL_PREFIX = "收尾："
# 「落在窗**外**的 anchor 行占比」的上限。名字里的 OUTSIDE 是刻意的：它原来叫
# ANCHOR_COVERAGE_MIN_RATIO（「覆盖率下限」），而代码里比的是 outside/total，
# 语义正好反过来 —— 读代码的人会以为 0.5 是「至少一半要被覆盖」。
#
# 实测真实 LLM 输出里 14/18 个 clip 至少有一条 anchor 落在窗外，多数只差 1-3 秒无害，
# 所以只在「过半 anchor 都在窗外」时才报——那种情况说明旁白讲的内容整段没有画面。
# 边界语义：**严格超过**才报。恰好一半在窗外是平手，不报（有测试锁着）。
ANCHOR_OUTSIDE_MAX_RATIO = 0.5


class ScriptValidationError(RuntimeError):
    """校验后剧本不可用，调用方应重试一次 LLM。

    `script` 带着**没通过校验的那一版**（可能是 None，比如节点数不足时连深拷贝都还没做）。
    这一份是给 pipeline 落盘用的：一次真实调用可达 561 秒，重试耗尽后原来什么都不留，
    用户既看不到模型到底写了什么，也无从判断是判据太严还是模型真的写错了。
    与 LLMSchemaError.raw_output 的分工：那个存的是「schema 都不合法的原始文本」，
    这个存的是「schema 合法但语义校验没过的 Script」。两者形态不同，所以落在两个产物槽位。
    """

    def __init__(self, message: str, *, script: Script | None = None) -> None:
        super().__init__(message)
        self.script = script


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


def _reject_reason(clip: Clip, track: DialogueTrack, cfg: ValidateConfig) -> str | None:
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
    if clip.duration < cfg.min_clip_seconds:
        return (
            f"clip {clip.start:.1f}-{clip.end:.1f} 只有 {clip.duration:.2f} 秒，"
            f"短于下限 {cfg.min_clip_seconds} 秒（进渲染就是一帧闪屏），过短丢弃"
        )
    for name, window in (("片头曲", track.op_range), ("片尾曲", track.ed_range)):
        ratio = _credits_overlap_ratio(clip, window)
        if ratio >= CREDITS_OVERLAP_MAX_RATIO:
            return (
                f"clip {clip.start:.1f}-{clip.end:.1f} 有 {ratio:.0%} 落在{name}内，丢弃"
            )
    return None


class AnchorIndex:
    """按 anchor 行号查 track.lines，一个 track 建一次、之后逐 clip O(1) 查。

    原实现 `_anchor_matches` 每个 clip 都把 `track.lines` 整个扫一遍。一集 400 行、
    二十来个 clip，再叠上 `_anchor_time` 与 `_check_anchor_coverage` 各扫一次，
    纯属重复劳动。

    两条必须保住的语义：

    1. **`merged_from` 里的旧行号也指向该行。** `merge_continuations` 把被并掉的行号
       记进 `merged_from`，而 LLM 拿到的对白块里可能还是旧行号，所以索引对每一行
       登记 `idx` 与 `merged_from` 的每个元素（与原来的
       `ln.idx in anchor_lines or any(m in anchor_lines for m in ln.merged_from)` 等价）。
    2. **返回顺序 = track.lines 顺序，且一行只出现一次。** 原实现是「按 track.lines
       顺序过滤」，一行被 idx 与 merged_from 同时命中时也只算一条，所以这里存位置、
       查完先去重再升序。

    值是 list 而不是单个位置：idx 是 cue 级的键，`split_dual_track` 从同一条 cue 拆出的
    台词与内心独白共享同一个 idx。
    """

    __slots__ = ("_lines", "_positions")

    def __init__(self, track: DialogueTrack) -> None:
        positions: dict[int, list[int]] = {}
        for position, line in enumerate(track.lines):
            for key in (line.idx, *line.merged_from):
                positions.setdefault(key, []).append(position)
        self._lines = track.lines
        self._positions = positions

    def matches(self, anchor_lines: Iterable[int]) -> list[DialogueLine]:
        hits = {p for key in anchor_lines for p in self._positions.get(key, ())}
        return [self._lines[position] for position in sorted(hits)]

    def earliest_start(self, anchor_lines: Iterable[int]) -> float | None:
        """匹配行里最早的 start。没有匹配行时 None。"""
        starts = [line.start for line in self.matches(anchor_lines)]
        if not starts:
            return None
        return min(starts)


def _anchor_indexes(tracks: dict[int, DialogueTrack]) -> dict[int, AnchorIndex]:
    return {episode: AnchorIndex(track) for episode, track in tracks.items()}



def _normalize_quote(text: str) -> str:
    """只留字母与数字（含 CJK 汉字与假名）。标点、空白、符号一律丢掉。

    判据用 unicodedata 的大类 L*/N*，与 render/tts.py 的 _is_pronounceable 同源：
    「读得出声的字符」正好也是「反查台词时该比对的字符」。
    """
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def _quote_matches(
    tracks: dict[int, DialogueTrack], episodes: list[int], quote: str
) -> list[DialogueLine]:
    """归一化（去标点/空白）后反查金句出处。找不到仍是合法情况（跨 cue 拼接／双轨半句）。

    原来是**整行完全相等**，标点差一个就找不到，而 `if not matches: continue` 让找不到
    的 hold 静默跳过整个检查。实测 61 个真实 hold 的定位率：

    - 整行完全相等（原实现）：41/61 = 67.2%
    - 归一化后整行相等：43/61 = 70.5%
    - 归一化后「金句是某行的子串」：44/61 = 72.1%
    - 再加上「某行是金句的一大半」（下面的反向包含）：46/61 = 75.4%

    反向包含必须带长度闸（QUOTE_FRAGMENT_MIN_*）：LLM 常把相邻两条字幕缝成一句金句，
    要认出这种情况就得允许「行 ⊂ 金句」，但不加闸的话「嗯」这种一个字的行会命中任何
    金句，把检查稀释成噪声（实测放开闸门后定位率虚高到 83.6%，单个 hold 最多命中 15 行）。
    """
    target = _normalize_quote(quote)
    if not target:
        return []
    minimum = max(QUOTE_FRAGMENT_MIN_CHARS, QUOTE_FRAGMENT_MIN_RATIO * len(target))
    found: list[DialogueLine] = []
    for episode in episodes:
        track = tracks.get(episode)
        if track is None:
            continue
        for ln in track.lines:
            line = _normalize_quote(ln.text)
            if not line:
                continue
            if target in line or (len(line) >= minimum and line in target):
                found.append(ln)
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


def _check_anchor_coverage(beat: Beat, indexes: dict[int, AnchorIndex]) -> list[str]:
    """clip 的时间窗必须装得下自己的 anchor_lines，否则旁白讲的内容没有画面。"""
    warnings: list[str] = []
    for clip in beat.clips:
        index = indexes.get(clip.episode)
        if index is None:
            continue
        lines = index.matches(clip.anchor_lines)
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


def _check_timeline_order(script: Script, cfg: ValidateConfig) -> list[str]:
    """A1：节点的画面起点应该跟正片时间轴同向前进。

    提示词（script/prompts/single_episode.md 的「硬性要求」一节）把「事件顺序必须与
    时间戳一致」列为废稿条件，但原来代码里**没有任何跨 beat 的时序检查**——beat 3 的
    clip 全在 60 秒、
    beat 4 全在 20 秒也照样通过。

    口径与豁免（都是实测定下来的，别凭感觉改）：

    - 排序键取「本 beat 全部 clip 起点的**最小值**」。实测 13 份真实 script.json 用这个
      口径只抓到 1 处逆序（saijo E02，倒退 230 秒，确实是真问题）；换成中位数会额外
      抓到一个只倒退 5.0 秒的抖动，那是噪声。
    - `hook` 与 `outro` 豁免：hook 本来就允许抓全片任何位置最抓人的画面（实测 saijo E06
      的 hook 起点 86 秒、E02 的 hook 起点 134 秒都在后面节点之前，但也有 hook 抓片尾
      定格的写法），outro 是收尾。
    - 只报 warning 不判错：倒叙是合法的创作手法，提示词自己也写了「用了倒叙就明确标出来」。
    """
    warnings: list[str] = []
    previous_start: float | None = None
    previous_label = ""
    for beat in script.beats:
        if beat.role in ("hook", "outro") or not beat.clips:
            continue
        start = min(clip.start for clip in beat.clips)
        if previous_start is not None:
            regression = previous_start - start
            if regression > cfg.timeline_regression_max_seconds:
                warnings.append(
                    f"{beat.label}：画面起点 {start:.1f} 秒比上一节点（{previous_label}，"
                    f"{previous_start:.1f} 秒）倒退了 {regression:.1f} 秒，"
                    f"超过 {cfg.timeline_regression_max_seconds:.0f} 秒。"
                    f"如果不是刻意的倒叙，事件顺序与时间戳就不一致了"
                )
        previous_start = start
        previous_label = beat.label
    return warnings


def _check_cue_offsets(beat: Beat, rate: str) -> list[str]:
    """A2：hold.at / sfx.at 是相对本节点旁白起点的偏移，不能落到本节点之外。

    模型层已经保证 `at >= 0` 与 `0 < duration <= 15`，缺的是**跨字段**这一条。超出去
    的后果是静默的：render/chunks.py 的 assign_holds 把 at 贴到「偏移最近的句边界」，
    而句偏移是本节点旁白的累计秒数，所以一个 at=300 的 hold 会被无声无息地按到最后
    一句后面。

    上界取 `beat_seconds(beat, rate=rate)`（旁白秒数 + 本节点全部留白秒数），也就是这个
    节点在成片里的总跨度。**`rate` 必须传**：它就是 `render.rate`，而 assign_holds 那边
    算同一个上界时是按 rate 缩放的。原来这里不传，于是 `render.rate != "+0%"` 的项目在
    script 阶段与 voice 阶段拿到两个不同的上界（实测 60 字旁白 + 一个 2 秒留白：
    不传 rate 15.556 秒、rate="+20%" 13.296 秒、"-20%" 18.944 秒），一个放过一个报警。

    实测 76 个 hold 与 51 个 sfx：`at / 纯旁白秒数` 的最大值分别是 1.004 与
    1.115（p50 0.647 / 0.603），按 beat_seconds 这个上界一个都不越界。换句话说
    「把留白放在这段旁白的最后」是正常创作，而 at 绝对值最大的那个 hold（36.0 秒）
    对应的节点旁白本身就有 35.9 秒——它不是 bug。
    """
    span = beat_seconds(beat, rate=rate)
    if span <= 0:
        return []
    warnings: list[str] = []
    for hold in beat.audio.holds:
        if hold.at > span:
            warnings.append(
                f"{beat.label}：留白落点 at={hold.at:.1f} 秒超出本节点跨度 "
                f"{span:.1f} 秒（旁白 + 留白），这段留白会被静默按到最后一句后面"
            )
    for cue in beat.audio.sfx:
        if cue.at > span:
            warnings.append(
                f"{beat.label}：音效落点 at={cue.at:.1f} 秒超出本节点跨度 {span:.1f} 秒"
            )
    return warnings


def _check_footage_budget(beat: Beat, cfg: ValidateConfig, rate: str) -> list[str]:
    """A3：本节点的画面总时长与旁白时长得在同一个量级。

    render/timeline.py 的 build_timeline 按 `ratio = 旁白秒数 / 画面秒数` 缩放本节点
    每一个 clip（`source_end = clip.start + clip.duration * ratio`）。ratio 远大于 1 时
    每段都要往后多吃几倍源片，吃到超出片长就被钳到片尾（那边「已钳到片尾」那条 warning
    就是这个），成片画面与旁白错位；ratio 远小于 1 时每段的尾巴被大幅截掉。

    在 script 阶段就能提前拦住，不用等到 timeline。阈值来自实测：85 个真实 beat 的
    拉伸倍率落在 **0.193–2.526**（画面/旁白比值 0.396–5.176，中位 1.319），
    上下界（4.0 / 0.125）各留约 1.6 倍余量，只拦数量级级别的配错。

    `rate` 的必要性同 `_check_cue_offsets`：分子是旁白秒数，语速一变整条阈值前提就被
    乘上 1/speed_factor(rate)。

    画面秒数走 `render/timeline.py 的 beat_clip_seconds`，也就是那边算 ratio 分母时用的
    **同一份实现** —— 那个文件的 docstring 写着「ratio 的分母只能有一处算法」，而这里
    原来有一份内联的 `sum(clip.duration for clip in beat.clips)`。
    """
    footage = beat_clip_seconds(beat)
    span = beat_seconds(beat, rate=rate)
    if footage <= 0 or span <= 0:
        return []
    stretch = span / footage
    if stretch > cfg.stretch_max:
        return [
            f"{beat.label}：画面只有 {footage:.1f} 秒，旁白要 {span:.1f} 秒，"
            f"渲染时每段要拉伸 {stretch:.1f} 倍（上限 {cfg.stretch_max:.1f}），"
            f"片段会被延到源片之外再钳到片尾，成片画面错位。请给这个节点补 clip"
        ]
    if stretch < cfg.stretch_min:
        return [
            f"{beat.label}：画面多达 {footage:.1f} 秒，旁白只有 {span:.1f} 秒，"
            f"渲染时每段只用得上 {stretch:.1%}（下限 {cfg.stretch_min:.1%}），"
            f"绝大部分画面会被截掉。请减少 clip 或加长旁白"
        ]
    return []


def _check_structure(script: Script, cfg: ValidateConfig) -> list[str]:
    """B1：提示词写明的节点结构约定（single_episode.md 的「节点结构」与
    「每个节点的 clips」两节）。

    原来这里只有 `cfg.min_beats` 一条下限（低于它直接判错重试），上限与结构一概不查。

    全部只给 warning，一条都不判错：这些是「写得合不合规格」的创作约定，违反了照样
    能出片，不值得烧掉一次几百秒的 LLM 调用。实测 13 份真实 script.json（10 集 saijo
    + akujo2 + 两份已提交样本）**全部满足**这四条：首节点 role=hook、末节点 role=outro
    且 label 以「收尾：」开头、6–7 个节点、恰好 1 个 climax。所以它们报出来一定是真的
    不合规格，不是判据太严。
    """
    beats = script.beats
    if not beats:
        return []
    warnings: list[str] = []
    if beats[0].role != "hook":
        warnings.append(
            f"首节点 {beats[0].label!r} 的 role 是 {beats[0].role!r}，"
            f"提示词要求固定 hook（用最抓人的画面开场）"
        )
    if beats[-1].role != "outro":
        warnings.append(
            f"末节点 {beats[-1].label!r} 的 role 是 {beats[-1].role!r}，提示词要求 outro"
        )
    elif not beats[-1].label.startswith(OUTRO_LABEL_PREFIX):
        warnings.append(
            f"末节点 label {beats[-1].label!r} 没有以「{OUTRO_LABEL_PREFIX}」开头"
        )
    if len(beats) > cfg.max_beats:
        warnings.append(
            f"节点数 {len(beats)} 超过上限 {cfg.max_beats}（提示词要求 5–8 个），"
            f"每个节点分到的时长会被摊薄"
        )
    climaxes = [beat.label for beat in beats if beat.role == "climax"]
    if len(climaxes) > 1:
        warnings.append(
            f"有 {len(climaxes)} 个 climax 节点（{'、'.join(climaxes)}），"
            f"提示词要求最多 1 个"
        )
    return warnings


def _check_narration(beat: Beat) -> list[str]:
    """B2：narration 不该是空的。

    内部 `Beat.narration` 刻意**不**加非空约束（人手清空某段旁白、只要画面不要解说是
    合法编辑，见 models.Beat.narration 与 render/tts.py 的 warning 降级），但 validate 层该说
    一声——LLM 那条路已经被 `LLMBeat.narration: NonBlankStr` 堵死，所以这条 warning
    只可能来自人工编辑。实测 85 个真实 beat 里 0 个空旁白。
    """
    if beat.narration.strip():
        return []
    return [f"{beat.label}：旁白为空，这一段不会有配音（只有画面）"]


def check_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
    *,
    cfg: ValidateConfig = DEFAULT_VALIDATE,
    rate: str = DEFAULT_RATE,
) -> list[str]:
    """纯读的语义校验。返回全部 warning，**不改** script、也不抛异常。

    reports 目前用不到，但保留在签名里：它与 repair_script 共用一套入参，
    调用方（single.py）拿同一组素材调两个函数，签名对齐比少一个参数更值。

    `rate` 是 `render.rate`（默认 `"+0%"`，speed_factor 恒为 1.0，所以默认路径与它
    加入之前逐点等价）。两条按秒数判的检查（_check_cue_offsets 的 hold/sfx 落点上界、
    _check_footage_budget 的拉伸倍率）都要按语速缩放，否则它们跟 voice 阶段真正用的
    口径分叉 —— 见那两个函数各自的 docstring。
    """
    warnings: list[str] = []
    indexes = _anchor_indexes(tracks)
    warnings.extend(_check_structure(script, cfg))
    warnings.extend(_check_timeline_order(script, cfg))
    for beat in script.beats:
        warnings.extend(_check_narration(beat))
        warnings.extend(_check_hold_quotes(beat, tracks))
        warnings.extend(_check_anchor_coverage(beat, indexes))
        warnings.extend(_check_cue_offsets(beat, rate))
        warnings.extend(_check_footage_budget(beat, cfg, rate))
    return warnings


def _apply_anchor(
    clip: Clip,
    track: DialogueTrack,
    index: AnchorIndex,
    label: str,
    cfg: ValidateConfig,
) -> list[str]:
    """按 anchor 行的字幕时间校准 clip 起点（就地改 clip，clip 已是深拷贝）。

    偏差超过 ANCHOR_OVERWRITE_MAX_SECONDS 时**不改**，只留 warning —— 那个量级说明
    两个来源必有一个系统性错了，而我们分不清是哪一个。
    """
    anchor_start = index.earliest_start(clip.anchor_lines)
    if anchor_start is None:
        return []
    drift = abs(anchor_start - clip.start)
    if drift <= cfg.anchor_tolerance_seconds:
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
        f"{cfg.anchor_tolerance_seconds:.0f} 秒，以字幕时间为准"
    ]


def repair_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
    *,
    cfg: ValidateConfig = DEFAULT_VALIDATE,
    rate: str = DEFAULT_RATE,
) -> tuple[Script, list[str]]:
    """丢掉不可用的 clip、按字幕覆写偏差过大的时间戳、回填 is_silent_highlight。

    返回 **新** Script（深拷贝后再改），入参一字不动。修不了的（节点数不足、
    某个节点的 clip 全灭）抛 ScriptValidationError，调用方重试一次 LLM。

    顺序是刻意的：**先**按 anchor 校准时间戳，**再**拿校准后的最终值跑窗口检查。
    原来是反的（先查、再覆写、只重查 end<=start），于是被 anchor 拽进片头曲或推出
    片长的 clip 会原样留到成片里。

    `rate` 目前修复路径本身用不到（这里的判据全是绝对秒数的窗口检查，跟语速无关），
    但跟 check_script / validate_script 保持同一套签名：调用方（single.py）手上就一个
    rate，三个入口形状一致才不会出现「传了两个、漏了第三个」。
    """
    if len(script.beats) < cfg.min_beats:
        raise ScriptValidationError(
            f"剧本节点数 {len(script.beats)} 少于下限 {cfg.min_beats}，重试",
            script=script,
        )

    repaired = script.model_copy(deep=True)
    warnings: list[str] = []
    indexes = _anchor_indexes(tracks)
    # 已经为「这一集缺 SignalReport」报过警的集号。按集去重而不是按 clip：一集
    # 二十来个 clip 会刷出二十条一模一样的 warning，反而把别的信息挤掉。
    reported_missing: set[int] = set()

    for beat in repaired.beats:
        kept: list[Clip] = []
        for clip in beat.clips:
            track = tracks.get(clip.episode)
            if track is None:
                warnings.append(f"{beat.label}：clip 引用了不存在的集数 {clip.episode}，丢弃")
                continue

            anchor_warnings = _apply_anchor(
                clip, track, indexes[clip.episode], beat.label, cfg
            )
            reason = _reject_reason(clip, track, cfg)
            if reason is not None:
                warnings.extend(anchor_warnings)
                warnings.append(f"{beat.label}：{reason}")
                continue
            warnings.extend(anchor_warnings)

            report = reports.get(clip.episode)
            if report is None and clip.episode not in reported_missing:
                # 原来这里是 `gaps = report.silent_gaps if report else []`：缺 report
                # 就静默退化成「本集没有静音间隙」，is_silent_highlight 全 False、
                # 一点痕迹都不留。而 single.py 只往 reports 里放**本集**一份，
                # 跨集 clip 必然走进这个分支，静音标记就这么静默丢了。
                reported_missing.add(clip.episode)
                warnings.append(
                    f"缺第 {clip.episode} 集的静音间隙信号（02_signals），"
                    f"这一集的 clip 一律不会被标成静音高光；"
                    f"跨集引用请先把那一集也跑过 signals 阶段"
                )
            gaps = report.silent_gaps if report else []
            clip.is_silent_highlight = any(
                _overlap(clip.start, clip.end, gap.start, gap.end)
                >= SILENT_OVERLAP_SECONDS
                for gap in gaps
            )
            kept.append(clip)

        if not kept:
            # B10 的判断：**刻意不**降级成「丢掉这个 beat、剩下的够数就继续」。
            #
            # 那样确实省钱（一次重试是几百秒的 LLM 调用），但代价是**静默丢内容**：
            # 这个 beat 的旁白会同时从成片、out/*.narration.txt 和对照表里消失，
            # 而用户很可能不会注意到少了一段。而且「某个节点的 clip 全灭」本身就是
            # 强信号——它意味着模型把整整一段的时间戳都编错了，那份稿子的其余部分
            # 也不值得信。
            #
            # 省钱的那一半改由 single.py 承担：它现在会把这份没过校验的稿子落盘到
            # 03_script/E{NN}.rejected.json，所以昂贵的调用不再是白花的。
            raise ScriptValidationError(
                f"{beat.label} 的所有 clip 都未通过校验，剧本不可用，重试",
                # 带上**修到一半**的那份：它已经反映了「哪些 clip 被丢了」，
                # 比原始输入更能说明模型错在哪。
                script=repaired,
            )
        beat.clips = kept

    return repaired, warnings


def validate_script(
    script: Script,
    tracks: dict[int, DialogueTrack],
    reports: dict[int, SignalReport],
    *,
    cfg: ValidateConfig = DEFAULT_VALIDATE,
    rate: str = DEFAULT_RATE,
) -> ValidationResult:
    """repair 一遍再 check 一遍。对外形状与历史一致，但**不再改动入参**。"""
    repaired, warnings = repair_script(script, tracks, reports, cfg=cfg, rate=rate)
    warnings.extend(check_script(repaired, tracks, reports, cfg=cfg, rate=rate))
    return ValidationResult(script=repaired, warnings=warnings)
