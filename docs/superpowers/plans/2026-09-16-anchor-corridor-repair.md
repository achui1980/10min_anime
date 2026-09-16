# Anchor 走廊修复与泛化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `validate.py` 里硬编码的 60 秒覆写上限换成基于叙事结构（beat 领地单调递增）的自校准“走廊检验”，同时补上体质检测闸门、OP/ED 静默失效告警、hold 金句模糊定位、以及把校验 warning 喂回 LLM 重试 prompt，一次性解决 saijo2 实测中 31% 的画面错位问题，且不引入新的绝对时长阈值（跨番剧/片长泛化）。

**Architecture:** 6 个改动全部集中在 `src/tenmin/script/validate.py`（Section 1-5）与 `src/tenmin/script/single.py`（Section 6）。核心是把「哪个 clip 时间戳可信」的判断从「离 anchor 差多少秒」换成「anchor 落在相邻已知健康节点围出的走廊里吗」，走廊的边界完全从当前剧本自身的健康 clip 反推，不依赖任何跨番剧的绝对秒数。

**Tech Stack:** Python 3.12、pydantic、pytest + pytest-asyncio、标准库 `difflib`（新引入，用于 hold 金句模糊匹配）、uv 管理依赖。

**Spec:** `docs/superpowers/specs/2026-09-16-anchor-corridor-repair-design.md`

## Global Constraints

- 新常量 `HEALTHY_ANCHOR_MIN_RATIO = 0.25` 不进 `config.py`，留在 `validate.py` 模块级（创作旋钮 vs 合法性边界的分家判据，AGENTS.md 明文规则）。
- 删除 `ANCHOR_OVERWRITE_MAX_SECONDS = 60.0` 这个常量及其定义前的长注释，不保留、不弃用、不迁移别名。
- `CREDITS_OVERLAP_MAX_RATIO = 0.5`、`QUOTE_FRAGMENT_MIN_CHARS = 4`、`QUOTE_FRAGMENT_MIN_RATIO = 0.6`、`ANCHOR_OUTSIDE_MAX_RATIO = 0.5`、`SILENT_OVERLAP_SECONDS = 1.0`、`OUTRO_LABEL_PREFIX = "收尾："` 全部保持不变。
- 新常量 `_QUOTE_SIMILARITY_MIN_RATIO = 0.6`（`difflib.SequenceMatcher.ratio()` 的模糊匹配阈值），模块级，不进 config。
- 走廊检验只对 `beat.role in ("act", "climax")` 的 beat 计算领地；`beat.role in ("hook", "outro")` 的 beat 内所有超容差 clip 直接免检重建（复用既有 `_check_timeline_order`/A1 已经在用的同一套 `beat.role` 判断，不新造 label 前缀判断）。
- hold 金句定位只提升算法（`difflib.SequenceMatcher` 兜底），**不做自动扩窗**——超出 clip 窗口的 hold 金句仍然只报 warning，不改时间戳。
- Warning 6（字幕折行过多）本次**不改**任何拆句/放宽 `max_lines` 逻辑，只做 Section 6（把 `check_script` 的 warning 喂回重试 prompt）这一件事。
- 语料回归验证（13 份 `work/` 下的真实 script.json）**不能**写成自动化 pytest —— `work/` 与 `tests/fixtures/generalize/` 都在 `.gitignore` 里，语料不随仓库分发。写成 Task 6 里的人工验收步骤（一次性脚本，跑完即弃，不提交）。
- 每个 Task 结束后必须跑一次 `uv run pytest tests/test_validate.py tests/test_single.py -q`，确认新增/修改测试与既有测试**同时**全部通过，不允许"先跑通新测试、回头再统一修旧测试"。
- 不在本次范围（禁止顺手改）：`src/tenmin/ingest/credits.py` 的 OP 检测算法本身、`render/timeline.py:66-69` 的 edge-tts `SentenceBoundary` 改造、`work/saijo2` 的 slug 命名错误、字幕折行的拆句/放宽行数结构性改动。

---

### Task 1: 体质检测闸门（Section 1）

**Files:**
- Modify: `src/tenmin/script/validate.py` （新增常量 + 新增函数 + 在 `repair_script` 里挂一个调用点，位于现有 `indexes = _anchor_indexes(tracks)` 之后、逐 beat 循环之前）
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes：既有 `AnchorIndex.matches(anchor_lines: list[int]) -> list[DialogueLine]`（`validate.py:179`）；既有 `ValidateConfig.anchor_tolerance_seconds`（`config.py:233`，默认 5.0）；既有 `ScriptValidationError(message, *, script=None)`（`validate.py:89`）。
- Produces：新函数 `_healthy_anchor_ratio(script: Script, indexes: dict[int, AnchorIndex], cfg: ValidateConfig) -> float`，供 Task 2 的走廊检验复用同一份"哪个 clip 算健康"的判据（即 `drift <= cfg.anchor_tolerance_seconds`）。新常量 `HEALTHY_ANCHOR_MIN_RATIO: float = 0.25`。

- [ ] **Step 1: 在 `validate.py` 里新增常量与函数**

在现有 `ANCHOR_OVERWRITE_MAX_SECONDS = 60.0`（第 56 行）那个常量块**之后**（不改动它，Task 2 才删）插入：

```python
# 体质检测闸门：整份剧本里「anchor 行时间与 clip.start 对得上」的比例。
# 13 份真实语料实测：9 份健康剧本（akujo2、saijo E01/E03-E10）全部 >= 16%；
# 唯一整集行号/时间体系不同源的坏样本（saijo E02，ingest 后 script.json 是针对
# 旧版 dialogue.json 生成的 stale 产物）只有 4%。0.25 卡在两者中间，且与番剧、
# 片长、模型均无关——它是集内相对比例，不是绝对秒数。
HEALTHY_ANCHOR_MIN_RATIO = 0.25


def _healthy_anchor_ratio(
    script: Script, indexes: dict[int, AnchorIndex], cfg: ValidateConfig
) -> float:
    """全剧本里「找得到 anchor 行」的 clip 中，drift 落在容差内的比例。
    找不到 anchor 行的 clip 不计入分母（它们本来就不受 anchor 校准约束）；
    分母为 0（整份剧本没有一个 clip 带得上 anchor）时视为满分通过，交给
    别的检查去处理，不在这里误判。"""
    total = 0
    healthy = 0
    for beat in script.beats:
        for clip in beat.clips:
            index = indexes.get(clip.episode)
            if index is None:
                continue
            matches = index.matches(clip.anchor_lines)
            if not matches:
                continue
            anchor_start = min(line.start for line in matches)
            total += 1
            if abs(anchor_start - clip.start) <= cfg.anchor_tolerance_seconds:
                healthy += 1
    return healthy / total if total else 1.0
```

- [ ] **Step 2: 在 `repair_script` 里挂调用点**

找到现有代码（`validate.py:552` 附近）：

```python
    repaired = script.model_copy(deep=True)
    warnings: list[str] = []
    indexes = _anchor_indexes(tracks)
```

改成：

```python
    repaired = script.model_copy(deep=True)
    warnings: list[str] = []
    indexes = _anchor_indexes(tracks)

    healthy_ratio = _healthy_anchor_ratio(repaired, indexes, cfg)
    if healthy_ratio < HEALTHY_ANCHOR_MIN_RATIO:
        raise ScriptValidationError(
            f"整份剧本只有 {healthy_ratio:.0%} 的 clip 时间戳与 anchor 行吻合"
            f"（阈值 {HEALTHY_ANCHOR_MIN_RATIO:.0%}），疑似整集行号/时间体系不同源"
            f"（比如剧本是针对旧版字幕生成的 stale 产物），逐条修复不可靠，"
            f"剧本不可用，重试",
            script=repaired,
        )
```

- [ ] **Step 3: 在 `tests/test_validate.py` 顶部的 import 里加上新常量**

把第一批 import：

```python
from tenmin.script.validate import (
    ANCHOR_OUTSIDE_MAX_RATIO,
    ANCHOR_OVERWRITE_MAX_SECONDS,
    CREDITS_OVERLAP_MAX_RATIO,
    AnchorIndex,
    ScriptValidationError,
    check_script,
    repair_script,
    validate_script,
)
```

改成（此时先只加新常量，`ANCHOR_OVERWRITE_MAX_SECONDS` 留到 Task 2 删除）：

```python
from tenmin.script.validate import (
    ANCHOR_OUTSIDE_MAX_RATIO,
    ANCHOR_OVERWRITE_MAX_SECONDS,
    CREDITS_OVERLAP_MAX_RATIO,
    HEALTHY_ANCHOR_MIN_RATIO,
    AnchorIndex,
    ScriptValidationError,
    check_script,
    repair_script,
    validate_script,
)
```

- [ ] **Step 4: 新增一个 filler 辅助函数（供本任务与后续任务复用）**

在 `make_script`/`run` 定义（`test_validate.py:69-99` 附近）之后插入：

```python
def _healthy_filler_rows(start_idx, positions):
    """造若干「drift=0 的健康 act 节点」：每个 position 对应一个 clip(pos, pos+5,
    anchors=[idx]) 和一条时间完全吻合的字幕行 dline(idx, pos, pos+5)。
    单独测 _apply_anchor 的某个分支时，剧本里往往只有一个带 anchor 的 clip；
    Section 1 的体质检测闸门是全局比例，分母只有 1 时任何 drift 都会把比例
    干到 0 或 1，容易被体质检测误拦。这个 helper 用来把分母垫高，让体质检测
    闸门只在真正需要它触发的测试里触发。"""
    rows = []
    lines = []
    for offset, pos in enumerate(positions):
        idx = start_idx + offset
        rows.append([clip(pos, pos + 5.0, anchors=[idx])])
        lines.append(dline(idx, pos, pos + 5.0))
    return rows, lines
```

- [ ] **Step 5: 修复会被体质检测闸门误拦的 5 个既有测试**

这 5 个测试目前只有 1 个带 anchor 的 clip、且这个 clip 本身 drift 很大（就是它们要测的东西），会被 Step 2 新加的闸门在到达 `_apply_anchor` 之前就拦成 `ScriptValidationError`。用 `_healthy_filler_rows` 把分母垫高到 >= 25%。

找到（`test_validate.py:164-173`）：

```python
def test_anchor_mismatch_beyond_tolerance_overwrites_start():
    track = make_track(lines=[dline(42, 140.0, 144.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    kept = result.script.beats[0].clips[0]
    assert kept.start == pytest.approx(140.0)
    assert kept.end == pytest.approx(146.0)
```

改成：

```python
def test_anchor_mismatch_beyond_tolerance_overwrites_start():
    filler_rows, filler_lines = _healthy_filler_rows(900, [300.0, 320.0])
    track = make_track(lines=[dline(42, 140.0, 144.0), *filler_lines])
    s = make_script([[clip(100.0, 106.0, anchors=[42])], *filler_rows], pad=False)
    result = run(s, track=track)
    kept = result.script.beats[0].clips[0]
    assert kept.start == pytest.approx(140.0)
    assert kept.end == pytest.approx(146.0)
```

找到（`test_validate.py:176-181`）：

```python
def test_anchor_mismatch_within_tolerance_keeps_start():
    track = make_track(lines=[dline(42, 103.0, 107.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    assert result.script.beats[0].clips[0].start == pytest.approx(100.0)
```

改成：

```python
def test_anchor_mismatch_within_tolerance_keeps_start():
    filler_rows, filler_lines = _healthy_filler_rows(900, [300.0, 320.0])
    track = make_track(lines=[dline(42, 103.0, 107.0), *filler_lines])
    s = make_script([[clip(100.0, 106.0, anchors=[42])], *filler_rows], pad=False)
    result = run(s, track=track)
    assert result.script.beats[0].clips[0].start == pytest.approx(100.0)
```

找到（`test_validate.py:184-191`）：

```python
def test_anchor_overwrite_clamped_to_episode_end():
    track = make_track(duration=1000.0, lines=[dline(42, 996.0, 999.0)])
    s = make_script([[clip(950.0, 980.0, anchors=[42])]])
    result = run(s, track=track)
    kept = result.script.beats[0].clips[0]
    assert kept.end == pytest.approx(1000.0)
```

改成：

```python
def test_anchor_overwrite_clamped_to_episode_end():
    filler_rows, filler_lines = _healthy_filler_rows(900, [500.0, 520.0])
    track = make_track(
        duration=1000.0, lines=[dline(42, 996.0, 999.0), *filler_lines]
    )
    s = make_script([[clip(950.0, 980.0, anchors=[42])], *filler_rows], pad=False)
    result = run(s, track=track)
    kept = result.script.beats[0].clips[0]
    assert kept.end == pytest.approx(1000.0)
```

找到（`test_validate.py:476-484`）：

```python
def test_anchor_overwrite_reruns_the_op_window_check():
    track = make_track(
        op=(153.486, 224.681), lines=[dline(1, 10.0, 12.0), dline(42, 210.0, 214.0)]
    )
    s = make_script([[clip(10.0, 15.0), clip(200.0, 220.0, anchors=[42])]])
    result = run(s, track=track)
    assert len(result.script.beats[0].clips) == 1
```

改成（保留原有断言逻辑，只加 filler）：

```python
def test_anchor_overwrite_reruns_the_op_window_check():
    filler_rows, filler_lines = _healthy_filler_rows(900, [500.0, 520.0])
    track = make_track(
        op=(153.486, 224.681),
        lines=[dline(1, 10.0, 12.0), dline(42, 210.0, 214.0), *filler_lines],
    )
    s = make_script(
        [[clip(10.0, 15.0), clip(200.0, 220.0, anchors=[42])], *filler_rows],
        pad=False,
    )
    result = run(s, track=track)
    assert len(result.script.beats[0].clips) == 1
```

找到（`test_validate.py:487-494`）：

```python
def test_anchor_overwrite_clamped_too_short_is_dropped():
    track = make_track(duration=1000.0, lines=[dline(42, 996.5, 997.0)])
    s = make_script([[clip(10.0, 15.0), clip(950.0, 980.0, anchors=[42])]])
    result = run(s, track=track)
    assert len(result.script.beats[0].clips) == 1
```

改成：

```python
def test_anchor_overwrite_clamped_too_short_is_dropped():
    filler_rows, filler_lines = _healthy_filler_rows(900, [500.0, 520.0])
    track = make_track(
        duration=1000.0, lines=[dline(42, 996.5, 997.0), *filler_lines]
    )
    s = make_script(
        [[clip(10.0, 15.0), clip(950.0, 980.0, anchors=[42])], *filler_rows],
        pad=False,
    )
    result = run(s, track=track)
    assert len(result.script.beats[0].clips) == 1
```

> 注：以上 5 个测试的具体断言取值（`kept.start`/`kept.end`）在本 Task 完成时还是走的**旧** `_apply_anchor` 逻辑（Task 2 才重写），所以断言应保持与现状一致，只是加 filler 行/rows。如果本 Task 单独跑测试时这些断言已经和上面写的不一致，以现有代码的真实输出为准，不要为了凑断言去改产品代码。

- [ ] **Step 6: 新增体质检测闸门本身的测试**

在 `test_anchor_overwrite_clamped_too_short_is_dropped` 之后插入：

```python
def test_healthy_anchor_ratio_below_threshold_raises():
    """5 个带 anchor 的 clip，只有 1 个 drift 在容差内 → 20% < 25%，
    判定为整集行号/时间体系不同源，直接拒绝重试，不逐条修。"""
    track = make_track(
        duration=2000.0,
        lines=[
            dline(1, 103.0, 107.0),
            dline(2, 260.0, 264.0),
            dline(3, 400.0, 404.0),
            dline(4, 550.0, 554.0),
            dline(5, 700.0, 704.0),
        ],
    )
    s = make_script(
        [
            [clip(100.0, 106.0, anchors=[1])],  # drift=3，健康
            [clip(200.0, 206.0, anchors=[2])],  # drift=60，不健康
            [clip(300.0, 306.0, anchors=[3])],  # drift=100，不健康
            [clip(450.0, 456.0, anchors=[4])],  # drift=100，不健康
            [clip(600.0, 606.0, anchors=[5])],  # drift=100，不健康
        ],
        pad=False,
    )
    with pytest.raises(ScriptValidationError, match="行号/时间体系不同源"):
        run(s, track=track)


def test_healthy_anchor_ratio_at_threshold_passes():
    """1/4 = 25%，恰好落在阈值上，不触发（严格小于才拒绝）。"""
    track = make_track(
        duration=2000.0,
        lines=[
            dline(1, 103.0, 107.0),
            dline(2, 260.0, 264.0),
            dline(3, 400.0, 404.0),
            dline(4, 550.0, 554.0),
        ],
    )
    s = make_script(
        [
            [clip(100.0, 106.0, anchors=[1])],  # drift=3，健康
            [clip(200.0, 206.0, anchors=[2])],  # drift=60，不健康
            [clip(300.0, 306.0, anchors=[3])],  # drift=100，不健康
            [clip(450.0, 456.0, anchors=[4])],  # drift=100，不健康
        ],
        pad=False,
    )
    result = run(s, track=track)  # 不应抛错
    assert result.script is not None


def test_healthy_anchor_ratio_is_vacuously_full_when_no_clip_has_anchors():
    """没有任何 clip 带 anchor（比如全靠静音间隙推的 beat）不该被体质检测拦下。"""
    s = make_script([[clip(300.0, 306.0)], [clip(400.0, 406.0)], [clip(500.0, 506.0)]])
    result = run(s)
    assert result.script is not None
```

- [ ] **Step 7: 跑测试**

Run: `uv run pytest tests/test_validate.py -q`
Expected: 全部 PASS（含刚改的 5 个 + 新增的 3 个）。

- [ ] **Step 8: Commit**

```bash
git add src/tenmin/script/validate.py tests/test_validate.py
git commit -m "feat(validate): 加体质检测闸门，健康 anchor 比例低于 25% 直接拒绝重试"
```

---

### Task 2: 走廊检验取代 60 秒硬上限（Section 2 + 3）

**Files:**
- Modify: `src/tenmin/script/validate.py`（删除 `ANCHOR_OVERWRITE_MAX_SECONDS`；新增 `_beat_territories`/`_corridor_for`；重写 `_apply_anchor`；重写 `repair_script` 的逐 clip 循环）
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes：Task 1 产出的 `_healthy_anchor_ratio` 使用的同一个健康判据 `drift <= cfg.anchor_tolerance_seconds`；`models.BeatRole = Literal["hook","act","climax","outro"]`（`models.py:117`）；既有 `AnchorIndex.matches`。
- Produces：`_apply_anchor(clip, track, index, label, cfg) -> list[str]` **签名不变**，但语义变成"被调用就无条件重建"（容差/走廊判断移到调用方）；`_beat_territories(beats: list[Beat], indexes: dict[int, AnchorIndex], cfg: ValidateConfig) -> list[tuple[float, float] | None]`；`_corridor_for(position: int, territories: list[tuple[float,float] | None], track_duration: float) -> tuple[float, float]`。后续 Task 3（credits 钳位）需要在这个新循环里插入调用点，务必保持本 Task 完成后循环体的调用顺序：`_apply_anchor`（如果决定重建）→ `_clamp_credits_overlap`（Task 3 新加）→ `_reject_reason`。

- [ ] **Step 1: 删除 `ANCHOR_OVERWRITE_MAX_SECONDS` 及其注释块**

删除第 41-56 行整段（常量定义与它前面的长注释），只留 `SILENT_OVERLAP_SECONDS = 1.0`（39 行）紧接着 `CREDITS_OVERLAP_MAX_RATIO` 的注释块（58 行起）。

- [ ] **Step 2: 新增 `_beat_territories` 与 `_corridor_for`**

在 `_anchor_indexes` 函数（`validate.py:191-192`）之后插入：

```python
def _beat_territories(
    beats: list[Beat], indexes: dict[int, AnchorIndex], cfg: ValidateConfig
) -> list[tuple[float, float] | None]:
    """每个 beat 的「领地」＝该 beat 内所有健康 clip（drift <= 容差）的 anchor
    行时间范围的并集，取 (最早 start, 最晚 end)。只对 role in ("act","climax")
    的 beat 算——hook/outro 天生允许跳出叙事顺序（预告片倒放、回忆闪回），拿
    它们的领地去卡邻居纯属噪声。13 份真实语料实测：排除 hook/outro 后，中间
    这些「阶段一..阶段N」的领地严格单调递增，这个不变量与番剧、片长、模型
    都无关，是走廊检验的地基。"""
    territories: list[tuple[float, float] | None] = []
    for beat in beats:
        if beat.role not in ("act", "climax"):
            territories.append(None)
            continue
        starts: list[float] = []
        ends: list[float] = []
        for clip in beat.clips:
            index = indexes.get(clip.episode)
            if index is None:
                continue
            matches = index.matches(clip.anchor_lines)
            if not matches:
                continue
            anchor_start = min(line.start for line in matches)
            if abs(anchor_start - clip.start) <= cfg.anchor_tolerance_seconds:
                starts.append(anchor_start)
                ends.append(max(line.end for line in matches))
        territories.append((min(starts), max(ends)) if starts else None)
    return territories


def _corridor_for(
    position: int,
    territories: list[tuple[float, float] | None],
    track_duration: float,
) -> tuple[float, float]:
    """position 号 beat 的走廊＝往前找最近一个有领地的 beat 的领地右端，往后
    找最近一个有领地的 beat 的领地左端。找不到就钳到片头/片尾。"""
    lower = 0.0
    for i in range(position - 1, -1, -1):
        if territories[i] is not None:
            lower = territories[i][1]
            break
    upper = track_duration
    for i in range(position + 1, len(territories)):
        if territories[i] is not None:
            upper = territories[i][0]
            break
    return (lower, upper)
```

- [ ] **Step 3: 重写 `_apply_anchor`**

找到（`validate.py:490-520`，现行完整版本）：

```python
def _apply_anchor(
    clip: Clip,
    track: DialogueTrack,
    index: AnchorIndex,
    label: str,
    cfg: ValidateConfig,
) -> list[str]:
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
```

改成（不再自己判断容差/上限——是否调用、调用后走廊内外的取舍都交给 `repair_script`；这个函数现在**只负责一件事**：按 anchor 行簇重建窗口）：

```python
def _apply_anchor(
    clip: Clip,
    track: DialogueTrack,
    index: AnchorIndex,
    label: str,
    cfg: ValidateConfig,
) -> list[str]:
    """无条件按 anchor 行簇重建 clip 窗口：start 取所有匹配行的最早 start，
    end 取「所有匹配行的最晚 end」与「start+原时长」两者的较大值（保证不会
    比原来的旁白配的画面更短），再钳到集数长度。是否应该调用这个函数（容差
    内不用改、走廊外不该信）由调用方 repair_script 决定。"""
    matches = index.matches(clip.anchor_lines)
    if not matches:
        return []
    anchor_start = min(line.start for line in matches)
    anchor_end = max(line.end for line in matches)
    duration = clip.duration
    new_start = anchor_start
    new_end = min(max(anchor_end, anchor_start + duration), track.duration)
    clip.start = new_start
    clip.end = new_end
    return [
        f"{label}：clip 起点 与 anchor 行时间 {new_start:.1f} 偏差超过 "
        f"{cfg.anchor_tolerance_seconds:.0f} 秒，以字幕时间为准"
    ]
```

- [ ] **Step 4: 重写 `repair_script` 的逐 clip 循环，接入走廊判断**

找到（`validate.py:552-573` 附近，Task 1 已经在 552 行之后插入了体质检测，这里紧接着往下）：

```python
    for beat in repaired.beats:
        kept: list[Clip] = []
        for clip in beat.clips:
            track = tracks.get(clip.episode)
            if track is None:
                warnings.append(f"{beat.label}：clip 引用了不存在的集数 {clip.episode}，丢弃")
                continue
            anchor_warnings = _apply_anchor(clip, track, indexes[clip.episode], beat.label, cfg)
            reason = _reject_reason(clip, track, cfg)
            if reason is not None:
                warnings.extend(anchor_warnings)
                warnings.append(f"{beat.label}：{reason}")
                continue
            warnings.extend(anchor_warnings)
```

改成：

```python
    territories = _beat_territories(repaired.beats, indexes, cfg)
    for beat_position, beat in enumerate(repaired.beats):
        kept: list[Clip] = []
        for clip in beat.clips:
            track = tracks.get(clip.episode)
            if track is None:
                warnings.append(f"{beat.label}：clip 引用了不存在的集数 {clip.episode}，丢弃")
                continue
            index = indexes[clip.episode]
            matches = index.matches(clip.anchor_lines)
            anchor_warnings: list[str] = []
            if matches:
                anchor_start = min(line.start for line in matches)
                drift = abs(anchor_start - clip.start)
                if drift > cfg.anchor_tolerance_seconds:
                    if beat.role in ("hook", "outro"):
                        anchor_warnings = _apply_anchor(clip, track, index, beat.label, cfg)
                    else:
                        lower, upper = _corridor_for(beat_position, territories, track.duration)
                        if lower <= anchor_start <= upper:
                            anchor_warnings = _apply_anchor(clip, track, index, beat.label, cfg)
                        else:
                            warnings.append(
                                f"{beat.label}：clip {clip.start:.1f}-{clip.end:.1f} 的 "
                                f"anchor 行时间 {anchor_start:.1f} 秒超出走廊 "
                                f"[{lower:.1f}, {upper:.1f}]，判定为离群（可能指向别的节点），"
                                f"丢弃而非按 anchor 重建"
                            )
                            continue
            reason = _reject_reason(clip, track, cfg)
            if reason is not None:
                warnings.extend(anchor_warnings)
                warnings.append(f"{beat.label}：{reason}")
                continue
            warnings.extend(anchor_warnings)
```

> 注意：这一段之后紧跟的静音间隙检查、`clip.is_silent_highlight` 回填、`kept.append(clip)` 保持原样不动（`validate.py:575-593`）。Task 3 会在 `reason = _reject_reason(...)` 这一行**之前**插入 `_clamp_credits_overlap(clip, track)` 调用，所以这里改完之后不要删掉 `reason = _reject_reason(clip, track, cfg)` 这一行，Task 3 是在它前面加一行，不是替换它。

- [ ] **Step 5: 把 `test_validate.py` 顶部 import 里的 `ANCHOR_OVERWRITE_MAX_SECONDS` 删掉**

```python
from tenmin.script.validate import (
    ANCHOR_OUTSIDE_MAX_RATIO,
    CREDITS_OVERLAP_MAX_RATIO,
    HEALTHY_ANCHOR_MIN_RATIO,
    AnchorIndex,
    ScriptValidationError,
    check_script,
    repair_script,
    validate_script,
)
```

- [ ] **Step 6: 删除锁常量的旧测试**

删除（`test_validate.py`）：

```python
def test_anchor_overwrite_cap_constant():
    assert ANCHOR_OVERWRITE_MAX_SECONDS == pytest.approx(60.0)
```

- [ ] **Step 7: 把两个"覆写上限"旧测试改写成走廊场景**

删除：

```python
def test_anchor_overwrite_beyond_the_cap_is_refused_with_a_warning():
    track = make_track(lines=[dline(42, 700.0, 704.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    kept = result.script.beats[0].clips[0]
    assert kept.start == pytest.approx(100.0), "幅度超过上限就不该改写"
    assert kept.end == pytest.approx(106.0)
    assert any("没有覆写" in w for w in result.warnings), result.warnings


def test_anchor_overwrite_within_the_cap_still_happens():
    track = make_track(lines=[dline(42, 130.0, 134.0)])
    s = make_script([[clip(100.0, 106.0, anchors=[42])]])
    result = run(s, track=track)
    assert result.script.beats[0].clips[0].start == pytest.approx(130.0)
```

替换成（5 行：hook / act1(健康,建立左领地) / act2(待处理) / act3(健康,建立右领地) / outro；两个测试共用同一套 track/rows，只改 act2 那个 clip 的 anchor）：

```python
def _corridor_test_track():
    return make_track(
        duration=2000.0,
        lines=[
            dline(10, 200.0, 204.0),
            dline(11, 206.0, 210.0),  # act1 领地：(200.0, 210.0)
            dline(20, 600.0, 604.0),  # 走廊内的位置，用于「重建」场景
            dline(30, 1200.0, 1204.0),
            dline(31, 1206.0, 1210.0),  # act3 领地：(1200.0, 1210.0)
            dline(40, 50.0, 54.0),  # 走廊外的位置（早于 act1 领地），用于「丢弃」场景
        ],
    )


def _corridor_test_rows(act2_clips):
    return [
        [clip(10.0, 16.0)],  # hook，无 anchor，占位
        [clip(202.0, 208.0, anchors=[10, 11])],  # act1，健康，drift=2
        act2_clips,  # act2，待处理
        [clip(1202.0, 1208.0, anchors=[30, 31])],  # act3，健康，drift=2
        [clip(1900.0, 1906.0)],  # outro，无 anchor，占位
    ]


def test_anchor_outside_the_corridor_is_dropped_not_rebuilt():
    """act2 唯一一个待处理 clip 的 anchor 落在走廊 [210.0, 1200.0] 之外
    （anchor=40 @50.0，早于 act1 的领地右端 210.0），判定为离群，丢弃而不是
    按 anchor 重建——无条件信 anchor 会把它错误地重建成一个提前剧透了 act1
    之前内容的画面。丢弃后 act2 还剩一个无 anchor 的健康 clip，beat 不会空。"""
    track = _corridor_test_track()
    rows = _corridor_test_rows(
        [clip(300.0, 306.0, anchors=[40]), clip(650.0, 656.0)]
    )
    s = make_script(rows, pad=False)
    result = run(s, track=track)
    act2_clips = result.script.beats[2].clips
    assert len(act2_clips) == 1
    assert act2_clips[0].start == pytest.approx(650.0), "留下的应是没有 anchor 的那个"
    assert any("走廊" in w and "丢弃" in w for w in result.warnings), result.warnings


def test_anchor_inside_the_corridor_is_rebuilt():
    """act2 唯一一个待处理 clip 的 anchor 落在走廊 [210.0, 1200.0] 内
    （anchor=20 @600.0），判定可信，按 anchor 重建。"""
    track = _corridor_test_track()
    rows = _corridor_test_rows([clip(300.0, 306.0, anchors=[20])])
    s = make_script(rows, pad=False)
    result = run(s, track=track)
    act2_clips = result.script.beats[2].clips
    assert len(act2_clips) == 1
    assert act2_clips[0].start == pytest.approx(600.0)
    assert act2_clips[0].end == pytest.approx(606.0)
```

- [ ] **Step 8: 新增 hook/outro 免检 + 健康剧本零改动的测试**

在上面两个测试之后插入：

```python
def test_hook_beat_rebuilds_regardless_of_corridor():
    """hook 免检：即便 anchor 时间与「走廊」完全不沾边（预告片本来就该从
    全集乱抓素材），只要 drift 超容差就直接按 anchor 重建，不做越境判断。"""
    track = _corridor_test_track()
    rows = _corridor_test_rows([clip(300.0, 306.0, anchors=[20])])
    rows[0] = [clip(50.0, 56.0, anchors=[40])]  # hook 里塞一个 anchor 远在 @50.0 的 clip
    s = make_script(rows, pad=False)
    result = run(s, track=track)
    hook_clip = result.script.beats[0].clips[0]
    assert hook_clip.start == pytest.approx(50.0)


def test_outro_beat_rebuilds_regardless_of_corridor():
    """outro 免检：同上，回收利用 hook 的素材做尾声蒙太奇，位置天然乱序。"""
    track = _corridor_test_track()
    rows = _corridor_test_rows([clip(300.0, 306.0, anchors=[20])])
    rows[4] = [clip(1900.0, 1906.0, anchors=[40])]
    s = make_script(rows, pad=False)
    result = run(s, track=track)
    outro_clip = result.script.beats[4].clips[0]
    assert outro_clip.start == pytest.approx(50.0)


def test_fully_healthy_script_gets_zero_corridor_changes():
    """所有 clip 都在容差内：走廊检验完全不触发，13 份真实语料里 9 份健康剧本
    的回归安全就是靠这一条保证的。"""
    track = _corridor_test_track()
    rows = _corridor_test_rows([clip(602.0, 608.0, anchors=[20])])  # drift=2，健康
    s = make_script(rows, pad=False)
    result = run(s, track=track)
    assert [c.start for beat in result.script.beats for c in beat.clips] == [
        10.0,
        202.0,
        602.0,
        1202.0,
        1900.0,
    ]
    assert not any("走廊" in w for w in result.warnings)
```

- [ ] **Step 9: 跑测试**

Run: `uv run pytest tests/test_validate.py -q`
Expected: 全部 PASS。

- [ ] **Step 10: Commit**

```bash
git add src/tenmin/script/validate.py tests/test_validate.py
git commit -m "feat(validate): 走廊检验取代 60 秒硬上限，anchor 重建改按行簇定窗"
```

---

### Task 3: OP/ED credits 不再静默 + ED 重叠钳位（Section 4）

**Files:**
- Modify: `src/tenmin/script/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes：Task 2 重写后的 `repair_script` 循环（`_apply_anchor`/走廊判断之后、`_reject_reason` 之前）；既有 `DialogueTrack.op_range`/`ed_range: tuple[float,float] | None`；既有 `_reject_reason(clip, track, cfg)`（不改）。
- Produces：`_clamp_credits_overlap(clip: Clip, track: DialogueTrack) -> None`（就地修改，无返回值）。

- [ ] **Step 1: 在 `_credits_overlap_ratio` 之后新增钳位函数**

在 `validate.py` 的 `_credits_overlap_ratio`（现第 113-117 行附近，Task 2 删常量后行号会往前移，以函数名定位而非行号）之后插入：

```python
def _clamp_credits_overlap(clip: Clip, track: DialogueTrack) -> None:
    """clip 部分压在片头/片尾曲上、但压的比例还没到 _reject_reason 的丢弃线
    （CREDITS_OVERLAP_MAX_RATIO=0.5）时，不要放任它继续半压半露——直接把
    压住的那一段钳掉。钳完之后 clip 与 OP/ED 的重叠比例总是 0，_reject_reason
    的 OP/ED 分支只会在「clip 整段都在片头/片尾曲内」（钳完时长非正）时触发，
    这正是我们想要的：能救的救，救不了的按现有的「时长非正」判据丢弃，不
    需要新增判据。"""
    if track.op_range is not None and clip.start < track.op_range[1]:
        clip.start = max(clip.start, track.op_range[1])
    if track.ed_range is not None and clip.end > track.ed_range[0]:
        clip.end = min(clip.end, track.ed_range[0])
```

- [ ] **Step 2: 在 `repair_script` 里加 credits 缺失告警的去重集合**

找到（`validate.py`，紧跟在 `warnings: list[str] = []` 之后，Task 1 加的体质检测代码之后）：

```python
    reported_missing: set[int] = set()
```

在它旁边加两个新集合：

```python
    reported_missing: set[int] = set()
    reported_missing_credits_op: set[int] = set()
    reported_missing_credits_ed: set[int] = set()
```

- [ ] **Step 3: 在逐 clip 循环里插入 credits 检查与钳位调用**

找到 Task 2 改完之后的这一段（循环体开头，`track = tracks.get(clip.episode)` 判断 None 之后）：

```python
            index = indexes[clip.episode]
            matches = index.matches(clip.anchor_lines)
```

在它之前插入：

```python
            if track.op_range is None and clip.episode not in reported_missing_credits_op:
                warnings.append(
                    f"第 {clip.episode} 集片头曲区间检测失败（op_range 为空），"
                    f"OP 重叠检查对这一集形同虚设"
                )
                reported_missing_credits_op.add(clip.episode)
            if track.ed_range is None and clip.episode not in reported_missing_credits_ed:
                warnings.append(
                    f"第 {clip.episode} 集片尾曲区间检测失败（ed_range 为空），"
                    f"ED 重叠检查对这一集形同虚设"
                )
                reported_missing_credits_ed.add(clip.episode)
```

然后找到（走廊判断之后）：

```python
            reason = _reject_reason(clip, track, cfg)
```

改成：

```python
            _clamp_credits_overlap(clip, track)
            reason = _reject_reason(clip, track, cfg)
```

- [ ] **Step 4: 新增测试**

在 Task 2 新增的走廊测试之后插入：

```python
def test_missing_op_or_ed_range_warns_once_per_episode():
    """op_range/ed_range 检测失败时曾经完全静默（_credits_overlap_ratio 直接
    返回 0.0，OP/ED 重叠检查形同虚设却没人知道）。现在两者都要各报一次，且
    多个 clip 引用同一集时不重复报。"""
    track = make_track(op=None, ed=None)
    s = make_script(
        [[clip(300.0, 306.0), clip(320.0, 326.0)]]
    )  # 两个 clip，同一集，都没 anchor
    result = run(s, track=track)
    op_warnings = [w for w in result.warnings if "片头曲区间检测失败" in w]
    ed_warnings = [w for w in result.warnings if "片尾曲区间检测失败" in w]
    assert len(op_warnings) == 1
    assert len(ed_warnings) == 1


def test_partial_ed_overlap_is_clamped_instead_of_dropped():
    """clip 压在 ED 上但比例只有 (515-500)/20=75%？不对，改小一点：clip 时长
    20 秒，压 15 秒进 ED，ratio=75%>=50% 本来就会被 _reject_reason 丢弃，不能
    验证钳位。这里故意把 clip 卡在刚好会被钳完之后完全脱离 ED 的位置：
    clip(495.0, 515.0)，ed_range=(500.0, 520.0)，重叠 15 秒/20 秒=75%——钳位在
    _reject_reason 之前跑，钳完 clip 变成 (495.0, 500.0)，与 ED 零重叠，
    _reject_reason 不会再把它当 ED 重叠丢弃。"""
    track = make_track(duration=600.0, op=(0.0, 5.0), ed=(500.0, 520.0))
    s = make_script([[clip(495.0, 515.0)]])
    result = run(s, track=track)
    kept = result.script.beats[0].clips
    assert len(kept) == 1
    assert kept[0].start == pytest.approx(495.0)
    assert kept[0].end == pytest.approx(500.0)
    assert not any("片尾曲内" in w for w in result.warnings)
```

- [ ] **Step 5: 跑测试**

Run: `uv run pytest tests/test_validate.py -q`
Expected: 全部 PASS（含既有的 B5 OP/ED 重叠比例测试，`test_validate.py:524-543`，因为它们测的是 ratio>=0.5 的整体丢弃场景，clamp 只处理 ratio<0.5 部分重叠，两者不冲突）。

- [ ] **Step 6: Commit**

```bash
git add src/tenmin/script/validate.py tests/test_validate.py
git commit -m "feat(validate): OP/ED 检测失败不再静默；部分压 credits 的 clip 改钳位不丢弃"
```

---

### Task 4: hold 金句模糊定位（Section 5）

**Files:**
- Modify: `src/tenmin/script/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes：既有 `_normalize_quote(text: str) -> str`（不改）；既有常量 `QUOTE_FRAGMENT_MIN_CHARS`/`QUOTE_FRAGMENT_MIN_RATIO`（不改）。
- Produces：`_quote_matches` 签名不变（`tracks, episodes, quote) -> list[DialogueLine]`），行为扩展：精确/子串匹配全部失败时，退回模糊匹配拿最佳单条结果。新常量 `_QUOTE_SIMILARITY_MIN_RATIO = 0.6`。

- [ ] **Step 1: 在 `validate.py` 顶部加 `import difflib`**

找到 import 区（`validate.py` 第 15-29 行附近），加一行：

```python
import difflib
```

- [ ] **Step 2: 在 `QUOTE_FRAGMENT_MIN_RATIO` 常量旁加新常量**

```python
QUOTE_FRAGMENT_MIN_CHARS = 4
QUOTE_FRAGMENT_MIN_RATIO = 0.6
# 精确子串匹配失败时的模糊匹配兜底阈值。只在子串匹配全部落空时才启用——
# 不想让「大致像」的误报盖过明确的精确匹配。0.6 与既有的
# QUOTE_FRAGMENT_MIN_RATIO 同值，都是「六成像就算数」的经验值。
_QUOTE_SIMILARITY_MIN_RATIO = 0.6
```

- [ ] **Step 3: 重写 `_quote_matches`**

找到（`validate.py:205-237`，现行完整版本）：

```python
def _quote_matches(
    tracks: dict[int, DialogueTrack], episodes: list[int], quote: str
) -> list[DialogueLine]:
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
```

改成：

```python
def _quote_matches(
    tracks: dict[int, DialogueTrack], episodes: list[int], quote: str
) -> list[DialogueLine]:
    target = _normalize_quote(quote)
    if not target:
        return []
    minimum = max(QUOTE_FRAGMENT_MIN_CHARS, QUOTE_FRAGMENT_MIN_RATIO * len(target))
    found: list[DialogueLine] = []
    best_fuzzy: tuple[float, DialogueLine] | None = None
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
                continue
            ratio = difflib.SequenceMatcher(None, target, line).ratio()
            if ratio >= _QUOTE_SIMILARITY_MIN_RATIO and (
                best_fuzzy is None or ratio > best_fuzzy[0]
            ):
                best_fuzzy = (ratio, ln)
    if found:
        return found
    return [best_fuzzy[1]] if best_fuzzy is not None else []
```

- [ ] **Step 4: 新增测试**

在既有 B9 quote 匹配测试组（`test_validate.py:861-895`）之后插入：

```python
def test_quote_matches_via_fuzzy_fallback_when_exact_substring_fails():
    """字幕行比金句多插了一个字，不构成子串关系（既不是 target 的子串，也不
    是反过来包含 target），精确匹配全部落空。difflib 在 14 个字里差 1 个字的
    相似度接近 0.93，远超 0.6 的门槛，应该被模糊匹配捞回来。"""
    tracks = {2: make_track(lines=[dline(1, 50.0, 53.0, text="以为你对我们这种普通人没有兴趣呢")])}
    matches = _quote_matches(tracks, [2], "以为你对我们这种普通人没兴趣呢")
    assert len(matches) == 1
    assert matches[0].idx == 1


def test_quote_matches_fuzzy_fallback_rejects_unrelated_text():
    """完全不相关的文本，相似度远低于 0.6，不该被模糊匹配硬凑。"""
    tracks = {2: make_track(lines=[dline(1, 50.0, 53.0, text="今天天气不错")])}
    matches = _quote_matches(tracks, [2], "以为你对我们这种普通人没兴趣呢")
    assert matches == []


def test_quote_matches_prefers_exact_over_fuzzy():
    """精确匹配存在时，即便有更高相似度的模糊候选，也只返回精确匹配的那批。"""
    tracks = {
        2: make_track(
            lines=[
                dline(1, 50.0, 53.0, text="以为你对我们这种普通人没兴趣呢"),
                dline(2, 60.0, 63.0, text="以为你对我们这种普通人没有兴趣呢"),
            ]
        )
    }
    matches = _quote_matches(tracks, [2], "以为你对我们这种普通人没兴趣呢")
    assert [m.idx for m in matches] == [1]
```

- [ ] **Step 5: 跑测试**

Run: `uv run pytest tests/test_validate.py -q`
Expected: 全部 PASS（含既有 B9 组 4 个测试——它们全部能被现有精确子串判据命中，不会滑到 fuzzy 分支，行为不变）。

- [ ] **Step 6: Commit**

```bash
git add src/tenmin/script/validate.py tests/test_validate.py
git commit -m "feat(validate): hold 金句定位加 difflib 模糊匹配兜底，不做自动扩窗"
```

---

### Task 5: warning 喂回重试 prompt（Section 6）

**Files:**
- Modify: `src/tenmin/script/single.py`
- Test: `tests/test_single.py`

**Interfaces:**
- Consumes：`tenmin.script.validate.check_script(script, tracks, reports, *, cfg, rate) -> list[str]`（纯函数，不抛异常，第 459-487 行，Task 1-4 均未改它的签名）；既有 `ScriptValidationError.script: Script | None`（`validate.py:89-101`）。
- Produces：不新增公开接口，只改 `generate_script` 内部重试循环拼 prompt 的方式。

- [ ] **Step 1: 在 `single.py` 顶部 import 里加 `check_script`**

找到：

```python
from tenmin.script.validate import ScriptValidationError, validate_script
```

改成：

```python
from tenmin.script.validate import ScriptValidationError, check_script, validate_script
```

- [ ] **Step 2: 改写重试循环的 `except` 块**

找到（`single.py:318-340` 附近，现行完整版本）：

```python
    while True:
        attempt += 1
        try:
            script, stage_warnings = await draft(prompt, label)
            break
        except ScriptValidationError as error:
            if attempt >= attempts:
                raise
            warnings.append(f"第 {attempt} 轮剧本校验失败，重试：{error}")
            label = f"校验失败，第 {attempt} 次重试"
            prompt = (
                _followup_prompt(
                    followup_base,
                    error.script,
                    "上一轮的问题",
                    f"{error}\n请在上一版基础上修掉这个问题，"
                    f"确保每个节点至少有一个有效 clip，且所有时间戳都落在正片范围内。",
                )
                if error.script is not None
                else f"{followup_base}\n\n## 上一轮的问题\n\n{error}\n请重新输出。"
            )
```

改成：

```python
    while True:
        attempt += 1
        try:
            script, stage_warnings = await draft(prompt, label)
            break
        except ScriptValidationError as error:
            if attempt >= attempts:
                raise
            warnings.append(f"第 {attempt} 轮剧本校验失败，重试：{error}")
            label = f"校验失败，第 {attempt} 次重试"
            stage_issues = (
                check_script(error.script, tracks, reports, cfg=cfg.validate_script, rate=rate)
                if error.script is not None
                else []
            )
            extra = ""
            if stage_issues:
                issues = "\n".join(f"- {w}" for w in stage_issues)
                extra = f"\n\n此外还有以下问题（尽量一并修掉）：\n{issues}"
            prompt = (
                _followup_prompt(
                    followup_base,
                    error.script,
                    "上一轮的问题",
                    f"{error}{extra}\n请在上一版基础上修掉这个问题，"
                    f"确保每个节点至少有一个有效 clip，且所有时间戳都落在正片范围内。",
                )
                if error.script is not None
                else f"{followup_base}\n\n## 上一轮的问题\n\n{error}{extra}\n请重新输出。"
            )
```

> `check_script` 是纯函数、不抛异常，`error.script` 是 `repair_script` 抛错时**已经部分修复**的那份（坏 clip 已按 Task 2/3 的新逻辑处理过），在这上面跑 `check_script` 是安全的，拿到的是"这份半成品剧本还有哪些结构性问题"。

- [ ] **Step 3: 新增测试**

在 `tests/test_single.py` 的 `test_retry_sends_the_previous_draft_back`（第 443-455 行）之后插入：

```python
@pytest.mark.asyncio
async def test_retry_prompt_includes_check_script_warnings_from_the_rejected_draft(
    cfg, track, report
):
    """b1 整个 beat 的唯一 clip 越界（9000-9080，远超 track.duration=1416.6），
    repair_script 会在处理到 b1 时因 kept 为空直接抛错，此时 b2 还没被处理过，
    它的 clip 带着一个「anchor 行完全落在 clip 窗口外」的毛病（Warning5，
    _check_anchor_coverage）留在 error.script 里。这条 warning 以前从来没有
    进过重试 prompt，现在应该被 check_script 捞出来一起交给模型。"""
    bad = LLMScript(
        beats=[
            llm_beat("b1", "Hook 开场", "hook", 360, 9000.0, 9080.0),
            llm_beat("b2", "阶段一", "act", 360, 300.0, 380.0, anchors=[1]),  # anchor@7.12-11.48，与窗口完全不重叠
            llm_beat("b3", "阶段二", "act", 360, 700.0, 780.0),
            llm_beat("b4", "收尾：完", "outro", 360, 900.0, 980.0),
        ]
    )
    provider = FakeProvider([bad, valid_llm_script()])
    await generate_script(cfg, track, report, provider)
    retry_prompt = provider.calls[1]["user"]
    assert "此外还有以下问题" in retry_prompt
    assert "缺画面" in retry_prompt


@pytest.mark.asyncio
async def test_retry_prompt_has_no_extra_block_when_check_script_finds_nothing(
    cfg, track, report
):
    """回归：`bad` 草稿本身没有触发任何 check_script 问题时，不应该无故多出
    一个空的「此外还有以下问题」块。"""
    bad = LLMScript(
        beats=[
            llm_beat("b1", "Hook 开场", "hook", 360, 9000.0, 9005.0),
            llm_beat("b2", "阶段一", "act", 360, 100.0, 180.0),
            llm_beat("b3", "收尾：完", "outro", 360, 300.0, 380.0),
        ]
    )
    provider = FakeProvider([bad, valid_llm_script()])
    await generate_script(cfg, track, report, provider)
    retry_prompt = provider.calls[1]["user"]
    assert "此外还有以下问题" not in retry_prompt
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_single.py -q`
Expected: 全部 PASS（含既有 `test_retry_sends_the_previous_draft_back`——它的 `bad` fixture 没有 anchor，`check_script` 对它跑出来是空列表，`extra=""`，原有断言不受影响）。

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/script/single.py tests/test_single.py
git commit -m "feat(single): 校验失败重试时把 check_script 的 warning 一并喂给模型"
```

---

### Task 6: 全量回归 + 真实语料人工验收 + 收尾

**Files:**
- Test: 无新增文件，跑既有全量测试；额外写一个**不提交**的临时验证脚本（放 `/tmp`，跑完删除）。

**Interfaces:**
- Consumes：Task 1-5 的全部改动。
- Produces：无新代码，本 Task 是验收关卡。

- [ ] **Step 1: 跑全量测试套件**

Run: `uv run pytest -q`
Expected: 全部 PASS（不跑 `-m llm`/`-m generalize`/`-m render`，那三类需要真实凭据/语料/ffmpeg，本次改动不涉及）。

- [ ] **Step 2: 跑 ruff**

Run: `uv run ruff check src/tenmin/script/validate.py src/tenmin/script/single.py tests/test_validate.py tests/test_single.py`
Expected: 无报错。若有格式问题，跑 `uv run ruff format` 后重新跑一次测试确认没改坏行为。

- [ ] **Step 3: 人工验收——用真实语料确认走廊检验的效果**

这一步**不写进仓库**，因为语料在 `.gitignore` 里的 `work/` 目录，不随仓库分发；只是本地验证今天改的代码对着真实数据表现如预期。写一个临时脚本（放 `/tmp/verify_corridor.py`），逐一读取 `work/{saijo,akujo2,saijo2}` 下所有 `03_script/*.script.json` 与对应 `01_dialogue/*.dialogue.json`、`02_signals/*.signals.json`，用改完的 `repair_script` 重跑一遍，打印每份剧本的 (a) 是否抛出体质检测错误 (b) 有多少 clip 被按 anchor 重建 (c) 有多少 clip 被判定走廊外丢弃。

验收标准（对照设计文档 Section 2 的实测表）：
- `saijo/E02` 应该抛出 `ScriptValidationError`，message 包含"行号/时间体系不同源"。
- `saijo2/E01` 应该有 8 处重建、1 处丢弃（原来 drift=-320.8 那个 `269.0-346.0` 的 clip）。
- 其余 9 份（`saijo/E01`、`saijo/E03`-`E10`、`akujo2`）应该**零改动**——即 warnings 列表里不出现"走廊"、"没有覆写"、"以字幕时间为准"这几个关键词，clip 时间戳与改动前落盘的 `03_script/*.script.json` 逐一相等。

跑完脚本、人工核对输出符合以上三条后，删除 `/tmp/verify_corridor.py`（不提交，Global Constraints 里已经说明原因）。如果某一份语料的结果与预期不符，回到对应 Task 补测试用例、修代码，不要在这一步直接改语料或悄悄放宽验收标准。

- [ ] **Step 4: 确认 6 个改动与设计文档的 7 条 Q&A 逐一对上**

过一遍 `docs/superpowers/specs/2026-09-16-anchor-corridor-repair-design.md` 的"已确认的设计决策"清单，确认：
1. 走廊检验取代 60 秒硬上限 —— Task 2 ✓
2. 离群 clip 直接丢弃+发 warning（不重建、不抛错）—— Task 2 Step 4 的 `continue` 分支 ✓
3. 体质检测阈值 0.25，不进 config —— Task 1 ✓
4. hold 定位只提算法不自动扩窗 —— Task 4（`_quote_matches` 只返回匹配行，没有改任何时间戳）✓
5. Warning 6 本次只喂回重试 prompt，不拆句/不放宽 `max_lines` —— Task 5，且 Task 5 完全没有touch `render/subtitles.py` ✓
6. credits 检测算法本次不修只发 warning —— Task 3 Step 2-3（`op_range is None`/`ed_range is None` 分支只 append warning，没有改 `ingest/credits.py`）✓
7. ED 重叠钳位不改常量 —— Task 3 Step 1（`_clamp_credits_overlap` 新增，`CREDITS_OVERLAP_MAX_RATIO` 原样保留）✓

- [ ] **Step 5: 最终 commit（如果 Step 2 的 ruff format 产生了额外改动）**

```bash
git status
# 如果 ruff format 改了东西：
git add -u
git commit -m "style: ruff format 收尾"
```

---

## 执行完成后

6 个 Task 全部走完、Step 全部打勾、Task 6 的人工验收三条标准全部核对通过后，这份计划视为完成。设计文档里"不在本次范围"的四项（OP credits 检测算法本身、edge-tts SentenceBoundary、`saijo2` slug 命名错误、Warning 6 的拆句/放宽行数）留给后续独立立项，不在本计划的验收范围内。
