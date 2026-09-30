# clip 拉伸比例触发返工（C-2）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `repair_script` 里新增一条比 `stretch_max`（4.0）更紧的门槛 `retry_stretch_max`（默认 1.5），扫完整篇剧本、收集全部拉伸过长的节点，一次性抛 `ScriptValidationError` 触发整篇重试（跟 `max_holds` 同一条路），不动现有 `_check_footage_budget` 的 warning 逻辑。

**Architecture:** `ValidateConfig` 新增一个字段；`repair_script` 在现有 `max_holds` 检查之后追加一段结构相同的聚合检查，复用 `_check_footage_budget` 已经在用的 `beat_clip_seconds`/`beat_seconds`。**这个默认阈值会跟 `tests/test_validate.py` 里贯穿全文件的「narration 40 字 + 5 秒 clip」这个共享 fixture 惯例正面相撞**（40 字 ≈ 8.9 秒旁白 / 5 秒画面 = 1.78 倍，超过默认 1.5）——本计划已经在实际跑过全量测试的基础上把这处 fixture 惯例改到安全区间，具体改法见 Task 2。

**Tech Stack:** Python 3.14、pytest、pydantic。

---

## 文件结构

- 改 `src/tenmin/config.py`：`ValidateConfig` 新增 `retry_stretch_max` 字段。
- 改 `src/tenmin/script/validate.py`：`repair_script` 新增聚合检查。
- 改 `tests/test_config.py`：现有的 `test_quality_knob_defaults`/`test_quality_knobs_reject_out_of_bounds` 两个测试各加一行/一个参数化用例。
- 改 `tests/test_validate.py`：`make_script`/`timeline_script`/`structure_script` 三处共享 fixture 的默认旁白字数收紧；`run()` 测试助手新增可选 `cfg` 参数；6 个既有测试改用这个新参数或改调 `check_script`；新增 8 个测试覆盖 spec §4 的全部要求。
- 改 `tests/test_single.py`：1 个既有测试的 fixture 参数微调（跟新检查的默认阈值撞车）。
- 改 `tests/test_e2e.py`：`project` fixture 追加一个 `validate_script.retry_stretch_max` 覆盖项（这个文件测的是 v1 链路接线，不是画面/旁白预算，本来就该跟这条新阈值无关）。

---

### Task 1: `ValidateConfig.retry_stretch_max` 字段

**Files:**
- Modify: `src/tenmin/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_config.py` 里找到这两个既有测试（约行 211-238），改成：

```python
def test_quality_knob_defaults():
    assert ValidateConfig().hold_clip_max_gap_seconds == 3.0
    assert ValidateConfig().retry_stretch_max == 1.5
    cfg = RenderConfig()
    assert (
        cfg.subtitle_soft_max_chars,
        cfg.hold_relative_lu,
        cfg.hold_gain_max_db,
        cfg.hold_silence_floor_lufs,
        cfg.hold_fade_seconds,
    ) == (36, 3.0, 6.0, -40.0, 0.1)
    assert (cfg.loudness_i, cfg.loudness_tp, cfg.loudness_lra) == (-14.0, -1.5, 11.0)


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (ValidateConfig, "hold_clip_max_gap_seconds", -1),
        (ValidateConfig, "retry_stretch_max", 0),
        (RenderConfig, "subtitle_soft_max_chars", 0),
        (RenderConfig, "hold_relative_lu", -1),
        (RenderConfig, "hold_gain_max_db", -1),
        (RenderConfig, "hold_silence_floor_lufs", 1),
        (RenderConfig, "hold_fade_seconds", -1),
        (RenderConfig, "loudness_i", 0),
        (RenderConfig, "loudness_tp", 1),
        (RenderConfig, "loudness_lra", 0),
    ],
)
def test_quality_knobs_reject_out_of_bounds(model, field, value):
    with pytest.raises(ValidationError):
        model.model_validate({field: value})
```

- [ ] **Step 2: 跑测试确认失败**

```bash
/Users/portz/.local/bin/uv run pytest tests/test_config.py -k "quality_knob" -v
```

预期：`test_quality_knob_defaults` 报 `AttributeError: 'ValidateConfig' object has no attribute 'retry_stretch_max'`；新的参数化用例 `[ValidateConfig-retry_stretch_max-0]` 同样报 `AttributeError`（`model_validate` 收到未知字段？不——`extra="forbid"` 会先报 `ValidationError`，但字段本身不存在，pydantic 会把 `retry_stretch_max` 当成未知字段报错，这也是一种 `ValidationError`，测试会**误绿**）。为确保 Step 1 是真失败，先只跑 `test_quality_knob_defaults` 确认它是 `AttributeError`（真失败）；第二个测试因为 `StrictModel(extra="forbid")` 会把不存在的字段当非法输入报 `ValidationError`，恰好符合 `pytest.raises(ValidationError)` 的预期，**不会先失败**——这一点要在 Step 4 之后用「删掉 Step 3 的实现、确认这条用例改成检查一个真实存在的字段类型错误时会挂」的方式反向确认（或者更简单：只把 `test_quality_knob_defaults` 当作本步骤唯一需要先见红的断言，第二个测试本身在字段存在之后才有意义地测「gt=0 拒绝 0」）。

- [ ] **Step 3: 实现**

编辑 `src/tenmin/config.py`，在 `ValidateConfig` 类里找到：

```python
    stretch_max: float = Field(default=4.0, gt=0)
    stretch_min: float = Field(default=0.125, gt=0)
```

改成：

```python
    stretch_max: float = Field(default=4.0, gt=0)
    stretch_min: float = Field(default=0.125, gt=0)
    # 比 stretch_max 更紧的门槛：超过它值得为这个节点重跑一次 LLM（走
    # validation_retries，跟 max_holds 同一条路），而不只是发个 warning。
    # stretch_max（4.0）依然是"数量级配错"的宽松上限，两条阈值刻意分层、互不影响。
    #
    # 取值依据真实缺陷案例（B 阶段人工评审 akujo E11 / saijo E02 成片）：akujo 28 段
    # clip 里 7 段拉伸超过 1.5 倍（最差 2.62 倍，画面被钳到片尾）；saijo beat7 拉伸
    # 1.88 倍。这些案例全部落在现有 stretch_max=4.0 之内、从未触发过 warning，
    # 说明"数量级配错"阈值拦不住真实可见的画面变形。
    #
    # 只拦拉伸过长方向：画面过剩被截断（stretch 过小）目前没有真实可见缺陷证据，
    # 继续只发 warning。
    retry_stretch_max: float = Field(default=1.5, gt=0)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
/Users/portz/.local/bin/uv run pytest tests/test_config.py -k "quality_knob" -v
```

预期：两个测试全部通过。

- [ ] **Step 5: 跑全量测试确认无副作用**

```bash
/Users/portz/.local/bin/uv run pytest tests/ -q
```

预期：与本任务开始前的基线一致（无新增失败）。这一步之所以不会有 fallout：`retry_stretch_max` 只是新增了一个字段，还没有任何代码读它，不改变任何现有行为。

- [ ] **Step 6: 提交**

```bash
/Users/portz/.local/bin/rtk git add src/tenmin/config.py tests/test_config.py
/Users/portz/.local/bin/rtk git commit -m "feat: ValidateConfig 新增 retry_stretch_max 字段"
```

---

### Task 2: `repair_script` 聚合检查 + 修复既有测试的 fixture 撞车 + 新增全部验收测试

**这是本计划里唯一有真实风险的一步**：默认 `retry_stretch_max=1.5` 会跟 `tests/test_validate.py` 里三处共享 fixture 助手（`make_script`/`timeline_script`/`structure_script`）以及 `tests/test_single.py`/`tests/test_e2e.py` 里各一处既有 fixture 撞车——这些 fixture 长期用「narration 40 字（≈8.9 秒）配 5 秒 clip」这个组合（stretch=1.78），此前从未触发任何检查（旧的 `stretch_max=4.0` 更宽），新增的 1.5 门槛会让它们全部命中。**下面每一处改动都已经在这个 worktree 里实际跑过 `uv run pytest tests/ -q` 验证到全绿（2242 passed, 32 skipped），不是理论推演**——照抄即可。

**Files:**
- Modify: `src/tenmin/script/validate.py`
- Modify: `tests/test_validate.py`
- Modify: `tests/test_single.py`
- Modify: `tests/test_e2e.py`

#### Step 1: 写失败测试（新增的 8 个验收测试 + 3 处共享 fixture 收紧）

在 `tests/test_validate.py` 里，先做三处**必须先动**的 fixture 收紧（否则下面新加的测试和大量既有测试会因为共享 fixture 的默认拉伸比例撞车而报错，跟本任务要验证的逻辑无关）：

**1a. `make_script`** 里（约行 71-102），把：

```python
                narration="旁白" * 20,
```

改成：

```python
                narration="旁白" * 15,
```

**1b. `timeline_script`**（约行 736-750 附近，用于 A1 时间轴单调性检查的独立 fixture 工厂）里同一行 `narration="旁白" * 20,` 也改成 `narration="旁白" * 15,`。

**1c. `structure_script`**（约行 950-970 附近，用于 A2 结构检查的独立 fixture 工厂）里同一行 `narration="旁白" * 20,` 也改成 `narration="旁白" * 15,`。

> 三处都是「30 字 ≈ 6.7 秒旁白」配「5 秒/10 秒 clip」，改完之后 stretch 落在 0.67–1.33 之间，安全落在新旧两条阈值（1.5 / 4.0）之内，不会污染任何跟拉伸比例无关的测试。

**1d.** 给 `run()` 测试助手（约行 105-110）新增一个可选 `cfg` 参数：

```python
def run(script, track=None, report=None, cfg=None):
    return validate_script(
        script,
        tracks={2: track or make_track()},
        reports={2: report or make_report()},
        cfg=cfg or DEFAULT_VALIDATE,
    )
```

**1e.** 有 4 个既有测试的 fixture 恰好构造出 stretch > 1.5（它们测的是别的东西——anchor 覆写钳到集尾、静音高光重叠阈值、`min_clip_seconds` 边界、`max_holds` 边界——拉伸比例只是这些场景的副产物）。给它们各传一个宽松的 `cfg`，不改它们原本要验证的逻辑：

`test_anchor_overwrite_clamped_to_episode_end`（约行 195-202）：

```python
def test_anchor_overwrite_clamped_to_episode_end():
    # 幅度 46 秒，落在 ANCHOR_OVERWRITE_MAX_SECONDS 之内，覆写照做。
    track = make_track(duration=1000.0, lines=[dline(42, 996.0, 999.0)])
    s = make_script([[clip(950.0, 980.0, anchors=[42])]])
    result = run(s, track=track, cfg=ValidateConfig(retry_stretch_max=100.0))
    kept = result.script.beats[0].clips[0]
    assert kept.end <= 1000.0
    assert kept.end > kept.start
```

`test_silent_highlight_needs_one_second_overlap`（约行 223-227）：

```python
def test_silent_highlight_needs_one_second_overlap():
    # 与间隙只重叠 0.633 秒。clip 本身给足 3 秒，避免撞上 min_clip_seconds。
    s = make_script([[clip(1326.0, 1329.0, silent=False)]])
    result = run(
        s,
        report=make_report(gaps=[(1328.367, 1348.18)]),
        cfg=ValidateConfig(retry_stretch_max=100.0),
    )
    assert result.script.beats[0].clips[0].is_silent_highlight is False
```

`test_clip_at_exactly_the_minimum_is_kept`（约行 724-726）：

```python
def test_clip_at_exactly_the_minimum_is_kept():
    s = make_script([[clip(50.0, 50.0 + DEFAULT_VALIDATE.min_clip_seconds)]])
    cfg = ValidateConfig(retry_stretch_max=100.0)
    assert len(run(s, cfg=cfg).script.beats[0].clips) == 1
```

`test_holds_at_the_cap_do_not_raise`（约行 1091-1101，注意这里加的 8 处 hold 每处 2 秒，会把这个节点的跨度从旁白的 6.7 秒推到 22.7 秒，footage 仍是 5 秒，stretch≈4.5）：

```python
def test_holds_at_the_cap_do_not_raise():
    """边界：恰好 max_holds 处不判错。上限取 8 而不是提示词原来写的 6，是因为
    tests/snapshots/saijo_e02.script.json（当质量基准用的样本）本身有 8 处 ——
    按 6 判错第一个被拦的就是自家黄金快照。提示词已同步改成 3–8。"""
    s = make_script([[clip(10.0, 15.0)]])
    s.beats[0].audio.holds = [
        Hold(at=1.0 + i, duration=2.0, quote="台词")
        for i in range(DEFAULT_VALIDATE.max_holds)
    ]
    result = run(
        s,
        track=make_track(lines=[dline(1, 11, 12)]),
        cfg=ValidateConfig(retry_stretch_max=100.0),
    )
    assert len(result.script.beats[0].audio.holds) == DEFAULT_VALIDATE.max_holds
```

**1f.** 有 2 个既有测试专门测 `_check_footage_budget` 自己的 warning 文案，它们构造的拉伸比例本来就故意超过 `stretch_max`（4.0），也必然超过新的 1.5；这两个测试的意图是验证 `check_script` 这条纯读路径的 warning 文案，不是验证 `repair_script`/`check_script` 谁先跑，所以改成直接调 `check_script`（绕开 `repair_script` 的新硬检查），跟 `test_footage_budget_uses_the_render_layer_clip_sum` 这个既有测试用的是同一个调用方式：

`test_beat_with_far_too_little_footage_warns`（约行 817-823）：

```python
def test_beat_with_far_too_little_footage_warns():
    """clip 太少时 render/timeline.py 会按 ratio = 旁白/画面 把每个 clip 往后延长，
    延到超出源片长再钳到片尾，成片画面错位。

    直接调 check_script（纯读），不走 run()/validate_script：这个比例本来就超过了
    retry_stretch_max（1.5），repair_script 会先抢着报错，这里要单独验证的是
    check_script 自己的这条 warning 文案，不是两者的先后关系（那条在
    repair_script 的测试里单独盖）。
    """
    # make_script 的旁白是 30 字 ≈ 6.7 秒；1.6 秒画面 → 拉伸 4.2 倍
    s = make_script([[clip(10.0, 11.6)]])
    warnings = check_script(s, {2: make_track()}, {2: make_report()})
    hits = [w for w in warnings if "画面只有" in w]
    assert len(hits) == 1, warnings
```

`test_beat_within_the_measured_stretch_range_does_not_warn`（约行 832-839）：

```python
def test_beat_within_the_measured_stretch_range_does_not_warn():
    """实测 85 个真实 beat 的 画面/旁白 比值落在 0.396–5.176（拉伸 0.19–2.53 倍），
    这整段区间都必须放过。

    直接调 check_script：ratio=0.40 时拉伸 2.5 倍，超过 retry_stretch_max（1.5），
    走 run()/validate_script 会被 repair_script 先抢着报错，这里要单独验证的是
    check_script 自己在 stretch_max/stretch_min 这两条更宽的边界内不报，跟新的
    retry 检查无关。
    """
    s = make_script([[clip(10.0, 25.0)]])
    seconds = beat_seconds(s.beats[0])
    for ratio in (0.40, 1.0, 5.0):
        one = make_script([[clip(10.0, 10.0 + seconds * ratio)]])
        warnings = check_script(one, {2: make_track()}, {2: make_report()})
        assert [w for w in warnings if "画面" in w] == [], ratio
```

**1g.** 现在在 `tests/test_validate.py` 的「A3：画面总时长 vs 旁白时长」小节末尾（`test_stretch_bounds_constants` 之后、`test_footage_budget_uses_the_render_layer_clip_sum` 之前，即约行 845 附近）新增 8 个测试：

```python
def test_single_overstretched_beat_raises_for_retry():
    """stretch 超过 retry_stretch_max（1.5）但仍在 stretch_max（4.0）之内——只有
    新检查会拦，check_script 的旧 warning 不会跑到（repair_script 先抢着报错）。"""
    s = make_script([[clip(10.0, 13.0)]])  # 30 字 ≈ 6.667 秒旁白 / 3 秒画面 = 2.2 倍
    with pytest.raises(ScriptValidationError) as exc:
        run(s)
    assert "拉伸变形" in str(exc.value)
    assert s.beats[0].label in str(exc.value)
    assert "2.2 倍" in str(exc.value)
    assert exc.value.script is not None


def test_multiple_overstretched_beats_are_all_listed_in_one_message():
    """跟 max_holds 同一个聚合模式：扫完整篇再报一次，一条消息列出全部违规节点。"""
    s = make_script([[clip(10.0, 13.0)], [clip(100.0, 102.0)]])
    # 节点1：6.667/3=2.2 倍；节点2：6.667/2=3.3 倍。两个都超过 1.5。
    with pytest.raises(ScriptValidationError) as exc:
        run(s)
    message = str(exc.value)
    assert s.beats[0].label in message
    assert s.beats[1].label in message
    assert "2 个节点" in message


def test_stretch_at_exactly_the_retry_threshold_does_not_raise():
    """边界：恰好 retry_stretch_max 处不判错（严格 >，不是 >=）。"""
    s = make_script([[clip(10.0, 14.0)]])
    s.beats[0].narration = "旁" * 27  # 27 字 / 4.5 = 6.0 秒，footage=4.0 秒 → 恰好 1.5 倍
    result = run(s)  # 不应该抛错
    assert result.script.beats[0].clips[0].end == pytest.approx(14.0)


def test_lowering_stretch_max_alone_does_not_trigger_the_retry_check():
    """两条阈值各自独立比较，不是嵌套关系：把 stretch_max 调到低于默认的
    retry_stretch_max，只会让旧 warning 先报，不会让新检查提前触发。"""
    s = make_script([[clip(10.0, 15.0)]])
    s.beats[0].narration = "旁" * 27  # 6.0 秒 / 5.0 秒 = 1.2 倍
    cfg = ValidateConfig(stretch_max=1.0)  # 低于 retry_stretch_max 默认值 1.5
    result = run(s, cfg=cfg)  # 1.2 <= 1.5，新检查不触发，不应该抛错
    assert any("画面只有" in w for w in result.warnings)  # 1.2 > stretch_max(1.0)，旧检查报


def test_understretched_beats_never_trigger_the_retry_check():
    """只拦拉伸过长方向：画面过剩（stretch 很小）无论多小都不触发新检查。"""
    s = make_script([[clip(10.0, 800.0)]])  # 790 秒画面 vs 6.667 秒旁白，stretch≈0.0084
    result = run(s)  # 不应该抛错
    assert any("画面多达" in w for w in result.warnings)  # 旧 warning 仍然照常报


def test_empty_narration_beat_does_not_crash_the_retry_check():
    """span<=0（空旁白、无留白）时新检查要跳过，不报错、不崩。"""
    s = make_script([[clip(10.0, 15.0)]])
    s.beats[0].narration = ""
    result = run(s)  # 不应该抛错、不应该除零
    assert result.script.beats[0].clips[0].end == pytest.approx(15.0)


def test_overstretched_error_carries_the_repaired_script():
    """跟 max_holds/clip 全灭两条既有失败路径一致：抛出的错误带着修到一半的那份，
    不是原始输入。"""
    s = make_script([[clip(10.0, 13.0)]])
    before = s.model_dump_json()
    with pytest.raises(ScriptValidationError) as exc:
        run(s)
    assert exc.value.script is not None
    assert exc.value.script.beats[0].label == s.beats[0].label
    assert s.model_dump_json() == before  # 入参一字不动


def test_validate_script_aborts_before_check_script_runs_for_overstretched_beats():
    """端到端：跟 test_too_many_holds_raises 同构——repair_script 先抛错，
    check_script 的画面/旁白 warning 根本不会被跑到。"""
    s = make_script([[clip(10.0, 13.0)]])
    with pytest.raises(ScriptValidationError):
        validate_script(s, {2: make_track()}, {2: make_report()})
```

- [ ] **Step 2: 跑测试确认新测试失败**

```bash
/Users/portz/.local/bin/uv run pytest tests/test_validate.py -k "overstretched or retry_threshold or lowering_stretch_max or understretched or empty_narration_beat" -v
```

预期：全部 `AttributeError: 'ValidateConfig' object has no attribute 'retry_stretch_max'`——不对，Task 1 已经把这个字段加上了，这里预期是**没有任何检查逻辑，所以 `pytest.raises(ScriptValidationError)` 的用例会因为没抛错而失败**（`Failed: DID NOT RAISE`），不抛错的用例（`test_stretch_at_exactly_the_retry_threshold_does_not_raise` 等）此刻应该已经能通过（因为还没有任何东西会抛错）——这是正常的、符合预期的部分绿。真正要看到红的是那 4 个 `pytest.raises` 用例。

- [ ] **Step 3: 实现**

编辑 `src/tenmin/script/validate.py`，在 `repair_script` 函数末尾（`max_holds` 检查之后、`return repaired, warnings` 之前，即紧跟在这一段之后）：

```python
    holds = sum(len(beat.audio.holds) for beat in repaired.beats)
    if holds > cfg.max_holds:
        raise ScriptValidationError(
            f"全片留白 {holds} 处，超过上限 {cfg.max_holds}（提示词要求 3–8 处）："
            f"每处 2–4 秒旁白静音，还要从旁白字数预算里扣，重试",
            script=repaired,
        )
```

追加：

```python
    overstretched: list[str] = []
    for beat in repaired.beats:
        footage = beat_clip_seconds(beat)
        span = beat_seconds(beat, rate=rate)
        if footage <= 0 or span <= 0:
            continue
        stretch = span / footage
        if stretch > cfg.retry_stretch_max:
            overstretched.append(f"{beat.label}（{stretch:.1f} 倍）")
    if overstretched:
        raise ScriptValidationError(
            f"{len(overstretched)} 个节点画面明显不够、会被拉伸变形："
            f"{'、'.join(overstretched)}（重试阈值 {cfg.retry_stretch_max:.1f} 倍），"
            f"重试",
            script=repaired,
        )
```

（函数体紧接着的 `return repaired, warnings` 保持不动。`beat_clip_seconds`/`beat_seconds` 两个函数已经在文件顶部导入，不需要新增 import。）

- [ ] **Step 4: 跑测试，观察 fixture 撞车的 fallout，逐个修**

```bash
/Users/portz/.local/bin/uv run pytest tests/test_validate.py -q
```

如果 Step 1 的三处 fixture 收紧（1a/1b/1c）和四处 cfg 覆盖（1e）以及两处改调 `check_script`（1f）都已经照抄，这一步应该直接是：

```
114 passed
```

如果不是 114 passed，说明抄漏了哪一处——对照本 Step 1 里给出的**逐字节代码块**重新核对，不要凭直觉另外调整数值（本计划的每一处数字都已经在这个 worktree 里实测验证过）。

- [ ] **Step 5: 跑全量测试，修另外两处跨文件 fallout**

```bash
/Users/portz/.local/bin/uv run pytest tests/ -q
```

预期会看到 `tests/test_single.py::test_generate_script_marks_silent_highlight` 和 `tests/test_e2e.py` 里全部用到 `project` fixture 的测试（13 个）报错，报错信息都是 `AssertionError: FakeProvider 的预置响应已用尽`——这是因为这两个文件各有一处共享 fixture，构造的 clip/旁白比例远超新阈值，触发了一次意料之外的 LLM 重试，把测试预置的唯一一份假响应用光了。逐个修：

**`tests/test_single.py`**：找到 `test_generate_script_marks_silent_highlight`（约行 534-541），改成：

```python
async def test_generate_script_marks_silent_highlight(cfg, track, report):
    # beat[2] 的字数改小：固定 10 秒 clip（1330-1340，刻意留在 ed_range 1348.18
    # 之前）配 216 字/48 秒旁白会拉伸 4.8 倍，超过 retry_stretch_max（1.5）触发
    # repair_script 的新硬失败；这里要验证的是 is_silent_highlight，跟画面/旁白
    # 预算无关，把这一个节点的字数减到跟 10 秒 clip 同量级（60 字 ≈ 13.3 秒），
    # 其余四个节点各加 39 字补足总量（原总量 5×216=1080 字，budget_rewrite 轮次
    # 判据按总时长算，字数变了总量不变才不会意外触发返工）。
    llm = valid_llm_script(chars_per_beat=(255, 255, 60, 255, 255))
    llm.beats[2].clips = [
        LLMClip(episode=2, start=1330.0, end=1340.0, visual="定格收尾", anchor_lines=[])
    ]
    provider = FakeProvider([llm])
    script, _ = await generate_script(cfg, track, report, provider)
    assert script.beats[2].clips[0].is_silent_highlight is True
```

**`tests/test_e2e.py`**：找到 `project` fixture（约行 102-120）里构造 `config` 字典的这一段：

```python
    config = {
        "show": "才女的侍从",
        "slug": "saijo",
        "mode": "single_episode",
        "target_seconds": 240,
        "locale": {"convert_traditional": True},
        "episodes": [{"number": 1, "srt": "srt/E02.srt"}],
        "glossary": {},
        "llm": {"provider": "gemini", "model": "gemini-3.6-flash"},
    }
```

改成：

```python
    config = {
        "show": "才女的侍从",
        "slug": "saijo",
        "mode": "single_episode",
        "target_seconds": 240,
        "locale": {"convert_traditional": True},
        "episodes": [{"number": 1, "srt": "srt/E02.srt"}],
        "glossary": {},
        "llm": {"provider": "gemini", "model": "gemini-3.6-flash"},
        # 这份 golden fixture 的 clip 时长是围着真实高光区间手工核对出来的，跟旁白
        # 字数完全不成比例（画面/旁白预算的 A3 早就超过 stretch_max，只是从没人看过
        # 那条 warning）。本文件测的是 v1 链路的产物接线，不是画面/旁白预算，
        # 放宽这一项阈值，别让新的重试检查打断这条早就存在的既有行为。
        "validate_script": {"retry_stretch_max": 100.0},
    }
```

- [ ] **Step 6: 再跑一次全量测试**

```bash
/Users/portz/.local/bin/uv run pytest tests/ -q
```

预期：`2250 passed, 32 skipped`（基线 2242 加上本任务新增的 8 个测试）。

- [ ] **Step 7: ruff 与格式检查**

```bash
/Users/portz/.local/bin/uv run ruff check src/tenmin/script/validate.py tests/test_validate.py tests/test_single.py tests/test_e2e.py
/Users/portz/.local/bin/rtk git diff --check
```

预期：均无输出（全部通过）。

- [ ] **Step 8: 提交**

```bash
/Users/portz/.local/bin/rtk git add src/tenmin/script/validate.py tests/test_validate.py tests/test_single.py tests/test_e2e.py
/Users/portz/.local/bin/rtk git commit -m "feat: clip 拉伸比例超限触发整篇重试"
```

---

## 自审（写完后再看一遍）

**spec 覆盖检查：**
- spec §1 新配置字段（`retry_stretch_max` 默认 1.5、`gt=0`、docstring 里的取值依据）→ Task 1 覆盖。
- spec §2 检查位置与聚合方式（放在 `repair_script` 的 `max_holds` 检查之后、复用 `beat_clip_seconds`/`beat_seconds`、`footage<=0 or span<=0` 跳过、扫全篇聚合一次报错、严格 `>`、带 `script=repaired`）→ Task 2 Step 3 逐字实现，`test_multiple_overstretched_beats_are_all_listed_in_one_message`/`test_overstretched_error_carries_the_repaired_script`/`test_empty_narration_beat_does_not_crash_the_retry_check` 覆盖对应断言。
- spec §3 与既有 warning 检查的关系（`(1.5,4.0]` 只触发新检查；`>4.0` 两者都该触发但 repair_script 先抢着报错；`<=1.5` 都不触发；`check_script` 完全不变）→ `test_single_overstretched_beat_raises_for_retry`（落在 (1.5,4.0] 区间）、`test_validate_script_aborts_before_check_script_runs_for_overstretched_beats`（验证 check_script 被抢先）、`test_lowering_stretch_max_alone_does_not_trigger_the_retry_check`（验证两条阈值独立比较）、`test_understretched_beats_never_trigger_the_retry_check` 逐条覆盖。
- spec §4 测试列表（pydantic 边界、单/多节点超限、恰好阈值不报、独立阈值、下溢方向不触发、退化 beat 不崩、`script=repaired`、端到端）→ Task 1 与 Task 2 的测试逐条对应，无遗漏。
- spec §5 风险（阈值样本量小、可调；额外一次 LLM 调用成本）不是代码改动，无需测试覆盖，已在 spec 文档里写明。

**占位符检查：** 无 TODO/TBD，所有代码块均为完整可运行代码，所有数值都已在本 worktree 实测跑绿（Task 2 的完整 diff 在写这份计划前已经用 `uv run pytest tests/ -q` 验证到 2242 passed / 0 failed，随后已把改动 `git checkout` 撤销，留给实现子代理走真正的 TDD 红绿流程去重新做一遍）。

**类型一致性检查：** `ScriptValidationError.__init__(self, message: str, *, script: Script | None = None)` 签名不变；新检查复用的 `beat_clip_seconds(beat: Beat) -> float`/`beat_seconds(beat: Beat, *, rate: str) -> float` 签名不变，均为已有函数、不改动其定义；`run()` 测试助手新增的 `cfg` 参数类型是 `ValidateConfig | None`，跟 `validate_script(..., cfg: ValidateConfig = DEFAULT_VALIDATE)` 的既有签名一致。

**范围检查：** 两个任务合计改动 2 个源文件（`config.py`、`validate.py`）+ 4 个测试文件，规模比 A/B 两个 plan 小得多，但 Task 2 因为要处理三处跨文件的 fixture 撞车而比 C-1 略大——这是必要的，不是范围蔓延：`retry_stretch_max` 默认值一旦生效，这些 fallout 是真实存在的，不修就没法让 Task 2 的产物在提交时全绿。
