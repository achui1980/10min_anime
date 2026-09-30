# 术语表回写过滤（C-1）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `merge_glossary` 回写累积术语表时，自动剪掉译名结尾的语气助词（剪一次、不循环），剪完只剩 0/1 字符就整条丢弃；每次剪裁/丢弃都产生一条中文 warning，经 `run_translate` → `run_pipeline` 的既有 warning 通道（与 voice/timeline 阶段同一模式）交付给用户。

**Architecture:** 在 `src/tenmin/translate/glossary.py` 新加一个模块级常量 `_TRAILING_PARTICLES` 和一个内部剪裁函数，`merge_glossary` 新增可选 `warnings` 出参、在原有 `setdefault` 逻辑前插入剪裁步骤。`run_translate` 的返回类型从 `TranslatedTrack` 改为 `tuple[TranslatedTrack, list[str]]`（对齐 `run_voice`/`run_timeline` 的既有模式），`run_pipeline` 的 `"translate" in wanted` 分支同步改成 `_, stage_warnings = await run_translate(...); warnings.extend(f"E{number:02d}：{w}" for w in stage_warnings)`。不新增任何 config 字段（语气词集是模块级合法性边界常量，不是创作旋钮）。

**Tech Stack:** Python 3.14、pytest、pydantic（`TranslatedTrack` 不变）。

---

### Task 1: `_TRAILING_PARTICLES` 剪裁规则 + `merge_glossary` 的 `warnings` 出参

**Files:**
- Modify: `src/tenmin/translate/glossary.py`
- Test: `tests/test_translate_glossary.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_translate_glossary.py` 末尾（`test_a_saved_table_is_a_fixed_point_of_merge` 之前或之后均可，建议紧跟 `test_merge_*` 系列之后、`test_manual_entries_override_accumulated_ones` 之前）追加：

```python
def test_merge_trims_a_trailing_particle_and_warns():
    """结尾的语气助词只剪一次、不循环检查新结尾。"""
    notices: list[str] = []
    merged = g.merge_glossary(
        {}, {"主ガビオ": "主嘉碑哦"}, warnings=notices
    )
    assert merged == {"主ガビオ": "主嘉碑"}
    assert len(notices) == 1
    assert "主ガビオ" in notices[0]
    assert "主嘉碑哦" in notices[0]
    assert "主嘉碑" in notices[0]


def test_merge_discards_an_entry_that_becomes_too_short_after_trimming():
    """剪完只剩 0 或 1 个字符，整条丢弃而不是保留单字。"""
    notices: list[str] = []
    merged = g.merge_glossary({}, {"帝": "哦"}, warnings=notices)
    assert merged == {}
    assert len(notices) == 1
    assert "帝" in notices[0]


def test_merge_discards_a_bare_particle_translation():
    """译名整个就是语气词（剪完剩 0 字符）也走丢弃分支。"""
    notices: list[str] = []
    merged = g.merge_glossary({}, {"x": "啦"}, warnings=notices)
    assert merged == {}
    assert len(notices) == 1


def test_merge_leaves_homographs_untouched_and_silent():
    """键等于值的日汉同形词（合法术语）不受影响、不产生 warning。"""
    notices: list[str] = []
    merged = g.merge_glossary({}, {"山田": "山田", "帝": "帝"}, warnings=notices)
    assert merged == {"山田": "山田", "帝": "帝"}
    assert notices == []


def test_merge_does_not_re_trim_accumulated_entries():
    """已经存在于累积表里的带语气词尾缀历史条目不被重新剪裁——只处理 fresh。"""
    notices: list[str] = []
    merged = g.merge_glossary({"主ガビオ": "主嘉碑哦"}, {}, warnings=notices)
    assert merged == {"主ガビオ": "主嘉碑哦"}
    assert notices == []


def test_merge_without_warnings_param_behaves_exactly_as_before():
    """不传 warnings 时行为与改动前逐字节一致，现有调用点无需修改。"""
    merged = g.merge_glossary({}, {"主ガビオ": "主嘉碑哦"})
    assert merged == {"主ガビオ": "主嘉碑"}


def test_merge_trims_only_the_last_character_once():
    """只剪一次：两个连续语气词只剪掉最后一个字符，不循环剪第二层。"""
    notices: list[str] = []
    merged = g.merge_glossary({}, {"x": "好啦啦"}, warnings=notices)
    assert merged == {"x": "好啦"}
    assert len(notices) == 1


def test_merge_conflict_after_trimming_keeps_accumulated_silently():
    """剪裁后撞上累积表已有的同一术语，走原有「累积的赢」逻辑，不产生额外 warning。"""
    notices: list[str] = []
    merged = g.merge_glossary(
        {"主ガビオ": "主嘉碑"}, {"主ガビオ": "主嘉碑哦"}, warnings=notices
    )
    assert merged == {"主ガビオ": "主嘉碑"}
    # 剪裁本身仍然发生并警告；只是 setdefault 不会覆盖已存在的键。
    assert len(notices) == 1
```

- [ ] **Step 2: 跑测试确认失败**

```bash
/Users/portz/.local/bin/uv run pytest tests/test_translate_glossary.py -k "trim or discard or homograph or leaves_homographs" -v
```

预期：`TypeError: merge_glossary() got an unexpected keyword argument 'warnings'`（新测试全部报这个错，因为 `merge_glossary` 还没有这个参数）。

- [ ] **Step 3: 实现**

编辑 `src/tenmin/translate/glossary.py`，在 `_clean` 函数下方（`load_glossary` 之前）新增模块级常量与内部函数：

```python
_TRAILING_PARTICLES = frozenset("哦呀啊呢吧啦嘛喀哟欸唉")
"""句尾语气助词/叹词集合。这是「数据坏了」一类的合法性边界，不是创作旋钮，所以不进
`RenderConfig`/`ValidateConfig`（判据见 AGENTS.md 关于 config 分家的说明）。

只在 `merge_glossary` 回写累积表时对 `fresh` 参数生效，不影响 `_clean`/`load_glossary`/
`effective_glossary` 的既有行为——那三个入口共享 `_clean`，动 `_clean` 本身会连累历史
已存条目的解释方式，范围过大。
"""


def _trim_trailing_particle(translation: str) -> tuple[str, bool]:
    """剪掉译名结尾的一个语气助词字符（只剪一次，不循环）。

    返回 `(剪裁后的字符串, 是否发生了剪裁)`。调用方据此决定要不要发 warning。
    """
    if translation and translation[-1] in _TRAILING_PARTICLES:
        return translation[:-1], True
    return translation, False
```

修改 `merge_glossary` 签名与函数体：

```python
def merge_glossary(
    accumulated: Mapping[str, str],
    fresh: Mapping[str, str],
    *,
    warnings: list[str] | None = None,
) -> dict[str, str]:
    """把这一集新认出来的词并进累积表。

    已经定下的译名**不许**被后面某一集改掉 —— 那正是累积要防的事（第 5 集把人名换个
    写法，成片看起来就像换了个角色）。所以冲突时保留累积的那个。

    两边都过 `_clean`：只洗新词的话，一条坏掉的累积条目会永远占着那个键、把后面每一集
    给出的好译名都挡在外面。

    对 `fresh` 里通过 `_clean` 的每一条译名，先剪一次结尾语气助词（ASR 听写误差经翻译
    放大后常见的垛词，例如"主ガビオ":"主嘉碑哦"）；剪完长度 ≤1 就整条丢弃，不并入
    返回的表。`accumulated` 侧不重新触发这条规则——历史累积表已经清洗过，不该被反复剪。
    可选的 `warnings` 出参收集每次剪裁/丢弃的中文提示，`None` 时（默认）不收集，现有
    调用点无需改动。

    返回新字典，不改入参。
    """
    merged = _clean(accumulated)
    for term, translation in _clean(fresh).items():
        trimmed, did_trim = _trim_trailing_particle(translation)
        if did_trim:
            if len(trimmed) <= 1:
                if warnings is not None:
                    warnings.append(
                        f"术语表：{term!r} 的译名 {translation!r} 疑似语气词或过短，"
                        "已丢弃"
                    )
                continue
            if warnings is not None:
                warnings.append(
                    f"术语表：{term!r} 的译名 {translation!r} 结尾疑似语气词，"
                    f"已修正为 {trimmed!r}"
                )
            translation = trimmed
        merged.setdefault(term, translation)
    return merged
```

- [ ] **Step 4: 跑测试确认通过**

```bash
/Users/portz/.local/bin/uv run pytest tests/test_translate_glossary.py -v
```

预期：全部通过（原有 26 个 + 新增 8 个）。

- [ ] **Step 5: 跑全量测试确认不动点性质与既有调用点未受影响**

```bash
/Users/portz/.local/bin/uv run pytest tests/ -q
```

预期：与本任务开始前的基线一致（无新增失败）。特别关注 `test_a_saved_table_is_a_fixed_point_of_merge`——已清洗的合法术语不带语气词尾缀，不会被重复剪裁，这个不动点性质应天然保留。

- [ ] **Step 6: 提交**

```bash
/Users/portz/.local/bin/rtk git add src/tenmin/translate/glossary.py tests/test_translate_glossary.py
/Users/portz/.local/bin/rtk git commit -m "feat: merge_glossary 回写时剪裁译名结尾语气助词"
```

---

### Task 2: `run_translate` 返回 `(TranslatedTrack, list[str])`，接线 `run_pipeline`

**Files:**
- Modify: `src/tenmin/pipeline.py:543-581`（`run_translate` 函体），`src/tenmin/pipeline.py:1363-1375`（`run_pipeline` 里 `"translate" in wanted` 分支）
- Test: `tests/test_pipeline.py`

**背景（写测试前必读）：** `run_translate` 目前的完整函体（第 543-581 行）：

```python
async def run_translate(
    cfg: ProjectConfig, provider: LLMProvider, episode: int
) -> TranslatedTrack:
    """..."""
    track = next(t for t in _load_tracks(cfg) if t.episode == episode)
    if track.source != "asr":
        return TranslatedTrack(episode=episode)

    paths = Paths(cfg.root)
    accumulated = load_glossary(paths.glossary)
    usage: list[UsageRecord] = []
    try:
        translated = await translate_track(
            cfg, track, provider, accumulated=accumulated, usage=usage
        )
    finally:
        write_usage(paths.zh_usage(episode), episode, usage)

    _write_json(paths.zh_lines(episode), translated.model_dump_json(indent=2))
    _write_text(paths.zh_subtitles(episode), render_zh_srt(track, translated))
    # merge 的方向是「累积的赢」：已经定下的译名不许被后面某一集改掉。喂给模型的那份表
    # 另有一个方向（手写的赢），那一步在 translate_track 内部做。
    save_glossary(paths.glossary, merge_glossary(accumulated, translated.glossary))
    return translated
```

`run_pipeline` 里的调用点（第 1363-1375 行）：

```python
            if "translate" in wanted:
                # **刻意不做多集并发**（script 上面那个有界预取窗口不往这里搬）：第 1 集
                # 写完累积术语表、第 2 集才读到含第 1 集的版本，并发会让累积失去意义，
                # 而且两个 task 会同时回写同一个文件。
                outputs = [paths.zh_lines(number), paths.zh_subtitles(number)]
                if force or not is_fresh(
                    "translate", number, outputs, _translate_inputs(paths, number)
                ):
                    reporter.stage_start("translate")
                    await run_translate(cfg, provider, number)
                    reporter.stage_done("translate")
                else:
                    reporter.stage_skip("translate")
```

对照的既有模式（`run_voice`，第 1411-1414 行）：

```python
                    _, stage_warnings = await run_voice(
                        cfg, tts_engine, episode=number, reporter=reporter
                    )
                    warnings.extend(stage_warnings)
```

`warnings` 是 `run_pipeline` 内部第 1182 行声明的 `warnings: list[str] = []`，在纵向循环外创建一次、循环内所有阶段共享同一个列表。

**现有 5 处直接调用 `run_translate` 的测试**（均在 `tests/test_pipeline.py`，行号可能因编辑漂移，以 `-k` 匹配函数名定位）：
- `test_run_translate_writes_all_three_artifacts`（约 2970 行）：`await run_translate(cfg, provider, 11)`，不接收返回值。
- `test_run_translate_accumulates_into_an_existing_glossary`（约 2983 行）：同上。
- `test_run_translate_leaves_an_unchanged_glossary_untouched`（约 2998 行）：调用两次，均不接收返回值。
- `test_run_translate_skips_a_native_subtitle_episode`（约 3017 行）：`result = await run_translate(cfg, FakeProvider([]), 2)`，**接收返回值**并断言 `result.episode == 2` 和 `result.lines == []`——这条必须改成接收 tuple 的第一个元素。
- `test_run_translate_writes_a_usage_file`（约 3252 行）与 `test_a_skipped_translate_writes_no_usage_file`（约 3261 行）：不接收返回值。

**Step-by-step:**

- [ ] **Step 1: 写失败测试**

在 `tests/test_pipeline.py` 里 `test_run_translate_skips_a_native_subtitle_episode` 之后（约 3033 行附近，在 `test_run_pipeline_runs_translate_for_a_transcribed_episode` 之前）新增：

```python
async def test_run_translate_returns_warnings_for_a_trimmed_glossary_entry(tmp_path):
    """新增：run_translate 把 merge_glossary 剪裁产生的 warning 一并返回。"""
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    provider = FakeProvider(
        [_translation_response({"主ガビオ": "主嘉碑哦"})]
    )

    _, stage_warnings = await run_translate(cfg, provider, 11)

    assert len(stage_warnings) == 1
    assert "主ガビオ" in stage_warnings[0]
    stored = json.loads(paths.glossary.read_text(encoding="utf-8"))
    assert stored == {"主ガビオ": "主嘉碑"}


async def test_run_translate_returns_no_warnings_when_glossary_is_clean(tmp_path):
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    provider = FakeProvider([_translation_response({"リディア": "莉迪亚"})])

    _, stage_warnings = await run_translate(cfg, provider, 11)

    assert stage_warnings == []


async def test_run_translate_returns_empty_warnings_when_skipped(tmp_path):
    cfg, paths = _project_with_dialogue(tmp_path, episode=2, source="srt")

    result, stage_warnings = await run_translate(cfg, FakeProvider([]), 2)

    assert result.episode == 2
    assert result.lines == []
    assert stage_warnings == []
```

修改既有的 `test_run_translate_skips_a_native_subtitle_episode`（约 3017-3032 行），把：

```python
    result = await run_translate(cfg, FakeProvider([]), 2)
```

改为：

```python
    result, _ = await run_translate(cfg, FakeProvider([]), 2)
```

其余断言（`result.episode == 2`、`result.lines == []`）保持不变。

新增一条 `run_pipeline` 层面的端到端测试（放在 `test_run_pipeline_runs_translate_for_a_transcribed_episode` 之后，约 3043 行之后）：

```python
async def test_run_pipeline_surfaces_glossary_trim_warnings_with_episode_prefix(tmp_path):
    """translate 阶段产生的语气词剪裁 warning 出现在最终 warnings 列表中，
    且带正确的 E{episode:02d}： 前缀（与 ingest_warnings/run_audio 的既有惯例一致）。"""
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")
    provider = FakeProvider([_translation_response({"主ガビオ": "主嘉碑哦"})])

    warnings = await run_pipeline(cfg, provider, only=["translate"])

    matches = [w for w in warnings if "主ガビオ" in w]
    assert len(matches) == 1
    assert matches[0].startswith("E11：")
```

- [ ] **Step 2: 跑测试确认失败**

```bash
/Users/portz/.local/bin/uv run pytest tests/test_pipeline.py -k "translate_returns_warnings or translate_returns_no_warnings or translate_returns_empty or surfaces_glossary_trim" -v
```

预期：新测试因 `run_translate` 仍返回裸 `TranslatedTrack` 而 `ValueError: too many values to unpack` 或 `AttributeError`（tuple 解包失败）报错；`surfaces_glossary_trim` 测试因 warning 未产生而断言失败。

- [ ] **Step 3: 实现**

编辑 `src/tenmin/pipeline.py`，把 `run_translate`（第 543-581 行）改为：

```python
async def run_translate(
    cfg: ProjectConfig, provider: LLMProvider, episode: int
) -> tuple[TranslatedTrack, list[str]]:
    """翻译一集：落译文轨、中文字幕，并把新认出的术语并回累积表。

    只对听写来的对白动手。手传的字幕与从视频里抽出来的软字幕轨都是片源自带的，本来
    就是观众能读的语言。判据用对白轨的 source 字段，刻意不做语言自动检测 —— 那是个
    会错的猜测，而 source 是一个确定的事实。

    跳过时连 `zh/` 目录都不建（早退发生在任何写盘之前），所以现有的繁中片源在磁盘上
    看不出这个阶段存在过。代价是这个函数每次运行都会被叫一遍：那一集的产物永远不会出现，
    而新鲜度判据（_translate_inputs + _is_fresh）只看文件 mtime、拿不到 source 字段，
    于是它每次都判「不新鲜」。一次读盘换「判据不必认识对白轨的内容」，划得来。
    读的量是 _load_tracks 的全量（cfg.episodes 里每一集的对白轨），跟 run_script 同一个
    口径 —— 十几集的 json 相对一次 LLM 调用可以忽略。

    已知边界：手传一份日语 SRT（或者软字幕轨恰好是日语）时，这一阶段不会跑，而且那份
    对白还会被繁转简改字。目前的片源都不是这种情况，真碰上了再说。

    返回 `(译文轨, warnings)`，与 `run_voice`/`run_timeline` 同一模式：warnings 目前
    只来自 `merge_glossary` 的语气词剪裁/丢弃提示，方便人工发现「本集新词被自动修正
    过」。跳过（非 asr 来源）时返回空列表，不产生任何提示噪音。
    """
    track = next(t for t in _load_tracks(cfg) if t.episode == episode)
    if track.source != "asr":
        return TranslatedTrack(episode=episode), []

    paths = Paths(cfg.root)
    accumulated = load_glossary(paths.glossary)
    usage: list[UsageRecord] = []
    try:
        translated = await translate_track(
            cfg, track, provider, accumulated=accumulated, usage=usage
        )
    finally:
        write_usage(paths.zh_usage(episode), episode, usage)

    _write_json(paths.zh_lines(episode), translated.model_dump_json(indent=2))
    _write_text(paths.zh_subtitles(episode), render_zh_srt(track, translated))
    # merge 的方向是「累积的赢」：已经定下的译名不许被后面某一集改掉。喂给模型的那份表
    # 另有一个方向（手写的赢），那一步在 translate_track 内部做。
    notices: list[str] = []
    save_glossary(
        paths.glossary, merge_glossary(accumulated, translated.glossary, warnings=notices)
    )
    return translated, notices
```

把 `run_pipeline` 里 `"translate" in wanted` 分支（第 1363-1375 行）改为：

```python
            if "translate" in wanted:
                # **刻意不做多集并发**（script 上面那个有界预取窗口不往这里搬）：第 1 集
                # 写完累积术语表、第 2 集才读到含第 1 集的版本，并发会让累积失去意义，
                # 而且两个 task 会同时回写同一个文件。
                outputs = [paths.zh_lines(number), paths.zh_subtitles(number)]
                if force or not is_fresh(
                    "translate", number, outputs, _translate_inputs(paths, number)
                ):
                    reporter.stage_start("translate")
                    _, stage_warnings = await run_translate(cfg, provider, number)
                    warnings.extend(f"E{number:02d}：{w}" for w in stage_warnings)
                    reporter.stage_done("translate")
                else:
                    reporter.stage_skip("translate")
```

- [ ] **Step 4: 更新现有直接调用点**

在 `tests/test_pipeline.py` 中，把以下 5 处调用改为接收 tuple（不接收返回值的用 `_` 或直接忽略）：

1. `test_run_translate_writes_all_three_artifacts`：`await run_translate(cfg, provider, 11)` → 保持原样即可（Python 允许丢弃协程结果，不接收返回值不会报错——**这一步实际上不需要改动**，因为没有 unpacking）。
2. `test_run_translate_accumulates_into_an_existing_glossary`：同上，不需要改动。
3. `test_run_translate_leaves_an_unchanged_glossary_untouched`：同上，不需要改动。
4. `test_run_translate_skips_a_native_subtitle_episode`：**必须改**（见 Step 1 已给出的具体改法）。
5. `test_run_translate_writes_a_usage_file` / `test_a_skipped_translate_writes_no_usage_file`：不需要改动。

（`monkeypatch.setattr("tenmin.pipeline.run_translate", spy_translate)` 那处的 `spy_translate` 已经返回 `TranslatedTrack(episode=episode, lines=[])`——需要改成返回 tuple：）

在 `_spy_on_script_prefetch_order`（约 3162 行附近）里，把：

```python
    async def spy_translate(cfg, provider, episode):
        order.append(f"translate{episode}")
        return TranslatedTrack(episode=episode, lines=[])
```

改为：

```python
    async def spy_translate(cfg, provider, episode):
        order.append(f"translate{episode}")
        return TranslatedTrack(episode=episode, lines=[]), []
```

- [ ] **Step 5: 跑测试确认通过**

```bash
/Users/portz/.local/bin/uv run pytest tests/test_pipeline.py -v
```

预期：全部通过。

- [ ] **Step 6: 跑全量测试**

```bash
/Users/portz/.local/bin/uv run pytest tests/ -q
```

预期：与基线一致，无新增失败。

- [ ] **Step 7: ruff 与格式检查**

```bash
/Users/portz/.local/bin/uv run ruff check src/tenmin/pipeline.py tests/test_pipeline.py
/Users/portz/.local/bin/rtk git diff --check
```

预期：均无输出（全部通过）。

- [ ] **Step 8: 提交**

```bash
/Users/portz/.local/bin/rtk git add src/tenmin/pipeline.py tests/test_pipeline.py
/Users/portz/.local/bin/rtk git commit -m "feat: run_translate 返回术语表剪裁警告并接入 run_pipeline"
```

---

## 自审（写完后再看一遍）

**spec 覆盖检查：**
- spec §1 语气助词剪裁规则（`_TRAILING_PARTICLES`、剪一次、≤1 字符丢弃、两种 warning 文案）→ Task 1 覆盖。
- spec §2 `merge_glossary` 接口变化（`warnings` 出参、`None` 时行为不变、`accumulated` 侧不重复过 `_clean` 之外的剪裁、不动点测试保持通过）→ Task 1 覆盖。
- spec §3 `run_translate`/`run_pipeline` 的 warning 传递（返回类型变化、`E{episode:02d}：` 前缀、早退路径空列表）→ Task 2 覆盖。
- spec §4 边界情况（剪裁后与累积表冲突走原有「累积的赢」、不产生额外 warning；语气词只剪一次不循环；键不参与过滤）→ Task 1 的 `test_merge_conflict_after_trimming_keeps_accumulated_silently`、`test_merge_trims_only_the_last_character_once` 覆盖。
- spec 测试小节列出的用例与本计划 Task 1/Task 2 的测试逐条对应，无遗漏。

**占位符检查：** 无 TODO/TBD，所有代码块均为完整可运行代码。

**类型一致性检查：** `merge_glossary` 返回类型不变（`dict[str, str]`），新增 `warnings: list[str] | None = None` 关键字参数；`run_translate` 返回类型从 `TranslatedTrack` 改为 `tuple[TranslatedTrack, list[str]]`，Task 2 的所有测试与 `run_pipeline` 调用点均按新类型解包，命名一致（`stage_warnings`，与 `run_voice`/`run_timeline` 调用处的变量名相同）。

**范围检查：** 两个任务合计改动 2 个源文件（`glossary.py`、`pipeline.py`）+ 1 个测试文件（`test_pipeline.py` 覆盖两个任务，`test_translate_glossary.py` 仅 Task 1），规模远小于 A/B 两个 plan，适合单一实现周期完成，不需要进一步拆分。
