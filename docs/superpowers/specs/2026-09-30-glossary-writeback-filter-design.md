# 术语表回写过滤设计（C-1）

## 背景与动机

`zh/glossary.json` 是跨集累积的专有名词表，翻译阶段把新认出的术语并回它，之后每一集的 script prompt 都会读到。实测 `work/akujo/zh/glossary.json` 里混进了 ASR 听写误差经翻译放大后的垃圾条目，例如 `"主ガビオ":"主嘉碧哦"`——日语原名 `主ガビオ` 被听错、又被翻译加了一个句尾语气助词「哦」，这个带语气词尾缀的错误译名此后会一直喂给 script 阶段的 LLM，直到有人手动清理。

同一张表里也有大量形如 `"山田":"山田"`（键等于值）的条目，但这些是合法的——日语汉字本来就与中文写法相同。因此"键等于值就丢弃"这条最初设想的规则被放弃：它会误杀大量正确术语。

语义跑偏类错误（例如 `"東欧":"铜"`、`"高級":"后宫"`，听感上跟正确读音相近但翻译完全不对）无法用机械规则识别对错，本设计不处理这类问题，依赖 `project.yaml` 里手写的 `glossary` 覆盖（`effective_glossary` 冲突时手写的赢）人工纠错。

## 目标

- 在**回写**累积表这一步（`merge_glossary` 处理 `fresh` 参数时）拦住带句尾语气助词的译名，自动剪掉尾缀而不是整条丢弃。
- 剪裁后如果译名退化成 0 或 1 个字符，整条丢弃而不并入累积表。
- 每次触发剪裁或丢弃都产生一条可读的中文 warning，通过既有的 stage-warning 通道（`run_pipeline` 的聚合 `warnings` 列表 → CLI 打印）交付给用户，方便发现"本集新词被自动修正过"。
- 不改变 `load_glossary`、`effective_glossary`、`_clean`、`save_glossary` 的现有行为；`accumulated` 参数（已经清洗过的历史累积表）不重新过滤。

## 非目标

- 不识别或修复语义跑偏类误译（如"東欧"→"铜"）——这类错误无法用规则判断对错，只能靠 `project.yaml` 手写 glossary 人工纠错。
- 不实现"键等于值就丢弃"规则——会误杀大量合法的日汉同形词条目（`山田`、`帝`、`友情`、`武田家`等）。
- 不改变 `_clean` 本身的清洗逻辑（strip 后非空字符串），也不影响 `load_glossary`/`effective_glossary` 读取历史累积数据的方式。
- 不处理 akujo 项目当前 `zh/glossary.json` 里已经存在的历史垃圾条目——那是一次性数据清理，随时可以直接编辑 `work/akujo/project.yaml` 的 `glossary` 字段手工覆盖，不需要本设计的代码改动。

## 设计

### 1. 语气助词剪裁规则

新增一个模块级常量集合（放在 `src/tenmin/translate/glossary.py`，属于"物理上不可能/数据坏了"一类的合法性边界，不是创作旋钮，不进 `RenderConfig`/`ValidateConfig`）：

```python
_TRAILING_PARTICLES = frozenset("哦呀啊呢吧啦嘛喔哟欸唉")
```

对 `merge_glossary` 收到的 `fresh` 参数中每一条译文（在 `_clean` 之后，即已保证是非空字符串）：

1. 若该译文的**最后一个字符**在 `_TRAILING_PARTICLES` 中，剪掉这一个字符。只剪一次，不循环——剪完不再检查新的结尾是否还是语气词。
2. 剪裁后若剩余长度 ≤ 1（即原文恰好是"语气词"或"单字+语气词"两种情况），整条丢弃，不进入返回的累积表。
3. 每次剪裁或丢弃都追加一条中文 warning：
   - 剪裁：`f"术语表：{term!r} 的译名 {original!r} 结尾疑似语气词，已修正为 {trimmed!r}"`
   - 丢弃：`f"术语表：{term!r} 的译名 {original!r} 疑似语气词或过短，已丢弃"`
4. 剪裁/丢弃只影响本次要并入累积表的 `fresh` 条目；已经存在于 `accumulated` 中的历史条目不重新触发这条规则（`merged = _clean(accumulated)` 这一步不变）。

### 2. `merge_glossary` 接口变化

```python
def merge_glossary(
    accumulated: Mapping[str, str],
    fresh: Mapping[str, str],
    *,
    warnings: list[str] | None = None,
) -> dict[str, str]:
```

- 新增可选的 `warnings` 出参（out-parameter），不改变返回类型、不改变现有 7 处调用点的签名兼容性（`warnings=None` 时行为与改动前完全一致，不产生任何警告收集开销之外的差异）。
- `merged = _clean(accumulated)` 这一步保持不变。
- 对 `_clean(fresh)` 的每一条，先过语气助词规则（剪裁或丢弃），再走原有的 `setdefault` 逻辑（累积的赢）。
- `test_a_saved_table_is_a_fixed_point_of_merge`（不动点测试）必须保持通过：已经过语气词清洗的合法术语不带这种尾缀，不会被重复剪裁，因此这个不动点性质天然保留，不需要额外处理。

### 3. `run_translate` 与 `run_pipeline` 的 warning 传递

`run_translate` 目前返回裸 `TranslatedTrack`，改为返回 `tuple[TranslatedTrack, list[str]]`，与 `run_voice`/`run_timeline` 的既有模式一致：

```python
async def run_translate(
    cfg: ProjectConfig, provider: LLMProvider, episode: int
) -> tuple[TranslatedTrack, list[str]]:
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
    notices: list[str] = []
    save_glossary(
        paths.glossary,
        merge_glossary(accumulated, translated.glossary, warnings=notices),
    )
    return translated, notices
```

`run_pipeline` 里的调用点（`"translate" in wanted` 分支）同步改为与 `voice`/`timeline` 阶段一致的接收方式，warning 加上 `E{episode:02d}：` 前缀后并入总 `warnings` 列表（与 `ingest_warnings`/`run_audio` 的既有惯例一致，方便区分批处理中不同集的警告）：

```python
if "translate" in wanted:
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

对早退路径（`track.source != "asr"`）返回空 warning 列表，不产生任何前缀噪音。

### 4. 边界情况

- 译文剪裁后如果撞上 `accumulated` 里已存在的同一术语（冲突），仍走原有"累积的赢"逻辑——`setdefault` 不会覆盖已存在的键，剪裁后的新值被丢弃，不产生额外 warning（这不是语气词问题，是术语定名冲突，属于现有行为，不在本设计范围）。
- `_TRAILING_PARTICLES` 只检查恰好一个字符的匹配；两个连续语气词（如"好啦啦"结尾"啦啦"）不会被完全剪掉，只剪最后一个字符变成"好啦"——这是用户在澄清问题中选定的行为（"剪完即停"），避免过度剪裁把合法多字词错误截断。
- 键（术语原文）本身不参与这条过滤规则，只处理值（译名）。

## 测试

- `tests/test_translate_glossary.py` 新增：
  - 单字符语气词结尾被剪裁且产生 warning（如 `"主ガビオ":"主嘉碧哦"` → `"主ガビオ":"主嘉碧"` + 一条 warning）。
  - 剪裁后长度 ≤1 的整条丢弃且产生 warning（如 `"帝":"哦"` → 该条目不出现在结果里）。
  - 不带语气词尾缀的合法条目（含键等于值的日汉同形词）不受影响、不产生 warning。
  - `accumulated` 中已存在的带语气词尾缀历史条目（假设清理前就已经写入过）不被重新剪裁——只处理 `fresh`。
  - `merge_glossary(accumulated, fresh)`（不传 `warnings`）行为与改动前逐字节一致，现有 7 处调用点无需修改。
  - `test_a_saved_table_is_a_fixed_point_of_merge` 不动点测试保持通过。
- `tests/test_pipeline.py` 新增/修改：
  - `run_translate` 返回类型变化后，现有直接调用 `run_translate` 的测试更新为接收 `tuple`。
  - `run_pipeline` 批处理路径下，`translate` 阶段产生的语气词剪裁 warning 出现在最终 `warnings` 列表中且带正确的 `E{episode:02d}：` 前缀；多集场景下前缀能正确区分来源集。
  - `track.source != "asr"` 早退路径确认返回空 warning，不产生任何多余通知。

## 风险

- 剪裁规则只是启发式，理论上存在合法译名恰好以这些字符结尾的极小概率误判（例如角色名恰好叫"小啦"）。这类误判可以通过 `project.yaml` 手写 `glossary` 覆盖修正，成本很低，不构成设计缺陷。
- `run_translate` 返回类型变化是一个小的公开接口变更，影响范围仅限 `pipeline.py` 内部调用点和直接测试该函数的测试文件，不影响 CLI 或其他外部调用方（`run_translate` 未在 `cli.py` 中被直接调用）。
