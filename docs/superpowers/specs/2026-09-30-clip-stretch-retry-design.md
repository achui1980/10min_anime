# clip 拉伸比例触发返工设计（C-2）

## 背景与动机

`render/timeline.py` 的 `build_timeline` 按 `ratio = 旁白秒数 / 画面秒数` 缩放某个节点（beat）里的每一个 clip：`source_end = clip.start + clip.duration * ratio`。ratio 远大于 1 时每段 clip 都要往后多吃几倍源片，吃到超出片长就被钳到片尾，成片画面与旁白错位；ratio 太大甚至会让画面看起来卡在某一帧不动（源片没有更多素材可用，只能反复钳在片尾）。

`script/validate.py` 的 `_check_footage_budget` 在 script 阶段就用同一份实现（`beat_clip_seconds`/`beat_seconds`）提前算出这个比值并检查，但目前**只发 warning，从不触发重试**。现有阈值（`ValidateConfig.stretch_max=4.0`、`stretch_min=0.125`）是刻意留得很松的："只拦数量级级别的配错"——docstring 里记录的依据是 85 个真实 beat 的拉伸倍率落在 0.193–2.526 之间，全部都在这两个阈值内，从未触发过 warning。

但实际人工评审真实成片（akujo E11、saijo E02）时发现的可见缺陷（B 阶段调研记录）恰恰落在这个"从未触发"的区间里：akujo 28 段 clip 里有 7 段被拉伸超过 1.5 倍（最差 2.62 倍，1069.1–1073.5 秒被拉伸到 1080.6 秒）；saijo beat7 旁白 39 秒对画面 20.7 秒（拉伸 1.88 倍）。这些案例都不会被现有的 4.0 上限拦住，说明"只拦数量级配错"的现有阈值与"值得为它重跑一次 LLM"的门槛是两件不同的事，不能直接把现有 warning 阈值升级成硬失败。

## 目标

- 新增一条更紧的阈值，专门用来判断"这个节点的画面已经不够用、需要重新生成剧本"，超过就在 `repair_script` 里抛 `ScriptValidationError`，触发现有 `validation_retries` 整篇重试机制（与 `max_holds` 同一条路径）。
- 现有 `_check_footage_budget` 的 warning 逻辑与 `stretch_max`/`stretch_min` 阈值完全不变，继续覆盖"数量级配错"这一更宽松的场景。
- 只拦"拉伸过长"（画面不够、被迫拉伸变形）方向。"画面过剩被截断"方向的可见缺陷证据不足，继续只发 warning，不新增硬性检查。

## 非目标

- 不改变 `stretch_max`/`stretch_min` 现有默认值，也不改变 `_check_footage_budget` 的 warning 文案或调用方式。
- 不处理"画面过剩被截断"（stretch 过小）方向的硬失败——现有真实缺陷案例全部是拉伸过长方向，没有证据支持对截断方向也加硬检查。
- 不尝试自动修复超限的 beat（比如自动延长旁白或裁剪 clip）。修复交给 LLM 重新生成，这里只负责判断"要不要重来"。
- 不改变 `validate_script`/`single.py` 现有的重试计数与轮次结构（`validation_retries`/`budget_rewrite_rounds`）——新检查复用的是已有的 `ScriptValidationError` 异常类型，调用方不需要感知这是一条新规则。

## 设计

### 1. 新配置字段

在 `ValidateConfig`（`src/tenmin/config.py`）里，紧跟在现有 `stretch_max`/`stretch_min` 字段之后新增：

```python
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

### 2. 检查位置与聚合方式

新检查放在 `repair_script`（`src/tenmin/script/validate.py`）里，紧邻现有 `max_holds` 检查（在 per-beat 主循环——anchor 校准、clip 去劣、holds 修复全部跑完——**之后**，使用 `repaired.beats` 的最终状态）：

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

- 计算逻辑完全复用 `_check_footage_budget` 已经在用的 `beat_clip_seconds`/`beat_seconds`（同一份实现，不新写），保证跟 warning 检查、跟 `render/timeline.py` 真正用的 ratio 分母同一口径。
- `footage <= 0 or span <= 0` 的跳过逻辑与 `_check_footage_budget` 完全一致（避免除零，且这种退化情况本身另有检查覆盖）。
- 采用与 `max_holds` 相同的"先扫完整个脚本、收集全部违规、末尾统一报一次错"的写法：如果有多个节点同时超限，一次重试的报错消息里会列出全部节点，让 LLM 一次性看到所有问题，不用一个个改。
- 严格 `>` 比较（不是 `>=`），跟现有 `stretch > cfg.stretch_max` 保持同一比较符——恰好等于阈值不算超限。
- 错误消息带 `script=repaired`（修到一半、已经过 anchor 校准和 clip 去劣的版本），与 `max_holds`、"某节点 clip 全灭" 两处失败路径的既有约定一致。

### 3. 与现有 warning 检查的关系

`_check_footage_budget`（纯读、`check_script` 里调用）完全不变：它仍然用 `stretch_max=4.0`/`stretch_min=0.125` 发 warning，覆盖比新阈值更宽松的场景。两条检查各自独立运作：

- `stretch` 在 `(1.5, 4.0]` 之间：`repair_script` 的新检查会触发重试（因为 `1.5 < stretch <= 4.0`），`_check_footage_budget` 的 warning 不会触发（因为 `stretch <= 4.0`）。
- `stretch > 4.0`：两条都会触发——但由于 `repair_script` 在 `check_script` 之前调用（`validate_script` 是 `repair_script` 后接 `check_script`），实际执行时 `ScriptValidationError` 会在 `repair_script` 里先抛出，整个 `validate_script` 调用直接中断重试，`check_script`（以及它的 warning）根本不会被跑到。这与 `max_holds` 现在的行为完全一致（`max_holds` 超限时同样会抢在 `check_script` 之前抛错）。
- `stretch <= 1.5`：都不触发。

## 测试

- `ValidateConfig.retry_stretch_max` 默认值、边界值（`gt=0`）的常规 pydantic 校验测试。
- `repair_script` 新增测试：
  - 单个 beat 的 `stretch` 超过 `retry_stretch_max` → 抛 `ScriptValidationError`，消息包含该 beat 的 label 与拉伸倍数。
  - 两个及以上 beat 同时超限 → 一条错误消息里包含全部超限 beat 的 label。
  - `stretch` 恰好等于 `retry_stretch_max` → 不抛错（严格 `>`）。
  - 用自定义 `ValidateConfig` 把 `stretch_max` 调到低于默认的 `retry_stretch_max`（例如 `stretch_max=1.0`），构造一个 `stretch≈1.2` 的 beat：验证只有旧的 warning 路径（`_check_footage_budget`）会报，`repair_script` 的新检查不会抛错（因为 `1.2 <= retry_stretch_max` 默认值 1.5）。用来确认两条检查是各自独立比较自己的阈值，不是互相包含或互相依赖的关系。
  - `stretch` 过小（画面过剩）无论多小都不触发新检查（只触发现有 warning，如果触发的话）。
  - `footage<=0` 或 `span<=0` 的退化 beat 不会导致新检查抛错或除零。
  - 抛出的 `ScriptValidationError` 带 `script=repaired`（修到一半的版本，而不是原始输入）。
- `check_script`（`_check_footage_budget` 的 warning 路径）现有测试不受影响，不需要修改。
- 端到端：`validate_script` 在有超限 beat 时抛错并中断，`check_script` 的 warning 不会被跑到（与 `max_holds` 现有的端到端测试同构）。

## 风险

- 1.5 倍这个阈值是从两部番、两集样本的人工评审得出的，样本量很小。如果换一部叙事节奏差异很大的番剧，可能会有一些"创作上刻意拉长的镜头"被误判为需要重试，浪费一次 LLM 调用。缓解手段：这是一个可调的 config 字段，不是模块级常量，遇到误判可以直接在 `project.yaml` 里调大 `retry_stretch_max`。
- 触发新检查意味着多花一次 LLM 调用（几百秒）。如果某一集经常触发这条重试却又修不好（比如源片本身素材匮乏，无论怎么重新生成剧本都绕不开某个画面稀缺的时间段），会导致重试耗尽 `validation_retries` 后仍然失败，需要人工介入调整该集的信号提取或素材范围。这是"整篇重试"机制本身就有的既有风险，不是这条新检查独有的。
