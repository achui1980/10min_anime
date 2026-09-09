# tenmin 进度条设计

## 1. 目标

`tenmin run` 跑一集要 4-10 分钟（`script`/`voice` 是网络调用，`render` 是真实视频编码），过程中终端完全静默，用户不知道卡在哪一步、还要等多久。本设计给 `tenmin run` 加一个可视化进度条，让用户随时看到：

- 当前在跑 8 个阶段（ingest/signals/script/docgen/voice/timeline/audio/render）里的哪一个
- 哪些阶段因为 mtime 新鲜度检查被跳过了（区别于真正执行完的）
- 两个耗时最长的阶段（`voice`、`render`）的内部进度（第几句配音 / 编码到百分之多少）
- 批量模式（不传 `--episode`，跑全部已注册集数）时，当前是第几集/共几集

不做的事：不改变现有的错误处理逻辑（红字 stderr 一行式报错保持不变，只是进度条会先冻结显示失败节点）；不做可暂停/可取消的交互式控制（Ctrl-C 直接退出，不做二次确认）；不做进度持久化或跨进程恢复。

## 2. 架构

### 2.1 新依赖

新增 `rich`（`pyproject.toml` `[project] dependencies` 追加 `"rich>=13"`）。选它而不是 `tqdm` 的原因：项目已经用 `typer` 做 CLI，`rich` 是 typer 生态里的标准搭档；`rich.progress.Progress` 原生支持多个嵌套/并列的进度条（外层"第几集"、中层"第几阶段"、内层"配音第几句/渲染百分之多少"），比 tqdm 的单一进度条模型更贴合这里的多层级结构。

### 2.2 ProgressReporter Protocol

不让 `pipeline.py`/`render/*.py` 直接依赖 `rich`（保持这些模块可以被非 CLI 场景复用、离线测试）。改为定义一个协议，和现有的 `LLMProvider`/`TTSEngine` 走一样的注入方式：

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ProgressReporter(Protocol):
    def stage_start(self, stage: str) -> None:
        """某阶段开始真正执行（不是被跳过）。"""
        ...

    def stage_skip(self, stage: str) -> None:
        """某阶段因为 mtime 新鲜度检查被跳过。"""
        ...

    def stage_done(self, stage: str) -> None:
        """某阶段执行完成。"""
        ...

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        """阶段内部子进度。目前只有 voice（第几句配音）和 render（百分之多少）会调用。"""
        ...

    def episode_start(self, number: int, index: int, total: int) -> None:
        """批量模式下，开始处理第 index/total 集（number 是集数）。单集模式不调用。"""
        ...
```

新增一个默认的无操作实现 `NullProgressReporter`，所有方法都是空函数体。

### 2.3 pipeline.py 的改动

`run_pipeline` 增加一个关键字参数：

```python
async def run_pipeline(
    cfg: ProjectConfig,
    provider: LLMProvider,
    *,
    from_stage: str = "ingest",
    only: Sequence[str] | None = None,
    force: bool = False,
    tts_engine: TTSEngine | None = None,
    episode: int | None = None,
    reporter: ProgressReporter | None = None,
) -> list[str]:
```

`reporter` 缺省为 `None` 时内部替换成 `NullProgressReporter()`。这保证现有全部测试（546 passed, 4 skipped）不用改一行就能继续通过——没人传 `reporter`，行为和以前完全一样。

每个阶段块的 `if "<stage>" in wanted:` 分支里，在决定"要不要跑"的判断处调用：

```python
if force or not _is_fresh(outputs, inputs):
    reporter.stage_start(stage_name)
    run_xxx(cfg, ...)
    reporter.stage_done(stage_name)
else:
    reporter.stage_skip(stage_name)
```

批量模式下（`episode is None`），在 `target_numbers` 的循环最外层调用一次 `reporter.episode_start(number, index, len(target_numbers))`。

### 2.4 voice 阶段子进度

`render/tts.py` 的 `synthesize_track` 已经有一个按 chunk 遍历的循环（`plan_chunks` 算出的 `planned` 列表长度就是总数）。改动只是在循环体里插入一行调用：

```python
for index, (text, hold_after) in enumerate(planned, start=1):
    reporter.substep("voice", index, len(planned), text[:20])
    ...
```

`synthesize_track` 需要新增一个 `reporter: ProgressReporter | None = None` 参数并沿用同样的"缺省 Null"模式。`run_voice`（pipeline.py）把自己收到的 `reporter` 转发进去。

### 2.5 render 阶段子进度：真实 ffmpeg 进度

这是唯一需要改动 ffmpeg 调用层内部实现的地方。`src/tenmin/render/ffmpeg.py` 的 `run(args)` 目前是一次性阻塞的 `subprocess.run(...)`，拿到完整 stdout/stderr 后才返回。

新增一个专门给 `render_video` 用的变体，而不是改动通用的 `run()`（`audio.py`/`preflight` 等调用方不需要进度，保持不变）：

```python
def run_with_progress(
    args: list[str],
    *,
    total_seconds: float,
    on_progress: Callable[[float], None] | None = None,
) -> str:
    """在 args 末尾自动插入 -progress pipe:1，流式读取 ffmpeg 的进度输出，
    每收到一个 out_time_ms 就换算成 0.0~1.0 的比例调用 on_progress。
    返回值和 run() 一样：完整 stderr 文本。非零退出同样抛 FFmpegError。
    """
```

`ffmpeg -progress pipe:1` 会往 stdout 按 `key=value` 格式周期性输出 `out_time_ms=<微秒数>`、`progress=continue|end` 等字段，一次编码结束时以 `progress=end` 收尾。`out_time_ms / 1_000_000 / total_seconds` 就是完成比例（钳制到 `[0, 1]`）。这里用 `subprocess.Popen` + 逐行读取 stdout，而不是 `subprocess.run`，因为要在编码过程中拿到中间输出。

`render_video`（`render/video.py`）新增可选参数 `reporter: ProgressReporter | None = None`，内部改用 `run_with_progress(args, total_seconds=timeline.total_seconds, on_progress=lambda p: reporter.substep("render", int(p * 100), 100, ""))`。`run_render`（pipeline.py）负责把自己收到的 `reporter` 转发进去。

### 2.6 cli.py 的改动

`run()` 命令里构造一个 `RichProgressReporter`（新模块 `src/tenmin/progress.py`），把它作为 `reporter=` 传进 `run_pipeline`。这个类内部维护一个 `rich.progress.Progress`（作为 context manager，`with progress:` 包住整个 `run_pipeline` 调用）和若干 `rich.progress.TaskID`，把 Protocol 的 5 个方法映射成 `progress.add_task`/`progress.update`/`progress.advance` 调用。

`RichProgressReporter` 是这个功能里唯一直接 import `rich` 的地方；其余所有模块只依赖 Protocol，方便离线单测（用一个记录调用的 `FakeReporter` 即可，不需要真的渲染终端）。

## 3. 可视化布局

### 3.1 单集模式

```
才女的侍从 E02
✓ ingest
✓ signals
⏭ script    (已是最新，跳过)
⏭ docgen    (已是最新，跳过)
▶ voice     配音中 [━━━━━━━━━━━━━━━━░░░░░░░░] 8/12 句
  timeline
  audio
  render
```

- `✓`（绿色）= 真正执行完成
- `⏭`（灰色）= 因 mtime 新鲜度跳过，未执行
- `▶` + 内嵌进度条 = 当前正在执行、且有子进度的阶段（只有 voice/render 会有）
- 纯灰色文字、无图标 = 还没轮到的阶段

`render` 阶段跑起来后长这样（带百分比和剩余时间估算，`rich.progress.Progress` 自带的 `TimeRemainingColumn` 直接可用）：

```
▶ render    渲染中 [━━━━━━━━━━━━━━━━━━░░░░░░] 76% (剩余 ~18s)
```

### 3.2 批量模式

在单集模式的面板上方加一层外框，显示"第几集/共几集"：

```
总进度：第 2/5 集

我是不才恶女 E02
✓ ingest
✓ signals
...
```

每一集跑完后，用 `rich.Live` 的刷新机制把这一集的面板"替换"成下一集的面板，而不是在终端里往下堆叠滚动（避免跑 5 集刷屏 40 行）。只有最顶层的"第 N/M 集"计数器在多集之间持续存在。

## 4. 错误与中断处理

- **正常捕获的异常**（`FileNotFoundError`/`ValueError`/`FFmpegError`/`RuntimeError`，即现有 `cli.py` 的 `except` 元组能接住的那些）：进度条冻结在出错的那个阶段，把该阶段的图标改成红色 `✗`，然后 `rich.Progress` 的 context manager 正常退出（保留最后一帧在 scrollback 里），紧接着现有的红色 stderr 一行报错逻辑照常打印，不做任何改动。
- **Ctrl-C（KeyboardInterrupt）**：不加任何自定义的"确认退出"提示。`rich.Progress` 作为 `with` 语句使用，Ctrl-C 会被其 `__exit__` 自动捕获并做清理（恢复光标可见性、恢复终端状态），之后中断正常向上传播，Python 正常退出。

## 5. 测试策略

延续项目"纯函数严格断言、I/O 边界打桩、不真的联网/真的编码"的一贯做法：

- `ProgressReporter`/`NullProgressReporter` 本身几乎没有逻辑，不需要专门测试；但要有一个 `FakeReporter`（记录每次调用到 `self.calls` 列表）加进 `tests/fakes.py`，供 `test_pipeline.py`/`test_render_tts.py`/`test_render_video.py` 断言"该阶段被跳过时调的是 `stage_skip` 不是 `stage_start`"这类行为。
- `run_with_progress` 的测试用一个 monkeypatch 过的假 `Popen`（stdout 逐行喂固定的 `out_time_ms=...`/`progress=end` 文本），断言 `on_progress` 被调用的参数序列，不真的跑 ffmpeg。
- `RichProgressReporter` 只做最小的烟雾测试（构造、调用 5 个方法不抛异常），不断言具体的终端渲染字符——`rich` 自己的渲染逻辑不是这个项目要测的东西，呼应 v1/v2 一贯的"我们只测自己拼的参数对不对，不测第三方库内部对不对"的原则。

## 6. 已知取舍 / 限制

1. `render` 的百分比来自 `out_time_ms / total_seconds`，如果时间轴发生了《design doc 2026-09-05》里提到的"画面被钳制导致画面比音频短"的情况，`total_seconds` 是音频时长，`out_time_ms` 走的是 ffmpeg 内部时间轴，两者在钳制场景下可能不完全对齐，导致百分比在结尾处出现轻微跳变或提前到 100%——这是已知的、可接受的展示层小瑕疵，不影响渲染本身是否成功。
2. 非单调片段顺序导致的高内存占用（见多集 CLI 相关的历史 follow-up）不受本设计影响，进度条只是忠实展示 ffmpeg 报告的进度，不会让这个问题变得更好或更坏。
3. 本设计只覆盖 `tenmin run`；`tenmin init`/`tenmin inspect` 不受影响，不加进度条。
