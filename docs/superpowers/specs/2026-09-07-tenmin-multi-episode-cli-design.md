# tenmin 多集项目与命令行参数化设计

日期：2026-09-07
状态：已确认设计，待编写实施计划

## 背景与目标

当前 tenmin 的每个项目（`work/<slug>/project.yaml`）在实践中只能完整跑通 **一集**：
`_only_episode(cfg)` 在管线代码里硬编码取 `cfg.episodes[0]`，`voice`/`timeline`/`audio`/`render`
四个阶段都只处理这一集；`script`/`docgen` 阶段产出的 `03_script/script.json`、
`out/解说方案.md`、`out/narration.txt` 也是**项目级共享单文件**，没有集数区分。

用户的诉求：
1. 一个番剧系列应该对应**一个** tenmin 项目，而不是每集建一个项目目录。
2. 运行命令应该能够直接通过参数指定要处理哪一集的 srt/视频文件，不必每次手动编辑
   `project.yaml` 的 `episodes:` 列表、手动把文件复制进 `srt/`、`video/` 目录。

本设计在保留现有单集处理逻辑正确性的前提下，把 tenmin 改造成"一个项目、多集復用"的模型，
并给 `tenmin run` 增加显式的集数与文件参数。

## 已确认的设计决策（与用户逐项确认，均已批准）

1. **产出文件全部按集数命名。** 不仅是已经按集数命名的文件（`01_dialogue/E01...`、
   `04_voice/E01/...`、`05_timeline/E01...`、`06_audio/E01...`、`07_render/E01.mp4`），
   目前项目级共享的三个文件也必须改为按集数命名：`03_script/script.json`、
   `out/解说方案.md`、`out/narration.txt`。
2. **集数必须显式传参，不做推断。** 新增 `--episode N`（必填整数），不从文件名解析、
   不自动累加。
3. **传入 `--srt`/`--video` 时自动注册。** 会把文件复制进
   `work/<slug>/srt/E{N:02d}.srt` 与 `work/<slug>/video/E{N:02d}.mp4`，并在
   `project.yaml` 的 `episodes:` 列表里新增或更新对应 `number: N` 的条目，使得之后
   `tenmin run <slug> --episode N`（不再传 `--srt`/`--video`）也能直接复现。
4. **`render:` 配置维持项目级共享，不做按集覆盖。** `duck_db`、`font_size`、
   `fade_out_seconds`、`outro_card_seconds`、`outro_message` 等继续是整个项目统一的
   一套值，适用于该项目下的所有集。
5. **不带新参数的 `tenmin run <slug>` 走批处理模式**：遍历 `project.yaml` 里
   `episodes:` 列表中已注册的**所有**集，按现有的 mtime 新鲜度判断 + `--force` 语义
   逐一跑完整 8 阶段管线。

## 设计详解

### 一、CLI 行为（`src/tenmin/cli.py`）

`tenmin run <slug>` 新增三个可选参数：`--episode N`、`--srt PATH`、`--video PATH`。
三种调用形态：

| 调用方式 | 行为 |
|---|---|
| `tenmin run saijo --episode 1 --srt E01.srt --video E01.mp4` | **注册并运行**：把两个文件复制进 `work/saijo/srt/E01.srt` + `work/saijo/video/E01.mp4`，在 `project.yaml` 的 `episodes:` 里新增/更新 `number: 1` 条目，然后只运行第 1 集的完整管线。 |
| `tenmin run saijo --episode 1`（不带 `--srt`/`--video`） | **只运行第 1 集**：要求该集已经在 `project.yaml` 中注册过，否则报错并提示先用上一行的方式注册。 |
| `tenmin run saijo`（不带任何新参数） | **批处理模式**：按 `project.yaml` 的 `episodes:` 列表顺序，逐集跑完整管线，遵循既有的 mtime 新鲜度跳过逻辑与 `--force`。 |

校验规则：
- `--srt` 与 `--video` 必须同时提供，不允许只传一个（只传一个视为参数错误）。
- 一旦提供了 `--srt`/`--video`，必须同时提供 `--episode`（否则报错，提示需要显式指定集数）。
- `--episode` 单独提供是合法的（对应"只运行已注册的某一集"场景）。

### 二、按集数产出路径（`src/tenmin/pipeline.py` 的 `Paths` 类）

三个当前**项目级共享**的单文件改为**按集数命名**：

| 修改前 | 修改后 |
|---|---|
| `03_script/script.json` | `03_script/E{episode:02d}.script.json` |
| `out/解说方案.md` | `out/E{episode:02d}.解说方案.md` |
| `out/narration.txt` | `out/E{episode:02d}.narration.txt` |

已经按集数命名的文件（`01_dialogue/E01...`、`04_voice/E01/...`、`05_timeline/E01...`、
`06_audio/E01...`、`07_render/E01.mp4`）保持不变。

代码层面需要的改动：

- `Paths.script`、`Paths.table`、`Paths.narration` 由无参属性改为
  `Paths.script(episode)`、`Paths.table(episode)`、`Paths.narration(episode)` 方法，
  接收集数参数。
- 移除 `_only_episode(cfg)`（当前硬编码取 `cfg.episodes[0]`），替换为
  `_find_episode(cfg, episode_number)`：按集数在 `cfg.episodes` 中查找对应的
  `EpisodeConfig`，找不到时抛出清晰的错误信息（提示需要先用
  `--srt`/`--video`/`--episode` 注册该集）。
- `run_script()`、`run_docgen()`、`run_voice()`、`run_timeline()`、`run_audio()`、
  `run_render()` 的函数签名都改为显式接收 `episode: int` 参数，不再隐式取
  `episodes[0]`。
- `run_pipeline()` 顶层编排逻辑：批处理模式下遍历 `cfg.episodes` 中所有集，逐一跑完整
  8 阶段管线；单集模式下只跑指定 `episode` 的 8 阶段。
- `run_ingest()`、`run_signals()` 保持不变（这两个阶段本来就遍历 `cfg.episodes`
  中的所有集，按集数写出各自的 dialogue/signals 文件，不受本次改动影响）。

**已有 Saijo E02 数据的迁移：**
`work/saijo/03_script/script.json` → `work/saijo/03_script/E02.script.json`；
`out/解说方案.md` → `out/E02.解说方案.md`；
`out/narration.txt` → `out/E02.narration.txt`。
需要一次性迁移（迁移脚本或代码里的兼容性检查），确保切换到新的按集路径时，这三个
已有的 E02 产出文件不会丢失或变成孤儿文件。

### 三、测试套件影响

- **`tests/test_pipeline.py`（影响最大）**：所有直接调用
  `run_script()`/`run_docgen()`/`run_voice()`/`run_timeline()`/`run_audio()`/
  `run_render()` 的测试都需要补上显式的 `episode=N` 参数；断言
  `Paths(...).script`/`.table`/`.narration` 为裸路径的测试需要改成
  `Paths(...).script(episode)` 等方法调用形式。像
  `test_run_timeline_uses_config_font_size` 这类单阶段测试：补上
  `episode=2`（与现有 fixture 保持一致），行为不变、测试继续通过。
  需要新增的测试：`_find_episode()` 在请求的集数未注册时抛出清晰错误；批处理模式下
  `run_pipeline()` 遍历多集；"注册并运行"流程（复制文件 + 更新 YAML）。
- **`tests/test_cli.py`**：为三种调用形态（注册并运行、运行已注册的集、批处理）新增
  测试，以及校验类错误的测试（只传 `--srt`/`--video` 之一、传了 `--srt`/`--video`
  却没传 `--episode`）。
- **`tests/test_config.py`**：基本不受影响——`ProjectConfig.episodes` 本来就支持
  列表，本次不改配置 schema 本身。
- **黄金/快照测试**（`tests/fixtures/saijo_e02.srt`、`tests/snapshots/saijo_e02.*`）：
  以内容而非路径命名规则为键，只要按集数的路径方法对第 2 集正确产出 `E02.*`
  （与现有文件名完全一致），就不受影响。
- 总体策略：只有直接涉及这 3 个新按集路径、或 6 个阶段函数签名的测试才需要改动——
  是一次外科手术式的改动，不是重写。

## 范围之外（本次不做）

- 不支持按集覆盖 `render:` 配置（已在决策 4 中明确排除）。
- 不做文件名自动解析集数、不做自动累加集数编号（已在决策 2 中明确排除）。
- 不改变现有单集内部的渲染/字幕/音频逻辑（那些是已完成并锁定的功能，与本设计无关）。
