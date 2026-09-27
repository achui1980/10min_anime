# 管线健壮性与成本（Spec A）

日期：2026-09-27　基线分支：`feat/asr-translate`

这是「优化三连」的第一份（A 管线健壮性与成本 → B 音频/字幕质量 → C 解说稿质量）。硬字幕 OCR 另开一份，不在这三份里。

## 背景与动机

2026-09-27 的调研拿到的证据：

- `pipeline.py:987-994` 把 `project.yaml` 加进了**每个**阶段的输入；`register_episode`（`pipeline.py:676`）每次登记都会重写这个文件；ingest 是全局阶段，会无条件重写所有集的 `dialogue.json`。结果是改一个 `render.font_size`、或者登记一集新番，整季的 script 全被判过期，每集重付一次 LLM。实例：saijo 的 `dialogue.json`（09-22）比 `script.json`（09-21）新。
- `llm.py:434-446` 的 `last_usage` 在生产代码里没有读取方，LLM 花了多少 token 看不到。
- 所有 config 都没有 `extra="forbid"`，`render: {font_sise: 99}` 这种拼错会被静默忽略。
- `cli.py:259` 的 `load_project`、`cli.py:277` 的 `register_episode` 在 `PIPELINE_ERRORS` 的 try 外面，路径写错、yaml 语法错会打出整页 traceback。Gemini 路径（默认 provider）不包装 `google.genai.errors.APIError`，也没有传输层退避。
- `register_episode` 走 `safe_load` → `safe_dump`，yaml 里的注释（包括 `init` 生成的 14 行说明）会被全部冲掉。
- 用户想先把每集的 `op_range`/`ed_range` 写进 `project.yaml`、之后再登记视频。现在做不到：只有 op/ed、没有 srt/video 的条目过不了 `EpisodeConfig._require_a_source`（`config.py:88-103`），整个项目都加载失败。

## 目标

1. 新鲜度只对「这个阶段真正读到的配置」敏感：改 render 的参数只重跑 render，登记新集不影响已有的集。
2. 每次 LLM 调用的用量落盘，方便看成本。
3. 配置字段拼错、YAML 键重复直接报错，报错信息是中文且点名字段。
4. 常见的用户错误不再打 traceback；Gemini 路径的错误与退避跟其他 provider 对齐。
5. 允许「预填」的集条目（只有 op/ed）；登记写回 yaml 时保留注释和顺序。

## 非目标

- 不改 `_is_fresh` 本身的 mtime 判据。
- 不重构 `run_pipeline` 的结构（StageSpec 表、拆分 `pipeline.py` 以后另做）。
- 不做 `doctor` / `status` / `clean` 命令。
- 不做按集覆盖 render 配置。

## 设计

### 1. 按阶段的配置切片

**机制。** 每次 `run_pipeline` 开始时，对每个阶段把它读到的那部分**解析后的**配置序列化成 JSON，写到 `work/<slug>/.config/<stage>.json`（按集的阶段是 `.config/E{NN}.<stage>.json`）。写法跟 `save_glossary` 一样：payload 跟盘上逐字节相同就不碰文件。然后阶段的新鲜度输入里用这个切片文件**替换掉** `cfg.config_path`，`_is_fresh` 的逻辑不动。

序列化的是 pydantic 模型的 `model_dump(mode="json")`，JSON 用 `sort_keys=True`、`ensure_ascii=False`、固定缩进。所以 yaml 里改注释、调顺序、把默认值显式写一遍，都不会触发重跑。

**切片内容。** 以整个子 config 为单位，再扣掉一张「运维旋钮排除表」。映射放在一个模块级常量表里（新模块 `src/tenmin/config_slices.py`），初稿如下；实现计划阶段要逐个 grep 各阶段实际读了哪些字段，以代码为准修正：

| 阶段 | 切片内容 |
|---|---|
| ingest | `locale`、`ingest`、`credits`、`asr`，加本集的 `EpisodeConfig` |
| translate | `llm`、`glossary`，加本集的 `EpisodeConfig` |
| signals | `signals` |
| script | `llm`、`validate_script`、`target_seconds`、`mode`、`glossary`、`show`、`render.rate` |
| docgen | `render.rate` |
| voice | `render` 里的 TTS 字段（`voice`、`rate`） |
| timeline | `render` 里的字幕与时间轴字段 |
| audio | `render` 里的混音字段 |
| render | `render` 里的编码、字幕样式、片尾字段，加本集的 `EpisodeConfig` |

`render` 被好几个阶段按字段拆开用，这是整节里唯一需要字段级映射的地方；其他子 config 都整段进切片。拆不清楚的字段宁可多挂一个阶段（多重跑一次），也不能漏挂（该重跑没重跑）。

**排除表**（不影响产物内容、从不触发重跑的运维旋钮）：`llm.timeout_seconds`、`llm.read_timeout_seconds`、`llm.total_timeout_seconds`、`llm.transport_max_attempts`、`llm.max_attempts`、`llm.script_concurrency`、`render.tts_max_attempts`、`render.tts_concurrency`、`render.tts_proxy`、`render.tts_connect_timeout`、`render.tts_receive_timeout`、`render.tts_chunk_timeout_seconds`、`render.ffmpeg_path`、`render.ffprobe_path`。`validation_retries`、`budget_rewrite_rounds` 会改变产物，**不**进排除表。

**完整性测试。** 递归遍历 `ProjectConfig.model_fields`（含子 config），断言每个叶子字段要么挂在至少一个阶段上，要么在排除表里。以后新增字段忘了登记，这个测试会直接失败。

**只放本集的 `EpisodeConfig`。** 登记第 11 集只改变 E11 的切片，E1~E10 的切片逐字节不变，不会被判过期。

**升级时不整季重跑。** 切片文件**第一次**创建时，把 mtime 设成 epoch 0。代价：升级前改过、但还没跑过的配置，升级后第一次运行检测不到。这是有意的取舍，写进升级说明。之后再修改的切片按正常 mtime 写。

**ingest / signals 跳过相同内容的写入。** `dialogue.json`、`signals.json` 的 payload 跟盘上逐字节相同就不写，保留旧 mtime，这样下游不会因为「内容没变、mtime 变了」被连带判过期。

### 2. LLM 用量落盘

script 阶段每跑一集，写一份 `03_script/E{NN}.usage.json`，内容是这一集所有调用的列表（首稿、语义校验重试、时长返工各一条），每条记 provider、model、轮次类型、prompt / completion / cached token 数（provider 没给的就写 `null`）、耗时（秒）。translate 阶段同样写 `zh/E{NN}.usage.json`。

这些文件只用来观察，**不是**任何阶段的新鲜度输入。用原子写。数据来源是现有的 `last_usage`，需要让 Gemini 和 OpenAI 兼容两条路径都把用量填上。

### 3. 配置严格校验

新建统一基类 `StrictModel(BaseModel)`，设 `model_config = ConfigDict(extra="forbid")`，`ProjectConfig`、`EpisodeConfig` 和 8 个子 config 全部继承它。`load_project` 捕获 `ValidationError`，把「不认识的字段」翻成中文报错，带上完整路径，例如：`project.yaml 里 render.font_sise 不是已知字段`。能给出拼写建议（`difflib.get_close_matches`）就给。

YAML 键重复时报错，例如同一层写了两个 `render:`。实现方式是自定义 loader，在构造 mapping 时检查键有没有重复（ruamel 的 round-trip 模式本身也会对重复键报错，两处要一致）。

### 4. CLI 与 Gemini 错误处理

- `cli.py` 里的 `load_project` 和 `register_episode` 挪进 `PIPELINE_ERRORS` 的 try。SRT 路径不存在、yaml 语法错、校验失败都给中文一行报错，退出码与现有管线错误一致。
- `GeminiProvider.complete` 把 `google.genai.errors.APIError` 包成 `LLMHTTPError`（带状态码与响应摘要）；网络连接类异常同样包进 `LLMError` 族。
- Gemini 路径接上现有的传输层退避：只对 429 / 5xx / 连接类异常重试，次数用 `transport_max_attempts`，退避走模块级 `_sleep` / `_rand`（测试 monkeypatch 这两个函数，不许直接写 `asyncio.sleep`）。其他 4xx 立即失败。

### 5. 预填集条目与保留注释

**预填条目。** 允许 `episodes` 里出现只有 `number` 和 `op_range`/`ed_range`、没有 srt/video 的条目。`_require_a_source` 的检查从 yaml 加载时挪到运行时：

- 批处理模式跳过这类条目，打印一行提示：`第 3 集还没有 video，已跳过`。
- `--episode 3` 不带 `--video`、而第 3 集是预填条目时，给中文报错，告诉用户要加 `--video`。
- `--episode 3 --video x.mkv` 会并进已有的预填条目，**保留** `op_range`/`ed_range`。现在的代码已经是这个行为，补测试锁住。
- `tenmin inspect` 把预填条目列出来，标注「未登记视频」。

**保留注释。** `register_episode` 写回 yaml 改用 `ruamel.yaml` 的 round-trip 模式：只改 `episodes` 这个序列里对应的条目（改已有条目的 srt/video 字段，或追加一个新条目），其余内容（包括注释、键的顺序、引号风格）原样保留。依然走原子写。`ruamel.yaml` 进主依赖。

## 测试

每一项都按 TDD 来：先写失败的测试，再实现。重点用例：

- 切片：映射完整性测试；改 `render.font_size` 只让 render 过期；改 `llm.timeout_seconds` 什么都不过期；改 yaml 注释什么都不过期；登记 E11 后 E1~E10 全部新鲜；第一次创建的切片 mtime 是 0；ingest 在内容不变时不改 `dialogue.json` 的 mtime。
- 用量：用 fake provider 跑一轮首稿加一轮返工，`usage.json` 里是两条记录；provider 不给用量时字段是 `null`。
- 严格校验：未知字段报错并点名路径与拼写建议；重复键报错。
- CLI：SRT 不存在、yaml 语法错都不出 traceback；Gemini 的 `APIError` 变成 `LLMHTTPError`；429 会退避重试，400 立即失败，测试里不真睡。
- 预填与注释：只填 op/ed 的条目能加载；批处理会跳过并提示；登记后 op/ed 还在；登记前后 yaml 的注释逐字保留。
- 端到端：在 `tests/` 的临时项目里依次改 render 字段、登记新集，断言每次只有预期的阶段重跑。
- 全量 `uv run pytest tests/ -q` 通过，MiniMax 的现有测试行为不变。

## 风险

- 字段映射挂漏会导致「该重跑没重跑」，比多重跑更难发现。缓解：完整性测试，加上拿不准就多挂。
- epoch 0 的升级取舍，见上。
- ruamel 的 round-trip 在极少见的 yaml 写法（锚点、多文档）上可能跟 PyYAML 的解析结果不同。读配置仍然用现有的解析路径，ruamel 只用在写回。
