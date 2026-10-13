# AGENTS.md

给在这个仓库里工作的 AI agent 看的说明。

## 项目是做什么的

tenmin（10 分钟看番剧）：把番剧字幕（SRT）或生肉视频 + 视频，自动加工成"解说方案"文档 + 配音 + 烧字幕的成片。9 个阶段，每个阶段读上游产物、写自己的产物，靠文件 mtime 决定要不要重跑：

```
ingest → translate → signals → script → docgen → voice → timeline → audio → render
```

- `ingest`：解析 SRT，生成对白轨（`01_dialogue/`）。**对白轨从哪来是四岔**（`ingest/resolve.py`）：手传 SRT → 视频里的软字幕轨直抽（`E{NN}.embedded.srt`，每次重抽）→ 声明了硬字幕时画面 OCR（`E{NN}.ocr.srt`，按 mtime 复用；只在 `cfg.hardsub_enabled(episode)` 为真时走，跑不了就报错、**不回落到语音转写**）→ 语音转写（`E{NN}.asr.srt`，按 mtime 复用）。四条都归一成一个 SRT 路径，所以 `build_track` 的形态不变；`DialogueTrack.source` 记的是 `"srt"` / `"ocr"` / `"asr"`（`ocr` 跟 `srt` 一样过 OpenCC 繁转简，`asr` 不过）。
- `translate`：按 `source` 分三路（判据是那个字段，不做语言检测）。`"srt"` 跳过，连 `zh/` 目录都不建；`"asr"` 把日语对白逐条译成简体中文（`zh/E{NN}.zh.json` + 交付物 `out/E{NN}.zh.srt`），并把新认出的专有名词并进项目级累积表 `zh/glossary.json`（会被 script 的 prompt 读）；`"ocr"` **不调 LLM、不碰 provider**，按 `translate/lines.py` 的 `passthrough_track`（筛选同 `select_translatable`，zh 取 ingest 已繁转简的原文）照样写那两份产物，不读不写累积表、不写 `zh/E{NN}.usage.json`。`pipeline.run_translate` 的 `provider` 因此标成 `LLMProvider | None`：srt/ocr 传 `None` 合法，asr 集拿到 `None` 在写任何文件之前抛中文 `ValueError`。但 `cli.run` 只要阶段里含 translate 就照旧构造 provider，所以全是 OCR 集的项目跑 `--only translate` 仍要 API key（刻意没改，YAGNI）。
- `signals`：识别静音间隙、语速变化等"高能点"信号（`02_signals/`）。
- `script`：调用 LLM，把信号转成分幕解说稿（`03_script/`）。
- `docgen`：把 script.json 渲染成人类可读的对照表 + 纯配音文本（`out/`）。
- `voice`：调用 TTS（edge-tts），把配音文本合成语音（`04_voice/`）。
- `timeline`：根据真实配音时长重新计算时间轴，生成字幕（`05_timeline/`）。
- `audio`：原声压低（ducking）+ 配音叠加混音（`06_audio/`）。
- `render`：一次性剪辑、拼接、烧字幕、合成音轨，出片（`07_render/`）。

一个 project（`work/<slug>/project.yaml`）对应**一部番**，可以登记多集。所有阶段产物都是按集号加前缀的（`Paths` 类，见 `src/tenmin/pipeline.py`），例如 `03_script/E02.script.json`、`out/E02.narration.txt`、`07_render/E02.mp4`。

## 关键工具规则（非常重要）

**跑本项目的 pipeline 命令和 pytest 时，一律用裸的 `uv run <cmd>`，绝对不要套 `rtk proxy` / `rtk pytest` / `rtk` 前缀。** 本项目的中文输出会把 rtk 的 UTF-8 抓取层搞崩。裸的 `rtk ls` / `rtk grep` / `rtk git` / `rtk read` 是安全的，可以正常用。

```bash
# 对：
uv run pytest tests/ -q
uv run tenmin run saijo --episode 2 --only script --force

# 错（会崩，别用）：
rtk pytest tests/
rtk proxy uv run tenmin run saijo
```

**`rtk find` 会拒绝 `-not` / `-exec`，而且 shell 层会把裸 `find` 也路由给 rtk。** 所以清 `__pycache__` 必须写绝对路径：

```bash
/usr/bin/find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} +
```

**做变异测试一律带 `PYTHONDONTWRITEBYTECODE=1`。** CPython 默认的 pyc 失效判据只看「源文件 mtime 的整秒值 + 字节数」，所以**等长**替换（比如把 `audio.wav` 改成 `audio.mp3`）且改坏与还原都发生在同一整秒内时，那两个记录值一个字节都不变 → 变异过的字节码被无限期当成有效，造成**假存活 / 假绿**。本分支真踩过一次。

如果需要重装 `.venv`（`uv sync --reinstall`），edge-tts 走公司 Zscaler MITM 代理会报 SSL 证书错——需要把 Zscaler CA 追加进 certifi 的 cacert.pem：

```bash
cat /path/to/zscaler_ca_bundle.pem >> .venv/lib/python3.14/site-packages/certifi/cacert.pem
```

## 代码结构

- `src/tenmin/config.py`：`ProjectConfig`（15 个字段）与它的 9 个子 config（`.locale` / `.llm` / `.ingest` / `.credits` / `.signals` / `.validate_script` / `.render` / `.asr` / `.ocr`）、`EpisodeConfig`，`Settings`（`BaseSettings`，读 `.env`，`env_prefix="TENMIN_"`）。**全项目所有「经验阈值」的唯一权威来源**；分家的判据是「这部番想要什么」的创作旋钮进 config，「物理上不可能／数据坏了」的合法性边界留在各模块的模块级常量里。各阶段模块只保留 `DEFAULT_XXX.field` 的模块级别名。改了子 config 的个数就来改这句数字（判据是 `ProjectConfig.model_fields` 里注解是 `BaseModel` 子类的那些）。

  这 11 个配置模型（项目、9 个子配置、集配置）继承 `StrictModel`（`extra="forbid"`）；`Settings` 除外，因为 `.env` 可含别的程序的变量。`load_project` 用 `_UniqueKeyLoader`（PyYAML SafeLoader + 同层重复键检查）；YAML 语法/重复键或 pydantic 校验失败都包装为 `ProjectConfigError(ValueError)`，点名文件和完整字段路径（如 `render.font_sise`、`episodes[0].op_rang`），未知字段给拼写建议。`EpisodeConfig` 可预填仅有集号/OP/ED、没有 srt/video 的条目；运行时按 `has_source` 判：批处理跳过并警告，单集运行要求 `--video`，inspect 标「未登记视频」；登记时合并同一条并保留 OP/ED。

  `AsrConfig` 只有 2 个字段（`model` / `language`），但有一条**不在代码里的操作约束**：`E{NN}.asr.srt` 的新鲜度只比源视频 mtime、**刻意不看 `AsrConfig`**，所以换了 `asr.model` 必须自己删那份 SRT。理由是它是一份人能手改的产物，按指纹失效会把手改静默冲掉。

  `OcrConfig` 有 9 个字段（`enabled` / `ai_correct` / `sample_fps` / `crop_top` / `crop_bottom` / `center_tolerance` / `similarity` / `min_frames` / `language`），同样有那条**不在代码里的操作约束**：`E{NN}.ocr.srt` 的新鲜度只比源视频 mtime、**刻意不看 `OcrConfig`**，改了 OCR 参数必须自己删那份 SRT（理由同上）。`crop_bottom` 可选，按原画面高度比例指定字幕带下沿；取帧仍裁到底，Vision 识别后按框中心的 y 过滤下层字幕，防止物理裁剪边缘漏认中文。「这一集带不带硬字幕」只有一个判定入口 `ProjectConfig.hardsub_enabled(episode)`：`EpisodeConfig.hardsub` 写了 true/false 就听它的，`None` 跟随 `ocr.enabled`。`config_slices` 在 `hardsub is None` 时**不**把这个键写进 EPISODE 切片——否则升级后 timeline/audio/render 的切片全部被改写、存量集白白重跑。OCR 引擎只有 Apple Vision（pyobjc，optional extra `ocr`，仅 macOS），pyobjc 的 import 只許写在 `ingest/ocr.py` 的 `_recognize` 函数体内。硬字幕片源会把**居中**的 OP/ED staff 字一起认进来（两侧的靠 `center_tolerance` 挡掉），所以这类片源要手填 `op_range` / `ed_range` 或 `credits.default_*`，交给三级回退去剔除。ingest 的配置切片含 `ocr`，所以改任何 `ocr.*` 旋钮都会让 ingest 重跑一次（便宜：缓存照旧复用、产物字节不变就不写，下游不连带），但**不会**让已有的 `.ocr.srt` 失效。

  **OP/ED 区间是三级回退**，改 ingest 的 credits 相关代码前先分清自己在哪一级：`EpisodeConfig.op_range`/`ed_range`（逐集手填）→ `CreditsConfig.default_op_range`/`default_ed_range`（项目级手填，ED 终点允许 `None` = 到片尾，在 `normalize._resolve_manual_range` 里按**这一集**的 duration 解析）→ `credits.find_credit_ranges` 的启发式推断。关键点：**前两级（手填）还会直接驱动 `in_credit_window`** —— 拿到确定区间时窗就是那两段（各留 `manual_window_margin`=5 秒余量），`credit_head_window`/`ed_keyword_window_seconds` 那对盲窗完全不参与；两级都空才走盲窗，且那条路与改动前**逐字节等价**（实测新旧代码各跑一遍，13 份 `01_dialogue/*.json` 哈希全同）。手填模式刻意不受 `credit_window_max_ratio` 约束（那是给盲窗兜底的，静默收缩用户的显式声明比覆盖过宽更难查）。实测收益：接住 3 条落在 300 秒盲窗外的 staff 行、同时救回 5 条落在盲窗内被规则 4/5 误杀的真台词；代价是填错会在**你填的区间内**误判。`normalize.credit_range_source()` 报告实际生效的是哪一级（`tenmin inspect` 用），它刻意复用 `_resolve_manual_range` 而不是自己再判一遍「字段填了没」——项目级默认可能填了却在某一集上解析不出合法区间。
- `src/tenmin/pipeline.py`：`Paths` 类（每阶段产物路径，全部按集号 `E{episode:02d}` 前缀），`STAGES` 列表（9 项，`translate` 在 index 1），`run_pipeline()` 顶层编排（支持单集/批量两种模式，靠 `episode: int | None` 区分），`register_episode()`（`--episode --video` 注册新集，`--srt` 可省；可选成对的 `op_range`/`ed_range` 只在这一集两个字段都空时写入，给 `tenmin import` 用；只给一半抛 `ValueError`）。

  **新鲜度按阶段配置切片**：`run_pipeline` 先写解析后的 `.config/E{NN}.<stage>.json`（signals 是项目级 `.config/signals.json`），替代整个 `project.yaml` 作为阶段输入。改阶段的配置读取点时核对 `src/tenmin/config_slices.py` 的 `STAGE_FIELDS`，同步更新对应阶段；`tests/test_config_slices.py` 只检测字段是否挂到任一阶段，抓不到「已挂在其他阶段的字段又被本阶段读取」。运维旋钮由 `EXCLUDED` 排除。切片内容不变不碰文件；首次创建 mtime 置 epoch 0，升级前已修改却未跑的配置首次不会失效旧产物；这种情况及删掉 `.config/` 后需 `--force`。

  ingest/signals **产物字节不变就不写**，避免下游因 mtime 变动连带重跑；全局阶段的新鲜度另由 `.config/{ingest,signals}.done` 记录上次完成时间及**每份产物 SHA-256 摘要**（`_is_fresh_stamped` 比对摘要和输入时间）。重跑前先将戳子置空，失败留下空戳子、下次必重跑；戳子不存在或是旧版阶段名纯文本时才退回按产物 mtime 判。`register_episode` 用 `ruamel.yaml` round-trip 只改目标集的 srt/video，保留其他字段、注释、键序、引号与原有序列缩进；配置读取仍用 PyYAML。登记 SRT 不存在时先报错，不拷贝也不写配置。

  批量模式的编排是**按集纵向**（P0-C 定的：中途失败要留下完整交付物，而不是一堆半成品）。script 阶段的多集并发（`llm.script_concurrency`，默认 1）是在这个纵向循环上加一个**有界预取窗口** —— 走到第 i 集时确保前 `i + concurrency` 集的 script task 都起了，然后 await 第 i 集那个。**刻意不把 script 抽成横向并发阶段**：那会直接推翻上面那条不变量（全部集的 script 跑完之前一集成片都不会有，而 script 恰好是最慢也最容易失败的阶段）。默认 1 的依据见 `config.LLMConfig.script_concurrency` 的注释（并发易撞 429、失败时在飞调用白花钱、audio/render 的同步 ffmpeg 会堵住事件循环）。失败收摊走 `_drain_script_tasks`，`try/finally` 包住整个纵向循环。
- `src/tenmin/cli.py`：Typer CLI（`tenmin init` / `tenmin run` / `tenmin inspect` / `tenmin ocr` / `tenmin import`）。`tenmin ocr` 是脱离项目的批量画面 OCR（见下面 `ingest/ocr_batch.py`），不走管线。`tenmin import <slug> <dir>`（函数名 `import_`，`import` 是关键字）扫描目录批量登记，见下面 `importer.py`：默认打印预览后要确认，`--yes` 跳过确认，`--dry-run` 只预览（含 LLM 兜底与 ffprobe，零写入），非 TTY 且没 `--yes` 报错退出；确认后先构造管线 provider（缺 key 时一个文件都不动），再**只对本次导入的集**逐集跑 `run_pipeline` 单集模式。`tenmin run` 支持四种用法：
  - `--episode N --srt <path> --video <path>`：注册新集并跑。
  - `--episode N --video <path>`（不带 `--srt`）：**生肉入口**，对白轨靠软字幕轨抽取、（声明了硬字幕时）画面 OCR 或语音转写拿。反过来「只传 `--srt`」非法（视频是渲染阶段的硬需求）。
  - `--episode N`（不带 srt/video）：重跑已注册的某一集。
  - 不带任何 flag：批处理模式，跑 project.yaml 里注册的所有集。
- `src/tenmin/script/llm.py`：LLM provider 抽象。`LLMProvider`（Protocol，`complete` 有两条 PEP 695 重载：传 schema 返回该 schema 实例）、`GeminiProvider`（原生 google.genai SDK）、`OpenAICompatibleProvider`（通用 OpenAI 兼容 chat/completions 流式接口，schema 写进 prompt + pydantic 校验 + 报错重试，不依赖 `response_format=json_schema`）、`MiniMaxProvider(OpenAICompatibleProvider)`（MiniMax 专属子类，多了 `thinking` 深度思考开关，走 `_extra_payload_fields()` hook 注入）。`build_provider(cfg, settings)` 工厂函数按 `cfg.provider`（`"gemini"` / `"minimax"` / `"openai_compatible"`）分支构造对应 provider。

  健壮性分成三层，改这个文件前先分清自己在动哪一层：
  1. **传输层**（OpenAI 兼容 `_stream_with_retries`、Gemini `_generate_with_retries`）：OpenAI 兼容只重试 HTTP 429/500/502/503/504，Gemini 重试 HTTP 429 和全部 500–599；两者都重试连接类异常，按 `transport_max_attempts` 指数退避 + 抖动（HTTP 响应另尊重 `Retry-After`）；其余 4xx 与非限流业务错误立即失败。Gemini 将 SDK `APIError` 按状态码判并包装为 `LLMHTTPError`，连接异常耗尽包装为 `LLMTransportError`；它显式传入自持有的 `httpx.AsyncClient`，绕开 SDK aiohttp 路径在 `attempts=1` 内仍会偷偷重试连接的行为，保证外层次数就是 HTTP 请求上限（连接异常识别仍兼容 httpx/aiohttp 两族）。
  2. **schema 修复层**（`complete_with_schema_repair`，provider 无关，两个 provider 共用）：校验失败就把「schema + 报错 + 截断后的坏输出」回灌重试，次数由 `max_attempts` 管。**纠错轮刻意不重发首轮那份 ~35k 字符的正文。**
  3. **异常族**：全部继承 `LLMError`（`RuntimeError` 子类，已进 `cli.py` 的 `PIPELINE_ERRORS`）。`LLMHTTPError` 把响应体摘要拼进消息，`LLMBusinessError` 管 HTTP 200 + `base_resp.status_code != 0`，`LLMSchemaError.raw_output` 带着最后一次的原始模型输出（由 `pipeline.run_script` 落到 `03_script/E{NN}.raw.txt`），`LLMFinishReasonError` 管「没有可用输出」的 finish_reason —— **两条 provider 路径共用它**（Gemini 的 `_check_gemini_finish` 与 OpenAI 兼容的 `_stream_once` 里那次截断判定），而且它刻意不被 schema 修复层网住：同一个 max_tokens 只会再截断一次。
  退避的 `_sleep` / `_rand` 是模块级函数，测试 monkeypatch 掉它们，所以**新增退避路径时不要改成直接 `asyncio.sleep`**，否则测试会真睡。
  **用量**：`LLMUsage` 累计一次 `complete()` 内的全部请求（含 `cached_tokens`）。`script/usage.py` 的 `track_call` 在调用返回后立刻读 `provider.last_usage`（中间不能插 await，否则并发会串账），记录每次首稿/语义重试/预算返工或翻译/修复调用；写入 `03_script/E{NN}.usage.json`、`zh/E{NN}.usage.json`。失败调用也记录 `ok: false`；用量文件只供成本观察，不参与新鲜度。
- `src/tenmin/render/`：`subtitles.py`（ASS 字幕生成，含手动 CJK 换行，因为 libass 不会按 CJK 字符边界自动换行）、`timeline.py`（时间轴重算 + 按句拆分字幕 cue）、`audio.py`（原声 ducking + 混音 + 淡出 + 结尾静音）、`video.py`（剪辑拼接烧字幕 + 淡出 + 结尾卡片）、`ffmpeg.py`（subprocess 封装，文本路径用 `text=True, errors="replace"`，因为老番源文件的容器元数据经常不是合法 UTF-8）。二进制例外有三个：`probe_video_size`（首条视频流宽高，给 OCR 算精确像素裁剪与每帧字节数）、`run_raw_frames`（rawvideo 走 stdout 按固定字节数逐帧回调，stderr 在另一个线程里排空——`showinfo` 的 pts 行刷得很凶，不并发排空会死锁；末尾残帧与非零退出都报 `FFmpegError`，回调抛异常时 kill 子进程）和 `run_pcm`（有界音频片段转为 s16le 单声道，`capture_output=True` 一次捕获二进制 stdout 并返回 bytes；非零退出时解码 stderr，连命令及日志尾部包装为 `FFmpegError`）。
- `src/tenmin/ingest/ocr.py`：画面硬字幕 OCR（`recognize(video, dest, *, ocr, ffmpeg_path, ffprobe_path)`，同步；识别本身在 `recognize_cues(...) -> list[RawCue]`，不落盘，`recognize` 只是它 + `atomic.write_text` + 完成提示，`tenmin ocr` 直接调 `recognize_cues`）。管线：`sample_step`（按源帧率换算「每几帧抽一帧」，`sample_fps` 是每秒采样数而不是步长）→ `frame_filter`（`select` + 精确像素 `crop` + 缩放到 `OCR_WIDTH=1280` 宽 + 灰度 + `showinfo`）→ `ffmpeg.run_raw_frames` → 每帧 `_recognize`（Apple Vision，pyobjc **只在这个函数体内 import**，ImportError → `OCRUnavailableError`；测试 monkeypatch 它）→ `frame_text`（只留居中行、按需筛下层字幕、要求含 CJK、同行按 x 排序、从上到下拼行、省略号归一，行尾单个 `.`/`。` 也算 `…`）→ `merge_frames`（相似度归组、允许 `_MAX_GAP_FRAMES=1` 个空帧、少于 `min_frames` 丢弃、众数投票同票取最早、末尾 = 最后一帧 + 采样间隔、重叠夹紧）→ `atomic.write_text`。**时间戳取 showinfo 的真实 pts，不按帧序号推算**（VFR 片源会算错），pts 行数 ≠ 帧数抛 `OCRError`；0 条 cue 抛 `OCRError` 且不写文件（否则空 SRT 会被当成可复用缓存）。开工前打预计耗时（时长 / 7），每 10% 打一行进度。`OCRError` 已进 `cli.PIPELINE_ERRORS`。
- FX 中日双语硬字幕的 OCR 混入日文及重复 cue，复现证据、已实施处理和画面文字的剩余边界见 `docs/fx-bilingual-ocr-followup.md`；处理这类上下分层字幕时先读该记录。
- `src/tenmin/ingest/ocr_correction.py`：可选文字校对（项目 `ocr.ai_correct` / 独立 `--ai-correct`，默认关）。仅 OCR 来源进入；每批 60 条加相邻两条只读上下文，模型只返回 id/text/uncertain/reason，程序保留时间与顺序；不确定行保留原文并写 review。成功稿用于 `build_track` 或独立交付，缺 key/调用/对齐失败警告后用原稿，下次普通运行重试。状态用原稿、视频身份、模型及机器输出摘要识别缓存和人工编辑；来源未变优先人工稿，来源变化保留旧人工稿并使用新原稿。同步 `run_ingest` / `run_batch` 保持默认路径，启用时走异步 corrected 入口；ingest 校对待重试时不盖成功戳。
- `src/tenmin/ingest/ocr_batch.py`：`tenmin ocr` 的实现（cli 那边只是薄包装）。`collect_videos`（目录只看一层、按 `VIDEO_SUFFIXES` 不分大小写挑、按名排序；显式文件不看扩展名；按 resolve 后路径去重；路径不存在抛 `FileNotFoundError`，一个都没有抛 `ValueError`）→ `output_path`（`<stem>.zh-Hans.srt` / `<stem>.zh-Hant.srt`，`-o` 目录或视频旁边）→ `run_batch`：复用判据直接用 `resolve._is_usable_cache`（非空且不比视频旧就跳过，`--force` 重跑），识别调 `ocr.recognize_cues`（测试 monkeypatch 它），默认用 `clean.to_simplified` 逐条繁转简，`atomic.write_text` 落盘。单个视频的 `OCRError` / `FFmpegError` / `FileNotFoundError` / `ValueError` 只记进 `BatchSummary.failed`、接着跑；`OCRUnavailableError`（`OCRError` 子类）先单独放行、整批中止。有失败时 CLI 退出码 1。刻意不读 project.yaml、不剔 OP/ED staff 字、只开放 `--crop-top` / `--crop-bottom` 两个旋钮。
- `src/tenmin/importer.py`：`tenmin import` 的实现，不启动流水线。`scan_directory`（单层、视频后缀复用 `ocr_batch.VIDEO_SUFFIXES`；SRT 按主体完全相等或「主体 + `.`/`_` + 语言后缀」配对，多候选按 `LANGUAGE_SUFFIXES` 顺序取一个、其余列为忽略，不做模糊匹配；`--srt-dir` 换字幕目录）→ `reference_hashes` 验图 → `is_extra`（NCOP/NCED/SP/OVA/PV/预告/Menu 跳过）→ `parse_episode_number`（NFKC 并剔掉方括号 CRC32 后，强模式 `E01`/`EP01`/`S01E01`/`第01集`/`第01話`（可带 `v2` 版本后缀），全不中才走弱模式：剔噪声 token 后被分隔符包住的 1–3 位数字；多个候选返回 `Ambiguous`）→ `resolve_ambiguous`（剩下的文件名**打包一次** LLM 调用；provider 只在有待判文件时才由 factory 构造，构造/调用失败一律按歧义跳过并 warning）→ 重复集号全部跳过 → 已登记且有来源的集跳过 → `match_credits` → `infer_ranges`（片长走 `pipeline._source_duration`，字幕区间走 `build_track`，与 ingest 同口径；字幕读取失败的集单列跳过）。

  **参考截图按侧覆盖**：`credits.op_reference` / `ed_reference`（`Path | None`，默认 `None`）仅用于 import；相对路径以 `cfg.root`（project.yaml 父目录）为基准，绝对路径照用。`build_import_plan` 将已配置侧传给 `scan_directory(reference_overrides=...)`，该侧全部目录候选（包括重复或坏图）忽略；未配置侧继续视频目录顶层发现：主体恰好为 `op` / `ed`，扩展名 `.png` / `.jpg` / `.jpeg`，均不分大小写，不查 `--srt-dir`。独立 `scan_directory` 保持原来的同侧多张 `ValueError`。所选截图必须在 LLM 兜底与任何导入写入前验证，显式路径缺失或读不出直接报错。两字段放进 `config_slices.EXCLUDED`，默认值也不进入 ingest payload，保证升级及改路径时存量切片逐字节不变；`tenmin run` 的 AV 行为保持现状。截图应选起点附近有辨识度的画面，避开黑场或通用 logo：同距取最早帧，会把它们误配到片头别处。

  **按侧选择来源**：有截图的侧用 `AvSide`（画面+音频），没有截图的侧才用字幕推断；截图匹配失败仍保留失败结果，**不回落字幕**。`match_credits` 跳过已有任一逐集手填区间的集；片长未知时跳过画面搜索、提供失败原因。OP 搜索 `[0, min(300, duration))`，ED 搜索 `[max(0, duration - 180), duration)`，接受画面距离 ≤12/64。搜索窗常量住在 importer，其余匹配常量住在 av_match，当前均为模块级常量，未新增 config 旋钮。

  **OP 长度**：多集截图匹配成功时，选距离最小者作音频参考（同距取集号小的），各集从各自画面起点读 150 秒包络，只读一次；参考与其他集逐一测长，剔除距原始中位数 >10 秒的值后再取中位数，所有可用集共用这个长度、终点夹到片长。只有一集时直接按 90 秒估计、不读音频；多集没有任何可测长度时也按 90 秒估计，两者都标记 `estimated`；已有测量却被全部剔除时判失败。参考集音频解码失败使全部已匹配 OP 失败；对比集音频失败只影响该集。**空 PCM / 不足一帧的包络也算音频失败**，不能混进「无法测长 → 90 秒估计」；缺 ffmpeg 的 `FFmpegBinaryError` 原样外抛。

  **ED 终点**：`_ed_sides` 独立选同规则参考集，从截图起点读到各自片尾（搜索保证 ≤180 秒），各集只读一次。`av_match.ed_endpoints` 沿用 OP 包络、±1 秒错位、一秒相关窗与两秒持续掉落，但先要求五秒连续匹配。掉落返回两条各自时间轴的歌曲终点；持续匹配耗尽只确认覆盖到的那一侧文件 EOF（容许不足一帧的量化尾差）。读段截断、较短集 EOF、无匹配或 EOF 前不足两秒的掉落都不能替较长集确认终点。参考集端点证据取稳健中位数，对比集各用自己的证据；没有证据或单集时仍到片尾并标 `estimated`，单集不读音频。解码/空 PCM/缺 binary 的策略与 OP 按侧相同。`RangePreview.op_length_estimated` 与 `ed_length_estimated` 分别传递状态；预览实测与估计分侧显示，ED 估计明确提示可能包含 C 段/预告。

  **OP/ED 必须一起写**：这两个字段直接驱动 `in_credit_window`，只写一半会把另一半的窗关掉，所以只在这一集两个字段都空、片长已知、项目级默认没生效、OP 与 ED 都识别成功且取整到 0.1 秒后仍合法时才写，否则两个都不写。已有手填区间原样保留，项目默认生效时保留其后续调整能力；补齐来源的预填集写区间会改 EPISODE 切片，预览里提示会重跑。预览按侧标「画面+音频」或「字幕推断」及可用的 `距离 N/64`，另列测量/估计、失败原因与是否写入；搜索时逐集打印 `正在匹配画面 i/N`。
- `src/tenmin/ingest/av_match.py`：截图与音频匹配，分纯计算和媒体 IO 两层。纯函数 `dhash` 将 9×8 灰度图左右比较编码成 64 位哈希，`hamming` 数不同位，`best_frame` 返回最小距离及索引、同距取首帧；最优帧本身不应用接受阈值，由 importer 判定。`audio_envelope` 将 8kHz s16le 单声道 PCM 转成 10 帧/秒、16 个对数频带能量（Hann 窗 + FFT，丢弃末尾单字节与不足一帧的采样）；`op_length` 先去各频带时间均值，在 ±1 秒找最佳错位，再用 1 秒窗算 Pearson 相关：先见过 ≥0.5 的匹配，之后持续 2 秒低于阈值才判结束，终点取首个低相关窗中心；没对齐或没持续掉落返回 `None`。`median_length` 实现上述异常值过滤与中位数。

  IO 层：`sample_hashes` 用 `fps` + 9×8 area 缩放 + 灰度 + `showinfo`，经 `run_raw_frames` 读帧；时间用 **start + 相对 PTS**，先验证完整帧数与时间戳数相等，再过滤到 `[start, end)`（fps 网格舍入可能补出窗外帧，提前过滤会掩盖错位）。`image_hash` 使用同一缩放/灰度管线且要求恰好一帧；`read_pcm` 调 `run_pcm` 固定 8kHz。`locate` 先 2fps 粗搜，再在最佳帧 ±1 秒内 10fps 细搜、夹到原搜索窗，细搜结果与粗搜最佳帧一起按时间排序后取最佳、同距取最早。
- `src/tenmin/render/tts.py`：TTS 层，结构上刻意跟 `script/llm.py` 对齐。改它之前先分清自己在动哪一层：
  1. **缓存身份**：chunk 文件名是 `chunk_{序号:03d}.{hash8}.mp3`，哈希 = sha256(`engine.fingerprint` + `\x00` + text)，`fingerprint` 含 voice 与 rate。**序号只为人工试听时可读，身份全靠哈希** —— 复用先按确切名字找，找不到就在同目录里按哈希 glob（chunk 数量一变序号全平移，但内容没变的不该重合成）。改这里会让 `work/` 下的存量 chunk 全部失效。
  2. **原子落盘 + 时长体检**（`EdgeTTSEngine.synthesize`）：`edge_tts.Communicate.save()` 是流式写，中断留截断 mp3。所以一律走 `tenmin.atomic` 的 `atomic_path`（全仓一套 `.part` 命名，`.part` 插在扩展名**之前**：`chunk_001.abc.part.mp3`）、`probe_duration` 体检通过才 `os.replace`。那个命名不会被 `_find_cached_chunk` 的 `chunk_*.{digest}.mp3` glob 命中（实测）。体检区间见 `_duration_bounds` 的 docstring（标定自 115 个真实 chunk）。
  3. **退避重试**（`synthesize_with_retry`）：模块级 `_sleep` / `_rand` 供测试 monkeypatch，参数与命名跟 llm.py 一套。`TypeError` / `ValueError` 判为不可重试（edge-tts 的参数校验）。**新增退避路径不要改成裸 `asyncio.sleep`**，否则测试会真睡。
  4. **输入健壮性**（`_plan_pronounceable`）：不含任何字母/数字的 chunk（切句留下的孤立 `'`）直接跳过，它带的 hold 折进前一个 chunk。
  5. `probe_duration` 是阻塞 subprocess，一律走 `asyncio.to_thread`。
  6. **并发合成**（`synthesize_track`，`render.tts_concurrency` 默认 4）：**固定 N 个 worker 抢一个共享游标**，刻意不是「每个 chunk 一个 task + Semaphore 限流」—— 后者在 concurrency=1 下也会把 N 个 task 全建出来，首个 chunk 失败时它们已经排在同一轮事件循环里、照样各发一次请求。四条不变量都有测试锁住：结果顺序 = 计划顺序（写预分配槽位，绝不 append，因为 `VoiceTrack.chunks` 的顺序决定 audio.py 的 adelay 偏移与字幕顺序）；同文本只合成一次（`digest_locks` 按内容哈希串起来，顺带关掉 `known_durations` 的竞态）；失败之后不再发新活；concurrency=1 与串行逐字节等价。TaskGroup 的 ExceptionGroup 必须解包成叶子异常（`_first_leaf`），否则 `cli.PIPELINE_ERRORS` 认不出 `TTSError`。默认值 4 的标定数据见 `config.RenderConfig.tts_concurrency` 的注释。

- `src/tenmin/script/single.py`：单集 LLM 调用编排（`generate_script()`），拼 prompt（模板 + few-shot 示例 + schema + 对白/信号数据）。轮次结构：首稿 → 最多 `llm.validation_retries` 次语义校验重试 → 最多 `llm.budget_rewrite_rounds` 轮时长返工。**返工轮跟首轮的 prompt 不一样**：摘掉 few-shot 范例（模型已经证明它会这个格式，而范例自己带着「不要学它的内容」的警告），但**必须**重发对白轨与高能点清单（占整份 prompt 的 84%，而重试要修的语义错只能对着对白原文才判得出来），另外把上一版的 `LLMScript` JSON 交回去让它做局部编辑 —— 不交回去的话 `budget.rewrite_instruction` 里那句「不要改动 clip 时间戳」是模型物理上做不到的要求。
- `src/tenmin/script/prompts/single_episode.md` 的**段落顺序是按 prefix 缓存实测标定的，别凭直觉重排**。顺序：完全静态的「素材格式说明 / 交付要求 / 输出格式」→ 本集素材（本期素材 / 术语表 / 高能点清单 / 对白轨）→ few-shot 范例 → 输出前自检。关键点是**范例必须留在全部素材之后**：返工轮除了摘掉范例什么都没动，所以范例排最后时返工轮的 prompt 就是首轮的一个**严格前缀**（实测 saijo 31.7k 字符可缓存）。「把静态段连范例一起前置」这个看起来更对的做法实测是 4 倍回退（一次 10 集批处理的可缓存字符占比 42.9% → 9.9%）：真正被反复发的前缀是「同集首轮↔返工轮共享的整份素材」，不是「跨集共享的那 2.5k 静态段」，而后者在默认模型（`gemini-3.6-flash`，implicit caching 门槛 4096 token）上大概率还够不到门槛。摘除范例的机制是「整节都注进 `{{example_block}}`」，不是字符串切割 —— 空值必须**逐字节**不留痕，多两个空行就足以让分叉点之后的缓存全部失效。
- `src/tenmin/script/prompt.py`：`{{name}}` 占位符渲染，**双向校验**（模板要的没传 → 报错；传了模板没用到 → 也报错，因为那是占位符名字敲错，会让整段内容静默丢失）。两个方向共用 `PromptTemplateError(KeyError, ValueError)`：KeyError 保历史语义，ValueError 让它落进 `cli.PIPELINE_ERRORS`。模板名是 `render_prompt` 的**位置参数**（占位符可能恰好叫 `template_name`）。`load_prompt` 带 `functools.cache`，改模板内容的测试用 `cache_clear()`。
- `src/tenmin/script/validate.py`：LLM 输出的唯一拦网。**分成两半，改之前先分清自己在动哪一半**：`check_script()` 是纯读（只返回 warning，一个字节都不改），`repair_script()` 是显式修复（先 `model_copy(deep=True)` 再改，返回**新** Script）。`validate_script()` 是两者的组合。三个入口都收 `rate`（= `render.rate`）：两条按秒数判的检查（hold/sfx 落点上界、画面/旁白拉伸倍率）必须跟 voice 阶段同口径，不传就会在非默认语速下分叉。拆开的动因是 single.py 的返工轮要「两版择优」，而原来 validate 就地改写并把同一个对象塞回结果，上一版根本没被保留下来。降级语义（丢弃单条 clip、保留其余、附一条 warning）是生产上的重要健壮性，别改成整篇作废。
- `src/tenmin/script/budget.py`：时长预算，全是纯函数。`SPEECH_RATE_CPS = 4.5` 是全项目唯一一份（render/tts.py 的时长体检、render/chunks.py 的句偏移、docgen/table.py 的估算列都从这里取），`speed_factor(rate)` 也住在这里、tts.py 反过来 import 它。字数换算成秒数的唯一入口是 `narration_seconds(text, rate=...)`，**反向**（秒→字）的唯一入口是 `narration_chars_for_seconds(seconds, rate=...)`；「旁白 + 留白」的跨度只有一份实现 `narration_span_seconds`（`beat_seconds` 是它的 Beat 包装，`render/chunks.py` 的 `assign_holds` 直接调散件版）；读全片估算的唯一入口是 `total_estimate()`（优先读存好的 `est_total_seconds`，缺了才重算）。`narration_chars` 对中文标点**全额计费是刻意的**，用 115 个真实 chunk 测过：打折只会让「实测/估算」的分布更散（docstring 里有完整数据）。
- `src/tenmin/translate/`：翻译阶段，4 个模块 + `prompts/`（`translate_lines.md` / `system.md`，走 `importlib.resources` 定位，渲染复用 `script.prompt.render_prompt` 但**不**复用 `load_prompt`——后者的 PROMPTS_DIR 硬绑在 `tenmin.script` 上）。
  - `lines.py`：一集对白的翻译编排（`translate_track()`）。**刻意不分批**（实测 396 条 dialogue、5~6k 字符一次调用塞得下，分批换来的是术语跨批不一致）。真正的风险是**对齐**：`select_translatable` 只挑 `SPEECH_KINDS`（dialogue + monologue，credits/screen_text/noise 不翻），返回的 id 是 `DialogueTrack.lines` 里的 **1-based 位置**而不是 `DialogueLine.idx`（后者在双轨字幕被 `split_dual_track` 拆开时会重复），`check_alignment` 按**集合**判进出相等（按条数判查不出「译了两遍 + 漏一条」）并抛 `LLMResponseFormatError`——那是硬契约，`complete_with_schema_repair` 只网这一族，抛别的就一次重试都没有。纠错轮**照旧重发整份正文**（跟 llm.py 那条「修复轮不重发正文」刻意相反：这里要补的是「第 137 条漏了」，模型得对着原文）。返回前用对白轨的集号盖掉模型填的 `episode`。一条 speech 行都没有时早退、不发请求。OCR 来源走同模块的 `passthrough_track`：不调 LLM，原文（ingest 已繁转简）直接当译文，选行与 id 口径同 `select_translatable`，glossary 留空。
  - `glossary.py`：跨集累积的专有名词表。**两个合并函数的优先级方向刻意相反**，别记混：`merge_glossary(accumulated, fresh)` 是**回写**用的，冲突时**累积的赢**（不许第 5 集把已定译名改掉）；`effective_glossary(accumulated, manual)` 是**喂模型**用的，冲突时 `project.yaml` 手写的那份赢（它是纠错入口）。全模块只有一个「条目算不算数」的判据 `_clean`（键值都得是 strip 后非空的 **str**，非字符串的值刻意丢掉而不是 `str()` 强转），读盘/合并/喂模型三个入口都过它、**save 不过**（它收的已经是 merge 洗过的）。`save_glossary` 在 payload 与盘上逐字节相同时**不碰文件**——这不是性能优化而是正确性：这张表是 script 的新鲜度输入（`pipeline._script_inputs`），无条件写的话 E10 的 translate 会把前 9 集刚写好的解说稿全部判旧。`load_glossary` 任何异常都返回空表（含 `UnicodeDecodeError`，它是 `ValueError` 子类、不被 `OSError`/`JSONDecodeError` 网住），两处 `is_file()` 守卫是为 FIFO 那种 `read_text` 会永久阻塞的路径留的（except 兜不住阻塞）。
  - `srt_writer.py`：中文字幕交付物（`render_zh_srt`）。分工是「翻译不碰时间，时间戳回对白轨取」。id 越界（`< 1` 或 `> len(lines)`）一律报错、与「译文空白就静默跳过」刻意不对称——多出来的 id 必然指到一行不是这句的时间，而 0/负数会被 Python 负下标静默配到末尾某行。按 `(start, end, id)` 三键排序，第三键是给 `split_dual_track` 拆出的同起止各段兜底（否则先后由模型回填顺序决定，同一素材换次调用就换个字节）。时间戳分隔符用逗号，靠 `timecode.format_timestamp(...).replace(".", ",")` 而不是重抄实现（重抄会漏掉 nan/inf 守卫与负数夹紧）。


## 测试

```bash
uv run pytest tests/ -q          # 全量跑，默认跳过需要真实 API key / 素材的标记测试
```

`tests/` 目录：`test_config.py`、`test_llm.py`、`test_pipeline.py`、`test_cli.py`、`test_config_slices.py`、`test_render_*.py` 等，共 54 个 Python 文件（按 `tests/**/*.py` glob，含 conftest.py / fakes.py / __init__.py）。pytest markers（`pyproject.toml` 里是**六个**）：

- `llm`：需要真实 LLM API key（默认跳过），跑法：`TENMIN_GEMINI_API_KEY=xxx uv run pytest -m llm`。
- `generalize`：需要额外的番剧 SRT fixture。
- `render`：需要真实视频 + 装了 libass 的 ffmpeg。
- `asr`：需要 `uv sync --extra asr`（会拽 torch，几个 G）+ 真实视频。要加载几个 G 的语音模型跑上几分钟。
- `ocr`：需要 macOS + `uv sync --extra ocr` + 一个真实的硬字幕片源，跑法：`TENMIN_OCR_SAMPLE_VIDEO=<片源路径> uv run pytest -m ocr`。整集识别要跑三四分钟。
- `av`：默认跳过，由 `tests/conftest.py` 的显式 marker 门和素材环境变量共同控制；跑法：`TENMIN_AV_SAMPLE_DIR=<素材目录> uv run pytest -m av tests/test_av_match.py`。目录须至少两集集号可解析且唯一的正片视频，每集恰好一份配对 SRT，并提供 OP/ED 截图；未配对素材、额外候选字幕、非正片均不接受。真实测试要求完整覆盖所有素材集、两侧画面距离 ≤12/64、OP 与 ED 均有跨集音频实测证据、OP 长度在 60–120 秒内、ED 终点合法且两侧可成对写入；验收素材 saijo E02/E12 另要求 ED 实测到 EOF。`uv run pytest -m render tests/test_av_match.py` 则用 lavfi 合成媒体验证抽帧/截图/PCM IO。

改动 provider 相关代码后，务必确认 MiniMax 的现有测试（`test_minimax_*`）**行为不变**——`OpenAICompatibleProvider` 的重构原则是零行为变更，只是代码结构拆分。

## 开发流程约定（本仓库沿用的模式）

新功能走 brainstorming → spec 文档（`docs/superpowers/specs/`）→ 实现计划（`docs/superpowers/plans/`）→ Subagent-Driven Development（独立 git worktree + 分任务派发实现/审查子代理）→ 全分支代码审查 → 合并回 `main`。历史 spec/plan 文档是"某个时间点的设计快照"，不代表当前代码状态——不要反向去改它们对齐现在的代码。

## 已知的、故意不修的问题

- 源视频自带的原生硬字幕（繁体中文，来自原始 ANIPLUS 流媒体版本）在画面底部，某些时间点会跟 tenmin 自己烧的字幕重叠。尝试过两种修复（不透明底框、全宽黑条），都被用户否决了；目前接受这个瑕疵，不再处理。
- `work/<slug>/project.yaml`（单个项目的 `render:` 配置）目前是整个 project 共享的，不支持按集覆盖。
- **位图字幕轨（Blu-ray PGS / DVD VobSub）走不通**：`has_subtitle_stream` 只判「有没有字幕轨」、判不了「是不是文本」，所以位图轨必然走进抽取路并在 ffmpeg 的跨族转码守卫上失败。`ingest/resolve.py` 认出那句报错并换成中文提示，但**刻意不自动回落到语音转写**（位图轨能 OCR，悄悄换成听写是把更好的素材丢了）。声明了硬字幕的集也一样：软字幕轨那一岔排在 OCR 前面，位图轨照样在抽取时报错、**不自动改走画面 OCR**，只是报错文案换成指向 `--srt` 或「去掉字幕轨走画面 OCR」。另外画面 OCR 只认画面底部居中的字幕，变形宽高比（SAR≠1）片源没处理。
