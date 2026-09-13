# AGENTS.md

给在这个仓库里工作的 AI agent 看的说明。

## 项目是做什么的

tenmin（10 分钟看番剧）：把番剧字幕（SRT）+ 视频，自动加工成"解说方案"文档 + 配音 + 烧字幕的成片。8 个阶段，每个阶段读上游产物、写自己的产物，靠文件 mtime 决定要不要重跑：

```
ingest → signals → script → docgen → voice → timeline → audio → render
```

- `ingest`：解析 SRT，生成对白轨（`01_dialogue/`）。
- `signals`：识别静音间隙、语速变化等"高能点"信号（`02_signals/`）。
- `script`：调用 LLM，把信号转成分幕解说稿（`03_script/`，唯一需要 LLM 的阶段）。
- `docgen`：把 script.json 渲染成人类可读的对照表 + 纯配音文本（`out/`）。
- `voice`：调用 TTS（edge-tts），把配音文本合成语音（`04_voice/`）。
- `timeline`：根据真实配音时长重新计算时间轴，生成字幕（`05_timeline/`）。
- `audio`：原声压低（ducking）+ 配音叠加混音（`06_audio/`）。
- `render`：一次性剪辑、拼接、烧字幕、合成音轨，出片（`07_render/`）。

一个 project（`work/<slug>/project.yaml`）对应**一部番**，可以登记多集。所有阶段产物都是按集号加前缀的（`Paths` 类，见 `src/tenmin/pipeline.py`），例如 `03_script/E02.script.json`、`out/E02.narration.txt`、`07_render/E02.mp4`。

## 关键工具规则（非常重要）

**跑本项目的 pipeline 命令和 pytest 时，一律用裸的 `uv run <cmd>`，绝对不要套 `rtk proxy` / `rtk pytest` / `rtk` 前缀。** 本项目的中文输出会把 rtk 的 UTF-8 抓取层搞崩。裸的 `rtk ls` / `rtk grep` / `rtk git` / `rtk read` / `rtk find` 是安全的，可以正常用。

```bash
# 对：
uv run pytest tests/ -q
uv run tenmin run saijo --episode 2 --only script --force

# 错（会崩，别用）：
rtk pytest tests/
rtk proxy uv run tenmin run saijo
```

如果需要重装 `.venv`（`uv sync --reinstall`），edge-tts 走公司 Zscaler MITM 代理会报 SSL 证书错——需要把 Zscaler CA 追加进 certifi 的 cacert.pem：

```bash
cat /path/to/zscaler_ca_bundle.pem >> .venv/lib/python3.14/site-packages/certifi/cacert.pem
```

## 代码结构

- `src/tenmin/config.py`：`ProjectConfig` 与它的 6 个子 config（`.ingest` / `.credits` / `.signals` / `.validate_script` / `.render` / `.llm`）、`EpisodeConfig`，`Settings`（`BaseSettings`，读 `.env`，`env_prefix="TENMIN_"`）。**全项目所有「经验阈值」的唯一权威来源**；分家的判据是「这部番想要什么」的创作旋钮进 config，「物理上不可能／数据坏了」的合法性边界留在各模块的模块级常量里。各阶段模块只保留 `DEFAULT_XXX.field` 的模块级别名。
- `src/tenmin/pipeline.py`：`Paths` 类（每阶段产物路径，全部按集号 `E{episode:02d}` 前缀），`STAGES` 列表，`run_pipeline()` 顶层编排（支持单集/批量两种模式，靠 `episode: int | None` 区分），`register_episode()`（`--episode --srt --video` 注册新集）。
- `src/tenmin/cli.py`：Typer CLI（`tenmin init` / `tenmin run` / `tenmin inspect`）。`tenmin run` 支持三种用法：
  - `--episode N --srt <path> --video <path>`：注册新集并跑。
  - `--episode N`（不带 srt/video）：重跑已注册的某一集。
  - 不带任何 flag：批处理模式，跑 project.yaml 里注册的所有集。
- `src/tenmin/script/llm.py`：LLM provider 抽象。`LLMProvider`（Protocol，`complete` 有两条 PEP 695 重载：传 schema 返回该 schema 实例）、`GeminiProvider`（原生 google.genai SDK）、`OpenAICompatibleProvider`（通用 OpenAI 兼容 chat/completions 流式接口，schema 写进 prompt + pydantic 校验 + 报错重试，不依赖 `response_format=json_schema`）、`MiniMaxProvider(OpenAICompatibleProvider)`（MiniMax 专属子类，多了 `thinking` 深度思考开关，走 `_extra_payload_fields()` hook 注入）。`build_provider(cfg, settings)` 工厂函数按 `cfg.provider`（`"gemini"` / `"minimax"` / `"openai_compatible"`）分支构造对应 provider。

  健壮性分成三层，改这个文件前先分清自己在动哪一层：
  1. **传输层**（`_stream_with_retries`）：429/5xx 与连接类异常走指数退避 + 抖动 + `Retry-After`，次数由 `transport_max_attempts` 管。其余 4xx 与非限流的业务错误码立即失败。
  2. **schema 修复层**（`_complete_with_schema_repair`，provider 无关，两个 provider 共用）：校验失败就把「schema + 报错 + 截断后的坏输出」回灌重试，次数由 `max_attempts` 管。**纠错轮刻意不重发首轮那份 ~35k 字符的正文。**
  3. **异常族**：全部继承 `LLMError`（`RuntimeError` 子类，已进 `cli.py` 的 `PIPELINE_ERRORS`）。`LLMHTTPError` 把响应体摘要拼进消息，`LLMBusinessError` 管 HTTP 200 + `base_resp.status_code != 0`，`LLMSchemaError.raw_output` 带着最后一次的原始模型输出（由 `pipeline.run_script` 落到 `03_script/E{NN}.raw.txt`）。
  退避的 `_sleep` / `_rand` 是模块级函数，测试 monkeypatch 掉它们，所以**新增退避路径时不要改成直接 `asyncio.sleep`**，否则测试会真睡。
- `src/tenmin/render/`：`subtitles.py`（ASS 字幕生成，含手动 CJK 换行，因为 libass 不会按 CJK 字符边界自动换行）、`timeline.py`（时间轴重算 + 按句拆分字幕 cue）、`audio.py`（原声 ducking + 混音 + 淡出 + 结尾静音）、`video.py`（剪辑拼接烧字幕 + 淡出 + 结尾卡片）、`ffmpeg.py`（subprocess 封装，所有调用都用 `text=True, errors="replace"`，因为老番源文件的容器元数据经常不是合法 UTF-8）。
- `src/tenmin/render/tts.py`：TTS 层，结构上刻意跟 `script/llm.py` 对齐。改它之前先分清自己在动哪一层：
  1. **缓存身份**：chunk 文件名是 `chunk_{序号:03d}.{hash8}.mp3`，哈希 = sha256(`engine.fingerprint` + `\x00` + text)，`fingerprint` 含 voice 与 rate。**序号只为人工试听时可读，身份全靠哈希** —— 复用先按确切名字找，找不到就在同目录里按哈希 glob（chunk 数量一变序号全平移，但内容没变的不该重合成）。改这里会让 `work/` 下的存量 chunk 全部失效。
  2. **原子落盘 + 时长体检**（`EdgeTTSEngine.synthesize`）：`edge_tts.Communicate.save()` 是流式写，中断留截断 mp3。所以一律先落 `.part`、`probe_duration` 体检通过才 `os.replace`。体检区间见 `_duration_bounds` 的 docstring（标定自 115 个真实 chunk）。
  3. **退避重试**（`synthesize_with_retry`）：模块级 `_sleep` / `_rand` 供测试 monkeypatch，参数与命名跟 llm.py 一套。`TypeError` / `ValueError` 判为不可重试（edge-tts 的参数校验）。**新增退避路径不要改成裸 `asyncio.sleep`**，否则测试会真睡。
  4. **输入健壮性**（`_plan_pronounceable`）：不含任何字母/数字的 chunk（切句留下的孤立 `'`）直接跳过，它带的 hold 折进前一个 chunk。
  5. `probe_duration` 是阻塞 subprocess，一律走 `asyncio.to_thread`（P2-B 的 TTS 并发要靠它）。

- `src/tenmin/script/single.py`：单集 LLM 调用编排（`generate_script()`），拼 prompt（模板 + few-shot 示例 + schema + 对白/信号数据）。轮次结构：首稿 → 最多 `llm.validation_retries` 次语义校验重试 → 最多 `llm.budget_rewrite_rounds` 轮时长返工。**返工轮跟首轮的 prompt 不一样**：摘掉 few-shot 范例（模型已经证明它会这个格式，而范例自己带着「不要学它的内容」的警告），但**必须**重发对白轨与高能点清单（占整份 prompt 的 84%，而重试要修的语义错只能对着对白原文才判得出来），另外把上一版的 `LLMScript` JSON 交回去让它做局部编辑 —— 不交回去的话 `budget.rewrite_instruction` 里那句「不要改动 clip 时间戳」是模型物理上做不到的要求。
- `src/tenmin/script/validate.py`：LLM 输出的唯一拦网。**分成两半，改之前先分清自己在动哪一半**：`check_script()` 是纯读（只返回 warning，一个字节都不改），`repair_script()` 是显式修复（先 `model_copy(deep=True)` 再改，返回**新** Script）。`validate_script()` 是两者的组合。拆开的动因是 single.py 的返工轮要「两版择优」，而原来 validate 就地改写并把同一个对象塞回结果，上一版根本没被保留下来。降级语义（丢弃单条 clip、保留其余、附一条 warning）是生产上的重要健壮性，别改成整篇作废。
- `src/tenmin/script/budget.py`：时长预算，全是纯函数。`SPEECH_RATE_CPS = 4.5` 是全项目唯一一份（render/tts.py 的时长体检、render/chunks.py 的句偏移、docgen/table.py 的估算列都从这里取），`speed_factor(rate)` 也住在这里、tts.py 反过来 import 它。字数换算成秒数的唯一入口是 `narration_seconds(text, rate=...)`；读全片估算的唯一入口是 `total_estimate()`（优先读存好的 `est_total_seconds`，缺了才重算）。`narration_chars` 对中文标点**全额计费是刻意的**，用 115 个真实 chunk 测过：打折只会让「实测/估算」的分布更散（docstring 里有完整数据）。

## 测试

```bash
uv run pytest tests/ -q          # 全量跑，默认跳过需要真实 API key / 素材的标记测试
```

`tests/` 目录：`test_config.py`、`test_llm.py`、`test_pipeline.py`、`test_cli.py`、`test_render_*.py` 等，共 28 个文件。pytest markers：

- `llm`：需要真实 LLM API key（默认跳过），跑法：`TENMIN_GEMINI_API_KEY=xxx uv run pytest -m llm`。
- `generalize`：需要额外的番剧 SRT fixture。
- `render`：需要真实视频 + 装了 libass 的 ffmpeg。

改动 provider 相关代码后，务必确认 MiniMax 的现有测试（`test_minimax_*`）**行为不变**——`OpenAICompatibleProvider` 的重构原则是零行为变更，只是代码结构拆分。

## 开发流程约定（本仓库沿用的模式）

新功能走 brainstorming → spec 文档（`docs/superpowers/specs/`）→ 实现计划（`docs/superpowers/plans/`）→ Subagent-Driven Development（独立 git worktree + 分任务派发实现/审查子代理）→ 全分支代码审查 → 合并回 `main`。历史 spec/plan 文档是"某个时间点的设计快照"，不代表当前代码状态——不要反向去改它们对齐现在的代码。

## 已知的、故意不修的问题

- 源视频自带的原生硬字幕（繁体中文，来自原始 ANIPLUS 流媒体版本）在画面底部，某些时间点会跟 tenmin 自己烧的字幕重叠。尝试过两种修复（不透明底框、全宽黑条），都被用户否决了；目前接受这个瑕疵，不再处理。
- `work/<slug>/project.yaml`（单个项目的 `render:` 配置）目前是整个 project 共享的，不支持按集覆盖。
