# OCR AI 校对实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本仓库上级指令禁止未经请求派发子代理，因此由当前会话内联执行。

**Goal:** 可选 AI 校对 OCR 字幕，成功时作为下游输入，失败时警告并用原稿继续、下次自动重试。

**Architecture:** 共享的 `ingest/ocr_correction.py` 管理逐条校对、对齐和来源摘要；独立批量命令和管线分别接入，复用原稿且保护人工编辑。管线全局 ingest 的戳必须在校对未完成时允许重试，现有 `source=ocr` 的 translate 仍直通。

**Tech Stack:** Python 3.14, Pydantic, Typer, tenmin 现有 LLMProvider 与 atomic 写盘、pytest。

**Spec:** `docs/superpowers/specs/2026-10-06-ocr-ai-correction-design.md`

## 执行记录

- 共享校对协议、缓存与人工稿保护已实现，分批失败整份回退。
- 独立 CLI 与管线入口已接入，集成测试覆盖降级后普通运行自动重试及中文字幕更新。
- 接入采用原稿路径的同集后缀，不新增 `Paths` 表项；现有 ingest 切片已经包含整个 `ocr`，无需修改 `config_slices.py`。
- 历史 FX 手校字幕未用于强制重跑；验收使用临时素材与可控 provider。

## Global Constraints

- 默认关闭；只有 OCR 来源走校对，不能将 OCR 失败改成 ASR 回退。
- 始终保留原稿和人工校对稿；不确定条目原样保留；不得让模型重排 cue/时间戳。
- 调用/对齐失败仅警告并回退 OCR 原稿；之后自动重试；原稿内容不变不得触发不必要的下游重跑。
- 本仓库测试与 pipeline 命令使用裸 `/Users/portz/.local/bin/uv run ...`，不包 rtk；既有未提交 OCR 工作与人工校过的 `work/fx/srt/fx-01.zh-Hans.srt` 不得覆盖。
- 不提交、推送，除非用户另行请求。

---

### Task 1: 校对协议与持久化状态

**Files:** Create `src/tenmin/ingest/ocr_correction.py`, `src/tenmin/ingest/prompts/ocr_correction.md`, `tests/test_ocr_correction.py`.

**Interfaces:** `async def correct_cues(cues: list[RawCue], provider: LLMProvider, *, cfg: LLMConfig) -> tuple[list[RawCue], list[ReviewItem]]`；`async def select_corrected(raw: Path, corrected: Path, state: Path, review: Path, *, provider_factory: Callable[[], LLMProvider], cfg: LLMConfig, source_identity: str) -> tuple[Path, list[str], bool]`（第三个值表示本次尚待 AI 重试）。模型只返回 `{id, text, uncertain, reason}`，原 cue 元数据由程序复制。`state` 存 raw sha256、source_identity、LLM 身份、机器写出的 corrected sha256 及成功状态；与盘上摘要不符即人工稿。

- [ ] 写失败测试：乱序/重复/漏 ID、空文本整份失败；单条 uncertain 保留原字而其他改字；provider 抛错回退；成功缓存与人工稿优先、原稿变化后保留旧稿并改用原稿；失败没有成功状态可供下次重试。
- [ ] 运行 `/Users/portz/.local/bin/uv run pytest tests/test_ocr_correction.py -q`，确认各失败原因是功能缺失。
- [ ] 实现逐条校对、持久化与原子写盘；长文本分批，每批保留相邻上下文，缺一批整份回退；失败时返回带原因的 warning，不吞掉 KeyboardInterrupt 或其它非预期程序错误。
- [ ] 运行目标测试直至全绿。

### Task 2: 独立 OCR 批量命令

**Files:** Modify `src/tenmin/ingest/ocr_batch.py`, `src/tenmin/cli.py`; test `tests/test_ocr_batch.py`, `tests/test_cli.py`.

**Interfaces:** 保留同步 `run_batch(...)` 的默认路径；增加 `async def run_batch_corrected(..., provider_factory: Callable[[], LLMProvider], llm_config: LLMConfig) -> BatchSummary`；`BatchSummary.warnings: list[str]`。CLI `--ai-correct`, `--llm-provider`, `--llm-model`；仅启用校对时用 `asyncio.run(run_batch_corrected(...))`，provider 按需创建并关闭。原始 SRT `<stem>.zh-Hans.raw.srt` 与最终 SRT 分开。

- [ ] 写失败测试：默认现有跳过语义、开启校对后原稿与最终输出分开、仅最终文件存在的存量文件先保留、API 缺 key/失败只 warning 且其它视频继续、第二次自动重试并不重 OCR、`--force` 保护人工字幕、两种语言互不覆盖。
- [ ] 运行目标测试看预期失败。
- [ ] 接入共享组件；provider key 从现有 `Settings` 读取，独立 CLI 使用显式 LLM 配置或默认 LLMConfig，输出 warning；先安全备份来源已变化的人工终稿再写新原稿。
- [ ] 运行 `/Users/portz/.local/bin/uv run pytest tests/test_ocr_batch.py tests/test_cli.py -q`。

### Task 3: 管线 OCR 接入和可重试 ingest

**Files:** Modify `src/tenmin/config.py`, `src/tenmin/config_slices.py`, `src/tenmin/pipeline.py`, `src/tenmin/cli.py`; test `tests/test_config.py`, `tests/test_config_slices.py`, `tests/test_pipeline.py`, `tests/test_cli.py`.

**Interfaces:** `OcrConfig.ai_correct: bool = False`；保留同步 `run_ingest(cfg) -> list[DialogueTrack]`，新增 `async def run_ingest_corrected(cfg, *, provider_factory: Callable[[], LLMProvider]) -> tuple[list[DialogueTrack], list[str], bool]`，只在启用校对时由异步 `run_pipeline` await（同步入口保持旧行为）。`run_pipeline` 可选传入 factory；CLI 只有 ingest 阶段需要 AI 时传工厂而不提前建连接。`Paths` 新按集属性另开校对稿、状态、review（检查路径布局冻结测试，再更新它的锁定表）。

- [ ] 写失败测试：只有 OCR 来源发 AI 请求；`--only ingest` 启用但缺 key warning 后继续；失败降级在第二次普通运行自动重试；成功后 `dialogue.json` 变化让依赖阶段更新；人工稿未变源可复用、源变化后回退新原稿；若同时运行 script，script 缺 key 仍须按原错误失败。
- [ ] 运行目标测试看预期失败。
- [ ] 在 resolve 返回 OCR 原稿之后、`build_track` 之前选择校对稿；全局 ingest 戳或新鲜度额外感知“校对未完成”，成功缓存不重新发请求；`source` 保持 `ocr`。对 `ocr.ai_correct` 新字段变更配置切片时避免升级项目其它阶段无故重跑。
- [ ] 运行 `/Users/portz/.local/bin/uv run pytest tests/test_pipeline.py tests/test_cli.py tests/test_config.py tests/test_config_slices.py -q`。

### Task 4: 集成验收和文档

**Files:** Modify `AGENTS.md`, `docs/fx-bilingual-ocr-followup.md`, `README.md`（仅与功能有关的入口说明）。

- [ ] 在临时项目 fixture 里跑 OCR 成功/不确定/失败降级/下次重试/人工编辑保留的集成测试，不对 FX 手校成果执行 `--force`。
- [ ] 检查新增路径的原稿/终稿稳定命名、状态原子性，失败不伪装为成功、warning 可读。
- [ ] 跑 `/Users/portz/.local/bin/uv run pytest tests/ -q`、`/Users/portz/.local/bin/uv run ruff check src/tenmin tests`、`git diff --check`，审查全部差异并准确汇报未解决的文字错字边界。
