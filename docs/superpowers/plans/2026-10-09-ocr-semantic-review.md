# OCR Semantic Review Implementation Plan

> 使用 executing-plans 在当前会话执行；未经用户要求不提交、不派发子代理。

**Goal:** 已启用 AI 校对时自动进行第二轮语义复核。
**Architecture:** 保留 correct_cues 首轮接口；新增 semantic_review 共用批次与编号校验，由 select_corrected 编排并记录降级状态。
**Tech Stack:** Python、Pydantic、pytest、现有 LLMProvider。
**Spec:** `docs/superpowers/specs/2026-10-09-ocr-semantic-review-design.md`

## 约束
条数、顺序、时间不变；第二轮失败保留第一轮；默认关闭 AI；保护人工稿；普通运行自动重试。

## 执行步骤
- [x] 添加语义修复、不确定候选、第二轮失败与重试测试，先验证失败。
- [x] 新建 `ingest/ocr_semantic.py` 与 `prompts/ocr_semantic.md`，输入原稿和首轮结果，返回复核 cues 和 ReviewItem；使用既有 `_verify` 校验全部编号。
- [x] select_corrected 首轮成功后调用 semantic_review；第二轮失败保存首轮，status=semantic_pending，返回 retry=True。缓存摘要加入新提示词。
- [x] 更新首轮测试假 provider 兼容第二轮，验证两个入口不改变其外部契约。
- [x] 更新 README，运行全量 pytest、改动文件 Ruff；真实 FX 输入另存验证，不覆盖手工稿。

实测 MiniMax-M3 两轮共 12 批完成，357 条时间保持不变，修改 62 条，18 条待复核。
编号缺漏按 llm.max_attempts 有界重试；明确数字发生变化时保留首轮并记录候选。
