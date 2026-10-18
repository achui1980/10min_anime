# Vision + RapidOCR 硬字幕识别

## 已确认的目标

将已有 `docs/poc/rapid_accel.py` 的复识别方案接入项目管线及独立 `tenmin ocr`，提供 `ocr.engine: vision_rapid`。默认 `vision` 保持现有行为。用户已批准该方案。

## 配置与入口

`OcrConfig.engine` 为 `Literal["vision", "vision_rapid"]`，默认 `vision`。项目配置经现有 ingest 配置切片传递；独立命令增加 `--engine vision|vision_rapid`，默认 `vision`。新增 `rapidocr` extra，使用新版 `rapidocr>=3.7,<4` 与 `onnxruntime>=1.20,<2`，实际 lock/sync 验证现有 Python 环境兼容性，不缩窄项目 Python 范围。

安装与运行：

```bash
uv run --extra ocr --extra rapidocr tenmin ocr "视频.mp4" --engine vision_rapid
```

两种模式均依赖 Apple Vision，因此组合模式仍只支持 macOS。RapidOCR 和 ONNX Runtime 惰性导入；默认 Vision 路径不加载它们。首次使用可能下载模型，后续复用库的本地模型缓存；不承诺新加离线 CLI。

## 数据流与模块边界

保留 `ocr.recognize_cues` 作为两个入口共用的门面，先完成现有 Vision 抽帧、真实 PTS 校验与合帧。仅在 `engine == "vision_rapid"` 时调用新模块 `ingest/rapid.py` 对得到的每条 cue 复识别。

`rapid.refine_cues(video, cues, *, ocr, ffmpeg_path, ffprobe_path) -> list[RawCue]` 每条取 `(start + end) / 2` 秒截图，裁剪高度、1280 宽灰度处理与 Vision 路径一致。逐条抽帧并处理，避免将整集图片保存在内存。单次调用只初始化一个 ONNX CPU 引擎，禁用方向分类和 CoreML/CUDA。

RapidOCR 的像素框转换成现有 `TextBox` 的归一化、自下向上坐标。同一行框按纵向中心归组、按 x 排序；横向重叠时只去掉确实相同的首尾文字，其他框保留空格。随后调用现有 `frame_text`，同时遵循 `center_tolerance`、`crop_bottom` 和 CJK 筛选。纯英文支持不在此次范围内。

复识别仅改变 cue.text，保留 idx、src_idx、start、end、clamped 与 cue 顺序。过滤后为空则保留该条 Vision 原文，并报告回落数量。不会因为空结果删行，也不会再次按文本合帧。下游繁简转换、AI 校对、ASR 佐证、名表与报告沿用现有实现。

## 错误与缓存

依赖缺失抛 `OCRUnavailableError`，提示安装 ocr 和 rapidocr extras。已知模型初始化、推理或媒体错误给出明确中文 OCR/FFmpeg 错误；不把错误当空结果回落，不吞中断或任意编程错误。识别完整成功后才沿用现有原子字幕写入。

保持现有人工字幕保护：OCR SRT 仍按视频 mtime 复用，切换 engine 不自动失效。项目重识别需删除该集 `.ocr.srt`；独立命令用 `--force`。文档说明校对稿的人工编辑保护仍生效，不能承诺 force 覆盖人工稿。

## 验收

- 默认配置、旧 Vision 结果与可选依赖隔离不变；非法 engine 清楚报错。
- 两入口能传递 engine；组合路径对每个 Vision cue 只复识别中点帧。
- 持久测试覆盖坐标转换、同行合并、重叠去重、保留空格、crop_bottom、空回落、元数据保持及失败不发布。
- 真实短视频验证媒体抽帧；使用已有药屋 S3E01 素材完整运行组合识别，记录总耗时、复识别耗时、回落条数与截图抽查，不宣称准确率。
- 更新 README 安装/命令/缓存说明与 AGENTS 配置字段数、模块约定；相关测试和全量 pytest 通过，改动文件 Ruff 通过。
