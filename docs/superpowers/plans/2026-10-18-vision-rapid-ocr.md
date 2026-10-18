# Vision + RapidOCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为项目管线及独立 OCR 命令提供已验证的 Vision 定时、RapidOCR 复识别模式。

**Architecture:** 保留 `ocr.recognize_cues` 门面，在 Vision 合帧后按配置调用新 `rapid.refine_cues`。独立模块负责 CPU 引擎、逐条中点抽帧、文本框转换及空结果回落，下游沿用现有字幕处理。

**Tech Stack:** Python、Pydantic、RapidOCR 3、ONNX Runtime CPU、FFmpeg、Apple Vision、pytest。

**Spec:** `docs/superpowers/specs/2026-10-18-vision-rapid-ocr-design.md`

## Global Constraints

- `engine` 仅允许 `vision`、`vision_rapid`，默认 `vision`。
- RapidOCR 和 ONNX Runtime 惰性导入；默认 Vision 路径不加载它们。
- 复识别仅改变 cue.text，保留 idx、src_idx、start、end、clamped 与 cue 顺序。
- 过滤后为空则保留该条 Vision 原文；运行错误明确报错，不当空结果处理。
- 遵循 `crop_top`、`crop_bottom`、`center_tolerance`；保持现有 CJK 筛选。
- 保持 OCR 缓存按视频 mtime 复用的约定，不按引擎切换静默覆盖字幕。
- pytest/项目命令使用裸 `uv run`；不运行 `ruff format`。未经新授权不提交或推送。
- 在独立 worktree 实现；原树已有五个未跟踪文件不是本任务产物。

---

### Task 1: 实现并验证 RapidOCR 复识别模块

**Files:**
- Create: `src/tenmin/ingest/rapid.py`
- Create: `tests/test_rapid.py`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Consumes: `ocr.TextBox`, `ocr.frame_text`, `ocr.OCR_WIDTH`, `RawCue`, `OcrConfig`, `render.ffmpeg` 媒体封装。
- Produces: `refine_cues(video: Path, cues: list[RawCue], *, ocr: OcrConfig, ffmpeg_path: str, ffprobe_path: str) -> list[RawCue]`。
- 模块内部使用可 monkeypatch 的引擎构造和中点抽帧函数；只在执行构造函数时导入 rapidocr。

- [ ] 先写测试并运行 `uv run pytest tests/test_rapid.py -q`，确认模块缺失导致失败。用两条带完整元数据的 cue 和 fake 引擎验证首条替换、第二条空回落；断言输入未修改和时间顺序保持。

```python
assert result[0].text == "已经"
assert result[1].text == cues[1].text
assert [(c.start, c.end, c.idx, c.src_idx, c.clamped) for c in result] == [
    (c.start, c.end, c.idx, c.src_idx, c.clamped) for c in cues
]
```

- [ ] 增加框转换和同行测试：左右两框拼回完整句子；重叠的「给我一」与「一个月饼」输出「给我一个月饼」；非重叠「喂」与「你们」保留空格；上下两行不误拼；非居中及 crop_bottom 下层字幕被过滤；返回 boxes/txts 为 None 时空回落。
- [ ] 添加 `rapidocr = ["rapidocr>=3.7,<4", "onnxruntime>=1.20,<2"]` extra，运行 `uv lock`，用 `uv run --extra rapidocr python -c 'import rapidocr, onnxruntime'` 验证。记录实际解析版本。确认 CPU 参数名称与该安装版本配置一致，显式禁用 GPU/CoreML，调用 `engine(image, use_cls=False)`。
- [ ] 实现模块。以 `dataclasses.replace(cue, text=text)` 保留 cue 元数据；像素框 y 转换为 `1 - y1 / height`。逐条通过现有 FFmpeg 封装抽一帧，裁剪采用与 `ocr.py` 一致的取偶计算；校验帧长度，灰度复制为连续三通道数组。只构造一次引擎；不缓存整集图片。
- [ ] 为缺依赖、模型初始化/推理的已知异常、坏帧和中断分别写测试。用依赖实际异常族转换已知失败，未知 ValueError/RuntimeError 不做无差别吞掉。成功路径打印复识别条数与回落数量。
- [ ] `uv run pytest tests/test_rapid.py tests/test_ocr.py -q` 与改动文件 Ruff 通过。若 OCR 测试文件命名不同，先用 glob 定位实际文件，不跳过既有回归。

### Task 2: 接入配置、共享入口与独立命令

**Files:**
- Modify: `src/tenmin/config.py`, `src/tenmin/ingest/ocr.py`, `src/tenmin/cli.py`
- Modify: `tests/test_config.py`, `tests/test_ocr_batch.py`, `tests/test_cli.py`
- Modify: 实际已有的 OCR 识别测试文件

**Interfaces:**
- Consumes: Task 1 的 `rapid.refine_cues`。
- Produces: `OcrConfig.engine` 与 `tenmin ocr --engine`，两入口继续复用 `ocr.recognize_cues`。

- [ ] 先写默认值/合法值/非法值测试和 CLI 参数传递测试。配置目标：

```python
engine: Literal["vision", "vision_rapid"] = "vision"
```

  独立命令构造 `OcrConfig(crop_top=crop_top, crop_bottom=crop_bottom, engine=engine)`；参数使用 Literal 或 Enum 让非法值在运行识别前报错。
- [ ] 在现有 Vision 合帧成功之后加组合分支：

```python
if ocr.engine == "vision_rapid":
    from tenmin.ingest.rapid import refine_cues
    cues = refine_cues(
        video, cues, ocr=ocr,
        ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path,
    )
```

- [ ] 测试默认模式不调用 refine、不导入 rapidocr/onnxruntime，组合模式传入 Vision 结果与配置；两入口传递自定义 FFmpeg 路径；复识别失败不会写最终 SRT。检查 `ocr_batch` 对 OCRUnavailableError 整批中断、普通 OCRError 单集失败的现有约定。
- [ ] 测试缓存：已有新鲜 SRT 切换 engine 仍复用，独立 `--force` 才调用组合识别；保持人工校对稿保护。检查 `config_slices` 的 ingest 已覆盖 ocr.engine，其余阶段不因新增默认字段白白失效。
- [ ] 运行 `uv run pytest tests/test_config.py tests/test_config_slices.py tests/test_ocr_batch.py tests/test_cli.py -q`，加上 OCR/rapid 单测；Ruff 检查改动文件。

### Task 3: 真实验收、文档与全分支审查

**Files:**
- Modify: `README.md`, `AGENTS.md`
- Create: `docs/vision-rapid-validation.md`
- Modify: `tests/test_rapid.py`（真实微型媒体回归）

**Interfaces:**
- Consumes: `tenmin ocr --engine vision_rapid` 和项目 `ocr.engine`。
- Produces: 可复现的运行说明与验收证据。

- [ ] 增加无需真实模型的微型媒体抽帧回归，ffmpeg 不存在时按现有模式 skip；测试实际裁剪尺寸与中点帧，不给模型联网。
- [ ] 用药屋 S3E01 原视频在全新 `work/` 输出目录运行，保护既有字幕：

```bash
uv run --extra ocr --extra rapidocr tenmin ocr "/Users/portz/Downloads/[BeanSub][Kusuriya no Hitorigoto S3][01_49][CHS][1080P][x264_AAC].mp4" --engine vision_rapid -o work/vision-rapid-validation
```

  记录整次墙钟耗时、Vision cue 数、复识别数量、回落数量与实际依赖版本。与 `work/ocr-compare/vision-hans.srt` 和 `work/ocr-accel4/cpu.srt` 对照时注明 Vision 默认 language 差异；核实元数据与本次 Vision 输出一致。抽查正确修复、新错误、空回落各自截图，不给出未测准确率。
- [ ] README 增加 extra 安装、YAML、独立命令、CPU/macOS限制、首次模型下载及缓存失效操作。AGENTS 更新 OcrConfig 字段数、rapid 模块边界、pyobjc import 仍只在原 `_recognize`；测试文件数按实际 glob 重算。
- [ ] 运行 `uv run pytest tests/ -q`、`uv lock --check`、改动文件 Ruff 和 `rtk git diff --check`。新环境无 rapidocr 时确认默认 CLI 导入可用。
- [ ] 组织全分支审查，核对设计每项验收、生产写入原子性、依赖隔离、字幕缓存以及 crop_bottom。修复后重跑覆盖测试；将结果与剩余边界写入验收记录。提交/推送由用户另行授权。

## 计划自审

配置与两个入口由 Task 2 覆盖；识别几何、空回落、元数据和异常由 Task 1 覆盖；缓存约定由 Task 2 测试及 Task 3 文档覆盖；真实素材、可选依赖、全回归和审查由 Task 3 覆盖。唯一共享实现接口为 `rapid.refine_cues`，生产入口始终为 `ocr.recognize_cues`。
