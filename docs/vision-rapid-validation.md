# Vision + RapidOCR 验收记录

验收日期：2026-10-18。范围：默认 Vision、组合模式的接口与缓存契约、真实媒体抽帧和药屋 S3E01
完整识别。结论：工程契约通过；组合识别既有修复也有退化，作为显式可选模式交付，不能据此声称准确率提升。

## 运行与证据位置

工作目录：`/Users/portz/js/10min_anime/.worktrees/vision-rapid`。
本次全部新验收文件位于该目录下的 `work/vision-rapid-validation/`（忽略的本地文件，不随补丁交付）。
素材：`/Users/portz/Downloads/[BeanSub][Kusuriya no Hitorigoto S3][01_49][CHS][1080P][x264_AAC].mp4`，
344620141 字节，mtime_ns 为 `1791981918345184031`。

普通复现命令（目标目录应为空，复跑现有结果需另选目录或显式 `--force`）：

```bash
uv run --extra ocr --extra rapidocr tenmin ocr \
  "/Users/portz/Downloads/[BeanSub][Kusuriya no Hitorigoto S3][01_49][CHS][1080P][x264_AAC].mp4" \
  --engine vision_rapid -o work/vision-rapid-validation
```

实际执行使用同目录的 `run_validation.py` 调用真实 Typer `app`，参数与上面相同；仅在
`rapid.refine_cues` 与 `frame_text` 边界记录输入、输出、过滤空结果和耗时，没有替换识别/媒体算法。
启动命令为 `uv run --extra ocr --extra rapidocr python work/vision-rapid-validation/run_validation.py`。
完整输出在 `run.log`，结构化原始证据在 `evidence.json`。未开启 AI 校对或 ASR，没有 LLM 调用。

## 实测结果

| 项目 | 本次观测 |
|---|---|
| CLI 调用墙钟 | 350.726 秒（约 5 分 51 秒） |
| RapidOCR 复识别墙钟 | 149.829 秒，包含模型初始化、逐条 ffmpeg 抽帧与推理 |
| 其余 CLI 时间 | 200.896 秒，主要为 Vision；没有单独计量纯 Vision 时间 |
| Vision 采样 | 5754 帧 |
| Vision cue / 复识别 cue / 最终 SRT cue | 342 / 342 / 342 |
| 过滤后空文本回落 | 31 条，保留对应 Vision 原文 |
| 繁转简后文本发生变化 | 106 条（含空格、标点变化，不等于修对数量） |
| CLI 汇总 | 成功 1，跳过 0，失败 0；退出码 0 |

计时从 Python 内进入 CLI 前开始，包含记录包装的开销；不含 uv 启动、依赖同步、Python 导入或
最终 evidence.json 序列化，不冒充外部进程端到端计时。此次 RapidOCR 日志显示检测和识别模型均已在
缓存且校验有效，**未验证首次模型下载耗时或断网首次启动**。

环境：macOS 26.7.1 arm64，Python 3.14.6，ffmpeg/ffprobe 9.0.1；RapidOCR 3.10.0、
ONNX Runtime 1.31.0、NumPy 2.5.3、pyobjc Vision/Quartz 12.2.2。
模型为 `PP-OCRv6_det_small.onnx` 与 `PP-OCRv6_rec_small.onnx`，日志路径在工作树虚拟环境的
`site-packages/rapidocr/models/`。实现配置固定 CPU，关闭分类与 CUDA/CoreML/DML/CANN；
本次没有额外采集 ORT session 的 provider 诊断。

`analyze.py` / `analysis.log` 核实了：

- 本次 Vision 与复识别结果的每条 `idx/src_idx/start/end/clamped` 及顺序完全相同。
- 31 个过滤空结果均使用 Vision 原文，没有删除 cue 或二次合帧。
- 最终 SRT 文本与本次复识别结果经 OpenCC 简化后一致；起止误差不超过 0.51ms（SRT 毫秒舍入）。

## 与历史文件对照

历史文件只读，均位于主工作树 `/Users/portz/js/10min_anime/work/`，没有覆盖或写回：

| 历史文件 | cue 数 | 与本次时间对一致数 | 与本次 Vision 简化文本不同 | 与本次最终文本不同 |
|---|---:|---:|---:|---:|
| `ocr-compare/vision-hans.srt` | 342 | 342 | 5 | 107 |
| `ocr-accel4/cpu.srt` | 342 | 342 | 107 | 9 |

两份文件 SHA-256 分别为：

```text
vision-hans.srt 2d6580a8bc632da43f71d37728e2334f30d4e280ce6a8e22039c754f4e1d73c8
cpu.srt         ca4de0e604071704f022869c045ca3ef89c3b7dd783ce5d9374fedd41e389e34
```

历史 `vision-hans` 使用 Vision `language="zh-Hans"`（见 `docs/poc/ocr_engine_compare.py`），
本次 CLI 使用默认 **`zh-Hant`** 再繁转简，因此旧文本不是本次 Vision 原始结果，不能把这 5 条差异
归因于组合模式。历史 `cpu.srt` 是 PoC 使用旧 Vision cue 的复识别稿，也不是本次相同配置的对照实验。
其 `timings.json` 记录抽帧 37.3 秒、推理 96.4 秒、33 次回落；本次是串行逐 cue 抽帧/推理、
31 次回落，不能把两次耗时或差异当成受控性能/准确率结论。

## 画面抽查

截图由 `screenshots.py` 在 cue 中点抽取，全画面缩放至 1280 宽，人工查看；文件名均相对本地验收目录。
以下文本按简体展示，cue 编号为本次 Vision 列表的 1-based 位置：

| 类型 | cue / 中点秒 | Vision → 组合结果 | 截图观察 |
|---|---|---|---|
| 正确修复 | 6 / 171.296 | `民众的矛头就必定会指向荔国皇而` → `民众的矛头就必定会指向荔国皇帝` | `repair-006.png`：画面末字确为“帝” |
| 新错误 | 13 / 202.577 | `因此受到皇帝赏识的家族` → `好 辰 学` | `regression-013.png`：底部对白清楚存在，结果却混入背景书法并丢失对白 |
| 空回落 | 3 / 161.411 | `对` → `对` | `fallback-003.png`：底部确有“对”；过滤后为空，保留 Vision 正确结果 |

第 13 条揭示**非空错误不会回落**：规则只能挡空结果，不能证明剩余文字就是对白。31 条回落指
过滤后为空，不能全部解释为检测器无框（运行日志仅部分条目报空检测）。这三张是定向抽样，
没有逐条金标准、随机抽样或完整错误标注，不能估计准确率、召回率或净收益。

## 自动验证与全分支审查

本次执行：

- `uv run pytest tests/ -q`：**3261 passed, 41 skipped**，22.46 秒；1 条既有 google-genai
  Python 3.14 弃用警告，日志 `full-tests.log`。
- `uv run pytest tests/test_rapid.py tests/test_ocr.py tests/test_ocr_batch.py tests/test_resolve.py tests/test_config.py tests/test_config_slices.py tests/test_cli.py -q`：
  **505 passed, 1 skipped**，3.28 秒，日志 `targeted-tests.log`。
- 移除 extras 后 `uv run pytest tests/test_rapid.py -q`：**33 passed**，1.38 秒。
- `uv lock --check`：113 packages，成功；全 feature 改动 Python 文件 Ruff 成功。
- `uv sync` 移除 extras 后，实际确认 `rapidocr` / `onnxruntime` 均不可发现；真实 CLI `--help`
  退出 0，`sys.modules` 未加载二者。第一次尝试检查时环境仍有 extras，前置断言失败；同步到基础
  环境后重查通过，这不是 CLI 缺陷。

Task 1 遗留 P3 已关闭：真实 3 秒、320×240 视频的上部固定黑色，底部字幕带随时间黑→白→灰。
测试检查 0.5/1.5/2.5 秒像素、实际裁剪后 264×1280×3 灰度连续数组，再调用真实 `refine_cues`
（仅替换模型），验证 `[0.5, 2.5]` 的中点确为白帧。缺 ffmpeg/ffprobe 时跳过；不导入或下载模型。
额外在进程内把 seek 固定为 0.5 秒，测试如期失败（`midpoint-mutation.log`），恢复后 33 条通过。

按设计逐项自审（本次没有独立审查代理）：

| 契约 | 审查/验证依据 |
|---|---|
| 默认行为、非法 engine | Literal 配置与 CLI；默认 Vision 分支导入守卫；基础环境真实 CLI |
| 两入口与下游 | 项目 resolver、普通/校对 batch 都经 `ocr.recognize_cues`，只在 Vision 校验与合帧后分派；繁简、校对、ASR、名表路径继续复用 |
| 抽帧与几何 | 真实时变媒体；像素框 bottom-up 转换、同行排序、仅重叠文字去重、非重叠保留空格测试 |
| crop_bottom | 与 Vision 使用相同取偶裁剪高度；框中心过滤复用 `frame_text`，含 1081 高度舍入回归；本次整集没有设置 crop_bottom，双语真实素材未测 |
| 元数据与空回落 | 单测及整集 evidence 对比；只复制 text、不修改原 cue、不二次合帧 |
| 错误与原子发布 | 模型已知异常包装、中断/编程异常传播；project/final/raw 都在完整识别后原子写，失败保留旧文件回归 |
| 可选依赖 | `_build_engine` 内惰性导入；pyobjc 仍仅原 `_recognize` 内导入；CPU/分类参数测试 |
| 缓存与人工稿 | SRT 只按源视频 mtime；独立 force/项目删除原稿、人工校对保护回归；engine 默认只改变 ingest 配置切片 |

未发现需要扩展 Task 3 范围的工程缺陷。真实抽查退化是此模式的质量边界，保留显式选择及默认 Vision；
没有增加新识别策略。首次模型下载、其他操作系统/Python 版本、其他番剧、变形宽高比与真实双语
crop_bottom 效果均未由本次整集验证覆盖。
