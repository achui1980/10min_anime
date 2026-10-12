# 用截图和音频定位 OP/ED 区间设计

日期：2026-10-12
状态：设计已获用户确认，等待用户审阅 spec
依赖：`tenmin import`（`feat/batch-directory-import` 分支上的 `src/tenmin/importer.py`）

## 背景

`tenmin import` 现在只靠字幕文本推断 OP/ED。OCR 字幕源上这条路基本走不通。以 saijo E12 为例：OP staff 字被认坏，台词又被误判成 credits，结果 OP 推到 93.9–133.9（手填值是 0–95）；1390–1480 秒没有字幕，ED 推不出来。

用户的方案：给一张 OP 开头截图和一张 ED 开头截图，用画面找起点，再用音频定终点。

## 已确认的决策

1. 画面与音频结合：截图定起点，音频跨集对齐定 OP 终点。
2. 参考素材只有两张起点截图：视频目录下的 `op.*`、`ed.*`（扩展名 `.png`/`.jpg`/`.jpeg`，不分大小写），`tenmin import` 自动发现。
3. 有截图的一侧走画面+音频，没截图的一侧退回字幕推断。OP 和 ED 各自独立选来源，写入规则不变：两个都成功才写。
4. 有截图但画面对不上时，该侧判「未识别」，**不退回字幕推断**。
5. 搜索窗：OP 只看每集前 5 分钟（0–300 秒），ED 只看最后 3 分钟。窗内没匹配就判未识别，不扩大搜索窗。片长短于搜索窗时夹到片长内。
6. OP 终点用跨集音频对齐求出的长度；ED 终点一律等于 ffprobe 片长，片长未知则 ED 失败、不写。

## 识别流程

### 起点（OP 与 ED 相同）

1. 截图和视频帧走同一条管线：缩到 9×8、转灰度，算 64 位 dHash。
2. 在搜索窗内每秒抽 2 帧，找与截图汉明距离最小的帧。
3. 在该帧前后各 1 秒内改用 10fps 细化，起点精确到 0.1 秒。
4. 最佳距离大于 12/64 判未识别，预览显示这个距离。

### OP 终点

1. 在起点匹配成功的集里取匹配距离最小的一集，读它起点后 150 秒的音频（单声道 8kHz），每秒算 10 帧对数频带能量，作为参考包络。
2. 其他集从各自的起点开始逐帧比较，相似度连续 2 秒低于阈值的位置就是该集的 OP 长度。
3. 剔除离中位数超过 10 秒的长度，再取中位数 L。每集 OP 终点 = 该集起点 + L。
4. 只有 1 集匹配成功时 L 取 90 秒，预览注明「OP 长度按 90 秒估计」。

## 模块边界

- 新模块 `src/tenmin/ingest/av_match.py`，不读 `project.yaml`，也不碰登记。
  - 纯函数：`dhash(gray9x8) -> int`、`best_frame(hashes, ref) -> (index, distance)`、`audio_envelope(pcm) -> ndarray`、`op_length(ref_env, env) -> float | None`、`median_length(lengths)`。
  - IO：`sample_hashes(video, start, end, fps)` 复用 `ffmpeg.run_raw_frames`；`read_pcm(video, start, duration)` 调新增的 `ffmpeg.run_pcm`（s16le、单声道、8kHz）；`image_hash(path)` 处理截图。
- `importer.py`：
  - `scan_directory` 把截图放进 `ScanResult.references`，不算未配对文件。
  - `build_import_plan` 分两遍：先逐集找起点，再汇总 OP 起点成功的集做跨集对齐求 L。
  - `infer_ranges` 按侧选来源。
  - `RangePreview` 新增 `op_source` / `ed_source`、`op_distance` / `ed_distance`、`op_length_estimated`；`render_preview` 在区间后标来源（「画面+音频」或「字幕推断」）和匹配距离（如 `5/64`）。
- 依赖：`numpy` 提为主依赖（现在只是 `uv.lock` 里的传递依赖）。不引入 OpenCV 或 fpcalc。

## 错误处理

- 截图读不了：整个命令报错退出，什么都不写。
- 某集抽帧或读音频抛 `FFmpegError`：该侧判未识别并写明原因，其他集照常继续。
- `FFmpegBinaryError` 照旧外抛。
- 画面匹配比字幕推断慢，只在有截图时才做；预览前逐集打印「正在匹配画面 N/M」。

## 测试

- 新增 `tests/test_av_match.py`。
  - 纯函数用合成数据：dHash 对亮度变化稳定、能区分不同画面；`best_frame` 的阈值；从合成能量包络找回段长；中位数剔除离群值。
  - IO 用 ffmpeg lavfi 生成色块加正弦音的短视频，验证时间戳和字节数，标 `render` marker。
- importer 层 monkeypatch av_match 的 IO 函数，覆盖：有截图走音画、无截图走字幕、截图对不上不退回字幕、单集 90 秒、截图损坏整体报错、预览中的来源与距离。
- 真实素材测试：新 marker `av`，用 `TENMIN_AV_SAMPLE_DIR`（saijo 目录），默认跳过。同步更新 `pyproject.toml` 的 marker 列表和 AGENTS.md（五个改为六个）。
- 命令：`uv run pytest tests/test_av_match.py tests/test_importer.py -q`、`uv run pytest tests/ -q`。

## 范围之外

- 自动截取参考图。
- 一部番有多首 OP（换季）。
- 只给音频片段做参考。
- ED 之后的 C 段（ED 终点一律取片长）。
- `tenmin run` 自动做音画匹配。
