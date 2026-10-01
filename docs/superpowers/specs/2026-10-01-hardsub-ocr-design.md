# tenmin 硬字幕 OCR：从画面里认出对白轨

日期：2026-10-01
状态：设计已确认，待实现

## 要解决的问题

ANi / Baha 这类 WEB-DL 片源没有字幕轨，繁体中文字幕直接烧在画面上（文件名带 `[CHT]`）。现在这种片源只能走语音转写，路径是日语 ASR → translate 译成中文。

实际用下来还要绕一圈。`work/akujo` 的做法是先跑 ASR + translate，再把 `out/E11.zh.srt` **手工登记回去**当 `srt/E11.srt` 用。翻出来的人名也要靠术语表一条条纠（主嘉碧→朱雅媚、铃琳/露琳→玲琳）。可画面上本来就有一份人工翻译好的中文字幕，信息最全，却被丢掉了。

本次要做的是：对声明了"带硬字幕"的片源，抽帧做 OCR 认出这份字幕，作为对白轨的来源。

`2026-09-21-tenmin-asr-translate-design.md` 当时把硬字幕 OCR 排除在外，理由是"成本和错误率比 ASR 高一个量级"。下面的实测推翻了这个判断。

## 前置验证（已完成）

**测试条件**
- 素材：`[ANi] 我是不才惡女 - 11 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4`。h264 1920×1080，23.976 fps，时长 1429.99 秒，码率 1.82 Mbps。
- 机器：macOS 26.6.2 arm64。
- 测试代码只放在临时目录，没有进仓库。

**画面特征**
- 字幕是白字黑边，水平居中，最多两行，纵向位置在画面高度的 81%–95%。
- 双人对白写成 `-A\n-B`。另有旁白式的信息卡，如 `（數日後）`。
- OP/ED 的日文 staff 字也出现在同一条底带里，大多在左右两侧。但有一部分是居中的：监督、原作，以及 ED 里最多 6 行的大块。

**做法**
1. ffmpeg 一条命令完成取帧：`select='not(mod(n\,6))',crop=iw:ih*0.28:0:ih*0.72,scale=1280:200,format=gray`，以 rawvideo 走管道输出。
2. 每帧转成 CGImage，交给 `VNRecognizeTextRequest`，参数为 accurate、`zh-Hant`、开启 language correction。

**实测数据**

| 项目 | 结果 |
|---|---|
| 速度 | 抽了 5715 帧（每 6 帧取 1 帧，约 4 fps），OCR 每帧 34.7 ms。整集墙钟 201 秒，约 7 倍实时，与 ASR 的约 8 倍同量级。ffmpeg 解码不是瓶颈 |
| 条数 | 388 条（同一集 ASR 出 371 条），不缺句 |
| 字幕显示时长 | 最短 0.75 秒，p5 为 1.0 秒，中位数约 2.25 秒 |
| 空帧 | 34% 的采样帧没有字 |
| 降到 2 fps | 丢 6 条，另有 50 条只被采到 2 帧（很脆弱）→ 4 fps 是合适的密度 |
| 准确率 | 随机抽 40 条，39 条逐字正确。错的 1 条是混进了居中的 OP staff 字（`監督\n好想趕快再見到她`） |
| 人名 | 朱慧月、玲琳、堯明全部正确，不需要术语表纠错 |

**实测中发现的规律**（设计直接依据这些）
- **多帧投票是准确率的关键。** 同一句字幕会被采到 4–10 帧。388 条里约 68 条在不同帧上认出过不同写法，取出现次数最多的那个，就能修掉单帧错字（未/末、日/目、又/叉、冷/泠），也能去掉前缀里的杂字（`C你…`）。少数条目是众数本身就错了（如 `表蓬友好`），只能人工修。
- **省略号必须先统一。** Vision 会把 `…` 认成 `•••`、`⋯`、`.`、句尾的 `。`。不先统一就比相似度，像"不要…"这种短句会被拆成 5 条。
- **只出现 1 帧的都是噪声**（`7000\n找`、`/1/7`、`MIMM`）。"至少 2 帧"加上"必须含汉字"两条规则就能清掉。
- **按像素变化检测字幕切换行不通。** 灰度帧差和字幕切换基本不相关，背景运动占主导。即使只跳过"几乎不变"的帧，也只能省约 27% 的 OCR 调用（每集约 55 秒），还会减少投票的帧数。所以"变化检测"和"跳过重复帧"两个方案都放弃。
- Vision 的 confidence 只有 0.3 / 0.5 / 1.0 几档，太粗，不拿它当判据。

## 关键设计决定

### 决定 1：显式声明，不做自动探测

由 `project.yaml` 声明"这部番带硬字幕"（项目级 `ocr.enabled`，可以逐集用 `episodes[].hardsub` 覆盖）。没声明的片源维持现在的行为。

不自动探测的原因：自动探测要先抽样再判断画面底部"有没有字"。画面里本来就会有字（片头 staff、招牌、信件），误判避免不了，而且每集都要多付一轮抽样的成本。一部番的片源通常来自同一个字幕组，在项目级声明一次就够了。

### 决定 2：对白来源的优先级

```
手传 SRT → 视频内软字幕轨（抽取）→ 声明了硬字幕（OCR）→ 语音转写
```

- 声明了硬字幕、但视频同时带字幕轨时，**仍然走字幕轨**。文本字幕轨是最准的素材。
- 声明了硬字幕、但 OCR 跑不了（没装 extra，或者不是 macOS）时，**直接报错，不悄悄回落到语音转写**。这跟"位图字幕轨不自动回落"是同一条原则：用户明确说了要用画面上那份更好的素材，静默换成听写等于把它丢了。

### 决定 3：引擎是 Apple Vision，只有这一个

- 用 `VNRecognizeTextRequest`，通过 pyobjc 调用。
- 依赖放在可选 extra `ocr` 里（`pyobjc-framework-Vision`、`pyobjc-framework-Quartz`），只有几 MB，不需要下载模型。
- 只支持 macOS，跟 ASR 的 mlx-whisper 一样。
- 沿用 ASR spec D4 的做法：只接一个引擎，不抽象引擎接口，导入延迟到真正用到时，导入失败给出可操作的报错。

### 决定 4：OCR 结果直接用，可以手改；不做 LLM 校对

- 产物是 `srt/E{NN}.ocr.srt`，内容是繁体原文。
- 新鲜度只跟源视频的 mtime 比，**刻意不看 `OcrConfig`**。这跟 `E{NN}.asr.srt` 的理由一样：它是一份人可以手改的产物，如果按配置指纹判失效，手改的内容会被静默冲掉。
- 代价是：改了 OCR 参数之后，要自己删掉这份 SRT 才会重新识别。
- 剩下的零星错字交给人工修，或者交给 script 阶段的 LLM 在理解剧情时自然消化。

### 决定 5：新增来源 `"ocr"`

`DialogueTrack.source` / `SubtitleSource.kind` 从 `Literal["srt", "asr"]` 扩成 `Literal["srt", "asr", "ocr"]`。下游按来源有三处不同处理：

| 处理 | srt | asr | ocr |
|---|---|---|---|
| OpenCC 繁转简（`normalize.py` 的 `convert_traditional and source == "srt"`） | 转 | 不转 | **转** |
| translate 阶段 | 跳过 | LLM 翻译 | **不调 LLM，直接产出简体 zh.srt（见决定 6）** |
| OP/ED 启发式推断是否可靠 | 可靠 | 不可靠，需要手填 | 不可靠，需要手填 |

不复用 `"srt"` 的原因有三个：来源可以追溯；`tenmin inspect` 能显示出来；以后要加 OCR 专属的处理（比如可疑行标记）时有地方挂。

关于 OP/ED：OCR 会把居中的 staff 字一起认进来。所以文档要写明，硬字幕片源应该手填 `op_range` / `ed_range`，或者项目级的 `credits.default_*`，交给现有的三级回退去剔除。

### 决定 6：OCR 来源也交付 `out/E{NN}.zh.srt`

translate 阶段遇到 `source == "ocr"` 的集时，**不调 LLM**：

- 从对白轨（已经繁转简）里按 `select_translatable` 选出 `SPEECH_KINDS` 行，也就是 dialogue 和 monologue；credits、screen_text、noise 不收，筛选规则跟 ASR 路径完全一致。
- 用这些行构造一个 `TranslatedTrack`：`zh` 取对白原文，`glossary` 为空。
- 照常写 `zh/E{NN}.zh.json` 和 `out/E{NN}.zh.srt`，后者复用 `render_zh_srt`。

这样做的好处：
- 产物集合与 `pipeline.run_pipeline` 里 translate 的 `outputs` 一致，新鲜度沿用 `_translate_inputs` 和 `is_fresh`，不另开一套判据。
- "`out/E{NN}.zh.srt` 由 translate 产出"这条始终成立。

需要注意的几点：
- 这条分支**不碰**累积术语表，也不写 `zh/E{NN}.usage.json`。
- 它也**不需要 LLM provider**。实现时要确认 run_translate 走这条分支时不会因为没有 API key 而失败。
- `source == "srt"` 照旧早退，连 `zh/` 目录都不建。

## OCR 管线（`src/tenmin/ingest/ocr.py`）

模块的结构仿照 `ingest/asr.py`：

- 异常族：`OCRError(RuntimeError)` 和 `OCRUnavailableError(OCRError)`，`OCRError` 加进 `cli.PIPELINE_ERRORS`。
- Vision 的调用收在模块级 `_recognize(...)` 里，在函数内部 import。测试会 monkeypatch 这个函数。
- 写盘走 `tenmin.atomic.write_text`。

### 1. 抽帧

- **步长**：`step = max(1, round(src_fps / ocr.sample_fps))`，`src_fps` 用 `ffmpeg.probe_frame_rate` 读。例如 23.976 / 24 fps 得 6，30 得 8，60 得 15。每次取到的都是一帧真实存在的帧，不会有插值或重复帧。
- **ffmpeg 滤镜**：`select='not(mod(n\,STEP))',crop=iw:ih*(1-crop_top):0:ih*crop_top,scale=1280:-2,format=gray`。
  - 裁剪按比例算，跟分辨率无关。
  - 缩放只固定宽度为 1280，高度按比例取偶数。720p 和 4K 交给 Vision 的都是同一个尺度的图。
- **输出**：`-f rawvideo` 走 stdout 管道，用 `-fps_mode passthrough` 保证一帧对一帧（不让 ffmpeg 为凑恒定帧率补帧或丢帧）。不在磁盘上写临时图片。
- 管道读取需要新加一个 ffmpeg 封装，放在 `render/ffmpeg.py` 里，沿用它的错误处理风格，失败时抛 `FFmpegError` 并带上 stderr 尾部。抽帧参数（crop、scale、select 怎么拼）属于业务逻辑，留在 `ocr.py`。
- **每帧的时间戳读真实 pts，不按序号推算。**
  - 滤镜链末尾接 `showinfo`，它会在 stderr 上为每个输出帧打一行 `pts_time:`，顺序与 stdout 上的帧一一对应。
  - 不用 `序号 × step / src_fps` 推算的原因：那个公式只对恒定帧率成立。番剧片源基本是恒定帧率，但 120 fps 这类高帧率文件常是录屏或补帧、属于可变帧率，按序号推算会让时间轴越跑越偏。读 pts 对恒定帧率片源没有任何坏处。
  - 可变帧率片源上，`select` 仍然按帧序号每 `step` 帧取一帧，所以每秒实际取到的帧数会随源帧率波动；投票和归并只看帧的先后，不受影响。
  - stdout（帧数据）和 stderr（pts）必须同时读，否则任一管道写满都会让 ffmpeg 卡死。实现上用一个线程读 stderr。
  - pts 行数与收到的帧数对不上时抛 `OCRError`，不猜。
  - 120 fps 片源每秒仍取 4 帧（步长 30），OCR 次数不变，但 ffmpeg 要解码全部帧，解码量约为 24 fps 的 5 倍。本次未实测，遇到这类片源时再测。

### 2. 单帧清洗：`frame_text(observations, *, center_tolerance) -> str`

1. 只保留文字框中心横坐标满足 `|x − 0.5| < center_tolerance` 的行（坐标按图宽归一化）。这一步去掉左右两侧的 staff 字。
2. 去掉不含汉字的行。
3. 剩下的行按从上到下排序后用 `\n` 拼接。注意 Vision 的 y 轴从下往上。
4. 统一省略号：把 `•••`、`⋯`、连续两个以上的 `.` / `。`、`・・・` 等写法换成 `…`。
5. 去掉首尾空白。结果为空串表示这一帧没有字幕。

### 3. 合并成 cue：`merge_frames(frames, *, interval, similarity, min_frames) -> list[RawCue]`

**归并规则**（按时间顺序扫描）：
- 当前帧和当前 cue 最后一个**非空帧**的 `difflib.SequenceMatcher.ratio()` 达到 `similarity` 时，归入同一个 cue。
- 中间允许夹 **1 个**空帧（`_MAX_GAP_FRAMES = 1`，作为模块常量），用来容忍单帧漏认。
- 不相似，或者连续空帧超过 1 个，当前 cue 结束。

**每个 cue 的定稿**：
- **文本**：所有帧里出现次数最多的写法。平票时取最早出现的那个，保证结果确定。
- **起点**：首帧的时间。
- **终点**：末帧的时间加一个 `interval`（也就是 `step / src_fps`）。
- **丢弃**：出现帧数少于 `min_frames` 的 cue。
- **去重叠**：如果上一条的终点晚于下一条的起点，把上一条的终点截到下一条的起点。实测会出现 `00:00:20,020` 对 `00:00:20,019` 这种毫秒级的重叠。

**输出格式**：复用 `asr.render_srt(cues)` 和 `RawCue`，时间戳格式与 ASR 产物完全一致。

### 4. 入口：`recognize(video, dest, *, ocr, ffmpeg_path, ffprobe_path) -> None`

- 开始时打印一行提示：`{video.name} 声明了硬字幕，开始识别画面字幕（约 N 分钟）`。N 按时长除以 7 估算。
- 每完成 10% 打印一次进度。
- 识别结果为 0 条时抛 `OCRError`，提示检查 `ocr.crop_top`，**不写出空文件**。
- 同步执行，跟 `asr.transcribe` 一致。

## 配置

### `OcrConfig`（`config.py`，`ProjectConfig.ocr`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `enabled` | `False` | 这部番的片源带硬字幕 |
| `sample_fps` | `4.0` | 目标取样密度（每秒帧数），实际步长会按源帧率取整 |
| `crop_top` | `0.72` | 裁剪区上沿占画面高度的比例，一直裁到画面底部 |
| `center_tolerance` | `0.08` | 文字框中心离水平中线的最大距离，按画面宽度的比例算 |
| `similarity` | `0.6` | 相邻帧文本相似度达到多少算同一句 |
| `min_frames` | `2` | 一个 cue 至少出现几帧，少于这个数丢弃 |
| `language` | `"zh-Hant"` | Vision 的识别语言 |

- 字段的合法范围在 pydantic 里校验：`0 < crop_top < 1`、`0 < center_tolerance ≤ 0.5`、`0 < similarity ≤ 1`、`sample_fps > 0`、`min_frames ≥ 1`。
- 默认值来自本 spec 的实测，把出处写进字段注释。
- 不进配置的是"物理上的实现细节"，留作模块常量：缩放宽度 1280、允许的空帧数 1、识别级别 accurate。

### `EpisodeConfig.hardsub: bool | None = None`

`None` 表示跟随 `ocr.enabled`。判定写成 `cfg.hardsub_enabled(episode)` 这一个入口，resolve 和 inspect 都调用它。

### 其他

- `ProjectConfig` 的子 config 从 8 个变成 9 个，`AGENTS.md` 里那个数字要同步改。
- `config_slices.STAGE_FIELDS["ingest"]` 加上 `"ocr"`。`EPISODE` 本来就在切片里，`hardsub` 随它一起进入。按决定 4，切片变化会让 ingest 重跑，但**不会**让 `.ocr.srt` 失效，这与 asr 的行为一致。
- `pyproject.toml` 加 extra `ocr = ["pyobjc-framework-Vision>=…", "pyobjc-framework-Quartz>=…"]`，下限取实现时的当前版本。再加 pytest marker `ocr`，并在 `tests/conftest.py` 的 `_GATED_MARKERS` 里登记。

## 接入点

- `ingest/resolve.py`
  - `_OCR_SUFFIX = ".ocr.srt"`。
  - `resolve_subtitle_source` 新增参数 `hardsub: bool` 和 `ocr_config`。软字幕轨分支之后、ASR 分支之前：如果 `hardsub` 为真，先看 OCR 缓存能不能用（判据同 `_is_usable_asr_cache`，泛化成一个按路径判断的函数），能用就复用，否则调用 `ocr.recognize`。返回 `SubtitleSource(path, "ocr")`。
- `pipeline.py`：`Paths` 新增 `ocr_cache`（`srt/E{NN}.ocr.srt`）。`run_ingest` 把这个路径、`hardsub` 和配置传下去。这份缓存**不进** ingest 的新鲜度输入，理由与 `asr_cache` 相同。
- `normalize.py`：繁转简的判断改成 `source in ("srt", "ocr")`。
- `pipeline.run_translate`：按决定 6 分支。
- `cli.py`：`PIPELINE_ERRORS` 加上 `OCRError`。`tenmin inspect` 显示每一集对白轨的来源（srt / asr / ocr），以及是否声明了硬字幕。
- 文档：
  - `AGENTS.md`：对白来源从三选一改成四选一；补上 `.ocr.srt` 的手改与失效约定；子 config 个数；marker 个数从四个改成五个。
  - `README.md`：补上硬字幕片源的用法，以及"要手填 OP/ED"的提醒。

## 测试

全部用替身，不依赖真实视频或 Vision。

- `tests/test_ocr.py`
  - 步长计算：23.976 / 24 / 30 / 60 / 120 fps，以及极低帧率时步长取 1。
  - pts 解析：从 `showinfo` 的 stderr 里按顺序取 `pts_time`；不均匀间隔（可变帧率）原样采用；pts 行数与帧数不一致时抛 `OCRError`。
  - `frame_text`：居中过滤、必须含汉字、多行从上到下拼接（y 轴从下往上）、各种省略号统一、空帧。
  - `merge_frames`：相似帧并成一条；夹 1 个空帧仍连续；连续 2 个空帧断开；多帧投票及平票取最早；少于 `min_frames` 丢弃；重叠截断；终点等于末帧时间加 interval。
  - `recognize`：monkeypatch 抽帧和 `_recognize`；缺依赖时抛 `OCRUnavailableError` 并带 `uv sync --extra ocr` 提示；0 条时报错且 dest 不存在；写盘走 atomic（源码卫生测试已有覆盖）。
- `tests/test_resolve.py`
  - 四路优先级；声明了硬字幕但有字幕轨时走字幕轨；OCR 缓存的新鲜度与手改保护；逐集 `hardsub` 覆盖项目级设置；未声明时行为不变。
  - 把 `test_the_source_kind_is_only_srt_or_asr` 改成锁定三值。
- `tests/test_normalize.py`：`source="ocr"` 会繁转简。
- `tests/test_pipeline.py`：ocr 来源的集，translate 不调用 provider，产出 `zh.json` 和简体 `zh.srt`，只收 speech 行，不碰术语表；srt 来源仍然早退。
- `tests/test_config*.py`：`OcrConfig` 的默认值与边界校验、`hardsub` 的覆盖规则、切片里有 ocr。
- `@pytest.mark.ocr`：用真实片源的冒烟测试（需要 macOS、extra 和视频），默认跳过。

## 不做

- LLM 校对 OCR 错字。
- 跳过重复帧、按变化检测取帧（实测否决）。
- 顶部字幕、竖排字幕、同屏多处字幕。
- 非 macOS 平台的 OCR 引擎。
- 位图字幕轨（PGS / VobSub）的 OCR：它现有的中文报错不变。
- 自动探测片源有没有硬字幕。
