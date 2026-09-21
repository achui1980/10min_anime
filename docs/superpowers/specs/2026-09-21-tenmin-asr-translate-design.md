# tenmin 生肉支持：ASR 取对白轨 + 翻译阶段

日期：2026-09-21
状态：设计已确认，待实现

## 要解决的问题

现在 tenmin 只吃自带字幕的片源：`EpisodeConfig.srt` 是必填字段，CLI 强制 `--srt` 和 `--video` 一起传。
手上很多片源是生肉（无软字幕轨），或者只有烧在画面上的硬字幕 —— 这两种目前完全跑不了。

本次要做两件事：

1. **只有视频也能跑**：没有 SRT 时自动拿到对白轨。
2. **日语对白也能出中文交付物**：生肉是日语，最终要的是中文。

## 前置验证（已完成）

用 `[ANi] 我是不才惡女 - 11 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4` 实测过：

- `ffprobe` 结果：video(h264) + audio(aac, 48kHz, 2ch, **语言 tag = jpn**) + mjpeg 封面。**零字幕轨**，`[CHT]` 是硬字幕。全片 1429.99 秒。
- 抽 5:00–7:00 的 120 秒单声道 16k wav，跑 `mlx_whisper.transcribe(path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="ja", word_timestamps=True)`：
  - 首次模型下载 202 秒（一次性，~1.6G 缓存在 `~/.cache/huggingface`）
  - 转写 120 秒音频耗时 15 秒 → **~8x 实时**，推算 24 分钟全片约 3 分钟
- 输出质量：干净的分句日语 cue，带 start/end。抽帧看硬字幕交叉验证两句，**语义级完全吻合、时间戳对得上**：

  | 硬字幕（繁中） | ASR 日语输出 |
  |---|---|
  | 想必讓此地陷入混亂並非您的本意 | この地を混乱させることは本意ではないはずです |
  | 我的脈搏是不是有力得讓人吃驚？ | びっくりするくらい元気な鼓動でしょ |

  字面级有少量听写误差（`どうすればいい` → `どうすからいい`、`ごまかせない` 被截成 `ごまかせな`）。**对 tenmin 无害**：对白只是喂给 script 阶段 LLM 当剧情理解材料，成片里烧的是解说文本、不是对白原文。
- 发现两个必须处理的点：音频末尾有 5 条 `[119.96 -> 119.96]` 零长度空 segment（whisper 边界 artifact）；whisper **不提供 speaker 标注**。

结论：ASR 路径可行，质量对下游够用。

## 关键设计决定

### 决定 1：翻译只给人看，管线内部保持日语原文

管线内部对白轨保持**日语原文**，script 阶段 LLM 直接读日语写中文解说。
另外产出一份**独立的中文 SRT** 作为纯交付物。

理由：翻译是有损转换，如果先翻成中文再喂 script，翻译误差会污染 script 的剧情判断。
而 script 的 prompt 明确要求「人物关系与动作方向必须以对白原文为准 …… 谁称呼谁、谁对谁用敬语」
（`script/prompts/single_episode.md:63-64`）—— **敬语是日语语法特征，翻成中文就丢了**。

### 决定 2：翻译是独立新阶段 `translate`

STAGES 变成：`ingest → translate → signals → script → docgen → voice → timeline → audio → render`

- 插在 ingest 之后：只有 ingest 产物变了才重跑翻译，改 script prompt 不重新付翻译费。
- signals 不依赖 translate（它只看时间戳和字数）。
- 代价：需要 LLM 的阶段从 1 个（script）变 2 个。
- ASR 本身**不是**新阶段，它进 `ingest` —— 它就是对白轨的另一个来源，`models.py:179` 的
  `source: Literal["srt", "asr"]` 早就留好位了。

### 决定 3：ASR 触发判据是自动推断三岔

```
传了 --srt                      → srt_parser 解析
没传 --srt，但视频有软字幕轨      → ffmpeg 抽成 SRT 再走 srt_parser（静默）
两者都无                        → 跑 whisper（先打一行告知）
```

中间那条**不能漏**：带软字幕轨的片源被白白拉去跑 3 分钟 ASR 还掉质量，而 ffprobe + ffmpeg
两条命令就能零成本零误差拿到。

第三条要在终端打一行「E11 无字幕轨，将对音轨转写，预计 3 分钟」再开跑。
不加 `--asr` flag：软字幕抽取是无损的，没必要让用户多打字；ASR 是唯一一处「有代价且有损」
的分枝，值得被看见，但不值得为它加一个每次都要记得传的 flag。

### 决定 4：ASR 依赖做 optional extra，引擎只做 mlx-whisper 一个

`pyproject.toml` 新增 `[project.optional-dependencies]`（仓里第一个 extras 节）：

```toml
asr = ["mlx-whisper>=0.4"]
```

**mlx-whisper 会把 torch 一起拽进来，几个 G。** 目前主依赖全是 typer/pydantic/httpx 这种轻量包，
不跑生肉的人不该背一整个机器学习栈。不跑生肉：`uv sync` 照旧；要跑生肉：`uv sync --extra asr`。

**只做 mlx-whisper，不做引擎抽象、不加 config 选引擎的字段。** 使用者只在 Apple Silicon Mac 上
自用，faster-whisper 的跳平台优势在这里不值钱。YAGNI。

两条强约束：

- **`import mlx_whisper` 必须写在函数体内**，不能放模块顶层 —— 否则没装 extra 的人连
  `tenmin --help` 都跑不起来。
- **没装 extra 却走到 ASR 分枝时要报可读的错**，告诉人跑 `uv sync --extra asr`，而不是丢一个裸 ImportError。

### 决定 5 / 6：产物路径

`translate` 的产物放在**不占编号**的 `zh/` 目录：

| 产物 | 路径 | 性质 |
|---|---|---|
| 逐条中文译文 | `zh/E{NN}.zh.json` | 中间产物 |
| 累积术语表 | `zh/glossary.json` | 中间产物，**项目级**（不带集号） |
| 中文字幕 | `out/E{NN}.zh.srt` | 交付物 |

`Paths` 现在是 `01_dialogue` → `07_render` 连续编号，**这些目录名字符串被全仓注释大量引用**
（`signals/gaps.py:78`、`script/validate.py:623,645`、`render/chunks.py:182`、`render/tts.py:291,313`、
`render/timeline.py:285`、`config.py:120`、`script/llm.py:146`、`cli.py:193`）。真按顺序插进去要重编号
02→08 六个目录，注释层面波及很广，还会让 `work/` 下已有的 13 集存量产物路径全部失配。

代价是阶段编号不再等于执行顺序 —— 但 `docgen` 写 `out/` 已经立了这个先例。

中文 SRT 进 `out/`，跟 `out/E{NN}.narration.txt`、`out/E{NN}.table.md` 并排 —— `out/` 已经是
「给人看的东西都在这」的约定。

### 决定 7：翻译用带 id 的结构化输出 + id 集合校验

实测 `work/saijo/01_dialogue/E02.dialogue.json` 是 408 条 cue / 3773 字（繁中，日语估 5–6k 字符），
**一次 LLM 调用完全塞得下，分批省钱不是真问题。**

真问题是**对齐**：408 条进去必须 408 条出来。LLM 漏一条或合并两条，整条轨的中文就会跟时间戳
静默错位，只有看成片字幕时才发现第 200 句之后全错。现有 `script` 阶段不吃这个风险（它输出自由
创作的分幕，没有一一对应约束）—— `translate` 是第一个有这种约束的阶段。

做法：输入 `[{id, ja}]`，schema 要求输出 `[{id, zh}]`，拿回来核对 id 集合是否完全相等，
缺/多/重就把差集拼成错误消息回灌重试。

这个校验**刻意放在 pydantic schema 之外**：schema 管不了「id 集合是否完整」这种跨条目约束，
而且回灌要告诉模型「你漏了第 137、298 行」这种具体信息。

### 决定 8：术语表由 translate 自动产出并跟集累积

日语路径下术语飘移有两根轴：

1. **translate 与 script 之间**：translate 一次调用译完整集，同一次调用内自洽；但 script 是
   另一次调用、读日语现场译，可能给出不同译名（`out/E11.zh.srt` 写「莉迪亚」、解说里写「莉蒂亚」）。
2. **集与集之间**：E01/E02 的 translate 调用互不知情。比第 1 条更难发现（单看一集完全自洽）。

做法：translate 的 schema 多一个字段「本集出现的专有名词 + 你选的中文译法」，落成项目级
`zh/glossary.json`。

- script 阶段读它 → 关掉第 1 根轴。
- E02 的 translate 调用把 E01 积下的表当输入塞回去（「这些已经定了，照用」）→ 关掉第 2 根轴，
  随集数收敛。
- `project.yaml` 的手写 `glossary` 作为**可选覆盖/纠错入口**，从「必须做的前置工作」降级成可选。

**架构后果：translate 从此是 script 的真上游**，不只是顺序在前 —— script 的 prompt 要读
`zh/glossary.json`，这是真数据依赖。决定 1/2 里「翻译只给人看、旁支」的措辞要按这一条理解。

术语表的现状（已查清）：`config.py:463` 的 `ProjectConfig.glossary: dict[str, str]` 是**纯手写、
无任何自动提取**，而且 `work/saijo/project.yaml:11` 是 `glossary: {}` —— 13 集跑下来从没填过。
它有两个消费点，语义不同：`ingest/clean.py:56` 的 `apply_glossary` 做字面 `str.replace`
（设计用途是繁转简之后的**同语言用词归一**，如 `侍从 → 管家`），`script/single.py:127` 的
`build_glossary_block` 渲染成 prompt 里的 `- key → value`。

### 决定 9：`{{dialogue_block}}` 只喂日语原文

曾考虑「日语原文 + 中文译文双语并排」。否掉的理由：

- 双语的主要卖点是术语一致，但决定 8 用一张自动产出的 `zh/glossary.json` 就解决了，不必把
  整条对白轨翻倍 —— 对白轨占整份 prompt 的 84%，双语会让一集 prompt 从 ~35k 涨到 ~50k+ 字符，
  返工轮跟着涨。
- 日语原文提供中文译文拿不到的东西（敬语、称呼），见决定 1。

唯一代价：prompt `:114` 要求「留白金句全部来自对白轨」，script 得自己把选中的日语句改写成
中文金句，可能跟 `out/E11.zh.srt` 里同一句的译法不同。**判定这不算错** —— 金句本来就是创作性
改写，要短要有冲击力，跟字幕直译风格本就该不同；人名靠术语表已经锁住了。

已确认的好消息：**`script/validate.py` 对 `dialogue` / `line.text` / `cue.text` 零引用** ——
它是纯结构/时长校验，跳语言完全不影响它。

### 决定 10：生肉路径下 OP/ED 靠手填

**`ingest/credits.py` 的 credits 识别在 ASR 路径下会整条链崩。** `is_credits` 的七条规则**全部**是
「字幕组打在屏幕上的文字」特征，ASR 一条都不会产出 —— whisper 转的是语音，不会念「作曲：XXX」，
而是把**主题曲歌词**转写成正常 cue：

- 规则 1 `©` / `(C)`（`credits.py:132`）
- 规则 2a `_KEYWORDS_ALWAYS`（`credits.py:22-34`）：製作委員会/作詞/作曲/編曲/フォント/主題歌 等
- 规则 2b `_KEYWORDS_IN_WINDOW`（`credits.py:40-53`）：製作/協力/作画/監督/脚本/演出/原作/STAFF/Studio
- 规则 3 书名号包裹 + 剧名字符重合、规则 4 纯人名罗列、规则 5 拉丁字母占比、规则 6 标题卡

后果链：

1. `is_credits` 对 ASR 输出几乎恒为 False；
2. `find_credit_ranges`（`credits.py:292`）的 OP 聚簇主路筛 `kind == "credits"` 必然空 → 退到
   `_op_from_silence` 静区兜底（`credits.py:268-289`）—— 但 OP 那 90 秒**有歌声被转写成 cue**，
   所以不是静区，兜底也失效；
3. ED **只走聚簇、刻意不做兜底**（`credits.py:321-324`，注释：「片尾前的长静场是真高光，兜底会吃掉它」）
   → ED 必然识别不到；
4. 于是 signals 拿主题曲歌词算静音间隙和语速，script 拿歌词当剧情素材 —— 成片可能在 OP 上插解说。

做法：**手填 `op_range` / `ed_range`**。`EpisodeConfig` 已经支持，而且它是三级回退的最高优先级
（填了就直接驱动 `in_credit_window`，盲窗完全不参与，各留 `manual_window_margin`=5s 余量，
见 `credits.py:206-249`）。零新代码；生肉路径下这从「可选优化」升级成「注册新集时的必填项」
（拖进播放器看一眼 OP 起止，30 秒）。

**后续可选增强（本次不做）**：让 LLM 认歌词 —— 顺路搭 translate 的车（那一趟本就要把整集 cue
喂进去，schema 多一个字段「哪几条是主题曲歌词」几乎零成本）。风险是多一个会错的自动判断，
所以留到手填用烦了再说。

## 各模块设计

### ingest 怎么拿到对白轨

在 `build_track` **上游**加一个取 cue 的解析层，`build_track` 本身签名不动：

```
resolve_cues(srt: Path | None, video: Path, *, cfg) -> tuple[list[RawCue], Literal["srt","asr"]]
  ├ srt 不为 None            → srt_parser.parse(srt)                → "srt"
  ├ ffprobe 视频有字幕轨      → ffmpeg 抽成临时 SRT → srt_parser.parse → "srt"
  └ 都没有                   → 打一行告知 → mlx-whisper 转写          → "asr"
```

三条要点：

- **第 2 条抽出来的也走 `srt_parser`**，不另写一套。ffmpeg 抽软字幕轨吐标准 SRT/ASS，转 SRT 后
  跟手传的没区别。这条分枝几乎不增加代码面，只多一个 ffprobe + 一个 ffmpeg 调用。
- **ASR 产物落盘缓存到 `srt/E{NN}.asr.srt`**（`srt/` 目录已存在，`register_episode` 就往那拷）。
  一集 3 分钟转写，`--force` 重跑 ingest 不该重付；落成 SRT 还意味着可被人手动修正，修完下次
  直接走第 1 条路。缓存新鲜度挂视频 mtime。
- **`source` 字段照 `models.py:179` 已留的位填**，并用它**派生**两件事（不是让用户在 project.yaml
  配 —— 它是「数据是什么语言」的事实，不是创作旋钮）：`source == "asr"` 时 `convert_traditional`
  强制 False；translate 据此知道要不要跑。

**为什么 `convert_traditional` 必须强制 False**：`ingest/clean.py` 的 `clean_text` 顺序固定为
「剥标签 → 繁转简 → 术语表」，`to_simplified`（`clean.py:36-39`）走 `OpenCC("t2s")`。日语文本过这一道
会被改字（`製作` → `制作` 等日语汉字被改成简体）。好消息是这已经是 config 旋钮不是硬编码：
`config.py:73` `LocaleConfig.convert_traditional: bool = True` → `pipeline.py:272` →
`normalize.py:163` → `normalize.py:207`，接线现成。

顺带确认的几条（都不用改）：`clean.py:101` 的 `_CJK` 含假名，所以 `is_suspect` 对日语正常；
`extract_prefix`（剔 `(角色名)` 前缀）与 `split_dual_track`（按括号前缀拆双轨）对 ASR 输出是空转
—— whisper 不产出括号说话人前缀，也不会把两句叠在同一时间区间。

**`duration` 不用动**：`pipeline.py:276` 本来就传 `_source_duration(cfg, episode)`，
`_source_duration`（`pipeline.py:250-261`）本来就是 ffprobe 源视频、`FFmpegBinaryError` 往上抛、
其余 `(FFmpegError, OSError)` 返 None；`normalize.py:168` 的 `duration: float | None = None`
缺省才回退 `max(cue.end)`。生肉路径有视频，这条自动就是精确值。

### `src/tenmin/ingest/asr.py`（新）

跟 `srt_parser.py` 并排 —— 同一层的另一个 cue 来源。

```python
def transcribe(video, *, model, language, cache) -> list[RawCue]
```

- **延迟 import**：`import mlx_whisper` 在函数体内，
  `except ImportError as e: raise ASRUnavailableError("这一集没有字幕，需要 ASR。请跑 uv sync --extra asr") from e`。
  `ASRUnavailableError` 继承 `RuntimeError`，进 `cli.py` 的 `PIPELINE_ERRORS`，跟 `LLMError`/`TTSError` 一套。
- **音轨抽取走已有的 `render/ffmpeg.py`**，不另写 subprocess（那里已统一
  `text=True, errors="replace"` 与 `FFmpegBinaryError`/`FFmpegError` 分层）。抽全片单声道 16k wav
  到临时文件，转完就删。
- **mlx-whisper 是阻塞调用**，`run_ingest` 在 async 上下文，所以走 `asyncio.to_thread`（跟
  `render/tts.py` 里 `probe_duration` 同样处理）。
- **尾部幻觉过滤就地做**：丢掉 `end <= start` 或文本不含任何假名/汉字/字母数字的 cue。
  **不指望下游 `is_noise` 接住** —— 那是清洗层，让脏数据流进去会污染 `ingest_warnings` 的统计。

新 config 节 `AsrConfig`（照现有六个子 config 的形状）：

```python
model: str = "mlx-community/whisper-large-v3-turbo"   # 实测 8x 实时，24 分钟片约 3 分钟
language: str = "ja"
```

**不加 `engine` 字段**（决定 4）。

### `src/tenmin/translate/`（新子包）

照 `script/` 的形状：

```
translate/
  __init__.py
  lines.py                     # translate_track()：拼 prompt、调 provider、id 对齐校验
  glossary.py                  # zh/glossary.json 的读写与合并
  srt_writer.py                # TranslatedTrack + DialogueTrack → out/E{NN}.zh.srt
  prompts/translate_lines.md
```

**刻意不复用 `script/prompts/`**：两个阶段的 prompt 生命周期无关，混一个目录会让「改翻译 prompt」
看起来像在动 script。

`lines.py` 直接吃 `script/llm.py` 的 `LLMProvider` protocol 和 `_complete_with_schema_repair`。
**后者要提成公开的**（去下划线）—— 它本就是 provider 无关的通用机制，只是此前只有一个消费者。

新增数据模型（放 `models.py`）：

```python
class TranslatedLine(BaseModel):
    id: int          # 对齐 DialogueLine 的行号
    zh: str

class TranslatedTrack(BaseModel):
    episode: int
    lines: list[TranslatedLine]
    glossary: dict[str, str]   # 本集新认出的 日语原文 → 中文译法
```

`glossary` 放在轨里而非单独一次调用，因为它跟译文是同一次 LLM 输出的两个字段 —— 分两次调用
无法保证译文用的就是它报上来的译名。

prompt 结构（照 `single_episode.md` 已标定的 prefix 缓存原则：静态段在前、本集素材在后）：

```
## 任务          翻译日语对白为简体中文
## 输出格式      schema
## 已定术语      {{glossary_block}}      ← 累积表，「这些已经定了，照用」
## 对白轨        {{lines_block}}         ← [{id, ja}]，占绝大部分
## 输出前自检    id 必须一条不漏
```

### translate 阶段的编排

**输入**：`01_dialogue/E{NN}.dialogue.json` + 项目级 `zh/glossary.json`
**输出**：`zh/E{NN}.zh.json`、`out/E{NN}.zh.srt`，并**回写** `zh/glossary.json`

**新鲜度（关键陷阱）**：`zh/glossary.json` 既是 translate 的输入又是输出，塞进它的 `is_fresh`
输入集会让它永远不新鲜。所以：

- **translate 的新鲜度只看 `01_dialogue/` 的 mtime**，glossary 不进它的输入集。
- **script 的输入集要加 `zh/glossary.json`**（决定 8，手改译名要让 script 重跑）。

**跳过条件**：`source == "asr"` 才跑，`source == "srt"` 整阶段跳过（现有 13 集繁中片源零影响，
`zh/` 目录都不建）。判据就是这一个字段，**不做语言自动检测**、也不去猜 SRT 里是什么语言。

这条规则的**已知边界**（接受，不在本次处理）：手传一份日语 SRT、或者视频自带的软字幕轨恰好是
日语，两种情况 `source` 都是 `"srt"`，于是既不会被翻译、还会被 `convert_traditional` 拿 OpenCC
改字。要处理就得引入语言检测，而目前手上的片源不存在这两种情况 —— 真撞上了再说。

**刻意不做多集并发**（script 有 `script_concurrency`，translate 不加）：E01 写完
`zh/glossary.json`、E02 才读到含 E01 的版本，并发会让累积失去意义，且两个 task 会同时回写同一文件。

### script 阶段的改动

只有一处：`script/prompts/single_episode.md:96` 的 `{{glossary_block}}` 内容从「中文 → 中文」
变成「日语原文 → 中文译法」（由 `zh/glossary.json` 渲染）。

`:94` 的标题和 `:63-64`/`:110`/`:114` 那些「以对白原文为准」的措辞**一个字都不改** ——
它们说的是「对白轨」，跳语言后依然成立，而 `:63-64` 的敬语要求在日语下才真正生效。

`script/single.py:127` 的 `build_glossary_block` 复用，只是喂进去的 dict 来源从 `cfg.glossary`
变成「`zh/glossary.json` 叠上 `cfg.glossary`（手写的覆盖自动的）」。

**`ingest/clean.py` 的 `apply_glossary` 不碰** —— 它吃的还是 `cfg.glossary`（同语言用词归一），
跟 `zh/glossary.json`（日→中）是两回事。刻意不让它们串到一起：串了就会把日语对白里的人名
替换成中文，直接违背决定 1。

### CLI 与 config

现状是 **`srt: Path` 必填、`video: Path | None = None` 可选**（`config.py:58-59`）—— 本次要把这个
必填/可选关系倒过来一半：

- `config.py:59`：`EpisodeConfig.srt: Path` → `Path | None = None`；`srt_path()`（`config.py:496-500`）
  跟着返回 `Path | None`。`video` 维持 `Path | None`。
- **新增 `EpisodeConfig` 的 model_validator：srt 和 video 至少要有一个，都没有就报错。**
  这条以前不需要（srt 必填自带这个保证），改成两个都可选之后必须显式补上。
- `cli.py:248-254` 两条校验改成三条：`--srt` 必须配 `--video`（视频是硬需求，ASR 和 render 都要它）；
  **只传 `--video` 合法**（生肉入口）；传了任一个就必须传 `--episode`（不变）。

  生肉登记命令：`tenmin run saijo --episode 11 --video <片>`
- `pipeline.py:420` `register_episode(cfg, *, episode, srt: Path | None, video)`：srt 为 None 就跳过
  `atomic.copy_file`，`EpisodeConfig.srt` 留空。`model_dump(exclude_none=True)` 已会把空 srt 从
  yaml 里省掉，序列化那半不用改。
- **新鲜度（关键一处）**：`pipeline.py:782/804` 的
  `srt_inputs = [cfg.srt_path(ep) for ep in cfg.episodes]` + `if force or not is_fresh(outputs, srt_inputs)`
  改成「每集取 srt 和 video 里存在的那些」。

  两个连带：`srt/E{NN}.asr.srt` 那份 ASR 缓存**不进** ingest 输入集（它是 ingest 自己的产物，
  其新鲜度在 `resolve_cues` 里对着 video mtime 单独判）；人手改了它想生效得 `--force`
  —— 有意为之，当输入会让「ASR 跑完写出缓存」立刻使 ingest 不新鲜。

已有的好地基：`pipeline.py:473` 注释说 `project.yaml` 是**每个阶段**的隐式输入
（`_is_fresh` 把它加进 inputs）。所以决定 8 里「手写 glossary 当覆盖入口」自动就有新鲜度，
不用额外接线。

### docgen：不改

`render_table(script)`（`docgen/table.py:92`）只吃 `Script`，从头到尾没读过对白轨。5 列是
`节点 | 原片截取时间戳 | 建议画面特征 | 分段解说文案 | 剪辑与原声处理`，`table.py:1` 那行注释写着
「列顺序是用户确认过的模板，不许改」。

那张表里**没有对白列**，也就没有可以配对中文的地方。硬加一列语义上也不通：一个 beat 对应若干
clip 区间，一个区间里可能跨 20 条对白，塞不进一格。

所以**中文的唯一人类交付口就是 `out/E{NN}.zh.srt`** —— 一份标准 SRT 可以直接拖进播放器跟视频
对着看，比在对照表里挤一列好用。`zh/E{NN}.zh.json` 的用途相应收窄为「`out/E{NN}.zh.srt` 的生成源
+ 累积 glossary 的中间态」。

### speaker：保持关闭

whisper 不提供 speaker 标注。所以 `merge_continuations`（跨行同句合并）维持默认关闭
（`build_track(..., merge_lines=False)`），**不引入 pyannote diarization**（YAGNI）。

这推翻了 2026-08-31 design spec 里「`merge_continuations` 保留为带开关的纯函数，供带说话人标注的
字幕源（v3 ASR 输出）显式启用」的预期 —— 那份 spec 当时假设 ASR 会带 speaker，实际不会。

## 测试策略

现有 `tests/` 40 个文件的路子是「纯函数密集测 + 外部依赖全 fake」，新增的跟着走。

**纯逻辑，正常单测（无 marker，默认跑）**

- `resolve_cues` 的三岔分枝：给 srt → 走 parser；ffprobe 报有字幕轨 → 走抽取；两者都无 → 调 ASR。
  **ffprobe/ffmpeg 全部 fake 掉**（`tests/fakes.py` 已有这套路子，`test_render_ffmpeg.py` 就是这么测的）。
- ASR 尾部幻觉过滤：喂一个含 `end <= start` 和纯符号 cue 的假 whisper 输出，断言被丢掉。
  **不需要真跑 whisper。**
- `translate/lines.py` 的 id 对齐校验：用 `tests/fakes.py` 的 fake provider 造「漏了第 137 条」
  「多了一条不存在的 id」「同一 id 重复两次」三种坏输出，断言各自触发重试且错误消息里带具体 id。
  **这是这个阶段最该锁死的不变量。**
- `translate/glossary.py` 的合并：自动表叠手写表、手写的覆盖自动的、空表降级。
- `translate/srt_writer.py`：`TranslatedTrack` + `DialogueTrack` → SRT 文本。时间戳格式复用
  `timecode.py`，这里只测拼装与转义。
- config：`EpisodeConfig` 那条新 model_validator（srt/video 至少有一个）；`AsrConfig` 默认值。
- **新鲜度**（`test_pipeline.py` 已经全是这类测试）：只有 video 的集算得出 ingest 该不该重跑；
  改 `zh/glossary.json` 让 script 不新鲜；改 `zh/glossary.json` **不**让 translate 不新鲜
  （那条自噬陷阱值得一个专门的测试锁住）；`source == "srt"` 且中文时 translate 整阶段跳过。
- **缺 extra 时的行为**：`monkeypatch` 让 `import mlx_whisper` 抛 ImportError，断言拿到
  `ASRUnavailableError` 且消息里有 `uv sync --extra asr`。这条不需要 marker，而它恰好是
  「没装 extra 的人」最容易撞上的路径。

**需要真依赖的，加 marker**

`pyproject.toml` 现有三个 marker（`llm` / `generalize` / `render`），第 41-50 行那段注释解释了
为何**不**用 `addopts` 按 marker 摘（门在代码里、比按 marker 摘更准）。ASR 跟着这个约定加第四个
marker `asr`：真跑 mlx-whisper + 真视频，默认跳过，门是「装了 extra 且给了素材」。
`test_marker_gate.py` 那个「marker 门自己的测试」要同步加一条。

**不测的**：whisper 的转写准确率。那是模型的事，我们测不了也不该测；前置验证阶段已经人工
交叉验证过一集够用。

## 明确不做的

- **硬字幕 OCR**。前置验证里为了交叉验证 ASR 质量，用过「ffmpeg 抽帧 + 裁字幕区 + 模型读图」
  这个土办法，但那是肉眼读的、样本只有 3 帧，**不是可复用方案**。真要做硬字幕提取需要正经
  OCR 管线（抽帧 → 去重 → OCR → 合并时间轴），成本和错误率比 ASR 高一个量级，而 ASR 已经够用。
- **引擎抽象 / faster-whisper**（决定 4）。
- **pyannote diarization 拿 speaker**。
- **翻译分批**（决定 7：整集一次调用塞得下）。
- **让 LLM 认主题曲歌词**（决定 10 的后续可选增强，手填用烦了再说）。
- **语言自动检测**。translate 的跳过判据用 `source` 字段，不猜。

## 实现顺序建议

1. `AsrConfig` + `pyproject.toml` 的 extras 节 + `ASRUnavailableError`（地基，无行为变化）
2. `ingest/asr.py` + `resolve_cues`（ingest 能吃生肉，但对白还是日语、下游未适配）
3. config/CLI/`register_episode`/新鲜度（`tenmin run --episode N --video <片>` 能跑通）
4. `models.py` 的 `TranslatedLine`/`TranslatedTrack` + `_complete_with_schema_repair` 提成公开
5. `translate/` 子包 + `run_translate` + STAGES 接线
6. script 的 glossary 来源切换 + prompt `:96` 的措辞
7. 端到端跑一集生肉验收
