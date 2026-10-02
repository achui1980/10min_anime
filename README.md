# 10 分钟看番剧 · tenmin

把番剧字幕 + 视频变成解说成片。输入 SRT 与源片，输出「分段文案与剪辑时间轴对照表」、
配音纯文本，以及配好音、烧好硬字幕的 mp4。

吃自带字幕的片源，也吃生肉：对白轨有四条来源（手传 SRT / 视频里的软字幕轨 / 画面硬字幕 OCR
/ 语音转写），生肉还会额外交付一份中文字幕。只想把一批视频的硬字幕认成 SRT 的话，
`tenmin ocr` 不用建项目就能批量跑（见「不建项目，单独批量 OCR」）。设计文档见
`docs/superpowers/specs/2026-08-31-10min-anime-design.md`，生肉那部分见
`docs/superpowers/specs/2026-09-21-tenmin-asr-translate-design.md`，硬字幕 OCR 见
`docs/superpowers/specs/2026-10-01-hardsub-ocr-design.md`。

## 核心思路

动画的演出语法是「该炫的时候不说话」。所以**长时间没有字幕的段落就是视觉高光**——
在《才女的侍从》第 2 集上验证：人工挑出的 7 个视觉高光里有 6 个落在 ≥3 秒的无字幕间隙里
（本集共检出 26 个这样的间隙，它们是候选池，由 `aggregate.py` 负责收敛）。
这条规则零成本、不需要看画面，是 v1 能只靠字幕干活的原因。

另外两条零成本信号：低字密度（说得慢 = 情绪爆发）、密度突变（叙事节奏换挡）。

## 安装

```bash
uv sync
export TENMIN_GEMINI_API_KEY=your-key
```

生肉片源（没有字幕、要靠语音转写）另外装一个 extra，它会把 torch 一起拽进来、几个 G：

```bash
uv sync --extra asr
```

不装也能跑 `tenmin --help` 和全部非转写用法；真需要转写时会报一句「跑一次
`uv sync --extra asr` 再试」而不是 traceback。只支持 mlx-whisper（Apple Silicon）。

画面上烧着字幕的片源（ANi / Baha 这类 `[CHT]` WEB-DL）走画面 OCR，再装一个只有几 MB 的 extra
（Apple Vision，只支持 macOS）：

```bash
uv sync --extra ocr
```

注意 extra 不叠加记忆：之后再跑一次不带参数的 `uv sync` 会把它卸掉。两个都要就写
`uv sync --extra asr --extra ocr`。不装也能跑 `tenmin --help`，真走到画面 OCR 时会报一句
「跑一次 `uv sync --extra ocr` 再试」。

渲染阶段需要编入 libass 的 ffmpeg（否则烧不了字幕）：

```bash
brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass
ffmpeg -hide_banner -filters | grep -w subtitles   # 能匹配到才算装对
```

## LLM 配置

默认用 Gemini（`TENMIN_GEMINI_API_KEY`）。也支持切到 MiniMax，在
`project.yaml` 里把 `llm.provider` 设成 `minimax`：

```yaml
llm:
  provider: minimax
  model: MiniMax-M3
```

对应的 key 放进 `.env`：

```
TENMIN_MINIMAX_API_KEY=your-key-here
```

### 使用其他 OpenAI 兼容模型（如 DeepSeek）

如果你有其他兼容 OpenAI chat/completions 接口的模型服务（比如 DeepSeek、
自建的开源模型服务），可以把 `llm.provider` 设成 `openai_compatible`，
并显式填写 `base_url`：

```yaml
llm:
  provider: openai_compatible
  model: deepseek-chat
  base_url: https://api.deepseek.com/v1
```

API key 放进 `.env`：

```
TENMIN_OPENAI_COMPATIBLE_API_KEY=your-key-here
```

注意：`openai_compatible` 目前只支持同时配置一个服务（一份 key，一个
project 用一个 base_url），且和 MiniMax 一样，schema 是写进 prompt
文本里再用 pydantic 校验的，不依赖服务端严格执行 `response_format`。

## 使用

```bash
uv run tenmin init saijo               # 创建 work/saijo/project.yaml 与 srt/
```

**一个 project 对应一部番，可以装多集**。往里加一集，用 `--episode`/`--srt`/`--video`
一次性注册并跑：

```bash
uv run tenmin run saijo --episode 2 --srt 你的字幕.srt --video 你的视频.mp4
```

这会把字幕拷进 `work/saijo/srt/E02.srt`（源片**留在原地**，只把它的绝对路径记进
`project.yaml`——源片实测 300MB~1.4GB，拷进 work/ 是纯冗余），在 `project.yaml` 的
`episodes:` 里补一条 `number: 2` 的记录，然后跑这一集的全链路。
`--srt` 必须配 `--video`（视频是渲染阶段的硬需求），且必须同时带 `--episode`。

**没有字幕的片源省掉 `--srt` 就行**，对白轨会自己找来源（见下一节；画面烧了字幕的片源要先在
`project.yaml` 里声明，见「硬字幕片源」）：

```bash
uv run tenmin run akujo --episode 11 --video 你的生肉.mp4
```

已经注册过的集，之后只需要带 `--episode` 就能重跑：

```bash
uv run tenmin run saijo --episode 2            # 重跑第 2 集
```

不带 `--episode` 时是**批处理模式**：把 `project.yaml` 里已注册的所有集都跑一遍
（按 mtime 跳过已是最新的阶段，跟单集模式一致）：

```bash
uv run tenmin run saijo                        # 跑 project.yaml 里的每一集
```

产物（每集独立一份，文件名带 `E{NN}` 前缀）：

- `work/saijo/out/E{NN}.解说方案.md` — 五列对照表，★ 标记纯字幕方案取不到的视觉高光
- `work/saijo/out/E{NN}.narration.txt` — 合并配音纯文本，直接丢给配音
- `work/saijo/03_script/E{NN}.script.json` — **唯一人工编辑面**，改完重跑 docgen 即可

`03_script/E{NN}.script.json` 管内容，`05_timeline/E{NN}.timeline.json` 管出片节奏。

```bash
uv run tenmin run saijo --episode 2 --only docgen --force   # 改完 script.json 重出文档，不调 LLM
uv run tenmin inspect saijo --episode 1         # 看无字幕间隙与高能点
uv run tenmin inspect saijo --episode 1 --suspect  # 看被标记为疑似 OCR 噪声的行
uv run tenmin run saijo --episode 1 --from signals   # 从指定阶段重跑某一集
uv run tenmin ocr ~/Downloads/番剧/ -o ~/srt/   # 不建项目，批量把硬字幕认成简体 SRT
```

只跑 v2 渲染部分（前 4 个阶段的产物照旧复用）：

```bash
uv run tenmin run akujo2 --from voice
```

手改过 `05_timeline/E02.timeline.json` 后只重新出片：

```bash
uv run tenmin run akujo2 --from audio --force
```

## 对白轨从哪来（四岔）

ingest 之前有一层来源解析，按「无损且便宜」排序取第一条成立的：

1. **手传的 SRT**（`--srt`，或 `project.yaml` 里那一集的 `srt:`）——人明确指定了，不猜。
2. **视频里的软字幕轨**——`ffmpeg` 一条命令抽成 `work/<slug>/srt/E{NN}.embedded.srt`，
   零成本零误差。**每次重抽**（抽取被打断留下的半份 SRT 在语法上合法，看不出是残骸）。
3. **画面硬字幕 OCR**——只在 `project.yaml` 声明了硬字幕时才走（见下文「硬字幕片源」），
   落成 `work/<slug>/srt/E{NN}.ocr.srt`（繁体原文）。一集 24 分钟约 3–4 分钟。
4. **语音转写**——前三条都不成立时才走，落成 `work/<slug>/srt/E{NN}.asr.srt`。
   开跑之前会打一行「没有字幕轨，只能走语音转写」，一集 24 分钟约 3 分钟。

第 1、2 条的对白轨记作 `source: "srt"`，第 3 条记作 `source: "ocr"`，第 4 条记作
`source: "asr"`（在 `01_dialogue/E{NN}.dialogue.json` 里，`tenmin inspect` 在对白轨那行
下面也会打出来）。这个字段决定两件事：**要不要繁转简**（srt 与 ocr 转；日语过 OpenCC 会被改字，
所以听写路径强制关掉）和 **translate 阶段怎么跑**（见下文）。

### 两个旋钮

```yaml
asr:
  model: mlx-community/whisper-large-v3-turbo
  # 源片语言。**韩语/英语片源必须改这里**，它不做自动检测。
  language: ja
```

### 换了模型要自己删缓存

`E{NN}.asr.srt` 的复用判据只有「文件非空 + mtime 不早于源视频」，**刻意不看 `asr` 配置**。
所以 **改完 `asr.model` 想重转，必须自己删掉那份 `.asr.srt`**：

```bash
rm work/<slug>/srt/E11.asr.srt
```

这么设计是因为那份 SRT 是**人可以手改**的产物（转差了就地改，改完 mtime 推过源视频、
下次照样复用）。按配置指纹失效就意味着改一次 `asr.model` 会把那些手改**静默冲掉**，
而「换了模型却没重转」打开文件就看得出来。

### 硬字幕片源（画面 OCR）

不做自动探测，要在 `project.yaml` 里声明。整部番都带硬字幕时写项目级的，个别集例外时逐集覆盖：

```yaml
ocr:
  enabled: true          # 这部番的片源带硬字幕
episodes:
  - number: 11
    video: /path/to/[ANi] 我是不才惡女 - 11 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4
  - number: 12
    video: /path/to/别的字幕组.mkv
    hardsub: false       # 只管这一集；不写 = 跟随 ocr.enabled
```

- 声明了硬字幕、但视频同时带软字幕轨时**照样走字幕轨**（文本字幕轨最准）。
- 声明了硬字幕、但 OCR 跑不了（没 `uv sync --extra ocr`、不是 macOS）时**直接报错**，不会
  悄悄换成语音转写——你明确说了要用画面上那份更好的素材。
- 其余旋钮（`sample_fps` / `crop_top` / `center_tolerance` / `similarity` / `min_frames` /
  `language`）的默认值来自一次实测，含义见 `src/tenmin/config.py` 的 `OcrConfig`。一条字幕都
  没认出来时会报错并提示检查 `ocr.crop_top`（裁剪区要框住画面底部的字幕）。
- **`E{NN}.ocr.srt` 跟 `.asr.srt` 一样只按 mtime 复用、不看 `ocr` 配置**：认错的字直接手改
  那份文件，下次照样复用；改了 `ocr.*` 参数想重认，得自己 `rm work/<slug>/srt/E11.ocr.srt`。
- **要手填 OP/ED 区间**（见「标注 OP / ED 区间」）：OCR 会把画面**居中**的 staff 字（监督、
  原作、ED 里的大块名单）一起认进来，左右两侧的才会被自动挡掉；剩下那批靠手填的
  `op_range` / `ed_range` 或 `credits.default_*` 去剔除。
- **那一集不能写 `srt:`**：手传 SRT 排在最前面，写了就不会走到 OCR。
- 声明了硬字幕、但片源带的是**位图**字幕轨（PGS / VobSub）时会报错：字幕轨分枝排在 OCR
  前面，抽不出来就停在那里，不会自动换到 OCR。提示会让你手传 `--srt`，或者把片源里的字幕轨
  去掉（比如 `ffmpeg -i in.mkv -map 0 -map -0:s -c copy out.mkv`），再跑就会走画面 OCR。

#### 上手：一集带硬字幕的片源

```bash
uv sync --extra ocr                                   # 1. 装 extra（仅 macOS）
uv run tenmin init akujo                              # 2. 新项目；已有项目跳过
# 3. 在 work/akujo/project.yaml 里写上 `ocr: {enabled: true}`（或在那一集写 `hardsub: true`），
#    并填好 op_range / ed_range
uv run tenmin run akujo --episode 11 --video "/path/to/[ANi] ... - 11 [...][CHT].mp4"
```

已登记过的集改成 OCR：删掉那一集的 `srt:`、加上 `hardsub: true`，然后
`uv run tenmin run akujo --episode 11 --force`。批处理（不带 `--episode`）照常可用，
每集各按自己的声明走。

跑的时候会先打一行预计耗时（约等于片长的 1/7，24 分钟一集约 3–4 分钟），之后每 10% 打一次进度：

```
  [ANi] 我是不才惡女 - 11 ....mp4 开始识别画面字幕（约 3 分钟）
  画面字幕识别 10%（571/5715 帧）
  ...
  画面字幕识别完成，387 条 → E11.ocr.srt
```

产物：

| 路径 | 是什么 |
|---|---|
| `work/<slug>/srt/E{NN}.ocr.srt` | 识别结果（**繁体原文**），可以手改，按 mtime 复用 |
| `work/<slug>/01_dialogue/E{NN}.dialogue.json` | 对白轨（已繁转简），`source: "ocr"` |
| `work/<slug>/out/E{NN}.zh.srt` | 简体中文字幕交付物，不调 LLM、不需要 API key |

之后的 signals → script → … → render 跟自带字幕的片源完全一样。核对来源：

```bash
uv run tenmin inspect akujo --episode 11
#   对白轨 E11：... 行，时长 ...s
#   来源：ocr（画面 OCR），硬字幕：已声明
```

实测（我是不才恶女 E11，1080p）：认出 387 条，抽 40 条核对 39 条一字不差，剩下 1 条混进了
居中的 OP staff 字——这就是要手填 OP/ED 的原因。人名比语音转写准得多（不需要术语表纠错）。

#### 旋钮（`ocr:`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `enabled` | `false` | 项目级声明这部番带硬字幕；逐集 `hardsub` 优先 |
| `sample_fps` | `4.0` | 每秒抽几帧去识别（与片源帧率无关，24/30/60/120fps 都是每秒 4 帧） |
| `crop_top` | `0.72` | 从画面高度的这个比例往下裁去识别；字幕位置偏高就**调小** |
| `center_tolerance` | `0.08` | 文字框中心离画面中线超过这个比例就丢掉（挡两侧 staff 字） |
| `similarity` | `0.6` | 相邻帧文字相似度不低于它就算同一条字幕 |
| `min_frames` | `2` | 少于这么多帧出现的识别结果当噪声丢掉 |
| `language` | `zh-Hant` | 识别语言（Vision 的语言代码） |

一条字幕都没认出来时会报错并提示检查 `ocr.crop_top`。

### 不建项目，单独批量 OCR：`tenmin ocr`

只想把一批视频的硬字幕认成 SRT、不需要解说成片时，用这个命令。它不读 `project.yaml`、不跑后面任何阶段，
同样要求 macOS + `uv sync --extra ocr`。

```bash
# 一个目录里的所有视频（只看这一层，认 .mp4 .mkv .mov .m4v .webm .avi .ts，按文件名排序）
uv run tenmin ocr ~/Downloads/番剧/

# 多个文件/目录混着传，输出统一放到一个目录（不存在会自动建）
uv run tenmin ocr a.mp4 b.mkv ~/Downloads/番剧/ -o ~/srt/

# 保留画面上的繁体原文（默认是繁转简）
uv run tenmin ocr ~/Downloads/番剧/ --traditional

# 字幕位置偏高，把裁剪区往上框；已有结果也重新识别
uv run tenmin ocr ep01.mp4 --crop-top 0.65 --force
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `-o` / `--output` | 视频旁边 | SRT 输出目录 |
| `--simplified` / `--traditional` | `--simplified` | 繁转简（OpenCC，与 ingest 同一个转换器）/ 保留繁体 |
| `--force` | 关 | 输出已存在也重新识别 |
| `--crop-top` | `0.72` | 同上表的 `crop_top`；其余旋钮用默认值 |

- 输出名带语言后缀：简体 `<文件名>.zh-Hans.srt`，繁体 `<文件名>.zh-Hant.srt`，两种都跑也不会互相覆盖。
- 输出已存在、非空且不比视频旧就跳过（打「已存在，跳过」），所以手改过的结果不会被冲掉；要重认加 `--force`。
- 某个视频失败（没认出字幕、ffmpeg 报错等）只记下来、接着跑下一个；最后打「成功 N，跳过 N，失败 N」，
  有失败时退出码为 1。没装 `ocr` extra 会直接中止。
- 它**不剔除 OP/ED 的 staff 字**（那要靠项目里的 `op_range` / `ed_range`），居中的 staff 字会留在 SRT 里。

#### 从 OCR 到成片：一条命令

先用 `tenmin ocr` 认出简体字幕，再把它当手传 SRT 交给 `tenmin run` 出片（项目要先 `tenmin init` 建好）。
下面以《才女的侍从》第 12 集为例，换片源只改 `V`、项目名和集号：

```bash
V="/Users/portz/Downloads/Video/[ANi] 才女的侍從 在滿是高嶺之花的貴族學校暗中照顧（毫無生活自理能力的）學院第一大小姐 - 12 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4"

uv run tenmin ocr "$V" -o work/saijo/srt/ \
  && uv run tenmin run saijo --episode 12 \
       --srt "work/saijo/srt/$(basename "$V" .mp4).zh-Hans.srt" --video "$V"
```

- 第一段约 3～4 分钟，产出 `work/saijo/srt/<片源名>.zh-Hans.srt`；第二段登记第 12 集并跑完 9 个阶段，
  成片在 `work/saijo/07_render/E12.mp4`，解说方案在 `work/saijo/out/E12.解说方案.md`（需要 LLM 的 key）。
- 整条命令可以重复跑：字幕已存在就跳过 OCR（手改过的不会被冲掉），后面各阶段按新鲜度复用。
- 跑完用 `uv run tenmin inspect saijo --episode 12` 看片头曲 / 片尾曲区间，不对就在那一集填上
  `op_range` / `ed_range`，再 `uv run tenmin run saijo --episode 12 --force`。
- 不想中间落一份 SRT 的话，也可以在那一集写 `hardsub: true`、不传 `--srt`，直接
  `uv run tenmin run saijo --episode 12 --video "$V"`（见上面「上手：一集带硬字幕的片源」）。

## translate 阶段：中文字幕 + 累积术语表

**只对 `source == "asr"` 的集真的翻译**（判据是那个字段，不做语言检测）。自带字幕的片源
（`source == "srt"`）本来就是观众读得懂的语言，这一阶段在磁盘上留不下任何痕迹——连 `zh/`
目录都不会建。画面 OCR 来的集（`source == "ocr"`）**不调 LLM**：认出来的字幕繁转简之后原样
交付成 `zh/E{NN}.zh.json` 与 `out/E{NN}.zh.srt`，不碰累积术语表，也不需要 API key。
（例外：`tenmin run` 只要本次要跑的阶段里有 translate，就会先把 LLM 客户端建出来，所以单跑
`--only translate` 时还是得配好 key；正常跑全链路时 script 阶段本来就要 key，没有区别。）

产物：

| 路径 | 是什么 |
|---|---|
| `work/<slug>/zh/E{NN}.zh.json` | 逐条译文（中间产物，带 id，对齐校验就盯它） |
| `work/<slug>/out/E{NN}.zh.srt` | **中文字幕交付物**，能直接拖进播放器 |
| `work/<slug>/zh/glossary.json` | **项目级**累积术语表（不带集号），日语原文 → 中文 |

累积表是跨集对齐译名用的：E02 的翻译调用会把 E01 积下的表当输入塞回去（「这些已经定了，
照用」），冲突时**累积的那个赢**——第 5 集换个写法会让成片看起来像换了个角色。它同时会被
**script 阶段的 prompt 读进去**，所以字幕里的人名和解说稿里的人名是同一套。

机器译错了要能盖掉它：在 `project.yaml` 里手写 `glossary`，**手写的赢**。

```yaml
glossary:
  リディア: 莉迪亚
```

`zh/glossary.json` 是 script 阶段的新鲜度输入，所以它的内容真变了才会刷 mtime
（否则一次 10 集的批处理会把前几集刚写好的解说稿全部判旧、下次白付好几次 LLM 费）。

生肉路径下**必须手填 OP/ED 区间**（见下一节），这不是可选优化：片尾识别的全部判据都是
「字幕组打在屏幕上的文字」特征（`©`、`作曲：`、`製作委員会`、人名罗列），而语音转写一条
都不会产出——它把主题曲**歌词**转成了正常对白，不填的话解说稿会拿歌词当剧情素材。

## 标注 OP / ED 区间

片头曲（OP）与片尾曲（ED）的区间有三级回退，**越靠前的优先**：

```yaml
show: 我是不才恶女
slug: akujo

credits:
  # 第 2 级：项目级手填，整季形态一致时只填这一处
  default_op_range: [300, 390]
  # ED 终点写 null = 到片尾（片长逐集不同，而 ED 起点通常是稳定结构）
  default_ed_range: [1290, null]

episodes:
  - number: 7
    srt: srt/E07.srt
    # 第 1 级：逐集手填，这集 OP 位置跟别集不一样
    op_range: [181, 260]
  - number: 8
    srt: srt/E08.srt
    # 什么都不填 → 第 3 级：从 staff 行自动推断
```

第 3 级的自动推断**不可靠**：它靠「把认出来的 staff 行聚成簇」反推，字幕组不打
staff 行、OP 位置异常时会静默返回 `None`。用 `tenmin inspect` 看实际生效的是哪一级：

```bash
uv run tenmin inspect akujo --episode 1
#   片头曲：(281.0, 368.0)（逐集手填）
#   片尾曲：(1354.0, 1429.95)（项目默认）
```

**手填区间不只是省事，它还会收紧 staff 行的识别窗。** 没有手填值时，识别 staff
行的激进规则（人名罗列、拉丁字母占比、有歧义的中文 staff 关键词）在「片头 0-300 秒
+ 片尾最后 80 秒」这个**盲窗**里生效 —— 于是 OP 开得晚的集会漏判 staff 行（它跑到
300 秒之后了），而片头的正常台词反倒会被误判成 staff。填上区间之后窗就是区间本身，
两个方向同时变好。实测一集真实素材：接住了 3 条落在 300 秒之后的 staff 行
（`副监督 野吕纯恵` / `监督 山崎みつえ` / `制作 ふかな悪女作委员会`），另有 5 条像
`此花同学 早安`、`欢迎回来 伊月` 这样的真台词不再被误判。

代价是**填错会在你填的区间内误判**，所以填完跑一次 `--only ingest` 再 `inspect`
核对一下分类计数。另外 `show` 必须写成真实剧名（简体）：标题卡（`《我是不才恶女》`
这种）是靠「书名号内容与剧名的字符重合率」认出来的，`show` 留着 `tenmin init` 写的
占位符会让这条规则完全失效。

## 阶段与产物

| 阶段 | 产物 | 是否调 LLM |
|---|---|---|
| ingest | `01_dialogue/E{NN}.dialogue.json` | 否（可能先跑一次画面 OCR 或语音转写，见上文四岔） |
| translate | `zh/E{NN}.zh.json`、`out/E{NN}.zh.srt`、`zh/glossary.json` | 只对 `source == "asr"` 的集调 LLM；`ocr` 的集直通交付，不调 |
| signals | `02_signals/E{NN}.signals.json` | 否 |
| script | `03_script/E{NN}.script.json` | **是** |
| docgen | `out/E{NN}.解说方案.md`、`out/E{NN}.narration.txt` | 否 |
| voice | `04_voice/E{NN}/chunk_*.mp3`、`04_voice/E{NN}.voice.json` | Edge-TTS 逐句配音，记录每段真实时长 |
| timeline | `05_timeline/E{NN}.timeline.json`、`05_timeline/E{NN}.ass` | 按真实配音时长重算画面时间轴，生成硬字幕 |
| audio | `06_audio/E{NN}.mixed.m4a` | 旁白盖在原声之上，旁白期间原声压低 |
| render | `07_render/E{NN}.mp4` | 一次编码完成切片、拼接、烧字幕、挂音轨 |

默认按 mtime 跳过已是最新的阶段，`--force` 强制重跑。

## 测试

```bash
uv run pytest -q                              # 默认：全部离线，不联网
uv run pytest -m llm                          # 真调 LLM 出快照（需 API key）
uv run pytest -m generalize                   # 泛化复验（需自备 SRT）
uv run pytest -m render     # 真跑渲染链路，需要真视频 + libass 版 ffmpeg
uv run pytest -m asr        # 真跑语音转写，需要 --extra asr + 真视频
TENMIN_OCR_SAMPLE_VIDEO=<片源> uv run pytest -m ocr   # 真跑画面 OCR，需要 macOS + --extra ocr + 硬字幕片源
```

## 已知限制

- **位图字幕轨（Blu-ray PGS / DVD VobSub）走不通**：上面第 2 条分枝只判「有没有字幕轨」，
  判不了「是不是文本」，于是位图轨会走进抽取路并在 ffmpeg 的跨族转码守卫上失败。会报一句
  中文提示（手传 `--srt`，或换一个没有字幕轨的片源让它走语音转写），但**不会自动回落到
  语音转写**——位图轨是能 OCR 的，悄悄换成听写等于把更好的素材丢了。声明了硬字幕时提示会改成
  「手传 `--srt`，或去掉字幕轨让它走画面 OCR」，同样不自动换路。
- 画面 OCR 只支持 macOS（Apple Vision），只认画面底部、水平居中的字幕；字幕在画面顶部或
  竖排的片源认不出来。变形宽高比（SAR ≠ 1，比如部分 DVD 源）没有专门处理。
- 行尾单个 `.` / `。` 会被当成识别错的省略号改成 `…`（Vision 常把 `…` 认成一个点）。
- 手传一份**日语** SRT（或软字幕轨恰好是日语）时 translate 不会跑，而且那份对白还会被
  繁转简改字。判据是 `source` 字段而不是语言检测。
- `work/<slug>/project.yaml` 的 `render:` 配置是整个 project 共享的，不支持按集覆盖。
- 源视频自带的原生硬字幕（某些片源有）在画面底部，某些时间点会跟 tenmin 烧的字幕重叠。

## 路线图

- **v1**（已完成）SRT → 对照表 + 配音文本
- **v2**（已完成）Edge-TTS 配音 + ffmpeg 切片拼接 + 混音 + 烧硬字幕 → 1920x1080 mp4
- **v3**（已完成）mlx-whisper 语音转写支持生肉 + `translate` 阶段交付中文字幕
- **v3.1**（本版）画面硬字幕 OCR（Apple Vision）作为对白轨的第四条来源；`tenmin ocr` 脱离项目批量识别，默认繁转简
- **v4** 本地 Web GUI（`script.json` 可视化编辑器）
- **v5** PySceneDetect + CLIP 视觉索引，整季 12 集压到 10 分钟
