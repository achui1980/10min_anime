# 10 分钟看番剧 — 设计文档

日期：2026-08-31
状态：已确认，待实现

## 1. 项目目标

把番剧字幕自动转换成可直接投产的**解说方案**：一张「分段文案与剪辑时间轴对照表」加一份「合并配音纯文本」。

对照表的每一行给出四件事：从原片哪几段时间截取画面、这些画面拍的是什么、配什么解说文案、原声与音效怎么处理。使用者拿着这张表就能完成剪辑，或者交给后续版本的渲染管线自动出片。

包名 `tenmin`，CLI 命令 `tenmin`。

## 2. 范围

### v1 交付（本文档的实现目标）

输入 SRT，输出三个文件：

| 文件 | 内容 |
|---|---|
| `解说方案.md` | 分段文案与剪辑时间轴对照表 |
| `narration.txt` | 合并配音纯文本，可直接粘贴进 TTS 工具 |
| `script.json` | 上述两者的结构化来源，人工可编辑 |

v1 **不碰视频文件**，不做 ASR、不做 TTS、不做渲染。全流程纯文本，秒级完成，可反复重跑调提示词。

理由：对照表和配音文本的质量决定了成片质量的上限，是整个项目价值的绝大部分。渲染是确定性的工程活，风险低，推后无损。

### v1 明确不做

- ASR（生肉转字幕）
- TTS 语音合成
- ffmpeg 切片与渲染
- 视觉索引（CLIP/SigLIP）
- GUI

### 后续版本路线

| 版本 | 增量 |
|---|---|
| v2 | Edge-TTS 合成 + ffmpeg 切片拼接 + 混音 + 烧硬字幕，出 1920x1080 mp4 |
| v3 | Faster-Whisper ASR，支持生肉输入 |
| v4 | 本地 Web GUI，`script.json` 的可视化编辑器 |
| v5（可选） | PySceneDetect + CLIP 视觉索引，画面精修 |

v1 的架构必须为这些留好接口，但不实现。

## 3. 已确认的产品决策

| 决策项 | 结论 |
|---|---|
| 使用者 | 自用，日后要 GUI |
| 压缩粒度 | 整季 → 10 分钟为最终目标；v1 先跑通单集 → 约 4 分钟，数据模型按整季设计 |
| 字幕来源 | 双模式：现成 SRT（v1）/ ASR 生成（v3）。汇聚成同一个标准化对白轨 |
| 画面定位 | 字幕粗定位 + 视觉细选的混合思路。v1 用三条零成本时间戳规则，视觉索引降级到 v5 |
| LLM | 云端 API，Provider 接口抽象 |
| TTS | Edge-TTS 起步，接口可插拔（v2） |
| 成片画幅 | 横屏 16:9，1920x1080（v2） |
| 音频 | 旁白为主 + 原声垫底 sidechain ducking（v2） |
| 解说字幕 | 烧硬字幕（v2） |

## 4. 架构

### 4.1 核心原则：artifact-driven pipeline

每个阶段只做一件事：读上游文件，写下游文件。没有共享内存状态，没有隐式依赖。

```
project.yaml
     │
 ①  ├─> 01_dialogue/E{NN}.dialogue.json    标准化对白轨
     │
 ②  ├─> 02_signals/E{NN}.signals.json      高能点清单（规则层）
     │
 ③  ├─> 03_script/script.json      ★人工可编辑  节点 + 旁白 + 片段 + 音频指示
     │
 ④  └─> out/解说方案.md
         out/narration.txt
```

`out/` 下的两个文件是纯派生产物。改 `script.json` 重跑 ④ 即可，不重新调用 LLM。

这个形态换来四件事：

1. **断点续跑** — 改提示词只重跑 ③，字幕解析和信号检测不用重跑。v3 加入 ASR 后这一点价值更大。
2. **人工干预点明确** — `script.json` 是唯一的人工编辑面。半自动化就落在这一个文件上。
3. **GUI 是免费的** — v4 的 GUI 本质是 `script.json` 的可视化编辑器加一个「重跑下一阶段」按钮，引擎不用改。
4. **可测试** — 每个阶段的单测 fixture 是上游 JSON，不需要真视频。

### 4.2 目录结构

```
10min_anime/
├── pyproject.toml
├── src/tenmin/
│   ├── __init__.py
│   ├── cli.py                  Typer CLI
│   ├── config.py               pydantic-settings，API key 从环境变量读
│   ├── models.py               全部 artifact 的 pydantic 模型
│   ├── pipeline.py             阶段编排 + 断点续跑
│   ├── ingest/
│   │   ├── srt_parser.py       编码嗅探 + SRT/ASS 解析
│   │   ├── clean.py            清洗规则集
│   │   ├── credits.py          OP/ED/staff 区间识别
│   │   └── normalize.py        汇聚成 DialogueTrack
│   ├── signals/
│   │   ├── gaps.py             无字幕间隙检测
│   │   ├── density.py          字密度 + 密度突变
│   │   ├── punctuation.py      标点情绪强度
│   │   └── aggregate.py        合成高能点清单
│   ├── script/
│   │   ├── llm.py              LLMProvider 协议 + 实现
│   │   ├── single.py           单集模式
│   │   ├── season.py           整季模式 map-reduce
│   │   ├── budget.py           时长预算校验与重写触发
│   │   └── prompts/            提示词模板（.md 文件，可独立迭代）
│   │       ├── single_episode.md
│   │       └── examples/
│   │           └── saijo_e02.md    黄金样本对照表，few-shot + 质量基线
│   └── docgen/
│       ├── table.py            script.json → 解说方案.md
│       └── narration.py        script.json → narration.txt
├── docs/superpowers/specs/
├── tests/
│   ├── fixtures/
│   │   └── saijo_e02.srt       黄金样本字幕：《才女的侍從》第2集，405 行
│   └── ...
└── work/<show-slug>/           运行时产物
```

### 4.3 可插拔接口

用 `typing.Protocol`，不用继承。

v1 只实现 `LLMProvider`。另外两个是为后续版本预留的形态说明，v1 不写这些代码，也不定义 `BeatDraft` / `EpisodeContext` / `SynthResult` 这些类型。

```python
class LLMProvider(Protocol):                          # v1
    async def complete(self, system: str, user: str,
                       schema: type[BaseModel] | None = None) -> Any: ...

class ShotSelector(Protocol):                         # v5 预留
    def select(self, beat: BeatDraft, ctx: EpisodeContext) -> list[Clip]: ...

class TTSEngine(Protocol):                            # v2 预留
    async def synth(self, text: str, out: Path) -> SynthResult: ...
```

### 4.4 技术栈

- Python 3.12，uv 管理依赖
- Typer — CLI
- Pydantic v2 — 全部 artifact 都是 pydantic 模型，schema 即文档，`model_json_schema()` 直接喂给 LLM 做结构化输出
- pysrt 或自写解析器 — SRT 解析
- charset-normalizer — 编码嗅探
- opencc-python-reimplemented — 繁简转换
- 不用 MoviePy（v2 直调 ffmpeg，MoviePy 处理 60+ 片段慢且内存占用高）

## 5. 数据模型

### 5.1 project.yaml

```yaml
show: "才女的侍從"
slug: "saijo"
mode: single_episode          # single_episode | season
target_seconds: 240           # 单集 240，整季 600
locale:
  convert_traditional: true   # 繁转简
episodes:
  - number: 2
    srt: "srt/E02.srt"
    op_range: null            # 留空则自动检测；[153.5, 226.0] 可手动覆盖
    ed_range: null
glossary:                     # 术语表，修正专有名词
  友成伊月: 友成伊月
  此花雛子: 此花雛子
  天王寺美麗: 天王寺美丽
  傑斯電器: 杰斯电器
llm:
  provider: gemini
  model: gemini-2.5-pro
```

### 5.2 DialogueTrack

```python
class DialogueLine(BaseModel):
    idx: int                  # 原 SRT 序号，用作锚点
    start: float              # 秒
    end: float
    text: str                 # 清洗后
    raw: str                  # 清洗前，保留用于排查
    speaker: str | None       # 从 (角色名) 前缀提取
    kind: Literal["dialogue", "monologue", "screen_text", "credits", "noise"]
    suspect: bool             # 疑似含 OCR 噪声，LLM 需谨慎对待

class DialogueTrack(BaseModel):
    episode: int
    source: Literal["srt", "asr"]
    duration: float
    op_range: tuple[float, float] | None
    ed_range: tuple[float, float] | None
    lines: list[DialogueLine]
```

`kind` 的设计意图：不删除任何行，只打标。`credits` 和 `noise` 在喂给 LLM 时过滤掉，但保留在 artifact 里可追溯。

### 5.3 Signals

```python
class Highlight(BaseModel):
    start: float
    end: float
    strength: int             # 1-5
    triggers: list[str]        # ["gap:19.8s", "density:0.92", "llm:修罗场"]
    summary: str               # 一句话说明这里发生了什么
    anchor_lines: list[int]    # 相关的 SRT 行号

class SignalReport(BaseModel):
    episode: int
    silent_gaps: list[tuple[float, float]]
    median_char_rate: float           # 全集字密度中位数，作为基线
    highlights: list[Highlight]
```

### 5.4 Script

这是核心 artifact，直接对应对照表的每一列。

```python
class Clip(BaseModel):
    episode: int
    start: float
    end: float
    visual: str               # 「建议画面特征」列。v1 给人看，v5 当视觉检索 query
    anchor_lines: list[int]
    is_silent_highlight: bool # 是否命中无字幕演出高光，表格里标 ★

class Hold(BaseModel):
    """旁白让位、原声顶上来的留白。必须计入时长预算。"""
    at: float                 # 相对于本段旁白起点的秒数
    duration: float
    quote: str                # 要突出的原声台词
    note: str

class SfxCue(BaseModel):
    at: float
    cue: Literal["impact", "whoosh", "comedy", "suspense", "uplift"]
    note: str

class AudioDirection(BaseModel):
    original_audio: Literal["duck", "mute", "full"] = "duck"
    sfx: list[SfxCue] = []
    holds: list[Hold] = []

class Beat(BaseModel):
    id: str                   # "beat_01"
    label: str                # 「节点」列，如 "阶段三：钱包、女厕与金发死对头"
    role: Literal["hook", "act", "climax", "outro"]
    narration: str            # 「分段解说文案」列
    clips: list[Clip]         # 「原片截取时间戳」列，列表支持跨时间点拼接
    audio: AudioDirection     # 「剪辑与原声处理」列
    est_seconds: float         # 按语速估算的时长，含 holds

class Script(BaseModel):
    show: str
    mode: Literal["single_episode", "season"]
    episodes: list[int]
    target_seconds: float
    est_total_seconds: float
    beats: list[Beat]
```

`clips` 是列表，这是刻意的：一个节点经常需要把片中相隔十几分钟的两三个画面拼在一起（例如 Hook 需要同时用到 00:01 的抱抱和 00:18 的泡澡）。

## 6. 模块设计

### 6.1 ingest — 字幕解析与清洗

真实字幕的脏数据密度很高。以黄金样本《才女的侍從》第 2 集（405 行）为例，实测出现的问题类型：

| 行号 | 内容 | 问题 |
|---|---|---|
| 51, 52 | `河原正信 有賀史英` | OP staff 名单 |
| 53 | `《才女的侍從 在滿是高嶺之花的…》` | 标题卡 |
| 54 | `製作『才女のお世話』製作委員会©…` | 版权声明 |
| 401, 403, 404, 405 | `HE IYA Synergy SP J.C.STAFF…` / `作詞 前島亞美 作曲·編曲…` | ED staff |
| 41 | `80-08 浙谷339 就請您動用這個吧` | 车牌 OCR 混入台词 |
| 67 | `258 置換積分法 這裡居然在二年級春天就教了 分` | 屏幕文字 + 页码混入 |
| 105 | `這也是我家訂的規矩\nboo` | 噪声后缀 |
| 187 | `"傑斯電器\n00` | 残缺 + 噪声 |
| 291 | `じやがいも\n%` | 日文原文 + 符号 |
| 375 | `-` | 纯符号空行 |
| 56, 57 | `(第二集當侍從的第一天) 我是友成伊月` | 标题卡与台词同行 |
| 200, 201, 202 | `剛才的回答相當精彩\n(伊月) 她隨時都被旁人包圍` | 台词与内心独白双轨叠加 |

最后一类最危险：不拆行的话 LLM 会把两个人的话当成同一个人说的。

清洗规则集，按顺序执行：

1. **编码嗅探** — charset-normalizer 探测，统一转 UTF-8，剥 BOM
2. **格式标签剥离** — ASS/SSA 的 `{\...}`、HTML 的 `<i>` `<b>` `<font>`
3. **繁简转换** — opencc `t2s`，受 `locale.convert_traditional` 控制
4. **术语表替换** — 应用 `project.yaml` 的 glossary
5. **说话人前缀提取** — 匹配 `^\((?P<name>[^)]{1,8})\)\s*` 提取到 `speaker`
6. **双轨拆行** — 一行内含换行且第二段带 `(角色名)` 前缀时，拆成两条 `DialogueLine`，共享时间区间，第二条标 `kind="monologue"`
7. **纯噪声行标记** — `^[\s\-–—•·%0-9]+$` 匹配则 `kind="noise"`
8. **staff/credits 识别** — 见下节
9. **OCR 噪声标记** — 行首或行中出现与上下文语义无关的数字串、车牌格式、单独的 `00` / `%` / `boo` 等，标 `suspect=True`，**不删除**
10. **跨行同句合并** — 上一行末尾无终止标点、与下一行间隔 < 0.3 秒、且 speaker 相同，合并为一条。**⚠️ 实现修正：这条规则在管线里默认关闭**（`build_track(..., merge_lines=False)`）。在黄金样本上实测证伪：`speaker` 相同这个主闸门在无 `(角色名)` 前缀的字幕上退化成 `None == None` 恒真；相邻 cue 间隔中位数只有 0.001 秒（229/404 个 < 0.3s）；全集仅 9/405 行以终止标点结尾。三条判据同时失效，实测触发 126 次合并并把三个不同说话人黏成一条。句内折行在这类字幕里本来就以 cue 内部的 `\n` 表示，已由折叠步骤处理。函数保留为带开关的纯函数，供带说话人标注的字幕源（v3 ASR 输出）显式启用。

第 9 条刻意保守。全自动识别 OCR 噪声不可靠，会误删真台词。策略是标记而非删除，在喂 LLM 时附带说明「suspect 行可能含屏幕文字噪声，请按上下文取舍」。LLM 有全局上下文，比正则更擅长判断 `80-08 浙谷339` 是噪声。

### 6.2 credits — OP/ED 识别

判定一行是 staff/credits 的特征（任一命中）：

- 含 `©` 或 `(C)`
- 含 `製作` `製作委員会` `作詞` `作曲` `編曲` `協力` `フォント` `STAFF` `Studio` `作画`
- 被 `《》` `『』` `「」` 包裹且与剧名字符串高度重合（标题卡）
- 纯人名罗列：整行由 2 个以上「2-4 个 CJK 字符」的词以空白分隔组成，且不含任何标点、动词、助词

连续（间隔 < 8 秒）的 staff 行聚成一簇。取簇的时间跨度作为候选区间：

- 落在片头 60–300 秒窗口内、跨度 60–100 秒的簇 → `op_range`
- 落在最后 120 秒内的簇 → `ed_range`

黄金样本预期结果：

- `op_range = [153.486, 224.681]` — 第 51 行 `河原正信 有賀史英` 起至第 54 行版权声明结束，跨度 71.2 秒
- `ed_range = [1348.180, 1416.622]` — 第 401 行 `HE IYA Synergy SP J.C.STAFF…` 起至第 405 行结束

`project.yaml` 的 `op_range` / `ed_range` 若非空则直接覆盖自动检测结果。

### 6.3 signals — 高能点检测

这是本设计中最重要的发现。人工看片挑出的本集视觉高光，及其对应的无字幕间隙：

| 间隙区间 | 时长 | 实际内容 |
|---|---|---|
| 00:02:14 → 00:02:33 | 18.9s | 贵皇学院全景仰拍演出 |
| 00:22:08 → 00:22:28 | 19.8s | 修罗场后的定格拉远收尾 |
| 00:20:33 → 00:20:47 | 13.4s | 女仆静音无声凝视的长镜头 |
| 00:12:17 → 00:12:25 | 8.3s | 天王寺美丽登场亮相演出 |
| 00:21:57 → 00:22:03 | 5.5s | 被认出后的震惊定格 |
| 00:18:20 → 00:18:25 | —（无间隙） | 泳装登场定格 |
| 00:21:09 → 00:21:13 | 3.7s | 递药丸后的反应镜头 |

**本集视觉高光 6/7 落在无字幕间隙里。** 这不是巧合——动画的演出语法就是「该炫的时候不说话」。唯一的例外是 `00:18:20 → 00:18:25`（泳装定格）：实测该区段是连续对白（334 `好宽敞` / 335 `伊月` / 336 `一起泡澡吧`），最大间隔只有 1.7s，字幕轨里**没有**间隙，所以纯时间戳规则拿不到它。

`00:20:33 → 00:20:47` 这条早期记的是 `00:20:35 → 00:20:47 (11.6s)`，那是把 idx 375 的孤立破折号 `-`（`kind="noise"`，1235.067–1235.399）当成语音边界量出来的。正确跳过非语音行后，间隙是 `[1233.606, 1246.995]` = **13.389s**。

推论：「长无字幕间隙 = 演出高光」这条启发式规则，计算成本是零（纯时间戳算术），在黄金样本上召回率 6/7 ≈ 86%。它比音频能量曲线更便宜，也比 CLIP 视觉索引更准。

⚠️ **但这张表不是「所有 ≥3s 无字幕间隙」的全集。** 实测本集共有 **26** 个 ≥3s 间隙（强度分布 `{4: 2, 3: 6, 2: 18}`），上表那 6 个有间隙的高光散落在时长排名 **1 / 2 / 3 / 7 / 11 / 18**，中间夹着 4 个更长的非高光间隙（10.2s / 9.8s / 8.9s / 8.1s）。也就是说 `gaps.py` 是**候选检测器**而非高光选择器，任何时长阈值都无法把这 6 个从 26 个里单独筛出来；收敛到 ~7 个高光是 `aggregate.py` 叠加密度信号并聚簇之后的事。


因此 v1 的信号层由三条零成本规则构成：

**a) 无字幕间隙检测**（`gaps.py`）

排除 `op_range` / `ed_range` 与首行之前的片头黑场后，计算相邻 `kind in {dialogue, monologue}` 行之间的间隔。

| 间隙时长 | 强度 |
|---|---|
| ≥ 15s | 4 |
| 8–15s | 3 |
| 3–8s | 2 |

**b) 低字密度检测**（`density.py`）

`char_rate = len(text) / (end - start)`。取全集中位数 `median_char_rate` 作基线。

`char_rate < 0.4 × median` 且 `duration ≥ 2.0s` → 强度 3，标记为情绪爆发。

黄金样本验证：第 269 行 `才沒有染呢我`，6.589 秒 / 6 字 = 0.91 字/秒，是全集最低值，对应天王寺被问染发后炸毛的尖叫镜头。

**c) 字幕密度突变**（`density.py`）

30 秒滑窗内的总字数序列，计算一阶差分的 z-score。`|z| > 1.5` 时标记。骤增对应连珠炮式争吵，骤降对应无台词高潮段。

**d) LLM 语义标注**（在 `script` 阶段内完成，不单独调用）

让 LLM 在读字幕的同一次调用里，为候选高能点打语义标签：`反转` `打脸` `破防` `名台词` `福利` `告白` `吐槽` `新角色登场` `伏笔`。

`aggregate.py` 把 a/b/c 三路信号按时间聚类合并（间隔 < 2 秒的信号归为同一个 `Highlight`），最终强度取该聚类内**最高单路强度，每多命中一路独立信号 +1，上限 5**。例如某处同时命中 `gap:19.8s`（强度 4）、`density:0.92`（强度 3）、`llm:修罗场`（强度 3），最终强度 = 4 + 2 = 5（截断到 5）。

黄金样本上的预期输出（前 5 条）：

```
强度  时间戳               触发信号                             内容
5    00:22:02-00:22:28   density:0.92 + gap:19.8s + llm:修罗场   "我好想你" 哭扑
5    00:20:58-00:21:15   llm:黄色梗 + gap:3.7s                  "硬不起來的藥"
4    00:18:24-00:18:27   gap:4.7s + llm:福利                    "一起泡澡吧"
4    00:14:50-00:14:58   density:0.91(全集最低) + llm:破防        "才沒有染呢我"
4    00:16:13-00:16:22   llm:反差萌                             洋芋片投喂
```

### 6.4 script — 剧本生成

**单集模式**（v1 主路径）：一次 LLM 调用。输入 = 清洗后的完整对白轨 + `SignalReport` + few-shot 样例。输出 = `Script`，用 pydantic schema 做结构化输出约束。

单集字幕约 400 行、6000 字符，加上信号清单和 few-shot 也在 3 万 token 以内，无需分片。

**整季模式**（数据模型与 `project.yaml` 已支持，实现在 v1 之后。v1 遇到 `mode: season` 时报错退出，提示「整季模式尚未实现，请使用 mode: single_episode」）：

1. `stage1` — 每集独立生成 `EpisodeDigest`（主线推进、关键转折、高能点 top-5、新增角色），12 集并行调用
2. `stage2` — 把 12 份 digest 一起喂进去，产出 `SeasonArc`：整季叙事骨架、15–25 个节点的划分与顺序、每个节点的时长预算
3. `stage3` — 按骨架逐节点定稿旁白，每次调用附带该节点涉及集数的完整字幕片段

**提示词组织**：`prompts/` 下是独立的 `.md` 文件，不硬编码在 Python 里。这样调提示词不需要改代码，也便于 diff。

`prompts/single_episode.md` 的核心要素：

- 角色设定：番剧解说号的写手，目标是让没看过的人想看
- 节点结构：`Hook 开场` + 若干 `阶段N：小标题` + `收尾`
- 每个节点必须给出 2–5 个 `clips`，其中至少一个来自无字幕高光区间
- `visual` 字段用「A ➔ B ➔ C」的形式描述画面序列
- `audio.holds` 用于把原片金句留出来给观众听，每期 3–6 处
- 明确禁止：剧透结局、复述字幕原文、使用「本片讲述了」这类总结腔
- suspect 行的处理说明
- **few-shot 样例**：《才女的侍從》第 2 集的完整对照表，已确认为质量模板。单一存放位置 `src/tenmin/script/prompts/examples/saijo_e02.md`，提示词和测试都引用这一份，不做副本

**时长预算**（`budget.py`）：

- 中文旁白语速基线 4.5 字/秒（Edge-TTS 中文女声默认速率的实测近似值）
- `est_seconds = len(narration) / 4.5 + sum(hold.duration for hold in holds)`
- holds 必须计入。5 个节点各留白 1 秒就是 10 秒，在 600 秒目标里占 1.7%，不算会飘
- 校验 `|est_total - target| / target > 0.12` 时，触发一轮重写：把超出/不足的量和具体节点的当前字数一起回传给 LLM，要求精简或扩写指定节点，其余节点保持不变

### 6.5 docgen — 文档生成

纯函数，无 LLM，无网络。

`table.py` — `Script` 渲染成 Markdown 表格，列顺序固定为：节点 / 原片截取时间戳 / 建议画面特征 / 分段解说文案 / 剪辑与原声处理。

- 时间戳格式 `HH:MM:SS.mmm`，多个 clip 之间用 `接` 连接并换行
- `is_silent_highlight` 为真的 clip 在时间戳后加 `★`
- 表格末尾附图例：`★ = 该片段命中无字幕演出高光区间，纯字幕方案取不到`
- 表头上方一行元信息：剧名、集数、旁白字数、估算时长

`narration.py` — 按 beat 顺序拼接 `narration`，段间空行分隔，不含任何标记，可直接全选粘贴进 TTS 工具。

## 7. CLI

```bash
tenmin init <slug>                    # 生成 project.yaml 骨架
tenmin run <slug>                     # 跑完整流程
tenmin run <slug> --from script       # 从指定阶段重跑
tenmin run <slug> --only docgen       # 只跑一个阶段
tenmin inspect <slug> --episode 2     # 打印清洗结果与信号报告，用于排查
```

阶段名：`ingest` / `signals` / `script` / `docgen`。

`run` 默认跳过已有产物的阶段（按文件 mtime 与上游比较），`--force` 强制重跑。

## 8. 测试策略

黄金样本《才女的侍從》第 2 集存为 `tests/fixtures/saijo_e02.srt`，405 行，时长 00:00:07–00:23:36。

**确定性阶段做严格断言：**

`ingest`
- 第 51、52、53、54、401、403、404、405 行 `kind == "credits"`
- 第 375 行 `kind == "noise"`
- 第 41、67、105、187、291 行 `suspect == True`
- 第 200 行拆成两条，第二条 `speaker == "伊月"` 且 `kind == "monologue"`
- 第 56、57 行的 `(第二集當侍從的第一天)` 前缀被剥离

`credits`
- `op_range` 起点 == 153.486，跨度在 60–100 秒区间内
- `ed_range` 起点 == 1348.180

`signals`
- 检出 7 个 ≥ 3 秒的无字幕间隙，起止时间误差 < 10ms
- 最长间隙 19.8 秒（00:22:08.367 → 00:22:28.180）
- 第 269 行的 `char_rate` 是全集最小值
- 强度 ≥ 4 的高能点里包含 00:22:02、00:20:58、00:18:24、00:14:50 四个时间点

`docgen`
- 给定手写的 `script.json` fixture，渲染出的 Markdown 逐字符匹配预期输出
- `narration.txt` 的字数等于所有 beat 的 narration 字数之和加上分隔空行

**LLM 阶段不做输出断言**，只做：
- schema 校验（返回值能被 `Script` 解析）
- 不变量校验（beat 数 ≥ 3、每个 beat 至少 1 个 clip、所有 clip 的时间戳落在字幕时长内且不落在 OP/ED 区间、`est_total` 在 target 的 ±12% 内）
- 快照留档，人工比对质量

## 9. 风险与应对

| 风险 | 应对 |
|---|---|
| LLM 引用的 `anchor_lines` 与 `clips` 时间戳不一致 | 后处理校验：用 `anchor_lines` 反查字幕时间，与 LLM 给的 `start`/`end` 比对，偏差 > 5 秒则以字幕时间为准并记 warning |
| LLM 编造不存在的时间戳 | 校验所有 clip 落在 `[0, duration]` 内且不在 OP/ED 区间，越界则丢弃该 clip；一个 beat 的 clip 全被丢弃时触发重试 |
| OCR 噪声被当成剧情写进旁白 | suspect 标记 + 提示词明确说明；黄金样本回归测试盯住 `80-08 浙谷339` 不出现在输出里 |
| 繁简转换破坏专有名词 | glossary 在繁简转换之后执行，可精确覆盖 |
| 时长估算与真实 TTS 偏差大 | v1 只能估算。v2 接上 Edge-TTS 后用实测时长回填，并据此校准 4.5 字/秒这个基线常数 |
| 提示词迭代导致质量回退 | 黄金样本的输出快照入 git，每次改提示词后 diff 人工过一遍 |
| 无字幕间隙规则在其他番上失效 | 这条规则基于动画演出语法，泛化性应该好，但只在一集上验证过。实现后立刻用 2–3 部不同类型的番（日常/战斗/悬疑）复验，若失效则把音频能量曲线提前到 v1.5 |

## 10. 附录：黄金样本的目标输出

`src/tenmin/script/prompts/examples/saijo_e02.md` 保存已确认的对照表全文。它同时承担两个角色：单集模式提示词的 few-shot 样例，以及质量基线。要点摘录：

- 7 个节点：Hook 开场 / 阶段一：入职即地狱 / 阶段二：假平民认证局 / 阶段三：钱包、女厕与金发死对头 / 阶段四：投喂式遛大小姐 / 阶段五：泡澡、监视与真心话 / 收尾：修罗场引爆
- 旁白 1,180 字，估算 4 分 22 秒
- 8 处 `holds`，全部指向原片金句（`就是會讓人硬不起來的藥` / `兩個叛徒` / `才沒有染呢我` / `給我吃` / `從一開始` 等）
- 6 个 clip 命中无字幕演出高光区间
