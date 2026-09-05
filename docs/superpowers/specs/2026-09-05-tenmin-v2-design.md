# 10分钟看番剧 v2 设计：把解说方案渲染成成品视频

## 1. 目标与边界

v1 的产出是一份**给人看的解说方案**（`解说方案.md` + `narration.txt` + `script.json`），人拿着它去剪辑软件里手工干活。v2 把这最后一步自动化：

**输入** `script.json`（v1 产物）+ 源视频文件
**输出** 一个可以直接上传的成品 mp4——画面已按 clip 剪好拼好、旁白已配音、原声已压低垫底、旁白字幕已烧进画面。

### v2 做的四件事

1. **配音**：把 `beat.narration` 用 Edge-TTS 合成成真实音频
2. **剪辑**：按 `beat.clips` 从源视频切片并拼接
3. **混音**：旁白在上，原声 ducking 垫底
4. **烧字幕**：旁白文本烧进画面

### v2 明确不做

| 不做的事 | 理由 |
|---|---|
| 音效（`sfx`） | 需要外部音效素材，引入版权问题。`script.json` 的 `sfx` 字段保留但本版不消费 |
| BGM | 同上，且 v1 spec 从未提及 |
| 原片对白字幕 | 解说视频的字幕是**旁白字幕**（观众跟着读解说），不是原片对白 |
| 精确金句同步 | 见 §3.3 |
| 视觉识别选镜头 | v5 |
| GUI | v4 |

---

## 2. 前提条件

### 2.1 ffmpeg 必须编入 libass

实测本机 ffmpeg 9.0.1（Homebrew 精简构建）的 configuration：

```
--enable-gpl --enable-libsvtav1 --enable-libopus --enable-libx264 --enable-libmp3lame
--enable-libdav1d --enable-libvmaf --enable-libvpx --enable-libx265 --enable-openssl
--enable-videotoolbox --enable-audiotoolbox --enable-neon
```

**没有 `--enable-libass`，也没有 `--enable-libfreetype`。**所以 `subtitles` 滤镜和 `drawtext` 滤镜都不存在，当前这个构建**烧不了任何文字**。切片、拼接、`sidechaincompress`、libx264、`h264_videotoolbox` 都可用。

v2 的实现前必须重装带 libass 的 ffmpeg。程序侧的对策见 §6.1：开跑前检查，不通过立即退出。

### 2.2 源视频

`project.yaml` 的 `episodes[]` 需新增 `video:` 字段指向视频文件。v1 只需要 SRT，v2 必须知道视频在哪。

黄金样本：`[LoliHouse] Futsutsuka na Akujo dewa Gozaimasu ga - 02 [WebRip 1080p HEVC-10bit AAC SRTx2].mkv`。`tests/fixtures/generalize/akujo_e02.srt` 就是从这个文件用 `ffmpeg -map 0:2 -c:s srt` 抽出来的，所以字幕时间戳与视频天然对齐。

源是 **1080p HEVC-10bit**，任意时间点切割无法 `-c copy`（关键帧对不齐），必须重编码。

---

## 3. 时间轴重算

这是 v2 的技术核心。v1 用 4.5 字/秒估算旁白时长，真实 TTS 长度一定不同，所以画面和声音必然对不齐。

### 3.1 实测出的主要矛盾

那份真实 `script.json`：**clip 总时长 626 秒，旁白估算只有 253 秒**。素材比旁白多 2.5 倍。所以主要矛盾**不是「旁白说不完」而是「画面放不完」**。

### 3.2 对齐规则：clip.start 硬，时长按比例缩放

```
本 beat 音频总时长 = 各 chunk 真实 TTS 时长之和 + 各 hold.duration 之和
ratio = 本 beat 音频总时长 / 本 beat 所有 clip 的原始总时长
每个 clip：source_start 不动，时长 × ratio
```

分子**包含留白的静音时长**。因为一个 beat 的画面总时长必须等于它的音频总时长（含停顿），否则留白期间就没画面可放。

- `ratio < 1`（常见情况，实测约 0.4）：每个 clip 按比例变短，**所有 clip 都出场**
- `ratio > 1`（旁白比素材长）：每个 clip 按比例变长，即「旁白没说完就接着往后放原片」
- 延长撞到源片尾：钳到片长，记 warning

**为什么按比例缩放而不是「顺序放到旁白结束就停」**：后者会让排在后面的 clip 完全消失。LLM 挑那几个镜头有叙事意图（Hook 的钩子镜头、climax 的情绪落点），砍掉等于毁了段落结构。按比例缩放保证每个镜头都出场，只是节奏更快。

`clip.start` 之所以是硬的：它经过 v1 `validate.py` 的 anchor 校正，是有依据的镜头起点；而 `clip.end` 是 LLM 猜的，没有依据。

### 3.3 hold（留白）处理

`hold.at` 是 LLM 按 4.5 字/秒估出来的位置，真实 TTS 一定漂移。做法：

1. 把 `beat.narration` 按句末标点切成句子
2. 按累计估算时长，找出离每个 `hold.at` 最近的句子边界
3. **每个 chunk 单独 TTS**，chunk 之间插入 `hold.duration` 长度的静音
4. 静音期间原声从 ducking 抬到全开

好处：每个 chunk 的真实时长直接量出来，不依赖 Edge-TTS 的 word-boundary API；每个 chunk 是独立落盘的产物，坏了只补那一个。

**取舍（明确记录）**：留白期间播的是「那一刻恰好在放的原声」，**不是金句本身的原声**。要精确对上金句，得让画面跳到金句的源时间戳，那是另一个数量级的复杂度，而且会破坏 clip 的连续性。所以 v2 的 hold 是**节奏装置**，不是精确金句同步。

v1 `validate.py` 的留白金句校验已经在推 LLM 把 hold 金句放在本 beat 的 clip 时间窗内（实测那次真跑 7/7 全部落在窗内），所以实际效果通常是对的，但不保证。

---

## 4. 四阶段流水线

按「贵不贵」分阶段，而不是按「像不像剪辑软件」分。

| 阶段 | 产物 | 成本 | 改动时重跑范围 |
|---|---|---|---|
| `voice` | `04_voice/E{NN}/chunk_*.mp3` + `E{NN}.voice.json` | Edge-TTS 网络调用 | 换音色 / 改文案 |
| `timeline` | `05_timeline/E{NN}.timeline.json` + `E{NN}.ass` | 纯计算，秒级 | 改对齐规则 |
| `audio` | `06_audio/E{NN}.mixed.m4a` | 音频编码，很便宜 | 改 ducking 参数 |
| `render` | `07_render/E{NN}.mp4` | **唯一一次视频编码**，几分钟 | 改字幕样式 |

关键决定：**视频只编码一次**。`render` 阶段的单个 ffmpeg 调用同时做 trim、concat、烧字幕、挂音轨。多编一次就多一次画质损失和几分钟等待。

代价：烧字幕出问题时要连着 trim/concat 一起重跑那几分钟。接受。

`timeline.json` 扮演 v1 里 `script.json` 那个角色——**唯一的人工可编辑面**。手改它再跑 `--from audio` 就能重出片。

### 被排除的两条路

- **每段切成独立文件再拼**：编码两次（切一次、拼一次），HEVC-10bit 任意点切割没法 `-c copy` 所以躲不掉。画质和时间都亏，且 18 个 1080p 片段落盘几个 GB。
- **一条巨型 `filter_complex` 一次成型**：上百个节点的 filtergraph，一处写错整条重跑，没法断点续跑。改一下字幕字号就要重新跑 TTS。和 v1 建立的分阶段模式完全冲突。

MoviePy 在 v1 spec 里已排除（自带编解码假设、性能差、HEVC-10bit 支持不稳）。

---

## 5. 模块与接口

### 5.1 新增 `render/` 包

沿用 v1「小文件、单一职责、纯函数尽量多」的路子：

```
src/tenmin/render/
  __init__.py
  tts.py         TTSEngine Protocol + EdgeTTSEngine
  chunks.py      旁白切句 + hold 定位              纯函数
  timeline.py    时间轴重算                        纯函数
  subtitles.py   生成 ASS 字幕                     纯函数
  ffmpeg.py      子进程封装：run / probe / has_filter / has_encoder
  audio.py       混音（调 ffmpeg）
  video.py       trim + concat + 烧字幕 + 挂音轨（调 ffmpeg）
```

### 5.2 TTS 藏在 Protocol 后面

与 v1 的 `LLMProvider` 完全同构：

```python
@runtime_checkable
class TTSEngine(Protocol):
    async def synthesize(self, text: str, out_path: Path) -> float:
        """合成一段语音，返回真实时长（秒）。"""
        ...
```

`EdgeTTSEngine(voice, rate)` 是唯一实现。

这个决定让 `chunks` / `timeline` / `subtitles` 三个纯函数模块 + 混音逻辑**全部能离线测试**——测试用 `FakeTTSEngine` 写一段指定长度的静音，不联网。v1 就是靠 `FakeProvider` 才做到 392 个测试里只有 1 个需要网络，v2 照搬这个结构。

### 5.3 数据模型（进 `models.py`）

| 模型 | 字段 |
|---|---|
| `VoiceChunk` | `beat_id: str` / `index: int` / `text: str` / `path: str` / `duration: float`（真实测量值） / `hold_after: float = 0.0`（该 chunk 之后的静音秒数） |
| `VoiceTrack` | `episode: int` / `chunks: list[VoiceChunk]` / `total_seconds: float` |
| `TimelineSegment` | `beat_id: str` / `source_start: float` / `source_end: float` / `timeline_start: float` / `timeline_end: float` |
| `SubtitleCue` | `start: float` / `end: float` / `text: str` |
| `Timeline` | `episode: int` / `segments: list[TimelineSegment]` / `subtitles: list[SubtitleCue]` / `narration_offsets: list[float]`（每个 chunk 在总时间轴上的落点） / `total_seconds: float` |

### 5.4 `project.yaml` 新增

```yaml
episodes:
  - number: 2
    srt: srt/E02.srt
    video: /Volumes/portz_extension_HD/Downloads/[LoliHouse] Futsutsuka na Akujo dewa Gozaimasu ga - 02 [WebRip 1080p HEVC-10bit AAC SRTx2].mkv

render:
  voice: zh-CN-YunxiNeural      # Edge-TTS 音色，可配不写死
  rate: "+0%"                    # Edge-TTS 语速
  video_encoder: libx264         # 或 h264_videotoolbox
  duck_db: -12.0                 # 原声压低多少 dB
  font_size: 48
```

音色默认云希（年轻男声、轻快、带调侃感，B 站番剧解说最常见）。

### 5.5 pipeline 扩展

`STAGES` 从 4 个扩到 8 个：

```python
STAGES = ["ingest", "signals", "script", "docgen", "voice", "timeline", "audio", "render"]
```

前 4 个阶段一字不改。`--from voice` 就能只跑 v2 部分——这正是 v1 那套 mtime 跳过机制现在的回报。

### 5.6 产物目录

```
work/<slug>/
  01_dialogue/E02.dialogue.json
  02_signals/E02.signals.json
  03_script/script.json              ★ v1 的人工编辑面
  out/解说方案.md
  out/narration.txt
  04_voice/E02/chunk_001.mp3 …
  04_voice/E02.voice.json
  05_timeline/E02.timeline.json      ★ v2 的人工编辑面
  05_timeline/E02.ass
  06_audio/E02.mixed.m4a
  07_render/E02.mp4                  ← 成品
```

---

## 6. 错误处理

原则：v2 每一步可能耗几分钟，所以**「早失败」比「能恢复」更重要**。

### 6.1 前置检查（开跑前一次性做完，任一不通过立即退出）

- `has_filter("subtitles")` → 不在就报「你的 ffmpeg 没编 libass，请 `brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass` 重装」。**绝不能跑完三分钟 TTS 才在最后一步炸。**
- `has_encoder(render.video_encoder)` → 不在就立即报。
- 源视频文件存在，且 `probe_duration` 能拿到时长。

### 6.2 运行期

| 情况 | 处理 |
|---|---|
| Edge-TTS 抖动 | 每个 chunk 独立重试 3 次；失败则报出「哪个 beat 的第几个 chunk、原文是什么」。chunk 独立落盘，重跑只补缺的那几个 |
| clip 延长撞源片尾 | 钳到片长 + warning，不报错 |
| segment 越界源片长 | 钳制 + warning |
| ffmpeg 非零退出 | 把 stderr **末尾 30 行**放进异常消息。ffmpeg 的真实错误永远在 stderr 尾部，只报 `exit 1` 等于没报 |

---

## 7. 测试策略

照搬 v1 那套，因为它验证过有效（392 passed，只有 1 个需要网络）。

| 层 | 怎么测 | 严格程度 |
|---|---|---|
| `chunks` / `timeline` / `subtitles` | 手造 `Script` + `VoiceTrack`，纯函数 | 严格断言；ASS 输出逐字符匹配 |
| TTS | `FakeTTSEngine` 写指定长度静音 | 不联网 |
| `audio.py` / `video.py` | **断言生成的 ffmpeg 命令行参数**，不真跑 | 逐参数匹配 |
| 黄金样本 | 用手上那份真 `script.json`（6 beats / 18 clips / 626 秒素材 / 253 秒旁白）做 timeline 重算断言 | 实测数字 |
| 端到端 | `@pytest.mark.render`，从 mkv 切 30 秒真跑全链路 | 无视频或无 libass 则 skip |

**ffmpeg 调用层测的是「命令行拼对了没」而不是「ffmpeg 干对了没」。**后者是 ffmpeg 自己的责任，我们测不了也不该测；前者才是我们会写错的地方。这跟 v1 里 `test_table.py` 用逐字符匹配测 markdown 渲染是同一个思路。

---

## 8. 已知局限

1. **留白不是精确金句同步**（§3.3）。留白期间播的是那一刻恰好在放的原声。
2. **不做音效**。`script.json` 的 `sfx` 字段被忽略。
3. **`clip.end` 会被覆盖**。LLM 给的结束时间只用来算比例，不直接采用。
4. **旁白字幕的断句沿用 TTS chunk 边界**，一个 chunk 一条字幕，字幕起止就是该 chunk 在总时间轴上的起止。**留白期间没有字幕**（那段没有旁白，屏幕上是干净画面 + 原声）。若某个 chunk 很长，字幕会在屏幕上停很久——v2 不做二次断句。
5. **只支持单集**。`mode: season` 在 v1 就是直接报错，v2 不改。

---

## 9. 版本路线图（更新）

- **v1（已完成）**：SRT → 解说方案对照表 + 配音文本 + `script.json`
- **v2（本设计）**：`script.json` + 源视频 → 成品 mp4（配音 / 剪辑 / 混音 / 烧字幕）
- **v3**：Faster-Whisper ASR，支持没有字幕的生肉
- **v4**：本地 Web GUI，`script.json` 与 `timeline.json` 的可视化编辑器
- **v5**：PySceneDetect + CLIP 视觉索引，用画面而不是字幕间隙选镜头
