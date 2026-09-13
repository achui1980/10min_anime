# 10 分钟看番剧 · tenmin

把番剧字幕 + 视频变成解说成片。输入 SRT 与源片，输出「分段文案与剪辑时间轴对照表」、
配音纯文本，以及配好音、烧好硬字幕的 mp4。

**还没有 ASR**，所以只吃自带字幕的片源（生肉见下面的路线图 v3）。设计文档见
`docs/superpowers/specs/2026-08-31-10min-anime-design.md`。

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
`--srt`/`--video` 必须一起传，且必须同时带 `--episode`。

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
```

只跑 v2 渲染部分（前 4 个阶段的产物照旧复用）：

```bash
uv run tenmin run akujo2 --from voice
```

手改过 `05_timeline/E02.timeline.json` 后只重新出片：

```bash
uv run tenmin run akujo2 --from audio --force
```

## 阶段与产物

| 阶段 | 产物 | 是否调 LLM |
|---|---|---|
| ingest | `01_dialogue/E{NN}.dialogue.json` | 否 |
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
```

## 路线图

- **v1**（已完成）SRT → 对照表 + 配音文本
- **v2**（本版）Edge-TTS 配音 + ffmpeg 切片拼接 + 混音 + 烧硬字幕 → 1920x1080 mp4
- **v3** Faster-Whisper ASR，支持生肉
- **v4** 本地 Web GUI（`script.json` 可视化编辑器）
- **v5** PySceneDetect + CLIP 视觉索引，整季 12 集压到 10 分钟
