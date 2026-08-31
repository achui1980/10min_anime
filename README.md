# 10 分钟看番剧 · tenmin

把番剧字幕变成解说方案。输入 SRT，输出「分段文案与剪辑时间轴对照表」+ 配音纯文本。

**v1 不碰视频文件。** 没有 ASR、没有 TTS、没有渲染。设计文档见
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

## 使用

```bash
uv run tenmin init saijo               # 创建 work/saijo/project.yaml 与 srt/
cp 你的字幕.srt work/saijo/srt/E02.srt  # 放字幕
$EDITOR work/saijo/project.yaml        # 填 show / episodes
uv run tenmin run saijo                # 跑全链路
```

产物：

- `work/saijo/out/解说方案.md` — 五列对照表，★ 标记纯字幕方案取不到的视觉高光
- `work/saijo/out/narration.txt` — 合并配音纯文本，直接丢给配音
- `work/saijo/03_script/script.json` — **唯一人工编辑面**，改完重跑 docgen 即可

```bash
uv run tenmin run saijo --only docgen --force   # 改完 script.json 重出文档，不调 LLM
uv run tenmin inspect saijo --episode 1         # 看无字幕间隙与高能点
uv run tenmin inspect saijo --episode 1 --suspect  # 看被标记为疑似 OCR 噪声的行
uv run tenmin run saijo --from signals          # 从指定阶段重跑
```

## 阶段与产物

| 阶段 | 产物 | 是否调 LLM |
|---|---|---|
| ingest | `01_dialogue/E{NN}.dialogue.json` | 否 |
| signals | `02_signals/E{NN}.signals.json` | 否 |
| script | `03_script/script.json` | **是** |
| docgen | `out/解说方案.md`、`out/narration.txt` | 否 |

默认按 mtime 跳过已是最新的阶段，`--force` 强制重跑。

## 测试

```bash
uv run pytest -q                              # 默认：全部离线，不联网
uv run pytest -m llm                          # 真调 LLM 出快照（需 API key）
uv run pytest -m generalize                   # 泛化复验（需自备 SRT）
```

## 路线图

- **v1**（本版）SRT → 对照表 + 配音文本
- **v2** Edge-TTS 配音 + ffmpeg 切片拼接 + 混音 + 烧硬字幕 → 1920x1080 mp4
- **v3** Faster-Whisper ASR，支持生肉
- **v4** 本地 Web GUI（`script.json` 可视化编辑器）
- **v5** PySceneDetect + CLIP 视觉索引，整季 12 集压到 10 分钟
