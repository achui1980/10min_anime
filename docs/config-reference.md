# project.yaml 配置参考

本文档逐字段说明 `work/<slug>/project.yaml` 支持的全部配置项。权威来源是
`src/tenmin/config.py`（`ProjectConfig` 及其子模型），本文档所有行号引用均指向该文件。
若本文档与代码不一致，以代码为准；改动配置字段后应同步更新本文档。

## 目录

1. [顶层字段](#1-顶层字段-projectconfigconfigpy441-454)
2. [episodes](#2-episodes-episodeconfigconfigpy57-69)
3. [llm](#3-llm-llmconfigconfigpy76-14314-字段)
4. [ingest](#4-ingest-ingestconfigconfigpy146-1563-字段)
5. [credits](#5-credits-creditsconfigconfigpy159-24718-字段)
6. [signals](#6-signals-signalsconfigconfigpy250-27111-字段)
7. [validate_script](#7-validate_script-yaml字段名如此类名是validateconfigconfigpy274-3126-字段)
8. [render](#8-render-renderconfigconfigpy315-43828-字段)
9. [环境变量（.env）](#补充环境变量env走settings类-configpy517-524)
10. [全量配置示例](#9-全量配置示例所有字段含默认值)

---

## 1. 顶层字段（`ProjectConfig`，config.py:441-454）

| 字段 | 默认值 | 说明 |
|---|---|---|
| `show` | 必填，无默认 | 剧名。**必须是真实简体剧名**——`credits.title_overlap_threshold` 靠它识别片头/片尾的标题卡（书名号内文本与它的字符重合率） |
| `slug` | 必填 | 项目标识，对应 `work/<slug>/` 目录 |
| `mode` | `single_episode` | 另一选项 `season`（整季模式）**类型上允许但运行期直接报错**：`pipeline.py:749-750` 第一行 `raise NotImplementedError`。填了会一跑就炸，不会在加载期拦截 |
| `target_seconds` | `240.0`（必须 >0） | 目标解说时长（秒），script 阶段的时长预算基准 |
| `locale.convert_traditional` | `true` | 是否把繁体转简体（ingest 阶段） |
| `episodes` | `[]` | 见第 2 节 |
| `glossary` | `{}` | 术语表，`dict[str, str]`，进 ingest 清洗和 script 提示词 |
| `llm` / `ingest` / `credits` / `signals` / `validate_script` / `render` | 各自的 Config() 默认实例 | 见下方各节 |

---

## 2. episodes（`EpisodeConfig`，config.py:57-69）

每一集一项：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `number` | 必填 | 集数 |
| `srt` | 必填 | 字幕路径，相对 `project.yaml` 所在目录解析（`config.py:481-485`），绝对路径原样使用 |
| `video` | `None` | 视频路径，同上解析规则；渲染阶段前必须补上，否则 `video_path()` 报错（config.py:487-496） |
| `op_range` | `None` | 逐集手填 OP 区间 `(start, end)`，**三级回退里最高优先级** |
| `ed_range` | `None` | 逐集手填 ED 区间 `(start, end)` |

校验（`_validate_credit_range`, config.py:21-34）：`start`/`end` 都不能为负，且 `start < end`。写反的区间不会报错但下游 `intervals.subtract` 会静默当空集处理，等于没填，只能靠肉眼核对成片才能发现。

---

## 3. llm（`LLMConfig`，config.py:76-143，14 字段）

| 字段 | 默认值 | 说明 |
|---|---|---|
| `provider` | `"gemini"` | 可选 `gemini`/`minimax`/`openai_compatible` |
| `model` | `"gemini-3.6-flash"` | 模型名 |
| `base_url` | `None` | `openai_compatible` provider 用 |
| `thinking` | `"disabled"` | 可选 `adaptive`/`disabled`。MiniMax-M3 默认打开深度思考会先吐一段被直接丢弃的 `<think>` 块（纯浪费耗时），默认关掉 |
| `temperature` | `None` | 不传等于用服务端默认值 |
| `max_output_tokens` | `None` | 同上 |
| `timeout_seconds` | `120.0` | httpx write 超时：把 ~35k 字符 prompt 推上去的上限 |
| `read_timeout_seconds` | `120.0` | **相邻两个 SSE chunk 之间**最多等多久，不是整段生成时长；每个 chunk 都会刷新计时 |
| `total_timeout_seconds` | `1200.0` | 单次 HTTP 请求整体截止。实测 MiniMax-M3 处理 ~35k 字符需 561 秒，留两倍余量 |
| `max_attempts` | `3` | schema 校验失败的**总**尝试次数（含首发）= 首发 + 2 次自修复轮 |
| `transport_max_attempts` | `4` | 429/5xx/连接失败的传输层重试总次数（含首发） |
| `budget_tolerance` | `0.12` | 时长预算容差 |
| `validation_retries` | `1` | 语义校验失败重试次数（**不含**首发） |
| `budget_rewrite_rounds` | `1` | 时长预算返工轮数，每轮一次完整 LLM 调用 |
| `script_concurrency` | `1` | 批量模式下 script 阶段的预取并发度。默认 1 有三条硬依据（config.py:126-142）：①默认 provider(gemini) 走 SDK 自己的重试，没有本项目传输层退避，并发容易撞 429 导致整批失败；②并发失败会白花已发出的 token；③`run_audio`/`run_render` 是同步 ffmpeg 调用会堵住事件循环，在飞的 LLM 流收不到 chunk（当前单集约 35 秒 < 120 秒超时还够用，但源片更长/机器更慢时是隐患）。`openai_compatible`/`minimax` 有完整传输层退避，可调到 2-4 |

---

## 4. ingest（`IngestConfig`，config.py:146-156，3 字段）

| 字段 | 默认值 | 说明 |
|---|---|---|
| `merge_max_gap` | `0.3`（秒） | 续行合并的最大时间间隔 |
| `merge_max_chars` | `40` | 续行合并的最大字符数 |
| `merge_max_line_seconds` | `4.0`（秒） | 单行时长 ≥ 此值视为完整句（往往是拖长音），不再往后合并 |

---

## 5. credits（`CreditsConfig`，config.py:159-247，18 字段）

全项目最复杂模块。OP/ED/staff 行识别走**三级回退**：`EpisodeConfig.op_range/ed_range`（逐集手填，最高优先级）→ 本节的 `default_op_range`/`default_ed_range`（项目级默认）→ `find_credit_ranges` 启发式聚簇推断（最后兜底，不可靠）。手填命中时会直接收紧识别窗（区间本身 + `manual_window_margin` 余量），完全绕开下面的盲窗字段。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `op_search_start` | `30.0`（秒） | 聚簇法找 OP 时，簇起点候选区间下界 |
| `op_search_end` | `300.0`（秒） | 候选区间上界 |
| `credit_head_window` | `300.0`（秒） | `in_credit_window` 的片头盲窗上界（历史上跟 `op_search_end` 共用同一数值，现已拆开独立可调） |
| `op_span_min` | `40.0`（秒） | 合格 OP 簇的最短跨度 |
| `op_span_max` | `120.0`（秒） | 合格 OP 簇的最长跨度 |
| `ed_cluster_tail_seconds` | `120.0`（秒） | 簇起点距片尾多少秒以内才算 ED 候选 |
| `ed_keyword_window_seconds` | `80.0`（秒） | `in_credit_window` 的片尾盲窗，刻意比 `ed_cluster_tail_seconds` 窄（真台词与 ED staff 行之间留余量） |
| `credit_window_max_ratio` | `0.5` | 片头窗+片尾窗覆盖率不能超过片长的这个比例，**只约束启发式盲窗，不约束手填区间** |
| `cluster_max_gap` | `35.0`（秒） | 相邻 credits 行间隔不超过它就并进同一簇 |
| `op_min_silent_span` | `60.0`（秒） | 静区兜底判定 OP 的最短静默时长 |
| `op_max_silent_span` | `120.0`（秒） | 最长静默时长（防止把无对白过场误判成 OP） |
| `title_overlap_threshold` | `0.6` | 书名号内文本与 `show` 字段的字符重合率门槛（识别标题卡） |
| `title_card_max_len` | `24` | 标题卡最大字符数 |
| `name_list_min_cjk` | `6` | 纯人名罗列识别的 CJK 字数门槛 |
| `name_list_many_segments` | `4` | 切分段数达到此值可放宽字数门槛 |
| `name_list_many_min_cjk` | `4` | 放宽后的 CJK 字数门槛 |
| `latin_ratio_threshold` | `0.6` | 拉丁字母占比门槛（识别英文 staff 行） |
| `latin_min_len` | `6` | 上述规则生效所需的最小非空白长度 |
| `default_op_range` | `None` | 项目级默认 OP 区间 `(start, end)` |
| `default_ed_range` | `None` | 项目级默认 ED 区间，**终点可写 `null` 表示“到片尾”**（因为片长逐集不同，ED 起点是稳定结构） |
| `manual_window_margin` | `5.0`（秒） | 手填区间两侧的余量宽度 |

实测收益（config.py:214-226）：手填模式接住 3 条落在盲窗外的 staff 行，救回 5 条落在盲窗内被误杀的真台词；代价是填错会在**你填的区间内**误判（比盲窗误判范围小得多，且是显式声明而非猜测）。

---

## 6. signals（`SignalsConfig`，config.py:250-271，11 字段）

| 字段 | 默认值 | 说明 |
|---|---|---|
| `min_gap_seconds` | `3.0`（秒） | 无字幕间隙成为“信号”的最小门槛（来自对《才女的侍从》第2集的人工验证：7个人工高光6个落在≥3秒间隙） |
| `gap_strong_seconds` | `15.0`（秒） | 间隙 ≥ 此值 → 强度 4 |
| `gap_medium_seconds` | `8.0`（秒） | 间隙 ≥ 此值 → 强度 3，否则强度 2 |
| `low_density_ratio` | `0.4` | 字密度低于均值比例的门槛 |
| `low_density_min_seconds` | `2.0`（秒） | 低密度信号最短持续时长 |
| `low_density_strength` | `3` | 低密度信号强度（受 `STRENGTH_MIN=1`/`STRENGTH_MAX=5` 约束） |
| `shift_window_seconds` | `30.0`（秒） | 语速变化检测的滑动窗口 |
| `shift_z_threshold` | `1.5` | 语速突变的 z-score 门槛 |
| `shift_strength` | `2` | 语速突变信号强度 |
| `min_separation` | `2.0`（秒） | 高能点聚合的最小间距 |
| `summary_max_chars` | `30` | 信号摘要最大字数 |

---

## 7. validate_script（yaml 字段名如此，类名是 `ValidateConfig`，config.py:274-312，6 字段）

| 字段 | 默认值 | 说明 |
|---|---|---|
| `min_beats` | `3` | 节点数下限，低于它直接判错重试（比提示词要求的 5-8 松，避免烧掉一次几百秒的 LLM 调用） |
| `max_beats` | `8` | 上限，超了只报 warning 不拦（实测 13 份样本节点数落在 6-7） |
| `anchor_tolerance_seconds` | `5.0`（秒） | clip 起点与字幕锚点时间的容差，超出以字幕时间为准 |
| `min_clip_seconds` | `1.5`（秒） | clip 最短可用时长，短于它直接丢弃（实测 263 个真实 clip 最短 3.09 秒，取一半留余量） |
| `timeline_regression_max_seconds` | `60.0`（秒） | 画面起点比上一节点倒退超过它才报 warning（倒叙是合法手法，只报不拦） |
| `stretch_max` | `4.0` | 画面/旁白拉伸倍率上限（实测 85 个真实 beat 落在 0.193-2.526，留约 1.6 倍余量） |
| `stretch_min` | `0.125` | 拉伸倍率下限 |

---

## 8. render（`RenderConfig`，config.py:315-438，30 字段）

覆盖面最广的一组，按用途分为 6 类。

### 8.1 TTS 基础
| 字段 | 默认值 | 说明 |
|---|---|---|
| `voice` | `"zh-CN-YunxiNeural"` | Edge-TTS 音色 |
| `rate` | `"+0%"` | 语速调整 |

### 8.2 视频编码
| 字段 | 默认值 | 说明 |
|---|---|---|
| `video_encoder` | `"libx264"` | 编码器 |
| `crf` | `"20"` | 恒定质量因子 |
| `preset` | `"medium"` | x264 preset。实测数据（config.py:349-359）：medium 比 faster 慢 7.3 秒但 PSNR 高约 1dB，渲染不是流水线瓶颈（script 阶段单次 LLM 调用实测 561 秒），不值得为几十秒换质量 |
| `tune` | `""` | x264 tune，空=不传（保持产物逐字节不变）。`"animation"` 对番剧线条画面确实更优但不是 Pareto 更优（耗时 +14~20%），故不做默认 |
| `videotoolbox_bitrate` | `"6000k"` | mac 硬件编码码率，非提速路径，只用于没有 libx264 的机器（实测产物体积是 libx264 medium 的 2.7 倍，PSNR 反而更低） |
| `width` / `height` | `1920` / `1080` | 与字幕 PlayRes 必须同源，否则字幕会被静默缩放 |

### 8.3 字幕
| 字段 | 默认值 | 说明 |
|---|---|---|
| `font_size` | `52`（> 0） | 字幕字号 |
| `subtitle_font_name` | `"Lantinghei SC"` | 字幕字体 |
| `subtitle_max_lines` | `2` | 单条字幕最多折几行，超了报 warning 不改产物；`0`=关闭该检查。实测 saijo 10集360条cue中2行占119条、3行占8条 |
| `subtitle_min_seconds` | `0.7`（秒） | 单条字幕最短显示时长，短于它报 warning；`0`=关闭。全季最短实际 cue 是 0.80 秒，故下界必须小于它 |

### 8.4 混音
| 字段 | 默认值 | 说明 |
|---|---|---|
| `duck_db` | `-12.0` | 原声压低分贝数（saijo 项目实例覆盖为 `-26.0`） |
| `audio_codec` | `"aac"` | 音频编码 |
| `audio_bitrate` | `"192k"` | 音频码率 |
| `limiter_ceiling` | `1.0` | alimiter 天花板（线性幅度，1.0=满刻度对未超标信号完全透明），想留 headroom 可调到 0.891(-1dB) |

### 8.5 片尾
| 字段 | 默认值 | 说明 |
|---|---|---|
| `fade_out_seconds` | `1.5`（秒） | 淡出时长 |
| `outro_card_seconds` | `3.0`（秒） | 片尾黑卡展示时长（saijo 项目实例覆盖为 `0` 即关闭） |
| `outro_message` | `"解说结束，谢谢观看"` | 片尾文案 |
| `outro_font_name` | `"Lantinghei SC"` | 片尾卡字体，与 `subtitle_font_name` 独立（卡片是纯 ASCII+中文标题，选择面更宽） |

### 8.6 TTS 健壮性
| 字段 | 默认值 | 说明 |
|---|---|---|
| `tts_max_attempts` | `3` | 单 chunk 合成失败重试总次数 |
| `tts_concurrency` | `4` | 同时在飞的 chunk 数。**实测选出的膝点值**（config.py:394-420）：115 个真实 chunk，并发1→297.3s，并发4→81.9s(3.6×)，并发8→36.2s，并发24→12.8s。收益膝点在4-8（1→4省76%可省时间，4→8只再省46秒）；且 Edge-TTS 是无公开契约的免费公共服务，不宜用过高并发突发请求；失败代价不对称（一个chunk彻底失败会中止整次运行） |
| `tts_proxy` | `None` | 公司网络走代理时需要 |
| `tts_connect_timeout` | `10`（秒） | socket 连接超时 |
| `tts_receive_timeout` | `60`（秒） | 单次 socket 读超时（不约束整段合成） |
| `tts_chunk_timeout_seconds` | `300.0`（秒） | 单 chunk 整体截止，补上面两个单次超时的漏洞。实测最长 25.6 秒音频，留十倍余量 |

### 8.7 其它
| 字段 | 默认值 | 说明 |
|---|---|---|
| `drift_tolerance` | `0.5`（秒） | 音频/视频时长漂移容差 |
| `ffmpeg_path` | `"ffmpeg"` | ffmpeg 二进制路径 |
| `ffprobe_path` | `"ffprobe"` | ffprobe 二进制路径 |

---

## 补充：环境变量（.env，走 `Settings` 类，config.py:517-524）

不在 `project.yaml` 里，通过 `.env`（`env_prefix="TENMIN_"`）或系统环境变量设置：

```
TENMIN_GEMINI_API_KEY=xxx              # provider=gemini 时必填
TENMIN_MINIMAX_API_KEY=xxx             # provider=minimax 时必填
TENMIN_OPENAI_COMPATIBLE_API_KEY=xxx   # provider=openai_compatible 时必填
```

按 `llm.provider` 对应取值，缺失对应 key 会在 CLI 层直接报错退出，不进入流水线。

仓库 `.gitignore` 已排除 `.env` 和整个 `work/` 目录，密钥与项目产物不会被提交进版本库。

---

## 9. 全量配置示例（所有字段，含默认值）

以下 YAML 覆盖 `ProjectConfig` 的**每一个**字段，取值全部是代码里的默认值（`op_range`/`ed_range`/`default_op_range`/`default_ed_range` 除外——它们默认是 `null`，这里给出示例值以展示写法）。实际项目通常只需要覆盖第 1 节模板里那几个常改字段（见 `tenmin init` 生成的模板），其余留空即可走默认值；这份示例是给"想知道某个字段到底叫什么、该填在哪一层"的场景用的参考文档，不建议直接复制整份去覆盖生产项目。

```yaml
# ============ 顶层字段（第 1 节） ============
show: 示例番剧名          # 必填，真实简体剧名，credits 识别标题卡要用
slug: example              # 必填，对应 work/example/
mode: single_episode       # single_episode | season（season 尚未实现，填了会运行期报错）
target_seconds: 240.0      # 目标解说时长（秒）

locale:
  convert_traditional: true  # 繁体转简体

episodes:                  # 第 2 节：EpisodeConfig 列表，每集手填一项
  - number: 1
    srt: srt/E01.srt        # 相对 project.yaml 所在目录解析
    video: video/E01.mp4    # 可选，渲染前必须补上
    op_range: null           # 例如 [30.5, 91.2]，逐集手填 OP 区间（最高优先级）
    ed_range: null           # 例如 [1348.2, 1416.6]，逐集手填 ED 区间

glossary: {}                # dict[str, str]，术语表，例如 {"伊月": "伊月"}

# ============ 第 3 节：llm（LLMConfig，14 字段） ============
llm:
  provider: gemini                     # gemini | minimax | openai_compatible
  model: gemini-3.6-flash
  base_url: null                       # openai_compatible 用
  thinking: disabled                   # adaptive | disabled
  temperature: null                    # null = 用服务端默认值
  max_output_tokens: null
  timeout_seconds: 120.0                # httpx write 超时
  read_timeout_seconds: 120.0           # 相邻 SSE chunk 间隔超时
  total_timeout_seconds: 1200.0         # 单次请求整体截止
  max_attempts: 3                       # schema 校验失败总尝试次数（含首发）
  transport_max_attempts: 4             # 429/5xx 传输层重试总次数（含首发）
  budget_tolerance: 0.12
  validation_retries: 1                 # 语义校验重试次数（不含首发）
  budget_rewrite_rounds: 1              # 时长预算返工轮数
  script_concurrency: 1                 # 批量模式下 script 阶段预取并发度

# ============ 第 4 节：ingest（IngestConfig，3 字段） ============
ingest:
  merge_max_gap: 0.3                    # 续行合并最大时间间隔（秒）
  merge_max_chars: 40                   # 续行合并最大字符数
  merge_max_line_seconds: 4.0           # 单行 >= 此值视为完整句，不再合并

# ============ 第 5 节：credits（CreditsConfig，21 字段） ============
credits:
  op_search_start: 30.0                 # 聚簇法找 OP，候选区间下界（秒）
  op_search_end: 300.0                  # 候选区间上界（秒）
  credit_head_window: 300.0             # in_credit_window 片头盲窗上界（秒）
  op_span_min: 40.0                     # 合格 OP 簇最短跨度（秒）
  op_span_max: 120.0                    # 合格 OP 簇最长跨度（秒）
  ed_cluster_tail_seconds: 120.0        # 簇起点距片尾多少秒内算 ED 候选
  ed_keyword_window_seconds: 80.0       # in_credit_window 片尾盲窗（秒）
  credit_window_max_ratio: 0.5          # 片头窗+片尾窗覆盖率上限（只约束启发式盲窗）
  cluster_max_gap: 35.0                 # 相邻 credits 行间隔阈值，超过不并簇（秒）
  op_min_silent_span: 60.0              # 静区兜底判定 OP 的最短静默时长（秒）
  op_max_silent_span: 120.0             # 最长静默时长
  title_overlap_threshold: 0.6          # 标题卡与 show 字段字符重合率门槛
  title_card_max_len: 24                # 标题卡最大字符数
  name_list_min_cjk: 6                  # 纯人名罗列识别的 CJK 字数门槛
  name_list_many_segments: 4            # 切分段数达到此值可放宽字数门槛
  name_list_many_min_cjk: 4             # 放宽后的 CJK 字数门槛
  latin_ratio_threshold: 0.6            # 拉丁字母占比门槛（识别英文 staff 行）
  latin_min_len: 6                      # 上述规则生效所需最小非空白长度
  default_op_range: null                # 例如 [30.5, 91.2]，项目级默认 OP 区间
  default_ed_range: null                # 例如 [1348.2, null]，终点 null=到片尾
  manual_window_margin: 5.0             # 手填区间两侧余量宽度（秒）

# ============ 第 6 节：signals（SignalsConfig，11 字段） ============
signals:
  min_gap_seconds: 3.0                  # 无字幕间隙成为"信号"的最小门槛（秒）
  gap_strong_seconds: 15.0              # 间隙 >= 此值 -> 强度 4
  gap_medium_seconds: 8.0               # 间隙 >= 此值 -> 强度 3
  low_density_ratio: 0.4                # 字密度低于均值比例的门槛
  low_density_min_seconds: 2.0          # 低密度信号最短持续时长（秒）
  low_density_strength: 3               # 低密度信号强度（1-5）
  shift_window_seconds: 30.0            # 语速变化检测滑动窗口（秒）
  shift_z_threshold: 1.5                # 语速突变 z-score 门槛
  shift_strength: 2                     # 语速突变信号强度（1-5）
  min_separation: 2.0                   # 高能点聚合最小间距（秒）
  summary_max_chars: 30                 # 信号摘要最大字数

# ============ 第 7 节：validate_script（ValidateConfig，7 字段） ============
validate_script:
  min_beats: 3                          # 节点数下限，低于直接判错重试
  max_beats: 8                          # 节点数上限，超了只报 warning
  anchor_tolerance_seconds: 5.0         # clip 起点与字幕锚点时间容差（秒）
  min_clip_seconds: 1.5                 # clip 最短可用时长，短于直接丢弃（秒）
  timeline_regression_max_seconds: 60.0 # 时间轴倒退超过此值才报 warning（秒）
  stretch_max: 4.0                      # 画面/旁白拉伸倍率上限
  stretch_min: 0.125                    # 拉伸倍率下限

# ============ 第 8 节：render（RenderConfig，30 字段） ============
render:
  # --- TTS 基础 ---
  voice: zh-CN-YunxiNeural
  rate: "+0%"
  # --- 视频编码 ---
  video_encoder: libx264
  crf: "20"
  preset: medium
  tune: ""                              # 空="不传"；番剧线条画面可选 "animation"
  videotoolbox_bitrate: 6000k           # 仅在机器没有 libx264 时用
  width: 1920
  height: 1080
  # --- 字幕 ---
  font_size: 52
  subtitle_font_name: Lantinghei SC
  subtitle_max_lines: 2                 # 0=关闭该检查
  subtitle_min_seconds: 0.7             # 0=关闭该检查
  # --- 混音 ---
  duck_db: -12.0                        # 原声压低分贝数
  audio_codec: aac
  audio_bitrate: 192k
  limiter_ceiling: 1.0                  # alimiter 天花板（线性幅度，1.0=满刻度）
  # --- 片尾 ---
  fade_out_seconds: 1.5
  outro_card_seconds: 3.0               # 0=关闭片尾黑卡
  outro_message: 解说结束，谢谢观看
  outro_font_name: Lantinghei SC
  # --- TTS 健壮性 ---
  tts_max_attempts: 3
  tts_concurrency: 4                    # 实测膝点值，公共 TTS 服务不建议调高
  tts_proxy: null                       # 例如 http://127.0.0.1:7890，公司代理用
  tts_connect_timeout: 10
  tts_receive_timeout: 60
  tts_chunk_timeout_seconds: 300.0
  # --- 其它 ---
  drift_tolerance: 0.5                  # 音频/视频时长漂移容差（秒）
  ffmpeg_path: ffmpeg
  ffprobe_path: ffprobe
```

