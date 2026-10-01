# 硬字幕 OCR 对白来源 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 `project.yaml` 声明了「带硬字幕」的片源，抽帧跑 Apple Vision OCR 认出画面底部那份繁中字幕，作为对白轨的第四条来源（`source="ocr"`），并在 translate 阶段不调 LLM、直接交付简体 `out/E{NN}.zh.srt`。

**Architecture:** 新模块 `ingest/ocr.py`（结构仿 `ingest/asr.py`）：ffmpeg 按步长取帧、裁底部、缩到 1280 宽灰度，rawvideo 走管道，pts 读 `showinfo`；每帧交给 Vision（pyobjc，延迟 import），单帧清洗 + 多帧投票归并成 `RawCue`，复用 `asr.render_srt` 落成 `srt/E{NN}.ocr.srt`。`ingest/resolve.py` 在软字幕轨分枝之后、语音转写之前插入 OCR 分枝（按 mtime 复用缓存、失败不回落）；`normalize` 对 `ocr` 照常繁转简；`pipeline.run_translate` 对 `ocr` 走 `passthrough_track`。

**Tech Stack:** Python 3.12+（venv 实为 3.14）、pydantic v2、typer、ffmpeg/ffprobe（已有封装 `render/ffmpeg.py`）、pyobjc-framework-Vision / -Quartz 12.2（optional extra `ocr`，仅 macOS）、pytest。

**Spec:** `docs/superpowers/specs/2026-10-01-hardsub-ocr-design.md`

## Global Constraints

- **跑 pytest 与 pipeline 命令一律用裸 `~/.local/bin/uv run <cmd>`**（`uv` 可能不在 PATH 上）。**绝不**套 `rtk pytest` / `rtk proxy` / 任何 `rtk` 前缀——本项目的中文输出会把 rtk 的 UTF-8 抓取层搞崩。git 直接用 `git`。
- **做变异测试一律带 `PYTHONDONTWRITEBYTECODE=1`**（等长替换 + 同一整秒内改坏又还原时，pyc 会被无限期当成有效，造成假绿）。
- **所有产物写盘走 `tenmin.atomic`**（`atomic.write_text` 等）。`tests/test_source_hygiene.py::test_artifact_writes_go_through_atomic` 按 AST 审计 `src/`，裸 `X.write_text()` / `write_bytes()` 会被判违规——连 `with atomic_path(dest) as staged: staged.write_text(...)` 这种写法也会被误判，所以一律用 `atomic.write_text(dest, text)`。
- **src/ 里禁止 `assert` 语句**（`test_no_assert_statements_in_src`）；所有 `open()` / `read_text()` / `write_text()` 显式 `encoding=`（`test_text_io_always_declares_encoding`）。
- **注释与 docstring 禁止 `模块.py:行号` 形式的内部交叉引用**（`test_no_line_number_cross_references`），要指路就写「`ingest/asr.py` 的 `render_srt`」这种符号名。
- **src/ 与 tests/ 的注释禁止内部任务代号**（`test_no_internal_task_codes_in_comments`：「P+数字-字母」「全角括号里的字母+数字」「Task + 空格 + 数字」都会被抓）。别把本计划的任务编号写进代码注释。
- **本项目刻意不引入 `logging`**：告知用户用 `print`，报错用 `raise`。
- **退避/重试路径不许裸 `asyncio.sleep`**（llm.py / tts.py 的 `_sleep` / `_rand` 是给测试 monkeypatch 的）。本功能是同步阻塞调用、不引入任何重试，所以只需「别新增 `asyncio.sleep`」。
- **测试只 monkeypatch 模块级函数**（`ocr._recognize`、`ocr.ffmpeg.run_raw_frames`、`resolve.ocr.recognize` 等）；默认测试套**不碰真实 Vision、不碰真实视频文件**。唯一例外是用 `ffmpeg -f lavfi` 现造的 2 秒画面（`shutil.which("ffmpeg") is None` 时 skip，跟 `tests/test_audio_quality.py` 同一惯例）。
- **OCR 引擎只有 Apple Vision，只支持 macOS**：依赖进 optional extra `ocr`（`pyobjc-framework-Vision`、`pyobjc-framework-Quartz`，下限 `>=12.2` 即实现时 PyPI 当前版 12.2.2，带 `sys_platform == 'darwin'` 环境标记）；`import objc / Quartz / Vision / Foundation` **只能写在 `ingest/ocr.py` 的 `_recognize` 函数体内**；`ImportError` 必须翻成 `OCRUnavailableError`，消息里带 `uv sync --extra ocr`。没装 extra 的人必须照常能跑 `tenmin --help` 与全部非 OCR 用法。
- **不做自动探测**：只在 `ProjectConfig.hardsub_enabled(episode)` 为真时走 OCR。优先级固定为：手传 SRT → 视频内软字幕轨（抽取）→ 声明了硬字幕（OCR）→ 语音转写。声明了硬字幕但 OCR 跑不了时**直接报错，不回落到语音转写**。
- **`srt/E{NN}.ocr.srt` 的新鲜度只比源视频 mtime，刻意不看 `OcrConfig`**（人能手改的产物，按指纹失效会把手改静默冲掉）；它**不进** ingest 的新鲜度输入。
- `DialogueTrack.source` / `SubtitleSource.kind` / `build_track(source=)` 三处统一为 `Literal["srt", "asr", "ocr"]`；OpenCC 判据为 `convert_traditional and source in ("srt", "ocr")`。
- translate 遇到 `source == "ocr"`：**不调 LLM、不需要 provider**，按 `select_translatable` 选 `SPEECH_KINDS` 行，`zh` 取对白原文、`glossary` 为空，照常写 `zh/E{NN}.zh.json` 与 `out/E{NN}.zh.srt`（复用 `render_zh_srt`）；**不碰**累积术语表，**不写** `zh/E{NN}.usage.json`；`source == "srt"` 照旧早退、连 `zh/` 都不建。
- `OcrConfig` 字段与默认值（逐字照 spec）：`enabled=False`、`sample_fps=4.0`、`crop_top=0.72`、`center_tolerance=0.08`、`similarity=0.6`、`min_frames=2`、`language="zh-Hant"`；校验 `0 < crop_top < 1`、`0 < center_tolerance ≤ 0.5`、`0 < similarity ≤ 1`、`sample_fps > 0`、`min_frames ≥ 1`。模块常量（不进配置）：缩放宽度 `OCR_WIDTH = 1280`、`_MAX_GAP_FRAMES = 1`、识别级别 accurate。
- `_ARTIFACTS` 已有条目的目录名与后缀是磁盘存量契约，一个字符都不许改，只许新增；新增按集产物必须同步进 `tests/test_pipeline.py` 的 `FROZEN_LAYOUT`。
- 提交信息沿用仓库风格：`feat: …` / `fix: …` / `test: …` / `docs: …`，中文描述。每个任务结束只 `git add` 本任务列出的文件。

## 与 spec 的偏离与澄清（实现时以本计划为准）

1. **抽帧滤镜用确定的整数像素，而不是 spec 写的 `crop=iw:ih*(1-crop_top):0:ih*crop_top,scale=1280:-2`。** rawvideo 管道里一帧有多少字节必须在 ffmpeg 开跑**之前**就知道，而 `ih*0.28` 的取整（crop 按色度采样向下取偶）与 `-2` 的四舍五入都是 ffmpeg 的实现细节。所以新增 `ffmpeg.probe_video_size`，由 `ocr.frame_filter` 算出 `crop=W:H':0:Y,scale=1280:H''`（H' 向下取偶、Y = H - H'、H'' 按比例取偶）。语义与 spec 完全一致（按比例裁到底、宽固定 1280、高按比例取偶），1080p 实测得 `crop=1920:302:0:778,scale=1280:202`。
2. **省略号统一额外收「行尾单个 `.` / `。`」。** spec 第 2 节第 4 步列的是「`•••`、`⋯`、连续两个以上的 `.` / `。`、`・・・` 等」，而前置验证一节写明 Vision 也会把 `…` 认成单个 `.` 与「句尾的 `。`」。本计划两者都收：任何含 `⋯`/`…` 的点号串、或连续两个以上的 `•·・.。` → `…`；行尾单个 `.`/`。` → `…`。单个 `·`（外文人名分隔符）不动。
3. **`EpisodeConfig.hardsub` 为 `None` 时不写进 EPISODE 切片。** spec 说「`EPISODE` 本来就在切片里，`hardsub` 随它一起进入」。但 EPISODE 记号同时挂在 timeline / audio / render 三个阶段上，无条件写 `"hardsub": null` 会让升级后**每一集**这三份切片都被改写、存量成片全部重渲。所以 `config_slices._episode_payload` 在 `hardsub is None` 时删掉这个键；显式写了 true/false 时照常进切片（ingest 因此重跑）。「跟随 `ocr.enabled`」那一半由 ingest 切片里的 `ocr` 子配置负责。
4. **`resolve_subtitle_source` 新增三个带默认值的关键字参数** `hardsub: bool = False`、`ocr_cache: Path | None = None`、`ocr_config: OcrConfig = DEFAULT_OCR`（spec 只点了 `hardsub` 与 `ocr_config`）。默认值保证现有 30 来个调用点/测试不用改；`hardsub=True` 而 `ocr_cache is None` 时抛 `ValueError`。`_OCR_SUFFIX = ".ocr.srt"` 照 spec 放进 resolve，但 OCR 落点的权威是 `pipeline.Paths.ocr_cache`，两者一致由 `test_the_ocr_cache_name_ends_with_ocr_srt` 核对。
5. **`_is_usable_asr_cache` 改名为 `_is_usable_cache`**（spec：「泛化成一个按路径判断的函数」——它本来就只收路径，泛化 = 改名 + docstring 说清两份缓存共用）。
6. **CLI 层不改 provider 的构造条件。** spec 决定 6 要求「run_translate 走这条分支时不会因为没有 API key 而失败」——本计划在 `run_translate` 层落实（OCR 分支一次都不碰 provider，测试直接传 `None`）。但 `cli.run` 现在只要阶段集合含 `translate` 就会 `build_provider`，所以对一个 OCR 项目单跑 `--only translate` 且没配 key 时，仍会在 CLI 层报「缺少 API key」。完整流水线本来就要为 script 阶段配 key，所以按 YAGNI 不动 CLI；见文末「待用户确认」。
7. **进度按估算总帧数打。** `floor(时长 × 源帧率 / 步长)` 作为分母、每跨过一个 10% 打一行；帧数很少时某些档位会被跳过（实测整集 5714/5715 帧，10 档全打）。
8. **默认测试套里有一条真 ffmpeg 的抽帧测试**（lavfi 现造 2 秒画面、Vision 换成假货），锁「滤镜 ffmpeg 真的认、帧大小与管道字节一致、pts 行数与帧数一致」——这三件事纯替身锁不住。没装 ffmpeg 时 skip。

---

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/tenmin/config.py` | 改 | 新增 `OcrConfig`（7 字段 + 校验 + 实测出处注释）、`ProjectConfig.ocr`、`EpisodeConfig.hardsub`、`ProjectConfig.hardsub_enabled()`、`DEFAULT_OCR` |
| `src/tenmin/config_slices.py` | 改 | `STAGE_FIELDS["ingest"]` 加 `"ocr"`；EPISODE 切片在 `hardsub is None` 时省掉该键 |
| `pyproject.toml` / `uv.lock` | 改 | extra `ocr`；pytest marker `ocr`；注释里 marker 个数四→五 |
| `tests/conftest.py` | 改 | `_GATED_MARKERS` 登记 `ocr` |
| `src/tenmin/render/ffmpeg.py` | 改 | 新增 `probe_video_size`、`run_raw_frames`（二进制 stdout 流式切帧 + stderr 并发排空线程） |
| `src/tenmin/ingest/ocr.py` | 新建 | 异常族、`TextBox`/`FrameText`、`sample_step`、`frame_filter`、`parse_showinfo_pts`、`normalize_ellipsis`、`frame_text`、`merge_frames`、`_recognize`（Vision）、`recognize`（编排 + 进度 + 原子落盘） |
| `src/tenmin/ingest/resolve.py` | 改 | 第四条分枝；`_OCR_SUFFIX`；`SubtitleSource.kind` 三值；`_is_usable_cache` |
| `src/tenmin/ingest/asr.py` | 改 | 模块 docstring 里那处 `_is_usable_asr_cache` 改名引用 |
| `src/tenmin/models.py` | 改 | `DialogueTrack.source` 三值 |
| `src/tenmin/ingest/normalize.py` | 改 | `build_track(source=)` 三值；`ocr` 也繁转简 |
| `src/tenmin/pipeline.py` | 改 | `_ARTIFACTS["ocr_cache"]` + `Paths.ocr_cache`；`run_ingest` 递 `hardsub`/`ocr_cache`/`ocr_config`；`run_translate` 的 OCR 分支 |
| `src/tenmin/translate/lines.py` | 改 | 新增 `passthrough_track` |
| `src/tenmin/cli.py` | 改 | `PIPELINE_ERRORS` 加 `OCRError`；`inspect` 显示来源与硬字幕声明；模板注释里的子配置清单 |
| `AGENTS.md` / `README.md` | 改 | 四岔来源、`.ocr.srt` 手改与失效约定、子 config 个数、marker 个数、硬字幕片源用法与「要手填 OP/ED」提醒 |
| `tests/test_config.py` / `tests/test_config_slices.py` / `tests/test_marker_gate.py` | 改 | 配置默认值、边界、覆盖规则、切片 |
| `tests/test_render_ffmpeg.py` | 改 | 两个新 ffmpeg 封装 |
| `tests/test_ocr.py` | 新建 | OCR 层全部单测 + 一条 lavfi 真 ffmpeg + 一条 `@pytest.mark.ocr` 真片源冒烟 |
| `tests/test_resolve.py` / `tests/test_normalize.py` / `tests/test_pipeline.py` / `tests/test_cli.py` | 改 | 四路优先级、缓存、繁转简、translate 直通、inspect |

---

### Task 1: `OcrConfig` + `EpisodeConfig.hardsub` + 切片 + extra / marker

**Files:**
- Modify: `src/tenmin/config.py:80-83`（EpisodeConfig 字段）、`:502-527` 之后（新增 OcrConfig 类）、`:544`（ProjectConfig.asr 之后）、`:571`（srt_path 之前）、`:607`（DEFAULT_ASR 之后）
- Modify: `src/tenmin/config_slices.py:26-27`、`:34`、`:132`、`:149`
- Modify: `src/tenmin/cli.py:122-124`（PROJECT_TEMPLATE_FIELDS docstring 的子配置清单）
- Modify: `pyproject.toml:25`、`:48-67`；`uv.lock`（`uv lock` 自动生成）
- Modify: `tests/conftest.py:12-19`
- Test: `tests/test_config.py`、`tests/test_config_slices.py`、`tests/test_marker_gate.py`

**Interfaces:**
- Consumes: 无（第一个任务）。
- Produces:
  - `tenmin.config.OcrConfig(StrictModel)`：`enabled: bool = False`、`sample_fps: float = 4.0 (gt=0)`、`crop_top: float = 0.72 (gt=0, lt=1)`、`center_tolerance: float = 0.08 (gt=0, le=0.5)`、`similarity: float = 0.6 (gt=0, le=1)`、`min_frames: int = 2 (ge=1)`、`language: str = "zh-Hant"`。
  - `tenmin.config.DEFAULT_OCR: OcrConfig`。
  - `EpisodeConfig.hardsub: bool | None = None`。
  - `ProjectConfig.ocr: OcrConfig`；`ProjectConfig.hardsub_enabled(episode: EpisodeConfig) -> bool`。
  - `config_slices.STAGE_FIELDS["ingest"]` 含 `"ocr"`；EPISODE 切片在 `hardsub is None` 时没有 `"hardsub"` 键。
  - pytest marker `ocr`（默认跳过，`-m ocr` 才放行）。

- [ ] **Step 1: 写失败测试（配置）**

`tests/test_config.py` 顶部 import 块（第 6–20 行）里在 `LocaleConfig,` 后面加一行 `OcrConfig,`，变成：

```python
from tenmin.config import (
    AsrConfig,
    CreditsConfig,
    EpisodeConfig,
    IngestConfig,
    LLMConfig,
    LocaleConfig,
    OcrConfig,
    ProjectConfig,
    ProjectConfigError,
    RenderConfig,
    Settings,
    SignalsConfig,
    ValidateConfig,
    load_project,
)
```

同文件 `test_every_config_model_forbids_extra_fields` 的参数表（约第 790–805 行）在 `(AsrConfig, {}),` 后加一行：

```python
        (OcrConfig, {}),
```

在 `# --- 生肉入口：一集可以只有视频 ----…` 这行注释（约第 620 行）**之前**插入整段：

```python
# --- 硬字幕 OCR ---------------------------------------------------------------


def test_ocr_config_defaults():
    """默认值出自 spec 那次实测（ANi《我是不才惡女》第 11 集），改它要有新的实测。"""
    cfg = OcrConfig()
    assert (
        cfg.enabled,
        cfg.sample_fps,
        cfg.crop_top,
        cfg.center_tolerance,
        cfg.similarity,
        cfg.min_frames,
        cfg.language,
    ) == (False, 4.0, 0.72, 0.08, 0.6, 2, "zh-Hant")


def test_project_config_carries_an_ocr_section():
    cfg = ProjectConfig(show="测试番", slug="test")
    assert isinstance(cfg.ocr, OcrConfig)
    assert cfg.ocr.enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("crop_top", 0),
        ("crop_top", 1),
        ("crop_top", -0.1),
        ("center_tolerance", 0),
        ("center_tolerance", 0.51),
        ("similarity", 0),
        ("similarity", 1.01),
        ("sample_fps", 0),
        ("sample_fps", -4),
        ("min_frames", 0),
    ],
)
def test_ocr_config_rejects_out_of_range_values(field, value):
    with pytest.raises(ValidationError):
        OcrConfig.model_validate({field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [("center_tolerance", 0.5), ("similarity", 1.0), ("min_frames", 1)],
)
def test_ocr_config_accepts_its_inclusive_upper_and_lower_bounds(field, value):
    """`≤ 0.5` / `≤ 1` / `≥ 1` 是闭区间，写成开区间会把合法的边界值拒掉。"""
    assert getattr(OcrConfig.model_validate({field: value}), field) == value


def test_a_bad_ocr_value_names_its_field_path(tmp_path):
    path = _write(tmp_path, MINIMAL + "ocr:\n  crop_top: 1.5\n")
    with pytest.raises(ProjectConfigError) as excinfo:
        load_project(path)
    assert "project.yaml 里 ocr.crop_top 的值不合法" in str(excinfo.value)


def test_episode_hardsub_defaults_to_following_the_project():
    assert EpisodeConfig(number=1).hardsub is None


@pytest.mark.parametrize(
    ("project_enabled", "episode_hardsub", "expected"),
    [
        (False, None, False),
        (True, None, True),
        (True, False, False),
        (False, True, True),
    ],
)
def test_hardsub_enabled_lets_an_episode_override_the_project(
    project_enabled, episode_hardsub, expected
):
    cfg = ProjectConfig(show="测试番", slug="test", ocr=OcrConfig(enabled=project_enabled))
    episode = EpisodeConfig(number=11, video=Path("/v/e11.mp4"), hardsub=episode_hardsub)
    assert cfg.hardsub_enabled(episode) is expected


def test_hardsub_loads_from_yaml(tmp_path):
    path = _write(
        tmp_path,
        "show: 某番\nslug: demo\nocr:\n  enabled: true\nepisodes:\n"
        "  - number: 1\n    video: /v/e01.mp4\n"
        "  - number: 2\n    video: /v/e02.mp4\n    hardsub: false\n",
    )
    cfg = load_project(path)
    assert [cfg.hardsub_enabled(e) for e in cfg.episodes] == [True, False]
```

（`_write` 与 `MINIMAL` 是本文件已有的助手，`_write` 定义在这段之后也没关系——测试函数在运行时才查名字。）

- [ ] **Step 2: 写失败测试（切片与 marker 门）**

`tests/test_config_slices.py` 的 `test_a_config_change_reaches_exactly_the_stages_that_read_it` 参数表里，在 `("asr.language", "en", {"ingest"}),`（第 192 行）后面加两行：

```python
        ("ocr.enabled", True, {"ingest"}),
        ("ocr.crop_top", 0.7, {"ingest"}),
```

同文件在 `def test_slice_paths(tmp_path):`（第 237 行）**之前**插入：

```python
def test_an_unset_hardsub_leaves_the_episode_slice_byte_identical(tmp_path):
    """hardsub 是后加的字段。没写它（None）的集，切片里不许多出这个键 —— 否则升级之后
    timeline / audio / render 的切片全部被改写，存量集白白整套重跑。"""
    cfg = _cfg(tmp_path)
    data = json.loads(slice_payload(cfg, "render", cfg.episodes[0]))
    assert "hardsub" not in data[EPISODE]


def test_a_declared_hardsub_reaches_ingest(tmp_path):
    cfg = _cfg(tmp_path)
    declared = cfg.episodes[0].model_copy(update={"hardsub": True})
    changed = _changed(cfg, cfg.model_copy(update={"episodes": [declared]}))
    assert "ingest" in changed
    data = json.loads(slice_payload(cfg, "ingest", declared))
    assert data[EPISODE]["hardsub"] is True
```

（已有的 `test_the_episode_token_embeds_this_episodes_config` 断言 EPISODE 切片**恰好**是 5 个键——它不改、必须继续绿，那就是「升级后切片逐字节不变」的回归锁。）

`tests/test_marker_gate.py` 文件末尾追加：

```python


def test_the_ocr_marker_is_gated_until_named():
    """ocr 用例要对一整集真实片源逐帧跑 Vision，一次普通 pytest 里不许意外跑到。"""
    from . import conftest

    default = _FakeItem("ocr")
    conftest.pytest_collection_modifyitems(_FakeConfig(None), [default])
    assert default.marks

    named = _FakeItem("ocr")
    conftest.pytest_collection_modifyitems(_FakeConfig("ocr"), [named])
    assert named.marks == []
```

- [ ] **Step 3: 跑测试确认失败**

Run: `~/.local/bin/uv run pytest tests/test_config.py tests/test_config_slices.py tests/test_marker_gate.py -q`
Expected: `tests/test_config.py` 收集期报 `ImportError: cannot import name 'OcrConfig' from 'tenmin.config'`；`test_config_slices.py` 的 `ocr.*` 两条参数报 `AttributeError: 'ProjectConfig' object has no attribute 'ocr'`，`test_a_declared_hardsub_reaches_ingest` 报 `"hardsub"` 相关的 KeyError/ValidationError；`test_the_ocr_marker_is_gated_until_named` 在 `assert default.marks` 处失败。

- [ ] **Step 4: 实现 config.py**

`EpisodeConfig` 里 `ed_range: tuple[float, float] | None = None`（第 83 行）之后、`@field_validator("op_range", "ed_range")` 之前插入：

```python
    # 这一集的画面上有没有烧进去的字幕。None = 跟随项目级的 ocr.enabled；写 true/false
    # 就只管这一集（同一部番里混着别的字幕组片源时用）。判定只有一个入口：
    # ProjectConfig.hardsub_enabled。
    hardsub: bool | None = None
```

在 `class AsrConfig` 整个类结束之后（第 527 行 `language: str = "ja"` 之后）、`class ProjectConfig(StrictModel):` 之前插入：

```python
class OcrConfig(StrictModel):
    """硬字幕 OCR 参数。只在「这部番声明了硬字幕、这一集又没有手传 SRT 和软字幕轨」时才用得上。

    跟 AsrConfig 一样刻意没有 engine 字段：只接 Apple Vision 一个引擎（pyobjc 调用，
    optional extra `uv sync --extra ocr`，只支持 macOS）。调用它的地方同样有两条硬约束：
    **import 必须写在函数体内**，**ImportError 必须翻成一句「跑 uv sync --extra ocr」**。

    下面的默认值全部出自一次实测：`[ANi] 我是不才惡女 - 11 [1080P][Baha][WEB-DL][CHT]`，
    h264 1920×1080、23.976 fps、1429.99 秒，macOS 26.6.2 arm64。整集 388 条（同集 ASR
    371 条）、随机抽 40 条 39 条逐字正确、墙钟 201 秒（约 7 倍实时）。

    不进配置的是物理上的实现细节，留作 ingest/ocr.py 的模块常量：缩放宽度 1280、允许夹在
    同一句里的空帧数 1、Vision 识别级别 accurate。

    **改了这里的任何参数都不会让已有的 `srt/E{NN}.ocr.srt` 失效**（新鲜度只比源视频
    mtime，理由见 ingest/resolve.py 的 _is_usable_cache），想重认必须自己删那份 SRT。
    """

    # 这部番的片源带硬字幕。不做自动探测：画面里本来就有字（片头 staff、招牌、信件），
    # 误判避免不了，而且每集都要多付一轮抽样。一部番的片源通常来自同一个字幕组，声明一次就够。
    enabled: bool = False
    # 目标取样密度（每秒帧数）。实际步长是 round(源帧率 / 这个值)，所以每次取到的都是
    # 真实存在的帧。实测字幕显示时长最短 0.75 秒、p5 为 1.0 秒；降到 2 fps 丢 6 条、另有
    # 50 条只被采到 2 帧（离 min_frames 只差一帧），4 fps 是合适的密度。
    sample_fps: float = Field(default=4.0, gt=0)
    # 裁剪区上沿占画面高度的比例，一直裁到画面底部。实测字幕纵向落在画面高度的 81%–95%，
    # 0.72 给上沿留了一整行的余量。
    crop_top: float = Field(default=0.72, gt=0, lt=1)
    # 文字框中心离水平中线的最大距离（按画面宽度的比例）。字幕水平居中，而 OP/ED 的日文
    # staff 字大多在左右两侧；0.08 刚好把两侧那批挡在外面。挡不住的是居中的那部分
    # staff（监督、原作、ED 的大块），那批交给手填的 op_range / ed_range 去剔除。
    center_tolerance: float = Field(default=0.08, gt=0, le=0.5)
    # 相邻帧文本的 difflib 相似度达到多少算同一句。单帧错字（未/末、日/目）只改一两个字，
    # 0.6 能把它们并进同一条 cue，交给多帧投票修掉。
    similarity: float = Field(default=0.6, gt=0, le=1)
    # 一条 cue 至少出现几帧。实测只出现 1 帧的全是噪声（`7000\n找`、`/1/7`、`MIMM`）。
    min_frames: int = Field(default=2, ge=1)
    # Vision 的识别语言。ANi / Baha 的片源是繁体中文。
    language: str = "zh-Hant"


```

`ProjectConfig` 里 `asr: AsrConfig = Field(default_factory=AsrConfig)`（第 544 行）之后加：

```python
    ocr: OcrConfig = Field(default_factory=OcrConfig)
```

`ProjectConfig` 里 `def srt_path(self, episode: EpisodeConfig) -> Path | None:`（第 571 行）**之前**插入：

```python
    def hardsub_enabled(self, episode: EpisodeConfig) -> bool:
        """这一集是否声明了硬字幕。逐集的 hardsub 写了就听它的，没写（None）跟随 ocr.enabled。

        这是全项目唯一的判定入口：ingest 选对白来源与 inspect 显示「是否声明了硬字幕」
        都调它，两处各写一遍迟早会分叉。
        """
        if episode.hardsub is not None:
            return episode.hardsub
        return self.ocr.enabled

```

`DEFAULT_ASR = AsrConfig()`（第 607 行）之后加一行：

```python
DEFAULT_OCR = OcrConfig()
```

- [ ] **Step 5: 实现 config_slices.py**

第 26–27 行：

```python
# 本集的完整 EpisodeConfig（number / srt / video / op_range / ed_range）。
EPISODE = "episode"
```

改成：

```python
# 本集的 EpisodeConfig（number / srt / video / op_range / ed_range，以及显式写了的 hardsub）。
EPISODE = "episode"
```

第 34 行：

```python
    "ingest": ("locale", "ingest", "credits", "asr", "glossary", "show", EPISODE),
```

改成：

```python
    "ingest": ("locale", "ingest", "credits", "asr", "ocr", "glossary", "show", EPISODE),
```

在 `def slice_path(root: Path, stage: str, episode: int | None) -> Path:`（第 132 行）**之前**插入：

```python
def _episode_payload(episode: EpisodeConfig) -> dict[str, Any]:
    """本集配置的切片内容。`hardsub` 没写（None）时整个键不出现。

    EPISODE 记号挂在 ingest、timeline、audio、render 四个阶段上。hardsub 是后加的字段，
    如果 None 也照常写成 `"hardsub": null`，升级之后每一集这四份切片都会多出一行而被改写，
    于是 timeline / audio / render 对全部存量集整套重跑（它们压根不读这个字段）。省掉
    None 让没用这个字段的项目切片逐字节不变；真写了 true/false 时它照常进切片，ingest
    因此重跑。「跟随项目级 ocr.enabled」那一半由 ingest 切片里的 ocr 子配置负责。
    """
    dumped = episode.model_dump(mode="json")
    if episode.hardsub is None:
        del dumped["hardsub"]
    return dumped


```

`slice_payload` 里（第 149 行）：

```python
            data[EPISODE] = episode.model_dump(mode="json")
```

改成：

```python
            data[EPISODE] = _episode_payload(episode)
```

- [ ] **Step 6: extra、marker、门**

`pyproject.toml` 第 25 行 `asr = ["mlx-whisper>=0.4"]` 之后插入：

```toml
# 硬字幕 OCR（片源把繁中字幕烧在画面上）。Apple Vision 经 pyobjc 调用，只有几 MB、不用下载
# 模型。只支持 macOS：环境标记让非 macOS 上的 `uv sync --extra ocr` 什么都不装，真需要 OCR
# 时报一句「跑 uv sync --extra ocr」（见 ingest/ocr.py 的 _recognize）。
ocr = [
    "pyobjc-framework-Vision>=12.2; sys_platform == 'darwin'",
    "pyobjc-framework-Quartz>=12.2; sys_platform == 'darwin'",
]
```

同文件 `[tool.pytest.ini_options]` 的注释块（第 48–61 行）做三处替换：
- `# 四个 marker 各自已经有更准的门，而且门都在代码里、不在这份配置里：` → `# 五个 marker 各自已经有更准的门，而且门都在代码里、不在这份配置里：`
- 在 `#   - asr：同一个 hook，同一个理由。这批用例要加载一个几个 G 的语音模型跑上几分钟。` 之后加两行：

```toml
#   - ocr：同一个 hook。这批用例要 macOS + ocr extra + 一个真实的硬字幕片源，整集识别
#     要跑上几分钟。
```

- `# 换句话说 addopts deselect 对 render / asr 是重复、对 llm 是退化、对 generalize` → `# 换句话说 addopts deselect 对 render / asr / ocr 是重复、对 llm 是退化、对 generalize`

`markers = [` 列表里 `"asr: …",`（第 66 行）之后加一行：

```toml
    "ocr: 需要 macOS、装了 ocr extra 与真实的硬字幕片源，默认跳过（跑法：TENMIN_OCR_SAMPLE_VIDEO=<片源> uv run pytest -m ocr）",
```

`tests/conftest.py` 第 12–19 行整段换成：

```python
# 默认跳过的 marker，以及跳过时给人看的那句话。
# 共同点是「一次普通 pytest 里意外跑到它会付真实代价」：render 要联网调 Edge-TTS 并
# 真编一段视频，asr 要一个几 G 的模型跑三分钟，ocr 要对一整集真实片源逐帧跑 Vision。
# llm 与 generalize 刻意不进这张表 —— 理由见 pytest_collection_modifyitems 的 docstring。
_GATED_MARKERS = {
    "render": "需要真实素材与 ffmpeg，跑法：uv run pytest -m render",
    "asr": "需要 asr extra 与真实视频，跑法：uv run pytest -m asr",
    "ocr": "需要 macOS、ocr extra 与真实硬字幕片源，跑法：uv run pytest -m ocr",
}
```

`src/tenmin/cli.py` 的 `PROJECT_TEMPLATE_FIELDS` docstring（第 122–124 行）里：

```
/ asr(2) 九十来个调参旋钮，全吐出来的 project.yaml 没人能读，而这个文件是用户的
```

改成：

```
/ asr(2) / ocr(7) 九十来个调参旋钮，全吐出来的 project.yaml 没人能读，而这个文件是用户的
```

然后更新锁文件：

Run: `~/.local/bin/uv lock`
Expected: 输出含 `Added pyobjc-framework-quartz v12.2.2` 与 `Added pyobjc-framework-vision v12.2.2`（以及 pyobjc-core / -cocoa / -coreml）；`uv.lock` 的 `provides-extras` 变成 `["asr", "ocr"]`。**不要**跑 `uv sync --extra ocr`——默认测试套必须在没装 extra 的环境里全绿。

- [ ] **Step 7: 跑测试确认通过**

Run: `~/.local/bin/uv run pytest tests/test_config.py tests/test_config_slices.py tests/test_marker_gate.py -q`
Expected: 全部 PASS（含原有的 `test_the_episode_token_embeds_this_episodes_config`）。

- [ ] **Step 8: 全量 + lint**

Run: `~/.local/bin/uv run pytest tests/ -q`
Expected: 0 failed（基线 2251 passed / 33 skipped，现在只增不减）。

Run: `~/.local/bin/uv run ruff check src/tenmin/config.py src/tenmin/config_slices.py src/tenmin/cli.py tests/conftest.py tests/test_config.py tests/test_config_slices.py tests/test_marker_gate.py`
Expected: `All checks passed!`

- [ ] **Step 9: Commit**

```bash
git add src/tenmin/config.py src/tenmin/config_slices.py src/tenmin/cli.py pyproject.toml uv.lock tests/conftest.py tests/test_config.py tests/test_config_slices.py tests/test_marker_gate.py
git commit -m "feat: 新增 OcrConfig 与逐集 hardsub 声明，登记 ocr extra 与 marker"
```

---

### Task 2: ffmpeg 层：`probe_video_size` 与 `run_raw_frames`

**Files:**
- Modify: `src/tenmin/render/ffmpeg.py:3-13`（import）、`:308-311` 之间（`probe_frame_rate` 之后插 `probe_video_size`）、文件末尾 `:755` 之后（追加 `run_raw_frames`）
- Test: `tests/test_render_ffmpeg.py:1-31`（import）与文件末尾

**Interfaces:**
- Consumes: 本文件已有的 `_probe_field`、`_require_binary`、`_drain`、`tail`、`FFmpegError`、`FFmpegBinaryError`、`STDERR_DRAIN_TIMEOUT_SECONDS`、`STDERR_TAIL_LINES`、`FFMPEG`、`FFPROBE`。
- Produces:
  - `ffmpeg.probe_video_size(path: Path, *, ffprobe: str = FFPROBE) -> tuple[int, int]`：第一条视频轨 `(宽, 高)`；文件不在抛 `FileNotFoundError`，读不出/非正抛 `FFmpegError`。
  - `ffmpeg.run_raw_frames(args: list[str], *, frame_size: int, on_frame: Callable[[bytes], None], ffmpeg: str = FFMPEG) -> str`：argv = `[ffmpeg, "-nostdin", "-nostats", *args]`；stdout 每满 `frame_size` 字节回调一次；返回 stderr 全文（UTF-8 + `errors="replace"`）；非零退出或末尾不足一帧抛 `FFmpegError`；回调抛异常时先 `kill` 再抛；`frame_size <= 0` 抛 `ValueError`。

- [ ] **Step 1: 写失败测试**

`tests/test_render_ffmpeg.py` 第 1–31 行的 import 改成（加 `import io`，加 `probe_video_size`、`run_raw_frames`）：

```python
from __future__ import annotations

import io
import shlex
import subprocess
import threading
from pathlib import Path
from unittest import mock

import pytest

from tenmin.cli import PIPELINE_ERRORS
from tenmin.render.ffmpeg import (
    FFmpegBinaryError,
    FFmpegError,
    extract_audio_track,
    extract_subtitle_track,
    font_available,
    has_audio_stream,
    has_encoder,
    has_filter,
    has_subtitle_stream,
    parse_names,
    preflight,
    probe_duration,
    probe_frame_rate,
    probe_video_size,
    progress_seconds,
    run,
    run_raw_frames,
    run_with_progress,
    tail,
)
```

文件末尾追加（本文件的 autouse fixture `pretend_binaries_exist` 已经把 `shutil.which` 桩成「都存在」，`FakeCompleted` 也是本文件已有的）：

```python


# --- probe_video_size ---


def test_probe_video_size_reads_width_and_height_of_the_first_video_stream(
    monkeypatch, tmp_path
):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"fake")
    seen: dict[str, list[str]] = {}

    def fake_run(args, **kwargs):
        seen["args"] = list(args)
        return FakeCompleted(stdout="1920,1080\n")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    assert probe_video_size(media) == (1920, 1080)
    assert seen["args"][seen["args"].index("-select_streams") + 1] == "v:0"
    assert "stream=width,height" in seen["args"]


@pytest.mark.parametrize("bad", ["", "N/A", "1920", "1920,1080,1", "0,1080", "1920,-2"])
def test_probe_video_size_rejects_unusable_values(monkeypatch, tmp_path, bad):
    media = tmp_path / "a.mkv"
    media.write_bytes(b"fake")
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.run",
        lambda args, **kwargs: FakeCompleted(stdout=f"{bad}\n"),
    )
    with pytest.raises(FFmpegError) as exc:
        probe_video_size(media)
    assert "a.mkv" in str(exc.value)


# --- run_raw_frames ---


class FakeBinaryPopen:
    """假的二进制模式 Popen：stdout / stderr 都是字节流。

    跟上面的 FakePopen 分开：run_raw_frames 不开 text 模式（stdout 是 rawvideo），
    stderr 由被测代码自己套 TextIOWrapper 解码，所以两条管道都必须是真的字节流对象。
    """

    def __init__(self, argv, *, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self.argv = list(argv)
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self._returncode = returncode
        self.killed = False

    def __enter__(self) -> FakeBinaryPopen:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        return self._returncode


def _fake_raw_popen(monkeypatch, **kwargs) -> list[FakeBinaryPopen]:
    """把 Popen 换成 FakeBinaryPopen，返回一个会记下每次 spawn 的列表。"""
    spawned: list[FakeBinaryPopen] = []

    def fake_popen(argv, **_popen_kwargs):
        process = FakeBinaryPopen(argv, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.Popen", fake_popen)
    return spawned


def test_run_raw_frames_hands_out_one_callback_per_frame(monkeypatch):
    spawned = _fake_raw_popen(monkeypatch, stdout=b"aaaabbbbcccc", stderr=b"pts_time:0\n")
    frames: list[bytes] = []

    stderr = run_raw_frames(
        ["-i", "in.mp4", "-f", "rawvideo", "-"], frame_size=4, on_frame=frames.append
    )

    assert frames == [b"aaaa", b"bbbb", b"cccc"]
    assert stderr == "pts_time:0\n"
    assert spawned[0].argv == [
        "ffmpeg", "-nostdin", "-nostats", "-i", "in.mp4", "-f", "rawvideo", "-",
    ]


def test_run_raw_frames_uses_the_configured_ffmpeg(monkeypatch):
    spawned = _fake_raw_popen(monkeypatch, stdout=b"")
    run_raw_frames(
        ["-i", "in.mp4"], frame_size=4, on_frame=lambda _f: None, ffmpeg="/opt/x/ffmpeg"
    )
    assert spawned[0].argv[0] == "/opt/x/ffmpeg"


def test_run_raw_frames_decodes_dirty_stderr_instead_of_crashing(monkeypatch):
    """老番源文件的容器元数据常年不是合法 UTF-8，跟 run 同一个理由用 errors="replace"。"""
    _fake_raw_popen(monkeypatch, stdout=b"", stderr=b"title: \xff\xfe\n")
    stderr = run_raw_frames(["-i", "in.mp4"], frame_size=4, on_frame=lambda _f: None)
    assert "\ufffd" in stderr


def test_run_raw_frames_raises_with_the_stderr_tail_on_failure(monkeypatch):
    _fake_raw_popen(monkeypatch, stdout=b"", stderr=b"Invalid argument\n", returncode=1)
    with pytest.raises(FFmpegError) as exc:
        run_raw_frames(["-i", "in.mp4"], frame_size=4, on_frame=lambda _f: None)
    assert "Invalid argument" in str(exc.value)
    assert "in.mp4" in str(exc.value)


def test_run_raw_frames_refuses_a_trailing_partial_frame(monkeypatch):
    """末尾不足一帧 = 帧大小算错了。静默丢掉的话前面每一帧其实都已经错位。"""
    _fake_raw_popen(monkeypatch, stdout=b"aaaabb")
    frames: list[bytes] = []
    with pytest.raises(FFmpegError) as exc:
        run_raw_frames(["-i", "in.mp4"], frame_size=4, on_frame=frames.append)
    assert "2 字节" in str(exc.value)


def test_run_raw_frames_kills_ffmpeg_when_the_callback_blows_up(monkeypatch):
    spawned = _fake_raw_popen(monkeypatch, stdout=b"aaaabbbb")

    def boom(_frame: bytes) -> None:
        raise RuntimeError("识别炸了")

    with pytest.raises(RuntimeError, match="识别炸了"):
        run_raw_frames(["-i", "in.mp4"], frame_size=4, on_frame=boom)
    assert spawned[0].killed


def test_run_raw_frames_requires_the_binary(monkeypatch):
    monkeypatch.setattr("tenmin.render.ffmpeg.shutil.which", lambda _name: None)
    with pytest.raises(FFmpegBinaryError):
        run_raw_frames(["-i", "in.mp4"], frame_size=4, on_frame=lambda _f: None)


def test_run_raw_frames_rejects_a_non_positive_frame_size():
    with pytest.raises(ValueError):
        run_raw_frames(["-i", "in.mp4"], frame_size=0, on_frame=lambda _f: None)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `~/.local/bin/uv run pytest tests/test_render_ffmpeg.py -q`
Expected: 收集期 `ImportError: cannot import name 'probe_video_size' from 'tenmin.render.ffmpeg'`。

- [ ] **Step 3: 实现**

`src/tenmin/render/ffmpeg.py` 的 import 区（第 3–13 行）在 `from __future__ import annotations` 与 `import re` 之间加 `import io`：

```python
from __future__ import annotations

import io
import re
import shlex
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import IO
```

在 `probe_frame_rate` 函数结束（第 308 行 `return rate`）之后、`# maxsize 从 1 提到 8：key 是可执行文件路径…`（第 311 行）之前插入：

```python
def probe_video_size(path: Path, *, ffprobe: str = FFPROBE) -> tuple[int, int]:
    """第一条视频轨的 (宽, 高)，像素。读不出、或不是两个正整数，一律抛错。

    `-select_streams v:0` 的理由同 probe_frame_rate：真实源片常带一条附图「视频」轨，
    不选流的话取到哪一条纯看运气。

    消费者是硬字幕 OCR 的抽帧（ingest/ocr.py）：它要按真实像素算出裁剪区与缩放后的高度，
    才知道 rawvideo 管道里一帧有多少字节。
    """
    raw = _probe_field(path, "stream=width,height", "画面尺寸", ffprobe=ffprobe, stream="v:0")
    try:
        width, height = (int(part) for part in raw.split(","))
    except ValueError as error:
        raise FFmpegError(f"ffprobe 读不出 {path} 的画面尺寸，输出是 {raw!r}") from error
    if width <= 0 or height <= 0:
        raise FFmpegError(f"ffprobe 报 {path} 的画面尺寸是 {raw!r}，这不可能是条能用的视频轨")
    return width, height


```

文件末尾（`run_with_progress` 之后）追加：

```python


def run_raw_frames(
    args: list[str],
    *,
    frame_size: int,
    on_frame: Callable[[bytes], None],
    ffmpeg: str = FFMPEG,
) -> str:
    """跑一条把 rawvideo 写到 stdout 的 ffmpeg，每凑满 frame_size 字节就回调一次 on_frame。

    返回 stderr 全文（已按 UTF-8 + errors="replace" 解码，理由同 run）。参数怎么拼（滤镜、
    输出格式、`-` 作为输出）是调用方的事，这一层只管管道与错误处理。

    帧是**流式**交出去的，不在内存里攒：一集 24 分钟按每秒 4 帧取、每帧 1280×202 灰度，
    合计约 1.5 GB。

    stderr 必须**并发**排空，理由同 run_with_progress：调用方往往开着 showinfo（每帧一行，
    一集几千行），一旦写满 64KB 的管道，ffmpeg 阻塞在写 stderr、这里阻塞在读 stdout，
    永久挂死。stdout 是二进制，所以 Popen 不开 text 模式，stderr 单独套一层 TextIOWrapper
    再交给 _drain。

    stdout 末尾剩下不足一帧的字节时抛 FFmpegError：那是「帧大小算错了」或「输出格式不是
    调用方以为的那个」，静默丢掉会让后面每一帧都错位。on_frame 抛异常（包括 Ctrl-C）时
    先 kill 子进程再往外抛，理由同 run_with_progress。
    """
    if frame_size <= 0:
        raise ValueError(f"frame_size 必须是正数，收到 {frame_size}")
    _require_binary(ffmpeg)
    argv = [ffmpeg, "-nostdin", "-nostats", *args]
    with subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            drained: list[str] = []
            stderr_text = io.TextIOWrapper(process.stderr, encoding="utf-8", errors="replace")
            drainer = threading.Thread(target=_drain, args=(stderr_text, drained), daemon=True)
            drainer.start()

            leftover = 0
            while True:
                frame = process.stdout.read(frame_size)
                if len(frame) < frame_size:
                    leftover = len(frame)
                    break
                on_frame(frame)

            returncode = process.wait()
            drainer.join(STDERR_DRAIN_TIMEOUT_SECONDS)
            stderr = "".join(drained)
        except BaseException:
            process.kill()
            raise
    if returncode != 0:
        raise FFmpegError(
            f"ffmpeg 执行失败（退出码 {returncode}）：{shlex.join(argv)}\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(stderr)}"
        )
    if leftover:
        raise FFmpegError(
            f"ffmpeg 的帧管道末尾剩下 {leftover} 字节，不足一帧（每帧 {frame_size} 字节）。"
            f"帧大小与输出格式对不上：{shlex.join(argv)}"
        )
    return stderr
```

- [ ] **Step 4: 跑测试确认通过**

Run: `~/.local/bin/uv run pytest tests/test_render_ffmpeg.py -q`
Expected: 全部 PASS。

- [ ] **Step 5: 全量 + lint**

Run: `~/.local/bin/uv run pytest tests/ -q`
Expected: 0 failed。

Run: `~/.local/bin/uv run ruff check src/tenmin/render/ffmpeg.py tests/test_render_ffmpeg.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/tenmin/render/ffmpeg.py tests/test_render_ffmpeg.py
git commit -m "feat: ffmpeg 封装新增画面尺寸探测与 rawvideo 管道逐帧读取"
```

---

### Task 3: `ingest/ocr.py` 纯函数：步长、滤镜几何、pts 解析、单帧清洗、多帧归并

**Files:**
- Create: `src/tenmin/ingest/ocr.py`
- Create: `tests/test_ocr.py`

**Interfaces:**
- Consumes: `tenmin.models.RawCue`（`idx: int, start: float, end: float, text: str`，可就地改字段）。
- Produces（全部在 `tenmin.ingest.ocr`）：
  - `OCR_WIDTH = 1280`、`_MAX_GAP_FRAMES = 1`、`_CJK`（汉字正则，`[\u3400-\u4dbf\u4e00-\u9fff]`）。
  - `class OCRError(RuntimeError)`、`class OCRUnavailableError(OCRError)`。
  - `@dataclass(frozen=True) TextBox(text: str, x: float, y: float, width: float, height: float)`——Vision 归一化坐标，y 轴从下往上，`x`/`y` 是左下角。
  - `@dataclass(frozen=True) FrameText(time: float, text: str)`——空串 = 这一帧没字幕。
  - `sample_step(src_fps: float, sample_fps: float) -> int`
  - `frame_filter(*, width: int, height: int, step: int, crop_top: float) -> tuple[str, int]`（滤镜链字符串，缩放后帧高）
  - `parse_showinfo_pts(stderr: str) -> list[float]`
  - `normalize_ellipsis(text: str) -> str`
  - `frame_text(observations: Sequence[TextBox], *, center_tolerance: float) -> str`
  - `merge_frames(frames: Sequence[FrameText], *, interval: float, similarity: float, min_frames: int) -> list[RawCue]`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_ocr.py`，内容如下（Task 4 会改它的 import 块并在末尾追加编排测试）：

```python
"""硬字幕 OCR 层。刻意不测 Vision 的识别准确率（那要 macOS + 真片源，见 -m ocr 那条），
只测我们自己那几层：步长、滤镜几何、pts 解析、单帧清洗、多帧归并、编排与报错。"""

from __future__ import annotations

import pytest

from tenmin.ingest import ocr
from tenmin.ingest.ocr import FrameText, TextBox

# --- 步长 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src_fps", "expected"),
    [(24000 / 1001, 6), (24.0, 6), (30.0, 8), (60.0, 15), (120.0, 30), (1.0, 1), (0.5, 1)],
)
def test_sample_step_rounds_to_a_real_frame(src_fps, expected):
    """每次取到的都是一帧真实存在的帧；源帧率比目标密度还低时每帧都取（步长不会是 0）。"""
    assert ocr.sample_step(src_fps, 4.0) == expected


# --- 滤镜几何 -------------------------------------------------------------------


def test_frame_filter_for_a_1080p_source():
    chain, out_height = ocr.frame_filter(width=1920, height=1080, step=6, crop_top=0.72)
    # 1080 × 0.28 = 302.4 → 302；起点 1080 - 302 = 778；1280 × 302 / 1920 = 201.3 → 202
    assert out_height == 202
    assert chain == (
        "select='not(mod(n\\,6))',crop=1920:302:0:778,scale=1280:202,format=gray,showinfo"
    )


@pytest.mark.parametrize(("width", "height"), [(1280, 720), (3840, 2160), (1440, 1080)])
def test_frame_filter_always_scales_to_the_fixed_width_and_an_even_height(width, height):
    """宽度固定、高度取偶数：rawvideo 一帧的字节数必须在开跑前就是确定的整数。"""
    chain, out_height = ocr.frame_filter(width=width, height=height, step=6, crop_top=0.72)
    assert f"scale={ocr.OCR_WIDTH}:{out_height}," in chain
    assert out_height % 2 == 0
    assert chain.endswith(",showinfo")


def test_frame_filter_crops_all_the_way_to_the_bottom():
    chain, _ = ocr.frame_filter(width=1920, height=1080, step=6, crop_top=0.5)
    assert "crop=1920:540:0:540," in chain


# --- pts 解析 -------------------------------------------------------------------

# 真实 ffmpeg 9.0.1 的 showinfo 输出（`-f lavfi -i testsrc ... ,showinfo`），含它在同一个
# 滤镜实例上打的 config / color_range 行与 Output 段落 —— 那些行不能被当成帧。
SHOWINFO_SAMPLE = """\
[Parsed_showinfo_4 @ 0xad8c34fc0] config in time_base: 1001/24000, frame_rate: 24000/1001
[Parsed_showinfo_4 @ 0xad8c34fc0] config out time_base: 0/0, frame_rate: 0/0
[Parsed_showinfo_4 @ 0xad8c34fc0] n:   0 pts:      0 pts_time:0       duration:      1 \
duration_time:0.0417083 fmt:gray cl:unspecified sar:1/1 s:1280x264 i:P iskey:1 type:I
[Parsed_showinfo_4 @ 0xad8c34fc0] color_range:pc color_space:unknown color_primaries:unknown
Output #0, rawvideo, to 'pipe:':
[Parsed_showinfo_4 @ 0xad8c34fc0] n:   1 pts:      6 pts_time:0.25025 duration:      1 \
duration_time:0.0417083 fmt:gray cl:unspecified sar:1/1 s:1280x264 i:P iskey:1 type:I
[Parsed_showinfo_4 @ 0xad8c34fc0] n:   2 pts:     12 pts_time:0.5005  duration:      1 \
duration_time:0.0417083 fmt:gray cl:unspecified sar:1/1 s:1280x264 i:P iskey:1 type:I
"""


def test_parse_showinfo_pts_reads_each_frame_in_order():
    assert ocr.parse_showinfo_pts(SHOWINFO_SAMPLE) == [0.0, 0.25025, 0.5005]


def test_parse_showinfo_pts_keeps_uneven_spacing_as_is():
    """可变帧率片源上间隔不均匀，原样采用 —— 不许按序号 × 步长重算。"""
    stderr = "".join(
        f"[Parsed_showinfo_4 @ 0x1] n: {i} pts: {i} pts_time:{t} duration: 1\n"
        for i, t in enumerate(["0", "0.2", "0.55", "0.6"])
    )
    assert ocr.parse_showinfo_pts(stderr) == [0.0, 0.2, 0.55, 0.6]


def test_parse_showinfo_pts_refuses_an_unreadable_timestamp():
    with pytest.raises(ocr.OCRError):
        ocr.parse_showinfo_pts("[Parsed_showinfo_4 @ 0x1] n: 0 pts: NOPTS pts_time:NOPTS\n")


def test_parse_showinfo_pts_of_nothing_is_empty():
    assert ocr.parse_showinfo_pts("") == []


# --- 单帧清洗 -------------------------------------------------------------------


def _box(text: str, *, center_x: float = 0.5, y: float = 0.2, width: float = 0.3) -> TextBox:
    return TextBox(text=text, x=center_x - width / 2, y=y, width=width, height=0.15)


def test_frame_text_keeps_a_centered_subtitle():
    assert ocr.frame_text([_box("我們走吧")], center_tolerance=0.08) == "我們走吧"


def test_frame_text_drops_staff_credits_on_the_sides():
    boxes = [_box("監督 山田", center_x=0.15), _box("我們走吧"), _box("原作 鈴木", center_x=0.85)]
    assert ocr.frame_text(boxes, center_tolerance=0.08) == "我們走吧"


def test_frame_text_center_tolerance_is_measured_from_the_box_center():
    """判的是框**中心**离中线多远，不是框的左边沿（宽框的左边沿离中线很远）。"""
    assert ocr.frame_text([_box("偏了", center_x=0.59)], center_tolerance=0.08) == ""
    assert ocr.frame_text([_box("差一點", center_x=0.57)], center_tolerance=0.08) == "差一點"
    wide = _box("一整行很長的字幕", center_x=0.5, width=0.8)
    assert ocr.frame_text([wide], center_tolerance=0.08) == "一整行很長的字幕"


def test_frame_text_requires_a_chinese_character():
    boxes = [_box("MIMM", y=0.6), _box("/1/7", y=0.3)]
    assert ocr.frame_text(boxes, center_tolerance=0.08) == ""


def test_frame_text_joins_lines_top_to_bottom_with_vision_y_pointing_up():
    """Vision 的 y 轴从下往上：y 大的那一行在画面上方，应该排在前面。"""
    boxes = [_box("-我才不要", y=0.1), _box("-快過來", y=0.5)]
    assert ocr.frame_text(boxes, center_tolerance=0.08) == "-快過來\n-我才不要"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("不要•••", "不要…"),
        ("不要⋯", "不要…"),
        ("不要...", "不要…"),
        ("不要..", "不要…"),
        ("不要。。。", "不要…"),
        ("不要・・・", "不要…"),
        ("不要……", "不要…"),
        ("不要…。", "不要…"),
        ("不要.", "不要…"),
        ("不要。", "不要…"),
        ("我…我不知道", "我…我不知道"),
        ("伊莉莎白·克洛", "伊莉莎白·克洛"),
        ("第3.5話", "第3.5話"),
    ],
)
def test_normalize_ellipsis(raw, expected):
    assert ocr.normalize_ellipsis(raw) == expected


def test_frame_text_normalizes_the_ellipsis_on_every_line():
    boxes = [_box("等等。", y=0.5), _box("不要•••", y=0.1)]
    assert ocr.frame_text(boxes, center_tolerance=0.08) == "等等…\n不要…"


def test_frame_text_of_an_empty_frame_is_empty():
    assert ocr.frame_text([], center_tolerance=0.08) == ""


def test_frame_text_strips_surrounding_whitespace():
    assert ocr.frame_text([_box("  我們走吧 ")], center_tolerance=0.08) == "我們走吧"


# --- 多帧归并 -------------------------------------------------------------------

INTERVAL = 0.25


def _frames(*texts: str) -> list[FrameText]:
    return [FrameText(time=i * INTERVAL, text=text) for i, text in enumerate(texts)]


def _merge(frames: list[FrameText], **overrides) -> list[tuple[float, float, str]]:
    kwargs = {"interval": INTERVAL, "similarity": 0.6, "min_frames": 2, **overrides}
    return [(c.start, c.end, c.text) for c in ocr.merge_frames(frames, **kwargs)]


def test_similar_frames_merge_into_one_cue():
    assert _merge(_frames("我們走吧", "我們走吧", "我們走吧")) == [(0.0, 0.75, "我們走吧")]


def test_the_cue_end_is_the_last_frame_plus_one_interval():
    cues = _merge(_frames("", "我們走吧", "我們走吧"))
    assert cues == [(0.25, 0.75, "我們走吧")]


def test_a_single_blank_frame_inside_a_subtitle_is_tolerated():
    assert _merge(_frames("我們走吧", "", "我們走吧")) == [(0.0, 0.75, "我們走吧")]


def test_two_blank_frames_end_the_cue():
    cues = _merge(_frames("我們走吧", "我們走吧", "", "", "我們走吧", "我們走吧"))
    assert cues == [(0.0, 0.5, "我們走吧"), (1.0, 1.5, "我們走吧")]


def test_a_dissimilar_frame_starts_a_new_cue():
    cues = _merge(_frames("我們走吧", "我們走吧", "明天見", "明天見"))
    assert cues == [(0.0, 0.5, "我們走吧"), (0.5, 1.0, "明天見")]


def test_similarity_compares_against_the_last_non_blank_frame():
    """夹了空帧之后，比的仍然是上一个**有字**的帧，不是那个空帧。"""
    cues = _merge(_frames("我們走吧", "", "我們走吧", "我們走吧"))
    assert cues == [(0.0, 1.0, "我們走吧")]


def test_the_most_frequent_spelling_wins_the_vote():
    """多帧投票修掉单帧错字（未/末）与前缀杂字。"""
    cues = _merge(_frames("還未結束", "還末結束", "還未結束", "C還未結束"))
    assert cues == [(0.0, 1.0, "還未結束")]


def test_a_tied_vote_goes_to_the_earliest_spelling():
    """平票取最早出现的那个，结果确定（同一份素材跑两次得同一个字节）。"""
    cues = _merge(_frames("還末結束", "還未結束", "還未結束", "還末結束"))
    assert cues == [(0.0, 1.0, "還末結束")]


def test_cues_seen_on_fewer_than_min_frames_are_dropped():
    """实测只出现 1 帧的全是噪声。"""
    cues = _merge(_frames("找", "", "", "我們走吧", "我們走吧"))
    assert cues == [(0.75, 1.25, "我們走吧")]


def test_min_frames_is_configurable():
    assert _merge(_frames("找"), min_frames=1) == [(0.0, 0.25, "找")]


def test_an_overlapping_end_is_clamped_to_the_next_start():
    """interval 比实际帧距长时（可变帧率），上一条的终点会越过下一条的起点。"""
    frames = [
        FrameText(time=0.0, text="我們走吧"),
        FrameText(time=0.25, text="我們走吧"),
        FrameText(time=0.4, text="明天見"),
        FrameText(time=0.65, text="明天見"),
    ]
    assert _merge(frames) == [(0.0, 0.4, "我們走吧"), (0.4, 0.9, "明天見")]


def test_merged_cues_are_numbered_from_one():
    cues = ocr.merge_frames(
        _frames("我們走吧", "我們走吧", "", "", "明天見", "明天見"),
        interval=INTERVAL,
        similarity=0.6,
        min_frames=2,
    )
    assert [c.idx for c in cues] == [1, 2]


def test_merging_nothing_gives_nothing():
    assert _merge([]) == []
    assert _merge(_frames("", "", "")) == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `~/.local/bin/uv run pytest tests/test_ocr.py -q`
Expected: 收集期 `ModuleNotFoundError: No module named 'tenmin.ingest.ocr'`。

- [ ] **Step 3: 实现**

新建 `src/tenmin/ingest/ocr.py`：

```python
"""硬字幕 OCR：把烧在画面底部的字幕认出来，落成一份 SRT。

只在「这部番声明了硬字幕（ocr.enabled / episodes[].hardsub），而这一集既没有手传 SRT、
视频里也没有软字幕轨」时才走到这里。跟 asr.py 并排、结构仿照它：产物落成普通 SRT
（`srt/E{NN}.ocr.srt`），落盘即缓存、人能直接手改，手改后照样按 `kind="ocr"` 复用
（判据见 resolve._is_usable_cache）。

管线分三段，前两段是纯函数：

1. 抽帧：ffmpeg 每 step 帧取 1 帧、裁出画面底部、缩到固定宽度的灰度图，rawvideo 走管道，
   每帧的时间戳读 showinfo 打在 stderr 上的真实 pts。
2. 单帧清洗（frame_text）：只留水平居中、含汉字的文字框，从上到下拼起来，统一省略号。
3. 合并成 cue（merge_frames）：相邻帧相似就归入同一句，多帧投票定文本。

默认参数与这些规则全部出自一次实测，数据见 docs/superpowers/specs/2026-10-01-hardsub-ocr-design.md
与 config.OcrConfig 的注释。
"""

from __future__ import annotations

import difflib
import itertools
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from tenmin.models import RawCue

# 交给 Vision 的图固定这么宽，高度按裁剪区的比例取偶数。720p 与 4K 片源交给 Vision 的
# 是同一个尺度的字；实测用的就是这个宽度（1080p 片源缩到 1280 宽）。
OCR_WIDTH = 1280

# 同一句字幕中间允许夹几个「没认出字」的空帧。1 = 容忍单帧漏认；再多就会把两句之间
# 正常的空档也吞掉。
_MAX_GAP_FRAMES = 1

# 认作「有字幕」的必要条件：至少含一个汉字（CJK 基本区 + 扩展 A）。实测的噪声帧
# （`/1/7`、`MIMM`）一个汉字都没有；含汉字的那类（`7000\n找`）靠 min_frames 清掉。
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

# Vision 对 `…` 的几种认法：`•••`、`⋯`、`...`、`・・・`、句尾一个 `.` 或 `。`。不先统一，
# 「不要…」这种短句会因为写法不同在相似度上被拆成好几条（实测拆成 5 条）。
#
# 第一支：一串点号类字符里只要含 `⋯` / `…` 就整串换成一个 `…`（顺带把 `……` 归一成 `…`，
# 投票才不会在两种写法之间分票）。第二支：没有 `⋯` / `…` 时，连续两个以上才算。单个 `·`
# 不动 —— 它在繁中字幕里是外文人名的分隔符（`伊莉莎白·克洛`）。
_ELLIPSIS_RUN = re.compile(r"[•·・.。⋯…]*[⋯…][•·・.。⋯…]*|[•·・.。]{2,}")
# 行尾单个 `.` / `。`。前置验证里 Vision 会把 `…` 认成单个 `.` 或句尾的 `。`，而繁中字幕
# 的行尾通常不打句号，所以按省略号处理。
_TRAILING_DOT = re.compile(r"[.。]$", re.MULTILINE)

# showinfo 给每个输出帧打的那一行：`n:   3 pts:     18 pts_time:0.75075 duration: ...`。
# 同一个滤镜实例还会打 `config in time_base` 与 `color_range` 这类行，它们不含这个形状。
_SHOWINFO_FRAME = re.compile(r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:(\S+)")


class OCRError(RuntimeError):
    """识别本身失败（帧与时间戳对不上、一条字幕都没认出来、Vision 报错）。"""


class OCRUnavailableError(OCRError):
    """需要 OCR，但 OCR 依赖没装（或者不是 macOS）。

    单独一个类型的理由同 asr.ASRUnavailableError：这是「环境缺东西」，消息里要直接给出
    补齐的命令。继承 OCRError，cli.PIPELINE_ERRORS 只登记父类。
    """


@dataclass(frozen=True)
class TextBox:
    """Vision 认出的一行字。坐标按图的宽高归一化到 0–1，**y 轴从下往上**（Vision 的约定）。"""

    text: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class FrameText:
    """一个采样帧：真实时间戳（秒）与清洗后的文本。空串 = 这一帧没有字幕。"""

    time: float
    text: str


def sample_step(src_fps: float, sample_fps: float) -> int:
    """每隔几帧取一帧。`max(1, round(源帧率 / 目标密度))`。

    取整而不是按秒抽：每次取到的都是一帧真实存在的帧，不会有插值或重复帧。23.976 / 24
    fps 得 6，30 得 8，60 得 15，120 得 30；源帧率比目标还低时每帧都取。
    """
    return max(1, round(src_fps / sample_fps))


def frame_filter(*, width: int, height: int, step: int, crop_top: float) -> tuple[str, int]:
    """抽帧滤镜链，以及缩放后一帧的高度（像素）。

    按源片的真实像素算出确定的整数，而不是写 `crop=iw:ih*0.28:...,scale=1280:-2` 让
    ffmpeg 自己算：rawvideo 管道里一帧有多少字节必须在 ffmpeg 开跑之前就知道，而 ffmpeg
    对表达式的取整（crop 按色度采样向下取偶、scale -2 的四舍五入）是它的实现细节。
    裁剪高度向下取偶数，起点 = 画面高度 - 裁剪高度（一直裁到底）；缩放后高度按比例取偶数。

    `select` 按帧序号取帧，所以可变帧率片源上每秒取到的帧数会随源帧率波动 —— 投票和归并
    只看帧的先后，不受影响。末尾的 showinfo 为每个输出帧在 stderr 打一行 pts_time。
    """
    crop_height = max(2, int(height * (1 - crop_top)) // 2 * 2)
    crop_y = height - crop_height
    out_height = max(2, round(OCR_WIDTH * crop_height / width / 2) * 2)
    chain = (
        f"select='not(mod(n\\,{step}))',"
        f"crop={width}:{crop_height}:0:{crop_y},"
        f"scale={OCR_WIDTH}:{out_height},"
        "format=gray,"
        "showinfo"
    )
    return chain, out_height


def parse_showinfo_pts(stderr: str) -> list[float]:
    """按出现顺序取出 showinfo 每一帧的 pts_time（秒）。

    读真实 pts 而不是按 `序号 × step / 源帧率` 推算：那个公式只对恒定帧率成立，可变帧率
    片源上时间轴会越跑越偏。读不出数（比如 `NOPTS`）就抛，不猜。
    """
    times: list[float] = []
    for match in _SHOWINFO_FRAME.finditer(stderr):
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError as error:
            raise OCRError(f"showinfo 给出的帧时间戳读不出来：{raw!r}") from error
        if not math.isfinite(value):
            raise OCRError(f"showinfo 给出的帧时间戳不是有限数：{raw!r}")
        times.append(value)
    return times


def normalize_ellipsis(text: str) -> str:
    """把 Vision 对省略号的各种认法统一成一个 `…`。规则见 _ELLIPSIS_RUN 的注释。"""
    return _TRAILING_DOT.sub("…", _ELLIPSIS_RUN.sub("…", text))


def frame_text(observations: Sequence[TextBox], *, center_tolerance: float) -> str:
    """一帧的文字框 → 这一帧的字幕文本。空串表示这一帧没有字幕。

    1. 只留文字框中心离水平中线不到 center_tolerance 的行（去掉左右两侧的 staff 字）。
    2. 去掉不含汉字的行。
    3. 从上到下排序后用换行拼接。Vision 的 y 轴从下往上，所以按框中心的 y **降序**排。
    4. 统一省略号，去掉首尾空白。
    """
    kept = [
        box
        for box in observations
        if abs(box.x + box.width / 2 - 0.5) < center_tolerance and _CJK.search(box.text)
    ]
    kept.sort(key=lambda box: box.y + box.height / 2, reverse=True)
    joined = "\n".join(box.text.strip() for box in kept)
    return normalize_ellipsis(joined).strip()


def _finalize(frames: Sequence[FrameText], *, idx: int, interval: float) -> RawCue:
    """一组同一句的帧 → 一条 cue。文本取众数，平票取最早出现的那个（结果确定）。"""
    texts = [frame.text for frame in frames]
    counts = Counter(texts)
    best = max(counts.values())
    text = next(candidate for candidate in texts if counts[candidate] == best)
    return RawCue(idx=idx, start=frames[0].time, end=frames[-1].time + interval, text=text)


def merge_frames(
    frames: Sequence[FrameText], *, interval: float, similarity: float, min_frames: int
) -> list[RawCue]:
    """按时间顺序把采样帧归并成 cue。

    - 当前帧跟当前 cue 最后一个**非空帧**的 `SequenceMatcher.ratio()` 达到 similarity 就
      归入同一句；中间最多允许夹 _MAX_GAP_FRAMES 个空帧（容忍单帧漏认）。
    - 不相似，或者连续空帧超过上限，当前 cue 结束。
    - 定稿：文本取多帧众数（修掉单帧错字与前缀杂字），起点 = 首帧时间，终点 = 末帧时间 +
      interval（一个采样间隔）；非空帧少于 min_frames 的丢弃（实测只出现 1 帧的都是噪声）。
    - 去重叠：上一条的终点晚于下一条的起点时截到下一条的起点（实测有毫秒级重叠）。
    """
    groups: list[list[FrameText]] = []
    current: list[FrameText] = []
    gap = 0

    def close() -> None:
        if len(current) >= min_frames:
            groups.append(list(current))
        current.clear()

    for frame in frames:
        if not frame.text:
            if current:
                gap += 1
                if gap > _MAX_GAP_FRAMES:
                    close()
            continue
        if current and (
            difflib.SequenceMatcher(None, current[-1].text, frame.text).ratio() >= similarity
        ):
            current.append(frame)
        else:
            close()
            current.append(frame)
        gap = 0
    close()

    cues = [
        _finalize(group, idx=index, interval=interval)
        for index, group in enumerate(groups, start=1)
    ]
    for previous, following in itertools.pairwise(cues):
        if previous.end > following.start:
            previous.end = following.start
    return cues
```

- [ ] **Step 4: 跑测试确认通过**

Run: `~/.local/bin/uv run pytest tests/test_ocr.py -q`
Expected: 50 passed。

- [ ] **Step 5: 全量 + lint**

Run: `~/.local/bin/uv run pytest tests/ -q`
Expected: 0 failed。

Run: `~/.local/bin/uv run ruff check src/tenmin/ingest/ocr.py tests/test_ocr.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/tenmin/ingest/ocr.py tests/test_ocr.py
git commit -m "feat: 硬字幕 OCR 的单帧清洗与多帧投票归并"
```

---

### Task 4: `ocr.recognize()` 编排 + Vision 调用 + CLI 错误网

**Files:**
- Modify: `src/tenmin/ingest/ocr.py`（Task 3 建的：import 块、常量区、文件末尾）
- Modify: `src/tenmin/cli.py:15-16`（import）、`:45-48`（PIPELINE_ERRORS 注释）、`:58-68`（PIPELINE_ERRORS）
- Modify: `tests/test_ocr.py`（import 块 + 末尾追加）

**Interfaces:**
- Consumes:
  - Task 1：`tenmin.config.OcrConfig`（`sample_fps` / `crop_top` / `center_tolerance` / `similarity` / `min_frames` / `language`），`DEFAULT_RENDER.ffmpeg_path` / `.ffprobe_path`。
  - Task 2：`ffmpeg.probe_video_size(path, *, ffprobe) -> tuple[int, int]`、`ffmpeg.run_raw_frames(args, *, frame_size, on_frame, ffmpeg) -> str`；已有 `ffmpeg.probe_frame_rate(path, *, ffprobe) -> float`、`ffmpeg.probe_duration(path, *, ffprobe) -> float`。
  - Task 3：本模块全部纯函数。
  - 已有：`tenmin.ingest.asr.render_srt(cues: Sequence[RawCue]) -> str`、`tenmin.atomic.write_text(path, text)`、`tenmin.ingest.srt_parser.load_srt_detailed(path)`（测试用）。
- Produces:
  - `ocr._recognize(frame: bytes, *, width: int, height: int, language: str) -> list[TextBox]`（模块级，测试 monkeypatch 它；Vision/Quartz/objc/Foundation 在函数体内 import，失败抛 `OCRUnavailableError`，消息含 `uv sync --extra ocr`）。
  - `ocr.recognize(video: Path, dest: Path, *, ocr: OcrConfig, ffmpeg_path: str = DEFAULT_RENDER.ffmpeg_path, ffprobe_path: str = DEFAULT_RENDER.ffprobe_path) -> None`：同步；开工前打 `  {video.name} 声明了硬字幕，开始识别画面字幕（约 N 分钟）`（N = max(1, round(时长 / 7 / 60))）；每跨过 10% 打一行进度；0 条 cue 抛 `OCRError`（消息含 `crop_top`）且不写 dest；pts 行数 ≠ 帧数抛 `OCRError`；成功时 `atomic.write_text(dest, render_srt(cues))`。
  - `cli.PIPELINE_ERRORS` 含 `OCRError`。

- [ ] **Step 1: 写失败测试**

`tests/test_ocr.py` 的 import 块（Task 3 写的那几行）整段换成：

```python
import builtins
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tenmin.config import OcrConfig
from tenmin.ingest import ocr
from tenmin.ingest.ocr import FrameText, TextBox
from tenmin.ingest.srt_parser import load_srt_detailed
```

文件末尾追加：

```python


# --- recognize：编排 ------------------------------------------------------------

# 在任何 monkeypatch 之前留一份真的 _recognize，给「缺依赖」那条用。
_REAL_RECOGNIZE = ocr._recognize

FPS = 24000 / 1001
STEP = 6
SAMPLE_INTERVAL = STEP / FPS
FRAME_SIZE = ocr.OCR_WIDTH * 202  # 1080p 片源、crop_top=0.72 时缩放后的帧


def _showinfo(times: list[float]) -> str:
    return "".join(
        f"[Parsed_showinfo_4 @ 0x1] n: {i} pts: {i * STEP} pts_time:{t} duration: 1\n"
        for i, t in enumerate(times)
    )


def _video(tmp_path: Path) -> Path:
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    return video


def _fake_ffmpeg(
    monkeypatch, texts: list[str], *, times: list[float] | None = None, duration: float = 60.0
) -> dict:
    """把 ffprobe 三连与抽帧换成假货。第 i 帧的首字节是 i，_recognize 的假货按它取 texts[i]。

    calls 里记下转发进来的参数：可执行文件路径与帧大小只能在调用当场查。
    """
    calls: dict = {"probes": [], "recognize": []}
    if times is None:
        times = [i * SAMPLE_INTERVAL for i in range(len(texts))]

    def probe(name, value):
        def fake(path, **kwargs):
            calls["probes"].append((name, Path(path), kwargs))
            return value

        return fake

    monkeypatch.setattr(ocr.ffmpeg, "probe_frame_rate", probe("fps", FPS))
    monkeypatch.setattr(ocr.ffmpeg, "probe_duration", probe("duration", duration))
    monkeypatch.setattr(ocr.ffmpeg, "probe_video_size", probe("size", (1920, 1080)))

    def fake_run_raw_frames(args, *, frame_size, on_frame, ffmpeg):
        calls["args"] = list(args)
        calls["frame_size"] = frame_size
        calls["ffmpeg"] = ffmpeg
        for index in range(len(texts)):
            on_frame(bytes([index]) + b"\0" * (frame_size - 1))
        return _showinfo(times)

    def fake_recognize(frame, *, width, height, language):
        calls["recognize"].append((width, height, language))
        text = texts[frame[0]]
        return [TextBox(text=text, x=0.35, y=0.2, width=0.3, height=0.15)] if text else []

    monkeypatch.setattr(ocr.ffmpeg, "run_raw_frames", fake_run_raw_frames)
    monkeypatch.setattr(ocr, "_recognize", fake_recognize)
    return calls


def test_recognize_writes_an_srt_of_the_merged_cues(tmp_path, monkeypatch):
    texts = ["", "我們走吧", "我們走吧", "我們走吧", "", "", "不要•••", "不要…"]
    _fake_ffmpeg(monkeypatch, texts)
    dest = tmp_path / "srt" / "E11.ocr.srt"

    ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    parsed = load_srt_detailed(dest)
    assert [cue.text for cue in parsed.cues] == ["我們走吧", "不要…"]
    first = parsed.cues[0]
    assert first.start == pytest.approx(SAMPLE_INTERVAL, abs=1e-3)
    assert first.end == pytest.approx(4 * SAMPLE_INTERVAL, abs=1e-3)


def test_recognize_uses_the_real_pts_not_the_frame_index(tmp_path, monkeypatch):
    """时间戳读 showinfo 的 pts，不按「序号 × 步长 / 帧率」推算。"""
    _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧"], times=[100.0, 100.25])
    dest = tmp_path / "E11.ocr.srt"

    ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    cue = load_srt_detailed(dest).cues[0]
    assert cue.start == pytest.approx(100.0)


def test_recognize_asks_ffmpeg_for_cropped_gray_frames(tmp_path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧"])
    video = _video(tmp_path)

    ocr.recognize(
        video,
        tmp_path / "E11.ocr.srt",
        ocr=OcrConfig(language="zh-Hans"),
        ffmpeg_path="/opt/x/ffmpeg",
        ffprobe_path="/opt/x/ffprobe",
    )

    args = calls["args"]
    assert args[args.index("-i") + 1] == str(video)
    assert args[args.index("-vf") + 1] == (
        "select='not(mod(n\\,6))',crop=1920:302:0:778,scale=1280:202,format=gray,showinfo"
    )
    assert args[args.index("-fps_mode") + 1] == "passthrough"
    assert args[args.index("-f") + 1] == "rawvideo"
    assert args[-1] == "-"
    assert calls["frame_size"] == FRAME_SIZE
    assert calls["ffmpeg"] == "/opt/x/ffmpeg"
    assert {kwargs["ffprobe"] for _, _, kwargs in calls["probes"]} == {"/opt/x/ffprobe"}
    assert {path for _, path, _ in calls["probes"]} == {video}
    assert calls["recognize"][0] == (ocr.OCR_WIDTH, 202, "zh-Hans")


def test_recognize_applies_the_configured_min_frames(tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, ["找", "", "", "我們走吧", "我們走吧"])
    dest = tmp_path / "E11.ocr.srt"

    ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig(min_frames=1))

    assert [cue.text for cue in load_srt_detailed(dest).cues] == ["找", "我們走吧"]


def test_recognize_applies_the_configured_center_tolerance(tmp_path, monkeypatch):
    """框中心在 0.65：默认容差 0.08 挡掉它（一条都没有 → 报错），放宽到 0.2 就收下。"""
    _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧"])
    off_center = [TextBox(text="我們走吧", x=0.5, y=0.2, width=0.3, height=0.15)]
    monkeypatch.setattr(ocr, "_recognize", lambda frame, **_: off_center)

    with pytest.raises(ocr.OCRError):
        ocr.recognize(_video(tmp_path), tmp_path / "E11.ocr.srt", ocr=OcrConfig())

    dest = tmp_path / "E12.ocr.srt"
    ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig(center_tolerance=0.2))
    assert [cue.text for cue in load_srt_detailed(dest).cues] == ["我們走吧"]


def test_recognize_announces_itself_before_the_first_frame(tmp_path, monkeypatch, capsys):
    """整集要跑几分钟。那句告知必须打在开工**之前**，否则几分钟静默会让人以为卡死了。"""
    calls = _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧"], duration=1430.0)
    seen_before: list[str] = []
    real_run = ocr.ffmpeg.run_raw_frames

    def spying_run(args, **kwargs):
        seen_before.append(capsys.readouterr().out)
        return real_run(args, **kwargs)

    monkeypatch.setattr(ocr.ffmpeg, "run_raw_frames", spying_run)

    ocr.recognize(_video(tmp_path), tmp_path / "E11.ocr.srt", ocr=OcrConfig())

    assert calls["recognize"]
    assert "e11.mp4 声明了硬字幕" in seen_before[0]
    # 1430 秒 / 7 倍实时 ≈ 204 秒 ≈ 3 分钟
    assert "约 3 分钟" in seen_before[0]


def test_recognize_reports_progress_every_ten_percent(tmp_path, monkeypatch, capsys):
    # 60 秒 × 23.976 / 6 ≈ 239 帧是进度的分母；假货喂满 239 帧
    texts = ["我們走吧"] * 239
    _fake_ffmpeg(monkeypatch, texts, duration=60.0)

    ocr.recognize(_video(tmp_path), tmp_path / "E11.ocr.srt", ocr=OcrConfig())

    out = capsys.readouterr().out
    marks = [f"{p}%" for p in range(10, 101, 10)]
    assert all(mark in out for mark in marks)
    assert out.count("画面字幕识别 ") == 10


def test_recognize_refuses_to_write_an_empty_srt(tmp_path, monkeypatch):
    """一条都没认出来多半是裁剪区没框住字幕；空文件会被当成合法缓存一直复用下去。"""
    _fake_ffmpeg(monkeypatch, ["", "", ""])
    dest = tmp_path / "E11.ocr.srt"

    with pytest.raises(ocr.OCRError) as excinfo:
        ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    assert "crop_top" in str(excinfo.value)
    assert not dest.exists()


def test_recognize_refuses_when_frames_and_timestamps_disagree(tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, ["我們走吧", "我們走吧", "我們走吧"], times=[0.0, 0.25])
    dest = tmp_path / "E11.ocr.srt"

    with pytest.raises(ocr.OCRError) as excinfo:
        ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    assert "3 帧" in str(excinfo.value)
    assert not dest.exists()


def _block_vision_imports(monkeypatch) -> None:
    real_import = builtins.__import__
    blocked = {"objc", "Quartz", "Vision", "Foundation"}

    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_recognize_without_the_extra_raises_an_actionable_error(tmp_path, monkeypatch):
    """没装 ocr extra（或者不是 macOS）时给一句能照着做的话，而不是裸 ImportError。"""
    _fake_ffmpeg(monkeypatch, ["我們走吧"])
    # _fake_ffmpeg 把 _recognize 换成了假货；这条要的恰恰是真的那个（它自己做 import）。
    monkeypatch.setattr(ocr, "_recognize", _REAL_RECOGNIZE)
    _block_vision_imports(monkeypatch)
    dest = tmp_path / "E11.ocr.srt"

    with pytest.raises(ocr.OCRUnavailableError) as excinfo:
        ocr.recognize(_video(tmp_path), dest, ocr=OcrConfig())

    assert "uv sync --extra ocr" in str(excinfo.value)
    assert not dest.exists()


def test_ocr_unavailable_is_an_ocr_error():
    assert issubclass(ocr.OCRUnavailableError, ocr.OCRError)


def test_ocr_errors_are_caught_by_the_cli_error_net():
    """逃出 PIPELINE_ERRORS 就意味着用户看到一整页 traceback。"""
    from tenmin.cli import PIPELINE_ERRORS

    assert issubclass(ocr.OCRError, PIPELINE_ERRORS)
    assert issubclass(ocr.OCRUnavailableError, PIPELINE_ERRORS)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg")
def test_recognize_drives_a_real_ffmpeg(tmp_path, monkeypatch):
    """用真 ffmpeg 跑一遍抽帧（lavfi 现造一段 2 秒的画面），只把 Vision 换成假货。

    锁的是假货锁不住的那几件事：滤镜链 ffmpeg 真的认、算出来的帧大小跟管道里的字节
    对得上、showinfo 的 pts 行数跟帧数一致。
    """
    video = tmp_path / "e11.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", "testsrc=size=1280x720:rate=24000/1001:duration=2",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    seen: list[tuple[int, int, int]] = []

    def fake_recognize(frame, *, width, height, language):
        seen.append((len(frame), width, height))
        return [TextBox(text="測試畫面", x=0.35, y=0.2, width=0.3, height=0.15)]

    monkeypatch.setattr(ocr, "_recognize", fake_recognize)
    dest = tmp_path / "E11.ocr.srt"

    ocr.recognize(video, dest, ocr=OcrConfig())

    # 2 秒 × 23.976 fps = 48 帧，每 6 帧取 1 帧 → 8 帧。720 × 0.28 = 201.6 → 向下取偶 200；
    # 源宽本来就是 1280，所以缩放后还是 1280×200。
    assert len(seen) == 8
    assert seen[0] == (ocr.OCR_WIDTH * 200, ocr.OCR_WIDTH, 200)
    cues = load_srt_detailed(dest).cues
    assert [cue.text for cue in cues] == ["測試畫面"]
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(8 * SAMPLE_INTERVAL, abs=1e-3)


@pytest.mark.ocr
def test_recognize_a_real_hardsub_episode(tmp_path):
    """真片源冒烟：macOS + `uv sync --extra ocr` + 一个带硬字幕的片源。

    跑法：TENMIN_OCR_SAMPLE_VIDEO=/path/to/[ANi]...[CHT].mp4 uv run pytest -m ocr
    只验「整条链路在真 Vision 上跑得通、认得出一批含汉字的 cue」，不验准确率。
    """
    sample = os.environ.get("TENMIN_OCR_SAMPLE_VIDEO")
    if not sample or not Path(sample).is_file():
        pytest.skip("设置 TENMIN_OCR_SAMPLE_VIDEO 指向一个带硬字幕的真实片源")
    dest = tmp_path / "E01.ocr.srt"

    ocr.recognize(Path(sample), dest, ocr=OcrConfig())

    cues = load_srt_detailed(dest).cues
    assert len(cues) > 50
    assert all(ocr._CJK.search(cue.text) for cue in cues)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `~/.local/bin/uv run pytest tests/test_ocr.py -q`
Expected: 收集期 `AttributeError: module 'tenmin.ingest.ocr' has no attribute '_recognize'`（来自模块级 `_REAL_RECOGNIZE = ocr._recognize`）。

- [ ] **Step 3: 实现 ocr.py**

把 Task 3 的 import 块：

```python
import difflib
import itertools
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from tenmin.models import RawCue
```

换成：

```python
import difflib
import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tenmin import atomic
from tenmin.config import DEFAULT_RENDER, OcrConfig
from tenmin.ingest.asr import render_srt
from tenmin.models import RawCue
from tenmin.render import ffmpeg
```

在 `_SHOWINFO_FRAME = re.compile(...)` 那一行之后、`class OCRError` 之前插入：

```python

# 开工前那句耗时估算用的倍率：实测整集墙钟 201 秒 / 片长 1429.99 秒，约 7 倍实时。
_REALTIME_FACTOR = 7.0
# 进度每完成这么多分之一打一行。
_PROGRESS_STEPS = 10
```

文件末尾（`merge_frames` 之后）追加：

```python


def _recognize(frame: bytes, *, width: int, height: int, language: str) -> list[TextBox]:
    """真正调用 Apple Vision 的那一下：一帧 8 位灰度 rawvideo → 认出来的文字框。

    import 刻意写在函数体内：pyobjc 在 pyproject 的 ocr extra 里、只装得上 macOS，没装它
    的人必须能正常跑 `tenmin --help` 和所有不碰 OCR 的用法。单独提成一个函数是为了让
    测试能换掉它。

    参数：accurate 识别级别、zh-Hant、开语言校正 —— 实测 40 条抽样 39 条逐字正确就是这组。
    Vision 的 confidence 只有 0.3 / 0.5 / 1.0 几档，太粗，不取。每帧包一层 autorelease
    pool：这是一个没有 run loop 的长循环，不包的话 Objective-C 的临时对象要到进程结束才释放。
    """
    try:
        import objc
        import Quartz
        import Vision
        from Foundation import NSData
    except ImportError as exc:
        raise OCRUnavailableError(
            "这一集声明了硬字幕（ocr.enabled 或 episodes[].hardsub），需要识别画面字幕，"
            "但 OCR 依赖没装（只支持 macOS）。跑一次 `uv sync --extra ocr` 再试。"
        ) from exc

    with objc.autorelease_pool():
        data = NSData.dataWithBytes_length_(frame, len(frame))
        provider = Quartz.CGDataProviderCreateWithCFData(data)
        image = Quartz.CGImageCreate(
            width,
            height,
            8,
            8,
            width,
            Quartz.CGColorSpaceCreateDeviceGray(),
            Quartz.kCGImageAlphaNone,
            provider,
            None,
            False,
            Quartz.kCGRenderingIntentDefault,
        )
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_([language])
        request.setUsesLanguageCorrection_(True)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise OCRError(f"Vision 识别失败：{error}")
        boxes: list[TextBox] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            rect = observation.boundingBox()
            boxes.append(
                TextBox(
                    text=str(candidates[0].string()),
                    x=float(rect.origin.x),
                    y=float(rect.origin.y),
                    width=float(rect.size.width),
                    height=float(rect.size.height),
                )
            )
        return boxes


def _progress_printer(total: int) -> Callable[[int], None]:
    """返回一个「第 done 帧做完了」的回调，每跨过一个 10% 打一行。total 不可靠时不打。"""
    printed = {"step": 0}

    def report(done: int) -> None:
        if total <= 0:
            return
        step = min(_PROGRESS_STEPS, done * _PROGRESS_STEPS // total)
        if step > printed["step"]:
            printed["step"] = step
            print(f"  画面字幕识别 {step * 100 // _PROGRESS_STEPS}%（{done}/{total} 帧）")

    return report


def recognize(
    video: Path,
    dest: Path,
    *,
    ocr: OcrConfig,
    ffmpeg_path: str = DEFAULT_RENDER.ffmpeg_path,
    ffprobe_path: str = DEFAULT_RENDER.ffprobe_path,
) -> None:
    """识别 video 画面底部的硬字幕，把结果写成 dest 这份 SRT（繁体原文）。

    同步阻塞调用，理由同 asr.transcribe：ingest 是全局阶段，跑它时没有别的任务在飞。
    可执行文件参数叫 `ffmpeg_path` / `ffprobe_path`，理由同 asr.transcribe（模块顶层的
    `ffmpeg` 名字已被导入的模块占了）。

    一条字幕都没认出来时抛 OCRError、不写出空文件：那多半是裁剪区没框住字幕
    （ocr.crop_top 不对），空文件会被当成一份「这一集没有对白」的合法缓存复用下去。
    """
    src_fps = ffmpeg.probe_frame_rate(video, ffprobe=ffprobe_path)
    duration = ffmpeg.probe_duration(video, ffprobe=ffprobe_path)
    width, height = ffmpeg.probe_video_size(video, ffprobe=ffprobe_path)
    step = sample_step(src_fps, ocr.sample_fps)
    interval = step / src_fps
    chain, out_height = frame_filter(width=width, height=height, step=step, crop_top=ocr.crop_top)

    minutes = max(1, round(duration / _REALTIME_FACTOR / 60))
    print(f"  {video.name} 声明了硬字幕，开始识别画面字幕（约 {minutes} 分钟）")

    texts: list[str] = []
    report = _progress_printer(math.floor(duration * src_fps / step))

    def on_frame(frame: bytes) -> None:
        boxes = _recognize(frame, width=OCR_WIDTH, height=out_height, language=ocr.language)
        texts.append(frame_text(boxes, center_tolerance=ocr.center_tolerance))
        report(len(texts))

    stderr = ffmpeg.run_raw_frames(
        [
            "-hide_banner",
            "-v",
            "info",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vf",
            chain,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        frame_size=OCR_WIDTH * out_height,
        on_frame=on_frame,
        ffmpeg=ffmpeg_path,
    )

    times = parse_showinfo_pts(stderr)
    if len(times) != len(texts):
        raise OCRError(
            f"{video.name} 抽出 {len(texts)} 帧，showinfo 却报了 {len(times)} 个时间戳，"
            "帧与时间对不上，不猜。"
        )
    cues = merge_frames(
        [FrameText(time=t, text=x) for t, x in zip(times, texts, strict=True)],
        interval=interval,
        similarity=ocr.similarity,
        min_frames=ocr.min_frames,
    )
    if not cues:
        raise OCRError(
            f"{video.name} 的画面上一条字幕都没认出来。检查 ocr.crop_top（现在是 "
            f"{ocr.crop_top}，裁剪区要框住字幕所在的画面底部），或者这个片源其实没有硬字幕。"
        )

    # 走 atomic.write_text，理由同 asr.transcribe（源码卫生审计认不出 atomic_path 的暂存名）。
    atomic.write_text(dest, render_srt(cues))
    print(f"  画面字幕识别完成，{len(cues)} 条 → {dest.name}")
```

说明（给实现者，不进代码）：`_recognize` 的 pyobjc 写法在 macOS 26 + pyobjc 12.2.2 上实测可用——`performRequests_error_` 返回 `(ok, error)` 二元组；`NSData` 来自 `Foundation`（pyobjc-framework-Cocoa，是 Quartz 的依赖，会被 extra 一起装上）；`VNRequestTextRecognitionLevelAccurate` 是 0。用 lavfi + drawtext 现造的画面跑通过整条链路，真实片源（spec 里那一集）跑出 387 条、墙钟 3 分 20 秒。

- [ ] **Step 4: 实现 cli.py 的错误网**

`src/tenmin/cli.py` 第 16 行 `from tenmin.ingest.normalize import credit_range_source` 之后加：

```python
from tenmin.ingest.ocr import OCRError
```

`PIPELINE_ERRORS` 上方注释里 `#   ASRUnavailableError 继承它。`（第 48 行）之后加：

```python
# - OCRError：声明了硬字幕的片源要走画面 OCR，而 OCR 依赖没装（子类
#   OCRUnavailableError，消息里带 `uv sync --extra ocr`）、帧与时间戳对不上、或者一条
#   字幕都没认出来。理由与登记方式同 ASRError。
```

`PIPELINE_ERRORS` 元组里 `ASRError,` 之后加一行 `OCRError,`：

```python
PIPELINE_ERRORS = (
    NotImplementedError,
    FileNotFoundError,
    ValueError,
    FFmpegError,
    ASRError,
    OCRError,
    ScriptValidationError,
    httpx.HTTPError,
    LLMError,
    TTSError,
)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `~/.local/bin/uv run pytest tests/test_ocr.py -q`
Expected: 63 passed, 1 skipped（skip 的是 `@pytest.mark.ocr` 那条；本机没装 ffmpeg 时 lavfi 那条也会 skip）。

再确认「没装 extra 也能用 CLI」：

Run: `~/.local/bin/uv run tenmin --help`
Expected: 正常打出帮助，退出码 0（`ingest/ocr.py` 顶层不 import 任何 pyobjc 模块）。

- [ ] **Step 6: 全量 + lint**

Run: `~/.local/bin/uv run pytest tests/ -q`
Expected: 0 failed。

Run: `~/.local/bin/uv run ruff check src/tenmin/ingest/ocr.py src/tenmin/cli.py tests/test_ocr.py`
Expected: `All checks passed!`

- [ ] **Step 7（可选，仅 macOS）: 真片源冒烟**

Run: `~/.local/bin/uv sync --extra ocr && TENMIN_OCR_SAMPLE_VIDEO="/Users/portz/Downloads/[ANi] 我是不才惡女 - 11 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4" ~/.local/bin/uv run pytest tests/test_ocr.py -m ocr -q`
Expected: 1 passed（约 3–4 分钟）。跑完 `~/.local/bin/uv sync` 卸掉 extra，保持默认环境干净。

- [ ] **Step 8: Commit**

```bash
git add src/tenmin/ingest/ocr.py src/tenmin/cli.py tests/test_ocr.py
git commit -m "feat: 硬字幕 OCR 识别入口（Apple Vision 逐帧识别并落成 SRT）"
```

---

### Task 5: 第四条来源接进 ingest：resolve 分枝、`Paths.ocr_cache`、`source="ocr"` 繁转简

**Files:**
- Modify: `src/tenmin/ingest/resolve.py:1-13`（模块 docstring）、`:20-28`（import 与后缀常量）、`:40-49`（SubtitleSource）、`:76-77`（`_is_usable_asr_cache` 改名）、`:92` 之后（docstring 补一句）、`:107-123`（签名与 docstring）、`:185-199`（新分枝）
- Modify: `src/tenmin/ingest/asr.py:7`（docstring 里的函数名）
- Modify: `src/tenmin/models.py:179`
- Modify: `src/tenmin/ingest/normalize.py:164`、`:189-191`、`:214-217`
- Modify: `src/tenmin/pipeline.py:136`、`:147-148`、`:241-243`、`:419-425`、`:476-487`
- Test: `tests/test_resolve.py`、`tests/test_normalize.py`、`tests/test_pipeline.py`

**Interfaces:**
- Consumes:
  - Task 1：`OcrConfig`、`DEFAULT_OCR`、`ProjectConfig.hardsub_enabled(episode) -> bool`、`cfg.ocr`。
  - Task 4：`ocr.recognize(video, dest, *, ocr, ffmpeg_path, ffprobe_path) -> None`、`ocr.OCRUnavailableError`。
- Produces:
  - `resolve.SubtitleSource.kind: Literal["srt", "asr", "ocr"]`；`resolve._OCR_SUFFIX = ".ocr.srt"`；`resolve._is_usable_cache(cache: Path, video: Path) -> bool`（原 `_is_usable_asr_cache`）。
  - `resolve.resolve_subtitle_source(srt, video, *, cache, asr_config, hardsub: bool = False, ocr_cache: Path | None = None, ocr_config: OcrConfig = DEFAULT_OCR, ffmpeg_path=…, ffprobe_path=…) -> SubtitleSource`：`hardsub=True` 时在软字幕轨分枝之后、转写之前走 OCR（缓存可用就复用，否则 `ocr.recognize(video, ocr_cache, ocr=ocr_config, ffmpeg_path=…, ffprobe_path=…)`），返回 `SubtitleSource(ocr_cache, "ocr")`；`hardsub=True` 且 `ocr_cache is None` 抛 `ValueError`。
  - `DialogueTrack.source: Literal["srt", "asr", "ocr"]`；`build_track(..., source: Literal["srt", "asr", "ocr"] = "srt")`，`ocr` 与 `srt` 一样繁转简。
  - `pipeline._ARTIFACTS["ocr_cache"] = ("srt", ".ocr.srt")`；`Paths.ocr_cache(episode: int) -> Path`（`srt/E{NN}.ocr.srt`）。
  - `run_ingest` 调 resolve 时额外传 `hardsub=cfg.hardsub_enabled(episode)`、`ocr_cache=paths.ocr_cache(n)`、`ocr_config=cfg.ocr`。

- [ ] **Step 1: 写失败测试（resolve）**

`tests/test_resolve.py`：

1. 第 1 行模块 docstring 改成：

```python
"""对白轨来源的四岔判据。ffprobe/ffmpeg/OCR/转写全部 fake —— 这一层的职责只是「选哪条路」。"""
```

2. 第 10 行 `from tenmin.config import AsrConfig` 改成 `from tenmin.config import AsrConfig, OcrConfig`。

3. `stub` fixture 里 `calls` 字典加一个键 `"recognize": [],`（放在 `"transcribe": [],` 之后）；`fake_transcribe` 之后、三行 `monkeypatch.setattr` 之前加一个假货，并在三行 setattr 之后再加一行。改完的 fixture 尾部是：

```python
    def fake_recognize(video, dest, **kwargs):
        calls["recognize"].append((Path(video), Path(dest)))
        calls["kwargs"]["recognize"] = kwargs
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text("1\n00:00:01,000 --> 00:00:02,000\n認出來的\n", encoding="utf-8")

    monkeypatch.setattr(resolve.ffmpeg, "has_subtitle_stream", fake_has_subtitle)
    monkeypatch.setattr(resolve.ffmpeg, "extract_subtitle_track", fake_extract)
    monkeypatch.setattr(resolve.asr, "transcribe", fake_transcribe)
    monkeypatch.setattr(resolve.ocr, "recognize", fake_recognize)
    return calls, state
```

4. `_cache` 助手（第 65–66 行）之后加：

```python
def _ocr_cache(tmp_path: Path) -> Path:
    return tmp_path / "srt" / "E11.ocr.srt"
```

5. 文件末尾的 `test_the_source_kind_is_only_srt_or_asr` 整个换成：

```python
def test_the_source_kind_is_only_srt_asr_or_ocr():
    """kind 是下游判「要不要繁转简」「translate 怎么处置」的唯一依据（日语过 OpenCC 会被
    改字），所以它的取值集合必须是封闭的。

    走 get_type_hints 而不是 `__annotations__`：resolve 有 `from __future__ import
    annotations`，直接读 `__annotations__` 拿到的是一个**未求值的 `ForwardRef`**
    （实测 Python 3.14.6：`ForwardRef("Literal['srt', 'asr']")`，`annotationlib.ForwardRef`
    类型，连 `isinstance(x, str)` 都是 False），`== Literal[...]` 永远为假 —— 也就是说
    那样写会得到一条永远红的测试。
    """
    from typing import Literal, get_args, get_type_hints

    field = get_type_hints(resolve.SubtitleSource)["kind"]
    assert field == Literal["srt", "asr", "ocr"]
    assert set(get_args(field)) == {"srt", "asr", "ocr"}
```

6. 文件末尾追加：

```python


# --- 声明了硬字幕：画面 OCR ----------------------------------------------------


def _resolve_hardsub(tmp_path: Path, video: Path, **overrides):
    kwargs = {
        "cache": _cache(tmp_path),
        "asr_config": AsrConfig(),
        "hardsub": True,
        "ocr_cache": _ocr_cache(tmp_path),
        "ocr_config": OcrConfig(),
        **overrides,
    }
    return resolve.resolve_subtitle_source(None, video, **kwargs)


def test_a_declared_hardsub_is_recognized_instead_of_transcribed(tmp_path, stub):
    calls, _ = stub
    video = _video(tmp_path)

    source = _resolve_hardsub(tmp_path, video)

    assert source == resolve.SubtitleSource(_ocr_cache(tmp_path), "ocr")
    assert calls["recognize"] == [(video, _ocr_cache(tmp_path))]
    assert calls["transcribe"] == []
    assert "認出來的" in source.path.read_text(encoding="utf-8")


def test_a_declared_hardsub_still_prefers_an_embedded_subtitle_track(tmp_path, stub):
    """文本字幕轨是最准的素材：声明了硬字幕也照样先抽它。"""
    calls, state = stub
    state["has_subtitle"] = True

    source = _resolve_hardsub(tmp_path, _video(tmp_path))

    assert source.kind == "srt"
    assert source.path.name.endswith(".embedded.srt")
    assert calls["recognize"] == []


def test_a_handed_srt_still_wins_over_a_declared_hardsub(tmp_path, stub):
    calls, _ = stub
    srt = tmp_path / "hand.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n手传\n", encoding="utf-8")

    source = resolve.resolve_subtitle_source(
        srt,
        _video(tmp_path),
        cache=_cache(tmp_path),
        asr_config=AsrConfig(),
        hardsub=True,
        ocr_cache=_ocr_cache(tmp_path),
    )

    assert source == resolve.SubtitleSource(srt, "srt")
    assert calls["probe"] == []
    assert calls["recognize"] == []


def test_without_a_hardsub_declaration_nothing_changes(tmp_path, stub):
    """没声明就维持原来的行为：哪怕磁盘上恰好躺着一份新鲜的 .ocr.srt 也不看它。"""
    calls, _ = stub
    video = _video(tmp_path)
    leftover = _ocr_cache(tmp_path)
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_text("1\n00:00:01,000 --> 00:00:02,000\n舊的\n", encoding="utf-8")
    _touch_relative_to(leftover, video, +10)

    source = resolve.resolve_subtitle_source(
        None, video, cache=_cache(tmp_path), asr_config=AsrConfig(), ocr_cache=leftover
    )

    assert source == resolve.SubtitleSource(_cache(tmp_path), "asr")
    assert calls["recognize"] == []
    assert calls["transcribe"]


def test_a_fresh_ocr_cache_is_reused_instead_of_recognizing_again(tmp_path, stub):
    """整集识别要几分钟，--force 重跑 ingest 不该重付。"""
    calls, _ = stub
    video = _video(tmp_path)
    cache = _ocr_cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("1\n00:00:01,000 --> 00:00:02,000\n既存\n", encoding="utf-8")
    _touch_relative_to(cache, video, +10)

    source = _resolve_hardsub(tmp_path, video)

    assert source == resolve.SubtitleSource(cache, "ocr")
    assert calls["recognize"] == []


def test_a_hand_edited_ocr_cache_survives_an_ocr_config_change(tmp_path, stub):
    """新鲜度刻意不看 OcrConfig：手改过的 .ocr.srt 不许因为改了一个阈值就被静默冲掉。"""
    calls, _ = stub
    video = _video(tmp_path)
    cache = _ocr_cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("1\n00:00:01,000 --> 00:00:02,000\n手改過的\n", encoding="utf-8")
    _touch_relative_to(cache, video, +10)

    source = _resolve_hardsub(tmp_path, video, ocr_config=OcrConfig(crop_top=0.6, similarity=0.8))

    assert calls["recognize"] == []
    assert "手改過的" in source.path.read_text(encoding="utf-8")


def test_a_stale_ocr_cache_is_recognized_again(tmp_path, stub):
    """换了片源（重新压制、换了个版本）就得重认。"""
    calls, _ = stub
    cache = _ocr_cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("舊的\n", encoding="utf-8")
    video = _video(tmp_path)
    _touch_relative_to(cache, video, -10)

    source = _resolve_hardsub(tmp_path, video)

    assert calls["recognize"]
    assert "認出來的" in source.path.read_text(encoding="utf-8")


def test_an_empty_ocr_cache_is_recognized_again(tmp_path, stub):
    """0 字节的缓存是上一次被打断留下的，不是产物。"""
    calls, _ = stub
    video = _video(tmp_path)
    cache = _ocr_cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("", encoding="utf-8")
    _touch_relative_to(cache, video, +10)

    _resolve_hardsub(tmp_path, video)

    assert calls["recognize"]


def test_an_ocr_failure_does_not_fall_back_to_transcription(monkeypatch, tmp_path, stub):
    """用户明确说了要用画面上那份更好的素材，静默换成听写等于把它丢了。"""
    calls, _ = stub

    def unavailable(video, dest, **kwargs):
        raise resolve.ocr.OCRUnavailableError("跑一次 `uv sync --extra ocr` 再试。")

    monkeypatch.setattr(resolve.ocr, "recognize", unavailable)

    with pytest.raises(resolve.ocr.OCRUnavailableError):
        _resolve_hardsub(tmp_path, _video(tmp_path))

    assert calls["transcribe"] == []


def test_a_declared_hardsub_needs_an_ocr_cache_path(tmp_path, stub):
    with pytest.raises(ValueError, match="ocr_cache"):
        _resolve_hardsub(tmp_path, _video(tmp_path), ocr_cache=None)


def test_the_configured_binaries_and_ocr_config_reach_recognize(tmp_path, stub):
    """跟转写那一路同理：形参叫 `ocr_config`，下层收的是 `ocr=`。转错了会静默退回默认值。"""
    calls, _ = stub
    config = OcrConfig(crop_top=0.65, language="zh-Hans")

    _resolve_hardsub(
        tmp_path,
        _video(tmp_path),
        ocr_config=config,
        ffmpeg_path="/opt/homebrew/bin/ffmpeg",
        ffprobe_path="/opt/homebrew/bin/ffprobe",
    )

    assert calls["kwargs"]["recognize"] == {
        "ocr": config,
        "ffmpeg_path": "/opt/homebrew/bin/ffmpeg",
        "ffprobe_path": "/opt/homebrew/bin/ffprobe",
    }
```

- [ ] **Step 2: 写失败测试（normalize 与 pipeline）**

`tests/test_normalize.py` 文件末尾追加：

```python


def test_an_ocr_source_goes_through_opencc(tmp_path):
    """画面 OCR 认出来的是繁体中文字幕，跟原生字幕一样要繁转简（不能跟着 asr 一起被关掉）。"""
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n我們說話\n", encoding="utf-8")

    track = build_track(srt, episode=1, source="ocr", convert_traditional=True)
    assert track.source == "ocr"
    assert "我们说话" in "".join(line.text for line in track.lines)


def test_an_ocr_source_respects_a_disabled_conversion(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n我們說話\n", encoding="utf-8")

    track = build_track(srt, episode=1, source="ocr", convert_traditional=False)
    assert "我們說話" in "".join(line.text for line in track.lines)
```

`tests/test_pipeline.py`：

1. 第 14 行 `from tenmin.ingest.resolve import SubtitleSource` 之前加一行 `from tenmin.ingest import resolve`。
2. `FROZEN_LAYOUT`（第 285–305 行）里 `"asr_cache": "srt/E02.asr.srt",` 之后加：

```python
    "ocr_cache": "srt/E02.ocr.srt",
```

3. 在 `def test_register_episode_without_an_srt_leaves_the_field_empty(tmp_path):`（第 2705 行）之前插入：

```python
def test_the_ocr_cache_name_ends_with_ocr_srt(tmp_path):
    """OCR 缓存的名字由 Paths 一处钉死，并且跟 resolve 那边记的后缀一致。

    它必须跟手传字幕（`E11.srt`）、转写缓存（`.asr.srt`）、软字幕轨抽出来的那份
    （`.embedded.srt`）都分得开：四份住同一个 srt/ 目录，复用策略各不相同。
    """
    path = Paths(tmp_path).ocr_cache(11)
    assert path.name.endswith(resolve._OCR_SUFFIX)
    assert path.parent == tmp_path / "srt"
    assert path != Paths(tmp_path).asr_cache(11)


```

4. 在 `test_ingest_inputs_exclude_the_asr_cache`（第 2826–2835 行）之后插入：

```python


def test_ingest_inputs_exclude_the_ocr_cache(tmp_path):
    """同上：OCR 缓存也是 ingest 自己的产物，算进输入会让 ingest 每次都重跑。"""
    cfg = _project_config(tmp_path)
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=11, srt=None, video=video)
    _file(Paths(cfg.root).ocr_cache(11), "1\n00:00:01,000 --> 00:00:02,000\n你好\n")

    assert all(".ocr.srt" not in str(p) for p in _ingest_inputs(cfg))
```

5. 在 `def test_ingest_never_runs_opencc_on_a_transcribed_track(...)`（第 2894 行）之前插入：

```python
def _fake_ocr_resolve(recorded: list[dict]):
    """同 _fake_resolve，但生肉集按 hardsub 分两条路：声明了就给 OCR 缓存、kind="ocr"。"""

    def fake(srt, video, **kwargs):
        recorded.append({"srt": srt, "video": video, **kwargs})
        if srt is not None:
            return SubtitleSource(srt, "srt")
        if kwargs["hardsub"]:
            return SubtitleSource(kwargs["ocr_cache"], "ocr")
        return SubtitleSource(kwargs["cache"], "asr")

    return fake


def test_ingest_hands_the_hardsub_declaration_to_the_source_layer(
    project, tmp_path, monkeypatch
):
    """项目级 ocr.enabled 经 hardsub_enabled 递下去，连同 OCR 缓存的落点与 OcrConfig。"""
    project.ocr.enabled = True
    cfg = _register_raw_episode(project, tmp_path, "1\n00:00:01,000 --> 00:00:02,000\n你好\n")
    _file(Paths(cfg.root).ocr_cache(11), "1\n00:00:01,000 --> 00:00:02,000\n你好\n")
    recorded: list[dict] = []
    monkeypatch.setattr("tenmin.pipeline.resolve_subtitle_source", _fake_ocr_resolve(recorded))

    tracks = run_ingest(cfg)

    raw = next(call for call in recorded if call["srt"] is None)
    assert raw["hardsub"] is True
    assert raw["ocr_cache"] == Paths(cfg.root).ocr_cache(11)
    assert raw["ocr_config"] is cfg.ocr
    assert next(t for t in tracks if t.episode == 11).source == "ocr"


def test_an_episode_can_opt_out_of_the_project_hardsub_declaration(
    project, tmp_path, monkeypatch
):
    """逐集 hardsub: false 盖过项目级 ocr.enabled: true（同一部番里混着别的字幕组片源）。"""
    project.ocr.enabled = True
    cfg = _register_raw_episode(project, tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nはい\n")
    episode = next(e for e in cfg.episodes if e.number == 11)
    episode.hardsub = False
    recorded: list[dict] = []
    monkeypatch.setattr("tenmin.pipeline.resolve_subtitle_source", _fake_ocr_resolve(recorded))

    tracks = run_ingest(cfg)

    raw = next(call for call in recorded if call["srt"] is None)
    assert raw["hardsub"] is False
    assert next(t for t in tracks if t.episode == 11).source == "asr"


def test_ingest_converts_an_ocr_track_to_simplified(project, tmp_path, monkeypatch):
    """OCR 认出来的是繁体字幕，run_ingest 要把 source="ocr" 递给 build_track 让它繁转简。"""
    project.ocr.enabled = True
    cfg = _register_raw_episode(project, tmp_path, "unused")
    _file(Paths(cfg.root).ocr_cache(11), "1\n00:00:01,000 --> 00:00:02,000\n我們說話\n")
    monkeypatch.setattr("tenmin.pipeline.resolve_subtitle_source", _fake_ocr_resolve([]))

    tracks = run_ingest(cfg)

    track = next(t for t in tracks if t.episode == 11)
    assert "我们说话" in "".join(line.text for line in track.lines)


```

- [ ] **Step 3: 跑测试确认失败**

Run: `~/.local/bin/uv run pytest tests/test_resolve.py tests/test_normalize.py tests/test_pipeline.py -q`
Expected: `test_resolve.py` 全部用到 `stub` 的用例 ERROR（`AttributeError: <module 'tenmin.ingest.resolve'> has no attribute 'ocr'`）；`test_normalize.py` 两条新用例报 pydantic `ValidationError`（`source` 不接受 `'ocr'`）；`test_pipeline.py` 的 `test_paths_layout_is_frozen[ocr_cache]` / `test_paths_exposes_exactly_the_frozen_artifacts` / `test_the_ocr_cache_name_ends_with_ocr_srt` 报 `AttributeError: 'Paths' object has no attribute 'ocr_cache'`，三条 ingest 用例在 `kwargs["hardsub"]` 处 `KeyError`。

- [ ] **Step 4: 实现 models / normalize / asr**

`src/tenmin/models.py` 第 179 行：

```python
    source: Literal["srt", "asr"] = "srt"
```

改成：

```python
    source: Literal["srt", "asr", "ocr"] = "srt"
```

`src/tenmin/ingest/normalize.py` 第 164 行 `    source: Literal["srt", "asr"] = "srt",` 改成 `    source: Literal["srt", "asr", "ocr"] = "srt",`。

同文件 `build_track` docstring 第 189–191 行：

```
    source 说的是这份对白从哪来（原生字幕 / 机器听写），不是文件格式 —— 三条来源路径
    给出的都是 SRT。它会被写进产物，下游靠它区分这两者：听写（"asr"）必然是源片的原生
    语言（日语），所以强制绕开 OpenCC，translate 阶段也按它判要不要跑。
```

改成：

```
    source 说的是这份对白从哪来（原生字幕 / 画面 OCR / 机器听写），不是文件格式 —— 四条
    来源路径给出的都是 SRT。它会被写进产物，下游靠它区分：听写（"asr"）必然是源片的原生
    语言（日语），所以强制绕开 OpenCC；画面 OCR（"ocr"）认出来的是繁体中文字幕，跟原生
    字幕一样照常繁转简。translate 阶段也按它判怎么处置。
```

同文件第 214–217 行：

```python
    # 听写来的对白是源片的原生语言（日语），过一遍繁转简会被改字（实测 `製作の話`
    # → `制作の话`）。这里强制关掉而不是要求调用方记得传：source 是「数据是什么」的
    # 事实，而 convert_traditional 是「想怎么处理繁体中文」的创作旋钮，前者该压住后者。
    convert = convert_traditional and source == "srt"
```

改成：

```python
    # 听写来的对白是源片的原生语言（日语），过一遍繁转简会被改字（实测 `製作の話`
    # → `制作の话`）。这里强制关掉而不是要求调用方记得传：source 是「数据是什么」的
    # 事实，而 convert_traditional 是「想怎么处理繁体中文」的创作旋钮，前者该压住后者。
    # OCR 认出来的是画面上的繁体中文字幕，跟原生字幕同样处置。
    convert = convert_traditional and source in ("srt", "ocr")
```

`src/tenmin/ingest/asr.py` 第 7 行里的 `resolve._is_usable_asr_cache` 改成 `resolve._is_usable_cache`。

- [ ] **Step 5: 实现 resolve.py**

模块 docstring（第 1–13 行）整段换成：

```python
"""对白轨从哪来。

四条路，按「无损且便宜」排序：

1. 手传的 SRT —— 人明确指定了，不猜。
2. 视频里的软字幕轨 —— ffmpeg 一条命令抽出来，零成本零误差。**这条分枝的存在是关键**：
   漏掉它会把一个自带字幕轨的片源白白拉去跑几分钟转写，还把质量换低了。声明了硬字幕的
   片源也照样先走这条：文本字幕轨是最准的素材。
3. 画面 OCR —— 只在 project.yaml 声明了「这部番带硬字幕」时才走（不做自动探测）。画面上
   那份是人工翻译好的中文字幕，比听写好得多。OCR 跑不了（没装 extra、不是 macOS）时
   **直接报错，不回落到语音转写**：用户明确说了要用画面上那份更好的素材。
4. 语音转写 —— 只有前三条都不成立时才走，而且要先告诉人一声。

四条路都归一成「一个 SRT 文件的路径」，所以下游 build_track 拿到的东西形态完全不变。
顺带的好处：OCR 与转写的结果落成 SRT 就等于缓存，也能被人手动修正。
"""
```

第 20–28 行（import 与两个后缀常量连同它们上面的注释）换成：

```python
from tenmin.config import DEFAULT_OCR, DEFAULT_RENDER, AsrConfig, OcrConfig
from tenmin.ingest import asr, ocr
from tenmin.render import ffmpeg

# 三份 SRT 的文件名后缀。它们**必须两两不同**：抽出来那份每次重写，OCR 与转写那两份按
# mtime 复用，共用一个名字既让人分不清手上这份是怎么来的，也会让「复用」那半边的判据落到
# 一份不该被信任的文件上。OCR 那份的路径由调用方给（pipeline.Paths.ocr_cache 是那个名字的
# 唯一权威），这里的后缀只是同一个约定的另一份记录，由
# tests/test_pipeline.py 的 test_the_ocr_cache_name_ends_with_ocr_srt 核对两边一致。
_ASR_SUFFIX = ".asr.srt"
_EMBEDDED_SUFFIX = ".embedded.srt"
_OCR_SUFFIX = ".ocr.srt"
```

`SubtitleSource`（第 40–49 行）的 docstring 与 `kind` 换成：

```python
class SubtitleSource(NamedTuple):
    """选中的对白轨来源。

    kind 不是「文件是什么格式」（四条路给出的都是 SRT），而是「这份对白是怎么来的」：
    原生字幕（srt）、画面 OCR（ocr）、机器听写（asr）。下游靠它决定要不要繁转简（srt 与
    ocr 转，日语听写过 OpenCC 会被改字所以不转）、translate 阶段怎么处置（srt 跳过、
    ocr 不调 LLM 直接交付简体字幕、asr 翻译）。
    """

    path: Path
    kind: Literal["srt", "asr", "ocr"]
```

第 76–77 行：

```python
def _is_usable_asr_cache(cache: Path, video: Path) -> bool:
    """这份**转写**结果还能用吗。
```

改成：

```python
def _is_usable_cache(cache: Path, video: Path) -> bool:
    """这份**转写或 OCR** 结果还能用吗。两者共用同一个判据，下面以转写为例说明。
```

同一个 docstring 里以「代价说清楚：换模型想重转必须自己删掉那份 `.asr.srt`。」结尾的那一行（第 92 行，开头是「解决），静默覆盖是无声的。」）之后加两行：

```
    OCR 那份 `.ocr.srt` 同理：**刻意不看 OcrConfig**，改了 crop_top / similarity 之类想
    重认，得自己删掉它。
```

`resolve_subtitle_source` 的签名与 docstring（第 107–123 行）换成：

```python
def resolve_subtitle_source(
    srt: Path | None,
    video: Path | None,
    *,
    cache: Path,
    asr_config: AsrConfig,
    hardsub: bool = False,
    ocr_cache: Path | None = None,
    ocr_config: OcrConfig = DEFAULT_OCR,
    ffmpeg_path: str = DEFAULT_RENDER.ffmpeg_path,
    ffprobe_path: str = DEFAULT_RENDER.ffprobe_path,
) -> SubtitleSource:
    """挑一条路，返回一份可解析的 SRT 及其来源类型。

    cache 是转写结果的落点（调用方给出，通常是 srt/E{NN}.asr.srt）。从软字幕轨抽出来
    的那份走同目录的另一个名字，见 _embedded_dest。ocr_cache 是 OCR 结果的落点（通常是
    srt/E{NN}.ocr.srt），只在 hardsub 为真时才用得上、也才必须给。

    hardsub 是「这一集声明了硬字幕」（调用方用 ProjectConfig.hardsub_enabled 算好再传），
    这一层不读 project.yaml。

    配置参数叫 `asr_config` / `ocr_config` 而不是跟下层一致的 `asr=` / `ocr=` —— 本模块
    顶层 `asr`、`ocr` 这两个名字已经被导入的模块占了，同名形参会把它们在函数体内遮掉。
    """
```

函数体末尾（第 185–199 行），把：

```python
        return SubtitleSource(embedded, "srt")

    if _is_usable_asr_cache(cache, video):
        return SubtitleSource(cache, "asr")

    # 这是三条路里唯一一条既费时间又有损的，所以让它被看见，而且必须打在调用**之前**
```

换成：

```python
        return SubtitleSource(embedded, "srt")

    if hardsub:
        if ocr_cache is None:
            raise ValueError(f"{video.name} 声明了硬字幕，但调用方没给 OCR 结果的落点（ocr_cache）")
        if _is_usable_cache(ocr_cache, video):
            return SubtitleSource(ocr_cache, "ocr")
        # 耗时告知由 recognize 自己打（那一行已经带着「声明了硬字幕」这个理由和预估分钟数），
        # 这里不再重复一遍。OCR 失败（含没装 extra）原样往外抛，不往下落到语音转写。
        ocr.recognize(
            video,
            ocr_cache,
            ocr=ocr_config,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )
        return SubtitleSource(ocr_cache, "ocr")

    if _is_usable_cache(cache, video):
        return SubtitleSource(cache, "asr")

    # 这是剩下几条路里唯一一条既费时间又有损的，所以让它被看见，而且必须打在调用**之前**
```

（后面的 `print(...)` 与 `asr.transcribe(...)` 不动。）

- [ ] **Step 6: 实现 pipeline.py**

第 136 行注释里的 `_is_usable_asr_cache` 改成 `_is_usable_cache`。

`_ARTIFACTS` 末尾 `"asr_cache": ("srt", ".asr.srt"),`（第 147 行）之后、`}` 之前加：

```python
    # 画面 OCR 的落点。跟 asr_cache 同样的双重身份（ingest 自己产出的缓存 + 下一次运行的
    # 输入，认错了就手改，按 mtime 复用、照样按 kind="ocr" 处置），同样住 srt/。后缀跟
    # ingest/resolve.py 的 _OCR_SUFFIX 保持一致，这里是唯一权威。
    "ocr_cache": ("srt", ".ocr.srt"),
```

`Paths.asr_cache`（第 241–243 行）之后加：

```python

    def ocr_cache(self, episode: int) -> Path:
        """画面 OCR 结果的落点，也是 resolve_subtitle_source 的 ocr_cache 参数。"""
        return self._artifact("ocr_cache", episode)
```

`_ingest_inputs` docstring 第 419–425 行：

```
    刻意**不**把 ingest 自己落在 srt/ 里的那两份产物算进来 —— 语音转写的缓存
    （Paths.asr_cache，`E{NN}.asr.srt`）与软字幕轨抽出来的那份（`E{NN}.embedded.srt`，
    见 ingest.resolve 的 _embedded_dest）：算进输入会让「解析完写出这份 SRT」这个动作
    立刻使 ingest 变得不新鲜，每次都重跑。两者各自的失效判据都在 ingest.resolve 里对着
    源视频判（缓存按 mtime 复用，抽出来那份每次重写）。当前实现两份都进不来（输入只来自
    episodes[].srt），所以这一段不是在描述一层真实过滤，而是给「顺手 glob 一下 srt/
    目录」这个改法留的警告 —— 两份都得排除，不是只排除缓存那一份。
```

改成：

```
    刻意**不**把 ingest 自己落在 srt/ 里的那三份产物算进来 —— 语音转写的缓存
    （Paths.asr_cache，`E{NN}.asr.srt`）、画面 OCR 的缓存（Paths.ocr_cache，
    `E{NN}.ocr.srt`）与软字幕轨抽出来的那份（`E{NN}.embedded.srt`，
    见 ingest.resolve 的 _embedded_dest）：算进输入会让「解析完写出这份 SRT」这个动作
    立刻使 ingest 变得不新鲜，每次都重跑。两者各自的失效判据都在 ingest.resolve 里对着
    源视频判（缓存按 mtime 复用，抽出来那份每次重写）。当前实现三份都进不来（输入只来自
    episodes[].srt），所以这一段不是在描述一层真实过滤，而是给「顺手 glob 一下 srt/
    目录」这个改法留的警告 —— 三份都得排除，不是只排除转写缓存那一份。
```

`run_ingest` 里（第 476–487 行）：

```python
        # 三条来源路径（手传 SRT / 视频内嵌软字幕轨 / 语音转写）都归一成一份 SRT，
```

改成：

```python
        # 四条来源路径（手传 SRT / 视频内嵌软字幕轨 / 画面 OCR / 语音转写）都归一成一份 SRT，
```

并在 `resolve_subtitle_source(...)` 调用里 `asr_config=cfg.asr,` 之后加三行：

```python
            hardsub=cfg.hardsub_enabled(episode),
            ocr_cache=paths.ocr_cache(episode.number),
            ocr_config=cfg.ocr,
```

- [ ] **Step 7: 跑测试确认通过**

Run: `~/.local/bin/uv run pytest tests/test_resolve.py tests/test_normalize.py tests/test_pipeline.py -q`
Expected: 全部 PASS（resolve 的老用例不传新参数，靠默认值 `hardsub=False` 行为不变）。

- [ ] **Step 8: 变异抽查（可选但推荐）**

把 resolve 里 `if hardsub:` 整段挪到 `if ffmpeg.has_subtitle_stream(...)` 之前，跑：

Run: `PYTHONDONTWRITEBYTECODE=1 ~/.local/bin/uv run pytest tests/test_resolve.py -q -k "embedded_subtitle_track"`
Expected: `test_a_declared_hardsub_still_prefers_an_embedded_subtitle_track` FAIL。还原后再跑一次确认 PASS。

- [ ] **Step 9: 全量 + lint**

Run: `~/.local/bin/uv run pytest tests/ -q`
Expected: 0 failed。

Run: `~/.local/bin/uv run ruff check src/tenmin/ingest/resolve.py src/tenmin/ingest/asr.py src/tenmin/ingest/normalize.py src/tenmin/models.py src/tenmin/pipeline.py tests/test_resolve.py tests/test_normalize.py tests/test_pipeline.py`
Expected: `All checks passed!`

- [ ] **Step 10: Commit**

```bash
git add src/tenmin/ingest/resolve.py src/tenmin/ingest/asr.py src/tenmin/ingest/normalize.py src/tenmin/models.py src/tenmin/pipeline.py tests/test_resolve.py tests/test_normalize.py tests/test_pipeline.py
git commit -m "feat: 声明了硬字幕的片源走画面 OCR 取对白轨（source=ocr，照常繁转简）"
```

---

### Task 6: translate 阶段的 OCR 直通（不调 LLM，交付简体 zh.srt）

**Files:**
- Modify: `src/tenmin/translate/lines.py:22`（import）、`:59-77` 之后（新增 `passthrough_track`）
- Modify: `src/tenmin/pipeline.py:53`（import）、`:543-568`（`run_translate` 的 docstring 与分支）
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes:
  - Task 5：`DialogueTrack.source` 可以是 `"ocr"`；ingest 已对 ocr 繁转简（`DialogueLine.text` 是简体）。
  - 已有：`translate.lines.select_translatable(track) -> list[tuple[int, DialogueLine]]`、`models.TranslatedLine(id: int, zh: str)`、`models.TranslatedTrack(episode, lines, glossary)`、`translate.srt_writer.render_zh_srt(track, translated) -> str`、`pipeline._write_json` / `_write_text`（都走 `atomic.write_text`）。
- Produces:
  - `translate.lines.passthrough_track(track: DialogueTrack) -> TranslatedTrack`：`lines = [TranslatedLine(id=position, zh=line.text) for position, line in select_translatable(track)]`，`glossary = {}`，`episode = track.episode`。
  - `pipeline.run_translate(cfg, provider, episode)` 的行为：`source == "srt"` 照旧早退（不建 `zh/`）；`source == "ocr"` 写 `zh/E{NN}.zh.json` 与 `out/E{NN}.zh.srt`、返回 `(passthrough 轨, [])`，**不读不写** `zh/glossary.json`、**不写** `zh/E{NN}.usage.json`、**不碰** `provider`（可以是 `None`）；`source == "asr"` 不变。

- [ ] **Step 1: 写失败测试**

`tests/test_pipeline.py` 里在 `def test_script_freshness_depends_on_the_glossary(tmp_path):`（第 3130 行）之前插入：

```python
def _ocr_project(tmp_path: Path) -> tuple[ProjectConfig, Paths]:
    """一集画面 OCR 来的对白轨（已繁转简）：两条台词、一条 staff、一条整行括注。"""
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="ocr")
    track = DialogueTrack(
        episode=11,
        source="ocr",
        duration=100.0,
        lines=[
            DialogueLine(idx=1, start=1.0, end=2.0, text="我们走吧", raw="我們走吧"),
            DialogueLine(
                idx=2, start=3.0, end=4.0, text="监督 山田", raw="監督 山田", kind="credits"
            ),
            DialogueLine(
                idx=3, start=5.0, end=6.0, text="（数日后）", raw="（數日後）", kind="screen_text"
            ),
            DialogueLine(
                idx=4, start=7.0, end=8.0, text="不要…", raw="不要…", kind="monologue"
            ),
        ],
    )
    _file(paths.dialogue(11), track.model_dump_json(indent=2))
    return cfg, paths


async def test_run_translate_passes_an_ocr_episode_through_without_the_llm(tmp_path):
    """画面上本来就是中文：不调模型，zh 直接取对白原文，只收 speech 行。

    FakeProvider 的预置响应给空列表，它被调用就抛。
    """
    cfg, paths = _ocr_project(tmp_path)
    provider = FakeProvider([])

    translated, stage_warnings = await run_translate(cfg, provider, 11)

    assert provider.calls == []
    assert stage_warnings == []
    assert [(line.id, line.zh) for line in translated.lines] == [(1, "我们走吧"), (4, "不要…")]
    stored = json.loads(paths.zh_lines(11).read_text(encoding="utf-8"))
    assert stored == {
        "episode": 11,
        "lines": [{"id": 1, "zh": "我们走吧"}, {"id": 4, "zh": "不要…"}],
        "glossary": {},
    }
    subtitles = paths.zh_subtitles(11).read_text(encoding="utf-8")
    assert subtitles == (
        "1\n00:00:01,000 --> 00:00:02,000\n我们走吧\n\n"
        "2\n00:00:07,000 --> 00:00:08,000\n不要…\n"
    )


async def test_run_translate_needs_no_provider_for_an_ocr_episode(tmp_path):
    """这条路压根不碰 provider：没配 API key（provider 构造不出来）也不许失败。"""
    cfg, paths = _ocr_project(tmp_path)

    await run_translate(cfg, None, 11)

    assert paths.zh_subtitles(11).is_file()


async def test_run_translate_leaves_the_glossary_and_usage_alone_for_ocr(tmp_path):
    """累积术语表是 script 的新鲜度输入，OCR 这条路连读带写都不许碰它；用量文件也不写。"""
    cfg, paths = _ocr_project(tmp_path)
    _file(paths.glossary, json.dumps({"玲琳": "玲琳"}, ensure_ascii=False))
    _shift_mtime(paths.glossary, -600.0)
    stamp = paths.glossary.stat().st_mtime

    await run_translate(cfg, FakeProvider([]), 11)

    assert paths.glossary.stat().st_mtime == stamp
    assert not paths.zh_usage(11).exists()


async def test_run_translate_on_an_ocr_episode_creates_no_glossary(tmp_path):
    cfg, paths = _ocr_project(tmp_path)

    await run_translate(cfg, FakeProvider([]), 11)

    assert not paths.glossary.exists()


async def test_run_pipeline_delivers_and_then_skips_an_ocr_translate(tmp_path):
    """接线：OCR 集的 translate 照常跑一次、产出两份产物；产物比对白轨新之后就跳过。"""
    cfg, paths = _ocr_project(tmp_path)
    reporter = FakeReporter()

    await run_pipeline(cfg, FakeProvider([]), only=["translate"], reporter=reporter)

    assert ("stage_start", "translate") in reporter.calls
    assert paths.zh_lines(11).is_file()
    assert paths.zh_subtitles(11).is_file()

    _shift_mtime(paths.zh_lines(11), 60.0)
    _shift_mtime(paths.zh_subtitles(11), 60.0)
    again = FakeReporter()
    await run_pipeline(cfg, FakeProvider([]), only=["translate"], reporter=again)
    assert ("stage_skip", "translate") in again.calls


```

（`_project_with_dialogue(source="ocr")` 只登记 video、不写 srt，正好是 OCR 集在 project.yaml 里的样子；随后整份对白轨被 `_ocr_project` 覆盖成带 credits / screen_text 的版本。）

- [ ] **Step 2: 跑测试确认失败**

Run: `~/.local/bin/uv run pytest tests/test_pipeline.py -q -k "ocr"`
Expected: `test_run_translate_passes_an_ocr_episode_through_without_the_llm` 在 `assert [(line.id, line.zh) ...] == [(1, "我们走吧"), (4, "不要…")]` 处失败（现在 `source != "asr"` 一律早退、返回空轨）；`test_run_translate_needs_no_provider_for_an_ocr_episode` 与 `test_run_pipeline_delivers_and_then_skips_an_ocr_translate` 在 `is_file()` 断言处失败。

- [ ] **Step 3: 实现 `passthrough_track`**

`src/tenmin/translate/lines.py` 第 22 行：

```python
from tenmin.models import SPEECH_KINDS, DialogueLine, DialogueTrack, TranslatedTrack
```

改成：

```python
from tenmin.models import (
    SPEECH_KINDS,
    DialogueLine,
    DialogueTrack,
    TranslatedLine,
    TranslatedTrack,
)
```

在 `select_translatable` 之后（第 77 行之后）、`def build_lines_block(` 之前插入：

```python


def passthrough_track(track: DialogueTrack) -> TranslatedTrack:
    """原文本来就是中文的一集（画面 OCR 来的，ingest 已经繁转简）：不调模型，zh 直接取原文。

    挑哪些行跟送去翻译的完全是同一套（select_translatable），id 也是同一个口径（轨里的
    1-based 位置），所以下游 render_zh_srt 与 zh.json 的形状跟 ASR 路径一模一样。glossary
    留空：这条路上没有模型替我们认专有名词，累积表也不该被它碰。
    """
    return TranslatedTrack(
        episode=track.episode,
        lines=[
            TranslatedLine(id=position, zh=line.text)
            for position, line in select_translatable(track)
        ],
    )
```

- [ ] **Step 4: 实现 `run_translate` 的分支**

`src/tenmin/pipeline.py` 第 53 行 `from tenmin.translate.lines import translate_track` 改成：

```python
from tenmin.translate.lines import passthrough_track, translate_track
```

`run_translate` docstring 开头那段（第 548–550 行）：

```
    只对听写来的对白动手。手传的字幕与从视频里抽出来的软字幕轨都是片源自带的，本来
    就是观众能读的语言。判据用对白轨的 source 字段，刻意不做语言自动检测 —— 那是个
    会错的猜测，而 source 是一个确定的事实。
```

换成：

```
    按对白轨的 source 字段分三路，刻意不做语言自动检测 —— 那是个会错的猜测，而 source
    是一个确定的事实：

    - "srt"：手传的字幕与从视频里抽出来的软字幕轨都是片源自带的，本来就是观众能读的
      语言，整个阶段跳过（见下一段）。
    - "ocr"：画面上认出来的中文字幕，ingest 已经繁转简。**不调 LLM**，按
      passthrough_track 原样交付 `zh/E{NN}.zh.json` 与简体的 `out/E{NN}.zh.srt`
      （产物集合与 run_pipeline 里 translate 的 outputs 一致，新鲜度照常）。这条路不碰
      累积术语表、不写 `zh/E{NN}.usage.json`，也压根不碰 provider —— 传 None 也行。
    - "asr"：听写来的日语对白，真的翻译。
```

同一 docstring 末尾那句（第 564 行）`过」。跳过（非 asr 来源）时返回空列表，不产生任何提示噪音。` 改成 `过」。srt 来源跳过、ocr 来源直通时都返回空列表，不产生任何提示噪音。`

函数体开头（第 566–570 行）：

```python
    track = next(t for t in _load_tracks(cfg) if t.episode == episode)
    if track.source != "asr":
        return TranslatedTrack(episode=episode), []

    paths = Paths(cfg.root)
```

换成：

```python
    track = next(t for t in _load_tracks(cfg) if t.episode == episode)
    if track.source == "srt":
        return TranslatedTrack(episode=episode), []

    paths = Paths(cfg.root)
    if track.source == "ocr":
        translated = passthrough_track(track)
        _write_json(paths.zh_lines(episode), translated.model_dump_json(indent=2))
        _write_text(paths.zh_subtitles(episode), render_zh_srt(track, translated))
        return translated, []
```

（其后 `accumulated = load_glossary(paths.glossary)` 起的 ASR 路径一字不动。）

- [ ] **Step 5: 跑测试确认通过**

Run: `~/.local/bin/uv run pytest tests/test_pipeline.py -q -k "translate"`
Expected: 全部 PASS，含原有的 `test_run_translate_skips_a_native_subtitle_episode`（srt 仍早退、不建 `zh/`）。

- [ ] **Step 6: 全量 + lint**

Run: `~/.local/bin/uv run pytest tests/ -q`
Expected: 0 failed。

Run: `~/.local/bin/uv run ruff check src/tenmin/translate/lines.py src/tenmin/pipeline.py tests/test_pipeline.py`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/tenmin/translate/lines.py src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "feat: translate 阶段对 OCR 来源不调 LLM，直接交付简体中文字幕"
```

---

### Task 7: `tenmin inspect` 显示来源与硬字幕声明 + 文档

**Files:**
- Modify: `src/tenmin/cli.py:349`（`_RANGE_SOURCE_LABEL` 之前新增 `_SOURCE_LABEL`）、`:403-404`（inspect 输出）
- Modify: `AGENTS.md:13-14`、`:55`、`:57`、`:59`、`:108-115`
- Modify: `README.md:6-9`、`:27-34`、`:147-181`、`:183-186`、`:264-265`、`:276-284`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes：Task 1 的 `ProjectConfig.hardsub_enabled(episode) -> bool`；Task 5 的 `DialogueTrack.source: Literal["srt", "asr", "ocr"]`。
- Produces：`tenmin inspect` 在「对白轨 E{NN}：…」那一行之后多打一行 `  来源：{source}（{中文标签}），硬字幕：已声明|未声明`，中文标签取自 `cli._SOURCE_LABEL = {"srt": "原生字幕", "ocr": "画面 OCR", "asr": "语音转写"}`。

- [ ] **Step 1: 写失败测试**

`tests/test_cli.py` 文件末尾追加：

```python


def _ocr_project_with_dialogue(work, *, hardsub_line: str = "") -> None:
    """一个声明了硬字幕的项目，外加一份已落盘的 OCR 来源对白轨（不跑 ingest）。"""
    from tenmin.models import DialogueLine, DialogueTrack

    root = work / "akujo"
    (root / "01_dialogue").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: 我是不才恶女\nslug: akujo\nocr:\n  enabled: true\n"
        f"episodes:\n- number: 11\n  video: /v/e11.mp4\n{hardsub_line}",
        encoding="utf-8",
    )
    track = DialogueTrack(
        episode=11,
        source="ocr",
        duration=1430.0,
        lines=[DialogueLine(idx=1, start=1.0, end=2.0, text="我们走吧", raw="我們走吧")],
    )
    (root / "01_dialogue" / "E11.dialogue.json").write_text(
        track.model_dump_json(indent=2), encoding="utf-8"
    )


def test_inspect_shows_the_dialogue_source_and_the_hardsub_declaration(work):
    _ocr_project_with_dialogue(work)
    result = runner.invoke(
        app, ["inspect", "akujo", "--work-dir", str(work), "--episode", "11", "--suspect"]
    )
    assert result.exit_code == 0, out(result)
    assert "来源：ocr（画面 OCR），硬字幕：已声明" in out(result)


def test_inspect_follows_an_episode_level_hardsub_override(work):
    _ocr_project_with_dialogue(work, hardsub_line="  hardsub: false\n")
    result = runner.invoke(
        app, ["inspect", "akujo", "--work-dir", str(work), "--episode", "11", "--suspect"]
    )
    assert result.exit_code == 0, out(result)
    assert "硬字幕：未声明" in out(result)


def test_inspect_labels_a_native_subtitle_source(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"])
    result = runner.invoke(
        app, ["inspect", "saijo", "--work-dir", str(work), "--episode", "2", "--suspect"]
    )
    assert result.exit_code == 0, out(result)
    assert "来源：srt（原生字幕），硬字幕：未声明" in out(result)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `~/.local/bin/uv run pytest tests/test_cli.py -q -k "inspect"`
Expected: 三条新用例在 `"来源：…" in out(result)` 处失败。

- [ ] **Step 3: 实现**

`src/tenmin/cli.py` 里 `# OP/ED 区间三级回退里实际生效的那一级…`（第 349 行）之前插入：

```python
# 对白轨来源（DialogueTrack.source）给 inspect 显示用。三种来源下游的处置各不相同
# （繁转简、translate 怎么跑、OP/ED 推断靠不靠得住），所以值得一眼看见。
_SOURCE_LABEL = {
    "srt": "原生字幕",
    "ocr": "画面 OCR",
    "asr": "语音转写",
}

```

`inspect` 里（第 403–404 行）：

```python
    typer.echo(f"对白轨 E{episode:02d}：{len(track.lines)} 行，时长 {track.duration:.3f}s")
    typer.echo("  分类：" + "、".join(f"{k}={v}" for k, v in sorted(counts.items())))
```

换成：

```python
    typer.echo(f"对白轨 E{episode:02d}：{len(track.lines)} 行，时长 {track.duration:.3f}s")
    # 「是否声明了硬字幕」读的是**现在**的 project.yaml，来源读的是上次 ingest 的产物；
    # 两者对不上（刚改了声明还没重跑 ingest、或者有软字幕轨优先）时并排摆着最好查。
    hardsub = ep_cfg is not None and cfg.hardsub_enabled(ep_cfg)
    typer.echo(
        f"  来源：{track.source}（{_SOURCE_LABEL[track.source]}），"
        f"硬字幕：{'已声明' if hardsub else '未声明'}"
    )
    typer.echo("  分类：" + "、".join(f"{k}={v}" for k, v in sorted(counts.items())))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `~/.local/bin/uv run pytest tests/test_cli.py -q`
Expected: 全部 PASS（原有 `test_inspect_prints_summary` 等不受影响——它们只断言子串）。

- [ ] **Step 5: 更新 AGENTS.md**

第 13 行（`ingest` 那条）整行换成：

```markdown
- `ingest`：解析 SRT，生成对白轨（`01_dialogue/`）。**对白轨从哪来是四岔**（`ingest/resolve.py`）：手传 SRT → 视频里的软字幕轨直抽（`E{NN}.embedded.srt`，每次重抽）→ 声明了硬字幕时画面 OCR（`E{NN}.ocr.srt`，按 mtime 复用；只在 `cfg.hardsub_enabled(episode)` 为真时走，跑不了就报错、**不回落到语音转写**）→ 语音转写（`E{NN}.asr.srt`，按 mtime 复用）。四条都归一成一个 SRT 路径，所以 `build_track` 的形态不变；`DialogueTrack.source` 记的是 `"srt"` / `"ocr"` / `"asr"`（`ocr` 跟 `srt` 一样过 OpenCC 繁转简，`asr` 不过）。
```

第 14 行（`translate` 那条）整行换成：

```markdown
- `translate`：按 `source` 分三路（判据是那个字段，不做语言检测）。`"srt"` 跳过，连 `zh/` 目录都不建；`"asr"` 把日语对白逐条译成简体中文（`zh/E{NN}.zh.json` + 交付物 `out/E{NN}.zh.srt`），并把新认出的专有名词并进项目级累积表 `zh/glossary.json`（会被 script 的 prompt 读）；`"ocr"` **不调 LLM、不碰 provider**，按 `translate/lines.py` 的 `passthrough_track`（筛选同 `select_translatable`，zh 取 ingest 已繁转简的原文）照样写那两份产物，不读不写累积表、不写 `zh/E{NN}.usage.json`。
```

第 55 行里 `` `ProjectConfig`（14 个字段）与它的 8 个子 config（`.locale` / `.llm` / `.ingest` / `.credits` / `.signals` / `.validate_script` / `.render` / `.asr`） `` 改成 `` `ProjectConfig`（15 个字段）与它的 9 个子 config（`.locale` / `.llm` / `.ingest` / `.credits` / `.signals` / `.validate_script` / `.render` / `.asr` / `.ocr`） ``。

第 57 行开头 `这 10 个配置模型（项目、8 个子配置、集配置）` 改成 `这 11 个配置模型（项目、9 个子配置、集配置）`。

第 59 行（`AsrConfig` 那段）之后插入一个空行和一段：

```markdown
  `OcrConfig` 有 7 个字段（`enabled` / `sample_fps` / `crop_top` / `center_tolerance` / `similarity` / `min_frames` / `language`），同样有那条**不在代码里的操作约束**：`E{NN}.ocr.srt` 的新鲜度只比源视频 mtime、**刻意不看 `OcrConfig`**，改了 OCR 参数必须自己删那份 SRT（理由同上）。「这一集带不带硬字幕」只有一个判定入口 `ProjectConfig.hardsub_enabled(episode)`：`EpisodeConfig.hardsub` 写了 true/false 就听它的，`None` 跟随 `ocr.enabled`。`config_slices` 在 `hardsub is None` 时**不**把这个键写进 EPISODE 切片——否则升级后 timeline/audio/render 的切片全部被改写、存量集白白重跑。OCR 引擎只有 Apple Vision（pyobjc，optional extra `ocr`，仅 macOS），pyobjc 的 import 只许写在 `ingest/ocr.py` 的 `_recognize` 函数体内。硬字幕片源会把**居中**的 OP/ED staff 字一起认进来（两侧的靠 `center_tolerance` 挡掉），所以这类片源要手填 `op_range` / `ed_range` 或 `credits.default_*`，交给三级回退去剔除。
```

第 108 行里 `共 46 个 Python 文件` 改成本任务完成后 `ls tests/*.py | wc -l` 的实数（实现时核对过是 48），`pyproject.toml` 里是**四个** 改成 `pyproject.toml` 里是**五个**；`asr` 那条 bullet（第 113 行）之后加：

```markdown
- `ocr`：需要 macOS + `uv sync --extra ocr` + 一个真实的硬字幕片源，跑法：`TENMIN_OCR_SAMPLE_VIDEO=<片源路径> uv run pytest -m ocr`。整集识别要跑三四分钟。
```

- [ ] **Step 6: 更新 README.md**

第 6–9 行换成：

```markdown
吃自带字幕的片源，也吃生肉：对白轨有四条来源（手传 SRT / 视频里的软字幕轨 / 画面硬字幕 OCR
/ 语音转写），生肉还会额外交付一份中文字幕。设计文档见
`docs/superpowers/specs/2026-08-31-10min-anime-design.md`，生肉那部分见
`docs/superpowers/specs/2026-09-21-tenmin-asr-translate-design.md`，硬字幕 OCR 见
`docs/superpowers/specs/2026-10-01-hardsub-ocr-design.md`。
```

第 34 行（`` `uv sync --extra asr` 再试」而不是 traceback。只支持 mlx-whisper（Apple Silicon）。 ``）之后插入：

````markdown

画面上烧着字幕的片源（ANi / Baha 这类 `[CHT]` WEB-DL）走画面 OCR，再装一个只有几 MB 的 extra
（Apple Vision，只支持 macOS）：

```bash
uv sync --extra ocr
```
````

第 147 行 `## 对白轨从哪来（三岔）` 改成 `## 对白轨从哪来（四岔）`；第 151–159 行（那个编号列表与它后面那段 `第 1、2 条的对白轨记作…`）换成：

```markdown
1. **手传的 SRT**（`--srt`，或 `project.yaml` 里那一集的 `srt:`）——人明确指定了，不猜。
2. **视频里的软字幕轨**——`ffmpeg` 一条命令抽成 `work/<slug>/srt/E{NN}.embedded.srt`，
   零成本零误差。**每次重抽**（抽取被打断留下的半份 SRT 在语法上合法，看不出是残骸）。
3. **画面硬字幕 OCR**——只在 `project.yaml` 声明了硬字幕时才走（见下文「硬字幕片源」），
   落成 `work/<slug>/srt/E{NN}.ocr.srt`（繁体原文）。一集 24 分钟约 3–4 分钟。
4. **语音转写**——前三条都不成立时才走，落成 `work/<slug>/srt/E{NN}.asr.srt`。
   开跑之前会打一行「没有字幕轨，只能走语音转写」，一集 24 分钟约 3 分钟。

第 1、2 条的对白轨记作 `source: "srt"`，第 3 条记作 `source: "ocr"`，第 4 条记作
`source: "asr"`（在 `01_dialogue/E{NN}.dialogue.json` 里，`tenmin inspect` 的第二行也会
打出来）。这个字段决定两件事：**要不要繁转简**（srt 与 ocr 转；日语过 OpenCC 会被改字，
所以听写路径强制关掉）和 **translate 阶段怎么跑**（见下文）。
```

在第 181 行（`而「换了模型却没重转」打开文件就看得出来。`）之后、`## translate 阶段：中文字幕 + 累积术语表` 之前插入：

````markdown

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
````

第 185–186 行（`**只对 `source == "asr"` 的集跑**…连 `zh/` 目录都不会建。`）换成：

```markdown
**只对 `source == "asr"` 的集真的翻译**（判据是那个字段，不做语言检测）。自带字幕的片源
（`source == "srt"`）本来就是观众读得懂的语言，这一阶段在磁盘上留不下任何痕迹——连 `zh/`
目录都不会建。画面 OCR 来的集（`source == "ocr"`）**不调 LLM**：认出来的字幕繁转简之后原样
交付成 `zh/E{NN}.zh.json` 与 `out/E{NN}.zh.srt`，不碰累积术语表，也不需要 API key。
```

第 264–265 行（阶段表里 ingest 与 translate 两行）换成：

```markdown
| ingest | `01_dialogue/E{NN}.dialogue.json` | 否（可能先跑一次画面 OCR 或语音转写，见上文四岔） |
| translate | `zh/E{NN}.zh.json`、`out/E{NN}.zh.srt`、`zh/glossary.json` | 只对 `source == "asr"` 的集调 LLM；`ocr` 的集直通交付，不调 |
```

第 283 行（`uv run pytest -m asr        # 真跑语音转写，需要 --extra asr + 真视频`）之后加一行：

```bash
TENMIN_OCR_SAMPLE_VIDEO=<片源> uv run pytest -m ocr   # 真跑画面 OCR，需要 macOS + --extra ocr + 硬字幕片源
```

- [ ] **Step 7: 全量 + lint**

Run: `~/.local/bin/uv run pytest tests/ -q`
Expected: 0 failed。

Run: `~/.local/bin/uv run ruff check src/tenmin/cli.py tests/test_cli.py`
Expected: `All checks passed!`

Run: `ls tests/*.py | wc -l`
Expected: 与 AGENTS.md 里刚写的数字一致。

- [ ] **Step 8: Commit**

```bash
git add src/tenmin/cli.py tests/test_cli.py AGENTS.md README.md
git commit -m "feat: inspect 显示对白来源与硬字幕声明，补硬字幕 OCR 文档"
```

---

## 收尾验收（全部任务完成后）

- [ ] `~/.local/bin/uv run pytest tests/ -q`：0 failed，skipped 比基线多 1（`-m ocr` 那条）。
- [ ] `~/.local/bin/uv run tenmin --help`：没装 `ocr` extra 时正常输出。
- [ ] `grep -rn "_is_usable_asr_cache" src tests` 无输出（改名彻底）。
- [ ] 仅 macOS、可选：`~/.local/bin/uv sync --extra ocr`，在 `work/akujo/project.yaml` 加 `ocr: {enabled: true}`，删掉手工登记的 `episodes[].srt`，跑 `~/.local/bin/uv run tenmin run akujo --episode 11 --only ingest,translate`（`--only translate` 在 CLI 层仍要 API key，见偏离第 6 条），确认 `srt/E11.ocr.srt` 生成、`tenmin inspect akujo --episode 11` 显示 `来源：ocr（画面 OCR），硬字幕：已声明`、`out/E11.zh.srt` 是简体。跑完 `~/.local/bin/uv sync` 恢复默认环境。

---

## 自查记录

**Spec 覆盖**（spec 条目 → 任务）：

| spec 条目 | 落点 |
|---|---|
| 决定 1：显式声明（项目级 `ocr.enabled` + 逐集 `episodes[].hardsub`），不自动探测 | Task 1（字段 + `hardsub_enabled`）、Task 5（resolve 只看 `hardsub`） |
| 决定 2：优先级 SRT → 软字幕轨 → OCR → ASR；有字幕轨仍走字幕轨；OCR 跑不了直接报错不回落 | Task 5（分枝位置 + `test_a_declared_hardsub_still_prefers_an_embedded_subtitle_track` + `test_an_ocr_failure_does_not_fall_back_to_transcription`） |
| 决定 3：只用 Apple Vision（`VNRecognizeTextRequest`，pyobjc），extra `ocr`，仅 macOS，延迟 import，可操作报错 | Task 1（extra）、Task 4（`_recognize` + `OCRUnavailableError` + 测试） |
| 决定 4：产物 `srt/E{NN}.ocr.srt` 繁体原文；新鲜度只比视频 mtime、不看 `OcrConfig`；可手改 | Task 4（落盘）、Task 5（`_is_usable_cache` + 手改保护测试）、Task 7（文档） |
| 决定 5：`source` / `kind` 扩成三值；OpenCC 对 ocr 也转；OP/ED 推断对 ocr 不可靠 → 文档写明要手填 | Task 5（Literal ×3 + normalize）、Task 7（AGENTS/README 提醒） |
| 决定 6：ocr 集 translate 不调 LLM，`select_translatable` 选行、zh=原文、glossary 空，写 zh.json + zh.srt（`render_zh_srt`）；不碰术语表、不写 usage；不需 provider；srt 照旧早退 | Task 6 |
| 管线·抽帧：`step = max(1, round(src_fps / sample_fps))`、`probe_frame_rate`；select/crop/scale(1280 宽、偶数高)/format=gray；`-f rawvideo` 管道 + `-fps_mode passthrough`、不落临时图片 | Task 3（`sample_step` / `frame_filter`）、Task 4（argv） |
| 管线·抽帧：新 ffmpeg 封装放 `render/ffmpeg.py`、失败抛 `FFmpegError` 带 stderr 尾部；参数拼法留在 `ocr.py` | Task 2（`run_raw_frames`）、Task 3/4 |
| 管线·抽帧：pts 读 `showinfo`、stdout/stderr 并发读（线程读 stderr）、行数对不上抛 `OCRError` | Task 2（排空线程）、Task 3（`parse_showinfo_pts`）、Task 4（计数核对） |
| 单帧清洗 `frame_text`：居中过滤、必须含汉字、从上到下（Vision y 向上）、省略号统一、strip | Task 3 |
| 合并 `merge_frames`：SequenceMatcher ≥ similarity、`_MAX_GAP_FRAMES = 1`、众数平票取最早、起点首帧、终点末帧 + interval、少于 min_frames 丢弃、去重叠截断；输出复用 `asr.render_srt` + `RawCue` | Task 3、Task 4 |
| 入口 `recognize`：开工提示（约 N 分钟，时长 / 7）、每 10% 进度、0 条抛 `OCRError` 提示 `crop_top` 且不写空文件、同步执行、`atomic.write_text` | Task 4 |
| 异常族 `OCRError` / `OCRUnavailableError`，进 `cli.PIPELINE_ERRORS` | Task 3（定义）、Task 4（登记） |
| `OcrConfig` 7 字段、默认值、范围校验、注释写出处；物理细节留模块常量 | Task 1、Task 3 |
| `EpisodeConfig.hardsub: bool \| None = None` + `cfg.hardsub_enabled(episode)` 单入口（resolve 与 inspect 都调） | Task 1、Task 5（run_ingest 调）、Task 7（inspect 调） |
| 子 config 8 → 9，AGENTS.md 同步 | Task 7 |
| `STAGE_FIELDS["ingest"]` 加 `"ocr"`；hardsub 随 EPISODE 进切片；切片变化不让 `.ocr.srt` 失效 | Task 1（偏离第 3 条：None 时省键） |
| pyproject extra + marker `ocr` + conftest `_GATED_MARKERS` | Task 1 |
| resolve：`_OCR_SUFFIX`、新增 `hardsub` / `ocr_config` 参数、缓存判据泛化、返回 `SubtitleSource(path, "ocr")` | Task 5（偏离第 4、5 条） |
| pipeline：`Paths.ocr_cache`；run_ingest 传路径 / hardsub / 配置；缓存不进 ingest 新鲜度输入 | Task 5（含 `test_ingest_inputs_exclude_the_ocr_cache`） |
| normalize：`source in ("srt", "ocr")` | Task 5 |
| cli：`PIPELINE_ERRORS` 加 `OCRError`；inspect 显示来源与硬字幕声明 | Task 4、Task 7 |
| 文档：AGENTS.md（四选一、`.ocr.srt` 手改与失效、子 config 个数、marker 四→五）；README（用法 + 要手填 OP/ED） | Task 7 |
| 测试：test_ocr（步长 23.976/24/30/60/120 与低帧率→1；pts 顺序/不均匀/对不上；frame_text 五项；merge_frames 七项；recognize 的 monkeypatch、缺依赖提示、0 条不写 dest） | Task 3、Task 4 |
| 测试：test_resolve（四路优先级、有字幕轨走字幕轨、OCR 缓存新鲜度与手改保护、未声明不变、kind 三值）；逐集 hardsub 覆盖项目级 | Task 5（覆盖规则在 test_config 与 test_pipeline 两层各锁一次） |
| 测试：test_normalize（ocr 繁转简）；test_pipeline（ocr translate 不调 provider、产出 zh.json 与简体 zh.srt、只收 speech 行、不碰术语表；srt 仍早退）；test_config*（默认值、边界、覆盖、切片含 ocr）；`@pytest.mark.ocr` 真片源冒烟 | Task 1、Task 4、Task 5、Task 6 |
| 「不做」六项（LLM 校对、跳帧/变化检测、顶部/竖排/多处字幕、非 macOS 引擎、位图轨 OCR、自动探测） | 本计划一处都没做；位图轨的中文报错（`resolve` 的 `_BITMAP_SUBTITLE_MARKER` 分枝）原样保留 |

**占位符扫描**：全文没有 TBD / TODO / 「类似 Task N」/ 「补充错误处理」；每个改代码的步骤都给了完整代码或逐字的「原文 → 新文」。

**类型与命名一致性**：`OcrConfig` 字段名在 Task 1 定义、Task 4（`ocr.sample_fps` 等）与 Task 5（`ocr_config=cfg.ocr`）一致；`recognize(video, dest, *, ocr, ffmpeg_path, ffprobe_path)` 在 Task 4 定义，Task 5 的调用与 `test_the_configured_binaries_and_ocr_config_reach_recognize` 断言的 kwargs 键（`ocr` / `ffmpeg_path` / `ffprobe_path`）一致；`run_raw_frames(args, *, frame_size, on_frame, ffmpeg)` 在 Task 2 定义，Task 4 的调用与假货签名一致；`Paths.ocr_cache` 在 Task 5 定义、`resolve_subtitle_source(ocr_cache=…)` 同名；`passthrough_track` 在 Task 6 定义并只在 Task 6 使用；`_SOURCE_LABEL` 的三个键与 `Literal["srt", "asr", "ocr"]` 一致。

**验证方式**：本计划的全部代码在仓库的一份临时副本里逐任务实现并跑过——每个任务结束时全量 `pytest` 0 failed、改动文件 `ruff check` 全过；在没装 `ocr` extra 的环境下全量也是 0 failed、`tenmin --help` 正常；装上 extra 后用 lavfi + drawtext 造的画面与 spec 那一集真实片源各跑过一遍 `recognize`（后者 387 条、墙钟 3 分 20 秒，spec 的 spike 是 388 条 / 201 秒）。

## 待用户确认

1. **CLI 对 OCR 项目单跑 `--only translate` 仍要 API key**（偏离第 6 条）。要不要把 `cli.run` 的 provider 构造条件改成「本次要跑的集里有 asr 来源才构造」？那要在 CLI 层先读对白轨，本计划按 YAGNI 没做。
2. **行尾单个 `.` / `。` 一律当省略号**（偏离第 2 条）。如果某个字幕组真的在行尾打句号，这条会把句号改成 `…`。
3. **`hardsub: null` 不进 EPISODE 切片**（偏离第 3 条），换来升级后零重渲；代价是切片内容不再是 `EpisodeConfig.model_dump()` 的原样。
