# 管线健壮性与成本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新鲜度只对「这个阶段真正读到的配置」敏感（改 render 参数不再整季重付 LLM、登记新集不影响已有集）；每次 LLM 调用的用量落盘；配置拼错/重复键给中文报错；常见用户错误不再出 traceback、Gemini 路径的错误与退避跟其他 provider 对齐；允许只写 op/ed 的「预填」集条目，登记写回 yaml 时保留注释。

**Architecture:** 新模块 `tenmin.config_slices` 在每次 `run_pipeline` 开跑前把每个阶段读到的那部分**解析后**配置落成 `work/<slug>/.config/*.json`（内容不变不碰文件、首次创建 mtime 设成 epoch 0），阶段的新鲜度输入里用切片替换 `project.yaml`；`_is_fresh` 本身不动。ingest / signals 两个全局阶段改成「内容不变不写」，并配一个「上次跑完」的戳子文件给它们自己判新鲜度。LLM 用量走 `provider.last_usage`（补上 cached token 与跨请求求和），由新模块 `tenmin.script.usage` 在每次 `complete()` 返回后立刻读、落到 `03_script/E{NN}.usage.json` / `zh/E{NN}.usage.json`。配置模型统一继承 `StrictModel(extra="forbid")`，`load_project` 用带重复键检查的 loader 并把所有失败翻成 `ProjectConfigError(ValueError)`。`register_episode` 改用 `ruamel.yaml` round-trip 只改 `episodes` 里对应的那一条。

**Tech Stack:** Python 3.12+（venv 是 3.14）、pydantic v2、PyYAML（读配置）、ruamel.yaml（新增主依赖，只用于写回）、typer、google-genai 2.20（`google.genai.errors.APIError`）、httpx / aiohttp（edge-tts 已把 aiohttp 拽进主依赖树）、pytest + pytest-asyncio（`asyncio_mode = "auto"`）。

---

**Spec:** `docs/superpowers/specs/2026-09-27-pipeline-robustness-design.md`

**基线：** `feat/asr-translate` 上 `uv run pytest tests/ -q` = **1916 passed, 32 skipped**（写本计划时实测）。另外实测过：给 10 个配置模型临时打开 `extra="forbid"` 后全量测试照样 1916 passed，`work/saijo`、`work/akujo` 两份真实 `project.yaml` 也没有任何模型不认识的键 —— Task 1 不会误伤存量。

## Global Constraints

- **`uv run` 不要套 rtk**：pytest 与 pipeline 命令一律裸 `uv run <cmd>`，绝不写 `rtk pytest` / `rtk proxy uv run`。裸的 `rtk ls` / `rtk grep` / `rtk git` 安全。
- **变异自检一律带 `PYTHONDONTWRITEBYTECODE=1`**（见 AGENTS.md：等长替换 + 同一整秒会让 pyc 假绿）。
- **src/ 里禁止 `assert`**（`test_no_assert_statements_in_src`）。前置条件写成 `if …: raise`。
- **注释与 docstring 禁止 `file.py:123` 形式的行号引用**（`test_no_line_number_cross_references`），要指路写模块名/函数名。
- **src/ 与 tests/ 的注释里禁止内部任务代号**（`test_no_internal_task_codes_in_comments`，正则含 `Task \d+`）。本计划里的「Task N」只能出现在计划文档里，**不许**抄进代码注释。
- **所有文本 IO 显式 `encoding=`**；**所有产物写盘走 `tenmin.atomic`**（`test_artifact_writes_go_through_atomic` 扫 src/ 里的 `.write_text(` / `.write_bytes(` / `copyfile(`，只豁免 `atomic.*` 接收者与 `atomic.py` 自身）。`os.utime` 不在审计名单里。
- **本项目不引入 logging**。诊断走 warnings 列表或异常消息。
- **`_ARTIFACTS` 的既有条目一个字符都不许改**，只许新增；新增按集产物要同步进 `tests/test_pipeline.py` 的 `FROZEN_LAYOUT`（`test_paths_exposes_exactly_the_frozen_artifacts` 双向锁）。项目级路径（`.config/` 下的切片与戳子）**不进** `Paths`，放在 `tenmin.config_slices` 里，免得撞上那条「Paths 的公开 callable == FROZEN_LAYOUT」的断言。
- **`tenmin.config_slices` 不许 import `tenmin.pipeline`**（pipeline 反过来 import 它，会成环）。
- **退避的 sleep / 抖动只许走 `llm._sleep` / `llm._rand`**，测试用 `tests/test_llm.py` 的 `sleeps` fixture monkeypatch 掉；不许在新路径上直接 `asyncio.sleep`。
- 提交信息沿用仓库风格：`feat: …` / `fix: …` / `test: …` / `docs: …`，中文。每个 Task 结束时全量测试必须是绿的。

## 与 spec 的偏离（实现时以本计划为准）

逐个 grep 过各阶段真实读到的字段之后，spec 的初稿映射表有几处跟代码对不上，按「以代码为准、拿不准就多挂、但不为 LLM 阶段白挂」修正：

1. **`render.font_size` 不只是 render 的输入。** 字号进的是 timeline 阶段写的 `05_timeline/E{NN}.ass`（`run_timeline` → `render_ass(font_size=…)`，外加 `check_cue_legibility` 的行数告警）。只挂在 render 上就是「改了字号，成片烧的还是旧 .ass」—— 恰好是 spec 自己说的「该重跑没重跑」。所以它挂在 timeline（真消费者）和 render（多挂）上；改字号的实际重跑集合是 **timeline → audio → render**（timeline 会重写 `timeline.json`，audio 以它为输入）。spec 测试清单里「改 `render.font_size` 只让 render 过期」改成两条：`render.crf`（只有 render 读）→ 只重跑 render；`render.font_size` → 只重跑 timeline/audio/render。
2. **ingest 的切片补上 `glossary` 与 `show`。** `run_ingest` 把 `cfg.glossary` 传进 `build_track`（`clean_text` 的术语替换），把 `cfg.show` 当 `show_title`（`is_credits` 的剧名比对）。
3. **translate 的切片只挂它真读的 6 个 llm 字段 + `glossary`，不挂整个 `llm`、也不挂本集 `EpisodeConfig`。** `translate_track` 只读 `cfg.glossary` 与 `cfg.llm.max_attempts`（后者在排除表里）；provider 由 `provider/model/base_url/thinking/temperature/max_output_tokens` 构造。挂整个 `llm` 会让「调 `budget_tolerance`」白重付一集翻译费；挂 `EpisodeConfig` 会让「源片挪了个位置、内容没变」也白重付 —— 对白轨是 translate 唯一的数据通路，本集的 op/ed/源片真变了会先经 ingest 改写对白轨、照样让它重跑。
4. **timeline / audio 也挂本集 `EpisodeConfig`**（spec 只给 render 挂了）：两者都读 `cfg.video_path(episode)`（timeline 探时长与帧率，audio 取原声）。
5. **`slug` 进排除表。** 它只决定 `work/<slug>/` 这个目录名，不进任何产物；完整性测试要求每个叶子都有归属，spec 的排除表漏了它。
6. **ingest / signals 额外引入一个「上次跑完」的戳子（`.config/ingest.done`、`.config/signals.done`）。** spec 只要求「内容相同就不写」。但只做这一半有个坑：改一个不影响产出的阈值（比如没有手填区间时的 `credits.manual_window_margin`）会让 ingest 重跑、产物一字不变、mtime 保持旧值 —— 于是产物永远比切片旧，ingest **每次**运行都重跑（signals 同理：只有一集的对白轨变了，其余集的 signals.json 不写，最旧的产物永远比那一集的对白轨旧）。戳子让这两个阶段按「上次跑完」判新鲜度，产物只查齐不齐、空不空；戳子不存在（升级前的项目）时退回按产物判，升级后第一次运行不白跑。`_is_fresh` 本身不改。
7. **`usage.json` 顶层是 `{"episode": N, "calls": [...]}` 而不是裸列表**，跟同目录的 `warnings.json`（`{"episode", "warnings"}`）一个形状；每条额外带 `requests`（这一轮底下实际发了几次 HTTP 请求，含传输层重试与 schema 修复轮）与 `ok`（这次调用成没成）。轮次类型的取值是 `draft` / `validation_retry` / `budget_rewrite` / `translate` / `translate_repair`。
8. **批处理跳过预填条目的那句提示走 warnings 通道**（CLI 末尾打 `[warn] 第 3 集还没有 video，已跳过`），跟本项目「诊断只走 warnings」的约定一致；只在批处理模式下提示。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/tenmin/config.py` | 改 | `StrictModel` 基类；10 个配置模型改继承它；`ProjectConfigError`、`DuplicateKeyError`、`_UniqueKeyLoader`；`load_project` 的中文报错；删 `_require_a_source`、加 `EpisodeConfig.has_source`；三处过期注释 |
| `src/tenmin/cli.py` | 改 | `run` / `inspect` 把 `load_project`、`register_episode`、`_find_episode` 挪进 `PIPELINE_ERRORS` 的网；`inspect` 列预填条目 |
| `src/tenmin/script/llm.py` | 改 | Gemini：`APIError` → `LLMHTTPError`、连接类异常 → `LLMTransportError`、传输层退避；`LLMUsage.cached_tokens`；`_StreamTally.usages` 跨请求求和；两条 provider 路径都填 cached token |
| `src/tenmin/script/usage.py` | **新建** | `UsageRecord`、`track_call`（每次 `complete()` 返回后立刻读 `last_usage`）、`write_usage` |
| `src/tenmin/script/single.py` | 改 | `generate_script(..., usage=)`，按轮次记用量 |
| `src/tenmin/translate/lines.py` | 改 | `translate_track(..., usage=)` |
| `src/tenmin/config_slices.py` | **新建** | `STAGE_FIELDS` 映射表、`EXCLUDED` 排除表、`slice_payload` / `slice_path` / `write_slice` / `write_slices` / `stamp_path` / `touch_stamp` |
| `src/tenmin/atomic.py` | 改 | `write_text_if_changed` |
| `src/tenmin/pipeline.py` | 改 | 两个 usage 产物路径；`_active_episodes`；`_find_episode` 认预填条目；`register_episode` 查 SRT 存在性 + ruamel 写回；`run_pipeline` 写切片、按切片判新鲜度、批处理跳过预填条目；`run_ingest` / `run_signals` 内容不变不写；`_is_fresh_stamped` |
| `pyproject.toml` / `uv.lock` | 改 | `uv add ruamel.yaml` |
| `AGENTS.md` | 改 | config / pipeline / llm 段落、测试文件数 |
| `tests/test_config.py` | 改 | 严格校验、重复键、预填条目 |
| `tests/test_cli.py` | 改 | 无 traceback 的错误路径、预填条目的 CLI 行为、init 注释保留 |
| `tests/test_llm.py` | 改 | Gemini 错误与退避、usage 求和与 cached token |
| `tests/test_single.py` | 改 | 按轮次记用量 |
| `tests/test_translate_lines.py` | 改 | 翻译轮次记用量 |
| `tests/test_atomic.py` | 改 | `write_text_if_changed` |
| `tests/test_config_slices.py` | **新建** | 映射完整性、排除表、切片内容、epoch 0 |
| `tests/test_pipeline.py` | 改 | usage 落盘、预填条目、ruamel 写回、切片接线、跳过相同写入、端到端 |
| `tests/test_source_hygiene.py` | 改 | 一段过期注释 |

### 已核实的「阶段 → 配置字段」映射（Task 7 的 `STAGE_FIELDS` 就是这张表）

| 阶段 | 切片内容 | 代码里的读取点 |
|---|---|---|
| ingest（按集） | `locale`、`ingest`、`credits`、`asr`、`glossary`、`show`、本集 `EpisodeConfig` | `run_ingest`：`resolve_subtitle_source(asr_config=cfg.asr, …)`、`build_track(glossary=cfg.glossary, convert_traditional=cfg.locale…, show_title=cfg.show, op_range/ed_range, ingest=cfg.ingest, credits=cfg.credits)`；`_source_duration` 读本集 video |
| translate（按集） | `glossary`、`llm.provider`、`llm.model`、`llm.base_url`、`llm.thinking`、`llm.temperature`、`llm.max_output_tokens` | `translate_track`：`effective_glossary(…, cfg.glossary)`、`cfg.llm.max_attempts`（排除）；`build_provider` 读那 6 个字段 |
| signals（项目级） | `signals` | `run_signals` → `build_report(track, cfg=cfg.signals)` |
| script（按集） | `llm`（扣排除项）、`validate_script`、`target_seconds`、`mode`、`glossary`、`show`、`render.rate` | `build_user_prompt`（show / target_seconds / llm.budget_tolerance / render.rate / glossary）、`generate_script`（validation_retries / budget_rewrite_rounds / budget_tolerance、`validate_script`）、`to_script`（show / target_seconds）；`mode` 只有 `run_pipeline` 入口在判，按「拿不准就多挂」留着 |
| docgen（按集） | `render.rate` | `render_table` / `render_narration` 其实**不读配置**（估算列用 script.json 里存好的 est_seconds），按 spec 多挂 |
| voice（按集） | `render.voice`、`render.rate` | `build_tts_engine`（voice / rate）、`synthesize_track(rate=…)` |
| timeline（按集） | `render.font_size`、`render.subtitle_font_name`、`render.width`、`render.height`、`render.subtitle_max_lines`、`render.subtitle_min_seconds`、`render.drift_tolerance`、本集 `EpisodeConfig` | `run_timeline`：`build_timeline(cfg=cfg.render)`（只读 drift_tolerance）、`check_cue_legibility`、`render_ass`、`cfg.video_path(episode_cfg)` |
| audio（按集） | `render.duck_db`、`render.fade_out_seconds`、`render.outro_card_seconds`、`render.audio_codec`、`render.audio_bitrate`、`render.limiter_ceiling`、本集 `EpisodeConfig` | `run_audio` → `mix_audio(...)` |
| render（按集） | `render.video_encoder`、`render.width`、`render.height`、`render.crf`、`render.preset`、`render.tune`、`render.videotoolbox_bitrate`、`render.fade_out_seconds`、`render.outro_card_seconds`、`render.outro_message`、`render.outro_font_name`、`render.font_size`、`render.subtitle_font_name`、`show`、本集 `EpisodeConfig` | `run_render` → `render_video(...)`（`outro_title` 用 `cfg.show`）；font_size / subtitle_font_name 是多挂的（真消费者是 timeline 写的 .ass） |

**排除表（`EXCLUDED`，不影响产物内容、从不触发重跑）：** `slug`、`llm.timeout_seconds`、`llm.read_timeout_seconds`、`llm.total_timeout_seconds`、`llm.transport_max_attempts`、`llm.max_attempts`、`llm.script_concurrency`、`render.tts_max_attempts`、`render.tts_concurrency`、`render.tts_proxy`、`render.tts_connect_timeout`、`render.tts_receive_timeout`、`render.tts_chunk_timeout_seconds`、`render.ffmpeg_path`、`render.ffprobe_path`。已核对这 14 个 llm/render 字段在 `config.py` 里全部存在，且没有别的同类旋钮（`RenderConfig` 30 个字段 = 22 个挂阶段 + 8 个排除；`LLMConfig` 15 个 = 9 个挂阶段 + 6 个排除）。

`llm.max_attempts` 的归属按 spec 放进排除表：它决定「这次调用成不成」（schema 修复轮数），不决定「成了之后的稿子长什么样」—— 首轮就合 schema 的输出跟第 3 轮修出来的合 schema 输出在产物形态上没有系统性差别。`validation_retries` / `budget_rewrite_rounds` 不一样：它们决定「采纳哪一版」（`_pick_better` 两版择优），所以留在 script 的切片里。

---
### Task 1: 配置模型一律 `extra="forbid"`，拼错字段与重复键给中文报错

**Files:**
- Modify: `src/tenmin/config.py`（imports；在 `_validate_open_credit_range` 之后新增 `StrictModel`；10 个模型的基类；文件末尾 `load_project` 前新增错误类型、loader 与报错翻译；重写 `load_project`）
- Modify: `tests/test_config.py`（`test_load_project_rejects_unknown_mode` 改断言；末尾追加新测试）
- Modify: `tests/test_cli.py`（`test_init_output_has_no_keys_the_models_do_not_know` 的 docstring 去掉过期描述）

- [ ] **Step 1: 写失败的测试**

`tests/test_config.py` 顶部 import 块改成：

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from tenmin.config import (
    AsrConfig,
    CreditsConfig,
    EpisodeConfig,
    IngestConfig,
    LLMConfig,
    LocaleConfig,
    ProjectConfig,
    ProjectConfigError,
    RenderConfig,
    Settings,
    SignalsConfig,
    ValidateConfig,
    load_project,
)
```

把 `test_load_project_rejects_unknown_mode` 整个换成：

```python
def test_load_project_rejects_unknown_mode(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(MINIMAL.replace("slug: demo", "slug: demo\nmode: whatever"), encoding="utf-8")
    with pytest.raises(ProjectConfigError) as excinfo:
        load_project(path)
    assert "project.yaml 里 mode 的值不合法" in str(excinfo.value)
```

文件末尾追加：

```python
# --- 严格校验：拼错的键必须报错，不能静默退回默认值 --------------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "project.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_field_names_its_full_path_and_a_suggestion(tmp_path):
    path = _write(tmp_path, MINIMAL + "render:\n  font_sise: 99\n")
    with pytest.raises(ProjectConfigError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "project.yaml 里 render.font_sise 不是已知字段" in message
    assert "是不是想写 font_size" in message


def test_unknown_field_inside_an_episode_names_the_list_index(tmp_path):
    path = _write(
        tmp_path,
        "show: 某番\nslug: demo\nepisodes:\n"
        "  - number: 1\n    srt: srt/E01.srt\n    op_rang: [1, 2]\n",
    )
    with pytest.raises(ProjectConfigError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "episodes[0].op_rang 不是已知字段" in message
    assert "是不是想写 op_range" in message


def test_unknown_top_level_field_is_rejected(tmp_path):
    path = _write(tmp_path, MINIMAL + "targt_seconds: 200\n")
    with pytest.raises(ProjectConfigError) as excinfo:
        load_project(path)
    assert "targt_seconds 不是已知字段，是不是想写 target_seconds" in str(excinfo.value)


def test_a_bad_value_names_the_field(tmp_path):
    path = _write(tmp_path, MINIMAL + "target_seconds: -1\n")
    with pytest.raises(ProjectConfigError) as excinfo:
        load_project(path)
    assert "project.yaml 里 target_seconds 的值不合法" in str(excinfo.value)


def test_a_yaml_syntax_error_is_a_project_config_error(tmp_path):
    path = _write(tmp_path, "show: [没闭合\nslug: demo\n")
    with pytest.raises(ProjectConfigError) as excinfo:
        load_project(path)
    assert "project.yaml 不是合法的 YAML" in str(excinfo.value)


def test_duplicate_top_level_keys_are_rejected(tmp_path):
    """PyYAML 默认让后写的静默盖掉先写的：两段 render: 只剩第二段生效。"""
    path = _write(
        tmp_path,
        "show: 某番\nslug: demo\nrender:\n  font_size: 40\nrender:\n  crf: '18'\n",
    )
    with pytest.raises(ProjectConfigError) as excinfo:
        load_project(path)
    message = str(excinfo.value)
    assert "render" in message
    assert "出现了两次" in message
    assert "第 3 行" in message
    assert "第 5 行" in message


def test_duplicate_nested_keys_are_rejected(tmp_path):
    path = _write(
        tmp_path, "show: 某番\nslug: demo\nrender:\n  font_size: 40\n  font_size: 50\n"
    )
    with pytest.raises(ProjectConfigError) as excinfo:
        load_project(path)
    assert "font_size" in str(excinfo.value)


def test_project_config_error_is_a_value_error():
    """cli.PIPELINE_ERRORS 靠 ValueError 那条网兜住它。"""
    assert issubclass(ProjectConfigError, ValueError)


@pytest.mark.parametrize(
    ("model", "required"),
    [
        (ProjectConfig, {"show": "某番", "slug": "demo"}),
        (EpisodeConfig, {"number": 1, "srt": "srt/E01.srt"}),
        (LocaleConfig, {}),
        (LLMConfig, {}),
        (IngestConfig, {}),
        (CreditsConfig, {}),
        (SignalsConfig, {}),
        (ValidateConfig, {}),
        (RenderConfig, {}),
        (AsrConfig, {}),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_every_config_model_forbids_extra_fields(model, required):
    with pytest.raises(ValidationError):
        model.model_validate({**required, "no_such_field": 1})


def test_settings_still_ignores_unrelated_env_vars(monkeypatch):
    """.env 里常年住着别的程序的变量，Settings 不许跟着变严。"""
    monkeypatch.setenv("TENMIN_SOMETHING_ELSE", "x")
    Settings()
```

`tests/test_cli.py` 里 `test_init_output_has_no_keys_the_models_do_not_know` 的 docstring 换成：

```python
    """独立于 pydantic 的 extra 设置，直接查生成的 yaml 里有没有模型不认识的键。

    config.py 的模型现在是 extra="forbid"（能 load 回来就说明每个键都认识），这条留着
    做一道不依赖 pydantic 配置的独立检查。
    """
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py -q`
Expected: 收集阶段 `ImportError: cannot import name 'ProjectConfigError'`。

- [ ] **Step 3: 实现**

`src/tenmin/config.py` 的 import 块换成：

```python
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from tenmin.models import STRENGTH_MAX, STRENGTH_MIN
```

在 `_validate_open_credit_range` 之后、`class EpisodeConfig` 之前插入：

```python
class StrictModel(BaseModel):
    """project.yaml 里全部模型的公共基类，只做一件事：extra="forbid"。

    pydantic 默认的 extra="ignore" 会把 `render: {font_sise: 99}` 这种拼错静默吞掉：
    值退回默认、旋钮成了哑的，而且没有任何提示。阶段间产物的模型早就是这么做的
    （models._StageModel），配置这边补齐同一条。报错怎么翻成人话见 load_project。

    Settings 刻意不继承它：.env 里常年住着别的程序的变量，那边必须保持 extra="ignore"。
    """

    model_config = ConfigDict(extra="forbid")
```

把下面 10 个类的基类从 `BaseModel` 改成 `StrictModel`（只改括号里那个名字，类体不动）：`EpisodeConfig`、`LocaleConfig`、`LLMConfig`、`IngestConfig`、`CreditsConfig`、`SignalsConfig`、`ValidateConfig`、`RenderConfig`、`AsrConfig`、`ProjectConfig`。

把文件里现有的 `def load_project(...)` 整个删掉，在 `DEFAULT_LLM = LLMConfig()` 之后、`class Settings` 之前插入：

```python
class ProjectConfigError(ValueError):
    """project.yaml 读不进来：YAML 语法错、同层键重复、字段拼错或取值不合法。

    继承 ValueError：cli.PIPELINE_ERRORS 里那条 ValueError 网就兜得住它，用户看到的是
    一行中文而不是 traceback。消息一律点名文件名与字段的完整路径。
    """


class DuplicateKeyError(yaml.YAMLError):
    """同一层映射里出现了两次同一个键。

    PyYAML 默认让后写的静默盖掉先写的 —— 两段 `render:` 只剩第二段生效，第一段里调的
    旋钮全成了哑的。
    """

    def __init__(self, key: object, first_line: int, second_line: int) -> None:
        self.key = key
        self.first_line = first_line
        self.second_line = second_line
        super().__init__(
            f"键 {key} 在同一层出现了两次（第 {first_line} 行与第 {second_line} 行），"
            "后写的会把先写的整段盖掉，请合并成一处"
        )


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader + 构造 mapping 时检查重复键。

    register_episode 写回 yaml 用的是 ruamel 的 round-trip 模式，它本身就拒绝重复键；
    这里让读配置的这条路跟它一致，同一份文件不会出现「读得进来、写不回去」。

    `<<` 合并键跳过（它合法地可以出现多次）；不可哈希的键交给父类去报它自己的错。
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: dict[Any, int] = {}
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                first = seen.get(key)
            except TypeError:
                continue
            line = key_node.start_mark.line + 1
            if first is not None:
                raise DuplicateKeyError(key, first, line)
            seen[key] = line
        return super().construct_mapping(node, deep=deep)


def _field_path(loc: tuple[int | str, ...]) -> str:
    """pydantic 的 loc → `render.font_sise` / `episodes[0].op_rang`。"""
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out = f"{out}.{part}" if out else str(part)
    return out


def _submodel(annotation: Any) -> type[BaseModel] | None:
    """字段注解里的子模型（`RenderConfig`、`list[EpisodeConfig]` 里的那个）。"""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _submodel(arg)
        if found is not None:
            return found
    return None


def _known_fields(loc: tuple[int | str, ...]) -> list[str]:
    """出错的那个键所在的模型认识哪些字段名。给拼写建议用。"""
    model: type[BaseModel] | None = ProjectConfig
    for part in loc[:-1]:
        if isinstance(part, int):
            continue
        if model is None or part not in model.model_fields:
            return []
        model = _submodel(model.model_fields[part].annotation)
    return list(model.model_fields) if model is not None else []


def _describe_validation_error(name: str, error: ValidationError) -> str:
    """把 pydantic 的逐条报错翻成「文件名 里 字段路径 ……」的中文，一条一行。"""
    lines: list[str] = []
    for item in error.errors():
        loc = tuple(item["loc"])
        where = _field_path(loc) or "顶层"
        if item["type"] == "extra_forbidden":
            message = f"{name} 里 {where} 不是已知字段"
            close = difflib.get_close_matches(str(loc[-1]), _known_fields(loc), n=1)
            if close:
                message += f"，是不是想写 {close[0]}？"
        elif item["type"] == "missing":
            message = f"{name} 里缺少必填字段 {where}"
        else:
            message = f"{name} 里 {where} 的值不合法：{item['msg']}"
        lines.append(message)
    return "\n".join(lines)


def load_project(path: Path) -> ProjectConfig:
    """读 project.yaml。任何「读不进来」都翻成 ProjectConfigError（文件不存在除外，
    那条保留 FileNotFoundError 的既有契约）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到项目配置: {path}")
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader) or {}
    except DuplicateKeyError as error:
        raise ProjectConfigError(f"{path.name} 里{error}") from error
    except yaml.YAMLError as error:
        raise ProjectConfigError(f"{path.name} 不是合法的 YAML：{error}") from error
    try:
        cfg = ProjectConfig.model_validate(data)
    except ValidationError as error:
        raise ProjectConfigError(_describe_validation_error(path.name, error)) from error
    return cfg.bind_root(path.parent)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -q`
Expected: 全部 PASS。

- [ ] **Step 5: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿（基线 1916 + 本任务新增）。

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/config.py tests/test_config.py tests/test_cli.py && rtk git commit -m "feat: 配置模型一律 extra=forbid，拼错字段与重复键给中文报错"
```

---

### Task 2: CLI 把读配置与登记挪进错误网，登记不存在的 SRT 给中文报错

**Files:**
- Modify: `src/tenmin/cli.py`（`run` 开头约 259–284 行；`inspect` 开头约 364 行）
- Modify: `src/tenmin/pipeline.py`（`register_episode` 函数体开头）
- Test: `tests/test_cli.py`、`tests/test_pipeline.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_cli.py` 末尾追加：

```python
# --- 读配置与登记也在错误网里：一行中文红字，不出 traceback ----------------


def _broken_project(work, text: str) -> None:
    root = work / "saijo"
    root.mkdir(parents=True)
    (root / "project.yaml").write_text(text, encoding="utf-8")


def test_run_reports_a_yaml_syntax_error_without_a_traceback(work):
    _broken_project(work, "show: [没闭合\nslug: saijo\n")
    result = runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"])
    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "不是合法的 YAML" in out(result)


def test_run_reports_an_unknown_field_without_a_traceback(work):
    _broken_project(work, "show: 某番\nslug: saijo\nrender:\n  font_sise: 99\n")
    result = runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"])
    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "render.font_sise 不是已知字段" in out(result)
    assert "font_size" in out(result)


def test_run_reports_a_missing_srt_without_a_traceback(work, golden_srt_path, tmp_path):
    root = _bootstrap(work, golden_srt_path)
    before = (root / "project.yaml").read_text(encoding="utf-8")
    video = tmp_path / "e03.mkv"
    video.write_bytes(b"fake")
    result = runner.invoke(
        app,
        [
            "run", "saijo", "--work-dir", str(work),
            "--episode", "3", "--srt", str(tmp_path / "nope.srt"), "--video", str(video),
            "--only", "ingest",
        ],
    )
    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "找不到要登记的字幕文件" in out(result)
    assert (root / "project.yaml").read_text(encoding="utf-8") == before


def test_inspect_reports_a_broken_project_without_a_traceback(work):
    _broken_project(work, "show: 某番\nslug: saijo\nrender:\n  font_sise: 99\n")
    result = runner.invoke(app, ["inspect", "saijo", "--work-dir", str(work), "--episode", "2"])
    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "render.font_sise 不是已知字段" in out(result)
```

`tests/test_pipeline.py` 末尾追加：

```python
def test_register_episode_rejects_a_missing_srt_before_touching_anything(tmp_path):
    cfg = _project_config(tmp_path)
    before = cfg.config_path.read_text(encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="找不到要登记的字幕文件"):
        register_episode(
            cfg, episode=3, srt=tmp_path / "nope.srt", video=tmp_path / "e03.mkv"
        )

    assert cfg.config_path.read_text(encoding="utf-8") == before
    assert [e.number for e in cfg.episodes] == [2]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli.py -q -k "without_a_traceback" && uv run pytest tests/test_pipeline.py::test_register_episode_rejects_a_missing_srt_before_touching_anything -q`
Expected: CLI 四条 FAIL（`_graceful` 为假：异常逃到顶层）；pipeline 那条 FAIL（报的是 shutil 的英文 errno，不匹配 `找不到要登记的字幕文件`）。

- [ ] **Step 3: 实现**

`src/tenmin/pipeline.py` 的 `register_episode` 里，在 docstring 之后、`relative_srt: Path | None = None` 之前插入：

```python
    # 在改动任何东西之前拦下：原来这里一路走到 atomic.copy_file 才炸，报的是 shutil 的
    # 英文 errno，而且那时 project.yaml 还没被碰过纯属侥幸（copy 在写 yaml 之前）。
    if srt is not None and not Path(srt).is_file():
        raise FileNotFoundError(f"找不到要登记的字幕文件：{srt}")
```

`src/tenmin/cli.py` 的 `run`：把 docstring 之后到 `only_stages = _parse_only(only)` 之前的那一段（从 `cfg = load_project(_project_file(work_dir, slug))` 到 `_find_episode` 那个 try/except 结束）整段换成：

```python
    project_file = _project_file(work_dir, slug)

    # 三条规则（原来是「--srt 与 --video 必须一起传」那一条对称的规则）：
    # - 传 --srt 必须配 --video：视频是 render 阶段的硬需求，只有字幕出不了片。
    # - 只传 --video 合法，这就是生肉入口（对白轨靠软字幕轨抽取或语音转写拿）。
    # - 传了 --video 就必须说这是第几集，否则没法登记进 project.yaml。
    # 这两条只看 flag，放在读配置之前：flag 组合错了没必要先去解析一遍 yaml。
    if srt is not None and video is None:
        typer.secho("传 --srt 时必须同时传 --video", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 消息只点 --video：上面那条已经把「只有 --srt」拦掉了，所以能走到这里必然带着
    # --video（--srt 可有可无）。原文案「传 --srt/--video 时」会在 --video 单飞的场合
    # 点一个用户压根没用的 flag。
    if video is not None and episode is None:
        typer.secho("传 --video 时必须同时传 --episode", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 读配置、登记、查集号都进 PIPELINE_ERRORS 的网：yaml 语法错、字段拼错、要登记的
    # 字幕不存在、集号没注册，都该是一行中文红字，而不是一整页 traceback。
    try:
        cfg = load_project(project_file)
        if video is not None:
            cfg = register_episode(cfg, episode=episode, srt=srt, video=video)
        if episode is not None:
            _find_episode(cfg, episode)
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
```

`inspect` 里 `cfg = load_project(_project_file(work_dir, slug))` 那一行换成：

```python
    project_file = _project_file(work_dir, slug)
    try:
        cfg = load_project(project_file)
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cli.py tests/test_pipeline.py -q`
Expected: 全部 PASS（含既有的 `test_run_episode_not_registered_fails`、`test_run_srt_without_video_fails` 等 —— 它们用 `out()` 合并 stdout/stderr，红字改走 stderr 不影响）。

- [ ] **Step 5: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/cli.py src/tenmin/pipeline.py tests/test_cli.py tests/test_pipeline.py && rtk git commit -m "fix: 读配置与登记挪进错误网，登记不存在的字幕给中文报错"
```

---

### Task 3: Gemini 路径包装 `APIError`、接上传输层退避

**Files:**
- Modify: `src/tenmin/script/llm.py`（imports；`_RETRYABLE_TRANSPORT_ERRORS` 之后新增 `_GEMINI_TRANSPORT_ERRORS`；`_check_gemini_finish` 之后新增两个 helper；重写 `GeminiProvider`；`build_provider` 的 gemini 分枝）
- Modify: `src/tenmin/config.py`（`LLMConfig.script_concurrency` 注释的第 1 条依据）
- Test: `tests/test_llm.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_llm.py` 末尾追加（`sleeps` fixture 已在本文件定义；`_FakeGeminiResponse` 也在本文件）：

```python
# --- Gemini：APIError 包成 LLMHTTPError，传输层退避跟 OpenAI 兼容那条同一套 ------


def _fake_gemini_sequence(monkeypatch, provider, script: list) -> list[dict]:
    """按序回放：元素是异常就抛出来，否则当成响应返回。"""
    calls: list[dict] = []
    queue = list(script)

    class FakeModels:
        async def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": contents})
            assert queue, "假 Gemini 的脚本已用尽"
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, _FakeGeminiResponse):
                return item
            return _FakeGeminiResponse(item)

    monkeypatch.setattr(
        provider, "_client", SimpleNamespace(aio=SimpleNamespace(models=FakeModels()))
    )
    return calls


def _api_error(code: int, message: str = "boom"):
    from google.genai import errors

    status = {400: "INVALID_ARGUMENT", 429: "RESOURCE_EXHAUSTED", 503: "UNAVAILABLE"}
    return errors.APIError(
        code,
        {"error": {"code": code, "message": message, "status": status.get(code, "UNKNOWN")}},
    )


@pytest.mark.asyncio
async def test_gemini_wraps_a_client_error_and_does_not_retry_it(monkeypatch, sleeps):
    provider = GeminiProvider(api_key="fake-key")
    calls = _fake_gemini_sequence(
        monkeypatch, provider, [_api_error(400, "API key not valid")]
    )

    with pytest.raises(LLMHTTPError) as excinfo:
        await provider.complete("SYS", "USR", Toy)

    assert excinfo.value.status_code == 400
    assert "API key not valid" in str(excinfo.value)
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_gemini_backs_off_on_429_then_succeeds(monkeypatch, sleeps):
    provider = GeminiProvider(api_key="fake-key")
    calls = _fake_gemini_sequence(
        monkeypatch, provider, [_api_error(429), _api_error(429), '{"value": 3}']
    )

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=3)
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_gemini_gives_up_on_5xx_after_transport_max_attempts(monkeypatch, sleeps):
    provider = GeminiProvider(api_key="fake-key", transport_max_attempts=3)
    calls = _fake_gemini_sequence(monkeypatch, provider, [_api_error(503)] * 3)

    with pytest.raises(LLMHTTPError) as excinfo:
        await provider.complete("SYS", "USR", Toy)

    assert excinfo.value.status_code == 503
    assert excinfo.value.attempts == 3
    assert "已尝试 3 次" in str(excinfo.value)
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_gemini_retries_an_httpx_connection_error_then_wraps_it(monkeypatch, sleeps):
    import httpx

    provider = GeminiProvider(api_key="fake-key", transport_max_attempts=2)
    error = httpx.ConnectError("connection refused")
    calls = _fake_gemini_sequence(monkeypatch, provider, [error, error])

    with pytest.raises(LLMTransportError) as excinfo:
        await provider.complete("SYS", "USR")

    assert "已尝试 2 次" in str(excinfo.value)
    assert "ConnectError" in str(excinfo.value)
    assert len(calls) == 2
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_gemini_retries_an_aiohttp_connection_error(monkeypatch, sleeps):
    """装了 aiohttp 时 google-genai 走 aiohttp（edge-tts 会把它拽进来），两族都得认。"""
    import aiohttp

    provider = GeminiProvider(api_key="fake-key")
    calls = _fake_gemini_sequence(
        monkeypatch,
        provider,
        [aiohttp.ClientConnectionError("reset by peer"), '{"value": 5}'],
    )

    assert await provider.complete("SYS", "USR", Toy) == Toy(value=5)
    assert len(calls) == 2
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_gemini_does_not_retry_an_unrelated_exception(monkeypatch, sleeps):
    provider = GeminiProvider(api_key="fake-key")
    _fake_gemini_sequence(monkeypatch, provider, [KeyError("sdk bug")])

    with pytest.raises(KeyError):
        await provider.complete("SYS", "USR", Toy)
    assert sleeps == []


def test_build_provider_wires_transport_max_attempts_into_gemini():
    provider = build_provider(
        LLMConfig(transport_max_attempts=2), Settings(gemini_api_key="k")
    )
    assert provider.transport_max_attempts == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_llm.py -q -k "gemini_wraps or gemini_backs_off or gemini_gives_up_on_5xx or gemini_retries or gemini_does_not_retry or transport_max_attempts_into_gemini"`
Expected: FAIL —— `GeminiProvider.__init__` 不认识 `transport_max_attempts`；其余几条抛的是裸 `APIError` / `ConnectError`，而且一次都不重试。

- [ ] **Step 3: 实现**

`src/tenmin/script/llm.py` 的第三方 import 段改成：

```python
import aiohttp
import httpx
from pydantic import BaseModel, ValidationError
```

在 `_RETRYABLE_TRANSPORT_ERRORS = (...)` 之后插入：

```python
# Gemini 那条路的连接类异常。google-genai 在 aiohttp 可用时（edge-tts 是主依赖，会把它
# 拽进来）走 aiohttp，否则走 httpx，所以两族都得认。TimeoutError 是 aiohttp 的
# ClientTimeout 到点时抛的（asyncio.TimeoutError 在 3.11 起就是内置 TimeoutError）。
# 刻意不收 aiohttp.ClientError 这个大父类：ClientResponseError 之类的 4xx 形态不该重试，
# 而 google-genai 自己会把 HTTP 错误翻成 APIError，单独判。
_GEMINI_TRANSPORT_ERRORS = (
    *_RETRYABLE_TRANSPORT_ERRORS,
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    TimeoutError,
)
```

在 `_check_gemini_finish` 之后、`class GeminiProvider` 之前插入：

```python
def _gemini_retry_after(error: Any) -> float | None:
    """APIError 挂着原始响应（httpx 或 aiohttp 的），两者的 headers 都大小写不敏感。"""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        return _parse_retry_after(headers.get("retry-after"))
    except (AttributeError, TypeError):
        return None


def _gemini_http_error(error: Any, model: str) -> LLMHTTPError:
    """google.genai.errors.APIError → LLMHTTPError（带状态码与响应摘要）。

    原来 Gemini 路径完全不包这一族：APIError 不在 cli.PIPELINE_ERRORS 里，一个 429 就是
    一整页 traceback，而且默认 provider 恰好是 gemini。
    """
    details = getattr(error, "details", None)
    if details is None:
        body = str(getattr(error, "message", None) or error)
    else:
        body = json.dumps(details, ensure_ascii=False, default=str)
    return LLMHTTPError(
        status_code=int(getattr(error, "code", 0) or 0),
        body=body,
        url=f"Gemini {model}",
        retry_after=_gemini_retry_after(error),
    )
```

把 `class GeminiProvider` 整个换成：

```python
class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        *,
        max_attempts: int = DEFAULT_LLM.max_attempts,
        transport_max_attempts: int = DEFAULT_LLM.transport_max_attempts,
        temperature: float | None = DEFAULT_LLM.temperature,
        max_output_tokens: int | None = DEFAULT_LLM.max_output_tokens,
    ):
        from google import genai

        self.model = model
        self.max_attempts = max_attempts
        self.transport_max_attempts = transport_max_attempts
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.last_usage: LLMUsage | None = None
        self._client = genai.Client(api_key=api_key)

    async def _generate(
        self,
        system: str,
        contents: str,
        schema: type[BaseModel] | None,
        tally: _StreamTally,
    ) -> Any:
        from google.genai import types

        # temperature / max_output_tokens 传 None 就是 GenerateContentConfig 自己的
        # 「不设置」默认值，所以这里无条件传，不用像 OpenAI 兼容那边那样挑着塞。
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        # 计数挂在这一次 complete() 自己的 tally 上：原来是 self._requests，每次 complete
        # 开头清零，并发下后开始的调用会把先开始那次的计数一起清掉。
        tally.requests += 1
        return await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config
        )

    async def _generate_with_retries(
        self,
        system: str,
        contents: str,
        schema: type[BaseModel] | None,
        tally: _StreamTally,
    ) -> Any:
        """在 _generate 外面套传输层重试。判据与退避跟
        OpenAICompatibleProvider._stream_with_retries 同一套：429 / 5xx / 连接类异常才重试，
        次数是 transport_max_attempts（含首发），其余 4xx 立即失败。

        google-genai 自己的重试默认是关的（HttpOptions.retry_options 为 None 时
        stop_after_attempt(1)），所以这一层不会跟 SDK 叠成乘法。
        """
        from google.genai import errors as genai_errors

        attempt = 0
        while True:
            attempt += 1
            error: Exception
            try:
                return await self._generate(system, contents, schema, tally)
            except genai_errors.APIError as exc:
                cause: BaseException = exc
                http_error = _gemini_http_error(exc, self.model)
                error = http_error
                retryable = http_error.status_code in RETRYABLE_STATUS_CODES
                retry_after = http_error.retry_after
            except _GEMINI_TRANSPORT_ERRORS as exc:
                cause = exc
                error = exc
                retryable = True
                retry_after = None
            if not retryable or attempt >= self.transport_max_attempts:
                raise self._exhausted(error, attempt) from cause
            tally.transport_retries += 1
            await _sleep(_backoff_delay(attempt, retry_after))

    def _exhausted(self, error: Exception, attempt: int) -> LLMError:
        """把最后一次失败翻译成带现场的 LLMError（形状跟 OpenAI 兼容那条路一致）。"""
        if isinstance(error, LLMHTTPError):
            return LLMHTTPError(
                status_code=error.status_code,
                body=error.body,
                url=error.url,
                retry_after=error.retry_after,
                attempts=attempt,
            )
        detail = str(error) or type(error).__name__
        return LLMTransportError(
            f"连接 Gemini 接口失败，已尝试 {attempt} 次仍不通"
            f"（model={self.model}）：{type(error).__name__}: {detail}"
        )

    async def _generate_checked(
        self,
        system: str,
        contents: str,
        schema: type[BaseModel] | None,
        tally: _StreamTally,
    ) -> Any:
        response = await self._generate_with_retries(system, contents, schema, tally)
        _check_gemini_finish(response)
        return response

    def _record_usage(self, response: Any, elapsed_seconds: float, requests: int) -> None:
        meta = getattr(response, "usage_metadata", None)
        self.last_usage = LLMUsage(
            prompt_tokens=_as_int(getattr(meta, "prompt_token_count", None)),
            completion_tokens=_as_int(getattr(meta, "candidates_token_count", None)),
            total_tokens=_as_int(getattr(meta, "total_token_count", None)),
            elapsed_seconds=elapsed_seconds,
            requests=requests,
        )

    def _repair_contents(self, schema: type[BaseModel], repair: RepairContext) -> str:
        return (
            f"{_REPAIR_HEADER}\n\n"
            "## 输出 JSON Schema（必须严格遵守）\n\n"
            f"```json\n{_schema_spec(schema)}\n```\n\n"
            "## 上一轮的输出\n\n"
            f"{repair.bad_output}\n\n"
            "## 校验报错\n\n"
            f"{repair.error}\n\n"
            "只输出修正后的完整 JSON 对象，不要解释，不要加 markdown 围栅。"
        )

    @overload
    async def complete(self, system: str, user: str, schema: None = None) -> str: ...

    @overload
    async def complete[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T: ...

    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        tally = _StreamTally()
        started = time.monotonic()
        last_response: Any = None
        try:
            if schema is None:
                last_response = await self._generate_checked(system, user, None, tally)
                return last_response.text

            async def send(repair: RepairContext | None) -> str:
                nonlocal last_response
                contents = (
                    user if repair is None else self._repair_contents(schema, repair)
                )
                last_response = await self._generate_checked(system, contents, schema, tally)
                text = last_response.text
                if text is None:
                    raise LLMResponseFormatError(
                        "Gemini 一个字都没返回（response.text is None），没有可校验的内容。"
                    )
                return text

            return await complete_with_schema_repair(
                send,
                schema,
                max_attempts=self.max_attempts,
                label=type(self).__name__,
                diagnostics=tally.summary,
            )
        finally:
            self._record_usage(last_response, time.monotonic() - started, tally.requests)
```

`build_provider` 的 gemini 分枝里 `GeminiProvider(...)` 的参数加一行 `transport_max_attempts=cfg.transport_max_attempts,`（放在 `max_attempts=cfg.max_attempts,` 之后）。

`src/tenmin/config.py` 里 `LLMConfig.script_concurrency` 注释的第 1 条（从 `# 1. 默认 provider 是 gemini` 到 `#    传输层退避（transport_max_attempts=4 + 抖动 + Retry-After），可以放心调到 2–4。` 那 6 行）换成：

```python
    # 1. **并发最容易撞 429**。三条 provider 路径现在都有我们自己的传输层退避
    #    （GeminiProvider._generate_with_retries 与 OpenAICompatibleProvider
    #    ._stream_with_retries 同一套判据：429/5xx/连接类异常，transport_max_attempts=4
    #    + 抖动 + Retry-After），但退避只兜得住秒级的限流窗，一个把 RPM 配额打满的并发度
    #    照样是整批失败。
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_llm.py -q`
Expected: 全部 PASS（含既有 Gemini 用例：`_fake_gemini` 替身的 `generate_content` 签名没变；`test_gemini_exposes_last_usage` 照旧读到 100/20/120）。

- [ ] **Step 5: 全量（重点看 MiniMax 用例零变化）**

Run: `uv run pytest tests/ -q`
Expected: 全绿；`uv run pytest tests/test_llm.py -q -k minimax` 条数与改动前一致、全 PASS。

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/script/llm.py src/tenmin/config.py tests/test_llm.py && rtk git commit -m "fix: Gemini 路径把 APIError 包成 LLMHTTPError 并接上传输层退避"
```

---
### Task 4: LLM 用量落盘（`03_script/E{NN}.usage.json`、`zh/E{NN}.usage.json`）

**Files:**
- Modify: `src/tenmin/script/llm.py`（`dataclasses` import；`LLMUsage` 加 `cached_tokens` 并改 docstring；`_StreamTally.usages` 与 `to_usage` 求和；`_stream_once` 收 usage 的写法；`GeminiProvider._generate` 收 usage、`complete` 改用 `tally.to_usage`、删 `_record_usage`）
- Create: `src/tenmin/script/usage.py`
- Modify: `src/tenmin/script/single.py`（`generate_script` 的 `usage` 参数与 `draft` 的轮次类型）
- Modify: `src/tenmin/translate/lines.py`（`translate_track` 的 `usage` 参数）
- Modify: `src/tenmin/pipeline.py`（`_ARTIFACTS` 两行、`Paths` 两个方法、`run_script` / `run_translate` 落盘）
- Modify: `src/tenmin/config.py`（`LLMConfig.script_concurrency` 注释第 4 条）
- Test: `tests/test_llm.py`、`tests/test_single.py`、`tests/test_translate_lines.py`、`tests/test_pipeline.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_llm.py` 末尾追加：

```python
# --- 用量：cached token 与「一次 complete() 底下全部请求」的合计 -------------------


@pytest.mark.asyncio
async def test_gemini_usage_sums_every_request_and_reads_cached_tokens(monkeypatch):
    provider = GeminiProvider(api_key="fake-key")
    first = SimpleNamespace(
        prompt_token_count=100,
        candidates_token_count=10,
        total_token_count=110,
        cached_content_token_count=60,
    )
    second = SimpleNamespace(
        prompt_token_count=30,
        candidates_token_count=5,
        total_token_count=35,
        cached_content_token_count=None,
    )
    _fake_gemini(
        monkeypatch,
        provider,
        [
            _FakeGeminiResponse('{"val": 1}', usage=first),
            _FakeGeminiResponse('{"value": 1}', usage=second),
        ],
    )

    await provider.complete("SYS", "USR", Toy)

    usage = provider.last_usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (130, 15, 145)
    assert usage.cached_tokens == 60
    assert usage.requests == 2


@pytest.mark.asyncio
async def test_gemini_usage_is_all_none_when_the_sdk_reports_nothing(monkeypatch):
    provider = GeminiProvider(api_key="fake-key")
    _fake_gemini(monkeypatch, provider, ['{"value": 1}'])

    await provider.complete("SYS", "USR", Toy)

    assert provider.last_usage.prompt_tokens is None
    assert provider.last_usage.cached_tokens is None
    assert provider.last_usage.requests == 1


@pytest.mark.asyncio
async def test_openai_compatible_usage_reads_cached_tokens(monkeypatch):
    _mock_httpx(
        monkeypatch,
        [
            _sse(
                _delta('{"value": 1}'),
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 5,
                        "total_tokens": 55,
                        "prompt_tokens_details": {"cached_tokens": 32},
                    },
                },
            )
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    await provider.complete("SYS", "USR", Toy)

    assert provider.last_usage.cached_tokens == 32


@pytest.mark.asyncio
async def test_deepseek_style_cache_hit_tokens_are_read(monkeypatch):
    _mock_httpx(
        monkeypatch,
        [
            _sse(
                _delta('{"value": 1}'),
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 5,
                        "total_tokens": 55,
                        "prompt_cache_hit_tokens": 40,
                    },
                },
            )
        ],
    )
    provider = OpenAICompatibleProvider(api_key="k", model="m", base_url="https://x.test/v1")

    await provider.complete("SYS", "USR", Toy)

    assert provider.last_usage.cached_tokens == 40


@pytest.mark.asyncio
async def test_openai_compatible_usage_sums_across_repair_rounds(monkeypatch):
    usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    _mock_httpx(
        monkeypatch,
        [
            _sse(_delta('{"val": 1}'), {"choices": [], "usage": usage}),
            _sse(_delta('{"value": 1}'), {"choices": [], "usage": usage}),
        ],
    )
    provider = MiniMaxProvider(api_key="secret")

    await provider.complete("SYS", "USR", Toy)

    assert provider.last_usage.prompt_tokens == 20
    assert provider.last_usage.total_tokens == 22
    assert provider.last_usage.requests == 2
```

`tests/test_single.py` 顶部 import 块追加：

```python
from tenmin.script.llm import LLMSchemaError, LLMUsage
from tenmin.script.usage import UsageRecord
```

`tests/test_single.py` 末尾追加：

```python
# --- 每次 LLM 调用一条用量记录 -----------------------------------------------


class _MeteredProvider(FakeProvider):
    """FakeProvider + 每次 complete() 把预置的 LLMUsage 挂到 last_usage 上。"""

    def __init__(self, responses, usages):
        super().__init__(responses)
        self._usages = list(usages)
        self.last_usage = None

    async def complete(self, system, user, schema=None):
        self.last_usage = self._usages.pop(0)
        return await super().complete(system, user, schema)


@pytest.mark.asyncio
async def test_generate_script_records_the_draft_and_the_budget_rewrite(cfg, track, report):
    too_long = valid_llm_script(chars_per_beat=(600, 600, 600))
    provider = _MeteredProvider(
        [too_long, valid_llm_script()],
        [
            LLMUsage(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                cached_tokens=800,
                elapsed_seconds=3.0,
                requests=1,
            ),
            LLMUsage(prompt_tokens=900, completion_tokens=150, total_tokens=1050, requests=2),
        ],
    )
    usage: list[UsageRecord] = []

    await generate_script(cfg, track, report, provider, usage=usage)

    assert [record.round for record in usage] == ["draft", "budget_rewrite"]
    assert usage[0].prompt_tokens == 1000
    assert usage[0].completion_tokens == 200
    assert usage[0].cached_tokens == 800
    assert usage[1].cached_tokens is None
    assert usage[1].requests == 2
    assert all(record.provider == "gemini" for record in usage)
    assert all(record.model == cfg.llm.model for record in usage)
    assert all(record.ok for record in usage)
    assert all(record.elapsed_seconds >= 0.0 for record in usage)


@pytest.mark.asyncio
async def test_generate_script_records_a_validation_retry(cfg, track, report):
    """两个节点低于 min_beats=3，首轮判错重试。FakeProvider 不报用量，字段就是 None。"""
    provider = FakeProvider([valid_llm_script(chars_per_beat=(216, 216)), valid_llm_script()])
    usage: list[UsageRecord] = []

    await generate_script(cfg, track, report, provider, usage=usage)

    assert [record.round for record in usage] == ["draft", "validation_retry"]
    assert usage[0].prompt_tokens is None
    assert usage[0].requests is None


@pytest.mark.asyncio
async def test_generate_script_records_a_call_that_failed(cfg, track, report):
    """失败路径也记：白烧了多少 token 正是这时候最想知道的数。"""

    class _Blowup:
        last_usage = LLMUsage(prompt_tokens=5, requests=3)

        async def complete(self, system, user, schema=None):
            raise LLMSchemaError("连续 3 次输出不符合 LLMScript", raw_output="x")

    usage: list[UsageRecord] = []
    with pytest.raises(LLMSchemaError):
        await generate_script(cfg, track, report, _Blowup(), usage=usage)

    assert len(usage) == 1
    assert usage[0].ok is False
    assert usage[0].prompt_tokens == 5
    assert usage[0].requests == 3


@pytest.mark.asyncio
async def test_generate_script_without_a_usage_sink_still_works(cfg, track, report):
    script, _ = await generate_script(cfg, track, report, FakeProvider([valid_llm_script()]))
    assert script.beats
```

`tests/test_translate_lines.py` 末尾追加：

```python
@pytest.mark.asyncio
async def test_translate_track_records_each_call_with_its_round(tmp_path):
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    provider = _ScriptedProvider(_payload([1]), _payload([1, 2]))
    usage = []

    await tl.translate_track(_cfg(tmp_path), track, provider, accumulated={}, usage=usage)

    assert [record.round for record in usage] == ["translate", "translate_repair"]
    assert all(record.ok for record in usage)
```

`tests/test_pipeline.py`：`FROZEN_LAYOUT` 里 `"script_warnings"` 那一行之后加 `"script_usage": "03_script/E02.usage.json",`，`"zh_subtitles"` 那一行之后加 `"zh_usage": "zh/E02.usage.json",`。文件末尾追加：

```python
# --- LLM 用量落盘（只用来观察成本，不是任何阶段的新鲜度输入）-----------------


@pytest.mark.asyncio
async def test_run_script_writes_a_usage_file(project):
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)

    data = json.loads(Paths(project.root).script_usage(2).read_text(encoding="utf-8"))
    assert data["episode"] == 2
    assert [call["round"] for call in data["calls"]] == ["draft"]
    call = data["calls"][0]
    assert call["provider"] == "gemini"
    assert call["prompt_tokens"] is None
    assert call["completion_tokens"] is None
    assert call["cached_tokens"] is None
    assert call["ok"] is True


@pytest.mark.asyncio
async def test_run_script_writes_the_usage_file_even_when_it_fails(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])

    with pytest.raises(LLMSchemaError):
        await run_script(project, _SchemaBlowupProvider("x"), episode=2)

    data = json.loads(Paths(project.root).script_usage(2).read_text(encoding="utf-8"))
    assert [call["ok"] for call in data["calls"]] == [False]


async def test_run_translate_writes_a_usage_file(tmp_path):
    cfg, paths = _project_with_dialogue(tmp_path, episode=11, source="asr")

    await run_translate(cfg, FakeProvider([_translation_response()]), 11)

    data = json.loads(paths.zh_usage(11).read_text(encoding="utf-8"))
    assert data["episode"] == 11
    assert [call["round"] for call in data["calls"]] == ["translate"]


async def test_a_skipped_translate_writes_no_usage_file(tmp_path):
    """繁中片源上 translate 是零动作，连 zh/ 目录都不许建。"""
    cfg, paths = _project_with_dialogue(tmp_path, episode=2, source="srt")

    await run_translate(cfg, FakeProvider([]), 2)

    assert not paths.zh_usage(2).exists()
    assert not (cfg.root / "zh").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_llm.py tests/test_single.py tests/test_translate_lines.py tests/test_pipeline.py -q`
Expected: `tests/test_single.py` 收集失败（`No module named 'tenmin.script.usage'`）；test_llm 的新用例 FAIL（`LLMUsage` 没有 `cached_tokens`、Gemini 只记最后一次响应）；test_pipeline 的 `FROZEN_LAYOUT` 双向断言与四条新用例 FAIL。

- [ ] **Step 3a: 实现 —— `llm.py` 的用量**

import 改成 `from dataclasses import dataclass, field`，并在 `from collections.abc import Awaitable, Callable` 那行加上 `Iterable`：`from collections.abc import Awaitable, Callable, Iterable`。

`LLMUsage` 整个换成：

```python
@dataclass(frozen=True)
class LLMUsage:
    """上一次 complete() 的用量与耗时。字段为 None = 服务端没给这个数。

    数字是**这一次 complete() 底下全部 HTTP 请求的合计**（schema 修复轮、带着 usage 返回
    的那几次请求都算进来）：原来 Gemini 只记最后一个响应、OpenAI 兼容只记最后一条流，
    修复轮烧掉的 token 就这样从账上消失了。cached_tokens 是 prompt 里命中服务端前缀缓存
    的那部分（Gemini 的 cached_content_token_count、OpenAI 的
    prompt_tokens_details.cached_tokens、DeepSeek 的 prompt_cache_hit_tokens）。

    **生产消费者是 script/usage.py 的 track_call**：它在每次 `await provider.complete()`
    返回后立刻读这个字段、中间没有任何 await，所以哪怕 `llm.script_concurrency > 1`、多集
    共用同一个 provider 实例，它读到的也是自己这一次的数（asyncio 只在 await 点切换任务，
    complete() 的 finally 赋值与调用方的读取在同一步里完成）。

    除此之外的时刻去读它就不可靠了：并发下 `last_usage` 是最后一个完成的那次赋的，既不是
    总量、也不一定是你关心的那一集 —— tests/test_llm.py 有一条用例把这个事实钉着。
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    elapsed_seconds: float = 0.0
    requests: int = 1
```

`_as_int` 之后插入：

```python
def _total(values: Iterable[int | None]) -> int | None:
    """逐请求的数加起来；一个都没报就是 None（不是 0 —— 0 会谎称「服务端说用了 0 个」）。"""
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _cached_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return _as_int(details.get("cached_tokens"))
    return _as_int(usage.get("prompt_cache_hit_tokens"))
```

`_StreamTally` 里 `usage: dict[str, Any] | None = None` 换成 `usages: list[dict[str, Any]] = field(default_factory=list)`，`to_usage` 换成：

```python
    def to_usage(self, elapsed_seconds: float) -> LLMUsage:
        return LLMUsage(
            prompt_tokens=_total(_as_int(u.get("prompt_tokens")) for u in self.usages),
            completion_tokens=_total(_as_int(u.get("completion_tokens")) for u in self.usages),
            total_tokens=_total(_as_int(u.get("total_tokens")) for u in self.usages),
            cached_tokens=_total(_cached_tokens(u) for u in self.usages),
            elapsed_seconds=elapsed_seconds,
            requests=self.requests,
        )
```

`OpenAICompatibleProvider._stream_once` 里：在 `finish_reason: str | None = None` 之后加一行 `stream_usage: dict[str, Any] | None = None`；把

```python
                        if isinstance(event.get("usage"), dict):
                            tally.usage = event["usage"]
```

换成

```python
                        # 一条流里只留最后一个 usage（个别实现每个 chunk 都带累计值）。
                        if isinstance(event.get("usage"), dict):
                            stream_usage = event["usage"]
```

并在 `async for line in response.aiter_lines():` 那个循环**之后**（仍在 `as response:` 块内，与 `async for` 同缩进）加：

```python
                    if stream_usage is not None:
                        tally.usages.append(stream_usage)
```

`GeminiProvider._generate` 的最后一句 `return await self._client.aio.models.generate_content(...)` 换成：

```python
        response = await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            # 折成 OpenAI 兼容那条路的 usage 形状，好让 _StreamTally.to_usage 一份实现两边共用。
            tally.usages.append(
                {
                    "prompt_tokens": getattr(meta, "prompt_token_count", None),
                    "completion_tokens": getattr(meta, "candidates_token_count", None),
                    "total_tokens": getattr(meta, "total_token_count", None),
                    "prompt_tokens_details": {
                        "cached_tokens": getattr(meta, "cached_content_token_count", None)
                    },
                }
            )
        return response
```

删掉 `GeminiProvider._record_usage`，`GeminiProvider.complete` 换成：

```python
    async def complete(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> Any:
        tally = _StreamTally()
        started = time.monotonic()
        try:
            if schema is None:
                response = await self._generate_checked(system, user, None, tally)
                return response.text

            async def send(repair: RepairContext | None) -> str:
                contents = (
                    user if repair is None else self._repair_contents(schema, repair)
                )
                response = await self._generate_checked(system, contents, schema, tally)
                text = response.text
                if text is None:
                    raise LLMResponseFormatError(
                        "Gemini 一个字都没返回（response.text is None），没有可校验的内容。"
                    )
                return text

            return await complete_with_schema_repair(
                send,
                schema,
                max_attempts=self.max_attempts,
                label=type(self).__name__,
                diagnostics=tally.summary,
            )
        finally:
            # 失败路径也记：「白烧了多少 token」正是这时候最想知道的数。
            self.last_usage = tally.to_usage(time.monotonic() - started)
```

`src/tenmin/config.py` 里 `LLMConfig.script_concurrency` 注释第 4 条（`# 4. **provider 的 usage 诊断字段在并发下不可靠**` 起那 3 行）换成：

```python
    # 4. `provider.last_usage` 在并发下只是「最后一个完成的那次」。这条不是「不该调高」的
    #    理由：落盘的 03_script/E{NN}.usage.json 不受影响（script/usage.py 的 track_call 在
    #    每次调用返回后立刻读，中间没有 await），只是别在别处拿 last_usage 当总量。
```

- [ ] **Step 3b: 实现 —— 新建 `src/tenmin/script/usage.py`**

```python
"""LLM 用量记录：每次 provider.complete() 一条，落到 03_script/ 与 zh/ 下的 usage.json。

只用来观察成本，**不是任何阶段的新鲜度输入**。数据来源是 provider.last_usage（两条
provider 路径都会填，见 script/llm.py 的 LLMUsage）；没有这个属性的 provider（测试替身、
库调用方自己的实现）照样记一条，token 字段写 null。
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from tenmin import atomic
from tenmin.config import LLMConfig

RoundKind = Literal["draft", "validation_retry", "budget_rewrite", "translate", "translate_repair"]


class UsageRecord(BaseModel):
    """一次 complete() 的用量。token 数为 None = provider 没给。

    requests 是这一轮底下实际发出的 HTTP 请求数（含传输层重试与 schema 修复轮）；
    elapsed_seconds 是调用方自己量的墙钟（provider 不报耗时也有数）。
    """

    provider: str
    model: str
    round: RoundKind
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    requests: int | None = None
    elapsed_seconds: float
    ok: bool


async def track_call[T](
    sink: list[UsageRecord] | None,
    provider: Any,
    llm: LLMConfig,
    kind: RoundKind,
    call: Awaitable[T],
) -> T:
    """await 一次 provider 调用，并往 sink 里追加一条记录（成功失败都记）。

    **读 last_usage 必须紧跟在 await 之后、中间不许再有 await**：asyncio 只在 await 点
    切换任务，所以 complete() 在 finally 里赋的值与这里的读取发生在同一步，别的并发任务
    插不进来。sink 为 None 时只是原样 await。
    """
    started = time.monotonic()
    ok = False
    try:
        result = await call
        ok = True
        return result
    finally:
        if sink is not None:
            usage = getattr(provider, "last_usage", None)
            sink.append(
                UsageRecord(
                    provider=llm.provider,
                    model=llm.model,
                    round=kind,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    cached_tokens=getattr(usage, "cached_tokens", None),
                    requests=getattr(usage, "requests", None),
                    elapsed_seconds=time.monotonic() - started,
                    ok=ok,
                )
            )


def write_usage(path: Path, episode: int, records: list[UsageRecord]) -> None:
    """落盘。形状跟同目录的 warnings.json 一致：{"episode": N, "calls": [...]}。"""
    payload = {
        "episode": episode,
        "calls": [record.model_dump(mode="json") for record in records],
    }
    atomic.write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
```

- [ ] **Step 3c: 实现 —— 调用点**

`src/tenmin/script/single.py`：import 段加 `from tenmin.script.usage import RoundKind, UsageRecord, track_call`。`generate_script` 签名的关键字参数末尾加 `usage: list[UsageRecord] | None = None,`，docstring 末尾加一段：

```python
    `usage` 给了就每次 LLM 调用往里追加一条 UsageRecord（首稿 / 语义校验重试 / 时长返工
    各记各的轮次类型），pipeline.run_script 拿它落 03_script/E{NN}.usage.json。
```

`draft` 换成：

```python
    async def draft(prompt: str, label: str, kind: RoundKind) -> tuple[Script, list[str]]:
        nonlocal round_index
        round_index += 1
        reporter.substep("script", round_index, total_rounds, label)
        llm_script = await track_call(
            usage, provider, llm, kind, provider.complete(SYSTEM_PROMPT, prompt, LLMScript)
        )
        result = validate_script(
            to_script(llm_script, cfg, track.episode),
            tracks,
            reports,
            cfg=cfg.validate_script,
            # validate 里两条按秒数判的检查（hold/sfx 落点上界、画面/旁白拉伸倍率）
            # 必须跟 voice 阶段同口径，见 validate.check_script 的 docstring。
            rate=rate,
        )
        return (
            apply_estimates(result.script, rate=rate),
            list(result.warnings),
        )
```

首稿循环：`prompt, label = first_prompt, "生成初稿"` 下一行加 `kind: RoundKind = "draft"`；`script, stage_warnings = await draft(prompt, label)` 改成 `await draft(prompt, label, kind)`；`label = f"校验失败，第 {attempt} 次重试"` 下一行加 `kind = "validation_retry"`。返工轮的 `draft(...)` 调用加第三个实参 `"budget_rewrite"`：

```python
            candidate, candidate_warnings = await draft(
                _followup_prompt(followup_base, script, "时长返工要求", instruction),
                f"时长返工第 {round_number} 轮",
                "budget_rewrite",
            )
```

`src/tenmin/translate/lines.py`：import 段加 `from tenmin.script.usage import UsageRecord, track_call`。`translate_track` 签名改成：

```python
async def translate_track(
    cfg: ProjectConfig,
    track: DialogueTrack,
    provider: LLMProvider,
    *,
    accumulated: Mapping[str, str],
    usage: list[UsageRecord] | None = None,
) -> TranslatedTrack:
```

`send` 换成：

```python
    async def send(repair: RepairContext | None) -> str:
        if repair is None:
            return await track_call(
                usage, provider, cfg.llm, "translate", provider.complete(system, body)
            )
        # 纠错轮**照旧重发整份正文**，刻意不学 llm.py 那条「修复轮完全不重发正文」的
        # 省法：那一层修的是纯格式问题，而这里要修的是「第 137、298 条漏了」，模型得
        # 对着原文才补得出那几条的译文。代价是重试一次就多发一份对白轨。
        return await track_call(
            usage,
            provider,
            cfg.llm,
            "translate_repair",
            provider.complete(
                system,
                f"{body}\n\n## 上一次的输出有问题\n\n"
                f"{repair.error}\n\n上一次的输出（可能被截断）：\n\n{repair.bad_output}\n",
            ),
        )
```

`src/tenmin/pipeline.py`：import 段加 `from tenmin.script.usage import UsageRecord, write_usage`。`_ARTIFACTS` 里 `"script_warnings"` 条目之后加：

```python
    # 每集 script 阶段每次真正跑完（含失败）都写：这一集全部 LLM 调用的用量，一次调用
    # 一条。只用来看成本，它不是任何阶段的输入或输出，不参与 _is_fresh。
    "script_usage": ("03_script", ".usage.json"),
```

`"zh_subtitles"` 条目之后加：

```python
    # translate 阶段的用量，形状同 script_usage。只在这一集真的走了翻译时才写（繁中片源
    # 上 translate 是零动作，连 zh/ 目录都不建）。不参与 _is_fresh。
    "zh_usage": ("zh", ".usage.json"),
```

`Paths` 里 `script_warnings` 方法之后加：

```python
    def script_usage(self, episode: int) -> Path:
        return self._artifact("script_usage", episode)
```

`zh_subtitles` 方法之后加：

```python
    def zh_usage(self, episode: int) -> Path:
        return self._artifact("zh_usage", episode)
```

`run_script` 里 `glossary = effective_glossary(...)` 之后到 `_write_json(paths.script(episode), ...)` 之前的 try 块改成（两条 except 分枝的内容原样保留，只在 `generate_script` 的调用加 `usage=usage`、并在整个 try 末尾加 finally）：

```python
    usage: list[UsageRecord] = []
    try:
        script, warnings = await generate_script(
            cfg, track, report, provider, reporter=reporter, glossary=glossary, usage=usage
        )
    except ScriptValidationError as error:
        # 跟下面 LLMSchemaError 的落盘同理：pipeline 是唯一知道产物往哪写的一层。
        # 一次真实调用可达 561 秒，重试耗尽后原来什么都不留。
        if error.script is None:
            raise
        rejected_path = paths.script_rejected(episode)
        _write_json(rejected_path, error.script.model_dump_json(indent=2))
        raise ScriptValidationError(
            f"{error}\n最后一版没通过校验的剧本已存到 {rejected_path}",
            script=error.script,
        ) from error
    except LLMSchemaError as error:
        # 落盘选在这一层：llm.py 不该知道 Paths（它是纯 provider 层，被单测直接实例化），
        # 而 single.py 只是拼 prompt 的无状态函数、同样拿不到项目根目录。pipeline 是
        # 「知道产物往哪写」的唯一一层，所以现场也在这里落。
        if not error.raw_output:
            raise
        raw_path = paths.script_raw(episode)
        _write_text(raw_path, error.raw_output)
        raise LLMSchemaError(
            f"{error}\n最后一次的原始模型输出已存到 {raw_path}",
            raw_output=error.raw_output,
        ) from error
    finally:
        # 成功失败都写：白烧了多少 token 正是失败时最想知道的数。它不是任何阶段的输入。
        write_usage(paths.script_usage(episode), episode, usage)
```

（两个 except 分枝的内容与改动前逐字相同，只是整块多了 `usage` 实参与末尾的 finally。）

`run_translate` 里 `accumulated = load_glossary(paths.glossary)` 与 `translated = await translate_track(...)` 两行换成：

```python
    accumulated = load_glossary(paths.glossary)
    usage: list[UsageRecord] = []
    try:
        translated = await translate_track(
            cfg, track, provider, accumulated=accumulated, usage=usage
        )
    finally:
        # 写在 source 判断之后：繁中片源在上面就早退了，连 zh/ 目录都不会被建出来。
        write_usage(paths.zh_usage(episode), episode, usage)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_llm.py tests/test_single.py tests/test_translate_lines.py tests/test_pipeline.py -q`
Expected: 全部 PASS（`test_last_usage_is_only_the_last_finished_call_under_concurrency` 照旧成立：每次 complete 有自己的 tally）。

- [ ] **Step 5: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/script/llm.py src/tenmin/script/usage.py src/tenmin/script/single.py src/tenmin/translate/lines.py src/tenmin/pipeline.py src/tenmin/config.py tests/test_llm.py tests/test_single.py tests/test_translate_lines.py tests/test_pipeline.py && rtk git commit -m "feat: 每次 LLM 调用的用量落盘到 usage.json"
```

---

### Task 5: 允许「预填」集条目（只有 op/ed），检查挪到运行时

**Files:**
- Modify: `src/tenmin/config.py`（`EpisodeConfig`：删 `_require_a_source`、加 `has_source`、改字段上方注释）
- Modify: `src/tenmin/pipeline.py`（新增 `_active_episodes`；`_ingest_inputs` / `run_ingest` / `_load_tracks` / `_load_reports` 改遍历它；`_find_episode` 认预填条目；`run_pipeline` 的集号解析与跳过提示）
- Modify: `src/tenmin/cli.py`（`inspect` 列预填条目）
- Test: `tests/test_config.py`、`tests/test_pipeline.py`、`tests/test_cli.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_config.py` 里 `test_episode_with_neither_srt_nor_video_is_rejected` 整个换成：

```python
def test_a_prefilled_episode_with_only_credit_ranges_loads(tmp_path):
    """用户想先把每集的 op/ed 写进 yaml、之后再登记视频。原来这种条目过不了加载期的
    校验，整个项目都读不进来。现在「有没有来源」是运行时的判据（has_source）。"""
    path = tmp_path / "project.yaml"
    path.write_text(
        MINIMAL + "  - number: 3\n    op_range: [10, 100]\n    ed_range: [1300, 1420]\n",
        encoding="utf-8",
    )
    cfg = load_project(path)
    prefilled = next(e for e in cfg.episodes if e.number == 3)
    assert prefilled.has_source is False
    assert prefilled.op_range == (10.0, 100.0)
    assert next(e for e in cfg.episodes if e.number == 1).has_source is True


def test_has_source_is_true_for_either_source():
    assert EpisodeConfig(number=1, srt=Path("srt/E01.srt")).has_source is True
    assert EpisodeConfig(number=1, video=Path("/v/e01.mkv")).has_source is True
    assert EpisodeConfig(number=1).has_source is False
```

`tests/test_pipeline.py` 末尾追加：

```python
# --- 预填集条目：只有 op/ed、还没有 srt/video --------------------------------


@pytest.mark.asyncio
async def test_batch_mode_skips_a_prefilled_episode_with_a_notice(project):
    project.episodes.append(EpisodeConfig(number=3, ed_range=(1300.0, 1420.0)))

    warnings = await run_pipeline(
        project, FakeProvider([fake_script_response()]), only=V1_STAGES
    )

    paths = Paths(project.root)
    assert "第 3 集还没有 video，已跳过" in warnings
    assert paths.script(2).exists()
    assert not paths.dialogue(3).exists()


@pytest.mark.asyncio
async def test_single_episode_mode_on_a_prefilled_episode_asks_for_a_video(project):
    project.episodes.append(EpisodeConfig(number=3, ed_range=(1300.0, 1420.0)))

    with pytest.raises(ValueError, match="--video"):
        await run_pipeline(project, FakeProvider([]), only=V1_STAGES, episode=3)


def test_registering_a_prefilled_episode_keeps_its_credit_ranges(tmp_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nepisodes:\n"
        "- number: 3\n  op_range: [10.0, 100.0]\n  ed_range: [1300.0, 1420.0]\n",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)
    video = tmp_path / "e03.mkv"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=3, srt=None, video=video)

    reloaded = load_project(yaml_path)
    (episode,) = reloaded.episodes
    assert episode.op_range == (10.0, 100.0)
    assert episode.ed_range == (1300.0, 1420.0)
    assert episode.video == video.resolve()
    assert episode.has_source is True
```

`tests/test_cli.py` 末尾追加：

```python
# --- 预填集条目的 CLI 行为 ---------------------------------------------------


def _bootstrap_with_prefilled(work, golden_srt_path):
    root = work / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "E02.srt").write_bytes(golden_srt_path.read_bytes())
    (root / "project.yaml").write_text(
        "show: 才女的侍从\nslug: saijo\nepisodes:\n"
        "- number: 2\n  srt: srt/E02.srt\n"
        "- number: 3\n  op_range: [10.0, 100.0]\n",
        encoding="utf-8",
    )
    return root


def test_batch_run_tells_the_user_it_skipped_a_prefilled_episode(work, golden_srt_path):
    root = _bootstrap_with_prefilled(work, golden_srt_path)
    result = runner.invoke(app, ["run", "saijo", "--work-dir", str(work), "--only", "ingest"])
    assert result.exit_code == 0, out(result)
    assert "第 3 集还没有 video，已跳过" in out(result)
    assert (root / "01_dialogue" / "E02.dialogue.json").exists()
    assert not (root / "01_dialogue" / "E03.dialogue.json").exists()


def test_running_a_prefilled_episode_without_a_video_asks_for_one(work, golden_srt_path):
    _bootstrap_with_prefilled(work, golden_srt_path)
    result = runner.invoke(
        app, ["run", "saijo", "--work-dir", str(work), "--episode", "3", "--only", "ingest"]
    )
    assert result.exit_code == 1
    assert _graceful(result), repr(result.exception)
    assert "--video" in out(result)


def test_inspect_lists_a_prefilled_episode_as_unregistered(work, golden_srt_path):
    _bootstrap_with_prefilled(work, golden_srt_path)
    result = runner.invoke(
        app, ["inspect", "saijo", "--work-dir", str(work), "--episode", "3"]
    )
    assert result.exit_code == 0, out(result)
    assert "未登记视频" in out(result)
    assert "--video" in out(result)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py tests/test_pipeline.py tests/test_cli.py -q -k "prefilled or has_source"`
Expected: FAIL —— `EpisodeConfig(number=3, …)` 与含预填条目的 yaml 都过不了 `_require_a_source`；`has_source` 不存在。

- [ ] **Step 3: 实现**

`src/tenmin/config.py` 的 `EpisodeConfig`：把 `number: int` 下面那段 11 行注释（从 `# 两个来源至少得有一个，由 _require_a_source 守着` 到 `# video 可空是历史约定…` 为止）换成：

```python
    # 两个来源都可空，而且可以**同时**为空：那是「预填」条目 —— 用户先把这一集的
    # op_range/ed_range 写进 yaml，之后再用 `--episode N --video …` 登记源片（登记会并进
    # 这一条、保留 op/ed）。「至少有一个来源」是运行时判据（has_source），由 pipeline 的
    # 批处理跳过、单集模式报错，不在加载期拦 —— 拦在加载期的话整个项目都读不进来。
    # srt 可空是为了「只有生肉视频」那条路（对白轨靠软字幕轨抽取或语音转写拿到）；
    # video 可空是历史约定（v1 只吃字幕、压根不碰视频文件，`--only ingest` 至今还这么用）。
```

删掉整个 `_require_a_source` 方法（含 `@model_validator(mode="after")` 装饰器），在 `_check_ranges` 之后加：

```python
    @property
    def has_source(self) -> bool:
        """srt 与 video 至少有一个。两个都没有 = 预填条目，不参与任何阶段。"""
        return self.srt is not None or self.video is not None
```

`model_validator` 在 config.py 里已经没有别的使用者了，从 pydantic 的 import 列表里删掉它（否则 ruff F401）。

`src/tenmin/pipeline.py`：在 `_source_duration` 之前插入：

```python
def _active_episodes(cfg: ProjectConfig) -> list[EpisodeConfig]:
    """登记过来源（srt 或 video）的集。预填条目（只写了 op/ed）不参与任何阶段。"""
    return [episode for episode in cfg.episodes if episode.has_source]
```

`_ingest_inputs`、`run_ingest`、`_load_tracks`、`_load_reports` 四个函数里的 `for episode in cfg.episodes:` 一律改成 `for episode in _active_episodes(cfg):`。

`_find_episode` 的循环体换成：

```python
    for episode in cfg.episodes:
        if episode.number == episode_number:
            if not episode.has_source:
                raise ValueError(
                    f"第 {episode_number} 集在 project.yaml 里只是预填条目（还没有 video）。"
                    f"请用 `tenmin run {cfg.slug} --episode {episode_number} "
                    "--video <视频路径>` 登记源片，已填的 op_range/ed_range 会保留。"
                )
            return episode
```

`run_pipeline` 里 `numbers = [ep.number for ep in cfg.episodes]` 换成：

```python
    active = _active_episodes(cfg)
    numbers = [ep.number for ep in active]
```

`if episode is None:\n        target_numbers = numbers` 那两行换成：

```python
    if episode is None:
        # 预填条目在批处理里跳过，但不能静默：用户会以为那一集跑过了。只在批处理模式下
        # 提示 —— 单集模式跑的是别的集，报一句无关的集号只是噪音。
        warnings.extend(
            f"第 {ep.number} 集还没有 video，已跳过"
            for ep in cfg.episodes
            if not ep.has_source
        )
        target_numbers = numbers
```

`src/tenmin/cli.py` 的 `inspect`：在 `paths = Paths(cfg.root)` 之后、`if not paths.dialogue(episode).exists():` 之前插入：

```python
    ep_cfg = next((ep for ep in cfg.episodes if ep.number == episode), None)
    prefilled = [ep for ep in cfg.episodes if not ep.has_source]
    if prefilled:
        typer.echo(
            "未登记视频的预填条目：" + "、".join(f"E{ep.number:02d}" for ep in prefilled)
        )
    if ep_cfg is not None and not ep_cfg.has_source:
        typer.echo(
            f"E{episode:02d}：未登记视频（op_range={ep_cfg.op_range}，ed_range={ep_cfg.ed_range}）"
        )
        typer.echo(f"登记源片：tenmin run {slug} --episode {episode} --video <源片路径>")
        return
```

并删掉后面原有的 `ep_cfg = next((ep for ep in cfg.episodes if ep.number == episode), None)` 那一行（已提前）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py tests/test_pipeline.py tests/test_cli.py -q`
Expected: 全部 PASS。

- [ ] **Step 5: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿（`tests/test_resolve.py::test_neither_srt_nor_video_is_an_error` 测的是 resolve 层自己的报错，不经过 EpisodeConfig，不受影响）。

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/config.py src/tenmin/pipeline.py src/tenmin/cli.py tests/test_config.py tests/test_pipeline.py tests/test_cli.py && rtk git commit -m "feat: 允许只填 op/ed 的预填集条目，批处理跳过并提示"
```

---
### Task 6: 登记写回 yaml 改用 ruamel round-trip，保留注释与顺序

**Files:**
- Modify: `pyproject.toml`、`uv.lock`（`uv add ruamel.yaml`）
- Modify: `src/tenmin/pipeline.py`（imports：删 `import yaml`、加 `io` 与 ruamel；`register_episode` 之前新增 `_write_back_episode`；`register_episode` 尾部与 docstring 的「向后兼容」一段）
- Test: `tests/test_pipeline.py`、`tests/test_cli.py`

- [ ] **Step 1: 加依赖**

Run: `uv add ruamel.yaml`
Expected: `pyproject.toml` 的 `dependencies` 多一行 `ruamel-yaml>=0.19.1`（或同义写法），`uv.lock` 更新。写本计划时实测的版本是 0.19.1。若公司代理导致 SSL 报错，按 AGENTS.md 的做法处理证书后重试，不要跳过这一步。

- [ ] **Step 2: 写失败的测试**

`tests/test_pipeline.py` 顶部 import 段加 `import difflib`，并把 `from tenmin.config import EpisodeConfig, ProjectConfig, load_project` 改成 `from tenmin.config import EpisodeConfig, ProjectConfig, ProjectConfigError, load_project`。末尾追加：

```python
# --- 登记写回 project.yaml：只动 episodes 里对应的那一条 --------------------

_COMMENTED_YAML = """\
# 我的番，这行注释要原样留着
show: "才女的侍从"   # 行尾注释，引号风格也得留着
slug: saijo
render:
  font_size: 60  # 字号
episodes:
  - number: 2
    srt: srt/E02.srt
    op_range: [153.5, 224.7]
  # 第 3 集先把片尾填上，视频还没下好
  - number: 3
    ed_range: [1300, 1420]
glossary:
  伊月: 伊月
"""


def _removed_lines(before: str, after: str) -> list[str]:
    return [
        line
        for line in difflib.ndiff(before.splitlines(), after.splitlines())
        if line.startswith("- ")
    ]


def test_register_episode_only_adds_lines_to_a_commented_yaml(tmp_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(_COMMENTED_YAML, encoding="utf-8")
    cfg = load_project(yaml_path)
    video3 = tmp_path / "e03.mkv"
    video3.write_bytes(b"fake")
    video4 = tmp_path / "e04.mkv"
    video4.write_bytes(b"fake")

    register_episode(cfg, episode=3, srt=None, video=video3)
    register_episode(cfg, episode=4, srt=None, video=video4)

    after = yaml_path.read_text(encoding="utf-8")
    assert _removed_lines(_COMMENTED_YAML, after) == []
    # 缩进风格跟着原文件走（这份是「  - 」+ 4 格），不是 ruamel 的默认值。
    assert f"    video: {video3.resolve()}" in after.splitlines()
    reloaded = load_project(yaml_path)
    assert [e.number for e in reloaded.episodes] == [2, 3, 4]
    episode3 = next(e for e in reloaded.episodes if e.number == 3)
    assert episode3.ed_range == (1300.0, 1420.0)
    assert episode3.video == video3.resolve()


def test_reregistering_changes_only_that_entrys_source_lines(tmp_path, golden_srt_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(_COMMENTED_YAML, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e02.mkv"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=2, srt=None, video=video)

    after = yaml_path.read_text(encoding="utf-8")
    # srt 字段被清掉（改走生肉），其它一行不少。
    assert _removed_lines(_COMMENTED_YAML, after) == ["-     srt: srt/E02.srt"]
    assert "    op_range: [153.5, 224.7]" in after.splitlines()


def test_the_write_back_parser_rejects_duplicate_keys_like_load_project(tmp_path):
    """两处对重复键的态度必须一致：不能出现「读得进来、写不回去」或者反过来。"""
    from ruamel.yaml import YAML
    from ruamel.yaml.constructor import DuplicateKeyError as RuamelDuplicateKeyError

    text = "show: 某番\nslug: demo\nrender:\n  crf: '18'\nrender:\n  crf: '20'\n"
    path = tmp_path / "project.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ProjectConfigError):
        load_project(path)
    with pytest.raises(RuamelDuplicateKeyError):
        YAML().load(text)
```

`tests/test_cli.py` 顶部 import 段加 `import difflib`，末尾追加：

```python
def test_registering_keeps_the_init_header_comments(work, tmp_path, monkeypatch):
    """init 生成的 14 行说明原来会被 register_episode 的 safe_dump 整段冲掉。"""
    from tenmin.cli import PROJECT_TEMPLATE_HEADER

    runner.invoke(app, ["init", "saijo", "--work-dir", str(work)])
    path = work / "saijo" / "project.yaml"
    before = path.read_text(encoding="utf-8")
    video = tmp_path / "e02.mkv"
    video.write_bytes(b"fake")

    async def fake_pipeline(cfg, provider, **kwargs):
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)
    result = runner.invoke(
        app,
        [
            "run", "saijo", "--work-dir", str(work),
            "--episode", "2", "--video", str(video), "--only", "ingest",
        ],
    )

    assert result.exit_code == 0, out(result)
    after = path.read_text(encoding="utf-8")
    assert after.startswith(PROJECT_TEMPLATE_HEADER.format(slug="saijo"))
    removed = [
        line
        for line in difflib.ndiff(before.splitlines(), after.splitlines())
        if line.startswith("- ")
    ]
    assert removed == ["- episodes: []"]
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -q -k "commented_yaml or reregistering or duplicate_keys_like or init_header_comments"`
Expected: 前两条与 CLI 那条 FAIL（`safe_dump` 把注释、引号、行尾注释全冲掉了）；`test_the_write_back_parser_rejects_duplicate_keys_like_load_project` PASS（它只是锁住两边一致，Task 1 已经让 load_project 那一半成立）。

- [ ] **Step 4: 实现**

`src/tenmin/pipeline.py` 的 import 段：删掉 `import yaml`（只有 `register_episode` 在用它），标准库段加 `import io`，第三方段加：

```python
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.util import load_yaml_guess_indent
```

在 `register_episode` 之前插入：

```python
def _write_back_episode(yaml_path: Path, entry: EpisodeConfig) -> None:
    """把这一集的 srt/video 写回 project.yaml，其余内容一个字节都不动。

    原来是 safe_load → 改 episodes → safe_dump 整份重写：注释（包括 init 生成的那 14 行
    说明）、键的顺序、引号风格全被冲掉。现在走 ruamel 的 round-trip：只改 episodes 序列里
    number 相同的那一条（改 srt/video 两个字段），没有就追加一条，其余节点原样回写。

    读配置仍然走 config.load_project（PyYAML + 重复键检查），ruamel 只用在这里。两边对
    重复键的态度一致（ruamel round-trip 默认拒绝），有测试锁着。

    缩进：ruamel 会按自己的设置重排**整份**文档的缩进，所以序列缩进从原文件里猜
    （`- ` 顶格还是缩两格），映射缩进固定 2 格（init 与 PyYAML 都是这个）。实测按这个
    配置回写，init 模板与手写的「  - number: 2」两种风格都只多出新增的那几行。
    width 放大是为了不让 ruamel 把长路径折行。
    """
    text = yaml_path.read_text(encoding="utf-8")
    _, sequence_indent, sequence_offset = load_yaml_guess_indent(text)
    round_trip = YAML()
    round_trip.preserve_quotes = True
    round_trip.width = 4096
    round_trip.indent(mapping=2, sequence=sequence_indent or 2, offset=sequence_offset or 0)
    data = round_trip.load(text)
    if data is None:
        data = CommentedMap()
    episodes = data.get("episodes")
    if episodes is None:
        episodes = CommentedSeq()
        data["episodes"] = episodes
    # init 写的是流式的 `episodes: []`，往里追加映射时得换成块式，否则整段挤成一行。
    episodes.fa.set_block_style()
    target = next(
        (
            item
            for item in episodes
            if isinstance(item, dict) and item.get("number") == entry.number
        ),
        None,
    )
    if target is None:
        target = CommentedMap()
        target["number"] = entry.number
        episodes.append(target)
    if entry.srt is None:
        target.pop("srt", None)
    else:
        target["srt"] = entry.srt.as_posix()
    target["video"] = str(entry.video)
    buffer = io.StringIO()
    round_trip.dump(data, buffer)
    # 原子写：这里是**改写**一个已有文件，中途被打断会把用户已注册的全部集数毁掉。
    atomic.write_text(yaml_path, buffer.getvalue())
```

`register_episode` 里从 `existing = next((e for e in cfg.episodes if e.number == episode), None)` 到函数末尾的 `return cfg` 整段换成：

```python
    existing = next((e for e in cfg.episodes if e.number == episode), None)
    if existing is not None:
        # 预填条目也走这条：只改 srt/video，op_range/ed_range 原样留着。
        existing.srt = relative_srt
        existing.video = video_source
        entry = existing
    else:
        entry = EpisodeConfig(number=episode, srt=relative_srt, video=video_source)
        cfg.episodes.append(entry)

    _write_back_episode(cfg.config_path, entry)
    return cfg
```

docstring 里「向后兼容：只有**本次登记的这一集**会被写成绝对路径。其余集的 srt/video 原样走各自 EpisodeConfig 的 model_dump 落盘，存量的相对路径（work/saijo/ 下 10 个已经拷好的 mp4）逐字节不变，video_path() 照旧按 project.yaml 所在目录解析。」那一段换成：

```
    向后兼容：只有**本次登记的这一集**的 srt/video 两个字段会被改写（video 写成绝对路径）。
    其余集、以及 yaml 里别的一切（注释、键序、引号）都由 _write_back_episode 原样保留，
    存量的相对路径（work/saijo/ 下 10 个已经拷好的 mp4）逐字节不变，video_path() 照旧按
    project.yaml 所在目录解析。
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -q`
Expected: 全部 PASS（含既有的 `test_register_episode_*`、`test_run_registers_new_episode_and_updates_yaml`、`test_register_episode_rewrites_project_yaml_atomically`）。

- [ ] **Step 6: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 7: 提交**

```bash
rtk git add pyproject.toml uv.lock src/tenmin/pipeline.py tests/test_pipeline.py tests/test_cli.py && rtk git commit -m "feat: 登记写回 project.yaml 改用 ruamel round-trip，保留注释与顺序"
```

---

### Task 7: `tenmin.config_slices`：阶段 → 配置字段映射、排除表、切片落盘

纯新增模块，这个任务不接进 `run_pipeline`（下一个任务接）。

**Files:**
- Modify: `src/tenmin/atomic.py`（新增 `write_text_if_changed`）
- Create: `src/tenmin/config_slices.py`
- Create: `tests/test_config_slices.py`
- Test: `tests/test_atomic.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_atomic.py`：import 段改成

```python
from __future__ import annotations

import os

import pytest

from tenmin.atomic import (
    PART_SUFFIX,
    atomic_path,
    copy_file,
    part_path,
    write_text,
    write_text_if_changed,
)
```

末尾追加：

```python
def test_write_text_if_changed_leaves_identical_content_untouched(tmp_path):
    target = tmp_path / "a.json"
    write_text(target, "x\n")
    os.utime(target, ns=(10**18, 10**18))

    assert write_text_if_changed(target, "x\n") is False
    assert target.stat().st_mtime_ns == 10**18


def test_write_text_if_changed_rewrites_different_content(tmp_path):
    target = tmp_path / "a.json"
    write_text(target, "x\n")

    assert write_text_if_changed(target, "y\n") is True
    assert target.read_text(encoding="utf-8") == "y\n"


def test_write_text_if_changed_creates_a_missing_file(tmp_path):
    target = tmp_path / "sub" / "a.json"

    assert write_text_if_changed(target, "x\n") is True
    assert target.read_text(encoding="utf-8") == "x\n"
```

新建 `tests/test_config_slices.py`：

```python
"""按阶段的配置切片：映射完整性、排除表、切片内容与落盘。

完整性测试是这个模块的主保险：ProjectConfig 新增一个字段却忘了挂到阶段上，
「改了它不重跑」这种失效比「多重跑一次」难发现得多。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.config_slices import (
    EPISODE,
    EXCLUDED,
    PROJECT_LEVEL_STAGES,
    STAGE_FIELDS,
    slice_path,
    slice_payload,
    write_slice,
    write_slices,
)
from tenmin.pipeline import STAGES


def _subconfig(name: str) -> type[BaseModel] | None:
    annotation = ProjectConfig.model_fields[name].annotation
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _leaves() -> set[str]:
    """ProjectConfig 的全部叶子字段（episodes 单独由 EPISODE 记号负责）。"""
    out: set[str] = set()
    for name in ProjectConfig.model_fields:
        if name == "episodes":
            continue
        sub = _subconfig(name)
        if sub is None:
            out.add(name)
        else:
            out |= {f"{name}.{field}" for field in sub.model_fields}
    return out


def _covered(entry: str) -> set[str]:
    if entry == EPISODE:
        return set()
    if "." in entry:
        return {entry}
    sub = _subconfig(entry)
    if sub is None:
        return {entry}
    return {f"{entry}.{field}" for field in sub.model_fields} - EXCLUDED


# --- 映射表本身 ---------------------------------------------------------------


def test_the_stage_table_covers_exactly_the_pipeline_stages():
    assert list(STAGE_FIELDS) == STAGES


def test_every_config_leaf_is_mapped_or_excluded():
    mapped = set().union(*(_covered(e) for fields in STAGE_FIELDS.values() for e in fields))
    assert _leaves() - mapped - EXCLUDED == set()


def test_the_tables_only_name_real_fields():
    leaves = _leaves()
    assert EXCLUDED <= leaves
    for stage, fields in STAGE_FIELDS.items():
        for entry in fields:
            if entry == EPISODE:
                continue
            assert entry in ProjectConfig.model_fields or entry in leaves, (stage, entry)


def test_no_stage_names_an_excluded_field_explicitly():
    explicit = {e for fields in STAGE_FIELDS.values() for e in fields if "." in e}
    assert explicit & EXCLUDED == set()


def test_the_output_changing_retry_knobs_are_not_excluded():
    """这两个决定「采纳哪一版稿子」，改了必须让 script 重跑。"""
    assert "llm.validation_retries" not in EXCLUDED
    assert "llm.budget_rewrite_rounds" not in EXCLUDED


def test_every_episode_field_reaches_ingest():
    assert EPISODE in STAGE_FIELDS["ingest"]


# --- 切片内容 -----------------------------------------------------------------


def _cfg(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "show": "才女的侍从",
            "slug": "saijo",
            "episodes": [{"number": 2, "srt": "srt/E02.srt", "op_range": [150, 220]}],
        }
    ).bind_root(tmp_path)


def _with(cfg: ProjectConfig, field: str, value: object) -> ProjectConfig:
    top, _, leaf = field.partition(".")
    if not leaf:
        return cfg.model_copy(update={top: value})
    sub = getattr(cfg, top)
    return cfg.model_copy(update={top: sub.model_copy(update={leaf: value})})


def _bump(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "x"
    if value is None:
        return "http://proxy.test:8080"
    raise TypeError(value)


def _payloads(cfg: ProjectConfig) -> dict[str, str]:
    episode = cfg.episodes[0]
    return {
        stage: slice_payload(cfg, stage, None if stage in PROJECT_LEVEL_STAGES else episode)
        for stage in STAGE_FIELDS
    }


def _changed(before: ProjectConfig, after: ProjectConfig) -> set[str]:
    old, new = _payloads(before), _payloads(after)
    return {stage for stage in old if old[stage] != new[stage]}


def test_a_payload_is_sorted_json_of_resolved_values(tmp_path):
    cfg = _cfg(tmp_path)
    payload = slice_payload(cfg, "voice", cfg.episodes[0])
    assert json.loads(payload) == {"render": {"rate": "+0%", "voice": "zh-CN-YunxiNeural"}}
    assert payload == (
        json.dumps(json.loads(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


def test_a_whole_subconfig_entry_drops_the_excluded_knobs(tmp_path):
    cfg = _cfg(tmp_path)
    data = json.loads(slice_payload(cfg, "script", cfg.episodes[0]))
    assert "timeout_seconds" not in data["llm"]
    assert "script_concurrency" not in data["llm"]
    assert data["llm"]["validation_retries"] == 2
    assert data["llm"]["budget_rewrite_rounds"] == 1


def test_the_episode_token_embeds_this_episodes_config(tmp_path):
    cfg = _cfg(tmp_path)
    data = json.loads(slice_payload(cfg, "ingest", cfg.episodes[0]))
    assert data[EPISODE] == {
        "number": 2,
        "srt": "srt/E02.srt",
        "video": None,
        "op_range": [150.0, 220.0],
        "ed_range": None,
    }


def test_a_per_episode_stage_needs_an_episode(tmp_path):
    with pytest.raises(ValueError):
        slice_payload(_cfg(tmp_path), "ingest", None)


@pytest.mark.parametrize("field", sorted(EXCLUDED))
def test_changing_an_excluded_knob_changes_no_slice(tmp_path, field):
    cfg = _cfg(tmp_path)
    top, _, leaf = field.partition(".")
    current = getattr(getattr(cfg, top), leaf) if leaf else getattr(cfg, top)
    assert _changed(cfg, _with(cfg, field, _bump(current))) == set()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("render.crf", "18", {"render"}),
        ("render.font_size", 60, {"timeline", "render"}),
        ("render.voice", "zh-CN-XiaoxiaoNeural", {"voice"}),
        ("render.rate", "+10%", {"script", "docgen", "voice"}),
        ("render.duck_db", -6.0, {"audio"}),
        ("render.fade_out_seconds", 2.0, {"audio", "render"}),
        ("llm.validation_retries", 5, {"script"}),
        ("llm.budget_tolerance", 0.2, {"script"}),
        ("llm.model", "gemini-x", {"translate", "script"}),
        ("credits.op_span_min", 30.0, {"ingest"}),
        ("asr.language", "en", {"ingest"}),
        ("signals.min_gap_seconds", 5.0, {"signals"}),
        ("validate_script.max_beats", 7, {"script"}),
        ("target_seconds", 200.0, {"script"}),
        ("glossary", {"伊月": "伊月君"}, {"ingest", "translate", "script"}),
        ("show", "别的番", {"ingest", "script", "render"}),
    ],
)
def test_a_config_change_reaches_exactly_the_stages_that_read_it(
    tmp_path, field, value, expected
):
    cfg = _cfg(tmp_path)
    assert _changed(cfg, _with(cfg, field, value)) == expected


def test_an_episode_change_reaches_ingest_and_the_source_video_stages(tmp_path):
    cfg = _cfg(tmp_path)
    moved = cfg.episodes[0].model_copy(update={"op_range": (140.0, 220.0)})
    assert _changed(cfg, cfg.model_copy(update={"episodes": [moved]})) == {
        "ingest",
        "timeline",
        "audio",
        "render",
    }


# --- 落盘 ---------------------------------------------------------------------


def test_slice_paths(tmp_path):
    assert slice_path(tmp_path, "script", 2) == tmp_path / ".config" / "E02.script.json"
    assert slice_path(tmp_path, "signals", None) == tmp_path / ".config" / "signals.json"
    with pytest.raises(ValueError):
        slice_path(tmp_path, "script", None)


def test_a_new_slice_is_backdated_to_epoch_zero(tmp_path):
    """升级后第一次运行不能让全部存量产物都比切片旧、整季重付一轮 LLM。"""
    path = tmp_path / ".config" / "E02.voice.json"
    write_slice(path, "{}\n")
    assert path.stat().st_mtime_ns == 0


def test_an_unchanged_slice_is_not_touched(tmp_path):
    path = tmp_path / ".config" / "E02.voice.json"
    write_slice(path, "{}\n")
    os.utime(path, ns=(10**18, 10**18))

    write_slice(path, "{}\n")

    assert path.stat().st_mtime_ns == 10**18


def test_a_changed_slice_gets_a_normal_mtime(tmp_path):
    path = tmp_path / ".config" / "E02.voice.json"
    write_slice(path, "{}\n")

    write_slice(path, '{"a": 1}\n')

    assert path.stat().st_mtime_ns > 0
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_write_slices_writes_one_file_per_stage_and_episode(tmp_path):
    cfg = _cfg(tmp_path)
    write_slices(cfg, cfg.episodes)
    names = sorted(p.name for p in (tmp_path / ".config").iterdir())
    expected = ["signals.json", *(f"E02.{s}.json" for s in STAGES if s != "signals")]
    assert names == sorted(expected)


def test_registering_another_episode_leaves_this_episodes_slices_alone(tmp_path):
    cfg = _cfg(tmp_path)
    write_slices(cfg, cfg.episodes)
    before = {p.name: p.read_bytes() for p in (tmp_path / ".config").iterdir()}

    cfg.episodes.append(EpisodeConfig(number=11, video=Path("/v/e11.mkv")))
    write_slices(cfg, cfg.episodes)

    after = {p.name: p.read_bytes() for p in (tmp_path / ".config").iterdir()}
    assert {name: after[name] for name in before} == before
    assert "E11.ingest.json" in after
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_atomic.py tests/test_config_slices.py -q`
Expected: 收集阶段 ImportError（`write_text_if_changed`、`tenmin.config_slices` 都不存在）。

- [ ] **Step 3: 实现**

`src/tenmin/atomic.py` 末尾追加：

```python
def write_text_if_changed(path: Path, text: str, *, encoding: str = "utf-8") -> bool:
    """内容跟盘上逐字节相同就不碰文件（保住旧 mtime），否则原子写。返回是否真的写了。

    「没变就不写」在本项目里是正确性而不是性能：下游阶段只比 mtime，内容一字没变、
    mtime 却刷新了，整条下游都会被判过期（translate.glossary.save_glossary 是同一个道理）。

    比的是字节而不是解码后的文本：换行符与编码上的任何差别都算「变了」。读盘失败
    （文件不存在、没权限）一律照写 —— 判不出「没变」就不许跳过。`is_file` 守卫：FIFO
    之类的路径上读操作会永久阻塞，except 兜不住阻塞。
    """
    path = Path(path)
    payload = text.encode(encoding)
    if path.is_file():
        try:
            if path.read_bytes() == payload:
                return False
        except OSError:
            pass
    write_text(path, text, encoding=encoding)
    return True
```

新建 `src/tenmin/config_slices.py`：

```python
"""按阶段的配置切片：每个阶段只对「它真正读到的那部分配置」敏感。

原来每个阶段的新鲜度输入里都有整个 project.yaml，于是改一个 render 参数、或者登记
一集新番（register_episode 会改写这个文件），整季的解说稿全被判过期，每集重付一次
LLM。现在 run_pipeline 开跑前把每个阶段读到的配置序列化成一份 JSON（切片），阶段的
新鲜度输入里用切片替换 project.yaml，pipeline._is_fresh 本身一个字都没改。

切片是**解析后**的配置（model_dump(mode="json")，sort_keys），所以改 yaml 注释、调键的
顺序、把默认值显式写一遍，都不会触发重跑。

三条规矩（改 STAGE_FIELDS 之前先读）：

1. **拿不准就多挂**。挂漏一个字段 = 改了它不重跑，用户拿到旧产物且没有任何提示；多挂
   一个只是多重跑一次。唯一的例外是 LLM 阶段（translate / script）：多挂一个字段就是
   多付一次调用的钱，所以那两行是逐字段核对过才写的。tests/test_config_slices.py 有一条
   完整性测试：ProjectConfig 的每个叶子字段要么挂在至少一个阶段上，要么在 EXCLUDED 里，
   新增字段忘了登记它会直接红。
2. **EXCLUDED 只收「不影响产物内容」的运维旋钮**（超时、重试次数、并发度、代理、二进制
   路径）。validation_retries / budget_rewrite_rounds 决定「采纳哪一版稿子」，不许进来。
3. **只放本集的 EpisodeConfig**。登记第 11 集只改 E11 的切片，E1~E10 的逐字节不变。

本模块不许 import tenmin.pipeline（pipeline 反过来 import 本模块，会成环）。
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tenmin import atomic
from tenmin.config import EpisodeConfig, ProjectConfig

SLICE_DIR = ".config"

# 特殊记号：本集的整个 EpisodeConfig（number / srt / video / op_range / ed_range）。
EPISODE = "episode"

# 切片不带集号的阶段。signals 的配置里没有任何按集的东西，它本来就是一次跑完全部集的
# 全局阶段。ingest 同样是全局阶段，但它读每一集的 op/ed 与源文件，所以切片按集分开 ——
# 否则登记新集会让它那唯一一份切片变掉。
PROJECT_LEVEL_STAGES = frozenset({"signals"})

# 阶段 → 它读到的配置。条目有三种：子 config 名（整段，扣掉 EXCLUDED 里的字段）、
# 「子 config.字段」（只要这一个）、顶层标量字段名；外加 EPISODE 记号。
# 键的顺序必须跟 pipeline.STAGES 一致（有测试锁着）。
STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    # run_ingest：resolve_subtitle_source 读 asr 与本集 srt/video；build_track 读 locale、
    # ingest、credits、glossary（clean_text 的术语替换）、show（is_credits 的剧名比对）与
    # 本集 op/ed；_source_duration 探本集 video。
    "ingest": ("locale", "ingest", "credits", "asr", "glossary", "show", EPISODE),
    # translate_track 只读 glossary（llm.max_attempts 在排除表里）；provider 由下面 6 个
    # 字段构造。刻意不挂整个 llm、也不挂本集 EpisodeConfig：这是一次 LLM 调用的钱，那些
    # 字段它一个都不读 —— 对白轨是它唯一的数据通路，本集 op/ed/源片真变了会先经 ingest
    # 改写对白轨，照样让它重跑。
    "translate": (
        "glossary",
        "llm.provider",
        "llm.model",
        "llm.base_url",
        "llm.thinking",
        "llm.temperature",
        "llm.max_output_tokens",
    ),
    "signals": ("signals",),
    # build_user_prompt / generate_script / to_script 读 show、target_seconds、glossary、
    # render.rate、整个 llm（budget_tolerance / validation_retries / budget_rewrite_rounds
    # 与 provider 构造字段）与 validate_script。mode 只有 run_pipeline 的入口判据在读
    # （season 直接抛），拿不准所以照挂。
    "script": (
        "llm",
        "validate_script",
        "target_seconds",
        "mode",
        "glossary",
        "show",
        "render.rate",
    ),
    # docgen 其实不读配置（对照表的估算列用的是 script.json 里存好的 est_seconds），
    # render.rate 是按「拿不准就多挂」留的。
    "docgen": ("render.rate",),
    # build_tts_engine 的 voice / rate，synthesize_track 用 rate 算留白落点。
    "voice": ("render.voice", "render.rate"),
    # run_timeline：render_ass（字号、字体、PlayRes）、check_cue_legibility（字号、宽度、
    # 行数与时长下限）、build_timeline（只读 drift_tolerance），外加本集源片（探时长与帧率）。
    "timeline": (
        "render.font_size",
        "render.subtitle_font_name",
        "render.width",
        "render.height",
        "render.subtitle_max_lines",
        "render.subtitle_min_seconds",
        "render.drift_tolerance",
        EPISODE,
    ),
    # run_audio → mix_audio，外加本集源片（原声轨）。
    "audio": (
        "render.duck_db",
        "render.fade_out_seconds",
        "render.outro_card_seconds",
        "render.audio_codec",
        "render.audio_bitrate",
        "render.limiter_ceiling",
        EPISODE,
    ),
    # run_render → render_video；show 进片尾卡标题。font_size / subtitle_font_name 的真正
    # 消费者是 timeline 写的 .ass，这里是多挂的。
    "render": (
        "render.video_encoder",
        "render.width",
        "render.height",
        "render.crf",
        "render.preset",
        "render.tune",
        "render.videotoolbox_bitrate",
        "render.fade_out_seconds",
        "render.outro_card_seconds",
        "render.outro_message",
        "render.outro_font_name",
        "render.font_size",
        "render.subtitle_font_name",
        "show",
        EPISODE,
    ),
}

# 不影响产物内容、从不触发重跑的运维旋钮。
EXCLUDED: frozenset[str] = frozenset(
    {
        # 项目标识，只决定 work/<slug>/ 这个目录名，不进任何产物。
        "slug",
        "llm.timeout_seconds",
        "llm.read_timeout_seconds",
        "llm.total_timeout_seconds",
        "llm.transport_max_attempts",
        # schema 修复轮数：决定「这次调用成不成」，不决定「成了之后的稿子长什么样」。
        "llm.max_attempts",
        "llm.script_concurrency",
        "render.tts_max_attempts",
        "render.tts_concurrency",
        "render.tts_proxy",
        "render.tts_connect_timeout",
        "render.tts_receive_timeout",
        "render.tts_chunk_timeout_seconds",
        "render.ffmpeg_path",
        "render.ffprobe_path",
    }
)

_SUBCONFIGS = frozenset(
    name
    for name, field in ProjectConfig.model_fields.items()
    if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)
)


def slice_path(root: Path, stage: str, episode: int | None) -> Path:
    """切片文件的位置：项目级阶段是 `.config/<stage>.json`，其余 `.config/E{NN}.<stage>.json`。

    集号前缀跟 pipeline.episode_stem 同一个形状；这里不 import 它，理由见模块 docstring。
    """
    if stage in PROJECT_LEVEL_STAGES:
        return Path(root) / SLICE_DIR / f"{stage}.json"
    if episode is None:
        raise ValueError(f"{stage} 阶段的配置切片是按集的，必须给集号")
    return Path(root) / SLICE_DIR / f"E{episode:02d}.{stage}.json"


def slice_payload(cfg: ProjectConfig, stage: str, episode: EpisodeConfig | None) -> str:
    """这个阶段读到的那部分解析后配置，序列化成固定格式的 JSON。"""
    dumped = cfg.model_dump(mode="json", exclude={"episodes"})
    data: dict[str, Any] = {}
    for entry in STAGE_FIELDS[stage]:
        if entry == EPISODE:
            if episode is None:
                raise ValueError(f"{stage} 阶段的切片含本集配置，必须给 episode")
            data[EPISODE] = episode.model_dump(mode="json")
        elif "." in entry:
            top, leaf = entry.split(".", 1)
            data.setdefault(top, {})[leaf] = dumped[top][leaf]
        elif entry in _SUBCONFIGS:
            data[entry] = {
                key: value
                for key, value in dumped[entry].items()
                if f"{entry}.{key}" not in EXCLUDED
            }
        else:
            data[entry] = dumped[entry]
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_slice(path: Path, payload: str) -> None:
    """payload 跟盘上逐字节相同就不碰文件；**第一次创建时把 mtime 设成 epoch 0**。

    epoch 0 是升级取舍：换成按切片判新鲜度之后，第一次运行时所有切片都是新文件，按正常
    mtime 写的话它们比全部存量产物都新，整季重跑一遍、重付一轮 LLM。代价是：升级前改过、
    但还没跑过的配置，升级后第一次运行检测不到（要 --force）。之后的修改按正常 mtime 写。
    同理，手动删掉 .config/ 之后重建出来的切片也是 epoch 0。
    """
    existed = path.exists()
    if atomic.write_text_if_changed(path, payload) and not existed:
        os.utime(path, ns=(0, 0))


def write_slices(cfg: ProjectConfig, episodes: Sequence[EpisodeConfig]) -> None:
    """给每个阶段写切片：项目级阶段一份，其余阶段 episodes 里每集一份。"""
    for stage in STAGE_FIELDS:
        if stage in PROJECT_LEVEL_STAGES:
            write_slice(slice_path(cfg.root, stage, None), slice_payload(cfg, stage, None))
            continue
        for episode in episodes:
            write_slice(
                slice_path(cfg.root, stage, episode.number),
                slice_payload(cfg, stage, episode),
            )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_atomic.py tests/test_config_slices.py -q`
Expected: 全部 PASS。

- [ ] **Step 5: 变异自检（完整性测试确实咬人）**

临时把 `STAGE_FIELDS["timeline"]` 里的 `"render.drift_tolerance",` 那一行删掉，然后：

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_config_slices.py::test_every_config_leaf_is_mapped_or_excluded -q`
Expected: FAIL，报 `{'render.drift_tolerance'}`。

把那一行原样加回去（这个文件此时还没提交，不能用 git checkout 还原），再跑一次同一条命令确认 PASS。

- [ ] **Step 6: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 7: 提交**

```bash
rtk git add src/tenmin/atomic.py src/tenmin/config_slices.py tests/test_atomic.py tests/test_config_slices.py && rtk git commit -m "feat: 新增按阶段的配置切片模块与映射完整性测试"
```

---

### Task 8: 切片接进 `run_pipeline`，替换 `project.yaml` 这个输入

**Files:**
- Modify: `src/tenmin/pipeline.py`（import `config_slices`；`_is_fresh` docstring 与注释；`run_pipeline` 的切片落盘、`is_fresh` 包装与全部 9 个阶段的调用点）
- Modify: `src/tenmin/config.py`（`config_path` docstring）
- Modify: `src/tenmin/cli.py`（`init` 里的一段注释）
- Modify: `tests/test_source_hygiene.py`（一段注释）
- Test: `tests/test_pipeline.py`（**删掉** `test_run_pipeline_reruns_every_stage_when_project_yaml_changes` 与 `test_run_pipeline_reruns_render_stages_when_project_yaml_changes`，换成下面的新用例）

- [ ] **Step 1: 写失败的测试**

`tests/test_pipeline.py` 里把 `test_run_pipeline_reruns_every_stage_when_project_yaml_changes` 与 `test_run_pipeline_reruns_render_stages_when_project_yaml_changes` 两个函数整个删掉，在原位置换成：

```python
@pytest.mark.asyncio
async def test_run_pipeline_reruns_the_stage_whose_config_changed(project):
    """改了 validate_script 只有读它的 script 重跑（docgen 因为 script.json 变了跟着
    重跑）；ingest / signals 一个字都没读它，必须跳过。"""
    provider = FakeProvider([fake_script_response(), fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)

    project.validate_script.max_beats = 7
    reporter = FakeReporter()
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)

    assert ("stage_skip", "ingest") in reporter.calls
    assert ("stage_skip", "signals") in reporter.calls
    assert ("stage_start", "script") in reporter.calls
    assert ("stage_start", "docgen") in reporter.calls
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_run_pipeline_reruns_ingest_when_its_config_changes(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest"])

    project.credits.op_span_min = 30.0
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=["ingest"], reporter=reporter)

    assert ("stage_start", "ingest") in reporter.calls


@pytest.mark.asyncio
async def test_touching_project_yaml_no_longer_reruns_anything(project):
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)

    _shift_mtime(project.config_path, 10.0)
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=V1_STAGES, reporter=reporter)

    for stage in V1_STAGES:
        assert ("stage_skip", stage) in reporter.calls, stage


@pytest.mark.asyncio
async def test_an_operational_knob_reruns_nothing(project):
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)

    project.llm.timeout_seconds = 999.0
    project.llm.script_concurrency = 3
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=V1_STAGES, reporter=reporter)

    for stage in V1_STAGES:
        assert ("stage_skip", stage) in reporter.calls, stage


@pytest.mark.asyncio
async def test_comment_and_default_value_edits_in_project_yaml_rerun_nothing(
    tmp_path, golden_srt_path
):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "E02.srt").write_bytes(golden_srt_path.read_bytes())
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\ntarget_seconds: 240\n"
        "episodes:\n- number: 2\n  srt: srt/E02.srt\n",
        encoding="utf-8",
    )
    await run_pipeline(
        load_project(yaml_path), FakeProvider([fake_script_response()]), only=V1_STAGES
    )

    edited = "# 加一行注释\n" + yaml_path.read_text(encoding="utf-8").replace(
        "target_seconds: 240", "target_seconds: 240.0  # 显式写一遍\nmode: single_episode"
    )
    yaml_path.write_text(edited, encoding="utf-8")
    _shift_mtime(yaml_path, 10.0)
    reporter = FakeReporter()
    await run_pipeline(
        load_project(yaml_path), FakeProvider([]), only=V1_STAGES, reporter=reporter
    )

    for stage in V1_STAGES:
        assert ("stage_skip", stage) in reporter.calls, stage


@pytest.mark.asyncio
async def test_run_pipeline_reruns_the_render_stages_when_the_voice_changes(
    project, monkeypatch
):
    """voice 之后的阶段吃的是各自的切片与上游产物：换配音音色 → voice 重跑 →
    voice.json 刷新 → timeline / audio / render 跟着重跑。"""
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    # voice 重跑时 synthesize_track 会复用上一轮落盘的 chunk 并用 ffprobe 量时长，
    # 而 FakeTTSEngine 写的是假 mp3 字节。
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 8.0)
    monkeypatch.setattr("tenmin.render.audio.run_with_progress", _touch_output)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)

    stages = ["voice", "timeline", "audio", "render"]
    await run_pipeline(
        project, FakeProvider([]), only=stages, tts_engine=FakeTTSEngine([8.0] * 3)
    )

    project.render.voice = "zh-CN-XiaoxiaoNeural"
    reporter = FakeReporter()
    await run_pipeline(
        project,
        FakeProvider([]),
        only=stages,
        tts_engine=FakeTTSEngine([8.0] * 3),
        reporter=reporter,
    )
    for stage in stages:
        assert ("stage_start", stage) in reporter.calls, stage
        assert ("stage_skip", stage) not in reporter.calls, stage


@pytest.mark.asyncio
async def test_run_pipeline_backdates_freshly_created_slices(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest"])

    slices = project.root / ".config"
    assert (slices / "signals.json").stat().st_mtime_ns == 0
    assert (slices / "E02.ingest.json").stat().st_mtime_ns == 0
    assert (slices / "E02.script.json").stat().st_mtime_ns == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_pipeline.py -q -k "config_changed or ingest_when_its_config or touching_project_yaml or operational_knob or comment_and_default or when_the_voice_changes or backdates"`
Expected: 前两条与 voice 那条 FAIL（内存里改配置、project.yaml 的 mtime 没动 → 全部 stage_skip）；`touching_project_yaml` 那条 FAIL（`FakeProvider 的预置响应已用尽`，因为 project.yaml 仍是输入）；`backdates` 那条 FAIL（`.config/` 不存在）。`operational_knob` 与 `comment_and_default` 两条此时可能直接 PASS（内存改动/同一次运行不触发旧判据），它们是给实现兜底的回归锁。

- [ ] **Step 3: 实现**

`src/tenmin/pipeline.py`：`from tenmin import atomic` 改成 `from tenmin import atomic, config_slices`。

`_is_fresh` docstring 的第 2 条换成：

```
    2. **inputs 必须包含这个阶段的配置切片**（run_pipeline 里的 is_fresh 包装自动加上，
       见 tenmin.config_slices）。漏了它就等于这个阶段的配置旋钮改了都不生效。原来这里
       放的是整个 project.yaml，结果是改任何一个旋钮、登记任何一集，全部阶段一起过期。
```

`_is_fresh` 函数体里那段注释（`# 刻意跳过而不是重跑：…` 共 4 行）换成：

```python
        # 刻意跳过而不是重跑：一个输入都不存在时重跑只可能崩（run_ingest 读不到 SRT）
        # 或产出垃圾，而跳过至少保住磁盘上已有的产物。配置切片进了 inputs 之后这个分支在
        # run_pipeline 里已经走不到（切片在开跑前必然已落盘），留着只为不给库调用方/单测
        # 埋 FileNotFoundError。
```

`run_pipeline` 里：把

```python
    def is_fresh(outputs: list[Path], inputs: list[Path]) -> bool:
        """每个阶段的判据都自动带上 project.yaml。

        它是所有阶段的隐式输入：ingest/credits/signals 的全部阈值、glossary、
        render 的字号与编码器都住在那里。漏掉它的话「改配置再重跑」会被全部
        stage_skip，用户拿到的产物跟改动前一模一样且没有任何提示。
        """
        return _is_fresh(outputs, [cfg.config_path, *inputs])
```

整段删掉；在 `target_numbers` 的 if/else 之后、`number_to_index = …` 之前插入：

```python
    # 开跑前把每个阶段读到的那部分配置落成切片（内容没变不碰文件）。放在集号解析之后：
    # --episode 指到一个没注册 / 预填的集时应该先报错，而不是先往磁盘上写东西。
    config_slices.write_slices(cfg, active)

    def is_fresh(
        stage: str, number: int | None, outputs: list[Path], inputs: list[Path]
    ) -> bool:
        """每个阶段的判据都自动带上它自己的配置切片（替换原来的整个 project.yaml）。

        漏掉切片的话「改配置再重跑」会被 stage_skip，用户拿到的产物跟改动前一模一样且
        没有任何提示；反过来把整个 project.yaml 放进来，则是改任何一个旋钮都整季重跑。
        """
        return _is_fresh(
            outputs, [config_slices.slice_path(cfg.root, stage, number), *inputs]
        )
```

ingest 分枝换成：

```python
    if "ingest" in wanted:
        outputs = [paths.dialogue(n) for n in numbers]
        # ingest 是全局阶段，但切片按集：每一集的切片都是它的输入。
        inputs = [
            *ingest_inputs,
            *(config_slices.slice_path(cfg.root, "ingest", n) for n in numbers),
        ]
        if force or not _is_fresh(outputs, inputs):
            reporter.stage_start("ingest")
            warnings.extend(ingest_warnings(run_ingest(cfg)))
            reporter.stage_done("ingest")
        else:
            reporter.stage_skip("ingest")
```

signals 分枝的判据改成 `if force or not is_fresh("signals", None, outputs, inputs):`。

其余 7 处判据逐一改成带阶段名与集号的形式：

- `_launch_scripts` 里：`if force or not is_fresh("script", number, [paths.script(number)], inputs):`
- translate：`if force or not is_fresh("translate", number, outputs, _translate_inputs(paths, number)):`
- docgen：`if force or not is_fresh("docgen", number, outputs, [paths.script(number)]):`
- voice：`if force or not is_fresh("voice", number, outputs, [paths.script(number)]):`
- timeline：`if force or not is_fresh("timeline", number, outputs, inputs):`
- audio：`if force or not is_fresh("audio", number, outputs, inputs):`
- render：`if force or not is_fresh("render", number, outputs, inputs):`

`_launch_scripts` docstring 里「project.yaml 不变，paths.script(n) 也只会被第 n 集自己的 task 写。」改成「配置切片在开跑前就写完了，paths.script(n) 也只会被第 n 集自己的 task 写。」

`src/tenmin/config.py` 的 `config_path` docstring 换成：

```python
        """project.yaml 自身的路径。

        register_episode 要改写它。它**不再**是任何阶段的新鲜度输入：每个阶段只看
        tenmin.config_slices 从解析后配置里切出来的那一份，改注释、改无关旋钮、登记新集
        都不会让别的阶段过期。

        文件名写死 "project.yaml"：CLI 的 _project_file 只会去找这个名字，
        register_episode 原本也是这么拼的，这里只是把这份假设收敛到一处。
        """
```

`src/tenmin/cli.py` 的 `init` 里那段注释的前两行

```python
    # 原子写，跟 register_episode 改写同一个文件时用的是同一层：project.yaml 是**每个
    # 阶段**的隐式输入（pipeline._is_fresh 把它加进 inputs），所以它属于「产物」。
```

换成

```python
    # 原子写，跟 register_episode 改写同一个文件时用的是同一层：project.yaml 是全部
    # 配置切片的来源、也是已登记集数的唯一记录，所以它属于「产物」。
```

`tests/test_source_hygiene.py` 里

```python
# `project.yaml` 就是个反例 —— 它看着像「配置」，实际是**每个阶段**的隐式输入
# （`_is_fresh` 把它加进 inputs），所以 register_episode 与 cli.init 都必须原子写。
```

换成

```python
# `project.yaml` 就是个反例 —— 它看着像「配置」，实际是全部配置切片（`.config/*.json`，
# 每个阶段的新鲜度输入）的来源，所以 register_episode 与 cli.init 都必须原子写。
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: 全部 PASS（既有的 `test_run_pipeline_skips_a_fresh_translate`、`test_ingest_freshness_survives_a_video_only_episode` 等：切片首次创建是 epoch 0，不会比它们手造的产物新）。

- [ ] **Step 5: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/pipeline.py src/tenmin/config.py src/tenmin/cli.py tests/test_pipeline.py tests/test_source_hygiene.py && rtk git commit -m "feat: 新鲜度改按阶段配置切片判，不再整份依赖 project.yaml"
```

---
### Task 9: ingest / signals 内容不变不写，全局阶段按「上次跑完」戳子判新鲜度

**Files:**
- Modify: `src/tenmin/config_slices.py`（新增 `stamp_path`、`touch_stamp`）
- Modify: `src/tenmin/pipeline.py`（`_is_fresh` 之后新增 `_is_fresh_stamped`；`run_ingest` / `run_signals` 的写盘；`run_pipeline` 的 ingest / signals 两个分枝）
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_pipeline.py` 的 `from tenmin.pipeline import (...)` 列表里（按字母序放在 `_is_fresh,` 之后）加 `_is_fresh_stamped,`。末尾追加：

```python
# --- 全局阶段：内容不变不写 + 按「上次跑完」判新鲜度 -------------------------


def test_stamped_freshness_falls_back_to_the_outputs_without_a_stamp(tmp_path):
    """升级前的项目没有戳子：退回按产物判，不因为缺戳子白跑一遍。"""
    src = _file(tmp_path / "in.txt")
    out = _file(tmp_path / "out.txt")
    stamp = tmp_path / "stage.done"
    assert _is_fresh_stamped([out], [src], stamp) is True
    _shift_mtime(src, 10.0)
    assert _is_fresh_stamped([out], [src], stamp) is False


def test_stamped_freshness_judges_by_the_stamp_when_present(tmp_path):
    """产物内容没变就没重写、mtime 停在旧值，但这个阶段确实刚跑过。"""
    src = _file(tmp_path / "in.txt")
    out = _file(tmp_path / "out.txt")
    stamp = _file(tmp_path / "stage.done", "stage\n")
    _shift_mtime(src, 10.0)
    _shift_mtime(stamp, 20.0)
    assert _is_fresh_stamped([out], [src], stamp) is True
    _shift_mtime(src, 20.0)
    assert _is_fresh_stamped([out], [src], stamp) is False


def test_stamped_freshness_still_requires_every_output(tmp_path):
    src = _file(tmp_path / "in.txt")
    stamp = _file(tmp_path / "stage.done", "stage\n")
    _shift_mtime(stamp, 20.0)
    assert _is_fresh_stamped([tmp_path / "gone.json"], [src], stamp) is False
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    assert _is_fresh_stamped([empty], [src], stamp) is False


@pytest.mark.asyncio
async def test_rerunning_ingest_and_signals_on_unchanged_input_keeps_their_mtimes(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])
    paths = Paths(project.root)
    before = (paths.dialogue(2).stat().st_mtime_ns, paths.signals(2).stat().st_mtime_ns)

    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"], force=True)

    after = (paths.dialogue(2).stat().st_mtime_ns, paths.signals(2).stat().st_mtime_ns)
    assert after == before


@pytest.mark.asyncio
async def test_a_no_op_ingest_config_change_invalidates_nothing_downstream(project):
    """没有手填区间时 manual_window_margin 不参与任何判定：ingest 该重跑（它的切片变了），
    但产出一字不变，下游一个都不许被连带判过期；而且下一次运行 ingest 自己也得是新鲜的
    （不能因为产物 mtime 停在旧值就每次都重跑）。"""
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)

    project.credits.manual_window_margin = 6.0
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=V1_STAGES, reporter=reporter)
    assert ("stage_start", "ingest") in reporter.calls
    for stage in ("signals", "script", "docgen"):
        assert ("stage_skip", stage) in reporter.calls, stage

    again = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=V1_STAGES, reporter=again)
    for stage in V1_STAGES:
        assert ("stage_skip", stage) in again.calls, stage


@pytest.mark.asyncio
async def test_run_pipeline_writes_the_global_stage_stamps(project):
    await run_pipeline(project, FakeProvider([]), only=["ingest", "signals"])
    assert (project.root / ".config" / "ingest.done").is_file()
    assert (project.root / ".config" / "signals.done").is_file()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_pipeline.py -q -k "stamped or unchanged_input_keeps or no_op_ingest or global_stage_stamps"`
Expected: 收集阶段 `ImportError: cannot import name '_is_fresh_stamped'`。

- [ ] **Step 3: 实现**

`src/tenmin/config_slices.py` 末尾追加：

```python
def stamp_path(root: Path, stage: str) -> Path:
    """全局阶段（ingest / signals）「上次跑完」的戳子。判据见 pipeline._is_fresh_stamped。"""
    return Path(root) / SLICE_DIR / f"{stage}.done"


def touch_stamp(path: Path) -> None:
    """阶段成功跑完后调用。要的是它的 mtime；内容非空即可（0 字节会被 _is_fresh 当成
    被打断的半截产物）。"""
    atomic.write_text(path, f"{path.stem}\n")
```

`src/tenmin/pipeline.py`：在 `_is_fresh` 之后插入：

```python
def _is_fresh_stamped(outputs: list[Path], inputs: list[Path], stamp: Path) -> bool:
    """给「内容不变就不写」的全局阶段（ingest / signals）用的新鲜度判据。

    这两个阶段在产出跟盘上逐字节相同时不改写产物、保住旧 mtime，下游才不会因为「内容
    没变、mtime 变了」被连带判过期（登记一集新番时 ingest 会把全部集重跑一遍，这条就是
    让已有的集纹丝不动的关键）。代价是产物的 mtime 不再代表「这个阶段上次跑完的时刻」：
    改一个不影响产出的阈值，重跑之后产物照样比切片旧，纯按产物判就会**每次**都重跑。
    所以这里按「上次跑完」的戳子判，产物只查齐不齐、空不空（_is_fresh 在没有输入时就是
    这个语义）。

    戳子不存在（升级前的项目、或者从没成功跑完过）时退回按产物判，升级后第一次运行不会
    因为缺戳子白跑一遍。
    """
    if not _is_fresh(outputs, []):
        return False
    if not stamp.is_file():
        return _is_fresh(outputs, inputs)
    return _is_fresh([stamp], inputs)
```

`run_ingest` 里 `_write_json(paths.dialogue(episode.number), track.model_dump_json(indent=2))` 换成：

```python
        # 内容不变就不写：保住旧 mtime，下游 signals/script 才不会被连带判过期。
        atomic.write_text_if_changed(
            paths.dialogue(episode.number), track.model_dump_json(indent=2)
        )
```

`run_signals` 里 `_write_json(paths.signals(track.episode), report.model_dump_json(indent=2))` 换成：

```python
        # 同 run_ingest：内容不变就不写，别让 script 因为 mtime 刷新白跑。
        atomic.write_text_if_changed(
            paths.signals(track.episode), report.model_dump_json(indent=2)
        )
```

`run_pipeline` 的 ingest 分枝换成：

```python
    if "ingest" in wanted:
        outputs = [paths.dialogue(n) for n in numbers]
        # ingest 是全局阶段，但切片按集：每一集的切片都是它的输入。
        inputs = [
            *ingest_inputs,
            *(config_slices.slice_path(cfg.root, "ingest", n) for n in numbers),
        ]
        stamp = config_slices.stamp_path(cfg.root, "ingest")
        if force or not _is_fresh_stamped(outputs, inputs, stamp):
            reporter.stage_start("ingest")
            warnings.extend(ingest_warnings(run_ingest(cfg)))
            config_slices.touch_stamp(stamp)
            reporter.stage_done("ingest")
        else:
            reporter.stage_skip("ingest")
```

signals 分枝换成：

```python
    if "signals" in wanted:
        outputs = [paths.signals(n) for n in numbers]
        inputs = [
            config_slices.slice_path(cfg.root, "signals", None),
            *(paths.dialogue(n) for n in numbers),
        ]
        stamp = config_slices.stamp_path(cfg.root, "signals")
        if force or not _is_fresh_stamped(outputs, inputs, stamp):
            reporter.stage_start("signals")
            run_signals(cfg)
            config_slices.touch_stamp(stamp)
            reporter.stage_done("signals")
        else:
            reporter.stage_skip("signals")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: 全部 PASS（`test_ingest_freshness_survives_a_video_only_episode` 与 `test_a_newer_srt_makes_ingest_rerun` 手造了产物、没有戳子，走的正是「退回按产物判」那条，结论不变）。

- [ ] **Step 5: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/config_slices.py src/tenmin/pipeline.py tests/test_pipeline.py && rtk git commit -m "feat: ingest 与 signals 内容不变不写，按上次跑完的戳子判新鲜度"
```

---

### Task 10: 端到端验收：每次改动只有预期的阶段重跑

纯测试任务。前面 9 个任务做完之后这些用例**应当直接通过**；哪条不过就说明前面有漏洞，回到对应任务修，不要在这里改判据迁就测试。

**Files:**
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 写测试**

`tests/test_pipeline.py` 末尾追加：

```python
# --- 端到端：每次改动只有预期的阶段重跑 ---------------------------------------


def _full_run_project(project, monkeypatch) -> None:
    """让 voice → render 在测试里跑得起来：空壳视频、假 ffprobe、假 ffmpeg。"""
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 25.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder, **_: 1400.0)
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 80.0)
    monkeypatch.setattr("tenmin.render.audio.run_with_progress", _touch_output)
    monkeypatch.setattr("tenmin.render.video.run_with_progress", _touch_output_with_progress)


async def _first_full_run(project) -> None:
    # 80 秒一个 chunk：这份 fixture 的旁白真实会有的长度（理由见
    # test_run_pipeline_batch_mode_runs_full_pipeline_for_all_episodes）。
    await run_pipeline(
        project, FakeProvider([fake_script_response()]), tts_engine=FakeTTSEngine([80.0] * 20)
    )


async def _stages_rerun(project) -> set[str]:
    """再跑一遍全流程，返回真正重跑了的阶段。

    FakeProvider([]) 与 FakeTTSEngine([])：谁要是真去调 LLM 或合成语音，当场
    AssertionError。translate 刻意扣掉：繁中片源上它的产物永远不存在，所以每次都「开跑」
    然后在 run_translate 里零动作早退（见 run_translate 的 docstring），它出现在这里不代表
    任何配置被判过期。
    """
    reporter = FakeReporter()
    await run_pipeline(
        project, FakeProvider([]), tts_engine=FakeTTSEngine([]), reporter=reporter
    )
    started = {call[1] for call in reporter.calls if call[0] == "stage_start"}
    return started - {"translate"}


@pytest.mark.asyncio
async def test_an_untouched_project_reruns_nothing(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)
    assert await _stages_rerun(project) == set()


@pytest.mark.asyncio
async def test_changing_crf_reruns_only_render(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)

    project.render.crf = "18"

    assert await _stages_rerun(project) == {"render"}


@pytest.mark.asyncio
async def test_changing_font_size_reruns_only_the_subtitle_consumers(project, monkeypatch):
    """字号进的是 timeline 写的 .ass：timeline 重跑 → timeline.json 刷新 → audio 跟着
    重跑 → render 重跑。script（LLM）与 voice（TTS）一个都不许动。"""
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)

    project.render.font_size = 60

    assert await _stages_rerun(project) == {"timeline", "audio", "render"}


@pytest.mark.asyncio
async def test_operational_knobs_rerun_nothing_end_to_end(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)

    project.llm.timeout_seconds = 5.0
    project.llm.transport_max_attempts = 2
    project.render.tts_concurrency = 8
    project.render.ffprobe_path = "/opt/custom/bin/ffprobe"

    assert await _stages_rerun(project) == set()


@pytest.mark.asyncio
async def test_registering_a_new_episode_leaves_the_existing_one_untouched(
    project, golden_srt_path, tmp_path, monkeypatch
):
    """登记第 1 集：ingest / signals 会把全部集重跑一遍，但第 2 集的产物一个字节、
    一个 mtime 都不许变，它的 LLM 一次都不许再调。"""
    await run_pipeline(project, FakeProvider([fake_script_response()]), only=V1_STAGES)
    paths = Paths(project.root)
    watched = [
        paths.dialogue(2),
        paths.signals(2),
        paths.script(2),
        paths.table(2),
        paths.narration(2),
    ]
    before = [p.stat().st_mtime_ns for p in watched]
    slice_dir = project.root / ".config"
    slices_before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in slice_dir.glob("E02.*.json")
    }

    video = tmp_path / "e01.mkv"
    video.write_bytes(b"fake")
    # 新集的源片是个假文件：别让 ingest 去真跑 ffprobe。
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 1500.0)
    register_episode(project, episode=1, srt=golden_srt_path, video=video)

    provider = FakeProvider([fake_script_response(episode=1)])
    await run_pipeline(project, provider, only=V1_STAGES)

    assert len(provider.calls) == 1
    assert "集数：第 1 集" in provider.calls[0]["user"]
    assert paths.script(1).exists()
    assert [p.stat().st_mtime_ns for p in watched] == before
    slices_after = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in slice_dir.glob("E02.*.json")
    }
    assert slices_after == slices_before
```

- [ ] **Step 2: 跑测试**

Run: `uv run pytest tests/test_pipeline.py -q -k "untouched_project or changing_crf or changing_font_size or operational_knobs_rerun or registering_a_new_episode"`
Expected: 5 条全部 PASS。

- [ ] **Step 3: 变异自检（这批验收测试真的咬人）**

把 `run_ingest` 里的 `atomic.write_text_if_changed(` 临时改回 `_write_json(`（两个实参不变），然后：

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_pipeline.py::test_registering_a_new_episode_leaves_the_existing_one_untouched -q`
Expected: FAIL（第 2 集的 dialogue mtime 变了，进而 script 被判过期、FakeProvider 用尽）。

改回去，再临时把 `config_slices.STAGE_FIELDS["timeline"]` 里的 `"render.font_size",` 删掉：

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_pipeline.py::test_changing_font_size_reruns_only_the_subtitle_consumers -q`
Expected: FAIL（只有 render 重跑，timeline 没重跑 —— 正是「成片烧的还是旧字号的 .ass」）。

两处都还原：`rtk git checkout src/tenmin/pipeline.py src/tenmin/config_slices.py`，再跑一次 Step 2 的命令确认 5 条 PASS。

- [ ] **Step 4: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
rtk git add tests/test_pipeline.py && rtk git commit -m "test: 端到端锁住改配置与登记新集时只有预期阶段重跑"
```

---

### Task 11: 更新 AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: 改 config.py 段落**

在「改了子 config 的个数就来改这句数字（判据是 `ProjectConfig.model_fields` 里注解是 `BaseModel` 子类的那些）。」这句之后另起一段（缩进两格，跟它下面 `AsrConfig` 那段同级）：

```markdown
  全部 10 个配置模型继承 `StrictModel`（`extra="forbid"`；`Settings` 刻意除外，`.env` 里住着别的程序的变量）。`load_project` 用 `_UniqueKeyLoader`（SafeLoader + 同层重复键检查）读 YAML，YAML 语法错 / 重复键 / pydantic 校验失败一律翻成 `ProjectConfigError(ValueError)`，消息点名文件与字段完整路径（`render.font_sise`、`episodes[0].op_rang`），未知字段带 difflib 拼写建议。`EpisodeConfig` 允许 srt 与 video 同时为空的**预填**条目（先把 op/ed 写进 yaml、以后再登记视频）；「有没有来源」是运行时判据 `has_source`：批处理跳过并出一条 warning，`--episode N` 报错要 `--video`，`tenmin inspect` 标「未登记视频」，登记会并进这一条、保留 op/ed。
```

- [ ] **Step 2: 改 pipeline.py 段落**

在「`register_episode()`（`--episode --video` 注册新集，`--srt` 可省）。」之后另起一段：

```markdown
  **新鲜度只看「这个阶段读到的配置」**（`src/tenmin/config_slices.py`）：`run_pipeline` 开跑前把每个阶段读到的那部分解析后配置落成 `work/<slug>/.config/E{NN}.<stage>.json`（signals 是项目级的 `.config/signals.json`），阶段的新鲜度输入里用切片替换了原来的整个 `project.yaml` —— 改注释、改无关旋钮、登记新集都不再让别的阶段过期。映射表 `STAGE_FIELDS` 是逐字段 grep 各阶段读取点得来的：**让某个阶段开始读一个新配置字段时，必须同步把它挂到这个阶段上**（`tests/test_config_slices.py` 的完整性测试只抓得到「新增字段没挂到任何阶段」，抓不到「已挂在别的阶段、这个阶段新开始读」）；运维旋钮（超时、重试次数、并发度、代理、二进制路径、slug）在 `EXCLUDED` 里。切片内容不变不碰文件；**第一次创建时 mtime 设成 epoch 0**，所以升级不会整季重跑，代价是升级前改过、还没跑的配置第一次检测不到，要 `--force`（手动删掉 `.config/` 也是同样效果）。ingest / signals 产物内容不变就不写，下游才不会被连带判过期；因此这两个全局阶段按 `.config/{ingest,signals}.done` 这个「上次跑完」的戳子判新鲜度（`_is_fresh_stamped`，戳子缺失时退回按产物判）。`register_episode` 写回 yaml 走 `ruamel.yaml` round-trip，只改 `episodes` 里那一条的 srt/video，注释、键序、引号原样保留，序列缩进从原文件里猜；读配置仍走 PyYAML。要登记的 SRT 不存在时在动任何东西之前就报错。
```

同一段落里「默认 1 的三条依据见 `config.LLMConfig.script_concurrency` 的注释（最硬的一条：默认 provider 是 gemini，而它那条路上没有我们自己的传输层退避）。」换成：

```markdown
默认 1 的依据见 `config.LLMConfig.script_concurrency` 的注释（并发最容易撞 429、失败时在飞的调用白花钱、audio/render 的同步 ffmpeg 会堵住事件循环）。
```

- [ ] **Step 3: 改 llm.py 段落**

「  1. **传输层**（`_stream_with_retries`）：429/5xx 与连接类异常走指数退避 + 抖动 + `Retry-After`，次数由 `transport_max_attempts` 管。其余 4xx 与非限流的业务错误码立即失败。」换成：

```markdown
  1. **传输层**（OpenAI 兼容的 `_stream_with_retries` 与 Gemini 的 `_generate_with_retries`，同一套判据与退避）：429/5xx 与连接类异常走指数退避 + 抖动 + `Retry-After`，次数由 `transport_max_attempts` 管。其余 4xx 与非限流的业务错误码立即失败。Gemini 那边 `google.genai.errors.APIError` 按状态码判并包成 `LLMHTTPError`，连接类异常认 httpx 与 aiohttp 两族（google-genai 装了 aiohttp 就走它，edge-tts 会把它拽进来），耗尽后包成 `LLMTransportError`。
```

「  退避的 `_sleep` / `_rand` 是模块级函数，测试 monkeypatch 掉它们，所以**新增退避路径时不要改成直接 `asyncio.sleep`**，否则测试会真睡。」之后另起一段：

```markdown
  **用量**：`LLMUsage` 是一次 `complete()` 底下全部请求的合计（含 `cached_tokens`）。`script/usage.py` 的 `track_call` 在每次调用返回后**立刻**读 `provider.last_usage`（中间不许插 await，并发下才读得准），落到 `03_script/E{NN}.usage.json`（首稿 / 语义校验重试 / 时长返工各一条）与 `zh/E{NN}.usage.json`。这两个文件只用来看成本，不是任何阶段的新鲜度输入；失败的调用也记（`ok: false`）。
```

- [ ] **Step 4: 改测试段落**

「`tests/` 目录：`test_config.py`、`test_llm.py`、`test_pipeline.py`、`test_cli.py`、`test_render_*.py` 等，共 45 个文件（含 conftest.py / fakes.py / __init__.py）。」里的 `45` 改成 `46`，并在 `test_cli.py`、之后加上 `test_config_slices.py`、。

Run: `rtk ls tests/*.py | wc -l`
Expected: `46`（跟改后的数字一致）。

- [ ] **Step 5: 全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
rtk git add AGENTS.md && rtk git commit -m "docs: AGENTS.md 补配置切片、用量落盘、严格校验与预填条目的说明"
```

---

## Spec 覆盖表

| Spec 要求 | Task |
|---|---|
| 按阶段切片写到 `.config/<stage>.json` / `.config/E{NN}.<stage>.json`，内容相同不碰文件 | 7（`write_slice` / `write_slices`）、8（接线） |
| 切片是 `model_dump(mode="json")`、`sort_keys`、`ensure_ascii=False`、固定缩进 | 7（`slice_payload`） |
| 新鲜度输入里用切片替换 `cfg.config_path`，`_is_fresh` 不动 | 8 |
| 映射放在 `config_slices.py` 的模块级常量表，逐字段 grep 修正 | 7（`STAGE_FIELDS`，见上文「已核实的映射」与「偏离」1–5） |
| 排除表（14 个运维旋钮），`validation_retries` / `budget_rewrite_rounds` 不排除 | 7（`EXCLUDED` + 两条测试） |
| 完整性测试：每个叶子要么挂阶段、要么在排除表 | 7（`test_every_config_leaf_is_mapped_or_excluded` + 变异自检） |
| 只放本集 `EpisodeConfig`，登记 E11 不动 E1~E10 的切片 | 7（`test_registering_another_episode_leaves_this_episodes_slices_alone`）、10 |
| 切片首次创建 mtime = epoch 0 | 7、8（`test_run_pipeline_backdates_freshly_created_slices`） |
| ingest / signals 相同内容不写 | 9（外加戳子，见「偏离」6） |
| script / translate 写 usage.json，每条 provider/model/轮次/三种 token/耗时，缺数写 null，原子写，非新鲜度输入 | 4 |
| Gemini 与 OpenAI 兼容两条路径都把用量填上（含 cached） | 4 |
| `StrictModel(extra="forbid")`，10 个模型继承 | 1 |
| `load_project` 把未知字段翻成中文、带完整路径与 difflib 建议 | 1 |
| YAML 重复键报错，跟 ruamel round-trip 一致 | 1、6（一致性测试） |
| `load_project` / `register_episode` 挪进 `PIPELINE_ERRORS` 的 try；SRT 不存在 / yaml 语法错 / 校验失败给中文一行报错 | 2 |
| Gemini `APIError` → `LLMHTTPError`，连接类异常进 `LLMError` 族 | 3 |
| Gemini 传输层退避：429/5xx/连接类重试、`transport_max_attempts`、模块级 `_sleep`/`_rand`、其他 4xx 立即失败 | 3 |
| 预填条目可加载，`_require_a_source` 挪到运行时 | 5 |
| 批处理跳过预填条目并提示 `第 3 集还没有 video，已跳过` | 5（CLI 与库两层都有测试） |
| `--episode 3` 不带 `--video` 且是预填条目 → 中文报错要 `--video` | 5 |
| `--episode 3 --video` 并进预填条目、保留 op/ed（补测试锁住） | 5、6 |
| `tenmin inspect` 列出预填条目、标「未登记视频」 | 5 |
| `register_episode` 用 ruamel round-trip 只改对应条目，保留注释/顺序/引号，原子写；ruamel 进主依赖 | 6 |
| 端到端：改 render 字段、改运维旋钮、改注释、登记新集，只有预期阶段重跑 | 8、9、10 |
| 全量 `uv run pytest tests/ -q` 通过，MiniMax 现有测试行为不变 | 每个 Task 的全量步骤；3（MiniMax 专门复核） |
