# 生肉支持实现计划（ASR 取对白轨 + translate 阶段）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 tenmin 在「只有一个日语生肉视频、没有 SRT」时也能跑完整条管线，并额外交付一份中文字幕。

**Architecture:** 两处扩张。（1）ingest 之前加一个「对白轨从哪来」的解析层：手传 SRT → 视频内嵌软字幕轨 → mlx-whisper 转写，三条路都归一成一个 SRT 文件路径，`build_track` 拿到的东西形态不变。（2）ingest 之后加一个新阶段 `translate`：把日语对白逐条译成简体中文，产出中文字幕交付物，并累积一份跨集共享的专有名词表回喂 script 阶段。

**Tech Stack:** Python 3.12+、pydantic v2、typer、mlx-whisper（optional extra，仅 Apple Silicon）、ffmpeg/ffprobe（已有封装）、现有 LLM provider 抽象。

**Spec:** `docs/superpowers/specs/2026-09-21-tenmin-asr-translate-design.md`

## Global Constraints

- **`uv run` 不要套 rtk**：本项目所有 pipeline 命令与 pytest 一律用裸 `uv run <cmd>`，绝不写 `rtk pytest` / `rtk proxy uv run`（中文输出会把 rtk 的 UTF-8 抓取层搞崩）。裸的 `rtk ls` / `rtk grep` / `rtk git` 安全。
- **src/ 里禁止 `assert` 语句**（`tests/test_source_hygiene.py::test_no_assert_statements_in_src` 会扫全部 `src/**/*.py`）。校验一律 `raise`。
- **注释与 docstring 禁止引用内部 `file.py:123` 形式的行号**（`test_no_line_number_cross_references`）。要指路就写模块名或函数名。
- **注释里禁止内部任务代号**（`test_no_internal_task_codes_in_comments`）。不要写 "Task 3" / "P0-C" 这类字样进 src/ 或 tests/。
- **所有 `open()` 必须显式 `encoding=`**（`test_text_io_always_declares_encoding`）。
- **所有阶段产物的写盘必须走 `tenmin.atomic`**（`test_artifact_writes_go_through_atomic`）。
- **本项目刻意不引入 `logging`**：要告知用户就 `print`，要报错就 `raise`，要带出诊断就进 warnings 列表。
- **`_ARTIFACTS` 里的目录名与后缀是磁盘上的存量契约**，一个字符都不许改（`work/` 下已有 13 集产物）。只许新增条目。
- **`tests/test_pipeline.py` 的 `FROZEN_LAYOUT` 必须与 `_ARTIFACTS` 严格同步**，且 `test_paths_exposes_exactly_the_frozen_artifacts` 断言 `Paths` 上可调用的公开名字集合 == `set(FROZEN_LAYOUT)`。新增**按集**产物要同步进 `FROZEN_LAYOUT`；新增**项目级**产物必须做成 `@property`（property 实例不是 callable，自动不进那个集合）。
- **`mlx-whisper` 只能延迟 import**（写在函数体内）。它在 `[project.optional-dependencies]` 的 `asr` 里，没装 extra 的人必须能正常跑 `tenmin --help` 和全部非 ASR 阶段。
- **繁转简（OpenCC）绝不能作用在日语文本上**。ASR 路径必须让 `convert_traditional` 强制为 False。
- 提交信息沿用仓库风格：`feat: …` / `fix: …` / `test: …` / `docs: …`，中文正文。

## 与 spec 的两处偏离（实现时以本计划为准）

1. **spec 写 `resolve_cues(...) -> tuple[list[RawCue], source]`，实际返回 `Path`。** 原因：`build_track` 内部第一件事就是 `load_srt_detailed(srt_path)`，返回 cue 列表必然要改它的签名（而 spec 要求签名不动）；而三条路本来都能给出一个 SRT 文件（手传的、从视频抽的、ASR 落盘缓存的），返回路径既保住签名又顺手满足「ASR 结果要能被人手动修正」。
2. **spec 写「`run_ingest` 在 async 上下文，所以 mlx-whisper 走 `asyncio.to_thread`」是错的。** `run_ingest` 是同步函数，被 `run_pipeline` 直接调用，而且 ingest 是全局阶段（跑它时没有别的 task 在飞）。ASR 保持同步阻塞调用，不引入 async 管道。

---

### Task 1: `AsrConfig` + optional extra + `asr` marker 门

地基任务，零行为变化。

**Files:**
- Modify: `src/tenmin/config.py`（新增 `AsrConfig`、`ProjectConfig.asr` 字段、`DEFAULT_ASR`）
- Modify: `pyproject.toml`（新增 `[project.optional-dependencies]`、新增 `asr` marker）
- Modify: `tests/conftest.py`（`selects_render_marker` 泛化成 `selects_marker`，门扩到两个 marker）
- Modify: `tests/test_marker_gate.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces: `tenmin.config.AsrConfig`（字段 `model: str`、`language: str`）、`tenmin.config.DEFAULT_ASR`、`ProjectConfig.asr: AsrConfig`；`tests.conftest.selects_marker(expression: str | None, marker: str) -> bool`

- [ ] **Step 1: 写失败的测试**

在 `tests/test_config.py` 末尾追加：

```python
def test_asr_config_defaults():
    from tenmin.config import AsrConfig

    cfg = AsrConfig()
    assert cfg.model == "mlx-community/whisper-large-v3-turbo"
    assert cfg.language == "ja"


def test_project_config_carries_an_asr_section():
    from tenmin.config import AsrConfig, ProjectConfig

    cfg = ProjectConfig(show="测试番", slug="test")
    assert isinstance(cfg.asr, AsrConfig)
    assert cfg.asr.language == "ja"


def test_asr_section_is_overridable_from_yaml_shaped_data():
    from tenmin.config import ProjectConfig

    cfg = ProjectConfig.model_validate(
        {"show": "测试番", "slug": "test", "asr": {"language": "en", "model": "tiny"}}
    )
    assert cfg.asr.language == "en"
    assert cfg.asr.model == "tiny"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py -k asr -v`
Expected: FAIL，`ImportError: cannot import name 'AsrConfig'`

- [ ] **Step 3: 加 `AsrConfig`**

在 `src/tenmin/config.py` 里 `RenderConfig` 之后、`ProjectConfig` 之前插入：

```python
class AsrConfig(BaseModel):
    """语音转写参数。只在「这一集既没有手传 SRT、视频里也没有软字幕轨」时才用得上。

    刻意没有 engine 字段：本项目只支持 mlx-whisper 一个引擎（作者只在 Apple Silicon
    Mac 上自用，faster-whisper 的跳平台优势买不到东西），多一层引擎抽象是为不存在的
    需求付复杂度。真要换引擎时再加这个字段，那时也才知道抽象该切在哪。
    """

    # 实测：这个模型在 M 系列芯片上约 8 倍实时速度，一集 24 分钟的番约 3 分钟转完，
    # 转出来的日语跟画面硬字幕交叉核对过，语义级吻合。再大的模型换不来可感知的收益
    # （对白只是喂给 script 阶段当剧情理解材料，不进成片）。
    model: str = "mlx-community/whisper-large-v3-turbo"
    # 源片语言。不做自动检测：这是「素材是什么」的事实，让人填一次比让机器每集猜一次可靠。
    language: str = "ja"
```

在 `ProjectConfig` 的字段里，`render` 之后加一行：

```python
    asr: AsrConfig = Field(default_factory=AsrConfig)
```

在 `DEFAULT_RENDER = RenderConfig()` 之后加一行：

```python
DEFAULT_ASR = AsrConfig()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -k asr -v`
Expected: PASS（3 passed）

- [ ] **Step 5: pyproject 加 extra 与 marker**

在 `pyproject.toml` 的 `[project]` 依赖节之后、`[tool.*]` 之前插入：

```toml
[project.optional-dependencies]
# 语音转写（生肉片源）。刻意做成 extra 而不是主依赖：mlx-whisper 会把 torch 一起
# 拽进来（几个 G），而绝大多数用法是喂一份自带字幕的片源、压根不需要它。不跑生肉的人
# `uv sync` 照旧轻装，要跑生肉才 `uv sync --extra asr`。
# 只列 mlx-whisper 一个引擎，理由见 config.AsrConfig 的 docstring。
asr = ["mlx-whisper>=0.4"]
```

把 `markers` 列表改成（追加第四条）：

```toml
markers = [
    "llm: 需要真实 LLM API key，默认跳过（跑法：TENMIN_GEMINI_API_KEY=xxx uv run pytest -m llm）",
    "generalize: 需要额外番剧 SRT fixture，默认跳过（把 SRT 放进 tests/fixtures/generalize/ 后自动生效）",
    "render: 需要真实视频文件与编入 libass 的 ffmpeg，默认跳过（跑法：uv run pytest -m render）",
    "asr: 需要装了 asr extra 与真实视频，默认跳过（跑法：uv run pytest -m asr）",
]
```

- [ ] **Step 6: 把 conftest 的 marker 门泛化到两个 marker**

`tests/conftest.py`：把 `selects_render_marker` 改名为 `selects_marker` 并加一个 `marker` 参数，同时把 hook 改成遍历。完整替换那两个函数：

```python
# 默认跳过的 marker，以及跳过时给人看的那句话。
# 共同点是「一次普通 pytest 里意外跑到它会付真实代价」：render 要联网调 Edge-TTS 并
# 真编一段视频，asr 要一个几 G 的模型跑三分钟。llm 与 generalize 刻意不进这张表 ——
# 理由见 pytest_collection_modifyitems 的 docstring。
_GATED_MARKERS = {
    "render": "需要真实素材与 ffmpeg，跑法：uv run pytest -m render",
    "asr": "需要 asr extra 与真实视频，跑法：uv run pytest -m asr",
}


def selects_marker(expression: str | None, marker: str) -> bool:
    """`-m` 表达式是否**正向**点名了某个 marker。

    原判据是子串匹配（`marker in expression`）。`-m "not render"` 在它下面当前是安全
    的（pytest 自己的 deselect 先生效，那些用例压根不进 items），所以那不是个现存 bug；
    但它对「以后加一个名字含 render 的 marker」零容错 —— `-m render_e2e` 会让子串命中、
    门打开，于是默认不该跑的 render 用例（真调 Edge-TTS、真编一段视频）被放进来。

    收紧成「按标识符切词，且不能是紧跟在 `not` 后面的那个」。刻意**不**实现完整的布尔
    表达式求值：这个门只需要回答「用户有没有明确点名它」，而方向一律取保守（拿不准就
    跳过）—— 它保护的是「别在一次普通 `pytest` 里意外联网/编码/跑模型」，误跳的代价是
    一句「跑法：uv run pytest -m xxx」，误跑的代价是网络一抖整个测试套变红。

    判据与用例在 tests/test_marker_gate.py。
    """
    tokens = _MARKER_TOKEN.findall(expression or "")
    return any(
        token == marker and (index == 0 or tokens[index - 1] != "not")
        for index, token in enumerate(tokens)
    )


def pytest_collection_modifyitems(config, items):
    """没有显式点名时，跳过 _GATED_MARKERS 里那些 marker 的用例。

    pyproject 里这些 marker 写的都是「默认跳过（跑法：uv run pytest -m xxx）」，执行
    这条约定的就是这个 hook。默认的 `uv run pytest` 若真去调一次 Edge-TTS（要联网）、
    真编一段 720p 视频、或真加载一个几 G 的语音模型：多花几分钟是小事，网络一抖整个
    测试套变红才是问题。

    刻意只管这张表里的，不管 llm / generalize：
    - llm 的门是「有没有 API key」，缺 key 时自己 skip，比按 marker 摘更准；
    - generalize 的约定是「把 SRT 放进 tests/fixtures/generalize/ 后**自动生效**」，
      按 marker 摘掉会把这条约定打死。
    """
    expression = config.getoption("-m")
    for marker, reason in _GATED_MARKERS.items():
        if selects_marker(expression, marker):
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
```

- [ ] **Step 7: 更新 marker 门的测试**

`tests/test_marker_gate.py`：把 import 与两个测试改成：

```python
from .conftest import selects_marker


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # 正向要 render
        ("render", True),
        (" render ", True),
        ("render and not slow", True),
        ("slow or render", True),
        ("not slow and render", True),
        # 没要
        ("", False),
        (None, False),
        ("not render", False),
        ("not  render", False),
        ("slow", False),
        # **子串匹配会判错、按词切不会**的那一组：这才是收紧的理由
        ("render_e2e", False),
        ("renderx", False),
        ("not render_e2e", False),
        ("prerender", False),
    ],
)
def test_selects_render_marker(expression, expected):
    assert selects_marker(expression, "render") is expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("asr", True),
        ("asr and not slow", True),
        ("not asr", False),
        ("", False),
        (None, False),
        ("render", False),
        # 两个门互不串：点名 render 不该把 asr 的门也打开
        ("asrx", False),
        ("not asr_e2e", False),
    ],
)
def test_selects_asr_marker(expression, expected):
    assert selects_marker(expression, "asr") is expected


def test_the_gate_is_wired_to_the_collection_hook():
    """守住这个测试自己：判据函数必须真的是 hook 在用的那一个。"""
    import inspect

    from . import conftest

    source = inspect.getsource(conftest.pytest_collection_modifyitems)
    assert "selects_marker" in source
```

- [ ] **Step 8: 跑全量测试**

Run: `uv run pytest tests/ -q`
Expected: 全绿（不该有任何既有测试变红 —— 这一步零行为变化）

- [ ] **Step 9: 提交**

```bash
rtk git add pyproject.toml src/tenmin/config.py tests/conftest.py tests/test_marker_gate.py tests/test_config.py
rtk git commit -m "feat: 加 AsrConfig、asr optional extra 与 asr marker 门"
```

---

### Task 2: ffmpeg 层的三个新探测/抽取函数

**Files:**
- Modify: `src/tenmin/render/ffmpeg.py`
- Test: `tests/test_render_ffmpeg.py`

**Interfaces:**
- Consumes: `ffmpeg._probe_field`、`ffmpeg.run`、`ffmpeg.FFmpegError`（均已存在）
- Produces:
  - `has_subtitle_stream(path: Path, *, ffprobe: str = FFPROBE) -> bool`
  - `extract_subtitle_track(video: Path, dest: Path, *, ffmpeg: str = FFMPEG) -> None`
  - `extract_audio_track(video: Path, dest: Path, *, ffmpeg: str = FFMPEG) -> None`

- [ ] **Step 1: 写失败的测试**

在 `tests/test_render_ffmpeg.py` 末尾追加（fake 的路子跟该文件既有的 `_probe_field` 测试一致 —— monkeypatch `subprocess.run`）：

```python
class _FakeCompleted:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_has_subtitle_stream_true_when_ffprobe_names_a_stream(tmp_path, monkeypatch):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        ffmpeg.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout="2\n")
    )
    assert ffmpeg.has_subtitle_stream(video) is True


def test_has_subtitle_stream_false_on_empty_stdout(tmp_path, monkeypatch):
    """没有字幕轨时 ffprobe 退出码仍是 0、stdout 为空 —— 只能看 stdout。"""
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(ffmpeg.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout="\n"))
    assert ffmpeg.has_subtitle_stream(video) is False


def test_has_subtitle_stream_selects_the_subtitle_stream(tmp_path, monkeypatch):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    seen: list[list[str]] = []

    def fake_run(args, **kwargs):
        seen.append(list(args))
        return _FakeCompleted(stdout="0\n")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    ffmpeg.has_subtitle_stream(video)
    assert "-select_streams" in seen[0]
    assert seen[0][seen[0].index("-select_streams") + 1] == "s"


def test_extract_subtitle_track_builds_an_srt_command(tmp_path, monkeypatch):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    dest = tmp_path / "out.srt"
    seen: list[list[str]] = []
    monkeypatch.setattr(ffmpeg, "run", lambda args, **k: seen.append(list(args)) or "")

    ffmpeg.extract_subtitle_track(video, dest)

    args = seen[0]
    assert args[args.index("-i") + 1] == str(video)
    assert args[args.index("-map") + 1] == "0:s:0"
    assert args[-1] == str(dest)
    assert "-y" in args


def test_extract_audio_track_builds_a_16k_mono_wav_command(tmp_path, monkeypatch):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    dest = tmp_path / "out.wav"
    seen: list[list[str]] = []
    monkeypatch.setattr(ffmpeg, "run", lambda args, **k: seen.append(list(args)) or "")

    ffmpeg.extract_audio_track(video, dest)

    args = seen[0]
    assert "-vn" in args
    assert args[args.index("-ac") + 1] == "1"
    assert args[args.index("-ar") + 1] == "16000"
    assert args[args.index("-c:a") + 1] == "pcm_s16le"
    assert args[-1] == str(dest)


def test_extract_helpers_reject_a_missing_video(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(ffmpeg.FFmpegError):
        ffmpeg.extract_audio_track(missing, tmp_path / "o.wav")
    with pytest.raises(ffmpeg.FFmpegError):
        ffmpeg.extract_subtitle_track(missing, tmp_path / "o.srt")
```

（若该测试文件顶部尚未 `import pytest` / `from tenmin.render import ffmpeg`，按文件既有写法补上。）

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_render_ffmpeg.py -k "subtitle_stream or extract_" -v`
Expected: FAIL，`AttributeError: module 'tenmin.render.ffmpeg' has no attribute 'has_subtitle_stream'`

- [ ] **Step 3: 实现三个函数**

在 `src/tenmin/render/ffmpeg.py` 里 `has_audio_stream` 之后追加：

```python
def has_subtitle_stream(path: Path, *, ffprobe: str = FFPROBE) -> bool:
    """视频里有没有软字幕轨。

    跟 has_audio_stream 同形。注意判据是**stdout 有没有内容**，不是退出码：ffprobe
    在「容器合法但没有这种流」时退出码仍为 0、只是什么都不打印。
    """
    return bool(
        _probe_field(path, "stream=index", "字幕轨", ffprobe=ffprobe, stream="s")
    )


def extract_subtitle_track(video: Path, dest: Path, *, ffmpeg: str = FFMPEG) -> None:
    """把第一条软字幕轨抽成 SRT。

    只抽第一条（`0:s:0`）：多语字幕的片源要选哪一条是个需要人判断的问题，而本项目遇到
    的片源要么零字幕轨、要么正好一条。真碰到多轨时宁可让人手传 --srt，也不猜。

    容器里可能是 ASS/SSA，`-c:s srt` 让 ffmpeg 负责转成 SRT（特效标记会被丢掉，而
    ingest 的清洗层本来就要剥掉它们）。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "-y",
            "-i",
            str(video),
            "-map",
            "0:s:0",
            "-c:s",
            "srt",
            str(dest),
        ],
        ffmpeg=ffmpeg,
    )


def extract_audio_track(video: Path, dest: Path, *, ffmpeg: str = FFMPEG) -> None:
    """抽一条 16kHz 单声道 PCM wav，喂给语音转写。

    这三个参数是 whisper 系模型的原生输入格式：它内部无论如何都要重采样到 16k 单声道，
    在这里一次做完比让模型库自己去解 mp4 更快也更可控（也顺手绕开「模型库找不到
    ffmpeg」这类环境问题 —— 本项目已经有一份 ffmpeg 路径配置了）。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        ffmpeg=ffmpeg,
    )
```

两个抽取函数都要先做「源文件存在吗」的前置检查。`run` 本身不做这件事（`_probe_field` 才做），所以在两个函数体的开头、`dest.parent.mkdir` 之前各加：

```python
    if not Path(video).is_file():
        raise FFmpegError(f"找不到源视频: {video}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_render_ffmpeg.py -k "subtitle_stream or extract_" -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 跑该文件全量，确认没碰坏既有的**

Run: `uv run pytest tests/test_render_ffmpeg.py -q`
Expected: 全绿

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/render/ffmpeg.py tests/test_render_ffmpeg.py
rtk git commit -m "feat: ffmpeg 层加软字幕轨探测与字幕/音轨抽取"
```

---

### Task 3: `ingest/asr.py` —— 语音转写与幻觉过滤

**Files:**
- Create: `src/tenmin/ingest/asr.py`
- Modify: `src/tenmin/cli.py`（`PIPELINE_ERRORS` 加 `ASRUnavailableError`）
- Test: `tests/test_asr.py`

**Interfaces:**
- Consumes: `tenmin.models.RawCue`、`tenmin.config.AsrConfig`、`tenmin.render.ffmpeg.extract_audio_track`、`tenmin.atomic.atomic_path`
- Produces:
  - `ASRUnavailableError(RuntimeError)`
  - `ASRError(RuntimeError)`
  - `segments_to_cues(segments: Iterable[Mapping[str, object]]) -> list[RawCue]`
  - `render_srt(cues: Sequence[RawCue]) -> str`
  - `transcribe(video: Path, dest: Path, *, asr: AsrConfig, ffmpeg_path: str = "ffmpeg") -> None`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_asr.py`：

```python
"""语音转写层。刻意不测 whisper 的转写准确率（那是模型的事，且要几 G 模型 + 真视频），
只测我们自己那几层：段落转 cue 的过滤规则、SRT 渲染、缺依赖时的报错。"""

from pathlib import Path

import pytest

from tenmin.ingest import asr


def test_segments_to_cues_keeps_normal_segments():
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.5, "text": "こんにちは"},
            {"start": 3.0, "end": 4.0, "text": "元気ですか"},
        ]
    )
    assert [(c.idx, c.start, c.end, c.text) for c in cues] == [
        (1, 1.0, 2.5, "こんにちは"),
        (2, 3.0, 4.0, "元気ですか"),
    ]


def test_segments_to_cues_drops_zero_length_tail_hallucinations():
    """实测 whisper 在音频末尾会吐一串 start == end 的空段落，是边界 artifact。

    在这一层就丢掉，不指望下游清洗层接住 —— 让脏数据流进去会污染 ingest 的
    「跳过了几个块」统计，那个数字是给人判断片源质量的。
    """
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.0, "text": "本物"},
            {"start": 119.96, "end": 119.96, "text": ""},
            {"start": 119.96, "end": 119.96, "text": " "},
            {"start": 119.96, "end": 119.96, "text": ""},
        ]
    )
    assert len(cues) == 1
    assert cues[0].text == "本物"


def test_segments_to_cues_drops_reversed_spans():
    cues = asr.segments_to_cues([{"start": 5.0, "end": 4.0, "text": "壊れてる"}])
    assert cues == []


def test_segments_to_cues_drops_pronunciation_free_text():
    """纯标点/纯符号的段落没有内容，留着只会在对白轨里占一行。"""
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.0, "text": "……"},
            {"start": 3.0, "end": 4.0, "text": "♪"},
            {"start": 5.0, "end": 6.0, "text": "、。"},
            {"start": 7.0, "end": 8.0, "text": "ありがとう"},
        ]
    )
    assert [c.text for c in cues] == ["ありがとう"]


def test_segments_to_cues_keeps_latin_and_digits():
    """歌名/型号这类内容是有效对白，不能被「没假名就丢」的规则误杀。"""
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.0, "text": "OK"},
            {"start": 3.0, "end": 4.0, "text": "2026"},
        ]
    )
    assert [c.text for c in cues] == ["OK", "2026"]


def test_segments_to_cues_strips_surrounding_whitespace():
    cues = asr.segments_to_cues([{"start": 1.0, "end": 2.0, "text": "  はい  "}])
    assert cues[0].text == "はい"


def test_segments_to_cues_renumbers_after_dropping():
    cues = asr.segments_to_cues(
        [
            {"start": 1.0, "end": 2.0, "text": "一"},
            {"start": 2.0, "end": 2.0, "text": ""},
            {"start": 3.0, "end": 4.0, "text": "二"},
        ]
    )
    assert [c.idx for c in cues] == [1, 2]


def test_render_srt_uses_comma_millisecond_separator():
    """SRT 的毫秒分隔符是逗号。timecode.format_timestamp 产出的是点号版本，
    它的 docstring 明令不要拿它写 SRT，所以这里自己拼。"""
    from tenmin.models import RawCue

    text = asr.render_srt(
        [
            RawCue(idx=1, start=1.5, end=2.25, text="はい"),
            RawCue(idx=2, start=3661.007, end=3662.0, text="いいえ"),
        ]
    )
    assert "00:00:01,500 --> 00:00:02,250" in text
    assert "01:01:01,007 --> 01:01:02,000" in text
    assert text.startswith("1\n")
    assert "はい" in text and "いいえ" in text


def test_render_srt_round_trips_through_the_parser():
    """最硬的不变量：渲染出来的东西必须能被本项目自己的 SRT 解析器吃回去。"""
    from tenmin.ingest.srt_parser import parse_srt
    from tenmin.models import RawCue

    original = [
        RawCue(idx=1, start=1.5, end=2.25, text="はい"),
        RawCue(idx=2, start=10.0, end=12.125, text="そうですね"),
    ]
    reparsed = parse_srt(asr.render_srt(original))
    assert [(c.start, c.end, c.text) for c in reparsed] == [
        (1.5, 2.25, "はい"),
        (10.0, 12.125, "そうですね"),
    ]


def test_render_srt_of_nothing_is_empty():
    assert asr.render_srt([]) == ""


def test_transcribe_without_the_extra_raises_a_actionable_error(tmp_path, monkeypatch):
    """没装 asr extra 时要给出能照着做的一句话，而不是一个裸 ImportError。"""
    import builtins

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_whisper":
            raise ImportError("No module named 'mlx_whisper'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", lambda *a, **k: None)

    from tenmin.config import AsrConfig

    with pytest.raises(asr.ASRUnavailableError) as excinfo:
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())
    assert "uv sync --extra asr" in str(excinfo.value)


def test_transcribe_writes_an_srt_from_the_model_output(tmp_path, monkeypatch):
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    dest = tmp_path / "sub" / "out.srt"
    seen: dict[str, object] = {}

    def fake_extract(src, wav, **kwargs):
        Path(wav).write_bytes(b"fake wav")

    def fake_load(**kwargs):
        seen.update(kwargs)
        return {
            "segments": [
                {"start": 0.5, "end": 1.5, "text": "おはよう"},
                {"start": 2.0, "end": 2.0, "text": ""},
            ]
        }

    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", fake_extract)
    monkeypatch.setattr(asr, "_run_model", fake_load)

    asr.transcribe(video, dest, asr=AsrConfig(model="m", language="ja"))

    assert dest.is_file()
    body = dest.read_text(encoding="utf-8")
    assert "おはよう" in body
    assert body.count("-->") == 1
    assert seen["path_or_hf_repo"] == "m"
    assert seen["language"] == "ja"


def test_transcribe_raises_when_the_model_finds_nothing(tmp_path, monkeypatch):
    """一条 cue 都没有意味着这一集彻底没对白轨，让它静默写个空文件会让后面所有阶段
    在莫名其妙的地方炸。在这里就报错。"""
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        asr.ffmpeg, "extract_audio_track", lambda src, wav, **k: Path(wav).write_bytes(b"x")
    )
    monkeypatch.setattr(asr, "_run_model", lambda **k: {"segments": []})

    with pytest.raises(asr.ASRError):
        asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())


def test_transcribe_cleans_up_the_temporary_wav(tmp_path, monkeypatch):
    from tenmin.config import AsrConfig

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    wavs: list[Path] = []

    def fake_extract(src, wav, **kwargs):
        wavs.append(Path(wav))
        Path(wav).write_bytes(b"x")

    monkeypatch.setattr(asr.ffmpeg, "extract_audio_track", fake_extract)
    monkeypatch.setattr(
        asr, "_run_model", lambda **k: {"segments": [{"start": 0.0, "end": 1.0, "text": "あ"}]}
    )

    asr.transcribe(video, tmp_path / "out.srt", asr=AsrConfig())

    assert wavs
    assert not wavs[0].exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_asr.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.ingest.asr'`

- [ ] **Step 3: 实现 `src/tenmin/ingest/asr.py`**

```python
"""语音转写：把一段视频的人声轨变成一份 SRT。

只在「这一集既没有手传 SRT、视频里也没有软字幕轨」时才走到这里。产物落成一份普通
SRT 而不是直接给出 cue 对象，有两个刻意的好处：转写一集要几分钟，落盘就等于缓存，
重跑 ingest 不用重付；而且 SRT 是人能直接改的格式，转差了可以手动修，改完下次就走
「手传 SRT」那条路。

跟 srt_parser 并排：它们是同一层的两个 cue 来源，上面由 resolve 决定走哪个。
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from ..atomic import atomic_path
from ..config import AsrConfig
from ..models import RawCue
from ..render import ffmpeg

# 「这段文本念得出来吗」。含任意假名、汉字、拉丁字母或数字就算有内容。
# 用途是丢掉 whisper 在音频边界吐出的空段落与纯标点段落（实测一集末尾能吐出好几条）。
# 判据刻意宽松：宁可留一条噪声让下游清洗层处理，也不要误杀「OK」「2026」这种短台词。
_PRONOUNCEABLE = re.compile(r"[0-9A-Za-z\u3040-\u30ff\u4e00-\u9fff]")


class ASRError(RuntimeError):
    """转写本身失败（模型报错、或者一条对白都没转出来）。"""


class ASRUnavailableError(ASRError):
    """需要转写，但转写依赖没装。

    单独一个类型是因为处置方式不同：这不是「数据坏了」，是「环境缺东西」，消息里要
    直接给出补齐的命令。继承 ASRError 让上层想一把网住时也能网住。
    """


def _format_timestamp(seconds: float) -> str:
    """SRT 的时间戳。

    毫秒分隔符是逗号。timecode.format_timestamp 产出的是点号版本（给人读的日志和
    文档用），它的 docstring 明令不要拿它写 SRT，所以这里自己拼一份。
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_cues(segments: Iterable[Mapping[str, object]]) -> list[RawCue]:
    """模型输出的段落列表 → RawCue 列表，顺带丢掉不能用的那些。

    三条过滤规则，都是实测踩到的：
    - end <= start：whisper 在音频末尾会吐一串 start == end 的零长度空段落，是解码
      边界的 artifact，不是内容。
    - 文本剥掉空白后为空。
    - 文本里没有任何可发音字符（纯标点、纯音符符号）。

    刻意在这一层就丢，而不是让它们流到下游的清洗层：清洗层丢掉的东西会计进
    「跳过了几个块」，而那个数字是给人判断**片源质量**的，混进模型的 artifact 就废了。

    idx 在过滤之后重新连续编号，跟 srt_parser 的语义一致（第几个**可用**的 cue）。
    """
    cues: list[RawCue] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text or not _PRONOUNCEABLE.search(text):
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        if end <= start:
            continue
        cues.append(RawCue(idx=len(cues) + 1, start=start, end=end, text=text))
    return cues


def render_srt(cues: Sequence[RawCue]) -> str:
    """RawCue 列表 → SRT 文本。

    产出的东西必须能被本项目自己的 srt_parser 吃回去（有测试守着这条往返）。
    """
    blocks = [
        f"{index}\n{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n{cue.text}\n"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def _run_model(**kwargs: object) -> Mapping[str, object]:
    """真正调用 mlx-whisper 的那一下。

    import 刻意写在函数体内、而不是模块顶层：转写依赖在 pyproject 的 asr extra 里，
    没装它的人（绝大多数用法）必须能正常跑 `tenmin --help` 和所有非转写阶段。放到
    顶层会让整个 ingest 包 import 不动。

    单独提成一个函数是为了让测试能换掉它而不必装几个 G 的模型。
    """
    try:
        import mlx_whisper
    except ImportError as exc:
        raise ASRUnavailableError(
            "这一集没有字幕（既没传 --srt，视频里也没有软字幕轨），需要语音转写，"
            "但转写依赖没装。跑一次 `uv sync --extra asr` 再试。"
        ) from exc
    return mlx_whisper.transcribe(**kwargs)


def transcribe(
    video: Path,
    dest: Path,
    *,
    asr: AsrConfig,
    ffmpeg_path: str = ffmpeg.FFMPEG,
) -> None:
    """转写 video 的人声轨，把结果写成 dest 这份 SRT。

    同步阻塞调用，不走 to_thread：ingest 是全局阶段，跑它的时候没有别的任务在飞，
    包一层 async 只会多一层看不出好处的管道。

    中间那份 wav 走系统临时目录、用完就删：它只是喂模型的入参，留在 work/ 下只会
    让人以为它是个产物（一集的 16k 单声道 wav 约 45 MB）。
    """
    print(f"  正在转写音轨（约需数分钟）: {video.name}")
    with tempfile.TemporaryDirectory(prefix="tenmin-asr-") as workdir:
        wav = Path(workdir) / "audio.wav"
        ffmpeg.extract_audio_track(video, wav, ffmpeg=ffmpeg_path)
        result = _run_model(
            audio=str(wav),
            path_or_hf_repo=asr.model,
            language=asr.language,
            word_timestamps=True,
        )

    raw_segments = result.get("segments") or []
    if not isinstance(raw_segments, Iterable):
        raise ASRError(f"转写结果里的 segments 不是个列表: {type(raw_segments)!r}")
    cues = segments_to_cues(raw_segments)
    if not cues:
        raise ASRError(
            f"转写 {video.name} 没得到任何对白。确认这个文件有人声轨，"
            f"或者手动传一份 --srt。"
        )

    with atomic_path(dest) as staged:
        staged.write_text(render_srt(cues), encoding="utf-8")
    print(f"  转写完成，{len(cues)} 条对白 → {dest.name}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_asr.py -v`
Expected: PASS（15 passed）

- [ ] **Step 5: 把新异常接进 CLI 的错误网**

`src/tenmin/cli.py`：在 import 区加 `from .ingest.asr import ASRError`，然后把 `ASRError` 加进 `PIPELINE_ERRORS` 元组（放在 `FFmpegError` 之后），并在那个元组上方的注释里补一句说明它管什么。

注意只加 `ASRError` 一个 —— `ASRUnavailableError` 继承它，自动被网住。

- [ ] **Step 6: 跑 CLI 测试与全量**

Run: `uv run pytest tests/test_cli.py tests/test_asr.py -q && uv run pytest tests/ -q`
Expected: 全绿

- [ ] **Step 7: 提交**

```bash
rtk git add src/tenmin/ingest/asr.py src/tenmin/cli.py tests/test_asr.py
rtk git commit -m "feat: 加语音转写层（mlx-whisper，含尾部幻觉过滤）"
```

---

### Task 4: `ingest/resolve.py` —— 对白轨从哪来的三岔

**Files:**
- Create: `src/tenmin/ingest/resolve.py`
- Test: `tests/test_resolve.py`

**Interfaces:**
- Consumes: `tenmin.ingest.asr.transcribe`、`tenmin.render.ffmpeg.has_subtitle_stream`、`tenmin.render.ffmpeg.extract_subtitle_track`、`tenmin.config.AsrConfig`
- Produces: `SubtitleSource`（NamedTuple，字段 `path: Path`、`kind: Literal["srt","asr"]`）、`resolve_subtitle_source(srt: Path | None, video: Path | None, *, cache: Path, asr: AsrConfig, ffmpeg_path: str, ffprobe_path: str) -> SubtitleSource`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_resolve.py`：

```python
"""对白轨来源的三岔判据。ffprobe/ffmpeg/转写全部 fake —— 这一层的职责只是「选哪条路」。"""

from pathlib import Path

import pytest

from tenmin.config import AsrConfig
from tenmin.ingest import resolve


@pytest.fixture
def stub(monkeypatch):
    """把三条路的出口都换成记账用的假货。"""
    calls: dict[str, list] = {"probe": [], "extract": [], "transcribe": []}
    state = {"has_subtitle": False}

    def fake_has_subtitle(path, **kwargs):
        calls["probe"].append(Path(path))
        return state["has_subtitle"]

    def fake_extract(video, dest, **kwargs):
        calls["extract"].append((Path(video), Path(dest)))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text("1\n00:00:01,000 --> 00:00:02,000\n抽出来的\n", encoding="utf-8")

    def fake_transcribe(video, dest, **kwargs):
        calls["transcribe"].append((Path(video), Path(dest)))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text("1\n00:00:01,000 --> 00:00:02,000\n転写した\n", encoding="utf-8")

    monkeypatch.setattr(resolve.ffmpeg, "has_subtitle_stream", fake_has_subtitle)
    monkeypatch.setattr(resolve.ffmpeg, "extract_subtitle_track", fake_extract)
    monkeypatch.setattr(resolve.asr, "transcribe", fake_transcribe)
    return calls, state


def _video(tmp_path: Path) -> Path:
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    return video


def _cache(tmp_path: Path) -> Path:
    return tmp_path / "srt" / "E11.asr.srt"


def test_a_handed_srt_wins_and_nothing_is_probed(tmp_path, stub):
    calls, _ = stub
    srt = tmp_path / "hand.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n手传\n", encoding="utf-8")

    source = resolve.resolve_subtitle_source(
        srt, _video(tmp_path), cache=_cache(tmp_path), asr=AsrConfig()
    )

    assert source.path == srt
    assert source.kind == "srt"
    assert calls["probe"] == []
    assert calls["transcribe"] == []


def test_a_handed_srt_works_without_any_video(tmp_path, stub):
    """视频对 render 阶段是硬需求，但对「对白轨从哪来」不是。"""
    srt = tmp_path / "hand.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n手传\n", encoding="utf-8")

    source = resolve.resolve_subtitle_source(
        srt, None, cache=_cache(tmp_path), asr=AsrConfig()
    )
    assert source == resolve.SubtitleSource(srt, "srt")


def test_an_embedded_subtitle_track_is_extracted_instead_of_transcribed(tmp_path, stub):
    """软字幕轨零成本零误差，比转写好得多 —— 这条分枝漏掉就是白付三分钟还掉质量。"""
    calls, state = stub
    state["has_subtitle"] = True
    cache = _cache(tmp_path)

    source = resolve.resolve_subtitle_source(
        None, _video(tmp_path), cache=cache, asr=AsrConfig()
    )

    assert source.kind == "srt"
    assert source.path.is_file()
    assert "抽出来的" in source.path.read_text(encoding="utf-8")
    assert calls["extract"]
    assert calls["transcribe"] == []


def test_transcription_is_the_last_resort(tmp_path, stub):
    calls, state = stub
    state["has_subtitle"] = False
    cache = _cache(tmp_path)

    source = resolve.resolve_subtitle_source(
        None, _video(tmp_path), cache=cache, asr=AsrConfig()
    )

    assert source == resolve.SubtitleSource(cache, "asr")
    assert calls["transcribe"] == [(tmp_path / "e11.mp4", cache)]


def test_a_fresh_cache_is_reused_instead_of_retranscribing(tmp_path, stub):
    """一集转写要几分钟，--force 重跑 ingest 不该重付。"""
    calls, state = stub
    video = _video(tmp_path)
    cache = _cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("1\n00:00:01,000 --> 00:00:02,000\n既存\n", encoding="utf-8")
    import os

    os.utime(cache, (video.stat().st_mtime + 10, video.stat().st_mtime + 10))

    source = resolve.resolve_subtitle_source(None, video, cache=cache, asr=AsrConfig())

    assert source.kind == "asr"
    assert "既存" in source.path.read_text(encoding="utf-8")
    assert calls["transcribe"] == []


def test_a_stale_cache_is_retranscribed(tmp_path, stub):
    """换了片源（重新压制、换了个版本）就得重转。"""
    calls, state = stub
    cache = _cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("旧的\n", encoding="utf-8")
    video = _video(tmp_path)
    import os

    os.utime(cache, (video.stat().st_mtime - 10, video.stat().st_mtime - 10))

    source = resolve.resolve_subtitle_source(None, video, cache=cache, asr=AsrConfig())

    assert "転写した" in source.path.read_text(encoding="utf-8")
    assert calls["transcribe"]


def test_an_empty_cache_is_retranscribed(tmp_path, stub):
    """0 字节的缓存是上一次被打断留下的，不是产物。"""
    calls, _ = stub
    video = _video(tmp_path)
    cache = _cache(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("", encoding="utf-8")
    import os

    os.utime(cache, (video.stat().st_mtime + 10, video.stat().st_mtime + 10))

    resolve.resolve_subtitle_source(None, video, cache=cache, asr=AsrConfig())
    assert calls["transcribe"]


def test_neither_srt_nor_video_is_an_error(tmp_path, stub):
    with pytest.raises(ValueError) as excinfo:
        resolve.resolve_subtitle_source(None, None, cache=_cache(tmp_path), asr=AsrConfig())
    assert "srt" in str(excinfo.value).lower() or "视频" in str(excinfo.value)


def test_a_missing_video_file_is_an_error(tmp_path, stub):
    with pytest.raises(FileNotFoundError):
        resolve.resolve_subtitle_source(
            None, tmp_path / "gone.mp4", cache=_cache(tmp_path), asr=AsrConfig()
        )


def test_a_missing_handed_srt_is_an_error(tmp_path, stub):
    with pytest.raises(FileNotFoundError):
        resolve.resolve_subtitle_source(
            tmp_path / "gone.srt", _video(tmp_path), cache=_cache(tmp_path), asr=AsrConfig()
        )


def test_the_subtitle_extraction_cache_is_not_the_asr_cache(tmp_path, stub):
    """从软字幕轨抽出来的东西不该占 ASR 缓存那个名字 —— 两者的失效条件一样，但
    混用会让人分不清「这份 SRT 是抽的还是转的」。"""
    calls, state = stub
    state["has_subtitle"] = True
    cache = _cache(tmp_path)

    source = resolve.resolve_subtitle_source(
        None, _video(tmp_path), cache=cache, asr=AsrConfig()
    )
    assert source.path != cache
    assert source.path.name.endswith(".embedded.srt")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_resolve.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.ingest.resolve'`

- [ ] **Step 3: 实现 `src/tenmin/ingest/resolve.py`**

```python
"""对白轨从哪来。

三条路，按「无损且便宜」排序：

1. 手传的 SRT —— 人明确指定了，不猜。
2. 视频里的软字幕轨 —— ffmpeg 一条命令抽出来，零成本零误差。**这条分枝的存在是关键**：
   漏掉它会把一个自带字幕轨的片源白白拉去跑几分钟转写，还把质量换低了。
3. 语音转写 —— 只有前两条都不成立时才走，而且要先告诉人一声（它是这三条里唯一一条
   既费时间又有损的）。

三条路都归一成「一个 SRT 文件的路径」，所以下游 build_track 拿到的东西形态完全不变。
顺带的好处：转写结果落成 SRT 就等于缓存，也能被人手动修正。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

from ..config import AsrConfig
from ..render import ffmpeg
from . import asr


class SubtitleSource(NamedTuple):
    """选中的对白轨来源。

    kind 不是「文件是什么格式」（三条路给出的都是 SRT），而是「这份对白是原生字幕
    还是机器听写的」。下游靠它决定要不要繁转简（日语过 OpenCC 会被改字）、要不要
    跑翻译阶段。
    """

    path: Path
    kind: Literal["srt", "asr"]


def _is_usable_cache(cache: Path, video: Path) -> bool:
    """这份缓存还能用吗。

    0 字节判为不可用：那是上一次被打断留下的残骸，不是产物（跟 pipeline 判断阶段
    新鲜度时的口径一致）。
    """
    if not cache.is_file() or cache.stat().st_size == 0:
        return False
    return cache.stat().st_mtime >= video.stat().st_mtime


def resolve_subtitle_source(
    srt: Path | None,
    video: Path | None,
    *,
    cache: Path,
    asr_config: AsrConfig,
    ffmpeg_path: str = ffmpeg.FFMPEG,
    ffprobe_path: str = ffmpeg.FFPROBE,
) -> SubtitleSource:
    """挑一条路，返回一份可解析的 SRT 及其来源类型。

    cache 是转写结果的落点（调用方给出，通常是 srt/E{NN}.asr.srt）。从软字幕轨抽出来
    的那份走同目录的另一个名字：两者的失效条件一样，但混用同一个文件名会让人分不清
    手上这份 SRT 是抽的还是转的。
    """
    if srt is not None:
        if not srt.is_file():
            raise FileNotFoundError(f"找不到字幕文件: {srt}")
        return SubtitleSource(srt, "srt")

    if video is None:
        raise ValueError("既没有字幕文件也没有源视频，无法得到对白轨")
    if not video.is_file():
        raise FileNotFoundError(f"找不到源视频: {video}")

    if ffmpeg.has_subtitle_stream(video, ffprobe=ffprobe_path):
        embedded = cache.with_name(cache.name.replace(".asr.srt", ".embedded.srt"))
        if not _is_usable_cache(embedded, video):
            ffmpeg.extract_subtitle_track(video, embedded, ffmpeg=ffmpeg_path)
        return SubtitleSource(embedded, "srt")

    if _is_usable_cache(cache, video):
        return SubtitleSource(cache, "asr")

    # 这是三条路里唯一一条既费时间又有损的，所以让它被看见。刻意不做成一个要用户每次
    # 记得传的 flag：软字幕那条是无损的，不值得为它多打字；值得被看见的只有这一条。
    print(f"  {video.name} 没有字幕轨，将对音轨做语音转写（约需数分钟）")
    asr.transcribe(video, cache, asr=asr_config, ffmpeg_path=ffmpeg_path)
    return SubtitleSource(cache, "asr")
```

注意：测试里调用时写的关键字是 `asr=AsrConfig()`，而实现里的参数名是 `asr_config`（模块里 `asr` 这个名字已经被导入的模块占了）。**把测试里全部 `asr=AsrConfig()` 改成 `asr_config=AsrConfig()`**，`test_asr.py` 里 `asr.transcribe(..., asr=AsrConfig())` 保持不变（那是另一个模块的参数名，那里没有命名冲突）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_resolve.py -v`
Expected: PASS（12 passed）

- [ ] **Step 5: 提交**

```bash
rtk git add src/tenmin/ingest/resolve.py tests/test_resolve.py
rtk git commit -m "feat: 加对白轨来源解析（手传 SRT / 软字幕轨 / 语音转写三岔）"
```

---

### Task 5: config 与 CLI 容纳「只有视频」

**Files:**
- Modify: `src/tenmin/config.py`（`EpisodeConfig.srt` 变可选 + 新 model_validator + `srt_path` 返回类型）
- Modify: `src/tenmin/cli.py`（`--srt`/`--video` 的校验规则）
- Modify: `src/tenmin/pipeline.py`（`register_episode` 的 srt 变可选）
- Test: `tests/test_config.py`、`tests/test_cli.py`、`tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1 的 `AsrConfig`
- Produces: `EpisodeConfig.srt: Path | None`；`ProjectConfig.srt_path(episode) -> Path | None`；`register_episode(cfg, *, episode: int, srt: Path | None, video: Path) -> ProjectConfig`

- [ ] **Step 1: 写失败的测试**

`tests/test_config.py` 末尾追加：

```python
def test_episode_may_have_only_a_video():
    from tenmin.config import EpisodeConfig

    episode = EpisodeConfig(number=11, video=Path("/tmp/e11.mp4"))
    assert episode.srt is None
    assert episode.video == Path("/tmp/e11.mp4")


def test_episode_may_have_only_an_srt():
    from tenmin.config import EpisodeConfig

    episode = EpisodeConfig(number=11, srt=Path("srt/E11.srt"))
    assert episode.video is None


def test_episode_with_neither_srt_nor_video_is_rejected():
    """两个字段都可选之后，「至少得有一个」这条保证就没人管了 —— 显式补上。
    没有它的话，一个空 episode 会一路飘到 ingest 才炸，且错误信息指不到根因。"""
    from pydantic import ValidationError

    from tenmin.config import EpisodeConfig

    with pytest.raises(ValidationError) as excinfo:
        EpisodeConfig(number=11)
    assert "srt" in str(excinfo.value)


def test_srt_path_is_none_for_a_video_only_episode(tmp_path):
    from tenmin.config import EpisodeConfig, ProjectConfig

    cfg = ProjectConfig(
        show="测试番",
        slug="test",
        episodes=[EpisodeConfig(number=11, video=tmp_path / "e11.mp4")],
    ).bind_root(tmp_path)
    assert cfg.srt_path(cfg.episodes[0]) is None
```

`tests/test_cli.py` 末尾追加（照该文件既有的 CliRunner 用法）：

```python
def test_run_accepts_a_video_without_an_srt(tmp_path, monkeypatch):
    """生肉入口：只有视频也能登记一集。"""
    project = _make_project(tmp_path)  # 该文件里已有的建 project 辅助函数
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    seen: dict[str, object] = {}

    def fake_register(cfg, *, episode, srt, video):
        seen.update({"episode": episode, "srt": srt, "video": video})
        return cfg

    monkeypatch.setattr(cli, "register_episode", fake_register)
    monkeypatch.setattr(cli, "run_pipeline", _noop_pipeline)

    result = runner.invoke(
        cli.app, ["run", str(project), "--episode", "11", "--video", str(video)]
    )

    assert result.exit_code == 0, result.output
    assert seen["srt"] is None
    assert seen["video"] == video


def test_run_rejects_an_srt_without_a_video(tmp_path):
    """反过来不行：视频是 render 阶段的硬需求。"""
    project = _make_project(tmp_path)
    srt = tmp_path / "e11.srt"
    srt.write_text("1\n", encoding="utf-8")

    result = runner.invoke(
        cli.app, ["run", str(project), "--episode", "11", "--srt", str(srt)]
    )

    assert result.exit_code == 1
    assert "video" in result.output


def test_run_still_requires_episode_when_registering(tmp_path):
    project = _make_project(tmp_path)
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")

    result = runner.invoke(cli.app, ["run", str(project), "--video", str(video)])

    assert result.exit_code == 1
    assert "--episode" in result.output
```

（`_make_project` / `_noop_pipeline` 用该文件已有的同类辅助；若名字不同，照现有命名改。）

`tests/test_pipeline.py` 末尾追加：

```python
def test_register_episode_without_an_srt_leaves_the_field_empty(tmp_path):
    cfg = _write_project(tmp_path)  # 该文件里已有的辅助
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")

    updated = register_episode(cfg, episode=11, srt=None, video=video)

    episode = next(e for e in updated.episodes if e.number == 11)
    assert episode.srt is None
    assert episode.video == video.resolve()
    assert not (tmp_path / "srt" / "E11.srt").exists()


def test_register_episode_without_an_srt_omits_it_from_the_yaml(tmp_path):
    import yaml

    cfg = _write_project(tmp_path)
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")

    register_episode(cfg, episode=11, srt=None, video=video)

    data = yaml.safe_load((tmp_path / "project.yaml").read_text(encoding="utf-8"))
    entry = next(e for e in data["episodes"] if e["number"] == 11)
    assert "srt" not in entry
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py tests/test_cli.py tests/test_pipeline.py -k "only_a_video or only_an_srt or neither or video_only or without_an_srt or without_a_video or requires_episode" -v`
Expected: FAIL

- [ ] **Step 3: 改 `EpisodeConfig`**

`src/tenmin/config.py` 里 `EpisodeConfig`：

```python
    # 两个来源至少得有一个，由 _require_a_source 守着。
    # srt 可空是为了「只有生肉视频」那条路（对白轨靠软字幕轨抽取或语音转写拿到）；
    # video 可空是历史约定（v1 只吃字幕、压根不碰视频文件）。
    srt: Path | None = None
    video: Path | None = None
```

并在 `_check_ranges` 之后追加：

```python
    @model_validator(mode="after")
    def _require_a_source(self) -> "EpisodeConfig":
        """至少要有 srt 或 video 之一。

        两个字段都可选之后这条保证就没别人管了。没有它的话，一个只写了 number 的
        episode 会一路飘到 ingest 才炸，而那时的错误信息指向的是「找不到文件」，
        指不到「你这一集压根没写来源」这个根因。
        """
        if self.srt is None and self.video is None:
            raise ValueError(
                f"第 {self.number} 集既没有 srt 也没有 video，至少要填一个"
            )
        return self
```

（`model_validator` 要按文件既有 import 风格加进 `from pydantic import ...`。）

`srt_path` 改成：

```python
    def srt_path(self, episode: EpisodeConfig) -> Path | None:
        """这一集的字幕路径，没配就返回 None。

        相对路径按 project.yaml 所在目录解析，绝对路径原样返回。返回 None 意味着
        对白轨要从视频里拿（软字幕轨抽取或语音转写），由 ingest 那边的 resolve 决定。
        """
        if episode.srt is None:
            return None
        if episode.srt.is_absolute():
            return episode.srt
        return self._root / episode.srt
```

- [ ] **Step 4: 改 CLI 的校验**

`src/tenmin/cli.py` 的 `run` 命令里，把原来那两条校验替换成：

```python
    # 三条规则：
    # - 传 --srt 必须配 --video（视频是 render 阶段的硬需求，只有字幕出不了片）
    # - 只传 --video 合法，这是生肉入口（对白轨靠软字幕轨或语音转写拿）
    # - 传了任一个就必须说这是第几集
    if srt is not None and video is None:
        typer.secho("传 --srt 时必须同时传 --video", fg=typer.colors.RED)
        raise typer.Exit(1)
    if video is not None and episode is None:
        typer.secho("传 --srt/--video 时必须同时传 --episode", fg=typer.colors.RED)
        raise typer.Exit(1)
    if video is not None:
        cfg = register_episode(cfg, episode=episode, srt=srt, video=video)
```

- [ ] **Step 5: 改 `register_episode`**

`src/tenmin/pipeline.py`：签名改成 `srt: Path | None`，并把拷字幕那段包起来：

```python
    relative_srt: Path | None = None
    if srt is not None:
        srt_dest = cfg.root / "srt" / f"E{episode:02d}.srt"
        atomic.copy_file(srt, srt_dest)
        relative_srt = srt_dest.relative_to(cfg.root)
```

下面构造/更新 `EpisodeConfig` 的地方用 `relative_srt`。`model_dump(exclude_none=True, mode="json")` 已经会把 None 的 srt 从 yaml 里省掉，序列化那半不用改。

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py tests/test_cli.py tests/test_pipeline.py -q`
Expected: 全绿。若既有测试因 `srt` 变可选而挂（例如断言过 `EpisodeConfig(number=1)` 会报 srt 缺失的那类），按新语义更新它们的断言。

- [ ] **Step 7: 跑全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿

- [ ] **Step 8: 提交**

```bash
rtk git add src/tenmin/config.py src/tenmin/cli.py src/tenmin/pipeline.py tests/test_config.py tests/test_cli.py tests/test_pipeline.py
rtk git commit -m "feat: 允许只用视频登记一集（srt 变可选）"
```

---

### Task 6: ingest 接线 —— `build_track` 认 source、`run_ingest` 走 resolve、新鲜度看 srt 或 video

**Files:**
- Modify: `src/tenmin/ingest/normalize.py`（`build_track` 加 `source` kwarg）
- Modify: `src/tenmin/pipeline.py`（`run_ingest` 用 resolve；ingest 新鲜度输入改成 srt-or-video）
- Test: `tests/test_normalize.py`、`tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 4 的 `resolve_subtitle_source` / `SubtitleSource`；Task 5 的 `srt_path` 返回 `Path | None`
- Produces: `build_track(..., source: Literal["srt","asr"] = "srt")`；`run_ingest` 的行为扩张（无签名变化）

- [ ] **Step 1: 写失败的测试**

`tests/test_normalize.py` 末尾追加：

```python
def test_build_track_records_the_source(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nはい\n", encoding="utf-8")

    track = build_track(srt, episode=1, source="asr")
    assert track.source == "asr"


def test_build_track_defaults_to_srt_source(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n你好\n", encoding="utf-8")

    assert build_track(srt, episode=1).source == "srt"


def test_an_asr_source_never_goes_through_opencc(tmp_path):
    """繁转简作用在日语上会改字（製作 → 制作 这类）。ASR 路径必须绕开它，
    而且不能依赖调用方记得传 convert_traditional=False。"""
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n製作の話\n", encoding="utf-8")

    track = build_track(srt, episode=1, source="asr", convert_traditional=True)
    assert "製作" in "".join(line.text for line in track.lines)
```

`tests/test_pipeline.py` 末尾追加：

```python
def test_ingest_stays_fresh_against_a_video_only_episode(tmp_path):
    """只有 video 的集也得能算出「该不该重跑」。原来的输入集合只取 srt，
    对这种集会得到一个空输入列表 —— 而空输入在新鲜度判据里等于「跳过」，
    于是换了片源也不会重跑。"""
    from tenmin.pipeline import _ingest_inputs

    cfg = _write_project(tmp_path)
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=11, srt=None, video=video)

    inputs = _ingest_inputs(cfg)
    assert video.resolve() in inputs


def test_ingest_inputs_include_both_when_both_exist(tmp_path):
    from tenmin.pipeline import _ingest_inputs

    cfg = _write_project(tmp_path)
    srt = tmp_path / "hand.srt"
    srt.write_text("1\n", encoding="utf-8")
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=11, srt=srt, video=video)

    inputs = _ingest_inputs(cfg)
    assert video.resolve() in inputs
    assert (tmp_path / "srt" / "E11.srt") in inputs


def test_ingest_inputs_exclude_the_asr_cache(tmp_path):
    """转写缓存是 ingest 自己的产物。把它算进 ingest 的输入会让
    「转写完写出缓存」这个动作立刻使 ingest 变得不新鲜 —— 每次都重跑。"""
    from tenmin.pipeline import _ingest_inputs

    cfg = _write_project(tmp_path)
    video = tmp_path / "e11.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=11, srt=None, video=video)

    inputs = _ingest_inputs(cfg)
    assert all(".asr.srt" not in str(p) for p in inputs)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_normalize.py tests/test_pipeline.py -k "source or opencc or ingest_inputs or video_only_episode" -v`
Expected: FAIL

- [ ] **Step 3: `build_track` 加 `source`**

`src/tenmin/ingest/normalize.py`：签名末尾加一个 kwarg

```python
    source: Literal["srt", "asr"] = "srt",
```

docstring 里补一段：

```
    source 说的是这份对白从哪来（原生字幕 / 机器听写），不是文件格式 —— 三条来源路径
    给出的都是 SRT。它会被写进产物，下游靠它判断这是什么语言的对白：翻译阶段只对
    听写来的日语对白动手，而繁转简（OpenCC）对日语是有害的。
```

在函数体里、算 `convert` 的地方：

```python
    # 听写来的对白是源片的原生语言（日语），过一遍繁转简会被改字（製作 → 制作 这类）。
    # 这里强制关掉而不是要求调用方记得传：source 是「数据是什么」的事实，
    # 而 convert_traditional 是「想怎么处理繁体中文」的创作旋钮，前者该压住后者。
    convert = convert_traditional and source == "srt"
```

然后把原来传给 `clean_text` 的 `convert=convert_traditional` 改成 `convert=convert`，并在构造 `DialogueTrack` 时带上 `source=source`。

（`Literal` 若尚未 import，按文件既有风格从 `typing` 引入。）

- [ ] **Step 4: `run_ingest` 改走 resolve**

`src/tenmin/pipeline.py`：在 `run_ingest` 上方新增一个辅助，并改 `run_ingest` 的循环体。

```python
def _ingest_inputs(cfg: ProjectConfig) -> list[Path]:
    """ingest 阶段的新鲜度输入。

    每集取「手传字幕」与「源视频」里存在的那些。原来只取字幕，对一个只有视频的集会
    得到空列表 —— 而空输入在新鲜度判据里等于「跳过」，于是换了片源也不会重跑。

    刻意**不**把语音转写的缓存（srt/E{NN}.asr.srt）算进来：它是 ingest 自己的产物，
    算进输入会让「转写完写出缓存」这个动作立刻使 ingest 变得不新鲜，每次都重跑。
    它自己的失效判据在 ingest.resolve 里，对着源视频的 mtime 单独判。
    """
    inputs: list[Path] = []
    for episode in cfg.episodes:
        srt = cfg.srt_path(episode)
        if srt is not None:
            inputs.append(srt)
        if episode.video is not None:
            inputs.append(cfg.video_path(episode))
    return inputs
```

`run_ingest` 的循环体改成：

```python
        source = resolve_subtitle_source(
            cfg.srt_path(episode),
            cfg.video_path(episode) if episode.video is not None else None,
            cache=cfg.root / "srt" / f"{episode_stem(episode.number)}.asr.srt",
            asr_config=cfg.asr,
            ffmpeg_path=cfg.render.ffmpeg_path,
            ffprobe_path=cfg.render.ffprobe_path,
        )
        track = build_track(
            source.path,
            episode=episode.number,
            source=source.kind,
            glossary=cfg.glossary,
            convert_traditional=cfg.locale.convert_traditional,
            show_title=cfg.show,
            op_range=episode.op_range,
            ed_range=episode.ed_range,
            duration=_source_duration(cfg, episode),
            ingest=cfg.ingest,
            credits=cfg.credits,
        )
```

（import `from .ingest.resolve import resolve_subtitle_source`。）

- [ ] **Step 5: ingest 的新鲜度换成新输入**

`run_pipeline` 里把 `srt_inputs = [cfg.srt_path(ep) for ep in cfg.episodes]` 改成：

```python
    ingest_inputs = _ingest_inputs(cfg)
```

并把 ingest 块里的 `is_fresh(outputs, srt_inputs)` 改成 `is_fresh(outputs, ingest_inputs)`。全文搜一遍 `srt_inputs` 确认没有别的用处。

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_normalize.py tests/test_pipeline.py -q`
Expected: 全绿

- [ ] **Step 7: 跑全量 + 确认存量产物没被改变**

Run: `uv run pytest tests/ -q`
Expected: 全绿

Run: `uv run tenmin run saijo --only ingest --force && rtk git status`
Expected: `work/saijo/01_dialogue/` 下 13 份 JSON 逐字节不变（git 看不到改动）。这一步是**回归闸门** —— 现有片源全部走「手传 SRT」那条路，`source` 默认 `"srt"`，行为必须完全等价。若 git 显示有改动，停下来查（很可能是 `source` 字段被写进了 JSON —— 那是预期的新增字段，确认 diff 里**只有**这一个键、值全是 `"srt"` 即可接受，并在提交信息里说明）。

- [ ] **Step 8: 提交**

```bash
rtk git add src/tenmin/ingest/normalize.py src/tenmin/pipeline.py tests/test_normalize.py tests/test_pipeline.py work/
rtk git commit -m "feat: ingest 走来源解析，新鲜度改看字幕或视频"
```

---

### Task 7: 翻译的数据模型与产物路径

**Files:**
- Modify: `src/tenmin/models.py`（`TranslatedLine`、`TranslatedTrack`）
- Modify: `src/tenmin/pipeline.py`（`_ARTIFACTS` 加两条、`Paths` 加两个方法与一个 property）
- Test: `tests/test_models.py`、`tests/test_pipeline.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `models.TranslatedLine`（字段 `id: int`、`zh: str`）
  - `models.TranslatedTrack`（字段 `episode: int`、`lines: list[TranslatedLine]`、`glossary: dict[str, str]`）
  - `Paths.zh_lines(episode) -> Path`（`zh/E{NN}.zh.json`）
  - `Paths.zh_subtitles(episode) -> Path`（`out/E{NN}.zh.srt`）
  - `Paths.glossary -> Path`（property，`zh/glossary.json`）

- [ ] **Step 1: 写失败的测试**

`tests/test_models.py` 末尾追加：

```python
def test_translated_track_round_trips():
    from tenmin.models import TranslatedLine, TranslatedTrack

    track = TranslatedTrack(
        episode=11,
        lines=[TranslatedLine(id=1, zh="你好"), TranslatedLine(id=2, zh="再见")],
        glossary={"リディア": "莉迪亚"},
    )
    restored = TranslatedTrack.model_validate_json(track.model_dump_json())
    assert restored == track


def test_translated_track_glossary_defaults_to_empty():
    from tenmin.models import TranslatedTrack

    assert TranslatedTrack(episode=1, lines=[]).glossary == {}
```

`tests/test_pipeline.py` 的 `FROZEN_LAYOUT` 里加两条：

```python
    "zh_lines": "zh/E02.zh.json",
    "zh_subtitles": "out/E02.zh.srt",
```

并追加：

```python
def test_glossary_path_is_project_level(tmp_path):
    """累积术语表不带集号：它的意义就是跨集共享。

    做成 property 而不是方法是刻意的 —— test_paths_exposes_exactly_the_frozen_artifacts
    断言 Paths 上可调用的公开名字集合正好等于按集产物那张表，property 不是 callable，
    自动落在那个集合之外。
    """
    paths = Paths(tmp_path)
    assert paths.glossary == tmp_path / "zh" / "glossary.json"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_models.py tests/test_pipeline.py -k "translated or glossary_path or frozen" -v`
Expected: FAIL

- [ ] **Step 3: 加数据模型**

`src/tenmin/models.py` 里 `DialogueTrack` 之后追加：

```python
class TranslatedLine(_StageModel):
    """一条对白的中文译文。

    id 是这条对白在 DialogueTrack.lines 里的位置（从 1 起），不是 DialogueLine.idx ——
    后者在双轨字幕被拆开时会重复（同一个 idx 配不同的 segment_index），拿它当键会撞。
    位置下标天然唯一，而且让「进去多少条就该出来多少条」这条校验变成一次集合相等判断。
    """

    id: int
    zh: str


class TranslatedTrack(_StageModel):
    """一集的中文译文轨，加这一集认出来的专有名词译法。

    glossary 跟译文放在同一个模型里、由同一次模型调用产出，是刻意的：分成两次调用就
    没法保证译文里用的就是它报上来的那个译名。
    """

    episode: int
    lines: list[TranslatedLine] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
```

（`_StageModel` 与 `Field` 按文件既有写法。若 `_StageModel` 是「不进 JSON 的纯内部模型」而 `TranslatedTrack` 需要落盘，改用该文件里落盘模型所用的基类 —— 以 `DialogueTrack` 的基类为准。）

- [ ] **Step 4: 加产物路径**

`src/tenmin/pipeline.py` 的 `_ARTIFACTS` 里，`narration` 之后加两条：

```python
    # 翻译阶段。目录刻意不占 0N 编号：现有的 01_dialogue → 07_render 是连续的，
    # 真按执行顺序插进去要把后面六个目录全部改名，而这些字符串是磁盘上的存量契约
    # （改一个字符，work/ 下已有的全部产物路径失配），同时全仓大量注释按名字引用它们。
    # docgen 把给人看的东西写进 out/ 已经立了「编号不等于执行顺序」这个先例。
    "zh_lines": ("zh", ".zh.json"),
    # 中文字幕是交付物，跟解说方案、配音文本并排放 out/：那个目录的约定是
    # 「给人看的东西都在这」，标准 SRT 可以直接拖进播放器。
    "zh_subtitles": ("out", ".zh.srt"),
```

`Paths` 里加两个方法（跟其它 14 个同形）与一个 property：

```python
    def zh_lines(self, episode: int) -> Path:
        return self._artifact("zh_lines", episode)

    def zh_subtitles(self, episode: int) -> Path:
        return self._artifact("zh_subtitles", episode)

    @property
    def glossary(self) -> Path:
        """跨集累积的专有名词表。

        项目级、不带集号 —— 它的全部意义就是让第 2 集知道第 1 集把人名译成了什么。
        刻意是 property 而不是方法：按集产物那张表有个测试断言 Paths 上可调用的公开
        名字集合正好等于表里的键，property 不是 callable，自动落在那个集合之外。
        """
        return self._root / "zh" / "glossary.json"
```

（`Paths.__init__` 存的根目录字段名以文件现状为准。）

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_models.py tests/test_pipeline.py -q`
Expected: 全绿

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/models.py src/tenmin/pipeline.py tests/test_models.py tests/test_pipeline.py
rtk git commit -m "feat: 加译文轨模型与翻译阶段产物路径"
```

---

### Task 8: schema 修复层提成公开、加输出后置校验钩子

**Files:**
- Modify: `src/tenmin/script/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: 无
- Produces: `complete_with_schema_repair[T: BaseModel](send, schema, *, max_attempts, label, diagnostics=None, check=None) -> T`（原 `_complete_with_schema_repair` 去下划线并加一个 `check` 参数）

- [ ] **Step 1: 写失败的测试**

`tests/test_llm.py` 末尾追加：

```python
@pytest.mark.asyncio
async def test_schema_repair_is_public():
    """翻译阶段要复用它。它本来就是 provider 无关的通用机制，此前只是恰好只有
    一个消费者。"""
    from tenmin.script import llm

    assert hasattr(llm, "complete_with_schema_repair")


@pytest.mark.asyncio
async def test_check_failure_triggers_a_repair_round():
    """schema 管不了跨条目的约束（比如「id 一条都不能漏」）。把那种校验交给 check，
    让它抛 LLMResponseFormatError 就能复用已有的回灌重试。"""
    from pydantic import BaseModel

    from tenmin.script.llm import (
        LLMResponseFormatError,
        complete_with_schema_repair,
    )

    class Box(BaseModel):
        n: int

    seen: list[object] = []
    payloads = ['{"n": 1}', '{"n": 2}']

    async def send(repair):
        seen.append(repair)
        return payloads[len(seen) - 1]

    def check(box: Box) -> None:
        if box.n != 2:
            raise LLMResponseFormatError("n 必须是 2，你给的是 1")

    result = await complete_with_schema_repair(
        send, Box, max_attempts=3, label="测试", check=check
    )

    assert result.n == 2
    assert seen[0] is None
    assert seen[1] is not None
    assert "n 必须是 2" in seen[1].error


@pytest.mark.asyncio
async def test_check_exhaustion_raises_schema_error_with_the_raw_output():
    from pydantic import BaseModel

    from tenmin.script.llm import (
        LLMResponseFormatError,
        LLMSchemaError,
        complete_with_schema_repair,
    )

    class Box(BaseModel):
        n: int

    async def send(repair):
        return '{"n": 1}'

    def check(box: Box) -> None:
        raise LLMResponseFormatError("永远不满意")

    with pytest.raises(LLMSchemaError) as excinfo:
        await complete_with_schema_repair(
            send, Box, max_attempts=2, label="测试", check=check
        )
    assert excinfo.value.raw_output == '{"n": 1}'


@pytest.mark.asyncio
async def test_no_check_means_the_old_behaviour():
    from pydantic import BaseModel

    from tenmin.script.llm import complete_with_schema_repair

    class Box(BaseModel):
        n: int

    calls = 0

    async def send(repair):
        nonlocal calls
        calls += 1
        return '{"n": 7}'

    result = await complete_with_schema_repair(send, Box, max_attempts=3, label="测试")
    assert result.n == 7
    assert calls == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_llm.py -k "schema_repair_is_public or check_failure or check_exhaustion or old_behaviour" -v`
Expected: FAIL，`AttributeError: module 'tenmin.script.llm' has no attribute 'complete_with_schema_repair'`

- [ ] **Step 3: 改名并加 `check`**

`src/tenmin/script/llm.py`：
1. `_complete_with_schema_repair` 改名为 `complete_with_schema_repair`（去掉前导下划线），两个内部调用点同步改。
2. 签名加一个参数：

```python
    check: Callable[[T], None] | None = None,
```

3. docstring 里补一段：

```
    check 是校验通过 schema 之后、返回之前的一道额外关卡。用途是 schema 表达不了的
    跨条目约束 —— 比如「输入 408 条就必须输出 408 条、id 一条不漏」这种。让它抛
    LLMResponseFormatError 就会走同一套回灌重试，而且回灌的消息由它自己写（能写成
    「你漏了第 137、298 行」这种模型照着就能改的具体话）。

    去掉前导下划线是因为它已经有了第二个消费者（翻译阶段）。它一直是 provider 无关的
    通用机制，此前只是恰好只有一个调用方。
```

4. 在核心循环里，把 `model_validate_json` 那一行拆成两步，`check` 放在同一个 `try` 内：

```python
            parsed = schema.model_validate_json(_extract_json(last_raw))
            if check is not None:
                check(parsed)
            return parsed
```

（`check` 必须在 `except (ValidationError, LLMResponseFormatError)` 覆盖的范围内 —— 它抛的就是 `LLMResponseFormatError`。）

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_llm.py -q`
Expected: 全绿（既有测试若引用了旧的私有名字，同步改）

- [ ] **Step 5: 跑全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/script/llm.py tests/test_llm.py
rtk git commit -m "refactor: schema 修复层提成公开并加输出后置校验钩子"
```

---

### Task 9: `translate/glossary.py` —— 累积术语表的读写与合并

**Files:**
- Create: `src/tenmin/translate/__init__.py`
- Create: `src/tenmin/translate/glossary.py`
- Test: `tests/test_translate_glossary.py`

**Interfaces:**
- Consumes: `tenmin.atomic`
- Produces:
  - `load_glossary(path: Path) -> dict[str, str]`
  - `save_glossary(path: Path, glossary: dict[str, str]) -> None`
  - `merge_glossary(accumulated: Mapping[str, str], fresh: Mapping[str, str]) -> dict[str, str]`
  - `effective_glossary(accumulated: Mapping[str, str], manual: Mapping[str, str]) -> dict[str, str]`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_translate_glossary.py`：

```python
"""累积术语表。纯文件读写与字典合并，没有外部依赖。"""

from pathlib import Path

from tenmin.translate import glossary as g


def test_loading_a_missing_file_gives_an_empty_table(tmp_path):
    """第一集跑之前这个文件不存在，那不是错误。"""
    assert g.load_glossary(tmp_path / "nope.json") == {}


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "zh" / "glossary.json"
    g.save_glossary(path, {"リディア": "莉迪亚", "ルーファス": "鲁弗斯"})
    assert g.load_glossary(path) == {"リディア": "莉迪亚", "ルーファス": "鲁弗斯"}


def test_saved_file_is_human_readable_utf8(tmp_path):
    path = tmp_path / "glossary.json"
    g.save_glossary(path, {"リディア": "莉迪亚"})
    body = path.read_text(encoding="utf-8")
    assert "莉迪亚" in body
    assert "\\u" not in body


def test_saved_keys_are_sorted(tmp_path):
    """稳定顺序让这个文件的 diff 可读 —— 它是人会去手动纠错的文件。"""
    path = tmp_path / "glossary.json"
    g.save_glossary(path, {"ロ": "罗", "イ": "伊", "ハ": "哈"})
    body = path.read_text(encoding="utf-8")
    assert body.index('"イ"') < body.index('"ハ"') < body.index('"ロ"')


def test_loading_a_corrupt_file_gives_an_empty_table(tmp_path):
    """这个文件坏了不该让整条管线停 —— 最坏情况是这一集的译名不跟前几集对齐。"""
    path = tmp_path / "glossary.json"
    path.write_text("{ 这不是 json", encoding="utf-8")
    assert g.load_glossary(path) == {}


def test_merge_keeps_the_accumulated_choice_for_known_terms():
    """已经定下的译名不许被后面某一集改掉 —— 那正是累积要防的事。"""
    merged = g.merge_glossary({"リディア": "莉迪亚"}, {"リディア": "莉蒂亚"})
    assert merged == {"リディア": "莉迪亚"}


def test_merge_adds_new_terms():
    merged = g.merge_glossary({"リディア": "莉迪亚"}, {"ルーファス": "鲁弗斯"})
    assert merged == {"リディア": "莉迪亚", "ルーファス": "鲁弗斯"}


def test_merge_ignores_blank_entries():
    merged = g.merge_glossary({}, {"リディア": "", "": "空键", "  ": "  "})
    assert merged == {}


def test_merge_strips_whitespace():
    merged = g.merge_glossary({}, {" リディア ": " 莉迪亚 "})
    assert merged == {"リディア": "莉迪亚"}


def test_merge_does_not_mutate_its_inputs():
    accumulated = {"リディア": "莉迪亚"}
    fresh = {"ルーファス": "鲁弗斯"}
    g.merge_glossary(accumulated, fresh)
    assert accumulated == {"リディア": "莉迪亚"}
    assert fresh == {"ルーファス": "鲁弗斯"}


def test_manual_entries_override_accumulated_ones():
    """手写表是纠错入口：机器译错了人要能盖掉它。"""
    effective = g.effective_glossary({"リディア": "莉蒂亚"}, {"リディア": "莉迪亚"})
    assert effective["リディア"] == "莉迪亚"


def test_effective_glossary_unions_both_sides():
    effective = g.effective_glossary({"イ": "伊"}, {"ロ": "罗"})
    assert effective == {"イ": "伊", "ロ": "罗"}


def test_effective_glossary_of_nothing_is_empty():
    assert g.effective_glossary({}, {}) == {}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_translate_glossary.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.translate'`

- [ ] **Step 3: 建包并实现**

`src/tenmin/translate/__init__.py`：

```python
"""翻译阶段：把听写来的日语对白逐条译成简体中文。

产出两样东西：一份中文字幕交付物，和一份跨集累积的专有名词表。后者会回喂给 script
阶段，让解说稿里的人名跟字幕里的写法一致。
"""
```

`src/tenmin/translate/glossary.py`：

```python
"""跨集累积的专有名词表。

存在的理由是译名漂移有两根轴：同一集里「字幕写莉迪亚、解说稿写莉蒂亚」（翻译与解说
是两次模型调用），以及集与集之间（每集的翻译调用互不知情，后者更难发现 —— 单看一集
完全自洽）。一份项目级的表把两根轴一起按住，而且随集数收敛。

手写的那份（project.yaml 里的 glossary）是纠错入口：机器译错了人要能盖掉它。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ..atomic import atomic_path


def load_glossary(path: Path) -> dict[str, str]:
    """读累积表。文件不存在或坏了都返回空表。

    两种情况都刻意不报错：第一集跑之前这个文件本来就不存在；而它坏掉的最坏后果只是
    「这一集的译名不跟前几集对齐」，不值得让整条管线停下来。
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def save_glossary(path: Path, glossary: Mapping[str, str]) -> None:
    """写累积表。

    键排序 + 不转义非 ASCII：这是个人会去手动纠错的文件，diff 得可读。
    """
    payload = json.dumps(dict(sorted(glossary.items())), ensure_ascii=False, indent=2)
    with atomic_path(path) as staged:
        staged.write_text(payload + "\n", encoding="utf-8")


def _clean(entries: Mapping[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in entries.items():
        term = str(key).strip()
        translation = str(value).strip()
        if term and translation:
            cleaned[term] = translation
    return cleaned


def merge_glossary(
    accumulated: Mapping[str, str], fresh: Mapping[str, str]
) -> dict[str, str]:
    """把这一集新认出来的词并进累积表。

    已经定下的译名**不许**被后面某一集改掉 —— 那正是累积要防的事（第 5 集把人名换个
    写法，成片看起来就像换了个角色）。所以冲突时保留累积的那个。

    返回新字典，不改入参。
    """
    merged = _clean(accumulated)
    for term, translation in _clean(fresh).items():
        merged.setdefault(term, translation)
    return merged


def effective_glossary(
    accumulated: Mapping[str, str], manual: Mapping[str, str]
) -> dict[str, str]:
    """喂给模型的那份表：累积的叠上手写的，手写的赢。

    手写表是纠错入口，它必须能盖掉机器的选择，否则「改了 project.yaml 却不生效」会是
    个很难查的问题。
    """
    return {**_clean(accumulated), **_clean(manual)}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_translate_glossary.py -v`
Expected: PASS（13 passed）

- [ ] **Step 5: 提交**

```bash
rtk git add src/tenmin/translate/__init__.py src/tenmin/translate/glossary.py tests/test_translate_glossary.py
rtk git commit -m "feat: 加跨集累积术语表的读写与合并"
```

---

### Task 10: `translate/srt_writer.py` —— 中文 SRT 交付物

**Files:**
- Create: `src/tenmin/translate/srt_writer.py`
- Test: `tests/test_translate_srt_writer.py`

**Interfaces:**
- Consumes: `models.DialogueTrack`、`models.TranslatedTrack`
- Produces: `render_zh_srt(track: DialogueTrack, translated: TranslatedTrack) -> str`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_translate_srt_writer.py`：

```python
"""中文字幕交付物的拼装。纯函数。"""

import pytest

from tenmin.models import DialogueLine, DialogueTrack, TranslatedLine, TranslatedTrack
from tenmin.translate.srt_writer import render_zh_srt


def _line(idx: int, start: float, end: float, text: str, kind: str = "dialogue"):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind=kind)


def _track(*lines: DialogueLine, duration: float = 100.0) -> DialogueTrack:
    return DialogueTrack(episode=11, source="asr", duration=duration, lines=list(lines))


def test_renders_timestamps_from_the_dialogue_track():
    """译文里只有 id 和中文，时间戳必须回到对白轨上取。"""
    track = _track(_line(1, 1.5, 2.25, "はい"), _line(2, 10.0, 12.125, "そうですね"))
    translated = TranslatedTrack(
        episode=11,
        lines=[TranslatedLine(id=1, zh="是的"), TranslatedLine(id=2, zh="说得对")],
    )

    body = render_zh_srt(track, translated)

    assert "00:00:01,500 --> 00:00:02,250" in body
    assert "是的" in body
    assert "00:00:10,000 --> 00:00:12,125" in body
    assert "说得对" in body


def test_blocks_are_numbered_from_one_consecutively():
    track = _track(_line(1, 1.0, 2.0, "あ"), _line(2, 3.0, 4.0, "い"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="啊"), TranslatedLine(id=2, zh="咦")]
    )
    body = render_zh_srt(track, translated)
    assert body.startswith("1\n")
    assert "\n2\n" in body


def test_untranslated_lines_are_skipped(tmp_path):
    """片头片尾那些行压根不会被送去翻译，字幕里也就不该出现空块。"""
    track = _track(
        _line(1, 1.0, 2.0, "作曲：某人", kind="credits"),
        _line(2, 3.0, 4.0, "本編のセリフ"),
    )
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=2, zh="正片台词")])

    body = render_zh_srt(track, translated)

    assert body.count("-->") == 1
    assert "正片台词" in body
    assert "作曲" not in body


def test_an_id_outside_the_track_is_an_error():
    """译文带了一个对白轨里不存在的 id 意味着上游对齐校验漏了 —— 别静默丢掉。"""
    track = _track(_line(1, 1.0, 2.0, "あ"))
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=99, zh="啊")])
    with pytest.raises(ValueError) as excinfo:
        render_zh_srt(track, translated)
    assert "99" in str(excinfo.value)


def test_output_is_ordered_by_time_not_by_translation_order():
    track = _track(_line(1, 5.0, 6.0, "後"), _line(2, 1.0, 2.0, "先"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="后"), TranslatedLine(id=2, zh="先")]
    )
    body = render_zh_srt(track, translated)
    assert body.index("先") < body.index("后")


def test_multiline_translation_is_kept_as_is():
    track = _track(_line(1, 1.0, 2.0, "あ"))
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=1, zh="第一行\n第二行")])
    body = render_zh_srt(track, translated)
    assert "第一行\n第二行" in body


def test_blank_translations_are_skipped():
    track = _track(_line(1, 1.0, 2.0, "あ"), _line(2, 3.0, 4.0, "い"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="   "), TranslatedLine(id=2, zh="咦")]
    )
    body = render_zh_srt(track, translated)
    assert body.count("-->") == 1
    assert "咦" in body


def test_an_empty_translation_track_gives_an_empty_file():
    track = _track(_line(1, 1.0, 2.0, "あ"))
    assert render_zh_srt(track, TranslatedTrack(episode=11, lines=[])) == ""


def test_round_trips_through_the_srt_parser():
    """最硬的不变量：产出的东西得能被本项目自己的解析器吃回去。"""
    from tenmin.ingest.srt_parser import parse_srt

    track = _track(_line(1, 1.5, 2.25, "はい"), _line(2, 10.0, 12.0, "いいえ"))
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="是"), TranslatedLine(id=2, zh="不是")]
    )
    cues = parse_srt(render_zh_srt(track, translated))
    assert [(c.start, c.end, c.text) for c in cues] == [(1.5, 2.25, "是"), (10.0, 12.0, "不是")]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_translate_srt_writer.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.translate.srt_writer'`

- [ ] **Step 3: 实现**

`src/tenmin/translate/srt_writer.py`：

```python
"""中文字幕交付物。

译文轨里只有「第几条对白 → 中文」，时间戳要回到对白轨上取 —— 这是刻意的分工：翻译
不碰时间，时间轴的唯一真相在对白轨里。
"""

from __future__ import annotations

from ..models import DialogueTrack, TranslatedTrack


def _format_timestamp(seconds: float) -> str:
    """SRT 时间戳。毫秒分隔符是逗号 —— timecode 那份是点号版本，不能用在这里。"""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_zh_srt(track: DialogueTrack, translated: TranslatedTrack) -> str:
    """对白轨的时间 + 译文轨的中文 → 一份可直接拖进播放器的 SRT。

    id 是对白在 track.lines 里的位置（从 1 起）。没有译文的行直接跳过（片头片尾那些
    行压根不会被送去翻译），译文为空白的也跳过 —— 空字幕块会让播放器显示一个空行。

    带了轨里不存在的 id 则报错：那意味着上游的对齐校验漏了，静默丢掉会让「译文少了
    一句」这种事永远查不出来。
    """
    total = len(track.lines)
    blocks: list[tuple[float, float, str]] = []
    for line in translated.lines:
        if line.id < 1 or line.id > total:
            raise ValueError(
                f"译文里的 id {line.id} 超出对白轨范围（共 {total} 行），"
                f"上游的对齐校验没拦住"
            )
        text = line.zh.strip()
        if not text:
            continue
        source = track.lines[line.id - 1]
        blocks.append((source.start, source.end, text))

    blocks.sort(key=lambda item: (item[0], item[1]))
    rendered = [
        f"{index}\n{_format_timestamp(start)} --> {_format_timestamp(end)}\n{text}\n"
        for index, (start, end, text) in enumerate(blocks, start=1)
    ]
    return "\n".join(rendered)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_translate_srt_writer.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: 提交**

```bash
rtk git add src/tenmin/translate/srt_writer.py tests/test_translate_srt_writer.py
rtk git commit -m "feat: 加中文字幕交付物的拼装"
```

---

### Task 11: `translate/lines.py` —— 逐条翻译与 id 对齐校验

这一任务的核心是 id 对齐：进去多少条就必须出来多少条。模型漏一条、合并两条，中文就会
跟时间戳整条错位，而那种错只有看成片字幕时才发现，且是从错位那一句往后全错。

**Files:**
- Create: `src/tenmin/translate/prompts/translate_lines.md`
- Create: `src/tenmin/translate/lines.py`
- Test: `tests/test_translate_lines.py`

**Interfaces:**
- Consumes: Task 8 的 `complete_with_schema_repair`、`LLMResponseFormatError`、`LLMProvider`；Task 7 的 `TranslatedTrack`；Task 9 的 `effective_glossary`；`script.prompt.render_prompt`
- Produces:
  - `select_translatable(track: DialogueTrack) -> list[tuple[int, DialogueLine]]`
  - `build_lines_block(selected: Sequence[tuple[int, DialogueLine]]) -> str`
  - `check_alignment(expected_ids: Collection[int], translated: TranslatedTrack) -> None`
  - `async translate_track(cfg: ProjectConfig, track: DialogueTrack, provider: LLMProvider, *, accumulated: Mapping[str, str]) -> TranslatedTrack`

- [ ] **Step 1: 写 prompt 模板**

创建 `src/tenmin/translate/prompts/translate_lines.md`：

```markdown
你是一名日语字幕译者。把下面这一集动画的日语对白逐条译成简体中文。

## 交付要求

- **逐条对应**：输入有多少条，输出就必须有多少条，`id` 一个都不能漏、不能多、不能重复。
- **不要合并、不要拆分**：即使两条在日语里是一句话被切开的，也各自译各自的。时间轴按条
  对齐，合并会让后面全部错位。
- 译成**简体中文**，口语化、自然，符合角色说话的语气。
- 保留语气词与停顿感，但不要加原文没有的内容。
- 专有名词（人名、地名、组织、称号）按「已定术语」里的写法；那里没有的，自己定一个
  自然的译法，并在 `glossary` 里报上来。
- 遇到听写错误（同音字错、句子被截断），按上下文推断本意再译，不要照着错字硬译。

## 输出格式

严格输出符合下面 JSON Schema 的 JSON，不要任何额外说明文字、不要 markdown 代码围栏。

```json
{{schema}}
```

`glossary` 只报**这一集里出现的**专有名词，键是日语原文、值是你选的中文译法。
「已定术语」里已经有的不用重复报。

## 已定术语（必须照这个写法）

{{glossary_block}}

## 对白轨

每行格式 `id | 日语原文`。

{{lines_block}}

## 输出前自检

1. 数一遍：输出的 `lines` 条数是否等于输入的行数？
2. 每个 `id` 是否都在输入里出现过，且只出现一次？
3. `glossary` 里的人名译法，是否跟 `lines` 里实际用的写法一致？
```

段落顺序照 script 那份模板已经标定过的原则：完全静态的说明在前、本集素材在后。这里
素材只有一份（对白轨），所以把它排在最后，让「任务 + 格式 + 术语表」那段成为跨集可
复用的前缀。

- [ ] **Step 2: 写失败的测试**

创建 `tests/test_translate_lines.py`：

```python
"""翻译阶段的编排与 id 对齐校验。

id 对齐是这个文件里最该锁死的东西：模型漏一条、合并两条，中文就跟时间戳整条错位，
而那种错只有看成片字幕时才发现，且是从错位那句往后全错。
"""

import json

import pytest

from tenmin.config import ProjectConfig
from tenmin.models import DialogueLine, DialogueTrack, TranslatedLine, TranslatedTrack
from tenmin.script.llm import LLMResponseFormatError, LLMSchemaError
from tenmin.translate import lines as tl


def _line(idx: int, text: str, kind: str = "dialogue", start: float = 0.0):
    return DialogueLine(
        idx=idx, start=start, end=start + 1.0, text=text, raw=text, kind=kind
    )


def _track(*items: DialogueLine) -> DialogueTrack:
    return DialogueTrack(episode=11, source="asr", duration=100.0, lines=list(items))


# ---- 选哪些行去翻译 ----


def test_only_dialogue_lines_are_translated():
    """片头片尾的 staff 名单、标题卡不是对白，翻它们是白花钱。"""
    track = _track(
        _line(1, "作曲：某人", kind="credits"),
        _line(2, "本編のセリフ"),
        _line(3, "タイトル", kind="title"),
    )
    selected = tl.select_translatable(track)
    assert [position for position, _ in selected] == [2]


def test_positions_are_one_based_indexes_into_the_track():
    """id 用位置下标而不是 DialogueLine.idx：后者在双轨字幕被拆开时会重复。"""
    track = _track(_line(7, "あ"), _line(7, "い"))
    selected = tl.select_translatable(track)
    assert [position for position, _ in selected] == [1, 2]


def test_a_track_with_no_dialogue_selects_nothing():
    assert tl.select_translatable(_track(_line(1, "x", kind="credits"))) == []


# ---- prompt 里的对白块 ----


def test_lines_block_is_id_pipe_text():
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    block = tl.build_lines_block(tl.select_translatable(track))
    assert block == "1 | はい\n2 | いいえ"


def test_lines_block_keeps_the_selected_ids_not_a_renumbering():
    track = _track(_line(1, "x", kind="credits"), _line(2, "はい"))
    block = tl.build_lines_block(tl.select_translatable(track))
    assert block == "2 | はい"


# ---- id 对齐校验（核心）----


def test_alignment_accepts_an_exact_match():
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="一"), TranslatedLine(id=2, zh="二")]
    )
    tl.check_alignment({1, 2}, translated)  # 不抛


def test_alignment_rejects_a_missing_id():
    translated = TranslatedTrack(episode=11, lines=[TranslatedLine(id=1, zh="一")])
    with pytest.raises(LLMResponseFormatError) as excinfo:
        tl.check_alignment({1, 137, 298}, translated)
    message = str(excinfo.value)
    assert "137" in message and "298" in message


def test_alignment_rejects_an_unknown_id():
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="一"), TranslatedLine(id=999, zh="?")]
    )
    with pytest.raises(LLMResponseFormatError) as excinfo:
        tl.check_alignment({1}, translated)
    assert "999" in str(excinfo.value)


def test_alignment_rejects_a_duplicated_id():
    """条数对上了但有重复 —— 单看数量查不出来，必须按集合判。"""
    translated = TranslatedTrack(
        episode=11, lines=[TranslatedLine(id=1, zh="一"), TranslatedLine(id=1, zh="壹")]
    )
    with pytest.raises(LLMResponseFormatError) as excinfo:
        tl.check_alignment({1, 2}, translated)
    assert "1" in str(excinfo.value)


def test_alignment_error_message_is_bounded():
    """回灌给模型的消息不能是三百个数字 —— 那会挤掉真正要修的内容。"""
    with pytest.raises(LLMResponseFormatError) as excinfo:
        tl.check_alignment(set(range(1, 401)), TranslatedTrack(episode=11, lines=[]))
    assert len(str(excinfo.value)) < 600


# ---- 编排 ----


class _ScriptedProvider:
    """按脚本逐次返回预设的响应，记下每次收到的 prompt。"""

    def __init__(self, *payloads: str):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    async def complete(self, prompt, *, schema=None, **kwargs):
        self.prompts.append(prompt)
        return self.payloads[len(self.prompts) - 1]


def _payload(ids, glossary=None) -> str:
    return json.dumps(
        {
            "episode": 11,
            "lines": [{"id": i, "zh": f"译{i}"} for i in ids],
            "glossary": glossary or {},
        },
        ensure_ascii=False,
    )


def _cfg(tmp_path) -> ProjectConfig:
    return ProjectConfig(show="测试番", slug="test").bind_root(tmp_path)


@pytest.mark.asyncio
async def test_translate_track_returns_the_aligned_result(tmp_path):
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    provider = _ScriptedProvider(_payload([1, 2], {"リディア": "莉迪亚"}))

    result = await tl.translate_track(
        _cfg(tmp_path), track, provider, accumulated={}
    )

    assert [line.id for line in result.lines] == [1, 2]
    assert result.glossary == {"リディア": "莉迪亚"}
    assert result.episode == 11


@pytest.mark.asyncio
async def test_translate_track_retries_a_misaligned_response(tmp_path):
    """第一轮漏了第 2 条，回灌重试后补齐。"""
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    provider = _ScriptedProvider(_payload([1]), _payload([1, 2]))

    result = await tl.translate_track(_cfg(tmp_path), track, provider, accumulated={})

    assert [line.id for line in result.lines] == [1, 2]
    assert len(provider.prompts) == 2


@pytest.mark.asyncio
async def test_translate_track_gives_up_after_the_retry_budget(tmp_path):
    track = _track(_line(1, "はい"), _line(2, "いいえ"))
    provider = _ScriptedProvider(*[_payload([1])] * 10)

    with pytest.raises(LLMSchemaError):
        await tl.translate_track(_cfg(tmp_path), track, provider, accumulated={})


@pytest.mark.asyncio
async def test_the_prompt_carries_the_accumulated_glossary(tmp_path):
    track = _track(_line(1, "はい"))
    provider = _ScriptedProvider(_payload([1]))

    await tl.translate_track(
        _cfg(tmp_path), track, provider, accumulated={"リディア": "莉迪亚"}
    )

    assert "莉迪亚" in provider.prompts[0]


@pytest.mark.asyncio
async def test_manual_glossary_overrides_the_accumulated_one_in_the_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.glossary = {"リディア": "莉蒂亚"}
    provider = _ScriptedProvider(_payload([1]))

    await tl.translate_track(
        cfg, _track(_line(1, "はい")), provider, accumulated={"リディア": "莉迪亚"}
    )

    assert "莉蒂亚" in provider.prompts[0]
    assert "莉迪亚" not in provider.prompts[0]


@pytest.mark.asyncio
async def test_a_track_without_dialogue_short_circuits(tmp_path):
    """一条对白都没有就别发请求。"""
    provider = _ScriptedProvider()
    result = await tl.translate_track(
        _cfg(tmp_path), _track(_line(1, "x", kind="credits")), provider, accumulated={}
    )
    assert result.lines == []
    assert provider.prompts == []
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_translate_lines.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.translate.lines'`

- [ ] **Step 4: 实现 `src/tenmin/translate/lines.py`**

```python
"""一集对白的翻译编排。

体量实测：一集约 400 条、日语 5~6k 字符，一次调用完全塞得下，所以刻意不分批。分批要
付的代价是「术语跨批不一致」加「调用次数乘以批数」，换来的是解决一个不存在的问题。

真正的风险不是成本而是**对齐**：进去多少条就必须出来多少条。模型漏一条、合并两条，
中文就跟时间戳整条错位，而那种错只有看成片字幕时才发现，且是从错位那句往后全错。所以
输出带 id，拿回来先核对 id 集合，不对就把缺的/多的报给模型重来。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from ..config import ProjectConfig
from ..models import DialogueLine, DialogueTrack, TranslatedTrack
from ..script.llm import (
    LLMProvider,
    LLMResponseFormatError,
    RepairContext,
    complete_with_schema_repair,
)
from ..script.prompt import load_prompt, render_prompt
from .glossary import effective_glossary

_PROMPTS = Path(__file__).parent / "prompts"
_TEMPLATE = "translate_lines.md"

# 报给模型的缺失/多余 id 最多列这么多个。回灌的消息要是三百个数字，真正要修的内容
# 就被挤出模型的注意力了 —— 列几个够它明白「你漏了东西，重新数一遍」就行。
_MAX_REPORTED_IDS = 12


def select_translatable(track: DialogueTrack) -> list[tuple[int, DialogueLine]]:
    """挑出要翻译的行，连同它们在轨里的位置（从 1 起）。

    只翻 kind == "dialogue" 的：片头片尾的 staff 名单、标题卡不是对白，翻它们是白
    花钱；而主题曲歌词在手填了 OP/ED 区间时也会落在这一类之外。

    位置下标而不是 DialogueLine.idx：后者在双轨字幕被拆开时会重复（同一个 idx 配不同
    的 segment_index），拿它当键会撞。位置天然唯一，也让对齐校验变成一次集合相等判断。
    """
    return [
        (position, line)
        for position, line in enumerate(track.lines, start=1)
        if line.kind == "dialogue"
    ]


def build_lines_block(selected: Sequence[tuple[int, DialogueLine]]) -> str:
    """对白轨渲染成 `id | 原文` 的清单。"""
    return "\n".join(f"{position} | {line.text}" for position, line in selected)


def _sample(ids: Collection[int]) -> str:
    ordered = sorted(ids)
    head = ordered[:_MAX_REPORTED_IDS]
    shown = "、".join(str(i) for i in head)
    if len(ordered) > len(head):
        shown += f" 等共 {len(ordered)} 个"
    return shown


def check_alignment(expected_ids: Collection[int], translated: TranslatedTrack) -> None:
    """译文的 id 集合必须跟送进去的完全相等。

    按集合判而不是按条数判：条数对上但有重复的情况（模型把某条译了两遍、漏了另一条）
    单看数量查不出来。

    抛 LLMResponseFormatError 是为了复用 schema 修复层那套回灌重试 —— 消息里带上具体
    缺了哪几个 id，模型照着就能改。
    """
    expected = set(expected_ids)
    got = [line.id for line in translated.lines]
    seen = set(got)

    missing = expected - seen
    unknown = seen - expected
    duplicated = {i for i in got if got.count(i) > 1}

    problems: list[str] = []
    if missing:
        problems.append(f"漏了这些 id: {_sample(missing)}")
    if unknown:
        problems.append(f"输出了输入里没有的 id: {_sample(unknown)}")
    if duplicated:
        problems.append(f"这些 id 出现了多次: {_sample(duplicated)}")
    if problems:
        raise LLMResponseFormatError(
            "译文没有逐条对齐（输入 "
            f"{len(expected)} 条，输出 {len(got)} 条）。"
            + "；".join(problems)
            + "。请重新输出全部条目，每条 id 恰好一次。"
        )


async def translate_track(
    cfg: ProjectConfig,
    track: DialogueTrack,
    provider: LLMProvider,
    *,
    accumulated: Mapping[str, str],
) -> TranslatedTrack:
    """翻译一集对白。

    accumulated 是前面几集攒下的术语表；project.yaml 里手写的那份会盖在它上面（手写
    是纠错入口，必须赢）。
    """
    selected = select_translatable(track)
    if not selected:
        return TranslatedTrack(episode=track.episode)

    glossary = effective_glossary(accumulated, cfg.glossary)
    expected_ids = {position for position, _ in selected}
    schema_json = TranslatedTrack.model_json_schema()
    template = load_prompt(_PROMPTS, _TEMPLATE)

    def build_prompt(repair: RepairContext | None) -> str:
        body = render_prompt(
            template,
            _TEMPLATE,
            schema=_json_dumps(schema_json),
            glossary_block=_glossary_block(glossary),
            lines_block=build_lines_block(selected),
        )
        if repair is None:
            return body
        # 纠错轮把「坏输出 + 报错」追加在后面。刻意不重发一份精简版正文：要修的是
        # 「哪几条漏了」，而模型得对着原文才能补出那几条的译文。
        return (
            f"{body}\n\n## 上一次的输出有问题\n\n"
            f"{repair.error}\n\n上一次的输出（可能被截断）：\n\n{repair.bad_output}\n"
        )

    async def send(repair: RepairContext | None) -> str:
        return await provider.complete(build_prompt(repair))

    return await complete_with_schema_repair(
        send,
        TranslatedTrack,
        max_attempts=cfg.llm.max_attempts,
        label=f"E{track.episode:02d} 翻译",
        check=lambda result: check_alignment(expected_ids, result),
    )
```

补两个小辅助（放在 `build_lines_block` 附近）：

```python
def _json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _glossary_block(glossary: Mapping[str, str]) -> str:
    """跟 script 阶段的术语表块同形，空表时给一句话而不是留白。"""
    if not glossary:
        return "（暂无已定术语，自己定并在 glossary 里报上来）"
    return "\n".join(f"- {term} → {translation}" for term, translation in sorted(glossary.items()))
```

实现时要核对三处现状并按实际情况调整：
- `load_prompt` / `render_prompt` 的确切签名（`render_prompt` 的模板名是**位置参数**）。若 `load_prompt` 只接受模块内固定目录，改成本模块自己读文件 + `functools.cache`。
- `cfg.llm` 上控制修复轮数的字段名（script 那边用的是「schema 修复次数」那个，不是校验重试次数）。以 `LLMConfig` 现状为准。
- `provider.complete` 的调用形态（script 那边怎么调就怎么调；这里刻意不传 `schema=`，因为 schema 已经写进 prompt 且我们要自己接管校验与重试）。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_translate_lines.py -v`
Expected: PASS（18 passed）

- [ ] **Step 6: 提交**

```bash
rtk git add src/tenmin/translate/lines.py src/tenmin/translate/prompts/translate_lines.md tests/test_translate_lines.py
rtk git commit -m "feat: 加逐条翻译与 id 对齐校验"
```

---

### Task 12: `translate` 阶段接进管线

**Files:**
- Modify: `src/tenmin/pipeline.py`（`STAGES`、`run_translate`、`run_pipeline` 接线、script 新鲜度加 glossary）
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 7 的 `Paths.zh_lines` / `Paths.zh_subtitles` / `Paths.glossary`；Task 9 的 glossary 读写；Task 10 的 `render_zh_srt`；Task 11 的 `translate_track`
- Produces: `async run_translate(cfg, provider, episode) -> TranslatedTrack`；`STAGES` 多一项 `"translate"`

- [ ] **Step 1: 写失败的测试**

`tests/test_pipeline.py` 追加：

```python
def test_translate_sits_between_ingest_and_signals():
    from tenmin.pipeline import STAGES

    assert STAGES.index("ingest") < STAGES.index("translate") < STAGES.index("signals")


@pytest.mark.asyncio
async def test_run_translate_writes_all_three_artifacts(tmp_path, monkeypatch):
    """译文轨（中间产物）、中文字幕（交付物）、累积术语表（回写）。"""
    cfg, paths = _project_with_asr_dialogue(tmp_path, episode=11)
    provider = _translation_provider({1: "是的", 2: "说得对"}, glossary={"リディア": "莉迪亚"})

    await run_translate(cfg, provider, 11)

    assert paths.zh_lines(11).is_file()
    assert paths.zh_subtitles(11).is_file()
    assert "是的" in paths.zh_subtitles(11).read_text(encoding="utf-8")
    assert json.loads(paths.glossary.read_text(encoding="utf-8")) == {"リディア": "莉迪亚"}


@pytest.mark.asyncio
async def test_run_translate_accumulates_into_an_existing_glossary(tmp_path, monkeypatch):
    """第 2 集要看得见第 1 集定下的译名，且不能把它改掉。"""
    cfg, paths = _project_with_asr_dialogue(tmp_path, episode=11)
    paths.glossary.parent.mkdir(parents=True, exist_ok=True)
    paths.glossary.write_text(
        json.dumps({"リディア": "莉迪亚"}, ensure_ascii=False), encoding="utf-8"
    )
    provider = _translation_provider(
        {1: "是的", 2: "说得对"}, glossary={"リディア": "莉蒂亚", "ルーファス": "鲁弗斯"}
    )

    await run_translate(cfg, provider, 11)

    stored = json.loads(paths.glossary.read_text(encoding="utf-8"))
    assert stored["リディア"] == "莉迪亚"
    assert stored["ルーファス"] == "鲁弗斯"


@pytest.mark.asyncio
async def test_run_translate_skips_a_native_subtitle_episode(tmp_path):
    """现有片源自带中文字幕，整个阶段不该跑，zh/ 目录都不该建。

    判据是对白轨的 source 字段，刻意不做语言自动检测：那是个会错的猜测，而
    source 是一个确定的事实。
    """
    cfg, paths = _project_with_srt_dialogue(tmp_path, episode=2)
    provider = _exploding_provider()

    await run_translate(cfg, provider, 2)

    assert not paths.zh_lines(2).exists()
    assert not paths.zh_subtitles(2).exists()


def test_script_freshness_depends_on_the_glossary(tmp_path):
    """手改了译名，解说稿该重跑 —— 术语表是 script 的真输入。"""
    from tenmin.pipeline import _script_inputs

    cfg, paths = _project_with_asr_dialogue(tmp_path, episode=11)
    assert paths.glossary in _script_inputs(paths, 11)


def test_translate_freshness_does_not_include_the_glossary(tmp_path):
    """术语表既是 translate 的输入又是它的输出。把它算进输入集，这个阶段就永远
    不新鲜 —— 每次都重跑，每次都重新付翻译费。"""
    from tenmin.pipeline import _translate_inputs

    cfg, paths = _project_with_asr_dialogue(tmp_path, episode=11)
    inputs = _translate_inputs(paths, 11)
    assert paths.dialogue(11) in inputs
    assert paths.glossary not in inputs
```

需要三个测试辅助（放在该文件的辅助区）：

```python
def _project_with_asr_dialogue(tmp_path, *, episode: int):
    """建一个 project，并落一份 source == "asr" 的对白轨产物。"""
    cfg = _write_project(tmp_path)
    video = tmp_path / f"e{episode}.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=episode, srt=None, video=video)
    paths = Paths(cfg.root)
    track = DialogueTrack(
        episode=episode,
        source="asr",
        duration=100.0,
        lines=[
            DialogueLine(idx=1, start=1.0, end=2.0, text="はい", raw="はい"),
            DialogueLine(idx=2, start=3.0, end=4.0, text="そうですね", raw="そうですね"),
        ],
    )
    target = paths.dialogue(episode)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(track.model_dump_json(indent=2), encoding="utf-8")
    return cfg, paths


def _project_with_srt_dialogue(tmp_path, *, episode: int):
    """同上，但对白轨是原生字幕来的。"""
    cfg = _write_project(tmp_path)
    srt = tmp_path / "hand.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n你好\n", encoding="utf-8")
    video = tmp_path / f"e{episode}.mp4"
    video.write_bytes(b"fake")
    cfg = register_episode(cfg, episode=episode, srt=srt, video=video)
    paths = Paths(cfg.root)
    track = DialogueTrack(
        episode=episode,
        source="srt",
        duration=100.0,
        lines=[DialogueLine(idx=1, start=1.0, end=2.0, text="你好", raw="你好")],
    )
    target = paths.dialogue(episode)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(track.model_dump_json(indent=2), encoding="utf-8")
    return cfg, paths


def _translation_provider(mapping: dict[int, str], *, glossary: dict[str, str] | None = None):
    class _P:
        async def complete(self, prompt, *, schema=None, **kwargs):
            return json.dumps(
                {
                    "episode": 11,
                    "lines": [{"id": i, "zh": zh} for i, zh in mapping.items()],
                    "glossary": glossary or {},
                },
                ensure_ascii=False,
            )

    return _P()


def _exploding_provider():
    class _P:
        async def complete(self, prompt, *, schema=None, **kwargs):
            raise AssertionError("这一集不该调用模型")

    return _P()
```

（`_exploding_provider` 里的 `AssertionError` 在 `tests/` 下是允许的 —— 禁 assert 那条只管 `src/`。若该文件已有同类「不该被调用」的假货，复用它。）

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_pipeline.py -k "translate or script_freshness" -v`
Expected: FAIL

- [ ] **Step 3: 加 `STAGES` 与两个输入集辅助**

`src/tenmin/pipeline.py`：`STAGES` 里 `"ingest"` 之后插入 `"translate"`。

新增两个辅助（放在 `_is_fresh` 附近）：

```python
def _translate_inputs(paths: Paths, episode: int) -> list[Path]:
    """翻译阶段的新鲜度输入。

    只看对白轨。累积术语表刻意**不**在里面：它既是这个阶段的输入又是它的输出，算进
    输入集会让这个阶段永远不新鲜 —— 每次都重跑，每次都重新付翻译费。

    代价是：手改了术语表不会让已经翻好的集自动重翻（要 --force）。这是有意的取舍 ——
    改译名的主要目的是让**后面**几集和解说稿用对写法，而解说稿那边的新鲜度是真的
    挂着术语表的（见 _script_inputs）。
    """
    return [paths.dialogue(episode)]


def _script_inputs(paths: Paths, episode: int) -> list[Path]:
    """解说稿阶段的新鲜度输入。

    术语表是这里的真输入：解说稿的 prompt 会读它，手动改了译名就该重写解说稿。
    文件不存在时会被新鲜度判据自己过滤掉（它只看存在的那些），所以无条件列进来是安全的。
    """
    return [paths.dialogue(episode), paths.signals(episode), paths.glossary]
```

把 `_launch_scripts` 里那行 `inputs = [paths.dialogue(number), paths.signals(number)]` 换成 `inputs = _script_inputs(paths, number)`。

- [ ] **Step 4: 实现 `run_translate`**

在 `run_ingest` 之后插入：

```python
async def run_translate(
    cfg: ProjectConfig, provider: LLMProvider, episode: int
) -> TranslatedTrack:
    """翻译一集：落译文轨、中文字幕，并把新认出的术语并回累积表。

    只对听写来的对白动手。手传的字幕与从视频里抽出来的软字幕轨都是片源自带的，本来
    就是观众能读的语言。判据用对白轨的 source 字段，刻意不做语言自动检测 —— 那是个
    会错的猜测，而 source 是一个确定的事实。

    已知边界：手传一份日语 SRT（或者软字幕轨恰好是日语）时，这一阶段不会跑，而且那份
    对白还会被繁转简改字。目前的片源都不是这种情况，真碰上了再说。
    """
    track = next(t for t in _load_tracks(cfg) if t.episode == episode)
    if track.source != "asr":
        return TranslatedTrack(episode=episode)

    paths = Paths(cfg.root)
    accumulated = load_glossary(paths.glossary)
    translated = await translate_track(cfg, track, provider, accumulated=accumulated)

    _write_json(paths.zh_lines(episode), translated.model_dump_json(indent=2))
    _write_text(paths.zh_subtitles(episode), render_zh_srt(track, translated))
    save_glossary(paths.glossary, merge_glossary(accumulated, translated.glossary))
    return translated
```

（import：`from .translate.glossary import load_glossary, merge_glossary, save_glossary`、`from .translate.lines import translate_track`、`from .translate.srt_writer import render_zh_srt`、`from .models import TranslatedTrack`。）

- [ ] **Step 5: 接进 `run_pipeline` 的纵向循环**

translate 是**按集**的、要 LLM 的阶段，所以它归在纵向循环里、紧跟 ingest 之后、signals 之前。在纵向循环里 script 块之前插入：

```python
            if "translate" in stages:
                outputs = [paths.zh_lines(number), paths.zh_subtitles(number)]
                if force or not is_fresh(outputs, _translate_inputs(paths, number)):
                    await run_translate(cfg, provider, number)
```

两处连带：
- `cli.py` 里那句 `if "script" in stages: provider = build_provider(...)` 要改成 `if {"script", "translate"} & set(stages):` —— translate 也要 provider。
- **translate 刻意不做多集并发**（script 有并发窗口，translate 不加）：第 1 集写完术语表、第 2 集才读到含第 1 集的版本，并发会让累积失去意义，而且两个任务会同时回写同一个文件。在插入处的注释里写明这一点。

另外 signals 是全局阶段（在纵向循环之前跑完所有集），而 translate 在纵向循环里 —— 所以实际执行顺序是 signals 先于 translate。这不违反任何依赖：**signals 不读对白文本，只看时间戳和字数**，跟语言无关。`STAGES` 里把 translate 排在 signals 之前，表达的是「逻辑上它紧跟 ingest」以及 `--only`/`--through` 的语义顺序。在 `STAGES` 那里加一条注释说明这个错位是已知且无害的。

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: 全绿

- [ ] **Step 7: 跑全量 + 确认现有项目不受影响**

Run: `uv run pytest tests/ -q`
Expected: 全绿

Run: `uv run tenmin run saijo --only translate && rtk git status`
Expected: 什么都不做（13 集全是 `source == "srt"`），`work/saijo/zh/` 不存在。

- [ ] **Step 8: 提交**

```bash
rtk git add src/tenmin/pipeline.py src/tenmin/cli.py tests/test_pipeline.py
rtk git commit -m "feat: translate 阶段接进管线，解说稿新鲜度挂上术语表"
```

---

### Task 13: script 阶段的术语表来源切到累积表

**Files:**
- Modify: `src/tenmin/pipeline.py`（`run_script` 读累积表并传下去）
- Modify: `src/tenmin/script/single.py`（`generate_script` 接受术语表入参）
- Modify: `src/tenmin/script/prompts/single_episode.md`（只改术语表那节的标题说明）
- Test: `tests/test_single.py`、`tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 9 的 `load_glossary` / `effective_glossary`
- Produces: `generate_script(cfg, track, report, provider, *, reporter=None, glossary=None)`（新增一个可选 kwarg；不传时退回 `cfg.glossary`，现有调用点零改动）

- [ ] **Step 1: 写失败的测试**

`tests/test_single.py` 追加：

```python
@pytest.mark.asyncio
async def test_generate_script_uses_the_supplied_glossary(tmp_path):
    """日语路径下，喂给解说稿的术语表是「日语原文 → 中文译法」，由翻译阶段攒出来的。"""
    cfg = _cfg(tmp_path)
    provider = _capturing_provider()

    await generate_script(
        cfg, _track(), _report(), provider, glossary={"リディア": "莉迪亚"}
    )

    assert "リディア → 莉迪亚" in provider.prompts[0]


@pytest.mark.asyncio
async def test_generate_script_falls_back_to_the_project_glossary(tmp_path):
    """不传时退回 project.yaml 里那份，现有的中文片源路径行为不变。"""
    cfg = _cfg(tmp_path)
    cfg.glossary = {"侍从": "管家"}
    provider = _capturing_provider()

    await generate_script(cfg, _track(), _report(), provider)

    assert "侍从 → 管家" in provider.prompts[0]
```

`tests/test_pipeline.py` 追加：

```python
@pytest.mark.asyncio
async def test_run_script_feeds_the_accumulated_glossary(tmp_path, monkeypatch):
    cfg, paths = _project_with_asr_dialogue(tmp_path, episode=11)
    _write_signals(paths, 11)  # 该文件已有的辅助；没有就照 _project_with_asr_dialogue 写一个
    paths.glossary.parent.mkdir(parents=True, exist_ok=True)
    paths.glossary.write_text(
        json.dumps({"リディア": "莉迪亚"}, ensure_ascii=False), encoding="utf-8"
    )
    seen: dict[str, object] = {}

    async def fake_generate(cfg_, track, report, provider, *, reporter=None, glossary=None):
        seen["glossary"] = glossary
        return _minimal_script(11), []

    monkeypatch.setattr("tenmin.pipeline.generate_script", fake_generate)

    await run_script(cfg, _exploding_provider(), 11)

    assert seen["glossary"] == {"リディア": "莉迪亚"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_single.py tests/test_pipeline.py -k "glossary" -v`
Expected: FAIL

- [ ] **Step 3: `generate_script` 接受术语表**

`src/tenmin/script/single.py`：签名末尾加

```python
    glossary: Mapping[str, str] | None = None,
```

在函数体里算实际用的那份：

```python
    # 日语路径下这份表是翻译阶段攒出来的「日语原文 → 中文译法」，由 pipeline 读盘后传
    # 进来。不传就退回 project.yaml 里手写的那份（中文片源路径，同语言的用词归一）。
    glossary_entries = dict(glossary) if glossary is not None else dict(cfg.glossary)
```

把原来渲染术语表块的地方改成用 `glossary_entries`。首轮与返工轮共用同一份（返工轮摘掉的只有 few-shot 范例）。

- [ ] **Step 4: `run_script` 读盘并传下去**

`src/tenmin/pipeline.py` 的 `run_script`，在调 `generate_script` 之前：

```python
    glossary = effective_glossary(load_glossary(Paths(cfg.root).glossary), cfg.glossary)
```

并把它作为 `glossary=glossary` 传给 `generate_script`。

（`effective_glossary` 让手写的那份盖住机器的选择 —— 手写表是纠错入口。空表时结果是空字典，`build_glossary_block` 已经能优雅降级。）

- [ ] **Step 5: 改 prompt 的术语表节说明**

`src/tenmin/script/prompts/single_episode.md`：只改术语表那一节的标题下那句说明，让它对两种语言都成立：

```markdown
## 术语表（专有名词必须按这个写法）

左边是对白里出现的原文写法，右边是解说稿里必须用的写法。对白是日语时，左边就是日语原文。
```

**`{{glossary_block}}` 占位符本身、以及「人物关系与动作方向必须以对白原文为准」「谁称呼谁、谁对谁用敬语」「画面描述与留白金句全部来自上面那份对白轨」这几处措辞一个字都不改** —— 它们说的是「对白轨」，跳语言后依然成立，而「敬语」那条要求恰好是在日语下才真正生效（中文译文里那个信息已经没了）。

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_single.py tests/test_pipeline.py -q`
Expected: 全绿

- [ ] **Step 7: 跑全量**

Run: `uv run pytest tests/ -q`
Expected: 全绿

- [ ] **Step 8: 提交**

```bash
rtk git add src/tenmin/pipeline.py src/tenmin/script/single.py src/tenmin/script/prompts/single_episode.md tests/test_single.py tests/test_pipeline.py
rtk git commit -m "feat: 解说稿的术语表改读跨集累积表"
```

---

### Task 14: 端到端验收

这一任务不写代码，只验收。它要回答的是「这条路真的走通了吗」，而前面 13 个任务的单测
全部是在 fake 后面跑的。

**Files:**
- Modify: `README.md`（现有措辞「**还没有 ASR**，所以只吃自带字幕的片源」要更正）
- Modify: `docs/config-reference.md`（补 `asr` 节）
- Test: 手动跑

**Interfaces:**
- Consumes: 前 13 个任务的全部产物
- Produces: 一份能播的成片 + 更新后的文档

- [ ] **Step 1: 装 extra**

Run: `uv sync --extra asr`
Expected: 成功。若因公司代理报 SSL 证书错，按 AGENTS.md 里那条把 Zscaler CA 追加进 certifi 的 cacert.pem。

- [ ] **Step 2: 拿生肉片源登记一集并只跑 ingest**

```bash
uv run tenmin run <slug> --episode 11 \
  --video "/Users/portz/Downloads/[ANi] 我是不才惡女 - 11 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4" \
  --only ingest
```

Expected:
- 终端打出「没有字幕轨，将对音轨做语音转写」那一行
- 约 3 分钟后打出「转写完成，N 条对白」
- `work/<slug>/srt/E11.asr.srt` 存在且能直接用播放器加载
- `work/<slug>/01_dialogue/E11.dialogue.json` 的 `source` 是 `"asr"`、`duration` 约 1430 秒
- 对白文本是**日语原文**，没有被繁转简改字（随手找一个含 `製`/`實`/`發` 这类字的句子看一眼）

- [ ] **Step 3: 手填 OP/ED 区间**

拖进播放器看一眼 OP 与 ED 的起止秒数，写进 `work/<slug>/project.yaml` 第 11 集的 `op_range` / `ed_range`。

这一步在生肉路径下是**必需的**，不是可选优化：片尾识别的全部判据都是「字幕组打在屏幕上的文字」特征（`©`、`作曲：`、`製作委員会`、人名罗列），而语音转写一条都不会产出 —— 它把主题曲**歌词**转成了正常对白。不填的话，signals 会拿歌词去算静音间隙和语速，解说稿会拿歌词当剧情素材，成片可能在片头曲上插解说。

Run: `uv run tenmin inspect <slug> --episode 11`
Expected: 报告里显示 OP/ED 区间来自手填（那一级优先于启发式推断）。

- [ ] **Step 4: 跑翻译**

Run: `uv run tenmin run <slug> --episode 11 --only translate`
Expected:
- `work/<slug>/zh/E11.zh.json` 的条数等于对白轨里 `kind == "dialogue"` 的行数（对齐校验是这条路上最该盯的东西）
- `work/<slug>/out/E11.zh.srt` 能拖进播放器，跟画面对得上
- `work/<slug>/zh/glossary.json` 里有几个人名，日语原文 → 中文

- [ ] **Step 5: 跑到出片**

Run: `uv run tenmin run <slug> --episode 11`
Expected: `work/<slug>/07_render/E11.mp4` 出来且能播。重点看三处：解说没插在 OP/ED 上；`out/E11.解说方案.md` 里的人名跟 `out/E11.zh.srt` 里一致；解说内容没把主题曲歌词当剧情。

- [ ] **Step 6: 跑一次全量回归，确认中文片源那条路没坏**

```bash
uv run pytest tests/ -q
uv run tenmin run saijo --force
rtk git status
```

Expected: 测试全绿；saijo 的 13 集产物除了 `01_dialogue/*.json` 里新增的 `source: "srt"` 键之外**逐字节不变**（`zh/` 目录不存在、`out/*.zh.srt` 不存在）。

- [ ] **Step 7: 更正文档**

`README.md`：把「**还没有 ASR**，所以只吃自带字幕的片源」改成描述现在的三条路（手传字幕 / 视频内嵌软字幕轨 / 语音转写），并写明生肉路径要 `uv sync --extra asr`、要手填 OP/ED、会额外产出 `out/E{NN}.zh.srt`。

`docs/config-reference.md`：补 `asr` 节的两个字段（`model`、`language`），照该文件既有条目的格式。

- [ ] **Step 8: 提交**

```bash
rtk git add README.md docs/config-reference.md work/
rtk git commit -m "docs: 更新生肉支持的说明与 asr 配置项"
```

---

## 自查记录

**Spec 覆盖**：spec 的 10 个决定逐条对到任务 —— 决定 1（翻译只给人看 + 独立中文 SRT）→ Task 10/12；决定 2（独立 translate 阶段）→ Task 12；决定 3（三岔判据 + 告知）→ Task 4；决定 4（optional extra + 只做 mlx-whisper + 延迟 import + 可读错误）→ Task 1/3；决定 5（`zh/` 不占编号）→ Task 7；决定 6（两个产物的确切路径）→ Task 7；决定 7（带 id 的结构化输出 + 对齐校验 + 回灌重试 + 不分批 + 术语表进 prompt）→ Task 8/11；决定 8（术语表自动累积 + script 新鲜度挂它）→ Task 9/12/13；决定 9（只喂日语原文）→ Task 13；决定 10（OP/ED 手填）→ Task 14 Step 3。spec 的测试策略逐条落在 Task 1~12 的测试里，第四个 marker `asr` 在 Task 1。spec 里「不做」的六项在本计划中确实一处都没做。

**类型一致性**：`SubtitleSource.kind`、`DialogueTrack.source`、`build_track(source=)` 三处都是 `Literal["srt","asr"]`；`TranslatedLine.id` 在 Task 7 定义、Task 10/11/12 一致地当作「`track.lines` 里的 1-based 位置」；`resolve_subtitle_source` 的配置参数名统一为 `asr_config`（避免跟导入的 `asr` 模块撞名，Task 4 Step 3 已标出测试要跟着改的地方）；`complete_with_schema_repair` 在 Task 8 去掉下划线、Task 11 按新名字用。

**两处与 spec 的偏离**已在文档开头单列，实现时以本计划为准。



