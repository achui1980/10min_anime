# tenmin v2 成片渲染 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 v1 产出的 `script.json` 加上源视频，自动渲染成可直接上传的 mp4（切片拼接 + Edge-TTS 配音 + 原声 ducking + 烧硬字幕）。

**Architecture:** 沿用 v1 的「阶段读上游文件、写自己的文件、靠 mtime 跳过」编排模式，在 `STAGES` 末尾追加 `voice / timeline / audio / render` 四个阶段。新增 `src/tenmin/render/` 包：`chunks.py`、`timeline.py`、`subtitles.py` 是纯函数（离线严格断言），`tts.py` 用 Protocol 把 Edge-TTS 藏在接口后面（测试用 `FakeTTSEngine`，不联网），`ffmpeg.py` 封装子进程，`audio.py`/`video.py` 只负责拼命令行（测试断言命令行参数，不真跑 ffmpeg）。视频只编码一次：`render` 阶段单次 ffmpeg 调用同时做 trim + concat + 烧字幕 + 挂音轨。

**Tech Stack:** Python 3.12+ / uv / pydantic v2 / typer / edge-tts / ffmpeg（必须编入 libass）/ pytest + pytest-asyncio（`asyncio_mode = "auto"`）/ ruff（line-length 100）

**核心算法（时间轴重算）：** 每个 beat 的音频总时长 =（各 chunk 真实 TTS 时长之和 + 各 hold.duration 之和）；`ratio = 音频总时长 / 本 beat 所有 clip 的原始总时长`；每个 clip 的 `start` 不动，时长 × ratio。实测 ratio ≈ 0.4（素材远多于旁白）。ratio > 1 时延长撞到源片尾就钳制 + warning。

---

## 明确不做的事（读代码时别自己加戏）

| 不做 | 原因 |
| --- | --- |
| 音效（`script.json` 的 `sfx` 字段） | 需要外部音效素材库，还有版权问题。字段**保留但本版不消费**，任何任务都不要读它 |
| BGM | 同上，v1 设计里也从未提及 |
| 原片对白字幕 | 解说视频烧的是**旁白**字幕（观众跟着解说读），不是原片对白 |
| 精确金句同步 | hold 期间播的是「那一刻恰好在放的原声」，不是 quote 本身的原声。要对上就得把画面跳到 quote 的源时间戳，复杂度再上一个量级且会打断 clip 连续性。本版 hold 是**节奏装置**，不是精确对轨 |
| 视觉识别选镜头 | v5 的事（PySceneDetect + CLIP） |
| GUI | v4 的事 |

其他已知局限，实现时不要试图「顺手修好」：

1. `clip.end` 会被覆盖——LLM 给的结束时间只用来算 ratio，不直接采用。`clip.start` 是硬的（v1 `validate.py` 做过锚点校正，有证据）。
2. 旁白字幕的切分完全跟随 TTS chunk 边界：一个 chunk = 一条字幕，字幕起止 = 该 chunk 在总时间轴上的起止。hold 期间没有字幕（那里没旁白，画面干净 + 原声顶上）。chunk 长则字幕停留久，本版**不做二次切分**。
3. 只支持单集。`mode: season` 在 v1 里已经直接报错，v2 不改这一点。
4. **不要给 ffmpeg 加 `-c copy` 想省编码时间。** 源片是 1080p HEVC-10bit，任意时间点切割关键帧对不齐，`-c copy` 出来的片子开头必然是花屏或黑帧。必须重编码，这是本方案接受的成本。

---

## File Structure

**新建：**

| 文件 | 职责 |
| --- | --- |
| `src/tenmin/render/__init__.py` | 空，不导出任何东西（跟 v1 其他子包一致） |
| `src/tenmin/render/ffmpeg.py` | 子进程封装：`run` / `probe_duration` / `has_filter` / `has_encoder` / `preflight` |
| `src/tenmin/render/chunks.py` | 纯函数：旁白切句 + hold 定位到句边界 |
| `src/tenmin/render/tts.py` | `TTSEngine` Protocol + `EdgeTTSEngine` + 重试 + `synthesize_track` |
| `src/tenmin/render/timeline.py` | 纯函数：时间轴重算，产出 `Timeline` |
| `src/tenmin/render/subtitles.py` | 纯函数：`Timeline.subtitles` → ASS 文本 |
| `src/tenmin/render/audio.py` | 拼混音 ffmpeg 命令行 + 执行 |
| `src/tenmin/render/video.py` | 拼渲染 ffmpeg 命令行 + 执行 |

**修改：**

| 文件 | 改动 |
| --- | --- |
| `pyproject.toml` | 加 `edge-tts` 依赖、加 `render` pytest marker |
| `src/tenmin/models.py` | 追加 `VoiceChunk` / `VoiceTrack` / `TimelineSegment` / `SubtitleCue` / `Timeline` |
| `src/tenmin/config.py` | `EpisodeConfig.video`、新增 `RenderConfig`、`ProjectConfig.render`、`video_path()` |
| `src/tenmin/pipeline.py` | `Paths` 新增 6 个路径、4 个 `run_*` 函数、`STAGES` 扩到 8 个、`run_pipeline` 加 `tts_engine` 参数 |
| `src/tenmin/cli.py` | 构造 `TTSEngine`、`PROJECT_TEMPLATE` 加 `video` 与 `render`、打印成品路径 |
| `tests/fakes.py` | 追加 `FakeTTSEngine` |
| `README.md` | 阶段表补 4 行、用法补 `--from voice` |

**新建测试（沿用扁平命名 `tests/test_<module>.py`）：** `test_render_ffmpeg.py`、`test_render_chunks.py`、`test_render_tts.py`、`test_render_timeline.py`、`test_render_subtitles.py`、`test_render_audio.py`、`test_render_video.py`、`test_render_golden.py`、`test_render_e2e.py`。另外往 `test_models.py`、`test_config.py`、`test_pipeline.py`、`test_cli.py` 里追加用例。

---

### Task 1: 依赖与骨架

**Files:**
- Modify: `pyproject.toml`
- Create: `src/tenmin/render/__init__.py`

- [ ] **Step 1: 加 edge-tts 依赖**

在 `pyproject.toml` 的 `[project] dependencies` 列表末尾（`"httpx>=0.27",` 之后）加一行：

```toml
    "edge-tts>=6.1",
```

- [ ] **Step 2: 加 render marker**

在 `pyproject.toml` 的 `[tool.pytest.ini_options] markers` 列表末尾（`generalize:` 那行之后）加一行：

```toml
    "render: 需要真实视频文件与编入 libass 的 ffmpeg，默认跳过（跑法：uv run pytest -m render）",
```

- [ ] **Step 3: 创建空包**

创建 `src/tenmin/render/__init__.py`，内容为空文件（0 字节）。v1 的 `ingest/`、`signals/`、`script/`、`docgen/` 的 `__init__.py` 都是空的，不要在这里写 re-export。

- [ ] **Step 4: 同步依赖并确认现有测试全绿**

```bash
uv sync
uv run pytest -q
```

Expected: 全部 PASS（v2 还没动任何现有逻辑）。同时确认 edge-tts 装上了：

```bash
uv run python -c "import edge_tts; print(edge_tts.__name__)"
```

Expected: 输出 `edge_tts`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/tenmin/render/__init__.py
git commit -m "chore: add edge-tts dependency and render package skeleton"
```

---

### Task 2: 数据模型

**Files:**
- Modify: `src/tenmin/models.py`（在文件末尾、`LLMScript` 之后追加）
- Test: `tests/test_models.py`

- [ ] **Step 1: 写失败的测试**

在 `tests/test_models.py` 末尾追加（文件顶部的 import 里补上这些名字）：

```python
from tenmin.models import (
    SubtitleCue,
    Timeline,
    TimelineSegment,
    VoiceChunk,
    VoiceTrack,
)


def test_voice_chunk_defaults():
    chunk = VoiceChunk(beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=3.0)
    assert chunk.hold_after == 0.0


def test_voice_track_holds_chunks():
    chunk = VoiceChunk(
        beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=3.0, hold_after=1.5
    )
    track = VoiceTrack(episode=2, chunks=[chunk], total_seconds=4.5)
    assert track.chunks[0].hold_after == 1.5
    assert track.total_seconds == 4.5


def test_timeline_segment_fields():
    seg = TimelineSegment(
        beat_id="b1", source_start=100.0, source_end=120.0, timeline_start=0.0, timeline_end=20.0
    )
    assert seg.source_end - seg.source_start == 20.0


def test_timeline_defaults_are_empty():
    timeline = Timeline(episode=2, total_seconds=0.0)
    assert timeline.segments == []
    assert timeline.subtitles == []
    assert timeline.narration_offsets == []


def test_timeline_roundtrips_json():
    timeline = Timeline(
        episode=2,
        segments=[
            TimelineSegment(
                beat_id="b1",
                source_start=100.0,
                source_end=120.0,
                timeline_start=0.0,
                timeline_end=20.0,
            )
        ],
        subtitles=[SubtitleCue(start=0.0, end=8.0, text="第一句。")],
        narration_offsets=[0.0],
        total_seconds=20.0,
    )
    restored = Timeline.model_validate_json(timeline.model_dump_json())
    assert restored == timeline
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_models.py -v -k "voice or timeline"`
Expected: FAIL，`ImportError: cannot import name 'VoiceChunk' from 'tenmin.models'`

- [ ] **Step 3: 写实现**

在 `src/tenmin/models.py` 末尾追加：

```python
# --- v2 渲染阶段的模型 ---


class VoiceChunk(BaseModel):
    """一段独立合成的旁白。duration 是 TTS 出来后实测的，不是估算。"""

    beat_id: str
    index: int
    text: str
    path: str
    duration: float
    hold_after: float = 0.0


class VoiceTrack(BaseModel):
    """一集的全部旁白 chunk。total_seconds 含 hold 静音。"""

    episode: int
    chunks: list[VoiceChunk] = Field(default_factory=list)
    total_seconds: float = 0.0


class TimelineSegment(BaseModel):
    """一段画面。source_* 是原片坐标，timeline_* 是成片坐标。"""

    beat_id: str
    source_start: float
    source_end: float
    timeline_start: float
    timeline_end: float

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start


class SubtitleCue(BaseModel):
    """一条烧进画面的旁白字幕。坐标是成片坐标。"""

    start: float
    end: float
    text: str


class Timeline(BaseModel):
    """v2 的人工编辑面。改完它跑 --from audio 就能重出片。"""

    episode: int
    segments: list[TimelineSegment] = Field(default_factory=list)
    subtitles: list[SubtitleCue] = Field(default_factory=list)
    narration_offsets: list[float] = Field(default_factory=list)
    total_seconds: float = 0.0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_models.py -v -k "voice or timeline"`
Expected: 5 个用例全 PASS

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/models.py tests/test_models.py
git commit -m "feat(models): add v2 voice and timeline models"
```

---

### Task 3: 配置扩展

**Files:**
- Modify: `src/tenmin/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写失败的测试**

在 `tests/test_config.py` 末尾追加（顶部 import 补 `EpisodeConfig`、`RenderConfig`）：

```python
from tenmin.config import EpisodeConfig, ProjectConfig, RenderConfig


def test_render_config_defaults():
    cfg = RenderConfig()
    assert cfg.voice == "zh-CN-YunxiNeural"
    assert cfg.rate == "+0%"
    assert cfg.video_encoder == "libx264"
    assert cfg.duck_db == -12.0
    assert cfg.font_size == 48


def test_project_config_has_render_defaults():
    cfg = ProjectConfig(show="剧名", slug="slug")
    assert cfg.render.voice == "zh-CN-YunxiNeural"


def test_project_config_reads_render_block():
    cfg = ProjectConfig.model_validate(
        {
            "show": "剧名",
            "slug": "slug",
            "render": {"voice": "zh-CN-XiaoxiaoNeural", "duck_db": -9.0},
        }
    )
    assert cfg.render.voice == "zh-CN-XiaoxiaoNeural"
    assert cfg.render.duck_db == -9.0
    # 没写的字段仍取默认值
    assert cfg.render.video_encoder == "libx264"


def test_video_path_resolves_relative_to_root(tmp_path):
    cfg = ProjectConfig.model_validate(
        {
            "show": "剧名",
            "slug": "slug",
            "episodes": [{"number": 2, "srt": "srt/E02.srt", "video": "video/E02.mkv"}],
        }
    ).bind_root(tmp_path)
    assert cfg.video_path(cfg.episodes[0]) == tmp_path.resolve() / "video/E02.mkv"


def test_video_path_passes_absolute_through(tmp_path):
    absolute = tmp_path / "elsewhere" / "E02.mkv"
    cfg = ProjectConfig.model_validate(
        {
            "show": "剧名",
            "slug": "slug",
            "episodes": [{"number": 2, "srt": "srt/E02.srt", "video": str(absolute)}],
        }
    ).bind_root(tmp_path)
    assert cfg.video_path(cfg.episodes[0]) == absolute


def test_video_path_without_video_raises():
    cfg = ProjectConfig.model_validate(
        {"show": "剧名", "slug": "slug", "episodes": [{"number": 2, "srt": "srt/E02.srt"}]}
    )
    with pytest.raises(ValueError) as exc:
        cfg.video_path(cfg.episodes[0])
    assert "第 2 集没有配置 video" in str(exc.value)


def test_episode_config_video_defaults_to_none():
    episode = EpisodeConfig(number=2, srt=Path("srt/E02.srt"))
    assert episode.video is None
```

`tests/test_config.py` 顶部若还没有 `import pytest` 和 `from pathlib import Path`，一并补上。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py -v -k "render or video"`
Expected: FAIL，`ImportError: cannot import name 'RenderConfig' from 'tenmin.config'`

- [ ] **Step 3: 写实现**

在 `src/tenmin/config.py` 里，`EpisodeConfig` 加 `video` 字段：

```python
class EpisodeConfig(BaseModel):
    number: int
    srt: Path
    video: Path | None = None
    op_range: tuple[float, float] | None = None
    ed_range: tuple[float, float] | None = None
```

在 `LLMConfig` 之后加 `RenderConfig`：

```python
class RenderConfig(BaseModel):
    """v2 渲染参数。voice 与 rate 直接喂 Edge-TTS。"""

    voice: str = "zh-CN-YunxiNeural"
    rate: str = "+0%"
    video_encoder: str = "libx264"
    duck_db: float = -12.0
    font_size: int = 48
```

在 `ProjectConfig` 的 `llm` 字段后面加：

```python
    render: RenderConfig = Field(default_factory=RenderConfig)
```

在 `ProjectConfig.srt_path` 后面加：

```python
    def video_path(self, episode: EpisodeConfig) -> Path:
        """源视频路径。相对路径按 project.yaml 所在目录解析。"""
        if episode.video is None:
            raise ValueError(
                f"第 {episode.number} 集没有配置 video，"
                "请在 project.yaml 的 episodes 里补上源视频路径"
            )
        if episode.video.is_absolute():
            return episode.video
        return self._root / episode.video
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v -k "render or video"`
Expected: 7 个用例全 PASS

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/config.py tests/test_config.py
git commit -m "feat(config): add render section and per-episode video path"
```

---

### Task 4: ffmpeg 纯解析函数

`ffmpeg -filters` / `-encoders` 的输出是「几列标志 + 名字 + 描述」，先把解析逻辑做成纯函数单独测，子进程留到下一个 Task。

**Files:**
- Create: `src/tenmin/render/ffmpeg.py`
- Test: `tests/test_render_ffmpeg.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_render_ffmpeg.py`：

```python
import pytest

from tenmin.render.ffmpeg import FFmpegError, parse_names, tail

FILTERS_SAMPLE = """Filters:
  T.. ass               V->V       Render ASS subtitles onto input video.
  ... concat            N->N       Concatenate audio and video streams.
  ..C subtitles         V->V       Render text subtitles onto input video.
  TSC volume            A->A       Change input volume.
"""

ENCODERS_SAMPLE = """Encoders:
 V..... libx264              libx264 H.264 / AVC
 V..... h264_videotoolbox    VideoToolbox H.264 Encoder
 A..... aac                  AAC (Advanced Audio Coding)
"""


def test_parse_names_from_filters():
    names = parse_names(FILTERS_SAMPLE)
    assert "subtitles" in names
    assert "concat" in names
    assert "volume" in names
    # 表头那行不该被当成名字
    assert "Filters:" not in names


def test_parse_names_from_encoders():
    names = parse_names(ENCODERS_SAMPLE)
    assert names == {"libx264", "h264_videotoolbox", "aac"}


def test_parse_names_on_empty_text():
    assert parse_names("") == set()


def test_tail_keeps_last_lines():
    text = "\n".join(str(i) for i in range(100))
    assert tail(text, lines=3) == "97\n98\n99"


def test_tail_shorter_than_limit_returns_all():
    assert tail("a\nb", lines=30) == "a\nb"


def test_tail_strips_trailing_blank_lines():
    assert tail("a\nb\n\n\n", lines=2) == "a\nb"


def test_ffmpeg_error_is_runtime_error():
    assert issubclass(FFmpegError, RuntimeError)
    with pytest.raises(RuntimeError):
        raise FFmpegError("boom")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_render_ffmpeg.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.render.ffmpeg'`

- [ ] **Step 3: 写实现**

创建 `src/tenmin/render/ffmpeg.py`：

```python
"""ffmpeg / ffprobe 子进程封装。解析部分是纯函数，单独测。"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
STDERR_TAIL_LINES = 30

# ffmpeg -filters / -encoders 每行形如 "  T.. ass  V->V  描述"，
# 标志列只由大写字母和点组成，名字是紧跟其后的第一个 token。
_NAME_LINE = re.compile(r"^\s*[A-Z.]{3,6}\s+(\S+)\s")


class FFmpegError(RuntimeError):
    """ffmpeg 非零退出。消息里必须带 stderr 尾部，否则等于没报错。"""


def parse_names(text: str) -> set[str]:
    """从 -filters / -encoders 的输出里抽出可用名字。"""
    return {m.group(1) for m in (_NAME_LINE.match(line) for line in text.splitlines()) if m}


def tail(text: str, lines: int = STDERR_TAIL_LINES) -> str:
    """取末尾若干行。ffmpeg 的真实错误永远在 stderr 尾部。"""
    stripped = text.rstrip("\n")
    if not stripped:
        return ""
    return "\n".join(stripped.splitlines()[-lines:])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_render_ffmpeg.py -v`
Expected: 7 个用例全 PASS

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/ffmpeg.py tests/test_render_ffmpeg.py
git commit -m "feat(render): add ffmpeg output parsing helpers"
```

---

### Task 5: ffmpeg 子进程层与前置检查

**Files:**
- Modify: `src/tenmin/render/ffmpeg.py`
- Test: `tests/test_render_ffmpeg.py`

- [ ] **Step 1: 写失败的测试**

在 `tests/test_render_ffmpeg.py` 末尾追加（顶部 import 改成把下面这些名字一起导进来）：

```python
from tenmin.render.ffmpeg import (
    FFmpegError,
    has_encoder,
    has_filter,
    parse_names,
    preflight,
    probe_duration,
    run,
    tail,
)


class FakeCompleted:
    """替代 subprocess.CompletedProcess，只带测试关心的三个字段。"""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_returns_stderr_on_success(monkeypatch):
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return FakeCompleted(stdout="", stderr="frame= 100")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    assert run(["-i", "a.mkv"]) == "frame= 100"
    assert seen["args"] == ["ffmpeg", "-i", "a.mkv"]


def test_run_raises_with_stderr_tail(monkeypatch):
    noise = "\n".join(f"line {i}" for i in range(50))

    def fake_run(args, **kwargs):
        return FakeCompleted(returncode=1, stderr=noise + "\nInvalid argument")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    with pytest.raises(FFmpegError) as exc:
        run(["-i", "a.mkv"])
    message = str(exc.value)
    assert "Invalid argument" in message
    assert "line 49" in message
    # 只留末尾 30 行，开头的噪声不该出现
    assert "line 0" not in message


def test_probe_duration_parses_csv(monkeypatch):
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return FakeCompleted(stdout="1425.501000\n")

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    assert probe_duration(Path("a.mkv")) == pytest.approx(1425.501)
    assert seen["args"][0] == "ffprobe"
    assert seen["args"][-1] == "a.mkv"


def test_probe_duration_raises_on_unparsable(monkeypatch):
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.run",
        lambda args, **kwargs: FakeCompleted(stdout="N/A\n"),
    )
    with pytest.raises(FFmpegError) as exc:
        probe_duration(Path("a.mkv"))
    assert "a.mkv" in str(exc.value)


def test_has_filter_and_has_encoder(monkeypatch):
    # available_filters / available_encoders 带 lru_cache，用例之间必须清缓存
    from tenmin.render import ffmpeg as ffmpeg_mod

    def fake_run(args, **kwargs):
        if "-filters" in args:
            return FakeCompleted(stdout=FILTERS_SAMPLE)
        return FakeCompleted(stdout=ENCODERS_SAMPLE)

    monkeypatch.setattr("tenmin.render.ffmpeg.subprocess.run", fake_run)
    ffmpeg_mod.available_filters.cache_clear()
    ffmpeg_mod.available_encoders.cache_clear()
    assert has_filter("subtitles") is True
    assert has_filter("nosuchfilter") is False
    assert has_encoder("libx264") is True
    assert has_encoder("libx265") is False
    ffmpeg_mod.available_filters.cache_clear()
    ffmpeg_mod.available_encoders.cache_clear()


def test_preflight_reports_missing_libass(monkeypatch, tmp_path):
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name: False)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", lambda path: 100.0)
    with pytest.raises(RuntimeError) as exc:
        preflight(video, "libx264")
    assert "libass" in str(exc.value)


def test_preflight_reports_missing_encoder(monkeypatch, tmp_path):
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name: False)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", lambda path: 100.0)
    with pytest.raises(RuntimeError) as exc:
        preflight(video, "h264_videotoolbox")
    assert "h264_videotoolbox" in str(exc.value)


def test_preflight_reports_missing_video(monkeypatch, tmp_path):
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name: True)
    with pytest.raises(FileNotFoundError) as exc:
        preflight(tmp_path / "missing.mkv", "libx264")
    assert "missing.mkv" in str(exc.value)


def test_preflight_returns_source_duration(monkeypatch, tmp_path):
    video = tmp_path / "E02.mkv"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tenmin.render.ffmpeg.has_filter", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.has_encoder", lambda name: True)
    monkeypatch.setattr("tenmin.render.ffmpeg.probe_duration", lambda path: 1425.5)
    assert preflight(video, "libx264") == pytest.approx(1425.5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_render_ffmpeg.py -v`
Expected: FAIL，`ImportError: cannot import name 'run' from 'tenmin.render.ffmpeg'`

- [ ] **Step 3: 写实现**

在 `src/tenmin/render/ffmpeg.py` 末尾追加：

```python
def run(args: list[str]) -> str:
    """跑 ffmpeg，返回 stderr（ffmpeg 的进度与日志都在 stderr）。"""
    completed = subprocess.run([FFMPEG, *args], capture_output=True, text=True)
    if completed.returncode != 0:
        raise FFmpegError(
            f"ffmpeg 退出码 {completed.returncode}，命令：\n"
            f"{FFMPEG} {' '.join(args)}\n\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(completed.stderr)}"
        )
    return completed.stderr


def probe_duration(path: Path) -> float:
    """用 ffprobe 读时长（秒）。"""
    args = [
        FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(path),
    ]
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        raise FFmpegError(
            f"ffprobe 读不出 {path} 的时长，退出码 {completed.returncode}：\n"
            f"{tail(completed.stderr)}"
        )
    try:
        return float(completed.stdout.strip())
    except ValueError as error:
        raise FFmpegError(
            f"ffprobe 读不出 {path} 的时长，输出是 {completed.stdout.strip()!r}"
        ) from error


@lru_cache(maxsize=1)
def available_filters() -> set[str]:
    completed = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True)
    return parse_names(completed.stdout)


@lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    completed = subprocess.run(
        [FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return parse_names(completed.stdout)


def has_filter(name: str) -> bool:
    return name in available_filters()


def has_encoder(name: str) -> bool:
    return name in available_encoders()


def preflight(video: Path, video_encoder: str) -> float:
    """开跑前一次性检查，返回源片时长。任一项不满足立刻抛错。

    渲染动辄几分钟，绝不能跑完 TTS 才在最后一步炸掉。
    """
    if not has_filter("subtitles"):
        raise RuntimeError(
            "你的 ffmpeg 没编 libass，subtitles 滤镜不可用，烧不了字幕。\n"
            "请重装：brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass"
        )
    if not has_encoder(video_encoder):
        raise RuntimeError(
            f"你的 ffmpeg 没有编码器 {video_encoder}，请改 project.yaml 的 render.video_encoder，"
            "或重装 ffmpeg"
        )
    if not Path(video).is_file():
        raise FileNotFoundError(f"找不到源视频 {video}，请检查 project.yaml 的 episodes[].video")
    return probe_duration(Path(video))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_render_ffmpeg.py -v`
Expected: 全部 PASS（Task 4 的 7 个 + 本 Task 的 8 个）

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/ffmpeg.py tests/test_render_ffmpeg.py
git commit -m "feat(render): add ffmpeg subprocess layer and preflight checks"
```

---

### Task 6: 旁白切句与 hold 定位（纯函数）

`hold.at` 是 LLM 按 4.5 字/秒估的，真实 TTS 一定会漂。做法：把 `beat.narration` 按句末标点切句，按累计估算时长找离 `hold.at` 最近的句边界，每个 chunk 单独 TTS，chunk 之间插入 `hold.duration` 长的静音。

**Files:**
- Create: `src/tenmin/render/chunks.py`
- Test: `tests/test_render_chunks.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_render_chunks.py`：

```python
import pytest

from tenmin.models import AudioDirection, Beat, Hold
from tenmin.render.chunks import (
    assign_holds,
    plan_chunks,
    sentence_offsets,
    split_sentences,
)


def test_split_sentences_keeps_punctuation():
    assert split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]


def test_split_sentences_keeps_tail_without_punctuation():
    assert split_sentences("第一句。没有句号的尾巴") == ["第一句。", "没有句号的尾巴"]


def test_split_sentences_merges_repeated_punctuation():
    assert split_sentences("真的吗？！接着说。") == ["真的吗？！", "接着说。"]


def test_split_sentences_ignores_blank_input():
    assert split_sentences("   \n  ") == []


def test_sentence_offsets_accumulate_estimated_duration():
    # 每句 4 个字，4.5 字/秒 → 每句 0.888… 秒，offsets 是「该句结束时刻」
    offsets = sentence_offsets(["第一句。", "第二句。", "第三句。"])
    assert offsets == pytest.approx([0.8889, 1.7778, 2.6667], abs=1e-4)


def test_assign_holds_picks_nearest_boundary():
    sentences = ["第一句。", "第二句。", "第三句。"]
    holds = [Hold(at=2.0, duration=2.0, quote="金句")]
    # offsets 是 [0.889, 1.778, 2.667]；2.0 离 1.778 最近 → 索引 1
    assert assign_holds(sentences, holds) == {1: 2.0}


def test_assign_holds_sums_holds_on_same_boundary():
    sentences = ["第一句。", "第二句。"]
    holds = [Hold(at=1.7, duration=2.0, quote="甲"), Hold(at=1.9, duration=1.0, quote="乙")]
    assert assign_holds(sentences, holds) == {1: 3.0}


def test_assign_holds_on_empty_sentences_returns_empty():
    assert assign_holds([], [Hold(at=1.0, duration=1.0, quote="金句")]) == {}


def test_plan_chunks_merges_sentences_between_holds():
    beat = Beat(
        id="b1",
        label="Hook",
        role="hook",
        narration="第一句。第二句。第三句。",
        audio=AudioDirection(holds=[Hold(at=2.0, duration=2.0, quote="金句")]),
    )
    assert plan_chunks(beat) == [("第一句。第二句。", 2.0), ("第三句。", 0.0)]


def test_plan_chunks_without_holds_is_one_chunk():
    beat = Beat(id="b1", label="Hook", role="hook", narration="第一句。第二句。")
    assert plan_chunks(beat) == [("第一句。第二句。", 0.0)]


def test_plan_chunks_on_empty_narration_returns_empty():
    beat = Beat(id="b1", label="Hook", role="hook", narration="")
    assert plan_chunks(beat) == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_render_chunks.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.render.chunks'`

- [ ] **Step 3: 写实现**

创建 `src/tenmin/render/chunks.py`：

```python
"""旁白切句与 hold 定位。全是纯函数。

每个 chunk 独立 TTS：真实时长直接测得，不依赖 Edge-TTS 的 word boundary；
单个 chunk 失败只需补它一个。
"""

from __future__ import annotations

from tenmin.models import Beat, Hold
from tenmin.script.budget import SPEECH_RATE_CPS, narration_chars

# 一句 = 若干非句末字符 + 一串句末标点；末尾没标点的残句单独成句。
_SENTENCE_ENDINGS = "。！？!?"


def split_sentences(text: str) -> list[str]:
    """按句末标点切句，标点跟着前一句。连续标点算同一句末。"""
    sentences: list[str] = []
    buffer: list[str] = []
    pending_end = False
    for char in text.strip():
        if char == "\n":
            continue
        if char in _SENTENCE_ENDINGS:
            buffer.append(char)
            pending_end = True
            continue
        if pending_end:
            sentences.append("".join(buffer))
            buffer = []
            pending_end = False
        buffer.append(char)
    if buffer:
        sentences.append("".join(buffer))
    return [s for s in (item.strip() for item in sentences) if s]


def sentence_offsets(sentences: list[str]) -> list[float]:
    """每句「结束时刻」的估算值（秒），用 v1 的 4.5 字/秒。"""
    offsets: list[float] = []
    cursor = 0.0
    for sentence in sentences:
        cursor += narration_chars(sentence) / SPEECH_RATE_CPS
        offsets.append(cursor)
    return offsets


def assign_holds(sentences: list[str], holds: list[Hold]) -> dict[int, float]:
    """把每个 hold 落到最近的句边界上。返回 {句索引: 该句之后的静音秒数}。"""
    if not sentences:
        return {}
    offsets = sentence_offsets(sentences)
    assigned: dict[int, float] = {}
    for hold in holds:
        # 平手取靠前的边界：min 遇到相等的 key 保留第一个
        index = min(range(len(offsets)), key=lambda i: abs(offsets[i] - hold.at))
        assigned[index] = assigned.get(index, 0.0) + hold.duration
    return assigned


def plan_chunks(beat: Beat) -> list[tuple[str, float]]:
    """把一个 beat 切成 [(要合成的文本, 该 chunk 之后的静音秒数)]。"""
    sentences = split_sentences(beat.narration)
    if not sentences:
        return []
    hold_after = assign_holds(sentences, beat.audio.holds)
    chunks: list[tuple[str, float]] = []
    buffer: list[str] = []
    for index, sentence in enumerate(sentences):
        buffer.append(sentence)
        silence = hold_after.get(index)
        if silence is not None:
            chunks.append(("".join(buffer), silence))
            buffer = []
    if buffer:
        chunks.append(("".join(buffer), 0.0))
    return chunks
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_render_chunks.py -v`
Expected: 11 个用例全 PASS

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/chunks.py tests/test_render_chunks.py
git commit -m "feat(render): add narration chunking with hold placement"
```

---

### Task 7: TTS Protocol、重试与整集合成

**Files:**
- Create: `src/tenmin/render/tts.py`
- Modify: `tests/fakes.py`
- Test: `tests/test_render_tts.py`

- [ ] **Step 1: 写 FakeTTSEngine**

在 `tests/fakes.py` 末尾追加（顶部 import 补 `from pathlib import Path`）：

```python
class FakeTTSEngine:
    """测试用假 TTS engine。绝不联网，按调用顺序返回预置时长。"""

    def __init__(self, durations: list[float]):
        self.durations = list(durations)
        self.calls: list[dict[str, Any]] = []

    async def synthesize(self, text: str, out_path: Path) -> float:
        self.calls.append({"text": text, "out_path": out_path})
        if not self.durations:
            raise AssertionError("FakeTTSEngine 的预置时长已用尽")
        duration = self.durations.pop(0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake mp3")
        return duration


class FlakyTTSEngine:
    """前 fail_times 次抛错，之后返回 duration。测重试用。"""

    def __init__(self, fail_times: int, duration: float = 3.0):
        self.fail_times = fail_times
        self.duration = duration
        self.attempts = 0

    async def synthesize(self, text: str, out_path: Path) -> float:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(f"网络抖动 {self.attempts}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake mp3")
        return self.duration
```

- [ ] **Step 2: 写失败的测试**

创建 `tests/test_render_tts.py`：

```python
import pytest

from tenmin.config import RenderConfig
from tenmin.models import AudioDirection, Beat, Hold, Script
from tenmin.render.tts import (
    TTS_MAX_ATTEMPTS,
    EdgeTTSEngine,
    TTSEngine,
    build_tts_engine,
    synthesize_track,
    synthesize_with_retry,
)

from .fakes import FakeTTSEngine, FlakyTTSEngine


def sample_script() -> Script:
    return Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(
                id="b1",
                label="Hook",
                role="hook",
                narration="第一句。第二句。",
                audio=AudioDirection(holds=[Hold(at=1.0, duration=1.5, quote="金句")]),
            ),
            Beat(id="b2", label="收尾", role="outro", narration="第三句。"),
        ],
    )


def test_fake_engine_satisfies_protocol():
    assert isinstance(FakeTTSEngine([1.0]), TTSEngine)


def test_edge_engine_satisfies_protocol():
    assert isinstance(EdgeTTSEngine(voice="zh-CN-YunxiNeural", rate="+0%"), TTSEngine)


def test_build_tts_engine_uses_render_config():
    engine = build_tts_engine(RenderConfig(voice="zh-CN-XiaoxiaoNeural", rate="+10%"))
    assert isinstance(engine, EdgeTTSEngine)
    assert engine.voice == "zh-CN-XiaoxiaoNeural"
    assert engine.rate == "+10%"


async def test_synthesize_with_retry_recovers(tmp_path):
    engine = FlakyTTSEngine(fail_times=2, duration=3.0)
    duration = await synthesize_with_retry(
        engine, "第一句。", tmp_path / "chunk_001.mp3", label="beat b1 的第 1 个 chunk"
    )
    assert duration == 3.0
    assert engine.attempts == 3


async def test_synthesize_with_retry_gives_up_after_max_attempts(tmp_path):
    engine = FlakyTTSEngine(fail_times=TTS_MAX_ATTEMPTS)
    with pytest.raises(RuntimeError) as exc:
        await synthesize_with_retry(
            engine, "第一句。", tmp_path / "chunk_001.mp3", label="beat b1 的第 1 个 chunk"
        )
    message = str(exc.value)
    assert "beat b1 的第 1 个 chunk" in message
    assert "第一句。" in message
    assert engine.attempts == TTS_MAX_ATTEMPTS


async def test_synthesize_track_builds_chunks(tmp_path):
    engine = FakeTTSEngine([3.0, 4.0, 5.0])
    track, warnings = await synthesize_track(sample_script(), 2, tmp_path, engine)
    assert warnings == []
    assert [c.path for c in track.chunks] == [
        "chunk_001.mp3",
        "chunk_002.mp3",
        "chunk_003.mp3",
    ]
    assert [c.beat_id for c in track.chunks] == ["b1", "b1", "b2"]
    assert [c.index for c in track.chunks] == [1, 2, 1]
    assert [c.text for c in track.chunks] == ["第一句。", "第二句。", "第三句。"]
    assert [c.duration for c in track.chunks] == [3.0, 4.0, 5.0]
    assert [c.hold_after for c in track.chunks] == [1.5, 0.0, 0.0]
    # 3 + 1.5 + 4 + 5
    assert track.total_seconds == pytest.approx(13.5)
    assert track.episode == 2


async def test_synthesize_track_writes_files(tmp_path):
    engine = FakeTTSEngine([3.0, 4.0, 5.0])
    await synthesize_track(sample_script(), 2, tmp_path, engine)
    assert (tmp_path / "chunk_001.mp3").exists()
    assert (tmp_path / "chunk_003.mp3").exists()


async def test_synthesize_track_reuses_existing_chunk(tmp_path, monkeypatch):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "chunk_001.mp3").write_bytes(b"already there")
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 9.0)
    engine = FakeTTSEngine([4.0, 5.0])
    track, _ = await synthesize_track(sample_script(), 2, tmp_path, engine)
    # 第一个 chunk 复用磁盘上的文件，engine 只被调了 2 次
    assert len(engine.calls) == 2
    assert track.chunks[0].duration == 9.0


async def test_synthesize_track_reuse_false_resynthesizes(tmp_path, monkeypatch):
    (tmp_path / "chunk_001.mp3").write_bytes(b"already there")
    monkeypatch.setattr("tenmin.render.tts.probe_duration", lambda path: 9.0)
    engine = FakeTTSEngine([3.0, 4.0, 5.0])
    track, _ = await synthesize_track(sample_script(), 2, tmp_path, engine, reuse=False)
    assert len(engine.calls) == 3
    assert track.chunks[0].duration == 3.0


async def test_synthesize_track_warns_on_empty_narration(tmp_path):
    script = Script(
        show="剧名",
        episodes=[2],
        beats=[Beat(id="b1", label="Hook", role="hook", narration="  ")],
    )
    track, warnings = await synthesize_track(script, 2, tmp_path, FakeTTSEngine([]))
    assert track.chunks == []
    assert len(warnings) == 1
    assert "b1" in warnings[0]
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_render_tts.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.render.tts'`

- [ ] **Step 4: 写实现**

创建 `src/tenmin/render/tts.py`：

```python
"""TTS。跟 v1 的 LLMProvider 完全同构：真实引擎藏在 Protocol 后面，测试用假引擎。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from tenmin.config import RenderConfig
from tenmin.models import Script, VoiceChunk, VoiceTrack
from tenmin.render.chunks import plan_chunks
from tenmin.render.ffmpeg import probe_duration

TTS_MAX_ATTEMPTS = 3


@runtime_checkable
class TTSEngine(Protocol):
    async def synthesize(self, text: str, out_path: Path) -> float:
        """合成一段语音，返回真实时长（秒）。"""
        ...


class EdgeTTSEngine:
    def __init__(self, voice: str = "zh-CN-YunxiNeural", rate: str = "+0%") -> None:
        self.voice = voice
        self.rate = rate

    async def synthesize(self, text: str, out_path: Path) -> float:
        import edge_tts

        out_path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        await communicate.save(str(out_path))
        return probe_duration(out_path)


def build_tts_engine(cfg: RenderConfig) -> TTSEngine:
    return EdgeTTSEngine(voice=cfg.voice, rate=cfg.rate)


async def synthesize_with_retry(
    engine: TTSEngine, text: str, out_path: Path, *, label: str
) -> float:
    """单个 chunk 重试 TTS_MAX_ATTEMPTS 次。失败时报清楚是哪个 chunk、原文是什么。"""
    last_error = ""
    for _ in range(TTS_MAX_ATTEMPTS):
        try:
            return await engine.synthesize(text, out_path)
        except Exception as error:  # noqa: BLE001 - 网络层什么都可能抛
            last_error = str(error)
    raise RuntimeError(
        f"{label} 连续 {TTS_MAX_ATTEMPTS} 次合成失败：{text!r}\n{last_error}"
    )


async def synthesize_track(
    script: Script,
    episode: int,
    voice_dir: Path,
    engine: TTSEngine,
    *,
    reuse: bool = True,
) -> tuple[VoiceTrack, list[str]]:
    """合成整集旁白。chunk 独立落盘，重跑只补缺的那几个。"""
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[VoiceChunk] = []
    warnings: list[str] = []
    serial = 0
    for beat in script.beats:
        planned = plan_chunks(beat)
        if not planned:
            warnings.append(f"beat {beat.id} 没有旁白文本，已跳过配音")
            continue
        for index, (text, hold_after) in enumerate(planned, start=1):
            serial += 1
            filename = f"chunk_{serial:03d}.mp3"
            out_path = voice_dir / filename
            label = f"beat {beat.id} 的第 {index} 个 chunk"
            if reuse and out_path.is_file() and out_path.stat().st_size > 0:
                duration = probe_duration(out_path)
            else:
                duration = await synthesize_with_retry(engine, text, out_path, label=label)
            chunks.append(
                VoiceChunk(
                    beat_id=beat.id,
                    index=index,
                    text=text,
                    path=filename,
                    duration=duration,
                    hold_after=hold_after,
                )
            )
    total = sum(chunk.duration + chunk.hold_after for chunk in chunks)
    return VoiceTrack(episode=episode, chunks=chunks, total_seconds=total), warnings
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_render_tts.py -v`
Expected: 10 个用例全 PASS。确认没有联网：整个文件不该出现超过 1 秒的用例。

- [ ] **Step 6: Commit**

```bash
git add src/tenmin/render/tts.py tests/test_render_tts.py tests/fakes.py
git commit -m "feat(render): add TTS protocol, retry and per-episode synthesis"
```

---

### Task 8: 时间轴重算（纯函数，全案核心）

**算法：** 逐 beat 处理。本 beat 音频总时长 = 各 chunk 真实时长之和 + 各 hold 之和；`ratio = 音频总时长 / 本 beat clip 原始总时长`；每个 clip 的 `source_start` 不动，时长 × ratio。

**两个游标：** `audio_cursor` 每个 chunk 走 `duration` 再走 `hold_after`，驱动字幕时刻与 `narration_offsets`；`picture_cursor` 按每段画面实际时长走，驱动 `timeline_start/end`。正常情况两者收敛到同一个值；被钳制时会分叉，最后统一报 warning。

**Files:**
- Create: `src/tenmin/render/timeline.py`
- Test: `tests/test_render_timeline.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_render_timeline.py`：

```python
import pytest

from tenmin.models import AudioDirection, Beat, Clip, Hold, Script, VoiceChunk, VoiceTrack
from tenmin.render.timeline import (
    beat_audio_seconds,
    beat_clip_seconds,
    build_timeline,
    chunks_by_beat,
    scale_ratio,
)


def one_beat_script(clips: list[Clip], holds: list[Hold] | None = None) -> Script:
    return Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(
                id="b1",
                label="Hook",
                role="hook",
                narration="第一句。第二句。",
                clips=clips,
                audio=AudioDirection(holds=holds or []),
            )
        ],
    )


def two_chunk_track() -> VoiceTrack:
    chunks = [
        VoiceChunk(
            beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3",
            duration=8.0, hold_after=2.0,
        ),
        VoiceChunk(
            beat_id="b1", index=2, text="第二句。", path="chunk_002.mp3", duration=10.0
        ),
    ]
    return VoiceTrack(episode=2, chunks=chunks, total_seconds=20.0)


def test_beat_audio_seconds_includes_holds():
    assert beat_audio_seconds(two_chunk_track().chunks) == pytest.approx(20.0)


def test_beat_clip_seconds_sums_clip_durations():
    beat = one_beat_script(
        [Clip(episode=2, start=100.0, end=140.0), Clip(episode=2, start=200.0, end=220.0)]
    ).beats[0]
    assert beat_clip_seconds(beat) == pytest.approx(60.0)


def test_scale_ratio_shrinks_when_material_is_longer():
    assert scale_ratio(20.0, 40.0) == pytest.approx(0.5)


def test_scale_ratio_grows_when_narration_is_longer():
    assert scale_ratio(30.0, 10.0) == pytest.approx(3.0)


def test_scale_ratio_on_zero_clip_seconds_is_zero():
    assert scale_ratio(20.0, 0.0) == 0.0


def test_chunks_by_beat_groups_in_order():
    track = VoiceTrack(
        episode=2,
        chunks=[
            VoiceChunk(beat_id="b1", index=1, text="a", path="chunk_001.mp3", duration=1.0),
            VoiceChunk(beat_id="b2", index=1, text="b", path="chunk_002.mp3", duration=2.0),
            VoiceChunk(beat_id="b1", index=2, text="c", path="chunk_003.mp3", duration=3.0),
        ],
    )
    grouped = chunks_by_beat(track)
    assert [c.text for c in grouped["b1"]] == ["a", "c"]
    assert [c.text for c in grouped["b2"]] == ["b"]


def test_build_timeline_scales_single_clip():
    script = one_beat_script([Clip(episode=2, start=100.0, end=140.0)])
    timeline, warnings = build_timeline(script, two_chunk_track(), source_duration=1400.0)
    assert warnings == []
    assert len(timeline.segments) == 1
    seg = timeline.segments[0]
    assert seg.beat_id == "b1"
    assert seg.source_start == pytest.approx(100.0)
    assert seg.source_end == pytest.approx(120.0)
    assert seg.timeline_start == pytest.approx(0.0)
    assert seg.timeline_end == pytest.approx(20.0)
    assert timeline.total_seconds == pytest.approx(20.0)
    assert timeline.episode == 2


def test_build_timeline_places_subtitles_and_offsets():
    script = one_beat_script([Clip(episode=2, start=100.0, end=140.0)])
    timeline, _ = build_timeline(script, two_chunk_track(), source_duration=1400.0)
    assert [(c.start, c.end, c.text) for c in timeline.subtitles] == [
        (0.0, 8.0, "第一句。"),
        (10.0, 20.0, "第二句。"),
    ]
    # hold 期间没有字幕，画面干净、只有原声
    assert timeline.narration_offsets == pytest.approx([0.0, 10.0])


def test_build_timeline_scales_two_clips_proportionally():
    script = one_beat_script(
        [Clip(episode=2, start=100.0, end=140.0), Clip(episode=2, start=200.0, end=220.0)]
    )
    chunks = [
        VoiceChunk(beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=30.0)
    ]
    track = VoiceTrack(episode=2, chunks=chunks, total_seconds=30.0)
    timeline, warnings = build_timeline(script, track, source_duration=1400.0)
    assert warnings == []
    first, second = timeline.segments
    # ratio = 30 / 60 = 0.5
    assert (first.source_start, first.source_end) == pytest.approx((100.0, 120.0))
    assert (first.timeline_start, first.timeline_end) == pytest.approx((0.0, 20.0))
    assert (second.source_start, second.source_end) == pytest.approx((200.0, 210.0))
    assert (second.timeline_start, second.timeline_end) == pytest.approx((20.0, 30.0))


def test_build_timeline_clamps_at_source_end_with_warning():
    script = one_beat_script([Clip(episode=2, start=100.0, end=110.0)])
    chunks = [
        VoiceChunk(beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=30.0)
    ]
    track = VoiceTrack(episode=2, chunks=chunks, total_seconds=30.0)
    timeline, warnings = build_timeline(script, track, source_duration=120.0)
    # ratio = 3.0，本该切到 130，源片只有 120
    assert timeline.segments[0].source_end == pytest.approx(120.0)
    assert timeline.segments[0].timeline_end == pytest.approx(20.0)
    assert any("钳到片尾" in w for w in warnings)
    # 画面 20s 与音频 30s 相差超过 0.5s，应额外报一条
    assert any("相差超过" in w for w in warnings)


def test_build_timeline_warns_on_beat_without_chunks():
    script = one_beat_script([Clip(episode=2, start=100.0, end=140.0)])
    timeline, warnings = build_timeline(
        script, VoiceTrack(episode=2, chunks=[]), source_duration=1400.0
    )
    assert timeline.segments == []
    assert any("没有配音 chunk" in w for w in warnings)


def test_build_timeline_warns_on_beat_without_clips():
    script = one_beat_script([])
    timeline, warnings = build_timeline(script, two_chunk_track(), source_duration=1400.0)
    assert timeline.segments == []
    # 音频照旧推进，字幕仍然产出
    assert len(timeline.subtitles) == 2
    assert timeline.total_seconds == pytest.approx(20.0)
    assert any("没有可用的 clip" in w for w in warnings)


def test_build_timeline_drops_clip_starting_past_source_end():
    script = one_beat_script([Clip(episode=2, start=200.0, end=240.0)])
    timeline, warnings = build_timeline(script, two_chunk_track(), source_duration=150.0)
    assert timeline.segments == []
    assert any("超出源片长" in w for w in warnings)


def test_build_timeline_segments_are_contiguous_and_monotonic():
    script = Script(
        show="剧名",
        episodes=[2],
        beats=[
            Beat(
                id="b1",
                label="Hook",
                role="hook",
                narration="第一句。",
                clips=[Clip(episode=2, start=100.0, end=140.0)],
            ),
            Beat(
                id="b2",
                label="收尾",
                role="outro",
                narration="第二句。",
                clips=[Clip(episode=2, start=300.0, end=340.0)],
            ),
        ],
    )
    track = VoiceTrack(
        episode=2,
        chunks=[
            VoiceChunk(beat_id="b1", index=1, text="第一句。", path="chunk_001.mp3", duration=10.0),
            VoiceChunk(beat_id="b2", index=1, text="第二句。", path="chunk_002.mp3", duration=20.0),
        ],
        total_seconds=30.0,
    )
    timeline, warnings = build_timeline(script, track, source_duration=1400.0)
    assert warnings == []
    starts = [s.timeline_start for s in timeline.segments]
    assert starts == sorted(starts)
    for previous, current in zip(timeline.segments, timeline.segments[1:], strict=False):
        assert current.timeline_start == pytest.approx(previous.timeline_end)
    assert timeline.segments[-1].timeline_end == pytest.approx(timeline.total_seconds)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_render_timeline.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.render.timeline'`

- [ ] **Step 3: 写实现**

创建 `src/tenmin/render/timeline.py`：

```python
"""时间轴重算。全是纯函数。

对齐规则：clip.start 硬（它过了 v1 validate.py 的 anchor 校正，是有据可查的镜头起点），
clip 时长按 ratio 缩放（clip.end 是 LLM 猜的，只用来算比例）。
"""

from __future__ import annotations

from tenmin.models import (
    Beat,
    Script,
    SubtitleCue,
    Timeline,
    TimelineSegment,
    VoiceChunk,
    VoiceTrack,
)

# 画面与音频总时长的容忍差。超过就报 warning，不报错。
DRIFT_TOLERANCE = 0.5


def beat_audio_seconds(chunks: list[VoiceChunk]) -> float:
    """本 beat 的音频总时长。含 hold 静音——静音期间也得有画面。"""
    return sum(chunk.duration + chunk.hold_after for chunk in chunks)


def beat_clip_seconds(beat: Beat) -> float:
    return sum(clip.duration for clip in beat.clips)


def scale_ratio(audio_seconds: float, clip_seconds: float) -> float:
    if clip_seconds <= 0:
        return 0.0
    return audio_seconds / clip_seconds


def chunks_by_beat(track: VoiceTrack) -> dict[str, list[VoiceChunk]]:
    grouped: dict[str, list[VoiceChunk]] = {}
    for chunk in track.chunks:
        grouped.setdefault(chunk.beat_id, []).append(chunk)
    return grouped


def build_timeline(
    script: Script, track: VoiceTrack, source_duration: float
) -> tuple[Timeline, list[str]]:
    """按 beat 逐段重算画面时长，产出成片时间轴。"""
    grouped = chunks_by_beat(track)
    segments: list[TimelineSegment] = []
    subtitles: list[SubtitleCue] = []
    offsets: list[float] = []
    warnings: list[str] = []
    audio_cursor = 0.0
    picture_cursor = 0.0

    for beat in script.beats:
        chunks = grouped.get(beat.id, [])
        if not chunks:
            warnings.append(f"beat {beat.id} 没有配音 chunk，已跳过")
            continue

        # 音频游标：字幕与旁白落点都由它驱动
        for chunk in chunks:
            offsets.append(audio_cursor)
            subtitles.append(
                SubtitleCue(start=audio_cursor, end=audio_cursor + chunk.duration, text=chunk.text)
            )
            audio_cursor += chunk.duration + chunk.hold_after

        audio_seconds = beat_audio_seconds(chunks)
        clip_seconds = beat_clip_seconds(beat)
        if clip_seconds <= 0:
            warnings.append(f"beat {beat.id} 没有可用的 clip，该段将没有画面")
            continue

        ratio = scale_ratio(audio_seconds, clip_seconds)
        for clip in beat.clips:
            if clip.start >= source_duration:
                warnings.append(
                    f"beat {beat.id} 的 clip 起点 {clip.start:.1f}s 超出源片长 "
                    f"{source_duration:.1f}s，已丢弃该段"
                )
                continue
            source_end = clip.start + clip.duration * ratio
            if source_end > source_duration:
                warnings.append(
                    f"beat {beat.id} 的 clip {clip.start:.1f}s 延长到 {source_end:.1f}s "
                    f"超出源片长 {source_duration:.1f}s，已钳到片尾"
                )
                source_end = source_duration
            if source_end <= clip.start:
                warnings.append(
                    f"beat {beat.id} 的 clip {clip.start:.1f}s 缩放后时长为 0，已丢弃该段"
                )
                continue
            length = source_end - clip.start
            segments.append(
                TimelineSegment(
                    beat_id=beat.id,
                    source_start=clip.start,
                    source_end=source_end,
                    timeline_start=picture_cursor,
                    timeline_end=picture_cursor + length,
                )
            )
            picture_cursor += length

    if abs(picture_cursor - audio_cursor) > DRIFT_TOLERANCE:
        warnings.append(
            f"画面总时长 {picture_cursor:.1f}s 与音频总时长 {audio_cursor:.1f}s "
            f"相差超过 {DRIFT_TOLERANCE}s，成片尾部会有画面缺失或黑屏，请检查 timeline.json"
        )

    timeline = Timeline(
        episode=track.episode,
        segments=segments,
        subtitles=subtitles,
        narration_offsets=offsets,
        total_seconds=audio_cursor,
    )
    return timeline, warnings
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_render_timeline.py -v`
Expected: 14 个用例全 PASS

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/timeline.py tests/test_render_timeline.py
git commit -m "feat(render): add ratio-based timeline recalculation"
```

---

### Task 9: ASS 字幕生成（纯函数）

ASS 的时间格式是 `H:MM:SS.cc`（厘秒、小时不补零），跟 `timecode.format_timestamp` 的 `HH:MM:SS.mmm` 不一样，所以这里必须自己写格式化函数，不要复用。

**Files:**
- Create: `src/tenmin/render/subtitles.py`
- Test: `tests/test_render_subtitles.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_render_subtitles.py`：

```python
from tenmin.models import SubtitleCue
from tenmin.render.subtitles import escape_text, format_ass_time, render_ass

EXPECTED = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, \
BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, \
BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Narration,Source Han Sans SC,48,&H00FFFFFF,&H000000FF,&H00000000,\
&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:08.00,Narration,,0,0,0,,第一句。
Dialogue: 0,0:00:10.00,0:00:20.00,Narration,,0,0,0,,第二句。
"""


def test_format_ass_time_at_zero():
    assert format_ass_time(0.0) == "0:00:00.00"


def test_format_ass_time_centiseconds():
    assert format_ass_time(3.5) == "0:00:03.50"


def test_format_ass_time_rounds_to_centiseconds():
    assert format_ass_time(3661.239) == "1:01:01.24"


def test_format_ass_time_clamps_negative():
    assert format_ass_time(-1.0) == "0:00:00.00"


def test_escape_text_converts_newline():
    assert escape_text("上一行\n下一行") == "上一行\\N下一行"


def test_escape_text_strips_braces():
    # ASS 用花括号写覆盖标签，正文里的花括号必须去掉
    assert escape_text("这里{有}花括号") == "这里有花括号"


def test_render_ass_matches_expected_output():
    cues = [
        SubtitleCue(start=0.0, end=8.0, text="第一句。"),
        SubtitleCue(start=10.0, end=20.0, text="第二句。"),
    ]
    assert render_ass(cues) == EXPECTED


def test_render_ass_honours_font_size():
    out = render_ass([SubtitleCue(start=0.0, end=1.0, text="喂")], font_size=64)
    assert "Style: Narration,Source Han Sans SC,64," in out


def test_render_ass_honours_font_name():
    out = render_ass([SubtitleCue(start=0.0, end=1.0, text="喂")], font_name="PingFang SC")
    assert "Style: Narration,PingFang SC,48," in out


def test_render_ass_without_cues_still_has_headers():
    out = render_ass([])
    assert "[Events]" in out
    assert "Dialogue:" not in out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_render_subtitles.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tenmin.render.subtitles'`

- [ ] **Step 3: 写实现**

创建 `src/tenmin/render/subtitles.py`：

```python
"""生成 ASS 字幕。全是纯函数，输出逐字符可测。

ASS 的时间是 H:MM:SS.cc（厘秒、小时不补零），跟 timecode.format_timestamp 不同，
所以这里单独写格式化函数。
"""

from __future__ import annotations

from tenmin.models import SubtitleCue

DEFAULT_FONT_NAME = "Source Han Sans SC"
DEFAULT_FONT_SIZE = 48
PLAY_RES_X = 1920
PLAY_RES_Y = 1080

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
# 白字黑边、底部居中、四周留 60px。Alignment 2 = 底部居中。
_STYLE_TAIL = "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,60,1"


def format_ass_time(seconds: float) -> str:
    """秒 → H:MM:SS.cc。负数按 0 处理。"""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    hours, rem = divmod(total_cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def escape_text(text: str) -> str:
    """ASS 正文转义：换行用 \\N，花括号是覆盖标签的定界符必须去掉。"""
    cleaned = text.replace("{", "").replace("}", "")
    return cleaned.replace("\r\n", "\n").replace("\n", "\\N").strip()


def render_ass(
    cues: list[SubtitleCue],
    *,
    font_size: int = DEFAULT_FONT_SIZE,
    font_name: str = DEFAULT_FONT_NAME,
) -> str:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {PLAY_RES_X}",
        f"PlayResY: {PLAY_RES_Y}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        _STYLE_FORMAT,
        f"Style: Narration,{font_name},{font_size},{_STYLE_TAIL}",
        "",
        "[Events]",
        _EVENT_FORMAT,
    ]
    for cue in cues:
        lines.append(
            f"Dialogue: 0,{format_ass_time(cue.start)},{format_ass_time(cue.end)},"
            f"Narration,,0,0,0,,{escape_text(cue.text)}"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_render_subtitles.py -v`
Expected: 10 个用例全 PASS。如果 `test_render_ass_matches_expected_output` 失败，用 `pytest -vv` 看逐字符 diff——`EXPECTED` 里的反斜杠续行必须跟实现里的字符串拼接结果完全一致。

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/subtitles.py tests/test_render_subtitles.py
git commit -m "feat(render): add ASS subtitle rendering"
```

---

### Task 10: 混音（`render/audio.py`）

原声按 timeline 的 segment 切片拼接，有旁白的区间压低（ducking），留白区间回到全开；旁白 chunk 按 `narration_offsets` 延迟后叠上去。

**这个阶段测的是「ffmpeg 命令行拼对了没」，不真跑 ffmpeg。** ffmpeg 自己干得对不对是它的责任，我们测不了也不该测；命令行拼错才是我们会犯的错。

**Files:**
- Create: `src/tenmin/render/audio.py`
- Test: `tests/test_render_audio.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_render_audio.py`：

```python
import pytest

from tenmin.models import SubtitleCue, Timeline, TimelineSegment, VoiceChunk, VoiceTrack
from tenmin.render.audio import (
    build_mix_args,
    duck_gain,
    duck_volume_expr,
    mix_audio,
)


def make_timeline() -> Timeline:
    """两个 segment、三个旁白 chunk，画面总时长与音频总时长都是 30 秒。"""
    return Timeline(
        episode=2,
        segments=[
            TimelineSegment(
                beat_id="b1",
                source_start=100.0,
                source_end=120.0,
                timeline_start=0.0,
                timeline_end=20.0,
            ),
            TimelineSegment(
                beat_id="b2",
                source_start=200.0,
                source_end=210.0,
                timeline_start=20.0,
                timeline_end=30.0,
            ),
        ],
        subtitles=[
            SubtitleCue(start=0.0, end=8.0, text="第一句"),
            SubtitleCue(start=10.0, end=20.0, text="第二句"),
            SubtitleCue(start=20.0, end=30.0, text="第三句"),
        ],
        narration_offsets=[0.0, 10.0, 20.0],
        total_seconds=30.0,
    )


def make_track() -> VoiceTrack:
    return VoiceTrack(
        episode=2,
        chunks=[
            VoiceChunk(
                beat_id="b1",
                index=1,
                text="第一句",
                path="chunk_001.mp3",
                duration=8.0,
                hold_after=2.0,
            ),
            VoiceChunk(
                beat_id="b1",
                index=2,
                text="第二句",
                path="chunk_002.mp3",
                duration=10.0,
            ),
            VoiceChunk(
                beat_id="b2",
                index=1,
                text="第三句",
                path="chunk_003.mp3",
                duration=10.0,
            ),
        ],
        total_seconds=30.0,
    )


EXPECTED_GRAPH = (
    "[0:a]atrim=start=100.000:end=120.000,asetpts=PTS-STARTPTS[o0];"
    "[0:a]atrim=start=200.000:end=210.000,asetpts=PTS-STARTPTS[o1];"
    "[o0][o1]concat=n=2:v=0:a=1[orig];"
    "[orig]volume='if(gt(between(t,0.000,8.000)+between(t,10.000,20.000)"
    "+between(t,20.000,30.000),0),0.2512,1.0000)':eval=frame[ducked];"
    "[1:a]adelay=delays=0:all=1[n0];"
    "[2:a]adelay=delays=10000:all=1[n1];"
    "[3:a]adelay=delays=20000:all=1[n2];"
    "[n0][n1][n2]amix=inputs=3:normalize=0[voice];"
    "[ducked][voice]amix=inputs=2:normalize=0[mix]"
)


def build(tmp_path, **overrides):
    kwargs = {
        "video": tmp_path / "source.mkv",
        "timeline": make_timeline(),
        "track": make_track(),
        "voice_dir": tmp_path / "04_voice" / "E02",
        "out_path": tmp_path / "06_audio" / "E02.mixed.m4a",
        "duck_db": -12.0,
    }
    kwargs.update(overrides)
    return build_mix_args(**kwargs)


def test_duck_gain_zero_db_is_unity():
    assert duck_gain(0.0) == pytest.approx(1.0)


def test_duck_gain_minus_twelve_db():
    assert duck_gain(-12.0) == pytest.approx(0.2512, abs=1e-4)


def test_duck_volume_expr_without_cues_stays_full():
    assert duck_volume_expr([], 0.2512) == "1.0000"


def test_duck_volume_expr_lists_every_cue_window():
    cues = [SubtitleCue(start=0.0, end=8.0, text="a"), SubtitleCue(start=10.0, end=20.0, text="b")]
    assert duck_volume_expr(cues, 0.2512) == (
        "if(gt(between(t,0.000,8.000)+between(t,10.000,20.000),0),0.2512,1.0000)"
    )


def test_build_mix_args_inputs_video_then_every_chunk(tmp_path):
    args = build(tmp_path)
    voice_dir = tmp_path / "04_voice" / "E02"
    assert args[:3] == ["-y", "-i", str(tmp_path / "source.mkv")]
    assert args[3:5] == ["-i", str(voice_dir / "chunk_001.mp3")]
    assert args[5:7] == ["-i", str(voice_dir / "chunk_002.mp3")]
    assert args[7:9] == ["-i", str(voice_dir / "chunk_003.mp3")]


def test_build_mix_args_filter_graph_matches_expected(tmp_path):
    args = build(tmp_path)
    graph = args[args.index("-filter_complex") + 1]
    assert graph == EXPECTED_GRAPH


def test_build_mix_args_maps_mix_and_encodes_aac(tmp_path):
    args = build(tmp_path)
    out = tmp_path / "06_audio" / "E02.mixed.m4a"
    assert args[-7:] == ["-map", "[mix]", "-c:a", "aac", "-b:a", "192k", str(out)]


def test_build_mix_args_single_chunk_skips_voice_amix(tmp_path):
    timeline = make_timeline()
    timeline.segments = timeline.segments[:1]
    timeline.subtitles = timeline.subtitles[:1]
    timeline.narration_offsets = [0.0]
    track = make_track()
    track.chunks = track.chunks[:1]
    args = build(tmp_path, timeline=timeline, track=track)
    graph = args[args.index("-filter_complex") + 1]
    assert "amix=inputs=1" not in graph
    assert graph.endswith("[ducked][n0]amix=inputs=2:normalize=0[mix]")


def test_build_mix_args_rejects_empty_timeline(tmp_path):
    timeline = make_timeline()
    timeline.segments = []
    with pytest.raises(ValueError) as exc:
        build(tmp_path, timeline=timeline)
    assert "segment" in str(exc.value)


def test_build_mix_args_rejects_empty_track(tmp_path):
    track = make_track()
    track.chunks = []
    with pytest.raises(ValueError) as exc:
        build(tmp_path, track=track)
    assert "chunk" in str(exc.value)


def test_build_mix_args_rejects_offset_count_mismatch(tmp_path):
    timeline = make_timeline()
    timeline.narration_offsets = [0.0]
    with pytest.raises(ValueError):
        build(tmp_path, timeline=timeline)


def test_mix_audio_runs_ffmpeg_and_returns_path(tmp_path, monkeypatch):
    from tenmin.render import audio as audio_module

    seen: list[list[str]] = []

    def fake_run(args):
        seen.append(list(args))
        return ""

    monkeypatch.setattr(audio_module, "run", fake_run)
    out_path = tmp_path / "06_audio" / "E02.mixed.m4a"
    result = audio_module.mix_audio(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        track=make_track(),
        voice_dir=tmp_path / "04_voice" / "E02",
        out_path=out_path,
        duck_db=-12.0,
    )
    assert result == out_path
    assert out_path.parent.is_dir()
    assert seen[0][0] == "-y"
    assert seen[0][-1] == str(out_path)


def test_mix_audio_is_exported():
    assert callable(mix_audio)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_render_audio.py -v`
Expected: 收集阶段就 FAIL，`ModuleNotFoundError: No module named 'tenmin.render.audio'`

- [ ] **Step 3: 写实现**

创建 `src/tenmin/render/audio.py`：

```python
"""混音：原声按 timeline 切片拼接后压低，旁白按 offset 延迟叠上去。

只负责拼 ffmpeg 命令行。命令行对不对由测试逐参数断言，ffmpeg 干得对不对是它自己的事。
"""

from __future__ import annotations

from pathlib import Path

from tenmin.models import SubtitleCue, Timeline, VoiceTrack
from tenmin.render.ffmpeg import run

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
FULL_VOLUME = 1.0


def duck_gain(duck_db: float) -> float:
    """把 dB 换成线性增益。-12dB ≈ 0.2512。"""
    return 10 ** (duck_db / 20)


def duck_volume_expr(cues: list[SubtitleCue], gain: float) -> str:
    """有旁白的区间压到 gain，其余（留白）回到原声全开。

    没有任何旁白时返回常量 1，让 volume 滤镜变成空操作。
    """
    if not cues:
        return f"{FULL_VOLUME:.4f}"
    windows = "+".join(f"between(t,{cue.start:.3f},{cue.end:.3f})" for cue in cues)
    return f"if(gt({windows},0),{gain:.4f},{FULL_VOLUME:.4f})"


def build_mix_args(
    *,
    video: Path,
    timeline: Timeline,
    track: VoiceTrack,
    voice_dir: Path,
    out_path: Path,
    duck_db: float,
) -> list[str]:
    """拼出混音用的 ffmpeg 参数列表（不含 ffmpeg 本身）。"""
    if not timeline.segments:
        raise ValueError("timeline 里没有任何 segment，无法混音")
    if not track.chunks:
        raise ValueError("voice track 里没有任何 chunk，无法混音")
    if len(track.chunks) != len(timeline.narration_offsets):
        raise ValueError(
            f"chunk 数 {len(track.chunks)} 与 narration_offsets 数 "
            f"{len(timeline.narration_offsets)} 不一致，timeline 与 voice 产物不匹配"
        )

    parts: list[str] = []
    for index, segment in enumerate(timeline.segments):
        parts.append(
            f"[0:a]atrim=start={segment.source_start:.3f}:end={segment.source_end:.3f},"
            f"asetpts=PTS-STARTPTS[o{index}]"
        )
    origin_labels = "".join(f"[o{i}]" for i in range(len(timeline.segments)))
    parts.append(f"{origin_labels}concat=n={len(timeline.segments)}:v=0:a=1[orig]")

    expr = duck_volume_expr(timeline.subtitles, duck_gain(duck_db))
    parts.append(f"[orig]volume='{expr}':eval=frame[ducked]")

    chunk_paths: list[str] = []
    for index, (chunk, offset) in enumerate(
        zip(track.chunks, timeline.narration_offsets, strict=True)
    ):
        chunk_paths.append(str(voice_dir / chunk.path))
        parts.append(
            f"[{index + 1}:a]adelay=delays={int(round(offset * 1000))}:all=1[n{index}]"
        )

    if len(track.chunks) == 1:
        voice_label = "[n0]"
    else:
        voice_labels = "".join(f"[n{i}]" for i in range(len(track.chunks)))
        parts.append(f"{voice_labels}amix=inputs={len(track.chunks)}:normalize=0[voice]")
        voice_label = "[voice]"
    parts.append(f"[ducked]{voice_label}amix=inputs=2:normalize=0[mix]")

    args = ["-y", "-i", str(video)]
    for path in chunk_paths:
        args.extend(["-i", path])
    args.extend(
        [
            "-filter_complex",
            ";".join(parts),
            "-map",
            "[mix]",
            "-c:a",
            AUDIO_CODEC,
            "-b:a",
            AUDIO_BITRATE,
            str(out_path),
        ]
    )
    return args


def mix_audio(
    *,
    video: Path,
    timeline: Timeline,
    track: VoiceTrack,
    voice_dir: Path,
    out_path: Path,
    duck_db: float,
) -> Path:
    """真跑 ffmpeg 混音，返回产物路径。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        build_mix_args(
            video=video,
            timeline=timeline,
            track=track,
            voice_dir=voice_dir,
            out_path=out_path,
            duck_db=duck_db,
        )
    )
    return out_path
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_render_audio.py -v`
Expected: 11 个用例全 PASS。若 `test_build_mix_args_filter_graph_matches_expected` 失败，用 `pytest -vv` 看 diff——分隔符是 `;`，标签顺序、小数位数（时间 3 位、增益 4 位）都必须一致。

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/audio.py tests/test_render_audio.py
git commit -m "feat(render): mix ducked original audio with narration"
```

---

### Task 11: 视频渲染（`render/video.py`）

一次 ffmpeg 调用完成 trim + concat + 烧字幕 + 挂音轨。视频只编码一次——编两次等于多掉一次画质、多等几分钟。

**Files:**
- Create: `src/tenmin/render/video.py`
- Test: `tests/test_render_video.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_render_video.py`：

```python
from pathlib import Path

import pytest

from tenmin.models import Timeline, TimelineSegment
from tenmin.render.video import (
    build_render_args,
    escape_filter_path,
    quality_args,
    render_video,
)


def make_timeline() -> Timeline:
    return Timeline(
        episode=2,
        segments=[
            TimelineSegment(
                beat_id="b1",
                source_start=100.0,
                source_end=120.0,
                timeline_start=0.0,
                timeline_end=20.0,
            ),
            TimelineSegment(
                beat_id="b2",
                source_start=200.0,
                source_end=210.0,
                timeline_start=20.0,
                timeline_end=30.0,
            ),
        ],
        subtitles=[],
        narration_offsets=[],
        total_seconds=30.0,
    )


def build(tmp_path, **overrides):
    kwargs = {
        "video": tmp_path / "source.mkv",
        "timeline": make_timeline(),
        "audio": tmp_path / "06_audio" / "E02.mixed.m4a",
        "ass": tmp_path / "05_timeline" / "E02.ass",
        "out_path": tmp_path / "07_render" / "E02.mp4",
        "encoder": "libx264",
    }
    kwargs.update(overrides)
    return build_render_args(**kwargs)


def test_escape_filter_path_wraps_in_single_quotes():
    assert escape_filter_path(Path("/tmp/work/E02.ass")) == "'/tmp/work/E02.ass'"


def test_escape_filter_path_escapes_single_quote():
    assert escape_filter_path(Path("/tmp/it's/E02.ass")) == "'/tmp/it\\'s/E02.ass'"


def test_escape_filter_path_keeps_spaces_and_brackets():
    """源片名里带方括号和空格是常态，单引号包住就够了。"""
    assert escape_filter_path(Path("/a b/[LoliHouse] x.ass")) == "'/a b/[LoliHouse] x.ass'"


def test_quality_args_software_encoder_uses_crf():
    assert quality_args("libx264") == ["-crf", "20", "-preset", "medium"]


def test_quality_args_videotoolbox_uses_bitrate():
    """videotoolbox 不认 -crf/-preset，只能给码率。"""
    assert quality_args("h264_videotoolbox") == ["-b:v", "6000k"]


def test_build_render_args_filter_graph_matches_expected(tmp_path):
    args = build(tmp_path)
    graph = args[args.index("-filter_complex") + 1]
    ass = tmp_path / "05_timeline" / "E02.ass"
    assert graph == (
        "[0:v]trim=start=100.000:end=120.000,setpts=PTS-STARTPTS,"
        "scale=1920:1080,setsar=1[v0];"
        "[0:v]trim=start=200.000:end=210.000,setpts=PTS-STARTPTS,"
        "scale=1920:1080,setsar=1[v1];"
        "[v0][v1]concat=n=2:v=1:a=0[vcat];"
        f"[vcat]subtitles=filename='{ass}'[vout]"
    )


def test_build_render_args_inputs_video_then_audio(tmp_path):
    args = build(tmp_path)
    assert args[:5] == [
        "-y",
        "-i",
        str(tmp_path / "source.mkv"),
        "-i",
        str(tmp_path / "06_audio" / "E02.mixed.m4a"),
    ]


def test_build_render_args_maps_burned_video_and_mixed_audio(tmp_path):
    args = build(tmp_path)
    assert args[args.index("-map") : args.index("-map") + 4] == [
        "-map",
        "[vout]",
        "-map",
        "1:a",
    ]


def test_build_render_args_copies_audio_and_adds_faststart(tmp_path):
    args = build(tmp_path)
    out = tmp_path / "07_render" / "E02.mp4"
    assert args[-13:] == [
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(out),
    ]


def test_build_render_args_honours_custom_resolution(tmp_path):
    args = build(tmp_path, width=1280, height=720)
    graph = args[args.index("-filter_complex") + 1]
    assert "scale=1280:720" in graph
    assert "scale=1920:1080" not in graph


def test_build_render_args_rejects_empty_timeline(tmp_path):
    timeline = make_timeline()
    timeline.segments = []
    with pytest.raises(ValueError) as exc:
        build(tmp_path, timeline=timeline)
    assert "segment" in str(exc.value)


def test_render_video_runs_ffmpeg_and_returns_path(tmp_path, monkeypatch):
    from tenmin.render import video as video_module

    seen: list[list[str]] = []

    def fake_run(args):
        seen.append(list(args))
        return ""

    monkeypatch.setattr(video_module, "run", fake_run)
    out_path = tmp_path / "07_render" / "E02.mp4"
    result = render_video(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        audio=tmp_path / "06_audio" / "E02.mixed.m4a",
        ass=tmp_path / "05_timeline" / "E02.ass",
        out_path=out_path,
        encoder="libx264",
    )
    assert result == out_path
    assert out_path.parent.is_dir()
    assert seen[0][-1] == str(out_path)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_render_video.py -v`
Expected: 收集阶段就 FAIL，`ModuleNotFoundError: No module named 'tenmin.render.video'`

- [ ] **Step 3: 写实现**

创建 `src/tenmin/render/video.py`：

```python
"""视频渲染：一次 ffmpeg 调用完成 trim + concat + 烧字幕 + 挂音轨。

视频只编码一次。编两次等于多掉一次画质、多等几分钟。
代价是烧字幕出错时要连带重跑切片拼接——接受。
"""

from __future__ import annotations

from pathlib import Path

from tenmin.models import Timeline
from tenmin.render.ffmpeg import run

WIDTH = 1920
HEIGHT = 1080
PIX_FMT = "yuv420p"
CRF = "20"
PRESET = "medium"
VIDEOTOOLBOX_BITRATE = "6000k"


def escape_filter_path(path: Path) -> str:
    """filtergraph 里的路径整体用单引号包住，反斜杠与单引号再转义一次。

    源片名常带方括号和空格（`[LoliHouse] ... .mkv`），单引号包住就不用逐字符转义。
    """
    text = str(path).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def quality_args(encoder: str) -> list[str]:
    """videotoolbox 不认 -crf/-preset，只能给码率。"""
    if encoder.endswith("videotoolbox"):
        return ["-b:v", VIDEOTOOLBOX_BITRATE]
    return ["-crf", CRF, "-preset", PRESET]


def build_render_args(
    *,
    video: Path,
    timeline: Timeline,
    audio: Path,
    ass: Path,
    out_path: Path,
    encoder: str,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> list[str]:
    """拼出渲染用的 ffmpeg 参数列表（不含 ffmpeg 本身）。"""
    if not timeline.segments:
        raise ValueError("timeline 里没有任何 segment，无法渲染")

    parts: list[str] = []
    for index, segment in enumerate(timeline.segments):
        parts.append(
            f"[0:v]trim=start={segment.source_start:.3f}:end={segment.source_end:.3f},"
            f"setpts=PTS-STARTPTS,scale={width}:{height},setsar=1[v{index}]"
        )
    labels = "".join(f"[v{i}]" for i in range(len(timeline.segments)))
    parts.append(f"{labels}concat=n={len(timeline.segments)}:v=1:a=0[vcat]")
    parts.append(f"[vcat]subtitles=filename={escape_filter_path(ass)}[vout]")

    return [
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[vout]",
        "-map",
        "1:a",
        "-c:v",
        encoder,
        *quality_args(encoder),
        "-pix_fmt",
        PIX_FMT,
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(out_path),
    ]


def render_video(
    *,
    video: Path,
    timeline: Timeline,
    audio: Path,
    ass: Path,
    out_path: Path,
    encoder: str,
) -> Path:
    """真跑 ffmpeg 渲染，返回成品路径。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        build_render_args(
            video=video,
            timeline=timeline,
            audio=audio,
            ass=ass,
            out_path=out_path,
            encoder=encoder,
        )
    )
    return out_path
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_render_video.py -v`
Expected: 11 个用例全 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/video.py tests/test_render_video.py
git commit -m "feat(render): trim, concat and burn subtitles in one ffmpeg pass"
```

---

### Task 12: pipeline 接入 voice / timeline / audio / render 四个阶段

前 4 个阶段一个字都不改。这一步只做三件事：`Paths` 加 6 个路径、加 4 个 `run_*` 函数、`run_pipeline` 尾部加 4 个 `if` 块。

**Files:**
- Modify: `src/tenmin/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: 改掉两个会失败的旧断言，并写新的失败测试**

先把 `tests/test_pipeline.py` 顶部的 import 换成下面这版（新增 `asyncio` / `Path` / 四个 `run_*` / 五个模型 / `FakeTTSEngine`）：

```python
import asyncio
import json
from pathlib import Path

import pytest

from tenmin.config import ProjectConfig
from tenmin.models import (
    AudioDirection,
    Beat,
    Clip,
    Hold,
    LLMBeat,
    LLMClip,
    LLMScript,
    Script,
)
from tenmin.pipeline import (
    STAGES,
    Paths,
    run_audio,
    run_docgen,
    run_ingest,
    run_pipeline,
    run_render,
    run_signals,
    run_timeline,
    run_voice,
    stages_from,
)

from .fakes import FakeProvider, FakeTTSEngine
```

再把这两个旧断言改成 8 阶段版本（原来只列到 `docgen`）：

```python
def test_stage_names():
    assert STAGES == [
        "ingest",
        "signals",
        "script",
        "docgen",
        "voice",
        "timeline",
        "audio",
        "render",
    ]


def test_stages_from_middle():
    assert stages_from("signals") == [
        "signals",
        "script",
        "docgen",
        "voice",
        "timeline",
        "audio",
        "render",
    ]
```

然后在 `tests/test_pipeline.py` 末尾追加下面这一整段。

`render_script()` 的数字是精心挑的，跟 Task 10 / Task 11 的 fixture 完全对齐：
b1 旁白「第一句。第二句。」切成两句各 4 字，估算落点 `[0.889, 1.778]`，`hold.at=1.0`
最近的句界是 index 0，所以第一个 chunk 后面挂 2.0 秒静音；
`FakeTTSEngine([8.0, 10.0, 10.0])` 给出 b1 音频 `8 + 2 + 10 = 20` 秒、b2 音频 `10` 秒；
b1 素材 40 秒 → ratio 0.5 → 截 100→120；b2 素材 20 秒 → ratio 0.5 → 截 200→210；
画面总长 30 秒，正好等于音频总长 30 秒。

```python
def _write_script(path: Path, script: Script) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script.model_dump_json(indent=2), encoding="utf-8")


def _touch_output(args: list[str]) -> str:
    """假的 ffmpeg：不跑编码，只把输出文件创建出来。"""
    out = Path(args[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"")
    return ""


def render_script() -> Script:
    """两个 beat、两个 clip 的最小剧本，配 FakeTTSEngine([8.0, 10.0, 10.0]) 用。"""
    return Script(
        show="才女的侍从",
        episodes=[2],
        target_seconds=30.0,
        beats=[
            Beat(
                id="b1",
                label="Hook",
                role="hook",
                narration="第一句。第二句。",
                clips=[Clip(episode=2, start=100.0, end=140.0)],
                audio=AudioDirection(
                    holds=[Hold(at=1.0, duration=2.0, quote="第一句")]
                ),
            ),
            Beat(
                id="b2",
                label="收尾",
                role="outro",
                narration="第三句。",
                clips=[Clip(episode=2, start=200.0, end=220.0)],
            ),
        ],
    )


def _prepare_video(cfg: ProjectConfig) -> Path:
    """造一个空壳视频文件并写进 config，供需要 video_path 的阶段用。"""
    video = cfg.root / "E02.mkv"
    video.write_bytes(b"")
    cfg.episodes[0].video = Path("E02.mkv")
    return video


def test_paths_render_layout(tmp_path):
    paths = Paths(tmp_path / "akujo2")
    assert paths.voice_dir(2).name == "E02"
    assert paths.voice_dir(2).parent.name == "04_voice"
    assert paths.voice(2).name == "E02.voice.json"
    assert paths.voice(2).parent.name == "04_voice"
    assert paths.timeline(2).name == "E02.timeline.json"
    assert paths.timeline(2).parent.name == "05_timeline"
    assert paths.subtitles(2).name == "E02.ass"
    assert paths.subtitles(2).parent.name == "05_timeline"
    assert paths.mixed_audio(2).name == "E02.mixed.m4a"
    assert paths.mixed_audio(2).parent.name == "06_audio"
    assert paths.video(2).name == "E02.mp4"
    assert paths.video(2).parent.name == "07_render"


@pytest.mark.asyncio
async def test_run_voice_writes_voice_json(project):
    paths = Paths(project.root)
    _write_script(paths.script, render_script())
    engine = FakeTTSEngine([8.0, 10.0, 10.0])

    track, warnings = await run_voice(project, engine)

    assert warnings == []
    assert [chunk.path for chunk in track.chunks] == [
        "chunk_001.mp3",
        "chunk_002.mp3",
        "chunk_003.mp3",
    ]
    assert track.total_seconds == pytest.approx(30.0)
    assert paths.voice(2).exists()
    assert (paths.voice_dir(2) / "chunk_001.mp3").exists()


@pytest.mark.asyncio
async def test_run_voice_without_script_raises(project):
    with pytest.raises(FileNotFoundError):
        await run_voice(project, FakeTTSEngine([]))


@pytest.mark.asyncio
async def test_run_voice_without_engine_raises(project):
    _write_script(Paths(project.root).script, render_script())
    with pytest.raises(ValueError) as exc:
        await run_voice(project, None)
    assert "TTS" in str(exc.value)


@pytest.mark.asyncio
async def test_run_timeline_writes_timeline_and_ass(project):
    paths = Paths(project.root)
    _write_script(paths.script, render_script())
    await run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0]))

    timeline, warnings = run_timeline(project, source_duration=1400.0)

    assert warnings == []
    assert len(timeline.segments) == 2
    assert timeline.segments[0].source_end == pytest.approx(120.0)
    assert timeline.segments[1].source_end == pytest.approx(210.0)
    assert timeline.segments[1].timeline_end == pytest.approx(30.0)
    assert timeline.narration_offsets == pytest.approx([0.0, 10.0, 20.0])
    assert paths.timeline(2).exists()
    assert paths.subtitles(2).read_text(encoding="utf-8").startswith("[Script Info]")


def test_run_timeline_without_voice_raises(project):
    _write_script(Paths(project.root).script, render_script())
    with pytest.raises(FileNotFoundError):
        run_timeline(project, source_duration=1400.0)


def test_run_timeline_probes_source_when_duration_missing(project, monkeypatch):
    _write_script(Paths(project.root).script, render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0])))
    video = _prepare_video(project)
    calls: list[Path] = []

    def fake_probe(path):
        calls.append(Path(path))
        return 1400.0

    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)

    timeline, warnings = run_timeline(project)

    assert calls == [video]
    assert warnings == []
    assert timeline.total_seconds == pytest.approx(30.0)


def test_run_timeline_uses_config_font_size(project):
    _write_script(Paths(project.root).script, render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0])))
    project.render.font_size = 72

    run_timeline(project, source_duration=1400.0)

    ass = Paths(project.root).subtitles(2).read_text(encoding="utf-8")
    assert "Source Han Sans SC,72," in ass


def test_run_audio_invokes_ffmpeg(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script, render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0])))
    run_timeline(project, source_duration=1400.0)
    _prepare_video(project)
    captured: list[list[str]] = []

    def fake_run(args):
        captured.append(list(args))
        return _touch_output(args)

    monkeypatch.setattr("tenmin.render.audio.run", fake_run)

    out = run_audio(project)

    assert out == paths.mixed_audio(2)
    assert out.exists()
    assert captured[0][:2] == ["-y", "-i"]
    assert captured[0][-1] == str(paths.mixed_audio(2))
    assert "amix=inputs=2:normalize=0[mix]" in captured[0][captured[0].index("-filter_complex") + 1]


def test_run_audio_without_timeline_raises(project):
    _write_script(Paths(project.root).script, render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0])))
    _prepare_video(project)
    with pytest.raises(FileNotFoundError):
        run_audio(project)


def test_run_render_invokes_ffmpeg(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script, render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0])))
    run_timeline(project, source_duration=1400.0)
    _prepare_video(project)
    paths.mixed_audio(2).parent.mkdir(parents=True, exist_ok=True)
    paths.mixed_audio(2).write_bytes(b"")
    captured: list[list[str]] = []

    def fake_run(args):
        captured.append(list(args))
        return _touch_output(args)

    monkeypatch.setattr("tenmin.render.video.run", fake_run)

    out = run_render(project)

    assert out == paths.video(2)
    assert out.exists()
    assert "-movflags" in captured[0]
    assert captured[0][-1] == str(paths.video(2))


def test_run_render_without_audio_raises(project):
    _write_script(Paths(project.root).script, render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8.0, 10.0, 10.0])))
    run_timeline(project, source_duration=1400.0)
    _prepare_video(project)
    with pytest.raises(FileNotFoundError) as exc:
        run_render(project)
    assert "audio 阶段" in str(exc.value)


@pytest.mark.asyncio
async def test_run_pipeline_from_voice_runs_render_stages(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script, render_script())
    _prepare_video(project)
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path: 1400.0)
    monkeypatch.setattr("tenmin.pipeline.preflight", lambda video, encoder: 1400.0)
    monkeypatch.setattr("tenmin.render.audio.run", _touch_output)
    monkeypatch.setattr("tenmin.render.video.run", _touch_output)

    warnings = await run_pipeline(
        project,
        FakeProvider([]),
        from_stage="voice",
        tts_engine=FakeTTSEngine([8.0, 10.0, 10.0]),
    )

    assert warnings == []
    assert paths.voice(2).exists()
    assert paths.timeline(2).exists()
    assert paths.subtitles(2).exists()
    assert paths.mixed_audio(2).exists()
    assert paths.video(2).exists()


@pytest.mark.asyncio
async def test_run_pipeline_skips_voice_when_fresh(project):
    _write_script(Paths(project.root).script, render_script())
    engine = FakeTTSEngine([8.0, 10.0, 10.0])

    await run_pipeline(project, FakeProvider([]), only=["voice"], tts_engine=engine)
    assert len(engine.calls) == 3

    # 预置时长已用尽：真的再合成一次就会 AssertionError
    await run_pipeline(project, FakeProvider([]), only=["voice"], tts_engine=engine)
    assert len(engine.calls) == 3


@pytest.mark.asyncio
async def test_run_pipeline_voice_only_skips_preflight(project, monkeypatch):
    _write_script(Paths(project.root).script, render_script())

    def boom(video, encoder):
        raise AssertionError("只跑 voice 不该做 ffmpeg 前置检查")

    monkeypatch.setattr("tenmin.pipeline.preflight", boom)

    await run_pipeline(
        project,
        FakeProvider([]),
        only=["voice"],
        tts_engine=FakeTTSEngine([8.0, 10.0, 10.0]),
    )

    assert Paths(project.root).voice(2).exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL —— `ImportError: cannot import name 'run_audio' from 'tenmin.pipeline'`。

- [ ] **Step 3: 改 `src/tenmin/pipeline.py`**

把文件顶部的 import 与 `STAGES` 换成这版：

```python
"""阶段编排。每个阶段读上游文件、写自己的文件，靠 mtime 决定是否跳过。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.docgen.narration import render_narration
from tenmin.docgen.table import render_table
from tenmin.ingest.normalize import build_track
from tenmin.models import DialogueTrack, Script, SignalReport, Timeline, VoiceTrack
from tenmin.render.audio import mix_audio
from tenmin.render.ffmpeg import preflight, probe_duration
from tenmin.render.subtitles import render_ass
from tenmin.render.timeline import build_timeline
from tenmin.render.tts import TTSEngine, synthesize_track
from tenmin.render.video import render_video
from tenmin.script.llm import LLMProvider
from tenmin.script.single import generate_script
from tenmin.signals.aggregate import build_report

STAGES = [
    "ingest",
    "signals",
    "script",
    "docgen",
    "voice",
    "timeline",
    "audio",
    "render",
]
```

在 `Paths` 类里，`narration` 属性后面追加 6 个方法：

```python
    def voice_dir(self, episode: int) -> Path:
        return self.root / "04_voice" / f"E{episode:02d}"

    def voice(self, episode: int) -> Path:
        return self.root / "04_voice" / f"E{episode:02d}.voice.json"

    def timeline(self, episode: int) -> Path:
        return self.root / "05_timeline" / f"E{episode:02d}.timeline.json"

    def subtitles(self, episode: int) -> Path:
        return self.root / "05_timeline" / f"E{episode:02d}.ass"

    def mixed_audio(self, episode: int) -> Path:
        return self.root / "06_audio" / f"E{episode:02d}.mixed.m4a"

    def video(self, episode: int) -> Path:
        return self.root / "07_render" / f"E{episode:02d}.mp4"
```

在 `run_docgen` 后面追加下面这一整段（4 个 loader + 4 个阶段函数）：

```python
def _only_episode(cfg: ProjectConfig) -> EpisodeConfig:
    """v2 只做单集。season 模式在 run_pipeline 入口就被拦掉了。"""
    if not cfg.episodes:
        raise ValueError("project.yaml 的 episodes 是空的，至少要配一集")
    return cfg.episodes[0]


def _load_script(cfg: ProjectConfig) -> Script:
    path = Paths(cfg.root).script
    if not path.exists():
        raise FileNotFoundError(f"缺少剧本产物 {path}，请先跑 script 阶段")
    return Script.model_validate_json(path.read_text(encoding="utf-8"))


def _load_voice(cfg: ProjectConfig, episode: int) -> VoiceTrack:
    path = Paths(cfg.root).voice(episode)
    if not path.exists():
        raise FileNotFoundError(f"缺少配音产物 {path}，请先跑 voice 阶段")
    return VoiceTrack.model_validate_json(path.read_text(encoding="utf-8"))


def _load_timeline(cfg: ProjectConfig, episode: int) -> Timeline:
    path = Paths(cfg.root).timeline(episode)
    if not path.exists():
        raise FileNotFoundError(f"缺少时间轴产物 {path}，请先跑 timeline 阶段")
    return Timeline.model_validate_json(path.read_text(encoding="utf-8"))


async def run_voice(
    cfg: ProjectConfig, engine: TTSEngine | None
) -> tuple[VoiceTrack, list[str]]:
    if engine is None:
        raise ValueError(
            "voice 阶段需要 TTS engine，请检查 project.yaml 的 render.voice 配置"
        )
    paths = Paths(cfg.root)
    episode = _only_episode(cfg).number
    script = _load_script(cfg)
    track, warnings = await synthesize_track(
        script, episode, paths.voice_dir(episode), engine
    )
    _write_json(paths.voice(episode), track.model_dump_json(indent=2))
    return track, warnings


def run_timeline(
    cfg: ProjectConfig, *, source_duration: float | None = None
) -> tuple[Timeline, list[str]]:
    paths = Paths(cfg.root)
    episode_cfg = _only_episode(cfg)
    episode = episode_cfg.number
    script = _load_script(cfg)
    track = _load_voice(cfg, episode)
    if source_duration is None:
        source_duration = probe_duration(cfg.video_path(episode_cfg))
    timeline, warnings = build_timeline(script, track, source_duration)
    _write_json(paths.timeline(episode), timeline.model_dump_json(indent=2))
    _write_text(
        paths.subtitles(episode),
        render_ass(timeline.subtitles, font_size=cfg.render.font_size),
    )
    return timeline, warnings


def run_audio(cfg: ProjectConfig) -> Path:
    paths = Paths(cfg.root)
    episode_cfg = _only_episode(cfg)
    episode = episode_cfg.number
    timeline = _load_timeline(cfg, episode)
    track = _load_voice(cfg, episode)
    return mix_audio(
        video=cfg.video_path(episode_cfg),
        timeline=timeline,
        track=track,
        voice_dir=paths.voice_dir(episode),
        out_path=paths.mixed_audio(episode),
        duck_db=cfg.render.duck_db,
    )


def run_render(cfg: ProjectConfig) -> Path:
    paths = Paths(cfg.root)
    episode_cfg = _only_episode(cfg)
    episode = episode_cfg.number
    timeline = _load_timeline(cfg, episode)
    audio = paths.mixed_audio(episode)
    if not audio.exists():
        raise FileNotFoundError(f"缺少混音产物 {audio}，请先跑 audio 阶段")
    ass = paths.subtitles(episode)
    if not ass.exists():
        raise FileNotFoundError(f"缺少字幕产物 {ass}，请先跑 timeline 阶段")
    return render_video(
        video=cfg.video_path(episode_cfg),
        timeline=timeline,
        audio=audio,
        ass=ass,
        out_path=paths.video(episode),
        encoder=cfg.render.video_encoder,
    )
```

最后改 `run_pipeline`：签名多一个关键字参数，`docgen` 块之后追加前置检查与 4 个阶段块。

```python
async def run_pipeline(
    cfg: ProjectConfig,
    provider: LLMProvider,
    *,
    from_stage: str = "ingest",
    only: Sequence[str] | None = None,
    force: bool = False,
    tts_engine: TTSEngine | None = None,
) -> list[str]:
```

`if "docgen" in wanted:` 块的后面、`return warnings` 的前面，插入：

```python
    # 前置检查放在 voice 之前：绝不能跑完几分钟 TTS，最后一步才发现 ffmpeg 没编 libass。
    if {"audio", "render"} & set(wanted):
        preflight(cfg.video_path(_only_episode(cfg)), cfg.render.video_encoder)

    if "voice" in wanted:
        outputs = [paths.voice(n) for n in numbers]
        if force or not _is_fresh(outputs, [paths.script]):
            _, stage_warnings = await run_voice(cfg, tts_engine)
            warnings.extend(stage_warnings)

    if "timeline" in wanted:
        outputs = [paths.timeline(n) for n in numbers] + [
            paths.subtitles(n) for n in numbers
        ]
        inputs = [paths.script] + [paths.voice(n) for n in numbers]
        if force or not _is_fresh(outputs, inputs):
            _, stage_warnings = run_timeline(cfg)
            warnings.extend(stage_warnings)

    if "audio" in wanted:
        outputs = [paths.mixed_audio(n) for n in numbers]
        inputs = [paths.timeline(n) for n in numbers] + [
            paths.voice(n) for n in numbers
        ]
        if force or not _is_fresh(outputs, inputs):
            run_audio(cfg)

    if "render" in wanted:
        outputs = [paths.video(n) for n in numbers]
        inputs = (
            [paths.mixed_audio(n) for n in numbers]
            + [paths.subtitles(n) for n in numbers]
            + [paths.timeline(n) for n in numbers]
        )
        if force or not _is_fresh(outputs, inputs):
            run_render(cfg)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 原有 17 个用例（其中 2 个断言已更新）加新增 15 个，全 PASS。

再跑一遍全量，确认没碰坏 v1：

Run: `uv run pytest -q`
Expected: 全 PASS，无 skip 之外的失败。

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): wire voice, timeline, audio and render stages"
```

---

### Task 13: CLI 接线

`run` 命令不加新选项——阶段本来就走 `--from` / `--only` / `--force`。只需要在 `"voice" in stages` 时造 TTS engine，并把 `FFmpegError` 也纳入友好报错。

**Files:**
- Modify: `src/tenmin/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_cli.py` 顶部需要这些 import（文件里已有的不用重复加）：

```python
import yaml
from typer.testing import CliRunner

from tenmin.cli import app
from tenmin.render.ffmpeg import FFmpegError

runner = CliRunner()
```

在末尾追加：

```python
def _minimal_project(tmp_path):
    root = tmp_path / "akujo2"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: 我是不才恶女\n"
        "slug: akujo2\n"
        "episodes:\n"
        "- number: 2\n"
        "  srt: srt/E02.srt\n",
        encoding="utf-8",
    )
    return root


def test_init_template_has_video_and_render(tmp_path):
    result = runner.invoke(app, ["init", "akujo2", "--work-dir", str(tmp_path)])

    assert result.exit_code == 0
    data = yaml.safe_load(
        (tmp_path / "akujo2" / "project.yaml").read_text(encoding="utf-8")
    )
    assert data["episodes"][0]["video"] == "video/E02.mkv"
    assert data["render"]["voice"] == "zh-CN-YunxiNeural"
    assert data["render"]["rate"] == "+0%"
    assert data["render"]["video_encoder"] == "libx264"
    assert data["render"]["duck_db"] == -12.0
    assert data["render"]["font_size"] == 48
    assert (tmp_path / "akujo2" / "video").is_dir()


def test_run_passes_tts_engine_when_voice_wanted(tmp_path, monkeypatch):
    _minimal_project(tmp_path)
    sentinel = object()
    monkeypatch.setattr("tenmin.cli.build_tts_engine", lambda cfg: sentinel)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "voice"]
    )

    assert result.exit_code == 0
    assert captured["tts_engine"] is sentinel


def test_run_skips_tts_engine_when_voice_not_wanted(tmp_path, monkeypatch):
    _minimal_project(tmp_path)

    def boom(cfg):
        raise AssertionError("只跑 docgen 不该造 TTS engine")

    monkeypatch.setattr("tenmin.cli.build_tts_engine", boom)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "docgen"]
    )

    assert result.exit_code == 0
    assert captured["tts_engine"] is None


def test_run_reports_ffmpeg_error(tmp_path, monkeypatch):
    _minimal_project(tmp_path)

    async def boom(cfg, provider, **kwargs):
        raise FFmpegError("你的 ffmpeg 没编 libass")

    monkeypatch.setattr("tenmin.cli.run_pipeline", boom)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "render"]
    )

    assert result.exit_code == 1
    assert "没编 libass" in result.output


def test_run_prints_mp4_path(tmp_path, monkeypatch):
    root = _minimal_project(tmp_path)
    mp4 = root / "07_render" / "E02.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"")

    async def fake_pipeline(cfg, provider, **kwargs):
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)

    result = runner.invoke(
        app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "render"]
    )

    assert result.exit_code == 0
    assert "成品视频" in result.output
    assert str(mp4) in result.output
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL —— `AttributeError: <module 'tenmin.cli'> has no attribute 'build_tts_engine'`，以及模板断言的 `KeyError: 'video'`。

- [ ] **Step 3: 改 `src/tenmin/cli.py`**

顶部 import 加两行：

```python
from tenmin.render.ffmpeg import FFmpegError
from tenmin.render.tts import build_tts_engine
```

`PROJECT_TEMPLATE` 换成：

```python
PROJECT_TEMPLATE = {
    "show": "剧名",
    "slug": "slug",
    "mode": "single_episode",
    "target_seconds": 240,
    "locale": {"convert_traditional": True},
    "episodes": [{"number": 2, "srt": "srt/E02.srt", "video": "video/E02.mkv"}],
    "glossary": {},
    "llm": {"provider": "gemini", "model": "gemini-3.6-flash"},
    "render": {
        "voice": "zh-CN-YunxiNeural",
        "rate": "+0%",
        "video_encoder": "libx264",
        "duck_db": -12.0,
        "font_size": 48,
    },
}
```

`init` 命令里，建 `srt/` 的那行后面再建 `video/`，并把结尾提示补一句：

```python
    (root / "srt").mkdir(parents=True, exist_ok=True)
    (root / "video").mkdir(parents=True, exist_ok=True)
```

```python
    typer.echo(f"把字幕放进 {root / 'srt'}、源视频放进 {root / 'video'}，")
    typer.echo(f"改好 project.yaml 后跑 tenmin run {slug}")
```

`run` 命令的 docstring 与函数体改成：

```python
    """跑流水线：ingest -> signals -> script -> docgen -> voice -> timeline -> audio -> render。"""
```

`provider` 那段之后插入 engine 构造，并把调用与 except 换掉：

```python
    tts_engine = None
    if "voice" in stages:
        tts_engine = build_tts_engine(cfg.render)

    try:
        warnings = asyncio.run(
            run_pipeline(
                cfg,
                provider,
                from_stage=from_stage,
                only=[only] if only else None,
                force=force,
                tts_engine=tts_engine,
            )
        )
    except (NotImplementedError, FileNotFoundError, ValueError, FFmpegError) as error:
        typer.secho(str(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
```

函数末尾的产物打印补上成品视频：

```python
    paths = Paths(cfg.root)
    if paths.table.exists():
        typer.echo(f"对照表：{paths.table}")
        typer.echo(f"配音文本：{paths.narration}")
    for episode in cfg.episodes:
        mp4 = paths.video(episode.number)
        if mp4.exists():
            typer.echo(f"成品视频：{mp4}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 原有用例加新增 5 个，全 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/cli.py tests/test_cli.py
git commit -m "feat(cli): build TTS engine on demand and report the rendered mp4"
```

---

### Task 14: 黄金样本时间轴断言

用手上那份真 `script.json`（6 beats / 18 clips）跑一遍时间轴重算，锁住「每个镜头都还在、画面总长等于音频总长」这两条不能退化的性质。

**刻意不断言 626 秒 / 253 秒这类绝对数字**：设计文档里那两个数字来自另一份 `script.json`，而 `narration` 一个字的改动就会让总时长变。断言结构不变量比断言快照数字稳。

**Files:**
- Create: `tests/fixtures/akujo_e02.script.json`（从真实产物拷）
- Create: `tests/test_render_golden.py`

- [ ] **Step 1: 把真实产物拷成 fixture**

```bash
cp work/akujo2/03_script/script.json tests/fixtures/akujo_e02.script.json
```

确认一下形状（应该输出 `6 18`）：

```bash
uv run python -c "
import json
d = json.load(open('tests/fixtures/akujo_e02.script.json', encoding='utf-8'))
print(len(d['beats']), sum(len(b['clips']) for b in d['beats']))
"
```

- [ ] **Step 2: 写失败的测试**

Create `tests/test_render_golden.py`:

```python
"""黄金样本：真 script.json 的时间轴重算。断言结构不变量，不断言快照数字。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tenmin.models import Script, VoiceChunk, VoiceTrack
from tenmin.render.chunks import plan_chunks
from tenmin.render.subtitles import render_ass
from tenmin.render.timeline import (
    beat_audio_seconds,
    beat_clip_seconds,
    build_timeline,
    chunks_by_beat,
    scale_ratio,
)
from tenmin.script.budget import SPEECH_RATE_CPS, narration_chars

FIXTURES = Path(__file__).parent / "fixtures"

# 真实剧集约 24 分钟；样本里最后一个 clip 结束在 1314 秒，1440 足够宽松。
SOURCE_DURATION = 1440.0


@pytest.fixture(scope="module")
def golden_script() -> Script:
    path = FIXTURES / "akujo_e02.script.json"
    if not path.exists():
        pytest.skip("缺少 tests/fixtures/akujo_e02.script.json")
    return Script.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_voice(golden_script) -> VoiceTrack:
    """不联网：按 4.5 字/秒给每个 chunk 一个确定时长，模拟一次配音结果。"""
    chunks: list[VoiceChunk] = []
    serial = 0
    for beat in golden_script.beats:
        for index, (text, hold_after) in enumerate(plan_chunks(beat), start=1):
            serial += 1
            chunks.append(
                VoiceChunk(
                    beat_id=beat.id,
                    index=index,
                    text=text,
                    path=f"chunk_{serial:03d}.mp3",
                    duration=narration_chars(text) / SPEECH_RATE_CPS,
                    hold_after=hold_after,
                )
            )
    total = sum(chunk.duration + chunk.hold_after for chunk in chunks)
    return VoiceTrack(episode=2, chunks=chunks, total_seconds=total)


def test_golden_script_shape(golden_script):
    assert len(golden_script.beats) == 6
    assert sum(len(beat.clips) for beat in golden_script.beats) == 18


def test_golden_timeline_keeps_every_clip(golden_script, golden_voice):
    timeline, warnings = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    assert len(timeline.segments) == 18
    assert {segment.beat_id for segment in timeline.segments} == {
        beat.id for beat in golden_script.beats
    }
    assert warnings == []


def test_golden_every_beat_shrinks(golden_script, golden_voice):
    """实测结论：素材远多于旁白，所以每个 beat 的 ratio 都应该落在 (0, 1)。"""
    by_beat = chunks_by_beat(golden_voice)
    for beat in golden_script.beats:
        ratio = scale_ratio(
            beat_audio_seconds(by_beat[beat.id]), beat_clip_seconds(beat)
        )
        assert 0.0 < ratio < 1.0, f"{beat.id} 的 ratio={ratio}"


def test_golden_source_starts_untouched(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    expected = [clip.start for beat in golden_script.beats for clip in beat.clips]
    assert [segment.source_start for segment in timeline.segments] == pytest.approx(
        expected
    )


def test_golden_timeline_is_contiguous(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    assert timeline.segments[0].timeline_start == pytest.approx(0.0)
    for previous, current in zip(
        timeline.segments, timeline.segments[1:], strict=False
    ):
        assert current.timeline_start == pytest.approx(previous.timeline_end)
        assert current.timeline_end > current.timeline_start


def test_golden_picture_matches_audio(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    picture = sum(segment.duration for segment in timeline.segments)
    assert picture == pytest.approx(timeline.total_seconds, abs=0.01)
    assert timeline.total_seconds == pytest.approx(golden_voice.total_seconds, abs=0.01)


def test_golden_subtitles_cover_every_chunk(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    assert len(timeline.subtitles) == len(golden_voice.chunks)
    assert len(timeline.narration_offsets) == len(golden_voice.chunks)
    for cue in timeline.subtitles:
        assert cue.end > cue.start
        assert cue.text.strip() == cue.text


def test_golden_ass_renders(golden_script, golden_voice):
    timeline, _ = build_timeline(golden_script, golden_voice, SOURCE_DURATION)

    ass = render_ass(timeline.subtitles)
    assert "[Script Info]" in ass
    assert ass.count("\nDialogue: ") == len(timeline.subtitles)
```

- [ ] **Step 3: 跑测试**

Run: `uv run pytest tests/test_render_golden.py -v`
Expected: 8 个用例全 PASS。

如果 `test_golden_every_beat_shrinks` 挂了，说明某个 beat 的旁白反而比素材长——那不是 bug，是这份剧本的真实形态，把该 beat 的 ratio 打出来核对后，把断言改成 `assert ratio > 0.0` 并在注释里写明是哪个 beat 例外。**其他七个用例任何一个挂都是真 bug，必须回去修实现。**

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/akujo_e02.script.json tests/test_render_golden.py
git commit -m "test(render): pin timeline invariants against the golden script"
```

---

### Task 15: 端到端真跑 + 文档

最后一步：真调 Edge-TTS、真调 ffmpeg，只跑第一个 beat（约 30 秒）验证整条链路。默认 skip，靠 `-m render` 打开。

**Files:**
- Create: `tests/test_render_e2e.py`
- Modify: `README.md`

- [ ] **Step 1: 写端到端测试**

Create `tests/test_render_e2e.py`:

```python
"""端到端：真 Edge-TTS + 真 ffmpeg 跑一遍。默认 skip，跑法 uv run pytest -m render。

要求 project.yaml 里的 episodes[].video 写绝对路径——测试会把项目根换到 tmp 目录，
相对路径会失效。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tenmin.config import load_project
from tenmin.models import Script, Timeline
from tenmin.pipeline import Paths, run_audio, run_render, run_timeline, run_voice
from tenmin.render.ffmpeg import has_encoder, has_filter, probe_duration
from tenmin.render.tts import build_tts_engine

pytestmark = pytest.mark.render

PROJECT_FILE = Path("work") / "akujo2" / "project.yaml"


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    if not PROJECT_FILE.exists():
        pytest.skip(f"缺少 {PROJECT_FILE}")

    original = load_project(PROJECT_FILE)
    if not original.episodes or original.episodes[0].video is None:
        pytest.skip("project.yaml 没配 episodes[].video")
    video = original.video_path(original.episodes[0])
    if not video.is_absolute():
        pytest.skip("端到端测试要求 video 写绝对路径")
    if not video.exists():
        pytest.skip(f"源视频不存在：{video}")
    if not has_filter("subtitles"):
        pytest.skip(
            "ffmpeg 没编 libass，装法："
            "brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass"
        )
    if not has_encoder(original.render.video_encoder):
        pytest.skip(f"ffmpeg 没有 {original.render.video_encoder} 编码器")

    source_script = Paths(original.root).script
    if not source_script.exists():
        pytest.skip(f"缺少 {source_script}，请先跑 script 阶段")

    # 换到临时根目录，只留第一个 beat，绝不污染真实产物。
    root = tmp_path_factory.mktemp("render_e2e")
    cfg = load_project(PROJECT_FILE).bind_root(root)
    script = Script.model_validate_json(source_script.read_text(encoding="utf-8"))
    script.beats = script.beats[:1]
    trimmed = Paths(root).script
    trimmed.parent.mkdir(parents=True, exist_ok=True)
    trimmed.write_text(script.model_dump_json(indent=2), encoding="utf-8")

    asyncio.run(run_voice(cfg, build_tts_engine(cfg.render)))
    run_timeline(cfg)
    run_audio(cfg)
    return cfg, run_render(cfg)


def test_e2e_produces_mp4(rendered):
    _, mp4 = rendered
    assert mp4.exists()
    assert mp4.stat().st_size > 0


def test_e2e_keeps_every_intermediate_artifact(rendered):
    cfg, _ = rendered
    paths = Paths(cfg.root)
    episode = cfg.episodes[0].number
    assert paths.voice(episode).exists()
    assert paths.timeline(episode).exists()
    assert paths.subtitles(episode).exists()
    assert paths.mixed_audio(episode).exists()


def test_e2e_duration_matches_timeline(rendered):
    cfg, mp4 = rendered
    episode = cfg.episodes[0].number
    timeline = Timeline.model_validate_json(
        Paths(cfg.root).timeline(episode).read_text(encoding="utf-8")
    )
    assert probe_duration(mp4) == pytest.approx(timeline.total_seconds, abs=1.0)
```

- [ ] **Step 2: 确认默认跳过**

Run: `uv run pytest tests/test_render_e2e.py -q`
Expected: `3 deselected` 或 `3 skipped`（取决于 marker 过滤方式），**不能有 failed**。

- [ ] **Step 3: 装好 libass 版 ffmpeg 后真跑一遍**

```bash
brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass
ffmpeg -hide_banner -filters | grep -w subtitles
```
Expected: 能匹配到 `subtitles` 一行。匹配不到就别往下跑。

给 `work/akujo2/project.yaml` 的 episode 补上绝对路径的 `video:`，然后：

Run: `uv run pytest -m render -v`
Expected: 3 个用例 PASS（会真的联网合成语音、真的编码几十秒视频，跑几分钟正常）。

- [ ] **Step 4: 更新 README.md**

「环境准备」一节补 ffmpeg 前置条件：

```markdown
渲染阶段需要编入 libass 的 ffmpeg（否则烧不了字幕）：

```bash
brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass
ffmpeg -hide_banner -filters | grep -w subtitles   # 能匹配到才算装对
```
```

阶段表补 4 行：

```markdown
| voice | `04_voice/E{NN}/chunk_*.mp3`、`04_voice/E{NN}.voice.json` | Edge-TTS 逐句配音，记录每段真实时长 |
| timeline | `05_timeline/E{NN}.timeline.json`、`05_timeline/E{NN}.ass` | 按真实配音时长重算画面时间轴，生成硬字幕 |
| audio | `06_audio/E{NN}.mixed.m4a` | 旁白盖在原声之上，旁白期间原声压低 |
| render | `07_render/E{NN}.mp4` | 一次编码完成切片、拼接、烧字幕、挂音轨 |
```

用法一节补两条：

```markdown
只跑 v2 渲染部分（前 4 个阶段的产物照旧复用）：

```bash
uv run tenmin run akujo2 --from voice
```

手改过 `05_timeline/E02.timeline.json` 后只重新出片：

```bash
uv run tenmin run akujo2 --from audio --force
```
```

测试一节补一条：

```markdown
```bash
uv run pytest -m render     # 真跑渲染链路，需要真视频 + libass 版 ffmpeg
```
```

「人工编辑面」的说明补一句：`03_script/script.json` 管内容，`05_timeline/E{NN}.timeline.json` 管出片节奏。

- [ ] **Step 5: 全量回归**

Run: `uv run pytest -q`
Expected: 全 PASS，render 与 llm 标记的用例 skip/deselect。

Run: `uv run ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add tests/test_render_e2e.py README.md
git commit -m "test(render): add opt-in end-to-end render check and document v2 stages"
```
