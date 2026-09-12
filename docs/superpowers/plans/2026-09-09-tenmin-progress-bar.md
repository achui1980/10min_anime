# Progress Bar for `tenmin run` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `tenmin run` a visual progress bar (via `rich`) showing which of the 8 pipeline stages is running, which were skipped due to mtime freshness, sub-progress for the two slow stages (`voice`: nth sentence/total; `render`: real ffmpeg percent via `-progress pipe:1`), and in batch mode which episode number out of how many.

**Architecture:** A new `ProgressReporter` Protocol (5 methods: `stage_start`/`stage_skip`/`stage_done`/`substep`/`episode_start`) is threaded through `run_pipeline` as an optional keyword-only param, defaulting to a no-op `NullProgressReporter` so all existing tests and library callers are unaffected. `pipeline.py`, `render/tts.py`, and `render/video.py` only depend on the Protocol, never on `rich` directly — this keeps them offline-testable with a `FakeReporter`. The only file that imports `rich` is a new `src/tenmin/rich_progress.py`, wired into `cli.py`. `render/ffmpeg.py` gains a new `run_with_progress()` function (in addition to the existing blocking `run()`) that streams ffmpeg's own `-progress pipe:1` output to compute real percent-complete.

**Tech Stack:** Python 3.12, `rich>=13` (new dependency), `typer` (existing CLI framework), `pytest`/`pytest-asyncio` (existing test stack), `uv` (package manager).

---

## File Structure

**New files:**

| File | Purpose |
|---|---|
| `src/tenmin/progress.py` | `ProgressReporter` Protocol + `NullProgressReporter` |
| `src/tenmin/rich_progress.py` | `RichProgressReporter` — the only file that imports `rich` |
| `tests/test_progress.py` | Tests for `NullProgressReporter` and `FakeReporter` |
| `tests/test_rich_progress.py` | Smoke test for `RichProgressReporter` |

**Modified files:**

| File | Change |
|---|---|
| `pyproject.toml` | Add `rich>=13` dependency |
| `tests/fakes.py` | Add `FakeReporter` |
| `src/tenmin/pipeline.py` | `run_pipeline` gains `reporter=` param; wires `stage_start`/`stage_skip`/`stage_done`/`episode_start` into all 8 stage blocks; `run_voice`/`run_render` gain `reporter=` forwarding |
| `src/tenmin/render/tts.py` | `synthesize_track` gains `reporter=` param + `substep` calls in its per-chunk loop |
| `src/tenmin/render/ffmpeg.py` | New `run_with_progress()` function |
| `src/tenmin/render/video.py` | `render_video` gains `reporter=` param, switches from `run()` to `run_with_progress()` |
| `src/tenmin/cli.py` | Constructs `RichProgressReporter`, passes `reporter=` into `run_pipeline` |
| `tests/test_pipeline.py` | New tests for stage events, episode_start, voice/render substep forwarding |
| `tests/test_render_tts.py` | New test for substep reporting |
| `tests/test_render_ffmpeg.py` | New `FakePopen` + tests for `run_with_progress` |
| `tests/test_render_video.py` | Modify existing ffmpeg-invocation test to patch `run_with_progress`; add substep test |
| `tests/test_cli.py` | New test asserting a `ProgressReporter` is passed to `run_pipeline` |

---

### Task 1: 依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the `rich` dependency**

In `pyproject.toml`, in the `[project] dependencies` list, add a new line right after `"edge-tts>=6.1",`:

```toml
dependencies = [
    "typer>=0.12",
    "pydantic>=2.7",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "charset-normalizer>=3.3",
    "opencc-python-reimplemented>=0.1.7",
    "google-genai>=1.33",
    "httpx>=0.27",
    "edge-tts>=6.1",
    "rich>=13",
]
```

- [ ] **Step 2: Sync and verify import**

Run: `uv sync`
Then run: `uv run python -c "import rich; print(rich.__version__)"`
Expected: prints a version string (e.g. `13.9.4`), no error.

- [ ] **Step 3: Verify no regressions**

Run: `uv run pytest -q`
Expected: same pass/skip counts as before this change (no new failures).

Run: `uv run ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add rich dependency for progress bars"
```

---

### Task 2: `ProgressReporter` Protocol + `NullProgressReporter`

**Files:**
- Create: `src/tenmin/progress.py`
- Test: `tests/test_progress.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_progress.py`:

```python
from tenmin.progress import NullProgressReporter, ProgressReporter


def test_null_reporter_is_protocol_conformant():
    assert isinstance(NullProgressReporter(), ProgressReporter)


def test_null_reporter_accepts_all_calls_without_error():
    reporter = NullProgressReporter()
    reporter.stage_start("ingest")
    reporter.stage_skip("signals")
    reporter.stage_done("script")
    reporter.substep("voice", 1, 3, "第一句")
    reporter.episode_start(2, 1, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_progress.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenmin.progress'`

- [ ] **Step 3: Write the implementation**

Create `src/tenmin/progress.py`:

```python
"""进度上报的抽象接口。pipeline.py 和 render/*.py 只依赖这个 Protocol，
从不直接依赖 rich，这样它们离线测试时用一个 FakeReporter 就够了。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressReporter(Protocol):
    def stage_start(self, stage: str) -> None:
        """某阶段真的开始执行了（不是被 mtime 跳过）。"""
        ...

    def stage_skip(self, stage: str) -> None:
        """某阶段因为产物已是最新，被跳过。"""
        ...

    def stage_done(self, stage: str) -> None:
        """某阶段执行完毕。"""
        ...

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        """阶段内部的细粒度进度。目前只有 voice（按句）和 render（按 ffmpeg 百分比）会调用。"""
        ...

    def episode_start(self, number: int, index: int, total: int) -> None:
        """批量模式下，开始处理第几集（共几集）。单集模式不会调用这个方法。"""
        ...


class NullProgressReporter:
    """默认的空实现。所有方法都不做事，保证不传 reporter 时行为完全不变。"""

    def stage_start(self, stage: str) -> None:
        pass

    def stage_skip(self, stage: str) -> None:
        pass

    def stage_done(self, stage: str) -> None:
        pass

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        pass

    def episode_start(self, number: int, index: int, total: int) -> None:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_progress.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/progress.py tests/test_progress.py
git commit -m "feat(progress): add ProgressReporter protocol and null implementation"
```

---

### Task 3: `FakeReporter`

**Files:**
- Modify: `tests/fakes.py`
- Test: `tests/test_progress.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_progress.py`:

```python
from .fakes import FakeReporter


def test_fake_reporter_satisfies_protocol():
    assert isinstance(FakeReporter(), ProgressReporter)


def test_fake_reporter_records_every_call():
    reporter = FakeReporter()
    reporter.stage_start("ingest")
    reporter.stage_skip("signals")
    reporter.stage_done("script")
    reporter.substep("voice", 1, 3, "第一句")
    reporter.episode_start(2, 1, 3)
    assert reporter.calls == [
        ("stage_start", "ingest"),
        ("stage_skip", "signals"),
        ("stage_done", "script"),
        ("substep", "voice", 1, 3, "第一句"),
        ("episode_start", 2, 1, 3),
    ]
```

Also move the `from tenmin.progress import NullProgressReporter, ProgressReporter` line already at the top of `tests/test_progress.py` so that the new `from .fakes import FakeReporter` line sits directly below it (standard two-import-groups convention already used elsewhere in the test suite: third-party/stdlib group first, blank line, then relative-import group).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_progress.py -v`
Expected: FAIL with `ImportError: cannot import name 'FakeReporter' from 'tests.fakes'`

- [ ] **Step 3: Write the implementation**

Append to `tests/fakes.py` (at the end of the file):

```python
class FakeReporter:
    """测试用假 progress reporter。记录每次调用，方便断言顺序和参数。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def stage_start(self, stage: str) -> None:
        self.calls.append(("stage_start", stage))

    def stage_skip(self, stage: str) -> None:
        self.calls.append(("stage_skip", stage))

    def stage_done(self, stage: str) -> None:
        self.calls.append(("stage_done", stage))

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        self.calls.append(("substep", stage, current, total, label))

    def episode_start(self, number: int, index: int, total: int) -> None:
        self.calls.append(("episode_start", number, index, total))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_progress.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/fakes.py tests/test_progress.py
git commit -m "test: add FakeReporter for offline progress-reporting tests"
```

---

### Task 4: Wire stage events + `episode_start` into `run_pipeline`

**Files:**
- Modify: `src/tenmin/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_pipeline.py`, change the import line:

```python
from .fakes import FakeProvider, FakeTTSEngine
```

to:

```python
from .fakes import FakeProvider, FakeReporter, FakeTTSEngine
```

Then append these three tests to the end of the file:

```python
@pytest.mark.asyncio
async def test_run_pipeline_reports_stage_start_and_done(project):
    reporter = FakeReporter()
    provider = FakeProvider([fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    calls = reporter.calls
    assert ("stage_start", "ingest") in calls
    assert ("stage_done", "ingest") in calls
    assert ("stage_start", "signals") in calls
    assert ("stage_done", "signals") in calls
    assert ("episode_start", 2, 1, 1) in calls
    assert ("stage_start", "script") in calls
    assert ("stage_done", "script") in calls
    assert ("stage_start", "docgen") in calls
    assert ("stage_done", "docgen") in calls
    # ingest 必须先于 signals，signals 必须先于 script
    assert calls.index(("stage_done", "ingest")) < calls.index(("stage_start", "signals"))
    assert calls.index(("stage_done", "signals")) < calls.index(("stage_start", "script"))


@pytest.mark.asyncio
async def test_run_pipeline_reports_stage_skip_on_second_run(project):
    provider = FakeProvider([fake_script_response(), fake_script_response()])
    await run_pipeline(project, provider, only=V1_STAGES)
    reporter = FakeReporter()
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    calls = reporter.calls
    assert ("stage_skip", "ingest") in calls
    assert ("stage_skip", "signals") in calls
    assert ("stage_skip", "script") in calls
    assert ("stage_skip", "docgen") in calls
    assert ("stage_start", "ingest") not in calls
    assert ("stage_start", "script") not in calls


@pytest.mark.asyncio
async def test_run_pipeline_batch_mode_reports_episode_start_for_each_episode(project):
    project.episodes.append(EpisodeConfig(number=1, srt=project.episodes[0].srt))
    reporter = FakeReporter()
    provider = FakeProvider([fake_script_response(episode=2), fake_script_response(episode=1)])
    await run_pipeline(project, provider, only=V1_STAGES, reporter=reporter)
    calls = reporter.calls
    assert ("episode_start", 2, 1, 2) in calls
    assert ("episode_start", 1, 2, 2) in calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -k "reports_stage or batch_mode_reports_episode" -v`
Expected: FAIL with `TypeError: run_pipeline() got an unexpected keyword argument 'reporter'`

- [ ] **Step 3: Write the implementation**

In `src/tenmin/pipeline.py`, add a new import line. The existing import block has (in order):

```python
from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.docgen.narration import render_narration
from tenmin.docgen.table import render_table
from tenmin.ingest.normalize import build_track
from tenmin.models import DialogueTrack, Script, SignalReport, Timeline, VoiceTrack
from tenmin.render.audio import mix_audio
```

Insert `from tenmin.progress import NullProgressReporter, ProgressReporter` between the `tenmin.models` line and the `tenmin.render.audio` line:

```python
from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.docgen.narration import render_narration
from tenmin.docgen.table import render_table
from tenmin.ingest.normalize import build_track
from tenmin.models import DialogueTrack, Script, SignalReport, Timeline, VoiceTrack
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.audio import mix_audio
```

Now rewrite `run_pipeline`'s body. Replace the entire function (from `async def run_pipeline(` through the final `return warnings`) with:

```python
async def run_pipeline(
    cfg: ProjectConfig,
    provider: LLMProvider,
    *,
    from_stage: str = "ingest",
    only: Sequence[str] | None = None,
    force: bool = False,
    tts_engine: TTSEngine | None = None,
    episode: int | None = None,
    reporter: ProgressReporter | None = None,
) -> list[str]:
    """返回本次运行累积的 warnings。

    only 传阶段名序列，只跑这些阶段（CLI 的 --only 传单元素列表，
    端到端测试会传 ["ingest", "signals"] 这样的多元素列表）。
    episode 传具体集数只跑那一集；不传则批量跑 cfg.episodes 里注册的所有集。
    """
    if cfg.mode == "season":
        raise NotImplementedError("整季模式尚未实现，请使用 mode: single_episode")

    reporter = reporter or NullProgressReporter()

    if only is not None:
        unknown = [stage for stage in only if stage not in STAGES]
        if unknown:
            raise ValueError(f"未知阶段 {unknown[0]!r}，可选：{', '.join(STAGES)}")
        wanted = [stage for stage in STAGES if stage in only]
    else:
        wanted = stages_from(from_stage)

    paths = Paths(cfg.root)
    numbers = [ep.number for ep in cfg.episodes]
    srt_inputs = [cfg.srt_path(ep) for ep in cfg.episodes]
    warnings: list[str] = []

    if episode is None:
        target_numbers = numbers
    else:
        _find_episode(cfg, episode)  # 找不到会抛 ValueError（"没有注册"）
        target_numbers = [episode]

    number_to_index = {number: idx for idx, number in enumerate(target_numbers, start=1)}

    if "ingest" in wanted:
        outputs = [paths.dialogue(n) for n in numbers]
        if force or not _is_fresh(outputs, srt_inputs):
            reporter.stage_start("ingest")
            run_ingest(cfg)
            reporter.stage_done("ingest")
        else:
            reporter.stage_skip("ingest")

    if "signals" in wanted:
        outputs = [paths.signals(n) for n in numbers]
        inputs = [paths.dialogue(n) for n in numbers]
        if force or not _is_fresh(outputs, inputs):
            reporter.stage_start("signals")
            run_signals(cfg)
            reporter.stage_done("signals")
        else:
            reporter.stage_skip("signals")

    for number in target_numbers:
        if episode is None:
            reporter.episode_start(number, number_to_index[number], len(target_numbers))

        if "script" in wanted:
            inputs = [paths.dialogue(number), paths.signals(number)]
            if force or not _is_fresh([paths.script(number)], inputs):
                reporter.stage_start("script")
                _, stage_warnings = await run_script(cfg, provider, episode=number)
                warnings.extend(stage_warnings)
                reporter.stage_done("script")
            else:
                reporter.stage_skip("script")

        if "docgen" in wanted:
            outputs = [paths.table(number), paths.narration(number)]
            if force or not _is_fresh(outputs, [paths.script(number)]):
                reporter.stage_start("docgen")
                run_docgen(cfg, episode=number)
                reporter.stage_done("docgen")
            else:
                reporter.stage_skip("docgen")

    # 前置检查放在 voice 之前，批量模式下要给每一集都做前置检查
    if {"audio", "render"} & set(wanted):
        for number in target_numbers:
            preflight(cfg.video_path(_find_episode(cfg, number)), cfg.render.video_encoder)

    for number in target_numbers:
        if episode is None:
            reporter.episode_start(number, number_to_index[number], len(target_numbers))

        if "voice" in wanted:
            outputs = [paths.voice(number)]
            if force or not _is_fresh(outputs, [paths.script(number)]):
                reporter.stage_start("voice")
                assert tts_engine is not None
                _, stage_warnings = await run_voice(cfg, tts_engine, episode=number)
                warnings.extend(stage_warnings)
                reporter.stage_done("voice")
            else:
                reporter.stage_skip("voice")

        if "timeline" in wanted:
            outputs = [paths.timeline(number), paths.subtitles(number)]
            inputs = [paths.script(number), paths.voice(number)]
            if force or not _is_fresh(outputs, inputs):
                reporter.stage_start("timeline")
                _, stage_warnings = run_timeline(cfg, episode=number)
                warnings.extend(stage_warnings)
                reporter.stage_done("timeline")
            else:
                reporter.stage_skip("timeline")

        if "audio" in wanted:
            outputs = [paths.mixed_audio(number)]
            inputs = [paths.timeline(number), paths.voice(number)]
            if force or not _is_fresh(outputs, inputs):
                reporter.stage_start("audio")
                run_audio(cfg, episode=number)
                reporter.stage_done("audio")
            else:
                reporter.stage_skip("audio")

        if "render" in wanted:
            outputs = [paths.video(number)]
            inputs = [paths.mixed_audio(number), paths.subtitles(number), paths.timeline(number)]
            if force or not _is_fresh(outputs, inputs):
                reporter.stage_start("render")
                run_render(cfg, episode=number)
                reporter.stage_done("render")
            else:
                reporter.stage_skip("render")

    return warnings
```

Note: `run_voice(cfg, tts_engine, episode=number)` and `run_render(cfg, episode=number)` calls are left UNCHANGED in this task (no `reporter=` forwarded yet) — that forwarding is added in Tasks 5 and 7 respectively, once those functions actually know what to do with a reporter.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -k "reports_stage or batch_mode_reports_episode" -v`
Expected: `3 passed`

Run: `uv run pytest -q`
Expected: same total pass count as before plus 3 (no regressions — every existing test that calls `run_pipeline` without `reporter=` gets the `NullProgressReporter` default and behaves identically).

Run: `uv run ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): report stage start/skip/done and episode progress"
```

---

### Task 5: Voice sub-progress

**Files:**
- Modify: `src/tenmin/render/tts.py`
- Modify: `src/tenmin/pipeline.py`
- Test: `tests/test_render_tts.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_render_tts.py`, change the import line:

```python
from .fakes import FakeTTSEngine, FlakyTTSEngine
```

to:

```python
from .fakes import FakeReporter, FakeTTSEngine, FlakyTTSEngine
```

Then append this test to the end of the file:

```python
@pytest.mark.asyncio
async def test_synthesize_track_reports_substep_progress(tmp_path):
    engine = FakeTTSEngine([3.0, 4.0, 5.0])
    reporter = FakeReporter()
    await synthesize_track(sample_script(), 2, tmp_path, engine, reporter=reporter)
    assert reporter.calls == [
        ("substep", "voice", 1, 3, "第一句。"),
        ("substep", "voice", 2, 3, "第二句。"),
        ("substep", "voice", 3, 3, "第三句。"),
    ]
```

In `tests/test_pipeline.py`, append this test:

```python
@pytest.mark.asyncio
async def test_run_voice_reports_substep_progress(project):
    _write_script(Paths(project.root).script(2), render_script())
    engine = FakeTTSEngine([8.0, 10.0, 10.0])
    reporter = FakeReporter()
    await run_voice(project, engine, episode=2, reporter=reporter)
    substeps = [call for call in reporter.calls if call[0] == "substep"]
    assert len(substeps) == 3
    assert substeps[-1] == ("substep", "voice", 3, 3, "第三句。")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_render_tts.py::test_synthesize_track_reports_substep_progress -v`
Expected: FAIL with `TypeError: synthesize_track() got an unexpected keyword argument 'reporter'`

- [ ] **Step 3: Write the implementation**

In `src/tenmin/render/tts.py`, change the import lines from:

```python
from tenmin.config import RenderConfig
from tenmin.models import Script, VoiceChunk, VoiceTrack
from tenmin.render.chunks import plan_chunks
from tenmin.render.ffmpeg import probe_duration
```

to:

```python
from tenmin.config import RenderConfig
from tenmin.models import Beat, Script, VoiceChunk, VoiceTrack
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.chunks import plan_chunks
from tenmin.render.ffmpeg import probe_duration
```

Replace `synthesize_track`'s signature and body entirely:

```python
async def synthesize_track(
    script: Script,
    episode: int,
    voice_dir: Path,
    engine: TTSEngine,
    *,
    reuse: bool = True,
    reporter: ProgressReporter | None = None,
) -> tuple[VoiceTrack, list[str]]:
    reporter = reporter or NullProgressReporter()
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    planned_by_beat: list[tuple[Beat, list[tuple[str, float]]]] = []
    for beat in script.beats:
        planned = plan_chunks(beat)
        if not planned:
            warnings.append(f"beat {beat.id} 没有旁白文本，已跳过配音")
            continue
        planned_by_beat.append((beat, planned))

    total_chunks = sum(len(planned) for _, planned in planned_by_beat)

    chunks: list[VoiceChunk] = []
    serial = 0
    for beat, planned in planned_by_beat:
        for index, (text, hold_after) in enumerate(planned, start=1):
            serial += 1
            reporter.substep("voice", serial, total_chunks, text[:20])
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

Note the substep label uses `text[:20]` — for the fixture's short sentences (`"第一句。"` etc.) this is the whole string verbatim, which is why the test above asserts exact matches.

Now update `src/tenmin/pipeline.py`. Change `run_voice`'s signature and body from:

```python
async def run_voice(cfg: ProjectConfig, engine: TTSEngine, episode: int) -> tuple[VoiceTrack, list[str]]:
    paths = Paths(cfg.root)
    script = _load_script(cfg, episode)
    track, warnings = await synthesize_track(script, episode, paths.voice_dir(episode), engine)
    paths.voice(episode).parent.mkdir(parents=True, exist_ok=True)
    paths.voice(episode).write_text(track.model_dump_json(indent=2), encoding="utf-8")
    return track, warnings
```

to (adding the `reporter` parameter and forwarding it):

```python
async def run_voice(
    cfg: ProjectConfig,
    engine: TTSEngine,
    episode: int,
    reporter: ProgressReporter | None = None,
) -> tuple[VoiceTrack, list[str]]:
    paths = Paths(cfg.root)
    script = _load_script(cfg, episode)
    track, warnings = await synthesize_track(
        script, episode, paths.voice_dir(episode), engine, reporter=reporter
    )
    paths.voice(episode).parent.mkdir(parents=True, exist_ok=True)
    paths.voice(episode).write_text(track.model_dump_json(indent=2), encoding="utf-8")
    return track, warnings
```

(If the exact body of `run_voice` you find in the file differs slightly in variable names from the snippet above, keep the existing body's logic exactly as-is — only add the `reporter` parameter to the signature and add `reporter=reporter` to the `synthesize_track(...)` call.)

In the same file, inside `run_pipeline`'s `"voice" in wanted` block (written in Task 4), change:

```python
                _, stage_warnings = await run_voice(cfg, tts_engine, episode=number)
```

to:

```python
                _, stage_warnings = await run_voice(cfg, tts_engine, episode=number, reporter=reporter)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_render_tts.py tests/test_pipeline.py -k "substep_progress" -v`
Expected: `2 passed`

Run: `uv run pytest -q`
Expected: no regressions.

Run: `uv run ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/tts.py src/tenmin/pipeline.py tests/test_render_tts.py tests/test_pipeline.py
git commit -m "feat(render): report per-chunk progress during voice synthesis"
```

---

### Task 6: `run_with_progress` — streaming ffmpeg progress

**Files:**
- Modify: `src/tenmin/render/ffmpeg.py`
- Test: `tests/test_render_ffmpeg.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render_ffmpeg.py`:

```python
class FakePopen:
    """假的 subprocess.Popen，逐行喂 stdout，不真的起进程。"""

    class _Stderr:
        def __init__(self, text: str):
            self._text = text

        def read(self) -> str:
            return self._text

    def __init__(self, lines: list[str], returncode: int = 0, stderr: str = ""):
        self.stdout = iter(lines)
        self.stderr = FakePopen._Stderr(stderr)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def test_run_with_progress_reports_fraction_from_out_time_ms(monkeypatch):
    lines = [
        "frame=1\n",
        "out_time_ms=5000000\n",
        "progress=continue\n",
        "out_time_ms=10000000\n",
        "progress=end\n",
    ]
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(lines),
    )
    seen: list[float] = []
    run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=20.0, on_progress=seen.append)
    assert seen == [0.25, 0.5, 1.0]


def test_run_with_progress_raises_with_stderr_tail(monkeypatch):
    noise = "\n".join(f"line {i}" for i in range(50))
    stderr = noise + "\nInvalid argument\n"
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(["progress=end\n"], returncode=1, stderr=stderr),
    )
    with pytest.raises(FFmpegError) as exc:
        run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=10.0)
    message = str(exc.value)
    assert "Invalid argument" in message
    assert "line 0" not in message


def test_run_with_progress_clamps_fraction_to_one(monkeypatch):
    lines = ["out_time_ms=999999999\n"]
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(lines),
    )
    seen: list[float] = []
    run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=5.0, on_progress=seen.append)
    assert seen == [1.0]


def test_run_with_progress_works_without_on_progress_callback(monkeypatch):
    monkeypatch.setattr(
        "tenmin.render.ffmpeg.subprocess.Popen",
        lambda args, **kwargs: FakePopen(["out_time_ms=1000000\n", "progress=end\n"]),
    )
    result = run_with_progress(["-i", "in.mp4", "out.mp4"], total_seconds=1.0)
    assert result == ""
```

Also update the import line at the top of `tests/test_render_ffmpeg.py`. It currently reads:

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
```

Change it to add `run_with_progress` alphabetically after `run`:

```python
from tenmin.render.ffmpeg import (
    FFmpegError,
    has_encoder,
    has_filter,
    parse_names,
    preflight,
    probe_duration,
    run,
    run_with_progress,
    tail,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_render_ffmpeg.py -k run_with_progress -v`
Expected: FAIL with `ImportError: cannot import name 'run_with_progress' from 'tenmin.render.ffmpeg'`

- [ ] **Step 3: Write the implementation**

In `src/tenmin/render/ffmpeg.py`, change the import block from:

```python
from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path
```

to (adding `Callable`):

```python
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
```

Append this function to the end of the file (after `preflight`):

```python
def run_with_progress(
    args: list[str],
    *,
    total_seconds: float,
    on_progress: Callable[[float], None] | None = None,
) -> str:
    """跟 run() 一样跑 ffmpeg，但额外加 -progress pipe:1，流式解析进度，
    每读到一条 out_time_ms 就换算成 0.0~1.0 的比例回调 on_progress。

    坑：ffmpeg 的 out_time_ms 字段名字带 "ms"，但实际单位是微秒（众所周知的
    ffmpeg 老 bug/历史遗留），所以换算要除以 1_000_000 而不是 1_000。
    """
    process = subprocess.Popen(
        [FFMPEG, *args, "-progress", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    for line in process.stdout:
        line = line.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key == "out_time_ms":
            try:
                microseconds = int(value)
            except ValueError:
                continue
            if on_progress is not None and total_seconds > 0:
                fraction = microseconds / 1_000_000 / total_seconds
                on_progress(max(0.0, min(1.0, fraction)))
        elif key == "progress" and value == "end":
            if on_progress is not None:
                on_progress(1.0)

    stderr = process.stderr.read()
    returncode = process.wait()
    if returncode != 0:
        command = " ".join([FFMPEG, *args])
        raise FFmpegError(
            f"ffmpeg 执行失败（退出码 {returncode}）：{command}\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(stderr)}"
        )
    return stderr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_render_ffmpeg.py -k run_with_progress -v`
Expected: `4 passed`

Run: `uv run pytest -q`
Expected: no regressions.

Run: `uv run ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/ffmpeg.py tests/test_render_ffmpeg.py
git commit -m "feat(render): add run_with_progress for streaming ffmpeg percent-complete"
```

---

### Task 7: Render sub-progress

**Files:**
- Modify: `src/tenmin/render/video.py`
- Modify: `src/tenmin/pipeline.py`
- Test: `tests/test_render_video.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_render_video.py`, find the existing test:

```python
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

Replace it with (patching `run_with_progress` instead of `run`):

```python
def test_render_video_runs_ffmpeg_and_returns_path(tmp_path, monkeypatch):
    from tenmin.render import video as video_module

    seen: list[list[str]] = []

    def fake_run_with_progress(args, *, total_seconds, on_progress=None):
        seen.append(list(args))
        return ""

    monkeypatch.setattr(video_module, "run_with_progress", fake_run_with_progress)
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

Then append a new test to the end of the file:

```python
def test_render_video_reports_substep_progress(tmp_path, monkeypatch):
    from tenmin.render import video as video_module

    from .fakes import FakeReporter

    captured: dict[str, float] = {}

    def fake_run_with_progress(args, *, total_seconds, on_progress=None):
        captured["total_seconds"] = total_seconds
        if on_progress is not None:
            on_progress(0.5)
        return ""

    monkeypatch.setattr(video_module, "run_with_progress", fake_run_with_progress)
    reporter = FakeReporter()
    render_video(
        video=tmp_path / "source.mkv",
        timeline=make_timeline(),
        audio=tmp_path / "06_audio" / "E02.mixed.m4a",
        ass=tmp_path / "05_timeline" / "E02.ass",
        out_path=tmp_path / "07_render" / "E02.mp4",
        encoder="libx264",
        reporter=reporter,
    )
    assert captured["total_seconds"] == pytest.approx(30.0)
    assert reporter.calls == [("substep", "render", 50, 100, "")]
```

In `tests/test_pipeline.py`, append this test:

```python
def test_run_render_reports_substep_progress(project, monkeypatch):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    engine = FakeTTSEngine([8.0, 10.0, 10.0])
    asyncio.run(run_voice(project, engine, episode=2))
    run_timeline(project, episode=2, source_duration=1400.0)
    _prepare_video(project)
    paths.mixed_audio(2).parent.mkdir(parents=True, exist_ok=True)
    paths.mixed_audio(2).write_bytes(b"")

    def fake_run_with_progress(args, *, total_seconds, on_progress=None):
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        if on_progress is not None:
            on_progress(1.0)
        return ""

    monkeypatch.setattr("tenmin.render.video.run_with_progress", fake_run_with_progress)
    reporter = FakeReporter()
    run_render(project, episode=2, reporter=reporter)
    assert ("substep", "render", 100, 100, "") in reporter.calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_render_video.py::test_render_video_reports_substep_progress -v`
Expected: FAIL with `TypeError: render_video() got an unexpected keyword argument 'reporter'`

- [ ] **Step 3: Write the implementation**

In `src/tenmin/render/video.py`, change the import block from:

```python
from __future__ import annotations

from pathlib import Path

from tenmin.models import Timeline
from tenmin.render.ffmpeg import run
```

to:

```python
from __future__ import annotations

from pathlib import Path

from tenmin.models import Timeline
from tenmin.progress import NullProgressReporter, ProgressReporter
from tenmin.render.ffmpeg import run_with_progress
```

Change `render_video`'s signature and body. It currently looks like:

```python
def render_video(
    *,
    video: Path,
    timeline: Timeline,
    audio: Path,
    ass: Path,
    out_path: Path,
    encoder: str,
    fade_out_seconds: float = 0.0,
    outro_seconds: float = 0.0,
    outro_title: str = "",
    outro_message: str = "",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = build_render_args(
        video=video,
        timeline=timeline,
        audio=audio,
        ass=ass,
        out_path=out_path,
        encoder=encoder,
        fade_out_seconds=fade_out_seconds,
        outro_seconds=outro_seconds,
        outro_title=outro_title,
        outro_message=outro_message,
    )
    run(args)
    return out_path
```

Replace it with:

```python
def render_video(
    *,
    video: Path,
    timeline: Timeline,
    audio: Path,
    ass: Path,
    out_path: Path,
    encoder: str,
    fade_out_seconds: float = 0.0,
    outro_seconds: float = 0.0,
    outro_title: str = "",
    outro_message: str = "",
    reporter: ProgressReporter | None = None,
) -> Path:
    reporter = reporter or NullProgressReporter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = build_render_args(
        video=video,
        timeline=timeline,
        audio=audio,
        ass=ass,
        out_path=out_path,
        encoder=encoder,
        fade_out_seconds=fade_out_seconds,
        outro_seconds=outro_seconds,
        outro_title=outro_title,
        outro_message=outro_message,
    )
    total_seconds = timeline.total_seconds + outro_seconds

    def _on_progress(fraction: float) -> None:
        reporter.substep("render", int(fraction * 100), 100, "")

    run_with_progress(args, total_seconds=total_seconds, on_progress=_on_progress)
    return out_path
```

(If the existing body's exact variable names differ, keep everything else the same — only the two lines `run(args)` -> `run_with_progress(...)` and the added `reporter` parameter/default/`_on_progress` closure matter.)

Now update `src/tenmin/pipeline.py`. Change `run_render`'s signature and body from:

```python
def run_render(cfg: ProjectConfig, episode: int) -> Path:
    episode_cfg = _find_episode(cfg, episode)
    paths = Paths(cfg.root)
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
        fade_out_seconds=cfg.render.fade_out_seconds,
        outro_seconds=cfg.render.outro_card_seconds,
        outro_title=f"{cfg.show} · EP{episode:02d}",
        outro_message=cfg.render.outro_message,
    )
```

to (adding `reporter` and forwarding it):

```python
def run_render(cfg: ProjectConfig, episode: int, reporter: ProgressReporter | None = None) -> Path:
    episode_cfg = _find_episode(cfg, episode)
    paths = Paths(cfg.root)
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
        fade_out_seconds=cfg.render.fade_out_seconds,
        outro_seconds=cfg.render.outro_card_seconds,
        outro_title=f"{cfg.show} · EP{episode:02d}",
        outro_message=cfg.render.outro_message,
        reporter=reporter,
    )
```

(Again, if the exact existing body differs in details unrelated to this change, preserve it — only the signature and the added `reporter=reporter` kwarg on the `render_video(...)` call matter.)

In the same file, inside `run_pipeline`'s `"render" in wanted` block (written in Task 4), change:

```python
                run_render(cfg, episode=number)
```

to:

```python
                run_render(cfg, episode=number, reporter=reporter)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_render_video.py tests/test_pipeline.py -k "render" -v`
Expected: all render-related tests pass, including the two new ones and the modified `test_render_video_runs_ffmpeg_and_returns_path`.

Run: `uv run pytest -q`
Expected: no regressions.

Run: `uv run ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/render/video.py src/tenmin/pipeline.py tests/test_render_video.py tests/test_pipeline.py
git commit -m "feat(render): report ffmpeg encode progress during render"
```

---

### Task 8: `RichProgressReporter` + CLI wiring

**Files:**
- Create: `src/tenmin/rich_progress.py`
- Modify: `src/tenmin/cli.py`
- Test: `tests/test_rich_progress.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rich_progress.py`:

```python
from tenmin.progress import ProgressReporter
from tenmin.rich_progress import RichProgressReporter


def test_rich_progress_reporter_satisfies_protocol():
    assert isinstance(RichProgressReporter(), ProgressReporter)


def test_rich_progress_reporter_handles_full_call_sequence():
    """纯粹的烟雾测试：只断言不抛异常，绝不断言终端渲染出的字符
    （rich 内部怎么画是它自己的事，我们只测「调用对不对」）。"""
    with RichProgressReporter() as reporter:
        reporter.episode_start(1, 1, 2)
        reporter.stage_start("ingest")
        reporter.stage_done("ingest")
        reporter.stage_skip("signals")
        reporter.stage_start("voice")
        reporter.substep("voice", 1, 3, "第一句")
        reporter.substep("voice", 2, 3, "第二句")
        reporter.substep("voice", 3, 3, "第三句")
        reporter.stage_done("voice")
        reporter.stage_start("render")
        reporter.substep("render", 50, 100, "")
        reporter.substep("render", 100, 100, "")
        reporter.stage_done("render")
        reporter.episode_start(2, 2, 2)
        reporter.stage_start("ingest")
        reporter.stage_done("ingest")
```

Append this test to `tests/test_cli.py`:

```python
def test_run_passes_progress_reporter(tmp_path, monkeypatch):
    from tenmin.progress import ProgressReporter

    _minimal_project(tmp_path)
    captured: dict[str, object] = {}

    async def fake_pipeline(cfg, provider, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "akujo2", "--work-dir", str(tmp_path), "--only", "docgen"])
    assert result.exit_code == 0, out(result)
    assert isinstance(captured["reporter"], ProgressReporter)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rich_progress.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenmin.rich_progress'`

- [ ] **Step 3: Write the implementation**

Create `src/tenmin/rich_progress.py`:

```python
"""基于 rich 的进度条实现。全项目唯一一个直接依赖 rich 的模块——
pipeline.py 和 render/*.py 都只依赖 progress.py 的 Protocol，方便离线测试。"""

from __future__ import annotations

from typing import Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

STAGE_LABELS: dict[str, str] = {
    "ingest": "ingest",
    "signals": "signals",
    "script": "script",
    "docgen": "docgen",
    "voice": "voice",
    "timeline": "timeline",
    "audio": "audio",
    "render": "render",
}


class RichProgressReporter:
    """把 ProgressReporter 的 5 个方法映射到 rich.progress.Progress 的任务上。"""

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        self._episode_task: TaskID | None = None
        self._stage_tasks: dict[str, TaskID] = {}
        self._substep_tasks: dict[str, TaskID] = {}

    def __enter__(self) -> RichProgressReporter:
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._progress.__exit__(exc_type, exc, tb)

    def episode_start(self, number: int, index: int, total: int) -> None:
        label = f"总进度：第 {index}/{total} 集 (E{number:02d})"
        if self._episode_task is None:
            self._episode_task = self._progress.add_task(
                label, total=total, completed=index - 1
            )
        else:
            self._progress.update(
                self._episode_task, description=label, completed=index - 1
            )

    def stage_start(self, stage: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        task_id = self._progress.add_task(f"▶ {label}", total=None)
        self._stage_tasks[stage] = task_id

    def stage_skip(self, stage: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        self._progress.add_task(f"⏭ {label}（已是最新，跳过）", total=1, completed=1)

    def stage_done(self, stage: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        task_id = self._stage_tasks.pop(stage, None)
        if task_id is None:
            task_id = self._progress.add_task(f"✓ {label}", total=1, completed=1)
        else:
            self._progress.update(
                task_id, description=f"✓ {label}", total=1, completed=1
            )

    def substep(self, stage: str, current: int, total: int, label: str) -> None:
        description = f"  {stage} {label}".rstrip()
        task_id = self._substep_tasks.get(stage)
        if task_id is None:
            task_id = self._progress.add_task(description, total=total, completed=current)
            self._substep_tasks[stage] = task_id
        else:
            self._progress.update(
                task_id, description=description, total=total, completed=current
            )
```

Now update `src/tenmin/cli.py`. Add a new import line, placed alphabetically after `from tenmin.render.tts import build_tts_engine` and before `from tenmin.script.llm import build_provider`:

```python
from tenmin.render.tts import build_tts_engine
from tenmin.rich_progress import RichProgressReporter
from tenmin.script.llm import build_provider
```

Then change the `try`/`except` block around the pipeline call. It currently reads:

```python
    try:
        warnings = asyncio.run(
            run_pipeline(
                cfg, provider, from_stage=from_stage, only=[only] if only else None,
                force=force, tts_engine=tts_engine, episode=episode,
            )
        )
    except (NotImplementedError, FileNotFoundError, ValueError, FFmpegError) as error:
        typer.secho(str(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
```

Change it to wrap the pipeline call in a `with RichProgressReporter() as reporter:` block *inside* the `try`:

```python
    try:
        with RichProgressReporter() as reporter:
            warnings = asyncio.run(
                run_pipeline(
                    cfg, provider, from_stage=from_stage, only=[only] if only else None,
                    force=force, tts_engine=tts_engine, episode=episode, reporter=reporter,
                )
            )
    except (NotImplementedError, FileNotFoundError, ValueError, FFmpegError) as error:
        typer.secho(str(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
```

Putting the `with` block inside the `try` (rather than the other way around) matters: if `run_pipeline` raises one of the caught exceptions, Python unwinds the `with` block's `__exit__` (which calls `rich.Progress.__exit__`, restoring the terminal — cursor visibility, etc.) *before* control reaches the `except` clause. This means the progress display always cleans up properly, then the existing red-stderr error line prints exactly as before.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rich_progress.py tests/test_cli.py -v`
Expected: all pass, including the 2 new `test_rich_progress.py` tests and the new `test_run_passes_progress_reporter` in `test_cli.py`.

Run: `uv run pytest -q`
Expected: no regressions — total pass count increases by exactly the new tests added across all 8 tasks.

Run: `uv run ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/rich_progress.py src/tenmin/cli.py tests/test_rich_progress.py tests/test_cli.py
git commit -m "feat(cli): show a rich progress bar while tenmin run executes"
```

---

## Known Limitations (carried over from the design doc)

1. Render's percentage comes from `out_time_ms / total_seconds`. If the timeline was clamped (audio longer than picture — a pre-existing, separately-tracked issue), `total_seconds` is the audio duration while ffmpeg's own internal encode timeline may not perfectly match near the end. This can cause a minor display glitch (percentage jumping or reaching 100% slightly early); it does not affect whether the render itself succeeds.
2. This design only touches `tenmin run`. `tenmin init` and `tenmin inspect` are unaffected — no progress bar is added to them.
3. No pause/cancel interactivity beyond what already exists: Ctrl-C exits normally (rich's own context-manager cleanup handles terminal restoration); there is no custom confirm-exit prompt.
