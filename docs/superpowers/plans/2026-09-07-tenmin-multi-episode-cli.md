# Multi-Episode CLI Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one tenmin project (one `project.yaml`) represent a whole anime series and manage multiple episodes, with `tenmin run <slug> --episode N --srt PATH --video PATH` registering + running a new episode, `tenmin run <slug> --episode N` re-running an already-registered episode, and bare `tenmin run <slug>` batch-processing every registered episode.

**Architecture:** Convert the three currently-shared `Paths` properties (`script`, `table`, `narration`) into per-episode-keyed methods. Replace `_only_episode(cfg)` (which hardcoded `cfg.episodes[0]`) with `_find_episode(cfg, episode_number)`. Thread an explicit `episode: int` parameter through `run_script`, `run_docgen`, `run_voice`, `run_timeline`, `run_audio`, `run_render`. Give the top-level `run_pipeline` orchestrator an `episode: int | None = None` parameter: `None` means batch mode (loop every `cfg.episodes` entry through stages 3-8), a concrete int means single-episode mode (run stages 3-8 for just that episode). Extend the `tenmin run` Typer command with `--episode`, `--srt`, `--video` options that validate the three invocation modes and, when `--srt`/`--video` are given, copy the files into the project directory and update `project.yaml`'s `episodes:` list before running.

**Tech Stack:** Python 3.14, Typer (CLI), Pydantic (config models), pytest (tests), PyYAML (project.yaml read/write), existing `uv run` tooling.

**Reference spec:** `docs/superpowers/specs/2026-09-07-tenmin-multi-episode-cli-design.md` (approved by user).

**Tooling rule for every verification step in this plan:** always run pipeline/pytest commands with plain `uv run <cmd>` via the bash tool directly. Never use `rtk proxy`, `rtk pytest`, or a bare `rtk` prefix for these — Chinese-language stdout crashes rtk's UTF-8 capture layer. Plain `rtk ls`/`rtk grep`/`rtk git` are fine for simple file/git operations.

---

### Task 1: Convert `Paths.script`/`Paths.table`/`Paths.narration` from properties to per-episode methods

**Files:**
- Modify: `src/tenmin/pipeline.py:45-55`
- Test: `tests/test_pipeline.py:111-121` (`test_paths_layout`)

- [ ] **Step 1: Write the failing test**

Replace the existing `test_paths_layout` test in `tests/test_pipeline.py` (lines 111-121) with:

```python
def test_paths_layout(tmp_path):
    paths = Paths(tmp_path)
    assert paths.script(2).name == "E02.script.json"
    assert paths.script(2).parent.name == "03_script"
    assert paths.table(2).name == "E02.解说方案.md"
    assert paths.table(2).parent.name == "out"
    assert paths.narration(2).name == "E02.narration.txt"
    assert paths.narration(2).parent.name == "out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py::test_paths_layout -v`
Expected: FAIL with `TypeError: 'PosixPath' object is not callable` (since `.script` is still a property, calling `.script(2)` fails).

- [ ] **Step 3: Write minimal implementation**

In `src/tenmin/pipeline.py`, replace lines 45-55 (the three `@property` definitions for `script`, `table`, `narration`) with:

```python
    def script(self, episode: int) -> Path:
        return self.root / "03_script" / f"E{episode:02d}.script.json"

    def table(self, episode: int) -> Path:
        return self.root / "out" / f"E{episode:02d}.解说方案.md"

    def narration(self, episode: int) -> Path:
        return self.root / "out" / f"E{episode:02d}.narration.txt"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py::test_paths_layout -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): make Paths.script/table/narration per-episode methods"
```

---

### Task 2: Replace `_only_episode` with `_find_episode`

**Files:**
- Modify: `src/tenmin/pipeline.py:171-175`
- Test: `tests/test_pipeline.py` (new tests, add near the bottom of the file, after existing helper tests)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py` (append near the end of the file):

```python
def test_find_episode_returns_matching_config(project):
    episode_cfg = _find_episode(project, 2)
    assert episode_cfg.number == 2


def test_find_episode_raises_when_not_registered(project):
    with pytest.raises(ValueError, match="没有注册"):
        _find_episode(project, 99)
```

Make sure `_find_episode` is imported at the top of the test file alongside the other pipeline imports (find the existing `from tenmin.pipeline import (...)` block and add `_find_episode` to it).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -k test_find_episode -v`
Expected: FAIL with `ImportError: cannot import name '_find_episode'`

- [ ] **Step 3: Write minimal implementation**

In `src/tenmin/pipeline.py`, replace the `_only_episode` function (lines 171-175) with:

```python
def _find_episode(cfg: ProjectConfig, episode_number: int) -> EpisodeConfig:
    """按集数查找已注册的 episode 配置。

    找不到时报错并提示用户先用 --srt/--video/--episode 注册。
    """
    for episode in cfg.episodes:
        if episode.number == episode_number:
            return episode
    raise ValueError(
        f"第 {episode_number} 集还没有注册。"
        f"请先用 `tenmin run <slug> --episode {episode_number} "
        "--srt <srt路径> --video <视频路径>` 注册这一集。"
    )
```

Then update every call site that referenced `_only_episode(cfg)` — these will be fixed in Tasks 3-8 below as each function gains an explicit `episode` parameter. For now, just confirm no remaining references to `_only_episode` exist anywhere else:

Run: `rtk grep "_only_episode" src/tenmin/pipeline.py`
Expected at this point: 6 remaining matches inside `run_voice`, `run_timeline`, `run_audio`, `run_render`, and `run_pipeline` — these are fixed in the following tasks, not this one. Do not touch them yet.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -k test_find_episode -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): add _find_episode, keep _only_episode callers pending"
```

---

### Task 3: Add explicit `episode` parameter to `run_script`

**Files:**
- Modify: `src/tenmin/pipeline.py:153-158`
- Test: `tests/test_pipeline.py` (existing tests that call `run_script`)

- [ ] **Step 1: Update the implementation first (this task has no new behavior to test-first; it's a signature change — write the updated call sites as the "test", per Step 2)**

In `src/tenmin/pipeline.py`, replace `run_script` (lines 153-158):

```python
async def run_script(
    cfg: ProjectConfig, provider: LLMProvider, episode: int
) -> tuple[Script, list[str]]:
    tracks = _load_tracks(cfg)
    reports = _load_reports(cfg)
    track = next(t for t in tracks if t.episode == episode)
    report = next(r for r in reports if r.episode == episode)
    script, warnings = await generate_script(cfg, track, report, provider)
    _write_json(Paths(cfg.root).script(episode), script.model_dump_json(indent=2))
    return script, warnings
```

Note: check the actual attribute name for episode number on the `report`/`track` objects before finalizing — grep to confirm:

Run: `rtk grep "class DialogueTrack" -A 10 src/tenmin/models.py` and `rtk grep "class SignalReport" -A 10 src/tenmin/models.py` (or whatever the report class is actually named — confirm via `rtk grep "def run_signals" -A 10 src/tenmin/pipeline.py`) to verify the field is literally `.episode` on both. If the field is named differently (e.g. `.episode_number`), adjust the `next(...)` filters above accordingly.

- [ ] **Step 2: Update existing test call sites in `tests/test_pipeline.py`**

Find every direct call to `run_script(project, provider)` (async calls, likely inside `test_run_pipeline_...` tests that call the orchestrator, and possibly a dedicated `test_run_script_...` test if one exists — grep first):

Run: `rtk grep "run_script(" tests/test_pipeline.py`

For every direct call found (not calls that go through `run_pipeline`), add `, episode=2`, e.g. `await run_script(project, provider, episode=2)`.

- [ ] **Step 3: Run the full pipeline test file to check for breakage**

Run: `uv run pytest tests/test_pipeline.py -v 2>&1 | tail -60`
Expected: Multiple failures at this point (expected — `run_docgen`, `run_voice`, etc. still call `_only_episode` and other now-broken things; `run_pipeline` itself still calls `run_script(cfg, provider)` without `episode=`). This is fine — do not try to make everything pass yet. Confirm specifically that no failure is an `ImportError` or `SyntaxError` (those would indicate a typo), only `TypeError: run_script() missing ...` or similar signature-mismatch errors are expected right now.

- [ ] **Step 4: Commit as a checkpoint (tests still red overall, but this task's isolated change is correct)**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): add explicit episode param to run_script"
```

---

### Task 4: Add explicit `episode` parameter to `run_docgen`

**Files:**
- Modify: `src/tenmin/pipeline.py:161-168`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update implementation**

Replace `run_docgen` (lines 161-168):

```python
def run_docgen(cfg: ProjectConfig, episode: int) -> Script:
    paths = Paths(cfg.root)
    script_path = paths.script(episode)
    if not script_path.exists():
        raise FileNotFoundError(f"缺少剧本 {script_path}，请先跑 script 阶段")
    script = Script.model_validate_json(script_path.read_text())
    _write_text(paths.table(episode), render_table(script))
    _write_text(paths.narration(episode), render_narration(script))
    return script
```

- [ ] **Step 2: Update test call sites**

Run: `rtk grep "run_docgen(" tests/test_pipeline.py`

For the `test_run_docgen_without_script_raises` test (around line 241-243), change `run_docgen(project)` to `run_docgen(project, episode=2)`. For any other direct `run_docgen(...)` call, do the same, and fix any `paths.script`/`paths.table`/`paths.narration` reference in the same test to use `(2)` call form (e.g. `_write_script(paths.script(2), ...)`, `assert paths.table(2).exists()`).

- [ ] **Step 3: Run test file, expect continued red overall (progress check)**

Run: `uv run pytest tests/test_pipeline.py -v 2>&1 | tail -60`
Expected: fewer/different failures than before, still not all green — that's fine, continue.

- [ ] **Step 4: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): add explicit episode param to run_docgen"
```

---

### Task 5: Add explicit `episode` parameter to `run_voice`

**Files:**
- Modify: `src/tenmin/pipeline.py:178-213` (includes `_load_script` at 178-182 and `run_voice` at 199-213)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update `_load_script` to take an episode param**

Replace `_load_script` (lines 178-182):

```python
def _load_script(cfg: ProjectConfig, episode: int) -> Script:
    path = Paths(cfg.root).script(episode)
    if not path.exists():
        raise FileNotFoundError(f"缺少剧本 {path}，请先跑 script 阶段")
    return Script.model_validate_json(path.read_text())
```

- [ ] **Step 2: Update `run_voice`**

Replace `run_voice` (lines 199-213):

```python
async def run_voice(
    cfg: ProjectConfig, engine: TTSEngine, episode: int
) -> tuple[VoiceTrack, list[str]]:
    paths = Paths(cfg.root)
    script = _load_script(cfg, episode)
    track, warnings = await synthesize_track(
        script, episode, paths.voice_dir(episode), engine
    )
    _write_json(paths.voice(episode), track.model_dump_json(indent=2))
    return track, warnings
```

- [ ] **Step 3: Update test call sites**

Run: `rtk grep -n "run_voice(\|_load_script(" tests/test_pipeline.py`

For every direct `run_voice(project, engine)` call, change to `run_voice(project, engine, episode=2)`. For every `_load_script(project)` call (if any exist in tests directly), change to `_load_script(project, 2)`. Also fix any `paths.script` reference in the same tests to `paths.script(2)`.

- [ ] **Step 4: Run test file, progress check**

Run: `uv run pytest tests/test_pipeline.py -v 2>&1 | tail -60`

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): add explicit episode param to run_voice and _load_script"
```

---

### Task 6: Add explicit `episode` parameter to `run_timeline`

**Files:**
- Modify: `src/tenmin/pipeline.py:216-232`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update implementation**

Replace `run_timeline` (lines 216-232):

```python
def run_timeline(
    cfg: ProjectConfig, episode: int, *, source_duration: float | None = None
) -> tuple[Timeline, list[str]]:
    paths = Paths(cfg.root)
    episode_cfg = _find_episode(cfg, episode)
    script = _load_script(cfg, episode)
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
```

- [ ] **Step 2: Update test call sites**

Run: `rtk grep -n "run_timeline(" tests/test_pipeline.py`

For every direct call, e.g. `run_timeline(project, source_duration=1400.0)`, change to `run_timeline(project, episode=2, source_duration=1400.0)`.

- [ ] **Step 3: Run test file, progress check**

Run: `uv run pytest tests/test_pipeline.py -v 2>&1 | tail -60`

- [ ] **Step 4: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): add explicit episode param to run_timeline"
```

---

### Task 7: Add explicit `episode` parameter to `run_audio`

**Files:**
- Modify: `src/tenmin/pipeline.py:235-250`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update implementation**

Replace `run_audio` (lines 235-250):

```python
def run_audio(cfg: ProjectConfig, episode: int) -> Path:
    paths = Paths(cfg.root)
    episode_cfg = _find_episode(cfg, episode)
    timeline = _load_timeline(cfg, episode)
    track = _load_voice(cfg, episode)
    return mix_audio(
        video=cfg.video_path(episode_cfg),
        timeline=timeline,
        track=track,
        voice_dir=paths.voice_dir(episode),
        out_path=paths.mixed_audio(episode),
        duck_db=cfg.render.duck_db,
        fade_out_seconds=cfg.render.fade_out_seconds,
        outro_seconds=cfg.render.outro_card_seconds,
    )
```

- [ ] **Step 2: Update test call sites**

Run: `rtk grep -n "run_audio(" tests/test_pipeline.py`

For every direct call, e.g. `run_audio(project)`, change to `run_audio(project, episode=2)`.

- [ ] **Step 3: Run test file, progress check**

Run: `uv run pytest tests/test_pipeline.py -v 2>&1 | tail -60`

- [ ] **Step 4: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): add explicit episode param to run_audio"
```

---

### Task 8: Add explicit `episode` parameter to `run_render`

**Files:**
- Modify: `src/tenmin/pipeline.py:253-275`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update implementation**

Replace `run_render` (lines 253-275):

```python
def run_render(cfg: ProjectConfig, episode: int) -> Path:
    paths = Paths(cfg.root)
    episode_cfg = _find_episode(cfg, episode)
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

- [ ] **Step 2: Update test call sites**

Run: `rtk grep -n "run_render(" tests/test_pipeline.py`

For every direct call, e.g. `run_render(project)`, change to `run_render(project, episode=2)`.

- [ ] **Step 3: Confirm no more `_only_episode` references remain**

Run: `rtk grep "_only_episode" src/tenmin/pipeline.py`
Expected: only remaining match(es) should be inside `run_pipeline` itself (fixed in Task 9). If any match is inside `run_script`/`run_docgen`/`run_voice`/`run_timeline`/`run_audio`/`run_render`, that's a bug in this or an earlier task — go back and fix it.

- [ ] **Step 4: Run test file, progress check**

Run: `uv run pytest tests/test_pipeline.py -v 2>&1 | tail -80`
Expected: most failures should now be in `run_pipeline`-orchestrated tests (Task 9 territory) — direct single-stage-function tests should mostly be passing now.

- [ ] **Step 5: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): add explicit episode param to run_render"
```

---

### Task 9: Give `run_pipeline` an `episode: int | None = None` parameter with batch/single mode logic

**Files:**
- Modify: `src/tenmin/pipeline.py:278-366`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write a new failing test for batch mode with multiple episodes**

Add to `tests/test_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_run_pipeline_batch_mode_processes_all_episodes(project, golden_srt_path):
    # register a second episode by copying the same golden SRT under a new number
    second_srt = project.root / "srt" / "E01.srt"
    second_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    project.episodes.append(EpisodeConfig(number=1, srt=Path("srt/E01.srt")))

    provider = FakeProvider(fake_script_response())
    warnings = await run_pipeline(project, provider, only=V1_STAGES)

    paths = Paths(project.root)
    assert paths.script(1).exists()
    assert paths.script(2).exists()
    assert paths.table(1).exists()
    assert paths.table(2).exists()


@pytest.mark.asyncio
async def test_run_pipeline_single_episode_mode_processes_only_that_episode(
    project, golden_srt_path
):
    second_srt = project.root / "srt" / "E01.srt"
    second_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    project.episodes.append(EpisodeConfig(number=1, srt=Path("srt/E01.srt")))

    provider = FakeProvider(fake_script_response())
    await run_pipeline(project, provider, only=V1_STAGES, episode=1)

    paths = Paths(project.root)
    assert paths.script(1).exists()
    assert not paths.script(2).exists()
```

Check the imports at the top of `tests/test_pipeline.py` — add `EpisodeConfig` to the existing `from tenmin.config import (...)` block if it isn't already imported, and confirm `golden_srt_path` is an existing fixture (grep for it: `rtk grep "golden_srt_path" tests/test_pipeline.py` — if it doesn't exist under that exact name, find the actual fixture/variable name used to build the `srt/E02.srt` file in the `project` fixture and adapt these two new tests to reuse that same source, e.g. by reading `project.root / "srt" / "E02.srt"` as the source content instead of a separate fixture).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -k "batch_mode or single_episode_mode" -v`
Expected: FAIL — `test_run_pipeline_batch_mode_processes_all_episodes` fails because `paths.script(1)` doesn't exist (current code only processes `cfg.episodes[0]`, i.e. episode 2, and ignores episode 1 entirely since `run_script`/`run_docgen` inside `run_pipeline` haven't been updated yet); `test_run_pipeline_single_episode_mode_processes_only_that_episode` fails with `TypeError: run_pipeline() got an unexpected keyword argument 'episode'`.

- [ ] **Step 3: Write minimal implementation**

Replace `run_pipeline` (lines 278-366) in `src/tenmin/pipeline.py`. First locate the current full body via:

Run: `rtk read src/tenmin/pipeline.py --offset 278 --limit 90` (or use the Read tool) to get the exact current text before editing, since the per-stage blocks need surgical modification rather than a blind rewrite (to preserve the freshness/`--force`/warning-collection logic that already works).

Then apply this new structure — the function signature and the top-level loop change, while the freshness-check style within each stage block is preserved but made per-episode:

```python
async def run_pipeline(
    cfg: ProjectConfig,
    provider: LLMProvider | None,
    *,
    from_stage: str = "ingest",
    only: list[str] | None = None,
    force: bool = False,
    tts_engine: TTSEngine | None = None,
    episode: int | None = None,
) -> list[str]:
    if cfg.mode == "season":
        raise NotImplementedError("season 模式还没做，先用 single_episode")

    wanted = set(only) if only else set(stages_from(from_stage))
    unknown = wanted - set(STAGES)
    if unknown:
        raise ValueError(f"未知阶段：{sorted(unknown)}")

    paths = Paths(cfg.root)
    warnings: list[str] = []

    if episode is None:
        target_numbers = [e.number for e in cfg.episodes]
    else:
        _find_episode(cfg, episode)  # raises if not registered
        target_numbers = [episode]

    if "ingest" in wanted:
        run_ingest(cfg)

    if "signals" in wanted:
        run_signals(cfg)

    for number in target_numbers:
        if "script" in wanted:
            inputs = [paths.dialogue(number), paths.signals(number)]
            if force or not _is_fresh([paths.script(number)], inputs):
                assert provider is not None
                _, stage_warnings = await run_script(cfg, provider, episode=number)
                warnings.extend(stage_warnings)

        if "docgen" in wanted:
            if force or not _is_fresh(
                [paths.table(number), paths.narration(number)], [paths.script(number)]
            ):
                run_docgen(cfg, episode=number)

    if {"voice", "timeline", "audio", "render"} & wanted:
        # preflight only needs to run once per invocation, not once per episode
        first_number = target_numbers[0]
        preflight(cfg.video_path(_find_episode(cfg, first_number)), cfg.render.video_encoder)

    for number in target_numbers:
        if "voice" in wanted:
            outputs = [paths.voice(number)]
            if force or not _is_fresh(outputs, [paths.script(number)]):
                assert tts_engine is not None
                _, stage_warnings = await run_voice(cfg, tts_engine, episode=number)
                warnings.extend(stage_warnings)

        if "timeline" in wanted:
            outputs = [paths.timeline(number), paths.subtitles(number)]
            if force or not _is_fresh(outputs, [paths.voice(number)]):
                _, stage_warnings = run_timeline(cfg, episode=number)
                warnings.extend(stage_warnings)

        if "audio" in wanted:
            outputs = [paths.mixed_audio(number)]
            if force or not _is_fresh(outputs, [paths.timeline(number)]):
                run_audio(cfg, episode=number)

        if "render" in wanted:
            outputs = [paths.video(number)]
            if force or not _is_fresh(outputs, [paths.mixed_audio(number)]):
                run_render(cfg, episode=number)

    return warnings
```

**Important:** the exact freshness-input lists above (e.g. `[paths.dialogue(number), paths.signals(number)]`) are a best-effort reconstruction based on the Task-1-through-8 audit. Before finalizing, diff this against the ACTUAL current per-stage blocks read in Step 3 above (via the `rtk read`/Read tool call) to make sure no existing freshness input or warning-collection detail is dropped — e.g. if the current code also compares against `srt_inputs`/checks `cfg.srt_path(e)` mtimes for the `script` stage's freshness inputs, preserve that in the per-episode version (`paths.dialogue`/`paths.signals` are themselves already derived from the srt file, so this is likely already covered, but confirm rather than assume).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -k "batch_mode or single_episode_mode" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the entire test_pipeline.py file**

Run: `uv run pytest tests/test_pipeline.py -v 2>&1 | tail -100`
Expected: all tests pass. Fix any remaining stragglers (most likely leftover bare `.script`/`.table`/`.narration` references in tests not yet caught, or a missed `episode=` kwarg on a direct stage-function call) before moving on — do not proceed to Task 10 until this file is fully green.

- [ ] **Step 6: Commit**

```bash
git add src/tenmin/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): run_pipeline supports batch mode and single-episode mode"
```

---

### Task 10: Run the full test suite and fix any remaining fallout

**Files:**
- Modify: whichever files the failures point to (most likely none beyond what's already touched, but `src/tenmin/cli.py`'s `run()` function prints `paths.table`/`paths.narration` as bare properties at lines 129-132 — this will now be a hard `TypeError` crash, not just a test failure, since `Path` objects aren't callable)

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -100`

- [ ] **Step 2: Fix `src/tenmin/cli.py`'s `run()` function's now-broken property access**

Open `src/tenmin/cli.py` around lines 129-136 (the code printing `paths.table`/`paths.narration`/`paths.video(episode.number)` after a successful `run_pipeline` call). At this point in the plan, `run()` still doesn't have an `episode` parameter of its own yet (that's Task 11) — for now, make this block loop over `cfg.episodes` (all of them) using the per-episode methods, matching the existing loop-over-`cfg.episodes`-for-video pattern already present for `paths.video`:

```python
    for episode_cfg in cfg.episodes:
        table_path = paths.table(episode_cfg.number)
        narration_path = paths.narration(episode_cfg.number)
        if table_path.exists():
            typer.echo(f"对照表：{table_path}")
        if narration_path.exists():
            typer.echo(f"配音文本：{narration_path}")
        video_path = paths.video(episode_cfg.number)
        if video_path.exists():
            typer.echo(f"成品视频：{video_path}")
```

(This is a temporary/intermediate correctness fix ahead of Task 11's real episode-scoping — Task 11 will narrow this loop to only the episode(s) actually processed in that invocation.)

- [ ] **Step 3: Re-run the full suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -100`
Expected: all tests pass (515+ passed, 0 failed — matching the pre-refactor baseline count, since no tests were removed, only updated).

- [ ] **Step 4: Commit**

```bash
git add src/tenmin/cli.py
git commit -m "fix(cli): update run() output printing for per-episode Paths methods"
```

---

### Task 11: Add `--episode`, `--srt`, `--video` options to `tenmin run` with validation

**Files:**
- Modify: `src/tenmin/cli.py:75-136`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for validation errors**

Add to `tests/test_cli.py`:

```python
def test_run_srt_without_video_fails(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app,
        [
            "run",
            "saijo",
            "--work-dir",
            str(work),
            "--episode",
            "1",
            "--srt",
            str(golden_srt_path),
        ],
    )
    assert result.exit_code != 0
    assert "--srt" in out(result) and "--video" in out(result)


def test_run_srt_video_without_episode_fails(work, golden_srt_path, tmp_path):
    _bootstrap(work, golden_srt_path)
    fake_video = tmp_path / "E01.mp4"
    fake_video.write_bytes(b"fake")
    result = runner.invoke(
        app,
        [
            "run",
            "saijo",
            "--work-dir",
            str(work),
            "--srt",
            str(golden_srt_path),
            "--video",
            str(fake_video),
        ],
    )
    assert result.exit_code != 0
    assert "--episode" in out(result)


def test_run_episode_not_registered_fails(work, golden_srt_path):
    _bootstrap(work, golden_srt_path)
    result = runner.invoke(
        app,
        ["run", "saijo", "--work-dir", str(work), "--episode", "99"],
    )
    assert result.exit_code != 0
    assert "没有注册" in out(result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k "srt_without_video or srt_video_without_episode or episode_not_registered" -v`
Expected: FAIL — `Error: No such option: --episode` (the option doesn't exist yet on `run`), or similar.

- [ ] **Step 3: Write minimal implementation**

Open `src/tenmin/cli.py`. Add three new `typer.Option` parameters to the `run()` function signature (lines 75-87ish — find the exact current parameter list first via the Read tool since it must be edited precisely, not blindly appended):

```python
    episode: int | None = typer.Option(
        None, "--episode", help="要处理的集数；配合 --srt/--video 可注册新的一集"
    ),
    srt: Path | None = typer.Option(
        None, "--srt", help="要注册的字幕文件路径，需配合 --episode 和 --video"
    ),
    video: Path | None = typer.Option(
        None, "--video", help="要注册的视频文件路径，需配合 --episode 和 --srt"
    ),
```

Then, right after `cfg = load_project(_project_file(work_dir, slug))` (near the top of the function body), add validation + registration logic:

```python
    if (srt is None) != (video is None):
        typer.secho("--srt 和 --video 必须一起传", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if srt is not None and episode is None:
        typer.secho("传 --srt/--video 时必须同时传 --episode", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if srt is not None and video is not None:
        cfg = register_episode(cfg, episode=episode, srt=srt, video=video)

    if episode is not None:
        try:
            _find_episode_for_cli(cfg, episode)
        except ValueError as error:
            typer.secho(str(error), fg=typer.colors.RED)
            raise typer.Exit(code=1) from error
```

Note: `_find_episode_for_cli` here is a thin wrapper to avoid importing the pipeline-internal `_find_episode` (which is prefixed with `_` and lives in `pipeline.py`) directly into `cli.py`'s public surface — simplest is to just import `_find_episode` from `tenmin.pipeline` directly since it's already used elsewhere in this module for other purposes, or add a small public alias. Check the existing imports in `cli.py`:

Run: `rtk grep "^from tenmin.pipeline import" src/tenmin/cli.py`

Add `_find_episode` to that import line, and simplify the snippet above to call `_find_episode(cfg, episode)` directly instead of a wrapper.

Then update the `asyncio.run(run_pipeline(...))` call (further down in the function) to pass `episode=episode` through:

```python
        warnings = asyncio.run(
            run_pipeline(
                cfg,
                provider,
                from_stage=from_stage,
                only=[only] if only else None,
                force=force,
                tts_engine=tts_engine,
                episode=episode,
            )
        )
```

`register_episode` itself is implemented in Task 12 — for this task, just add a placeholder import and a minimal stub so the file is syntactically valid; Task 12 will fill it in properly. Add near the top of `cli.py`, alongside other imports:

```python
from tenmin.pipeline import register_episode  # noqa: F401  (implemented in Task 12)
```

(If `register_episode` doesn't exist yet at this point, this import will fail — so for this task ONLY, temporarily inline a no-op version directly in `cli.py` instead of importing it, to keep Task 11 self-contained and testable in isolation:)

```python
def _register_episode_placeholder(cfg, *, episode, srt, video):
    raise NotImplementedError("register_episode not implemented yet — see Task 12")
```

and call `cfg = _register_episode_placeholder(cfg, episode=episode, srt=srt, video=video)` instead of `register_episode(...)` for now. Task 12 replaces this placeholder with the real implementation and removes the `NotImplementedError` stub.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k "srt_without_video or srt_video_without_episode or episode_not_registered" -v`
Expected: PASS (3 passed) — the first two tests never reach the placeholder (they fail validation before registration is attempted); the third test doesn't pass `--srt`/`--video` at all, so it also never touches the placeholder.

- [ ] **Step 5: Run the full CLI test file**

Run: `uv run pytest tests/test_cli.py -v 2>&1 | tail -60`
Expected: all existing tests still pass (none of them pass `--episode`/`--srt`/`--video`, so the new validation branches are never triggered for them — they should be unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/tenmin/cli.py tests/test_cli.py
git commit -m "feat(cli): add --episode/--srt/--video validation to tenmin run"
```

---

### Task 12: Implement `register_episode` (auto-register: copy files + update `project.yaml`)

**Files:**
- Modify: `src/tenmin/pipeline.py` (add new function, near `_find_episode` or at the end of the file)
- Modify: `src/tenmin/cli.py` (replace the Task 11 placeholder with the real import/call)
- Test: `tests/test_pipeline.py` (new tests), `tests/test_cli.py` (new end-to-end register+run test)

- [ ] **Step 1: Write failing tests for `register_episode` in `tests/test_pipeline.py`**

```python
def test_register_episode_copies_files_and_appends_yaml_entry(tmp_path, golden_srt_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "video").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nmode: single_episode\n"
        "target_seconds: 240\nepisodes:\n- number: 2\n  srt: srt/E02.srt\n"
        "  video: video/E02.mp4\n",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)

    source_srt = tmp_path / "incoming_E01.srt"
    source_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_video = tmp_path / "incoming_E01.mp4"
    source_video.write_bytes(b"fake video bytes")

    updated_cfg = register_episode(
        cfg, episode=1, srt=source_srt, video=source_video
    )

    assert (root / "srt" / "E01.srt").exists()
    assert (root / "video" / "E01.mp4").exists()
    assert len(updated_cfg.episodes) == 2
    new_entry = next(e for e in updated_cfg.episodes if e.number == 1)
    assert new_entry.srt == Path("srt/E01.srt")
    assert new_entry.video == Path("video/E01.mp4")

    # reload from disk to confirm the yaml file itself was updated
    reloaded = load_project(yaml_path)
    assert len(reloaded.episodes) == 2
    assert any(e.number == 1 for e in reloaded.episodes)
    assert any(e.number == 2 for e in reloaded.episodes)


def test_register_episode_updates_existing_entry_in_place(tmp_path, golden_srt_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "video").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(
        "show: 才女的侍从\nslug: saijo\nmode: single_episode\n"
        "target_seconds: 240\nepisodes:\n- number: 2\n  srt: srt/E02.srt\n"
        "  video: video/E02.mp4\n",
        encoding="utf-8",
    )
    cfg = load_project(yaml_path)

    source_srt = tmp_path / "replacement_E02.srt"
    source_srt.write_text(golden_srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_video = tmp_path / "replacement_E02.mp4"
    source_video.write_bytes(b"replacement video bytes")

    updated_cfg = register_episode(
        cfg, episode=2, srt=source_srt, video=source_video
    )

    assert len(updated_cfg.episodes) == 1
    assert (root / "video" / "E02.mp4").read_bytes() == b"replacement video bytes"
```

Add `register_episode`, `load_project`, `Path` imports to the top of `tests/test_pipeline.py` if not already present (check `rtk grep "^from tenmin" tests/test_pipeline.py` first).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -k test_register_episode -v`
Expected: FAIL with `ImportError: cannot import name 'register_episode'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/tenmin/pipeline.py` (near `_find_episode`, or at the end of the file — check that `yaml` is already imported at the top of this file; if not, add `import yaml`):

```python
def register_episode(
    cfg: ProjectConfig, *, episode: int, srt: Path, video: Path
) -> ProjectConfig:
    """把外部传入的 srt/video 拷进项目目录，并把这一集写进 project.yaml。

    如果这一集已经注册过，就覆盖 srt/video 路径（保留其它字段）；
    否则追加一条新的 episode 记录。返回更新后的 ProjectConfig（root 已绑定）。
    """
    srt_dest = cfg.root / "srt" / f"E{episode:02d}.srt"
    video_dest = cfg.root / "video" / f"E{episode:02d}.mp4"
    srt_dest.parent.mkdir(parents=True, exist_ok=True)
    video_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(srt, srt_dest)
    shutil.copyfile(video, video_dest)

    relative_srt = srt_dest.relative_to(cfg.root)
    relative_video = video_dest.relative_to(cfg.root)

    existing = next((e for e in cfg.episodes if e.number == episode), None)
    if existing is not None:
        existing.srt = relative_srt
        existing.video = relative_video
    else:
        cfg.episodes.append(
            EpisodeConfig(number=episode, srt=relative_srt, video=relative_video)
        )

    yaml_path = cfg.root / "project.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["episodes"] = [
        {
            "number": e.number,
            "srt": str(e.srt),
            **({"video": str(e.video)} if e.video else {}),
        }
        for e in cfg.episodes
    ]
    yaml_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    return cfg
```

Check whether `shutil` is already imported at the top of `pipeline.py` — if not, add `import shutil`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -k test_register_episode -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Wire the real `register_episode` into `cli.py`, replacing the Task 11 placeholder**

In `src/tenmin/cli.py`: remove the `_register_episode_placeholder` function entirely. Add `register_episode` to the existing `from tenmin.pipeline import (...)` block. Change the call site from `cfg = _register_episode_placeholder(cfg, episode=episode, srt=srt, video=video)` to:

```python
    if srt is not None and video is not None:
        cfg = register_episode(cfg, episode=episode, srt=srt, video=video)
```

- [ ] **Step 6: Write an end-to-end CLI test for the register+run flow**

Add to `tests/test_cli.py`:

```python
def test_run_registers_new_episode_and_updates_yaml(work, golden_srt_path, tmp_path):
    _bootstrap(work, golden_srt_path)
    fake_video = tmp_path / "E01_source.mp4"
    fake_video.write_bytes(b"fake video bytes")

    result = runner.invoke(
        app,
        [
            "run",
            "saijo",
            "--work-dir",
            str(work),
            "--episode",
            "1",
            "--srt",
            str(golden_srt_path),
            "--video",
            str(fake_video),
            "--only",
            "ingest",
        ],
    )

    assert result.exit_code == 0, out(result)
    assert (work / "saijo" / "srt" / "E01.srt").exists()
    assert (work / "saijo" / "video" / "E01.mp4").exists()

    reloaded_yaml = (work / "saijo" / "project.yaml").read_text(encoding="utf-8")
    assert "number: 1" in reloaded_yaml
```

- [ ] **Step 7: Run the new test and the full CLI test file**

Run: `uv run pytest tests/test_cli.py -v 2>&1 | tail -80`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/tenmin/pipeline.py src/tenmin/cli.py tests/test_pipeline.py tests/test_cli.py
git commit -m "feat(pipeline,cli): implement register_episode auto-registration flow"
```

---

### Task 13: Narrow the CLI's post-run output printing to only the episode(s) actually processed

**Files:**
- Modify: `src/tenmin/cli.py` (the block added in Task 10 Step 2)

- [ ] **Step 1: Update the implementation**

Replace the Task-10 loop (`for episode_cfg in cfg.episodes: ...`) with logic that only prints for the episode(s) actually touched this run:

```python
    if episode is not None:
        printed_numbers = [episode]
    else:
        printed_numbers = [e.number for e in cfg.episodes]

    for number in printed_numbers:
        table_path = paths.table(number)
        narration_path = paths.narration(number)
        if table_path.exists():
            typer.echo(f"对照表：{table_path}")
        if narration_path.exists():
            typer.echo(f"配音文本：{narration_path}")
        video_path = paths.video(number)
        if video_path.exists():
            typer.echo(f"成品视频：{video_path}")
```

- [ ] **Step 2: Run the full CLI test file**

Run: `uv run pytest tests/test_cli.py -v 2>&1 | tail -80`
Expected: all pass (existing tests that check for `paths.table`/`paths.video` printouts, e.g. `test_run_prints_mp4_path`, should be unaffected since they only ever register/process one episode, so `printed_numbers` is still just `[2]` for them).

- [ ] **Step 3: Commit**

```bash
git add src/tenmin/cli.py
git commit -m "fix(cli): only print output paths for the episode(s) actually processed"
```

---

### Task 14: Run the entire test suite end-to-end

**Files:** none (verification-only task)

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest tests/ -q 2>&1 | tail -60`
Expected: all tests pass, 0 failed. Compare the total passed count against the pre-refactor baseline (515 passed, 9 skipped) — the count should now be higher (new tests added in Tasks 2, 9, 11, 12), with 0 failed and the same 9 skipped.

- [ ] **Step 2: If any failures remain, fix them now**

Do not proceed to Task 15 until this is fully green.

---

### Task 15: Migrate the existing `work/saijo` project's on-disk outputs to the new per-episode filenames

**Files:**
- `work/saijo/03_script/script.json` → `work/saijo/03_script/E02.script.json`
- `work/saijo/out/解说方案.md` → `work/saijo/out/E02.解说方案.md`
- `work/saijo/out/narration.txt` → `work/saijo/out/E02.narration.txt`

- [ ] **Step 1: Confirm the files exist at their old locations**

Run: `rtk ls /Users/portz/js/10min_anime/work/saijo/03_script/ /Users/portz/js/10min_anime/work/saijo/out/`

- [ ] **Step 2: Move each file to its new per-episode name**

```bash
mv /Users/portz/js/10min_anime/work/saijo/03_script/script.json /Users/portz/js/10min_anime/work/saijo/03_script/E02.script.json
mv "/Users/portz/js/10min_anime/work/saijo/out/解说方案.md" "/Users/portz/js/10min_anime/work/saijo/out/E02.解说方案.md"
mv /Users/portz/js/10min_anime/work/saijo/out/narration.txt /Users/portz/js/10min_anime/work/saijo/out/E02.narration.txt
```

- [ ] **Step 3: Verify the new paths exist and old ones are gone**

Run: `rtk ls /Users/portz/js/10min_anime/work/saijo/03_script/ /Users/portz/js/10min_anime/work/saijo/out/`
Expected: `E02.script.json`, `E02.解说方案.md`, `E02.narration.txt` present; old bare-named files gone.

- [ ] **Step 4: Smoke-test the real pipeline against the migrated project**

Run: `uv run tenmin run saijo --episode 2 --only docgen --force` (plain bash tool, NOT `rtk proxy`/`rtk` prefix)
Expected: EXIT 0, and `uv run tenmin run saijo --episode 2 --only docgen` regenerates `E02.解说方案.md`/`E02.narration.txt` from the migrated `E02.script.json` without errors — confirming the new per-episode paths work end-to-end against the live project, not just in the test suite's tmp_path fixtures.

- [ ] **Step 5: Also smoke-test bare batch-mode invocation (no `--episode`) still works for this single-episode project**

Run: `uv run tenmin run saijo --only docgen --force`
Expected: EXIT 0, same output regenerated — confirms batch mode with exactly one registered episode behaves identically to before this whole refactor.

- [ ] **Step 6: Commit the migration (the moved files themselves, if `work/` is tracked by git — check first)**

Run: `rtk git status --short work/saijo/` to see if `work/` is gitignored or tracked.

If tracked:
```bash
git add work/saijo/03_script/E02.script.json "work/saijo/out/E02.解说方案.md" work/saijo/out/E02.narration.txt
git commit -m "chore: migrate saijo E02 outputs to per-episode filenames"
```

If `work/` is gitignored, no commit is needed for this step — just note that the migration was a filesystem-only operation.

---

## Self-Review Checklist

- [x] **Spec coverage:** All 5 approved Q&A decisions and all 3 design sections from `docs/superpowers/specs/2026-09-07-tenmin-multi-episode-cli-design.md` are covered — per-episode-keyed outputs (Tasks 1, 15), explicit `--episode` flag (Task 11), auto-register (Task 12), project-wide shared render config (untouched, no task needed since design says NOT to change this), batch mode (Task 9), test suite impact (Tasks 1-13 update tests inline as each change is made).
- [x] **Placeholder scan:** No "TBD"/"TODO" left in final code — the one intentional placeholder (Task 11's `_register_episode_placeholder`) is explicitly removed in Task 12 Step 5, and is called out as temporary/scoped in the task text itself, not a genuine unresolved gap.
- [x] **Type/signature consistency:** `episode: int` parameter name and position is consistent across `run_script`, `run_docgen`, `run_voice`, `run_timeline`, `run_audio`, `run_render`, `run_pipeline` (as `episode: int | None = None`), `_find_episode(cfg, episode_number)`, and `register_episode(cfg, *, episode, srt, video)`. `Paths.script/table/narration` all take `episode: int` as their sole positional parameter, matching `Paths.dialogue/signals/voice/voice_dir/timeline/subtitles/mixed_audio/video`'s existing convention.
- [x] **Scope check:** Single cohesive feature (multi-episode CLI support), no unrelated refactoring bundled in. Out-of-scope items from the spec (per-episode render override, filename-based episode inference, changes to render/subtitle/audio internals) are correctly NOT touched by any task.

---

Plan complete and saved to `docs/superpowers/plans/2026-09-07-tenmin-multi-episode-cli.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
