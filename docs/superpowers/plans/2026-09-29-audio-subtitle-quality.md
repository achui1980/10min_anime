# 音频与字幕质量 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 确保留白播放所引原声并平衡音量，控制全片响度和真峰值，软拆长字幕。

**Architecture:** 剧本修复只延长当前节点同集相邻镜头；时间轴用实测 chunk 时长、真实画面映射验证留白并迭代撤销；音频只读持久化的时间轴 offsets/保留窗，同图测量、逐窗增益、双遍归一和编码后核验。字幕仅拆显示 cue，不改 TTS chunk。

**Tech Stack:** Python ≥3.12、Pydantic v2、pytest、ruff、ffmpeg/ffprobe、临时生成 WAV/视频。

**Spec:** `docs/superpowers/specs/2026-09-29-audio-subtitle-quality-design.md`（SHA `94feaa9`）。

## Global Constraints

- 在 `feat/pipeline-robustness` 的本 worktree 实施；不调用真实 LLM/TTS、不碰正式 `work/`，不改 MP3 身份、旁白文本、硬字幕位置。
- 裸 `uv run pytest ...` / `uv run ruff check ...`；PATH 不含 uv 时用 `/Users/portz/.local/bin/uv run ...`，绝不包 rtk；变异测试设 `PYTHONDONTWRITEBYTECODE=1`。
- 旋钮默认：邻近 3s、字幕软限 36 字、相对旁白 ±3 LU、增益上限 ±6dB、静音下限 −40 LUFS、淡变 0.1s；全片 I=−14 LUFS、TP≤−1.5 dBTP、LRA=11。机械门槛：覆盖率 ≥70%，端点误差 ≤0.25s（不计入实际覆盖秒数）。
- `check_script` 纯读；`repair_script` 深拷贝；`build_timeline` 返回 `(Timeline, warnings)`；`run_audio`/`mix_audio` 保持返回 `Path`，新 warnings 以可选可变列表出参交给 `run_pipeline`。所有测试先红再绿；精确图快照随真实图变化更新，不能删除测试。

---

## File map

| 路径 | 职责 |
| --- | --- |
| `src/tenmin/config.py`, `src/tenmin/config_slices.py` | 严格旋钮、阶段新鲜度 |
| `src/tenmin/models.py` | `HoldWindow`，`Timeline.hold_windows` 缺省空列表 |
| `src/tenmin/script/validate.py` | 复用金句匹配和 clip 合法性判据，逐 hold 保守修复 |
| `src/tenmin/render/timeline.py` | 显示 cue 拆分、真实映射、留白固定点重算 |
| `src/tenmin/render/audio.py` | 人工窗口体检、响度测量、窗口增益、双遍归一及原子发布 |
| `src/tenmin/render/subtitles.py` | 更新不可安全拆分的可读性警告；ASS 样式保持 |
| `src/tenmin/pipeline.py` | 对白输入/新鲜度、warnings 与 ffprobe 参数接线 |
| `tests/test_config.py`, `tests/test_config_slices.py`, `tests/test_models.py`, `tests/test_validate.py`, `tests/test_render_timeline.py`, `tests/test_render_subtitles.py`, `tests/test_render_audio.py`, `tests/test_pipeline.py`, `tests/test_config_wiring.py`, `tests/test_audio_quality.py`（新建） | 按层红绿及生成媒体集成测试 |

### Task 1: 配置、切片与兼容模型

**Files:** Modify `src/tenmin/config.py:302-424`, `src/tenmin/config_slices.py:46-75`, `src/tenmin/models.py:409-435`; Test `tests/test_config.py`, `tests/test_config_slices.py`, `tests/test_models.py`.

**Interfaces:** Produce `ValidateConfig.hold_clip_max_gap_seconds`, `RenderConfig.subtitle_soft_max_chars/hold_relative_lu/hold_gain_max_db/hold_silence_floor_lufs/hold_fade_seconds/loudness_i/loudness_tp/loudness_lra`; `HoldWindow(beat_id,hold_index,quote,episode,source_start,source_end,start,end)`.

- [ ] **Step 1: Red test** append to named tests (existing imports include `pytest`, `ValidationError`, `RenderConfig`, `ValidateConfig`, `_cfg`, `_with`, `_changed`):

```python
# tests/test_config.py
def test_quality_knobs_are_bounded():
    assert ValidateConfig().hold_clip_max_gap_seconds == 3
    cfg = RenderConfig()
    assert (cfg.subtitle_soft_max_chars, cfg.hold_relative_lu, cfg.hold_gain_max_db,
            cfg.hold_silence_floor_lufs, cfg.hold_fade_seconds) == (36, 3, 6, -40, 0.1)
    assert (cfg.loudness_i, cfg.loudness_tp, cfg.loudness_lra) == (-14, -1.5, 11)
    for model, name, value in [(ValidateConfig, "hold_clip_max_gap_seconds", -1),
                               (RenderConfig, "subtitle_soft_max_chars", 0),
                               (RenderConfig, "hold_gain_max_db", -1),
                               (RenderConfig, "hold_fade_seconds", -1),
                               (RenderConfig, "loudness_tp", 1),
                               (RenderConfig, "loudness_lra", 0)]:
        with pytest.raises(ValidationError):
            model.model_validate({name: value})

# tests/test_config_slices.py
@pytest.mark.parametrize(("field", "value", "expected"), [
    ("validate_script.hold_clip_max_gap_seconds", 4.0, {"script"}),
    ("render.subtitle_soft_max_chars", 30, {"timeline"}),
    ("render.hold_relative_lu", 2, {"audio"}),
    ("render.hold_gain_max_db", 5, {"audio"}),
    ("render.hold_silence_floor_lufs", -42, {"audio"}),
    ("render.hold_fade_seconds", 0.2, {"audio"}),
    ("render.loudness_i", -16, {"audio"}),
    ("render.loudness_tp", -2, {"audio"}),
    ("render.loudness_lra", 9, {"audio"}),
])
def test_quality_field_affects_its_first_consumer(tmp_path, field, value, expected):
    cfg = _cfg(tmp_path)
    assert _changed(cfg, _with(cfg, field, value)) == expected

# tests/test_models.py: import HoldWindow and Timeline from tenmin.models
def test_old_timeline_has_no_windows_and_new_timeline_round_trips():
    assert Timeline.model_validate_json('{"episode":2}').hold_windows == []
    window = HoldWindow(beat_id="b", hold_index=0, quote="金句", episode=2,
                        source_start=10, source_end=11, start=2, end=3)
    item = Timeline(episode=2, hold_windows=[window])
    assert Timeline.model_validate_json(item.model_dump_json()).hold_windows == [window]
```

- [ ] **Step 2: Confirm red.** `uv run pytest tests/test_config.py::test_quality_knobs_are_bounded tests/test_config_slices.py::test_quality_field_affects_its_first_consumer tests/test_models.py::test_old_timeline_has_no_windows_and_new_timeline_round_trips -q`; expected FAIL missing fields/import.
- [ ] **Step 3: Implement** insert these actual fields into existing classes (no other changes to defaults):

```python
# config.ValidateConfig
hold_clip_max_gap_seconds: float = Field(default=3.0, ge=0)
# config.RenderConfig
subtitle_soft_max_chars: int = Field(default=36, gt=0)
hold_relative_lu: float = Field(default=3.0, ge=0)
hold_gain_max_db: float = Field(default=6.0, ge=0)
hold_silence_floor_lufs: float = Field(default=-40.0, le=0)
hold_fade_seconds: float = Field(default=0.1, ge=0)
loudness_i: float = Field(default=-14.0, lt=0)
loudness_tp: float = Field(default=-1.5, le=0)
loudness_lra: float = Field(default=11.0, gt=0)

# models.py before Timeline
class HoldWindow(_StageModel):
    beat_id: str
    hold_index: int = Field(ge=0)
    quote: str
    episode: int
    source_start: float
    source_end: float
    start: float
    end: float

# models.Timeline
hold_windows: list[HoldWindow] = Field(default_factory=list)

# config_slices.STAGE_FIELDS: script already includes entire validate_script;
# add "render.subtitle_soft_max_chars" to timeline tuple;
# add the following exact entries to audio tuple:
"render.hold_relative_lu", "render.hold_gain_max_db",
"render.hold_silence_floor_lufs", "render.hold_fade_seconds",
"render.loudness_i", "render.loudness_tp", "render.loudness_lra",
```

- [ ] **Step 4: Green.** `uv run pytest tests/test_config.py tests/test_config_slices.py tests/test_models.py -q`; expected PASS. `test_config_wiring.py` is run after actual consumers in subsequent tasks (it detects as-yet unwired fields).
- [ ] **Step 5: Commit.** `git add src/tenmin/config.py src/tenmin/config_slices.py src/tenmin/models.py tests/test_config.py tests/test_config_slices.py tests/test_models.py && git commit -m "feat: define audio and subtitle quality contracts"`.

### Task 2: 剧本留白只在安全范围内延长已有镜头

**Files:** Modify `src/tenmin/script/validate.py:211-267,597-654`; Test `tests/test_validate.py:257-320,957-994`.

**Interfaces:** `locate_hold_line(tracks: dict[int, DialogueTrack], episodes: list[int], quote: str) -> tuple[int,DialogueLine] | None` shared by Task 4. Exact duplicates `(episode,start,end)` collapse; multiple different positions ambiguous. `_quote_matches` stays unchanged. Fully cover a line, not just overlap. Per beat/episode, do not move hard starts, add clips or reorder clips.

- [ ] **Step 1: Red tests** append (existing test helpers `make_script`, `clip`, `make_track`, `make_report`, `dline`, `with_holds`, `hold`):

```python
def test_repair_extends_nearest_clip_without_mutating_script():
    s = with_holds(make_script([[clip(10, 20), clip(30, 35)]]), [hold("听我说清楚")])
    before = s.model_dump_json()
    repaired, warnings = repair_script(s, {2: make_track(lines=[
        dline(1, 21, 22, "听我说清楚")])}, {2: make_report()})
    assert [c.end for c in repaired.beats[0].clips] == [22, 35]
    assert len(repaired.beats[0].audio.holds) == 1
    assert s.model_dump_json() == before
    assert any("延长" in w for w in warnings)

@pytest.mark.parametrize(("begin", "end", "op", "reason"), [
    (24.01, 25, None, "邻近"), (9, 11, None, "起点"),
    (21, 23, (16, 24), "片头"), (21, 27, None, "顺序"),
])
def test_repair_discards_unsafe_hold(begin, end, op, reason):
    s = with_holds(make_script([[clip(10, 20), clip(25, 30)]]), [hold("听我说清楚")])
    repaired, warnings = repair_script(s, {2: make_track(op=op, lines=[
        dline(1, begin, end, "听我说清楚")])}, {2: make_report()})
    assert repaired.beats[0].audio.holds == []
    assert [(c.start, c.end) for c in repaired.beats[0].clips] == [(10, 20), (25, 30)]
    assert any(reason in w and "留白" in w for w in warnings)

def test_ambiguous_quote_and_missing_quote_are_discarded():
    s = with_holds(make_script([[clip(10, 20)]]), [hold("听我说清楚"), hold("不存在")])
    track = make_track(lines=[dline(1, 11, 12, "听我说清楚"),
                              dline(2, 18, 19, "听我说清楚")])
    result, warnings = repair_script(s, {2: track}, {2: make_report()})
    assert result.beats[0].audio.holds == []
    assert any("多个" in w for w in warnings) and any("定位" in w for w in warnings)
```

Update old tests expecting “outside clip only warn”, “touch edge is inside”, “missing quote silently ignored”, “any of multiple matches succeeds”, and quote tests near end of `tests/test_validate.py`; assert removal/full coverage instead. Other tests using placeholder `quote="金句"` must supply a matching line if they assert preserved hold count; keep the tests of `check_script` unchanged to verify pure read.
- [ ] **Step 2: Red.** `uv run pytest tests/test_validate.py -q`; expected FAIL new tests and old warning semantics.
- [ ] **Step 3: Implement** after existing `_quote_matches` and call `_repair_holds` after `beat.clips = kept` in `repair_script`; use same `_reject_reason` as normal clips:

```python
def locate_hold_line(tracks: dict[int, DialogueTrack], episodes: list[int],
                     quote: str) -> tuple[int, DialogueLine] | None:
    hits = [(ep, ln) for ep in episodes for ln in _quote_matches(tracks, [ep], quote)
            if ln.kind in ("dialogue", "monologue") and ln.end > ln.start]
    unique = {(ep, ln.start, ln.end): (ep, ln) for ep, ln in hits}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _repair_holds(beat: Beat, tracks: dict[int, DialogueTrack], cfg: ValidateConfig) -> list[str]:
    warnings: list[str] = []
    kept = []
    episodes = list(dict.fromkeys(c.episode for c in beat.clips))
    for item in beat.audio.holds:
        located = locate_hold_line(tracks, episodes, item.quote)
        if located is None:
            matches = {(ep, ln.start, ln.end) for ep in episodes
                       for ln in _quote_matches(tracks, [ep], item.quote)}
            why = "多个矛盾出处" if len(matches) > 1 else "无法可靠定位"
            warnings.append(f"{beat.label}：留白「{item.quote}」{why}，已撤销")
            continue
        episode, line = located
        choices = [(i, c) for i, c in enumerate(beat.clips) if c.episode == episode]
        if any(c.start <= line.start and line.end <= c.end for _, c in choices):
            kept.append(item)
            continue
        candidates: list[tuple[float, int, Clip]] = []
        failures: list[str] = []
        for index, clip in choices:
            if line.start < clip.start:
                failures.append("金句位于镜头硬起点之前")
                continue
            gap = max(0.0, line.start - clip.end)
            if gap > cfg.hold_clip_max_gap_seconds:
                failures.append("超出邻近距离")
                continue
            end = max(clip.end, line.end)
            following = next((c for c in beat.clips[index + 1:]
                              if c.episode == episode), None)
            if following is not None and following.episode == episode and end > following.start:
                failures.append("会破坏镜头顺序")
                continue
            candidate = clip.model_copy(update={"end": end})
            reason = _reject_reason(candidate, tracks[episode], cfg)
            if reason:
                failures.append(reason)
                continue
            candidates.append((end - clip.end, index, candidate))
        if not candidates:
            warnings.append(f"{beat.label}：留白「{item.quote}」无法安全修复"
                            f"（{'；'.join(dict.fromkeys(failures)) or '没有同集邻近镜头'}），已撤销")
            continue
        _, index, proposed = min(candidates, key=lambda x: (x[0], x[1]))
        beat.clips[index] = proposed
        kept.append(item)
        warnings.append(f"{beat.label}：为留白「{item.quote}」延长镜头至 {proposed.end:.2f}s")
    beat.audio.holds = kept
    return warnings

# inside repair_script after beat.clips = kept:
warnings.extend(_repair_holds(beat, tracks, cfg))
```

`_check_hold_quotes` should warn only on unique quote not fully covered or ambiguous/missing (no mutation). Add explicit tests for different episodes of identical text, different candidate magnitudes, exactly 3s eligible, source-duration limit, and existing anchor-corrected hard start. The OP fixture has original overlap 4/10 (<50%) but proposed overlap 7/13 (>50%), so only extension fails `_reject_reason`; the following clip starts outside [16,24]. The next same-episode clip may have a different-episode clip between it and the candidate: search all later same-episode clips. Reject cross-beat sourcing by never looking outside `beat.clips`.
- [ ] **Step 4: Green.** `uv run pytest tests/test_validate.py tests/test_single.py -q`; expected PASS after replacing outdated expectations.
- [ ] **Step 5: Commit.** `git add src/tenmin/script/validate.py tests/test_validate.py && git commit -m "feat: repair safe hold footage and discard uncertain quotes"`.

### Task 3: 软拆长字幕显示 cue

**Files:** Modify `src/tenmin/render/timeline.py:56-100,228-231`, `src/tenmin/render/subtitles.py:258-305`, `src/tenmin/pipeline.py:932-939`; Test `tests/test_render_timeline.py`, `tests/test_render_subtitles.py`.

**Interfaces:** `sentence_cues(chunk, start, *, cfg: RenderConfig=DEFAULT_RENDER)`; `_split_display(text, cap, min_seconds, duration)` retains original text. Split preferably after `，、；,;` in 60–100% cap window. If no safe cut, leave long original for legibility warning. Preserve existing sentence_cues exact last end.

- [ ] **Step 1: Red tests** add imports `RenderConfig`, `check_cue_legibility` in `tests/test_render_timeline.py`:

```python
@pytest.mark.parametrize(("text", "seconds", "split"), [
    ("一二三四五六七八，九十一二三四五六。", 8.0, True),
    ("这是一段没有任何停顿的很长很长很长的旁白", 8.0, False),
    ("一二三四五六七八九十，啊", 8.0, False),
    ("一二三四五六七八，九十一二三四五六", 1.0, False),
])
def test_display_cues_split_without_loss_or_flash(text, seconds, split):
    cfg = RenderConfig(subtitle_soft_max_chars=12, subtitle_min_seconds=0.7)
    chunk = VoiceChunk(beat_id="b1", index=1, text=text, path="c.mp3", duration=seconds)
    cues = sentence_cues(chunk, 3, cfg=cfg)
    assert (len(cues) > 1) is split
    assert "".join(c.text for c in cues) == text
    assert cues[0].start == 3 and cues[-1].end == 3 + seconds
    assert all(a.end == b.start for a, b in pairwise(cues))
    if split:
        assert all(c.end - c.start >= 0.7 for c in cues)
    else:
        assert any("过长" in w for w in check_cue_legibility(
            cues, max_chars=12, max_lines=0, min_seconds=0))
```

- [ ] **Step 2: Red.** `uv run pytest tests/test_render_timeline.py::test_display_cues_split_without_loss_or_flash -q`; expected FAIL unexpected cfg/max_chars keywords.
- [ ] **Step 3: Implement:**

```python
# render/timeline.py; keep old sentence_cues weighting/cursor/last-end logic
_DISPLAY_STOPS = "，、；,;"


def _split_display(text: str, cap: int, min_seconds: float, duration: float) -> list[str]:
    if len(text) <= cap:
        return [text]
    pieces: list[str] = []
    rest = text
    while len(rest) > cap:
        lower = max(1, int(cap * 0.6))
        cuts = [i for i in range(lower, min(cap, len(rest) - 1) + 1)
                if rest[i - 1] in _DISPLAY_STOPS]
        if not cuts:
            return [text]
        cut = cuts[-1]
        pieces.append(rest[:cut])
        rest = rest[cut:]
    pieces.append(rest)
    while len(pieces) > 1 and sum(ch.isalnum() for ch in pieces[-1]) <= 2:
        pieces[-2] += pieces.pop()
    if len(pieces) == 1 or any(sum(ch.isalnum() for ch in p) <= 2 for p in pieces):
        return [text]
    if any(duration * len(p) / len(text) < min_seconds for p in pieces):
        return [text]
    return pieces

# Replace sentence_cues signature and initial `sentences = split_sentences(chunk.text)`;
# preserve existing rest of function unchanged:
def sentence_cues(chunk: VoiceChunk, start: float, *,
                  cfg: RenderConfig = DEFAULT_RENDER) -> list[SubtitleCue]:
    base = split_sentences(chunk.text)
    total_weight = sum(narration_chars(sentence) for sentence in base)
    sentences = [part for sentence in base
                 for part in _split_display(sentence, cfg.subtitle_soft_max_chars,
                                            cfg.subtitle_min_seconds,
                                            chunk.duration * narration_chars(sentence) /
                                            max(1, total_weight))]
# Existing `if not sentences:` through `return cues` stays identical:
# weights -> ratio -> contiguous cursor -> cues[-1].end = start+chunk.duration.

# build_timeline old loop: subtitles.extend(sentence_cues(chunk, audio_cursor, cfg=cfg))
```

In `subtitles.check_cue_legibility` add keyword `max_chars: int = DEFAULT_RENDER.subtitle_soft_max_chars`, inside loop append `f"字幕 {stamp} 仍有 {len(cue.text)} 字，过长且无法安全软拆，建议人工改稿"` when `max_chars>0 and len(cue.text)>max_chars`. Update old docstring warning-only rationale; do not alter `render_ass`. In `pipeline.run_timeline` pass `max_chars=cfg.render.subtitle_soft_max_chars`.
- [ ] **Step 4: Green.** `uv run pytest tests/test_render_timeline.py tests/test_render_subtitles.py -q`; expected PASS. Check `split_sentences` in each fixture has one base sentence; adjust fixtures to avoid `。` before end if needed.
- [ ] **Step 5: Commit.** `git add src/tenmin/render/timeline.py src/tenmin/render/subtitles.py src/tenmin/pipeline.py tests/test_render_timeline.py tests/test_render_subtitles.py && git commit -m "feat: split long display cues at safe pauses"`.

### Task 4: 按真实配音建立留白身份并验证静音窗

**Files:** Modify `src/tenmin/render/timeline.py:178-307`; Test `tests/test_render_timeline.py`.

**Interfaces:** `build_timeline(script, track, source_duration, *, dialogue: DialogueTrack | None=None, frame_rate=None, cfg=DEFAULT_RENDER) -> tuple[Timeline,list[str]]`. Without `dialogue`, existing direct callers keep current behavior. When provided, only this episode's dialogue can locate quotes (`locate_hold_line` from Task 2); original `VoiceTrack` never mutated. `HoldWindow` fields from Task 1. Mechanical constants `HOLD_MIN_COVERAGE=.70`, `HOLD_EDGE_TOLERANCE=.25`. Use original chunk order, not `VoiceChunk.index` as a global index. Retraction zeros the *working copy's* hold_after and reruns all timeline computations. Note that current two_chunk_track has 2s hold_after but a script with no Hold; with dialogue passed that is an orphan and MUST be revoked.

- [ ] **Step 1: Add red tests** to `tests/test_render_timeline.py` (new imports `DialogueLine, DialogueTrack`, use existing helpers):

```python
def quote_track(start=108.0, end=109.0):
    return DialogueTrack(episode=2, duration=1000, lines=[
        DialogueLine(idx=1, start=start, end=end, text="关键台词", raw="关键台词")])


def hold_script():
    # assign_holds(rate=+0%) maps at=1 to first sentence boundary.
    return one_beat_script([Clip(episode=2, start=100, end=120)],
                           [Hold(at=1, duration=2, quote="关键台词")])


def test_quote_in_actual_silence_is_retained():
    track = two_chunk_track()
    timeline, warnings = build_timeline(hold_script(), track, 1000, dialogue=quote_track())
    assert warnings == []
    assert [(w.beat_id, w.hold_index, w.episode, w.start, w.end) for w in
            timeline.hold_windows] == [("b1", 0, 2, 8, 10)]
    assert timeline.hold_windows[0].source_start == 108
    assert track.chunks[0].hold_after == 2


def test_quote_outside_actual_silence_retracts_and_recalculates():
    timeline, warnings = build_timeline(hold_script(), two_chunk_track(), 1000,
                                        dialogue=quote_track(114, 115))
    assert timeline.hold_windows == []
    assert timeline.narration_offsets == pytest.approx([0, 8])
    assert timeline.total_seconds == pytest.approx(18)
    assert timeline.subtitles[-1].end == pytest.approx(18)
    assert timeline.segments[-1].timeline_end == pytest.approx(18)
    assert any("撤销" in w for w in warnings)


def test_two_holds_merged_in_one_voice_chunk_revoke_entire_silence():
    script = one_beat_script([Clip(episode=2, start=100, end=120)], [
        Hold(at=1, duration=1, quote="关键台词"), Hold(at=1, duration=1, quote="关键台词")])
    timeline, warnings = build_timeline(script, two_chunk_track(), 1000, dialogue=quote_track())
    assert timeline.total_seconds == pytest.approx(18)
    assert timeline.hold_windows == []
    assert any("合并" in w and "撤销" in w for w in warnings)


@pytest.mark.parametrize(("start", "end", "keep"), [
    (108, 109, True), (107.76, 109.76, True),
    (107.74, 109.74, False), (109, 111, False),
])
def test_coverage_threshold_is_inclusive_without_counting_tolerance(start, end, keep):
    # Actual source [108,110]; >.25s left mismatch fails even if coverage >70%.
    timeline, _ = build_timeline(hold_script(), two_chunk_track(), 1000,
                                 dialogue=quote_track(start, end))
    assert bool(timeline.hold_windows) is keep
```

Test exact 70% separately on the primitive: `assert hold_coverage([(0, 7)], 0, 10) == pytest.approx(.7)` and `assert hold_coverage([(0, 6.99)], 0, 10) < .7`; the 0.25s endpoint gate makes a 30%-missing quote impossible in this 2s integration fixture. Add tests for a second beat changing offsets on withdrawal, a quote in a clip of a different episode, a window containing another chunk's narration, and orphan hold_after. No test needs real media.
- [ ] **Step 2: Red.** `uv run pytest tests/test_render_timeline.py::test_quote_in_actual_silence_is_retained tests/test_render_timeline.py::test_quote_outside_actual_silence_retracts_and_recalculates -q`; expected FAIL (`dialogue` keyword missing). For these tests use `Clip(episode=2,start=100,end=120)` (20s source divided by 20s audio => ratio=1) and `Hold(at=1,...)`: nearest sentence boundary is first sentence (4 chars / 4.5 cps ≈0.89s), so voice chunk #1 owns the hold.
- [ ] **Step 3: Implement pure mapping primitive** near `align_to_frame`; use it both for acceptance and Task 6 manual-window audit:

```python
HOLD_MIN_COVERAGE = 0.70
HOLD_EDGE_TOLERANCE = 0.25


def source_intervals(segments: list[TimelineSegment], start: float,
                     end: float) -> list[tuple[float, float]]:
    intervals = []
    for seg in segments:
        lo, hi = max(start, seg.timeline_start), min(end, seg.timeline_end)
        if hi <= lo or seg.timeline_end <= seg.timeline_start:
            continue
        ratio = (seg.source_end - seg.source_start) / (seg.timeline_end - seg.timeline_start)
        intervals.append((seg.source_start + (lo - seg.timeline_start) * ratio,
                          seg.source_start + (hi - seg.timeline_start) * ratio))
    return intervals


def hold_coverage(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    from tenmin.intervals import merge_intervals
    if end <= start:
        return 0.0
    return sum(max(0, min(hi, end) - max(lo, start))
               for lo, hi in merge_intervals(intervals)) / (end - start)
```

- [ ] **Step 4: Implement identity reconstruction** in timeline.py; import `assign_holds`, `HoldWindow`, `DialogueTrack` and `locate_hold_line`. The following helper consumes chunks *per beat in the same order as `chunks_by_beat`*; returns `(global chunk index, list[hold indices], error)`. `hold_after` is compared with the original track, not a partially retracted copy. A hard mismatch is an ambiguous/orphan chunk, not a reason to mutate voice.json:

```python
def _chunk_holds(script: Script, track: VoiceTrack, cfg: RenderConfig
                 ) -> dict[int, tuple[list[int], str | None]]:
    result: dict[int, tuple[list[int], str | None]] = {}
    for beat in script.beats:
        sentences = split_sentences(beat.narration)
        if not sentences:
            continue
        assigned = assign_holds(sentences, beat.audio.holds, rate=cfg.rate)
        # Reuse assign_holds' nearest-boundary rule for identity; determine
        # which hold shares each boundary by replaying with one hold at a time.
        owners: dict[int, list[int]] = {}
        for index, item in enumerate(beat.audio.holds):
            boundary = next(iter(assign_holds(sentences, [item], rate=cfg.rate)))
            owners.setdefault(boundary, []).append(index)
        cursor = 0
        indices = [i for i, chunk in enumerate(track.chunks) if chunk.beat_id == beat.id]
        for global_index in indices:
            chunk = track.chunks[global_index]
            begin = cursor
            while cursor < len(sentences) and "".join(sentences[begin:cursor + 1]) != chunk.text:
                cursor += 1
            if cursor == len(sentences):
                result[global_index] = ([], "chunk 文本与剧本句界不符")
                continue
            cursor += 1
            boundary = cursor - 1
            group = owners.get(boundary, [])
            expected = assigned.get(boundary, 0.0)
            hidden = any(owners.get(k) for k in range(begin, boundary))
            error = ("chunk 跳过了中间留白边界" if hidden else
                     "合并的留白身份不唯一" if len(group) > 1 else
                     "留白身份或时长不一致" if abs(chunk.hold_after - expected) > 0.001 or
                     (chunk.hold_after > 0 and not group) else None)
            result[global_index] = (group, error)
    for i, chunk in enumerate(track.chunks):
        if chunk.hold_after > 0 and i not in result:
            result[i] = ([], "找不到该配音 chunk 的留白身份")
    return result
```

The standalone `assign_holds` with one hold may change its span-warning only (not its boundary); it is deterministic. `assigned` sums by sentence; if `owners` for an intermediate sentence boundary is nonempty but the current chunk spans past that boundary (the voice product cannot be mapped reliably), reject the enclosing chunk. Ensure `chunk.text` may contain merged punctuation with no loss; any mismatch is conservative retraction. If a group exists but `chunk.hold_after == 0`, do not create a window; warn identity mismatch only if script implies a positive hold. Add tests for mismatched identity and missing beat.
- [ ] **Step 5: Implement fixed-point wrapper**: rename current `build_timeline` body to `_build_once`; add optional `segment_episodes: list[int] | None = None` keyword and, **exactly where the current code appends a successful `TimelineSegment`**, also `segment_episodes.append(clip.episode)` if list supplied. This metadata stays in memory (do not change TimelineSegment's disk shape); it avoids guessing a clip's episode by duplicate start times. Otherwise keep the old body except `sentence_cues(...,cfg=cfg)`. Implement `_assess_holds` below:

```python
def _assess_holds(script: Script, original: VoiceTrack, working: VoiceTrack,
                  dialogue: DialogueTrack, timeline: Timeline, cfg: RenderConfig,
                  segment_episodes: list[int]
                  ) -> tuple[list[HoldWindow], dict[int, str]]:
    from tenmin.script.validate import locate_hold_line
    by_id = {beat.id: beat for beat in script.beats}
    identities = _chunk_holds(script, original, cfg)
    windows: list[HoldWindow] = []
    rejected: dict[int, str] = {}
    for i, chunk in enumerate(working.chunks):
        if chunk.hold_after <= 0:
            continue
        owners, error = identities.get(i, ([], "找不到留白身份"))
        if error or len(owners) != 1 or i >= len(timeline.narration_offsets):
            rejected[i] = f"beat {chunk.beat_id} 合并/孤立留白已撤销：{error or '数量不唯一'}"
            continue
        beat = by_id[chunk.beat_id]
        index = owners[0]
        quote = beat.audio.holds[index].quote
        located = locate_hold_line({dialogue.episode: dialogue}, [working.episode], quote)
        if located is None:
            rejected[i] = f"beat {chunk.beat_id} 留白「{quote}」同集原声无法定位，已撤销"
            continue
        _, line = located
        begin = timeline.narration_offsets[i] + chunk.duration
        end = begin + chunk.hold_after
        voice_intervals = [(offset, offset + item.duration) for offset, item in
                           zip(timeline.narration_offsets, working.chunks, strict=True)]
        touched = [(seg, ep) for seg, ep in
                   zip(timeline.segments, segment_episodes, strict=True)
                   if seg.timeline_start < end and seg.timeline_end > begin]
        played = source_intervals([seg for seg, _ in touched], begin, end)
        source_ok = bool(touched) and all(ep == working.episode and
                                           seg.beat_id == beat.id for seg, ep in touched)
        continuous = len(played) == 1 or all(
            abs(played[k][1] - played[k + 1][0]) <= 1e-6
            for k in range(len(played) - 1))
        if (not source_ok or not played or
                not continuous or
                any(begin < b and end > a for j, (a, b) in enumerate(voice_intervals)
                    if j != i) or
                hold_coverage(played, line.start, line.end) < HOLD_MIN_COVERAGE):
            rejected[i] = f"beat {chunk.beat_id} 留白「{quote}」静音窗内原声覆盖不足/串音/跨集，已撤销"
            continue
        # Tolerance only allows a <=0.25s endpoint mismatch; never expands
        # the real played interval before hold_coverage.
        if (min(lo for lo, _ in played) > line.start + HOLD_EDGE_TOLERANCE or
                max(hi for _, hi in played) < line.end - HOLD_EDGE_TOLERANCE):
            rejected[i] = f"beat {chunk.beat_id} 留白「{quote}」原声端点偏差过大，已撤销"
            continue
        windows.append(HoldWindow(beat_id=beat.id, hold_index=index, quote=quote,
                                  episode=working.episode, source_start=line.start,
                                  source_end=line.end, start=begin, end=end))
    return windows, rejected


def build_timeline(script: Script, track: VoiceTrack, source_duration: float, *,
                   dialogue: DialogueTrack | None = None, frame_rate: float | None = None,
                   cfg: RenderConfig = DEFAULT_RENDER) -> tuple[Timeline, list[str]]:
    if dialogue is None:
        return _build_once(script, track, source_duration, frame_rate=frame_rate, cfg=cfg)
    if dialogue.episode != track.episode:
        raise ValueError(f"对白轨集号 {dialogue.episode} 与配音集号 {track.episode} 不一致")
    working = track.model_copy(deep=True)
    removed: list[str] = []
    # Number of positive chunk holds strictly decreases on each unsuccessful pass.
    for _ in range(1 + sum(c.hold_after > 0 for c in track.chunks)):
        episodes: list[int] = []
        timeline, warnings = _build_once(script, working, source_duration,
                                         frame_rate=frame_rate, cfg=cfg,
                                         segment_episodes=episodes)
        windows, rejected = _assess_holds(script, track, working, dialogue,
                                          timeline, cfg, episodes)
        for index, reason in sorted(rejected.items()):
            working.chunks[index].hold_after = 0.0
            removed.append(reason)
        if not rejected:
            timeline.hold_windows = windows
            return timeline, removed + warnings
    raise ValueError("留白固定点未收敛")
```

For a gap between played source intervals, the code uses `abs(...) <= 1e-6` for continuity (0.25s applies solely to subtitle timecode endpoints). `hold_coverage` returns summed coverage only for *continuous* playback, never for disjoint clips. Store one window per global chunk; a deleted window no longer appears in final `windows`. Ensure old warnings from intermediate iterations are discarded; only last `_build_once` warnings included. `by_id[chunk.beat_id]` must be guarded: unknown beat is an orphan hold and should be rejected rather than `KeyError`.
- [ ] **Step 6: Green.** `uv run pytest tests/test_render_timeline.py -q`; expected PASS, including old no-dialogue tests. Check a 2-beat/2-hold case leaves valid earlier hold intact while the later one is revoked; assert original `model_dump_json()` unchanged.
- [ ] **Step 6a: Explicit convergence and cross-episode tests** in `tests/test_render_timeline.py`:

```python
def test_retraction_rechecks_remaining_hold_after_ratio_change():
    first = Beat(id="a", label="a", role="hook", narration="第一句。第二句。",
                 clips=[Clip(episode=2, start=100, end=120)],
                 audio=AudioDirection(holds=[Hold(at=1, duration=2, quote="关键台词")]))
    second = Beat(id="b", label="b", role="outro", narration="第三句。第四句。",
                  clips=[Clip(episode=2, start=200, end=220)],
                  audio=AudioDirection(holds=[Hold(at=1, duration=2, quote="不存在的句子")]))
    script = Script(show="剧名", episodes=[2], beats=[first, second])
    track = VoiceTrack(episode=2, chunks=[
        VoiceChunk(beat_id="a", index=1, text="第一句。", path="1.mp3", duration=8, hold_after=2),
        VoiceChunk(beat_id="a", index=2, text="第二句。", path="2.mp3", duration=10),
        VoiceChunk(beat_id="b", index=1, text="第三句。", path="3.mp3", duration=8, hold_after=2),
        VoiceChunk(beat_id="b", index=2, text="第四句。", path="4.mp3", duration=10),
    ])
    before = track.model_dump_json()
    timeline, warnings = build_timeline(script, track, 1000, dialogue=quote_track())
    assert timeline.narration_offsets == pytest.approx([0, 10, 20, 28])
    assert timeline.total_seconds == pytest.approx(38)
    assert [w.beat_id for w in timeline.hold_windows] == ["a"]
    assert sum("撤销" in w for w in warnings) == 1
    assert track.model_dump_json() == before


def test_cross_episode_clip_cannot_supply_current_video_audio():
    script = hold_script()
    script.beats[0].clips[0].episode = 3
    timeline, warnings = build_timeline(script, two_chunk_track(), 1000, dialogue=quote_track())
    assert timeline.hold_windows == []
    assert timeline.total_seconds == pytest.approx(18)
    assert any("跨集" in w for w in warnings)


def test_orphan_voice_silence_is_removed_without_quote_guess():
    script = one_beat_script([Clip(episode=2, start=100, end=120)])
    timeline, warnings = build_timeline(script, two_chunk_track(), 1000,
                                        dialogue=quote_track())
    assert timeline.total_seconds == pytest.approx(18)
    assert any("孤立" in w for w in warnings)
```

Run `uv run pytest tests/test_render_timeline.py::test_retraction_rechecks_remaining_hold_after_ratio_change tests/test_render_timeline.py::test_cross_episode_clip_cannot_supply_current_video_audio tests/test_render_timeline.py::test_orphan_voice_silence_is_removed_without_quote_guess -q`; expected PASS. The first fixture removes only b's hold: a stays valid, and b's last chunk moves from 30 to 28. For an additional within-beat recomputation test, place two holds in *separate chunks of the same beat* and pick a quote that is initially covered but, after the first hold is removed, becomes uncovered because its beat-wide scale shrinks; assert both holds ultimately revoked. Changing a previous beat's duration changes later offsets but **not** the later beat-local ratio, so that would not exercise this convergence rule.
- [ ] **Step 7: Commit.** `git add src/tenmin/render/timeline.py tests/test_render_timeline.py && git commit -m "feat: reject holds that miss real playback windows"`.

### Task 5: 时间轴读对白、按集新鲜度及警告接线

**Files:** Modify `src/tenmin/pipeline.py:881-952,1355-1364`, `src/tenmin/config_slices.py:57-66`; Test `tests/test_pipeline.py`, `tests/test_config_slices.py`.

**Interfaces:** `run_timeline` loads `Paths.dialogue(episode)` as `DialogueTrack` even with supplied source_duration (no video needed for the unit test); calls Task 4 `build_timeline(..., dialogue=...)`; `run_pipeline` timeline freshness includes dialogue file; `STAGE_FIELDS['timeline']` includes `render.rate` because identity mapping reads cfg.rate. A dialogue from any other episode must raise `ValueError` before publishing a timeline; otherwise same-index quotes would be misattributed.

- [ ] **Step 1: Write red tests** in `tests/test_pipeline.py`, using `project` fixture and existing `_write_script`, `render_script`, `_file`, `Paths`, `FakeTTSEngine`:

```python
def test_run_timeline_requires_current_dialogue(project):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8, 10, 10]), episode=2))
    with pytest.raises(FileNotFoundError, match="对白轨"):
        run_timeline(project, 2, source_duration=1400)


def test_run_timeline_reads_current_dialogue_and_writes_compat_windows(project):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8, 10, 10]), episode=2))
    _file(paths.dialogue(2), DialogueTrack(episode=2, duration=1400).model_dump_json())
    timeline, _ = run_timeline(project, 2, source_duration=1400)
    assert "hold_windows" in paths.timeline(2).read_text(encoding="utf-8")
    assert timeline.hold_windows == []


def test_run_timeline_rejects_dialogue_from_another_episode(project):
    paths = Paths(project.root)
    _write_script(paths.script(2), render_script())
    asyncio.run(run_voice(project, FakeTTSEngine([8, 10, 10]), episode=2))
    _file(paths.dialogue(2), DialogueTrack(episode=3, duration=1400).model_dump_json())
    with pytest.raises(ValueError, match="集号"):
        run_timeline(project, 2, source_duration=1400)
    assert not paths.timeline(2).exists()


@pytest.mark.asyncio
async def test_editing_dialogue_invalidates_timeline_but_not_llm_or_tts(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)
    path = Paths(project.root).dialogue(2)
    _shift_mtime(path, 100)
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([]), only=["timeline", "audio", "render"],
                       tts_engine=FakeTTSEngine([]), reporter=reporter)
    starts = {call[1] for call in reporter.calls if call[0] == "stage_start"}
    assert starts == {"timeline", "audio", "render"}
    assert all(call[1] != "voice" for call in reporter.calls)
```

The `only` argument isolates this freshness check; `FakeProvider([])` and `FakeTTSEngine([])` ensure neither service can be called. `_full_run_project` normally runs ingest first, so dialogue exists. The old `dialogue` mtime update may also invalidate the *global* signals stamp if selected; here signals is excluded. Current `run_timeline` unconditionally writes timeline.json and ASS when it runs, so audio/render also rerun. Update all **existing** direct `run_timeline` tests under `tests/test_pipeline.py` and `tests/test_config_wiring.py` to write a minimal `DialogueTrack` to temporary Paths before invoking it; never stub production to fabricate a missing dialogue. Use existing `_write_voice_and_script` helper in config_wiring and a local `_file(...)` for dialogue.
- [ ] **Step 2: Red.** `uv run pytest tests/test_pipeline.py::test_run_timeline_requires_current_dialogue tests/test_pipeline.py::test_run_timeline_reads_current_dialogue_and_writes_compat_windows -q`; expected FAIL first does not raise / second lacks windows.
- [ ] **Step 3: Implement exact connection:**

```python
# pipeline.run_timeline after loading script and voice, before probing video
dialogue_path = paths.dialogue(episode)
if not dialogue_path.exists():
    raise FileNotFoundError(f"缺少对白轨 {dialogue_path}，请先跑 ingest 阶段")
dialogue = DialogueTrack.model_validate_json(dialogue_path.read_text(encoding="utf-8"))
if dialogue.episode != episode:
    raise ValueError(f"对白轨集号 {dialogue.episode} 与目标集 {episode} 不一致")
# existing build_timeline call adds `dialogue=dialogue`
timeline, warnings = build_timeline(
    script, track, source_duration, dialogue=dialogue, frame_rate=frame_rate, cfg=cfg.render
)
# run_pipeline timeline stage:
inputs = [paths.script(number), paths.voice(number), paths.dialogue(number), *video_inputs]
# config_slices.STAGE_FIELDS['timeline']: add "render.rate" (voice still has it).
```

- [ ] **Step 4: Green.** `uv run pytest tests/test_pipeline.py tests/test_config_wiring.py tests/test_config_slices.py -q`; expected PASS after adjusting old integration fixtures to provide minimal temp dialogue. All work remains in pytest tmp dirs.
- [ ] **Step 5: Commit.** `git add src/tenmin/pipeline.py src/tenmin/config_slices.py tests/test_pipeline.py tests/test_config_wiring.py tests/test_config_slices.py && git commit -m "feat: include dialogue in timeline playback freshness"`.

### Task 6: 验证人工编辑的保留窗，组装逐窗增益图

**Files:** Modify `src/tenmin/render/audio.py:84-252`; Test `tests/test_render_audio.py`.

**Interfaces:** `_valid_hold_windows(timeline: Timeline, track: VoiceTrack) -> tuple[list[tuple[int,HoldWindow]],list[str]]` retains *original index* in timeline.hold_windows; `build_mix_args(..., hold_gains: dict[int,float]|None=None, hold_fade_seconds: float=0.1)` takes original indices; when no gains the exact old graph/snapshot remains. Invalid manually edited window skips *only* its gain, warns via `mix_audio` in Task 8. Audio `adelay` already takes `timeline.narration_offsets` (keep).

- [ ] **Step 1: Red tests** append to `tests/test_render_audio.py` (existing `build`, `make_timeline`, `make_track`):

```python
def test_manually_moved_window_is_ignored_and_valid_second_window_keeps_identity():
    from tenmin.models import HoldWindow
    from tenmin.render.audio import _valid_hold_windows
    timeline = make_timeline()
    bad = HoldWindow(beat_id="b1", hold_index=0, quote="a", episode=2,
                     source_start=108, source_end=109, start=7, end=10)
    good = HoldWindow(beat_id="b1", hold_index=1, quote="b", episode=2,
                      source_start=108, source_end=109, start=8, end=10)
    timeline.hold_windows = [bad, good]
    kept, warnings = _valid_hold_windows(timeline, make_track())
    assert kept == [(1, good)]
    assert len(warnings) == 1 and "留白窗" in warnings[0]


def test_window_gain_is_only_inside_silence_and_before_ducking(tmp_path):
    from tenmin.models import HoldWindow
    timeline = make_timeline()
    timeline.hold_windows = [HoldWindow(beat_id="b1", hold_index=0, quote="a",
        episode=2, source_start=108, source_end=109, start=8, end=10)]
    args = build(tmp_path, timeline=timeline, hold_gains={0: -6}, hold_fade_seconds=0.1)
    graph = args[args.index("-filter_complex") + 1]
    assert "between(t,8.000,10.000)" in graph
    assert "0.501187" in graph  # -6 dB linear amplitude
    assert graph.index("[holdbalanced]") < graph.index("[ducked]")


def test_audio_delays_follow_final_timeline_not_original_voice_hold_after(tmp_path):
    timeline = make_timeline()
    timeline.narration_offsets = [0, 8, 18]
    args = build(tmp_path, timeline=timeline)
    graph = args[args.index("-filter_complex") + 1]
    assert "adelay=delays=8000" in graph and "adelay=delays=18000" in graph
```

- [ ] **Step 2: Red.** `uv run pytest tests/test_render_audio.py::test_manually_moved_window_is_ignored_and_valid_second_window_keeps_identity tests/test_render_audio.py::test_window_gain_is_only_inside_silence_and_before_ducking -q`; expected FAIL missing helper/keyword.
- [ ] **Step 3: Implement** import `math`, `HoldWindow`, `source_intervals`, `hold_coverage`. To validate windows, the exact same beat's segments must cover the whole window in timeline coordinates and the chunk immediately preceding the window must end there; do not accept a random non-narration gap. Keep original list indices:

```python
def _valid_hold_windows(timeline: Timeline, track: VoiceTrack
                        ) -> tuple[list[tuple[int, HoldWindow]], list[str]]:
    kept: list[tuple[int, HoldWindow]] = []
    warnings: list[str] = []
    if len(timeline.narration_offsets) != len(track.chunks):
        return [], ["timeline 与 voice 数量不符，跳过留白窗单独增益"]
    narration = [(offset, offset + chunk.duration, chunk.beat_id) for offset, chunk in
                 zip(timeline.narration_offsets, track.chunks, strict=True)]
    for index, w in enumerate(timeline.hold_windows):
        fields = (w.start, w.end, w.source_start, w.source_end)
        matching = [(j, a, b) for j, (a, b, beat) in enumerate(narration)
                    if beat == w.beat_id and abs(b - w.start) <= 0.001]
        finite = all(math.isfinite(x) for x in fields)
        overlaps = ([(lo, hi) for lo, hi in source_intervals(
            [seg for seg in timeline.segments if seg.beat_id == w.beat_id], w.start, w.end)]
            if finite else [])
        same_identity = all((w.beat_id, w.hold_index) != (old.beat_id, old.hold_index)
                            for _, old in kept)
        touched = sorted((max(w.start, seg.timeline_start), min(w.end, seg.timeline_end))
                         for seg in timeline.segments if seg.beat_id == w.beat_id and
                         seg.timeline_start < w.end and seg.timeline_end > w.start)
        foreign = any(seg.beat_id != w.beat_id and seg.timeline_start < w.end and
                      seg.timeline_end > w.start for seg in timeline.segments)
        source_contiguous = all(abs(a[1] - b[0]) <= 1e-6 for a, b in
                                zip(overlaps, overlaps[1:], strict=False))
        contiguous = (bool(touched) and abs(touched[0][0] - w.start) <= 0.001 and
                      abs(touched[-1][1] - w.end) <= 0.001 and
                      all(abs(a[1] - b[0]) <= 0.001 for a, b in
                          zip(touched, touched[1:], strict=False)))
        fit = (finite and same_identity and contiguous and not foreign and source_contiguous
               and w.episode == timeline.episode
               and 0 <= w.start < w.end <= timeline.total_seconds
               and w.source_start < w.source_end and len(matching) == 1
               and (matching[0][0] + 1 == len(narration) or
                    timeline.narration_offsets[matching[0][0] + 1] >= w.end - 0.001)
               and all(w.end <= a or w.start >= b for a, b, _ in narration)
               and all(w.end <= old.start or w.start >= old.end for _, old in kept)
               and sum(max(0, min(seg.timeline_end, w.end) - max(seg.timeline_start, w.start))
                       for seg in timeline.segments if seg.beat_id == w.beat_id) >=
                   w.end - w.start - 0.001
               and overlaps
               and hold_coverage(overlaps, w.source_start, w.source_end) >= 0.70
               and (min(lo for lo, _ in overlaps) <= w.source_start + 0.25)
               and (max(hi for _, hi in overlaps) >= w.source_end - 0.25))
        if fit:
            kept.append((index, w))
        else:
            warnings.append(f"留白窗 {w.beat_id}/{w.hold_index} 与片段/旁白空档不一致，跳过单独增益")
    return kept, warnings


def hold_gain_expr(windows: list[tuple[int, HoldWindow]], gains: dict[int, float],
                   fade: float) -> str:
    expr = "1"
    for index, w in reversed(windows):
        if index not in gains:
            continue
        amp = 10 ** (gains[index] / 20)
        edge = min(fade, (w.end - w.start) / 2)
        envelope = (f"{amp:.6f}" if edge == 0 else
            f"if(lt(t,{w.start + edge:.3f}),1+({amp:.6f}-1)*(t-{w.start:.3f})/{edge:.3f},"
            f"if(gt(t,{w.end - edge:.3f}),1+({amp:.6f}-1)*({w.end:.3f}-t)/{edge:.3f},{amp:.6f}))")
        expr = f"if(between(t,{w.start:.3f},{w.end:.3f}),{envelope},{expr})"
    return expr
```

`build_mix_args` adds optional `hold_gains`/`hold_fade_seconds`; if gains nonempty, call `_valid_hold_windows`, filter to keyed indices, append `[orig]volume='{hold_gain_expr(...)}':eval=frame[holdbalanced]`, then replace *only* ducking input `[orig]` with `[holdbalanced]`. Existing no-gains `EXPECTED_GRAPH` must be byte-identical. In `mix_audio` collect `_valid_hold_windows` warnings once in Task 8. Guard nonfinite user-edited bounds before `source_intervals`; a NaN must never reach coverage. `source_contiguous` rejects a manually edited window spanning a different source scene; `foreign` rejects a different beat's picture in this window. Neither prevents the normal mix from continuing.
- [ ] **Step 4: Green.** `uv run pytest tests/test_render_audio.py -q`; expected PASS including exact graph assertions.
- [ ] **Step 5: Commit.** `git add src/tenmin/render/audio.py tests/test_render_audio.py && git commit -m "feat: apply retained hold gain only in valid audio gaps"`.

### Task 7: 测量同图中的原声留白和预归一化旁白

**Files:** Modify `src/tenmin/render/audio.py`; Test `tests/test_render_audio.py`, Create `tests/test_audio_quality.py`.

**Interfaces:** `_parse_ebur128_i(stderr: str) -> float|None`; `gain_for_window(source_lufs:float|None, voice_lufs:float|None, *, cfg:RenderConfig)->tuple[float,str|None]`; `_measure_hold_gains(..., windows: list[tuple[int,HoldWindow]], ...) -> tuple[dict[int,float],list[str]]`. Meter reads actual output-coordinate origin audio of **the same concat graph** before per-hold gain and normalization; narration reference uses the same voice mixing and delays but only selected spoken intervals. Do not simply probe original video's source interval: scaled/concatenated picture/audio changes the samples playing under the window. Return zero gain for source below floor or measurement error; never assert a loud music bed proves audible dialogue.

- [ ] **Step 1: Write red tests** in `tests/test_render_audio.py` and the new media test file:

```python
# tests/test_render_audio.py: add RenderConfig import
@pytest.mark.parametrize(("source", "voice", "gain", "warn"), [
    (-25, -32, -4, False),  # upper bound -29, down 4 dB
    (-37, -30, 4, False),   # lower bound -33, up 4 dB
    (-50, -30, 0, True),
    (None, -30, 0, True),
    (-30, None, 0, True),
    (-10, -32, -6, False),  # capped adjustment
])
def test_window_gain_meets_nearest_relative_bound_without_amplifying_noise(
    source, voice, gain, warn
):
    from tenmin.render.audio import gain_for_window
    actual, message = gain_for_window(source, voice, cfg=RenderConfig())
    assert actual == gain
    assert (message is not None) is warn

def test_ebur128_parser_ignores_invalid_and_uses_last_integrated_report():
    from tenmin.render.audio import _parse_ebur128_i
    assert _parse_ebur128_i("I: -29.0 LUFS\nI: -23.5 LUFS") == -23.5
    assert _parse_ebur128_i("I: -inf LUFS") is None

# tests/test_audio_quality.py (the only new test file)
import shutil
from pathlib import Path

import pytest

from tenmin.render.ffmpeg import run
from tenmin.render.audio import _parse_ebur128_i


def test_generated_sine_meter_and_silence(tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    from subprocess import run as spawn
    sine = tmp_path / "voice.wav"
    spawn(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi",
           "-i", "sine=frequency=440:duration=2:sample_rate=48000",
           "-c:a", "pcm_s16le", str(sine)], check=True)
    stderr = run(["-hide_banner", "-i", str(sine), "-af", "ebur128=peak=true",
                  "-f", "null", "-"], ffmpeg="ffmpeg")
    assert _parse_ebur128_i(stderr) is not None
    assert _parse_ebur128_i("I: -inf LUFS") is None
```

- [ ] **Step 2: Red.** `uv run pytest tests/test_render_audio.py::test_window_gain_meets_nearest_relative_bound_without_amplifying_noise tests/test_audio_quality.py -q`; expected FAIL missing functions.
- [ ] **Step 3: Implement parse and decision** (`audio.py` import `math,re`, `RenderConfig`, `run` from `tenmin.render.ffmpeg`):

```python
_I_LINE = re.compile(r"^\s*I:\s*([-+]?\d+(?:\.\d+)?|-inf)\s*LUFS\s*$", re.M | re.I)


def _parse_ebur128_i(stderr: str) -> float | None:
    hits = _I_LINE.findall(stderr)
    if not hits:
        return None
    value = float(hits[-1])
    return value if math.isfinite(value) else None


def gain_for_window(source_lufs: float | None, voice_lufs: float | None, *,
                    cfg: RenderConfig) -> tuple[float, str | None]:
    if source_lufs is None or source_lufs < cfg.hold_silence_floor_lufs:
        return 0.0, "留白近乎无声或测不出有效响度，保持原声；响度不能证明有对白"
    if voice_lufs is None:
        return 0.0, "旁白参考响度无法测量，跳过留白单窗校准"
    lower = voice_lufs - cfg.hold_relative_lu
    upper = voice_lufs + cfg.hold_relative_lu
    delta = (lower - source_lufs if source_lufs < lower else
             upper - source_lufs if source_lufs > upper else 0.0)
    return max(-cfg.hold_gain_max_db, min(cfg.hold_gain_max_db, delta)), None
```

- [ ] **Step 4: Build reusable meter graph**: refactor the *construction* inside `build_mix_args` into shared `_origin_parts(timeline)->list[str]` (all `[0:a]atrim` + concat), `_voice_parts(track,offsets)->tuple[list[str],str]` (adelay + optional voice amix), `_final_mix_parts(...)->list[str]` (ducking, amix, length, fade, outro, limiter). Do not copy/paste the origin or voice implementation into meters. `build_mix_args` joins all three to get its **unchanged** no-gain snapshot. An origin meter joins **only** `_origin_parts` and `[orig]atrim=start=WSTART:end=WEND,asetpts=PTS-STARTPTS,ebur128=peak=true[meter]`; output-coordinate window means trim after concat, not at source timestamps. A voice meter joins **only** `_voice_parts` and `voice_label+aselect='between(t,...) + ...',asetpts=N/SR/TB,ebur128=peak=true[meter]` (speech-only reference), with the same input list but without an unused branch: ffmpeg rejects unconnected filter outputs. Origin is full volume in the hold window; the ordinary ducking expression there evaluates to 1. Map only `[meter]`, `-f null -`. `run` returns stderr; `_parse_ebur128_i` reads the final summary. Implement actual argument composition:

```python
def _meter_args(video: Path, timeline: Timeline, track: VoiceTrack,
                voice_dir: Path, *, window: HoldWindow | None) -> list[str]:
    inputs = ["-i", str(video)]
    inputs.extend(x for chunk in track.chunks for x in ("-i", str(voice_dir / chunk.path)))
    if window is not None:
        parts = _origin_parts(timeline)
        parts.append(f"[orig]atrim=start={window.start:.3f}:end={window.end:.3f},"
                     "asetpts=PTS-STARTPTS,ebur128=peak=true[meter]")
    else:
        parts, label = _voice_parts(track, timeline.narration_offsets)
        speech = "+".join(f"between(t,{at:.3f},{at + chunk.duration:.3f})"
                          for at, chunk in zip(timeline.narration_offsets,
                                                track.chunks, strict=True))
        parts.append(f"{label}aselect='{speech}',asetpts=N/SR/TB,"
                     "ebur128=peak=true[meter]")
    return ["-hide_banner", *inputs, "-filter_complex", ";".join(parts),
            "-map", "[meter]", "-f", "null", "-"]


def _measure_hold_gains(video: Path, timeline: Timeline, track: VoiceTrack,
                        voice_dir: Path, windows: list[tuple[int, HoldWindow]], *,
                        cfg: RenderConfig, ffmpeg: str) -> tuple[dict[int, float], list[str]]:
    warnings: list[str] = []
    gains: dict[int, float] = {}
    try:
        voice_lufs = _parse_ebur128_i(run(_meter_args(video, timeline, track, voice_dir,
                                                      window=None), ffmpeg=ffmpeg))
    except FFmpegError:
        voice_lufs = None
    if voice_lufs is None:
        return gains, ["旁白参考响度无法测量，留白窗保持原声"]
    for index, window in sorted(windows, key=lambda pair: (pair[1].start, pair[0])):
        try:
            source = _parse_ebur128_i(run(_meter_args(video, timeline, track, voice_dir,
                                                     window=window), ffmpeg=ffmpeg))
        except FFmpegError:
            source = None
        gain, warning = gain_for_window(source, voice_lufs, cfg=cfg)
        if warning:
            warnings.append(f"留白窗 {window.beat_id}/{window.hold_index}：{warning}")
        if gain:
            gains[index] = gain
    return gains, warnings
```

Required tests use monkeypatched `audio.run` to assert origin meter graph has `[orig]atrim=start=...:end=...` and voice meter has `aselect`, with no loudnorm or encoded intermediate; first window `FFmpegError` still measures the next. Generate a 2s audio-only source WAV and a voice WAV in tmp_path; build one segment timeline and assert both meter values have finite LUFS. Use identical source/voice paths as the actual full mix passes.
- [ ] **Step 5: Implement per-window handling** in `_measure_hold_gains`: iterate indexed valid windows in sorted `(start,index)` order; measure voice reference once, then each origin interval; catch `FFmpegError` *only for that window*; emit warning and continue. When reference measurement fails, warn once and return `{}`. Store only nonzero gains keyed by original timeline index. Add test first window throws `FFmpegError` and second succeeds; neither missing window nor invalid manual window should block mixing.
- [ ] **Step 6: Green.** `uv run pytest tests/test_render_audio.py tests/test_audio_quality.py -q`; expected PASS, including existing byte-exact no-gain graph. If meter `ebur128` summary prints `I: -inf`, parser returns None; preserve unboosted source.
- [ ] **Step 7: Commit.** `git add src/tenmin/render/audio.py tests/test_render_audio.py tests/test_audio_quality.py && git commit -m "feat: measure actual hold windows against narration"`.

### Task 8: 双遍响度归一、输入守卫与编码后验收

**Files:** Modify `src/tenmin/render/audio.py:255-302`; Test `tests/test_render_audio.py`, `tests/test_audio_quality.py`.

**Interfaces:** `_loudnorm_stats(stderr: str)->dict[str,float]|None` parses finite `input_i,input_tp,input_lra,input_thresh,target_offset`; `_loudnorm_filter(cfg:RenderConfig, measured:dict[str,float]|None)->str`. `mix_audio(..., cfg:RenderConfig=DEFAULT_RENDER, warnings:list[str]|None=None, ffprobe:str=DEFAULT_RENDER.ffprobe_path) -> Path` retains existing parameters. First pass full finalized graph → loudnorm to null (no lossy intermediate); second pass same inputs/graph + measured params → AAC in atomic `.part.m4a`. Final meter probes *encoded part*, not graph output, including TP. Fatal failure does not replace previous file. All-silent stats → encode unnormalized full graph with warning, skip invalid measured params. On impossible I/TP combination keep TP and warn measured I deviation (>1 LU). Duration tolerance 0.1s and TP tolerance 0.1 dB (e.g. max measured TP = −1.4 for configured −1.5).

- [ ] **Step 1: Red tests** append in `tests/test_render_audio.py`:

```python
def test_loudnorm_stats_ignores_nonfinite_and_reads_final_json():
    from tenmin.render.audio import _loudnorm_stats
    good = ('{"input_i":"-23.9","input_tp":"-3","input_lra":"4",'
            '"input_thresh":"-34","target_offset":"0.1"}')
    assert _loudnorm_stats("header\n" + good)["input_i"] == -23.9
    assert _loudnorm_stats(good.replace('"-23.9"', '"-inf"')) is None


def test_input_changed_between_two_passes_keeps_old_audio(tmp_path, monkeypatch):
    from tenmin.atomic import part_path
    from tenmin.render import audio as module
    from tenmin.render.ffmpeg import FFmpegError
    video, voice_dir, out = tmp_path / "video.mkv", tmp_path / "voice", tmp_path / "out.m4a"
    video.write_bytes(b"original")
    voice_dir.mkdir()
    for chunk in make_track().chunks:
        (voice_dir / chunk.path).write_bytes(b"voice")
    out.write_bytes(b"good old audio")
    before = out.stat().st_mtime_ns
    def meter(args, **kwargs):
        video.write_bytes(b"changed")
        return ('{"input_i":"-23","input_tp":"-3","input_lra":"4",'
                '"input_thresh":"-34","target_offset":"0"}')
    monkeypatch.setattr(module, "run", meter)
    with pytest.raises(FFmpegError, match="输入.*变化"):
        module.mix_audio(video=video, timeline=make_timeline(), track=make_track(),
                         voice_dir=voice_dir, out_path=out, duck_db=-12)
    assert out.read_bytes() == b"good old audio"
    assert out.stat().st_mtime_ns == before
    assert not part_path(out).exists()
```

Expand `tests/test_audio_quality.py` to generate source `sine` WAV and voice WAV, create `Timeline(episode=2, segments=[TimelineSegment(beat_id="b1", source_start=0, source_end=3, timeline_start=0, timeline_end=3)], narration_offsets=[0], total_seconds=3)` and `VoiceTrack(episode=2,chunks=[VoiceChunk(beat_id="b1",index=1,text="配音",path="voice.wav",duration=3)])`. Invoke `mix_audio(video=source, timeline=..., track=..., voice_dir=tmp_path, out_path=tmp_path/"out.m4a", duck_db=-12, fade_out_seconds=0, outro_seconds=0)`; meter final file using `loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json` to parse **input** values: integrated within ±1 LU when tone is feasible; TP ≤−1.4 dBTP; `probe_duration` within 0.1s. Generate silent source/voice using `anullsrc`, assert warning and resulting silence; generate high-crest pulse (mix short high-amplitude tone into low sine) to check TP still capped and I discrepancy warns. Inject failure at each subprocess boundary (meter, encode, encoded recheck, duration probe), assert previous bytes/mtime and no `.part` file. Tests run by default, do not mark `render` (marker would skip them).
- [ ] **Step 2: Red.** `uv run pytest tests/test_render_audio.py::test_loudnorm_stats_ignores_nonfinite_and_reads_final_json tests/test_render_audio.py::test_input_changed_between_two_passes_keeps_old_audio tests/test_audio_quality.py -q`; expected FAIL.
- [ ] **Step 3: Implement stats and two-pass filter**; add imports `json`, `FFmpegError`, `probe_duration`, `run` to audio.py:

```python
def _loudnorm_stats(stderr: str) -> dict[str, float] | None:
    decoder = json.JSONDecoder()
    for pos in range(len(stderr) - 1, -1, -1):
        if stderr[pos] != "{":
            continue
        try:
            item, _ = decoder.raw_decode(stderr[pos:])
            fields = {key: float(item[key]) for key in
                      ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
        except (ValueError, KeyError, TypeError):
            continue
        return fields if all(math.isfinite(x) for x in fields.values()) else None
    return None


def _loudnorm_filter(cfg: RenderConfig, measured: dict[str, float] | None = None) -> str:
    base = f"loudnorm=I={cfg.loudness_i:g}:TP={cfg.loudness_tp:g}:LRA={cfg.loudness_lra:g}"
    if measured is None:
        return base + ":print_format=json"
    return (base + f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:linear=false:print_format=json")


def _input_identity(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
```

Construct first/second args from the *same* shared `_origin_parts + _voice_parts + _final_mix_parts` used by `build_mix_args`; append loudnorm after `[limited]` with `;[limited]{filter}[normalized]`, map `[normalized]` for encode. First pass run `ffmpeg.run([...,'-f','null','-'])`; stats from stderr. If no finite stats, append all-silence warning and encode the unnormalized `[limited]` branch. Source and every chunk identity before first pass, after first pass, and after encode before publication; on mismatch `raise FFmpegError("混音输入在两遍之间发生变化，拒绝发布")`. All actual encode args write to `atomic_path(out_path)` through existing `run_with_progress(... total_seconds=timeline.output_seconds(outro_seconds))`. For final encoded measurement use `run(['-hide_banner','-i',str(part),'-af',_loudnorm_filter(cfg),'-f','null','-'])`; check `input_tp` <= `cfg.loudness_tp + 0.1`; warn when `abs(input_i-cfg.loudness_i)>1` or none on all silence; check `abs(probe_duration(part,ffprobe=ffprobe)-timeline.output_seconds(outro_seconds))<=0.1`. If measurement output malformed and audio is **not** all silent, fail with `FFmpegError` instead of publishing unverified peak. Wrap all final checks INSIDE `atomic_path`. `linear=false` means dynamic loudnorm respects TP if linear gain impossible; keep existing `alimiter` before loudnorm, never clamp after it and lie about TP.

The concrete flow for the body of `mix_audio` is (insert *after* the signature's existing `reporter`, `ffmpeg` parameters the new `cfg`, `warnings`, `ffprobe` keyword parameters from the Interfaces block; `full_args` is a local helper):

```python
reporter = reporter or NullProgressReporter()
notices: list[str] = []
kept, audit = _valid_hold_windows(timeline, track)
notices.extend(audit)
sources = [video, *(voice_dir / chunk.path for chunk in track.chunks)]
identity = _input_identity(sources)
gains, gain_warnings = _measure_hold_gains(video, timeline, track, voice_dir, kept,
                                           cfg=cfg, ffmpeg=ffmpeg) if kept else ({}, [])
notices.extend(gain_warnings)

def full_args(dest: str, *, measured: dict[str, float] | None,
              silent: bool = False) -> list[str]:
    args = build_mix_args(video=video, timeline=timeline, track=track,
                          voice_dir=voice_dir, out_path=Path(dest), duck_db=duck_db,
                          fade_out_seconds=fade_out_seconds, outro_seconds=outro_seconds,
                          audio_codec=audio_codec, audio_bitrate=audio_bitrate,
                          limiter_ceiling=limiter_ceiling, hold_gains=gains,
                          hold_fade_seconds=cfg.hold_fade_seconds)
    graph_index = args.index("-filter_complex") + 1
    if not silent:
        args[graph_index] += f";[limited]{_loudnorm_filter(cfg, measured)}[normalized]"
        args[args.index("-map") + 1] = "[normalized]"
    if dest == "-":
        args = args[:args.index("-c:a")] + ["-f", "null", "-"]
    return args

# Include all meter runs in input guard, not just the main two passes.
if _input_identity(sources) != identity:
    raise FFmpegError("混音输入在测量期间发生变化，拒绝发布")
stats = _loudnorm_stats(run(full_args("-", measured=None), ffmpeg=ffmpeg))
if _input_identity(sources) != identity:
    raise FFmpegError("混音输入在两遍之间发生变化，拒绝发布")
if stats is None:
    notices.append("整片无有效响度统计（全静音），按原静音编码")
with atomic_path(out_path) as part:
    args = full_args(str(part), measured=stats, silent=stats is None)
    run_with_progress(args, total_seconds=timeline.output_seconds(outro_seconds),
                      on_progress=percent_reporter(reporter, "audio"), ffmpeg=ffmpeg)
    if _input_identity(sources) != identity:
        raise FFmpegError("混音输入在编码期间发生变化，拒绝发布")
    measured_out = _loudnorm_stats(run(["-hide_banner", "-i", str(part), "-af",
                                        _loudnorm_filter(cfg), "-f", "null", "-"],
                                       ffmpeg=ffmpeg))
    if measured_out is None and stats is not None:
        raise FFmpegError("编码后响度/真峰值读不出，拒绝发布")
    if measured_out is not None:
        if measured_out["input_tp"] > cfg.loudness_tp + 0.1:
            raise FFmpegError("编码后真峰值超出配置上限，拒绝发布")
        if abs(measured_out["input_i"] - cfg.loudness_i) > 1.0:
            notices.append(f"成片实测 {measured_out['input_i']:.1f} LUFS，因峰值限制偏离目标")
    actual_seconds = probe_duration(part, ffprobe=ffprobe)
    if abs(actual_seconds - timeline.output_seconds(outro_seconds)) > 0.1:
        raise FFmpegError(f"编码后音轨时长 {actual_seconds:.2f}s 与时间轴不符，拒绝发布")
    if _input_identity(sources) != identity:
        raise FFmpegError("混音输入在成品体检期间发生变化，拒绝发布")
if warnings is not None:
    warnings.extend(notices)
return out_path
```

`full_args("-", ...)` must not include an earlier `-c:a/-b:a` before `-f null`; when building the list slice, use the first `-c:a` index as in the code. `run_with_progress`'s output argument remains the same-directory `.part.m4a`, preserving ffmpeg muxer inference. A zero-signal final `loudnorm` JSON can contain `-inf`: in that one case treat absent measured_out as expected, not an error. `_input_identity` uses size+nanosecond mtime and checks again before replace (an input changed with identical size+reset mtime is outside this practical guard; do not claim it detects such tampering). The first-pass `loudnorm` with no `measured_` parameters is a **meter**: append `[normalized]` to null, parse the `input_*` fields; the second pass alone encodes.
- [ ] **Step 4: Adapt old mock tests** in `tests/test_render_audio.py` / `tests/test_pipeline.py`: current mock ffmpeg writes a byte and has no real loudnorm JSON, so inject fake `run` returning finite JSON, `probe_duration` returning `timeline.output_seconds(...)`, and `run_with_progress` writing fake part bytes. Assert two graph passes use identical `-i` inputs and exactly the same base mix graph before appended loudnorm; check final measurement reads `.part` and previous file survives injected failures.
- [ ] **Step 5: Green.** `uv run pytest tests/test_render_audio.py tests/test_audio_quality.py tests/test_pipeline.py tests/test_config_wiring.py -q`; expected PASS. If AAC overshoots configured TP, use a small *internal* safety margin for loudnorm target and retain hard verification of configured ceiling; never publish over-limit file.
- [ ] **Step 6: Commit.** `git add src/tenmin/render/audio.py tests/test_render_audio.py tests/test_audio_quality.py tests/test_pipeline.py tests/test_config_wiring.py && git commit -m "feat: normalize final mix with encoded true-peak verification"`.

### Task 9: 输出警告流、按集端到端失效验证与最终检查

**Files:** Modify `src/tenmin/pipeline.py:955-976,1366-1375`, `tests/test_pipeline.py`, `tests/test_config_wiring.py`, `tests/test_config_slices.py`.

**Interfaces:** `run_audio(cfg:ProjectConfig, episode:int, reporter:ProgressReporter|None=None, warnings:list[str]|None=None)->Path`; `mix_audio(..., cfg=cfg.render, warnings=warnings, ffprobe=cfg.render.ffprobe_path)`; `run_pipeline` continues returning `list[str]`. No new warning file. End-to-end fresh skips LLM/TTS for subtitle or audio-only tuning; script knob causes script and dependent phases to rerun. Config slice comparison tests only list first affected stage; pipeline tests check transitive invalidation.

- [ ] **Step 1: Write red integration tests** using `tests/test_pipeline.py` existing `_full_run_project`, `_first_full_run`, `_stages_rerun`, `project`:

```python
@pytest.mark.asyncio
async def test_subtitle_limit_reruns_timeline_audio_render_only(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)
    project.render.subtitle_soft_max_chars = 25
    assert await _stages_rerun(project) == {"timeline", "audio", "render"}


@pytest.mark.asyncio
async def test_loudness_change_reruns_audio_render_only(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)
    project.render.loudness_i = -16
    assert await _stages_rerun(project) == {"audio", "render"}


@pytest.mark.asyncio
async def test_clip_gap_change_reruns_script_and_dependent_stages(project, monkeypatch):
    _full_run_project(project, monkeypatch)
    await _first_full_run(project)
    project.validate_script.hold_clip_max_gap_seconds = 4
    # This particular test explicitly supplies a fake provider for *one*
    # re-run; FakeTTSEngine used for downstream only, no network.
    reporter = FakeReporter()
    await run_pipeline(project, FakeProvider([fake_script_response()]),
                       tts_engine=FakeTTSEngine([80.0] * 20), reporter=reporter)
    started = {call[1] for call in reporter.calls if call[0] == "stage_start"}
    assert "script" in started
    assert {"docgen", "voice", "timeline", "audio", "render"} <= started


def test_audio_warnings_reach_existing_list(project, monkeypatch):
    from tenmin.models import Timeline, VoiceTrack
    paths = Paths(project.root)
    _prepare_video(project)
    _file(paths.timeline(2), Timeline(episode=2).model_dump_json())
    _file(paths.voice(2), VoiceTrack(episode=2).model_dump_json())
    def fake_mix_audio(**kwargs):
        kwargs["warnings"].append("E02：留白原声近乎无声")
        return kwargs["out_path"]
    monkeypatch.setattr("tenmin.pipeline.mix_audio", fake_mix_audio)
    notices = []
    assert run_audio(project, 2, warnings=notices) == paths.mixed_audio(2)
    assert notices == ["E02：留白原声近乎无声"]
```

- [ ] **Step 2: Red.** `uv run pytest tests/test_pipeline.py::test_audio_warnings_reach_existing_list tests/test_pipeline.py::test_loudness_change_reruns_audio_render_only -q`; expected FAIL missing warnings argument or fake first pass assumptions.
- [ ] **Step 3: Implement wiring:**

```python
def run_audio(cfg: ProjectConfig, episode: int,
              reporter: ProgressReporter | None = None,
              warnings: list[str] | None = None) -> Path:
    paths = Paths(cfg.root)
    episode_cfg = _find_episode(cfg, episode)
    timeline = _load_timeline(cfg, episode)
    track = _load_voice(cfg, episode)
    return mix_audio(
        video=cfg.video_path(episode_cfg), timeline=timeline, track=track,
        voice_dir=paths.voice_dir(episode), out_path=paths.mixed_audio(episode),
        duck_db=cfg.render.duck_db, fade_out_seconds=cfg.render.fade_out_seconds,
        outro_seconds=cfg.render.outro_card_seconds,
        audio_codec=cfg.render.audio_codec, audio_bitrate=cfg.render.audio_bitrate,
        limiter_ceiling=cfg.render.limiter_ceiling, reporter=reporter,
        ffmpeg=cfg.render.ffmpeg_path, ffprobe=cfg.render.ffprobe_path,
        cfg=cfg.render, warnings=warnings,
    )

# run_pipeline audio stage, before reporter.stage_done:
run_audio(cfg, episode=number, reporter=reporter, warnings=warnings)
```

In `mix_audio`, collect `_valid_hold_windows` warnings, `_measure_hold_gains` warnings, and final loudness deviation warnings by `warnings.extend(...)` when list is not None; distinguish `[]` from None (do not use `warnings or []`). Config wiring test now checks `cfg.render` reaches `mix_audio` and `ffprobe` is forwarded. One episode's warnings should include its `E{number:02d}` prefix when emitted via `run_audio` (wrap by extending a local list) and never leak to other episodes. No new `Paths` or project-level warning artifacts.
- [ ] **Step 4: Green.** `uv run pytest tests/test_pipeline.py tests/test_config_slices.py tests/test_config_wiring.py -q`; expected PASS; `FakeProvider([])`/`FakeTTSEngine([])` guarantees subtitle/audio-only changes never call external services. For the script-knob test compare artifact mtimes and provider calls, not only reporter events (an identical script can still be rewritten by `run_script` and invalidates downstream because current run_script unconditionally writes).
- [ ] **Step 5: Final full checks** `uv run pytest tests/ -q` then `uv run ruff check src/ tests/` then `git status --short && git diff --check && git diff --stat`. Expected pytest PASS (no real LLM/TTS/render/asr), ruff PASS, diff clean whitespace. Run generated-media tests explicitly even if the full suite reports skips: `uv run pytest tests/test_audio_quality.py -q` (no gated marker); if actual ffmpeg missing, report that separately.
- [ ] **Step 6: Commit.** `git add src/tenmin/pipeline.py tests/test_pipeline.py tests/test_config_wiring.py tests/test_config_slices.py && git commit -m "test: lock stage freshness and audio warning delivery"`. Finally `git status --short` expected clean. Any mutation run must use `PYTHONDONTWRITEBYTECODE=1 uv run pytest ...`.

## 自审（写计划时已执行）

- **Spec §1:** Task 2 的匹配/同集/合法性/深拷贝；Task 4 的 chunk 身份、合并歧义、真实音画映射、70%/0.25s、迭代撤销、窗口持久化；Task 5 的对白输入与警告交付。没有改写 voice.json/script.json。
- **Spec §2:** Task 6 人工编辑窗体检；Task 7 预归一化同图声源与配音量测、静音降级和逐窗渐变；Task 8 全图两遍、无损首遍、输入不变守卫、原子产物、编码后 TP/I/时长体检和全静音回退；Task 9 汇集警告。响度本身无法识别语音，须在警告/注释明确说明。
- **Spec §3–5:** Task 3 只拆显示 cue；Task 1/5/9 覆盖严格旋钮、配置切片、新鲜度及分集离线测试。全量测试、ruff、git diff/status 在 Task 9。
- **接口核对:** `run_audio` 与 `mix_audio` 均仍返回 `Path`；timeline 仍返回二元组；`_valid_hold_windows` 保留原列表索引；audio 不用原 VoiceTrack 的 hold_after 重算 offsets；`STAGE_FIELDS['timeline']` 增加 `render.rate` 因 Task 4 使用它。
- **审查注意:** Task 7–8 在同一张图上重用 `_origin_parts/_voice_parts/_final_mix_parts`，第一遍与第二遍都必须实际 ffmpeg 生成媒体验证；精确图快照不因代码重排被放宽。Task 4 的对齐按句文本重建身份；遇到旧产物人工编辑或相同边界合并时宁可撤销整块。运行中不做源片是否真含对白的声学识别。
