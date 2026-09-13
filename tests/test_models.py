from pathlib import Path

import pytest
from pydantic import ValidationError

from tenmin.models import (
    HOLD_MAX_SECONDS,
    AudioDirection,
    Beat,
    Clip,
    DialogueLine,
    DialogueTrack,
    Highlight,
    Hold,
    LLMBeat,
    LLMClip,
    LLMScript,
    Script,
    SfxCue,
    Signal,
    SignalReport,
    SubtitleCue,
    Timeline,
    TimelineSegment,
    VoiceChunk,
    VoiceTrack,
)


def test_dialogue_line_defaults():
    line = DialogueLine(idx=1, start=7.12, end=11.48, text="你好", raw="你好")
    assert line.speaker is None
    assert line.kind == "dialogue"
    assert line.suspect is False
    assert line.merged_from == []
    assert line.duration == pytest.approx(4.36)


def test_dialogue_line_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        DialogueLine(idx=1, start=0.0, end=1.0, text="x", raw="x", kind="singing")


def test_dialogue_track_defaults():
    track = DialogueTrack(episode=2, duration=1416.622)
    assert track.source == "srt"
    assert track.op_range is None
    assert track.ed_range is None
    assert track.lines == []


def test_signal_duration():
    sig = Signal(start=134.6, end=153.486, source="gap", strength=4, detail="gap:18.9s")
    assert sig.duration == pytest.approx(18.886, abs=1e-3)
    assert sig.anchor_lines == []


def test_signal_report_defaults():
    report = SignalReport(episode=2)
    assert report.silent_gaps == []
    assert report.highlights == []
    assert report.median_char_rate == 0.0


def test_highlight_strength_bounds():
    with pytest.raises(ValidationError):
        Highlight(start=0.0, end=1.0, strength=6)
    with pytest.raises(ValidationError):
        Highlight(start=0.0, end=1.0, strength=0)
    assert Highlight(start=0.0, end=1.0, strength=5).triggers == []


def test_clip_defaults():
    clip = Clip(episode=2, start=1100.0, end=1104.7)
    assert clip.visual == ""
    assert clip.is_silent_highlight is False
    assert clip.duration == pytest.approx(4.7)


def test_audio_direction_defaults():
    audio = AudioDirection()
    assert audio.original_audio == "duck"
    assert audio.sfx == []
    assert audio.holds == []


def test_beat_and_script_defaults():
    beat = Beat(
        id="b1",
        label="Hook 开场",
        role="hook",
        narration="开场文案",
        clips=[Clip(episode=2, start=7.12, end=11.48)],
        audio=AudioDirection(
            holds=[Hold(at=1.0, duration=3.0, quote="金句")],
            sfx=[SfxCue(at=0.5, cue="impact")],
        ),
    )
    assert beat.est_seconds == 0.0
    script = Script(show="才女的侍从", episodes=[2], beats=[beat])
    assert script.mode == "single_episode"
    assert script.target_seconds == pytest.approx(240.0)
    assert script.est_total_seconds == 0.0


def test_llm_script_schema_has_no_computed_fields():
    """喂给 LLM 的 schema 不能包含我们自己算的字段。"""
    schema = LLMScript.model_json_schema()
    dumped = str(schema)
    assert "est_seconds" not in dumped
    assert "est_total_seconds" not in dumped
    assert "is_silent_highlight" not in dumped
    assert "beats" in schema["properties"]


# --- Hold / SfxCue 的时间与时长约束 ---


def test_hold_rejects_negative_at():
    with pytest.raises(ValidationError):
        Hold(at=-0.1, duration=3.0, quote="金句")


def test_hold_accepts_zero_at():
    assert Hold(at=0.0, duration=3.0, quote="金句").at == 0.0


def test_hold_rejects_non_positive_duration():
    """duration=0 的留白在预算里占 0 秒、在成片里插 0 秒静音，等于这条 hold 不存在。"""
    with pytest.raises(ValidationError):
        Hold(at=1.0, duration=0.0, quote="金句")
    with pytest.raises(ValidationError):
        Hold(at=1.0, duration=-2.0, quote="金句")


def test_hold_rejects_duration_above_hard_ceiling():
    """LLM 把 3 写成 30 会往成片里插一整段死寂，还让时长预算彻底失真。"""
    with pytest.raises(ValidationError):
        Hold(at=1.0, duration=HOLD_MAX_SECONDS + 0.1, quote="金句")


def test_hold_accepts_real_world_duration_range():
    """实测真实产出全部落在 2.0–4.0 秒，上界本身也必须放过。"""
    for duration in (2.0, 2.5, 3.0, 3.5, 4.0, HOLD_MAX_SECONDS):
        assert Hold(at=1.0, duration=duration, quote="金句").duration == duration


def test_sfx_cue_rejects_negative_at():
    with pytest.raises(ValidationError):
        SfxCue(at=-1.0, cue="impact")


# --- Beat.id：渲染阶段的连接键 ---


def test_beat_rejects_blank_id():
    for bad in ("", "   ", "\n"):
        with pytest.raises(ValidationError):
            Beat(id=bad, label="Hook", role="hook", narration="文案")


def test_llm_beat_rejects_blank_id():
    for bad in ("", "   "):
        with pytest.raises(ValidationError):
            LLMBeat(id=bad, label="Hook", role="hook", narration="文案")


def test_script_rejects_duplicate_beat_ids():
    """重复 id 会让 render/timeline.py 按 beat_id 聚合配音时静默串台。"""
    beats = [
        Beat(id="b1", label="Hook", role="hook", narration="甲"),
        Beat(id="b1", label="阶段一", role="act", narration="乙"),
    ]
    with pytest.raises(ValidationError):
        Script(show="剧名", episodes=[2], beats=beats)


def test_llm_script_rejects_duplicate_beat_ids():
    beats = [
        LLMBeat(id="b1", label="Hook", role="hook", narration="甲"),
        LLMBeat(id="b1", label="阶段一", role="act", narration="乙"),
    ]
    with pytest.raises(ValidationError):
        LLMScript(beats=beats)


def test_script_accepts_distinct_beat_ids():
    beats = [
        Beat(id="b1", label="Hook", role="hook", narration="甲"),
        Beat(id="b2", label="阶段一", role="act", narration="乙"),
    ]
    assert len(Script(show="剧名", episodes=[2], beats=beats).beats) == 2


# --- narration：严在 LLM 侧，宽在内部侧 ---


def test_llm_beat_rejects_blank_narration():
    """空 narration 会渲染出一个有画面没声音的节点，LLM 侧直接判错触发重试。"""
    for bad in ("", "   "):
        with pytest.raises(ValidationError):
            LLMBeat(id="b1", label="Hook", role="hook", narration=bad)


def test_internal_beat_still_allows_blank_narration():
    """内部 Beat 刻意保持宽松：人手改 script.json 清空某段旁白是合法编辑，
    由 render/tts.py 给出 warning 降级，不该让整份 script.json 读不进来。"""
    assert Beat(id="b1", label="Hook", role="hook", narration="  ").narration == "  "


# --- extra="forbid"：拼错的键必须报错，不能静默退回默认值 ---


def test_llm_models_forbid_extra_fields():
    with pytest.raises(ValidationError):
        LLMScript.model_validate({"beats": [], "bogus": 1})
    with pytest.raises(ValidationError):
        LLMClip.model_validate(
            {"episode": 2, "start": 1.0, "end": 2.0, "visual": "画面", "bogus": 1}
        )


def test_hold_forbids_misspelled_duration_key():
    """`dur` 被静默忽略的话，收到的是 duration 默认值而不是一个报错。"""
    with pytest.raises(ValidationError):
        Hold.model_validate({"at": 1.0, "dur": 3.0, "quote": "金句"})


def test_internal_script_models_forbid_extra_fields():
    with pytest.raises(ValidationError):
        Script.model_validate({"show": "剧名", "episodes": [2], "bogus": 1})


# --- 喂给 Gemini 的 schema 只能用 types.Schema 认识的关键字 ---


def _walk_schema_keys(node, found):
    if isinstance(node, dict):
        found.update(node.keys())
        for value in node.values():
            _walk_schema_keys(value, found)
    elif isinstance(node, list):
        for value in node:
            _walk_schema_keys(value, found)


def test_llm_schema_avoids_keywords_gemini_rejects():
    """google.genai 的 types.Schema 没有 exclusiveMinimum/exclusiveMaximum 字段，
    pydantic 的 gt/lt 会生成它们，于是 t_schema(LLMScript) 在发请求时直接 ValidationError。
    所以 LLM 侧的「必须大于 0」只能用 ge + AfterValidator 表达，不能用 gt。"""
    found: set[str] = set()
    _walk_schema_keys(LLMScript.model_json_schema(), found)
    assert "exclusiveMinimum" not in found
    assert "exclusiveMaximum" not in found


# --- 新约束不能把真实产物判成非法 ---


@pytest.mark.parametrize(
    "name", ["../tests/fixtures/akujo_e02.script.json", "../tests/snapshots/saijo_e02.script.json"]
)
def test_committed_real_scripts_still_validate(name):
    path = (Path(__file__).parent / name).resolve()
    if not path.exists():
        pytest.skip(f"缺少 {path.name}")
    script = Script.model_validate_json(path.read_text(encoding="utf-8"))
    assert script.beats


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


def test_timeline_output_seconds_is_the_body_plus_the_outro_card():
    """成片总长只有这一处定义。原来 render/video.py 与 render/audio.py 各算一份。"""
    timeline = Timeline(episode=2, total_seconds=30.0)
    assert timeline.output_seconds(0.0) == pytest.approx(30.0)
    assert timeline.output_seconds(3.0) == pytest.approx(33.0)


def test_timeline_output_seconds_ignores_a_negative_outro():
    """卡片时长配成负数不该把成片算短（config 拦得住，但这里是唯一真相，自己也得站得住）。"""
    assert Timeline(episode=2, total_seconds=30.0).output_seconds(-5.0) == pytest.approx(30.0)


def test_timeline_frame_rate_defaults_to_none():
    """存量 timeline.json 没有这个键，None = 「没探到」而不是「帧率是 0」。"""
    assert Timeline(episode=2).frame_rate is None
