import pytest

from tenmin.config import ProjectConfig
from tenmin.models import (
    DialogueLine,
    DialogueTrack,
    Highlight,
    LLMBeat,
    LLMClip,
    LLMScript,
    Signal,
    SignalReport,
)
from tenmin.script.single import (
    build_dialogue_block,
    build_glossary_block,
    build_highlight_block,
    generate_script,
    to_script,
)

from .fakes import FakeProvider


def dline(idx, start, end, text, kind="dialogue", speaker=None, suspect=False):
    return DialogueLine(
        idx=idx,
        start=start,
        end=end,
        text=text,
        raw=text,
        kind=kind,
        speaker=speaker,
        suspect=suspect,
    )


@pytest.fixture
def track():
    return DialogueTrack(
        episode=2,
        duration=1416.6,
        op_range=(153.486, 224.681),
        ed_range=(1348.18, 1416.622),
        lines=[
            dline(1, 7.12, 11.48, "你是谁"),
            dline(2, 12.0, 14.0, "我是侍从", speaker="伊月"),
            dline(3, 20.0, 22.0, "她随时都被旁人包围", kind="monologue", speaker="伊月"),
            dline(4, 30.0, 32.0, "80-08 浙谷339", suspect=True),
            dline(5, 40.0, 42.0, "河原正信 有贺史英", kind="credits"),
            dline(6, 50.0, 51.0, "-", kind="noise"),
        ],
    )


@pytest.fixture
def report():
    return SignalReport(
        episode=2,
        median_char_rate=5.0,
        silent_gaps=[
            Signal(start=1328.367, end=1348.18, source="gap", strength=4, detail="gap:19.8s")
        ],
        highlights=[
            Highlight(
                start=1328.367,
                end=1348.18,
                strength=4,
                triggers=["gap:19.8s"],
                summary="无台词演出段 19.8s",
                anchor_lines=[1],
            )
        ],
    )


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "e02.srt").write_text("1\n", encoding="utf-8")
    return ProjectConfig.model_validate(
        {
            "show": "才女的侍从",
            "slug": "saijo",
            "target_seconds": 240,
            "episodes": [{"number": 2, "srt": "e02.srt"}],
            "glossary": {"伊月": "伊月"},
        }
    )


def llm_beat(bid, label, role, chars, start, end, anchors=()):
    return LLMBeat(
        id=bid,
        label=label,
        role=role,
        narration="啊" * chars,
        clips=[
            LLMClip(
                episode=2, start=start, end=end, visual="画面 ➔ 特写", anchor_lines=list(anchors)
            )
        ],
    )


def valid_llm_script(chars_per_beat=(360, 360, 360)):
    roles = ["hook", "act", "outro"]
    labels = ["Hook 开场", "阶段一：入职即地狱", "收尾：修罗场引爆"]
    return LLMScript(
        beats=[
            # 10/310/610 三个起点全部避开 op_range (153.486, 224.681) 与 ed_range
            # 只有 beat1 anchor 到第 1 行（偏差 2.88s 在 5s 容差内）；
            # track 只有 6 行覆盖 7-51s，物理上覆盖不到 310s/610s
            # clip 长度按 chars/4.5 给足：原来固定 5 秒，对 360 字（80 秒）的旁白
            # 就是 16 倍拉伸，validate 的 A3「画面/旁白预算」会报 warning。clip 长度
            # 对这些测试是无关变量，给成跟旁白同量级才不会掩盖真正要断言的东西。
            llm_beat(
                f"b{i + 1}", labels[i], roles[i], chars,
                10.0 + i * 300, 10.0 + i * 300 + chars / 4.5,
                anchors=[1] if i == 0 else [],
            )
            for i, chars in enumerate(chars_per_beat)
        ]
    )


# --- prompt 素材块 ---


def test_dialogue_block_skips_credits_and_noise(track):
    text = build_dialogue_block(track)
    assert "河原正信" not in text
    assert "你是谁" in text


def test_dialogue_block_includes_line_numbers_and_kind(track):
    text = build_dialogue_block(track)
    assert "3 |" in text
    assert "monologue" in text


def test_dialogue_block_marks_suspect_lines(track):
    line = next(ln for ln in build_dialogue_block(track).splitlines() if "80-08" in ln)
    assert "?" in line


def test_dialogue_block_uses_readable_timestamps(track):
    assert "00:00:07.120" in build_dialogue_block(track)


def test_dialogue_block_fills_missing_speaker(track):
    line = next(ln for ln in build_dialogue_block(track).splitlines() if "你是谁" in ln)
    assert "| - |" in line


def test_highlight_block_lists_strength_and_triggers(report):
    text = build_highlight_block(report)
    assert "gap:19.8s" in text
    assert "强度 4" in text
    assert "00:22:08.367" in text


def test_highlight_block_empty_report_says_none():
    text = build_highlight_block(SignalReport(episode=2))
    assert "无" in text


def test_glossary_block_formats_pairs():
    assert "伊月 → 伊月" in build_glossary_block({"伊月": "伊月"})


def test_glossary_block_empty_says_none():
    assert "无" in build_glossary_block({})


# --- to_script ---


def test_to_script_maps_fields(cfg):
    script = to_script(valid_llm_script(), cfg, 2)
    assert script.show == "才女的侍从"
    assert script.mode == "single_episode"
    assert script.episodes == [2]
    assert script.target_seconds == pytest.approx(240.0)
    assert len(script.beats) == 3
    assert script.beats[0].label == "Hook 开场"
    assert script.beats[0].clips[0].visual == "画面 ➔ 特写"


def test_to_script_moves_audio_fields_into_audio_direction(cfg):
    llm = valid_llm_script()
    llm.beats[0].original_audio = "mute"
    script = to_script(llm, cfg, 2)
    assert script.beats[0].audio.original_audio == "mute"
    assert script.beats[0].audio.holds == []


def test_to_script_records_only_the_episode_being_generated():
    """单集流程只能记本集。原来写的是 cfg.episodes 的全部集数，于是 project.yaml 一旦
    登记了 ≥2 集，每一集的 script.episodes 都是「全季」，对照表标题全变成「整季」。"""
    multi = ProjectConfig.model_validate(
        {
            "show": "才女的侍从",
            "slug": "saijo",
            "target_seconds": 240,
            "episodes": [
                {"number": 1, "srt": "e01.srt"},
                {"number": 2, "srt": "e02.srt"},
                {"number": 3, "srt": "e03.srt"},
            ],
        }
    )
    script = to_script(valid_llm_script(), multi, 2)
    assert script.episodes == [2]


def test_multi_episode_project_table_title_is_not_season():
    """episodes bug 的用户可见后果：对照表标题。"""
    from tenmin.docgen.table import render_table

    multi = ProjectConfig.model_validate(
        {
            "show": "才女的侍从",
            "slug": "saijo",
            "episodes": [{"number": n, "srt": f"e{n:02d}.srt"} for n in (1, 2, 3)],
        }
    )
    title = render_table(to_script(valid_llm_script(), multi, 2)).splitlines()[0]
    assert title == "# 才女的侍从 第 2 集 解说方案"
    assert "整季" not in title


@pytest.mark.asyncio
async def test_generate_script_records_only_current_episode(track, report):
    """端到端：注册了 3 集的 project，跑 E02 出来的 script.episodes 必须是 [2]。"""
    multi = ProjectConfig.model_validate(
        {
            "show": "才女的侍从",
            "slug": "saijo",
            "target_seconds": 240,
            "episodes": [{"number": n, "srt": f"e{n:02d}.srt"} for n in (1, 2, 3)],
        }
    )
    provider = FakeProvider([valid_llm_script()])
    script, _ = await generate_script(multi, track, report, provider)
    assert script.episodes == [2]


# --- generate_script ---


@pytest.mark.asyncio
async def test_generate_script_happy_path(cfg, track, report):
    provider = FakeProvider([valid_llm_script()])
    script, warnings = await generate_script(cfg, track, report, provider)
    assert len(script.beats) == 3
    assert script.est_total_seconds == pytest.approx(240.0)
    assert warnings == []
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_generate_script_sends_schema_and_materials(cfg, track, report):
    provider = FakeProvider([valid_llm_script()])
    await generate_script(cfg, track, report, provider)
    call = provider.calls[0]
    assert call["schema"] is LLMScript
    assert "才女的侍从" in call["user"]
    assert "你是谁" in call["user"]
    assert "gap:19.8s" in call["user"]
    assert "Hook 开场" in call["user"]  # few-shot 范例已注入


@pytest.mark.asyncio
async def test_generate_script_retries_once_on_validation_error(cfg, track, report):
    bad = LLMScript(
        beats=[
            llm_beat("b1", "Hook 开场", "hook", 360, 9000.0, 9005.0),
            llm_beat("b2", "阶段一", "act", 360, 100.0, 105.0),
            llm_beat("b3", "收尾", "outro", 360, 200.0, 205.0),
        ]
    )
    provider = FakeProvider([bad, valid_llm_script()])
    script, _ = await generate_script(cfg, track, report, provider)
    assert len(provider.calls) == 2
    assert len(script.beats) == 3


@pytest.mark.asyncio
async def test_generate_script_gives_up_after_second_validation_error(cfg, track, report):
    bad = LLMScript(beats=[llm_beat("b1", "Hook 开场", "hook", 360, 9000.0, 9005.0)])
    provider = FakeProvider([bad, bad])
    from tenmin.script.validate import ScriptValidationError

    with pytest.raises(ScriptValidationError):
        await generate_script(cfg, track, report, provider)
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_generate_script_rewrites_when_over_budget(cfg, track, report):
    too_long = valid_llm_script(chars_per_beat=(600, 600, 600))  # 1800 字 = 400s
    provider = FakeProvider([too_long, valid_llm_script()])
    script, _ = await generate_script(cfg, track, report, provider)
    assert len(provider.calls) == 2
    assert "精简" in provider.calls[1]["user"]
    assert script.est_total_seconds == pytest.approx(240.0)


@pytest.mark.asyncio
async def test_generate_script_accepts_second_draft_even_if_still_off(cfg, track, report):
    too_long = valid_llm_script(chars_per_beat=(600, 600, 600))
    provider = FakeProvider([too_long, too_long])
    script, warnings = await generate_script(cfg, track, report, provider)
    assert len(provider.calls) == 2
    assert any("容差" in w for w in warnings)
    assert script.est_total_seconds == pytest.approx(400.0)


@pytest.mark.asyncio
async def test_generate_script_propagates_validation_warnings(cfg, track, report):
    llm = valid_llm_script()
    llm.beats[1].clips.append(LLMClip(episode=2, start=9000.0, end=9005.0, visual="假的"))
    provider = FakeProvider([llm])
    _, warnings = await generate_script(cfg, track, report, provider)
    assert any("越界" in w for w in warnings)


@pytest.mark.asyncio
async def test_generate_script_marks_silent_highlight(cfg, track, report):
    llm = valid_llm_script()
    llm.beats[2].clips = [
        LLMClip(episode=2, start=1330.0, end=1340.0, visual="定格收尾", anchor_lines=[])
    ]
    provider = FakeProvider([llm])
    script, _ = await generate_script(cfg, track, report, provider)
    assert script.beats[2].clips[0].is_silent_highlight is True
