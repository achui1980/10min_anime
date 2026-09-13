import pytest

from tenmin.config import ProjectConfig
from tenmin.models import (
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
    SfxCue,
    Signal,
    SignalReport,
)
from tenmin.script.single import (
    SYSTEM_PROMPT,
    build_dialogue_block,
    build_glossary_block,
    build_highlight_block,
    build_user_prompt,
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


# --- E7：SYSTEM_PROMPT 搬进 prompts/system.md ---


def test_system_prompt_comes_from_the_prompt_file():
    from tenmin.script.prompt import load_prompt

    assert load_prompt("system.md").strip() == SYSTEM_PROMPT.strip()


# --- E8：to_script 的字段映射不能静默漏字段 ---


def test_to_script_covers_every_llm_field():
    """原来是手工逐字段搬运，给 Clip/Beat 新增字段时会静默丢失（不报错、不传值）。"""
    llm = LLMScript(
        beats=[
            LLMBeat(
                id="b1",
                label="Hook 开场",
                role="hook",
                narration="旁白",
                clips=[
                    LLMClip(
                        episode=2, start=1.0, end=9.0, visual="A ➔ B", anchor_lines=[3, 4]
                    )
                ],
                original_audio="mute",
                holds=[Hold(at=1.5, duration=2.0, quote="金句", note="备注")],
                sfx=[SfxCue(at=0.5, cue="impact", note="砸")],
            )
        ]
    )
    cfg = ProjectConfig(show="X", slug="x")
    beat = to_script(llm, cfg, 2).beats[0]
    # LLMBeat 的字段要么原样落在 Beat 上，要么落进 Beat.audio；LLMClip 的全落在 Clip 上。
    beat_side = set(Beat.model_fields) | set(AudioDirection.model_fields)
    assert set(LLMBeat.model_fields) <= beat_side
    assert set(LLMClip.model_fields) <= set(Clip.model_fields)

    # 逐字段比值，一个都不许漏。这是 E8 的核心断言：只看「字段名对得上」抓不到
    # 「字段名对得上但根本没赋值」。
    llm_dump = llm.beats[0].model_dump()
    merged = beat.model_dump() | beat.audio.model_dump()
    for name in LLMBeat.model_fields:
        if name == "clips":
            continue
        assert merged[name] == llm_dump[name], name
    for name in LLMClip.model_fields:
        assert beat.clips[0].model_dump()[name] == llm_dump["clips"][0][name], name


def test_to_script_does_not_let_the_llm_fill_computed_fields():
    """LLM* 镜像模型刻意不含 est_seconds / est_total_seconds / is_silent_highlight
    （models.py:284 的注释说明这是设计意图），所以映射不能盲目全字段对拷。"""
    computed = {"est_seconds", "est_total_seconds", "is_silent_highlight"}
    assert computed & set(LLMBeat.model_fields) == set()
    assert computed & set(LLMClip.model_fields) == set()
    assert computed & set(LLMScript.model_fields) == set()
    script = to_script(valid_llm_script(), ProjectConfig(show="X", slug="x"), 2)
    assert script.est_total_seconds == 0.0
    assert script.beats[0].est_seconds == 0.0
    assert script.beats[0].clips[0].is_silent_highlight is False


# --- E9：对白块要标出合并来源 ---


def test_dialogue_block_marks_merged_source_line_numbers():
    """validate.py 的 anchor 匹配**会认** merged_from 里的旧行号，而对白块原来不输出
    它们——模型看不到这些号，双向信息不对称。"""
    track = DialogueTrack(
        episode=2,
        duration=100.0,
        lines=[
            DialogueLine(
                idx=7,
                start=1.0,
                end=3.0,
                text="合并后的一整句",
                raw="合并后的一整句",
                merged_from=[7, 8, 9],
            )
        ],
    )
    line = build_dialogue_block(track)
    assert "7" in line
    assert "8" in line and "9" in line


def test_dialogue_block_omits_the_marker_when_nothing_was_merged(track):
    assert "+" not in build_dialogue_block(track)


# --- E4：重试轮/重写轮把上一版交回模型，且不再重发 few-shot 范例 ---


@pytest.mark.asyncio
async def test_retry_sends_the_previous_draft_back(cfg, track, report):
    bad = LLMScript(
        beats=[
            llm_beat("b1", "Hook 开场", "hook", 360, 9000.0, 9080.0),
            llm_beat("b2", "阶段一", "act", 360, 100.0, 180.0),
            llm_beat("b3", "收尾：完", "outro", 360, 300.0, 380.0),
        ]
    )
    provider = FakeProvider([bad, valid_llm_script()])
    await generate_script(cfg, track, report, provider)
    retry = provider.calls[1]["user"]
    assert "## 上一版输出" in retry
    assert '"id": "b1"' in retry, "上一版的 JSON 必须原样交回去"


@pytest.mark.asyncio
async def test_followup_rounds_drop_the_few_shot_example(cfg, track, report):
    """few-shot 范例只教「格式、语气、节奏」，而模型这时已经交出过一份合 schema 的稿子，
    格式它显然学会了；范例自己还带着「不要学它的内容」的警告，去掉只减少污染风险。"""
    too_long = valid_llm_script(chars_per_beat=(600, 600, 600))
    provider = FakeProvider([too_long, valid_llm_script()])
    await generate_script(cfg, track, report, provider)
    first, rewrite = provider.calls[0]["user"], provider.calls[1]["user"]
    assert "## 参考范例" in first
    assert "## 参考范例" not in rewrite


@pytest.mark.asyncio
async def test_followup_rounds_still_resend_the_dialogue_track(cfg, track, report):
    """对白轨占整份 prompt 的 78.2%，但**必须**重发：重试要修的语义错误（时间戳越界、
    人物关系写反、事件顺序）全部只能对着对白原文才判得出来。"""
    too_long = valid_llm_script(chars_per_beat=(600, 600, 600))
    provider = FakeProvider([too_long, valid_llm_script()])
    await generate_script(cfg, track, report, provider)
    rewrite = provider.calls[1]["user"]
    assert "你是谁" in rewrite  # 对白轨
    assert "gap:19.8s" in rewrite  # 高能点清单


# --- E2/E3：两版择优 + warning 跟着被采纳的版本 ---


@pytest.mark.asyncio
async def test_rewrite_keeps_the_first_draft_when_the_second_is_worse(cfg, track, report):
    """原来无条件用新稿替换旧稿，即使新稿偏差更大。"""
    first = valid_llm_script(chars_per_beat=(430, 430, 430))  # 1290 字 = 286.7s，+19.4%
    worse = valid_llm_script(chars_per_beat=(700, 700, 700))  # 2100 字 = 466.7s，+94.4%
    provider = FakeProvider([first, worse])
    script, warnings = await generate_script(cfg, track, report, provider)
    assert script.est_total_seconds == pytest.approx(286.666, abs=0.01)
    assert any("采纳" in w and "首版" in w for w in warnings), warnings


@pytest.mark.asyncio
async def test_rewrite_adopts_the_second_draft_when_it_is_better(cfg, track, report):
    first = valid_llm_script(chars_per_beat=(700, 700, 700))
    better = valid_llm_script(chars_per_beat=(430, 430, 430))
    provider = FakeProvider([first, better])
    script, warnings = await generate_script(cfg, track, report, provider)
    assert script.est_total_seconds == pytest.approx(286.666, abs=0.01)
    assert any("采纳" in w and "重写版" in w for w in warnings), warnings


@pytest.mark.asyncio
async def test_warnings_belong_to_the_adopted_draft_only(cfg, track, report):
    """真实后果：首版触发重写时，首版的 validate warning（「clip 落在片头曲内，丢弃」）
    已经进了列表，而那份稿子随后被丢弃——用户看到的是在描述一份**不存在的稿子**。"""
    first = valid_llm_script(chars_per_beat=(700, 700, 700))
    first.beats[1].clips.append(LLMClip(episode=2, start=160.0, end=200.0, visual="片头曲"))
    better = valid_llm_script(chars_per_beat=(430, 430, 430))
    provider = FakeProvider([first, better])
    _, warnings = await generate_script(cfg, track, report, provider)
    assert not any("片头" in w for w in warnings), warnings


@pytest.mark.asyncio
async def test_warnings_of_the_kept_first_draft_are_reported(cfg, track, report):
    first = valid_llm_script(chars_per_beat=(430, 430, 430))
    first.beats[1].clips.append(LLMClip(episode=2, start=160.0, end=200.0, visual="片头曲"))
    worse = valid_llm_script(chars_per_beat=(700, 700, 700))
    provider = FakeProvider([first, worse])
    _, warnings = await generate_script(cfg, track, report, provider)
    assert any("片头" in w for w in warnings), warnings


# --- E5：重试次数/重写轮数/tolerance 可配 ---


@pytest.mark.asyncio
async def test_validation_retries_comes_from_config(cfg, track, report):
    bad = LLMScript(beats=[llm_beat("b1", "Hook 开场", "hook", 360, 9000.0, 9080.0)])
    cfg.llm.validation_retries = 2
    provider = FakeProvider([bad, bad, valid_llm_script()])
    script, _ = await generate_script(cfg, track, report, provider)
    assert len(provider.calls) == 3
    assert len(script.beats) == 3


@pytest.mark.asyncio
async def test_budget_rewrite_rounds_comes_from_config(cfg, track, report):
    too_long = valid_llm_script(chars_per_beat=(600, 600, 600))
    cfg.llm.budget_rewrite_rounds = 0
    provider = FakeProvider([too_long])
    _, warnings = await generate_script(cfg, track, report, provider)
    assert len(provider.calls) == 1
    assert any("容差" in w for w in warnings)


@pytest.mark.asyncio
async def test_budget_tolerance_comes_from_config(cfg, track, report):
    """1080 字 = 240s 正好达标；把容差收到 0 之后 +0.0% 仍然不超，
    但 1290 字（+19.4%）在默认 12% 下要重写、把容差放到 30% 就不该重写。"""
    cfg.llm.budget_tolerance = 0.30
    provider = FakeProvider([valid_llm_script(chars_per_beat=(430, 430, 430))])
    _, warnings = await generate_script(cfg, track, report, provider)
    assert len(provider.calls) == 1
    assert warnings == []


# --- E1：校验失败时把最后一版稿子留下来 ---


@pytest.mark.asyncio
async def test_final_validation_error_carries_the_rejected_script(cfg, track, report):
    """原来第二次仍抛 ScriptValidationError 时异常直接冒出，这次几百秒的昂贵调用
    产物一点没留。"""
    from tenmin.script.validate import ScriptValidationError

    bad = LLMScript(beats=[llm_beat("b1", "Hook 开场", "hook", 360, 9000.0, 9080.0)])
    provider = FakeProvider([bad, bad])
    with pytest.raises(ScriptValidationError) as exc:
        await generate_script(cfg, track, report, provider)
    assert exc.value.script is not None
    assert exc.value.script.beats[0].id == "b1"


# --- E6：script 阶段的心跳 ---


@pytest.mark.asyncio
async def test_generate_script_reports_each_round(cfg, track, report):
    class Recorder:
        def __init__(self):
            self.substeps = []

        def substep(self, stage, current, total, label):
            self.substeps.append((stage, current, total, label))

        def __getattr__(self, name):
            return lambda *a, **k: None

    too_long = valid_llm_script(chars_per_beat=(600, 600, 600))
    provider = FakeProvider([too_long, valid_llm_script()])
    recorder = Recorder()
    await generate_script(cfg, track, report, provider, reporter=recorder)
    stages = [s[0] for s in recorder.substeps]
    assert stages == ["script", "script"]
    assert "初稿" in recorder.substeps[0][3]
    assert "返工" in recorder.substeps[1][3]


# --- D3：提示词里的语速/容差/字数预算不再是第二份真相 ---


def test_prompt_injects_the_tolerance_from_config(cfg, track, report):
    cfg.llm.budget_tolerance = 0.05
    text = build_user_prompt(cfg, track, report)
    assert "正负 5%" in text
    assert "12%" not in text


def test_prompt_injects_the_speech_rate(cfg, track, report):
    assert "4.5 字/秒" in build_user_prompt(cfg, track, report)


def test_prompt_char_budget_deducts_the_hold_reserve(cfg, track, report):
    """提示词原来写「合计约 target × 4.5 字」，跟 budget.budget_chars 犯的是同一个错：
    没扣掉留白占走的时间。实测 13 份真实产物的全片留白是 10.5–21.0 秒。"""
    from tenmin.script.single import HOLD_RESERVE_SECONDS

    text = build_user_prompt(cfg, track, report)
    assert str(int((240.0 - HOLD_RESERVE_SECONDS) * 4.5)) in text


def test_prompt_char_budget_follows_the_tts_rate(cfg, track, report):
    from tenmin.script.single import HOLD_RESERVE_SECONDS

    cfg.render.rate = "+20%"
    text = build_user_prompt(cfg, track, report)
    assert str(int((240.0 - HOLD_RESERVE_SECONDS) * 4.5 * 1.2)) in text
