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
    build_credits_block,
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


def test_dialogue_block_gives_plain_seconds(track):
    """模板要求 `clip.start` / `clip.end` 填**秒数**，而这份清单原来给的是
    `00:00:07.120`。模型每写一个 clip 都得做一次 60 进制换算，错一次的量级是 60 或 600
    秒 —— 超过 validate 的 ANCHOR_OVERWRITE_MAX_SECONDS(60)，于是不纠正只 warning，
    接着大概率被当成越界 clip 丢掉。顺带：秒数比时间码短一半，408 行省约 4.9k 字符。"""
    line = next(ln for ln in build_dialogue_block(track).splitlines() if "你是谁" in ln)
    assert line.startswith("1 | 7.1 - 11.5 |")


def test_dialogue_block_does_not_leak_clock_timestamps(track):
    """两套单位共存比单给一套更糟：模型会分不清该抄哪个。"""
    assert "00:00:07.120" not in build_dialogue_block(track)


def test_dialogue_block_fills_missing_speaker(track):
    line = next(ln for ln in build_dialogue_block(track).splitlines() if "你是谁" in ln)
    assert "| - |" in line


def test_highlight_block_lists_strength_and_triggers(report):
    text = build_highlight_block(report)
    assert "gap:19.8s" in text
    assert "强度 4" in text
    assert "1328.4 - 1348.2" in text


def test_highlight_block_does_not_leak_clock_timestamps(report):
    assert "00:22:08.367" not in build_highlight_block(report)


def test_highlight_block_prints_precomputed_anchor_lines(report):
    """`Highlight.anchor_lines` 是 signals/gaps.py 的 _piece_anchors 已经算好的
    「紧邻这段间隙前后的对白行号」，但原来一个字都不印，模板 :35 却让模型自己去 400 行
    对白轨里找同一份东西。不印是纯粹的信息浪费，而模型找错的后果是 clip 被 anchor 校正
    搬到别处（validate.py 的 R2）。"""
    assert "锚点行 1" in build_highlight_block(report)


def test_highlight_block_omits_the_anchor_column_when_there_are_none(report):
    """两头都被 OP/ED 裁掉的间隙拿不到锚点行（_piece_anchors 要求端点浮点相等）。
    那种情况下印一个空列只是噪声。"""
    bare = SignalReport(
        episode=2,
        highlights=[
            Highlight(
                start=10.0,
                end=20.0,
                strength=2,
                triggers=["gap:10.0s"],
                summary="无台词演出段 10.0s",
            )
        ],
    )
    assert "锚点行" not in build_highlight_block(bare)


def test_highlight_block_empty_report_says_none():
    text = build_highlight_block(SignalReport(episode=2))
    assert "无" in text


def test_glossary_block_formats_pairs():
    assert "伊月 → 伊月" in build_glossary_block({"伊月": "伊月"})


def test_glossary_block_empty_says_none():
    assert "无" in build_glossary_block({})


# --- credits block ---
#
# 模板两次要求「不要落在片头曲/片尾曲区间」（clips 小节与自检第 6 条），而
# validate.py 的执行方式是静默丢 clip：与 OP/ED 重叠达 CREDITS_OVERLAP_MAX_RATIO
# (0.5) 就整条扔掉，丢光一个节点的 clip 还会抛 ScriptValidationError、烧掉一整轮
# 重试。track 里本来就有 op_range / ed_range，不给模型等于让它盲猜。


def test_credits_block_prints_both_ranges_in_seconds(track):
    text = build_credits_block(track)
    assert "153.5 - 224.7" in text
    assert "1348.2 - 1416.6" in text


def test_credits_block_uses_the_same_unit_as_clip_timestamps(track):
    """必须是纯秒数：模型要照着它判断自己写的 clip.start 落没落进禁区。"""
    assert "00:02:33" not in build_credits_block(track)


def test_credits_block_marks_a_missing_range_instead_of_dropping_it(track):
    """
    只推断出一段时，缺的那段要显式说「未识别」。

    静默省略会让模型把「没提片尾曲」读成「本集没有片尾曲」，从而放心去切最后
    90 秒——那恰好是丢 clip 最集中的区域。
    """
    only_op = track.model_copy(update={"ed_range": None})
    text = build_credits_block(only_op)
    assert "153.5 - 224.7" in text
    assert "未识别" in text


def test_credits_block_falls_back_when_nothing_was_inferred(track):
    """
    两段都没推出来时给一句可执行的回避指引。

    这条分支是常态而非边缘：credits.py 的 OP 推断与静音兜底共用 [30, 300] 的搜索
    窗，OP 起点早于 30 秒的番、或把 OP 歌词打成字幕的番，两条路都推不出来。
    """
    blind = track.model_copy(update={"op_range": None, "ed_range": None})
    text = build_credits_block(blind)
    assert "未识别" in text
    assert "153.5" not in text


def test_prompt_injects_the_credit_ranges(cfg, track, report):
    assert "1348.2 - 1416.6" in build_user_prompt(cfg, track, report)


# --- schema 字段与 prompt 的一致性 ---
#
# 这三条锁的是「模板描述的形状」必须等于「LLMScript 真实的形状」。不一致的代价不是
# 质量下降而是硬失败：_StageModel 是 extra="forbid"，字面照着模板写就是
# ValidationError，在 gemini 之外的 provider 上还会白烧一轮 schema 修复。


def test_prompt_does_not_describe_a_nested_audio_object(cfg, track, report):
    """
    LLMBeat 是平的：original_audio / holds / sfx 直接挂在节点上（models.py:334-336）。

    模板原先写 `audio.original_audio` / `audio.sfx`，而 LLMBeat 上并没有 `audio`
    这个字段，且 extra="forbid" —— 模型照着写就是 ValidationError。嵌套的
    `hold.at` / `sfx.cue` 是对的（Hold / SfxCue 确实是对象），不要连它们一起改。
    """
    prompt = build_user_prompt(cfg, track, report)
    assert "audio.original_audio" not in prompt
    assert "audio.sfx" not in prompt
    assert "`original_audio`" in prompt
    assert "`sfx`" in prompt


def test_prompt_tells_the_model_that_every_beat_needs_an_id(cfg, track, report):
    """
    LLMBeat.id 是必填的 NonBlankStr，且重复 id 会被 model_validator 直接拒掉
    （models.py:329,342-345）——那是硬失败，不是 warning。模板此前一个字没提。
    """
    assert "`id`" in build_user_prompt(cfg, track, report)


def test_prompt_tells_the_model_what_clip_episode_should_be(cfg, track, report):
    """
    LLMClip.episode 是必填的（models.py:321），填错则 validate 静默丢掉该 clip
    （R1），丢光一个节点就抛 ScriptValidationError。模板此前只有「集数：第 N 集」
    这一行素材，没说过它要被抄进每个 clip。
    """
    assert "`clip.episode`" in build_user_prompt(cfg, track, report)


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


# --- clip 上界的单位不能跟「正片秒数」冲突 ---


def test_prompt_states_the_clip_upper_bound_in_seconds(cfg, track, report):
    """`clip.start/end 是正片秒数，必须落在 0 到 X 之内` 里的 X 原来填的是
    duration_readable（"23 分 36 秒"），跟同一句话里的「秒数」单位冲突，逼模型自己
    换算——正是 validate.py 那条「clip 越界丢弃」在兜的坑。现在填纯秒数。"""
    text = build_user_prompt(cfg, track, report)
    bound = str(int(track.duration))  # 向下取整：validate 判的是 clip.end > track.duration
    assert f"`0` 到 `{bound}` 秒" in text


def test_prompt_still_carries_the_human_readable_duration(cfg, track, report):
    """秒数是给模型算时间戳用的，人类可读串在「本期素材」里该留着——它是唯一一处
    让人（读 prompt 排查问题的人）一眼看出这集有多长的地方。"""
    from tenmin.timecode import readable_seconds

    assert readable_seconds(track.duration) in build_user_prompt(cfg, track, report)


# --- 静态段前置，范例留在末尾（prefix 缓存） ---

_DYNAMIC_ZONE_HEADING = "## 本期素材"


def _at(text: str, heading: str) -> int:
    """二级/三级标题在整份 prompt 里的位置。带上行首换行，免得 `## 高能点清单`
    误命中静态段里的 `### 高能点清单`。"""
    return text.index(f"\n{heading}\n")


def _static_head(text: str) -> str:
    """整份 prompt 里「一个字都不随集数变」的那段前缀。"""
    return text[: _at(text, _DYNAMIC_ZONE_HEADING) + 1]


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def test_static_head_comes_before_every_dynamic_block(cfg, track, report):
    """交付要求与输出格式原来排在动态素材**后面**，任何一集变化都让它们的缓存前缀失效。"""
    text = build_user_prompt(cfg, track, report)
    static_end = _at(text, _DYNAMIC_ZONE_HEADING)
    for heading in (
        "## 素材格式说明",
        "## 交付要求",
        "### 硬性要求（违反任一条即为废稿）",
        "## 输出格式",
    ):
        assert _at(text, heading) < static_end, heading
    for heading in ("## 术语表（专有名词必须按这个写法）", "## 高能点清单", "## 对白轨"):
        assert _at(text, heading) > static_end, heading


def test_static_head_is_byte_identical_across_episodes(cfg, track, report):
    """缓存前缀的唯一判据：换一集之后这段前缀必须**逐字节相同**。"""
    other_track = DialogueTrack(
        episode=7,
        duration=1402.5,
        lines=[dline(1, 1.0, 2.0, "完全不同的一集")],
    )
    other_report = SignalReport(episode=7, median_char_rate=4.0)
    head = _static_head(build_user_prompt(cfg, track, report))
    other_head = _static_head(build_user_prompt(cfg, other_track, other_report))
    assert head == other_head
    # 交付要求 + 输出格式 + 素材格式说明 合起来是 2KB 量级，不能只剩个招呼语。
    assert len(head) > 2000


def test_followup_prompt_is_a_strict_prefix_of_the_first_round(cfg, track, report):
    """**本任务里最重的一条不变量。** 返工轮除了摘掉 few-shot 范例什么都没动，所以只要
    范例排在全部素材之后，返工轮的整份素材就是首轮的一个严格前缀（实测 saijo 10 集
    31.6k 字符全部可缓存）。把范例跟其余静态段一起前置会让两轮在第 2.4k 字符处分叉，
    后面 29k 素材白发一遍 —— 实测「一次 10 集批处理里可缓存的字符占比」从 43.0% 掉到
    9.9%，差 4 倍且方向是反的。"""
    first = build_user_prompt(cfg, track, report)
    followup = build_user_prompt(cfg, track, report, with_example=False)
    shared = _common_prefix(first, followup)
    # 分叉点就在 `## 参考范例` 这个标题上（两个标题共享的 "## " 让共同前缀再多 3 字节）。
    assert shared >= _at(first, "## 参考范例") + 1
    assert shared > _at(first, "## 对白轨"), "共同前缀必须一直延伸到对白轨之后"


def test_example_section_sits_after_all_of_the_episode_material(cfg, track, report):
    text = build_user_prompt(cfg, track, report)
    assert _at(text, "## 参考范例") > _at(text, "## 对白轨")


def test_hard_requirements_sit_next_to_the_output_format(cfg, track, report):
    """审查结论：5 条废稿条件排在末尾、而 3KB 范例又在它们之后，
    长上下文里位置最劣。重排后它们必须紧贴「输出格式」。"""
    text = build_user_prompt(cfg, track, report)
    between = text[_at(text, "### 硬性要求（违反任一条即为废稿）") : _at(text, "## 输出格式")]
    assert "\n## " not in between, "硬性要求与输出格式之间不许再插别的二级小节"


def test_self_check_is_the_last_section_of_the_prompt(cfg, track, report):
    """静态段前置 + 范例留在末尾之后，最后一眼看到的东西不能是「不要学它的内容」的范例。"""
    for with_example in (True, False):
        text = build_user_prompt(cfg, track, report, with_example=with_example)
        tail = text[_at(text, "## 输出前自检") :]
        assert "\n## " not in tail[1:], "自检必须是最后一节"
        assert "JSON schema" in tail
        assert str(int(track.duration)) in tail, "clip 上界要在末尾复述一次"


# --- 报错消息要指名道姓是哪份模板 ---


def test_prompt_errors_name_the_template_file(cfg, track, report, monkeypatch):
    """render_prompt 只认得到「一个 str 模板」，模板名得由调用点告诉它 —— 不告诉的话
    报错消息里是 `<inline>`，排查的人不知道该去改哪份 md。"""
    from tenmin.script.prompt import PromptTemplateError

    monkeypatch.setattr("tenmin.script.single.load_prompt", lambda name: "{{typo_here}}")
    with pytest.raises(PromptTemplateError) as exc:
        build_user_prompt(cfg, track, report)
    assert "single_episode.md" in str(exc.value)
    assert "typo_here" in str(exc.value)
