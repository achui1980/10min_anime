from pathlib import Path

import pytest

from tenmin.docgen.table import (
    LEGEND,
    escape_cell,
    render_audio_cell,
    render_table,
    render_timestamp_cell,
)
from tenmin.models import AudioDirection, Beat, Clip, Hold, Script, SfxCue


def clip(start, end, visual="画面", silent=False, episode=2):
    return Clip(
        episode=episode, start=start, end=end, visual=visual, is_silent_highlight=silent
    )


def beat(label, narration, clips, audio=None, role="act", bid="b1"):
    return Beat(
        id=bid,
        label=label,
        role=role,
        narration=narration,
        clips=list(clips),
        audio=audio or AudioDirection(),
    )


def script(beats, show="才女的侍从", episodes=(2,)):
    return Script(show=show, episodes=list(episodes), beats=list(beats))


# --- 单元格渲染 ---


def test_escape_cell_escapes_pipe():
    assert escape_cell("a|b") == "a\\|b"


def test_escape_cell_escapes_angle_brackets():
    """narration 里的 `<` 会被 markdown 渲染器当 HTML 标签吃掉，整段文案消失。"""
    assert escape_cell("他说<很生气>") == "他说&lt;很生气&gt;"


def test_escape_cell_escapes_backtick():
    assert escape_cell("按 `Enter`") == "按 \\`Enter\\`"


def test_escape_cell_escapes_ampersand_before_entities():
    """先转 & 再转 <>，否则原文里的 `&lt;` 会被渲染成 `<`。"""
    assert escape_cell("A&B") == "A&amp;B"
    assert escape_cell("&lt;") == "&amp;lt;"


def test_escape_cell_keeps_generated_br_intact():
    """换行转成的 <br> 是我们自己生成的标签，不能被 <> 转义连带干掉。"""
    assert escape_cell("a\nb") == "a<br>b"


def test_escape_cell_converts_newline_to_br():
    assert escape_cell("a\nb") == "a<br>b"


def test_escape_cell_strips_outer_whitespace():
    assert escape_cell("  文案  ") == "文案"


def test_timestamp_cell_single_clip():
    assert render_timestamp_cell([clip(7.12, 11.48)]) == "00:00:07.120 - 00:00:11.480"


def test_timestamp_cell_marks_silent_highlight():
    assert render_timestamp_cell([clip(1100.1, 1104.8, silent=True)]).endswith(" ★")


def test_timestamp_cell_joins_with_jie():
    cell = render_timestamp_cell([clip(7.12, 11.48), clip(1100.1, 1104.8, silent=True)])
    assert cell == "00:00:07.120 - 00:00:11.480<br>接 00:18:20.100 - 00:18:24.800 ★"


def test_timestamp_cell_empty_clips():
    assert render_timestamp_cell([]) == "-"


def test_audio_cell_duck_default():
    assert render_audio_cell(AudioDirection()) == "原声压低垫底"


def test_audio_cell_mute():
    assert render_audio_cell(AudioDirection(original_audio="mute")) == "原声静音"


def test_audio_cell_full():
    assert render_audio_cell(AudioDirection(original_audio="full")) == "原声全开"


def test_audio_cell_includes_holds_and_sfx():
    audio = AudioDirection(
        holds=[Hold(at=1.0, duration=3.0, quote="就是会让人硬不起来的药")],
        sfx=[SfxCue(at=1.5, cue="impact")],
    )
    assert render_audio_cell(audio) == (
        "原声压低垫底<br>留白 3.0s：「就是会让人硬不起来的药」<br>音效 impact @1.5s"
    )


def test_audio_cell_multiple_holds():
    audio = AudioDirection(
        holds=[Hold(at=1.0, duration=3.0, quote="甲"), Hold(at=8.0, duration=2.5, quote="乙")]
    )
    assert render_audio_cell(audio) == "原声压低垫底<br>留白 3.0s：「甲」<br>留白 2.5s：「乙」"


# --- 整表渲染 ---


def test_render_table_exact_output():
    s = script(
        [
            beat(
                "Hook 开场",
                "文案甲",
                [clip(7.12, 11.48, "伊月被扑倒 ➔ 大小姐居高临下")],
                audio=AudioDirection(holds=[Hold(at=1.0, duration=3.0, quote="金句")]),
                role="hook",
                bid="b1",
            ),
            beat(
                "阶段一：入职即地狱",
                "文案乙",
                [clip(100.0, 105.0, "全景"), clip(1330.0, 1340.0, "定格", silent=True)],
                bid="b2",
            ),
        ]
    )
    assert render_table(s) == (
        "# 才女的侍从 第 2 集 解说方案\n"
        "\n"
        "旁白 6 字 · 估算时长 4 秒 · 2 个节点\n"
        "\n"
        "| 节点 | 原片截取时间戳 | 建议画面特征 | 分段解说文案 | 剪辑与原声处理 |\n"
        "|---|---|---|---|---|\n"
        "| Hook 开场 | 00:00:07.120 - 00:00:11.480 | 伊月被扑倒 ➔ 大小姐居高临下 |"
        " 文案甲 | 原声压低垫底<br>留白 3.0s：「金句」 |\n"
        "| 阶段一：入职即地狱 | 00:01:40.000 - 00:01:45.000<br>"
        "接 00:22:10.000 - 00:22:20.000 ★ | 全景 ➔ 定格 | 文案乙 | 原声压低垫底 |\n"
        "\n"
        "★ = 该片段命中无字幕演出高光区间，纯字幕方案取不到\n"
    )


def test_render_table_ends_with_single_newline():
    out = render_table(script([beat("Hook 开场", "文案", [clip(1.0, 2.0)], role="hook")]))
    assert out.endswith("取不到\n")
    assert not out.endswith("\n\n")


def test_render_table_season_title():
    s = script([beat("Hook 开场", "文案", [clip(1.0, 2.0)], role="hook")], episodes=(1, 2, 3))
    assert s.show in render_table(s).splitlines()[0]
    assert "整季" in render_table(s).splitlines()[0]


def test_render_table_meta_line_uses_minutes_above_sixty_seconds():
    long_narration = "啊" * (45 * 22)  # 990 字 = 220 秒
    s = script([beat("Hook 开场", long_narration, [clip(1.0, 2.0)], role="hook")])
    meta = render_table(s).splitlines()[2]
    assert meta == "旁白 990 字 · 估算时长 3 分 40 秒 · 1 个节点"


def test_render_table_counts_all_beats_narration():
    s = script(
        [
            beat("Hook 开场", "啊" * 10, [clip(1.0, 2.0)], role="hook", bid="b1"),
            beat("阶段一", "啊" * 20, [clip(3.0, 4.0)], bid="b2"),
        ]
    )
    assert "旁白 30 字" in render_table(s)


def test_render_table_visual_joins_multiple_clips():
    s = script(
        [beat("Hook 开场", "文案", [clip(1.0, 2.0, "甲"), clip(3.0, 4.0, "乙")], role="hook")]
    )
    assert "甲 ➔ 乙" in render_table(s)


def test_render_table_escapes_pipe_in_narration():
    s = script([beat("Hook 开场", "他说|然后", [clip(1.0, 2.0)], role="hook")])
    assert "他说\\|然后" in render_table(s)


def test_render_table_empty_visual_shows_dash():
    s = script([beat("Hook 开场", "文案", [clip(1.0, 2.0, "")], role="hook")])
    row = [ln for ln in render_table(s).splitlines() if ln.startswith("| Hook")][0]
    assert row.split(" | ")[2] == "-"


def test_legend_constant_is_in_output():
    s = script([beat("Hook 开场", "文案", [clip(1.0, 2.0)], role="hook")])
    assert LEGEND in render_table(s)


def test_render_table_no_beats_still_renders_header():
    out = render_table(script([]))
    assert "| 节点 | 原片截取时间戳" in out
    assert "0 个节点" in out


# --- 未知 original_audio 的兜底 ---


def test_audio_cell_falls_back_on_unknown_original_audio():
    """给 OriginalAudio Literal 加成员时，忘了同步 _ORIGINAL_AUDIO_LABELS 不该 KeyError。"""
    audio = AudioDirection.model_construct(original_audio="karaoke", sfx=[], holds=[])
    assert render_audio_cell(audio) == "原声处理 karaoke"


# --- 估算时长的单一数据源 ---


def test_render_table_prefers_stored_estimates():
    """budget.apply_estimates 已经把估算写进 script.json 了，对照表不该再自己算一遍：
    用户手改 script.json 后两个数字会不一致。"""
    b = beat("Hook 开场", "啊" * 10, [clip(1.0, 2.0)], role="hook")
    b.est_seconds = 123.0
    s = script([b])
    s.est_total_seconds = 123.0
    assert "估算时长 2 分 3 秒" in render_table(s)


def test_render_table_falls_back_when_estimates_missing():
    """est_* 还是默认的 0（budget 没跑过）时回退到重算，输出与旧行为一致。"""
    s = script([beat("Hook 开场", "啊" * 45, [clip(1.0, 2.0)], role="hook")])
    assert s.est_total_seconds == 0.0
    assert "估算时长 10 秒" in render_table(s)


def test_render_table_stored_and_recomputed_agree_on_real_script():
    """正常流程里 budget 一定跑过，两条路径必须给出同一个数字 —— 这条改动不改变现有输出。"""
    path = Path(__file__).parent / "fixtures" / "akujo_e02.script.json"
    if not path.exists():
        pytest.skip("缺少 akujo_e02.script.json")
    stored = Script.model_validate_json(path.read_text(encoding="utf-8"))
    assert stored.est_total_seconds > 0
    recomputed = stored.model_copy(deep=True)
    recomputed.est_total_seconds = 0.0
    for b in recomputed.beats:
        b.est_seconds = 0.0
    assert render_table(stored) == render_table(recomputed)
