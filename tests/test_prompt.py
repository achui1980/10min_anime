import pytest

from tenmin.script.prompt import PROMPTS_DIR, load_prompt, render_prompt


def test_render_replaces_placeholder():
    assert render_prompt("你好 {{name}}", name="世界") == "你好 世界"


def test_render_replaces_all_occurrences():
    assert render_prompt("{{a}}-{{a}}", a="x") == "x-x"


def test_render_leaves_json_braces_untouched():
    template = 'schema: {"beats": [{"id": 1}]} 目标 {{target}}'
    out = render_prompt(template, target="240")
    assert '{"beats": [{"id": 1}]}' in out
    assert "240" in out


def test_render_missing_variable_raises():
    with pytest.raises(KeyError) as exc:
        render_prompt("你好 {{name}}", other="x")
    assert "name" in str(exc.value)


def test_render_ignores_unused_kwargs():
    assert render_prompt("固定文本", unused="x") == "固定文本"


def test_load_prompt_reads_file():
    text = load_prompt("single_episode.md")
    assert "Hook 开场" in text
    assert "{{episode_number}}" in text


def test_load_prompt_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("nope.md")


def test_single_episode_prompt_declares_all_placeholders():
    text = load_prompt("single_episode.md")
    for name in (
        "show",
        "episode_number",
        "target_seconds",
        "duration_readable",
        "duration_seconds",
        "glossary_block",
        "highlight_block",
        "dialogue_block",
        "example_block",
    ):
        assert f"{{{{{name}}}}}" in text, name


def test_golden_example_asset_exists_and_has_seven_beats():
    path = PROMPTS_DIR / "examples" / "saijo_e02.md"
    assert path.exists(), "黄金样本 few-shot 资产缺失，见 Task 13 的人工资产门禁说明"
    text = path.read_text(encoding="utf-8")
    for label in (
        "Hook 开场",
        "阶段一：入职即地狱",
        "阶段二：假平民认证局",
        "阶段三：钱包、女厕与金发死对头",
        "阶段四：投喂式遛大小姐",
        "阶段五：泡澡、监视与真心话",
        "收尾：修罗场引爆",
    ):
        assert label in text, label


def test_golden_example_asset_has_silent_highlight_markers():
    text = (PROMPTS_DIR / "examples" / "saijo_e02.md").read_text(encoding="utf-8")
    assert text.count("★") >= 6


def test_golden_example_asset_has_holds():
    text = (PROMPTS_DIR / "examples" / "saijo_e02.md").read_text(encoding="utf-8")
    assert text.count("留白") >= 6


def test_golden_example_asset_has_no_ocr_garbage():
    text = (PROMPTS_DIR / "examples" / "saijo_e02.md").read_text(encoding="utf-8")
    assert "80-08" not in text
    assert "浙谷339" not in text
