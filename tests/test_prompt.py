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


def test_render_rejects_unused_kwargs():
    """原来这里锁的是反的行为（「多传的变量被忽略」）。那正是 P2-C 第 4 项要修的 bug：
    模板里的占位符名字敲错时，整段内容会静默丢失。见下面 P2-C-4 那一组测试。"""
    from tenmin.script.prompt import PromptTemplateError

    with pytest.raises(PromptTemplateError):
        render_prompt("固定文本", unused="x")


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


# --- P2-C-4：模板校验必须是双向的 ---


def test_render_rejects_a_variable_the_template_never_uses():
    """md 里把 {{dialogue_block}} 写错成 {{dialog_block}} 时，传进来的 dialogue_block
    原来被**静默忽略**：模型收到一份没有字幕轨的 prompt，照样凭空编时间戳生成，要到
    validate 层才可能发现。打字错没有任何合法用途，所以报错而不是 warning。"""
    from tenmin.script.prompt import PromptTemplateError

    with pytest.raises(PromptTemplateError) as exc:
        render_prompt("只用了 {{dialog_block}}", "fake.md", dialogue_block="正文", dialog_block="x")
    assert "dialogue_block" in str(exc.value)
    assert "fake.md" in str(exc.value)


def test_render_unused_variable_error_lists_the_placeholders_it_did_find():
    from tenmin.script.prompt import PromptTemplateError

    with pytest.raises(PromptTemplateError) as exc:
        render_prompt("{{a}}", "fake.md", a="1", b="2")
    assert "{{a}}" in str(exc.value)


def test_render_missing_variable_error_names_the_template_and_the_candidates():
    """原来只 `raise KeyError(name)`：光一个变量名，既不知道是哪份模板，也不知道
    调用点到底传了什么。"""
    with pytest.raises(KeyError) as exc:
        render_prompt("你好 {{name}}", "greeting.md", nome="x")
    message = str(exc.value)
    assert "name" in message
    assert "greeting.md" in message
    assert "nome" in message


def test_template_errors_land_in_the_cli_error_funnel():
    """模板与调用点对不上是打字错，但用户该看到一行红字而不是 traceback。
    KeyError 不在 cli.PIPELINE_ERRORS 里，ValueError 在。"""
    from tenmin.script.prompt import PromptTemplateError

    assert issubclass(PromptTemplateError, ValueError)
    assert issubclass(PromptTemplateError, KeyError)


# --- P2-C-5：资源加载 ---


def test_load_prompt_is_cached():
    """范例是 3KB 的常驻资产，每次拼 prompt 都读一遍盘没有意义。"""
    load_prompt.cache_clear()
    first = load_prompt("single_episode.md")
    second = load_prompt("single_episode.md")
    assert first is second
    assert load_prompt.cache_info().hits >= 1


def test_load_prompt_cache_can_be_cleared():
    """加了缓存就必须留一个清缓存的口子，否则想改模板内容的测试没法写。"""
    load_prompt("single_episode.md")
    load_prompt.cache_clear()
    assert load_prompt.cache_info().currsize == 0


def test_load_prompt_refuses_to_escape_the_prompts_dir():
    with pytest.raises(ValueError, match="prompts"):
        load_prompt("../single.py")
