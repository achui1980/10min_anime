"""conftest 的 render marker 门。

`pyproject` 里 render 这个 marker 写的是「默认跳过（跑法：uv run pytest -m render）」，
执行这条约定的是 `conftest.pytest_collection_modifyitems`。它原来的判据是**子串**匹配
（`if "render" in (config.getoption("-m") or "")`）：

- `-m "not render"` 当前**是安全的**（pytest 自己的 deselect 先生效，那些用例压根不
  进 items），所以它不是个现存 bug；
- 但对「以后加一个名字含 render 的 marker」零容错：`-m render_e2e` 会让子串命中、
  门打开，于是默认不该跑的 render 用例（真调 Edge-TTS、真编一段视频）被放进来。

收紧之后的判据是「按标识符切词 + 排除被 `not` 否否的那个」。方向刻意保守：拿不准就跳过
—— 这个门保护的是「别在一次普通 `pytest` 里意外联网/编码」。
"""

from __future__ import annotations

import pytest

from .conftest import selects_render_marker


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # 正向要 render
        ("render", True),
        (" render ", True),
        ("render and not slow", True),
        ("slow or render", True),
        ("not slow and render", True),
        # 没要
        ("", False),
        (None, False),
        ("not render", False),
        ("not  render", False),
        ("slow", False),
        # **子串匹配会判错、按词切不会**的那一组：这才是收紧的理由
        ("render_e2e", False),
        ("renderx", False),
        ("not render_e2e", False),
        ("prerender", False),
    ],
)
def test_selects_render_marker(expression, expected):
    assert selects_render_marker(expression) is expected


def test_the_gate_is_wired_to_the_collection_hook():
    """守住这个测试自己：判据函数必须真的是 hook 在用的那一个。"""
    import inspect

    from . import conftest

    source = inspect.getsource(conftest.pytest_collection_modifyitems)
    assert "selects_render_marker" in source
