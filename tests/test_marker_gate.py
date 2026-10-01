"""conftest 的默认跳过 marker 门（render 与 asr）。

`pyproject` 里这两个 marker 写的都是「默认跳过（跑法：uv run pytest -m xxx）」，
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

from .conftest import selects_marker


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
    assert selects_marker(expression, "render") is expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # 正向要 asr
        ("asr", True),
        ("asr and not slow", True),
        # 没要
        ("not asr", False),
        ("", False),
        (None, False),
        # 两个门互不串：点名 render 不该把 asr 的门也打开
        ("render", False),
        # 子串组：同样是「子串匹配会判错、按词切不会」，跟 render 表末尾那一组同理
        ("asrx", False),
        ("not asr_e2e", False),
    ],
)
def test_selects_asr_marker(expression, expected):
    assert selects_marker(expression, "asr") is expected


def test_the_gate_is_wired_to_the_collection_hook():
    """守住这个测试自己：判据函数必须真的是 hook 在用的那一个。"""
    import inspect

    from . import conftest

    source = inspect.getsource(conftest.pytest_collection_modifyitems)
    assert "selects_marker" in source


# --- hook 本体：两条门互不串 -------------------------------------------------
#
# 下面两个替身存在的唯一理由是「这个不变量只有走 hook 本体才证得到」。
# pytest 真正的 Config 与 Item 在测试里造不起来（Config 要一整轮 argparse + ini 解析 +
# 插件注册，Item 要一个 collector 树和一个真实的测试函数），而 hook 只从它们身上各取
# 一样东西：Config 的 `getoption("-m")`、Item 的 `keywords` 与 `add_marker`。
# 这三样用替身顶掉，剩下的就是 hook 自己的形状。
#
# 为什么不能只测 selects_marker：泛化真正改的是 hook 的形状 —— 从一个 `return` 变成
# per-marker 的 `continue`。把它写回「`if any(selects_marker(expression, m) for m in
# _GATED_MARKERS): return`」，selects_marker 的两张用例表照样全绿，而 `-m render` 会把
# asr 的门一起打开。上面那个 grep 源码的测试也拦不住（`selects_marker` 字样仍在）。
# 实测过：换成那个形态，下面这个测试会在 `render` 与 `asr` 两条参数上变红。


class _FakeConfig:
    """只提供 hook 要的那一个方法：`getoption("-m")`。"""

    def __init__(self, expression: str | None) -> None:
        self._expression = expression

    def getoption(self, name: str) -> str | None:
        # hook 只该问 `-m`。问别的说明 hook 变了形状，这个替身就不再是等价的了。
        assert name == "-m"
        return self._expression


class _FakeItem:
    """只提供 hook 要的那两样：`keywords` 与 `add_marker`。"""

    def __init__(self, *keywords: str) -> None:
        self.keywords = set(keywords)
        self.marks: list[object] = []

    def add_marker(self, mark: object) -> None:
        self.marks.append(mark)


@pytest.mark.parametrize(
    ("expression", "render_skipped", "asr_skipped"),
    [
        (None, True, True),
        # 点名一个门不该把另一个门也打开 —— 这三条是 selects_marker 的单测证不到的，
        # 它们守的是 hook 里那个 per-marker continue
        ("render", False, True),
        ("asr", True, False),
        ("render or asr", False, False),
    ],
)
def test_the_gate_opens_one_marker_at_a_time(expression, render_skipped, asr_skipped):
    from . import conftest

    render_item, asr_item = _FakeItem("render"), _FakeItem("asr")
    conftest.pytest_collection_modifyitems(_FakeConfig(expression), [render_item, asr_item])
    assert bool(render_item.marks) is render_skipped
    assert bool(asr_item.marks) is asr_skipped


def test_the_ocr_marker_is_gated_until_named():
    """ocr 用例要对一整集真实片源逐帧跑 Vision，一次普通 pytest 里不许意外跑到。"""
    from . import conftest

    default = _FakeItem("ocr")
    conftest.pytest_collection_modifyitems(_FakeConfig(None), [default])
    assert default.marks

    named = _FakeItem("ocr")
    conftest.pytest_collection_modifyitems(_FakeConfig("ocr"), [named])
    assert named.marks == []
