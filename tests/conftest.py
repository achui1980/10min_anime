import re
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# `-m` 表达式里的标识符。按**词**切而不是子串匹配 —— 见 selects_render_marker。
_MARKER_TOKEN = re.compile(r"\w+")


def selects_render_marker(expression: str | None) -> bool:
    """`-m` 表达式是否**正向**点名了 render 这个 marker。

    原判据是子串匹配（`"render" in expression`）。`-m "not render"` 在它下面当前是安全
    的（pytest 自己的 deselect 先生效，那些用例压根不进 items），所以那不是个现存 bug；
    但它对「以后加一个名字含 render 的 marker」零容错 —— `-m render_e2e` 会让子串命中、
    门打开，于是默认不该跑的 render 用例（真调 Edge-TTS、真编一段视频）被放进来。

    收紧成「按标识符切词，且不能是紧跟在 `not` 后面的那个」。刻意**不**实现完整的布尔
    表达式求值：这个门只需要回答「用户有没有明确点名 render」，而方向一律取保守
    （拿不准就跳过）—— 它保护的是「别在一次普通 `pytest` 里意外联网/编码」，误跳的代价
    是一句「跑法：uv run pytest -m render」，误跑的代价是网络一抖整个测试套变红。

    判据与用例在 tests/test_marker_gate.py。
    """
    tokens = _MARKER_TOKEN.findall(expression or "")
    return any(
        token == "render" and (index == 0 or tokens[index - 1] != "not")
        for index, token in enumerate(tokens)
    )


def pytest_collection_modifyitems(config, items):
    """没有显式 `-m render` 时跳过 render 标记的用例。

    pyproject 里 render 这个 marker 写的是「默认跳过（跑法：uv run pytest -m render）」，
    但实际上没有任何东西在执行这条约定 —— 它一直是靠 test_render_e2e.py 的 fixture
    自己 skip 掉的（它写死了一个压根没配源片的项目，于是永远 skip）。那个 fixture 修好
    之后，默认的 `uv run pytest` 就会真的去调一次 Edge-TTS（要联网）并真编一段 720p
    视频：多花几秒是小事，网络一抖整个测试套变红才是问题。

    刻意只管 render，不管 llm / generalize：
    - llm 的门是「有没有 API key」，缺 key 时自己 skip，比按 marker 摘更准；
    - generalize 的约定是「把 SRT 放进 tests/fixtures/generalize/ 后**自动生效**」，
      按 marker 摘掉会把这条约定打死。
    """
    if selects_render_marker(config.getoption("-m")):
        return
    skip = pytest.mark.skip(reason="需要真实素材与 ffmpeg，跑法：uv run pytest -m render")
    for item in items:
        if "render" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def golden_srt_path() -> Path:
    path = FIXTURES / "saijo_e02.srt"
    if not path.exists():
        pytest.skip("缺少 tests/fixtures/saijo_e02.srt")
    return path


@pytest.fixture(scope="session")
def golden_track(golden_srt_path):
    pytest.importorskip("tenmin.ingest.normalize")
    from tenmin.ingest.normalize import build_track

    return build_track(golden_srt_path, episode=2, show_title="才女的侍从")


@pytest.fixture(scope="session")
def lines_for_idx():
    """按原始 SRT 序号取行。一个序号可能对应多行（双轨拆行），也可能被并进别人的 merged_from。"""

    def _lookup(track, idx: int):
        return [ln for ln in track.lines if ln.idx == idx or idx in ln.merged_from]

    return _lookup
