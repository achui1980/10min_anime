"""跨模块的源码不变量。用 AST 扫 src/tenmin/，不是文本 grep。

这里只放「一处漏了就会在别人的机器上炸、而本机测试永远绿」的规则。
目前只有一条：文本 I/O 必须显式写 encoding。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "tenmin"

# 这些调用不写 encoding= 时，Python 3.12 会退化到 locale.getpreferredencoding()。
# 本项目的产物（对照表、旁白、dialogue.json、project.yaml）全是中文，在非 UTF-8
# locale（Windows 简中默认 cp936、部分 CI 容器是 POSIX/ascii）下读写就直接
# UnicodeDecodeError/UnicodeEncodeError。本机是 UTF-8，所以这类漏写永远测不出来。
#
# 内建 `open` 也被守（见 _called_name）：它是这条规则最容易漏的形态，而 `open(p, "w")`
# 与 `p.open("w")` 的失败模式一模一样。
TEXT_IO_METHODS = frozenset({"read_text", "write_text"})

# 唯一豁免的接收者：`tenmin.atomic`。`atomic.write_text` 不是 stdlib 的那个 ——
# 它自己的签名把 encoding 钉成了 "utf-8"（下面 test_atomic_write_text_pins_utf8
# 就是守这一条的），所以调用点不写 encoding= 也不可能退化到 locale。
# 刻意做成「白名单 + 一条验证白名单前提的测试」而不是直接放宽规则：规则的本意是
# 「绝不让 locale 决定编码」，这个豁免不违反本意，而且前提是被断言住的。
# atomic.py 自己照旧被扫（它内部那句 tmp.write_text 是显式传 encoding 的）。
UTF8_PINNED_MODULES = frozenset({"atomic"})


def _receiver_name(func: ast.Attribute) -> str | None:
    return func.value.id if isinstance(func.value, ast.Name) else None


def _source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py"))


def _called_name(func: ast.expr) -> str | None:
    """被调用者的「方法/函数名」。

    两种形态都要认，缺一个这条守卫就有洞：
    - `ast.Attribute`（`path.read_text(...)` / `path.open(...)`）→ `.attr`
    - `ast.Name`（**内建** `open(...)`）→ `.id`

    原实现只认前者，于是 `name == "open"` 那一支对裸 `open(p, "w")` 永不生效 ——
    而模块 docstring 与 TEXT_IO_METHODS 旁边的注释都把 `open` 列为被守对象。
    """
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _offenders(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = _called_name(func)
        if name is None:
            continue
        if name in TEXT_IO_METHODS or name == "open":
            # 豁免只对「方法调用」有意义（`atomic.write_text(...)`）；裸 `open()` 没有
            # 接收者，_receiver_name 只接受 ast.Attribute，所以这里先判形态。
            if isinstance(func, ast.Attribute) and _receiver_name(func) in (
                UTF8_PINNED_MODULES
            ):
                continue
            keywords = {kw.arg for kw in node.keywords}
            if "encoding" not in keywords:
                out.append(f"{path.name}:{node.lineno} {name}()")
    return out


def test_atomic_write_text_pins_utf8():
    """守住上面那条豁免的前提：atomic.write_text 的 encoding 必须默认 UTF-8。

    这个默认值一旦被改成 None / locale，_offenders 的豁免就会变成一个静默的漏洞。
    """
    import inspect

    from tenmin import atomic

    default = inspect.signature(atomic.write_text).parameters["encoding"].default
    assert default == "utf-8"


# --- 守卫本身有洞：裸 open() 那一支（M4）------------------------------------
#
# `_offenders` 原来是 `name = func.attr if isinstance(func, ast.Attribute) else None`
# 紧跟一句 `if name is None: continue`，所以 `name == "open"` 那一支**只对
# `path.open(...)` 生效，对内建 `open()` 永不生效** —— 裸 `open(p, "w")` 的
# `node.func` 是 `ast.Name`。而模块 docstring 与 TEXT_IO_METHODS 旁边的注释都把
# `open` 列为被守对象。目前 src/ 里没有裸 open，所以这是潜在漏洞而不是现存违规。


def _offenders_of(source: str, tmp_path: Path) -> list[str]:
    path = tmp_path / "synthetic.py"
    path.write_text(source, encoding="utf-8")
    return _offenders(path)


def test_audit_catches_a_bare_open_without_encoding(tmp_path):
    assert _offenders_of("def f(p):\n    return open(p, 'w')\n", tmp_path) == [
        "synthetic.py:2 open()"
    ]


def test_audit_accepts_a_bare_open_that_declares_encoding(tmp_path):
    assert _offenders_of(
        "def f(p):\n    return open(p, 'w', encoding='utf-8')\n", tmp_path
    ) == []


def test_audit_ignores_a_bare_call_that_is_not_open(tmp_path):
    """只有 `open` 这个名字算，别把 `read_text(x)` 这种同名参数的调用也算进去。"""
    assert _offenders_of("def f(p):\n    return dict(p)\n", tmp_path) == []


def test_audit_still_catches_the_attribute_form(tmp_path):
    """`path.open(...)` 那一支不能被回归。"""
    assert _offenders_of("def f(p):\n    return p.open('w')\n", tmp_path) == [
        "synthetic.py:2 open()"
    ]


def test_src_has_python_files():
    """守住这个测试自己：SRC 路径写错时上面那条会假绿。"""
    assert len(_source_files()) > 5


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_text_io_always_declares_encoding(path: Path):
    assert _offenders(path) == []


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_assert_statements_in_src(path: Path):
    """src/ 里不许有 assert。

    `python -O` 会把 assert 整句剥掉，于是被它当作控制流/前置条件用的地方在生产
    模式下静默失效，错误漂到很远的地方才以别的异常炸出来。本仓库已经踩过两次
    （srt_parser 的 `assert match is not None` 退化成 AttributeError、
    run_pipeline 的 `assert tts_engine is not None` 退化成 None 漂进 TTS）。
    前置条件请写成显式的 `if ...: raise`，并选一个 cli.py 会捕获的异常类型。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert lines == [], f"{path.name} 第 {lines} 行有 assert"
