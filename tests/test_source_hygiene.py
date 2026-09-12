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
TEXT_IO_METHODS = frozenset({"read_text", "write_text"})


def _source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py"))


def _offenders(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if name is None:
            continue
        if name in TEXT_IO_METHODS or name == "open":
            keywords = {kw.arg for kw in node.keywords}
            if "encoding" not in keywords:
                out.append(f"{path.name}:{node.lineno} {name}()")
    return out


def test_src_has_python_files():
    """守住这个测试自己：SRC 路径写错时上面那条会假绿。"""
    assert len(_source_files()) > 5


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_text_io_always_declares_encoding(path: Path):
    assert _offenders(path) == []
