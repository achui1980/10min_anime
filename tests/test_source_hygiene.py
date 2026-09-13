"""跨模块的源码不变量。用 AST 扫 src/tenmin/，不是文本 grep。

这里只放「一处漏了就会在别人的机器上炸、而本机测试永远绿」的规则。目前三条：

1. 文本 I/O 必须显式写 encoding。
2. src/ 里不许有 assert（`python -O` 会把它整句剥掉）。
3. 产物写入必须走 `tenmin.atomic`，不许直接 `Path.write_text` / `write_bytes` /
   `shutil.copyfile`。
"""

from __future__ import annotations

import ast
import re
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


# --- 产物写入必须走 tenmin.atomic（M5）--------------------------------------
#
# `pipeline._is_fresh` 只比 mtime。被 Ctrl-C 或 ffmpeg 中途失败留下的半截产物 mtime
# 恰好最新，于是下一次运行把它判成「已是最新」整段跳过，一个截断的 .m4a/.mp4/.json
# 就这样一路进成片、全程零警告（见 tenmin/atomic.py 的模块 docstring）。
#
# 「全部产物写入走原子写」这条不变量在 M5 之前**没有任何测试或审计守着**，而
# `cli.init` 就是它唯一的缺口。

# 直接调用等于绕过原子写的那些 API。
_NON_ATOMIC_WRITES = frozenset({"write_text", "write_bytes", "copyfile"})

# 唯一豁免的文件：`atomic.py` 自己 —— 它就是那一层实现，`tmp.write_text` /
# `shutil.copyfile(src, tmp)` 写的都是 `.part` 临时文件，正式路径由 `os.replace` 落。
_ATOMIC_EXEMPT_FILES = frozenset({"atomic.py"})

# 豁免的**接收者**：`atomic.write_text(...)` 形态上也是 `X.write_text(...)`，
# 但它就是原子版本本身。
_ATOMIC_RECEIVERS = frozenset({"atomic"})

# 「已知非产物」白名单：{(文件名, 方法名): 理由}。
#
# **目前是空的。** 往里加之前先问一句：这个文件真的不是任何阶段的输入吗？
# `project.yaml` 就是个反例 —— 它看着像「配置」，实际是**每个阶段**的隐式输入
# （`_is_fresh` 把它加进 inputs），所以 register_episode 与 cli.init 都必须原子写。
_NON_ARTIFACT_WRITES: dict[tuple[str, str], str] = {}


def _non_atomic_writes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        if name not in _NON_ATOMIC_WRITES:
            continue
        if _receiver_name(node.func) in _ATOMIC_RECEIVERS:
            continue
        if (path.name, name) in _NON_ARTIFACT_WRITES:
            continue
        out.append(f"{path.name}:{node.lineno} {name}()")
    return out


@pytest.mark.parametrize(
    "path",
    [p for p in _source_files() if p.name not in _ATOMIC_EXEMPT_FILES],
    ids=lambda p: p.name,
)
def test_artifact_writes_go_through_atomic(path: Path):
    assert _non_atomic_writes(path) == [], (
        f"{path.name} 直接写盘了。产物写入必须走 tenmin.atomic"
        "（write_text / copy_file / atomic_path），否则半截产物的 mtime 会让"
        "_is_fresh 把它当成品跳过。"
    )


def test_the_atomic_audit_can_actually_see_a_violation(tmp_path):
    """守住上面那条审计自己：它在 M5 之前对 `cli.init` 报的就是这个形状。"""
    path = tmp_path / "synthetic.py"
    path.write_text(
        "from pathlib import Path\n"
        "def f(p: Path):\n"
        "    p.write_text('x', encoding='utf-8')\n",
        encoding="utf-8",
    )
    assert _non_atomic_writes(path) == ["synthetic.py:3 write_text()"]


def test_the_atomic_audit_lets_the_atomic_helper_through(tmp_path):
    path = tmp_path / "synthetic.py"
    path.write_text(
        "from tenmin import atomic\ndef f(p):\n    atomic.write_text(p, 'x')\n",
        encoding="utf-8",
    )
    assert _non_atomic_writes(path) == []


# --- 注释里不许写 `模块.py:行号` 这种交叉引用（N1）--------------------------
#
# 审查时逐条核过 13 处这种引用，**正确率 0/13** —— 它们全部在后续重构里漂掉了，
# 而且漂得毫无痕迹（读注释的人会跳到一段完全无关的代码，然后怀疑自己）。行号是
# 这个仓库里最容易过期的东西：一次 63 个 commit 的优化就能把整份文件推走几十行。
#
# 正确写法是**符号名引用**：`render/timeline.py 的 chunks_by_beat`、
# `single_episode.md 的「硬性要求」一节`。它们跟着重命名一起被 grep 到，不跟着行号漂。
_LINE_REFERENCE = re.compile(r"[\w/]+\.(?:py|md):\d+")


def _internal_file_names() -> set[str]:
    """src/tenmin/ 底下所有 .py 与 .md 的**文件名**（含 prompts/*.md）。"""
    return {p.name for p in SRC.rglob("*") if p.suffix in {".py", ".md"}}


def _line_references(path: Path) -> list[str]:
    """本文件里指向**本项目自己**某个文件某一行的引用。

    白名单是结构性的、不用手维护：只有「文件名在 src/tenmin/ 底下真的存在」才算违规。
    edge-tts 那几处（`communicate.py:616`、`data_classes.py:38`）形态一样，但那两个
    文件不属于本项目 —— 引用**第三方库**的行号是合理的（它是钉在某个版本上的证据，
    而我们不重构它），所以自动放过。
    """
    internal = _internal_file_names()
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _LINE_REFERENCE.finditer(line):
            name = match.group(0).rsplit("/", 1)[-1].split(":")[0]
            if name in internal:
                out.append(f"{path.name}:{lineno} → {match.group(0)}")
    return out


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_line_number_cross_references(path: Path):
    assert _line_references(path) == [], (
        f"{path.name} 里有指向本项目文件某一行的引用。行号一定会漂 —— "
        "请改成符号名引用（`render/timeline.py 的 chunks_by_beat`）。"
    )


def test_the_line_reference_audit_lets_third_party_refs_through(tmp_path):
    """守住上面那条白名单：edge-tts 的 `communicate.py:616` 必须被放过。"""
    path = tmp_path / "synthetic.py"
    path.write_text("# 见 communicate.py:616 与 data_classes.py:38\n", encoding="utf-8")
    assert _line_references(path) == []


def test_the_line_reference_audit_catches_an_internal_ref(tmp_path):
    """而指向本项目的必须被抓到，两种写法（带目录/不带目录）都算。"""
    path = tmp_path / "synthetic.py"
    path.write_text(
        "# 见 render/timeline.py:107\n# 也见 models.py:257\n", encoding="utf-8"
    )
    assert _line_references(path) == [
        "synthetic.py:1 → render/timeline.py:107",
        "synthetic.py:2 → models.py:257",
    ]


# --- 注释里不许留内部任务代号（N7）------------------------------------------
#
# 形如「P + 一位数字 + 短横 + 一个大写字母」的代号曾经散在 60 多处注释里。它们对读代码
# 的人**毫无意义**：那些计划文档是某个时间点的快照，代号既不指向代码里的任何东西，也不
# 告诉你那件事到底做了什么，还常年过期（`_is_fresh` 的 docstring 把已经落地的产物原子写
# 写成「不在这里做，只能靠那个任务」，而它早就落地了）。
#
# 描述一个已经做完的改动，正确写法是说清**它做了什么**：写「产物原子写
# （tenmin.atomic）」，不写代号。
#
# 下面这条正则刻意不含任何字面代号，所以本文件不会自我命中。
_TASK_CODE = re.compile(r"\bP\d-[A-Z]\b|\bP\d 的\b")

# 也扫 tests/：那边原来占了三分之二。
_ALL_AUDITED_DIRS = (SRC, Path(__file__).resolve().parent)


def _audited_python_files() -> list[Path]:
    seen: dict[Path, None] = {}
    for base in _ALL_AUDITED_DIRS:
        for path in sorted(base.rglob("*.py")):
            seen.setdefault(path, None)
    return list(seen)


def _task_codes(path: Path) -> list[str]:
    out: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        out.extend(f"{path.name}:{lineno} {m.group(0)}" for m in _TASK_CODE.finditer(line))
    return out


@pytest.mark.parametrize("path", _audited_python_files(), ids=lambda p: p.name)
def test_no_internal_task_codes_in_comments(path: Path):
    assert _task_codes(path) == [], (
        f"{path.name} 里留了内部任务代号。请改成说清那件事做了什么 —— 代号对读代码的人"
        "毫无意义，而且计划文档只是某个时间点的快照。"
    )


def test_the_task_code_audit_can_see_a_violation(tmp_path):
    """守住上面那条审计自己（它的正则不含任何字面代号，所以不会自我命中）。"""
    path = tmp_path / "synthetic.py"
    code = "P" + "1-G"
    path.write_text(f"# 那是 {code} 的范围\n", encoding="utf-8")
    assert _task_codes(path) == [f"synthetic.py:1 {code}"]


def test_the_task_code_audit_covers_the_tests_directory():
    """tests/ 那边原来占了三分之二的代号，必须也在扫描范围里。"""
    names = {p.name for p in _audited_python_files()}
    assert "pipeline.py" in names
    assert "test_pipeline.py" in names
