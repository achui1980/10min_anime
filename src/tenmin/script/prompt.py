"""提示词加载与渲染。占位符语法 {{name}}，不与 JSON 花括号冲突。"""

from __future__ import annotations

import re
from functools import cache
from importlib.resources import files
from pathlib import Path

# 走 importlib.resources 而不是 `Path(__file__).parent`：后者假定包一定是解包在文件
# 系统上的。当前 hatchling wheel 确实如此，所以这不是修 bug，是把「资源怎么定位」这件事
# 交给标准库那条唯一正确的路——tenmin.script 是有 __init__.py 的常规包，files() 对它
# 返回的就是一个 pathlib.Path，下游 `PROMPTS_DIR / "examples" / ...` 的用法一字不改。
PROMPTS_DIR = Path(str(files("tenmin.script") / "prompts"))

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class PromptTemplateError(KeyError, ValueError):
    """模板与调用点对不上：模板要的变量没传，或者传了模板里根本不存在的变量。

    同时继承 KeyError 与 ValueError 是刻意的：
    - KeyError 保住「缺变量抛 KeyError」这条历史行为（也是最贴切的语义）。
    - ValueError 让它落进 cli.PIPELINE_ERRORS —— KeyError **不在**那个元组里，所以
      一个模板打字错原来是一路 traceback 糊到用户脸上，而不是一行红字。

    KeyError.__str__ 会把消息 repr 一遍（多行消息整个变成 \\n 转义串），所以覆盖掉。
    """

    def __str__(self) -> str:
        return str(self.args[0])


def render_prompt(template: str, template_name: str = "<inline>", /, **variables: object) -> str:
    """把 {{name}} 替换成 variables[name]。

    **双向校验**，两个方向都是硬报错：
    - 模板要的变量没传 → 原来就报（只是消息里光有个变量名）。
    - 传了但模板里没有对应占位符 → 原来**静默忽略**。md 里把 `{{dialogue_block}}` 敲成
      `{{dialog_block}}`，模型就会收到一份没有字幕轨的 prompt 照样生成（凭空编时间戳），
      要到 validate 层才可能发现。这种情况没有任何合法用途，所以是错，不是 warning。

    `template_name` 只进报错消息，是**位置参数**（`/` 之后才是 **variables）：写成关键字
    参数的话，一个恰好叫 template_name 的占位符就会把它顶掉。
    """
    used: set[str] = set()

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise PromptTemplateError(
                f"提示词模板 {template_name} 要的变量 {{{{{name}}}}} 没传。"
                f"调用点传了：{', '.join(sorted(variables)) or '（什么都没传）'}"
            )
        used.add(name)
        return str(variables[name])

    rendered = _PLACEHOLDER.sub(_sub, template)

    unused = sorted(set(variables) - used)
    if unused:
        placeholders = sorted({f"{{{{{name}}}}}" for name in _PLACEHOLDER.findall(template)})
        raise PromptTemplateError(
            f"提示词模板 {template_name} 里没有用到这些变量：{', '.join(unused)}。"
            f"模板里的占位符只有：{', '.join(placeholders) or '（一个都没有）'}。"
            f"通常是模板里的占位符名字敲错了 —— 那会让整段内容悄悄丢失。"
        )
    return rendered


@cache
def load_prompt(name: str) -> str:
    """读一份提示词资产。**带缓存**：范例是 3KB 的常驻资产，每拼一次 prompt 读一遍盘
    没有意义（一集要拼 1–3 次）。

    缓存是 `functools.cache`，所以要改模板内容的测试可以 `load_prompt.cache_clear()`。
    """
    path = PROMPTS_DIR / name
    # 全是内部调用，风险低；但 `name` 直接拼进路径这件事本身值一行校验 —— 静默读到
    # 包外的文件比响亮报错糟得多。
    if not path.resolve().is_relative_to(PROMPTS_DIR.resolve()):
        raise ValueError(f"提示词名 {name!r} 跑到 prompts 目录外面去了：{path}")
    if not path.is_file():
        raise FileNotFoundError(f"提示词文件不存在：{path}")
    return path.read_text(encoding="utf-8")
