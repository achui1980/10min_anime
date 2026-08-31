"""提示词加载与渲染。占位符语法 {{name}}，不与 JSON 花括号冲突。"""

from __future__ import annotations

import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render_prompt(template: str, **variables: object) -> str:
    """把 {{name}} 替换成 variables[name]。缺变量直接 KeyError，不静默留坑。"""

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise KeyError(name)
        return str(variables[name])

    return _PLACEHOLDER.sub(_sub, template)


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"提示词文件不存在：{path}")
    return path.read_text(encoding="utf-8")
