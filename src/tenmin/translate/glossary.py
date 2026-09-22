"""跨集累积的专有名词表。

存在的理由是译名漂移有两根轴：同一集里「字幕写莉迪亚、解说稿写莉蒂亚」（翻译与解说
是两次模型调用），以及集与集之间（每集的翻译调用互不知情，后者更难发现 —— 单看一集
完全自洽）。一份项目级的表把两根轴一起按住，而且随集数收敛。

手写的那份（project.yaml 里的 glossary）是纠错入口：机器译错了人要能盖掉它。

**全模块只有一个「条目算不算数」的判据**，就是 `_clean`：键与值都得是去掉首尾空白后
非空的字符串。读盘、合并、喂模型三个入口都过它，所以「存盘再读回来」是 `merge_glossary`
的不动点（`test_a_saved_table_is_a_fixed_point_of_merge` 钉住了这一条）—— 下一集的累积
表是「读回上一集的表 + 本集新词」，这个不动点性质就是译名不在集与集之间漂的前提。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from tenmin import atomic


def _clean(entries: Mapping[str, str]) -> dict[str, str]:
    """筛出可用条目。键与值都 strip 过，两边任一为空就整条丢掉。

    非字符串的值刻意**丢掉而不是 `str()` 强转**：这个文件是人会手改的，一条改成
    `null` 或数字的条目强转出来是译名 `"None"` / `"3"`，会原样喂进 prompt 当成「已经
    定下的译法」，比少一条难查得多。声明的入参类型本来就是 `Mapping[str, str]`，强转
    是在回答一个没人问的问题。
    """
    cleaned: dict[str, str] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        term = key.strip()
        translation = value.strip()
        if term and translation:
            cleaned[term] = translation
    return cleaned


def load_glossary(path: Path) -> dict[str, str]:
    """读累积表。文件不存在或坏了都返回空表。

    两种情况都刻意不报错：第一集跑之前这个文件本来就不存在；而它坏掉的最坏后果只是
    「这一集的译名不跟前几集对齐」，不值得让整条管线停下来。

    `UnicodeDecodeError` 也得捕：它是 `ValueError` 的子类而不是 `OSError` 或
    `JSONDecodeError` 的，漏掉它的话一份被非 UTF-8 编辑器存回去的表会把异常漏出去、
    停掉整条管线 —— 正是这个函数发誓不做的事。

    第一句 `is_file` 检查在行为上是**冗余**的（实测：缺文件 → `FileNotFoundError`、
    路径是目录 → `IsADirectoryError`，两个都是 `OSError` 子类，下面那组捕获照样兜住
    并返回空表）。留着它纯粹是为了把「文件不存在不是错误」这条意图写在显眼处，而不是
    藏在一个 except 子句里；所以删掉它不会让任何测试变红，别把那当成「这里没被测到」。
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        # 合法 JSON 但不是对象（`["リディア"]`）。少了这句会在 `.items()` 上炸
        # AttributeError，而那不在上面那组捕获里。
        return {}
    return _clean(data)


def save_glossary(path: Path, glossary: Mapping[str, str]) -> None:
    """写累积表。

    键排序 + 不转义非 ASCII：这是个人会去手动纠错的文件，diff 得可读。

    不用自己 mkdir：`atomic.write_text` 会代建父目录（实测），而这个表住在 `zh/` 底下，
    第一次存盘时那个目录还不存在。
    """
    payload = json.dumps(dict(sorted(glossary.items())), ensure_ascii=False, indent=2)
    atomic.write_text(path, payload + "\n")


def merge_glossary(
    accumulated: Mapping[str, str], fresh: Mapping[str, str]
) -> dict[str, str]:
    """把这一集新认出来的词并进累积表。

    已经定下的译名**不许**被后面某一集改掉 —— 那正是累积要防的事（第 5 集把人名换个
    写法，成片看起来就像换了个角色）。所以冲突时保留累积的那个。

    两边都过 `_clean`：只洗新词的话，一条坏掉的累积条目会永远占着那个键、把后面每一集
    给出的好译名都挡在外面。

    返回新字典，不改入参。
    """
    merged = _clean(accumulated)
    for term, translation in _clean(fresh).items():
        merged.setdefault(term, translation)
    return merged


def effective_glossary(
    accumulated: Mapping[str, str], manual: Mapping[str, str]
) -> dict[str, str]:
    """喂给模型的那份表：累积的叠上手写的，手写的赢。

    手写表是纠错入口，它必须能盖掉机器的选择，否则「改了 project.yaml 却不生效」会是
    个很难查的问题。
    """
    return {**_clean(accumulated), **_clean(manual)}
