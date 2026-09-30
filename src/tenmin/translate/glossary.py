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

    键那一半的判据（`isinstance(key, str)`）跟值那一半**不对称**：走 `load_glossary`
    那条路时它恒为真（JSON 对象的键必然是 str），只有 `merge_glossary` /
    `effective_glossary` 的调用方 —— 它们收的是人写的 Python dict —— 才可能真的踩到它。
    所以别把它读成给 load 路径准备的守卫。实测：单独删掉这半个判据，全量测试仍然全绿，
    也就是说没有任何测试喂过非 str 的键。
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


_TRAILING_PARTICLES = frozenset("哦呀啊呢吧啦嘛喀哟欸唉")
"""句尾语气助词/叹词集合。这是"数据坏了"一类的合法性边界，不是创作旋钮，所以不进
`RenderConfig`/`ValidateConfig`。

只在 `merge_glossary` 回写累积表时对 `fresh` 参数生效，不影响 `_clean`/`load_glossary`/
`effective_glossary` 的既有行为——那三个入口共享 `_clean`，动 `_clean` 本身会连累历史
已存条目的解释方式，范围过大。
"""


def _trim_trailing_particle(translation: str) -> tuple[str, bool]:
    """剪掉译名结尾的一个语气助词字符（只剪一次，不循环）。

    返回 `(剪裁后的字符串, 是否发生了剪裁)`。调用方据此决定要不要发 warning。
    """
    if translation and translation[-1] in _TRAILING_PARTICLES:
        return translation[:-1], True
    return translation, False


def load_glossary(path: Path) -> dict[str, str]:
    """读累积表。文件不存在或坏了都返回空表。

    两种情况都刻意不报错：第一集跑之前这个文件本来就不存在；而它坏掉的最坏后果只是
    「这一集的译名不跟前几集对齐」，不值得让整条管线停下来。

    `UnicodeDecodeError` 也得捕：它是 `ValueError` 的子类而不是 `OSError` 或
    `JSONDecodeError` 的，漏掉它的话一份被非 UTF-8 编辑器存回去的表会把异常漏出去、
    停掉整条管线 —— 正是这个函数发誓不做的事。

    第一句 `is_file` 检查只对**缺文件与目录**这两种情况是冗余的（实测：缺文件 →
    `FileNotFoundError`、路径是目录 → `IsADirectoryError`，两个都是 `OSError` 子类，
    下面那组捕获照样兜住并返回空表）。所以删掉它不会让任何测试变红 —— 但别据此当成
    「没测到、可以删」：对 FIFO / 字符设备它**不冗余**。那种路径 `is_file()` 为 False 而
    `exists()` 为 True（实测 `os.mkfifo` 建出来的就是这个组合），少了这一句
    `read_text()` 会在 open 上**永久阻塞**、根本不抛异常，下面那组 except 兜不住。
    顺带它也把「文件不存在不是错误」这条意图写在了显眼处，而不是藏在一个 except 子句里。
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
    """写累积表。**内容与盘上现有内容逐字节相同时直接返回，不碰文件。**

    键排序 + 不转义非 ASCII：这是个人会去手动纠错的文件，diff 得可读。

    不用自己 mkdir：`atomic.write_text` 会代建父目录（实测），而这个表住在 `zh/` 底下，
    第一次存盘时那个目录还不存在。

    「没变就不写」不是性能优化，是正确性：这个文件是 script 阶段的新鲜度输入
    （见 pipeline._script_inputs），而每一集的 translate 都会把整张表回写一遍。无条件
    写的话，一次 E01…E10 的批处理跑完，最后几集的 translate 会把前面几集**已经写好的**
    解说稿全部判旧，下一次运行白付好几次 LLM 费（一次调用实测可达 561 秒）。真有新词时
    照写、照刷 mtime —— 那种重跑是该付的（prompt 里的术语表确实变了）。

    比的是**自己将要写下去的那份 payload**，不是入参：入参经过排序与 JSON 序列化才成为
    磁盘内容，拿字典去跟文件比根本对不上。读盘失败（文件不存在、坏了、不是 UTF-8）一律
    退回「照写」—— 判不出「没变」就不许跳过。`is_file` 那道守卫跟 `load_glossary` 里那句
    同源：FIFO 之类的路径上 `read_text` 会在 open 上永久阻塞，而 except 兜不住阻塞。

    **刻意不再过一遍 `_clean`** —— 所以模块 docstring 里说的那三个入口不含 save。理由是
    传进来的表本身来自 `merge_glossary`，已经洗过了。这件事没有任何测试钉住：将来要是有
    人把生数据直接喂给 save，「洗不洗」得重新想一遍，别默认它已经洗过。
    """
    payload = (
        json.dumps(dict(sorted(glossary.items())), ensure_ascii=False, indent=2) + "\n"
    )
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == payload:
                return
        except (UnicodeDecodeError, OSError):
            pass
    atomic.write_text(path, payload)


def merge_glossary(
    accumulated: Mapping[str, str],
    fresh: Mapping[str, str],
    *,
    warnings: list[str] | None = None,
) -> dict[str, str]:
    """把这一集新认出来的词并进累积表。

    已经定下的译名**不许**被后面某一集改掉 —— 那正是累积要防的事（第 5 集把人名换个
    写法，成片看起来就像换了个角色）。所以冲突时保留累积的那个。

    两边都过 `_clean`：只洗新词的话，一条坏掉的累积条目会永远占着那个键、把后面每一集
    给出的好译名都挡在外面。

    对 `fresh` 里通过 `_clean` 的每一条译名，先剪一次结尾语气助词（ASR 听写误差经翻译
    放大后常见的垃词，例如"主ガビオ":"主嘉碑哦"）；剪完长度 ≤1 就整条丢弃，不并入
    返回的表。`accumulated` 侧不重新触发这条规则——历史累积表已经清洗过，不该被反复剪。
    可选的 `warnings` 出参收集每次剪裁/丢弃的中文提示，`None` 时（默认）不收集，现有
    调用点无需改动。

    返回新字典，不改入参。
    """
    merged = _clean(accumulated)
    for term, translation in _clean(fresh).items():
        trimmed, did_trim = _trim_trailing_particle(translation)
        if did_trim:
            if len(trimmed) <= 1:
                if warnings is not None:
                    warnings.append(
                        f"术语表：{term!r} 的译名 {translation!r} 疑似语气词或过短，"
                        "已丢弃"
                    )
                continue
            if warnings is not None:
                warnings.append(
                    f"术语表：{term!r} 的译名 {translation!r} 结尾疑似语气词，"
                    f"已修正为 {trimmed!r}"
                )
            translation = trimmed
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
