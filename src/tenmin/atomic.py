"""产物原子落盘：先写同目录的临时文件，成功了才 `os.replace` 到正式路径。

**为什么必须有这一层**：`pipeline._is_fresh` 只比 mtime。被 Ctrl-C 或 ffmpeg 中途
失败留下的半截产物，mtime 恰好是最新的，于是下一次运行会把它判成「已是最新」整段
跳过，一个截断的 `.m4a`/`.mp4`/`.json` 就这样一路进成片，全程零警告。`_is_fresh` 里
那条「0 字节判不新鲜」只挡得住「刚 open 就被打断」，挡不住「写了一半」。
`os.replace` 是同文件系统内的原子操作，所以正式路径上永远只有完整内容。

**为什么放在包根、而不是 render/ 或 pipeline.py 里**：`pipeline` 已经
`from tenmin.render.audio import mix_audio`，所以 `render/*` 反过来 import
`pipeline` 会直接成环；而这套逻辑 `pipeline`（写 json/txt/拷 SRT/改写 yaml）和
`render/{audio,video}`（包 ffmpeg 输出）两边都要用。跟 `tenmin/intervals.py`
同样是一个「不 import 任何 tenmin 模块」的叶子层，谁都可以放心依赖。

结构刻意跟 `render/tts.py` 的 `EdgeTTSEngine.synthesize` 对齐（同样的「异常路径
unlink、成功路径 replace」），而且那边现在也直接用本模块的 `atomic_path` —— 全仓
只有一套 `.part` 命名（历史上是两套，见 `part_path`）。
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

PART_SUFFIX = ".part"


def part_path(path: Path) -> Path:
    """正式路径对应的临时文件路径。

    两条硬约束：

    1. **必须同目录**。`os.replace` 跨文件系统直接抛 `OSError`，而本项目的产物目录
       常年住在外置盘上（`/Volumes/...`），拿 `tempfile.gettempdir()` 必然踩中。
    2. **必须保留原扩展名**。ffmpeg 靠输出文件的扩展名推断容器格式，写成
       `E02.mixed.m4a.part` 会让它报 `Unable to find a suitable output format`。
       所以 `.part` 插在扩展名**之前**：`E02.mixed.m4a` → `E02.mixed.part.m4a`。

       `render/tts.py` 历史上自己拼的是另一套（`out_path.name + ".part"`，也就是
       `chunk_001.abc.mp3.part`）—— edge-tts 不推容器格式，所以那样也能用，但全仓两套
       命名而 docstring 却声称「同一个 `.part` 记号」。现在它也走本模块，只剩这一套。
       安全性已实测：`.part.mp3` **不会**被 `tts._find_cached_chunk` 的
       `chunk_*.{digest}.mp3` glob 命中（那个模式要求以 `.{digest}.mp3` 收尾）。

    刻意不掺 pid / 随机数：确定的名字才能在下一次运行时被认出来并清掉（见
    `atomic_path`）。同一集并发跑两份本来就会互相踩产物，不是这一层该解决的问题。
    """
    return path.with_name(f"{path.stem}{PART_SUFFIX}{path.suffix}")


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """交出一个临时路径；块正常结束就原子改名到 `path`，出任何事就把它删掉。

    捕获 `BaseException` 而不是 `Exception`：`KeyboardInterrupt` 正是这套机制存在的
    头号原因，而它不是 `Exception` 的子类（`SystemExit` 同理）。

    进块前先清掉可能残留的旧 `.part`：上一次运行如果是在 `replace` 之前被 `kill -9`
    掉的，那个文件还在，直接往上写会得到「新数据覆盖旧数据的前半段」这种更坏的形态。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path(path)
    tmp.unlink(missing_ok=True)
    try:
        yield tmp
        # replace 也在 try 里：它自己失败（盘满、权限、目标是个目录）时同样不该把临时
        # 文件留在磁盘上。os.replace 是原子的，所以它抛异常就意味着「什么都没发生」，
        # 删掉 tmp 是安全的。
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """`Path.write_text` 的原子版本。"""
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding=encoding)


def copy_file(src: str | PathLike[str], dest: str | PathLike[str]) -> None:
    """`shutil.copyfile` 的原子版本。

    用在 `register_episode` 拷 SRT 上：那份 SRT 是 ingest 阶段的输入，半截的字幕
    文件会静默产出一条缺对白的对白轨（`srt_parser` 对截断输入不报错）。
    """
    with atomic_path(Path(dest)) as tmp:
        shutil.copyfile(src, tmp)
