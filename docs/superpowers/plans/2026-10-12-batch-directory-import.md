# 批量目录导入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `tenmin import <slug> <directory>`：一次扫描目录里的视频与配对 SRT、识别集号（正则优先、LLM 兜底）、用现有 `build_track` 预览完整 OP/ED 区间，确认后登记（OP+ED 都推断成功才写区间）并只对本次导入的集逐集跑管线。

**Architecture:** 新模块 `src/tenmin/importer.py` 负责「算出要导入什么」（扫描配对 → 集号解析 → LLM 兜底 → 冲突分类 → 区间推断 → 预览文本）与「按计划登记」，不启动流水线；`pipeline.register_episode` 增加可选的 `op_range`/`ed_range`（仅当这一集两个字段都空时写，ruamel round-trip 写回）；`cli.py` 新增 `@app.command("import")` 的 `import_`，处理确认/`--yes`/`--dry-run`/非 TTY，确认后逐集调用现有 `run_pipeline` 单集模式。

**Tech Stack:** Python 3.14、uv、Typer（`CliRunner` 测试）、pydantic（LLM 兜底 schema）、ruamel.yaml（round-trip 写回）、pytest（`asyncio_mode = "auto"`）、ruff。

**工具规则（来自 AGENTS.md，必须遵守）：** pytest 与 tenmin 命令一律用裸 `uv run ...`，**绝不**套 `rtk` / `rtk proxy` / `rtk pytest` 前缀（中文输出会把 rtk 的抓取层搞崩）。`rtk git` / `rtk ls` / `rtk grep` 可以用。源码/测试里**不许**写 `模块.py:行号` 这种交叉引用、不许写「Task + 数字」这种任务代号（`tests/test_source_hygiene.py` 会扫），`src/` 里不许出现 `assert` 语句。

---

## File Structure

| 文件 | 动作 | 职责 |
| --- | --- | --- |
| `src/tenmin/importer.py` | Create | 目录导入的全部纯逻辑：`scan_directory`（单层枚举 + 主体/语言后缀配对）、`parse_episode_number` / `is_extra`（强/弱正则、噪声 token、非正片关键词）、`resolve_ambiguous`（一次打包的 LLM 兜底）、`round_range` / `infer_ranges`（复用 `build_track` 与 `_source_duration`）、`build_import_plan`（四类分拣 + 重复集号）、`apply_import_plan`（逐集 `register_episode`）、`render_preview`（预览文本）。不启动流水线。 |
| `src/tenmin/pipeline.py` | Modify | `register_episode` 增加 `op_range` / `ed_range` 关键字参数（必须成对；只在这一集两个字段都空时写）；`_write_back_episode` 增加 `write_ranges` 开关，把两个区间以 flow 序列写回。 |
| `src/tenmin/cli.py` | Modify | 新增 `import` 命令（函数 `import_`）、`_stdin_is_interactive`、`_run_imported_and_close`；把 `run` 末尾打印产物路径的那段抽成 `_print_artifacts` 供两个命令共用。 |
| `tests/test_importer.py` | Create | importer 的离线单测（临时目录、golden SRT、monkeypatch 掉 ffprobe、`tests/fakes.py` 的 `FakeProvider`）。 |
| `tests/test_pipeline.py` | Modify | `register_episode` 写区间的新测试（复用文件里已有的 `_COMMENTED_YAML` 与 `_removed_lines`）。 |
| `tests/test_cli.py` | Modify | `tenmin import` 的交互/副作用测试（复用已有的 `runner`、`out`）。 |
| `AGENTS.md` | Modify | 补 `tenmin import` 命令与 `importer.py` 模块说明、`register_episode` 新参数、测试文件数。 |

已核对过的现有事实（实现时直接依赖）：

- `tests/fixtures/saijo_e02.srt` 用 `build_track(..., show_title="才女的侍从", duration=1420.0)` 得到 `op_range=(153.486, 224.681)`、`ed_range=(1348.18, 1416.622)`；`duration=1500.0` 时 `ed_range=None`（实测）。
- `pipeline._source_duration(cfg, episode)` 内部调用模块级 `probe_duration(path, ffprobe=...)`，先查 `path.is_file()`；`FFmpegBinaryError` 外抛，其余 `FFmpegError`/`OSError`/没写 video/文件不在 → `None`。测试 monkeypatch `tenmin.pipeline.probe_duration`。
- `build_provider` 缺 key 抛 `RuntimeError`，`openai_compatible` 缺 `base_url` 抛 `ValueError`。
- `FakeProvider(responses)` 遇到 dict 且传了 schema 时走 `schema.model_validate(payload)`，并把每次调用记进 `.calls`。
- `CliRunner` 的 stdin 不是 TTY；`typer.confirm` 读 `runner.invoke(..., input="y\n")`。

---

### Task 1: 目录扫描与视频/SRT 配对

**Files:**
- Create: `src/tenmin/importer.py`
- Test: `tests/test_importer.py`

- [ ] **Step 1: Write the failing test**

新建 `tests/test_importer.py`：

```python
from pathlib import Path

import pytest

from tenmin.importer import LANGUAGE_SUFFIXES, scan_directory


def _touch(path: Path, data: bytes = b"fake") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_scan_pairs_exact_stem_case_insensitively(tmp_path):
    video = _touch(tmp_path / "Show 01.MKV")
    srt = _touch(tmp_path / "show 01.SRT")

    scan = scan_directory(tmp_path)

    assert [(p.video, p.srt, p.ignored_srts) for p in scan.pairs] == [(video, srt, ())]
    assert scan.unpaired_videos == []
    assert scan.unpaired_srts == []


def test_scan_only_looks_at_the_top_level(tmp_path):
    _touch(tmp_path / "nested" / "Show 02.mkv")
    _touch(tmp_path / "nested" / "Show 02.srt")

    scan = scan_directory(tmp_path)

    assert scan.pairs == []
    assert scan.unpaired_videos == []
    assert scan.unpaired_srts == []


@pytest.mark.parametrize("separator", [".", "_"])
@pytest.mark.parametrize("suffix", LANGUAGE_SUFFIXES)
def test_scan_accepts_a_language_suffix(tmp_path, separator, suffix):
    video = _touch(tmp_path / "Show 01.mp4")
    srt = _touch(tmp_path / f"Show 01{separator}{suffix}.srt")

    scan = scan_directory(tmp_path)

    assert [(p.video, p.srt) for p in scan.pairs] == [(video, srt)]


def test_scan_picks_the_highest_priority_candidate_and_reports_the_rest(tmp_path):
    _touch(tmp_path / "Show 01.mkv")
    exact = _touch(tmp_path / "Show 01.srt")
    hans = _touch(tmp_path / "Show 01.zh-Hans.srt")
    ja = _touch(tmp_path / "Show 01.ja.srt")

    scan = scan_directory(tmp_path)

    assert len(scan.pairs) == 1
    assert scan.pairs[0].srt == exact
    assert scan.pairs[0].ignored_srts == (hans, ja)
    assert scan.unpaired_srts == []


def test_scan_language_priority_follows_the_list_order(tmp_path):
    _touch(tmp_path / "Show 01.mkv")
    jpn = _touch(tmp_path / "Show 01.jpn.srt")
    tc = _touch(tmp_path / "Show 01_tc.srt")

    scan = scan_directory(tmp_path)

    assert scan.pairs[0].srt == tc
    assert scan.pairs[0].ignored_srts == (jpn,)


def test_scan_lists_unpaired_files_and_does_no_fuzzy_matching(tmp_path):
    lonely_video = _touch(tmp_path / "Show 03.mkv")
    lonely_srt = _touch(tmp_path / "Show 03 v2.srt")
    _touch(tmp_path / "notes.txt")

    scan = scan_directory(tmp_path)

    assert scan.pairs == []
    assert scan.unpaired_videos == [lonely_video]
    assert scan.unpaired_srts == [lonely_srt]


def test_scan_pairs_from_a_separate_srt_dir(tmp_path):
    video = _touch(tmp_path / "videos" / "Show 04.mkv")
    srt = _touch(tmp_path / "subs" / "Show 04.zh-Hant.srt")
    _touch(tmp_path / "videos" / "Show 04.srt")

    scan = scan_directory(tmp_path / "videos", tmp_path / "subs")

    assert [(p.video, p.srt) for p in scan.pairs] == [(video, srt)]


def test_scan_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到目录"):
        scan_directory(tmp_path / "nope")


def test_scan_rejects_a_file_given_as_directory(tmp_path):
    file = _touch(tmp_path / "Show 01.mkv")
    with pytest.raises(ValueError, match="不是目录"):
        scan_directory(file)


def test_scan_rejects_a_missing_srt_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到目录"):
        scan_directory(tmp_path, tmp_path / "subs")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_importer.py -q`
Expected: FAIL（collection error：`ModuleNotFoundError: No module named 'tenmin.importer'`）

- [ ] **Step 3: Write minimal implementation**

新建 `src/tenmin/importer.py`：

```python
"""`tenmin import`：扫描一个目录，把视频与同名 SRT 配对、识别集号、预览 OP/ED 区间。

只负责「算出要导入什么」和「按用户确认过的计划登记」，不启动流水线 —— 那是 cli 的事。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tenmin.ingest.ocr_batch import VIDEO_SUFFIXES

# 视频主体后面允许跟的语言后缀。顺序就是同一视频命中多个候选 SRT 时的优先级
# （主体完全相等的排在它们全部之前）。必须放开：`tenmin ocr` 自己的产物就是
# `<stem>.zh-Hans.srt` / `<stem>.zh-Hant.srt`，下载来的字幕也普遍带这类后缀。
LANGUAGE_SUFFIXES = ("zh-Hans", "zh-Hant", "sc", "tc", "chs", "cht", "zh", "ja", "jp", "jpn")
_LANGUAGE_SEPARATORS = (".", "_")


@dataclass(frozen=True)
class Pair:
    """一个视频与选中的那份 SRT；ignored_srts 是同样配得上、但优先级更低的候选。"""

    video: Path
    srt: Path
    ignored_srts: tuple[Path, ...] = ()


@dataclass
class Scan:
    pairs: list[Pair] = field(default_factory=list)
    unpaired_videos: list[Path] = field(default_factory=list)
    unpaired_srts: list[Path] = field(default_factory=list)


def _require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"找不到目录：{path}")
    if not path.is_dir():
        raise ValueError(f"不是目录：{path}")


def _srt_rank(video_stem: str, srt: Path) -> int | None:
    """这份 SRT 能不能配给这个视频；能就返回优先级（越小越优先），不能返回 None。

    大小写不敏感。只认「主体完全相等」与「主体 + 分隔符 + 一个语言后缀」两种形态，
    刻意不做模糊匹配：配错一集比少配一集难查得多。
    """
    stem = srt.name[: -len(".srt")].casefold()
    base = video_stem.casefold()
    if stem == base:
        return 0
    for rank, suffix in enumerate(LANGUAGE_SUFFIXES, start=1):
        for separator in _LANGUAGE_SEPARATORS:
            if stem == f"{base}{separator}{suffix.casefold()}":
                return rank
    return None


def scan_directory(video_dir: Path, srt_dir: Path | None = None) -> Scan:
    """单层枚举视频与 SRT 并配对。srt_dir 省略时从视频目录里找字幕。

    视频后缀复用 `ocr_batch.VIDEO_SUFFIXES`，字幕只认 `.srt`，两者都大小写不敏感。
    结果按文件名排序，保证预览与 LLM 兜底的输入顺序稳定。
    """
    _require_dir(video_dir)
    if srt_dir is not None:
        _require_dir(srt_dir)
    srt_root = srt_dir if srt_dir is not None else video_dir
    videos = sorted(
        (p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda p: p.name,
    )
    srts = sorted(
        (p for p in srt_root.iterdir() if p.is_file() and p.suffix.lower() == ".srt"),
        key=lambda p: p.name,
    )
    scan = Scan()
    claimed: set[Path] = set()
    for video in videos:
        ranked: list[tuple[int, Path]] = []
        for srt in srts:
            rank = _srt_rank(video.stem, srt)
            if rank is not None:
                ranked.append((rank, srt))
        if not ranked:
            scan.unpaired_videos.append(video)
            continue
        ranked.sort(key=lambda item: (item[0], item[1].name))
        scan.pairs.append(
            Pair(
                video=video,
                srt=ranked[0][1],
                ignored_srts=tuple(srt for _, srt in ranked[1:]),
            )
        )
        claimed.update(srt for _, srt in ranked)
    scan.unpaired_srts = [srt for srt in srts if srt not in claimed]
    return scan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_importer.py -q`
Expected: PASS（全部用例通过）

- [ ] **Step 5: Commit**

```bash
rtk git add src/tenmin/importer.py tests/test_importer.py && rtk git commit -m "feat: 增加批量导入的目录扫描与字幕配对"
```

---

### Task 2: 集号解析（强/弱正则、噪声 token、非正片识别）

**Files:**
- Modify: `src/tenmin/importer.py`
- Test: `tests/test_importer.py`

- [ ] **Step 1: Write the failing test**

把 `tests/test_importer.py` 顶部的 import 块替换为：

```python
from pathlib import Path

import pytest

from tenmin.importer import (
    LANGUAGE_SUFFIXES,
    Ambiguous,
    is_extra,
    parse_episode_number,
    scan_directory,
)
```

在文件末尾追加：

```python
@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("Show E01", 1),
        ("Show EP02", 2),
        ("Show.S01E03.1080p.WEB", 3),
        ("Show 第04集", 4),
        ("Show 第05話", 5),
        ("Show 第05话", 5),
        ("Show 第０６話", 6),
        ("Show ＥＰ07", 7),
    ],
)
def test_parse_strong_patterns(stem, expected):
    assert parse_episode_number(stem) == expected


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("[ANi] 才女的侍从 - 11 [1080P][Baha][WEB-DL][AAC AVC][CHT]", 11),
        ("Show_08_x264_10bit", 8),
        ("Show 2024 - 09 [H.264 AAC2.0]", 9),
        ("Show.4K.12", 12),
        ("[Group] Show [HEVC FLAC OPUS] [10]", 10),
    ],
)
def test_parse_weak_pattern_skips_noise_tokens(stem, expected):
    assert parse_episode_number(stem) == expected


def test_parse_returns_none_when_no_number_is_left():
    assert parse_episode_number("Kimi no Na wa") is None


def test_parse_reports_ambiguous_weak_candidates():
    assert parse_episode_number("86 - 11") == Ambiguous((11, 86))


def test_parse_reports_conflicting_strong_matches_as_ambiguous():
    assert parse_episode_number("Show E01 E02") == Ambiguous((1, 2))


@pytest.mark.parametrize(
    "stem",
    [
        "Show NCOP",
        "[Group] Show NCED1",
        "Show SP01",
        "[Group] Show OVA",
        "Show PV",
        "Show 预告",
        "Show 預告",
        "Show Menu",
    ],
)
def test_is_extra_flags_non_main_material(stem):
    assert is_extra(stem) is True


@pytest.mark.parametrize("stem", ["Spy x Family - 01", "Show - 01", "Nova E02"])
def test_is_extra_leaves_main_episodes_alone(stem):
    assert is_extra(stem) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_importer.py -q`
Expected: FAIL（`ImportError: cannot import name 'Ambiguous' from 'tenmin.importer'`）

- [ ] **Step 3: Write minimal implementation**

把 `src/tenmin/importer.py` 的 import 块替换为：

```python
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from tenmin.ingest.ocr_batch import VIDEO_SUFFIXES
```

在 `scan_directory` 之后追加：

```python
@dataclass(frozen=True)
class Ambiguous:
    """文件名里剩下不止一个候选集号（升序去重），交给 LLM 兜底。"""

    candidates: tuple[int, ...]


# 第一级强模式。全部在 NFKC 归一（全角数字/字母 → 半角）之后匹配。
# `E` 前面不许是字母，挡掉 `WEB`、`THE01` 这类误命中；S01E01 也会被第二条命中，
# 两条给出同一个数字，去重后不构成歧义。
_STRONG_PATTERNS = (
    re.compile(r"S[0-9]{1,2}E([0-9]{1,3})(?![0-9])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])EP?([0-9]{1,3})(?![0-9])", re.IGNORECASE),
    re.compile(r"第\s*([0-9]{1,3})\s*[集話话]"),
)

# 第二级之前先剔掉的噪声 token：分辨率、编码、位深、音频编码（含声道写法 AAC2.0）、
# 四位年份。两侧不许紧挨字母数字，免得啃掉标题的一部分。
_NOISE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[0-9]{3,4}p|4k|[xh]\.?26[45]|avc|hevc|(?:aac|flac|opus)(?:[0-9]\.[0-9])?"
    r"|[0-9]{1,2}bit|(?:19|20)[0-9]{2}"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# 第二级的分隔符：spec 列的 `-`、空白、`[`、`]`、`.`、`_`，外加中日文命名常见的
# 全角/半角圆括号与 `【】`（只会让更多「被分隔符包住的数字」露出来，不会制造新的歧义判定）。
_WEAK_SEPARATORS = re.compile(r"[-\s\[\]._()（）【】]+")
_WEAK_NUMBER = re.compile(r"[0-9]{1,3}")

# 非正片素材。ASCII 关键词两侧不许是字母（`Spy` 不算 SP、`Nova` 不算 OVA），
# 后面跟数字照样算（NCED1、SP01）。
_EXTRA = re.compile(
    r"(?<![A-Za-z])(?:NCOP|NCED|SP|OVA|PV|Menu)(?![A-Za-z])|预告|預告",
    re.IGNORECASE,
)


def _normalize(stem: str) -> str:
    return unicodedata.normalize("NFKC", stem)


def is_extra(stem: str) -> bool:
    """文件名主体是不是 NCOP / NCED / SP / OVA / PV / 预告 / Menu 这类非正片素材。"""
    return _EXTRA.search(_normalize(stem)) is not None


def _verdict(numbers: list[int]) -> int | Ambiguous | None:
    unique = sorted(set(numbers))
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return Ambiguous(tuple(unique))


def parse_episode_number(stem: str) -> int | Ambiguous | None:
    """从文件名主体里认集号。纯函数，完全离线。

    强模式（E01 / EP01 / S01E01 / 第01集 / 第01話）任一命中就只看强模式；全部未命中
    才走弱模式：剔掉噪声 token 后，被分隔符包围的 1–3 位数字。剩一个 → 集号；
    一个都没有 → None；不止一个 → Ambiguous。**绝不**按排序顺序补集号。
    """
    text = _normalize(stem)
    strong = [int(m.group(1)) for pattern in _STRONG_PATTERNS for m in pattern.finditer(text)]
    if strong:
        return _verdict(strong)
    cleaned = _NOISE.sub(" ", text)
    weak = [
        int(token) for token in _WEAK_SEPARATORS.split(cleaned) if _WEAK_NUMBER.fullmatch(token)
    ]
    return _verdict(weak)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_importer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add src/tenmin/importer.py tests/test_importer.py && rtk git commit -m "feat: 增加批量导入的集号正则解析与非正片识别"
```

---

### Task 3: LLM 兜底 `resolve_ambiguous`

**Files:**
- Modify: `src/tenmin/importer.py`
- Test: `tests/test_importer.py`

- [ ] **Step 1: Write the failing test**

把 `tests/test_importer.py` 顶部的 import 块替换为：

```python
import json
from pathlib import Path

import pytest

from tenmin.importer import (
    EXTRA,
    LANGUAGE_SUFFIXES,
    Ambiguous,
    is_extra,
    parse_episode_number,
    resolve_ambiguous,
    scan_directory,
)

from .fakes import FakeProvider
```

在文件末尾追加：

```python
async def test_resolve_ambiguous_sends_one_batched_call():
    provider = FakeProvider(
        [{"files": [{"id": 1, "episode": 11}, {"id": 2}, {"id": 3, "extra": True}]}]
    )

    verdicts = await resolve_ambiguous(["86 - 11", "Kimi no Na wa", "Mystery Bonus"], provider)

    assert verdicts == [11, None, EXTRA]
    assert len(provider.calls) == 1
    payload = json.loads(provider.calls[0]["user"])
    assert payload == {
        "files": [
            {"id": 1, "filename": "86 - 11"},
            {"id": 2, "filename": "Kimi no Na wa"},
            {"id": 3, "filename": "Mystery Bonus"},
        ]
    }
    assert provider.calls[0]["schema"] is not None


async def test_resolve_ambiguous_treats_missing_or_invalid_answers_as_unknown():
    provider = FakeProvider(
        [{"files": [{"id": 2, "episode": 0}, {"id": 2, "episode": 5}, {"id": 99, "episode": 3}]}]
    )

    verdicts = await resolve_ambiguous(["a", "b"], provider)

    assert verdicts == [None, None]


async def test_resolve_ambiguous_with_no_stems_makes_no_call():
    provider = FakeProvider([])

    assert await resolve_ambiguous([], provider) == []
    assert provider.calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_importer.py -q`
Expected: FAIL（`ImportError: cannot import name 'EXTRA' from 'tenmin.importer'`）

- [ ] **Step 3: Write minimal implementation**

把 `src/tenmin/importer.py` 的 import 块替换为：

```python
from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from tenmin.ingest.ocr_batch import VIDEO_SUFFIXES
from tenmin.script.llm import LLMProvider
```

在 `parse_episode_number` 之后追加：

```python
EXTRA: Literal["extra"] = "extra"
"""LLM 兜底判为「非正片」时 resolve_ambiguous 返回的标记。"""

_RESOLVE_SYSTEM = """\
你在整理同一部番剧同一季的视频文件。请为下面每个文件名判断它是正片第几集：
- 是正片：episode 填集号（正整数），extra 填 false。
- 不是正片（NCOP、NCED、SP、OVA、PV、预告、菜单等）：episode 填 null，extra 填 true。
- 无法确定：episode 填 null，extra 填 false。不要按文件的排列顺序猜集号。
按输入里的 id 逐条返回，不要遗漏，也不要新增。"""


class _EpisodeGuess(BaseModel):
    id: int
    episode: int | None = None
    extra: bool = False


class _EpisodeGuesses(BaseModel):
    files: list[_EpisodeGuess]


async def resolve_ambiguous(
    stems: Sequence[str], provider: LLMProvider
) -> list[int | Literal["extra"] | None]:
    """把正则认不出/认出多个候选的文件名**打包成一次调用**交给 LLM。

    返回值与 stems 逐位对齐：集号、`EXTRA`（非正片）或 None（仍无法确定）。
    模型漏答、答了不在输入里的 id、给出非正整数集号，一律当「无法确定」；同一个 id
    答了两次取第一次。provider 抛的异常原样外抛，由调用方决定怎么降级。
    """
    if not stems:
        return []
    user = json.dumps(
        {"files": [{"id": i, "filename": stem} for i, stem in enumerate(stems, start=1)]},
        ensure_ascii=False,
    )
    response = await provider.complete(_RESOLVE_SYSTEM, user, _EpisodeGuesses)
    by_id: dict[int, _EpisodeGuess] = {}
    for guess in response.files:
        by_id.setdefault(guess.id, guess)
    verdicts: list[int | Literal["extra"] | None] = []
    for i in range(1, len(stems) + 1):
        guess = by_id.get(i)
        if guess is None:
            verdicts.append(None)
        elif guess.extra:
            verdicts.append(EXTRA)
        elif guess.episode is not None and guess.episode >= 1:
            verdicts.append(guess.episode)
        else:
            verdicts.append(None)
    return verdicts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_importer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add src/tenmin/importer.py tests/test_importer.py && rtk git commit -m "feat: 增加批量导入集号的 LLM 兜底"
```

---
### Task 4: 区间推断 `round_range` / `infer_ranges`

**Files:**
- Modify: `src/tenmin/importer.py`
- Test: `tests/test_importer.py`

- [ ] **Step 1: Write the failing test**

把 `tests/test_importer.py` 顶部的 import 块替换为：

```python
import json
from pathlib import Path

import pytest

from tenmin.config import EpisodeConfig, ProjectConfig, load_project
from tenmin.importer import (
    EXTRA,
    LANGUAGE_SUFFIXES,
    Ambiguous,
    infer_ranges,
    is_extra,
    parse_episode_number,
    resolve_ambiguous,
    round_range,
    scan_directory,
)
from tenmin.ingest.normalize import build_track
from tenmin.render.ffmpeg import FFmpegBinaryError, FFmpegError

from .fakes import FakeProvider
```

在文件末尾追加（这几个 helper 后面的任务也会用）：

```python
def _project(tmp_path: Path, extra_yaml: str = "episodes: []\n") -> ProjectConfig:
    root = tmp_path / "work" / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: 才女的侍从\nslug: saijo\n" + extra_yaml, encoding="utf-8"
    )
    return load_project(root / "project.yaml")


def _probe(monkeypatch, result):
    """把 pipeline._source_duration 用的 ffprobe 换成固定结果（数字）或固定异常。"""

    def fake_probe(path, *, ffprobe):
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)


@pytest.fixture
def fixed_duration(monkeypatch):
    _probe(monkeypatch, 1420.0)


def test_round_range_rounds_half_up_to_a_tenth():
    assert round_range((153.486, 224.681)) == (153.5, 224.7)
    assert round_range((0.25, 0.35)) == (0.3, 0.4)


def test_round_range_rejects_a_collapsed_range():
    assert round_range((10.02, 10.04)) is None
    assert round_range((10.06, 10.04)) is None


def test_infer_ranges_matches_build_track_and_writes_both(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)

    track = build_track(
        golden_srt_path,
        episode=2,
        show_title=cfg.show,
        glossary=cfg.glossary,
        convert_traditional=cfg.locale.convert_traditional,
        duration=1420.0,
        ingest=cfg.ingest,
        credits=cfg.credits,
    )
    assert preview.duration == 1420.0
    assert preview.op_range == track.op_range
    assert preview.ed_range == track.ed_range
    assert preview.write == ((153.5, 224.7), (1348.2, 1416.6))
    assert preview.note is None


def test_infer_ranges_writes_nothing_when_only_op_is_found(tmp_path, golden_srt_path, monkeypatch):
    _probe(monkeypatch, 1500.0)
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)

    assert preview.op_range == (153.486, 224.681)
    assert preview.ed_range is None
    assert preview.write is None
    assert "都不写" in preview.note


def test_infer_ranges_writes_nothing_when_duration_is_unknown(
    tmp_path, golden_srt_path, monkeypatch
):
    _probe(monkeypatch, FFmpegError("moov atom not found"))
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)

    assert preview.duration is None
    assert preview.write is None
    assert "片长未知" in preview.note


def test_infer_ranges_lets_a_misconfigured_ffprobe_through(tmp_path, golden_srt_path, monkeypatch):
    _probe(monkeypatch, FFmpegBinaryError("找不到 ffprobe"))
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    with pytest.raises(FFmpegBinaryError):
        infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)


def test_infer_ranges_keeps_existing_manual_ranges(tmp_path, golden_srt_path, fixed_duration):
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e03.mkv")
    existing = EpisodeConfig(number=3, ed_range=(1300.0, 1420.0))

    preview = infer_ranges(cfg, number=3, srt=golden_srt_path, video=video, existing=existing)

    assert preview.ed_range == (1300.0, 1420.0)
    assert preview.write is None
    assert "已有手填区间" in preview.note


def test_infer_ranges_does_not_pin_project_defaults(tmp_path, golden_srt_path, fixed_duration):
    cfg = _project(tmp_path, "credits:\n  default_op_range: [150, 225]\nepisodes: []\n")
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)

    assert preview.op_range == (150.0, 225.0)
    assert preview.write is None
    assert "项目级" in preview.note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_importer.py -q`
Expected: FAIL（`ImportError: cannot import name 'infer_ranges' from 'tenmin.importer'`）

- [ ] **Step 3: Write minimal implementation**

把 `src/tenmin/importer.py` 的 import 块替换为：

```python
from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.ingest.normalize import build_track, credit_range_source
from tenmin.ingest.ocr_batch import VIDEO_SUFFIXES
from tenmin.pipeline import _source_duration
from tenmin.script.llm import LLMProvider
```

在 `resolve_ambiguous` 之后追加：

```python
_TENTH = Decimal("0.1")


def round_range(value: tuple[float, float]) -> tuple[float, float] | None:
    """四舍五入到 0.1 秒（半入，经 str 走 Decimal，避免二进制浮点的 0.25 → 0.2）。

    取整后 start >= end 返回 None：调用方把它当推断失败，不写入。
    """
    start, end = (
        float(Decimal(str(x)).quantize(_TENTH, rounding=ROUND_HALF_UP)) for x in value
    )
    if start >= end:
        return None
    return (start, end)


@dataclass(frozen=True)
class RangePreview:
    """一集的区间预览。

    op_range / ed_range 是 build_track 实际给出的值（含已有手填/项目默认），给人看；
    write 是确认后要写进 project.yaml 的那一对（已取整），None 表示一个都不写，
    note 说明为什么不写。
    """

    duration: float | None
    op_range: tuple[float, float] | None
    ed_range: tuple[float, float] | None
    write: tuple[tuple[float, float], tuple[float, float]] | None
    note: str | None = None


def infer_ranges(
    cfg: ProjectConfig,
    *,
    number: int,
    srt: Path,
    video: Path,
    existing: EpisodeConfig | None = None,
) -> RangePreview:
    """用 ingest 同一套 build_track 推断这一集的 OP/ED，并决定写不写。

    片长走 pipeline._source_duration：`FFmpegBinaryError`（ffprobe 配错）照旧外抛，
    其余探测失败降级为 None —— 此时 build_track 会退化到字幕末尾，ED 口径不可信，
    所以不写。existing 是预填条目：把它已有的区间原样喂给 build_track，预览才跟
    之后真正的 ingest 一致。

    写入规则（OP/ED 两个字段直接驱动 credits 识别窗，只写一半会把另一半的窗关掉）：
    已有任一手填值 → 不写；片长未知 → 不写；项目级默认区间在这一集生效 → 不写
    （那不是推断值，钉成逐集值只会让项目默认以后改不动这一集）；OP 或 ED 任一推不出
    → 都不写；取整后任一区间无效 → 都不写。
    """
    duration = _source_duration(cfg, EpisodeConfig(number=number, video=Path(video).resolve()))
    manual_op = existing.op_range if existing is not None else None
    manual_ed = existing.ed_range if existing is not None else None
    track = build_track(
        srt,
        episode=number,
        source="srt",
        glossary=cfg.glossary,
        convert_traditional=cfg.locale.convert_traditional,
        show_title=cfg.show,
        op_range=manual_op,
        ed_range=manual_ed,
        duration=duration,
        ingest=cfg.ingest,
        credits=cfg.credits,
    )

    def preview(write=None, note=None) -> RangePreview:
        return RangePreview(
            duration=duration,
            op_range=track.op_range,
            ed_range=track.ed_range,
            write=write,
            note=note,
        )

    if manual_op is not None or manual_ed is not None:
        return preview(note="已有手填区间，原样保留")
    if duration is None:
        return preview(note="片长未知，不写区间")
    sources = {
        credit_range_source(None, cfg.credits.default_op_range, track.duration),
        credit_range_source(None, cfg.credits.default_ed_range, track.duration),
    }
    if sources != {"inferred"}:
        return preview(note="项目级默认区间生效，不写逐集区间")
    if track.op_range is None or track.ed_range is None:
        return preview(note="OP 与 ED 未能都识别，两个都不写")
    op = round_range(track.op_range)
    ed = round_range(track.ed_range)
    if op is None or ed is None:
        return preview(note="取整到 0.1 秒后区间无效，两个都不写")
    return preview(write=(op, ed))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_importer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add src/tenmin/importer.py tests/test_importer.py && rtk git commit -m "feat: 增加批量导入的 OP/ED 区间推断预览"
```

---

### Task 5: `register_episode` 可选写入 OP/ED 区间

**Files:**
- Modify: `src/tenmin/pipeline.py`（`_write_back_episode`、`register_episode`）
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_pipeline.py` 里 `test_reregistering_changes_only_that_entrys_source_lines` 之后追加（`_COMMENTED_YAML`、`_removed_lines`、`load_project`、`register_episode`、`pytest` 都已在该文件里）：

```python
def _tiny_srt(path: Path) -> Path:
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n台词\n", encoding="utf-8")
    return path


def test_register_episode_writes_both_ranges_for_a_new_entry(tmp_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(_COMMENTED_YAML, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e04.mkv"
    video.write_bytes(b"fake")

    register_episode(
        cfg,
        episode=4,
        srt=_tiny_srt(tmp_path / "e04.srt"),
        video=video,
        op_range=(90.0, 180.5),
        ed_range=(1348.2, 1416.6),
    )

    after = yaml_path.read_text(encoding="utf-8")
    assert _removed_lines(_COMMENTED_YAML, after) == []
    lines = after.splitlines()
    assert "    op_range: [90.0, 180.5]" in lines
    assert "    ed_range: [1348.2, 1416.6]" in lines
    episode4 = next(e for e in load_project(yaml_path).episodes if e.number == 4)
    assert episode4.op_range == (90.0, 180.5)
    assert episode4.ed_range == (1348.2, 1416.6)
    assert (root / "srt" / "E04.srt").is_file()


def test_register_episode_fills_ranges_of_an_empty_prefilled_entry(tmp_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    before = "show: 才女的侍从\nslug: saijo\nepisodes:\n- number: 5\n"
    yaml_path.write_text(before, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e05.mkv"
    video.write_bytes(b"fake")

    register_episode(
        cfg,
        episode=5,
        srt=_tiny_srt(tmp_path / "e05.srt"),
        video=video,
        op_range=(90.0, 180.5),
        ed_range=(1348.2, 1416.6),
    )

    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    assert "  op_range: [90.0, 180.5]" in lines
    assert "  ed_range: [1348.2, 1416.6]" in lines
    reloaded = load_project(yaml_path)
    assert [e.number for e in reloaded.episodes] == [5]
    assert reloaded.episodes[0].op_range == (90.0, 180.5)


def test_register_episode_never_overrides_a_partial_manual_range(tmp_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(_COMMENTED_YAML, encoding="utf-8")
    cfg = load_project(yaml_path)
    video = tmp_path / "e03.mkv"
    video.write_bytes(b"fake")

    register_episode(
        cfg,
        episode=3,
        srt=_tiny_srt(tmp_path / "e03.srt"),
        video=video,
        op_range=(90.0, 180.5),
        ed_range=(1348.2, 1416.6),
    )

    after = yaml_path.read_text(encoding="utf-8")
    assert _removed_lines(_COMMENTED_YAML, after) == []
    assert "[90.0, 180.5]" not in after
    episode3 = next(e for e in load_project(yaml_path).episodes if e.number == 3)
    assert episode3.op_range is None
    assert episode3.ed_range == (1300.0, 1420.0)


def test_register_episode_rejects_half_a_range_before_touching_anything(tmp_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(_COMMENTED_YAML, encoding="utf-8")
    cfg = load_project(yaml_path)

    with pytest.raises(ValueError, match="同时"):
        register_episode(
            cfg,
            episode=4,
            srt=_tiny_srt(tmp_path / "e04.srt"),
            video=tmp_path / "e04.mkv",
            op_range=(90.0, 180.5),
        )

    assert not (root / "srt" / "E04.srt").exists()
    assert yaml_path.read_text(encoding="utf-8") == _COMMENTED_YAML


def test_register_episode_rejects_an_inverted_range_before_copying(tmp_path):
    root = tmp_path / "saijo"
    (root / "srt").mkdir(parents=True)
    yaml_path = root / "project.yaml"
    yaml_path.write_text(_COMMENTED_YAML, encoding="utf-8")
    cfg = load_project(yaml_path)

    with pytest.raises(ValueError):
        register_episode(
            cfg,
            episode=4,
            srt=_tiny_srt(tmp_path / "e04.srt"),
            video=tmp_path / "e04.mkv",
            op_range=(180.5, 90.0),
            ed_range=(1348.2, 1416.6),
        )

    assert not (root / "srt" / "E04.srt").exists()
    assert yaml_path.read_text(encoding="utf-8") == _COMMENTED_YAML
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -q -k "register_episode_writes_both or empty_prefilled or partial_manual_range or half_a_range or inverted_range"`
Expected: FAIL（`TypeError: register_episode() got an unexpected keyword argument 'op_range'`）

- [ ] **Step 3: Write minimal implementation**

`src/tenmin/pipeline.py` 里把 `_write_back_episode` 的签名和 docstring 首段替换为：

```python
def _write_back_episode(
    yaml_path: Path, entry: EpisodeConfig, *, write_ranges: bool = False
) -> None:
    """只更新这一集的来源字段（write_ranges 时连同 OP/ED 区间），保留其它节点的注释、顺序与格式。
```

（docstring 其余两段原样保留。）把函数末尾这三行：

```python
    target["video"] = str(entry.video)
    buffer = io.StringIO()
    round_trip.dump(data, buffer)
```

替换为：

```python
    target["video"] = str(entry.video)
    if write_ranges:
        # flow 序列写成 `op_range: [153.5, 224.7]`，跟手写的形状一致。
        for key in ("op_range", "ed_range"):
            value = CommentedSeq(list(getattr(entry, key)))
            value.fa.set_flow_style()
            target[key] = value
    buffer = io.StringIO()
    round_trip.dump(data, buffer)
```

把 `register_episode` 的签名替换为：

```python
def register_episode(
    cfg: ProjectConfig,
    *,
    episode: int,
    srt: Path | None,
    video: Path,
    op_range: tuple[float, float] | None = None,
    ed_range: tuple[float, float] | None = None,
) -> ProjectConfig:
```

在 docstring 末尾（「…project.yaml 所在目录解析。」那段之后、结束的 `"""` 之前）追加一段：

```
    op_range / ed_range 是 `tenmin import` 用的推断建议，必须成对给：这两个字段直接
    驱动 credits 识别窗，只写一半会把另一半的窗彻底关掉。只有这一集现有的 op_range 与
    ed_range **都为空**时才写入；任一已有手填值时两个都不动，绝不覆盖人工配置。
```

把函数体里这两句：

```python
    _parse_project_yaml(cfg.config_path)
    # 在拷贝字幕或改写配置之前检查输入，给出可读的错误并避免部分登记。
    if srt is not None and not Path(srt).is_file():
```

替换为：

```python
    _parse_project_yaml(cfg.config_path)
    # 在拷贝字幕或改写配置之前检查输入，给出可读的错误并避免部分登记。
    if (op_range is None) != (ed_range is None):
        raise ValueError("op_range 与 ed_range 必须同时给出：只写一半会关掉另一半的识别窗")
    if op_range is not None:
        # 跟加载 project.yaml 时同一套区间校验（起点 < 终点、不为负），抛 ValueError。
        EpisodeConfig(number=episode, op_range=op_range, ed_range=ed_range)
    if srt is not None and not Path(srt).is_file():
```

把函数末尾这一段：

```python
    existing = next((e for e in cfg.episodes if e.number == episode), None)
    if existing is not None:
        # 预填条目也走这条：只改 srt/video，op_range/ed_range 原样留着。
        existing.srt = relative_srt
        existing.video = video_source
        entry = existing
    else:
        entry = EpisodeConfig(number=episode, srt=relative_srt, video=video_source)
        cfg.episodes.append(entry)

    _write_back_episode(cfg.config_path, entry)
```

替换为：

```python
    existing = next((e for e in cfg.episodes if e.number == episode), None)
    write_ranges = False
    if existing is not None:
        # 预填条目也走这条：只改 srt/video，已有的 op_range/ed_range 原样留着；
        # 两个都空时才接受这次给的建议。
        existing.srt = relative_srt
        existing.video = video_source
        if op_range is not None and existing.op_range is None and existing.ed_range is None:
            existing.op_range = op_range
            existing.ed_range = ed_range
            write_ranges = True
        entry = existing
    else:
        entry = EpisodeConfig(
            number=episode,
            srt=relative_srt,
            video=video_source,
            op_range=op_range,
            ed_range=ed_range,
        )
        write_ranges = op_range is not None
        cfg.episodes.append(entry)

    _write_back_episode(cfg.config_path, entry, write_ranges=write_ranges)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -q -k "register"`
Expected: PASS（新增 5 条 + 既有全部 register 相关用例）

- [ ] **Step 5: Commit**

```bash
rtk git add src/tenmin/pipeline.py tests/test_pipeline.py && rtk git commit -m "feat: 登记集数时支持成对写入 OP/ED 区间"
```

---

### Task 6: 导入计划 `build_import_plan` 与 `apply_import_plan`

**Files:**
- Modify: `src/tenmin/importer.py`
- Test: `tests/test_importer.py`

- [ ] **Step 1: Write the failing test**

把 `tests/test_importer.py` 顶部的 import 块替换为：

```python
import json
from pathlib import Path

import pytest

from tenmin.config import EpisodeConfig, ProjectConfig, load_project
from tenmin.importer import (
    EXTRA,
    LANGUAGE_SUFFIXES,
    Ambiguous,
    apply_import_plan,
    build_import_plan,
    infer_ranges,
    is_extra,
    parse_episode_number,
    resolve_ambiguous,
    round_range,
    scan_directory,
)
from tenmin.ingest.normalize import build_track
from tenmin.render.ffmpeg import FFmpegBinaryError, FFmpegError
from tenmin.script.llm import LLMError

from .fakes import FakeProvider
```

在文件末尾追加：

```python
def _season(tmp_path: Path, golden_srt_path: Path, stems: list[str]) -> Path:
    season = tmp_path / "season"
    season.mkdir(exist_ok=True)
    for stem in stems:
        _touch(season / f"{stem}.mkv")
        _touch(season / f"{stem}.srt", golden_srt_path.read_bytes())
    return season


def _no_llm():
    raise AssertionError("正则已全部命中，不该构造 provider")


class _ClosingProvider(FakeProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FailingProvider:
    async def complete(self, system, user, schema=None):
        raise LLMError("HTTP 429 限流")


async def test_plan_never_builds_a_provider_when_regex_resolves_everything(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [1, 2]
    assert plan.items[0].ranges.write == ((153.5, 224.7), (1348.2, 1416.6))
    assert plan.items[0].prefilled is False
    assert plan.warnings == []


async def test_plan_sends_only_unresolved_names_to_the_llm(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(
        tmp_path, golden_srt_path, ["86 - 11", "Kimi no Na wa", "Mystery Bonus", "Show E01"]
    )
    provider = _ClosingProvider(
        [{"files": [{"id": 1, "episode": 11}, {"id": 2}, {"id": 3, "extra": True}]}]
    )

    plan = await build_import_plan(cfg, season, provider_factory=lambda: provider)

    assert [item.number for item in plan.items] == [1, 11]
    assert [p.name for p in plan.extras] == ["Mystery Bonus.mkv"]
    assert [s.path.name for s in plan.ambiguous] == ["Kimi no Na wa.mkv"]
    sent = json.loads(provider.calls[0]["user"])["files"]
    assert [f["filename"] for f in sent] == ["86 - 11", "Kimi no Na wa", "Mystery Bonus"]
    assert provider.closed is True


async def test_plan_skips_ambiguous_files_when_the_provider_cannot_be_built(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["86 - 11", "Show E01"])

    def factory():
        raise RuntimeError("缺少 Gemini API key，请设置环境变量 TENMIN_GEMINI_API_KEY")

    plan = await build_import_plan(cfg, season, provider_factory=factory)

    assert [item.number for item in plan.items] == [1]
    assert [s.path.name for s in plan.ambiguous] == ["86 - 11.mkv"]
    assert any("TENMIN_GEMINI_API_KEY" in w for w in plan.warnings)


async def test_plan_skips_ambiguous_files_when_the_llm_call_fails(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["86 - 11", "Show E01"])

    plan = await build_import_plan(cfg, season, provider_factory=_FailingProvider)

    assert [item.number for item in plan.items] == [1]
    assert [s.path.name for s in plan.ambiguous] == ["86 - 11.mkv"]
    assert any("429" in w for w in plan.warnings)


async def test_plan_skips_extras_without_asking_the_llm(tmp_path, golden_srt_path, fixed_duration):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show NCOP", "Show SP01", "Show PV", "Show E01"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [1]
    assert sorted(p.name for p in plan.extras) == ["Show NCOP.mkv", "Show PV.mkv", "Show SP01.mkv"]


async def test_plan_skips_every_file_that_shares_an_episode_number(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show EP01 v2", "Show E02"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [2]
    assert sorted(s.path.name for s in plan.ambiguous) == ["Show E01.mkv", "Show EP01 v2.mkv"]
    assert all("重复" in s.reason for s in plan.ambiguous)


async def test_plan_skips_registered_episodes_and_fills_prefilled_ones(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(
        tmp_path,
        "episodes:\n"
        "- number: 1\n  srt: srt/E01.srt\n"
        "- number: 2\n  ed_range: [1300, 1420]\n"
        "- number: 3\n",
    )
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02", "Show E03"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [s.path.name for s in plan.registered] == ["Show E01.mkv"]
    by_number = {item.number: item for item in plan.items}
    assert sorted(by_number) == [2, 3]
    assert by_number[2].prefilled is True
    assert by_number[2].ranges.write is None
    assert by_number[2].ranges.ed_range == (1300.0, 1420.0)
    assert by_number[3].prefilled is True
    assert by_number[3].ranges.write == ((153.5, 224.7), (1348.2, 1416.6))
    assert by_number[3].invalidates_outputs is True
    assert by_number[2].invalidates_outputs is False


async def test_plan_keeps_unpaired_files_out_of_the_way(tmp_path, golden_srt_path, fixed_duration):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01"])
    _touch(season / "Show E02.mkv")
    _touch(season / "Other E05.srt")

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [1]
    assert [p.name for p in plan.unpaired_videos] == ["Show E02.mkv"]
    assert [p.name for p in plan.unpaired_srts] == ["Other E05.srt"]


async def test_plan_records_an_unreadable_episode_and_keeps_going(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"])

    original = infer_ranges

    def fake_infer(cfg, *, number, srt, video, existing=None):
        if number == 1:
            raise ValueError("字幕编码无法识别")
        return original(cfg, number=number, srt=srt, video=video, existing=existing)

    monkeypatch.setattr("tenmin.importer.infer_ranges", fake_infer)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [2]
    assert [s.path.name for s in plan.failed] == ["Show E01.mkv"]
    assert "字幕编码无法识别" in plan.failed[0].reason


async def test_plan_lets_a_misconfigured_ffprobe_through(tmp_path, golden_srt_path, monkeypatch):
    _probe(monkeypatch, FFmpegBinaryError("找不到 ffprobe"))
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01"])

    with pytest.raises(FFmpegBinaryError):
        await build_import_plan(cfg, season, provider_factory=_no_llm)


async def test_plan_with_unknown_duration_still_imports_without_ranges(
    tmp_path, golden_srt_path, monkeypatch
):
    _probe(monkeypatch, FFmpegError("moov atom not found"))
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [1]
    assert plan.items[0].ranges.write is None


async def test_apply_registers_items_and_writes_only_the_planned_ranges(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path, "episodes:\n- number: 2\n  ed_range: [1300, 1420]\n")
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"])
    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    apply_import_plan(cfg, plan)

    reloaded = load_project(cfg.config_path)
    by_number = {e.number: e for e in reloaded.episodes}
    assert by_number[1].srt == Path("srt/E01.srt")
    assert by_number[1].video == (season / "Show E01.mkv").resolve()
    assert by_number[1].op_range == (153.5, 224.7)
    assert by_number[1].ed_range == (1348.2, 1416.6)
    assert by_number[2].op_range is None
    assert by_number[2].ed_range == (1300.0, 1420.0)
    assert (cfg.root / "srt" / "E01.srt").read_bytes() == golden_srt_path.read_bytes()
    assert (cfg.root / "srt" / "E02.srt").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_importer.py -q`
Expected: FAIL（`ImportError: cannot import name 'apply_import_plan' from 'tenmin.importer'`）

- [ ] **Step 3: Write minimal implementation**

把 `src/tenmin/importer.py` 的 import 块替换为：

```python
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.ingest.normalize import build_track, credit_range_source
from tenmin.ingest.ocr_batch import VIDEO_SUFFIXES
from tenmin.pipeline import _source_duration, register_episode
from tenmin.script.llm import LLMProvider
```

在 `infer_ranges` 之后追加：

```python
@dataclass(frozen=True)
class Skipped:
    """跳过的一个视频及原因（给预览用）。"""

    path: Path
    reason: str


@dataclass(frozen=True)
class ImportItem:
    number: int
    pair: Pair
    prefilled: bool
    ranges: RangePreview

    @property
    def invalidates_outputs(self) -> bool:
        """补齐来源的预填集要写区间：EPISODE 配置切片含 op_range/ed_range，
        这一集已有的 ingest/timeline/audio/render 产物会失效重跑，预览要提示。"""
        return self.prefilled and self.ranges.write is not None


@dataclass
class ImportPlan:
    items: list[ImportItem] = field(default_factory=list)
    registered: list[Skipped] = field(default_factory=list)
    ambiguous: list[Skipped] = field(default_factory=list)
    failed: list[Skipped] = field(default_factory=list)
    extras: list[Path] = field(default_factory=list)
    unpaired_videos: list[Path] = field(default_factory=list)
    unpaired_srts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# LLM 兜底调用失败的形态：LLMError（RuntimeError 子类）、没被包装的 httpx 传输异常、
# 以及 schema 校验失败的 pydantic ValidationError（ValueError 子类）。
_LLM_CALL_FAILURES = (RuntimeError, ValueError, httpx.HTTPError)


def _describe(error: BaseException) -> str:
    return str(error) or type(error).__name__


async def _llm_numbers(
    pending: list[Pair],
    provider_factory: Callable[[], LLMProvider],
    plan: ImportPlan,
) -> list[tuple[int, Pair]]:
    """正则没认出来的那批文件交给 LLM。provider 只在这里、且确实有待判文件时才构造。

    构造失败（缺 key）或调用失败都不中断命令：这批文件按歧义跳过并给一条 warning，
    正则已判定的集照常导入。
    """
    if not pending:
        return []

    def skip_all(reason: str) -> list[tuple[int, Pair]]:
        plan.ambiguous.extend(Skipped(pair.video, reason) for pair in pending)
        return []

    try:
        provider = provider_factory()
    except (RuntimeError, ValueError) as error:
        plan.warnings.append(
            f"LLM 兜底不可用（{_describe(error)}），{len(pending)} 个文件按集号无法确定跳过"
        )
        return skip_all("文件名里认不出集号，LLM 兜底不可用")
    try:
        verdicts = await resolve_ambiguous([pair.video.stem for pair in pending], provider)
    except _LLM_CALL_FAILURES as error:
        plan.warnings.append(
            f"LLM 兜底调用失败（{_describe(error)}），{len(pending)} 个文件按集号无法确定跳过"
        )
        return skip_all("文件名里认不出集号，LLM 兜底调用失败")
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()

    numbered: list[tuple[int, Pair]] = []
    for pair, verdict in zip(pending, verdicts, strict=True):
        if verdict == EXTRA:
            plan.extras.append(pair.video)
        elif verdict is None:
            plan.ambiguous.append(Skipped(pair.video, "文件名里认不出集号，LLM 也无法确定"))
        else:
            numbered.append((verdict, pair))
    return numbered


async def build_import_plan(
    cfg: ProjectConfig,
    video_dir: Path,
    srt_dir: Path | None = None,
    *,
    provider_factory: Callable[[], LLMProvider],
) -> ImportPlan:
    """扫描、识别集号、分拣，并对可导入集做区间推断。全程只读，不写任何文件。

    分拣顺序：非正片 → 集号（正则，再 LLM 兜底）→ 重复集号（同号文件全部跳过，
    不靠排序挑一个）→ 已登记且有来源（跳过，不替换来源、不覆盖区间）→ 其余可导入
    （预填条目补齐来源、保留已有区间）。`FFmpegBinaryError` 原样外抛。
    """
    scan = scan_directory(video_dir, srt_dir)
    plan = ImportPlan(
        unpaired_videos=list(scan.unpaired_videos), unpaired_srts=list(scan.unpaired_srts)
    )
    numbered: list[tuple[int, Pair]] = []
    pending: list[Pair] = []
    for pair in scan.pairs:
        if is_extra(pair.video.stem):
            plan.extras.append(pair.video)
            continue
        verdict = parse_episode_number(pair.video.stem)
        if isinstance(verdict, int):
            numbered.append((verdict, pair))
        else:
            pending.append(pair)
    numbered.extend(await _llm_numbers(pending, provider_factory, plan))

    counts = Counter(number for number, _ in numbered)
    by_number = {episode.number: episode for episode in cfg.episodes}
    for number, pair in sorted(numbered, key=lambda item: (item[0], item[1].video.name)):
        if counts[number] > 1:
            plan.ambiguous.append(Skipped(pair.video, f"集号 {number} 重复，同号文件都跳过"))
            continue
        existing = by_number.get(number)
        if existing is not None and existing.has_source:
            plan.registered.append(
                Skipped(pair.video, f"第 {number} 集已登记，跳过（不替换来源、不覆盖区间）")
            )
            continue
        try:
            ranges = infer_ranges(
                cfg, number=number, srt=pair.srt, video=pair.video, existing=existing
            )
        except (ValueError, OSError) as error:
            plan.failed.append(Skipped(pair.video, f"字幕读不出来：{_describe(error)}"))
            continue
        plan.items.append(
            ImportItem(number=number, pair=pair, prefilled=existing is not None, ranges=ranges)
        )
    return plan


def apply_import_plan(cfg: ProjectConfig, plan: ImportPlan) -> ProjectConfig:
    """按确认过的计划逐集登记：拷 SRT 到 srt/E{NN}.srt、记视频绝对路径、写区间建议。

    每一集各自原子写回 project.yaml，不做跨集事务：中途失败时已登记的集保留。
    """
    for item in plan.items:
        op_range, ed_range = item.ranges.write if item.ranges.write is not None else (None, None)
        cfg = register_episode(
            cfg,
            episode=item.number,
            srt=item.pair.srt,
            video=item.pair.video,
            op_range=op_range,
            ed_range=ed_range,
        )
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_importer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add src/tenmin/importer.py tests/test_importer.py && rtk git commit -m "feat: 增加批量导入计划的分拣与登记"
```

---
### Task 7: 预览渲染 `render_preview`

**Files:**
- Modify: `src/tenmin/importer.py`
- Test: `tests/test_importer.py`

- [ ] **Step 1: Write the failing test**

把 `tests/test_importer.py` 顶部 `from tenmin.importer import (...)` 那一段替换为：

```python
from tenmin.importer import (
    EXTRA,
    LANGUAGE_SUFFIXES,
    Ambiguous,
    ImportItem,
    ImportPlan,
    Pair,
    RangePreview,
    Skipped,
    apply_import_plan,
    build_import_plan,
    infer_ranges,
    is_extra,
    parse_episode_number,
    render_preview,
    resolve_ambiguous,
    round_range,
    scan_directory,
)
```

在文件末尾追加：

```python
_WRITE = ((153.5, 224.7), (1348.2, 1416.6))


def _item(
    number: int,
    *,
    write=_WRITE,
    note=None,
    prefilled=False,
    duration=1420.0,
    ed=(1348.18, 1416.622),
    ignored=(),
) -> ImportItem:
    pair = Pair(
        video=Path(f"/season/Show E{number:02d}.mkv"),
        srt=Path(f"/season/Show E{number:02d}.srt"),
        ignored_srts=tuple(Path(f"/season/{name}") for name in ignored),
    )
    ranges = RangePreview(
        duration=duration, op_range=(153.486, 224.681), ed_range=ed, write=write, note=note
    )
    return ImportItem(number=number, pair=pair, prefilled=prefilled, ranges=ranges)


def test_preview_shows_pairing_ranges_and_the_write_decision():
    plan = ImportPlan(items=[_item(1, ignored=["Show E01.ja.srt"])])

    text = "\n".join(render_preview(plan))

    assert "可导入 1 集" in text
    assert "E01  视频：Show E01.mkv  字幕：Show E01.srt" in text
    assert "片长 1420.0 秒" in text
    assert "OP：153.5–224.7 秒" in text
    assert "ED：1348.2–1416.6 秒" in text
    assert "将写入区间：op_range [153.5, 224.7]，ed_range [1348.2, 1416.6]" in text
    assert "忽略的候选字幕：Show E01.ja.srt" in text
    assert "重跑" not in text


def test_preview_marks_unknown_duration_and_unrecognised_ranges():
    plan = ImportPlan(
        items=[_item(2, write=None, note="片长未知，不写区间", duration=None, ed=None)]
    )

    text = "\n".join(render_preview(plan))

    assert "片长未知" in text
    assert "ED：未识别" in text
    assert "不写区间：片长未知，不写区间" in text


def test_preview_warns_that_prefilled_episodes_will_rerun():
    plan = ImportPlan(
        items=[
            _item(3, prefilled=True),
            _item(4, prefilled=True, write=None, note="已有手填区间，原样保留"),
        ]
    )

    lines = render_preview(plan)

    assert "E03" in "\n".join(lines)
    warnings = [line for line in lines if "重跑" in line]
    assert len(warnings) == 1
    assert "E03" in warnings[0]
    assert "预填条目" in "\n".join(lines)


def test_preview_lists_every_skipped_category():
    plan = ImportPlan(
        registered=[Skipped(Path("/s/Show E05.mkv"), "第 5 集已登记，跳过")],
        ambiguous=[Skipped(Path("/s/86 - 11.mkv"), "文件名里认不出集号，LLM 兜底不可用")],
        failed=[Skipped(Path("/s/Show E06.mkv"), "字幕读不出来：坏了")],
        extras=[Path("/s/Show NCOP.mkv")],
        unpaired_videos=[Path("/s/Show E07.mkv")],
        unpaired_srts=[Path("/s/Other.srt")],
    )

    text = "\n".join(render_preview(plan))

    assert "可导入 0 集" in text
    assert "已登记跳过：\n  Show E05.mkv：第 5 集已登记，跳过" in text
    assert "集号无法确定跳过：\n  86 - 11.mkv：文件名里认不出集号，LLM 兜底不可用" in text
    assert "字幕读取失败跳过：\n  Show E06.mkv：字幕读不出来：坏了" in text
    assert "非正片跳过：\n  Show NCOP.mkv" in text
    assert "未配对视频：\n  Show E07.mkv" in text
    assert "未配对字幕：\n  Other.srt" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_importer.py -q`
Expected: FAIL（`ImportError: cannot import name 'render_preview' from 'tenmin.importer'`）

- [ ] **Step 3: Write minimal implementation**

在 `src/tenmin/importer.py` 末尾追加：

```python
def _format_range(value: tuple[float, float] | None) -> str:
    if value is None:
        return "未识别"
    return f"{value[0]:.1f}–{value[1]:.1f} 秒"


def render_preview(plan: ImportPlan) -> list[str]:
    """逐项预览：集数、视频、SRT、片长、OP/ED、是否写入区间，以及各类跳过项。"""
    lines = [f"可导入 {len(plan.items)} 集："]
    for item in plan.items:
        ranges = item.ranges
        label = "（预填条目，补齐来源）" if item.prefilled else ""
        lines.append(
            f"  E{item.number:02d}  视频：{item.pair.video.name}  "
            f"字幕：{item.pair.srt.name}{label}"
        )
        duration = "片长未知" if ranges.duration is None else f"片长 {ranges.duration:.1f} 秒"
        lines.append(
            f"       {duration}  OP：{_format_range(ranges.op_range)}  "
            f"ED：{_format_range(ranges.ed_range)}"
        )
        if ranges.write is not None:
            op, ed = ranges.write
            lines.append(f"       将写入区间：op_range {list(op)}，ed_range {list(ed)}")
        else:
            lines.append(f"       不写区间：{ranges.note}")
        if item.pair.ignored_srts:
            lines.append(
                "       忽略的候选字幕：" + "、".join(p.name for p in item.pair.ignored_srts)
            )
        if item.invalidates_outputs:
            lines.append(
                f"       提示：写入区间会改变 E{item.number:02d} 的配置切片，"
                "它已有的 ingest/timeline/audio/render 产物会失效重跑"
            )
    for title, skipped in (
        ("已登记跳过", plan.registered),
        ("集号无法确定跳过", plan.ambiguous),
        ("字幕读取失败跳过", plan.failed),
    ):
        if skipped:
            lines.append(f"{title}：")
            lines.extend(f"  {entry.path.name}：{entry.reason}" for entry in skipped)
    for title, paths in (
        ("非正片跳过", plan.extras),
        ("未配对视频", plan.unpaired_videos),
        ("未配对字幕", plan.unpaired_srts),
    ):
        if paths:
            lines.append(f"{title}：")
            lines.extend(f"  {path.name}" for path in paths)
    return lines
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_importer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add src/tenmin/importer.py tests/test_importer.py && rtk git commit -m "feat: 增加批量导入的预览文本"
```

---

### Task 8: CLI `tenmin import`

**Files:**
- Modify: `src/tenmin/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_cli.py` 末尾追加（`runner`、`out`、`pytest`、`yaml` 已在文件里；`golden_srt_path` 来自 `tests/conftest.py`）：

```python
# --- tenmin import ------------------------------------------------------------


def _import_project(tmp_path, episodes="episodes: []\n"):
    root = tmp_path / "work" / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: 才女的侍从\nslug: saijo\n" + episodes, encoding="utf-8"
    )
    return root


def _import_season(tmp_path, golden_srt_path, stems):
    season = tmp_path / "season"
    season.mkdir()
    for stem in stems:
        (season / f"{stem}.mkv").write_bytes(b"fake video")
        (season / f"{stem}.srt").write_bytes(golden_srt_path.read_bytes())
    return season


@pytest.fixture
def import_env(monkeypatch):
    """假管线（记下每次的 episode）+ 假 provider/TTS + 固定片长。返回 episode 记录。"""
    episodes: list[object] = []

    async def fake_pipeline(cfg, provider, **kwargs):
        episodes.append(kwargs.get("episode"))
        return []

    def fake_probe(path, *, ffprobe):
        return 1420.0

    monkeypatch.setattr("tenmin.cli.run_pipeline", fake_pipeline)
    monkeypatch.setattr("tenmin.cli.build_provider", lambda llm, settings: object())
    monkeypatch.setattr("tenmin.cli.build_tts_engine", lambda cfg: object())
    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)
    return episodes


def _import_args(tmp_path, season, *extra):
    return ["import", "saijo", str(season), "--work-dir", str(tmp_path / "work"), *extra]


def test_import_yes_registers_and_runs_only_the_imported_episodes(
    tmp_path, golden_srt_path, import_env
):
    root = _import_project(tmp_path, "episodes:\n- number: 2\n  srt: srt/E02.srt\n")
    season = _import_season(tmp_path, golden_srt_path, ["Show E01", "Show E02", "Show E03"])

    result = runner.invoke(app, _import_args(tmp_path, season, "--yes"))

    assert result.exit_code == 0, out(result)
    assert import_env == [1, 3]
    assert "已登记跳过" in out(result)
    data = yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8"))
    by_number = {e["number"]: e for e in data["episodes"]}
    assert sorted(by_number) == [1, 2, 3]
    assert by_number[1]["op_range"] == [153.5, 224.7]
    assert by_number[1]["ed_range"] == [1348.2, 1416.6]
    assert by_number[1]["video"] == str((season / "Show E01.mkv").resolve())
    assert "video" not in by_number[2]
    assert (root / "srt" / "E01.srt").is_file()
    assert (root / "srt" / "E03.srt").is_file()


def test_import_dry_run_has_no_side_effects(tmp_path, golden_srt_path, import_env):
    root = _import_project(tmp_path)
    season = _import_season(tmp_path, golden_srt_path, ["Show E01"])
    before = (root / "project.yaml").read_text(encoding="utf-8")

    result = runner.invoke(app, _import_args(tmp_path, season, "--dry-run"))

    assert result.exit_code == 0, out(result)
    assert "OP：153.5–224.7 秒" in out(result)
    assert "dry-run" in out(result)
    assert import_env == []
    assert (root / "project.yaml").read_text(encoding="utf-8") == before
    assert list((root / "srt").iterdir()) == []
    assert not (root / ".config").exists()


def test_import_without_yes_on_a_non_tty_refuses(tmp_path, golden_srt_path, import_env):
    root = _import_project(tmp_path)
    season = _import_season(tmp_path, golden_srt_path, ["Show E01"])
    before = (root / "project.yaml").read_text(encoding="utf-8")

    result = runner.invoke(app, _import_args(tmp_path, season))

    assert result.exit_code == 1
    assert "--yes" in out(result)
    assert import_env == []
    assert (root / "project.yaml").read_text(encoding="utf-8") == before


def test_import_declined_confirmation_changes_nothing(
    tmp_path, golden_srt_path, import_env, monkeypatch
):
    monkeypatch.setattr("tenmin.cli._stdin_is_interactive", lambda: True)
    root = _import_project(tmp_path)
    season = _import_season(tmp_path, golden_srt_path, ["Show E01"])
    before = (root / "project.yaml").read_text(encoding="utf-8")

    result = runner.invoke(app, _import_args(tmp_path, season), input="n\n")

    assert result.exit_code == 0, out(result)
    assert "已取消" in out(result)
    assert import_env == []
    assert (root / "project.yaml").read_text(encoding="utf-8") == before
    assert list((root / "srt").iterdir()) == []


def test_import_accepted_confirmation_imports(
    tmp_path, golden_srt_path, import_env, monkeypatch
):
    monkeypatch.setattr("tenmin.cli._stdin_is_interactive", lambda: True)
    root = _import_project(tmp_path)
    season = _import_season(tmp_path, golden_srt_path, ["Show E01"])

    result = runner.invoke(app, _import_args(tmp_path, season), input="y\n")

    assert result.exit_code == 0, out(result)
    assert import_env == [1]
    assert (root / "srt" / "E01.srt").is_file()


def test_import_with_nothing_importable_asks_nothing(tmp_path, golden_srt_path, import_env):
    root = _import_project(tmp_path)
    season = tmp_path / "season"
    season.mkdir()
    (season / "Show E01.mkv").write_bytes(b"fake video")
    before = (root / "project.yaml").read_text(encoding="utf-8")

    result = runner.invoke(app, _import_args(tmp_path, season))

    assert result.exit_code == 0, out(result)
    assert "没有可导入的集" in out(result)
    assert "未配对视频" in out(result)
    assert import_env == []
    assert (root / "project.yaml").read_text(encoding="utf-8") == before


def test_import_a_bad_episode_does_not_block_the_good_ones(
    tmp_path, golden_srt_path, import_env
):
    _import_project(tmp_path)
    season = _import_season(
        tmp_path, golden_srt_path, ["Show E01", "Show E02", "Show EP02 v2", "Show NCOP"]
    )
    (season / "Show E09.mkv").write_bytes(b"fake video")

    result = runner.invoke(app, _import_args(tmp_path, season, "--yes"))

    assert result.exit_code == 0, out(result)
    assert import_env == [1]
    assert "重复" in out(result)
    assert "非正片跳过" in out(result)


def test_import_rejects_a_missing_directory(tmp_path, import_env):
    _import_project(tmp_path)

    result = runner.invoke(app, _import_args(tmp_path, tmp_path / "nope", "--yes"))

    assert result.exit_code == 1
    assert "找不到目录" in out(result)


def test_import_rejects_a_missing_srt_dir(tmp_path, golden_srt_path, import_env):
    _import_project(tmp_path)
    season = _import_season(tmp_path, golden_srt_path, ["Show E01"])

    result = runner.invoke(
        app, _import_args(tmp_path, season, "--srt-dir", str(tmp_path / "subs"), "--yes")
    )

    assert result.exit_code == 1
    assert "找不到目录" in out(result)


def test_import_pairs_from_srt_dir(tmp_path, golden_srt_path, import_env):
    root = _import_project(tmp_path)
    season = tmp_path / "season"
    season.mkdir()
    (season / "Show E01.mkv").write_bytes(b"fake video")
    subs = tmp_path / "subs"
    subs.mkdir()
    (subs / "Show E01.zh-Hans.srt").write_bytes(golden_srt_path.read_bytes())

    result = runner.invoke(app, _import_args(tmp_path, season, "--srt-dir", str(subs), "--yes"))

    assert result.exit_code == 0, out(result)
    assert import_env == [1]
    assert (root / "srt" / "E01.srt").read_bytes() == golden_srt_path.read_bytes()


def test_import_missing_project_fails(tmp_path):
    result = runner.invoke(
        app, ["import", "nope", str(tmp_path), "--work-dir", str(tmp_path / "work")]
    )

    assert result.exit_code == 1
    assert "project.yaml" in out(result)


def test_import_reports_pipeline_errors_as_one_line(
    tmp_path, golden_srt_path, import_env, monkeypatch
):
    _import_project(tmp_path)
    season = _import_season(tmp_path, golden_srt_path, ["Show E01"])

    async def broken_pipeline(cfg, provider, **kwargs):
        raise ValueError("第 1 集的对白轨坏了")

    monkeypatch.setattr("tenmin.cli.run_pipeline", broken_pipeline)

    result = runner.invoke(app, _import_args(tmp_path, season, "--yes"))

    assert result.exit_code == 1
    assert "第 1 集的对白轨坏了" in out(result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -q -k import_`
Expected: FAIL（exit code 2，`No such command 'import'`）

- [ ] **Step 3: Write minimal implementation**

`src/tenmin/cli.py`：

1. import 区：在 `import asyncio` 下面加一行 `import sys`；把 `from tenmin import atomic` 改为 `from tenmin import atomic, importer`。

2. 把 `run` 函数末尾这一段：

```python
    paths = Paths(cfg.root)
    if episode is not None:
        printed_numbers = [episode]
    else:
        printed_numbers = [e.number for e in cfg.episodes if e.has_source]

    for number in printed_numbers:
        table_path = paths.table(number)
        narration_path = paths.narration(number)
        if table_path.exists():
            typer.echo(f"对照表：{table_path}")
        if narration_path.exists():
            typer.echo(f"配音文本：{narration_path}")
        video_path = paths.video(number)
        if video_path.exists():
            typer.echo(f"成品视频：{video_path}")
```

替换为：

```python
    if episode is not None:
        printed_numbers = [episode]
    else:
        printed_numbers = [e.number for e in cfg.episodes if e.has_source]
    _print_artifacts(cfg, printed_numbers)
```

3. 在 `_run_pipeline_and_close` 之前加：

```python
def _print_artifacts(cfg: ProjectConfig, numbers: list[int]) -> None:
    """跑完之后给出交付物路径（不存在的不打）。run 与 import 共用。"""
    paths = Paths(cfg.root)
    for number in numbers:
        table_path = paths.table(number)
        narration_path = paths.narration(number)
        if table_path.exists():
            typer.echo(f"对照表：{table_path}")
        if narration_path.exists():
            typer.echo(f"配音文本：{narration_path}")
        video_path = paths.video(number)
        if video_path.exists():
            typer.echo(f"成品视频：{video_path}")
```

4. 在 `ocr` 命令的那组 `OCR_*` 常量之前（即 `inspect` 函数之后）加：

```python
IMPORT_DIRECTORY_ARGUMENT = typer.Argument(..., help="视频所在目录（只看第一层，不递归）")
IMPORT_SRT_DIR_OPTION = typer.Option(
    None, "--srt-dir", help="字幕不在视频目录时，从这个目录配对"
)
IMPORT_YES_OPTION = typer.Option(False, "--yes", help="跳过确认，直接登记并运行（脚本 / CI 用）")
IMPORT_DRY_RUN_OPTION = typer.Option(
    False, "--dry-run", help="只预览：不写任何文件、不运行管线"
)


def _stdin_is_interactive() -> bool:
    """单独成函数，测试才能把 CliRunner 的非 TTY stdin 伪装成终端。"""
    return sys.stdin.isatty()


async def _run_imported_and_close(
    cfg: ProjectConfig, plan: importer.ImportPlan, provider: Any, **kwargs: Any
) -> list[str]:
    """登记本次导入的集，再逐集跑单集模式管线，最后关掉 provider 的连接池。

    刻意逐集调用而不是一次批处理：批处理会把存量集也过一遍，`zh/glossary.json` 这类
    项目级输入一变动，存量集就会连带重跑。
    """
    try:
        cfg = importer.apply_import_plan(cfg, plan)
        warnings: list[str] = []
        for item in plan.items:
            warnings.extend(await run_pipeline(cfg, provider, episode=item.number, **kwargs))
        return warnings
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()


@app.command("import")
def import_(
    slug: str,
    directory: Path = IMPORT_DIRECTORY_ARGUMENT,
    work_dir: Path = WORK_DIR_OPTION,
    srt_dir: Path | None = IMPORT_SRT_DIR_OPTION,
    yes: bool = IMPORT_YES_OPTION,
    dry_run: bool = IMPORT_DRY_RUN_OPTION,
) -> None:
    """扫描目录批量导入：配对视频与 SRT、识别集号、预览 OP/ED，确认后登记并逐集运行。"""
    project_file = _project_file(work_dir, slug)
    # 扫描、集号解析（含 LLM 兜底）、ffprobe 与区间推断都是只读的，--dry-run 也照跑：
    # 预览不完整就没有意义。
    try:
        cfg = load_project(project_file)
        plan = asyncio.run(
            importer.build_import_plan(
                cfg,
                directory,
                srt_dir,
                provider_factory=lambda: build_provider(cfg.llm, Settings()),
            )
        )
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error

    for line in importer.render_preview(plan):
        typer.echo(line)
    for warning in plan.warnings:
        typer.secho(f"[warn] {warning}", fg="yellow")

    if not plan.items:
        typer.echo("没有可导入的集，未做任何改动。")
        return
    if dry_run:
        typer.echo("--dry-run：只预览，未写入任何文件，也未运行管线。")
        return
    if not yes:
        if not _stdin_is_interactive():
            typer.secho("当前不是交互终端，无法确认；确认要导入请加 --yes", fg="red", err=True)
            raise typer.Exit(code=1)
        if not typer.confirm(f"导入以上 {len(plan.items)} 集并运行管线？", default=False):
            typer.echo("已取消，未做任何改动。")
            return

    # 先把 provider 建好再登记：缺 key 时一个文件都不动，而不是登记完才在管线里炸。
    try:
        provider = build_provider(cfg.llm, Settings())
    except (RuntimeError, ValueError) as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    tts_engine = build_tts_engine(cfg.render)

    try:
        with RichProgressReporter() as reporter:
            warnings = asyncio.run(
                _run_imported_and_close(
                    cfg, plan, provider, tts_engine=tts_engine, reporter=reporter
                )
            )
    except PIPELINE_ERRORS as error:
        typer.secho(_error_message(error), fg="red", err=True)
        raise typer.Exit(code=1) from error

    for warning in warnings:
        typer.secho(f"[warn] {warning}", fg="yellow")
    _print_artifacts(cfg, [item.number for item in plan.items])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -q`
Expected: PASS（新增 `import_` 用例与既有 `run`/`inspect`/`ocr` 用例全部通过 —— 后者守住 `_print_artifacts` 抽取没改行为）

- [ ] **Step 5: Commit**

```bash
rtk git add src/tenmin/cli.py tests/test_cli.py && rtk git commit -m "feat: 增加 tenmin import 批量目录导入命令"
```

---

### Task 9: 文档与全量验证

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: 更新 AGENTS.md**

1. 把 `src/tenmin/pipeline.py` 条目里的 `` `register_episode()`（`--episode --video` 注册新集，`--srt` 可省）。`` 替换为：

```
`register_episode()`（`--episode --video` 注册新集，`--srt` 可省；可选成对的 `op_range`/`ed_range` 只在这一集两个字段都空时写入，给 `tenmin import` 用）。
```

2. 把 `src/tenmin/cli.py` 条目开头的 `` Typer CLI（`tenmin init` / `tenmin run` / `tenmin inspect` / `tenmin ocr`）。`tenmin ocr` 是脱离项目的批量画面 OCR（见下面 `ingest/ocr_batch.py`），不走管线。 `` 替换为：

```
Typer CLI（`tenmin init` / `tenmin run` / `tenmin inspect` / `tenmin ocr` / `tenmin import`）。`tenmin ocr` 是脱离项目的批量画面 OCR（见下面 `ingest/ocr_batch.py`），不走管线。`tenmin import <slug> <dir>`（函数名 `import_`，`import` 是关键字）扫描目录批量登记，见下面 `importer.py`：默认打印预览后要确认，`--yes` 跳过确认，`--dry-run` 只预览（含 LLM 兜底与 ffprobe，零写入），非 TTY 且没 `--yes` 报错退出；确认后**只对本次导入的集**逐集跑 `run_pipeline` 单集模式。
```

3. 在 `src/tenmin/ingest/ocr_batch.py` 那一条之后插入新条目：

```
- `src/tenmin/importer.py`：`tenmin import` 的实现，不启动流水线。`scan_directory`（单层、视频后缀复用 `ocr_batch.VIDEO_SUFFIXES`；SRT 按主体完全相等或「主体 + `.`/`_` + 语言后缀」配对，多候选按 `LANGUAGE_SUFFIXES` 顺序取一个、其余列为忽略，不做模糊匹配；`--srt-dir` 换字幕目录）→ `is_extra`（NCOP/NCED/SP/OVA/PV/预告/Menu 跳过）→ `parse_episode_number`（NFKC 后强模式 `E01`/`EP01`/`S01E01`/`第01集`/`第01話`，全不中才走弱模式：剔噪声 token 后被分隔符包住的 1–3 位数字；多个候选返回 `Ambiguous`）→ `resolve_ambiguous`（剩下的文件名**打包一次** LLM 调用；provider 只在有待判文件时才由 factory 构造，构造/调用失败一律按歧义跳过并 warning）→ 重复集号全部跳过 → 已登记且有来源的集跳过 → `infer_ranges`（片长走 `pipeline._source_duration`，区间走 `build_track`，与 ingest 同口径）。**OP/ED 必须一起写**：这两个字段直接驱动 `in_credit_window`，只写一半会把另一半的窗关掉，所以只在这一集两个字段都空、片长已知、项目级默认没生效、OP 与 ED 都推断成功且取整到 0.1 秒后仍合法时才写，否则两个都不写；补齐来源的预填集写区间会改 EPISODE 切片，预览里提示会重跑。
```

4. 把 `tests/` 段落里的 `共 49 个 Python 文件` 改成实际数目：先跑 `/bin/ls tests/*.py | wc -l`（本计划完成后应为 53），把输出的数字填进去。

- [ ] **Step 2: 全量测试**

Run: `uv run pytest tests/ -q`
Expected: PASS（默认跳过的 render/asr/ocr marker 用例显示为 skipped，无 failed）

- [ ] **Step 3: Lint**

Run: `uv run ruff check src tests`
Expected: 只剩一条**改动前就存在**的 `tests/test_render_subtitles.py` E501（main 上 `uv run ruff check src tests` 本来就报它，不属于本计划范围，别顺手改）；本计划新增/修改的文件零报错。若出现 `I001`（import 排序），跑 `uv run ruff check --fix` 只修本计划改动的文件后再确认。

- [ ] **Step 4: 手工冒烟（可选，需真实素材）**

Run: `uv run tenmin import <slug> "<季目录>" --dry-run`
Expected: 打印「可导入 N 集」与每集 OP/ED 预览，`git status` 里 `work/` 下没有任何改动。

- [ ] **Step 5: Commit**

```bash
rtk git add AGENTS.md && rtk git commit -m "docs: AGENTS.md 补 tenmin import 与 importer 模块说明"
```

---

## Self-Review（对照 spec 逐条核对）

| Spec 要求 | 落点 |
| --- | --- |
| 独立命令 `tenmin import <slug> <directory>`，`@app.command("import")` + `import_` | Task 8 |
| 只扫第一层、`VIDEO_SUFFIXES` 大小写不敏感 | Task 1（`test_scan_only_looks_at_the_top_level`、`..._case_insensitively`） |
| 配对：主体相等 / 主体 + `.`/`_` + 10 种语言后缀；多候选按列表顺序取第一个、预览显示选中与忽略；不做模糊匹配 | Task 1（参数化 20 例、优先级、模糊不配）、Task 7（「忽略的候选字幕」） |
| `--srt-dir` | Task 1、Task 8（`test_import_pairs_from_srt_dir`、缺失报错） |
| 强模式 E01/EP01/S01E01/第01集/第01話（半角/全角） | Task 2 |
| 弱模式 + 噪声 token（1080P/720P/2160P/4K/x264/x265/H264/H265/AVC/HEVC/8bit/10bit/AAC/FLAC/OPUS/四位年份）；`[ANi] … - 11 [1080P]…` 取 11 | Task 2 |
| 歧义（多个候选）与无命中 → 打包一次 LLM 调用 | Task 2（`Ambiguous`）、Task 3、Task 6 |
| provider 只在需要兜底时构造（`build_provider(cfg.llm, settings)`）；构造/调用失败不中断，受影响文件按歧义跳过 | Task 6（`_no_llm` 断言、缺 key、LLMError 三个分支）、Task 8（factory 为 `build_provider(cfg.llm, Settings())`） |
| 非正片（NCOP/NCED/SP/OVA/PV/预告/Menu）规则排除 + LLM 显式标记，跳过并列出 | Task 2、Task 3（`EXTRA`）、Task 6、Task 7 |
| 只导入配对成功的；未配对视频/字幕列出跳过 | Task 1、Task 6、Task 7、Task 8 |
| 区间推断复用 `build_track` → `find_credit_ranges`；预览与 `build_track` 一致；未识别显示「未识别」 | Task 4（`test_infer_ranges_matches_build_track...`）、Task 7 |
| 写入规则：两个都空才写；两个都推断成功才写；取整 0.1 秒；取整后 `start >= end` 视为失败 | Task 4（`round_range` 与各分支）、Task 5（API 层成对 + 不覆盖）、Task 6（apply） |
| 已完整登记的集跳过；预填条目补齐来源、保留已有区间 | Task 6、Task 5 |
| 片长：ffprobe 沿用 `_source_duration` 语义（`FFmpegBinaryError` 外抛、其余降级 None）；片长未知显式标注、不写区间但仍登记 | Task 4、Task 6（`..._unknown_duration_still_imports...`、`..._misconfigured_ffprobe_through`）、Task 7 |
| 默认确认；拒绝零副作用；`--yes`；`--dry-run` 照常推断但零写入；非 TTY 无 `--yes` 报错 | Task 8 |
| 确认后登记（视频绝对路径、SRT 拷到 `srt/E{NN}.srt`）、逐集 `run_pipeline` 单集模式，不触发全量批处理 | Task 6（apply）、Task 8（`import_env == [1, 3]`） |
| 没有可导入项时只展示、不确认、不跑 | Task 8（`test_import_with_nothing_importable_asks_nothing`） |
| 重复集号列出并跳过；不得按排序补集号 | Task 6、Task 2 |
| 预填集写区间会改 EPISODE 切片 → 预览提示重跑 | Task 6（`invalidates_outputs`）、Task 7 |
| `register_episode` 扩展：ruamel round-trip，保留注释/键序/其他字段，绝不覆盖已有 OP/ED | Task 5（`_removed_lines == []`、partial manual range 不动） |
| 目录/项目错误 → 单行中文错误、非零退出；登记/管线错误走 `PIPELINE_ERRORS` | Task 1、Task 8 |
| 异常集不阻塞有效集 | Task 6（unreadable、LLM 失败、重复）、Task 8（`..._bad_episode_does_not_block...`） |
| 测试命令 `uv run pytest tests/test_importer.py tests/test_cli.py tests/test_pipeline.py -q` 与全量 | Task 9 |

实现过程中需要拍板的 spec 空白（已在代码里落定）：

1. **项目级 `credits.default_*_range` 生效时不写逐集区间**（`infer_ranges` 的 `credit_range_source` 判定）：那不是推断值，钉成逐集值会让以后改项目默认对这一集失效；预览仍显示生效的区间。
2. **字幕读不出来（`build_track` 抛 `ValueError`/`OSError`）单列「字幕读取失败跳过」**，不登记：登记了 ingest 也会在同一处失败，并会挡住整批的全局 ingest 阶段。
3. **弱模式分隔符在 spec 列表外加了圆括号与 `【】`**，噪声 token 额外认 `AAC2.0` 这类声道写法：只会让更多真实命名被正则直接认出，不改变歧义判定规则。
4. **重复集号**：同号的所有文件都跳过（不挑其中一个），哪怕号是 LLM 给的。
5. **登记前先构造管线用的 provider**：缺 key 时一个文件都不动，而不是登记完才在管线里报错。
6. `register_episode` 在 API 层强制 `op_range`/`ed_range` 成对，并在拷贝 SRT 之前用 `EpisodeConfig` 的同一套校验挡住非法区间。
