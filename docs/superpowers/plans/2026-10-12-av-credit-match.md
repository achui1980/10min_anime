# 截图与音频定位 OP/ED Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tenmin import` 在视频目录里发现 `op.*` / `ed.*` 参考截图时，按侧改用「截图定起点 + 跨集音频对齐定 OP 长度、ED 终点取片长」来定 OP/ED 区间；没截图的一侧照旧字幕推断，有截图却对不上的一侧判未识别、不退回字幕推断，写入规则（两侧都成功才写）不变。

**Architecture:** 新模块 `src/tenmin/ingest/av_match.py` 放纯函数（`dhash` / `hamming` / `best_frame` / `audio_envelope` / `op_length` / `median_length`）和 IO（`sample_hashes` 走 `ffmpeg.run_raw_frames`、`image_hash`、`read_pcm` 走新增的 `ffmpeg.run_pcm`、`locate` 粗搜 + 细化），不读 project.yaml。`importer.py` 里 `scan_directory` 把截图放进 `Scan.references`；`build_import_plan` 先算截图哈希（读不出整个报错），分拣完交给 `match_credits` 两遍处理（逐集找起点 → 汇总 OP 起点成功的集做跨集音频对齐求 L），结果以 `AvSide` 按侧传进 `infer_ranges`；`RangePreview` 增加来源/距离/估计字段，`render_preview` 标「画面+音频」/「字幕推断」与 `5/64`。CLI 不用改：截图读不出抛 `ValueError`，`import_` 现有的 `PIPELINE_ERRORS` 分支已经会打一行红字、退出码 1、什么都不写。

**Tech Stack:** Python 3.14、uv、numpy（提为主依赖）、ffmpeg/ffprobe（lavfi 合成测试素材）、pytest（`asyncio_mode = "auto"`）、ruff。

**工具规则（来自 AGENTS.md，必须遵守）：** pytest 与 tenmin 命令一律用裸 `uv run ...`，**绝不**套 `rtk` / `rtk proxy` / `rtk pytest` 前缀（中文输出会把 rtk 的抓取层搞崩）。`rtk git` / `rtk ls` / `rtk grep` 可以用。做变异测试时带 `PYTHONDONTWRITEBYTECODE=1`。源码/测试里**不许**写 `模块.py:行号` 这种交叉引用、不许写「Task + 数字」这种任务代号（`tests/test_source_hygiene.py` 会扫），`src/` 里不许出现 `assert` 语句。

---

## File Structure

| 文件 | 动作 | 职责 |
| --- | --- | --- |
| `pyproject.toml` / `uv.lock` | Modify | numpy 提为主依赖；注册 `av` marker、更新 marker 说明注释。 |
| `src/tenmin/ingest/av_match.py` | Create | 截图与音频定位：dHash 与最佳帧、音频包络与跨集 OP 长度、抽帧/截图/读音频 IO、`locate`。阈值是模块级常量。 |
| `src/tenmin/render/ffmpeg.py` | Modify | 新增 `run_pcm`：s16le 单声道 PCM 走 stdout 一次读回。 |
| `src/tenmin/importer.py` | Modify | `Scan.references` + `_find_references`；`AvSide`、`SOURCE_LABELS`、`RangePreview` 新字段；`infer_ranges` 按侧选来源；`reference_hashes`、`match_credits`（两遍）；`build_import_plan` 接入；`render_preview` 标来源与距离、OP 估计。 |
| `tests/test_av_match.py` | Create | 纯函数合成数据测试；IO 拼装（monkeypatch 子进程/`run_raw_frames`）；lavfi 合成短视频的真 ffmpeg 测试（`render` marker）；真实素材测试（`av` marker）。 |
| `tests/test_importer.py` | Modify | 截图扫描、按侧来源、预览标注；`_FakeAv` 替换 av_match 的三个 IO 函数覆盖两遍流程的各分支。 |
| `tests/test_cli.py` | Modify | 截图读不出 → 退出码 1、零写入。 |
| `tests/conftest.py` / `tests/test_marker_gate.py` | Modify | `av` 进默认跳过门，并测它。 |
| `AGENTS.md` | Modify | 新模块与 importer 截图流程说明、`run_pcm`、marker 五→六、测试文件数 53→54。 |

已核对过的现有事实（实现时直接依赖）：

- `tests/fixtures/saijo_e02.srt` 用 `build_track(..., duration=1420.0)` 推出 `op_range=(153.486, 224.681)`、`ed_range=(1348.18, 1416.622)`；测试里 monkeypatch `tenmin.pipeline.probe_duration` 固定片长（`tests/test_importer.py` 的 `_probe` / `fixed_duration`）。
- `ffmpeg.run_raw_frames(args, *, frame_size, on_frame, ffmpeg)` 返回 stderr 全文；`_require_binary` 找不到二进制抛 `FFmpegBinaryError`（`FFmpegError` 子类），`_require_source_video` 源片不在抛 `FFmpegError`。
- `cli.import_` 对 `build_import_plan` 抛出的 `PIPELINE_ERRORS`（含 `ValueError`、`FFmpegError`）打一行红字、退出码 1。
- 真实片源（saijo E02 / E12）带一条附图「视频」轨（`240x320`、`1200x800`），抽帧必须 `-map 0:v:0`。
- 已在临时副本上用 saijo E02/E12 实测过整套算法：OP 截图（E12 第 6 秒的帧）在 E02 定到 137.0 秒（距离 1/64）、E12 定到 6.0 秒；ED 截图两集距离 0/64；OP 长度两个方向都是 90.1 秒；OP 内 1 秒窗相关全部 ≥0.93，OP 之后全部 ≤0.49；两集合计约 11 秒。
- `uv run ruff check src tests` 在改动前就有一条 `tests/test_render_subtitles.py` 的 E501，不属于本计划，别顺手改；本计划只要求改动过的文件零报错。

---

### Task 1: numpy 提为主依赖

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`（由 `uv lock` 生成）

- [ ] **Step 1: 把 numpy 加进 dependencies**

numpy 现在只是 `uv.lock` 里的传递依赖（默认环境里 `import numpy` 直接失败），av_match 要直接用它。

在 `pyproject.toml` 里把下面这段（全文唯一）：

```toml
    "ruamel-yaml>=0.19.1",
]
```

替换为：

```toml
    "ruamel-yaml>=0.19.1",
    "numpy>=2.0",
]
```

- [ ] **Step 2: 更新锁文件**

Run: `uv lock`

Expected: 退出码 0（`Resolved ... packages`）；`uv.lock` 里 tenmin 的 dependencies 多出 `{ name = "numpy" }`

- [ ] **Step 3: 确认能导入**

Run: `uv sync && uv run python -c "import numpy; print(numpy.__version__)"`

Expected: 打印 `2.x.y`（当前锁定 2.5.3）

- [ ] **Step 4: Commit**

```bash
rtk git add pyproject.toml uv.lock && rtk git commit -m "build: numpy 提为主依赖"
```

---

### Task 2: av_match：dHash 与最佳帧

**Files:**
- Create: `src/tenmin/ingest/av_match.py`
- Create: `tests/test_av_match.py`

- [ ] **Step 1: Write the failing test**

新建 `tests/test_av_match.py`：

```python
import numpy as np
import pytest

from tenmin.ingest import av_match

# --- 纯函数：dHash ------------------------------------------------------------------


def _image(seed: int) -> np.ndarray:
    """一张 9×8 的随机灰度图，取值留在 30–200，给亮度平移留出不截断的余量。"""
    rng = np.random.default_rng(seed)
    return rng.integers(30, 200, size=(av_match.HASH_HEIGHT, av_match.HASH_WIDTH))


def test_dhash_compares_horizontal_neighbours_row_by_row():
    # 每行从左到右递减 → 每次比较都是「左边更亮」→ 64 位全是 1
    falling = np.tile(np.arange(9, 0, -1), (8, 1))
    rising = np.tile(np.arange(9), (8, 1))

    assert av_match.dhash(falling) == (1 << 64) - 1
    assert av_match.dhash(rising) == 0


def test_dhash_accepts_raw_frame_bytes():
    image = _image(1)

    assert av_match.dhash(bytes(image.astype(np.uint8).ravel())) == av_match.dhash(image)


def test_dhash_rejects_a_wrong_pixel_count():
    with pytest.raises(ValueError, match="72"):
        av_match.dhash(bytes(71))


def test_dhash_is_stable_under_brightness_and_contrast_changes():
    image = _image(2)
    brighter = image + 40
    flatter = (image * 0.6 + 20).round()

    reference = av_match.dhash(image)
    assert av_match.hamming(av_match.dhash(brighter), reference) == 0
    assert av_match.hamming(av_match.dhash(flatter), reference) <= 2


def test_dhash_tells_different_pictures_apart():
    distances = [
        av_match.hamming(av_match.dhash(_image(10)), av_match.dhash(_image(seed)))
        for seed in range(11, 21)
    ]

    assert min(distances) > av_match.MAX_DISTANCE


def test_hamming_counts_differing_bits():
    assert av_match.hamming(0b1011, 0b0001) == 2
    assert av_match.hamming(0, (1 << 64) - 1) == 64


# --- 纯函数：best_frame ----------------------------------------------------------------


def test_best_frame_returns_the_closest_frame_and_its_distance():
    ref = 0
    hashes = [0b1111, 0b1, 0b11, 0b1]

    assert av_match.best_frame(hashes, ref) == (1, 1)


def test_best_frame_reports_a_distance_above_the_threshold_without_judging_it():
    ref = 0
    far = (1 << (av_match.MAX_DISTANCE + 1)) - 1

    index, distance = av_match.best_frame([far], ref)

    assert (index, distance) == (0, av_match.MAX_DISTANCE + 1)
    assert distance > av_match.MAX_DISTANCE


def test_best_frame_rejects_an_empty_list():
    with pytest.raises(ValueError, match="没有可比较的帧"):
        av_match.best_frame([], 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_av_match.py -q`

Expected: FAIL — 收集阶段 `ImportError: cannot import name 'av_match' from 'tenmin.ingest'`

- [ ] **Step 3: Write minimal implementation**

新建 `src/tenmin/ingest/av_match.py`：

```python
"""用参考截图与音频定位 OP/ED：截图定起点，跨集音频对齐定 OP 长度。

给 `tenmin import` 用，不读 project.yaml、不碰登记。分两半：

- 纯函数（单独测）：`dhash`、`hamming`、`best_frame`、`audio_envelope`、`op_length`、
  `median_length`。
- IO：`sample_hashes`（视频帧 → dHash，走 `ffmpeg.run_raw_frames`）、`image_hash`
  （截图 → dHash，同一条缩放管线）、`read_pcm`（走 `ffmpeg.run_pcm`）、`locate`
  （粗搜 + 细化）。importer 的测试 monkeypatch 前三个，所以 `locate` 必须经模块全局名
  调用它们。

阈值全是模块级常量：它们是「这个算法怎么判」的合法性边界，不是「这部番想要什么」的
创作旋钮，所以不进 config。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# --- 画面：dHash ----------------------------------------------------------------
# 缩到 9×8 灰度，每行相邻两像素比大小，8 行 × 8 次比较 = 64 位。
HASH_WIDTH = 9
HASH_HEIGHT = 8
HASH_BYTES = HASH_WIDTH * HASH_HEIGHT
HASH_BITS = (HASH_WIDTH - 1) * HASH_HEIGHT
# 最佳距离大于它判未识别（spec：12/64）。
MAX_DISTANCE = 12


def dhash(gray: bytes | Sequence[int] | np.ndarray) -> int:
    """9×8 灰度图（行优先，72 个 0–255 值）→ 64 位 dHash。

    每行从左到右比较相邻两像素，左边更亮记 1。整体亮度平移与对比度拉伸不改变
    大小关系，所以截图与视频帧之间的色彩范围差异（PNG 全范围 vs 视频有限范围）
    基本不影响结果。
    """
    if isinstance(gray, bytes | bytearray):
        pixels = np.frombuffer(bytes(gray), dtype=np.uint8)
    else:
        pixels = np.asarray(gray)
    if pixels.size != HASH_BYTES:
        raise ValueError(f"dHash 需要 {HASH_BYTES} 个像素，收到 {pixels.size} 个")
    grid = pixels.reshape(HASH_HEIGHT, HASH_WIDTH).astype(np.int16)
    value = 0
    for bit in (grid[:, :-1] > grid[:, 1:]).ravel():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """两个 dHash 不同的位数，0–64。"""
    return (a ^ b).bit_count()


def best_frame(hashes: Sequence[int], ref: int) -> tuple[int, int]:
    """与 ref 汉明距离最小的那一帧：(下标, 距离)。同距离取最早的一帧。

    阈值判定（`MAX_DISTANCE`）留给调用方：预览要显示未过阈值时的最佳距离。
    """
    if not hashes:
        raise ValueError("没有可比较的帧")
    distances = [hamming(value, ref) for value in hashes]
    best = min(distances)
    return distances.index(best), best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_av_match.py -q`

Expected: PASS（`9 passed`）

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/tenmin/ingest/av_match.py tests/test_av_match.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
rtk git add src/tenmin/ingest/av_match.py tests/test_av_match.py && rtk git commit -m "feat: 增加 dHash 与最佳帧匹配"
```

---

### Task 3: av_match：音频包络、跨集 OP 长度、中位数

**Files:**
- Modify: `src/tenmin/ingest/av_match.py`
- Modify: `tests/test_av_match.py`

- [ ] **Step 1: Write the failing test**

合成数据：每帧 16 个频带的正态噪声当包络（相互独立的噪声相关约 0，同一段噪声相关为 1），模拟「前 N 秒共享 OP、之后各集自己的内容」。

追加到 `tests/test_av_match.py` 末尾（与前文空两行）：

```python
# --- 纯函数：audio_envelope ------------------------------------------------------------


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def test_audio_envelope_has_ten_frames_per_second_and_drops_the_tail():
    pcm = _pcm(np.zeros(av_match.SAMPLE_RATE * 3 + 123))

    envelope = av_match.audio_envelope(pcm)

    assert envelope.shape == (30, av_match.BANDS)
    assert np.all(envelope == 0)


def test_audio_envelope_puts_a_tone_in_the_right_band():
    t = np.arange(av_match.SAMPLE_RATE * 2) / av_match.SAMPLE_RATE
    pcm = _pcm(8000 * np.sin(2 * np.pi * 440 * t))

    envelope = av_match.audio_envelope(pcm)

    # 440 Hz 落在第 44 个 bin，即 [40, 53) 那个频带
    assert int(np.argmax(envelope.mean(axis=0))) == 8


def test_audio_envelope_of_empty_input_is_empty():
    assert av_match.audio_envelope(b"").shape == (0, av_match.BANDS)


# --- 纯函数：op_length / median_length -------------------------------------------------


def _noise_envelope(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(5.0, 1.0, size=(round(seconds * av_match.ENVELOPE_RATE), av_match.BANDS))


def _episode(op: np.ndarray, op_seconds: float, seed: int, *, lead_frames: int = 0, gain=0.0):
    """前 lead_frames 帧是别的内容，接着是 op 的前 op_seconds 秒，之后换成本集自己的内容。"""
    frames = round(op_seconds * av_match.ENVELOPE_RATE)
    rest = _noise_envelope(av_match.REFERENCE_SECONDS, seed)
    lead = rest[:lead_frames]
    body = np.concatenate([lead, op[:frames] + gain, rest[lead_frames + frames :]])
    return body[: len(op)]


def test_op_length_finds_where_the_shared_audio_ends():
    ref = _noise_envelope(av_match.REFERENCE_SECONDS, 0)
    other = _episode(ref, 90.0, seed=1)

    assert av_match.op_length(ref, other) == pytest.approx(90.0, abs=0.2)


def test_op_length_ignores_a_loudness_difference():
    ref = _noise_envelope(av_match.REFERENCE_SECONDS, 0)
    other = _episode(ref, 72.0, seed=2, gain=1.5)

    assert av_match.op_length(ref, other) == pytest.approx(72.0, abs=0.2)


def test_op_length_absorbs_a_small_start_offset():
    ref = _noise_envelope(av_match.REFERENCE_SECONDS, 0)
    # 这一集的起点早估了 0.4 秒：OP 在它自己的时间轴上晚 4 帧开始、晚 4 帧结束
    other = _episode(ref, 85.0, seed=3, lead_frames=4)

    assert av_match.op_length(ref, other) == pytest.approx(85.4, abs=0.2)


def test_op_length_ignores_a_short_dip_inside_the_op():
    ref = _noise_envelope(av_match.REFERENCE_SECONDS, 0)
    other = _episode(ref, 90.0, seed=4)
    # OP 中间 1 秒被音效盖住：没到连续 2 秒，不算结束
    other[300:310] = _noise_envelope(1.0, 5)

    assert av_match.op_length(ref, other) == pytest.approx(90.0, abs=0.2)


def test_op_length_is_none_when_the_two_never_line_up():
    ref = _noise_envelope(av_match.REFERENCE_SECONDS, 0)
    other = _noise_envelope(av_match.REFERENCE_SECONDS, 6)

    assert av_match.op_length(ref, other) is None


def test_op_length_is_none_when_the_op_outlasts_the_reference_audio():
    ref = _noise_envelope(av_match.REFERENCE_SECONDS, 0)

    assert av_match.op_length(ref, ref.copy()) is None


def test_median_length_drops_outliers_before_taking_the_median():
    assert av_match.median_length([89.9, 90.1, 90.0, 60.0, 130.0]) == pytest.approx(90.0)


def test_median_length_keeps_values_within_ten_seconds():
    assert av_match.median_length([80.0, 90.0, 99.0]) == pytest.approx(90.0)


def test_median_length_of_nothing_is_none():
    assert av_match.median_length([]) is None


def test_median_length_is_none_when_everything_is_an_outlier():
    assert av_match.median_length([60.0, 90.0]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_av_match.py -q`

Expected: FAIL — `AttributeError: module 'tenmin.ingest.av_match' has no attribute 'audio_envelope'`（以及 op_length / median_length）

- [ ] **Step 3: Write minimal implementation**

先换 import 块，再把音频部分追加到文件末尾。

在 `src/tenmin/ingest/av_match.py` 里把下面这段（全文唯一）：

```python
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
```

替换为：

```python
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from statistics import median

import numpy as np
```

追加到 `src/tenmin/ingest/av_match.py` 末尾（与前文空两行）：

```python
# --- 音频：包络与跨集对齐 ------------------------------------------------------------
SAMPLE_RATE = 8000
# 包络每秒 10 帧（每帧 800 个采样点），对齐精度 0.1 秒，跟起点精度一致。
ENVELOPE_RATE = 10
_HOP = SAMPLE_RATE // ENVELOPE_RATE
# 16 个对数间隔的频带，约 40 Hz – 4 kHz（rfft 的第 4 – 400 个 bin，每个 bin 10 Hz）。
BANDS = 16
_BAND_EDGES = np.geomspace(4, _HOP // 2 + 1, BANDS + 1).astype(int)
# 参考集从起点读这么长的音频（spec：150 秒）。
REFERENCE_SECONDS = 150.0
# 相似度按 1 秒窗口算 Pearson 相关；连续 2 秒低于阈值判 OP 结束。阈值的依据：saijo
# E02/E12 实测，OP 内的窗口相关全部 ≥0.93，OP 之后全部 ≤0.49。
SIMILARITY_WINDOW = ENVELOPE_RATE
SIMILARITY_THRESHOLD = 0.5
DROP_SECONDS = 2.0
# 起点各自有 ±0.1 秒的量化误差，再留余量：先在 ±1 秒内找最佳错位，再逐窗比较。
MAX_LAG_FRAMES = ENVELOPE_RATE
ALIGN_FRAMES = 20 * ENVELOPE_RATE
# 跨集长度离中位数超过这么多秒的剔除（spec：10 秒）。
OUTLIER_SECONDS = 10.0
# 只有一集匹配成功、或一个长度都测不出时，OP 长度按这个估计（spec：90 秒）。
DEFAULT_OP_LENGTH = 90.0


def audio_envelope(pcm: bytes) -> np.ndarray:
    """s16le 单声道 8 kHz PCM → (帧数, BANDS) 的对数频带能量，每秒 ENVELOPE_RATE 帧。

    末尾不足一帧的采样丢掉。每帧加 Hann 窗做 rfft，按 `_BAND_EDGES` 把功率谱求和，
    取 log10(能量 + 1)（+1 让静音帧是 0 而不是负无穷）。
    """
    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float64)
    count = samples.size // _HOP
    if count == 0:
        return np.zeros((0, BANDS))
    frames = samples[: count * _HOP].reshape(count, _HOP) * np.hanning(_HOP)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    energies = np.stack(
        [power[:, lo:hi].sum(axis=1) for lo, hi in itertools.pairwise(_BAND_EDGES)], axis=1
    )
    return np.log10(energies + 1.0)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """两段等形状包络拉平后的 Pearson 相关。任一段是常数（方差 0）时记 0。"""
    x = a.ravel() - a.mean()
    y = b.ravel() - b.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator == 0:
        return 0.0
    return float(x @ y) / denominator


def _span(ref: np.ndarray, env: np.ndarray, lag: int) -> tuple[int, int]:
    """错位 lag（ref[i] 对 env[i + lag]）时 ref 上能比较的下标区间 [first, stop)。"""
    return max(0, -lag), min(len(ref), len(env) - lag)


def _best_lag(ref: np.ndarray, env: np.ndarray) -> int:
    """在 ±MAX_LAG_FRAMES 内找让开头 ALIGN_FRAMES 帧最相关的错位。同分取绝对值小的。"""
    best_lag, best_score = 0, -math.inf
    for lag in sorted(range(-MAX_LAG_FRAMES, MAX_LAG_FRAMES + 1), key=abs):
        first, stop = _span(ref, env, lag)
        stop = min(stop, first + ALIGN_FRAMES)
        if stop - first < SIMILARITY_WINDOW:
            continue
        score = _correlation(ref[first:stop], env[first + lag : stop + lag])
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag


def op_length(ref_env: np.ndarray, env: np.ndarray) -> float | None:
    """从起点对齐的两段包络里找 env 那一集的 OP 长度（秒，从它自己的起点算）。

    1. 两段各自减去每个频带的均值（去掉两集共同的频谱倾斜，否则毫不相干的两段音乐
       也会因为「低频总比高频响」而高度相关）。
    2. `_best_lag` 找最佳错位，吸收两集起点的量化误差。
    3. 逐个 1 秒窗口算相关；先要见过至少一个过阈值的窗口（两集确实对上了），之后
       第一次「连续 DROP_SECONDS 秒都低于阈值」的位置就是 OP 结束。窗口在 OP 内外各占
       一半时相关约等于阈值，所以终点 = 那串低值的首个窗口 + 半个窗口。

    一直没对上、或者读到末尾都没掉下去（OP 比参考音频还长），返回 None。
    """
    ref = ref_env - ref_env.mean(axis=0) if len(ref_env) else ref_env
    other = env - env.mean(axis=0) if len(env) else env
    lag = _best_lag(ref, other)
    first, stop = _span(ref, other, lag)
    hold = round(DROP_SECONDS * ENVELOPE_RATE)
    matched = False
    run = 0
    for i in range(first, stop - SIMILARITY_WINDOW + 1):
        value = _correlation(
            ref[i : i + SIMILARITY_WINDOW], other[i + lag : i + lag + SIMILARITY_WINDOW]
        )
        if value >= SIMILARITY_THRESHOLD:
            matched = True
            run = 0
            continue
        if not matched:
            continue
        run += 1
        if run == hold:
            end = i - hold + 1 + lag + SIMILARITY_WINDOW // 2
            return end / ENVELOPE_RATE
    return None


def median_length(lengths: Sequence[float]) -> float | None:
    """剔除离中位数超过 OUTLIER_SECONDS 的长度后再取中位数。空表或全被剔除返回 None。"""
    if not lengths:
        return None
    center = median(lengths)
    kept = [value for value in lengths if abs(value - center) <= OUTLIER_SECONDS]
    if not kept:
        return None
    return median(kept)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_av_match.py -q`

Expected: PASS（`22 passed`）

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/tenmin/ingest/av_match.py tests/test_av_match.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
rtk git add src/tenmin/ingest/av_match.py tests/test_av_match.py && rtk git commit -m "feat: 增加音频包络与跨集 OP 长度对齐"
```

---

### Task 4: ffmpeg.run_pcm

**Files:**
- Modify: `src/tenmin/render/ffmpeg.py`（文件末尾追加）
- Modify: `tests/test_av_match.py`

- [ ] **Step 1: Write the failing test**

替换掉 `subprocess.run` 与 `shutil.which`，不需要真 ffmpeg。

在 `tests/test_av_match.py` 里把下面这段（全文唯一）：

```python
import numpy as np
import pytest

from tenmin.ingest import av_match
```

替换为：

```python
import subprocess

import numpy as np
import pytest

from tenmin.ingest import av_match
from tenmin.render import ffmpeg
from tenmin.render.ffmpeg import FFmpegError
```

追加到 `tests/test_av_match.py` 末尾（与前文空两行）：

```python
# --- ffmpeg.run_pcm（替换掉子进程，不需要真 ffmpeg） -----------------------------


def test_run_pcm_builds_a_mono_s16le_pipe(monkeypatch, tmp_path):
    video = tmp_path / "v.mkv"
    video.write_bytes(b"fake")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b"\x01\x00\x02\x00", stderr=b"")

    monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    pcm = ffmpeg.run_pcm(video, start=1.5, duration=2.0, sample_rate=8000, ffmpeg="ff")

    assert pcm == b"\x01\x00\x02\x00"
    argv, kwargs = calls[0]
    assert argv[0] == "ff"
    assert argv[argv.index("-ss") + 1] == "1.500"
    assert argv[argv.index("-t") + 1] == "2.000"
    assert argv[argv.index("-ac") + 1] == "1"
    assert argv[argv.index("-ar") + 1] == "8000"
    assert argv[argv.index("-f") + 1] == "s16le"
    assert argv[-1] == "-"
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert "text" not in kwargs


def test_run_pcm_reports_a_failure_with_the_stderr_tail(monkeypatch, tmp_path):
    video = tmp_path / "v.mkv"
    video.write_bytes(b"fake")

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr="坏了\n".encode())

    monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    with pytest.raises(FFmpegError, match="坏了"):
        ffmpeg.run_pcm(video, start=0.0, duration=1.0, sample_rate=8000)


def test_run_pcm_rejects_a_missing_video(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: f"/bin/{name}")

    with pytest.raises(FFmpegError, match="找不到源视频"):
        ffmpeg.run_pcm(tmp_path / "nope.mkv", start=0.0, duration=1.0, sample_rate=8000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_av_match.py -q -k run_pcm`

Expected: FAIL — `AttributeError: module 'tenmin.render.ffmpeg' has no attribute 'run_pcm'`

- [ ] **Step 3: Write minimal implementation**

追加到 `src/tenmin/render/ffmpeg.py` 末尾（与前文空两行）：

```python
def run_pcm(
    video: Path,
    *,
    start: float,
    duration: float,
    sample_rate: int,
    ffmpeg: str = FFMPEG,
) -> bytes:
    """从 start 起读 duration 秒音频，返回 s16le 单声道 PCM 原始字节（不落盘）。

    给 `ingest/av_match.py` 的跨集音频对齐用：一次最多 150 秒 × 8 kHz × 2 字节 ≈ 2.4 MB，
    直接攒在内存里，不走 run_raw_frames 的流式回调（PCM 没有「一帧多少字节」的边界）。
    `-ss` 在 `-i` 前做输入端定位；`-vn` 省掉视频解码。stdout 是二进制所以不开 text，
    stderr 按 UTF-8 + errors="replace" 解码，理由同 run。
    """
    _require_binary(ffmpeg)
    _require_source_video(video)
    argv = [
        ffmpeg,
        "-nostdin",
        "-nostats",
        "-hide_banner",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-c:a",
        "pcm_s16le",
        "-",
    ]
    completed = subprocess.run(argv, capture_output=True, stdin=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise FFmpegError(
            f"ffmpeg 执行失败（退出码 {completed.returncode}）：{shlex.join(argv)}\n"
            f"stderr 末尾 {STDERR_TAIL_LINES} 行：\n{tail(_decode(completed.stderr))}"
        )
    return completed.stdout
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_av_match.py -q -k run_pcm`

Expected: PASS（`3 passed`）

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/tenmin/render/ffmpeg.py tests/test_av_match.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
rtk git add src/tenmin/render/ffmpeg.py tests/test_av_match.py && rtk git commit -m "feat: ffmpeg 增加 run_pcm 读原始 PCM"
```

---

### Task 5: av_match：抽帧、截图、读音频与起点定位

**Files:**
- Modify: `src/tenmin/ingest/av_match.py`
- Modify: `tests/test_av_match.py`

- [ ] **Step 1: Write the failing test**

前半段替换掉 `run_raw_frames` / `run_pcm`，默认就跑；后半段用 lavfi 合成 6 秒「旋转的 testsrc2 + 440 Hz 正弦音」，验证真实时间戳、定位精度与字节数，标 `render` marker（默认跳过）。

在 `tests/test_av_match.py` 里把下面这段（全文唯一）：

```python
import subprocess

import numpy as np
import pytest

from tenmin.ingest import av_match
from tenmin.render import ffmpeg
from tenmin.render.ffmpeg import FFmpegError
```

替换为：

```python
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tenmin.ingest import av_match
from tenmin.render import ffmpeg
from tenmin.render.ffmpeg import FFmpegError
```

追加到 `tests/test_av_match.py` 末尾（与前文空两行）：

```python
# --- IO 的拼装（替换掉 ffmpeg 子进程，不需要真 ffmpeg） --------------------------------


def _showinfo(times: list[float]) -> str:
    return "".join(
        f"[Parsed_showinfo_3 @ 0x1] n:{i:4d} pts:{round(t * 10):7d} pts_time:{t:g} duration:1\n"
        for i, t in enumerate(times)
    )


def test_sample_hashes_adds_the_window_start_to_showinfo_pts(monkeypatch):
    calls = []

    def fake_run_raw_frames(args, *, frame_size, on_frame, ffmpeg):
        calls.append(args)
        for _ in range(3):
            on_frame(bytes(frame_size))
        return _showinfo([0.0, 0.5, 1.0])

    monkeypatch.setattr(ffmpeg, "run_raw_frames", fake_run_raw_frames)

    frames = av_match.sample_hashes(Path("/v.mkv"), 10.0, 11.5, 2.0, ffmpeg_path="ff")

    assert frames == [(10.0, 0), (10.5, 0), (11.0, 0)]
    args = calls[0]
    assert args[args.index("-ss") + 1] == "10.000"
    assert args[args.index("-t") + 1] == "1.500"
    assert args[args.index("-map") + 1] == "0:v:0"
    assert args[args.index("-vf") + 1] == "fps=2,scale=9:8:flags=area,format=gray,showinfo"


def test_sample_hashes_rejects_a_frame_and_timestamp_mismatch(monkeypatch):
    def fake_run_raw_frames(args, *, frame_size, on_frame, ffmpeg):
        on_frame(bytes(frame_size))
        return _showinfo([0.0, 0.5])

    monkeypatch.setattr(ffmpeg, "run_raw_frames", fake_run_raw_frames)

    with pytest.raises(FFmpegError, match="1 帧"):
        av_match.sample_hashes(Path("/v.mkv"), 0.0, 1.0, 2.0)


def test_sample_hashes_of_an_empty_window_runs_nothing(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("空窗口不该启动 ffmpeg")

    monkeypatch.setattr(ffmpeg, "run_raw_frames", fail)

    assert av_match.sample_hashes(Path("/v.mkv"), 5.0, 5.0, 2.0) == []


def test_locate_refines_around_the_coarse_best_and_prefers_the_earliest_tie(monkeypatch):
    target = 0

    def fake_sample(video, start, end, fps, *, ffmpeg_path):
        # 静止镜头从 12.3 秒开始、持续 3 秒；其他帧都离得很远
        frames = []
        step = 1 / fps
        t = start
        while t < end - 1e-9:
            near = 12.3 - 1e-9 <= t < 15.3
            frames.append((round(t, 3), target if near else (1 << 40) - 1))
            t = round(t + step, 6)
        return frames

    monkeypatch.setattr(av_match, "sample_hashes", fake_sample)

    match = av_match.locate(Path("/v.mkv"), target, start=0.0, end=300.0)

    assert match == av_match.Match(time=12.3, distance=0)


def test_locate_of_an_empty_window_is_none(monkeypatch):
    monkeypatch.setattr(av_match, "sample_hashes", lambda *a, **k: [])

    assert av_match.locate(Path("/v.mkv"), 0, start=0.0, end=0.0) is None


def test_image_hash_wants_exactly_one_frame(monkeypatch):
    def fake_run_raw_frames(args, *, frame_size, on_frame, ffmpeg):
        return ""

    monkeypatch.setattr(ffmpeg, "run_raw_frames", fake_run_raw_frames)

    with pytest.raises(FFmpegError, match="0 帧"):
        av_match.image_hash(Path("/op.png"))


def test_read_pcm_asks_for_8khz(monkeypatch):
    calls = []

    def fake_run_pcm(video, *, start, duration, sample_rate, ffmpeg):
        calls.append((video, start, duration, sample_rate, ffmpeg))
        return b""

    monkeypatch.setattr(ffmpeg, "run_pcm", fake_run_pcm)

    av_match.read_pcm(Path("/v.mkv"), 3.0, 150.0, ffmpeg_path="ff")

    assert calls == [(Path("/v.mkv"), 3.0, 150.0, 8000, "ff")]


# --- 真 ffmpeg：lavfi 合成的短视频（render marker，默认跳过） ---------------------------


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    """6 秒旋转的 testsrc2 画面 + 440 Hz 正弦音，160×120、24 fps。

    必须转起来：testsrc2 只有一小块在动，缩到 9×8 之后前后几秒的 dHash 一模一样，
    定位会落到最早那一帧（实测 3.0 秒的截图被定到 1.0 秒）。每秒转 60° 时离 3.0 秒
    超过 0.25 秒的帧距离都在 21 以上。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("没有 ffmpeg")
    path = tmp_path_factory.mktemp("av") / "clip.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=24:duration=6,rotate=a=PI*t/3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.mark.render
def test_sample_hashes_reads_real_timestamps(synthetic_video):
    frames = av_match.sample_hashes(synthetic_video, 1.0, 5.0, 2.0)

    assert [t for t, _ in frames] == pytest.approx([1.0 + 0.5 * i for i in range(8)], abs=0.05)


@pytest.mark.render
def test_locate_finds_a_screenshot_to_a_tenth_of_a_second(synthetic_video, tmp_path):
    shot = tmp_path / "op.png"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            "3.0",
            "-i",
            str(synthetic_video),
            "-frames:v",
            "1",
            str(shot),
        ],
        check=True,
    )

    match = av_match.locate(synthetic_video, av_match.image_hash(shot), start=0.0, end=6.0)

    assert match is not None
    assert match.time == pytest.approx(3.0, abs=0.1)
    # 截图是 RGB 的 PNG、视频帧是 yuv420p，转灰度的路径不同，实测距离 3
    assert match.distance <= av_match.MAX_DISTANCE


@pytest.mark.render
def test_image_hash_rejects_a_corrupted_screenshot(tmp_path):
    broken = tmp_path / "op.png"
    broken.write_bytes(b"not an image")

    with pytest.raises(FFmpegError):
        av_match.image_hash(broken)


@pytest.mark.render
def test_read_pcm_returns_the_requested_number_of_bytes(synthetic_video):
    pcm = av_match.read_pcm(synthetic_video, 1.0, 2.0)

    # 2 秒 × 8000 Hz × 2 字节，重采样允许差几个采样点
    assert abs(len(pcm) - 32000) <= 160
    envelope = av_match.audio_envelope(pcm)
    assert int(np.argmax(envelope.mean(axis=0))) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_av_match.py -q`

Expected: FAIL — `AttributeError: module 'tenmin.ingest.av_match' has no attribute 'sample_hashes'`（以及 locate / image_hash / read_pcm）

- [ ] **Step 3: Write minimal implementation**

在 `src/tenmin/ingest/av_match.py` 里把下面这段（全文唯一）：

```python
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from statistics import median

import numpy as np
```

替换为：

```python
from __future__ import annotations

import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from tenmin.render import ffmpeg
```

追加到 `src/tenmin/ingest/av_match.py` 末尾（与前文空两行）：

```python
# --- IO：抽帧、截图、读音频 ----------------------------------------------------------
# 粗搜每秒 2 帧；在粗搜最佳帧前后各 1 秒内改用每秒 10 帧细化，起点精确到 0.1 秒。
COARSE_FPS = 2.0
FINE_FPS = 10.0
FINE_RADIUS_SECONDS = 1.0
# 截图与视频帧走同一条缩放链。flags=area 是按面积平均，9×8 这种极端缩小时最稳。
_SCALE_CHAIN = f"scale={HASH_WIDTH}:{HASH_HEIGHT}:flags=area,format=gray"
# showinfo 每个输出帧一行，形状同 ingest/ocr.py 的 _SHOWINFO_FRAME。
_SHOWINFO_FRAME = re.compile(r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:(\S+)")


def _hash_frames(args: list[str], *, ffmpeg_path: str) -> tuple[list[int], str]:
    hashes: list[int] = []
    stderr = ffmpeg.run_raw_frames(
        args,
        frame_size=HASH_BYTES,
        on_frame=lambda frame: hashes.append(dhash(frame)),
        ffmpeg=ffmpeg_path,
    )
    return hashes, stderr


def sample_hashes(
    video: Path, start: float, end: float, fps: float, *, ffmpeg_path: str = ffmpeg.FFMPEG
) -> list[tuple[float, int]]:
    """[start, end) 内每秒 fps 帧的 (时间, dHash)。end <= start 返回空表、不启动 ffmpeg。

    `-ss` 放在 `-i` 前（输入端快速定位），输出时间戳从 0 起算，所以绝对时间 =
    start + showinfo 的 pts_time。时间读真实 pts 而不是按序号推算，理由同 ingest/ocr.py。
    `-map 0:v:0` 是因为真实片源常带一条附图「视频」轨（saijo 两集都有）。
    帧数与时间戳数对不上抛 FFmpegError（importer 把它记成这一集这一侧未识别）。
    """
    if end <= start:
        return []
    hashes, stderr = _hash_frames(
        [
            "-hide_banner",
            "-v",
            "info",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={fps:g},{_SCALE_CHAIN},showinfo",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        ffmpeg_path=ffmpeg_path,
    )
    times: list[float] = []
    for match in _SHOWINFO_FRAME.finditer(stderr):
        try:
            value = float(match.group(1))
        except ValueError as error:
            raise ffmpeg.FFmpegError(
                f"{video} 的帧时间戳读不出来：{match.group(1)!r}"
            ) from error
        times.append(value)
    if len(times) != len(hashes):
        raise ffmpeg.FFmpegError(
            f"{video} 抽出 {len(hashes)} 帧，showinfo 却报了 {len(times)} 个时间戳"
        )
    return [(start + t, value) for t, value in zip(times, hashes, strict=True)]


def image_hash(path: Path, *, ffmpeg_path: str = ffmpeg.FFMPEG) -> int:
    """参考截图 → dHash，与视频帧走同一条缩放链。读不出抛 FFmpegError。"""
    hashes, _ = _hash_frames(
        [
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            _SCALE_CHAIN,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        ffmpeg_path=ffmpeg_path,
    )
    if len(hashes) != 1:
        raise ffmpeg.FFmpegError(f"截图 {path} 解码出 {len(hashes)} 帧，应该正好 1 帧")
    return hashes[0]


def read_pcm(
    video: Path, start: float, duration: float, *, ffmpeg_path: str = ffmpeg.FFMPEG
) -> bytes:
    """从 start 起读 duration 秒音频：s16le、单声道、SAMPLE_RATE。"""
    return ffmpeg.run_pcm(
        video, start=start, duration=duration, sample_rate=SAMPLE_RATE, ffmpeg=ffmpeg_path
    )


@dataclass(frozen=True)
class Match:
    """一次起点搜索的结果：最佳帧的时间（秒）与汉明距离。是否过阈值由调用方判。"""

    time: float
    distance: int


def locate(
    video: Path, ref: int, *, start: float, end: float, ffmpeg_path: str = ffmpeg.FFMPEG
) -> Match | None:
    """在 [start, end) 里找与截图最像的帧：先每秒 COARSE_FPS 帧粗搜，再在最佳帧前后各
    FINE_RADIUS_SECONDS 秒内每秒 FINE_FPS 帧细化。窗口里一帧都没有返回 None。

    细化结果与粗搜最佳帧放在一起按时间排序后再取最小距离：同距离取最早，对应「静止的
    开场镜头」里真正的第一帧。
    """
    coarse = sample_hashes(video, start, end, COARSE_FPS, ffmpeg_path=ffmpeg_path)
    if not coarse:
        return None
    index, _ = best_frame([value for _, value in coarse], ref)
    center = coarse[index][0]
    fine = sample_hashes(
        video,
        max(start, center - FINE_RADIUS_SECONDS),
        min(end, center + FINE_RADIUS_SECONDS),
        FINE_FPS,
        ffmpeg_path=ffmpeg_path,
    )
    candidates = sorted([*fine, coarse[index]], key=lambda item: item[0])
    index, distance = best_frame([value for _, value in candidates], ref)
    return Match(time=candidates[index][0], distance=distance)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_av_match.py -q`

Expected: PASS（`32 passed, 4 skipped` —— 4 个是 render marker）

- [ ] **Step 5: 跑 lavfi 真 ffmpeg 测试**

Run: `uv run pytest tests/test_av_match.py -q -m render`

Expected: PASS（`4 passed`）。需要 PATH 里有带 libx264/aac 的 ffmpeg；没有 ffmpeg 时 fixture 会 skip

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/tenmin/ingest/av_match.py tests/test_av_match.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
rtk git add src/tenmin/ingest/av_match.py tests/test_av_match.py && rtk git commit -m "feat: 增加截图与视频帧的 dHash 抽取和起点定位"
```

---

### Task 6: 导入扫描识别参考截图

**Files:**
- Modify: `src/tenmin/importer.py`（`Scan`、`scan_directory`）
- Modify: `tests/test_importer.py`

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_importer.py` 末尾（与前文空两行）：

```python
# --- 参考截图：画面 + 音频 -----------------------------------------------------------


def test_scan_finds_reference_screenshots_case_insensitively(tmp_path):
    op = _touch(tmp_path / "OP.PNG")
    ed = _touch(tmp_path / "ed.Jpeg")
    _touch(tmp_path / "cover.jpg")
    _touch(tmp_path / "op.txt")

    scan = scan_directory(tmp_path)

    assert scan.references == {"op": op, "ed": ed}
    assert scan.unpaired_videos == []
    assert scan.unpaired_srts == []


def test_scan_only_looks_for_screenshots_in_the_video_dir(tmp_path):
    _touch(tmp_path / "videos" / "Show 04.mkv")
    _touch(tmp_path / "subs" / "Show 04.srt")
    _touch(tmp_path / "subs" / "op.png")

    scan = scan_directory(tmp_path / "videos", tmp_path / "subs")

    assert scan.references == {}


def test_scan_rejects_two_screenshots_for_the_same_side(tmp_path):
    _touch(tmp_path / "op.png")
    _touch(tmp_path / "op.jpg")

    with pytest.raises(ValueError, match=r"op\.jpg、op\.png"):
        scan_directory(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_importer.py -q -k screenshots`

Expected: FAIL — `AttributeError: 'Scan' object has no attribute 'references'`，以及 `DID NOT RAISE`

- [ ] **Step 3: Write minimal implementation**

在 `src/tenmin/importer.py` 里把下面这段（全文唯一）：

```python
@dataclass
class Scan:
    pairs: list[Pair] = field(default_factory=list)
    unpaired_videos: list[Path] = field(default_factory=list)
    unpaired_srts: list[Path] = field(default_factory=list)
```

替换为：

```python
@dataclass
class Scan:
    pairs: list[Pair] = field(default_factory=list)
    unpaired_videos: list[Path] = field(default_factory=list)
    unpaired_srts: list[Path] = field(default_factory=list)
    # 参考截图：键是 "op" / "ed"。不是视频也不是字幕，所以不进任何未配对列表。
    references: dict[str, Path] = field(default_factory=dict)


# 参考截图：视频目录下主体叫 op / ed 的图片（主体与后缀都不分大小写）。
REFERENCE_SIDES = ("op", "ed")
REFERENCE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def _find_references(video_dir: Path) -> dict[str, Path]:
    """视频目录里的 OP / ED 参考截图。同一侧不止一张时报错：拿哪张都是猜。"""
    found: dict[str, list[Path]] = {side: [] for side in REFERENCE_SIDES}
    for path in video_dir.iterdir():
        side = path.stem.casefold()
        if path.is_file() and side in found and path.suffix.lower() in REFERENCE_SUFFIXES:
            found[side].append(path)
    references: dict[str, Path] = {}
    for side, paths in found.items():
        if len(paths) > 1:
            names = "、".join(sorted(path.name for path in paths))
            raise ValueError(f"{side.upper()} 参考截图不止一张（{names}），只留一张再导入")
        if paths:
            references[side] = paths[0]
    return references
```

在 `src/tenmin/importer.py` 里把下面这段（全文唯一）：

```python
    结果按文件名排序，保证预览与 LLM 兜底的输入顺序稳定。
    """
```

替换为：

```python
    结果按文件名排序，保证预览与 LLM 兜底的输入顺序稳定。视频目录里的 `op.*` / `ed.*`
    截图（`.png` / `.jpg` / `.jpeg`）放进 `references`，只从视频目录找，不看 srt_dir。
    """
```

在 `src/tenmin/importer.py` 里把下面这段（全文唯一）：

```python
    scan.unpaired_srts = [srt for srt in srts if srt not in claimed]
    return scan
```

替换为：

```python
    scan.unpaired_srts = [srt for srt in srts if srt not in claimed]
    scan.references = _find_references(video_dir)
    return scan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_importer.py -q -k screenshots`

Expected: PASS（`3 passed`）

- [ ] **Step 5: 整个 importer 测试文件不回退**

Run: `uv run pytest tests/test_importer.py -q`

Expected: PASS，无 failed

- [ ] **Step 6: Commit**

```bash
rtk git add src/tenmin/importer.py tests/test_importer.py && rtk git commit -m "feat: 导入扫描识别 OP/ED 参考截图"
```

---

### Task 7: 区间预览按侧选来源、标来源与距离

**Files:**
- Modify: `src/tenmin/importer.py`（import、`RangePreview` / `infer_ranges`、`render_preview`）
- Modify: `tests/test_importer.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_importer.py` 里把下面这段（全文唯一）：

```python
    Ambiguous,
    ImportItem,
```

替换为：

```python
    Ambiguous,
    AvSide,
    ImportItem,
```

追加到 `tests/test_importer.py` 末尾（与前文空两行）：

```python
def test_infer_ranges_requires_both_sides_even_when_one_comes_from_the_picture(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")
    ed_miss = AvSide(range=None, distance=20, reason="ED 画面对不上（最佳距离 20/64，超过 12/64）")

    preview = infer_ranges(
        cfg,
        number=2,
        srt=golden_srt_path,
        video=video,
        op_av=AvSide(range=(10.0, 100.0), distance=3),
        ed_av=ed_miss,
    )

    assert preview.op_range == (10.0, 100.0)
    assert preview.ed_range is None
    assert preview.write is None
    assert preview.note == (
        "OP 与 ED 未能都识别，两个都不写（ED 画面对不上（最佳距离 20/64，超过 12/64））"
    )


def test_preview_labels_sources_and_distances():
    pair = Pair(video=Path("/season/Show E01.mkv"), srt=Path("/season/Show E01.srt"))
    ranges = RangePreview(
        duration=1420.0,
        op_range=(12.0, 102.0),
        ed_range=(1348.18, 1416.622),
        write=((12.0, 102.0), (1348.2, 1416.6)),
        op_source="av",
        ed_source="subtitle",
        op_distance=5,
    )
    plan = ImportPlan(items=[ImportItem(number=1, pair=pair, prefilled=False, ranges=ranges)])

    text = "\n".join(render_preview(plan))

    assert "OP：12.0–102.0 秒（画面+音频，5/64）  ED：1348.2–1416.6 秒（字幕推断）" in text
    assert "估计" not in text


def test_preview_shows_the_distance_of_an_unrecognised_side():
    pair = Pair(video=Path("/season/Show E01.mkv"), srt=Path("/season/Show E01.srt"))
    ranges = RangePreview(
        duration=1420.0,
        op_range=None,
        ed_range=(1330.0, 1420.0),
        write=None,
        note="OP 与 ED 未能都识别，两个都不写",
        op_source="av",
        ed_source="av",
        op_distance=15,
        ed_distance=0,
    )
    plan = ImportPlan(items=[ImportItem(number=1, pair=pair, prefilled=False, ranges=ranges)])

    text = "\n".join(render_preview(plan))

    assert "OP：未识别（画面+音频，15/64）" in text
    assert "ED：1330.0–1420.0 秒（画面+音频，0/64）" in text


def test_infer_ranges_marks_subtitle_sources_without_screenshots(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)

    assert (preview.op_source, preview.ed_source) == ("subtitle", "subtitle")
    assert (preview.op_distance, preview.ed_distance) == (None, None)
    assert preview.op_length_estimated is False


def test_infer_ranges_takes_each_side_from_its_own_source(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(
        cfg,
        number=2,
        srt=golden_srt_path,
        video=video,
        op_av=AvSide(range=(10.04, 100.06), distance=3),
    )

    assert preview.op_range == (10.04, 100.06)
    assert preview.ed_range == (1348.18, 1416.622)
    assert (preview.op_source, preview.ed_source) == ("av", "subtitle")
    assert preview.op_distance == 3
    assert preview.write == ((10.0, 100.1), (1348.2, 1416.6))


def test_infer_ranges_keeps_manual_ranges_unlabelled_even_with_a_screenshot(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e03.mkv")
    existing = EpisodeConfig(number=3, ed_range=(1300.0, 1420.0))

    preview = infer_ranges(
        cfg,
        number=3,
        srt=golden_srt_path,
        video=video,
        existing=existing,
        ed_av=AvSide(range=(1330.0, 1420.0), distance=0),
    )

    assert preview.ed_range == (1300.0, 1420.0)
    assert (preview.op_source, preview.ed_source) == (None, None)
    assert preview.write is None


def test_preview_notes_an_estimated_op_length():
    pair = Pair(video=Path("/season/Show E01.mkv"), srt=Path("/season/Show E01.srt"))
    ranges = RangePreview(
        duration=1420.0,
        op_range=(12.0, 102.0),
        ed_range=(1330.0, 1420.0),
        write=((12.0, 102.0), (1330.0, 1420.0)),
        op_source="av",
        ed_source="av",
        op_distance=0,
        ed_distance=0,
        op_length_estimated=True,
    )
    plan = ImportPlan(items=[ImportItem(number=1, pair=pair, prefilled=False, ranges=ranges)])

    assert "       OP 长度按 90 秒估计" in render_preview(plan)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_importer.py -q`

Expected: FAIL — 收集阶段 `ImportError: cannot import name 'AvSide' from 'tenmin.importer'`

- [ ] **Step 3: Write minimal implementation**

`RangePreview` 的新字段都有默认值，已有的 `RangePreview(...)` 构造（`_item` 等）不用改；没截图时来源是 `subtitle`，预览里区间后多一个「（字幕推断）」，已有断言（如 `"OP：153.5–224.7 秒" in text`）是子串匹配，照样通过。

在 `src/tenmin/importer.py` 里把下面这段（全文唯一）：

```python
from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.ingest.normalize
```

替换为：

```python
from tenmin.config import EpisodeConfig, ProjectConfig
from tenmin.ingest import av_match
from tenmin.ingest.normalize
```

在 `src/tenmin/importer.py` 里把下面这段（全文唯一）：

```python
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

替换为：

```python
@dataclass(frozen=True)
class AvSide:
    """一侧（OP 或 ED）的画面+音频结果。

    range 是 None 表示未识别，reason 写原因（会拼进预览的「不写区间」说明）；
    distance 是截图与最佳帧的汉明距离，没比过画面时是 None；estimated 只对 OP 有意义：
    长度没法跨集测，按 `av_match.DEFAULT_OP_LENGTH` 估计。
    """

    range: tuple[float, float] | None
    distance: int | None = None
    reason: str | None = None
    estimated: bool = False


# RangePreview.op_source / ed_source 的取值及预览里的说法。None = 区间来自配置（手填或
# 项目级默认），不标来源。
SOURCE_LABELS = {"av": "画面+音频", "subtitle": "字幕推断"}


@dataclass(frozen=True)
class RangePreview:
    """一集的区间预览。

    op_range / ed_range 是这一侧实际采用的值（有截图的一侧取画面+音频结果，否则取
    build_track 的结果，含已有手填/项目默认），给人看；write 是确认后要写进 project.yaml
    的那一对（已取整），None 表示一个都不写，note 说明为什么不写。op_source / ed_source
    见 SOURCE_LABELS；op_distance / ed_distance 是画面匹配距离（0–64）。
    """

    duration: float | None
    op_range: tuple[float, float] | None
    ed_range: tuple[float, float] | None
    write: tuple[tuple[float, float], tuple[float, float]] | None
    note: str | None = None
    op_source: str | None = None
    ed_source: str | None = None
    op_distance: int | None = None
    ed_distance: int | None = None
    op_length_estimated: bool = False


def infer_ranges(
    cfg: ProjectConfig,
    *,
    number: int,
    srt: Path,
    video: Path,
    existing: EpisodeConfig | None = None,
    op_av: AvSide | None = None,
    ed_av: AvSide | None = None,
) -> RangePreview:
    """推断这一集的 OP/ED，并决定写不写。

    按侧选来源：传了 op_av / ed_av 的一侧（目录里有那张截图）用画面+音频结果，
    **未识别也不退回字幕推断**；没传的一侧用 ingest 同一套 build_track 推断。

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

    def from_config(note: str) -> RangePreview:
        return RangePreview(
            duration=duration,
            op_range=track.op_range,
            ed_range=track.ed_range,
            write=None,
            note=note,
        )

    if manual_op is not None or manual_ed is not None:
        return from_config("已有手填区间，原样保留")

    op = op_av.range if op_av is not None else track.op_range
    ed = ed_av.range if ed_av is not None else track.ed_range

    def preview(write=None, note=None) -> RangePreview:
        return RangePreview(
            duration=duration,
            op_range=op,
            ed_range=ed,
            write=write,
            note=note,
            op_source="av" if op_av is not None else "subtitle",
            ed_source="av" if ed_av is not None else "subtitle",
            op_distance=op_av.distance if op_av is not None else None,
            ed_distance=ed_av.distance if ed_av is not None else None,
            op_length_estimated=op_av is not None and op_av.estimated,
        )

    if duration is None:
        return preview(note="片长未知，不写区间")
    sources = {
        credit_range_source(None, cfg.credits.default_op_range, track.duration),
        credit_range_source(None, cfg.credits.default_ed_range, track.duration),
    }
    if sources != {"inferred"}:
        return from_config("项目级默认区间生效，不写逐集区间")
    if op is None or ed is None:
        reasons = [
            side.reason
            for side in (op_av, ed_av)
            if side is not None and side.range is None and side.reason
        ]
        detail = f"（{'；'.join(reasons)}）" if reasons else ""
        return preview(note=f"OP 与 ED 未能都识别，两个都不写{detail}")
    rounded_op = round_range(op)
    rounded_ed = round_range(ed)
    if rounded_op is None or rounded_ed is None:
        return preview(note="取整到 0.1 秒后区间无效，两个都不写")
    return preview(write=(rounded_op, rounded_ed))
```

在 `src/tenmin/importer.py` 里把下面这段（全文唯一）：

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
            f"  E{item.number:02d}  视频：{item.pair.video.name}  字幕：{item.pair.srt.name}{label}"
        )
        duration = "片长未知" if ranges.duration is None else f"片长 {ranges.duration:.1f} 秒"
        lines.append(
            f"       {duration}  OP：{_format_range(ranges.op_range)}  "
            f"ED：{_format_range(ranges.ed_range)}"
        )
```

替换为：

```python
def _format_range(value: tuple[float, float] | None) -> str:
    if value is None:
        return "未识别"
    return f"{value[0]:.1f}–{value[1]:.1f} 秒"


def _format_side(
    value: tuple[float, float] | None, source: str | None, distance: int | None
) -> str:
    """区间后面标来源与匹配距离：`0.0–95.0 秒（画面+音频，5/64）`。来源未知时只给区间。"""
    text = _format_range(value)
    if source is None:
        return text
    label = SOURCE_LABELS[source]
    if distance is not None:
        label += f"，{distance}/{av_match.HASH_BITS}"
    return f"{text}（{label}）"


def render_preview(plan: ImportPlan) -> list[str]:
    """逐项预览：集数、视频、SRT、片长、OP/ED（含来源与匹配距离）、是否写入区间，以及各类跳过项。"""
    lines = [f"可导入 {len(plan.items)} 集："]
    for item in plan.items:
        ranges = item.ranges
        label = "（预填条目，补齐来源）" if item.prefilled else ""
        lines.append(
            f"  E{item.number:02d}  视频：{item.pair.video.name}  字幕：{item.pair.srt.name}{label}"
        )
        duration = "片长未知" if ranges.duration is None else f"片长 {ranges.duration:.1f} 秒"
        op = _format_side(ranges.op_range, ranges.op_source, ranges.op_distance)
        ed = _format_side(ranges.ed_range, ranges.ed_source, ranges.ed_distance)
        lines.append(f"       {duration}  OP：{op}  ED：{ed}")
        if ranges.op_length_estimated:
            lines.append(f"       OP 长度按 {av_match.DEFAULT_OP_LENGTH:g} 秒估计")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_importer.py tests/test_cli.py -q`

Expected: PASS，无 failed

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/tenmin/importer.py tests/test_importer.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
rtk git add src/tenmin/importer.py tests/test_importer.py && rtk git commit -m "feat: 区间预览按侧标注来源与匹配距离"
```

---

### Task 8: 导入时用截图与音频定位 OP/ED（两遍匹配）

**Files:**
- Modify: `src/tenmin/importer.py`（import、`build_import_plan`，新增 `reference_hashes` / `match_credits` 及其辅助函数）
- Modify: `tests/test_importer.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`_FakeAv` 替换 `av_match.image_hash` / `sample_hashes` / `read_pcm`：画面按「镜头表」合成 dHash，音频用同一段种子噪声当共享 OP、之后换成各集自己的噪声，所以 `audio_envelope` / `op_length` / `median_length` 走的是真实现。`fake_infer` 改成 `**kwargs` 透传，因为 `build_import_plan` 现在会多传 `op_av` / `ed_av`。

在 `tests/test_importer.py` 里把下面这段（全文唯一）：

```python
import pytest

from tenmin.config import EpisodeConfig, ProjectConfig, load_project
```

替换为：

```python
import numpy as np
import pytest

from tenmin.config import EpisodeConfig, ProjectConfig, load_project
```

在 `tests/test_importer.py` 里把下面这段（全文唯一）：

```python
from tenmin.ingest.normalize import build_track
```

替换为：

```python
from tenmin.ingest import av_match
from tenmin.ingest.normalize import build_track
```

在 `tests/test_importer.py` 里把下面这段（全文唯一）：

```python
    def fake_infer(cfg, *, number, srt, video, existing=None):
        if number == 1:
            raise ValueError("字幕编码无法识别")
        return original(cfg, number=number, srt=srt, video=video, existing=existing)
```

替换为：

```python
    def fake_infer(cfg, *, number, **kwargs):
        if number == 1:
            raise ValueError("字幕编码无法识别")
        return original(cfg, number=number, **kwargs)
```

追加到 `tests/test_importer.py` 末尾（与前文空两行）：

```python
_OP_HASH = 0
_ED_HASH = (1 << 64) - 1
_OTHER_HASH = (1 << 32) - 1  # 离两张截图都是 32
_NEAR_MISS_HASH = (1 << 15) - 1  # 离 OP 截图 15，过不了 12/64


class _FakeAv:
    """替换 av_match 的三个 IO 函数。

    scenes：{视频主体: [(起, 止, dHash), ...]}，不在任何镜头里的帧是 _OTHER_HASH。
    op_lengths：{视频主体: 这一集与参考 OP 共享的音频秒数}；所有集的 OP 用同一段噪声，
    之后换成各集自己的噪声。broken：抽帧时抛 FFmpegError 的视频主体。
    """

    def __init__(self, scenes, op_lengths=None, *, broken=(), broken_audio=(), bad_image=False):
        self.scenes = scenes
        self.op_lengths = op_lengths or {}
        self.broken = set(broken)
        self.broken_audio = set(broken_audio)
        self.bad_image = bad_image
        self.windows: list[tuple[str, float, float, float]] = []
        self.audio_reads: list[tuple[str, float]] = []

    def image_hash(self, path, *, ffmpeg_path):
        if self.bad_image:
            raise FFmpegError("ffmpeg 执行失败（退出码 1）\nInvalid data found")
        return {"op": _OP_HASH, "ed": _ED_HASH}[path.stem.casefold()]

    def sample_hashes(self, video, start, end, fps, *, ffmpeg_path):
        stem = Path(video).stem
        self.windows.append((stem, start, end, fps))
        if stem in self.broken:
            raise FFmpegError("ffmpeg 执行失败（退出码 1）\nmoov atom not found")
        frames = []
        i = 0
        while (t := round(start + i / fps, 3)) < end - 1e-9:
            value = next(
                (h for lo, hi, h in self.scenes.get(stem, []) if lo - 1e-9 <= t < hi),
                _OTHER_HASH,
            )
            frames.append((t, value))
            i += 1
        return frames

    def read_pcm(self, video, start, duration, *, ffmpeg_path):
        stem = Path(video).stem
        self.audio_reads.append((stem, start))
        if stem in self.broken_audio:
            raise FFmpegError("ffmpeg 执行失败（退出码 1）\naudio broken")
        rate = av_match.SAMPLE_RATE
        total = round(duration * rate)
        shared = np.random.default_rng(0).normal(0, 3000, total)
        own = np.random.default_rng(int(stem[-2:]) + 100).normal(0, 3000, total)
        cut = round(self.op_lengths.get(stem, 0.0) * rate)
        samples = np.concatenate([shared[:cut], own[cut:]])
        return samples.astype("<i2").tobytes()

    def install(self, monkeypatch):
        monkeypatch.setattr(av_match, "image_hash", self.image_hash)
        monkeypatch.setattr(av_match, "sample_hashes", self.sample_hashes)
        monkeypatch.setattr(av_match, "read_pcm", self.read_pcm)
        return self


def _with_screenshots(season: Path, *sides: str) -> Path:
    for side in sides:
        _touch(season / f"{side}.png")
    return season


def _scenes(op_start: float, ed_start: float, *, op_hash=_OP_HASH):
    return [(op_start, op_start + 3.0, op_hash), (ed_start, ed_start + 3.0, _ED_HASH)]


async def test_plan_locates_both_sides_by_screenshot_and_audio(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch, capsys
):
    cfg = _project(tmp_path)
    season = _with_screenshots(
        _season(tmp_path, golden_srt_path, ["Show E01", "Show E02", "Show E03"]), "op", "ed"
    )
    fake = _FakeAv(
        {
            "Show E01": _scenes(12.0, 1330.0),
            "Show E02": _scenes(40.5, 1325.2),
            "Show E03": _scenes(100.3, 1340.0),
        },
        {"Show E01": 90.0, "Show E02": 90.0, "Show E03": 90.0},
    ).install(monkeypatch)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    by_number = {item.number: item.ranges for item in plan.items}
    assert sorted(by_number) == [1, 2, 3]
    for number, start, ed_start in ((1, 12.0, 1330.0), (2, 40.5, 1325.2), (3, 100.3, 1340.0)):
        ranges = by_number[number]
        assert (ranges.op_source, ranges.ed_source) == ("av", "av")
        assert (ranges.op_distance, ranges.ed_distance) == (0, 0)
        assert ranges.op_range[0] == start
        assert ranges.op_range[1] == pytest.approx(start + 90.0, abs=0.3)
        assert ranges.ed_range == (ed_start, 1420.0)
        assert ranges.op_length_estimated is False
        assert ranges.write is not None
        assert ranges.write[1] == (ed_start, 1420.0)
    # 参考集是距离最小、集号最小的 E01，读了它和另外两集各一次
    assert [stem for stem, _ in fake.audio_reads] == ["Show E01", "Show E02", "Show E03"]
    printed = capsys.readouterr().out
    assert "正在匹配画面 1/3" in printed
    assert "正在匹配画面 3/3" in printed


async def test_plan_searches_only_the_first_five_and_last_three_minutes(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    season = _with_screenshots(_season(tmp_path, golden_srt_path, ["Show E01"]), "op", "ed")
    fake = _FakeAv({"Show E01": _scenes(12.0, 1330.0)}).install(monkeypatch)

    await build_import_plan(cfg, season, provider_factory=_no_llm)

    coarse = [(start, end) for _, start, end, fps in fake.windows if fps == av_match.COARSE_FPS]
    assert coarse == [(0.0, 300.0), (1240.0, 1420.0)]


async def test_plan_clamps_search_windows_to_a_short_video(
    tmp_path, golden_srt_path, monkeypatch
):
    _probe(monkeypatch, 100.0)
    cfg = _project(tmp_path)
    season = _with_screenshots(_season(tmp_path, golden_srt_path, ["Show E01"]), "op", "ed")
    fake = _FakeAv({"Show E01": _scenes(5.0, 60.0)}).install(monkeypatch)

    await build_import_plan(cfg, season, provider_factory=_no_llm)

    coarse = [(start, end) for _, start, end, fps in fake.windows if fps == av_match.COARSE_FPS]
    assert coarse == [(0.0, 100.0), (0.0, 100.0)]


async def test_plan_uses_subtitles_for_the_side_without_a_screenshot(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    season = _with_screenshots(_season(tmp_path, golden_srt_path, ["Show E01"]), "op")
    _FakeAv({"Show E01": _scenes(12.0, 1330.0)}).install(monkeypatch)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    ranges = plan.items[0].ranges
    assert (ranges.op_source, ranges.ed_source) == ("av", "subtitle")
    assert ranges.op_range == (12.0, 102.0)
    assert ranges.op_length_estimated is True
    assert ranges.ed_range == (1348.18, 1416.622)
    assert ranges.ed_distance is None
    assert ranges.write == ((12.0, 102.0), (1348.2, 1416.6))


async def test_plan_does_not_fall_back_to_subtitles_when_the_screenshot_does_not_match(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    season = _with_screenshots(
        _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"]), "op", "ed"
    )
    _FakeAv(
        {
            "Show E01": _scenes(12.0, 1330.0),
            "Show E02": _scenes(40.0, 1330.0, op_hash=_NEAR_MISS_HASH),
        },
        {"Show E01": 90.0, "Show E02": 90.0},
    ).install(monkeypatch)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    e02 = {item.number: item.ranges for item in plan.items}[2]
    assert e02.op_source == "av"
    assert e02.op_range is None  # 字幕推断能给出 153.486–224.681，但不许退回去
    assert e02.op_distance == 15
    assert e02.write is None
    assert "15/64" in e02.note
    assert "都不写" in e02.note


async def test_plan_estimates_ninety_seconds_with_a_single_matched_episode(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    season = _with_screenshots(_season(tmp_path, golden_srt_path, ["Show E01"]), "op", "ed")
    fake = _FakeAv({"Show E01": _scenes(12.0, 1330.0)}).install(monkeypatch)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    ranges = plan.items[0].ranges
    assert ranges.op_range == (12.0, 102.0)
    assert ranges.op_length_estimated is True
    assert ranges.write == ((12.0, 102.0), (1330.0, 1420.0))
    assert fake.audio_reads == []
    assert "OP 长度按 90 秒估计" in "\n".join(render_preview(plan))


async def test_plan_drops_an_outlier_op_length(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    stems = ["Show E01", "Show E02", "Show E03", "Show E04"]
    season = _with_screenshots(_season(tmp_path, golden_srt_path, stems), "op")
    _FakeAv(
        {stem: _scenes(10.0, 1330.0) for stem in stems},
        {"Show E01": 90.0, "Show E02": 90.0, "Show E03": 90.0, "Show E04": 60.0},
    ).install(monkeypatch)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    for item in plan.items:
        assert item.ranges.op_range[1] == pytest.approx(100.0, abs=0.3), item.number


async def test_plan_marks_only_the_broken_episode_side_unrecognised(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    season = _with_screenshots(
        _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"]), "op", "ed"
    )
    _FakeAv({"Show E01": _scenes(12.0, 1330.0)}, broken={"Show E02"}).install(monkeypatch)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    by_number = {item.number: item.ranges for item in plan.items}
    assert by_number[1].write == ((12.0, 102.0), (1330.0, 1420.0))
    assert by_number[2].op_range is None
    assert by_number[2].ed_range is None
    assert by_number[2].write is None
    assert "抽帧失败：ffmpeg 执行失败（退出码 1）" in by_number[2].note
    assert "moov" not in by_number[2].note  # 只放第一行


async def test_plan_marks_an_episode_whose_audio_cannot_be_read(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    stems = ["Show E01", "Show E02", "Show E03"]
    season = _with_screenshots(_season(tmp_path, golden_srt_path, stems), "op", "ed")
    _FakeAv(
        {stem: _scenes(10.0, 1330.0) for stem in stems},
        {"Show E01": 90.0, "Show E02": 90.0, "Show E03": 90.0},
        broken_audio={"Show E03"},
    ).install(monkeypatch)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    by_number = {item.number: item.ranges for item in plan.items}
    assert by_number[1].op_range[1] == pytest.approx(100.0, abs=0.3)
    assert by_number[2].op_range[1] == pytest.approx(100.0, abs=0.3)
    assert by_number[3].op_range is None
    assert "OP 音频读不出来" in by_number[3].note


async def test_plan_lets_a_misconfigured_ffmpeg_through_while_matching(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    season = _with_screenshots(_season(tmp_path, golden_srt_path, ["Show E01"]), "op")
    fake = _FakeAv({}).install(monkeypatch)

    def missing_binary(*args, **kwargs):
        raise FFmpegBinaryError("找不到 ffmpeg")

    monkeypatch.setattr(av_match, "sample_hashes", missing_binary)

    with pytest.raises(FFmpegBinaryError):
        await build_import_plan(cfg, season, provider_factory=_no_llm)
    assert fake.windows == []


async def test_plan_refuses_a_corrupted_screenshot_before_anything_else(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    season = _with_screenshots(_season(tmp_path, golden_srt_path, ["86 - 11"]), "op")
    fake = _FakeAv({}, bad_image=True).install(monkeypatch)

    with pytest.raises(ValueError, match="参考截图读不出来"):
        await build_import_plan(cfg, season, provider_factory=_no_llm)
    assert fake.windows == []


async def test_plan_with_unknown_duration_skips_matching_and_writes_nothing(
    tmp_path, golden_srt_path, monkeypatch
):
    _probe(monkeypatch, FFmpegError("moov atom not found"))
    cfg = _project(tmp_path)
    season = _with_screenshots(_season(tmp_path, golden_srt_path, ["Show E01"]), "ed")
    fake = _FakeAv({}).install(monkeypatch)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    ranges = plan.items[0].ranges
    assert fake.windows == []
    assert ranges.ed_source == "av"
    assert ranges.ed_range is None
    assert ranges.write is None
    assert "片长未知" in ranges.note


async def test_plan_does_not_match_episodes_with_manual_ranges(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch, capsys
):
    cfg = _project(tmp_path, "episodes:\n- number: 2\n  ed_range: [1300, 1420]\n")
    season = _with_screenshots(
        _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"]), "op", "ed"
    )
    fake = _FakeAv({"Show E01": _scenes(12.0, 1330.0)}).install(monkeypatch)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert {stem for stem, *_ in fake.windows} == {"Show E01"}
    e02 = {item.number: item.ranges for item in plan.items}[2]
    assert e02.ed_range == (1300.0, 1420.0)
    assert e02.op_source is None
    assert "已有手填区间" in e02.note
    assert "正在匹配画面 1/1" in capsys.readouterr().out
```

追加到 `tests/test_cli.py` 末尾（与前文空两行）：

```python
def test_import_refuses_a_corrupted_screenshot_and_writes_nothing(
    tmp_path, golden_srt_path, import_env, monkeypatch
):
    root = _import_project(tmp_path)
    season = _import_season(tmp_path, golden_srt_path, ["Show E01"])
    (season / "op.png").write_bytes(b"not an image")
    before = (root / "project.yaml").read_text(encoding="utf-8")

    def broken_image(path, *, ffmpeg_path):
        raise FFmpegError("ffmpeg 执行失败（退出码 1）\nInvalid data found when processing input")

    monkeypatch.setattr("tenmin.ingest.av_match.image_hash", broken_image)

    result = runner.invoke(app, _import_args(tmp_path, season, "--yes"))

    assert result.exit_code == 1
    assert "参考截图读不出来" in out(result)
    assert "op.png" in out(result)
    assert import_env == []
    assert (root / "project.yaml").read_text(encoding="utf-8") == before
    assert list((root / "srt").iterdir()) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_importer.py tests/test_cli.py -q`

Expected: FAIL — 新增的 `test_plan_*` 断言失败（来源仍是 `subtitle`、没有「正在匹配画面」输出），`test_import_refuses_a_corrupted_screenshot_and_writes_nothing` 退出码是 0

- [ ] **Step 3: Write minimal implementation**

最后一处替换把原来「分拣 + 逐集 infer_ranges」的循环拆成「先收集 candidates → `match_credits` → 再逐集 infer_ranges」，并在 `build_import_plan` 之后定义搜索窗常量与新函数。

在 `src/tenmin/importer.py` 里把下面这段（全文唯一）：

```python
from tenmin.pipeline import _source_duration, register_episode
from tenmin.script.llm import LLMProvider
```

替换为：

```python
from tenmin.pipeline import _source_duration, register_episode
from tenmin.render.ffmpeg import FFmpegBinaryError, FFmpegError
from tenmin.script.llm import LLMProvider
```

在 `src/tenmin/importer.py` 里把下面这段（全文唯一）：

```python
    （预填条目补齐来源、保留已有区间）。`FFmpegBinaryError` 原样外抛。
    """
    scan = scan_directory(video_dir, srt_dir)
```

替换为：

```python
    （预填条目补齐来源、保留已有区间）。`FFmpegBinaryError` 原样外抛。

    目录里有 op / ed 参考截图时，先算截图的 dHash（读不出整个报错，在 LLM 兜底之前），
    分拣完再对可导入集做一遍 `match_credits`，结果按侧交给 infer_ranges。
    """
    scan = scan_directory(video_dir, srt_dir)
    references = reference_hashes(cfg, scan.references)
```

在 `src/tenmin/importer.py` 里把下面这段（全文唯一）：

```python
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
```

替换为：

```python
    numbered.extend(await _llm_numbers(pending, provider_factory, plan))

    counts = Counter(number for number, _ in numbered)
    by_number = {episode.number: episode for episode in cfg.episodes}
    candidates: list[tuple[int, Pair, EpisodeConfig | None]] = []
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
        candidates.append((number, pair, existing))

    matched = match_credits(cfg, candidates, references)
    for number, pair, existing in candidates:
        op_av, ed_av = matched.get(number, (None, None))
        try:
            ranges = infer_ranges(
                cfg,
                number=number,
                srt=pair.srt,
                video=pair.video,
                existing=existing,
                op_av=op_av,
                ed_av=ed_av,
            )
        except (ValueError, OSError) as error:
            plan.failed.append(Skipped(pair.video, f"字幕读不出来：{_describe(error)}"))
            continue
        plan.items.append(
            ImportItem(number=number, pair=pair, prefilled=existing is not None, ranges=ranges)
        )
    return plan


# 起点搜索窗（spec：OP 只看前 5 分钟，ED 只看最后 3 分钟；片长更短时夹到片长内）。
OP_SEARCH_SECONDS = 300.0
ED_SEARCH_SECONDS = 180.0


def reference_hashes(cfg: ProjectConfig, references: dict[str, Path]) -> dict[str, int]:
    """参考截图 → dHash。任何一张读不出就抛 ValueError（整个命令报错，什么都不写）。

    `FFmpegBinaryError`（ffmpeg 配错）原样外抛，不包装成「截图坏了」。
    """
    hashes: dict[str, int] = {}
    for side, path in references.items():
        try:
            hashes[side] = av_match.image_hash(path, ffmpeg_path=cfg.render.ffmpeg_path)
        except FFmpegBinaryError:
            raise
        except FFmpegError as error:
            raise ValueError(
                f"参考截图读不出来：{path}（{_first_line(error)}）。换一张或删掉它再导入"
            ) from error
    return hashes


def _first_line(error: BaseException) -> str:
    """FFmpegError 的消息带整段 stderr 尾部，预览里只放第一行。"""
    return _describe(error).splitlines()[0]


def _locate_side(
    cfg: ProjectConfig, video: Path, ref: int, *, label: str, start: float, end: float
) -> tuple[av_match.Match | None, AvSide | None]:
    """在一个搜索窗里找一侧的起点。成功返回 (Match, None)；失败返回 (None, 未识别的 AvSide)。"""
    try:
        match = av_match.locate(
            video, ref, start=start, end=end, ffmpeg_path=cfg.render.ffmpeg_path
        )
    except FFmpegBinaryError:
        raise
    except FFmpegError as error:
        return None, AvSide(range=None, reason=f"{label} 抽帧失败：{_first_line(error)}")
    if match is None:
        return None, AvSide(range=None, reason=f"{label} 搜索窗内没有画面")
    if match.distance > av_match.MAX_DISTANCE:
        return None, AvSide(
            range=None,
            distance=match.distance,
            reason=(
                f"{label} 画面对不上（最佳距离 {match.distance}/{av_match.HASH_BITS}，"
                f"超过 {av_match.MAX_DISTANCE}/{av_match.HASH_BITS}）"
            ),
        )
    return match, None


def _envelope(cfg: ProjectConfig, video: Path, start: float):
    return av_match.audio_envelope(
        av_match.read_pcm(
            video, start, av_match.REFERENCE_SECONDS, ffmpeg_path=cfg.render.ffmpeg_path
        )
    )


def _op_sides(
    cfg: ProjectConfig, starts: dict[int, tuple[Path, float, av_match.Match]]
) -> dict[int, AvSide]:
    """第二遍：OP 起点匹配成功的集做跨集音频对齐，求统一的 OP 长度 L。

    参考集取匹配距离最小的一集（同距离取集号小的），读它起点后 REFERENCE_SECONDS 秒；
    其余集各自从起点读同样长度，用 `op_length` 测长度，`median_length` 剔除离群值后取
    中位数。只有一集成功、或一个长度都没测出来时，L 按 DEFAULT_OP_LENGTH 估计。某集读
    音频失败，只有那一集的 OP 判未识别。
    """
    sides: dict[int, AvSide] = {}

    def side(number: int, length: float, *, estimated: bool) -> AvSide:
        _, duration, match = starts[number]
        end = min(match.time + length, duration)
        return AvSide(range=(match.time, end), distance=match.distance, estimated=estimated)

    if len(starts) == 1:
        (number,) = starts
        return {number: side(number, av_match.DEFAULT_OP_LENGTH, estimated=True)}

    ref_number = min(starts, key=lambda number: (starts[number][2].distance, number))
    ref_video, _, ref_match = starts[ref_number]
    try:
        ref_env = _envelope(cfg, ref_video, ref_match.time)
    except FFmpegBinaryError:
        raise
    except FFmpegError as error:
        reason = f"OP 参考集 E{ref_number:02d} 的音频读不出来：{_first_line(error)}"
        return {
            number: AvSide(range=None, distance=starts[number][2].distance, reason=reason)
            for number in starts
        }

    lengths: list[float] = []
    usable = [ref_number]
    for number, (video, _, match) in starts.items():
        if number == ref_number:
            continue
        try:
            env = _envelope(cfg, video, match.time)
        except FFmpegBinaryError:
            raise
        except FFmpegError as error:
            sides[number] = AvSide(
                range=None,
                distance=match.distance,
                reason=f"OP 音频读不出来：{_first_line(error)}",
            )
            continue
        usable.append(number)
        length = av_match.op_length(ref_env, env)
        if length is not None:
            lengths.append(length)

    if not lengths:
        length, estimated = av_match.DEFAULT_OP_LENGTH, True
    else:
        median = av_match.median_length(lengths)
        if median is None:
            shown = "、".join(f"{value:.1f}" for value in sorted(lengths))
            reason = f"各集 OP 长度相差太大（{shown} 秒），定不了终点"
            for number in usable:
                sides[number] = AvSide(
                    range=None, distance=starts[number][2].distance, reason=reason
                )
            return sides
        length, estimated = median, False
    for number in usable:
        sides[number] = side(number, length, estimated=estimated)
    return sides


def match_credits(
    cfg: ProjectConfig,
    candidates: Sequence[tuple[int, Pair, EpisodeConfig | None]],
    references: dict[str, int],
) -> dict[int, tuple[AvSide | None, AvSide | None]]:
    """有截图的一侧用画面+音频定区间。返回 {集号: (OP 结果, ED 结果)}，没截图的一侧是 None。

    第一遍逐集找起点（OP 在前 OP_SEARCH_SECONDS 秒、ED 在最后 ED_SEARCH_SECONDS 秒，
    窗口夹到片长内），ED 终点一律取片长；第二遍见 `_op_sides`。已有手填区间的集不匹配
    （infer_ranges 反正原样保留）。片长未知的集不匹配，有截图的一侧直接判未识别。
    每集开始前打一行「正在匹配画面 N/M」。
    """
    if not references:
        return {}
    op_ref = references.get("op")
    ed_ref = references.get("ed")
    eligible = [
        (number, pair)
        for number, pair, existing in candidates
        if existing is None or (existing.op_range is None and existing.ed_range is None)
    ]
    results: dict[int, tuple[AvSide | None, AvSide | None]] = {}
    op_starts: dict[int, tuple[Path, float, av_match.Match]] = {}
    for index, (number, pair) in enumerate(eligible, start=1):
        print(f"正在匹配画面 {index}/{len(eligible)}")
        video = Path(pair.video).resolve()
        duration = _source_duration(cfg, EpisodeConfig(number=number, video=video))
        if duration is None:
            unknown = AvSide(range=None, reason="片长未知，没做画面匹配")
            results[number] = (
                unknown if op_ref is not None else None,
                unknown if ed_ref is not None else None,
            )
            continue
        op_side: AvSide | None = None
        ed_side: AvSide | None = None
        if op_ref is not None:
            match, op_side = _locate_side(
                cfg, video, op_ref, label="OP", start=0.0, end=min(OP_SEARCH_SECONDS, duration)
            )
            if match is not None:
                op_starts[number] = (video, duration, match)
        if ed_ref is not None:
            match, ed_side = _locate_side(
                cfg,
                video,
                ed_ref,
                label="ED",
                start=max(0.0, duration - ED_SEARCH_SECONDS),
                end=duration,
            )
            if match is not None:
                ed_side = AvSide(range=(match.time, duration), distance=match.distance)
        results[number] = (op_side, ed_side)
    if op_starts:
        for number, side in _op_sides(cfg, op_starts).items():
            results[number] = (side, results[number][1])
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_importer.py tests/test_cli.py -q`

Expected: PASS，无 failed

- [ ] **Step 5: 全量测试**

Run: `uv run pytest tests/ -q`

Expected: PASS，无 failed（render/asr/ocr marker 用例显示为 skipped）

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/tenmin/importer.py tests/test_importer.py tests/test_cli.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
rtk git add src/tenmin/importer.py tests/test_importer.py tests/test_cli.py && rtk git commit -m "feat: 导入时用截图与音频定位 OP/ED"
```

---

### Task 9: `av` marker 与真实素材测试

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Modify: `tests/test_marker_gate.py`
- Modify: `tests/test_av_match.py`

- [ ] **Step 1: Write the failing test**

真实素材测试要的目录：同一部番的几集视频 + 配对 SRT + `op.png`（OP 里一个有辨识度的画面，**别用纯黑帧或片头厂牌**——saijo 的 MBS 厂牌在 OP 之前就出现，跟 OP 音频对不上）+ `ed.png`。

追加到 `tests/test_marker_gate.py` 末尾（与前文空两行）：

```python
def test_the_av_marker_is_gated_until_named():
    """av 用例要对一整个真实素材目录逐集抽帧、读音频，一次普通 pytest 里不许意外跑到。"""
    from . import conftest

    default = _FakeItem("av")
    conftest.pytest_collection_modifyitems(_FakeConfig(None), [default])
    assert default.marks

    named = _FakeItem("av")
    conftest.pytest_collection_modifyitems(_FakeConfig("av"), [named])
    assert named.marks == []
```

在 `tests/test_av_match.py` 里把下面这段（全文唯一）：

```python
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tenmin.ingest import av_match
from tenmin.render import ffmpeg
from tenmin.render.ffmpeg import FFmpegError
```

替换为：

```python
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tenmin.ingest import av_match
from tenmin.render import ffmpeg
from tenmin.render.ffmpeg import FFmpegError
```

追加到 `tests/test_av_match.py` 末尾（与前文空两行）：

```python
# --- 真实素材（av marker，默认跳过） ----------------------------------------------------


@pytest.mark.av
@pytest.mark.skipif(
    not os.environ.get("TENMIN_AV_SAMPLE_DIR"), reason="需要 TENMIN_AV_SAMPLE_DIR 指向素材目录"
)
async def test_real_season_locates_op_and_ed_by_screenshot_and_audio(tmp_path):
    from tenmin.config import load_project
    from tenmin.importer import build_import_plan

    sample = Path(os.environ["TENMIN_AV_SAMPLE_DIR"])
    root = tmp_path / "work" / "sample"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: sample\nslug: sample\nepisodes: []\n", encoding="utf-8"
    )
    cfg = load_project(root / "project.yaml")

    def no_llm():
        raise RuntimeError("真实素材测试不调 LLM")

    plan = await build_import_plan(cfg, sample, provider_factory=no_llm)

    assert plan.items, "素材目录里没有可导入的集"
    for item in plan.items:
        ranges = item.ranges
        assert (ranges.op_source, ranges.ed_source) == ("av", "av"), item.number
        assert ranges.op_distance is not None and ranges.op_distance <= av_match.MAX_DISTANCE
        assert ranges.ed_distance is not None and ranges.ed_distance <= av_match.MAX_DISTANCE
        assert ranges.op_range is not None and ranges.ed_range is not None
        assert 60.0 <= ranges.op_range[1] - ranges.op_range[0] <= 120.0
        assert ranges.ed_range[1] == ranges.duration
        assert ranges.write is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_marker_gate.py -q`

Expected: FAIL — `test_the_av_marker_is_gated_until_named` 的 `assert default.marks` 不成立

- [ ] **Step 3: Write minimal implementation**

在 `pyproject.toml` 里把下面这段（全文唯一）：

```toml
# 五个 marker 各自已经有更准的门，而且门都在代码里、不在这份配置里：
```

替换为：

```toml
# 六个 marker 各自已经有更准的门，而且门都在代码里、不在这份配置里：
```

在 `pyproject.toml` 里把下面这段（全文唯一）：

```toml
#   - ocr：同一个 hook。这批用例要 macOS + ocr extra + 一个真实的硬字幕片源，整集识别
#     要跑上几分钟。
```

替换为：

```toml
#   - ocr：同一个 hook。这批用例要 macOS + ocr extra + 一个真实的硬字幕片源，整集识别
#     要跑上几分钟。
#   - av：同一个 hook，外加「有没有 TENMIN_AV_SAMPLE_DIR」的 skipif。这批用例要对一整
#     个真实素材目录逐集抽帧、读音频。
```

在 `pyproject.toml` 里把下面这段（全文唯一）：

```toml
# 换句话说 addopts deselect 对 render / asr / ocr 是重复
```

替换为：

```toml
# 换句话说 addopts deselect 对 render / asr / ocr / av 是重复
```

在 `pyproject.toml` 里把下面这段（全文唯一）：

```toml
    "ocr: 需要 macOS、装了 ocr extra 与真实的硬字幕片源，默认跳过（跑法：TENMIN_OCR_SAMPLE_VIDEO=<片源> uv run pytest -m ocr）",
```

替换为：

```toml
    "ocr: 需要 macOS、装了 ocr extra 与真实的硬字幕片源，默认跳过（跑法：TENMIN_OCR_SAMPLE_VIDEO=<片源> uv run pytest -m ocr）",
    "av: 需要一个放了视频、SRT 与 op/ed 截图的真实素材目录，默认跳过（跑法：TENMIN_AV_SAMPLE_DIR=<目录> uv run pytest -m av）",
```

在 `tests/conftest.py` 里把下面这段（全文唯一）：

```python
    "ocr": "需要 macOS、ocr extra 与真实硬字幕片源，跑法：uv run pytest -m ocr",
}
```

替换为：

```python
    "ocr": "需要 macOS、ocr extra 与真实硬字幕片源，跑法：uv run pytest -m ocr",
    "av": "需要真实素材目录，跑法：TENMIN_AV_SAMPLE_DIR=<目录> uv run pytest -m av",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_marker_gate.py tests/test_av_match.py -q`

Expected: PASS，`av` 用例 skipped，且没有 `PytestUnknownMarkWarning: Unknown pytest.mark.av`

- [ ] **Step 5: 真实素材（可选，需要素材目录）**

Run: `TENMIN_AV_SAMPLE_DIR=<saijo 素材目录> uv run pytest tests/test_av_match.py -m av -q`

Expected: PASS（`1 passed`，saijo E02+E12 两集约 10 秒）。没有素材就跳过这一步

- [ ] **Step 6: Commit**

```bash
rtk git add pyproject.toml tests/conftest.py tests/test_marker_gate.py tests/test_av_match.py && rtk git commit -m "test: 增加 av marker 与真实素材测试"
```

---

### Task 10: AGENTS.md 与全量验证

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: 更新 AGENTS.md**

五处改动：`render/` 条目补 `run_pcm`；`importer.py` 条目末尾补截图流程并在其后插入 `ingest/av_match.py` 新条目；测试文件数；marker 个数；marker 列表补 `av`。

在 `AGENTS.md` 里把下面这段（全文唯一）：

```markdown
OCR 用的两个是例外的二进制路径：`probe_video_size`（首条视频流宽高，给 OCR 算精确像素裁剪与每帧字节数）和 `run_raw_frames`
```

替换为：

```markdown
OCR 与音画匹配用的三个是例外的二进制路径：`probe_video_size`（首条视频流宽高，给 OCR 算精确像素裁剪与每帧字节数）、`run_pcm`（s16le 单声道 PCM 走 stdout 一次读回内存，给 `ingest/av_match.py`；非零退出报 `FFmpegError`）和 `run_raw_frames`
```

在 `AGENTS.md` 里把下面这段（全文唯一）：

```markdown
补齐来源的预填集写区间会改 EPISODE 切片，预览里提示会重跑。
```

替换为：

```markdown
补齐来源的预填集写区间会改 EPISODE 切片，预览里提示会重跑。

  **参考截图**（视频目录下的 `op.*` / `ed.*`，`.png`/`.jpg`/`.jpeg` 不分大小写，进 `Scan.references`、不算未配对；同一侧多张报错）：有截图的一侧改走画面+音频（`match_credits`，见下面 `ingest/av_match.py`），没截图的一侧照旧字幕推断，**有截图但对不上判未识别、不退回字幕推断**；写入规则不变（两侧都成功才写）。截图读不出在 LLM 兜底之前整个报错（`ValueError`），某集抽帧/读音频的 `FFmpegError` 只让那一侧未识别，`FFmpegBinaryError` 外抛。搜索窗 OP 前 300 秒、ED 后 180 秒（夹到片长），ED 终点 = 片长，片长未知不匹配。OP 长度 L：取距离最小的一集当参考读 150 秒音频，其余集测长度，剔除离中位数 >10 秒的再取中位数；只有一集或一个都测不出时按 90 秒估计（`op_length_estimated`）；全被剔除（各集长度对不上）时 OP 全部未识别。已有手填区间的集不匹配。预览区间后标「画面+音频」/「字幕推断」与距离（`5/64`），匹配前逐集打印「正在匹配画面 N/M」。
- `src/tenmin/ingest/av_match.py`：截图与音频定位 OP/ED，不读 project.yaml。纯函数 `dhash`（9×8 灰度 → 64 位）、`hamming`、`best_frame`（同距离取最早）、`audio_envelope`（8 kHz PCM → 每秒 10 帧 × 16 个对数频带的 log 能量）、`op_length`（各自减频带均值去掉频谱倾斜 → ±1 秒内找最佳错位 → 1 秒窗 Pearson 相关，先对上过、再连续 2 秒低于 0.5 判结束）、`median_length`；IO `sample_hashes`（`fps` + `scale=9:8:flags=area` + `showinfo`，`-map 0:v:0`，时间 = 窗口起点 + 真实 pts）、`image_hash`（截图走同一条缩放链）、`read_pcm`、`locate`（每秒 2 帧粗搜 + 最佳帧 ±1 秒每秒 10 帧细化）。阈值（`MAX_DISTANCE`=12 等）是模块级常量、不进 config。importer 的测试 monkeypatch `image_hash` / `sample_hashes` / `read_pcm`，所以 `locate` 必须经模块全局名调它们。实测 saijo E02/E12：OP 长度两向都是 90.1 秒，对上时窗口相关 ≥0.93、对不上时 ≤0.49；两集合计约 11 秒。参考截图别用纯黑帧或 OP 之前的片头厂牌（saijo 的 MBS 厂牌画面对得上、音频对不上）。
```

在 `AGENTS.md` 里把下面这段（全文唯一）：

```markdown
共 53 个 Python 文件
```

替换为：

```markdown
共 54 个 Python 文件
```

在 `AGENTS.md` 里把下面这段（全文唯一）：

```markdown
pytest markers（`pyproject.toml` 里是**五个**）：
```

替换为：

```markdown
pytest markers（`pyproject.toml` 里是**六个**）：
```

在 `AGENTS.md` 里把下面这段（全文唯一）：

```markdown
- `ocr`：需要 macOS + `uv sync --extra ocr` + 一个真实的硬字幕片源，跑法：`TENMIN_OCR_SAMPLE_VIDEO=<片源路径> uv run pytest -m ocr`。整集识别要跑三四分钟。
```

替换为：

```markdown
- `ocr`：需要 macOS + `uv sync --extra ocr` + 一个真实的硬字幕片源，跑法：`TENMIN_OCR_SAMPLE_VIDEO=<片源路径> uv run pytest -m ocr`。整集识别要跑三四分钟。
- `av`：需要一个放了视频、SRT 与 `op.png`/`ed.png` 的真实素材目录（比如 saijo），跑法：`TENMIN_AV_SAMPLE_DIR=<目录> uv run pytest -m av`。没设环境变量时自己 skip。
```

- [ ] **Step 2: 核对测试文件数**

Run: `/bin/ls tests/*.py | wc -l`

Expected: `54`

- [ ] **Step 3: 全量测试**

Run: `uv run pytest tests/ -q`

Expected: PASS，无 failed

- [ ] **Step 4: Lint 改动过的文件**

Run: `uv run ruff check src/tenmin/ingest/av_match.py src/tenmin/importer.py src/tenmin/render/ffmpeg.py tests/test_av_match.py tests/test_importer.py tests/test_cli.py tests/test_marker_gate.py tests/conftest.py`

Expected: `All checks passed!`

- [ ] **Step 5: 手工冒烟（可选，需真实素材）**

Run: `uv run tenmin import <slug> "<放了 op.png/ed.png 的季目录>" --dry-run`

Expected: 先逐集打印「正在匹配画面 N/M」，预览里每集 `OP：…（画面+音频，d/64）  ED：…（画面+音频，d/64）`；`git status` 里 `work/` 下没有任何改动

- [ ] **Step 6: Commit**

```bash
rtk git add AGENTS.md && rtk git commit -m "docs: AGENTS.md 补截图与音频定位 OP/ED 说明"
```

---

## Self-Review（对照 spec 逐条核对）

| Spec 要求 | 落点 |
| --- | --- |
| 截图定起点：9×8 灰度 64 位 dHash；窗内每秒 2 帧粗搜、最佳帧 ±1 秒 10 fps 细化到 0.1 秒 | Task 2（`dhash` / `best_frame`）、Task 5（`sample_hashes` / `locate`，lavfi 实测定位误差 ≤0.1 秒） |
| 最佳距离 >12/64 判未识别，预览显示距离 | Task 2（`MAX_DISTANCE`）、Task 7（`_format_side`）、Task 8（`test_plan_does_not_fall_back...` 断言 `15/64`） |
| OP 终点：距离最小的一集当参考，起点后 150 秒、单声道 8 kHz、每秒 10 帧对数频带能量；其他集连续 2 秒低于阈值处为长度；剔除离中位数 >10 秒再取中位数；单集 90 秒并标注 | Task 3（`audio_envelope` / `op_length` / `median_length`）、Task 8（`_op_sides`、离群/单集/读音频失败测试） |
| ED 终点 = ffprobe 片长；片长未知 ED 失败、不写 | Task 8（`match_credits` 的 `(match.time, duration)` 与「片长未知，没做画面匹配」） |
| 搜索窗 OP 0–300 秒、ED 最后 180 秒，窗内没匹配不扩大，短片夹到片长 | Task 8（`test_plan_searches_only...`、`test_plan_clamps_search_windows...`） |
| 截图 `op.*`/`ed.*`（png/jpg/jpeg 不分大小写）自动发现、不算未配对 | Task 6 |
| 按侧独立选来源；有截图对不上不退回字幕；写入规则两个都成功才写 | Task 7（`infer_ranges`）、Task 8 |
| `RangePreview` 的 `op_source`/`ed_source`、`op_distance`/`ed_distance`、`op_length_estimated`；预览标「画面+音频」/「字幕推断」与距离 | Task 7 |
| `build_import_plan` 两遍；`infer_ranges` 按侧选来源 | Task 8、Task 7 |
| 截图读不了：整个命令报错、什么都不写 | Task 8（`reference_hashes`，importer 与 CLI 两层测试） |
| 某集 `FFmpegError` 只让该侧未识别并写原因；`FFmpegBinaryError` 外抛 | Task 8（`_locate_side`、`_op_sides`，对应测试） |
| 只有截图时才做画面匹配；逐集打印「正在匹配画面 N/M」 | Task 8（`match_credits` 空 references 直接返回 `{}`；capsys 断言） |
| 新模块不读 project.yaml、不碰登记；纯函数与 IO 分开 | Task 2/3/5 |
| `numpy` 提为主依赖，不引入 OpenCV / fpcalc | Task 1 |
| `ffmpeg.run_pcm`（s16le、单声道、8 kHz） | Task 4 |
| 测试：纯函数合成数据；lavfi IO 测试标 `render`；importer monkeypatch IO；`av` marker + `TENMIN_AV_SAMPLE_DIR`；marker 与 AGENTS.md 五→六 | Task 2–5、Task 8、Task 9、Task 10 |

## 实现中替 spec 拍板的空白

1. **同一侧有多张截图**（如 `op.png` 和 `op.jpg`）：`scan_directory` 直接报 `ValueError`，不猜哪张。截图只从视频目录找，`--srt-dir` 不影响。
2. **截图读不出的报错类型**是 `ValueError`（消息「参考截图读不出来：<路径>（ffmpeg 报错第一行）」），在 LLM 兜底之前就抛，CLI 现有的 `PIPELINE_ERRORS` 分支直接接住；`FFmpegBinaryError` 不包装、原样外抛。
3. **已有手填区间的集不做画面匹配**（反正原样保留，省时间），也不计入「正在匹配画面 N/M」的 M；它和项目级默认生效的集一样，预览不标来源（`op_source`/`ed_source` 为 None）。项目级默认生效的集仍会做匹配（是否生效要等 `build_track` 才知道），但不写。
4. **片长未知的集不做画面匹配**：有截图的一侧直接判未识别（「片长未知，没做画面匹配」），既然 ED 写不了、两侧又必须一起写，匹配 OP 只是白花时间。
5. **OP 起点成功不止一集、但一个长度都测不出**（`op_length` 全是 None）时，与单集同样按 90 秒估计并标注；**测出了长度但全部互为离群**（比如只有 60 与 90 两个）时，OP 全部判未识别（「各集 OP 长度相差太大」），不拿 90 秒糊过去。
6. **参考集音频读不出**：所有 OP 都判未识别；**非参考集音频读不出**：只有那一集的 OP 判未识别，其余集照常用 L。参考集与同距离并列时取集号最小的一集；参考集自身不贡献长度。
7. **相似度的定义**（spec 只说「相似度」）：两集各自减去频带均值后，1 秒窗口内 16 个频带 × 10 帧拉平做 Pearson 相关，阈值 0.5；比较前先在 ±1 秒内找最佳错位吸收起点量化误差；必须先对上过（至少一个窗口过阈值）才开始找下降点。OP 终点 = 低值串的首个窗口 + 半个窗口。阈值与窗口长度用 saijo E02/E12 实测标定（OP 内 ≥0.93、OP 后 ≤0.49）。
8. **起点同距离取最早帧**（粗搜与细化都是），对应静止开场镜头真正的第一帧；细化候选里也放进粗搜最佳帧，保证细化不会比粗搜更差。
9. **OP 终点夹到片长内**（`min(起点 + L, 片长)`）；**ED 终点取整后可能比 ffprobe 片长多出不到 0.05 秒**（`round_range` 是四舍五入，例如 1420.074667 → 1420.1），下游按区间求交，无害，沿用现有取整规则不特殊处理。
10. **阈值都是 av_match 的模块级常量**、不进 `config`：它们是算法的判定边界而不是「这部番想要什么」的创作旋钮（AGENTS.md 的分家判据）。`cli.py` 不用改。
11. **lavfi 测试素材用旋转的 testsrc2**：静止的 testsrc2 只有一小块在动，缩到 9×8 后前后几秒哈希完全一样（实测 3.0 秒的截图被定到 1.0 秒）。PNG 截图与 yuv420p 视频帧走不同的转灰度路径，实测距离 3，所以 lavfi 测试只断言距离 ≤12。
