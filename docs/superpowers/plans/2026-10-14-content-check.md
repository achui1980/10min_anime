# 最终成片内容检查 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增独立本地视频内容检查命令，并在用户开启时于每集最终成片之后生成同口径的截图、评级和候选报告。

**Architecture:** `content_check/` 提供同步共享服务，负责真实 PTS 抽帧、固定 MobileNet 模型、原始分数缓存和 HTML 报告。CLI 与按集纵向管线只是入口；管线单独捕获可预期检查错误并继续下一集，检查服务自己不决定批处理退出策略。推理缓存与阈值报告分离，最后原子发布 `results.json` 标记成功。

**Tech Stack:** Python >=3.12、Pydantic 2、Typer、FFmpeg/ffprobe、可选 ONNX Runtime / NumPy / Pillow / huggingface-hub、pytest、静态 HTML/JavaScript。

**Spec:** `docs/superpowers/specs/2026-10-14-content-check-design.md`（用户已批准）。

## Global Constraints

- 模型 repo `deepghs/anime_dbrating`，revision `7af21db648acdeb74f5c334abda9dd7403407b3c`，仅 `mobilenetv3_large_100_v0_ls0.2`，CPUExecutionProvider，4 线程。
- 已实测权重 SHA-256：`c7d6fd0cd71c48616fa87fe2bca87f8e1775e9a3b96d12797e2144fca3543362`；下载文件名 `mobilenetv3_large_100_v0_ls0.2/model.onnx`、同目录 `meta.json`。
- 配置默认 `enabled=false`、`sample_interval=1.0`、`sensitive_threshold=0.5`、`adult_threshold=0.5`；间隔有限且 >0，阈值有限且在 [0,1]。
- 四类 `general/sensitive/questionable/explicit` 是图片评级分数，不是校准色情概率或严重程度。
- 候选规则 `sensitive >= sensitive_threshold OR questionable + explicit >= adult_threshold`，排序 `max(sensitive, questionable + explicit)`；相邻候选点 <=1.5×间隔归组，不宣称精确边界。
- 项目只检查 `07_render/E{NN}.mp4`，输出 `08_check/E{NN}/{results.json,report.html,frames/}`；保留所有存量路径。
- 首帧也采样；时间来自 FFmpeg showinfo 的真实 PTS；截图宽 640，保持显示方向和显示比例；RGB / bilinear / CHW / [-1,1]，动态空间维度 384×384；不重复 softmax。
- 命中只报告并继续。预期执行失败警告并继续后续集；不得捕获取消、KeyboardInterrupt 或未知编程错误伪装成功。
- 独立命令一个视频，不读项目、不调用 LLM/TTS；候选仍 exit 0，执行失败 exit 1，格式错误沿用 Typer。
- 可选 extra 名 `content-check`；未进入检查不导入其第三方依赖，不访问模型服务；已缓存默认优先本地，`--offline` 禁止模型联网。
- 所有正式文本产物使用 `tenmin.atomic`；编码 UTF-8；临时抽帧隔离、结果最后发布、用户目录无关文件不删除。
- 测试和 pipeline 命令用裸 `uv run ...`，不套 rtk；其他 shell 命令按仓库约定使用 rtk。不运行 `ruff format`。
- 实施时创建独立 worktree，按任务实现/审查，最后全分支审查。只有用户明确请求提交时才执行 git commit；本计划不授权提交或合并。

## File Map and Dependency Order

| 文件 | 唯一职责 |
|---|---|
| `src/tenmin/config.py` | 四个创作配置字段与默认实例 |
| `src/tenmin/content_check/__init__.py` | 轻量公共导出，无额外依赖导入 |
| `src/tenmin/content_check/models.py` | 结果、身份和返回值的数据契约 |
| `src/tenmin/content_check/candidates.py` | 候选判定和采样点归组纯函数 |
| `src/tenmin/content_check/frames.py` | FFmpeg 抽帧和真实 PTS 对齐 |
| `src/tenmin/content_check/inference.py` | 模型缓存、下载、摘要验证和逐帧推理 |
| `src/tenmin/content_check/report.py` | 安全的本地 HTML 渲染 |
| `src/tenmin/content_check/report.html` | 静态页面模板，经 importlib.resources 加载 |
| `src/tenmin/content_check/service.py` | 缓存完整性、目录所有权、临时运行和原子发布 |
| `src/tenmin/pipeline.py` | 阶段选择、路径及按集调用/失败继续 |
| `src/tenmin/config_slices.py` | check 专用配置切片 |
| `src/tenmin/cli.py` | 独立命令与本次项目检查结果打印 |
| `pyproject.toml`, `uv.lock` | 可选依赖及实际兼容解析 |
| `README.md`, `AGENTS.md` | 安装、使用、阶段和缓存约定 |

任务依赖：1 → 2/3/4 → 5 → 6 → 7 → 8。日常测试完全 fake 外部模型和媒体运行，不下载模型。

## Shared Interfaces

所有类型定义在任务 1；签名在各任务重复引用，避免独立实现时猜名字。

```python
# models.py: all BaseModel classes use extra="forbid".
class ContentCheckError(RuntimeError): ...

class Scores(BaseModel):
    general: float
    sensitive: float
    questionable: float
    explicit: float

class FrameScore(BaseModel):
    time: float
    image: str  # POSIX relative path: frames/frame_000001.jpg
    label: Literal["general", "sensitive", "questionable", "explicit"]
    scores: Scores

class CandidateRange(BaseModel):
    first: float
    last: float
    count: int

class CacheIdentity(BaseModel):
    video_path: str  # resolved absolute path
    video_size: int
    video_mtime_ns: int
    repository: str
    revision: str
    model: str
    model_sha256: str
    preprocessing_version: str  # "rgb-bilinear-chw-v1"
    sampling_version: str  # "select-relative-pts-dar-v1"
    sample_interval: float

class CheckManifest(BaseModel):
    schema_version: Literal[1] = 1
    identity: CacheIdentity
    providers: list[str]
    extraction_seconds: float
    inference_seconds: float
    sensitive_threshold: float
    adult_threshold: float
    frames: list[FrameScore]
    candidate_count: int
    candidate_ranges: list[CandidateRange]

@dataclass(frozen=True)
class SampledFrame:
    path: Path
    time: float

@dataclass(frozen=True)
class ModelFiles:
    weights: Path
    metadata: Path
    sha256: str

@dataclass(frozen=True)
class CheckOutcome:
    status: Literal["completed", "cached", "report_rebuilt"]
    candidate_count: int
    frame_count: int
    report_path: Path
    results_path: Path
```

`Scores` 要求四项有限且 [0,1]、和与 1 的差 <=0.001；`FrameScore.time` 有限非负。
Manifest 校验帧非空、时间非递减、图片路径唯一且严格为 `frames/frame_\d{6,}.jpg`，
label 等于按固定类序取 argmax；拒绝路径穿越、绝对路径和非预期 schema。
缓存中的候选字段仍校验范围/计数，加载后按当前阈值重新求值，不把它们当原始事实。

### Task 1: 配置、数据契约和候选规则

**Files:**
- Modify: `src/tenmin/config.py`
- Create: `src/tenmin/content_check/__init__.py`, `models.py`, `candidates.py`
- Create: `tests/test_content_check_candidates.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: 上述类型；`ContentCheckConfig(StrictModel)` 和 `DEFAULT_CONTENT_CHECK`。
- Produces: `screen_frames(frames: Sequence[FrameScore], cfg: ContentCheckConfig) -> tuple[list[FrameScore], list[CandidateRange]]`。

- [ ] **Step 1: 写失败测试：配置有限性、等号边界与归组。**

```python
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_interval_rejects_invalid(value):
    with pytest.raises(ValidationError):
        ContentCheckConfig(sample_interval=value)

def test_candidate_boundary_and_gap():
    frames = [
        FrameScore(time=t, image=f"frames/frame_{i:06d}.jpg", label="sensitive",
                   scores=Scores(general=.2, sensitive=.5, questionable=.2, explicit=.1))
        for i, t in enumerate([.021, 1.521, 3.022], 1)
    ]
    selected, ranges = screen_frames(frames, ContentCheckConfig())
    assert selected == frames
    assert [(r.first, r.last, r.count) for r in ranges] == [
        (.021, 1.521, 2), (3.022, 3.022, 1)
    ]
```

补独立 adult=.5 命中、两项未达阈值、零候选、按时间而非输入顺序归组、输入不被修改；
配置未知键拒绝、project 默认关闭、两个阈值的 -epsilon/1+epsilon/nan/inf。

- [ ] **Step 2: 运行 red。** `uv run pytest tests/test_content_check_candidates.py tests/test_config.py -q`；新入口尚不存在应失败。
- [ ] **Step 3: 实现契约和最小纯函数。**

```python
class ContentCheckConfig(StrictModel):
    enabled: bool = False
    sample_interval: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    sensitive_threshold: float = Field(default=.5, ge=0, le=1, allow_inf_nan=False)
    adult_threshold: float = Field(default=.5, ge=0, le=1, allow_inf_nan=False)

# ProjectConfig: content_check = Field(default_factory=ContentCheckConfig)
# Module defaults: DEFAULT_CONTENT_CHECK = ContentCheckConfig()

selected = sorted(
    (f for f in frames if f.scores.sensitive >= cfg.sensitive_threshold
     or f.scores.questionable + f.scores.explicit >= cfg.adult_threshold),
    key=lambda f: f.time,
)
ranges = []
for frame in selected:
    if ranges and frame.time - ranges[-1].last <= 1.5 * cfg.sample_interval:
        ranges[-1] = CandidateRange(first=ranges[-1].first, last=frame.time,
                                    count=ranges[-1].count + 1)
    else:
        ranges.append(CandidateRange(first=frame.time, last=frame.time, count=1))
return selected, ranges
```

模型定义用实际 imports、Field/model_validator 完成 Shared Interfaces 的全部校验；错误信息点名坏字段。

- [ ] **Step 4: 运行 green 及源码约束。** `uv run pytest tests/test_content_check_candidates.py tests/test_config.py tests/test_source_hygiene.py -q`。
- [ ] **Step 5: 审查。** 核对 .5 等号、概率不被重归一化、类型不导入可选依赖；仅在用户授权提交后提交本任务文件。

### Task 2: 真实时间戳与正确显示比例抽帧

**Files:**
- Create: `src/tenmin/content_check/frames.py`, `tests/test_content_check_frames.py`

**Interfaces:**
- Consumes: `SampledFrame`, `ContentCheckError`。
- Produces: `extract_frames(video: Path, destination: Path, *, interval: float, ffmpeg: str, ffprobe: str) -> list[SampledFrame]`。
- Produces: `frame_filter(interval: float) -> str`；`parse_pts(stderr: str) -> list[float]`。

- [ ] **Step 1: 写失败测试，fake `tenmin.content_check.frames.run`（复用 render.ffmpeg.run）返回日志。**

```python
def test_parse_real_pts():
    stderr = (
        "[Parsed_showinfo_2 @ a] n: 0 pts: 0 pts_time:0.0213333\n"
        "[Parsed_showinfo_2 @ a] n: 1 pts: 9 pts_time:1.187\n"
    )
    assert parse_pts(stderr) == [.0213333, 1.187]
    assert "showinfo" in frame_filter(1)
    assert "setsar=1" in frame_filter(1)
```

fake run 创建两张文件再返回两条日志，断言文件顺序对 PTS；另测零帧、日志少一条、倒退/非有限时间失败，
路径含空格参数保持 list，FFmpegError 能包装为 ContentCheckError。

- [ ] **Step 2: red。** `uv run pytest tests/test_content_check_frames.py -q`。
- [ ] **Step 3: 实现抽帧。**

```python
filters = (
    f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',"
    "scale=640:trunc(640/dar/2)*2,setsar=1,showinfo"
)
args = ["-hide_banner", "-nostdin", "-y", "-i", str(video),
        "-map", "0:v:0", "-an", "-vf", filters, "-fps_mode", "vfr",
        "-q:v", "2", str(destination / "frame_%06d.jpg")]
stderr = run(args, ffmpeg=ffmpeg)
```

先 `probe_duration(video, ffprobe=ffprobe)` 确认视频可读。使用 FFmpeg 默认 autorotate，
测试有旋转元数据的真实样片；showinfo 位于采样之后，不用 frame index 推时间。
时间口径使用普通 `-i` 解码的播放相对时间（不传 `-copyts`）；用带非零起始 PTS 的真实夹具验证首帧接近 0，
若该 FFmpeg build 保留容器偏移，依据 ffprobe start_time 归一化并升级 sampling_version，不猜偏移。
保留音视频首帧相对偏差，不无条件减首采样帧时间。复用 ffmpeg.run 的解码替换/错误诊断；不需要 libass preflight。

- [ ] **Step 4: green。** `uv run pytest tests/test_content_check_frames.py tests/test_render_ffmpeg.py -q`。
- [ ] **Step 5: 审查。** 检查 DAR/旋转/PTS 证据，所有输出仅写临时 frames 目录；授权后提交。

### Task 3: 可选本地模型与惰性依赖

**Files:**
- Create: `src/tenmin/content_check/inference.py`, `tests/test_content_check_inference.py`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Consumes: `ModelFiles`, `SampledFrame`, `FrameScore`, `Scores`, `ContentCheckError`。
- Produces: `load_model_files(*, offline: bool) -> ModelFiles`。
- Produces: `score_frames(frames: Sequence[SampledFrame], files: ModelFiles) -> list[FrameScore]`。
- Produces: `REPO`, `REVISION`, `MODEL`, `MODEL_SHA256`, `PREPROCESSING_VERSION`, `SAMPLING_VERSION` 常量，供服务构造身份；取值如下。

- [ ] **Step 1: 写失败测试，本地缓存优先且离线绝不发远端请求。**

```python
def test_cached_download_never_queries_remote(monkeypatch, tmp_path):
    calls = []
    def download(repo, filename, *, revision, local_files_only):
        calls.append(local_files_only)
        assert local_files_only
        return str(tmp_path / Path(filename).name)
    # 用 sys.modules 的 fake huggingface_hub 注入 download；造匹配摘要的 fixture，
    # monkeypatch inference.MODEL_SHA256 为 fixture sha256。
    install_fake_hub(monkeypatch, download)
    create_fake_model_files(tmp_path)
    load_model_files(offline=False)
    assert calls == [True, True]
```

在本测试文件定义 `install_fake_hub`（SimpleNamespace 放入 sys.modules）和 `create_fake_model_files`
（权重写 b"fake"，meta labels 写固定四类，摘要 hashlib.sha256(b"fake").hexdigest()）。
补缓存 miss 在线回退、offline miss 提示、摘要不符、labels 不符、缺依赖中文安装提示；
fake ORT session 检查 4线程、CPU、输入动态384、输出已为[.1,.2,.3,.4]不做softmax。

- [ ] **Step 2: red。** `uv run pytest tests/test_content_check_inference.py -q`。
- [ ] **Step 3: 实现模型下载与推理。**

```python
REPO = "deepghs/anime_dbrating"
REVISION = "7af21db648acdeb74f5c334abda9dd7403407b3c"
MODEL = "mobilenetv3_large_100_v0_ls0.2"
MODEL_SHA256 = "c7d6fd0cd71c48616fa87fe2bca87f8e1775e9a3b96d12797e2144fca3543362"
PREPROCESSING_VERSION = "rgb-bilinear-chw-v1"
SAMPLING_VERSION = "select-relative-pts-dar-v1"

# imports of numpy/PIL/onnxruntime/huggingface_hub live in function bodies.
try:
    path = hf_hub_download(REPO, filename, revision=REVISION, local_files_only=True)
except LocalEntryNotFoundError as error:
    if offline:
        raise ContentCheckError("离线模型缺失；先联网运行检查以下载模型") from error
    path = hf_hub_download(REPO, filename, revision=REVISION, local_files_only=False)

options = ort.SessionOptions()
options.intra_op_num_threads = 4
session = ort.InferenceSession(str(files.weights), sess_options=options,
                               providers=["CPUExecutionProvider"])
info = session.get_inputs()[0]
height, width = info.shape[2:]
if not isinstance(height, int) or not isinstance(width, int):
    height = width = 384
with Image.open(frame.path) as image:
    array = np.asarray(image.convert("RGB").resize((width, height),
                       Image.Resampling.BILINEAR), dtype=np.float32) / 255
tensor = ((array.transpose(2, 0, 1) - .5) / .5)[None].copy()
values = session.run(["output"], {info.name: tensor})[0][0]
```

校验输入rank/三通道、输出维度和 Shared Interfaces 的分数；image 用 `frames/<文件名>`，
label 固定类序 argmax。只在第三方下载/ORT/图片解码边界把已知异常转 ContentCheckError，
不包整个函数的 Exception；ImportError 提示 `uv sync --extra content-check`。

- [ ] **Step 4: 加 extra 并实际解析兼容性。**

```toml
content-check = [
    "onnxruntime>=1.20,<2",
    "numpy>=2,<3",
    "pillow>=11,<13",
    "huggingface-hub>=0.30,<2",
]
```

执行 `rtk proxy uv lock`，随后 `rtk proxy uv sync --extra content-check`。
在当前 Python（包括项目使用的3.14）运行依赖 import；如 wheel 不支持，仅为 extra 的不支持平台/版本添加环境标记，
并在错误中给出可运行的 Python 版本，不改项目全局 requires-python。
需要 NumPy/Pillow 的测试使用 importorskip，其他模型缓存/惰性导入测试在无 extra 环境也必须运行。
fake session 测试使用本地生成图片而非模型下载。

- [ ] **Step 5: green 与导入隔离。** `uv run pytest tests/test_content_check_inference.py tests/test_source_hygiene.py -q`；
独立无 extra 环境 import tenmin.cli 后确认 sys.modules 无 onnxruntime/numpy/PIL/huggingface_hub。
- [ ] **Step 6: 审查。** 检查依赖版本证据、固定摘要及下载缓存语义，授权后提交。

### Task 4: 本地报告的安全渲染与交互

**Files:**
- Create: `src/tenmin/content_check/report.py`, `report.html`, `tests/test_content_check_report.py`

**Interfaces:**
- Consumes: `CheckManifest`。
- Produces: `render_report(manifest: CheckManifest) -> str`（只返回文本，不落盘）。

- [ ] **Step 1: 写失败测试，视频名无法关闭 script 标签。**

```python
def test_report_escapes_embedded_json(valid_manifest):
    manifest = valid_manifest.model_copy(deep=True)
    manifest.identity.video_path = '</script><img src=x onerror="alert(1)">.mp4'
    html = render_report(manifest)
    assert manifest.identity.video_path not in html
    assert "\\u003c/script>" in html
    assert "textContent" in html
    assert "https://" not in html
```

`valid_manifest` 在 `tests/test_content_check_report.py` fixture 定义：按 Shared Interfaces
完整填一个general帧、固定identity、CPU、耗时0、阈值.5、候选0，不依赖生产缓存。
补阈值从manifest初始化、四类英文和中文解释、分数非概率文案、相对图片路径、范围非精确文案。

- [ ] **Step 2: red。** `uv run pytest tests/test_content_check_report.py -q`。
- [ ] **Step 3: 从 POC 提取页面，适配 Manifest，而非导入 POC。**

```python
template = resources.files("tenmin.content_check").joinpath("report.html").read_text(
    encoding="utf-8"
)
data = manifest.model_dump_json().replace("<", "\\u003c")
return template.replace("__DATA__", data)
```

页面沿用 POC DOM id `s/a/all/sort/count/grid/more/segments`，60张分页，risk与time排序。
JS `report.identity.sample_interval` 决定归组，阈值初始化来自manifest，
视频名/时间/label全部textContent；零候选显示“已检查，未发现达到当前阈值的候选”。
验证非法/空阈值显示参数提示而非默默当0，修改页面阈值不写文件。
顶部显示路径、模型revision、采样数、耗时、provider。

- [ ] **Step 4: green。** `uv run pytest tests/test_content_check_report.py -q`。
- [ ] **Step 5: 审查。** HTML不引用CDN，图片和JSON渲染安全；最终浏览器检查在任务8，授权后提交。

### Task 5: 可复用检查服务与中断一致性

**Files:**
- Create: `src/tenmin/content_check/service.py`, `tests/test_content_check_service.py`
- Modify: `src/tenmin/content_check/__init__.py`（仅导出公共服务和结果类型）

**Interfaces:**
- Consumes: 前四任务的全部类型；`extract_frames`、`load_model_files`、`score_frames`、`screen_frames`、`render_report`。
- Produces: `check_video(video: Path, output_dir: Path, *, cfg: ContentCheckConfig, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", force: bool = False, offline: bool = False) -> CheckOutcome`。
- Internal: `load_cached(output_dir: Path, identity: CacheIdentity) -> CheckManifest | None`；仅 regular files、有限/合法schema、所有截图存在。

- [ ] **Step 1: 创建完整 fake 检查fixture及失败测试。**

```python
@pytest.fixture
def fake_check(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    calls = {"extract": 0, "score": 0}
    def extract(video, destination, **kwargs):
        calls["extract"] += 1
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "frame_000001.jpg"
        path.write_bytes(b"image")
        return [SampledFrame(path=path, time=.021)]
    def score(frames, files):
        calls["score"] += 1
        return [FrameScore(time=.021, image="frames/frame_000001.jpg", label="general",
                           scores=Scores(general=.7, sensitive=.1, questionable=.1,
                                         explicit=.1))]
    monkeypatch.setattr(service, "extract_frames", extract)
    monkeypatch.setattr(service, "score_frames", score)
    monkeypatch.setattr(service, "load_model_files", lambda **kw: ModelFiles(
        weights=tmp_path / "model.onnx", metadata=tmp_path / "meta.json", sha256="a" * 64))
    return video, tmp_path / "check", calls

def test_threshold_rebuild_reuses_scores(fake_check):
    video, output, calls = fake_check
    first = check_video(video, output, cfg=ContentCheckConfig())
    second = check_video(video, output, cfg=ContentCheckConfig(sensitive_threshold=.1))
    assert first.candidate_count == 0
    assert second.candidate_count == 1
    assert second.status == "report_rebuilt"
    assert calls == {"extract": 1, "score": 1}
```

补完全命中cached且mtime不变、HTML丢失重建、视频size/mtime/path变化、interval/model摘要/version变化、
缺截图/空JSON/坏JSON/错误schema/越界图片路径/NaN分数触发重新检测、force重跑。
补非拥有非空目录拒绝、不删除notes.txt、发布中途OSError后无有效results、
KeyboardInterrupt传播且不完成、零帧失败、模型不可用时旧报告不被作为本次成功返回。

- [ ] **Step 2: red。** `uv run pytest tests/test_content_check_service.py -q`。
- [ ] **Step 3: 实现服务及发布顺序。**

```python
# Outline inside check_video; failures at known I/O boundaries become ContentCheckError.
if not video.is_file():
    raise ContentCheckError(f"视频文件不存在或不是普通文件：{video}")
files = load_model_files(offline=offline)
identity = CacheIdentity(video_path=str(video.resolve()), video_size=video.stat().st_size,
    video_mtime_ns=video.stat().st_mtime_ns, repository=REPO, revision=REVISION,
    model=MODEL, model_sha256=files.sha256, preprocessing_version=PREPROCESSING_VERSION,
    sampling_version=SAMPLING_VERSION, sample_interval=cfg.sample_interval)
cached = None if force else load_cached(output_dir, identity)
if cached is not None:
    selected, ranges = screen_frames(cached.frames, cfg)
    # thresholds unchanged + existing regular HTML => cached, no writes.
    # otherwise copy manifest with current thresholds/candidates and atomically republish report.
```

目录首次为空/不存在时，写固定 `.tenmin-content-check` 所有权标记（版本1）；
已有非空目录无该合法标记时拒绝。检查 output_dir、frames 不是symlink，
引用图片resolve不能越过frames；不得通过坏JSON删除任意路径。
`TemporaryDirectory(dir=output_dir.parent)` 确保同文件系统，临时frames不混旧文件。
全重跑：临时抽帧→临时推理→校验Manifest→渲染字符串→使旧results失效→
仅替换工具管理的frames→atomic report.html→最后atomic results.json。
报告重建也先失效完成标记，再atomic HTML，最后atomic JSON；过程中出错不返回成功。
仅删除之前工具拥有的固定frames目录，不递归清空用户output_dir；无关文件保留。
发布前复查视频stat身份没有在运行中变化，变化时报检查失败，防止旧帧挂新身份。
坏缓存以miss处理，文件读取守卫避免FIFO阻塞；原始有效缓存无需重新推理。

- [ ] **Step 4: green。** `uv run pytest tests/test_content_check_service.py tests/test_content_check_candidates.py tests/test_content_check_report.py tests/test_source_hygiene.py -q`。
- [ ] **Step 5: 审查。** 手动核对每一个rename/write异常窗口，缓存阈值不进入identity；授权后提交。

### Task 6: check 阶段接入纵向管线

**Files:**
- Modify: `src/tenmin/pipeline.py`, `src/tenmin/config_slices.py`
- Modify: `tests/test_pipeline.py`, `tests/test_config_slices.py`
- Create: `tests/test_pipeline_content_check.py`

**Interfaces:**
- Consumes: `check_video`, `CheckOutcome`, `ContentCheckError`、`cfg.content_check`。
- Produces: `Paths.check_dir(episode: int) -> Path`；`STAGES`末尾check。
- Produces: `run_check(cfg: ProjectConfig, episode: int, *, force: bool = False) -> CheckOutcome`。
- Extend: `run_pipeline(..., check_results: dict[int, CheckOutcome] | None = None) -> list[str]`；
  收集器仅写本次成功结果，开头clear；provider注解为 `LLMProvider | None`（check-only已有CLI传None）。

- [ ] **Step 1: 写失败测试：显式检查不跑上游，失败继续下一集。**

```python
async def test_explicit_check_runs_when_disabled(tmp_path, monkeypatch):
    cfg = ProjectConfig(show="test", slug="test",
        episodes=[EpisodeConfig(number=2, video="missing-source.mp4")]).bind_root(tmp_path)
    calls = []
    def fake(cfg, episode, *, force=False):
        calls.append(episode)
        return CheckOutcome("completed", 1, 2,
            Paths(cfg.root).check_dir(episode) / "report.html",
            Paths(cfg.root).check_dir(episode) / "results.json")
    monkeypatch.setattr(pipeline, "run_check", fake)
    results = {}
    warnings = await run_pipeline(cfg, None, only=["check"], check_results=results)
    assert calls == [2]
    assert results[2].candidate_count == 1
    assert not warnings
```

两集fake_check第一集抛ContentCheckError、第二集成功，断言两个调用都发生且warnings点名E02；
第一集RuntimeError/CancelledError必须传播。补enabled=false全流程不调服务，enabled=true在render之后调用，
render fresh仍调用服务、`--only render`不check、`--from check`主动执行、
`only=[]`不动作、缺成片提示render、缺源片但已有成片可检查。
用FakeReporter验证每集stage顺序和episode_done；失败不发check stage_done。

- [ ] **Step 2: red。** `uv run pytest tests/test_pipeline_content_check.py -q`。
- [ ] **Step 3: 最小接入，无全管线重构。**

```python
# _ARTIFACTS addition: "check_dir": ("08_check", "")
# Paths.check_dir returns self._artifact("check_dir", episode)
# STAGE_FIELDS addition: "check": ("content_check",)
# no EPISODE/source video fields: check reads final rendered video only.

explicit_check = ((only is not None and "check" in only)
                  or (only is None and from_stage == "check"))
if "check" in wanted and (cfg.content_check.enabled or explicit_check):
    reporter.stage_start("check")
    try:
        outcome = run_check(cfg, number, force=force)
    except ContentCheckError as error:
        warnings.append(f"E{number:02d}：本次检查失败／报告未更新：{error}")
    else:
        if check_results is not None:
            check_results[number] = outcome
        reporter.substep("check", outcome.frame_count, outcome.frame_count,
            f"候选 {outcome.candidate_count} 帧；报告 {outcome.report_path}")
        reporter.stage_done("check")
```

在render分支之后、episode_done之前插入；无mtime outer skip，由服务核实完整性。
关闭时stage_skip check并输出“检查未启用”（CLI任务7负责清楚打印，与缓存命中区别）。
run_check验证`paths.video`存在，缺失转ContentCheckError并提示先render；
只传最终视频、cfg.content_check、cfg.render.ffmpeg_path/ffprobe_path、force。
check-only不能扩展audio/render preflight条件，也不探源视频。
无需新增RichProgressReporter方法（现有stage/substep已支持任意阶段）。

- [ ] **Step 4: 更新冻结契约期望。** STAGES/stages_from期待追加check；FROZEN_LAYOUT只新增check_dir，
原条目逐字不变；config_slices完整性增加check，测试四个content_check变化只改变check payload。
- [ ] **Step 5: green。** `uv run pytest tests/test_pipeline_content_check.py tests/test_pipeline.py tests/test_config_slices.py tests/test_rich_progress.py -q`。
- [ ] **Step 6: 审查。** 纵向循环/预取取消语义保持，错误只在check边界捕获，授权后提交。

### Task 7: 独立命令和项目结果展示

**Files:**
- Modify: `src/tenmin/cli.py`, `tests/test_cli.py`
- Create: `tests/test_cli_content_check.py`

**Interfaces:**
- Consumes: `check_video`, `ContentCheckConfig`, `DEFAULT_CONTENT_CHECK`, `DEFAULT_RENDER`, `CheckOutcome`, `ContentCheckError`。
- Produces: Typer `check-content VIDEO [--output DIR] [--interval FLOAT] [--sensitive-threshold FLOAT] [--adult-threshold FLOAT] [--force] [--offline]`。
- Consumes: 任务6 `run_pipeline(check_results=...)`；收集器只含本次成功结果。

- [ ] **Step 1: 写失败测试：有候选exit0，预期错误exit1。**

```python
def test_independent_command_needs_no_project_or_provider(tmp_path, monkeypatch):
    video = tmp_path / "视频.mp4"
    video.write_bytes(b"video")
    seen = []
    def fake(video, output_dir, **kwargs):
        seen.append((video, output_dir, kwargs))
        return CheckOutcome("completed", 3, 10, output_dir / "report.html",
                            output_dir / "results.json")
    monkeypatch.setattr(cli, "check_video", fake)
    monkeypatch.setattr(cli, "build_provider", lambda *a, **kw: pytest.fail("LLM called"))
    result = runner.invoke(app, ["check-content", str(video), "--offline"])
    assert result.exit_code == 0, out(result)
    assert "候选 3" in out(result)
    assert seen[0][1] == video.with_name("视频.content-check")
    assert seen[0][2]["offline"] is True
```

在本测试文件定义runner与out合并stdout/stderr（沿用test_cli.py方式），补output/interval/阈值/force传参、
非法nan/inf/阈值错误、ContentCheckError exit1、cached也打印、零候选已检查文案。
项目`--only check`/`--from check`禁止build_provider/build_tts_engine调用，
检查失败旧report存在仍不打印成功链接；默认关闭打印“检查未启用”。
增加入口一致性测试：两个入口传同一视频与同一配置，fake服务记录参数，
断言interval/阈值一致、项目入口视频为Paths.video；服务返回相同候选数，两入口打印相同数量。

- [ ] **Step 2: red。** `uv run pytest tests/test_cli_content_check.py -q`。
- [ ] **Step 3: 加命令与打印。**

```python
@app.command("check-content")
def check_content(
    video: Path,
    output: Path | None = typer.Option(None, "--output"),
    interval: float = typer.Option(DEFAULT_CONTENT_CHECK.sample_interval, "--interval"),
    sensitive_threshold: float = typer.Option(
        DEFAULT_CONTENT_CHECK.sensitive_threshold, "--sensitive-threshold"),
    adult_threshold: float = typer.Option(DEFAULT_CONTENT_CHECK.adult_threshold,
                                         "--adult-threshold"),
    force: bool = typer.Option(False, "--force"),
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    destination = output if output is not None else video.with_name(
        f"{video.stem}.content-check")
    try:
        cfg = ContentCheckConfig(sample_interval=interval,
            sensitive_threshold=sensitive_threshold, adult_threshold=adult_threshold)
        outcome = check_video(video, destination, cfg=cfg,
            ffmpeg=DEFAULT_RENDER.ffmpeg_path, ffprobe=DEFAULT_RENDER.ffprobe_path,
            force=force, offline=offline)
    except (ContentCheckError, ValueError) as error:
        typer.secho(str(error), fg="red", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"已检查 {outcome.frame_count} 帧；候选 {outcome.candidate_count} 帧")
    typer.echo(f"检查报告：{outcome.report_path}")
```

验证错误只捕获配置ValidationError（可把上例ValueError收窄为ValidationError），
服务运行未知ValueError不能被误吞。run创建本地check_results字典传管线，
只从该字典打印本次成功报告，不通过旧文件exists推断成功。
正常默认流程关闭时打印未启用；显式only render无额外check提示。
PROJECT_TEMPLATE_FIELDS加content_check，使init生成可见开关，所有默认来源仍是config。
run docstring/help追加check，force说明包含缓存。

- [ ] **Step 4: green。** `uv run pytest tests/test_cli_content_check.py tests/test_cli.py tests/test_pipeline_content_check.py -q`。
- [ ] **Step 5: 审查。** 同一服务两个入口参数一致，输出success与failed不混淆，授权后提交。

### Task 8: 实际成片验证、文档和全分支验收

**Files:**
- Modify: `README.md`, `AGENTS.md`
- Create: `docs/content-check-validation.md`（实际环境、命令、结果与效果边界）
- 不改 POC 和既有历史设计快照。

**Interfaces:** 使用已实现CLI及检查结果，不新增生产接口。

- [ ] **Step 1: 运行相关测试后做真实离线短成片。**

```bash
uv run pytest tests/test_content_check_candidates.py tests/test_content_check_frames.py tests/test_content_check_inference.py tests/test_content_check_report.py tests/test_content_check_service.py tests/test_pipeline_content_check.py tests/test_cli_content_check.py -q
uv run --extra content-check tenmin check-content work/saijo/07_render/E02.mp4 --output work/content-check-validation --offline
uv run --extra content-check tenmin check-content work/saijo/07_render/E02.mp4 --output work/content-check-validation --offline
uv run --extra content-check tenmin check-content work/saijo/07_render/E02.mp4 --output work/content-check-validation --sensitive-threshold 0.6 --offline
```

先用Read确认真实成片存在。若没有成片，只在work/里从原片取30秒样片（标注是测试片，
不冒充项目成片）：确认work父目录后运行
`rtk ffmpeg -hide_banner -nostdin -y -ss 1160 -i work/saijo/video/E02.mp4 -t 30 -c:v libx264 -an work/content-check-sample.mp4`，
三个CLI命令的视频参数改为该样片。记录第一次帧数/耗时/四类分数，第二次不抽帧推理，
第三次只重建。最后项目现有成片可用时跑`uv run tenmin run saijo --episode 2 --only check`证明无需API。

- [ ] **Step 2: 验证真实媒体几何与时钟。** 使用FFmpeg生成 `testsrc2=size=320x240:rate=10:duration=3` 的
SAR=2视频，另用remux写旋转90度metadata及起始PTS=5的片段；调用extract_frames，
ffprobe核对JPEG显示比例，时间对播放起点，不按采样序号推算。样片仅在work验证目录，
记录具体FFmpeg版本与观察结果；发现失败按systematic-debugging修复后重跑对应测试。

- [ ] **Step 3: 浏览器验证。** 先Chrome DevTools；连接不可用时用Playwright已安装Chrome
`chromium.launch(channel="chrome", headless=True)`打开file报告。
断言无pageerror；默认候选数与JSON相同；s/a改1→候选0；all显示全帧；time首帧正确；
more从60到120（真实样片不足120时使用任务4fixture生成150帧报告）；
图片`naturalWidth>0`；页面阈值变化不改results.json。
项目真实检查可运行时，将其results与同一成片的独立检查results对比，
只比较frames中的time/label/scores及当前阈值候选，不比较耗时和输出路径。
人工查看最高候选和普通帧，描述误报，不发布准确率/召回率。

- [ ] **Step 4: 文档写入实际安装和运行示例。**

```yaml
content_check:
  enabled: true
  sample_interval: 1.0
  sensitive_threshold: 0.5
  adult_threshold: 0.5
```

README列 `uv sync --extra content-check`、项目全流程/only check/only render、独立命令、输出路径、
首次联网及离线范围、阈值仅重建、force身份边界、失败继续且非零候选不改退出码。
AGENTS阶段9→10、子config9→10、模型11→12、ProjectConfig字段15→16；
测试文件数量与marker数量根据最终代码实数核对，不凭计划预估。
增加content_check模块及完成标记/异常族约定，遵writing-for-agents技能后修改AGENTS。
验证记录明确本次模型/环境与人工观察，未通过项写真实原因。

- [ ] **Step 5: 最终回归和lint。**

```bash
uv run pytest tests/ -q
rtk proxy uv run ruff check src/tenmin/content_check src/tenmin/config.py src/tenmin/config_slices.py src/tenmin/pipeline.py src/tenmin/cli.py tests/test_content_check_candidates.py tests/test_content_check_frames.py tests/test_content_check_inference.py tests/test_content_check_report.py tests/test_content_check_service.py tests/test_pipeline_content_check.py tests/test_cli_content_check.py tests/test_config.py tests/test_config_slices.py tests/test_pipeline.py tests/test_cli.py
rtk git diff --check
rtk git status --short
rtk git diff --stat
```

记录passed/skipped/failed和lint结果，不把跳过的真实模型测试当验证通过。
在未安装extra的隔离环境运行基础CLI/help与默认关闭的pipeline fake回归，证明可选依赖无侵入。

- [ ] **Step 6: 全分支审查与交付。** 调用requesting-code-review技能按批准设计审查全部任务，
修复发现后只重跑受影响测试及必要全回归。交付独立命令、项目开关、实际报告链接与验证证据。
用户授权提交时逐任务文件定向stage，不包含原有 `.coverage/ED.PNG/ed_start.png/op.png/op_start.png`；
合并回main需用户明确决定。

## Self-Review Coverage

| 设计章节 | 实现任务 / 证据 |
|---|---|
| 目标、模型边界、模块接口 | Global Constraints、1、3、6、7、8 |
| 配置和触发 | 1、6、7；显式only/from测试 |
| 独立CLI/退出码/失败继续 | 6、7；两集错误传播测试 |
| 抽帧、PTS、显示比例、模型预处理 | 2、3、8真实媒体几何检查 |
| 产物/缓存/阈值重建/中断 | 5；发布各异常窗口测试 |
| 报告与安全转义 | 4、8浏览器证据 |
| extra/Python/默认无依赖联网 | 3、7、8隔离环境验证 |
| 文档、默认pytest、lint、人工抽查 | 8 |

接口统一使用 `sample_interval`（配置/identity）与 `interval`（独立CLI/抽帧函数参数），
`CheckOutcome.candidate_count`只指候选帧数，范围数量取candidate_ranges长度；
管线本次结果收集器不读取旧报告推断成功。模板数据契约为CheckManifest，不兼容POC JSON，
POC路径与正式报告目录互相独立。
