"""本地模型身份、缓存优先、预处理和第三方错误边界。"""

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ConnectError

from tenmin.content_check import model
from tenmin.content_check.models import ContentCheckError, ModelFiles, SampledFrame

LABELS = ["general", "sensitive", "questionable", "explicit"]


class LocalEntryNotFoundError(Exception):
    pass


class HfHubHTTPError(Exception):
    pass


class OrtFailure(Exception):
    pass


def install_fake_hub(monkeypatch, download):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", SimpleNamespace(
        LocalEntryNotFoundError=LocalEntryNotFoundError, HfHubHTTPError=HfHubHTTPError,
    ))


def create_fake_model_files(tmp_path, monkeypatch):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"fake")
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps({"labels": LABELS}), encoding="utf-8")
    digest = hashlib.sha256(b"fake").hexdigest()
    monkeypatch.setattr(model, "MODEL_SHA256", digest)
    return ModelFiles(weights=weights, metadata=metadata, sha256=digest)


def test_pinned_identity():
    assert model.REPO == "deepghs/anime_dbrating"
    assert model.REVISION == "7af21db648acdeb74f5c334abda9dd7403407b3c"
    assert model.MODEL == "mobilenetv3_large_100_v0_ls0.2"
    assert model.MODEL_SHA256 == "c7d6fd0cd71c48616fa87fe2bca87f8e1775e9a3b96d12797e2144fca3543362"
    assert model.PREPROCESSING_VERSION == "rgb-bilinear-chw-v1"
    assert model.SAMPLING_VERSION == "select-relative-pts-dar-v1"


@pytest.mark.parametrize("offline", [False, True])
def test_cached_download_never_queries_remote(monkeypatch, tmp_path, offline):
    files = create_fake_model_files(tmp_path, monkeypatch)
    calls = []

    def download(repo, filename, *, revision, local_files_only):
        calls.append((filename, local_files_only))
        assert repo == model.REPO and revision == model.REVISION
        assert local_files_only
        return str(tmp_path / Path(filename).name)

    install_fake_hub(monkeypatch, download)
    assert model.load_model_files(offline=offline) == files
    assert calls == [(f"{model.MODEL}/model.onnx", True), (f"{model.MODEL}/meta.json", True)]


@pytest.mark.parametrize("missing", ["model.onnx", "meta.json"])
def test_online_falls_back_only_for_missing_file(monkeypatch, tmp_path, missing):
    files = create_fake_model_files(tmp_path, monkeypatch)
    calls = []

    def download(repo, filename, *, revision, local_files_only):
        name = Path(filename).name
        calls.append((name, local_files_only))
        if name == missing and local_files_only:
            raise LocalEntryNotFoundError("not cached")
        return str(tmp_path / name)

    install_fake_hub(monkeypatch, download)
    assert model.load_model_files(offline=False) == files
    expected = [("model.onnx", True), ("meta.json", True)]
    expected.insert(1 if missing == "model.onnx" else 2, (missing, False))
    assert calls == expected


@pytest.mark.parametrize("missing", ["model.onnx", "meta.json"])
def test_offline_miss_does_not_download(monkeypatch, tmp_path, missing):
    create_fake_model_files(tmp_path, monkeypatch)
    calls = []

    def download(repo, filename, *, revision, local_files_only):
        calls.append(local_files_only)
        assert local_files_only
        if Path(filename).name == missing:
            raise LocalEntryNotFoundError("not cached")
        return str(tmp_path / Path(filename).name)

    install_fake_hub(monkeypatch, download)
    with pytest.raises(ContentCheckError, match=r"离线.*联网") as caught:
        model.load_model_files(offline=True)
    assert isinstance(caught.value.__cause__, LocalEntryNotFoundError)
    assert calls and all(calls)


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("corruption", ["digest", "labels", "json", "structure", "missing"])
def test_corrupt_cache_is_rejected_without_redownload(monkeypatch, tmp_path, offline, corruption):
    files = create_fake_model_files(tmp_path, monkeypatch)
    if corruption == "digest":
        files.weights.write_bytes(b"tampered")
    elif corruption == "labels":
        files.metadata.write_text(json.dumps({"labels": LABELS[::-1]}), encoding="utf-8")
    elif corruption == "json":
        files.metadata.write_text("{", encoding="utf-8")
    elif corruption == "structure":
        files.metadata.write_text("[]", encoding="utf-8")
    else:
        files.weights.unlink()
    calls = []

    def download(repo, filename, *, revision, local_files_only):
        calls.append(local_files_only)
        return str(tmp_path / Path(filename).name)

    install_fake_hub(monkeypatch, download)
    with pytest.raises(ContentCheckError, match=r"摘要|元数据|读取"):
        model.load_model_files(offline=offline)
    assert all(calls)


@pytest.mark.parametrize("error", [
    HfHubHTTPError("403"), OSError("disk denied"), ConnectError("connection failed"),
])
def test_expected_download_failure_has_chinese_diagnostic(monkeypatch, error):
    def download(*args, **kwargs):
        raise error

    install_fake_hub(monkeypatch, download)
    with pytest.raises(
        ContentCheckError, match=r"模型.*(403|disk denied|connection failed)",
    ) as caught:
        model.load_model_files(offline=False)
    assert caught.value.__cause__ is error


def test_download_programming_error_propagates(monkeypatch):
    def download(*args, **kwargs):
        raise TypeError("bug")

    install_fake_hub(monkeypatch, download)
    with pytest.raises(TypeError, match="bug"):
        model.load_model_files(offline=False)


@pytest.mark.parametrize("detail", [
    "File reconstruction error: CAS Client Error: Request error: connection reset, domain: cas",
    "File reconstruction error: CAS Client Error: Request middleware error: request failed",
    "File reconstruction error: CAS Client Error: IO error: disk full",
    "File reconstruction error: IO Error: permission denied",
    "File reconstruction error: Corrupted Reconstruction: hash mismatch",
    "File size mismatch: expected 20 bytes but downloaded 10 bytes",
])
def test_xet_background_download_failure_is_wrapped(monkeypatch, detail):
    error = RuntimeError(f"Task error: {detail}")
    calls = []

    def download(repo, filename, *, revision, local_files_only):
        calls.append(local_files_only)
        if local_files_only:
            raise LocalEntryNotFoundError("cache miss")
        raise error

    install_fake_hub(monkeypatch, download)
    with pytest.raises(ContentCheckError, match="模型文件下载失败") as caught:
        model.load_model_files(offline=False)
    assert caught.value.__cause__ is error
    assert str(error) in str(caught.value)
    assert calls == [True, False]


@pytest.mark.parametrize("error", [
    RuntimeError("programming bug"),
    RuntimeError("Task error: background task panicked"),
    RuntimeError("Task error: File reconstruction error: Internal Error: invariant failed"),
    RuntimeError("Task error: File reconstruction error: Task Join Error: task cancelled"),
    RuntimeError("Task error: User cancelled: aborted"),
    RuntimeError("Operation cancelled: shutdown"),
    TypeError("programming bug"),
    KeyboardInterrupt(),
])
def test_online_fallback_does_not_mask_bugs_or_cancellation(monkeypatch, error):
    calls = []

    def download(repo, filename, *, revision, local_files_only):
        calls.append(local_files_only)
        if local_files_only:
            raise LocalEntryNotFoundError("cache miss")
        raise error

    install_fake_hub(monkeypatch, download)
    with pytest.raises(type(error)) as caught:
        model.load_model_files(offline=False)
    assert caught.value is error
    assert calls == [True, False]


def test_xet_shaped_runtime_error_outside_online_download_propagates(monkeypatch):
    error = RuntimeError("Task error: File reconstruction error: IO Error: disk full")

    def download(*args, **kwargs):
        assert kwargs["local_files_only"]
        raise error

    install_fake_hub(monkeypatch, download)
    with pytest.raises(RuntimeError) as caught:
        model.load_model_files(offline=False)
    assert caught.value is error


def test_missing_hub_has_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ContentCheckError, match="uv sync --extra content-check"):
        model.load_model_files(offline=True)


def test_imports_are_lazy_in_fresh_process():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import tenmin.cli; "
         "import tenmin.content_check.model; "
         "blocked = {'numpy', 'PIL', 'onnxruntime', 'huggingface_hub'}; "
         "assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)"],
        capture_output=True, text=True, errors="replace",
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image")
    files = create_fake_model_files(tmp_path, monkeypatch)
    frame_path = tmp_path / "frame_000001.jpg"
    # PNG content avoids JPEG loss when checking exact RGB normalization.
    image.new("RGBA", (7, 5), (255, 0, 128, 100)).save(frame_path, format="PNG")
    state = SimpleNamespace(
        shape=[1, 3, "height", "width"], output=np.array([[.1, .2, .3, .4]], dtype=np.float32),
        calls=[], tensors=[], init_error=None, run_error=None,
    )

    class Session:
        def __init__(self, path, *, sess_options, providers):
            state.calls.append((path, sess_options.intra_op_num_threads, providers))
            if state.init_error:
                raise state.init_error

        def get_inputs(self):
            return [SimpleNamespace(name="pixels", shape=state.shape)]

        def run(self, names, feed):
            assert names == ["output"]
            state.tensors.append(feed["pixels"])
            if state.run_error:
                raise state.run_error
            return [state.output]

    errors = SimpleNamespace(**dict.fromkeys(
        ["Fail", "InvalidArgument", "InvalidGraph", "InvalidProtobuf", "NoSuchFile",
         "NotImplemented", "RuntimeException", "EngineError", "EPFail"], OrtFailure,
    ))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=Session,
    ))
    monkeypatch.setitem(sys.modules, "onnxruntime.capi.onnxruntime_pybind11_state", errors)
    return np, files, SampledFrame(frame_path, 1.25), state


@pytest.mark.parametrize("shape,expected", [
    ([1, 3, "h", "w"], (1, 3, 384, 384)),
    ([1, 3, 2, 4], (1, 3, 2, 4)),
    ([1, 3, 2, "w"], (1, 3, 384, 384)),
])
def test_preprocessing_cpu_threads_and_raw_scores(runtime, shape, expected):
    np, files, frame, state = runtime
    state.shape = shape
    scored = model.score_frames([frame, SampledFrame(frame.path, 3.7)], files)
    assert state.calls == [(str(files.weights), 4, ["CPUExecutionProvider"])]
    assert len(state.tensors) == 2
    tensor = state.tensors[0]
    assert tensor.shape == expected and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 0, 0], [1, -1, (128 / 255 - .5) / .5], atol=1e-7)
    assert [s.time for s in scored] == [1.25, 3.7]
    assert scored[0].image == "frames/frame_000001.jpg"
    assert scored[0].label == "explicit"
    assert list(scored[0].scores.model_dump().values()) == pytest.approx([.1, .2, .3, .4])


def test_tied_argmax_uses_fixed_class_order(runtime):
    np, files, frame, state = runtime
    state.output = np.array([[.4, .4, .1, .1]])
    assert model.score_frames([frame], files)[0].label == "general"


def test_resize_matches_bilinear_for_nonuniform_rgb_image(runtime):
    np, files, frame, state = runtime
    image = pytest.importorskip("PIL.Image")
    pixels = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
                      dtype=np.uint8)
    original = image.fromarray(pixels)
    original.save(frame.path, format="PNG")
    state.shape = [1, 3, 5, 7]
    model.score_frames([frame], files)
    resized = np.asarray(original.resize((7, 5), image.Resampling.BILINEAR), dtype=np.float32)
    expected = ((resized / 255 - .5) / .5).transpose(2, 0, 1)[None]
    np.testing.assert_array_equal(state.tensors[0], expected)


@pytest.mark.parametrize("shape", [[3, 384, 384], [1, 4, 384, 384], [1, 3, 0, 4]])
def test_bad_input_shape_is_expected_failure(runtime, shape):
    _, files, frame, state = runtime
    state.shape = shape
    with pytest.raises(ContentCheckError, match="输入"):
        model.score_frames([frame], files)
    assert state.tensors == []


@pytest.mark.parametrize("values", [
    [[.1, .2, .7]], [.1, .2, .3, .4], [[.1, .2, .3, .4], [.1, .2, .3, .4]],
    [[float("nan"), .2, .3, .4]], [[float("inf"), .2, .3, .4]],
    [[-.1, .2, .3, .6]], [[1.1, 0, 0, -.1]], [[.1, .1, .1, .1]],
])
def test_invalid_output_is_not_normalized(runtime, values):
    np, files, frame, state = runtime
    state.output = np.array(values)
    with pytest.raises(ContentCheckError, match=r"输出|分数"):
        model.score_frames([frame], files)


@pytest.mark.parametrize("boundary", ["init_error", "run_error"])
def test_known_ort_error_is_wrapped(runtime, boundary):
    _, files, frame, state = runtime
    error = OrtFailure("invalid graph")
    setattr(state, boundary, error)
    with pytest.raises(ContentCheckError, match=r"模型.*invalid graph") as caught:
        model.score_frames([frame], files)
    assert caught.value.__cause__ is error


def test_runtime_programming_error_propagates(runtime):
    _, files, frame, state = runtime
    state.run_error = TypeError("bug")
    with pytest.raises(TypeError, match="bug"):
        model.score_frames([frame], files)


def test_bad_image_is_expected_failure(runtime):
    _, files, frame, _ = runtime
    frame.path.write_bytes(b"not an image")
    with pytest.raises(ContentCheckError, match=r"图片.*frame_000001"):
        model.score_frames([frame], files)


@pytest.mark.parametrize("dependency", ["numpy", "PIL", "onnxruntime"])
def test_missing_inference_dependency_has_install_guidance(monkeypatch, dependency, tmp_path):
    monkeypatch.setitem(sys.modules, dependency, None)
    files = ModelFiles(tmp_path / "model.onnx", tmp_path / "meta.json", "unused")
    with pytest.raises(ContentCheckError, match="uv sync --extra content-check"):
        model.score_frames([SampledFrame(tmp_path / "frame_000001.jpg", 0)], files)


def test_reload_does_not_import_optional_dependencies(monkeypatch):
    for name in ("numpy", "PIL", "onnxruntime", "huggingface_hub"):
        monkeypatch.setitem(sys.modules, name, None)
    importlib.reload(model)
