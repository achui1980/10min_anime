"""RapidOCR cue refinement: text changes must not change the Vision timeline."""

import builtins
import importlib
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tenmin.config import OcrConfig
from tenmin.ingest import rapid
from tenmin.ingest.ocr import OCRError, OCRUnavailableError, TextBox, frame_text
from tenmin.models import RawCue


def output(*items):
    return SimpleNamespace(
        boxes=np.array([points for points, _ in items]),
        txts=[text for _, text in items],
    )


def rectangle(text, x0=400, y0=20, x1=880, y1=60):
    return ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text)


def test_refine_preserves_metadata_order_and_empty_fallback(monkeypatch, capsys):
    cues = [
        RawCue(idx=7, src_idx=42, start=3.1, end=4.9, text="己经", clamped=True),
        RawCue(idx=2, src_idx=None, start=1, end=2, text="原文", clamped=False),
    ]
    original = deepcopy(cues)
    results = iter([output(rectangle("已经")), SimpleNamespace(boxes=None, txts=None)])
    calls = []
    built = []

    def engine(image, *, use_cls):
        assert use_cls is False
        calls.append(image)
        return next(results)

    def build():
        built.append(True)
        return engine

    mids = []

    def frame(video, midpoint, *, chain, height, ffmpeg_path):
        mids.append(midpoint)
        return np.zeros((height, 1280, 3), dtype=np.uint8)

    monkeypatch.setattr(rapid, "_build_engine", build)
    monkeypatch.setattr(rapid, "_midpoint_frame", frame)
    monkeypatch.setattr(rapid.ffmpeg, "probe_video_size", lambda *a, **kw: (1920, 1080))
    result = rapid.refine_cues(
        Path("video.mp4"), cues, ocr=OcrConfig(), ffmpeg_path="ff", ffprobe_path="probe"
    )
    assert result[0].text == "已经"
    assert result[1].text == cues[1].text
    assert [(c.start, c.end, c.idx, c.src_idx, c.clamped) for c in result] == [
        (c.start, c.end, c.idx, c.src_idx, c.clamped) for c in cues
    ]
    assert cues == original
    assert len(built) == 1
    assert len(calls) == 2
    assert mids == [4, 1.5]
    assert "2 条" in capsys.readouterr().out


def text(result, *, min_center_y=0, tolerance=0.2):
    return frame_text(
        rapid._merge_lines(rapid._boxes(result, width=1280, height=200)),
        center_tolerance=tolerance, min_center_y=min_center_y,
    )


def test_pixel_coordinates_use_bottom_up_y():
    box = rapid._boxes(output(rectangle("字", 320, 20, 960, 60)), width=1280, height=200)[0]
    assert box == TextBox("字", 0.25, 0.7, 0.5, 0.2)


@pytest.mark.parametrize(("items", "expected"), [
    ([rectangle("已经", 350, 20, 600, 60), rectangle("到了", 650, 20, 930, 60)], "已经 到了"),
    ([rectangle("给我一", 350, 20, 700, 60), rectangle("一个月饼", 650, 20, 930, 60)],
     "给我一个月饼"),
    ([rectangle("喂", 350, 20, 600, 60), rectangle("你们", 650, 20, 930, 60)], "喂 你们"),
    ([rectangle("上行", 400, 20, 880, 60), rectangle("下行", 400, 100, 880, 140)],
     "上行\n下行"),
    ([rectangle("给我", 350, 20, 700, 60), rectangle("月饼", 650, 20, 930, 60)], "给我 月饼"),
    ([rectangle("一", 350, 20, 600, 60), rectangle("一", 650, 20, 930, 60)], "一 一"),
])
def test_rows_and_only_geometrically_overlapping_text_deduplicated(items, expected):
    assert text(output(*reversed(items))) == expected


def test_center_bottom_and_cjk_filters():
    result = output(rectangle("边字", 0, 65, 100, 85), rectangle("中文"),
                    rectangle("日文漢字", 400, 150, 880, 190),
                    rectangle("English", 400, 80, 880, 120))
    assert text(result, min_center_y=0.4) == "中文"


@pytest.mark.parametrize(("boxes", "txts"), [(None, None), (None, ["字"]), ([], None)])
def test_missing_results_are_empty(boxes, txts):
    assert rapid._boxes(SimpleNamespace(boxes=boxes, txts=txts), width=1280, height=200) == []


def test_refine_uses_rounded_crop_bottom_boundary(monkeypatch):
    cfg = OcrConfig(crop_top=0.72, crop_bottom=0.9)
    monkeypatch.setattr(rapid.ffmpeg, "probe_video_size", lambda *a, **kw: (1920, 1081))
    def midpoint(*a, **kw):
        assert "crop=1920:302:0:779,scale=1280:202,format=gray" in kw["chain"]
        assert kw["height"] == 202
        assert kw["ffmpeg_path"] == "ff"
        return np.zeros((202, 1280, 3))

    monkeypatch.setattr(rapid, "_midpoint_frame", midpoint)
    monkeypatch.setattr(rapid, "_build_engine", lambda: lambda *a, **kw: output(rectangle("字")))
    seen = []
    original = rapid.frame_text

    def capture(boxes, **kwargs):
        seen.append(kwargs["min_center_y"])
        return original(boxes, **kwargs)

    monkeypatch.setattr(rapid, "frame_text", capture)
    rapid.refine_cues(Path("v"), [RawCue(idx=1, start=0, end=1, text="旧")],
                      ocr=cfg, ffmpeg_path="ff", ffprobe_path="probe")
    assert seen == [pytest.approx(1081 * 0.1 / 302)]


def test_midpoint_frame_uses_media_wrapper_and_contiguous_gray_triplets(monkeypatch):
    seen = []

    def run(args, *, frame_size, on_frame, ffmpeg):
        seen.append((args, frame_size, ffmpeg))
        on_frame(bytes([23]) * frame_size)
        return ""

    monkeypatch.setattr(rapid.ffmpeg, "run_raw_frames", run)
    image = rapid._midpoint_frame(Path("v.mp4"), 1.25, chain="crop=test", height=2,
                                  ffmpeg_path="custom-ff")
    assert image.shape == (2, 1280, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert np.all(image == 23)
    args, size, binary = seen[0]
    assert args[args.index("-ss") + 1] == "1.250"
    assert args[args.index("-frames:v") + 1] == "1"
    assert args[args.index("-vf") + 1] == "crop=test"
    assert size == 2560
    assert binary == "custom-ff"


@pytest.mark.parametrize("frames", [[], [b"short"], [bytes(2560), bytes(2560)]])
def test_bad_or_missing_or_multiple_frames_fail(monkeypatch, frames):
    def run(*a, on_frame, **kw):
        for frame in frames:
            on_frame(frame)
        return ""

    monkeypatch.setattr(rapid.ffmpeg, "run_raw_frames", run)
    with pytest.raises(OCRError, match="帧"):
        rapid._midpoint_frame(Path("v"), 0.5, chain="gray", height=2, ffmpeg_path="ff")


def test_import_is_lazy_and_missing_dependency_is_actionable(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.startswith(("rapidocr", "onnxruntime")):
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    importlib.reload(rapid)
    with pytest.raises(OCRUnavailableError, match="--extra ocr --extra rapidocr"):
        rapid._build_engine()


@pytest.fixture
def backend(monkeypatch):
    # Stub only optional IO/inference boundaries; keep exception identities distinct.
    class DownloadFileException(Exception):
        pass

    class ONNXRuntimeError(Exception):
        pass

    class InvalidGraph(Exception):
        pass

    modules = {
        "rapidocr": SimpleNamespace(RapidOCR=None),
        "rapidocr.utils.download_file": SimpleNamespace(
            DownloadFileException=DownloadFileException,
        ),
        "rapidocr.inference_engine.onnxruntime.main": SimpleNamespace(
            ONNXRuntimeError=ONNXRuntimeError,
        ),
        "onnxruntime.capi.onnxruntime_pybind11_state": SimpleNamespace(
            Fail=InvalidGraph, InvalidArgument=InvalidGraph, NoSuchFile=InvalidGraph,
            NoModel=InvalidGraph, EngineError=InvalidGraph, RuntimeException=InvalidGraph,
            InvalidProtobuf=InvalidGraph, ModelLoaded=InvalidGraph, NotImplemented=InvalidGraph,
            InvalidGraph=InvalidGraph, EPFail=InvalidGraph,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "onnxruntime.capi", SimpleNamespace(
        onnxruntime_pybind11_state=modules["onnxruntime.capi.onnxruntime_pybind11_state"],
    ))
    return modules["rapidocr"], [DownloadFileException, ONNXRuntimeError, InvalidGraph]


def test_build_engine_cpu_only_and_cls_disabled(backend):
    module, _ = backend
    params_seen = []

    def constructor(*, params):
        params_seen.append(params)
        return lambda image, **kw: kw

    module.RapidOCR = constructor
    engine = rapid._build_engine()
    assert engine(None, use_cls=False) == {"use_cls": False}
    params = params_seen[0]
    for provider in ("cuda", "coreml", "dml", "cann"):
        assert params[f"EngineConfig.onnxruntime.use_{provider}"] is False
    assert params["Global.use_cls"] is False


@pytest.mark.parametrize("stage", ["init", "infer"])
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_known_backend_errors_are_wrapped_not_fallback(backend, stage, kind):
    module, errors = backend
    error = errors[kind]("broken model")

    def failing(*a, **kw):
        raise error

    module.RapidOCR = failing if stage == "init" else lambda **kw: failing
    with pytest.raises(OCRError, match="RapidOCR") as caught:
        engine = rapid._build_engine()
        engine(None, use_cls=False)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("stage", ["init", "infer"])
@pytest.mark.parametrize("error", [ValueError("bug"), RuntimeError("bug"), KeyboardInterrupt()])
def test_programming_errors_and_interruptions_propagate(backend, stage, error):
    module, _ = backend

    def failing(*a, **kw):
        raise error

    module.RapidOCR = failing if stage == "init" else lambda **kw: failing
    with pytest.raises(type(error)) as caught:
        engine = rapid._build_engine()
        engine(None, use_cls=False)
    assert caught.value is error


def test_real_short_video_midpoint_crop(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe unavailable")
    video = tmp_path / "sample.mp4"
    rapid.ffmpeg.run([
        "-v", "error", "-f", "lavfi", "-i",
        "nullsrc=size=320x240:rate=10,"
        "geq=lum='if(lt(Y,172),16,if(lt(T,1),16,if(lt(T,2),235,81)))':cb=128:cr=128",
        "-t", "3", "-c:v", "mpeg4", str(video),
    ])
    width, height = rapid.ffmpeg.probe_video_size(video)
    chain, out_height = rapid.frame_filter(width=width, height=height, step=1, crop_top=0.72)
    # The upper picture stays black; only the subtitle band changes black → white
    # → gray. A wrong crop or a seek to either cue endpoint cannot pass.
    for timestamp, lower, upper in [(0.5, 0, 10), (1.5, 245, 256), (2.5, 65, 90)]:
        image = rapid._midpoint_frame(video, timestamp, chain=chain, height=out_height,
                                      ffmpeg_path="ffmpeg")
        assert image.shape == (264, 1280, 3)
        assert image.flags.c_contiguous
        assert np.all(image == image[:, :, :1])
        assert lower <= image.mean() < upper

    def engine(image, *, use_cls):
        assert use_cls is False
        assert image.shape == (264, 1280, 3)
        assert image.mean() > 245
        return output(rectangle("中点"))

    # Exercise the real facade-to-extractor path, stubbing inference only.
    monkeypatch.setattr(rapid, "_build_engine", lambda: engine)
    cue = RawCue(idx=9, src_idx=7, start=0.5, end=2.5, text="原文", clamped=True)
    result = rapid.refine_cues(video, [cue], ocr=OcrConfig(),
                               ffmpeg_path="ffmpeg", ffprobe_path="ffprobe")
    assert result == [cue.model_copy(update={"text": "中点"})]


def test_media_error_is_not_text_fallback(monkeypatch):
    error = rapid.ffmpeg.FFmpegError("bad decode")

    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(rapid.ffmpeg, "run_raw_frames", fail)
    with pytest.raises(rapid.ffmpeg.FFmpegError) as caught:
        rapid._midpoint_frame(Path("v"), 0.5, chain="gray", height=2, ffmpeg_path="ff")
    assert caught.value is error
