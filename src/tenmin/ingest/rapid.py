"""CPU-only RapidOCR refinement of Vision cues, without changing their timeline."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from tenmin.config import OcrConfig
from tenmin.ingest.ocr import (
    OCR_WIDTH,
    OCRError,
    OCRUnavailableError,
    TextBox,
    frame_filter,
    frame_text,
)
from tenmin.models import RawCue
from tenmin.render import ffmpeg


def _build_engine():
    """Load optional libraries here; translate only their known operational errors.

    RapidOCR 3.10 loads models on first inference, not necessarily in __init__.
    Its ONNXRuntimeError wraps session.run failures; initialization instead exposes
    ORT's native exceptions. Generic ValueError/RuntimeError must escape unchanged.
    """
    try:
        from onnxruntime.capi import onnxruntime_pybind11_state as ort
        from rapidocr import RapidOCR
        from rapidocr.inference_engine.onnxruntime.main import ONNXRuntimeError
        from rapidocr.utils.download_file import DownloadFileException
    except ImportError as exc:
        raise OCRUnavailableError(
            "Vision + RapidOCR 需要可选依赖（Vision 只支持 macOS）。"
            "请运行 `uv sync --extra ocr --extra rapidocr` 后重试。"
        ) from exc

    known_errors = (
        DownloadFileException, ONNXRuntimeError,
        ort.Fail, ort.InvalidArgument, ort.NoSuchFile, ort.NoModel, ort.EngineError,
        ort.RuntimeException, ort.InvalidProtobuf, ort.ModelLoaded, ort.NotImplemented,
        ort.InvalidGraph, ort.EPFail, OSError,
    )
    params = {
        "Global.use_cls": False,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.use_coreml": False,
        "EngineConfig.onnxruntime.use_dml": False,
        "EngineConfig.onnxruntime.use_cann": False,
    }
    try:
        engine = RapidOCR(params=params)
    except known_errors as exc:
        raise OCRError(f"RapidOCR 模型初始化失败：{exc}") from exc

    def recognize(image, *, use_cls=False):
        try:
            return engine(image, use_cls=use_cls)
        except known_errors as exc:
            raise OCRError(f"RapidOCR 模型加载或推理失败：{exc}") from exc

    return recognize


def _midpoint_frame(
    video: Path, midpoint: float, *, chain: str, height: int, ffmpeg_path: str
) -> np.ndarray:
    """Read exactly one gray frame through the shared media error/cleanup layer."""
    image = None
    frame_size = OCR_WIDTH * height

    def on_frame(raw: bytes) -> None:
        nonlocal image
        if image is not None or len(raw) != frame_size:
            raise OCRError(f"{video.name} 中点抽帧失败：帧数量或字节数不正确。")
        gray = np.frombuffer(raw, dtype=np.uint8).reshape(height, OCR_WIDTH)
        image = np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))

    ffmpeg.run_raw_frames(
        [
            "-hide_banner", "-v", "error", "-ss", f"{midpoint:.3f}",
            "-i", str(video), "-map", "0:v:0", "-frames:v", "1",
            "-vf", chain, "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        frame_size=frame_size, on_frame=on_frame, ffmpeg=ffmpeg_path,
    )
    if image is None:
        raise OCRError(f"{video.name} 在 {midpoint:.3f} 秒没有抽到中点帧。")
    return image


def _boxes(result: Any, *, width: int, height: int) -> list[TextBox]:
    """Convert RapidOCR pixel quadrilaterals to Vision's normalized bottom-up boxes."""
    if result.boxes is None or result.txts is None:
        return []
    boxes = []
    for points, text in zip(result.boxes, result.txts, strict=True):
        xs, ys = points[:, 0], points[:, 1]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        boxes.append(TextBox(text, x0 / width, 1 - y1 / height,
                             (x1 - x0) / width, (y1 - y0) / height))
    return boxes


def _merge_lines(boxes: list[TextBox]) -> list[TextBox]:
    """Group row centers before filtering; deduplicate only overlapping text ends."""
    rows: list[list[TextBox]] = []
    for box in sorted(boxes, key=lambda b: -(b.y + b.height / 2)):
        center = box.y + box.height / 2
        for row in rows:
            ref = row[0]
            if abs(center - (ref.y + ref.height / 2)) < min(ref.height, box.height) / 2:
                row.append(box)
                break
        else:
            rows.append([box])
    merged = []
    for row in rows:
        row.sort(key=lambda b: b.x)
        x0 = min(b.x for b in row)
        x1 = max(b.x + b.width for b in row)
        y0 = min(b.y for b in row)
        y1 = max(b.y + b.height for b in row)
        text = row[0].text
        for previous, box in pairwise(row):
            overlap = 0
            if box.x < previous.x + previous.width:
                overlap = next(
                    (n for n in range(min(len(text), len(box.text)), 0, -1)
                     if text.endswith(box.text[:n])), 0,
                )
            text += box.text[overlap:] if overlap else " " + box.text
        merged.append(TextBox(text, x0, y0, x1 - x0, y1 - y0))
    return merged


def refine_cues(
    video: Path, cues: list[RawCue], *, ocr: OcrConfig,
    ffmpeg_path: str, ffprobe_path: str,
) -> list[RawCue]:
    """Refine each cue's midpoint once, falling back only for empty filtered text."""
    if not cues:
        print("  RapidOCR 复识别完成，0 条，空文本回落 0 条")
        return []
    engine = _build_engine()
    width, height = ffmpeg.probe_video_size(video, ffprobe=ffprobe_path)
    # step=1 selects every frame; -frames:v 1 bounds each midpoint extraction.
    chain, out_height = frame_filter(width=width, height=height, step=1, crop_top=ocr.crop_top)
    crop_height = max(2, int(height * (1 - ocr.crop_top)) // 2 * 2)
    min_center_y = 0.0 if ocr.crop_bottom is None else height * (1 - ocr.crop_bottom) / crop_height
    refined = []
    fallback = 0
    for cue in cues:
        image = _midpoint_frame(
            video, (cue.start + cue.end) / 2, chain=chain, height=out_height,
            ffmpeg_path=ffmpeg_path,
        )
        result = engine(image, use_cls=False)
        boxes = _merge_lines(_boxes(result, width=OCR_WIDTH, height=out_height))
        text = frame_text(boxes, center_tolerance=ocr.center_tolerance, min_center_y=min_center_y)
        if not text:
            text = cue.text
            fallback += 1
        # RawCue is a Pydantic model, not a dataclass; copy all metadata intact.
        refined.append(cue.model_copy(update={"text": text}))
    print(f"  RapidOCR 复识别完成，{len(refined)} 条，空文本回落 {fallback} 条")
    return refined
