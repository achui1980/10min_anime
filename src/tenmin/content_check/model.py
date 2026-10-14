"""固定本地图片评级模型的缓存加载与逐帧推理。"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from tenmin.content_check.models import (
    ContentCheckError,
    FrameScore,
    ModelFiles,
    SampledFrame,
    Scores,
)

REPO = "deepghs/anime_dbrating"
REVISION = "7af21db648acdeb74f5c334abda9dd7403407b3c"
MODEL = "mobilenetv3_large_100_v0_ls0.2"
MODEL_SHA256 = "c7d6fd0cd71c48616fa87fe2bca87f8e1775e9a3b96d12797e2144fca3543362"
PREPROCESSING_VERSION = "rgb-bilinear-chw-v1"
SAMPLING_VERSION = "select-relative-pts-dar-v1"
_LABELS = ("general", "sensitive", "questionable", "explicit")
_INSTALL = "内容检查依赖不可用；请运行 uv sync --extra content-check"
# hf-xet 1.6 的 TaskError 丢失异常类型；仅识别已知传输/文件错误的 Display 前缀，
# 不把同属 RuntimeError 的 panic、内部错误、取消当成可预期下载失败。
_XET_DOWNLOAD_ERRORS = (
    "Task error: File reconstruction error: CAS Client Error: Request error: ",
    "Task error: File reconstruction error: CAS Client Error: Request middleware error: ",
    "Task error: File reconstruction error: CAS Client Error: IO error: ",
    "Task error: File reconstruction error: IO Error: ",
    "Task error: File reconstruction error: Corrupted Reconstruction: ",
    "Task error: File size mismatch: expected ",
)


def load_model_files(*, offline: bool) -> ModelFiles:
    """每份文件先查固定 revision 的缓存；只有在线缓存缺失才下载。"""
    try:
        from httpx import TransportError
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    except ImportError as error:
        raise ContentCheckError(_INSTALL) from error

    paths = []
    for name in ("model.onnx", "meta.json"):
        filename = f"{MODEL}/{name}"
        try:
            try:
                path = hf_hub_download(REPO, filename, revision=REVISION, local_files_only=True)
            except LocalEntryNotFoundError as error:
                if offline:
                    raise ContentCheckError("离线模型缺失；先联网运行检查以下载模型") from error
                try:
                    path = hf_hub_download(
                        REPO, filename, revision=REVISION, local_files_only=False,
                    )
                except RuntimeError as error:
                    if not str(error).startswith(_XET_DOWNLOAD_ERRORS):
                        raise
                    raise ContentCheckError(f"模型文件下载失败（{filename}）：{error}") from error
        except (HfHubHTTPError, TransportError, OSError) as error:
            raise ContentCheckError(f"模型文件下载失败（{filename}）：{error}") from error
        paths.append(Path(path))

    weights, metadata = paths
    try:
        with weights.open("rb", encoding=None) as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        raw_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContentCheckError(f"模型文件读取或元数据解析失败：{error}") from error
    if digest != MODEL_SHA256:
        raise ContentCheckError("模型权重摘要不符；请检查或清理损坏的模型缓存后联网重试")
    if not isinstance(raw_metadata, dict) or raw_metadata.get("labels") != list(_LABELS):
        raise ContentCheckError(
            "模型元数据 labels 不符；要求 general/sensitive/questionable/explicit 类序",
        )
    return ModelFiles(weights=weights, metadata=metadata, sha256=digest)


def score_frames(frames: Sequence[SampledFrame], files: ModelFiles) -> list[FrameScore]:
    """CPU 单会话逐帧推理；保留模型原始分数，不执行 softmax。"""
    try:
        import numpy as np
        import onnxruntime as ort
        from onnxruntime.capi.onnxruntime_pybind11_state import (
            EngineError,
            EPFail,
            Fail,
            InvalidArgument,
            InvalidGraph,
            InvalidProtobuf,
            NoSuchFile,
            NotImplemented,
            RuntimeException,
        )
        from PIL import Image
        from PIL.Image import open as open_image
    except ImportError as error:
        raise ContentCheckError(_INSTALL) from error

    ort_errors = (EPFail, EngineError, Fail, InvalidArgument, InvalidGraph,
                  InvalidProtobuf, NoSuchFile, NotImplemented, RuntimeException)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    try:
        session = ort.InferenceSession(str(files.weights), sess_options=options,
                                       providers=["CPUExecutionProvider"])
    except ort_errors as error:
        raise ContentCheckError(f"模型加载失败：{error}") from error
    inputs = session.get_inputs()
    if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 3:
        raise ContentCheckError("模型输入必须为单个 rank-4 三通道 NCHW 张量")
    info = inputs[0]
    height, width = info.shape[2:]
    if any(isinstance(size, int) and size <= 0 for size in (height, width)):
        raise ContentCheckError("模型输入空间维度必须为正数")
    if not isinstance(height, int) or not isinstance(width, int):
        height = width = 384

    results = []
    for frame in frames:
        try:
            # Pillow 的 open 是二进制图片 API，别让文本 I/O 守卫误判。
            with open_image(frame.path) as image:
                array = np.asarray(image.convert("RGB").resize(
                    (width, height), Image.Resampling.BILINEAR,
                ), dtype=np.float32) / 255
        except (OSError, Image.DecompressionBombError) as error:
            raise ContentCheckError(f"图片解码失败（{frame.path}）：{error}") from error
        tensor = ((array.transpose(2, 0, 1) - .5) / .5)[None].copy()
        try:
            outputs = session.run(["output"], {info.name: tensor})
        except ort_errors as error:
            raise ContentCheckError(f"模型推理失败（{frame.path}）：{error}") from error
        if (len(outputs) != 1 or not isinstance(outputs[0], np.ndarray)
                or outputs[0].shape != (1, 4)
                or not np.issubdtype(outputs[0].dtype, np.floating)):
            raise ContentCheckError("模型输出必须为 (1, 4) 浮点分数张量")
        values = outputs[0][0]
        try:
            scores = Scores(**dict(zip(_LABELS, map(float, values), strict=True)))
        except ValidationError as error:
            raise ContentCheckError(f"模型输出分数无效（{frame.path}）：{error}") from error
        results.append(FrameScore(
            time=frame.time, image=f"frames/{frame.path.name}",
            label=_LABELS[int(np.argmax(values))], scores=scores,
        ))
    return results
