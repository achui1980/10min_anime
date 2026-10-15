"""PROTOTYPE（可丢弃）：同一抽帧/过滤/合帧管线下，对比不同单帧 OCR 引擎。

用法：
  uv run --extra ocr --with rapidocr_onnxruntime python docs/poc/ocr_engine_compare.py \
      <视频> -o work/ocr-compare --engine vision-hans --engine rapid
引擎：vision-hant / vision-hans / rapid。每个引擎写 <out>/<engine>.srt 与耗时。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tenmin.config import OcrConfig
from tenmin.ingest import ocr
from tenmin.render import ffmpeg as ff

_rapid = None


def _rapid_recognize(frame: bytes, *, width: int, height: int, language: str) -> list[ocr.TextBox]:
    global _rapid
    if _rapid is None:
        from rapidocr_onnxruntime import RapidOCR

        _rapid = RapidOCR()
    img = np.frombuffer(frame, dtype=np.uint8).reshape(height, width)
    img = np.ascontiguousarray(np.stack([img] * 3, axis=-1))
    result, _ = _rapid(img)
    boxes = []
    for points, text, _score in result or []:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        boxes.append(
            ocr.TextBox(
                text=text,
                x=x0 / width,
                y=1 - y1 / height,  # 转成 Vision 约定：y 自下而上，取框下沿
                width=(x1 - x0) / width,
                height=(y1 - y0) / height,
            )
        )
    return boxes


def _srt(cues) -> str:
    def ts(s: float) -> str:
        ms = round(s * 1000)
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    return "".join(f"{i}\n{ts(c.start)} --> {ts(c.end)}\n{c.text}\n\n" for i, c in enumerate(cues, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--engine", action="append", required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = ocr._recognize
    timings = {}
    for engine in args.engine:
        language = "zh-Hant" if engine == "vision-hant" else "zh-Hans"
        ocr._recognize = _rapid_recognize if engine == "rapid" else original
        started = time.perf_counter()
        cues = ocr.recognize_cues(
            args.video,
            ocr=OcrConfig(language=language),
            ffmpeg_path=ff.FFMPEG,
            ffprobe_path=ff.FFPROBE,
        )
        timings[engine] = {"seconds": round(time.perf_counter() - started, 1), "cues": len(cues)}
        (args.output / f"{engine}.srt").write_text(_srt(cues), encoding="utf-8")
        print(engine, timings[engine], flush=True)
    timings_path = args.output / "timings.json"
    old = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    timings_path.write_text(json.dumps(old | timings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
