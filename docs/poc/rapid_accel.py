"""PROTOTYPE（可丢弃）：只识别每条字幕中间帧的新版 RapidOCR，对比 CPU 与 CoreML。

时间段取自已有的 Vision 原稿 SRT；每条 cue 只在中点取一帧（裁剪/缩放/灰度与 ocr.py 相同），
关掉方向分类。每种模式写 <out>/<mode>.srt，耗时写 <out>/timings.json。

用法：
  uv run --extra ocr --with "rapidocr>=3.7" python docs/poc/rapid_accel.py <视频> \
      --cues work/ocr-compare/vision-hans.srt -o work/ocr-accel --mode cpu --mode coreml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from tenmin.config import OcrConfig
from tenmin.ingest import ocr
from tenmin.ingest.srt_parser import load_srt
from tenmin.render import ffmpeg as ff


def _frames(video: Path, cues, cfg: OcrConfig) -> tuple[list[np.ndarray], int]:
    width, height = ff.probe_video_size(video, ffprobe=ff.FFPROBE)
    crop_h = max(2, int(height * (1 - cfg.crop_top)) // 2 * 2)
    out_h = max(2, round(ocr.OCR_WIDTH * crop_h / width / 2) * 2)
    vf = f"crop={width}:{crop_h}:0:{height - crop_h},scale={ocr.OCR_WIDTH}:{out_h},format=gray"
    frames = []
    for cue in cues:
        mid = (cue.start + cue.end) / 2
        raw = subprocess.run(
            [ff.FFMPEG, "-v", "error", "-ss", f"{mid:.3f}", "-i", str(video), "-frames:v", "1",
             "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True,
        ).stdout
        img = np.frombuffer(raw[: ocr.OCR_WIDTH * out_h], dtype=np.uint8).reshape(out_h, ocr.OCR_WIDTH)
        frames.append(np.ascontiguousarray(np.stack([img] * 3, axis=-1)))
    return frames, out_h


def _boxes(result, width: int, height: int) -> list[ocr.TextBox]:
    if result.boxes is None or result.txts is None:
        return []
    boxes = []
    for points, text in zip(result.boxes, result.txts, strict=True):
        xs, ys = points[:, 0], points[:, 1]
        x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
        boxes.append(ocr.TextBox(text=text, x=x0 / width, y=1 - y1 / height,
                                 width=(x1 - x0) / width, height=(y1 - y0) / height))
    return boxes


def _merge_lines(boxes: list[ocr.TextBox]) -> list[ocr.TextBox]:
    """把垂直方向重叠过半的框按 x 顺序拼成一整行（空格分隔），再交给居中过滤。"""
    lines: list[list[ocr.TextBox]] = []
    for box in sorted(boxes, key=lambda b: -(b.y + b.height / 2)):
        center = box.y + box.height / 2
        for line in lines:
            ref = line[0]
            if abs(center - (ref.y + ref.height / 2)) < max(ref.height, box.height) / 2:
                line.append(box)
                break
        else:
            lines.append([box])
    merged = []
    for line in lines:
        line.sort(key=lambda b: b.x)
        x0 = min(b.x for b in line)
        x1 = max(b.x + b.width for b in line)
        y0 = min(b.y for b in line)
        y1 = max(b.y + b.height for b in line)
        text = line[0].text
        for prev, box in zip(line, line[1:]):
            if box.x < prev.x + prev.width:  # 横向重叠：去掉重复识别出的首尾字
                k = next((n for n in range(min(len(text), len(box.text)), 0, -1)
                          if text.endswith(box.text[:n])), 0)
                text += box.text[k:] if k else " " + box.text
            else:
                text += " " + box.text
        merged.append(ocr.TextBox(text=text, x=x0, y=y0,
                                  width=x1 - x0, height=y1 - y0))
    return merged


def _ts(s: float) -> str:
    ms = round(s * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--cues", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--mode", action="append", choices=["cpu", "coreml"], required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from rapidocr import RapidOCR

    cfg = OcrConfig()
    cues = load_srt(args.cues)
    started = time.perf_counter()
    frames, out_h = _frames(args.video, cues, cfg)
    timings = {"extract": {"seconds": round(time.perf_counter() - started, 1), "frames": len(frames)}}
    print("extract", timings["extract"], flush=True)

    for mode in args.mode:
        params = {"EngineConfig.onnxruntime.use_coreml": True} if mode == "coreml" else {}
        engine = RapidOCR(params=params)
        engine(frames[0], use_cls=False)  # 预热：CoreML 首次编译模型不计入
        started = time.perf_counter()
        lines = []
        fallback = 0
        for cue, img in zip(cues, frames, strict=True):
            boxes = _merge_lines(_boxes(engine(img, use_cls=False), ocr.OCR_WIDTH, out_h))
            text = ocr.frame_text(boxes, center_tolerance=cfg.center_tolerance, min_center_y=0.0)
            if not text:
                text = cue.text
                fallback += 1
            lines.append(f"{len(lines) + 1}\n{_ts(cue.start)} --> {_ts(cue.end)}\n{text or ''}\n\n")
        timings[mode] = {"seconds": round(time.perf_counter() - started, 1), "cues": len(lines),
                         "vision_fallback": fallback}
        (args.output / f"{mode}.srt").write_text("".join(lines), encoding="utf-8")
        print(mode, timings[mode], flush=True)

    (args.output / "timings.json").write_text(json.dumps(timings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
