"""PROTOTYPE：对比几个 mlx-whisper 日语模型的转写耗时与结果。

用法：
    uv run --extra asr python docs/poc/asr_compare.py <视频> -o work/asr-compare \
        --model mlx-community/whisper-large-v3-turbo \
        --model mlx-community/whisper-large-v3-mlx \
        --model kaiinui/kotoba-whisper-v2.0-mlx

每个模型写 <out>/<模型短名>.ja.asr.srt，timings.json 记录耗时与条数。
已存在的 SRT 不重跑（删掉即可重跑）。
"""

import argparse
import json
import time
from pathlib import Path

from tenmin.config import AsrConfig
from tenmin.ingest import asr
from tenmin.ingest.srt_parser import load_srt
from tenmin.render import ffmpeg as ff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timings_path = args.output / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.is_file() else {}
    for model in args.model:
        name = model.split("/")[-1]
        dest = args.output / f"{name}.ja.asr.srt"
        if not dest.is_file():
            began = time.perf_counter()
            asr.transcribe(
                args.video, dest, asr=AsrConfig(model=model), ffmpeg_path=ff.FFMPEG
            )
            timings[name] = {"model": model, "seconds": round(time.perf_counter() - began, 1)}
        timings.setdefault(name, {"model": model})["cues"] = len(load_srt(dest))
        timings_path.write_text(json.dumps(timings, ensure_ascii=False, indent=2))
        print(name, timings[name], flush=True)


if __name__ == "__main__":
    main()
