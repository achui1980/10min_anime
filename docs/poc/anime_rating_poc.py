# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy>=2,<3", "pillow>=11,<13", "onnxruntime>=1.20,<2", "huggingface-hub>=0.30,<2"]
# ///
"""Throwaway POC: offline anime frame rating, NOT a validated moderation system."""

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "deepghs/anime_dbrating"
REVISION = "7af21db648acdeb74f5c334abda9dd7403407b3c"
MODELS = {
    "swin": "swinv2pv3_v0_ls0.2",
    "mobile": "mobilenetv3_large_100_v0_ls0.2",
    "caformer": "caformer_s36_v0_ls0.2",
}


def extract_frames(args, frames_dir):
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", str(args.start),
               "-i", str(args.video), "-map", "0:v:0", "-an"]
    if args.duration is not None:
        command += ["-t", str(args.duration)]
    # select retains sampled frame PTS, unlike fps which duplicates/re-times frames.
    filters = (f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{args.interval})',"
               "showinfo,scale=640:-2")
    command += ["-vf", filters, "-fps_mode", "vfr", "-q:v", "2",
                str(frames_dir / "frame_%06d.jpg")]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    timestamps = [float(value) + args.start for value in re.findall(
        r"\[Parsed_showinfo[^\]]*\].*?\bn:\s*\d+.*?\bpts_time:([-+\d.eE]+)",
        result.stderr,
    )]
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames or len(frames) != len(timestamps):
        raise RuntimeError(f"Frame/PTS mismatch: {len(frames)} / {len(timestamps)}")
    return list(zip(frames, timestamps, strict=True))


def write_report(destination, payload):
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>动漫分级 POC</title>
<style>
body{font-family:system-ui;margin:24px;background:#f6f6f6;color:#222}
label{display:inline-block;margin:8px}input[type=number]{width:75px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
article{background:white;padding:12px;border:1px solid #ddd}img{width:100%}
table{width:100%;font-variant-numeric:tabular-nums}td:last-child{text-align:right}
button{padding:6px 14px;margin:12px}small{color:#555}
</style>
<h1>动漫分级 POC · 本地模型</h1>
<p>仅为图片评级粗筛，候选 ≠ 色情。请对照截图人工判断；未采样到的画面无法评估。</p>
<p id="summary"></p>
<label>sensitive 阈值 <input id="s" type="number" min="0" max="1" step="0.05" value="0.5"></label>
<label>questionable + explicit 阈值 <input id="a" type="number" min="0" max="1" step="0.05" value="0.5"></label>
<label><input id="all" type="checkbox"> 显示所有采样帧（检查漏报）</label>
<label>排序 <select id="sort"><option value="risk">敏感分数优先</option><option value="time">时间顺序</option></select></label>
<p id="count"></p><div id="segments"></div><div class="grid" id="grid"></div>
<button id="more">显示更多</button>
<script>
const report=__DATA__;
const labels={general:'普通',sensitive:'敏感',questionable:'成人倾向',explicit:'露骨'};
const $=id=>document.getElementById(id);
const stamp=t=>`${Math.floor(t/60).toString().padStart(2,'0')}:${(t%60).toFixed(2).padStart(5,'0')}`;
const risk=f=>Math.max(f.scores.sensitive,f.scores.questionable+f.scores.explicit);
let limit=60;
$('summary').textContent=`${report.video} | ${report.model} | ${report.frames.length} 帧 | 间隔 ${report.interval}s | 提取 ${report.extraction_seconds.toFixed(1)}s / 推理 ${report.inference_seconds.toFixed(1)}s | ${report.providers.join(', ')}`;
function render(){
 const candidates=report.frames.filter(f=>f.scores.sensitive>=Number($('s').value)||f.scores.questionable+f.scores.explicit>=Number($('a').value));
 const groups=[];
 for(const f of candidates){
  const previous=groups.at(-1);
  if(previous&&f.time-previous.last<=report.interval*1.5){previous.last=f.time;previous.count++;}
  else groups.push({first:f.time,last:f.time,count:1});
 }
 $('segments').textContent='候选采样点范围（非精确片段边界）：'+(groups.map(g=>`${stamp(g.first)}–${stamp(g.last)} (${g.count}帧)`).join('；')||'无');
 const shown=[...($('all').checked?report.frames:candidates)];
 shown.sort($('sort').value==='time'?(x,y)=>x.time-y.time:(x,y)=>risk(y)-risk(x));
 $('count').textContent=`候选 ${candidates.length}/${report.frames.length} 帧；当前显示 ${Math.min(limit,shown.length)}/${shown.length}。分数不是经过校准的色情概率。`;
 $('grid').replaceChildren();
 for(const f of shown.slice(0,limit)){
  const card=document.createElement('article');
  const img=document.createElement('img');img.src=f.image;img.loading='lazy';
  const title=document.createElement('h3');title.textContent=`${stamp(f.time)} · ${labels[f.label]}`;
  const table=document.createElement('table');
  for(const [label,score] of Object.entries(f.scores)){
   const row=table.insertRow();row.insertCell().textContent=label;row.insertCell().textContent=score.toFixed(4);
  }
  card.append(img,title,table);$('grid').append(card);
 }
 $('more').hidden=limit>=shown.length;
}
for(const id of ['s','a','all','sort'])$(id).addEventListener('change',()=>{limit=60;render();});
$('more').onclick=()=>{limit+=60;render();};render();
</script></html>""".replace("__DATA__", data)
    (destination / "report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="swin")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--offline", action="store_true", help="Use cached model only")
    args = parser.parse_args()
    if (not args.video.is_file() or args.interval <= 0 or args.start < 0
            or (args.duration is not None and args.duration <= 0)):
        parser.error("Video must exist; interval/duration must be positive; start >= 0")
    # Each run has its own directory, so old frames cannot contaminate the report.
    args.output.mkdir(parents=True, exist_ok=False)
    frames_dir = args.output / "frames"
    frames_dir.mkdir()
    model_name = MODELS[args.model]
    print(f"Loading {REPO}/{model_name} (offline={args.offline})", flush=True)
    files = {name: hf_hub_download(REPO, f"{model_name}/{name}", revision=REVISION,
                                  local_files_only=args.offline)
             for name in ("model.onnx", "meta.json")}
    labels = json.loads(Path(files["meta.json"]).read_text())["labels"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(files["model.onnx"], sess_options=options,
                                   providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    _, channels, height, width = input_info.shape
    assert channels == 3
    # Official imgutils defaults to 384x384 for dynamic spatial dimensions.
    if not isinstance(height, int) or not isinstance(width, int):
        height = width = 384
    print(f"Input {input_info.shape}; extracting frames…", flush=True)
    started = time.perf_counter()
    sampled = extract_frames(args, frames_dir)
    extraction_seconds = time.perf_counter() - started
    results = []
    started = time.perf_counter()
    for index, (frame, timestamp) in enumerate(sampled, 1):
        with Image.open(frame) as image:
            # Match imgutils.generic.classify._img_encode: RGB, bilinear, (x-.5)/.5.
            array = np.asarray(image.convert("RGB").resize((width, height),
                               Image.Resampling.BILINEAR), dtype=np.float32) / 255
        tensor = ((array.transpose(2, 0, 1) - 0.5) / 0.5)[None].copy()
        scores = session.run(["output"], {input_info.name: tensor})[0][0]
        if not np.isfinite(scores).all() or not np.isclose(scores.sum(), 1, atol=1e-3):
            raise RuntimeError(f"Unexpected model output: {scores}")
        score_map = dict(zip(labels, map(float, scores), strict=True))
        results.append({"time": timestamp, "image": str(frame.relative_to(args.output)),
                        "label": labels[int(np.argmax(scores))], "scores": score_map})
        if index == 1 or index % 25 == 0 or index == len(sampled):
            print(f"{index}/{len(sampled)} frames; {time.perf_counter()-started:.1f}s", flush=True)
    inference_seconds = time.perf_counter() - started
    payload = {
        "prototype": True, "video": str(args.video.resolve()), "model": model_name,
        "repository": REPO, "revision": REVISION,
        "model_sha256": hashlib.sha256(Path(files["model.onnx"]).read_bytes()).hexdigest(),
        "providers": session.get_providers(), "interval": args.interval,
        "start": args.start, "duration": args.duration,
        "extraction_seconds": extraction_seconds, "inference_seconds": inference_seconds,
        "counts": dict(Counter(frame["label"] for frame in results)), "frames": results,
    }
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output, payload)
    print(json.dumps({key: payload[key] for key in (
        "counts", "extraction_seconds", "inference_seconds")}, ensure_ascii=False), flush=True)
    print(f"Report: {args.output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
