"""PROTOTYPE（可丢弃）：两份同时间轴 OCR 原稿，只列文字不一致的行，附中点截图与字级 diff。

用法：
  uv run python docs/poc/ocr_diff_report.py <视频> work/ocr-compare/vision-hans.srt \
      work/ocr-accel3/cpu.srt -o work/ocr-diff
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from tenmin.ingest import ocr_audit
from tenmin.ingest.srt_parser import load_srt
from tenmin.render import ffmpeg as ff

PAGE = """<!doctype html><meta charset="utf-8"><title>OCR 对比</title>
<style>body{font:15px sans-serif;margin:20px;background:#f6f6f6}
.c{background:#fff;margin:0 0 14px;padding:10px;border-radius:6px}
img{max-width:960px;display:block;margin:6px 0}del{background:#fcc}ins{background:#cfc;text-decoration:none}
.t{color:#888}.b{margin:2px 0}</style>
<h2>__TITLE__</h2><div id="list"></div><script>
const DATA = __DATA__;
const list = document.getElementById('list');
function line(label, diff, keep) {
  const d = document.createElement('div'); d.className = 'b';
  d.append(label + '：');
  for (const [op, t] of diff) {
    if (op === 'equal') d.append(t);
    else if (op === keep) { const e = document.createElement(op === 'delete' ? 'del' : 'ins'); e.textContent = t; d.append(e); }
  }
  return d;
}
for (const it of DATA.items) {
  const c = document.createElement('div'); c.className = 'c';
  const t = document.createElement('div'); t.className = 't'; t.textContent = `#${it.id} ${it.time}`;
  const img = document.createElement('img'); img.src = it.image; img.loading = 'lazy';
  c.append(t, img, line(DATA.a, it.diff, 'delete'), line(DATA.b, it.diff, 'insert'));
  list.append(c);
}
</script>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    frames = args.output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    a, b = load_srt(args.a), load_srt(args.b)
    if len(a) != len(b):
        raise SystemExit(f"条数不同：{len(a)} vs {len(b)}")
    items = []
    for i, (x, y) in enumerate(zip(a, b, strict=True), 1):
        if x.text == y.text:
            continue
        mid = (x.start + x.end) / 2
        dest = frames / f"{i:04d}.jpg"
        if not dest.exists():
            ocr_audit._extract_frame(args.video, mid, dest, ffmpeg=ff.FFMPEG)
        items.append({"id": i, "time": f"{int(mid // 60):02d}:{mid % 60:05.2f}",
                      "image": f"frames/{dest.name}", "diff": ocr_audit.char_diff(x.text, y.text)})
    data = json.dumps({"a": args.a.stem, "b": args.b.stem, "items": items}, ensure_ascii=False)
    title = f"{args.a.name} vs {args.b.name}：不一致 {len(items)}/{len(a)}"
    page = PAGE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data.replace("<", "\\u003c"))
    (args.output / "report.html").write_text(page, encoding="utf-8")
    print(title, args.output / "report.html")


if __name__ == "__main__":
    main()
