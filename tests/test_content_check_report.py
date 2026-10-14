import json
import re
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from tenmin.config import ContentCheckConfig
from tenmin.content_check.candidates import screen_frames
from tenmin.content_check.models import CacheIdentity, CheckManifest, FrameScore, Scores


@pytest.fixture
def valid_manifest():
    return CheckManifest(
        identity=CacheIdentity(
            video_path="/videos/E01.mp4", video_size=100, video_mtime_ns=123,
            repository="deepghs/anime_dbrating",
            revision="7af21db648acdeb74f5c334abda9dd7403407b3c",
            model="mobilenetv3_large_100_v0_ls0.2", model_sha256="a" * 64,
            preprocessing_version="rgb-bilinear-chw-v1",
            sampling_version="select-relative-pts-dar-v1", sample_interval=1,
        ),
        providers=["CPUExecutionProvider"], extraction_seconds=0, inference_seconds=0,
        sensitive_threshold=0.5, adult_threshold=0.5,
        frames=[FrameScore(time=0, image="frames/frame_000001.jpg", label="general",
                           scores=Scores(general=1, sensitive=0, questionable=0, explicit=0))],
        candidate_count=0, candidate_ranges=[],
    )


def render(manifest):
    from tenmin.content_check.report import render_report

    return render_report(manifest)


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = {}
        self.scripts = []
        self.current_script = None
        self.external = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.elements[attrs["id"]] = attrs
        if tag == "script":
            self.current_script = ""
        if tag in {"script", "link", "iframe", "img"}:
            self.external.extend(attrs[k] for k in ("src", "href") if k in attrs)

    def handle_data(self, data):
        if self.current_script is not None:
            self.current_script += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.scripts.append(self.current_script)
            self.current_script = None


# Execute the delivered page's actual script with a minimal DOM, without a browser,
# dependencies, network or model downloads. Task8 verifies real browser/layout behavior.
DOM = r"""
class Element {
  constructor(tag) { this.tag=tag; this.children=[]; this.value=''; this.checked=false;
    this.hidden=false; this.listeners={}; this.textContent=''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=[...nodes]; }
  insertRow() { const row=new Element('tr'); this.append(row); return row; }
  insertCell() { const cell=new Element('td'); this.append(cell); return cell; }
  addEventListener(event, fn) { this.listeners[event]=fn; }
}
const elements={};
for (const [id, attrs] of Object.entries(ATTRS)) {
  elements[id]=new Element('div'); elements[id].value=attrs.value || '';
}
elements.sort.value='risk';
const document={getElementById: id=>elements[id], createElement: tag=>new Element(tag)};
const snapshot=()=>({summary:elements.summary.textContent, count:elements.count.textContent,
  segments:elements.segments.textContent, error:elements.error.textContent,
  s:elements.s.value, a:elements.a.value, more:elements.more.hidden,
  cards:elements.grid.children.map(card=>({image:card.children[0].src,
    alt:card.children[0].alt, title:card.children[1].textContent,
    scores:card.children[2].children.map(row=>row.children.map(cell=>cell.textContent))}))});
"""


def run_page(manifest, actions=""):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for non-browser report interaction tests")
    doc = Document(render(manifest))
    script = "const ATTRS=" + json.dumps(doc.elements) + ";\n" + DOM
    script += "\n".join(doc.scripts) + "\n" + actions
    script += "\nconsole.log(JSON.stringify(snapshot()));"
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def with_frames(manifest, values, *, interval=1, sensitive=0.5, adult=0.5):
    frames = []
    for index, (time, scores) in enumerate(values, 1):
        names = ("general", "sensitive", "questionable", "explicit")
        frames.append(FrameScore(
            time=time, image=f"frames/frame_{index:06d}.jpg",
            label=names[max(range(4), key=lambda i: scores[i])],
            scores=Scores(**dict(zip(names, scores, strict=True))),
        ))
    cfg = ContentCheckConfig(sample_interval=interval, sensitive_threshold=sensitive,
                             adult_threshold=adult)
    selected, ranges = screen_frames(frames, cfg)
    data = manifest.model_dump()
    data["identity"]["sample_interval"] = interval
    data.update(frames=frames, sensitive_threshold=sensitive, adult_threshold=adult,
                candidate_count=len(selected), candidate_ranges=ranges)
    return CheckManifest(**data)


def test_report_escapes_embedded_json(valid_manifest):
    manifest = valid_manifest.model_copy(deep=True)
    manifest.identity.video_path = '/videos/</script><img src=x onerror="alert(1)">.mp4'
    html = render(manifest)
    assert manifest.identity.video_path not in html
    assert "\\u003c/script>" in html
    assert "textContent" in html
    assert "https://" not in html
    doc = Document(html)
    assert len(doc.scripts) == 1
    assert not doc.external
    assert "innerHTML" not in html
    embedded = re.search(r"const report=(.*);", html).group(1)
    assert CheckManifest.model_validate_json(embedded) == manifest


def test_report_displays_hostile_path_as_text(valid_manifest):
    manifest = valid_manifest.model_copy(deep=True)
    manifest.identity.video_path = '/videos/</script><img src=x onerror="alert(1)">.mp4'
    assert run_page(manifest)["summary"].startswith(manifest.identity.video_path)


def test_report_preserves_all_data_and_is_pure(valid_manifest, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = valid_manifest.model_dump_json()
    html = render(valid_manifest)
    embedded = re.search(r"const report=(.*);", html).group(1)
    assert CheckManifest.model_validate_json(embedded) == valid_manifest
    assert valid_manifest.model_dump_json() == before
    assert list(tmp_path.iterdir()) == []
    assert "__DATA__" not in html
    assert {"s", "a", "all", "sort", "count", "grid", "more", "segments"} <= (
        Document(html).elements.keys()
    )


def test_initial_thresholds_metadata_and_zero_candidates(valid_manifest):
    manifest = with_frames(valid_manifest, [(0, (1, 0, 0, 0))], sensitive=0.75, adult=0.9)
    state = run_page(manifest)
    assert float(state["s"]) == 0.75
    assert float(state["a"]) == 0.9
    for text in [manifest.identity.video_path, manifest.identity.model,
                 manifest.identity.revision, "CPUExecutionProvider", "1 帧", "0.0s"]:
        assert text in state["summary"]
    assert "候选 0/1" in state["count"]
    assert "已检查，未发现达到当前阈值的候选" in state["count"]
    assert state["cards"] == []
    assert state["more"] is True
    html = render(manifest)
    for text in ["general", "普通", "sensitive", "敏感", "questionable", "成人倾向",
                 "explicit", "露骨", "不是经过校准的色情概率", "非精确片段边界"]:
        assert text in html


def test_threshold_boundary_sorting_and_interval_match_python_contract(valid_manifest):
    manifest = with_frames(valid_manifest, [
        (0, (0.2, 0.5, 0.2, 0.1)), (3, (0.4, 0.1, 0.25, 0.25)),
        (6.001, (0.1, 0.1, 0.2, 0.6)), (7, (1, 0, 0, 0)),
    ], interval=2)
    state = run_page(manifest)
    assert [c["image"] for c in state["cards"]] == [
        "frames/frame_000003.jpg", "frames/frame_000001.jpg", "frames/frame_000002.jpg",
    ]
    assert "00:00.00–00:03.00 (2帧)" in state["segments"]
    assert "00:06.00–00:06.00 (1帧)" in state["segments"]
    state = run_page(manifest, "elements.sort.value='time'; elements.sort.listeners.change();")
    assert [c["image"] for c in state["cards"]] == [f.image for f in manifest.frames[:3]]
    assert state["cards"][0]["scores"] == [
        ["general · 普通", "0.2000"], ["sensitive · 敏感", "0.5000"],
        ["questionable · 成人倾向", "0.2000"], ["explicit · 露骨", "0.1000"],
    ]
    assert "sensitive · 敏感" in state["cards"][0]["title"]


def test_all_frames_pagination_and_live_threshold_changes(valid_manifest):
    manifest = with_frames(valid_manifest, [(i, (1, 0, 0, 0)) for i in range(125)])
    actions = "elements.all.checked=true; elements.all.listeners.change();"
    state = run_page(manifest, actions)
    assert len(state["cards"]) == 60
    assert state["more"] is False
    state = run_page(manifest, actions + "elements.more.onclick(); elements.more.onclick();")
    assert len(state["cards"]) == 125
    assert state["more"] is True
    state = run_page(manifest, actions + "elements.more.onclick(); "
                     "elements.s.value='0'; elements.s.listeners.input();")
    assert len(state["cards"]) == 60
    assert "候选 125/125" in state["count"]
    assert manifest.sensitive_threshold == 0.5


@pytest.mark.parametrize("value", ["", " ", "nope", "NaN", "Infinity", "-0.1", "1.1"])
@pytest.mark.parametrize("field", ["s", "a"])
def test_invalid_threshold_has_parameter_message_and_recovers(valid_manifest, field, value):
    actions = f"elements.{field}.value={json.dumps(value)}; elements.{field}.listeners.input();"
    state = run_page(valid_manifest, actions)
    assert "阈值必须为 0 到 1 之间的有限数" in state["error"]
    assert state["cards"] == []
    assert "已检查，未发现" not in state["count"]
    assert state["more"] is True
    state = run_page(valid_manifest, actions +
                     f"elements.{field}.value='0.5'; elements.{field}.listeners.input();")
    assert state["error"] == ""
    assert "已检查，未发现" in state["count"]
