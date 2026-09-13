"""证明 config 层的阈值真的被下游消费了，不只是"定义了个字段"。

逐字段的接线测试（ingest / credits / signals / render / llm）在下半部分；
文件开头那两条是**自动化审计**，扫 src/ 的 AST，不用逐个字段手写：

- `test_every_render_and_llm_field_is_read_off_a_config_object`：字段至少有一处
  真的从 config 对象上读出来（挡「定义了个字段就忘了」）。
- `test_config_aliases_are_only_default_parameter_values`：模块级别名只许当默认
  参数值用（挡「半接线」—— 函数体里直接读别名，那个位置的 config 覆盖永远失效）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tenmin.config import (
    CreditsConfig,
    IngestConfig,
    LLMConfig,
    ProjectConfig,
    RenderConfig,
    SignalsConfig,
)
from tenmin.ingest.credits import find_credit_ranges, in_credit_window
from tenmin.ingest.normalize import build_track
from tenmin.models import DialogueLine, DialogueTrack, SubtitleCue
from tenmin.pipeline import (
    Paths,
    run_audio,
    run_ingest,
    run_render,
    run_signals,
    run_timeline,
    run_voice,
)
from tenmin.render.subtitles import max_cells_per_line, render_ass
from tenmin.render.tts import build_tts_engine
from tenmin.render.video import build_render_args, quality_args
from tenmin.signals.aggregate import build_report
from tenmin.signals.density import find_density_shifts, find_low_density
from tenmin.signals.gaps import find_silent_gaps

SRT_TWO_HALVES = """1
00:00:00,000 --> 00:00:01,000
前半句

2
00:00:01,100 --> 00:00:02,000
后半句
"""

# --- 自动化接线审计（挡「定义了个字段但没接线」这一整类复发）---

SRC = Path(__file__).resolve().parent.parent / "src" / "tenmin"

# 权威定义住在 config.py，所以它自己不算消费点。
_AUDIT_SKIP_FILES = frozenset({"config.py"})

# 「这是一个 config 对象」的接收者名字。覆盖三种真实形态：
#   cfg.render.width / cfg.llm.max_attempts   → 接收者末段是 render / llm
#   llm.validation_retries（llm = cfg.llm）    → 接收者是 llm
#   cfg.voice（cfg: RenderConfig，build_tts_engine）→ 接收者是 cfg
#   DEFAULT_RENDER.ffmpeg_path                 → 接收者是那个默认实例
_CONFIG_RECEIVERS = frozenset({"cfg", "render", "llm", "DEFAULT_RENDER", "DEFAULT_LLM"})

# 派生模块级别名的那两个默认实例。
_DEFAULT_INSTANCES = {"DEFAULT_RENDER": "render", "DEFAULT_LLM": "llm"}

# 「已知不消费」白名单：{(子块, 字段名): 理由}。
#
# **目前是空的，这是刻意的** —— RenderConfig 与 LLMConfig 现在每一个字段都有真实
# 消费点。往里加东西时必须写清理由（比如「纯诊断字段，只挂在 provider 上供人读」），
# 而且理由要能回答一句话：为什么这个字段存在却不该有任何代码读它。
_KNOWN_UNCONSUMED: dict[tuple[str, str], str] = {}


def _audit_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name not in _AUDIT_SKIP_FILES)


def _receiver_tail(node: ast.expr) -> str | None:
    """`cfg.render` → "render"、`DEFAULT_RENDER` → "DEFAULT_RENDER"，别的返回 None。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _alias_definitions(tree: ast.AST) -> dict[str, tuple[str, str, int]]:
    """模块级 `NAME = DEFAULT_RENDER.field` → {别名: (子块, 字段, 行号)}。"""
    out: dict[str, tuple[str, str, int]] = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Attribute):
            continue
        base = node.value.value
        if not isinstance(base, ast.Name) or base.id not in _DEFAULT_INSTANCES:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = (
                    _DEFAULT_INSTANCES[base.id],
                    node.value.attr,
                    node.lineno,
                )
    return out


def _default_value_nodes(tree: ast.AST) -> set[int]:
    """所有落在「默认参数值」位置的表达式节点 id（含它们的子节点）。"""
    marked: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is None:
                continue
            for inner in ast.walk(default):
                marked.add(id(inner))
    return marked


def test_audit_sees_the_real_sources():
    """守住上面两条审计自己：SRC 写错时它们会假绿。"""
    names = {p.name for p in _audit_files()}
    assert "pipeline.py" in names
    assert "config.py" not in names
    assert len(names) > 10


def test_every_render_and_llm_field_is_read_off_a_config_object():
    """RenderConfig / LLMConfig 的每个字段都得有一处「从 config 对象上读」的代码。

    只算真正的读取：config.py 自己不算（那是定义），模块级别名的**定义式**
    （`WIDTH = DEFAULT_RENDER.width`）也不算 —— 它只是给默认参数值起了个名字，
    不代表 project.yaml 里配的值到得了任何地方。
    """
    wanted = {
        ("render", name) for name in RenderConfig.model_fields
    } | {("llm", name) for name in LLMConfig.model_fields}
    seen: set[tuple[str, str]] = set()
    for path in _audit_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        alias_lines = {line for _, _, line in _alias_definitions(tree).values()}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.lineno in alias_lines:
                continue
            tail = _receiver_tail(node.value)
            if tail is None or tail not in _CONFIG_RECEIVERS:
                continue
            block = {"render": "render", "DEFAULT_RENDER": "render"}.get(tail)
            if block is None:
                block = {"llm": "llm", "DEFAULT_LLM": "llm"}.get(tail)
            if block is None:  # 裸 cfg：两个子块都算它一份
                seen.add(("render", node.attr))
                seen.add(("llm", node.attr))
                continue
            seen.add((block, node.attr))
    missing = sorted(wanted - seen - set(_KNOWN_UNCONSUMED))
    assert missing == [], (
        f"这些 config 字段在 src/tenmin/ 里没有任何读取点（=哑字段）：{missing}"
    )


def test_config_aliases_are_only_default_parameter_values():
    """模块级别名（`X = DEFAULT_RENDER.field`）只许出现在**默认参数值**位置。

    这条是「半接线」的通用探测器，也是它存在的唯一理由。`WIDTH`/`CRF` 那一族的
    正确形状是「别名当默认值 + 真值由 pipeline 按参数传进来」，函数**体**里直接读
    别名就意味着那个位置永远只能拿到 config 的默认值 —— project.yaml 里的覆盖静默
    失效，而且失效得毫无痕迹（渲染照样成功，只是字体/码率不是你配的那个）。

    历史事故：`render/video.py` 的片尾卡 drawtext 直接读 `OUTRO_FONT_NAME`，而
    `build_render_args` 压根没有 `outro_font_name` 参数；同时 pipeline 把配置值递给
    了 preflight 去查字体 —— 于是「查的字体」和「用的字体」是两个不同的值。
    """
    offenders: list[str] = []
    for path in _audit_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _alias_definitions(tree)
        if not aliases:
            continue
        allowed = _default_value_nodes(tree)
        alias_lines = {line for _, _, line in aliases.values()}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or node.id not in aliases:
                continue
            if node.lineno in alias_lines or id(node) in allowed:
                continue
            block, field, _ = aliases[node.id]
            offenders.append(f"{path.name}:{node.lineno} {node.id}（={block}.{field}）")
    assert offenders == [], (
        "这些模块级 config 别名被用在了默认参数值之外的位置，"
        f"那个取值点上 project.yaml 的覆盖是静默失效的：{offenders}"
    )


def dline(idx: int, start: float, end: float, text: str = "喂", kind: str = "dialogue"):
    return DialogueLine(idx=idx, start=start, end=end, text=text, raw=text, kind=kind)


def track(lines: list[DialogueLine], duration: float) -> DialogueTrack:
    return DialogueTrack(episode=1, duration=duration, lines=lines)


# --- IngestConfig 真的被 build_track 消费 ---


def test_ingest_config_merge_max_gap_is_honoured(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(SRT_TWO_HALVES, encoding="utf-8")

    # 两条 cue 间隔 0.1s < 默认 merge_max_gap=0.3 → 合并成一条。
    merged = build_track(srt, episode=1, merge_lines=True)
    assert len(merged.lines) == 1

    # 把阈值压到 0.05 就不该再合并。
    split = build_track(
        srt, episode=1, merge_lines=True, ingest=IngestConfig(merge_max_gap=0.05)
    )
    assert len(split.lines) == 2


def test_ingest_config_merge_max_chars_is_honoured(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(SRT_TWO_HALVES, encoding="utf-8")
    split = build_track(
        srt, episode=1, merge_lines=True, ingest=IngestConfig(merge_max_chars=5)
    )
    assert len(split.lines) == 2


def test_ingest_config_merge_max_line_seconds_is_honoured(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(SRT_TWO_HALVES, encoding="utf-8")
    split = build_track(
        srt, episode=1, merge_lines=True, ingest=IngestConfig(merge_max_line_seconds=0.5)
    )
    assert len(split.lines) == 2


# --- CreditsConfig 真的被 credits.py 消费 ---


def test_credits_config_head_window_is_honoured():
    duration = 1000.0
    assert in_credit_window(250.0, duration) is True
    narrow = CreditsConfig(credit_head_window=100.0)
    assert in_credit_window(250.0, duration, cfg=narrow) is False


def test_credits_config_ed_keyword_window_is_honoured():
    duration = 1000.0
    # 距片尾 90s，默认 ed_keyword_window_seconds=80 → 窗外。
    assert in_credit_window(910.0, duration) is False
    wide = CreditsConfig(ed_keyword_window_seconds=100.0)
    assert in_credit_window(910.0, duration, cfg=wide) is True


def test_credits_config_op_span_window_is_honoured():
    lines = [dline(1, 100.0, 130.0, "監督 山田太郎", kind="credits")]
    # 30s 跨度落在默认 op_span_min=40 之外 → 认不出 OP。
    assert find_credit_ranges(lines, duration=1000.0)[0] is None
    loose = CreditsConfig(op_span_min=10.0)
    assert find_credit_ranges(lines, duration=1000.0, cfg=loose)[0] == (100.0, 130.0)


def test_credits_config_ed_cluster_tail_is_honoured():
    lines = [dline(1, 800.0, 860.0, "作詞 某某", kind="credits")]
    # 起点距片尾 200s，默认 ed_cluster_tail_seconds=120 → 不算 ED。
    assert find_credit_ranges(lines, duration=1000.0)[1] is None
    loose = CreditsConfig(ed_cluster_tail_seconds=300.0)
    assert find_credit_ranges(lines, duration=1000.0, cfg=loose)[1] == (800.0, 860.0)


def test_credits_config_reaches_credits_through_build_track(tmp_path):
    """CreditsConfig 必须一路从 build_track 传到 credits.py，不能半路丢掉。"""
    srt = tmp_path / "a.srt"
    srt.write_text(
        "1\n00:01:40,000 --> 00:02:10,000\n監督 山田太郎\n\n"
        "2\n00:16:30,000 --> 00:16:40,000\n结束\n",
        encoding="utf-8",
    )
    assert build_track(srt, episode=1).op_range is None
    loose = build_track(srt, episode=1, credits=CreditsConfig(op_span_min=10.0))
    assert loose.op_range == (100.0, 130.0)


# --- SignalsConfig 真的被 signals/ 消费 ---


def test_signals_config_min_gap_seconds_is_honoured():
    t = track([dline(1, 0.0, 2.0), dline(2, 7.0, 9.0)], duration=9.0)
    assert len(find_silent_gaps(t)) == 1
    assert find_silent_gaps(t, cfg=SignalsConfig(min_gap_seconds=10.0)) == []


def test_signals_config_gap_strength_thresholds_are_honoured():
    t = track([dline(1, 0.0, 2.0), dline(2, 12.0, 14.0)], duration=14.0)
    # 10s 间隙：默认 (15, 8) 分档 → strength 3。
    assert find_silent_gaps(t)[0].strength == 3
    strong = SignalsConfig(gap_strong_seconds=9.0, gap_medium_seconds=5.0)
    assert find_silent_gaps(t, cfg=strong)[0].strength == 4


def test_signals_config_low_density_ratio_is_honoured():
    t = track(
        [
            dline(1, 0.0, 2.0, "十个字十个字"),
            dline(2, 2.0, 4.0, "十个字十个字"),
            dline(3, 4.0, 9.0, "少"),
        ],
        duration=9.0,
    )
    assert [s.anchor_lines for s in find_low_density(t)] == [[3]]
    # ratio 压到 0 就没有任何一行低于阈值。
    assert find_low_density(t, cfg=SignalsConfig(low_density_ratio=0.0)) == []


def test_signals_config_low_density_strength_is_honoured():
    t = track(
        [
            dline(1, 0.0, 2.0, "十个字十个字"),
            dline(2, 2.0, 4.0, "十个字十个字"),
            dline(3, 4.0, 9.0, "少"),
        ],
        duration=9.0,
    )
    assert find_low_density(t)[0].strength == 3
    tuned = SignalsConfig(low_density_strength=5)
    assert find_low_density(t, cfg=tuned)[0].strength == 5


def test_signals_config_shift_z_threshold_is_honoured():
    lines = [
        dline(idx, idx * 30.0, idx * 30.0 + 1.0, "字" * chars)
        for idx, chars in enumerate([3, 3, 3, 3, 12, 3, 3, 3])
    ]
    t = track(lines, duration=240.0)
    assert find_density_shifts(t) != []
    calm = SignalsConfig(shift_z_threshold=99.0)
    assert find_density_shifts(t, cfg=calm) == []


def test_signals_config_summary_max_chars_is_honoured():
    long_text = "一二三四五六七八九十" * 5
    t = track(
        [
            dline(1, 0.0, 1.0, "字" * 20),
            dline(2, 1.0, 2.0, "字" * 20),
            dline(3, 4.0, 30.0, long_text),
        ],
        duration=30.0,
    )
    default = build_report(t)
    short = build_report(t, cfg=SignalsConfig(summary_max_chars=4))
    assert max(len(h.summary) for h in default.highlights) == 30
    assert max(len(h.summary) for h in short.highlights) == 4


def test_signals_config_min_separation_is_honoured():
    t = track(
        [dline(1, 0.0, 2.0), dline(2, 6.0, 8.0), dline(3, 12.0, 14.0)], duration=14.0
    )
    # 两个 4s 间隙相距 2s，默认 min_separation=2.0 恰好把它们聚成一个 highlight。
    assert len(build_report(t).highlights) == 1
    assert len(build_report(t, cfg=SignalsConfig(min_separation=0.5)).highlights) == 2


# --- RenderConfig 的 width/height/subtitle_font_name 真的被消费 ---


def test_render_ass_honours_width_and_height():
    out = render_ass([SubtitleCue(start=0.0, end=1.0, text="喂")], width=1280, height=720)
    assert "PlayResX: 1280" in out
    assert "PlayResY: 720" in out


def test_max_cells_per_line_honours_width():
    assert max_cells_per_line(52, width=1280) < max_cells_per_line(52, width=1920)


def test_render_ass_wraps_against_configured_width():
    """断行必须跟着分辨率走，否则窄画布下字幕会冲出画面。"""
    cue = SubtitleCue(start=0.0, end=5.0, text="一二三四五六七八九十" * 5)
    narrow = render_ass([cue], width=960)
    wide = render_ass([cue], width=1920)
    assert narrow.count("\\N") > wide.count("\\N")


def test_pipeline_timeline_passes_render_config_into_subtitles(tmp_path, monkeypatch):
    from tenmin.render import subtitles as subtitles_module

    seen: dict[str, object] = {}
    real = subtitles_module.render_ass

    def spy(cues, **kwargs):
        seen.update(kwargs)
        return real(cues, **kwargs)

    monkeypatch.setattr("tenmin.pipeline.render_ass", spy)

    cfg = _minimal_project(tmp_path)
    cfg.render.width = 1280
    cfg.render.height = 720
    cfg.render.subtitle_font_name = "PingFang SC"
    _write_voice_and_script(cfg)

    run_timeline(cfg, episode=1, source_duration=100.0)
    assert seen["width"] == 1280
    assert seen["height"] == 720
    assert seen["font_name"] == "PingFang SC"
    assert seen["font_size"] == cfg.render.font_size


def test_build_render_args_scale_defaults_match_render_config():
    """字幕的 PlayRes 与视频的 scale 必须来自同一个 width/height，否则字幕会静默缩放。"""
    from pathlib import Path

    defaults = RenderConfig()
    args = build_render_args(
        video=Path("in.mkv"),
        timeline=_stub_timeline(),
        audio=Path("a.m4a"),
        ass=Path("s.ass"),
        out_path=Path("out.mp4"),
        encoder="libx264",
    )
    assert f"scale={defaults.width}:{defaults.height}" in " ".join(args)
    assert f"PlayResX: {defaults.width}" in render_ass([])
    assert f"PlayResY: {defaults.height}" in render_ass([])


# --- pipeline 层把 cfg 的子块传下去 ---


def test_run_ingest_and_signals_use_project_config(tmp_path):
    cfg = _minimal_project(tmp_path)
    cfg.signals.min_gap_seconds = 999.0
    run_ingest(cfg)
    reports = run_signals(cfg)
    assert reports[0].silent_gaps == []
    assert Paths(cfg.root).signals(1).is_file()


# --- render 的编码参数真的被消费（P2-A 第 2 项）---
#
# 这五个字段（crf / preset / videotoolbox_bitrate / audio_codec / audio_bitrate）
# 以前只被 video.py / audio.py 通过 DEFAULT_RENDER 的模块别名读，所以 project.yaml
# 里的按项目覆盖是**静默失效**的。下面这一组测试就是那件事的回归锁。


def test_quality_args_honours_crf_and_preset():
    assert quality_args("libx264", crf="18", preset="slow") == [
        "-crf",
        "18",
        "-preset",
        "slow",
    ]


def test_quality_args_appends_tune_only_when_configured():
    assert "-tune" not in quality_args("libx264")
    assert quality_args("libx264", tune="animation")[-2:] == ["-tune", "animation"]


def test_quality_args_honours_videotoolbox_bitrate():
    assert quality_args("h264_videotoolbox", videotoolbox_bitrate="9000k") == [
        "-b:v",
        "9000k",
    ]


def test_build_render_args_honours_encoder_knobs(tmp_path):
    args = build_render_args(
        video=tmp_path / "in.mkv",
        timeline=_stub_timeline(),
        audio=tmp_path / "a.m4a",
        ass=tmp_path / "s.ass",
        out_path=tmp_path / "out.mp4",
        encoder="libx264",
        crf="17",
        preset="veryfast",
        tune="animation",
    )
    assert args[args.index("-crf") + 1] == "17"
    assert args[args.index("-preset") + 1] == "veryfast"
    assert args[args.index("-tune") + 1] == "animation"


def test_build_mix_args_honours_audio_codec_and_bitrate(tmp_path):
    args = _mix_args(tmp_path, audio_codec="libopus", audio_bitrate="128k")
    assert args[args.index("-c:a") + 1] == "libopus"
    assert args[args.index("-b:a") + 1] == "128k"


def test_build_mix_args_honours_limiter_ceiling(tmp_path):
    default = _mix_args(tmp_path)
    graph = default[default.index("-filter_complex") + 1]
    assert "alimiter=limit=1:level=false:latency=true" in graph
    lowered = _mix_args(tmp_path, limiter_ceiling=0.891)
    graph = lowered[lowered.index("-filter_complex") + 1]
    assert "alimiter=limit=0.891:level=false:latency=true" in graph


def test_run_render_wires_the_encoder_knobs(tmp_path, monkeypatch):
    """cfg.render 的编码参数必须一路走到 build_render_args。"""
    from tenmin.render import video as video_module

    seen: dict[str, object] = {}
    real = video_module.build_render_args

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr("tenmin.pipeline.render_video", _capturing_render_video(seen))
    cfg = _minimal_project(tmp_path)
    cfg.render.crf = "16"
    cfg.render.preset = "slower"
    cfg.render.tune = "animation"
    cfg.render.videotoolbox_bitrate = "7000k"
    _write_timeline_and_inputs(cfg)
    run_render(cfg, episode=1)
    assert seen["crf"] == "16"
    assert seen["preset"] == "slower"
    assert seen["tune"] == "animation"
    assert seen["videotoolbox_bitrate"] == "7000k"


def test_run_render_takes_the_frame_rate_from_the_timeline_artifact(tmp_path, monkeypatch):
    """片尾卡的帧率来自 timeline.json 记的那个，不是 render 阶段自己再探一次。

    段边界就是按那个帧率对齐的，两处必须是同一个数字。
    """
    seen: dict[str, object] = {}
    monkeypatch.setattr("tenmin.pipeline.render_video", _capturing_render_video(seen))
    cfg = _minimal_project(tmp_path)
    _write_timeline_and_inputs(cfg, frame_rate=23.976023976023978)
    run_render(cfg, episode=1)
    assert seen["frame_rate"] == pytest.approx(23.976023976023978)


def test_run_timeline_records_the_probed_frame_rate(tmp_path, monkeypatch):
    """帧率必须进产物：render 阶段与帧对齐都读它。"""
    monkeypatch.setattr("tenmin.pipeline.probe_duration", lambda path, **_: 100.0)
    monkeypatch.setattr("tenmin.pipeline.probe_frame_rate", lambda path, **_: 23.976)
    cfg = _minimal_project(tmp_path)
    _write_voice_and_script(cfg)
    timeline, _ = run_timeline(cfg, episode=1)
    assert timeline.frame_rate == pytest.approx(23.976)
    reloaded = Paths(cfg.root).timeline(1).read_text(encoding="utf-8")
    assert '"frame_rate": 23.976' in reloaded


def test_run_audio_wires_the_audio_knobs(tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    def fake_mix_audio(**kwargs):
        seen.update(kwargs)
        return kwargs["out_path"]

    monkeypatch.setattr("tenmin.pipeline.mix_audio", fake_mix_audio)
    cfg = _minimal_project(tmp_path)
    cfg.render.audio_codec = "libopus"
    cfg.render.audio_bitrate = "96k"
    cfg.render.limiter_ceiling = 0.7
    _write_timeline_and_inputs(cfg)
    run_audio(cfg, episode=1)
    assert seen["audio_codec"] == "libopus"
    assert seen["audio_bitrate"] == "96k"
    assert seen["limiter_ceiling"] == 0.7


# --- helpers ---


def _mix_args(tmp_path, **overrides):
    from tenmin.models import VoiceChunk, VoiceTrack
    from tenmin.render.audio import build_mix_args

    kwargs: dict[str, object] = {
        "video": tmp_path / "in.mkv",
        "timeline": _stub_timeline(offsets=[0.0]),
        "track": VoiceTrack(
            episode=1,
            chunks=[
                VoiceChunk(
                    beat_id="b1",
                    index=1,
                    text="喂",
                    path="chunk_001.mp3",
                    duration=5.0,
                )
            ],
        ),
        "voice_dir": tmp_path / "04_voice" / "E01",
        "out_path": tmp_path / "out.m4a",
        "duck_db": -12.0,
    }
    kwargs.update(overrides)
    return build_mix_args(**kwargs)


def _capturing_render_video(seen: dict[str, object]):
    def fake_render_video(**kwargs):
        seen.update(kwargs)
        return kwargs["out_path"]

    return fake_render_video


def _write_timeline_and_inputs(cfg: ProjectConfig, frame_rate: float | None = None) -> None:
    """run_audio / run_render 需要的上游产物（内容不重要，它们的消费者被替掉了）。"""
    from tenmin.models import VoiceChunk, VoiceTrack

    paths = Paths(cfg.root)
    timeline = _stub_timeline(offsets=[0.0])
    timeline.frame_rate = frame_rate
    timeline.episode = 1
    paths.timeline(1).parent.mkdir(parents=True, exist_ok=True)
    paths.timeline(1).write_text(timeline.model_dump_json(), encoding="utf-8")
    paths.subtitles(1).write_text("[Script Info]\n", encoding="utf-8")
    track = VoiceTrack(
        episode=1,
        chunks=[
            VoiceChunk(beat_id="b1", index=1, text="喂", path="chunk_001.mp3", duration=5.0)
        ],
    )
    paths.voice(1).parent.mkdir(parents=True, exist_ok=True)
    paths.voice(1).write_text(track.model_dump_json(), encoding="utf-8")
    paths.mixed_audio(1).parent.mkdir(parents=True, exist_ok=True)
    paths.mixed_audio(1).write_bytes(b"\x00")


def _stub_timeline(offsets: list[float] | None = None):
    from tenmin.models import Timeline, TimelineSegment

    return Timeline(
        episode=1,
        total_seconds=10.0,
        segments=[
            TimelineSegment(
                beat_id="b1",
                source_start=0.0,
                source_end=10.0,
                timeline_start=0.0,
                timeline_end=10.0,
            )
        ],
        subtitles=[],
        narration_offsets=offsets or [],
    )


def _minimal_project(tmp_path) -> ProjectConfig:
    srt = tmp_path / "srt" / "E01.srt"
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n喂\n\n2\n00:00:20,000 --> 00:00:22,000\n嗯\n",
        encoding="utf-8",
    )
    video = tmp_path / "video" / "E01.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"")
    return ProjectConfig.model_validate(
        {
            "show": "剧名",
            "slug": "slug",
            "episodes": [{"number": 1, "srt": "srt/E01.srt", "video": "video/E01.mp4"}],
        }
    ).bind_root(tmp_path)


def _write_voice_and_script(cfg: ProjectConfig) -> None:
    from tenmin.models import Beat, Clip, Script, VoiceChunk, VoiceTrack

    paths = Paths(cfg.root)
    narration = "一二三四五六七八九十" * 5
    script = Script(
        # 原来这里写的是 episode=1 / title="标题"，Script 上并没有这两个字段，
        # extra="ignore" 时被静默丢掉（episodes 实际是 []）。改成真实字段名。
        show="剧名",
        episodes=[1],
        beats=[
            Beat(
                id="b1",
                label="开场",
                role="hook",
                narration=narration,
                clips=[Clip(episode=1, start=0.0, end=10.0)],
                est_seconds=5.0,
            )
        ],
    )
    paths.script(1).parent.mkdir(parents=True, exist_ok=True)
    paths.script(1).write_text(script.model_dump_json(), encoding="utf-8")

    voice = VoiceTrack(
        episode=1,
        total_seconds=5.0,
        chunks=[
            VoiceChunk(
                beat_id="b1",
                index=1,
                text=narration,
                path="chunk_001.mp3",
                duration=5.0,
                hold_after=0.0,
            )
        ],
    )
    paths.voice(1).parent.mkdir(parents=True, exist_ok=True)
    paths.voice(1).write_text(voice.model_dump_json(), encoding="utf-8")


# --- render.tts_* 接线 ---


def test_build_tts_engine_wires_proxy_and_timeouts():
    engine = build_tts_engine(
        RenderConfig(
            tts_proxy="http://127.0.0.1:8080",
            tts_connect_timeout=3,
            tts_receive_timeout=17,
            tts_chunk_timeout_seconds=44.0,
        )
    )
    assert engine.proxy == "http://127.0.0.1:8080"
    assert engine.connect_timeout == 3
    assert engine.receive_timeout == 17
    assert engine.chunk_timeout_seconds == 44.0


async def test_run_voice_wires_tts_max_attempts(tmp_path, monkeypatch):
    """render.tts_max_attempts 必须真的到达 synthesize_with_retry 的重试循环。"""
    from tenmin.render import tts as tts_module

    from .fakes import FlakyTTSEngine

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(tts_module, "_sleep", no_sleep)

    cfg = _minimal_project(tmp_path)
    cfg.render.tts_max_attempts = 2
    _write_voice_and_script(cfg)
    Paths(cfg.root).voice(1).unlink()

    engine = FlakyTTSEngine(fail_times=99)
    with pytest.raises(tts_module.TTSError):
        await run_voice(cfg, engine, episode=1)
    assert engine.attempts == 2


async def test_run_voice_wires_tts_concurrency(tmp_path):
    """render.tts_concurrency 必须真的到达 synthesize_track 的 worker 池。

    原来它是个哑字段：定义了、测了 default，但 pipeline 压根不读它。
    """
    import asyncio

    from tenmin.models import Beat, Clip, Script

    class _PeakEngine:
        fingerprint = "fake|voice|+0%"

        def __init__(self):
            self.in_flight = 0
            self.peak = 0

        async def synthesize(self, text: str, out_path: Path) -> float:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            await asyncio.sleep(0.02)
            self.in_flight -= 1
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake mp3")
            return 2.0

    cfg = _minimal_project(tmp_path)
    cfg.render.tts_concurrency = 3
    # 一个 beat 只会切出一个 chunk（plan_chunks 只在 hold 处断开），所以要 6 个 beat。
    script = Script(
        show="剧名",
        episodes=[1],
        beats=[
            Beat(
                id=f"b{i}",
                label="开场",
                role="hook",
                narration=f"第{i}句话在这里。",
                clips=[Clip(episode=1, start=float(i), end=float(i) + 10.0)],
            )
            for i in range(6)
        ],
    )
    paths = Paths(cfg.root)
    paths.script(1).parent.mkdir(parents=True, exist_ok=True)
    paths.script(1).write_text(script.model_dump_json(), encoding="utf-8")

    engine = _PeakEngine()
    await run_voice(cfg, engine, episode=1)
    assert engine.peak == 3


async def test_run_voice_wires_ffprobe_path_into_the_reuse_path(tmp_path, monkeypatch):
    """render.ffprobe_path 必须也到达**复用** chunk 那条路上的时长体检。

    engine 那条路（EdgeTTSEngine.synthesize 的体检）早就接线了，复用那条路一直用的是
    模块默认 "ffprobe"。这个半接线的失败模式最难查：只配了 ffprobe_path 的用户第一次
    跑（全新合成）好的，第二次跑（chunk 全命中缓存）才炸。
    """
    from tenmin.render import tts as tts_module

    from .fakes import FakeTTSEngine

    cfg = _minimal_project(tmp_path)
    cfg.render.ffprobe_path = "/opt/x/ffprobe"
    _write_voice_and_script(cfg)
    Paths(cfg.root).voice(1).unlink()

    # 先跑一轮把 chunk 落到盘上（FakeTTSEngine 自己报时长，不碰 ffprobe）。
    await run_voice(cfg, FakeTTSEngine([2.0] * 8), episode=1)
    # voice.json 拿掉，复用路径就没有现成时长可用，只能去 probe。
    Paths(cfg.root).voice(1).unlink()

    seen: list[str] = []

    def spy(path, *, ffprobe="ffprobe"):
        seen.append(ffprobe)
        return 2.0

    monkeypatch.setattr(tts_module, "probe_duration", spy)
    await run_voice(cfg, FakeTTSEngine([]), episode=1)

    assert seen, "复用路径压根没去 probe，这个测试没测到东西"
    assert set(seen) == {"/opt/x/ffprobe"}


def test_run_timeline_surfaces_subtitle_legibility_warnings(tmp_path):
    """字幕可读性检查（P2-E C2）要接到 timeline 阶段的 warnings 上。

    检查住在 render/subtitles.py（它才知道字号与画布宽度），而拿得到 cfg 与 warnings
    的是 run_timeline —— render/timeline.py 压根不认识字体。
    """
    cfg = _minimal_project(tmp_path)
    cfg.render.subtitle_max_lines = 1
    _write_voice_and_script(cfg)

    _, warnings = run_timeline(cfg, episode=1, source_duration=100.0)
    assert any("字幕" in w and "行" in w for w in warnings)


def test_run_timeline_legibility_check_can_be_switched_off(tmp_path):
    cfg = _minimal_project(tmp_path)
    cfg.render.subtitle_max_lines = 0
    cfg.render.subtitle_min_seconds = 0
    _write_voice_and_script(cfg)

    _, warnings = run_timeline(cfg, episode=1, source_duration=100.0)
    assert not any("字幕" in w for w in warnings)
