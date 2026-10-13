import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from tenmin import importer
from tenmin.config import EpisodeConfig, ProjectConfig, load_project
from tenmin.importer import (
    EXTRA,
    LANGUAGE_SUFFIXES,
    Ambiguous,
    ImportItem,
    ImportPlan,
    Pair,
    RangePreview,
    Skipped,
    apply_import_plan,
    build_import_plan,
    infer_ranges,
    is_extra,
    parse_episode_number,
    render_preview,
    resolve_ambiguous,
    round_range,
    scan_directory,
)
from tenmin.ingest import av_match
from tenmin.ingest.normalize import build_track
from tenmin.render.ffmpeg import FFmpegBinaryError, FFmpegError
from tenmin.script.llm import LLMError

from .fakes import FakeProvider


def _touch(path: Path, data: bytes = b"fake") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_scan_pairs_exact_stem_case_insensitively(tmp_path):
    video = _touch(tmp_path / "Show 01.MKV")
    srt = _touch(tmp_path / "show 01.SRT")

    scan = scan_directory(tmp_path)

    assert [(p.video, p.srt, p.ignored_srts) for p in scan.pairs] == [(video, srt, ())]
    assert scan.unpaired_videos == []
    assert scan.unpaired_srts == []


def test_scan_only_looks_at_the_top_level(tmp_path):
    _touch(tmp_path / "nested" / "Show 02.mkv")
    _touch(tmp_path / "nested" / "Show 02.srt")

    scan = scan_directory(tmp_path)

    assert scan.pairs == []
    assert scan.unpaired_videos == []
    assert scan.unpaired_srts == []


@pytest.mark.parametrize("separator", [".", "_"])
@pytest.mark.parametrize("suffix", LANGUAGE_SUFFIXES)
def test_scan_accepts_a_language_suffix(tmp_path, separator, suffix):
    video = _touch(tmp_path / "Show 01.mp4")
    srt = _touch(tmp_path / f"Show 01{separator}{suffix}.srt")

    scan = scan_directory(tmp_path)

    assert [(p.video, p.srt) for p in scan.pairs] == [(video, srt)]


def test_scan_picks_the_highest_priority_candidate_and_reports_the_rest(tmp_path):
    _touch(tmp_path / "Show 01.mkv")
    exact = _touch(tmp_path / "Show 01.srt")
    hans = _touch(tmp_path / "Show 01.zh-Hans.srt")
    ja = _touch(tmp_path / "Show 01.ja.srt")

    scan = scan_directory(tmp_path)

    assert len(scan.pairs) == 1
    assert scan.pairs[0].srt == exact
    assert scan.pairs[0].ignored_srts == (hans, ja)
    assert scan.unpaired_srts == []


def test_scan_language_priority_follows_the_list_order(tmp_path):
    _touch(tmp_path / "Show 01.mkv")
    jpn = _touch(tmp_path / "Show 01.jpn.srt")
    tc = _touch(tmp_path / "Show 01_tc.srt")

    scan = scan_directory(tmp_path)

    assert scan.pairs[0].srt == tc
    assert scan.pairs[0].ignored_srts == (jpn,)


def test_scan_lists_unpaired_files_and_does_no_fuzzy_matching(tmp_path):
    lonely_video = _touch(tmp_path / "Show 03.mkv")
    lonely_srt = _touch(tmp_path / "Show 03 v2.srt")
    _touch(tmp_path / "notes.txt")

    scan = scan_directory(tmp_path)

    assert scan.pairs == []
    assert scan.unpaired_videos == [lonely_video]
    assert scan.unpaired_srts == [lonely_srt]


def test_scan_pairs_from_a_separate_srt_dir(tmp_path):
    video = _touch(tmp_path / "videos" / "Show 04.mkv")
    srt = _touch(tmp_path / "subs" / "Show 04.zh-Hant.srt")
    _touch(tmp_path / "videos" / "Show 04.srt")

    scan = scan_directory(tmp_path / "videos", tmp_path / "subs")

    assert [(p.video, p.srt) for p in scan.pairs] == [(video, srt)]


def test_scan_finds_references_case_insensitively_without_unpaired_files(tmp_path):
    video = _touch(tmp_path / "Show 01.mkv")
    srt = _touch(tmp_path / "Show 01.srt")
    op = _touch(tmp_path / "OP.PNG")
    ed = _touch(tmp_path / "ed.Jpeg")
    _touch(tmp_path / "cover.jpg")
    _touch(tmp_path / "op.txt")
    _touch(tmp_path / "nested" / "op.jpg")
    _touch(tmp_path / "ed.png" / "notes.txt")

    scan = scan_directory(tmp_path)

    assert scan.references == {"op": op, "ed": ed}
    assert [(p.video, p.srt) for p in scan.pairs] == [(video, srt)]
    assert scan.unpaired_videos == []
    assert scan.unpaired_srts == []


def test_scan_ignores_references_in_a_separate_srt_dir(tmp_path):
    video = _touch(tmp_path / "videos" / "Show 01.mkv")
    srt = _touch(tmp_path / "subs" / "Show 01.srt")
    _touch(tmp_path / "subs" / "op.png")

    scan = scan_directory(tmp_path / "videos", tmp_path / "subs")

    assert scan.references == {}
    assert [(p.video, p.srt) for p in scan.pairs] == [(video, srt)]
    assert scan.unpaired_videos == []
    assert scan.unpaired_srts == []


def test_scan_rejects_duplicate_references_with_sorted_names(tmp_path):
    _touch(tmp_path / "op.png")
    _touch(tmp_path / "op.jpg")

    with pytest.raises(ValueError) as error:
        scan_directory(tmp_path)

    assert str(error.value) == "OP 参考截图不止一张（op.jpg、op.png），只留一张再导入"


def test_scan_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到目录"):
        scan_directory(tmp_path / "nope")


def test_scan_rejects_a_file_given_as_directory(tmp_path):
    file = _touch(tmp_path / "Show 01.mkv")
    with pytest.raises(ValueError, match="不是目录"):
        scan_directory(file)


def test_scan_rejects_a_missing_srt_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到目录"):
        scan_directory(tmp_path, tmp_path / "subs")


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("Show E01", 1),
        ("Show EP02", 2),
        ("Show.S01E03.1080p.WEB", 3),
        ("Show 第04集", 4),
        ("Show 第05話", 5),
        ("Show 第05话", 5),
        ("Show 第０６話", 6),
        ("Show ＥＰ07", 7),
    ],
)
def test_parse_strong_patterns(stem, expected):
    assert parse_episode_number(stem) == expected


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("[ANi] 才女的侍从 - 11 [1080P][Baha][WEB-DL][AAC AVC][CHT]", 11),
        ("Show_08_x264_10bit", 8),
        ("Show 2024 - 09 [H.264 AAC2.0]", 9),
        ("Show.4K.12", 12),
        ("[Group] Show [HEVC FLAC OPUS] [10]", 10),
    ],
)
def test_parse_weak_pattern_skips_noise_tokens(stem, expected):
    assert parse_episode_number(stem) == expected


def test_parse_returns_none_when_no_number_is_left():
    assert parse_episode_number("Kimi no Na wa") is None


def test_parse_reports_ambiguous_weak_candidates():
    assert parse_episode_number("86 - 11") == Ambiguous((11, 86))


def test_parse_reports_conflicting_strong_matches_as_ambiguous():
    assert parse_episode_number("Show E01 E02") == Ambiguous((1, 2))


@pytest.mark.parametrize(
    "stem",
    [
        "Show NCOP",
        "[Group] Show NCED1",
        "Show SP01",
        "[Group] Show OVA",
        "Show PV",
        "Show 预告",
        "Show 預告",
        "Show Menu",
    ],
)
def test_is_extra_flags_non_main_material(stem):
    assert is_extra(stem) is True


@pytest.mark.parametrize("stem", ["Spy x Family - 01", "Show - 01", "Nova E02"])
def test_is_extra_leaves_main_episodes_alone(stem):
    assert is_extra(stem) is False


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("Show - 05 [E3A1B2C4]", 5),
        ("[Group] Show - 07 [1080p][e3a1b2c4]", 7),
        ("Show S01E05 [E3A1B2C4]", 5),
        ("Show E05v2", 5),
        ("Show S01E05v2", 5),
        ("Show E3A1", None),
    ],
)
def test_parse_strong_pattern_ignores_crc_tags_and_allows_version_suffix(stem, expected):
    assert parse_episode_number(stem) == expected


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("Show - 03v2", 3),
        ("Show - 03V2 [1080p]", 3),
        ("Show - 08 [DDP5.1]", 8),
        ("Show - 08 [DD5.1]", 8),
        ("Show - 08 [AAC 5.1ch]", 8),
        ("Show - 08 [2.0ch]", 8),
    ],
)
def test_parse_weak_pattern_version_suffix_and_channel_tokens(stem, expected):
    assert parse_episode_number(stem) == expected


async def test_resolve_ambiguous_sends_one_batched_call():
    provider = FakeProvider(
        [{"files": [{"id": 1, "episode": 11}, {"id": 2}, {"id": 3, "extra": True}]}]
    )

    verdicts = await resolve_ambiguous(["86 - 11", "Kimi no Na wa", "Mystery Bonus"], provider)

    assert verdicts == [11, None, EXTRA]
    assert len(provider.calls) == 1
    payload = json.loads(provider.calls[0]["user"])
    assert payload == {
        "files": [
            {"id": 1, "filename": "86 - 11"},
            {"id": 2, "filename": "Kimi no Na wa"},
            {"id": 3, "filename": "Mystery Bonus"},
        ]
    }
    assert provider.calls[0]["schema"] is not None


async def test_resolve_ambiguous_treats_missing_or_invalid_answers_as_unknown():
    provider = FakeProvider(
        [{"files": [{"id": 2, "episode": 0}, {"id": 2, "episode": 5}, {"id": 99, "episode": 3}]}]
    )

    verdicts = await resolve_ambiguous(["a", "b"], provider)

    assert verdicts == [None, None]


async def test_resolve_ambiguous_with_no_stems_makes_no_call():
    provider = FakeProvider([])

    assert await resolve_ambiguous([], provider) == []
    assert provider.calls == []


def _project(tmp_path: Path, extra_yaml: str = "episodes: []\n") -> ProjectConfig:
    root = tmp_path / "work" / "saijo"
    (root / "srt").mkdir(parents=True)
    (root / "project.yaml").write_text(
        "show: 才女的侍从\nslug: saijo\n" + extra_yaml, encoding="utf-8"
    )
    return load_project(root / "project.yaml")


def _probe(monkeypatch, result):
    """把 pipeline._source_duration 用的 ffprobe 换成固定结果（数字）或固定异常。"""

    def fake_probe(path, *, ffprobe):
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr("tenmin.pipeline.probe_duration", fake_probe)


@pytest.fixture
def fixed_duration(monkeypatch):
    _probe(monkeypatch, 1420.0)


def test_round_range_rounds_half_up_to_a_tenth():
    assert round_range((153.486, 224.681)) == (153.5, 224.7)
    assert round_range((0.25, 0.35)) == (0.3, 0.4)


def test_round_range_rejects_a_collapsed_range():
    assert round_range((10.02, 10.04)) is None
    assert round_range((10.06, 10.04)) is None


def test_infer_ranges_matches_build_track_and_writes_both(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)

    track = build_track(
        golden_srt_path,
        episode=2,
        show_title=cfg.show,
        glossary=cfg.glossary,
        convert_traditional=cfg.locale.convert_traditional,
        duration=1420.0,
        ingest=cfg.ingest,
        credits=cfg.credits,
    )
    assert preview.duration == 1420.0
    assert preview.op_range == track.op_range
    assert preview.ed_range == track.ed_range
    assert preview.write == ((153.5, 224.7), (1348.2, 1416.6))
    assert preview.note is None


def test_infer_ranges_writes_nothing_when_only_op_is_found(tmp_path, golden_srt_path, monkeypatch):
    _probe(monkeypatch, 1500.0)
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)

    assert preview.op_range == (153.486, 224.681)
    assert preview.ed_range is None
    assert preview.write is None
    assert "都不写" in preview.note


def test_infer_ranges_writes_nothing_when_duration_is_unknown(
    tmp_path, golden_srt_path, monkeypatch
):
    _probe(monkeypatch, FFmpegError("moov atom not found"))
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)

    assert preview.duration is None
    assert preview.write is None
    assert "片长未知" in preview.note


def test_infer_ranges_lets_a_misconfigured_ffprobe_through(tmp_path, golden_srt_path, monkeypatch):
    _probe(monkeypatch, FFmpegBinaryError("找不到 ffprobe"))
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e02.mkv")

    with pytest.raises(FFmpegBinaryError):
        infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)


def test_infer_ranges_keeps_existing_manual_ranges(tmp_path, golden_srt_path, fixed_duration):
    cfg = _project(tmp_path)
    video = _touch(tmp_path / "e03.mkv")
    existing = EpisodeConfig(number=3, ed_range=(1300.0, 1420.0))

    preview = infer_ranges(cfg, number=3, srt=golden_srt_path, video=video, existing=existing)

    assert preview.ed_range == (1300.0, 1420.0)
    assert preview.write is None
    assert "已有手填区间" in preview.note


def test_infer_ranges_does_not_pin_project_defaults(tmp_path, golden_srt_path, fixed_duration):
    cfg = _project(tmp_path, "credits:\n  default_op_range: [150, 225]\nepisodes: []\n")
    video = _touch(tmp_path / "e02.mkv")

    preview = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)

    assert preview.op_range == (150.0, 225.0)
    assert preview.write is None
    assert "项目级" in preview.note


def test_av_side_is_frozen_and_defaults_are_optional():
    assert hasattr(importer, "AvSide")
    side = importer.AvSide((100.0, 190.0))
    assert (side.distance, side.reason, side.estimated) == (None, None, False)
    with pytest.raises(FrozenInstanceError):
        side.range = None


def test_infer_ranges_combines_av_op_with_subtitle_ed(tmp_path, golden_srt_path, fixed_duration):
    assert hasattr(importer, "AvSide")
    preview = infer_ranges(
        _project(tmp_path),
        number=2,
        srt=golden_srt_path,
        video=_touch(tmp_path / "e02.mkv"),
        op_av=importer.AvSide((100.25, 190.35), distance=4, estimated=True),
    )

    assert preview.op_range == (100.25, 190.35)
    assert preview.ed_range == (1348.18, 1416.622)
    assert preview.write == ((100.3, 190.4), (1348.2, 1416.6))
    assert (preview.op_source, preview.ed_source) == ("av", "subtitle")
    assert (preview.op_distance, preview.ed_distance) == (4, None)
    assert preview.op_length_estimated is True


def test_infer_ranges_failed_av_ed_never_falls_back_to_subtitles(
    tmp_path, golden_srt_path, fixed_duration
):
    assert hasattr(importer, "AvSide")
    preview = infer_ranges(
        _project(tmp_path),
        number=2,
        srt=golden_srt_path,
        video=_touch(tmp_path / "e02.mkv"),
        ed_av=importer.AvSide(None, distance=28, reason="ED 截图未匹配"),
    )

    assert preview.op_range == (153.486, 224.681)
    assert preview.ed_range is None
    assert preview.write is None
    assert (preview.op_source, preview.ed_source) == ("subtitle", "av")
    assert (preview.op_distance, preview.ed_distance) == (None, 28)
    assert preview.note == "OP 与 ED 未能都识别，两个都不写（ED 截图未匹配）"


def test_infer_ranges_joins_only_failed_av_reasons(tmp_path, golden_srt_path, fixed_duration):
    assert hasattr(importer, "AvSide")
    preview = infer_ranges(
        _project(tmp_path),
        number=2,
        srt=golden_srt_path,
        video=_touch(tmp_path / "e02.mkv"),
        op_av=importer.AvSide(None, reason="OP 截图未匹配"),
        ed_av=importer.AvSide(None, reason="ED 音频未识别"),
    )
    assert preview.write is None
    assert preview.note == "OP 与 ED 未能都识别，两个都不写（OP 截图未匹配；ED 音频未识别）"


def test_infer_ranges_without_av_labels_subtitle_sources(tmp_path, golden_srt_path, fixed_duration):
    preview = infer_ranges(
        _project(tmp_path),
        number=2,
        srt=golden_srt_path,
        video=_touch(tmp_path / "e02.mkv"),
    )
    assert hasattr(preview, "op_source")
    assert (preview.op_source, preview.ed_source) == ("subtitle", "subtitle")
    assert (preview.op_distance, preview.ed_distance) == (None, None)
    assert preview.op_length_estimated is False
    assert preview.write == _WRITE


@pytest.mark.parametrize("manual_side", ["op", "ed"])
def test_infer_ranges_manual_side_preserves_track_and_ignores_av(
    tmp_path, golden_srt_path, fixed_duration, manual_side
):
    assert hasattr(importer, "AvSide")
    cfg = _project(tmp_path)
    existing = EpisodeConfig(number=2, **{f"{manual_side}_range": (100.0, 190.0)})
    video = _touch(tmp_path / "e02.mkv")
    original = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video, existing=existing)
    preview = infer_ranges(
        cfg,
        number=2,
        srt=golden_srt_path,
        video=video,
        existing=existing,
        op_av=importer.AvSide((200.0, 290.0), distance=0, estimated=True),
        ed_av=importer.AvSide(None, distance=30, reason="未匹配"),
    )
    assert preview == original
    assert (preview.op_source, preview.ed_source) == (None, None)
    assert (preview.op_distance, preview.ed_distance) == (None, None)
    assert preview.op_length_estimated is False
    assert preview.write is None


def test_infer_ranges_project_defaults_preserve_unlabelled_track_despite_av(
    tmp_path, golden_srt_path, fixed_duration
):
    assert hasattr(importer, "AvSide")
    cfg = _project(tmp_path, "credits:\n  default_op_range: [150, 225]\nepisodes: []\n")
    video = _touch(tmp_path / "e02.mkv")
    original = infer_ranges(cfg, number=2, srt=golden_srt_path, video=video)
    preview = infer_ranges(
        cfg,
        number=2,
        srt=golden_srt_path,
        video=video,
        op_av=importer.AvSide((200.0, 290.0), distance=0, estimated=True),
    )
    assert preview == original
    assert (preview.op_source, preview.ed_source) == (None, None)
    assert preview.op_length_estimated is False
    assert preview.write is None


def test_infer_ranges_unknown_duration_never_writes_av_ranges(
    tmp_path, golden_srt_path, monkeypatch
):
    assert hasattr(importer, "AvSide")
    _probe(monkeypatch, FFmpegError("moov atom not found"))
    preview = infer_ranges(
        _project(tmp_path),
        number=2,
        srt=golden_srt_path,
        video=_touch(tmp_path / "e02.mkv"),
        op_av=importer.AvSide((100.0, 190.0)),
        ed_av=importer.AvSide((1300.0, 1400.0)),
    )
    assert preview.duration is None
    assert preview.op_range == (100.0, 190.0)
    assert preview.ed_range == (1300.0, 1400.0)
    assert preview.write is None
    assert preview.note == "片长未知，不写区间"


def test_infer_ranges_collapsed_av_range_prevents_both_writes(
    tmp_path, golden_srt_path, fixed_duration
):
    assert hasattr(importer, "AvSide")
    preview = infer_ranges(
        _project(tmp_path),
        number=2,
        srt=golden_srt_path,
        video=_touch(tmp_path / "e02.mkv"),
        op_av=importer.AvSide((10.02, 10.04)),
    )
    assert preview.write is None
    assert preview.note == "取整到 0.1 秒后区间无效，两个都不写"


def _season(tmp_path: Path, golden_srt_path: Path, stems: list[str]) -> Path:
    season = tmp_path / "season"
    season.mkdir(exist_ok=True)
    for stem in stems:
        _touch(season / f"{stem}.mkv")
        _touch(season / f"{stem}.srt", golden_srt_path.read_bytes())
    return season


def _no_llm():
    raise AssertionError("正则已全部命中，不该构造 provider")


class _ClosingProvider(FakeProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FailingProvider:
    async def complete(self, system, user, schema=None):
        raise LLMError("HTTP 429 限流")


async def test_plan_never_builds_a_provider_when_regex_resolves_everything(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [1, 2]
    assert plan.items[0].ranges.write == ((153.5, 224.7), (1348.2, 1416.6))
    assert plan.items[0].prefilled is False
    assert plan.warnings == []


async def test_plan_sends_only_unresolved_names_to_the_llm(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(
        tmp_path, golden_srt_path, ["86 - 11", "Kimi no Na wa", "Mystery Bonus", "Show E01"]
    )
    provider = _ClosingProvider(
        [{"files": [{"id": 1, "episode": 11}, {"id": 2}, {"id": 3, "extra": True}]}]
    )

    plan = await build_import_plan(cfg, season, provider_factory=lambda: provider)

    assert [item.number for item in plan.items] == [1, 11]
    assert [p.name for p in plan.extras] == ["Mystery Bonus.mkv"]
    assert [s.path.name for s in plan.ambiguous] == ["Kimi no Na wa.mkv"]
    sent = json.loads(provider.calls[0]["user"])["files"]
    assert [f["filename"] for f in sent] == ["86 - 11", "Kimi no Na wa", "Mystery Bonus"]
    assert provider.closed is True


async def test_plan_skips_ambiguous_files_when_the_provider_cannot_be_built(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["86 - 11", "Show E01"])

    def factory():
        raise RuntimeError("缺少 Gemini API key，请设置环境变量 TENMIN_GEMINI_API_KEY")

    plan = await build_import_plan(cfg, season, provider_factory=factory)

    assert [item.number for item in plan.items] == [1]
    assert [s.path.name for s in plan.ambiguous] == ["86 - 11.mkv"]
    assert any("TENMIN_GEMINI_API_KEY" in w for w in plan.warnings)


async def test_plan_skips_ambiguous_files_when_the_llm_call_fails(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["86 - 11", "Show E01"])

    plan = await build_import_plan(cfg, season, provider_factory=_FailingProvider)

    assert [item.number for item in plan.items] == [1]
    assert [s.path.name for s in plan.ambiguous] == ["86 - 11.mkv"]
    assert any("429" in w for w in plan.warnings)


async def test_plan_skips_extras_without_asking_the_llm(tmp_path, golden_srt_path, fixed_duration):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show NCOP", "Show SP01", "Show PV", "Show E01"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [1]
    assert sorted(p.name for p in plan.extras) == ["Show NCOP.mkv", "Show PV.mkv", "Show SP01.mkv"]


async def test_plan_skips_every_file_that_shares_an_episode_number(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show EP01 v2", "Show E02"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [2]
    assert sorted(s.path.name for s in plan.ambiguous) == ["Show E01.mkv", "Show EP01 v2.mkv"]
    assert all("重复" in s.reason for s in plan.ambiguous)


async def test_plan_skips_registered_episodes_and_fills_prefilled_ones(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(
        tmp_path,
        "episodes:\n"
        "- number: 1\n  srt: srt/E01.srt\n"
        "- number: 2\n  ed_range: [1300, 1420]\n"
        "- number: 3\n",
    )
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02", "Show E03"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [s.path.name for s in plan.registered] == ["Show E01.mkv"]
    by_number = {item.number: item for item in plan.items}
    assert sorted(by_number) == [2, 3]
    assert by_number[2].prefilled is True
    assert by_number[2].ranges.write is None
    assert by_number[2].ranges.ed_range == (1300.0, 1420.0)
    assert by_number[3].prefilled is True
    assert by_number[3].ranges.write == ((153.5, 224.7), (1348.2, 1416.6))
    assert by_number[3].invalidates_outputs is True
    assert by_number[2].invalidates_outputs is False


async def test_plan_keeps_unpaired_files_out_of_the_way(tmp_path, golden_srt_path, fixed_duration):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01"])
    _touch(season / "Show E02.mkv")
    _touch(season / "Other E05.srt")

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [1]
    assert [p.name for p in plan.unpaired_videos] == ["Show E02.mkv"]
    assert [p.name for p in plan.unpaired_srts] == ["Other E05.srt"]


async def test_plan_records_an_unreadable_episode_and_keeps_going(
    tmp_path, golden_srt_path, fixed_duration, monkeypatch
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"])

    original = infer_ranges

    def fake_infer(cfg, *, number, srt, video, existing=None, **kwargs):
        if number == 1:
            raise ValueError("字幕编码无法识别")
        return original(cfg, number=number, srt=srt, video=video, existing=existing, **kwargs)

    monkeypatch.setattr("tenmin.importer.infer_ranges", fake_infer)

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [2]
    assert [s.path.name for s in plan.failed] == ["Show E01.mkv"]
    assert "字幕编码无法识别" in plan.failed[0].reason


async def test_plan_lets_a_misconfigured_ffprobe_through(tmp_path, golden_srt_path, monkeypatch):
    _probe(monkeypatch, FFmpegBinaryError("找不到 ffprobe"))
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01"])

    with pytest.raises(FFmpegBinaryError):
        await build_import_plan(cfg, season, provider_factory=_no_llm)


async def test_plan_with_unknown_duration_still_imports_without_ranges(
    tmp_path, golden_srt_path, monkeypatch
):
    _probe(monkeypatch, FFmpegError("moov atom not found"))
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01"])

    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    assert [item.number for item in plan.items] == [1]
    assert plan.items[0].ranges.write is None


async def test_apply_registers_items_and_writes_only_the_planned_ranges(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path, "episodes:\n- number: 2\n  ed_range: [1300, 1420]\n")
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"])
    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)

    apply_import_plan(cfg, plan)

    reloaded = load_project(cfg.config_path)
    by_number = {e.number: e for e in reloaded.episodes}
    assert by_number[1].srt == Path("srt/E01.srt")
    assert by_number[1].video == (season / "Show E01.mkv").resolve()
    assert by_number[1].op_range == (153.5, 224.7)
    assert by_number[1].ed_range == (1348.2, 1416.6)
    assert by_number[2].op_range is None
    assert by_number[2].ed_range == (1300.0, 1420.0)
    assert (cfg.root / "srt" / "E01.srt").read_bytes() == golden_srt_path.read_bytes()
    assert (cfg.root / "srt" / "E02.srt").is_file()


@pytest.mark.parametrize("source_yaml", ["  video: video/new.mkv\n", "  srt: srt/E02.srt\n"])
async def test_apply_rejects_sources_registered_after_preview_before_any_copy(
    tmp_path, golden_srt_path, fixed_duration, source_yaml
):
    cfg = _project(tmp_path, "episodes:\n- number: 2\n")
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"])
    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)
    assert [item.number for item in plan.items] == [1, 2]
    current_yaml = "show: 才女的侍从\nslug: saijo\nepisodes:\n- number: 2\n" + source_yaml
    cfg.config_path.write_text(current_yaml, encoding="utf-8")
    destination = _touch(cfg.root / "srt" / "E02.srt", b"manually registered subtitles")

    with pytest.raises(ValueError, match="第 2 集已登记"):
        apply_import_plan(cfg, plan)

    assert cfg.config_path.read_text(encoding="utf-8") == current_yaml
    assert destination.read_bytes() == b"manually registered subtitles"
    assert not (cfg.root / "srt" / "E01.srt").exists()


async def test_apply_carries_current_config_and_manual_ranges_through_multiple_items(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02"])
    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)
    cfg.config_path.write_text(
        "show: 才女的侍从\nslug: saijo\nrender:\n  rate: '+20%'\nepisodes:\n"
        "- number: 1\n  op_range: [10, 100]\n  hardsub: true\n"
        "- number: 2\n  op_range: [20, 110]\n  ed_range: [1300, 1420]\n"
        "- number: 9\n  video: video/E09.mkv\n",
        encoding="utf-8",
    )

    updated = apply_import_plan(cfg, plan)

    reloaded = load_project(cfg.config_path)
    assert updated == reloaded
    assert updated.render.rate == "+20%"
    by_number = {e.number: e for e in updated.episodes}
    assert sorted(by_number) == [1, 2, 9]
    assert by_number[1].op_range == (10.0, 100.0)
    assert by_number[1].ed_range is None
    assert by_number[1].hardsub is True
    assert by_number[2].op_range == (20.0, 110.0)
    assert by_number[2].ed_range == (1300.0, 1420.0)
    assert by_number[9].video == Path("video/E09.mkv")
    for number in (1, 2):
        assert by_number[number].video == (season / f"Show E{number:02d}.mkv").resolve()
        destination = cfg.root / "srt" / f"E{number:02d}.srt"
        assert destination.read_bytes() == golden_srt_path.read_bytes()


async def test_apply_validates_current_config_before_copying(
    tmp_path, golden_srt_path, fixed_duration
):
    cfg = _project(tmp_path)
    season = _season(tmp_path, golden_srt_path, ["Show E01"])
    plan = await build_import_plan(cfg, season, provider_factory=_no_llm)
    current_yaml = "show: 才女的侍从\nslug: saijo\nrender:\n  font_sise: 30\nepisodes: []\n"
    cfg.config_path.write_text(current_yaml, encoding="utf-8")

    with pytest.raises(ValueError, match=r"render\.font_sise"):
        apply_import_plan(cfg, plan)

    assert cfg.config_path.read_text(encoding="utf-8") == current_yaml
    assert not (cfg.root / "srt" / "E01.srt").exists()


_WRITE = ((153.5, 224.7), (1348.2, 1416.6))


def _item(
    number: int,
    *,
    write=_WRITE,
    note=None,
    prefilled=False,
    duration=1420.0,
    ed=(1348.18, 1416.622),
    ignored=(),
) -> ImportItem:
    pair = Pair(
        video=Path(f"/season/Show E{number:02d}.mkv"),
        srt=Path(f"/season/Show E{number:02d}.srt"),
        ignored_srts=tuple(Path(f"/season/{name}") for name in ignored),
    )
    ranges = RangePreview(
        duration=duration, op_range=(153.486, 224.681), ed_range=ed, write=write, note=note
    )
    return ImportItem(number=number, pair=pair, prefilled=prefilled, ranges=ranges)


def test_preview_shows_pairing_ranges_and_the_write_decision():
    plan = ImportPlan(items=[_item(1, ignored=["Show E01.ja.srt"])])

    text = "\n".join(render_preview(plan))

    assert "可导入 1 集" in text
    assert "E01  视频：Show E01.mkv  字幕：Show E01.srt" in text
    assert "片长 1420.0 秒" in text
    assert "OP：153.5–224.7 秒" in text
    assert "ED：1348.2–1416.6 秒" in text
    assert "将写入区间：op_range [153.5, 224.7]，ed_range [1348.2, 1416.6]" in text
    assert "忽略的候选字幕：Show E01.ja.srt" in text
    assert "重跑" not in text


def test_preview_marks_unknown_duration_and_unrecognised_ranges():
    plan = ImportPlan(
        items=[_item(2, write=None, note="片长未知，不写区间", duration=None, ed=None)]
    )

    text = "\n".join(render_preview(plan))

    assert "片长未知" in text
    assert "ED：未识别" in text
    assert "不写区间：片长未知，不写区间" in text


@pytest.mark.parametrize("ed", [(1348.18, 1416.622), None])
def test_preview_labels_both_av_sides_with_distances_even_when_failed(ed):
    item = _item(2, ed=ed, write=None, note="两个都不写")
    assert hasattr(item.ranges, "op_source")
    item = replace(
        item,
        ranges=replace(
            item.ranges,
            op_source="av",
            ed_source="av",
            op_distance=0,
            ed_distance=28,
        ),
    )
    text = "\n".join(render_preview(ImportPlan(items=[item])))
    assert "OP：153.5–224.7 秒（画面+音频，距离 0/64）" in text
    expected_ed = "未识别" if ed is None else "1348.2–1416.6 秒"
    assert f"ED：{expected_ed}（画面+音频，距离 28/64）" in text


def test_preview_shows_subtitle_label_without_distance_and_estimated_op_line():
    item = _item(2)
    assert hasattr(item.ranges, "op_source")
    item = replace(
        item,
        ranges=replace(
            item.ranges,
            op_source="av",
            ed_source="subtitle",
            op_length_estimated=True,
        ),
    )
    lines = render_preview(ImportPlan(items=[item]))
    assert "OP：153.5–224.7 秒（画面+音频）" in lines[2]
    assert "ED：1348.2–1416.6 秒（字幕推断）" in lines[2]
    assert "距离" not in lines[2]
    assert "       OP 长度按 90 秒估计" in lines
    assert "OP 长度按" not in "\n".join(render_preview(ImportPlan(items=[_item(1)])))


def test_preview_warns_that_prefilled_episodes_will_rerun():
    plan = ImportPlan(
        items=[
            _item(3, prefilled=True),
            _item(4, prefilled=True, write=None, note="已有手填区间，原样保留"),
        ]
    )

    lines = render_preview(plan)

    assert "E03" in "\n".join(lines)
    warnings = [line for line in lines if "重跑" in line]
    assert len(warnings) == 1
    assert "E03" in warnings[0]
    assert "预填条目" in "\n".join(lines)


def test_preview_lists_every_skipped_category():
    plan = ImportPlan(
        registered=[Skipped(Path("/s/Show E05.mkv"), "第 5 集已登记，跳过")],
        ambiguous=[Skipped(Path("/s/86 - 11.mkv"), "文件名里认不出集号，LLM 兜底不可用")],
        failed=[Skipped(Path("/s/Show E06.mkv"), "字幕读不出来：坏了")],
        extras=[Path("/s/Show NCOP.mkv")],
        unpaired_videos=[Path("/s/Show E07.mkv")],
        unpaired_srts=[Path("/s/Other.srt")],
    )

    text = "\n".join(render_preview(plan))

    assert "可导入 0 集" in text
    assert "已登记跳过：\n  Show E05.mkv：第 5 集已登记，跳过" in text
    assert "集号无法确定跳过：\n  86 - 11.mkv：文件名里认不出集号，LLM 兜底不可用" in text
    assert "字幕读取失败跳过：\n  Show E06.mkv：字幕读不出来：坏了" in text
    assert "非正片跳过：\n  Show NCOP.mkv" in text
    assert "未配对视频：\n  Show E07.mkv" in text
    assert "未配对字幕：\n  Other.srt" in text


@pytest.fixture
def av_season(tmp_path, golden_srt_path, monkeypatch, fixed_duration):
    """只替换媒体 IO；画面搜索与音频包络、长度、中位数计算都用真实实现。"""
    cfg = _project(tmp_path)
    cfg.render.ffmpeg_path = "custom-ffmpeg"
    season = _season(tmp_path, golden_srt_path, ["Show E01", "Show E02", "Show E03"])
    for side in ("op", "ed"):
        _touch(season / f"{side}.png")
    state = {
        "starts": {1: 12.0, 2: 40.5, 3: 100.3},
        "ends": {1: 1330.0, 2: 1325.2, 3: 1340.0},
        "lengths": {1: 90, 2: 90, 3: 90},
        "distances": {},
        "frame_errors": {},
        "audio_errors": {},
        "windows": [],
        "reads": [],
        "images": [],
        "empty": False,
    }
    shared = np.random.default_rng(42).normal(0, 5000, 150 * av_match.SAMPLE_RATE)

    def number(video):
        return parse_episode_number(video.stem)

    def image_hash(path, *, ffmpeg_path):
        assert ffmpeg_path == "custom-ffmpeg"
        state["images"].append(path.name)
        return 0 if path.stem == "op" else (1 << 64) - 1

    def sample_hashes(video, start, end, fps, *, ffmpeg_path):
        assert ffmpeg_path == "custom-ffmpeg"
        n = number(video)
        state["windows"].append((n, start, end, fps))
        if n in state["frame_errors"]:
            raise state["frame_errors"][n]
        if state["empty"]:
            return []
        samples = []
        for i in range(int(np.ceil((end - start) * fps))):
            time = round(start + i / fps, 3)
            if not start <= time < end:
                continue
            value = (1 << 32) - 1
            if state["starts"][n] <= time < state["starts"][n] + 2:
                value = (1 << state["distances"].get(n, 0)) - 1
            if state["ends"][n] <= time < state["ends"][n] + 2:
                value = (1 << 64) - 1
            samples.append((time, value))
        return samples

    def read_pcm(video, start, duration, *, ffmpeg_path):
        assert ffmpeg_path == "custom-ffmpeg"
        n = number(video)
        state["reads"].append((n, start, duration))
        if n in state["audio_errors"]:
            raise state["audio_errors"][n]
        samples = np.random.default_rng(n).normal(0, 5000, len(shared))
        stop = state["lengths"][n] * av_match.SAMPLE_RATE
        samples[:stop] = shared[:stop]
        return samples.astype("<i2").tobytes()

    monkeypatch.setattr(av_match, "image_hash", image_hash)
    monkeypatch.setattr(av_match, "sample_hashes", sample_hashes)
    monkeypatch.setattr(av_match, "read_pcm", read_pcm)
    return cfg, season, state


async def _av_plan(av_season):
    cfg, season, _ = av_season
    return await build_import_plan(cfg, season, provider_factory=_no_llm)


async def test_av_plan_matches_three_episodes_and_reads_reference_once(av_season, capsys):
    plan = await _av_plan(av_season)
    _, _, state = av_season
    for item in plan.items:
        ranges = item.ranges
        assert ranges.op_range[0] == pytest.approx(state["starts"][item.number], abs=0.1)
        assert ranges.op_range[1] - ranges.op_range[0] == pytest.approx(90, abs=0.3)
        assert ranges.ed_range == (state["ends"][item.number], 1420)
        assert (ranges.op_source, ranges.ed_source) == ("av", "av")
        assert (ranges.op_distance, ranges.ed_distance) == (0, 0)
        assert ranges.write is not None
        assert not ranges.op_length_estimated
    assert [read[0] for read in state["reads"]] == [1, 2, 3]
    assert all(read[2] == 150 for read in state["reads"])
    coarse = [(n, start, end) for n, start, end, fps in state["windows"] if fps == 2]
    assert coarse == [(n, start, end) for n in (1, 2, 3) for start, end in ((0, 300), (1240, 1420))]
    assert capsys.readouterr().out.splitlines() == [f"正在匹配画面 {n}/3" for n in (1, 2, 3)]


@pytest.mark.parametrize("missing", ["op", "ed"])
async def test_av_missing_reference_uses_subtitle_for_that_side(av_season, missing):
    _, season, _ = av_season
    (season / f"{missing}.png").unlink()
    plan = await _av_plan(av_season)
    ranges = plan.items[0].ranges
    assert getattr(ranges, f"{missing}_source") == "subtitle"
    provided = "ed" if missing == "op" else "op"
    assert getattr(ranges, f"{provided}_source") == "av"
    assert getattr(ranges, f"{missing}_range") == (
        (153.486, 224.681) if missing == "op" else (1348.18, 1416.622)
    )


async def test_av_near_miss_does_not_fall_back(av_season):
    av_season[2]["distances"][2] = 15
    ranges = (await _av_plan(av_season)).items[1].ranges
    assert ranges.op_range is None
    assert ranges.op_distance == 15
    assert ranges.write is None
    assert "OP 画面对不上（最佳距离 15/64，超过 12/64）" in ranges.note


async def test_av_single_episode_estimates_without_audio_and_clamps_short_video(
    av_season, monkeypatch
):
    _, season, state = av_season
    for n in (2, 3):
        (season / f"Show E{n:02d}.mkv").unlink()
    state["starts"][1] = 40
    state["ends"][1] = 95
    _probe(monkeypatch, 100)
    ranges = (await _av_plan(av_season)).items[0].ranges
    assert ranges.op_range == (40, 100)
    assert ranges.ed_range == (95, 100)
    assert ranges.op_length_estimated
    assert state["reads"] == []
    assert [(start, end) for _, start, end, fps in state["windows"] if fps == 2] == [
        (0, 100),
        (0, 100),
    ]


async def test_av_median_discards_sixty_second_outlier(av_season):
    _, season, state = av_season
    _touch(season / "Show E04.mkv")
    _touch(season / "Show E04.srt", (season / "Show E01.srt").read_bytes())
    state["starts"][4], state["ends"][4], state["lengths"][4] = 20, 1330, 60
    plan = await _av_plan(av_season)
    assert len(plan.items) == 4
    for item in plan.items:
        assert item.ranges.op_range[1] - item.ranges.op_range[0] == pytest.approx(90, abs=0.3)
        assert not item.ranges.op_length_estimated


async def test_av_inconsistent_lengths_fail_all_usable_sides(av_season):
    av_season[2]["lengths"][3] = 60
    plan = await _av_plan(av_season)
    for item in plan.items:
        assert item.ranges.op_range is None
        assert "各集 OP 长度相差太大（" in item.ranges.note
        assert "秒），定不了终点" in item.ranges.note


async def test_av_no_measured_lengths_estimates_all(av_season):
    av_season[2]["lengths"] = {1: 150, 2: 150, 3: 150}
    for item in (await _av_plan(av_season)).items:
        assert item.ranges.op_range[1] - item.ranges.op_range[0] == pytest.approx(90)
        assert item.ranges.op_length_estimated


@pytest.mark.parametrize("kind", ["frames", "audio"])
async def test_av_broken_episode_does_not_block_others(av_season, kind):
    state = av_season[2]
    state["frame_errors" if kind == "frames" else "audio_errors"][2] = FFmpegError(
        "坏媒体\n详细错误"
    )
    plan = await _av_plan(av_season)
    assert len(plan.items) == 3
    assert plan.items[0].ranges.write is not None
    assert plan.items[2].ranges.write is not None
    broken = plan.items[1].ranges
    assert broken.op_range is None
    assert ("OP 抽帧失败：坏媒体" if kind == "frames" else "OP 音频读不出来：坏媒体") in broken.note
    assert "详细错误" not in broken.note


async def test_av_reference_audio_failure_fails_all_starts(av_season):
    state = av_season[2]
    state["audio_errors"][1] = FFmpegError("坏音频\n细节")
    for item in (await _av_plan(av_season)).items:
        assert item.ranges.op_range is None
        assert "OP 参考集 E01 的音频读不出来：坏音频" in item.ranges.note
    assert [read[0] for read in state["reads"]] == [1]


@pytest.mark.parametrize("pcm", [b"", b"\x00" * 1599], ids=["empty", "less-than-one-frame"])
@pytest.mark.parametrize("number", [1, 2], ids=["reference", "peer"])
async def test_av_decoded_audio_without_envelope_frames_fails(av_season, monkeypatch, pcm, number):
    original = av_match.read_pcm

    def read_pcm(video, start, duration, *, ffmpeg_path):
        if parse_episode_number(video.stem) == number:
            av_season[2]["reads"].append((number, start, duration))
            return pcm
        return original(video, start, duration, ffmpeg_path=ffmpeg_path)

    monkeypatch.setattr(av_match, "read_pcm", read_pcm)
    plan = await _av_plan(av_season)

    for item in plan.items:
        ranges = item.ranges
        if number == 1 or item.number == number:
            assert ranges.op_range is None
            assert ranges.write is None
            assert ranges.op_distance == 0
            assert ranges.ed_range is not None
            reason = "OP 参考集 E01 的音频读不出来：" if number == 1 else "OP 音频读不出来："
            assert reason in ranges.note
            assert not ranges.op_length_estimated
        else:
            assert ranges.write is not None
            assert ranges.op_range[1] - ranges.op_range[0] == pytest.approx(90, abs=0.3)
            assert not ranges.op_length_estimated
    assert [read[0] for read in av_season[2]["reads"]] == ([1] if number == 1 else [1, 2, 3])


async def test_av_reference_is_best_distance_then_episode_number(av_season):
    state = av_season[2]
    state["distances"][1] = 1
    await _av_plan(av_season)
    assert [read[0] for read in state["reads"]] == [2, 1, 3]


@pytest.mark.parametrize("layer", ["image", "frames", "reference_audio", "audio"])
async def test_av_binary_error_propagates(av_season, monkeypatch, layer):
    error = FFmpegBinaryError("找不到 ffmpeg")
    state = av_season[2]
    if layer == "image":

        def broken(*args, **kwargs):
            raise error

        monkeypatch.setattr(av_match, "image_hash", broken)
    elif layer == "frames":
        state["frame_errors"][2] = error
    else:
        state["audio_errors"][1 if layer == "reference_audio" else 2] = error
    with pytest.raises(FFmpegBinaryError, match="找不到 ffmpeg"):
        await _av_plan(av_season)


async def test_av_corrupt_reference_is_rejected_before_llm(av_season, monkeypatch):
    cfg, season, _ = av_season
    _touch(season / "Mystery.mkv")
    _touch(season / "Mystery.srt")

    def broken(*args, **kwargs):
        raise FFmpegError("无效图片\n细节")

    monkeypatch.setattr(av_match, "image_hash", broken)
    with pytest.raises(ValueError) as error:
        await build_import_plan(cfg, season, provider_factory=_no_llm)
    assert (
        str(error.value)
        == f"参考截图读不出来：{season / 'op.png'}（无效图片）。换一张或删掉它再导入"
    )


async def test_av_unknown_duration_does_no_media_matching(av_season, monkeypatch):
    _probe(monkeypatch, FFmpegError("片长未知"))
    for item in (await _av_plan(av_season)).items:
        assert item.ranges.op_range is None
        assert item.ranges.ed_range is None
        assert item.ranges.write is None
    assert av_season[2]["windows"] == []
    assert av_season[2]["reads"] == []


@pytest.mark.parametrize("side", ["op", "ed"])
async def test_av_manual_range_excludes_episode_from_matching(av_season, side, capsys):
    cfg, _, state = av_season
    cfg.episodes = [EpisodeConfig(number=2, **{f"{side}_range": (100, 190)})]
    plan = await _av_plan(av_season)
    assert plan.items[1].ranges.write is None
    assert getattr(plan.items[1].ranges, f"{side}_range") == (100, 190)
    assert {n for n, *_ in state["windows"]} == {1, 3}
    assert [read[0] for read in state["reads"]] == [1, 3]
    assert capsys.readouterr().out.splitlines() == ["正在匹配画面 1/2", "正在匹配画面 2/2"]


async def test_av_empty_search_window_fails_without_subtitle_fallback(av_season):
    av_season[2]["empty"] = True
    for item in (await _av_plan(av_season)).items:
        assert item.ranges.op_range is None
        assert item.ranges.ed_range is None
        assert "OP 搜索窗内没有画面" in item.ranges.note
        assert "ED 搜索窗内没有画面" in item.ranges.note
    assert av_season[2]["reads"] == []
