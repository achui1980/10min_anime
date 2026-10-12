import json
from pathlib import Path

import pytest

from tenmin.config import EpisodeConfig, ProjectConfig, load_project
from tenmin.importer import (
    EXTRA,
    LANGUAGE_SUFFIXES,
    Ambiguous,
    infer_ranges,
    is_extra,
    parse_episode_number,
    resolve_ambiguous,
    round_range,
    scan_directory,
)
from tenmin.ingest.normalize import build_track
from tenmin.render.ffmpeg import FFmpegBinaryError, FFmpegError

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
