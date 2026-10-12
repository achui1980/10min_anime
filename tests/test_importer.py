from pathlib import Path

import pytest

from tenmin.importer import (
    LANGUAGE_SUFFIXES,
    Ambiguous,
    is_extra,
    parse_episode_number,
    scan_directory,
)


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
