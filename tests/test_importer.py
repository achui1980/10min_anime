from pathlib import Path

import pytest

from tenmin.importer import LANGUAGE_SUFFIXES, scan_directory


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
