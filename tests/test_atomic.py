"""产物原子写。核心不变量：路径上要么是完整的旧内容/不存在，要么是完整的新内容。"""

from __future__ import annotations

import pytest

from tenmin.atomic import PART_SUFFIX, atomic_path, copy_file, part_path, write_text


def test_part_path_keeps_suffix_so_ffmpeg_can_infer_container(tmp_path):
    """`.part` 必须插在扩展名之前：ffmpeg 靠扩展名推断输出容器格式。"""
    assert part_path(tmp_path / "E02.mixed.m4a").name == "E02.mixed.part.m4a"
    assert part_path(tmp_path / "E02.mp4").name == "E02.part.mp4"


def test_part_path_stays_in_the_same_directory():
    """跨文件系统的 os.replace 会抛 OSError，所以临时文件必须同目录。"""
    target = "/tmp/deep/dir/E02.mixed.m4a"
    from pathlib import Path

    assert part_path(Path(target)).parent == Path(target).parent


def test_part_path_handles_suffixless_names(tmp_path):
    assert part_path(tmp_path / "artifact").name == f"artifact{PART_SUFFIX}"


def test_write_text_creates_parents_and_writes(tmp_path):
    target = tmp_path / "out" / "E02.narration.txt"
    write_text(target, "旁白正文")
    assert target.read_text(encoding="utf-8") == "旁白正文"


def test_write_text_leaves_no_part_file_behind(tmp_path):
    target = tmp_path / "E02.timeline.json"
    write_text(target, "{}")
    assert not part_path(target).exists()
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_path_yields_a_temp_path_then_renames(tmp_path):
    target = tmp_path / "06_audio" / "E02.mixed.m4a"
    with atomic_path(target) as tmp:
        assert tmp == part_path(target)
        tmp.write_bytes(b"\x00\x01")
        # 关键：块内目标路径还不存在，半成品不会被 _is_fresh 当成产物
        assert not target.exists()
    assert target.read_bytes() == b"\x00\x01"


def test_atomic_path_removes_the_temp_file_when_the_body_raises(tmp_path):
    target = tmp_path / "E02.mp4"
    with pytest.raises(RuntimeError):
        with atomic_path(target) as tmp:
            tmp.write_bytes(b"half written")
            raise RuntimeError("ffmpeg 挂了")
    assert not target.exists()
    assert not part_path(target).exists()


def test_atomic_path_cleans_up_on_keyboard_interrupt(tmp_path):
    """Ctrl-C 是这套机制存在的头号原因，而它不是 Exception 的子类。"""
    target = tmp_path / "E02.mixed.m4a"
    with pytest.raises(KeyboardInterrupt):
        with atomic_path(target) as tmp:
            tmp.write_bytes(b"truncated")
            raise KeyboardInterrupt
    assert not target.exists()
    assert not part_path(target).exists()


def test_atomic_path_keeps_the_previous_artifact_when_the_body_raises(tmp_path):
    """失败时旧产物必须原封不动 —— 这正是「-y 直接写目标路径」做不到的。"""
    target = tmp_path / "E02.mixed.m4a"
    good = "上一轮的好产物".encode()
    target.write_bytes(good)
    before = target.stat().st_mtime_ns
    with pytest.raises(RuntimeError):
        with atomic_path(target):
            raise RuntimeError("boom")
    assert target.read_bytes() == good
    assert target.stat().st_mtime_ns == before


def test_atomic_path_clears_a_stale_part_file_from_an_earlier_crash(tmp_path):
    target = tmp_path / "E02.mp4"
    stale = part_path(target)
    stale.write_bytes("上次被打断留下的垃圾".encode())
    with atomic_path(target) as tmp:
        assert not tmp.exists()
        tmp.write_bytes(b"new")
    assert target.read_bytes() == b"new"


def test_atomic_path_overwrites_an_existing_artifact(tmp_path):
    target = tmp_path / "E02.script.json"
    target.write_text("旧的", encoding="utf-8")
    with atomic_path(target) as tmp:
        tmp.write_text("新的", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "新的"


def test_copy_file_is_atomic(tmp_path):
    src = tmp_path / "src.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:02,000\n台词\n", encoding="utf-8")
    dest = tmp_path / "srt" / "E02.srt"
    copy_file(src, dest)
    assert dest.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")
    assert not part_path(dest).exists()


def test_copy_file_keeps_the_old_copy_when_the_source_disappears(tmp_path):
    dest = tmp_path / "E02.srt"
    dest.write_text("旧字幕", encoding="utf-8")
    with pytest.raises(OSError):
        copy_file(tmp_path / "missing.srt", dest)
    assert dest.read_text(encoding="utf-8") == "旧字幕"
    assert not part_path(dest).exists()
