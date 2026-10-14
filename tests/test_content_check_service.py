"""Shared service cache and publication guarantees, without models or network."""

import json
import os
from pathlib import Path

import pytest

from tenmin.config import ContentCheckConfig
from tenmin.content_check import service
from tenmin.content_check.models import (
    CheckManifest,
    ContentCheckError,
    FrameScore,
    ModelFiles,
    SampledFrame,
    Scores,
)


@pytest.fixture
def fake_check(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    calls = {"extract": 0, "score": 0}

    def extract(video, destination, **kwargs):
        assert kwargs == {"interval": interval[0], "ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}
        assert destination.parent.parent == tmp_path
        assert not destination.exists()
        calls["extract"] += 1
        destination.mkdir(parents=True)
        path = destination / "frame_000001.jpg"
        path.write_bytes(b"image")
        return [SampledFrame(path=path, time=.021)]

    def score(frames, files):
        calls["score"] += 1
        assert frames[0].path.read_bytes() == b"image"
        return [FrameScore(time=.021, image="frames/frame_000001.jpg", label="general",
                           scores=Scores(general=.7, sensitive=.1, questionable=.1,
                                         explicit=.1))]

    interval = [1.0]
    monkeypatch.setattr(service, "extract_frames", extract)
    monkeypatch.setattr(service, "score_frames", score)
    monkeypatch.setattr(service, "load_model_files", lambda **kw: ModelFiles(
        weights=tmp_path / "model.onnx", metadata=tmp_path / "meta.json", sha256="a" * 64))
    return video, tmp_path / "check", calls, interval


def test_public_service_exists():
    from tenmin.content_check import CheckOutcome, check_video
    assert check_video is service.check_video
    assert CheckOutcome.__dataclass_params__.frozen


def run(fake_check, **kwargs):
    video, output, _, interval = fake_check
    cfg = kwargs.pop("cfg", ContentCheckConfig())
    interval[0] = cfg.sample_interval
    return service.check_video(video, output, cfg=cfg, **kwargs)


def manifest(output):
    return CheckManifest.model_validate_json((output / "results.json").read_text(encoding="utf-8"))


def test_threshold_rebuild_reuses_scores(fake_check):
    first = run(fake_check)
    original = manifest(fake_check[1])
    second = run(fake_check, cfg=ContentCheckConfig(sensitive_threshold=.1))
    assert first.candidate_count == 0
    assert second.candidate_count == 1
    assert second.status == "report_rebuilt"
    updated = manifest(fake_check[1])
    assert updated.frames == original.frames
    assert updated.identity == original.identity
    assert updated.extraction_seconds == original.extraction_seconds
    assert updated.inference_seconds == original.inference_seconds
    assert fake_check[2] == {"extract": 1, "score": 1}


def test_exact_cache_no_writes_or_network(fake_check, monkeypatch):
    first = run(fake_check)
    paths = list(fake_check[1].rglob("*"))
    before = {p: p.stat().st_mtime_ns for p in paths}
    # The real loader already performs local-only cache lookup first; preserve offline.
    def local_model(*, offline):
        assert offline is True
        return ModelFiles(Path("weights"), Path("meta"), "a" * 64)
    monkeypatch.setattr(service, "load_model_files", local_model)
    second = run(fake_check, offline=True)
    assert second.status == "cached"
    assert second.report_path == first.report_path
    assert second.frame_count == 1
    assert second.results_path == fake_check[1] / "results.json"
    assert {p: p.stat().st_mtime_ns for p in paths} == before
    assert fake_check[2] == {"extract": 1, "score": 1}


def test_missing_html_rebuilds(fake_check):
    run(fake_check)
    (fake_check[1] / "report.html").unlink()
    assert run(fake_check).status == "report_rebuilt"
    assert fake_check[2] == {"extract": 1, "score": 1}


@pytest.mark.parametrize("change", ["size", "mtime", "path", "interval", "digest",
                                      "revision", "preprocessing", "sampling", "force"])
def test_identity_changes_rerun(fake_check, monkeypatch, change):
    run(fake_check)
    video, _, _, _ = fake_check
    kwargs = {}
    if change == "size":
        video.write_bytes(b"longer video")
    elif change == "mtime":
        os.utime(video, ns=(video.stat().st_atime_ns, video.stat().st_mtime_ns + 1_000_000))
    elif change == "path":
        new = video.with_name("other.mp4")
        new.write_bytes(video.read_bytes())
        fake_check = (new, *fake_check[1:])
    elif change == "interval":
        kwargs["cfg"] = ContentCheckConfig(sample_interval=2)
    elif change == "digest":
        monkeypatch.setattr(service, "load_model_files", lambda **kw: ModelFiles(
            video, video, "b" * 64))
    elif change == "force":
        kwargs["force"] = True
    else:
        name = {"revision": "REVISION", "preprocessing": "PREPROCESSING_VERSION",
                "sampling": "SAMPLING_VERSION"}[change]
        monkeypatch.setattr(service, name, "changed")
    assert run(fake_check, **kwargs).status == "completed"
    assert fake_check[2] == {"extract": 2, "score": 2}


@pytest.mark.parametrize("damage", ["missing_image", "empty", "json", "schema", "escape",
                                      "nan", "fifo", "image_fifo", "image_escape"])
def test_invalid_cache_reruns(fake_check, tmp_path, damage):
    run(fake_check)
    output = fake_check[1]
    results = output / "results.json"
    image = output / "frames/frame_000001.jpg"
    data = json.loads(results.read_text(encoding="utf-8"))
    if damage == "missing_image":
        image.unlink()
    elif damage == "empty":
        results.write_text("", encoding="utf-8")
    elif damage == "json":
        results.write_text("{", encoding="utf-8")
    elif damage == "fifo":
        results.unlink()
        os.mkfifo(results)
    elif damage == "image_fifo":
        image.unlink()
        os.mkfifo(image)
    elif damage == "image_escape":
        image.unlink()
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(b"keep")
        image.symlink_to(outside)
    else:
        if damage == "schema":
            data["schema_version"] = 2
        elif damage == "escape":
            data["frames"][0]["image"] = "../notes.txt"
        elif damage == "nan":
            data["frames"][0]["scores"]["general"] = float("nan")
        results.write_text(json.dumps(data), encoding="utf-8")
    assert run(fake_check).status == "completed"
    assert fake_check[2] == {"extract": 2, "score": 2}
    assert service.load_cached(output, manifest(output).identity) is not None
    if damage == "image_escape":
        assert outside.read_bytes() == b"keep"


def test_unknown_nonempty_directory_refused(fake_check):
    output = fake_check[1]
    output.mkdir()
    (output / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ContentCheckError, match="目录"):
        run(fake_check)
    assert fake_check[2] == {"extract": 0, "score": 0}
    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_owned_rerun_preserves_unknown_files(fake_check):
    run(fake_check)
    notes = fake_check[1] / "notes.txt"
    notes.write_text("keep", encoding="utf-8")
    (fake_check[1] / "frames/stale.jpg").write_bytes(b"stale")
    run(fake_check, force=True)
    assert notes.read_text(encoding="utf-8") == "keep"
    assert not (fake_check[1] / "frames/stale.jpg").exists()


@pytest.mark.parametrize("target", ["output", "frames", "marker"])
def test_symlink_directories_and_marker_refused(fake_check, tmp_path, target):
    run(fake_check)
    output = fake_check[1]
    path = {"output": output, "frames": output / "frames",
            "marker": output / ".tenmin-content-check"}[target]
    moved = tmp_path / f"saved-{target}"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ContentCheckError):
        run(fake_check, force=True)
    assert fake_check[2] == {"extract": 1, "score": 1}


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("boundary", ["invalidate", "remove_frames", "frames", "report.html",
                                        "results.json"])
def test_publication_io_failure_never_leaves_new_completion(
    fake_check, monkeypatch, rebuild, boundary,
):
    if rebuild and boundary in {"remove_frames", "frames"}:
        pytest.skip("report-only rebuild does not replace frames")
    run(fake_check)
    output = fake_check[1]
    identity = manifest(output).identity
    old = (output / "results.json").read_bytes()
    replace = Path.replace
    unlink = Path.unlink
    rmtree = service.shutil.rmtree

    def replace_fail(self, target):
        if Path(target) == output / boundary:
            assert not (output / "results.json").exists()
            raise OSError("publication fault")
        return replace(self, target)

    def unlink_fail(self, *args, **kwargs):
        if boundary == "invalidate" and self == output / "results.json":
            raise OSError("invalidate fault")
        return unlink(self, *args, **kwargs)

    def remove_fail(path, *args, **kwargs):
        if boundary == "remove_frames" and Path(path) == output / "frames":
            assert not (output / "results.json").exists()
            raise OSError("remove fault")
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", replace_fail)
    monkeypatch.setattr(Path, "unlink", unlink_fail)
    monkeypatch.setattr(service.shutil, "rmtree", remove_fail)
    with pytest.raises(ContentCheckError, match=r"失败.*报告未更新"):
        run(fake_check, cfg=ContentCheckConfig(sensitive_threshold=.1), force=not rebuild)
    if boundary == "invalidate":
        assert (output / "results.json").read_bytes() == old
    else:
        assert service.load_cached(output, identity) is None


@pytest.mark.parametrize("error", [KeyboardInterrupt, TypeError, AssertionError, RuntimeError])
@pytest.mark.parametrize("boundary", ["extract_frames", "score_frames", "render_report", "publish"])
def test_interrupt_and_programmer_errors_propagate(fake_check, monkeypatch, error, boundary):
    def fail(*args, **kwargs):
        raise error("injected")
    if boundary == "publish":
        replace = Path.replace
        def replace_fail(self, target):
            if Path(target).name == "report.html":
                raise error("injected")
            return replace(self, target)
        monkeypatch.setattr(Path, "replace", replace_fail)
    else:
        monkeypatch.setattr(service, boundary, fail)
    with pytest.raises(error, match="injected"):
        run(fake_check)
    assert not (fake_check[1] / "results.json").exists()
    assert not list(fake_check[0].parent.glob("tenmin-content-check-*/"))


def test_zero_frames_fails(fake_check, monkeypatch):
    monkeypatch.setattr(service, "extract_frames", lambda *a, **kw: [])
    with pytest.raises(ContentCheckError, match="帧"):
        run(fake_check)
    assert not (fake_check[1] / "results.json").exists()


def test_unavailable_model_does_not_return_old_success(fake_check, monkeypatch):
    run(fake_check)
    def unavailable(**kw):
        raise ContentCheckError("离线模型缺失")
    monkeypatch.setattr(service, "load_model_files", unavailable)
    with pytest.raises(ContentCheckError, match="离线模型缺失"):
        run(fake_check)


def test_video_change_during_scoring_fails_before_publication(fake_check, monkeypatch):
    run(fake_check)
    old = (fake_check[1] / "results.json").read_bytes()
    score = service.score_frames
    def changing_video(*args):
        result = score(*args)
        fake_check[0].write_bytes(b"changed during check")
        return result
    monkeypatch.setattr(service, "score_frames", changing_video)
    with pytest.raises(ContentCheckError, match=r"视频.*变化"):
        run(fake_check, force=True)
    assert (fake_check[1] / "results.json").read_bytes() == old


def test_invalid_video_is_expected_error(fake_check):
    fake_check[0].unlink()
    with pytest.raises(ContentCheckError, match="视频"):
        run(fake_check)


@pytest.mark.parametrize("name", ["report.html", "results.json"])
def test_payload_write_failure_preserves_previous_completion(fake_check, monkeypatch, name):
    run(fake_check)
    old = (fake_check[1] / "results.json").read_bytes()
    write = Path.write_text
    def fail(self, *args, **kwargs):
        if self.name == Path(name).with_name(f"{Path(name).stem}.part{Path(name).suffix}").name:
            raise OSError("write fault")
        return write(self, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(ContentCheckError, match=r"失败.*报告未更新"):
        run(fake_check, force=True)
    assert (fake_check[1] / "results.json").read_bytes() == old


@pytest.mark.parametrize("kind", ["wrong", "fifo", "missing", "bad_utf8"])
def test_invalid_ownership_marker_preserves_directory(fake_check, kind):
    run(fake_check)
    output = fake_check[1]
    marker = output / ".tenmin-content-check"
    marker.unlink()
    if kind == "wrong":
        marker.write_text("2\n", encoding="utf-8")
    elif kind == "fifo":
        os.mkfifo(marker)
    elif kind == "bad_utf8":
        marker.write_bytes(b"\xff")
    old = (output / "results.json").read_bytes()
    with pytest.raises(ContentCheckError):
        run(fake_check, force=True)
    assert (output / "results.json").read_bytes() == old


def test_video_change_during_payload_write_fails_before_invalidation(fake_check, monkeypatch):
    run(fake_check)
    old = (fake_check[1] / "results.json").read_bytes()
    write = Path.write_text
    def changing(self, *args, **kwargs):
        result = write(self, *args, **kwargs)
        if self.name == "results.part.json":
            fake_check[0].write_bytes(b"changed during publication preparation")
        return result
    monkeypatch.setattr(Path, "write_text", changing)
    with pytest.raises(ContentCheckError, match=r"视频.*变化"):
        run(fake_check, force=True)
    assert (fake_check[1] / "results.json").read_bytes() == old


@pytest.mark.parametrize("damage", ["count", "ranges"])
def test_schema_valid_stale_candidates_rebuild_without_inference(fake_check, monkeypatch, damage):
    score = service.score_frames
    def two_scores(frames, files):
        first = score(frames, files)[0]
        second = first.model_copy(update={"time": 1.021, "image": "frames/frame_000002.jpg"})
        frames[0].path.with_name("frame_000002.jpg").write_bytes(b"image two")
        return [first, second]
    monkeypatch.setattr(service, "score_frames", two_scores)
    cfg = ContentCheckConfig(sensitive_threshold=.1)
    run(fake_check, cfg=cfg)
    output = fake_check[1]
    original = manifest(output)
    data = original.model_dump()
    if damage == "count":
        data.update(candidate_count=0, candidate_ranges=[])
    else:
        data["candidate_ranges"] = [
            {"first": frame.time, "last": frame.time, "count": 1}
            for frame in original.frames
        ]
    invalid_summary = CheckManifest.model_validate(data)
    (output / "results.json").write_text(invalid_summary.model_dump_json(), encoding="utf-8")
    outcome = run(fake_check, cfg=cfg)
    assert outcome.status == "report_rebuilt"
    assert outcome.candidate_count == 2
    assert manifest(output) == original
    assert (output / "report.html").read_text(encoding="utf-8") == service.render_report(original)
    assert fake_check[2] == {"extract": 1, "score": 1}
    assert run(fake_check, cfg=cfg).status == "cached"


@pytest.mark.parametrize("rebuild", [False, True])
def test_original_video_symlink_retarget_before_publication_fails(
    fake_check, monkeypatch, rebuild,
):
    video, output, calls, interval = fake_check
    link = video.with_name("input.mp4")
    link.symlink_to(video)
    linked_check = (link, output, calls, interval)
    run(linked_check)
    old = (output / "results.json").read_bytes()
    other = video.with_name("other.mp4")
    other.write_bytes(video.read_bytes())
    os.utime(other, ns=(video.stat().st_atime_ns, video.stat().st_mtime_ns))
    write = Path.write_text
    def retarget(self, *args, **kwargs):
        result = write(self, *args, **kwargs)
        if self.name == "results.part.json":
            link.unlink()
            link.symlink_to(other)
        return result
    monkeypatch.setattr(Path, "write_text", retarget)
    with pytest.raises(ContentCheckError, match=r"视频.*变化"):
        run(linked_check, force=not rebuild, cfg=ContentCheckConfig(sensitive_threshold=.1))
    assert (output / "results.json").read_bytes() == old
    assert manifest(output).identity.video_path == str(video.resolve())


@pytest.mark.parametrize("rebuild", [False, True])
def test_publication_marker_invalid_utf8_is_expected_error(fake_check, monkeypatch, rebuild):
    run(fake_check)
    output = fake_check[1]
    old = (output / "results.json").read_bytes()
    write = Path.write_text
    def corrupt_marker(self, *args, **kwargs):
        result = write(self, *args, **kwargs)
        if self.name == "results.part.json":
            (output / ".tenmin-content-check").write_bytes(b"\xff")
        return result
    monkeypatch.setattr(Path, "write_text", corrupt_marker)
    with pytest.raises(ContentCheckError, match=r"失败.*报告未更新") as caught:
        run(fake_check, force=not rebuild, cfg=ContentCheckConfig(sensitive_threshold=.1))
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)
    assert (output / "results.json").read_bytes() == old


@pytest.mark.parametrize("error", [TypeError, AssertionError, RuntimeError, KeyboardInterrupt])
def test_publication_marker_unknown_read_errors_propagate(fake_check, monkeypatch, error):
    run(fake_check)
    output = fake_check[1]
    old = (output / "results.json").read_bytes()
    read = Path.read_text
    marker_reads = 0
    def fail_recheck(self, *args, **kwargs):
        nonlocal marker_reads
        if self == output / ".tenmin-content-check":
            marker_reads += 1
            if marker_reads == 2:
                raise error("marker programmer/interrupt fault")
        return read(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", fail_recheck)
    with pytest.raises(error, match="marker programmer/interrupt fault"):
        run(fake_check, force=True)
    assert (output / "results.json").read_bytes() == old
