from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from tenmin.config import ContentCheckConfig
from tenmin.content_check.candidates import screen_frames
from tenmin.content_check.models import (
    CacheIdentity,
    CandidateRange,
    CheckManifest,
    CheckOutcome,
    ContentCheckError,
    FrameScore,
    ModelFiles,
    SampledFrame,
    Scores,
)


def frame(time=0, index=1, scores=(0.2, 0.5, 0.2, 0.1)):
    names = ("general", "sensitive", "questionable", "explicit")
    return FrameScore(
        time=time,
        image=f"frames/frame_{index:06d}.jpg",
        label=names[max(range(4), key=lambda i: scores[i])],
        scores=Scores(**dict(zip(names, scores, strict=True))),
    )


def manifest_data():
    return {
        "identity": {
            "video_path": "/videos/E01.mp4",
            "video_size": 100,
            "video_mtime_ns": 123,
            "repository": "deepghs/anime_dbrating",
            "revision": "7af21db648acdeb74f5c334abda9dd7403407b3c",
            "model": "mobilenetv3_large_100_v0_ls0.2",
            "model_sha256": "c7d6fd0cd71c48616fa87fe2bca87f8e1775e9a3b96d12797e2144fca3543362",
            "preprocessing_version": "rgb-bilinear-chw-v1",
            "sampling_version": "select-relative-pts-dar-v1",
            "sample_interval": 1.0,
        },
        "providers": ["CPUExecutionProvider"],
        "extraction_seconds": 0.1,
        "inference_seconds": 0.2,
        "sensitive_threshold": 0.5,
        "adult_threshold": 0.5,
        "frames": [frame().model_dump()],
        "candidate_count": 1,
        "candidate_ranges": [{"first": 0, "last": 0, "count": 1}],
    }


def test_candidate_boundary_and_gap():
    frames = [frame(t, i) for i, t in enumerate([0.021, 1.521, 3.022], 1)]
    selected, ranges = screen_frames(frames, ContentCheckConfig())
    assert selected == frames
    assert [(r.first, r.last, r.count) for r in ranges] == [
        (0.021, 1.521, 2), (3.022, 3.022, 1)
    ]


def test_adult_sum_hits_equality_independently_of_sensitive():
    adult = frame(scores=(0.4, 0.1, 0.25, 0.25))
    selected, ranges = screen_frames([adult], ContentCheckConfig())
    assert selected == [adult]
    assert ranges == [CandidateRange(first=0, last=0, count=1)]


@pytest.mark.parametrize("frames", [[], [frame(scores=(0.4, 0.2, 0.2, 0.2))]])
def test_no_candidates_when_both_scores_are_below_threshold(frames):
    assert screen_frames(frames, ContentCheckConfig()) == ([], [])


def test_grouping_sorts_by_time_uses_interval_and_does_not_modify_input():
    frames = [frame(4, 1), frame(0, 2), frame(3, 3)]
    before = [f.model_dump() for f in frames]
    selected, ranges = screen_frames(tuple(frames), ContentCheckConfig(sample_interval=2))
    assert [f.time for f in selected] == [0, 3, 4]
    assert ranges == [CandidateRange(first=0, last=4, count=3)]
    assert [f.model_dump() for f in frames] == before
    assert selected[0] is frames[1]


def test_only_selected_points_bridge_ranges_and_thresholds_are_configurable():
    frames = [frame(0, 1), frame(1, 2, (0.7, 0.1, 0.1, 0.1)), frame(2, 3)]
    selected, ranges = screen_frames(frames, ContentCheckConfig())
    assert selected == [frames[0], frames[2]]
    assert [r.count for r in ranges] == [1, 1]
    assert screen_frames(frames, ContentCheckConfig(
        sensitive_threshold=0.6, adult_threshold=0.6
    )) == ([], [])
    assert screen_frames(frames, ContentCheckConfig(
        sensitive_threshold=0, adult_threshold=1
    ))[0] == frames


@pytest.mark.parametrize("field", ["general", "sensitive", "questionable", "explicit"])
@pytest.mark.parametrize("value", [-1e-9, 1 + 1e-9, float("nan"), float("inf")])
def test_scores_reject_nonfinite_or_out_of_range_values(field, value):
    data = dict.fromkeys(["general", "sensitive", "questionable", "explicit"], 0.25)
    data[field] = value
    with pytest.raises(ValidationError, match=field):
        Scores(**data)


@pytest.mark.parametrize("general", [0.2489, 0.2511])
def test_scores_reject_sum_outside_tolerance(general):
    with pytest.raises(ValidationError, match=r"general.*sensitive.*questionable.*explicit"):
        Scores(general=general, sensitive=0.25, questionable=0.25, explicit=0.25)


@pytest.mark.parametrize("general", [0.249, 0.2509, 0.251])
def test_scores_preserve_original_values_within_sum_tolerance(general):
    scores = Scores(general=general, sensitive=0.25, questionable=0.25, explicit=0.25)
    assert scores.general == general
    assert scores.sensitive == scores.questionable == scores.explicit == 0.25


@pytest.mark.parametrize("time", [-1, float("nan"), float("inf")])
def test_frame_rejects_invalid_time(time):
    data = frame().model_dump()
    data["time"] = time
    with pytest.raises(ValidationError, match="time"):
        FrameScore(**data)


@pytest.mark.parametrize("image", [
    "../frames/frame_000001.jpg", "/frames/frame_000001.jpg",
    "frames/../frame_000001.jpg", "frames\\frame_000001.jpg",
    "frames/frame_00001.jpg", "frames/frame_000001.jpg\n",
    "frames/frame_000001.png", "frames/frame_１２３４５６.jpg",
])
def test_frame_rejects_unsafe_or_unexpected_image_paths(image):
    data = frame().model_dump()
    data["image"] = image
    with pytest.raises(ValidationError, match="image"):
        FrameScore(**data)


def test_frame_label_uses_fixed_class_order_to_break_argmax_ties():
    tied = frame(scores=(0.25, 0.25, 0.25, 0.25))
    assert tied.label == "general"
    data = tied.model_dump()
    data["label"] = "sensitive"
    with pytest.raises(ValidationError, match="label"):
        FrameScore(**data)
    assert frame(index=1000000).image == "frames/frame_1000000.jpg"


@pytest.mark.parametrize(("field", "value"), [
    ("first", -1), ("last", float("nan")), ("last", float("inf")),
    ("last", -1), ("first", 1), ("count", 0), ("count", -1),
])
def test_candidate_range_rejects_invalid_bounds_or_count(field, value):
    data = {"first": 0, "last": 0, "count": 1}
    data[field] = value
    with pytest.raises(ValidationError, match=field):
        CandidateRange(**data)


def test_manifest_roundtrip_and_zero_candidates_are_valid():
    data = manifest_data()
    data.update(candidate_count=0, candidate_ranges=[])
    manifest = CheckManifest(**data)
    assert CheckManifest.model_validate_json(manifest.model_dump_json()) == manifest
    assert manifest.schema_version == 1
    assert screen_frames(manifest.frames, ContentCheckConfig())[0] == manifest.frames


@pytest.mark.parametrize(("field", "value"), [
    ("schema_version", 2), ("frames", []), ("candidate_count", -1),
    ("candidate_count", 2), ("extraction_seconds", -1),
    ("inference_seconds", float("inf")), ("sensitive_threshold", float("nan")),
    ("adult_threshold", 1.01),
])
def test_manifest_rejects_invalid_fields(field, value):
    data = manifest_data()
    data[field] = value
    with pytest.raises(ValidationError, match=field):
        CheckManifest(**data)


def test_manifest_rejects_decreasing_times_and_duplicate_images():
    data = manifest_data()
    data["frames"] = [frame(2, 1), frame(1, 2)]
    with pytest.raises(ValidationError, match=r"frames.*time"):
        CheckManifest(**data)
    data["frames"] = [frame(0, 1), frame(0, 1)]
    with pytest.raises(ValidationError, match=r"frames.*image"):
        CheckManifest(**data)
    data["frames"] = [frame(0, 1), frame(0, 2)]
    assert len(CheckManifest(**data).frames) == 2


@pytest.mark.parametrize("ranges", [
    [{"first": 0, "last": 1, "count": 1}],
    [{"first": 0, "last": 0, "count": 2}],
])
def test_manifest_rejects_candidate_ranges_outside_frame_bounds(ranges):
    data = manifest_data()
    data["candidate_ranges"] = ranges
    with pytest.raises(ValidationError, match="candidate_ranges"):
        CheckManifest(**data)


@pytest.mark.parametrize(("field", "value"), [
    ("video_size", -1), ("video_mtime_ns", -1),
    ("sample_interval", 0), ("sample_interval", float("nan")),
    ("sample_interval", float("inf")), ("video_path", "relative/E01.mp4"),
])
def test_cache_identity_rejects_invalid_fields(field, value):
    data = manifest_data()["identity"]
    data[field] = value
    with pytest.raises(ValidationError, match=field):
        CacheIdentity(**data)


@pytest.mark.parametrize(("model", "data"), [
    (Scores, {"general": 1, "sensitive": 0, "questionable": 0, "explicit": 0}),
    (FrameScore, frame().model_dump()),
    (CandidateRange, {"first": 0, "last": 0, "count": 1}),
    (CacheIdentity, manifest_data()["identity"]),
    (CheckManifest, manifest_data()),
])
def test_contract_models_reject_unknown_fields(model, data):
    with pytest.raises(ValidationError, match="unknown"):
        model(**data, unknown=True)


@pytest.mark.parametrize("value", [
    SampledFrame(path=Path("frame.jpg"), time=0),
    ModelFiles(weights=Path("model.onnx"), metadata=Path("meta.json"), sha256="abc"),
    CheckOutcome(status="completed", candidate_count=0, frame_count=1,
                 report_path=Path("report.html"), results_path=Path("results.json")),
])
def test_return_value_dataclasses_are_frozen(value):
    with pytest.raises(FrozenInstanceError):
        value.extra = True


def test_content_check_error_is_a_runtime_error():
    assert isinstance(ContentCheckError("检查失败"), RuntimeError)
