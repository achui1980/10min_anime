"""截图匹配的纯函数契约，不依赖视频或文件 IO。"""

import numpy as np
import pytest

from tenmin.ingest.av_match import (
    HASH_BITS,
    HASH_BYTES,
    HASH_HEIGHT,
    HASH_WIDTH,
    MAX_DISTANCE,
    best_frame,
    dhash,
    hamming,
)


def test_dhash_falling_rows_are_all_ones():
    assert (HASH_WIDTH, HASH_HEIGHT, HASH_BYTES, HASH_BITS, MAX_DISTANCE) == (9, 8, 72, 64, 12)
    gray = np.tile(np.arange(9, 0, -1), (8, 1))
    assert dhash(gray) == (1 << 64) - 1


def test_dhash_rising_rows_are_zero():
    gray = np.tile(np.arange(9), (8, 1))
    assert dhash(gray) == 0


def test_dhash_bytes_match_array():
    gray = np.random.default_rng(1).integers(30, 201, size=(8, 9), dtype=np.uint8)
    expected = dhash(gray)
    assert dhash(gray.tobytes()) == expected
    assert dhash(bytearray(gray.tobytes())) == expected
    assert dhash(gray.ravel().tolist()) == expected


def test_dhash_rejects_wrong_size():
    with pytest.raises(ValueError, match="72"):
        dhash(bytes(71))


def test_dhash_tolerates_brightness_and_contrast_changes():
    gray = np.random.default_rng(1).integers(30, 201, size=(8, 9))
    reference = dhash(gray)
    assert dhash(gray + 40) == reference
    assert hamming(dhash(np.round(gray * 0.6 + 20)), reference) <= 2


def test_dhash_unrelated_frames_exceed_threshold():
    reference = dhash(np.random.default_rng(10).integers(30, 201, size=(8, 9)))
    for seed in range(11, 21):
        gray = np.random.default_rng(seed).integers(30, 201, size=(8, 9))
        assert hamming(dhash(gray), reference) > MAX_DISTANCE


def test_hamming_counts_differing_bits():
    assert hamming(0b1011, 0b0001) == 2
    assert hamming(0, (1 << 64) - 1) == 64


def test_best_frame_keeps_first_tie_and_reports_distance_above_threshold():
    assert best_frame([15, 1, 3, 1], 0) == (1, 1)
    assert best_frame([(1 << 13) - 1], 0) == (0, 13)


def test_best_frame_rejects_empty_sequence():
    with pytest.raises(ValueError, match="没有可比较的帧"):
        best_frame([], 0)
