import pytest

from tenmin.timecode import format_timestamp, parse_timestamp


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("00:00:07,120", 7.12),
        ("00:00:07.120", 7.12),
        ("0:02:33,486", 153.486),
        ("00:22:28,180", 1348.18),
        ("01:00:00,000", 3600.0),
        ("00:00:00,5", 0.5),
    ],
)
def test_parse_timestamp(text, seconds):
    assert parse_timestamp(text) == pytest.approx(seconds, abs=1e-6)


def test_parse_timestamp_rejects_garbage():
    with pytest.raises(ValueError):
        parse_timestamp("not a timestamp")


@pytest.mark.parametrize(
    "seconds,text",
    [
        (7.12, "00:00:07.120"),
        (153.486, "00:02:33.486"),
        (1348.18, "00:22:28.180"),
        (3600.0, "01:00:00.000"),
        (0.0, "00:00:00.000"),
    ],
)
def test_format_timestamp(seconds, text):
    assert format_timestamp(seconds) == text


def test_roundtrip():
    for seconds in (0.0, 7.12, 153.486, 1416.622):
        assert parse_timestamp(format_timestamp(seconds)) == pytest.approx(seconds, abs=1e-6)


def test_format_clamps_negative():
    assert format_timestamp(-3.0) == "00:00:00.000"
