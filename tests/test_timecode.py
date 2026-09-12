import pytest

from tenmin.timecode import format_timestamp, parse_timestamp, readable_seconds


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


@pytest.mark.parametrize(
    "seconds,text",
    [
        (0.0, "0 秒"),
        (0.4, "0 秒"),
        (1.0, "1 秒"),
        (59.0, "59 秒"),
        (59.4, "59 秒"),
        (60.0, "1 分 0 秒"),
        (220.0, "3 分 40 秒"),
        (245.0, "4 分 5 秒"),
        (1416.622, "23 分 37 秒"),
    ],
)
def test_readable_seconds(seconds, text):
    assert readable_seconds(seconds) == text


def test_readable_seconds_clamps_negative():
    assert readable_seconds(-5.0) == "0 秒"


def test_readable_seconds_never_emits_sixty_seconds():
    """旧的 single.py 实现是 int(s//60) 分 + f"{rest:.0f}" 秒，rest=59.6 时会吐
    「23 分 60 秒」。合并后先 round 到整秒再 divmod，进位不会越界。"""
    assert readable_seconds(1439.6) == "24 分 0 秒"
