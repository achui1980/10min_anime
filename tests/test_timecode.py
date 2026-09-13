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


# --- 时间戳正则收敛成一份 ---


def test_timestamp_from_groups_matches_parse_timestamp():
    """解析器已经捕获过 4 个分组，不该再对同一子串跑一遍完整正则。"""
    from tenmin.timecode import timestamp_from_groups

    for text, groups in [
        ("00:00:07,120", ("00", "00", "07", "120")),
        ("0:02:33,486", ("0", "02", "33", "486")),
        ("00:00:00,5", ("00", "00", "00", "5")),
        ("01:00:00.000", ("01", "00", "00", "000")),
    ]:
        assert timestamp_from_groups(*groups) == parse_timestamp(text)


def test_timestamp_pattern_captures_four_groups_in_order():
    import re

    from tenmin.timecode import TIMESTAMP_PATTERN

    match = re.compile(TIMESTAMP_PATTERN).fullmatch("12:34:56,789")
    assert match is not None
    assert match.groups() == ("12", "34", "56", "789")


def test_parse_timestamp_rejects_four_digit_millis():
    """`,0000` 不是合法 SRT 毫秒。`_TS` 本来就拒（`\\s*$` 挡住多出来的那位），
    这条把它钉住，因为共享片段现在还要给 srt_parser 的非锚定搜索用。
    """
    with pytest.raises(ValueError):
        parse_timestamp("00:00:01,0000")


def test_parse_timestamp_rejects_minute_and_second_overflow():
    for text in ["00:75:30,000", "00:00:75,000"]:
        with pytest.raises(ValueError):
            parse_timestamp(text)


# --- nan / inf ---


def test_format_timestamp_rejects_nan_and_infinity():
    """原来靠 `int(round(...))` 自己炸：nan 抛 ValueError、inf 抛 OverflowError，
    消息都是 CPython 的「cannot convert float ...」，看不出是哪个数据坏了。

    OverflowError 不在 cli.PIPELINE_ERRORS 里，所以 inf 会直接把 traceback 糊到用户
    脸上；换成 ValueError 之后走的是正常的红字报错 + exit 1。
    """
    for value in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="秒数"):
            format_timestamp(value)


def test_readable_seconds_rejects_nan_and_infinity():
    """`readable_seconds` 也有同一个洞（`round(nan)` / `round(inf)`），
    而且连 `-inf` 都挡不住 —— format_timestamp 那边靠 `seconds < 0` 先夹到 0 侥幸躲过。
    """
    for value in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="秒数"):
            readable_seconds(value)


def test_negative_seconds_still_clamp_to_zero():
    assert format_timestamp(-5.0) == "00:00:00.000"
    assert readable_seconds(-5.0) == "0 秒"
