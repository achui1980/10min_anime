"""真跑一遍 ffmpeg 的 ebur128，确认 `_parse_ebur128_i` 认得真实输出的格式。

跟 test_render_audio.py 里那些纯图构造测试不同——这里刻意**不** monkeypatch
任何东西，就是要拿真 ffmpeg 的真 stderr 去验解析器。没装 ffmpeg 的机器上跳过，
不是失败：这条检查跟渲染成片（`render` marker）没关系，是纯粹的「解析器认不认
得真实工具的输出格式」，不该被那个更重的标记盖住，也不该在没有 ffmpeg 时报错。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tenmin.render.audio import _parse_ebur128_i
from tenmin.render.ffmpeg import run


def test_ebur128_parser_reads_a_real_ffmpeg_report(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("本机没有 ffmpeg，跳过（这条检查只关心解析器认不认得真实输出）")

    wav = tmp_path / "voice.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
            "-c:a", "pcm_s16le", str(wav),
        ],
        check=True,
    )

    stderr = run(
        ["-hide_banner", "-i", str(wav), "-af", "ebur128=peak=true", "-f", "null", "-"],
        ffmpeg="ffmpeg",
    )

    assert _parse_ebur128_i(stderr) is not None
    # 手造的静音场景不靠真 ffmpeg 跑——静音检测的深入覆盖留给后续接线，这里只是
    # 确认解析器本身认得 `-inf LUFS` 这种合法输出。
    assert _parse_ebur128_i("I: -inf LUFS") is None
