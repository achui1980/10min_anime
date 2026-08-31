from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def golden_srt_path() -> Path:
    path = FIXTURES / "saijo_e02.srt"
    if not path.exists():
        pytest.skip("缺少 tests/fixtures/saijo_e02.srt")
    return path


@pytest.fixture(scope="session")
def golden_track(golden_srt_path):
    pytest.importorskip("tenmin.ingest.normalize")
    from tenmin.ingest.normalize import build_track

    return build_track(golden_srt_path, episode=2, show_title="才女的侍从")


@pytest.fixture(scope="session")
def lines_for_idx():
    """按原始 SRT 序号取行。一个序号可能对应多行（双轨拆行），也可能被并进别人的 merged_from。"""

    def _lookup(track, idx: int):
        return [ln for ln in track.lines if ln.idx == idx or idx in ln.merged_from]

    return _lookup
