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
