from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def golden_srt_path() -> Path:
    path = FIXTURES / "saijo_e02.srt"
    if not path.exists():
        pytest.skip("缺少 tests/fixtures/saijo_e02.srt")
    return path
