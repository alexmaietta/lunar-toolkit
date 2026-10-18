"""Shared pytest fixtures (offline: everything comes from tests/fixtures)."""

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CUMINDEX_LBL = FIXTURES / "CUMINDEX_0066C.LBL"
CUMINDEX_TAB = FIXTURES / "CUMINDEX_head200.TAB"  # first 200 real records, 903 B each


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "tools(*names): runs ISIS / ASP / GDAL tools, skipped when any of *names* is missing "
        "(run with: pytest -m tools; set ASP_ROOT / ISISROOT or activate the environment)",
    )


def pytest_runtest_setup(item):
    from lunar_toolkit.tools import find_tool

    for marker in item.iter_markers("tools"):
        missing = [name for name in marker.args if find_tool(name) is None]
        if missing:
            pytest.skip(f"tools not found: {', '.join(missing)} (set ASP_ROOT or ISISROOT)")


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Never touch the real user data / cache dirs, and never pace or share host throttles."""
    from lunar_toolkit import downloader

    monkeypatch.setenv("LUNAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(downloader.MIN_INTERVAL_ENV, "0")
    monkeypatch.setattr(downloader, "COOLDOWN_FILE", tmp_path / "host_cooldowns.json")
    monkeypatch.setattr(downloader, "_HOSTS", {})


@pytest.fixture
def lbl_text() -> str:
    return CUMINDEX_LBL.read_text(encoding="latin-1")


@pytest.fixture(scope="session")
def built_db(tmp_path_factory) -> Path:
    """Parquet built from the 200-row fixture (session-scoped, offline)."""
    from lunar_toolkit.lroc.cumindex import build_parquet, parse_lbl_columns

    d = tmp_path_factory.mktemp("db")
    cols = parse_lbl_columns(CUMINDEX_LBL.read_text(encoding="latin-1"))
    build_parquet(
        [(CUMINDEX_TAB, cols)], d / "cumindex_edr.parquet", temp_dir=d / "tmp", memory_limit="512MB", wkt=True
    )
    return d


@pytest.fixture
def cumindex(built_db, tmp_path):
    """CumIndex over a private copy of the fixture database."""
    from lunar_toolkit.lroc.cumindex import CumIndex

    d = tmp_path / "db"
    shutil.copytree(built_db, d)
    return CumIndex(data_dir=d, cache_dir=tmp_path / "cache")
