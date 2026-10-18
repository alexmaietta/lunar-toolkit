"""Volume discovery and incremental update (offline)."""

from pathlib import Path

import platformdirs
import pytest
from conftest import CUMINDEX_TAB, FIXTURES

from lunar_toolkit.constants import find_parquet, get_data_dir
from lunar_toolkit.lroc.cumindex import (
    _volume_key,
    build_parquet,
    parse_lbl_columns,
    parse_s3_listing,
)


def test_parse_s3_listing():
    prefixes, token = parse_s3_listing((FIXTURES / "s3_listing_edr.xml").read_text())
    assert token is None
    vols = [p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes]
    assert "LROLRC_0001" in vols and "LROLRC_0066C" in vols


def test_volume_ordering():
    vols = ["LROLRC_0067A", "LROLRC_0066C", "LROLRC_0010", "LROLRC_0066A", "LROLRC_0068A"]
    assert sorted(vols, key=_volume_key) == [
        "LROLRC_0010",
        "LROLRC_0066A",
        "LROLRC_0066C",
        "LROLRC_0067A",
        "LROLRC_0068A",
    ]


def _rewrite_rows(data: bytes, n: int, old: bytes, new: bytes) -> bytes:
    rec = 903
    rows = [data[i * rec : (i + 1) * rec] for i in range(n)]
    return b"".join(r.replace(old, new) for r in rows)


def test_update_db_appends_and_dedups(tmp_path, lbl_text, monkeypatch):
    from lunar_toolkit.lroc.cumindex import CumIndex

    cols = parse_lbl_columns(lbl_text)
    db = tmp_path / "db"
    ci = CumIndex(data_dir=db, cache_dir=tmp_path / "cache")
    build_parquet([(CUMINDEX_TAB, cols)], ci.parquet_path, temp_dir=tmp_path / "tmp")
    ci._write_sidecar({"release": "LROLRC_0001", "rows": 200, "volumes": {"LROLRC_0001": {}}})

    data = CUMINDEX_TAB.read_bytes()
    # New volume: 10 re-released products + 1 brand-new product
    new = _rewrite_rows(data, 10, b"LROLRC_0001", b"LROLRC_0002")
    extra = _rewrite_rows(data, 1, b"LROLRC_0001", b"LROLRC_0002").replace(b"M101013765CE", b"M999999999CE")
    tab = tmp_path / "INDEX.TAB"
    tab.write_bytes(new + extra)
    lbl = tmp_path / "INDEX.LBL"
    lbl.write_text(lbl_text, encoding="latin-1")

    monkeypatch.setattr(ci, "discover_volumes", lambda: ["LROLRC_0001", "LROLRC_0002"])
    monkeypatch.setattr(
        ci._cache,
        "ensure_cached",
        lambda url, force=False, expected_md5=None: lbl if url.endswith(".LBL") else tab,
    )

    added = ci.update_db(memory_limit="1GB")
    assert added == ["LROLRC_0002"]
    n, n2 = ci.query("SELECT count(*), count(*) FILTER (VOLUME_ID = 'LROLRC_0002') FROM cumindex").iloc[0]
    assert n == 201 and n2 == 11
    assert "FP_MIN_LAT" in ci.query("SELECT * FROM cumindex LIMIT 1").columns
    info = ci.build_info()
    assert info["release"] == "LROLRC_0002"
    assert set(info["volumes"]) == {"LROLRC_0001", "LROLRC_0002"}
    # second run: nothing new
    assert ci.update_db() == []


def test_data_dir_resolution(tmp_path, monkeypatch):
    import lunar_toolkit.constants as c

    monkeypatch.delenv(c.DATA_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    user_dir = Path(platformdirs.user_data_dir(c.APP_NAME, appauthor=False))
    assert get_data_dir() == user_dir
    assert find_parquet("x.parquet") == user_dir / "x.parquet"
    assert find_parquet("x.parquet", tmp_path / "d") == tmp_path / "d" / "x.parquet"
    monkeypatch.setenv(c.DATA_DIR_ENV_VAR, str(tmp_path / "env"))
    assert get_data_dir() == tmp_path / "env"


@pytest.mark.network
def test_live_discover_volumes():
    from lunar_toolkit.lroc.cumindex import CumIndex

    vols = CumIndex("edr").discover_volumes()
    assert vols[0] == "LROLRC_0001"
    assert _volume_key(vols[-1]) >= _volume_key("LROLRC_0066C")
