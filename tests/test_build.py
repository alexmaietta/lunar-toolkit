"""DuckDB-native Parquet build on the 200-row CUMINDEX fixture."""

import duckdb
import pytest

from lunar_toolkit.constants import is_nac_id, nac_id, parse_nac_id, pds_url_from_file_spec
from lunar_toolkit.lroc.cumindex import clean_stale_build_files


def _sql(built_db, q):
    return duckdb.sql(q.replace("FROM T", f"FROM read_parquet('{built_db / 'cumindex_edr.parquet'}')"))


def test_row_count_and_types(built_db):
    rel = _sql(built_db, "SELECT * FROM T")
    assert len(rel) == 200
    types = dict(zip(rel.columns, map(str, rel.types)))
    assert types["CENTER_LATITUDE"] == "DOUBLE"
    assert types["ORBIT_NUMBER"] == "BIGINT"
    assert types["START_TIME"].startswith("TIMESTAMP")
    assert types["PRODUCT_ID"] == "VARCHAR"
    for c in ("FP_MIN_LAT", "FP_MAX_LAT", "FP_LON_WEST", "FP_LON_EAST", "FP_PS_XMIN", "FP_WKT"):
        assert c in types


def test_values_trimmed_and_typed(built_db):
    row = _sql(
        built_db,
        "SELECT VOLUME_ID, PRODUCT_ID, ORBIT_NUMBER, CENTER_LATITUDE, "
        "SPACECRAFT_CLOCK_START_COUNT, NAC_PREROLL_START_TIME "
        "FROM T WHERE PRODUCT_ID = 'M101013931LE'",
    ).fetchone()
    assert row[0] == "LROLRC_0001"
    assert row[1] == "M101013931LE"
    assert row[2] == 72
    assert row[3] == pytest.approx(-89.34)
    assert row[4] == "268067463:47077"
    assert row[5] is not None


def test_not_applicable_becomes_null(built_db):
    # WAC rows carry NAC_PREROLL_START_TIME = "N/A" and NAC temperatures 9999.999
    row = _sql(
        built_db,
        "SELECT NAC_PREROLL_START_TIME, NAC_TEMPERATURE_SCS FROM T WHERE PRODUCT_ID = 'M101013765CE'",
    ).fetchone()
    assert row == (None, None)


def test_sorted_by_center_latitude(built_db):
    lats = [r[0] for r in _sql(built_db, "SELECT CENTER_LATITUDE FROM T").fetchall()]
    vals = [x for x in lats if x is not None]
    assert vals == sorted(vals)


def test_pds_url_mapping():
    url = pds_url_from_file_spec("LRO-L-LROC-2-EDR-V1.0/LROLRC_0001/DATA/COM/2009181/NAC/M101013931LE.IMG")
    assert url == (
        "https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/"
        "lro-l-lroc-2-edr/LROLRC_0001/DATA/COM/2009181/NAC/M101013931LE.IMG"
    )


def test_lookup_and_enrich(cumindex):
    df = cumindex.lookup(["m101013931le", "M101013931RE", "MISSING"])
    assert sorted(df["PRODUCT_ID"]) == ["M101013931LE", "M101013931RE"]
    assert df["PDS_URL"].str.startswith("https://pds.mcp.nasa.gov/").all()
    assert df["LABEL_URL"].str.endswith(".xml").all()
    assert df["IS_NAC"].all()


def test_get_images(cumindex):
    col = cumindex.get_images(["M101013931LE"])
    assert len(col) == 1
    img = col[0]
    assert img.center_latitude == pytest.approx(-89.34)
    assert img.lines == 52224  # IMAGE_LINES
    assert img.pixel_scale == pytest.approx(0.991)  # RESOLUTION
    assert img.orbit_number == 72 and isinstance(img.orbit_number, int)
    assert img.mode_compression is True and img.temperature_scs is not None
    assert (img.side, img.product_type, img.pair_id, img.cdr_id) == (
        "L",
        "edr",
        "M101013931RE",
        "M101013931LC",
    )


def test_nac_ids():
    assert parse_nac_id(" m1096757719rc") == ("M1096757719", "R", "cdr")
    assert parse_nac_id("M1096757719L") == ("M1096757719", "L", "edr")
    assert parse_nac_id("M101016885CE") is None and parse_nac_id("M013178466SC") is None
    assert nac_id("M1096757719L") == "M1096757719LE"
    assert nac_id("M1096757719LE", side="R", product_type="cdr") == "M1096757719RC"
    assert is_nac_id("M1096757719LC") and not is_nac_id("M1096757719L")
    from lunar_toolkit.lroc.nac import NacImage

    assert NacImage("m1096757719r").product_id == "M1096757719RE"
    assert NacImage("M101016885CE").side is None


def test_nac_processor_flat_and_resumable(tmp_path):
    from lunar_toolkit.lroc.nac import NacProcessor
    from lunar_toolkit.tools import dry_run

    img = tmp_path / "M101013931LE.IMG"
    with dry_run() as cmds:
        tif = NacProcessor(tmp_path, resolution=2).process(img)
    assert tif == tmp_path / "M101013931LE.tif"
    gdal = next(c for c in cmds if c[0].endswith("gdal_translate"))
    assert "COMPRESS=DEFLATE" in gdal and str(tmp_path / "M101013931LE.map.cub") in map(str, gdal)
    tif.write_bytes(b"")
    with dry_run() as cmds:
        NacProcessor(tmp_path).process(img)
    assert cmds == []
    with dry_run() as cmds:
        NacProcessor(tmp_path, force=True).process(img)
    assert cmds


def test_to_observations(cumindex):
    obs = cumindex.to_observations(cumindex.lookup(["M101013931LE"]))
    from lunar_toolkit.observation import OBSERVATION_COLUMNS

    assert list(obs.columns) == OBSERVATION_COLUMNS
    assert obs.loc[0, "instrument"] == "LROC NAC"
    assert obs.loc[0, "url"].endswith("M101013931LE.IMG")


def test_load_preview_from_cached_tab(tmp_path, lbl_text):
    """load(nrows) parses a cached TAB head without any Parquet."""
    from conftest import CUMINDEX_TAB

    from lunar_toolkit.lroc.cumindex import CumIndex

    ci = CumIndex(data_dir=tmp_path / "none", cache_dir=tmp_path / "cache", release="LROLRC_0066C")
    lbl = ci._cache.cache_path(ci.lbl_url)
    lbl.parent.mkdir(parents=True)
    lbl.write_text(lbl_text, encoding="latin-1")
    tab = ci._cache.cache_path(ci.tab_url)
    tab.write_bytes(CUMINDEX_TAB.read_bytes())
    df = ci.load(nrows=5)
    assert len(df) == 5
    assert df.loc[0, "PRODUCT_ID"] == "M101013765CE"


def test_clean_stale_build_files(tmp_path):
    for name in ("cumindex_edr.build.duckdb", "cumindex_edr.build.duckdb.wal", "keep.parquet"):
        (tmp_path / name).write_bytes(b"x")
    removed = clean_stale_build_files(tmp_path)
    assert len(removed) == 2
    assert (tmp_path / "keep.parquet").exists()


def test_build_db_volume_only_offline(tmp_path, lbl_text, monkeypatch):
    """build_db end-to-end with the network replaced by local files."""
    from conftest import CUMINDEX_TAB

    from lunar_toolkit.lroc.cumindex import CumIndex

    ci = CumIndex(data_dir=tmp_path / "db", cache_dir=tmp_path / "cache")
    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "cumindex_edr.build.duckdb").write_bytes(b"stale")
    lbl_200 = lbl_text.replace("FILE_RECORDS            = 4142412", "FILE_RECORDS            = 200")
    lbl_200 = lbl_200.replace('MD5_CHECKSUM            = "a45d60e14a7fac114820f51bbe89b9ed"', "")

    def fake_cached(url, force=False, expected_md5=None):
        p = tmp_path / "cache" / url.rsplit("/", 1)[-1]
        p.parent.mkdir(exist_ok=True)
        if url.endswith(".LBL"):
            p.write_text(lbl_200, encoding="latin-1")
        else:
            p.write_bytes(CUMINDEX_TAB.read_bytes())
        return p

    monkeypatch.setattr(ci._cache, "ensure_cached", fake_cached)
    out = ci.build_db(release="LROLRC_0066C", volume_only=True, memory_limit="1GB")
    assert out.exists()
    assert not (tmp_path / "db" / "cumindex_edr.build.duckdb").exists()
    info = ci.build_info()
    assert info["release"] == "LROLRC_0066C"
    assert info["rows"] == 200
    assert info["source"] == "volume_index"
    assert "LROLRC_0066C" in info["volumes"]
