"""Footprint derivation and search across the 0/360 seam and near the poles."""

import math

import duckdb
import pytest

from lunar_toolkit.geometry import ps_xy, query_region
from lunar_toolkit.lroc.cumindex import CumIndex, _register_macros, footprint_sql

# (id, centre lat, centre lon, [(lat, lon) for UL, UR, LR, LL])
SYNTHETIC = [
    ("SEAM", 0.0, 0.0, [(1, 359.5), (1, 0.5), (-1, 0.5), (-1, 359.5)]),
    ("FARSIDE", 0.0, 180.0, [(1, 179.5), (1, 180.5), (-1, 180.5), (-1, 179.5)]),
    ("SPOLE", -89.9, 0.0, [(-89.5, 0), (-89.5, 90), (-89.5, 180), (-89.5, 270)]),
    ("SSTRIP", -88.5, 50.0, [(-88.0, 48), (-88.0, 52), (-89.0, 52), (-89.0, 48)]),
    ("NSTRIP", 85.0, 300.0, [(86.0, 299), (86.0, 301), (84.0, 301), (84.0, 299)]),
    ("SEAM_POLAR", -80.0, 0.0, [(-79, 358), (-79, 2), (-81, 2), (-81, 358)]),
]


@pytest.fixture(scope="module")
def synth_index(tmp_path_factory) -> CumIndex:
    d = tmp_path_factory.mktemp("synth")
    con = duckdb.connect()
    _register_macros(con)
    corner_cols = []
    for c in ("UPPER_LEFT", "UPPER_RIGHT", "LOWER_RIGHT", "LOWER_LEFT"):
        corner_cols += [f"{c}_LATITUDE DOUBLE", f"{c}_LONGITUDE DOUBLE"]
    con.execute(
        "CREATE TABLE t (PRODUCT_ID VARCHAR, CENTER_LATITUDE DOUBLE, CENTER_LONGITUDE DOUBLE, "
        + ", ".join(corner_cols)
        + ")"
    )
    for pid, clat, clon, corners in SYNTHETIC:
        vals = [pid, clat, clon] + [v for ll in corners for v in ll]
        con.execute(f"INSERT INTO t VALUES ({', '.join('?' * len(vals))})", vals)
    fp = footprint_sql(
        [c.split()[0] for c in corner_cols] + ["CENTER_LATITUDE", "CENTER_LONGITUDE"], wkt=True
    )
    out = d / "cumindex_edr.parquet"
    con.execute(f"COPY (SELECT *, {fp} FROM t) TO '{out}' (FORMAT PARQUET)")
    return CumIndex(data_dir=d, cache_dir=d / "cache")


def _ids(df):
    return set(df["PRODUCT_ID"])


def _fp(ci, pid):
    return ci.query(f"SELECT * FROM cumindex WHERE PRODUCT_ID = '{pid}'").iloc[0]


def test_seam_lon_range(synth_index):
    r = _fp(synth_index, "SEAM")
    assert r.FP_LON_WEST == pytest.approx(359.5)
    assert r.FP_LON_EAST == pytest.approx(0.5)
    assert r.FP_LON_SPAN == pytest.approx(1.0)
    assert not r.FP_CONTAINS_POLE
    assert r.FP_WKT.startswith("POLYGON((")


def test_pole_enclosing(synth_index):
    r = _fp(synth_index, "SPOLE")
    assert r.FP_CONTAINS_POLE
    assert r.FP_MIN_LAT == -90.0
    assert r.FP_LON_SPAN == 360.0
    assert r.FP_PS_HEMI == "S"
    assert r.FP_PS_XMIN < 0 < r.FP_PS_XMAX and r.FP_PS_YMIN < 0 < r.FP_PS_YMAX


def test_strip_near_pole_not_enclosing(synth_index):
    r = _fp(synth_index, "SSTRIP")
    assert not r.FP_CONTAINS_POLE
    assert r.FP_MIN_LAT == pytest.approx(-89.0)
    assert r.FP_LON_SPAN == pytest.approx(4.0)
    assert _fp(synth_index, "NSTRIP").FP_PS_HEMI == "N"
    assert _fp(synth_index, "SEAM").FP_PS_HEMI is None


@pytest.mark.parametrize("lon", [(355, 5), (-5, 5)])
def test_search_across_seam(synth_index, lon):
    ids = _ids(synth_index.search(lat=(-2, 2), lon=lon))
    assert ids == {"SEAM"}


def test_search_seam_polar_strip(synth_index):
    assert "SEAM_POLAR" in _ids(synth_index.search(lat=(-82, -78), lon=(359, 1)))
    assert "SEAM_POLAR" not in _ids(synth_index.search(lat=(-82, -78), lon=(10, 20)))


def test_search_farside(synth_index):
    assert _ids(synth_index.search(lat=(-2, 2), lon=(170, 190))) == {"FARSIDE"}


def test_search_south_cap(synth_index):
    ids = _ids(synth_index.search(lat=(-90, -89.8)))
    assert ids == {"SPOLE"}
    ids = _ids(synth_index.search(lat=(-90, -85)))
    assert ids == {"SPOLE", "SSTRIP"}


def test_point_at_pole(synth_index):
    assert _ids(synth_index.search(point=(-90, 0), radius_km=1)) == {"SPOLE"}


def test_point_radius_near_pole(synth_index):
    assert "SSTRIP" in _ids(synth_index.search(point=(-88.5, 50), radius_km=1))
    # Same latitude, opposite side of the pole → excluded (lon arc / PS box)
    assert "SSTRIP" not in _ids(synth_index.search(point=(-88.5, 230), radius_km=1))


def test_point_radius_north(synth_index):
    assert _ids(synth_index.search(point=(85, 300), radius_km=5)) == {"NSTRIP"}


def test_search_lon_range_at_seam(synth_index):
    assert "SEAM" in _ids(synth_index.search(lat=(-2, 2), lon=(355, 360)))


def test_query_region_cap():
    r = query_region(point=(-89.7, 129.8), radius_km=20)
    assert r["span"] == 360.0
    assert r["lat"][0] == -90.0
    assert "S" in r["ps"]


def test_query_region_lon_arc():
    r = query_region(point=(0, 0), radius_km=30.3)
    dlon = math.degrees(30.3 / 1737.4)
    assert r["west"] == pytest.approx(360 - dlon)
    assert r["span"] == pytest.approx(2 * dlon)


def test_ps_xy_pole_origin():
    x, y = ps_xy(-90, 123, "S")
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    x, _ = ps_xy(-89, 90, "S")
    assert x == pytest.approx(2 * 1737400 * math.tan(math.radians(0.5)), rel=1e-9)


def test_fixture_polar_and_seam(cumindex):
    """Real fixture rows: polar search and a seam search both return hits."""
    polar = cumindex.search(lat=(-90, -85))
    assert len(polar) >= 5
    assert (polar["FP_MIN_LAT"] <= -85).all()
    seam = cumindex.search(lat=(-90, 90), lon=(359, 1))
    assert len(seam) >= 1
