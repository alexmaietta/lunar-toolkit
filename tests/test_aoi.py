"""Area-of-interest input: loading, bounding regions and exact refinement."""

import argparse
import json

import pytest
from conftest import FIXTURES

from lunar_toolkit.geometry import (
    Aoi,
    aoi_mask,
    aoi_region,
    filter_aoi,
    load_aoi,
    ps_xy,
)
from lunar_toolkit.lroc.cumindex import region_sql
from lunar_toolkit.ode import OdeClient, extract_products, parse_product, to_dataframe

SQUARE = [[10.0, -5.0], [20.0, -5.0], [20.0, 5.0], [10.0, 5.0], [10.0, -5.0]]
SOUTH_CAP = "POLYGON((0 -88, 90 -88, 180 -88, 270 -88, 0 -88))"  # encloses the pole
SEAM_63 = "POLYGON((-2 60, 1.5 60, 1.5 66, -2 66, -2 60))"


def _prefilter(ci, aoi) -> set:
    sql = "SELECT PRODUCT_ID FROM cumindex WHERE " + region_sql(aoi_region(aoi))
    return set(ci.query(sql)["PRODUCT_ID"])


def _ids(df) -> set:
    return set(df["PRODUCT_ID"])


def _ode_df():
    res = json.loads((FIXTURES / "sc_cal.json").read_text())["ODEResults"]
    return res, to_dataframe([parse_product(p, "ShadowCam") for p in extract_products(res)])


# Loading


def test_load_formats_agree(tmp_path):
    geom = {"type": "Polygon", "coordinates": [SQUARE]}
    feature = {"type": "Feature", "properties": {}, "geometry": geom}
    (tmp_path / "f.geojson").write_text(json.dumps(feature))
    (tmp_path / "fc.json").write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    (tmp_path / "a.wkt").write_text("POLYGON((10 -5, 20 -5, 20 5, 10 5, 10 -5))\n")
    sources = [
        geom,
        tmp_path / "f.geojson",
        str(tmp_path / "fc.json"),
        tmp_path / "a.wkt",
        "POLYGON((10 -5, 20 -5, 20 5, 10 5, 10 -5))",
        json.dumps(geom),
    ]
    regions = [aoi_region(load_aoi(s)) for s in sources]
    for r in regions:
        assert r == {"lat": (-5.0, 5.0), "west": 10.0, "span": 10.0, "ps": {}}
    assert load_aoi(tmp_path / "a.wkt").source.endswith("a.wkt")


def test_load_passthrough_and_densify():
    aoi = load_aoi("POLYGON((10 -5, 20 -5, 20 5, 10 5, 10 -5))", densify_deg=1.0)
    assert isinstance(aoi, Aoi)
    assert load_aoi(aoi) is aoi
    assert len(aoi.polygons[0][0]) == 40  # 4 edges × 10 one-degree steps
    raw = load_aoi("POLYGON((10 -5, 20 -5, 20 5, 10 5, 10 -5))", densify_deg=None)
    assert len(raw.polygons[0][0]) == 4
    assert raw.wkt.startswith("MULTIPOLYGON(((10.000000 -5.000000")


def test_load_errors():
    with pytest.raises(ValueError, match="no Polygon"):
        load_aoi("POINT(10 10)")
    with pytest.raises(ValueError, match="GeoJSON/WKT"):
        load_aoi("not a polygon")
    with pytest.raises(ValueError, match="latitude"):
        load_aoi("POLYGON((10 80, 20 95, 30 80, 10 80))")


# Regions


def test_seam_conventions_agree():
    a = aoi_region("POLYGON((-3 -1, 3 -1, 3 1, -3 1, -3 -1))")
    b = aoi_region("POLYGON((357 -1, 3 -1, 3 1, 357 1, 357 -1))")
    for r in (a, b):
        assert r["west"] == pytest.approx(357.0)
        assert r["span"] == pytest.approx(6.0)
        assert r["lat"] == (-1.0, 1.0)


def test_antimeridian_multipolygon_arc():
    r = aoi_region(
        "MULTIPOLYGON(((170 0, 180 0, 180 10, 170 10, 170 0)),((-180 0, -170 0, -170 10, -180 10, -180 0)))"
    )
    assert r["west"] == pytest.approx(170.0)
    assert r["span"] == pytest.approx(20.0)


def test_disjoint_multipolygon_takes_shortest_cover():
    r = aoi_region("MULTIPOLYGON(((10 0, 20 0, 20 1, 10 1, 10 0)),((300 0, 310 0, 310 1, 300 1, 300 0)))")
    assert r["west"] == pytest.approx(300.0)
    assert r["span"] == pytest.approx(80.0)


def test_pole_enclosing_region():
    r = aoi_region(SOUTH_CAP)
    assert r["lat"] == (-90.0, -88.0)
    assert r["span"] == 360.0
    x0, x1, y0, y1 = r["ps"]["S"]
    edge = ps_xy(-88.0, 90.0, "S")[0]
    assert x0 < 0 < x1 and y0 < 0 < y1
    assert x1 == pytest.approx(edge, rel=0.02)
    assert "N" not in r["ps"]


def test_region_straddling_equator_falls_back_to_box():
    r = aoi_region("POLYGON((0 -70, 10 -70, 10 10, 0 10, 0 -70))")
    assert set(r["ps"]) == {"S"}
    x0, x1, y0, y1 = r["ps"]["S"]
    assert x1 > ps_xy(0.0, 10.0, "S")[0] * 0.99  # covers the equatorial edge


# CumIndex search


def test_search_south_pole_cap(cumindex):
    df = cumindex.search(aoi=SOUTH_CAP)
    assert _ids(df) == {
        "M101013765CE",
        "M101013931LE",
        "M101013931RE",
        "M101020920ME",
        "M101021090LE",
        "M101021090RE",
    }
    assert _ids(df) <= _prefilter(cumindex, SOUTH_CAP)


def test_search_polar_refinement_drops_bbox_hits(cumindex):
    # Inside M101013931RE's polar stereographic box, but off the strip itself.
    aoi = "POLYGON((30 -88.93, 31 -88.93, 31 -88.88, 30 -88.88, 30 -88.93))"
    assert "M101013931RE" in _prefilter(cumindex, aoi)
    assert "M101013931RE" not in _ids(cumindex.search(aoi=aoi))


def test_search_polar_seam_both_conventions(cumindex):
    want = {"M101014131CE", "M101021287ME"}  # WAC frames straddling 0° at −75°
    for wkt in (
        "POLYGON((-3 -76, 3 -76, 3 -74, -3 -74, -3 -76))",
        "POLYGON((357 -76, 3 -76, 3 -74, 357 -74, 357 -76))",
    ):
        assert _ids(cumindex.search(aoi=wkt)) == want


def test_search_lonlat_seam(cumindex):
    # Straddles 60°N, so the intersection runs in lon/lat across the seam
    assert _ids(cumindex.search(aoi=SEAM_63)) == {"M101016885CE"}
    miss = "POLYGON((1 60, 3 60, 3 62, 1 62, 1 60))"
    assert "M101016885CE" in _prefilter(cumindex, miss)
    assert cumindex.search(aoi=miss).empty


def test_search_geojson_file(cumindex, tmp_path):
    p = tmp_path / "aoi.geojson"
    p.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-2, 60], [1.5, 60], [1.5, 66], [-2, 66], [-2, 60]]],
                        },
                    }
                ],
            }
        )
    )
    assert _ids(cumindex.search(aoi=p)) == {"M101016885CE"}


def test_search_wkt_column_and_limit(cumindex):
    df = cumindex.search(aoi=SOUTH_CAP, columns=["EMISSION_ANGLE"], wkt=False)
    assert "FP_WKT_QUERY" not in df.columns and "FP_WKT" not in df.columns
    df = cumindex.search(aoi=SOUTH_CAP, columns=["EMISSION_ANGLE"], wkt=True)
    assert df["FP_WKT"].str.startswith("POLYGON").all()
    assert len(cumindex.search(aoi=SOUTH_CAP, limit=2)) == 2


def test_search_aoi_excludes_other_regions(cumindex):
    with pytest.raises(ValueError, match="either aoi"):
        cumindex.search(aoi=SOUTH_CAP, lat=(-90, -80))


# Fallback geometry and ODE


def test_corner_and_bbox_fallbacks(cumindex):
    df = cumindex.search(lat=(-90, -60))
    want = _ids(filter_aoi(df, SOUTH_CAP))
    assert _ids(filter_aoi(df.drop(columns=["FP_WKT"]), SOUTH_CAP)) == want  # corners
    bbox_only = df[["PRODUCT_ID", "FP_MIN_LAT", "FP_MAX_LAT", "FP_LON_WEST", "FP_LON_SPAN"]]
    assert want <= _ids(filter_aoi(bbox_only, SOUTH_CAP))  # boxes are conservative


def test_ode_footprint_refinement():
    _, df = _ode_df()
    near_270 = "POLYGON((268 -88.9, 272 -88.9, 272 -88.8, 268 -88.8, 268 -88.9))"
    assert aoi_mask(df, near_270) == [False, False, True]
    across_seam = "POLYGON((-10 -89, 10 -89, 10 -88.9, -10 -88.9, -10 -89))"
    assert aoi_mask(df, across_seam) == [True, True, False]
    no_wkt = df.drop(columns=["footprint_wkt"])  # min/max lat + lon_west/east boxes
    assert aoi_mask(no_wkt, near_270)[2]


def test_ode_build_params_and_search(monkeypatch):
    res, _ = _ode_df()
    aoi = "POLYGON((-10 -89, 10 -89, 10 -88.9, -10 -88.9, -10 -89))"
    client = OdeClient()
    params = client.build_params("shadowcam", "cal", aoi=aoi)
    assert (params["westernlon"], params["easternlon"]) == ("350", "10")
    assert (params["minlat"], params["maxlat"]) == ("-89", "-88.9")
    with pytest.raises(ValueError, match="either aoi"):
        client.build_params("shadowcam", aoi=aoi, point=(-89, 0))

    seen = []
    monkeypatch.setattr(client, "request", lambda p: seen.append(p) or res)
    df = client.search("shadowcam", "cal", aoi=aoi, max_results=10)
    assert list(df["product_id"]) == ["M013142801SC", "M013164210SC"]
    assert seen[0]["westernlon"] == "350"


# CLI


def _parser(module):
    parser = argparse.ArgumentParser()
    module.add_parser(parser.add_subparsers())
    return parser


def test_cli_search_aoi(cumindex, capsys):
    from lunar_toolkit.cli import search as cli

    args = _parser(cli).parse_args(
        ["search", "--aoi", SEAM_63, "--data-dir", str(cumindex.parquet_path.parent)]
    )
    assert cli.run(args) == 0
    out = capsys.readouterr().out
    assert "1 image(s)" in out and "M101016885CE" in out


def test_cli_aoi_is_exclusive():
    from lunar_toolkit.cli import ode_search, search

    with pytest.raises(SystemExit):
        _parser(search).parse_args(["search", "--aoi", SEAM_63, "--bbox", "0", "0", "1", "1"])
    args = _parser(ode_search).parse_args(["ode-search", "--aoi", "aoi.geojson"])
    assert args.aoi == "aoi.geojson" and args.bbox is None


def test_cli_pairs_aoi(cumindex, capsys):
    from lunar_toolkit.cli import pairs as cli

    d = str(cumindex.parquet_path.parent)
    args = _parser(cli).parse_args(["pairs", "--aoi", SOUTH_CAP, "--sfs", "--coverage", "--data-dir", d])
    assert cli.run(args) == 0
    out = capsys.readouterr().out
    assert "image(s) in region" in out and "selected" in out
    args = _parser(cli).parse_args(["pairs", "--aoi", SOUTH_CAP, "--conv", "0", "90", "--data-dir", d])
    assert cli.run(args) == 0
    assert "pair(s)" in capsys.readouterr().out


def test_region_dicts():
    box = load_aoi({"lat": (-1, 1), "lon": (359, 1)})
    r = aoi_region(box)
    assert (r["west"], r["span"], *r["lat"]) == pytest.approx((359.0, 2.0, -1.0, 1.0))
    cap = aoi_region({"point": (-89.9, 0.0), "radius_km": 10.0})
    assert cap["lat"][0] == -90.0 and cap["span"] == 360.0  # circle around the pole
    with pytest.raises(ValueError):
        load_aoi({"point": (0, 0), "radius_km": 0})
    with pytest.raises(ValueError):
        load_aoi({"radius_km": 3})


def test_long_wkt_and_odd_file_names(tmp_path):
    ring = ", ".join(f"{-0.8 + i * 0.0005:.7f} -86.0" for i in range(400))
    long_wkt = f"POLYGON(({ring}, 0.0 -85.9, -0.8 -86.0))"
    assert len(long_wkt) > 4096
    assert load_aoi(long_wkt).polygons
    wkt = "POLYGON((10 -80, 11 -80, 11 -79, 10 -79, 10 -80))"
    for name in ("aoi (1).wkt", "POLYGON_aoi.wkt"):
        path = tmp_path / name
        path.write_text(wkt)
        assert load_aoi(str(path)).wkt == load_aoi(wkt).wkt
