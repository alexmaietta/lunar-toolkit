"""Stereo pair ranking on synthetic observation frames (offline)."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES

from lunar_toolkit.constants import MOON_RADIUS_KM as R
from lunar_toolkit.geometry import aoi_geometry
from lunar_toolkit.observation import normalize_observations
from lunar_toolkit.stereo import (
    TangentPlane,
    azimuth_difference,
    convergence_angle,
    find_stereo_pairs,
    unit_vector,
)

ALT = 50.0


def _look_offset_deg(emission_deg: float, alt: float = ALT) -> float:
    """Sub-spacecraft offset (deg of arc) giving *emission_deg* at the target."""
    e = math.radians(emission_deg)
    # Triangle centre–ground–spacecraft: sin(gamma) = R sin(e) / (R + h).
    gamma = math.asin(R * math.sin(e) / (R + alt))
    return math.degrees(e - gamma)


def _obs(
    pid,
    lat,
    lon,
    look_east=0.0,
    half=0.05,
    inc=60.0,
    az=120.0,
    ps=0.5,
    t="2012-01-01",
    instrument="LROC NAC",
    wkt=False,
    geometry=True,
    **extra,
):
    """A square image of side 2*half deg centred on (lat, lon) viewed with
    emission *look_east* (deg, signed) from a spacecraft east/west of it."""
    off = _look_offset_deg(abs(look_east)) * (1 if look_east >= 0 else -1)
    rec = {
        "product_id": pid,
        "instrument": instrument,
        "start_time": t,
        "center_lat": lat,
        "center_lon": lon % 360,
        "min_lat": lat - half,
        "max_lat": lat + half,
        "lon_west": (lon - half) % 360,
        "lon_east": (lon + half) % 360,
        "incidence": inc,
        "emission": abs(look_east),
        "phase": None,
        "sub_solar_azimuth": az,
        "pixel_scale": ps,
    }
    if geometry:
        rec.update(
            sub_spacecraft_lat=lat,
            sub_spacecraft_lon=(lon + off / math.cos(math.radians(lat))) % 360,
            spacecraft_altitude=ALT,
        )
    if wkt:
        w, e, s, n = lon - half, lon + half, lat - half, lat + half
        rec["footprint_wkt"] = f"POLYGON (({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"
    rec.update(extra)
    return rec


def _df(*recs):
    return normalize_observations(list(recs))


def test_convergence_angle_symmetric():
    off = _look_offset_deg(15.0)
    conv, bh = convergence_angle((0.0, 0.0), (0.0, -off, ALT), (0.0, off, ALT))
    assert conv == pytest.approx(30.0, abs=1e-6)
    # baseline / slant range = 2 sin(conv / 2) for a symmetric pair
    assert bh == pytest.approx(2 * math.sin(math.radians(15.0)), rel=1e-6)


def test_convergence_angle_pushbroom_shift():
    """A nadir image's spacecraft follows the strip: a far ground point is
    still seen at ~nadir from its own image, so convergence stays ~0."""
    sc = (0.0, 0.0, ALT)
    conv, _ = convergence_angle((1.0, 0.0), sc, sc, centre1=(0.0, 0.0), centre2=(0.0, 0.0))
    assert conv == pytest.approx(0.0, abs=1e-6)
    conv_static, _ = convergence_angle((1.0, 0.0), sc, (2.0, 0.0, ALT))
    assert conv_static > 40.0


def test_ranking_and_filters():
    df = _df(
        _obs("M1000000001LE", 10, 20, look_east=0.0, t="2012-01-01"),
        _obs("M1000000002LE", 10, 20, look_east=25.0, t="2012-02-01"),  # 25° from nadir
        _obs("M1000000003LE", 10, 20, look_east=3.0, t="2012-03-01"),  # too small vs #1
        _obs("M1000000004LE", 10, 20, look_east=25.0, inc=85.0, t="2012-04-01"),  # incidence too far
        _obs("M1000000005LE", 10, 20, look_east=-20.0, az=300.0, t="2012-05-01"),  # azimuth too far
        _obs("M1000000006SC", 10, 20, look_east=25.0, t="2012-06-01", instrument="ShadowCam"),
    )
    pairs = find_stereo_pairs(df)
    ids = {frozenset(p) for p in zip(pairs.product_id_1, pairs.product_id_2)}
    assert frozenset({"M1000000001LE", "M1000000002LE"}) in ids
    assert frozenset({"M1000000001LE", "M1000000003LE"}) not in ids
    assert all("M1000000004LE" not in p for p in ids)
    assert all("M1000000005LE" not in p for p in ids)
    assert all("M1000000006SC" not in p for p in ids)
    best = pairs.iloc[0]
    assert {best.product_id_1, best.product_id_2} == {"M1000000001LE", "M1000000002LE"}
    assert best.convergence == pytest.approx(25.0, abs=0.2)
    assert best.overlap_frac == pytest.approx(1.0, abs=0.01)
    assert best.dt_days == pytest.approx(31.0)
    assert pairs["score"].is_monotonic_decreasing


def test_same_observation_halves_never_paired():
    df = _df(
        _obs("M1000000001LE", 0, 0, look_east=0.0, t="2012-01-01"),
        _obs("M1000000001RE", 0, 0, look_east=25.0, t="2012-02-01"),
    )
    assert find_stereo_pairs(df).empty


def test_overlap_fraction_wkt_and_bbox():
    a = _obs("A", 0, 0.0, look_east=0.0, t="2012-01-01", wkt=True)
    b = _obs("B", 0, 0.05, look_east=25.0, t="2012-02-01", wkt=True)  # shifted by half a width
    for recs in ([a, b], [{**a, "footprint_wkt": None}, {**b, "footprint_wkt": None}]):
        p = find_stereo_pairs(_df(*recs))
        assert len(p) == 1
        assert p.overlap_frac[0] == pytest.approx(0.5, abs=0.02)
        side = 0.1 * math.pi / 180 * R
        assert p.overlap_km2[0] == pytest.approx(0.5 * side * side, rel=0.03)
        assert p.overlap_lon[0] == pytest.approx(0.025, abs=0.002)


def test_seam_crossing_overlap():
    df = _df(
        _obs("A", 5, 359.98, look_east=0.0, t="2012-01-01"),
        _obs("B", 5, 0.02, look_east=30.0, t="2012-02-01"),
    )
    assert df.lon_west[0] > df.lon_east[0]
    p = find_stereo_pairs(df)
    assert len(p) == 1 and p.overlap_frac[0] > 0.5
    assert p.convergence[0] == pytest.approx(30.0, abs=1.0)


def test_polar_footprints():
    def polar(pid, look_north, t, **fp):
        off = _look_offset_deg(look_north)
        rec = _obs(
            pid,
            -89.95,
            0,
            t=t,
            geometry=False,
            emission=look_north,
            min_lat=-89.99,
            max_lat=-89.9,
            lon_west=340.0,
            lon_east=20.0,
        )  # wedge across 0/360
        rec.update(sub_spacecraft_lat=-89.95 + off, sub_spacecraft_lon=0.0, spacecraft_altitude=ALT, **fp)
        return rec

    df = _df(
        polar("A", 0.0, "2012-01-01"),
        polar("B", 25.0, "2012-02-01"),
        # pole-enclosing bbox: all longitudes south of -89.9
        polar("C", 20.0, "2012-03-01", min_lat=-90.0, max_lat=-89.9, lon_west=0.0, lon_east=360.0),
    )
    p = find_stereo_pairs(df)
    ids = {frozenset(x) for x in zip(p.product_id_1, p.product_id_2)}
    assert frozenset({"A", "B"}) in ids and frozenset({"A", "C"}) in ids
    ab = p[[{a, b} == {"A", "B"} for a, b in zip(p.product_id_1, p.product_id_2)]].iloc[0]
    assert ab.overlap_frac == pytest.approx(1.0, abs=0.01)
    assert ab.convergence == pytest.approx(25.0, abs=0.5)


def test_missing_geometry_bounds():
    df = _df(
        _obs("A", 0, 0, look_east=2.0, t="2012-01-01", geometry=False),
        _obs("B", 0, 0, look_east=20.0, t="2012-02-01", geometry=False),
    )
    p = find_stereo_pairs(df)
    assert len(p) == 1 and math.isnan(p.convergence[0])
    assert (p.convergence_min[0], p.convergence_max[0]) == pytest.approx((18.0, 22.0))
    assert find_stereo_pairs(df, require_geometry=True).empty


def test_aoi_clips_overlap():
    df = _df(
        _obs("A", 0, 0.0, look_east=0.0, t="2012-01-01", half=0.2),
        _obs("B", 0, 0.0, look_east=25.0, t="2012-02-01", half=0.2),
    )
    p = find_stereo_pairs(df, aoi={"point": (0.05, 0.05), "radius_km": 2.0})
    assert len(p) == 1 and p.aoi_coverage[0] == pytest.approx(1.0, abs=0.01)
    assert p.overlap_lat[0] == pytest.approx(0.05, abs=0.005)
    assert find_stereo_pairs(df, aoi={"point": (5.0, 5.0), "radius_km": 2.0}).empty


def test_geometry_helpers():
    assert azimuth_difference(350, 10) == pytest.approx(20)
    plane = TangentPlane(0.0, 0.0)
    sq = aoi_geometry({"lat": (-1, 1), "lon": (359, 1)}, plane.ring)
    assert sq.area == pytest.approx((2 * math.pi / 180 * R) ** 2, rel=0.01)
    half = aoi_geometry({"lat": (-1, 1), "lon": (0, 2)}, plane.ring)
    assert sq.intersection(half).area == pytest.approx(sq.area / 2, rel=0.02)
    assert np.allclose(np.linalg.norm(unit_vector([10, 20], [30, 40]), axis=1), 1.0)


def test_ode_lroc_index_record_geometry():
    from lunar_toolkit.ode import extract_products, parse_product

    res = json.loads((FIXTURES / "lroc_edr.json").read_text())["ODEResults"]
    r = parse_product(extract_products(res)[0], "LROC NAC")
    assert r["sub_solar_azimuth"] == pytest.approx(179.17)
    assert r["north_azimuth"] == pytest.approx(84.13)
    assert (r["sub_spacecraft_lat"], r["sub_spacecraft_lon"]) == pytest.approx((-0.71, 0.49))
    assert r["spacecraft_altitude"] == pytest.approx(79.72)


def test_cumindex_observations_geometry(cumindex):
    obs = cumindex.to_observations(cumindex.search(lat=(-90, -85), nac_only=True, wkt=True))
    assert obs["spacecraft_altitude"].notna().all()
    # Emission recomputed from the viewing geometry matches the index.
    from lunar_toolkit.stereo import angle_between

    g = R * unit_vector(obs.center_lat, obs.center_lon)
    s = (R + obs.spacecraft_altitude.to_numpy()[:, None]) * unit_vector(
        obs.sub_spacecraft_lat, obs.sub_spacecraft_lon
    )
    assert np.allclose(angle_between(g / R, s - g), obs.emission, atol=0.3)
    # Consecutive nadir orbits over the pole: overlap but tiny convergence.
    p = find_stereo_pairs(obs, convergence=(0, 90), min_overlap=0.0)
    assert len(p) > 0 and (p.convergence < 5).all()
    assert find_stereo_pairs(obs).empty
    assert isinstance(p.start_time_1[0], pd.Timestamp)


def test_aoi_forms_agree():
    df = _df(
        _obs("A", 0, 0.0, look_east=0.0, t="2012-01-01", half=0.2),
        _obs("B", 0, 0.15, look_east=25.0, t="2012-02-01", half=0.2),
    )
    box = find_stereo_pairs(df, aoi={"lat": (-0.1, 0.1), "lon": (359.9, 0.1)})
    wkt = find_stereo_pairs(df, aoi="POLYGON((-0.1 -0.1, 0.1 -0.1, 0.1 0.1, -0.1 0.1, -0.1 -0.1))")
    assert len(box) == len(wkt) == 1
    assert box.aoi_coverage[0] == pytest.approx(wkt.aoi_coverage[0]) == pytest.approx(0.75, abs=0.01)
