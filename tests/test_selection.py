"""Illumination-diverse selection on synthetic observation frames (offline)."""

import numpy as np
import pytest

from lunar_toolkit.observation import normalize_observations
from lunar_toolkit.selection import (
    aoi_coverage,
    coverage_matrix,
    kmedoids,
    select_for_coverage,
    select_illumination_diverse,
    select_kmedoids,
    select_kmedoids_footprints,
    selection_distances,
    sort_by_sun_azimuth,
)
from lunar_toolkit.stereo import azimuth_difference


def _obs(pid, az, inc=85.0, ps=1.0, lat=(-0.1, 0.1), lon=(359.9, 0.1), emission=1.0):
    return {
        "product_id": pid,
        "instrument": "LROC NAC",
        "start_time": "2015-01-01",
        "center_lat": sum(lat) / 2,
        "center_lon": 0.0,
        "min_lat": lat[0],
        "max_lat": lat[1],
        "lon_west": lon[0],
        "lon_east": lon[1],
        "incidence": inc,
        "emission": emission,
        "sub_solar_azimuth": az,
        "pixel_scale": ps,
    }


def _df(*recs):
    return normalize_observations(list(recs))


def test_sort_by_sun_azimuth():
    df = _df(_obs("a", 300), _obs("b", 10), _obs("c", None), _obs("d", 150))
    assert list(sort_by_sun_azimuth(df).product_id) == ["b", "d", "a", "c"]
    assert list(sort_by_sun_azimuth(df, start=200).product_id) == ["a", "b", "d", "c"]


def test_bins_pick_best_resolution_and_sort():
    df = _df(
        _obs("a1", 5, ps=2.0),
        _obs("a2", 25, ps=0.5),  # bin 0: a2 finer
        _obs("b1", 100, ps=1.0),
        _obs("b2", 101, ps=1.5),  # bin 90
        _obs("c1", 200, ps=1.0),
        _obs("x", None),  # no azimuth: skipped
    )
    sel = select_illumination_diverse(df, azimuth_bin_deg=30)
    assert list(sel.product_id) == ["a2", "b1", "c1"]
    assert list(sel.azimuth_bin) == [0, 90, 180]
    assert sel.index.tolist() == [1, 2, 4]  # original labels kept
    two = select_illumination_diverse(df, azimuth_bin_deg=30, per_bin=2)
    assert set(two.product_id) == {"a1", "a2", "b1", "b2", "c1"}


def test_incidence_and_emission_filters():
    df = _df(_obs("lo", 10, inc=40), _obs("hi", 100, inc=88), _obs("off", 200, inc=88, emission=20))
    sel = select_illumination_diverse(df, incidence=(80, 95), max_emission=5)
    assert list(sel.product_id) == ["hi"]


def test_incidence_bins():
    df = _df(_obs("a", 10, inc=82), _obs("b", 12, inc=88), _obs("c", 14, inc=89))
    sel = select_illumination_diverse(df, incidence_bin_deg=5)
    assert len(sel) == 2 and set(sel.incidence_bin) == {80, 85}


def test_greedy_enforces_separation():
    az = [0, 10, 25, 31, 45, 70, 95, 180, 200, 355]
    df = _df(*[_obs(f"i{k}", a, ps=1.0 + 0.01 * k) for k, a in enumerate(az)])
    sel = select_illumination_diverse(df, azimuth_bin_deg=30, method="greedy")
    a = sel.sub_solar_azimuth.to_numpy()
    d = azimuth_difference(a[:, None], a[None, :])
    assert (d[~np.eye(len(a), dtype=bool)] >= 30).all()
    assert sel.sub_solar_azimuth.is_monotonic_increasing
    assert "i0" in set(sel.product_id)  # best score accepted first


def test_max_images_keeps_spread():
    az = [0, 5, 90, 95, 180, 270]
    df = _df(*[_obs(f"i{a}", a) for a in az])
    sel = select_illumination_diverse(df, azimuth_bin_deg=1, max_images=4)
    assert len(sel) == 4
    a = sel.sub_solar_azimuth.to_numpy()
    assert {round(x / 90) % 4 for x in a} == {0, 1, 2, 3}


def test_aoi_coverage_and_prefer():
    aoi = {"lat": (-0.05, 0.05), "lon": (359.95, 0.05)}
    df = _df(
        _obs("full_coarse", 10, ps=2.0),  # covers the AOI
        _obs("half_fine", 20, ps=0.5, lat=(-0.1, 0.0)),  # half of it
        _obs("away", 30, lat=(10, 11), lon=(10, 11)),  # no overlap
    )
    cov = aoi_coverage(df, aoi)
    assert cov.tolist() == pytest.approx([1.0, 0.5, 0.0], abs=0.02)
    by_cov = select_illumination_diverse(df, aoi=aoi, prefer="coverage")
    assert list(by_cov.product_id) == ["full_coarse"]
    by_res = select_illumination_diverse(df, aoi=aoi, prefer="resolution")
    assert list(by_res.product_id) == ["half_fine"]
    assert select_illumination_diverse(df, aoi=aoi, min_aoi_coverage=0.9).product_id.tolist() == [
        "full_coarse"
    ]
    custom = select_illumination_diverse(df, aoi=aoi, prefer=lambda d: d["sub_solar_azimuth"])
    assert list(custom.product_id) == ["half_fine"]


def test_select_for_coverage_per_cell():
    aoi = {"lat": (-0.1, 0.1), "lon": (359.9, 0.1)}
    south = dict(lat=(-0.1, 0.0))
    north = dict(lat=(0.0, 0.1))
    df = _df(
        _obs("all_0", 0),
        _obs("all_5", 5, ps=0.5),  # redundant with all_0 (< 30° apart)
        _obs("south_90", 90, **south),
        _obs("north_90", 90, **north),
        _obs("north_180", 180, **north),
        _obs("south_95", 95, **south),  # redundant with south_90
    )
    sel = select_for_coverage(df, aoi, azimuth_sep_deg=30, views_per_cell=3, grid_km=0.3)
    ids = set(sel.product_id)
    assert len(ids & {"all_0", "all_5"}) == 1
    assert {"south_90", "north_90", "north_180"} <= ids
    assert "south_95" not in ids
    assert sel.sub_solar_azimuth.is_monotonic_increasing
    vf = sel.attrs["view_fractions"]
    assert vf[2] + vf[3] == pytest.approx(1.0) and vf[3] == pytest.approx(0.5, abs=0.05)
    first = sel.sort_values("selection_order").iloc[0]
    assert first.product_id in {"all_0", "all_5"} and first.cells_gained == pytest.approx(1.0)
    capped = select_for_coverage(df, aoi, max_images=2, grid_km=0.3)
    assert len(capped) == 2


def test_cumindex_fixture(cumindex):
    obs = cumindex.to_observations(cumindex.search(lat=(-90, -85), nac_only=True, wkt=True))
    sel = select_illumination_diverse(obs, azimuth_bin_deg=30, incidence=(80, 95))
    assert 0 < len(sel) <= 12
    assert sel.sub_solar_azimuth.is_monotonic_increasing
    pt = obs.iloc[0]
    sel2 = select_for_coverage(obs, {"point": (pt.center_lat, pt.center_lon), "radius_km": 1.0})
    assert len(sel2) >= 1 and sel2.attrs["n_cells"] > 0


def test_kmedoids_three_blobs_park_jun_and_random():
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    ring = 0.3 * np.array([[np.cos(t), np.sin(t)] for t in np.arange(6) * np.pi / 3])
    pts = np.concatenate([c + np.insert(ring, 3, 0.0, axis=0) for c in centres])  # centre at index 3
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    for init in ("build", "park_jun", "random"):
        med, lab = kmedoids(d, 3, init=init, seed=4, n_init=3)
        assert sorted(med.tolist()) == [3, 10, 17]
        assert [len(set(lab[i * 7 : i * 7 + 7])) for i in range(3)] == [1, 1, 1]
    again = kmedoids(d, 3, init="random", seed=4, n_init=3)
    assert np.array_equal(again[0], kmedoids(d, 3, init="random", seed=4, n_init=3)[0])
    assert kmedoids(d, 30)[0].tolist() == list(range(21))
    with pytest.raises(ValueError):
        kmedoids(d, 2, init="nope")


def test_selection_distances_normalised():
    cov = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0]], bool)
    d = selection_distances(cov, np.array([0.0, 0.0, 0.0, 180.0]))
    assert d[0, 1] == 0 and d[0, 2] == pytest.approx(1.0) and d[0, 3] == pytest.approx(1.0)
    assert d[2, 3] == pytest.approx(np.sqrt(2.0))
    assert np.allclose(d, d.T)
    only_az = selection_distances(cov, np.array([0.0, 0.0, 0.0, 180.0]), weights=(1.0, 0.0))
    assert only_az[0, 2] == 0


def _groups(cells=100, reps=3, azimuths=(0.0, 90.0, 180.0)):
    """reps copies of each (azimuth, half) image: left half / right half of a 1-D grid."""
    left = np.arange(cells) < cells // 2
    lit, az, group = [], [], []
    for a in azimuths:
        for g, half in enumerate((left, ~left)):
            for _ in range(reps):
                lit.append(half)
                az.append(a)
                group.append((a, g))
    return np.array(lit), np.array(az), group


def test_select_kmedoids_one_medoid_per_group():
    lit, az, group = _groups()
    res = select_kmedoids(lit, az, n_clusters=6)
    assert len(res.medoids) == 6 and len({group[i] for i in res.medoids}) == 6
    assert res.stage == ["medoid"] * 6  # already full coverage with 3 directions everywhere
    assert res.uncovered == 0 and res.poorly_lit == 0 and (res.directions == 3).all()
    assert len(set(res.labels)) == 6


def test_select_kmedoids_refinement_stages():
    lit, az, group = _groups()
    res = select_kmedoids(lit, az, n_clusters=1)
    assert res.stage[0] == "medoid" and res.stage[1] == "coverage"
    assert set(res.stage[2:]) == {"illumination"} and len(res.chosen) == 6
    assert len({group[i] for i in res.chosen}) == 6
    assert res.uncovered == 0 and res.poorly_lit == 0
    assert [h["stage"] for h in res.history] == res.stage[1:]
    two = select_kmedoids(lit, az, n_clusters=1, views=2)
    assert len(two.chosen) == 4 and (two.directions == 2).all()
    capped = select_kmedoids(lit, az, n_clusters=10, max_images=3)
    assert len(capped.chosen) == 3 and capped.stage == ["medoid"] * 3
    assert select_kmedoids(lit, az, n_clusters=1).chosen == res.chosen  # deterministic


def test_select_kmedoids_thresholds_and_nan():
    cells = 1000
    lit = np.zeros((4, cells), bool)
    lit[0, :990] = True  # 99 % covered
    lit[1, 990:996] = True  # +0.6 %
    lit[2, 996:] = True  # last 0.4 %
    lit[3] = True
    az = np.array([0.0, 10.0, 20.0, np.nan])  # the full-coverage image has no azimuth
    res = select_kmedoids(lit, az, n_clusters=1, max_poorly_lit=2.0)
    assert 3 not in res.chosen and res.labels[3] == -1
    assert res.chosen == [1, 0] and res.stage == ["medoid", "coverage"]  # 1 is the central image
    assert res.uncovered == pytest.approx(0.004)
    strict = select_kmedoids(lit, az, n_clusters=1, max_uncovered=0.0, max_poorly_lit=2.0)
    assert strict.chosen == [1, 0, 2] and strict.uncovered == 0


def test_select_kmedoids_footprints():
    aoi = {"lat": (-0.1, 0.1), "lon": (359.9, 0.1)}
    south = dict(lat=(-0.1, 0.0))
    north = dict(lat=(0.0, 0.1))
    df = _df(
        _obs("all_0", 0),
        _obs("all_5", 5),
        _obs("south_90", 90, **south),
        _obs("north_90", 90, **north),
        _obs("north_180", 180, **north),
        _obs("south_200", 200, **south),
        _obs("none", None),
    )
    cover, cx, cy = coverage_matrix(df, aoi, grid_km=0.3)
    assert cover.shape == (len(df), len(cx)) and len(cx) == len(cy)
    assert cover[0].all() and cover[2].sum() == pytest.approx(cover.shape[1] / 2, abs=cover.shape[1] * 0.1)
    sel = select_kmedoids_footprints(df, aoi, n_clusters=2, grid_km=0.3)
    assert "none" not in set(sel.product_id)
    assert sel.attrs["uncovered"] == 0 and sel.attrs["poorly_lit"] == 0
    assert set(sel.selection_stage) <= {"medoid", "coverage", "illumination"}
    assert len({"all_0", "all_5"} & set(sel.product_id)) == 1
    assert {"south_90", "north_90", "north_180", "south_200"} <= set(sel.product_id)
    assert sel.sub_solar_azimuth.is_monotonic_increasing
