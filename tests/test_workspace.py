"""DuckDB workspace: images, SQL stereo pairs vs find_stereo_pairs, BA tables, rejection views, CLI (offline)."""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_stereo import _df, _look_offset_deg, _obs

from lunar_toolkit import asp, cli, sfs_report, workspace
from lunar_toolkit.constants import MOON_RADIUS_KM as R
from lunar_toolkit.geometry import footprint_geometry, footprint_rings, ps_projection, ps_xy
from lunar_toolkit.observation import normalize_observations
from lunar_toolkit.sfs import SfsConfig, SfsImage, SfsPipeline, flag_images
from lunar_toolkit.stereo import PAIR_COLUMNS, TangentPlane, find_stereo_pairs
from lunar_toolkit.tools import dry_run
from lunar_toolkit.workspace import Workspace, WorkspaceError, find_stereo_pairs_duckdb

# LUNAR_REQUIRE_SPATIAL=1 (CI) turns the skip into failures
needs_spatial = pytest.mark.skipif(
    not os.environ.get("LUNAR_REQUIRE_SPATIAL") and not workspace.spatial_available(),
    reason="DuckDB spatial extension not available (INSTALL spatial needs network once)",
)


def _polar_frames(n, seed=0, radius_km=15.0, hemi="S", alt=50.0):
    """NAC-like 5 x 25 km strips near a pole, viewed up to 30 deg off-nadir."""
    rng = np.random.default_rng(seed)
    sgn = -1 if hemi == "S" else 1
    recs = []
    for k in range(n):
        r = radius_km * math.sqrt(rng.random())
        lat, lon = sgn * (90.0 - math.degrees(r / R)), rng.uniform(0, 360)
        pl = TangentPlane(lat, lon)
        rot = math.radians(rng.normal(0, 3))
        corners = [
            pl.unproject(dx * math.cos(rot) - dy * math.sin(rot), dx * math.sin(rot) + dy * math.cos(rot))
            for dx, dy in ((-2.5, 12.5), (2.5, 12.5), (2.5, -12.5), (-2.5, -12.5))
        ]
        e = rng.uniform(-30, 30)
        off = math.radians(_look_offset_deg(abs(e), alt)) * R * (1 if e >= 0 else -1)
        sclat, sclon = pl.unproject(off * math.cos(rot), off * math.sin(rot))
        ring = [(lo, la) for la, lo in corners]
        recs.append(
            {
                "product_id": f"M{1000000000 + k}{'LR'[k % 2]}E",
                "instrument": "LROC NAC",
                "start_time": pd.Timestamp("2010-01-01") + pd.Timedelta(seconds=int(rng.integers(0, 3e8))),
                "center_lat": lat,
                "center_lon": lon,
                "footprint_wkt": "POLYGON ((" + ", ".join(f"{a} {b}" for a, b in ring + ring[:1]) + "))",
                "incidence": rng.uniform(84, 90),
                "emission": abs(e),
                "sub_solar_azimuth": rng.uniform(0, 360),
                "sub_spacecraft_lat": sclat,
                "sub_spacecraft_lon": sclon,
                "spacecraft_altitude": alt,
                "pixel_scale": rng.uniform(0.5, 2.0),
            }
        )
    return normalize_observations(recs)


def _assert_same_pairs(py, sql, ordered=True):
    assert list(sql.columns) == PAIR_COLUMNS
    key = ["product_id_1", "product_id_2"]
    assert set(map(tuple, py[key].to_numpy())) == set(map(tuple, sql[key].to_numpy()))
    if ordered:
        assert list(py.index_1) == list(sql.index_1) and list(py.index_2) == list(sql.index_2)
    m = py.merge(sql, on=key, suffixes=("_py", "_sql"))
    assert len(m) == len(py)
    tol = {
        "convergence": 1e-6,
        "b_h": 1e-8,
        "overlap_frac": 2e-3,
        "aoi_coverage": 2e-3,
        "overlap_lat": 1e-3,
        "score": 1e-3,
        "dt_days": 1e-9,
        "incidence_diff": 1e-9,
        "azimuth_diff": 1e-9,
        "convergence_min": 1e-9,
        "convergence_max": 1e-9,
        "resolution_ratio": 1e-9,
    }
    for c, t in tol.items():
        a, b = m[f"{c}_py"].to_numpy(float), m[f"{c}_sql"].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), c
        assert np.allclose(a, b, atol=t, rtol=0, equal_nan=True), c
    a, b = m["overlap_km2_py"].to_numpy(float), m["overlap_km2_sql"].to_numpy(float)
    assert np.allclose(a, b, rtol=2e-3)
    assert (m["index_1_py"] == m["index_1_sql"]).all() and (m["index_2_py"] == m["index_2_sql"]).all()


@pytest.fixture
def ws(tmp_path):
    with Workspace(tmp_path / "w", memory_limit="256MB", threads=2) as w:
        yield w


STEREO_FRAMES = {
    "ranking": (
        [
            _obs("M1000000001LE", 10, 20, look_east=0.0, t="2012-01-01"),
            _obs("M1000000002LE", 10, 20, look_east=25.0, t="2012-02-01"),
            _obs("M1000000003LE", 10, 20, look_east=3.0, t="2012-03-01"),
            _obs("M1000000004LE", 10, 20, look_east=25.0, inc=85.0, t="2012-04-01"),
            _obs("M1000000005LE", 10, 20, look_east=-20.0, az=300.0, t="2012-05-01"),
            _obs("M1000000006SC", 10, 20, look_east=25.0, t="2012-06-01", instrument="ShadowCam"),
            _obs("M1000000002RE", 10, 20.05, look_east=0.0, t="2012-02-01"),
        ],
        {},
    ),
    "wkt": (
        [
            _obs("A", 0, 0.0, look_east=0.0, t="2012-01-01", wkt=True),
            _obs("B", 0, 0.05, look_east=25.0, t="2012-02-01", wkt=True),
        ],
        {},
    ),
    "seam": (
        [
            _obs("A", 5, 359.98, look_east=0.0, t="2012-01-01"),
            _obs("B", 5, 0.02, look_east=30.0, t="2012-02-01"),
        ],
        {},
    ),
    "no_geometry": (
        [
            _obs("A", 0, 0, look_east=2.0, t="2012-01-01", geometry=False),
            _obs("B", 0, 0, look_east=20.0, t="2012-02-01", geometry=False),
        ],
        {},
    ),
    "require_geometry": (
        [
            _obs("A", 0, 0, look_east=2.0, t="2012-01-01", geometry=False),
            _obs("B", 0, 0, look_east=20.0, t="2012-02-01", geometry=False),
        ],
        {"require_geometry": True},
    ),
    "aoi_circle": (
        [
            _obs("A", 0, 0.0, look_east=0.0, t="2012-01-01", half=0.2),
            _obs("B", 0, 0.0, look_east=25.0, t="2012-02-01", half=0.2),
        ],
        {"aoi": {"point": (0.05, 0.05), "radius_km": 2.0}},
    ),
    "aoi_box": (
        [
            _obs("A", 0, 0.0, look_east=0.0, t="2012-01-01", half=0.2),
            _obs("B", 0, 0.15, look_east=25.0, t="2012-02-01", half=0.2),
        ],
        {"aoi": "POLYGON((-0.1 -0.1, 0.1 -0.1, 0.1 0.1, -0.1 0.1, -0.1 -0.1))"},
    ),
    "aoi_miss": (
        [
            _obs("A", 0, 0.0, look_east=0.0, t="2012-01-01", half=0.2),
            _obs("B", 0, 0.0, look_east=25.0, t="2012-02-01", half=0.2),
        ],
        {"aoi": {"point": (5.0, 5.0), "radius_km": 2.0}},
    ),
    "resolution": (
        [
            _obs("A", 0, 0, look_east=0.0, t="2012-01-01", ps=0.5),
            _obs("B", 0, 0, look_east=25.0, t="2012-02-01", ps=2.0),
            _obs("C", 0, 0, look_east=-25.0, t="2012-03-01", ps=0.6),
        ],
        {"max_resolution_ratio": 1.5, "weights": {"resolution": 1.0}},
    ),
}


@needs_spatial
@pytest.mark.parametrize("case", sorted(STEREO_FRAMES))
def test_pairs_match_find_stereo_pairs(case):
    df = _df(*STEREO_FRAMES[case][0])
    kw = STEREO_FRAMES[case][1]
    py = find_stereo_pairs(df, **kw)
    sql = find_stereo_pairs_duckdb(df, **kw)
    _assert_same_pairs(py, sql)


@needs_spatial
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"convergence": (5.0, 40.0), "min_overlap": 0.3, "max_azimuth_diff": 60.0},
        {"aoi": {"point": (-89.9, 45.0), "radius_km": 6.0}, "min_aoi_coverage": 0.2},
    ],
)
def test_pairs_match_on_polar_frames(kw):
    df = _polar_frames(80, seed=3)
    df.index = df.index * 10 + 7  # non-default labels
    py = find_stereo_pairs(df, **kw)
    sql = find_stereo_pairs_duckdb(df, **kw)
    assert len(py) > 5
    _assert_same_pairs(py, sql, ordered=False)


@needs_spatial
def test_pairs_match_on_cumindex(cumindex):
    obs = cumindex.to_observations(cumindex.search(lat=(-90, -85), nac_only=True, wkt=True))
    kw = {"convergence": (0, 90), "min_overlap": 0.0}
    py, sql = find_stereo_pairs(obs, **kw), find_stereo_pairs_duckdb(obs, **kw)
    assert len(py) > 0
    _assert_same_pairs(py, sql, ordered=False)
    assert find_stereo_pairs_duckdb(obs).empty


@needs_spatial
def test_images_table(ws, cumindex):
    df = _polar_frames(5)
    assert ws.add_observations(df) == 5
    img = ws.query("SELECT *, ST_AsWKB(footprint) AS wkb FROM images ORDER BY idx")
    assert list(img.product_id) == list(df.product_id) and list(img.idx) == list(range(5))
    assert (img.hemi == "S").all() and img.footprint_wkt.notna().all()
    import shapely

    for row, wkb in zip(df.to_dict("records"), img.wkb):
        rings = footprint_rings(row, "footprint_wkt")
        expected = footprint_geometry(workspace._great_circle_rings(rings), ps_projection("S"))
        assert shapely.from_wkb(bytes(wkb)).equals_exact(expected, 1e-6)
        straight = footprint_geometry(rings, ps_projection("S"))
        assert expected.symmetric_difference(straight).area < 1e-3 * straight.area
    assert img.footprint_km2.to_numpy() == pytest.approx(125.0, rel=0.01)
    x, y = ps_xy(df.center_lat[0], df.center_lon[0], "S")
    assert img.xmin[0] < x < img.xmax[0] and img.ymin[0] < y < img.ymax[0]
    # Re-adding replaces rows (new idx); bbox-only rows get a lon/lat WKT
    box = _df(_obs("B1", -85.0, 10.0), _obs("N1", 85.0, 10.0, geometry=False))
    ws.add_observations(pd.concat([df.iloc[:2], box], ignore_index=True))
    img = ws.query("SELECT product_id, idx, hemi, footprint_wkt FROM images ORDER BY idx")
    assert len(img) == 7 and list(img.product_id[-4:]) == [*df.product_id[:2], "B1", "N1"]
    assert list(img.hemi[-2:]) == ["S", "N"] and img.footprint_wkt.str.startswith("POLYGON").all()
    rows = cumindex.search(lat=(-90, -85), nac_only=True, wkt=True)
    assert ws.add_index_rows(rows) == len(rows)
    n = ws.query("SELECT count(*) AS n FROM images WHERE footprint IS NOT NULL AND source = 'cumindex'").n[0]
    assert n == len(rows)


@needs_spatial
def test_pairs_table_and_subsets(ws):
    df = _polar_frames(40, seed=1)
    ws.add_observations(df)
    res = ws.pairs()
    stored = ws.query("SELECT * FROM pairs")
    assert len(stored) == len(res) > 0 and list(stored.columns) == PAIR_COLUMNS
    assert res["score"].is_monotonic_decreasing
    ids = list(df.product_id[:15])
    sub = ws.pairs(product_ids=ids, table=None)
    assert set(sub.product_id_1) | set(sub.product_id_2) <= set(ids)
    assert len(ws.query("SELECT * FROM pairs")) == len(res)
    assert ws.query("SELECT count(*) AS n FROM duckdb_tables() WHERE temporary").n[0] == 0


def _write_ba(prefix: Path, names, rng, offsets_p85=None, large=None):
    """Synthetic ASP bundle_adjust reports under *prefix*."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [f"/w/images/{n}.cal.echo.cub" for n in names]
    for key, scale in (("initial", 5.0), ("final", 0.4)):
        rows, raw = [], []
        for k, p in enumerate(paths):
            cnt = 0 if k == 1 else 40 + k
            norms = rng.exponential(scale, cnt)
            if key == "final" and large is not None:
                norms = np.r_[np.full(int(cnt * large[k]), 1.5), np.full(cnt - int(cnt * large[k]), 0.2)]
            ang = rng.uniform(0, 2 * np.pi, cnt)
            raw.append(f"{p}, {cnt}")
            raw += [f"{float(n * math.cos(a))!r}, {float(n * math.sin(a))!r}" for n, a in zip(norms, ang)]
            mean = float(np.mean(norms)) if cnt else float("nan")
            med = float(np.sort(norms)[cnt // 2]) if cnt else float("nan")
            rows.append(f"{p}, {mean}, {med}, {cnt}")
        Path(f"{prefix}-{key}_residuals_stats.txt").write_text(
            "# Pixel reprojection error per camera\n# Image, mean, median, count\n" + "\n".join(rows) + "\n"
        )
        Path(f"{prefix}-{key}_residuals_raw_pixels.txt").write_text("\n".join(raw) + "\n")
        pts = [f"{rng.uniform(0, 360)}, {rng.uniform(-89.9, -85)}, 12.5, {rng.exponential(scale)}, 2"]
        pts.append("10.0, -86.0, 1.0, 0.75, 3 # GCP")
        Path(f"{prefix}-{key}_residuals_pointmap.csv").write_text(
            "# lon, lat, height_above_datum, mean_residual, num_observations\n"
            "# Geodetic Datum --> Name: D_MOON  Spheroid: MOON  Semi-major axis: 1737400\n"
            + "\n".join(pts)
            + "\n"
        )
    p85 = offsets_p85 if offsets_p85 is not None else [0.5 + k for k in range(len(paths))]
    offs = [
        f"{p} -1 -1 -1 -1 -1 0" if k == 1 else f"{p} 0.2 0.4 0.8 {p85[k]} {p85[k] + 1} {100 + k}"
        for k, p in enumerate(paths)
    ]
    Path(f"{prefix}-mapproj_match_offset_stats.txt").write_text(
        "# Percentiles of distances between mapprojected matching pixels in an image and the others.\n"
        "# image_name 25% 50% 75% 85% 95% count\n" + "\n".join(offs) + "\n"
    )
    conv = [f"{paths[k]} {paths[k + 1]} 1.5 {2.5 + k} 3.5 {120 - k}" for k in range(len(paths) - 1)]
    Path(f"{prefix}-convergence_angles.txt").write_text(
        "# Convergence angle percentiles (in degrees) for each image pair having matches\n"
        "# left_image right_image 25% 50% 75% num_matches\n" + "\n".join(conv) + "\n"
    )
    Path(f"{prefix}-camera_offsets.txt").write_text(
        "# Per-image absolute horizontal and vertical change in camera center (meters)\n"
        + "".join(f"{p} {3.0 + k} {0.5 * k}\n" for k, p in enumerate(paths))
    )
    return paths


def test_load_bundle_adjust(tmp_path):
    rng = np.random.default_rng(0)
    prefix = tmp_path / "w" / "ba" / "run"
    names = ["M1LE", "M2LE", "M3RE", "M4LE"]
    _write_ba(prefix, names, rng)
    with Workspace(tmp_path / "w", spatial=False) as ws:
        counts = ws.load_bundle_adjust(prefix)
        assert ws.runs() == ["ba"]
        assert counts["ba_residual_stats"] == 8 and counts["ba_match_offsets"] == 4
        assert counts["ba_points"] == 4 and counts["ba_convergence"] == 3 and counts["ba_camera_offsets"] == 4
        assert counts["ba_pixel_residuals"] == 2 * sum(40 + k for k in (0, 2, 3))
        # Same per-image table as the report's file reader
        files = sfs_report.read_ba_reports(prefix)
        pd.testing.assert_frame_equal(ws.ba_report("ba"), files, check_dtype=False)
        pd.testing.assert_frame_equal(
            ws.convergence("ba"), sfs_report.read_convergence(Path(f"{prefix}-convergence_angles.txt"))
        )
        for key in ("initial", "final"):
            expected = sfs_report.read_pointmap_residuals(Path(f"{prefix}-{key}_residuals_pointmap.csv"))
            assert np.allclose(np.sort(ws.point_residuals("ba", key)), np.sort(expected))
        pts = ws.query("SELECT * FROM ba_points WHERE stage = 'final' ORDER BY kind")
        assert list(pts.kind) == ["gcp", "tie"] and list(pts.num_observations) == [3, 2]
        assert (pts.x[0], pts.y[0]) == pytest.approx(ps_xy(-86.0, 10.0, "S"))
        off = ws.query("SELECT * FROM ba_match_offsets WHERE product_id = 'M2LE'")
        assert off.p85.isna().all() and off.match_count[0] == 0
        stats = ws.query("SELECT * FROM ba_image_stats ORDER BY image")
        assert list(stats.product_id) == ["M1LE", "M2LE", "M3RE", "M4LE"]
        assert (stats.final_median < stats.initial_median).sum() == 3
        assert stats.camera_offset_horizontal_m.tolist() == [3.0, 4.0, 5.0, 6.0]
        # Reloading replaces the run; another prefix is another run
        ws.load_bundle_adjust(prefix)
        assert ws.query("SELECT count(*) AS n FROM ba_residual_stats").n[0] == 8
        _write_ba(tmp_path / "w" / "ba_refine" / "run", names, rng)
        ws.load_bundle_adjust(tmp_path / "w" / "ba_refine" / "run")
        assert ws.runs() == ["ba", "ba_refine"]
        summ = ws.summary().set_index("run")
        expected = sfs_report.ba_summary(files, 5.0, 1.8)
        assert summ.loc["ba", "images"] == 4
        assert summ.loc["ba", "final_median_mean"] == pytest.approx(expected["final_median_px"]["mean"])
        assert summ.loc["ba", "final_median_std"] == pytest.approx(expected["final_median_px"]["std"])
        assert summ.loc["ba", "offset_p85_max"] == pytest.approx(expected["offset_p85_m"]["max"])
        assert summ.loc["ba", "offset_p85_over_threshold"] == expected["offset_p85_over_threshold"]
        assert summ.loc["ba", "convergence_pairs"] == 3 and summ.loc["ba", "points_final"] == 2
        assert ws.load_bundle_adjust(tmp_path / "none" / "run") == {}


def _flag_images_reference(
    stats, norms, offsets, *, max_offset_p85, max_large_residual_fraction, large_px, min_matches
):
    """The pre-workspace Python implementation of sfs.flag_images."""
    flags = {}
    for name in set(stats) | set(norms) | set(offsets):
        counts = [s.count for s in (stats.get(name), offsets.get(name)) if s is not None]
        nm, off = norms.get(name), offsets.get(name)
        if counts and min(counts) < min_matches:
            flags[name] = f"{min(counts)} matches"
        elif max_offset_p85 is not None and off is not None and off.p85 > max_offset_p85:
            flags[name] = f"match offset p85 {off.p85:.2f} m > {max_offset_p85:g} m"
        elif max_large_residual_fraction is not None and nm is not None and len(nm):
            frac = float(np.mean(nm > large_px))
            if frac > max_large_residual_fraction:
                flags[name] = f"{frac:.0%} of residuals > {large_px:g} px"
    return flags


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "thresholds",
    [
        (1.8, 0.85, 1.0, 1),
        (1.2, 0.5, 0.5, 30),
        (None, None, 1.0, 1),
        (2.5, None, 1.0, 50),
    ],
)
def test_rejections_match_flag_images(seed, thresholds):
    rng = np.random.default_rng(seed)
    n = 40
    names = [f"M{k}LE.cal.echo.cub" for k in range(n)]
    stats, norms, offsets = {}, {}, {}
    for nm in names:
        cnt = int(rng.choice([0, 3, 40, 97, 200]))
        if rng.random() < 0.9:
            stats[nm] = asp.ResidualStats(f"/w/images/{nm}", 0.5, 0.4, cnt)
        if rng.random() < 0.8:
            norms[nm] = rng.exponential(rng.uniform(0.2, 2.0), cnt if rng.random() < 0.9 else 0)
        if rng.random() < 0.8:
            p85 = -1.0 if cnt == 0 else float(rng.uniform(0.3, 3.0))
            offsets[nm] = asp.MatchOffsetStats(f"/w/images/{nm}", 0.1, 0.2, 0.3, p85, p85 + 1, cnt)
    p85, frac, px, mm = thresholds
    kw = {
        "max_offset_p85": p85,
        "max_large_residual_fraction": frac,
        "large_residual_px": px,
        "min_matches": mm,
    }
    expected = _flag_images_reference(
        stats,
        norms,
        offsets,
        max_offset_p85=p85,
        max_large_residual_fraction=frac,
        large_px=px,
        min_matches=mm,
    )
    assert flag_images(stats, norms, offsets, **kw) == expected
    assert len(expected) > 0


def test_rejection_view_files_and_config(tmp_path):
    rng = np.random.default_rng(1)
    prefix = tmp_path / "ba" / "run"
    names = ["M1LE", "M2LE", "M3LE", "M4LE", "M5LE"]
    _write_ba(prefix, names, rng, offsets_p85=[0.5, 0, 2.4, 1.0, 0.9], large=[0.1, 0, 0.1, 0.9, 0.5])
    parsed = (
        asp.read_residual_stats(Path(f"{prefix}-final_residuals_stats.txt")),
        asp.read_residual_norms(Path(f"{prefix}-final_residuals_raw_pixels.txt")),
        asp.read_mapproj_offset_stats(Path(f"{prefix}-mapproj_match_offset_stats.txt")),
    )
    cfg = SfsConfig(max_offset_p85=1.8, max_large_residual_fraction=0.4)
    with Workspace(":memory:", spatial=False) as ws:
        ws.load_bundle_adjust(prefix, run="ba")
        ws.set_rejection_params(cfg)
        got = ws.rejections("ba")
        assert got == flag_images(*parsed, max_offset_p85=1.8, max_large_residual_fraction=0.4)
        assert set(got) == {
            "M2LE.cal.echo.cub",
            "M3LE.cal.echo.cub",
            "M4LE.cal.echo.cub",
            "M5LE.cal.echo.cub",
        }
        rules = ws.query("SELECT product_id, rule FROM image_rejections ORDER BY product_id")
        assert list(rules.rule) == ["matches", "offset", "residuals", "residuals"]
        at = ws.query("SELECT image, reason FROM image_rejections_at(1.8, 0.85, 1.0, 1) ORDER BY image")
        assert dict(zip(at.image, at.reason)) == flag_images(*parsed)
        with pytest.raises(TypeError):
            ws.set_rejection_params(max_offset=1.0)


def test_pipeline_filter_uses_workspace(tmp_path):
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif", SfsConfig(max_offset_p85=1.0, min_images=1))
    names = ["A", "B", "C"]
    p.images = [SfsImage(i, Path(f"/w/images/{i}.cal.echo.cub")) for i in names]
    p.all_images = list(p.images)
    _write_ba(p.refine_prefix, names, np.random.default_rng(2), offsets_p85=[0.5, 0, 1.5])
    removed = p.filter_images(p.refine_prefix, "refine")
    assert [(r.image.product_id, r.reason) for r in removed] == [
        ("B", "0 matches"),
        ("C", "match offset p85 1.50 m > 1 m"),
    ]
    with p.workspace(read_only=True) as ws:
        assert ws.runs() == ["ba_refine"]
        assert ws.query("SELECT max_offset_p85 FROM rejection_params").max_offset_p85[0] == 1.0


def test_dry_run_creates_no_workspace(tmp_path):
    obs = _polar_frames(2)
    p = SfsPipeline(
        tmp_path / "w", tmp_path / "ldem.tif", SfsConfig(registration="residuals"), observations=obs
    )
    with dry_run():
        p.run([tmp_path / f"{pid}.IMG" for pid in obs.product_id])
    assert not (tmp_path / "w" / workspace.DB_NAME).exists()


@needs_spatial
def test_pipeline_records_observations(tmp_path, monkeypatch):
    from lunar_toolkit import isis

    obs = _polar_frames(3)
    monkeypatch.setattr(isis, "prepare_lronac", lambda src, out, **kw: (out / f"{src.stem}.cub", None))
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif", observations=obs)
    p.prepare_images([tmp_path / f"{pid}.IMG" for pid in obs.product_id])
    with p.workspace(read_only=True) as ws:
        assert sorted(ws.query("SELECT product_id FROM images").product_id) == sorted(obs.product_id)


def test_report_reads_workspace(tmp_path):
    w = tmp_path / "w"
    _write_ba(w / "ba" / "run", ["M1LE", "M2LE", "M3LE"], np.random.default_rng(3))
    from_files = sfs_report.metrics(w, sections=["ba"])
    with Workspace(w, spatial=False) as ws:
        ws.load_bundle_adjust(w / "ba" / "run")
    for f in (w / "ba").iterdir():
        f.unlink()
    m = sfs_report.metrics(w, sections=["ba"])
    assert m.summary["ba"]["ba"]["source"] == "workspace"
    assert from_files.summary["ba"]["ba"]["source"] == "files"
    a, b = dict(from_files.summary["ba"]["ba"]), dict(m.summary["ba"]["ba"])
    a.pop("source"), b.pop("source")
    assert a == b
    pd.testing.assert_frame_equal(m.tables["ba_ba"], from_files.tables["ba_ba"], check_dtype=False)
    assert [h.total for h in m.histograms["pointmap_ba"]] == [
        h.total for h in from_files.histograms["pointmap_ba"]
    ]


def test_spatial_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_load_spatial", lambda con: "offline")
    with Workspace(tmp_path / "w") as ws:
        assert not ws.has_spatial
        with pytest.raises(WorkspaceError, match="INSTALL spatial"):
            ws.add_observations(_polar_frames(2))
        with pytest.raises(WorkspaceError):
            ws.pairs()
        _write_ba(tmp_path / "ba" / "run", ["A", "B"], np.random.default_rng(0))
        assert ws.load_bundle_adjust(tmp_path / "ba" / "run")["ba_points"] == 4
        assert set(ws.rejections("ba")) == {"B.cal.echo.cub"}


def test_cli_workspace(tmp_path, capsys):
    w = tmp_path / "w"
    _write_ba(w / "ba" / "run", ["M1LE", "M2LE"], np.random.default_rng(4))
    with Workspace(w, spatial=False) as ws:
        ws.load_bundle_adjust(w / "ba" / "run")
    assert cli.main(["workspace", str(w)]) == 0
    out = capsys.readouterr().out
    assert "ba_residual_stats" in out and "image_rejections" in out
    assert (
        cli.main(
            ["workspace", str(w / "lunar.duckdb"), "--sql", "SELECT image, reason FROM image_rejections"]
        )
        == 0
    )
    assert "0 matches" in capsys.readouterr().out
    csv = tmp_path / "out" / "stats.csv"
    pq = tmp_path / "out" / "stats.parquet"
    assert cli.main(["workspace", str(w), "--export", "ba_image_stats", str(csv)]) == 0
    assert cli.main(["workspace", str(w), "--export", "ba_image_stats", str(pq)]) == 0
    assert len(pd.read_csv(csv)) == len(pd.read_parquet(pq)) == 2
    assert cli.main(["workspace", str(w), "--sql", "SELECT 1 AS x", "--csv", str(tmp_path / "q.csv")]) == 0
    assert (tmp_path / "q.csv").read_text().splitlines() == ["x", "1"]
    capsys.readouterr()
    assert cli.main(["workspace", str(w), "--sql", "SELECT * FROM nope"]) == 1
    assert cli.main(["workspace", str(w), "--export", "nope", str(csv)]) == 1
    assert cli.main(["workspace", str(w), "--sql", "CREATE TABLE t (a INT)"]) == 1
    assert cli.main(["workspace", str(tmp_path / "missing")]) == 1
    assert "no workspace" in capsys.readouterr().err


@needs_spatial
def test_export_geometry(ws, tmp_path):
    ws.add_observations(_polar_frames(3))
    csv = ws.export("images", tmp_path / "images.csv")
    assert pd.read_csv(csv).footprint.str.startswith("POLYGON").all()
    assert len(pd.read_parquet(ws.export("images", tmp_path / "images.parquet"))) == 3
    with pytest.raises(ValueError):
        ws.export("images; DROP TABLE images", tmp_path / "x.csv")


@needs_spatial
def test_cli_pairs_engine(cumindex, tmp_path, capsys):
    base = ["pairs", "--bbox", "0", "-90", "360", "-85", "--conv", "0", "90", "--min-overlap", "0"]
    base += ["--data-dir", str(cumindex._explicit_data_dir)]
    res = {}
    for engine in ("python", "duckdb"):
        out = tmp_path / f"{engine}.csv"
        assert cli.main([*base, "--engine", engine, "--csv", str(out)]) == 0
        res[engine] = pd.read_csv(out)
    py, sql = res["python"], res["duckdb"]
    assert len(py) == len(sql) > 0
    assert set(zip(py.product_id_1, py.product_id_2)) == set(zip(sql.product_id_1, sql.product_id_2))
    assert np.allclose(
        py.sort_values(["product_id_1", "product_id_2"]).overlap_frac,
        sql.sort_values(["product_id_1", "product_id_2"]).overlap_frac,
        atol=2e-3,
    )
    assert cli.main([*base, "--workspace", str(tmp_path / "pws")]) == 0
    assert "Wrote images and pairs" in capsys.readouterr().out
    with Workspace(tmp_path / "pws", read_only=True) as ws:
        assert len(ws.query("SELECT * FROM pairs")) == len(py)
