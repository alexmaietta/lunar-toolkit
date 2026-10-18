"""SfS report: metric functions, BA report parsing, figures, HTML and the sfs-report CLI (offline)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from lunar_toolkit import cli, reference, sfs_products
from lunar_toolkit import sfs_report as sr
from lunar_toolkit.config import RunConfig, dump_config
from lunar_toolkit.sfs import SfsConfig
from lunar_toolkit.sfs_products import Grid, PointComparison, write_raster
from lunar_toolkit.tools import ToolError

X0, Y0, RES, W, H = -1750.0, 122750.0, 5.0, 40, 30
GRID = Grid(CRS.from_proj4(reference.PS_SRS), from_origin(X0, Y0, RES, RES), W, H)


def _terrain() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return 100 + 2.0 * xx + 0.5 * yy + 20 * np.exp(-((xx - 20) ** 2 + (yy - 15) ** 2) / 30)


def _raster(path: Path, array: np.ndarray, nodata=np.nan) -> Path:
    return write_raster(
        path, array.astype(np.float32 if array.dtype.kind == "f" else array.dtype), GRID, nodata
    )


def _ba_reports(prefix: Path, n: int = 4) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    names = [f"/w/images/M{k}LE.cal.echo.cub" for k in range(n)]
    for key, scale in (("initial", 10.0), ("final", 0.3)):
        rows = [
            f"{nm}, {scale * (k + 1):.3f}, {scale * (k + 0.5):.3f}, {100 + k}" for k, nm in enumerate(names)
        ]
        Path(f"{prefix}-{key}_residuals_stats.txt").write_text(
            "# Image, mean, median, count\n" + "\n".join(rows) + "\n"
        )
        pts = "\n".join(f"10.0, -85.9, 1.0, {scale * (1 + j % 3):.3f}, 2" for j in range(20))
        Path(f"{prefix}-{key}_residuals_pointmap.csv").write_text(
            "# lon, lat, height_above_datum, mean_residual, num_observations\n" + pts + "\n"
        )
    offs = [f"{nm} 0.2 0.4 0.8 {0.9 + k} {1.5 + k} {50 + k}" for k, nm in enumerate(names)]
    offs.append("/w/images/M9LE.cal.echo.cub -1 -1 -1 -1 -1 0")
    Path(f"{prefix}-mapproj_match_offset_stats.txt").write_text(
        "# Percentiles of distances between matching pixels after mapprojecting onto DEM.\n"
        "# image_name 25% 50% 75% 85% 95% count\n" + "\n".join(offs) + "\n"
    )
    Path(f"{prefix}-convergence_angles.txt").write_text(
        "# Convergence angle percentiles (in degrees) for each image pair having matches\n"
        "# left_image right_image 25% 50% 75% num_matches\n"
        f"{names[0]} {names[1]} 1.5 2.5 3.5 120\n{names[1]} {names[2]} 4 6 8 80\n"
    )


@pytest.fixture
def work(tmp_path) -> Path:
    """A synthetic SfS work dir: DEMs, products, BA reports, selection files, PGDA-like rasters."""
    w = tmp_path / "w"
    z = _terrain()
    rng = np.random.default_rng(0)
    prior = z + rng.normal(0, 0.5, z.shape)
    prior[0, :5] = np.nan
    _raster(w / "products" / "R_GLDELEV_5.tif", z)
    _raster(w / "dem" / "prior-5m.tif", prior)
    bins = np.zeros((H, W), np.uint8)
    bins[:, 10:20], bins[:, 20:30], bins[:, 30:] = 1, 2, 4
    _raster(w / "products" / "R_GLDSBCT_5.tif", bins, nodata=None)
    _raster(w / "products" / "R_GLDISGM_5_ENS.tif", np.full((H, W), 0.8))
    _raster(w / "products" / "R_GLDSSGM_5_ENS.tif", np.full((H, W), 1.5))
    _raster(w / "products" / "R_GLDISGM_5.tif", np.full((H, W), 0.2))
    rid = RunConfig(work_dir=w, dem="ldem.tif", sfs=SfsConfig(region_id="R", resolution=5.0))
    dump_config(rid, w / "config.yaml")
    _ba_reports(w / "ba" / "run")
    sel = w / "selection"
    sel.mkdir(parents=True)
    imgs = []
    for k in range(3):
        img = np.full((H, W), 0.1)
        img[:, : 10 * (k + 1)] = 0.0
        imgs.append(_raster(w / "prj_sfs" / f"M{k}LE.map.tif", img))
    pd.DataFrame(
        {
            "product_id": ["M0LE", "M1LE", "M2LE"],
            "sun_azimuth": [10.0, 100.0, 200.0],
            "sun_elevation": [1.0, 1.0, 1.0],
            "shadow_fraction": [0.7, 0.4, 0.1],
            "mapprojected": [str(p) for p in imgs],
            "selected": [True, False, True],
        }
    ).to_csv(w / "selection" / "selection.csv", index=False)
    (sel / "removed.csv").write_text(
        "product_id,stage,reason\nM9LE,ba,0 matches\nM8LE,ba,match offset p85 2.10 m > 1.8 m\n"
        "M7LE,refine,90% of residuals > 1 px\nM6LE,selection,97.0% shadowed\n"
    )
    ref = tmp_path / "ref"
    ldec = np.zeros((H, W))
    ldec[::3, ::3] = 2
    _raster(ref / "ldec.tif", ldec)
    _raster(ref / "toterr.tif", np.full((H, W), 0.5))
    _raster(ref / "slperr.tif", np.full((H, W), 2.0))
    _raster(ref / "published.tif", z + 0.25)
    pts = [
        (X0 + 7.5 + 5 * i, Y0 - 7.5 - 5 * j, z[1 + j, 1 + i] - 0.1)
        for i in range(0, 30, 3)
        for j in range(0, 25, 4)
    ]
    with open(ref / "lola.csv", "w") as f:
        f.write("# easting_m,northing_m,height_above_datum_m,rdr_id\n")
        f.writelines(f"{x:.3f},{y:.3f},{h:.3f},1\n" for x, y, h in pts)
    return w


@pytest.fixture
def no_geodiff(monkeypatch):
    def fail(*a, **kw):
        raise ToolError("geodiff not found")

    monkeypatch.setattr(sfs_products, "compare_to_points", fail)


def _inputs(work: Path, **kw) -> sr.ReportInputs:
    ref = work.parent / "ref"
    paths = {
        "ldec": ref / "ldec.tif",
        "toterr": ref / "toterr.tif",
        "slperr": ref / "slperr.tif",
        "lola_points": ref / "lola.csv",
        "published_dem": ref / "published.tif",
    }
    return sr.ReportInputs.from_work_dir(work, **{**paths, **kw})


def test_stats_and_split():
    s = sr.stats([1.0, -1.0, 3.0, np.nan])
    assert s["count"] == 3 and s["median"] == 1.0 and s["abs_mean"] == pytest.approx(5 / 3)
    assert s["rms"] == pytest.approx(np.sqrt(11 / 3)) and s["nmad"] == pytest.approx(1.4826 * 2)
    assert sr.stats([np.nan]) == {"count": 0}
    v = np.array([1.0, 2.0, 3.0, 4.0])
    out = sr.split_stats(v, np.array([0, 1, np.nan, 3]))
    assert out["all"]["count"] == 4 and out["ldec>0"]["count"] == 2 and out["ldec=0"]["count"] == 1


def test_binned_and_normal_summary():
    t = sr.binned(np.array([1.0, 2.0, 3.0, 10.0]), np.array([1.0, 2.0, 7.0, np.nan]), [0, 5, 10, 15])
    assert t["count"].tolist() == [2, 1, 0] and t["median"].iloc[0] == 1.5 and np.isnan(t["median"].iloc[2])
    z = np.random.default_rng(1).normal(0, 1, 20000)
    s = sr.normal_summary(z)
    assert s["std"] == pytest.approx(1, abs=0.03) and s["within_1"] == pytest.approx(0.683, abs=0.02)


def test_coverage_fractions():
    f = sr.coverage_fractions(np.array([[0, 1, 2, 3], [4, 5, 0, 0]]))
    assert f["pixels"] == 8 and f["frac_0"] == 3 / 8 and f["frac_3plus"] == 3 / 8 and f["frac_ge1"] == 5 / 8
    assert f["frac_ge2"] == 4 / 8


def test_histogram_clips_into_outer_bins():
    h = sr.Histogram.of("x", np.array([-100.0, 0.5, 100.0, np.nan]), [-1, 0, 1])
    assert h.counts.tolist() == [1, 2] and h.total == 3


def test_read_ba_reports(work):
    t = sr.read_ba_reports(work / "ba" / "run")
    assert t["image"].tolist() == [f"M{k}LE.cal.echo.cub" for k in (0, 1, 2, 3, 9)]
    assert t["initial_median"].iloc[1] == 15.0 and t["final_median"].iloc[1] == pytest.approx(0.45)
    assert t["offset_p85"].iloc[2] == pytest.approx(2.9) and np.isnan(t["offset_p85"].iloc[4])
    s = sr.ba_summary(t, 5.0, 1.8)
    assert s["images"] == 5 and s["offset_p85_over_threshold"] == 3
    assert s["initial_median_px"]["mean"] == pytest.approx(20.0) and s["frac_p95_below_resolution"] == 1.0
    assert sr.read_ba_reports(work / "nothing" / "run") is None
    conv = sr.read_convergence(work / "ba" / "run-convergence_angles.txt")
    assert conv["p50"].tolist() == [2.5, 6.0] and conv["left"].iloc[0] == "M0LE.cal.echo.cub"
    res = sr.read_pointmap_residuals(work / "ba" / "run-final_residuals_pointmap.csv")
    assert res.shape == (20,) and res.max() == pytest.approx(0.9)


def test_removed_images(work):
    df = sr.removed_images(work)
    assert df["cause"].tolist() == ["matches", "offset", "residuals", "shadow"]
    assert sr.removed_images(work.parent).empty


def test_sample_and_preview_grid():
    a = np.arange(W * H, dtype=float).reshape(H, W)
    v = sr.sample(a, GRID, np.array([X0 + 2.0, X0 + 7.0, X0 - 1.0]), np.array([Y0 - 2.0, Y0 - 7.0, Y0]))
    assert v[0] == 0 and v[1] == W + 1 and np.isnan(v[2])
    g = sr.preview_grid(GRID, 15)
    assert (g.width, g.height, g.res) == (14, 10, 15.0) and sr.preview_grid(GRID, 100) is GRID


def test_solar_bins_from_images(work):
    sel = sr.selection_table(work)
    bins = sr.solar_bins_from_images(sel["mapprojected"], sel["sun_azimuth"], GRID, 0.002)
    assert bins[0, 5] == 0 and bins[0, 15] == 1 and bins[0, 25] == 2 and bins[0, 35] == 3


def test_lola_differences_fallback_and_geodiff(work, tmp_path, no_geodiff, monkeypatch):
    dem = work / "products" / "R_GLDELEV_5.tif"
    pts = sr.lola_differences(dem, work.parent / "ref" / "lola.csv", tmp_path / "lola")
    assert len(pts) == 70 and pts["diff"].to_numpy() == pytest.approx(0.1, abs=1e-3)
    monkeypatch.setattr(
        sfs_products,
        "compare_to_points",
        lambda *a, **kw: PointComparison(np.array([0.0]), np.array([-85.98]), np.array([0.4])),
    )
    one = sr.lola_differences(dem, work.parent / "ref" / "lola.csv", tmp_path / "lola")
    assert one["diff"].tolist() == [0.4] and one["y"].iloc[0] > 0 and abs(one["x"].iloc[0]) < 1e-6


def test_metrics_all_sections(work, no_geodiff):
    m = sr.metrics(_inputs(work))
    assert m.skipped == {}
    s = m.summary
    assert s["ba"]["ba"]["convergence_deg"] == {"pairs": 2, "median": 4.25, "max": 6.0}
    assert (
        s["removed"]["kept"] == 3 and s["removed"]["selected"] == 2 and s["removed"]["bundle_adjusted"] == 5
    )
    assert s["removed"]["by_cause"]["shadow"] == 1
    cov = s["coverage"]["selected_full_res"]
    assert (
        cov["frac_0"] == 0.25
        and cov["frac_1"] == 0.25
        and cov["frac_2"] == 0.25
        and cov["frac_3plus"] == 0.25
    )
    assert s["coverage"]["kept"]["images"] == 3 and s["coverage"]["selected"]["images"] == 2
    prior = s["prior"]
    assert prior["all"]["count"] == W * H - 5 and abs(prior["all"]["mean"]) < 0.1
    assert prior["ldec>0"]["count"] + prior["ldec=0"]["count"] == prior["all"]["count"]
    assert s["lola"]["median"] == pytest.approx(0.1, abs=1e-3) and s["lola"]["count"] == 70
    assert m.tables["lola_vs_slope"]["count"].sum() > 0
    assert s["ensemble"]["elevation_std_m"]["median"] == pytest.approx(0.8)
    assert s["ensemble"]["ensemble_over_height_error"]["median"] == pytest.approx(4.0)
    assert sum(v["count"] for v in s["ensemble"]["by_slope"].values()) > 0
    assert s["uncertainty"]["elevation"]["std"] == pytest.approx(1.0, abs=0.15)
    assert "slope" in s["uncertainty"]
    assert s["published"]["elevation"]["median"] == pytest.approx(-0.25, abs=1e-4)
    json.dumps(sr._jsonable(m.summary))


def test_metrics_skips_missing_inputs(tmp_path):
    w = tmp_path / "empty"
    w.mkdir()
    m = sr.metrics(w)
    assert set(m.skipped) == set(sr.SECTIONS)
    with pytest.raises(ValueError, match="Unknown report sections"):
        sr.metrics(w, sections=["nope"])


def test_make_report(work, tmp_path, no_geodiff):
    out = sr.make_report(_inputs(work), tmp_path / "rep")
    assert out == tmp_path / "rep" / "report.html"
    figs = {p.name for p in (tmp_path / "rep" / "figures").iterdir()}
    assert {
        "hillshade.png",
        "slope_map.png",
        "prior_diff_map.png",
        "coverage_map.png",
        "coverage_bars.png",
        "ba_residuals_ba.png",
        "ba_before_after_ba.png",
        "match_offsets_ba.png",
        "pointmap_residuals_ba.png",
        "diff_histograms.png",
        "lola_vs_slope.png",
        "ensemble_map.png",
        "normalised_elevation.png",
        "normalised_slope.png",
        "published_diff_map.png",
    } <= figs
    tables = {p.name for p in (tmp_path / "rep" / "tables").iterdir()}
    assert {
        "ba_ba.csv",
        "removed.csv",
        "elevation_differences.csv",
        "coverage.csv",
        "lola_vs_slope.csv",
    } <= tables
    doc = json.loads((tmp_path / "rep" / "metrics.json").read_text())
    assert doc["skipped"] == {} and "prior_diff" in doc["histograms"]
    text = out.read_text()
    assert text.isascii() and text.count("data:image/png;base64,") == len(figs)
    assert "Table 2" in text and "tables/ba_ba.csv" in text
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


def test_report_partial_sections(work, tmp_path):
    out = sr.make_report(work, tmp_path / "rep", sections=["ba", "removed"])
    text = out.read_text()
    assert "Skipped" not in text and "Bundle adjustment" in text
    assert not (work / "report").exists()


def test_cli_sfs_report(work, tmp_path, capsys, no_geodiff, monkeypatch):
    ref = work.parent / "ref"
    got = {}

    def fake_pgda(region, kind, out_dir=None, download=True):
        got[kind] = out_dir
        return {"ldec": ref / "ldec.tif", "toterr": ref / "toterr.tif", "slperr": ref / "slperr.tif"}.get(
            kind, ref / f"missing_{kind}"
        )

    monkeypatch.setattr("lunar_toolkit.config.pgda_file", fake_pgda)
    rc = cli.main(
        ["sfs-report", str(work), "--region", "malapert", "--pgda", "ldec", "toterr", "slperr",
         "--reference-dir", str(tmp_path / "cache"), "--lola-points", str(ref / "lola.csv")]
    )  # fmt: skip
    assert rc == 0
    assert capsys.readouterr().out.strip() == f"Report -> {work / 'report' / 'report.html'}"
    assert set(got) == {"ldec", "toterr", "slperr"} and got["ldec"] == tmp_path / "cache" / "pgda"
    doc = json.loads((work / "report" / "metrics.json").read_text())
    assert "ldec>0" in doc["prior"] and "elevation" in doc["uncertainty"] and "published" in doc["skipped"]
    assert cli.main(["sfs-report", str(tmp_path / "nope")]) == 1
