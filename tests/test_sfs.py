"""SfS and stereo pipelines: command sequence, options, filters, selection and resume (offline)."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lunar_toolkit import asp, cli, sfs
from lunar_toolkit.observation import normalize_observations
from lunar_toolkit.sfs import (
    ASP_SFS,
    PRESETS,
    PolarStereographic,
    SfsConfig,
    SfsImage,
    SfsPipeline,
    flag_images,
    prefilter_candidates,
    select_by_illumination,
)
from lunar_toolkit.stereo_dem import stereo_dem
from lunar_toolkit.tools import dry_run

CALIBRATE = ["lronac2isis", "spiceinit", "lronaccal", "lronacecho", "isd_generate"]

SFS_STEPS = [
    *CALIBRATE,
    *CALIBRATE,
    "gdal_translate",
    "gdal_translate",
    "sfs",
    "mapproject",
    "mapproject",
    "parallel_bundle_adjust",
    "pc_align",
    "bundle_adjust",
    "bundle_adjust",
    "mapproject",
    "mapproject",
    "dem_mosaic",
    "gdalinfo",
    "reduce",
    "reduce",
    "parallel_sfs",
    "hillshade",
    "geodiff",
]


@pytest.fixture(autouse=True)
def _no_products(monkeypatch):
    """Use the built-in hillshade + diff unless a test opts into sfs_products."""
    real = sfs.importlib.import_module

    def fake(name, *a):
        if name.endswith("sfs_products"):
            raise ImportError("disabled in tests")
        return real(name, *a)

    monkeypatch.setattr(sfs.importlib, "import_module", fake)


def _run(tmp_path, cfg=None, sources=("A.IMG", "B.IMG"), **kw):
    cfg = cfg or SfsConfig(registration="residuals")
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif", cfg, **kw)
    with dry_run() as cmds:
        dem = p.run([tmp_path / s for s in sources])
    return p, dem, cmds


def _cmd(cmds, tool, k=0):
    return [c for c in cmds if c[0] == tool][k]


def _opt(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _obs(*rows):
    base = dict(
        instrument="LROC NAC",
        center_lat=-89.0,
        center_lon=0.0,
        footprint_wkt="POLYGON((-2 -89.2, 2 -89.2, 2 -88.8, -2 -88.8, -2 -89.2))",
        incidence=88.0,
        emission=1.0,
        sub_spacecraft_lat=-89.0,
        spacecraft_altitude=50.0,
        pixel_scale=1.0,
    )
    return normalize_observations(
        [
            dict(base, product_id=p, start_time=t, sub_spacecraft_lon=lon, sub_solar_azimuth=az)
            for p, t, lon, az in rows
        ]
    )


def test_full_sequence(tmp_path):
    p, dem, cmds = _run(tmp_path, SfsConfig(registration="residuals", projwin=(0, 100, 100, 0)))
    tools = [c[0] for c in cmds]
    assert [t for t in tools if t != "gdal_translate"] == [t for t in SFS_STEPS if t != "gdal_translate"]
    assert dem == tmp_path / "w" / "sfs" / "run-DEM-final.tif"
    w = tmp_path / "w"
    assert _cmd(cmds, "gdal_translate")[1:6] == ["-projwin", "0", "100", "100", "0"]
    prj = _cmd(cmds, "mapproject")
    assert _opt(prj, "--tr") == "1" and prj[prj.index("--t_projwin") + 1 : prj.index("--t_projwin") + 5] == [
        "0",
        "0",
        "100",
        "100",
    ]
    assert _opt(_cmd(cmds, "mapproject", 2), "--tr") == "5"
    assert _opt(_cmd(cmds, "mapproject", 2), "--bundle-adjust-prefix") == str(w / "ba_refine" / "run")
    align = _cmd(cmds, "bundle_adjust")
    assert "--apply-initial-transform-only" in align
    assert _opt(align, "--initial-transform") == str(w / "align" / "run-transform.txt")
    refine = _cmd(cmds, "bundle_adjust", 1)
    assert _opt(refine, "--match-files-prefix") == str(w / "ba" / "run")
    assert _opt(refine, "--input-adjustments-prefix") == str(w / "ba_align" / "run")
    assert _opt(refine, "--heights-from-dem") == str(p.dem)
    assert _opt(refine, "--mapproj-dem") == str(p.dem)
    run_sfs = _cmd(cmds, "parallel_sfs")
    assert _opt(run_sfs, "--bundle-adjust-prefix") == str(w / "sfs_cameras" / "run")
    assert str(w / "images" / "A.cal.echo.red5.cub") in run_sfs  # reduced cube with a re-mapped CSM camera
    assert str(w / "images" / "A.red5.json") in run_sfs
    assert "--use-approx-camera-models" not in run_sfs
    assert not w.exists()


def test_bertone_defaults(tmp_path):
    _, _, cmds = _run(tmp_path)
    ba = _cmd(cmds, "parallel_bundle_adjust")
    assert _opt(ba, "--overlap-limit") == "200"
    assert _opt(ba, "--camera-position-uncertainty") == "100,100"
    assert "--auto-overlap-params" not in ba
    assert _opt(ba, "--mapproj-dem").endswith("prior-5m.tif")
    assert "--match-first-to-last" in ba and "--solve-intrinsics" not in ba
    assert _opt(ba, "--mapprojected-data").split()[0].endswith("prj/A.map.tif")
    assert _opt(_cmd(cmds, "bundle_adjust", 1), "--camera-position-uncertainty") == "100,100"
    s = _cmd(cmds, "parallel_sfs")
    assert _opt(s, "--shadow-threshold") == "0.002"
    assert _opt(s, "--smoothness-weight") == "0.08"
    assert _opt(s, "--initial-dem-constraint-weight") == "0.001"
    assert _opt(s, "--max-iterations") == "5"
    assert "--float-albedo" not in s


def test_asp_preset(tmp_path):
    _, _, cmds = _run(tmp_path, SfsConfig(preset="asp", registration="residuals", reduce=None))
    ba = _cmd(cmds, "parallel_bundle_adjust")
    assert _opt(ba, "--overlap-limit") == "100"
    assert "--camera-position-uncertainty" not in ba and "--auto-overlap-params" not in ba
    s = _cmd(cmds, "parallel_sfs")
    assert _opt(s, "--shadow-threshold") == str(ASP_SFS["shadow_threshold"])
    assert _opt(s, "--initial-dem-constraint-weight") == "0.0025"
    assert set(PRESETS) == {"bertone", "asp"}
    with pytest.raises(ValueError, match="preset"):
        SfsConfig(preset="nope")


def test_option_overrides(tmp_path):
    cfg = SfsConfig(
        registration="residuals",
        bundle_adjust_options={"overlap_limit": 30, "datum": None, "auto_overlap_percent": None},
        refine_options={"heights_from_dem_uncertainty": 100},
        sfs_options={"smoothness_weight": 0.12, "float_albedo": True},
        processes=8,
    )
    _, _, cmds = _run(tmp_path, cfg)
    ba = _cmd(cmds, "parallel_bundle_adjust")
    assert _opt(ba, "--overlap-limit") == "30"
    assert "--datum" not in ba and "--auto-overlap-params" not in ba
    assert _opt(ba, "--processes") == "8"
    assert _opt(_cmd(cmds, "bundle_adjust", 1), "--heights-from-dem-uncertainty") == "100"
    s = _cmd(cmds, "parallel_sfs")
    assert _opt(s, "--smoothness-weight") == "0.12"
    assert "--float-albedo" in s


def test_reduce_modes(tmp_path):
    _, _, cmds = _run(tmp_path, SfsConfig(registration="residuals", reduce=None))
    s = _cmd(cmds, "parallel_sfs")
    assert "reduce" not in [c[0] for c in cmds]
    assert str(tmp_path / "w" / "images" / "A.json") in s and "--use-approx-camera-models" not in s
    assert _opt(s, "--bundle-adjust-prefix") == str(tmp_path / "w" / "ba_refine" / "run")

    obs = _obs(("A", "2010-01-01", 0, 10), ("B", "2011-01-01", 0, 50))
    obs.loc[0, "pixel_scale"] = 0.5
    _, _, cmds = _run(tmp_path, SfsConfig(registration="residuals"), observations=obs)
    reduces = [c for c in cmds if c[0] == "reduce"]
    assert [r[-1] for r in reduces] == ["lscale=10", "lscale=5"]

    _, _, cmds = _run(tmp_path, SfsConfig(registration="residuals", reduce=3))
    assert _cmd(cmds, "reduce")[-1] == "lscale=3"
    assert (
        SfsPipeline(tmp_path, tmp_path, SfsConfig(resolution=1.2)).reduce_factor(SfsImage("A", tmp_path)) == 1
    )


def test_sfs_inputs_copies_adjustments(tmp_path):
    cfg = SfsConfig(refine=False, registration="residuals", sfs_cameras="isis")
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif", cfg)
    im = SfsImage("A", tmp_path / "w" / "images" / "A.cal.echo.cub", tmp_path / "w" / "images" / "A.json")
    (tmp_path / "w" / "images").mkdir(parents=True)
    (tmp_path / "w" / "images" / "A.red5.cub").touch()
    (tmp_path / "w" / "ba_align").mkdir()
    (tmp_path / "w" / "ba_align" / "run-A.adjust").write_text("1 2 3\n1 0 0 0\n")
    cubes, cams, prefix, isis_cams = p.sfs_inputs([im])
    assert cubes == [tmp_path / "w" / "images" / "A.red5.cub"] and cams == [] and isis_cams
    assert (tmp_path / "w" / "sfs_cameras" / "run-A.red5.adjust").read_text() == "1 2 3\n1 0 0 0\n"
    assert prefix == tmp_path / "w" / "sfs_cameras" / "run"


def test_sfs_inputs_reduced_csm_cameras(tmp_path):
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif", SfsConfig(refine=False, registration="residuals"))
    images = tmp_path / "w" / "images"
    im = SfsImage("A", images / "A.cal.echo.cub", images / "A.json")
    images.mkdir(parents=True)
    for name in ("A.cal.echo.red5.cub", "A.red5.json"):  # already reduced: no tools needed
        (images / name).touch()
    (tmp_path / "w" / "ba_align").mkdir()
    (tmp_path / "w" / "ba_align" / "run-A.adjust").write_text("1 2 3\n1 0 0 0\n")
    cubes, cams, prefix, isis_cams = p.sfs_inputs([im])
    assert cubes == [images / "A.cal.echo.red5.cub"] and cams == [images / "A.red5.json"]
    assert not isis_cams and prefix == tmp_path / "w" / "sfs_cameras" / "run"
    assert (tmp_path / "w" / "sfs_cameras" / "run-A.red5.adjust").read_text() == "1 2 3\n1 0 0 0\n"


def test_published_cameras_skip_bundle_adjustment(tmp_path):
    cfg = SfsConfig(registration="residuals", cameras_from="published", region_id="A311")
    _, _, cmds = _run(tmp_path, cfg)
    tools = [c[0] for c in cmds]
    assert "parallel_bundle_adjust" not in tools and "bundle_adjust" not in tools and "pc_align" not in tools
    sfs_cmd = _cmd(cmds, "parallel_sfs")
    assert "--bundle-adjust-prefix" not in sfs_cmd
    assert str(tmp_path / "w" / "images" / "A.published.red5.json") in sfs_cmd


def test_kmedoids_selection_respects_nadir(tmp_path, monkeypatch):
    cfg = SfsConfig(selection="kmedoids", selection_clusters=2, max_sfs_slew=5.0)
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif", cfg)
    p.images = [
        sfs.SfsImage(n, tmp_path / f"{n}.cub", sun_azimuth=az, slew_angle=sl)
        for n, az, sl in (("A", 10, 0), ("B", 100, 0), ("C", 200, 20), ("D", 300, 1))
    ]
    lit = np.ones((4, 16), bool)
    monkeypatch.setattr(p, "_lit_masks", lambda: lit)
    chosen = p.select_images()
    assert "C" not in {im.product_id for im in chosen} and chosen


def test_stereo_registration(tmp_path):
    obs = _obs(("A", "2010-01-01", 0, 10), ("B", "2011-01-01", 40, 15), ("C", "2012-01-01", -30, 200))
    p, _, cmds = _run(tmp_path, SfsConfig(stereo_pairs=1), ("A.IMG", "B.IMG", "C.IMG"), observations=obs)
    w = tmp_path / "w"
    st = _cmd(cmds, "parallel_stereo")
    assert st[-1] == str(p.dem)
    assert str(w / "prj" / "A.map.tif") in st and str(w / "prj" / "B.map.tif") in st
    assert _opt(st, "--bundle-adjust-prefix") == str(w / "ba" / "run")
    assert len([c for c in cmds if c[0] == "parallel_stereo"]) == 1
    assert _opt(_cmd(cmds, "point2dem"), "--tr") == "5"
    mosaic = _cmd(cmds, "dem_mosaic")
    assert mosaic[-1] == str(w / "stereo" / "mosaic.tif")
    al = _cmd(cmds, "pc_align")
    assert al[-3:] == [str(w / "stereo" / "mosaic.tif"), "-o", str(w / "align" / "run")]
    assert "--compute-translation-only" in al and str(p.dem) in al


def test_stereo_registration_falls_back(tmp_path):
    obs = _obs(("A", "2010-01-01", 0, 10), ("B", "2011-01-01", 0, 15))
    _, _, cmds = _run(tmp_path, SfsConfig(), observations=obs)
    assert "parallel_stereo" not in [c[0] for c in cmds]
    assert _cmd(cmds, "pc_align")[-3].endswith("run-final_residuals_pointmap.csv")


def test_registration_none(tmp_path):
    _, _, cmds = _run(tmp_path, SfsConfig(registration="none", refine=False, reduce=None))
    tools = [c[0] for c in cmds]
    assert "pc_align" not in tools and "bundle_adjust" not in tools
    assert _opt(_cmd(cmds, "parallel_sfs"), "--bundle-adjust-prefix") == str(tmp_path / "w" / "ba" / "run")


def test_resume_skips_existing(tmp_path):
    w = tmp_path / "w"
    for rel in (
        "images/A.cal.echo.cub",
        "images/A.json",
        "images/B.cal.echo.cub",
        "images/B.json",
        "images/A.cal.echo.red5.cub",
        "images/A.red5.json",
        "images/B.cal.echo.red5.cub",
        "images/B.red5.json",
        "dem/prior.tif",
        "dem/prior-5m.tif",
        "prj/A.map.tif",
        "prj/B.map.tif",
        "ba/run-final_residuals_stats.txt",
        "align/run-transform.txt",
        "ba_align/run-applied-transform.txt",
        "ba_refine/run-final_residuals_stats.txt",
        "sfs_cameras/run-A.red5.adjust",
        "sfs_cameras/run-B.red5.adjust",
    ):
        (w / rel).parent.mkdir(parents=True, exist_ok=True)
        (w / rel).touch()
    _, _, cmds = _run(tmp_path)
    assert [c[0] for c in cmds] == [
        "sfs",
        "mapproject",
        "mapproject",
        "dem_mosaic",
        "gdalinfo",
        "parallel_sfs",
        "hillshade",
        "geodiff",
    ]


def test_sort_by_sun(tmp_path, monkeypatch):
    out = (
        "Sun azimuth and elevation for: /x/A.cal.echo.cub are 200 and 1 degrees.\n"
        "Sun azimuth and elevation for: /x/B.cal.echo.cub are 40 and 2 degrees.\n"
    )
    monkeypatch.setattr(asp, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif")
    with dry_run():
        p.prepare_images([tmp_path / "A.IMG", tmp_path / "B.IMG"])
    p.sort_by_sun()
    assert [im.product_id for im in p.images] == ["B", "A"]
    assert p.images[0].sun_azimuth == 40 and p.images[1].sun_elevation == 1


def test_flag_images():
    stats = {
        "A.cub": asp.ResidualStats("A.cub", 0.3, 0.2, 500),
        "B.cub": asp.ResidualStats("B.cub", 0.0, 0.0, 0),
        "C.cub": asp.ResidualStats("C.cub", 0.3, 0.2, 500),
        "D.cub": asp.ResidualStats("D.cub", 3.0, 2.0, 500),
    }
    norms = {"A.cub": np.full(100, 0.2), "D.cub": np.r_[np.full(90, 2.0), np.full(10, 0.2)]}
    off = {
        "A.cub": asp.MatchOffsetStats("A.cub", 0.2, 0.4, 0.8, 1.0, 2.0, 500),
        "C.cub": asp.MatchOffsetStats("C.cub", 0.5, 1.0, 1.5, 2.4, 4.0, 500),
    }
    flags = flag_images(stats, norms, off)
    assert set(flags) == {"B.cub", "C.cub", "D.cub"}
    assert flags["B.cub"] == "0 matches" and "2.40 m" in flags["C.cub"] and "90%" in flags["D.cub"]
    assert flag_images(stats, norms, off, max_offset_p85=None, max_large_residual_fraction=None) == {
        "B.cub": "0 matches"
    }


def test_filter_images_reads_reports(tmp_path):
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif")
    p.images = [SfsImage(i, Path(f"/w/images/{i}.cal.echo.cub")) for i in ("A", "B")]
    p.all_images = list(p.images)
    (tmp_path / "w" / "ba").mkdir(parents=True)
    (tmp_path / "w" / "ba" / "run-mapproj_match_offset_stats.txt").write_text(
        "# image_name 25% 50% 75% 85% 95% count\n"
        "/w/images/A.cal.echo.cub 0.2 0.4 0.9 1.2 2.5 5120\n"
        "/w/images/B.cal.echo.cub 0.5 1.1 1.7 2.1 3.0 4000\n"
    )
    removed = p.filter_images(p.ba_prefix, "ba")
    assert [r.image.product_id for r in removed] == ["B"] and [im.product_id for im in p.images] == ["A"]
    rows = (tmp_path / "w" / "selection" / "removed.csv").read_text().splitlines()
    assert rows[0] == "product_id,stage,reason" and rows[1].startswith("B,ba,match offset p85 2.10 m")
    assert p.filter_images(tmp_path / "none" / "run", "refine") == []


def test_select_by_illumination():
    lit = np.zeros((5, 6), bool)
    lit[0, :] = True
    lit[1, :3] = True
    lit[2, :] = True
    lit[3, 3:] = True
    lit[4, :] = True
    az = np.array([0.0, 10.0, 60.0, 100.0, 140.0])
    chosen, views = select_by_illumination(lit, az, views=3, azimuth_sep=30)
    assert chosen[:3] == [0, 2, 4] and 1 not in chosen and 3 not in chosen
    assert list(views) == [3] * 6
    chosen, _ = select_by_illumination(lit, az, max_images=1)
    assert chosen == [0]
    chosen, views = select_by_illumination(lit, np.array([np.nan, 10, 10, 10, 10]), views=3)
    assert len(chosen) == 1 and views.max() == 1


def _selection_pipeline(tmp_path, monkeypatch, cfg):
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif", cfg)
    p.images = [
        SfsImage(pid, Path(f"{pid}.cub"), sun_azimuth=az, mapprojected=tmp_path / f"{pid}.map.tif")
        for pid, az in (("A", 0.0), ("B", 5.0), ("C", 60.0), ("D", 120.0), ("E", 200.0))
    ]
    lit = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 0.0}
    monkeypatch.setattr(asp, "raster_bounds", lambda path: (0.0, 40.0, 40.0, 0.0))

    def fake_read(path, projwin=None, outsize=None, r="average"):
        assert projwin == (0.0, 40.0, 40.0, 0.0) and outsize == (cfg.selection_grid, cfg.selection_grid)
        data = np.full(outsize[::-1], lit[Path(path).name[0]])
        data[0, 0] = np.nan
        return asp.Raster(data, 0.0, 40.0, 1.0, -1.0)

    monkeypatch.setattr(asp, "read_raster", fake_read)
    return p


def test_select_images_coverage(tmp_path, monkeypatch):
    p = _selection_pipeline(tmp_path, monkeypatch, SfsConfig(selection_grid=4))
    selected = p.select_images()
    assert [im.product_id for im in selected] == ["A", "C", "D"]
    assert [im.product_id for im in p.unselected] == ["B"]
    assert [(r.image.product_id, r.stage) for r in p.removed] == [("E", "selection")]
    sel = tmp_path / "w" / "selection"
    assert (sel / "selected.txt").read_text() == "A\nC\nD\n"
    assert (sel / "unselected.txt").read_text() == "B\n"
    table = pd.read_csv(sel / "selection.csv")
    assert list(table["product_id"]) == ["A", "B", "C", "D"] and list(table["selected"]) == [
        True,
        False,
        True,
        True,
    ]
    assert table["mapprojected"][0] == str(tmp_path / "A.map.tif")


def test_select_images_image_subset(tmp_path, monkeypatch):
    p = _selection_pipeline(tmp_path, monkeypatch, SfsConfig(selection="image_subset", selection_grid=4))
    calls = []

    def fake_subset(images, out, threshold, **kw):
        calls.append((Path(out).name, [Path(i).name for i in images], kw["t_projwin"]))
        return [(str(images[0]), 10)]

    monkeypatch.setattr(asp, "image_subset", fake_subset)
    selected = p.select_images()
    assert [c[0] for c in calls] == [
        "image_subset_az000.txt",
        "image_subset_az060.txt",
        "image_subset_az120.txt",
    ]
    assert calls[0][1] == ["A.map.tif", "B.map.tif"] and calls[0][2] == (0.0, 0.0, 40.0, 40.0)
    assert [im.product_id for im in selected] == ["A", "C", "D"]


def test_qa_mosaics(tmp_path):
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif")
    p.images = [
        SfsImage(pid, Path(f"{pid}.cub"), sun_azimuth=az) for pid, az in (("A", 5), ("B", 29.9), ("C", 425))
    ]
    with dry_run() as cmds:
        out = p.qa_mosaics()
    assert out == {
        k: tmp_path / "w" / "qa" / f"{k}.tif" for k in ("max_lit", "max_lit_az000", "max_lit_az060")
    }
    assert all("--max" in c for c in cmds) and [len(c) for c in cmds] == [7, 6, 5]


def test_products_hook(tmp_path, monkeypatch):
    seen = {}

    class FakeProducts:
        @staticmethod
        def make_products(sfs_dem, prior_dem, out_dir, images, **kw):
            seen.update(sfs_dem=sfs_dem, prior=prior_dem, out_dir=out_dir, images=images, **kw)
            return {"elevation": out_dir / "x.tif"}

    monkeypatch.setattr(sfs.importlib, "import_module", lambda name: FakeProducts)
    p = SfsPipeline(tmp_path / "malapert", tmp_path / "ldem.tif")
    p.selected = [
        SfsImage("A", Path("A.cub"), sun_azimuth=12.0, pixel_scale=0.9, mapprojected=Path("A.map.tif"))
    ]
    out = p.products(p.sfs_dem)
    assert out == {"elevation": tmp_path / "malapert" / "products" / "x.tif"}
    assert seen["images"] == [(Path("A.map.tif"), 12.0, 0.9)] and seen["region_id"] == "malapert"
    assert seen["prior"] == p.dem and seen["resolution"] == 5.0 and seen["shadow_threshold"] == 0.002
    assert seen["height_error"] is None


def test_ensemble(tmp_path):
    cfg = SfsConfig(registration="residuals", projwin=(0, 5000, 5000, 0), ensemble_members=2, ensemble_size=2)
    p, _, cmds = _run(tmp_path, cfg, ("A.IMG", "B.IMG", "C.IMG", "D.IMG"))
    runs = [c for c in cmds if c[0] == "parallel_sfs"]
    assert len(runs) == 3
    assert runs[1][2] == str(tmp_path / "w" / "ensemble" / "prior.tif")
    assert runs[1][-1] == str(tmp_path / "w" / "ensemble" / "m00" / "run")
    crop = [c for c in cmds if c[0] == "gdal_translate"][-1]
    assert crop[1:6] == ["-projwin", "1250", "3750", "3750", "1250"]


def test_prefilter_candidates():
    df = pd.DataFrame(
        {
            "PRODUCT_ID": ["A", "B", "C", "D", "E"],
            "SLEW_ANGLE": [0.5, -4.9, 6.0, 1.0, None],
            "RESOLUTION": [1.0, 1.4, 1.0, 1.6, 1.0],
        }
    )
    assert list(prefilter_candidates(df)["PRODUCT_ID"]) == ["A", "B", "C", "E"]  # slewed kept for stereo
    assert list(prefilter_candidates(df, max_slew=5)["PRODUCT_ID"]) == ["A", "B"]
    obs = pd.DataFrame({"product_id": ["A", "B"], "pixel_scale": [1.0, 2.0]})
    assert list(prefilter_candidates(obs)["product_id"]) == ["A"]


def test_polar_stereographic():
    ps = PolarStereographic.from_proj4(
        "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
    )
    # PROJ (gdaltransform) gives 30326.4298062853 52526.9172366573 for lon 30, lat -88.
    assert ps.xy(-88.0, 30.0) == pytest.approx((30326.4298062853, 52526.9172366573), abs=1e-6)
    pn = PolarStereographic.from_proj4(
        "+proj=stere +lat_0=90 +lon_0=45 +lat_ts=90 +x_0=100 +y_0=-50 +a=1737400 +b=1737400"
    )
    assert pn.xy(88.0, 30.0) == pytest.approx((-15598.1152076628, -58636.1635380671), abs=1e-6)
    for bad in (
        "+proj=stere +lat_0=-9.6 +lon_0=15 +k=1 +R=1737400 +units=m",
        "+proj=eqc +lat_ts=0 +R=1737400",
        "+proj=stere +lat_0=-90 +lat_ts=-80 +R=1737400",
        "+proj=stere +lat_0=-90 +k=1",
    ):
        with pytest.raises(ValueError, match="polar stereographic"):
            PolarStereographic.from_proj4(bad)


def test_aoi_projwin(tmp_path, monkeypatch):
    monkeypatch.setattr(
        asp, "srs_proj4", lambda dem: "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m"
    )
    win = sfs.aoi_projwin("POLYGON((0 -88, 90 -88, 180 -88, 270 -88, 0 -88))", tmp_path / "d.tif", snap=5)
    r = PolarStereographic("S").xy(-88.0, 0.0)[1]
    assert win[0] == -win[2] and win[1] == -win[3] and win[2] >= r and win[2] % 5 == 0
    assert win[0] < win[2] and win[3] < win[1]


def test_stereo_dem_sequences(tmp_path):
    with dry_run() as cmds:
        dem = stereo_dem(tmp_path / "L.IMG", tmp_path / "R.IMG", tmp_path / "w")
    assert dem == tmp_path / "w" / "stereo" / "run-DEM.tif"
    assert [c[0] for c in cmds][-3:] == ["bundle_adjust", "parallel_stereo", "point2dem"]

    with dry_run() as cmds:
        dem = stereo_dem(
            tmp_path / "L.cub",
            tmp_path / "R.cub",
            tmp_path / "w2",
            prior_dem=tmp_path / "ldem.tif",
            csm=False,
        )
    tools = [c[0] for c in cmds]
    assert tools == [
        "bundle_adjust",
        "mapproject",
        "mapproject",
        "parallel_stereo",
        "point2dem",
        "pc_align",
        "point2dem",
    ]
    st = _cmd(cmds, "parallel_stereo")
    assert st[-1] == str(tmp_path / "ldem.tif")
    assert dem == tmp_path / "w2" / "align" / "run-trans_source-DEM.tif"


def test_stereo_dem_reuses_inputs(tmp_path):
    with dry_run() as cmds:
        dem = stereo_dem(
            tmp_path / "L.cub",
            tmp_path / "R.cub",
            tmp_path / "w",
            resolution=5,
            prior_dem=tmp_path / "ldem.tif",
            align=False,
            cameras=(tmp_path / "L.json", tmp_path / "R.json"),
            mapprojected=(tmp_path / "L.map.tif", tmp_path / "R.map.tif"),
            bundle_adjust_prefix=tmp_path / "ba" / "run",
        )
    assert [c[0] for c in cmds] == ["parallel_stereo", "point2dem"]
    st = cmds[0]
    assert st[-6:] == [
        str(tmp_path / "L.map.tif"),
        str(tmp_path / "R.map.tif"),
        str(tmp_path / "L.json"),
        str(tmp_path / "R.json"),
        str(tmp_path / "w" / "stereo" / "run"),
        str(tmp_path / "ldem.tif"),
    ]
    assert _opt(st, "--bundle-adjust-prefix") == str(tmp_path / "ba" / "run")
    assert dem == tmp_path / "w" / "stereo" / "run-DEM.tif"


def test_cli_sfs_dry_run(tmp_path, capsys):
    rc = cli.main(
        [
            "sfs",
            str(tmp_path / "A.IMG"),
            "M1234567890LE",
            "--dem",
            str(tmp_path / "d.tif"),
            "--work-dir",
            str(tmp_path / "w"),
            "--dry-run",
            "--registration",
            "residuals",
            "--reduce",
            "none",
            "--sfs-opt",
            "max_iterations=10",
        ]
    )
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("lronac2isis")
    assert any("M1234567890LE.IMG" in line for line in lines)
    assert any(line.startswith("parallel_sfs") and "--max-iterations 10" in line for line in lines)
    assert not any(line.startswith("reduce") for line in lines)
    assert not (tmp_path / "w").exists()


def test_cli_sfs_aoi_search(tmp_path, capsys, monkeypatch):
    seen = {}
    monkeypatch.setattr(sfs, "aoi_projwin", lambda aoi, dem, snap=None: (0.0, 100.0, 100.0, 0.0))

    def fake_search(aoi, **kw):
        seen.update(kw, aoi=aoi)
        return _obs(("M1LE", "2010-01-01", 0, 10), ("M2RE", "2011-01-01", 0, 50))

    monkeypatch.setattr(sfs, "search_images", fake_search)
    rc = cli.main(
        [
            "sfs",
            "--aoi",
            "aoi.geojson",
            "--dem",
            str(tmp_path / "d.tif"),
            "--work-dir",
            str(tmp_path / "w"),
            "--registration",
            "residuals",
            "--max-slew",
            "3",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "AOI -> projwin 0 100 100 0" in out and "2 NAC images" in out
    assert seen["aoi"] == "aoi.geojson" and seen["max_slew"] == 3 and seen["max_resolution"] == 1.5
    assert "M1LE.IMG" in out and "M2RE.IMG" in out and "-projwin 0 100 100 0" in out


def test_cli_sfs_needs_sources(tmp_path, capsys):
    assert cli.main(["sfs", "--dem", str(tmp_path / "d.tif")]) == 1
    assert "No sources" in capsys.readouterr().err


def _strip_obs(n, slews, az0=0.0):
    """*n* overlapping N-S NAC-like strips over a polar AOI, Sun azimuths spread from *az0*."""
    rows = []
    for k in range(n):
        lon = 10.0 + 0.002 * k
        rows.append(
            {
                "product_id": f"M{1000 + k:09d}LE",
                "instrument": "LROC NAC",
                "center_lat": -86.0,
                "center_lon": lon,
                "min_lat": -86.2,
                "max_lat": -85.8,
                "lon_west": lon - 0.3,
                "lon_east": lon + 0.3,
                "incidence": 88.0,
                "emission": abs(slews[k]),
                "slew_angle": slews[k],
                "sub_solar_azimuth": (az0 + 37.0 * k) % 360.0,
                "pixel_scale": 1.0 + 0.01 * k,
                "start_time": f"2010-01-{1 + k % 28:02d}",
            }
        )
    return normalize_observations(rows)


def test_cap_candidates_keeps_budget_and_nadir_coverage(monkeypatch):
    obs = _strip_obs(12, [0.0] * 10 + [12.0, -12.0])
    aoi = {"lat": (-86.05, -85.95), "lon": (9.9, 10.1)}
    pair = pd.DataFrame({"product_id_1": [obs.product_id[10]], "product_id_2": [obs.product_id[0]]})
    monkeypatch.setattr(sfs, "_candidate_pairs", lambda *a, **k: pair)
    capped = sfs.cap_candidates(obs, aoi, 6)
    ids = list(capped["product_id"])
    assert len(ids) == 6
    assert obs.product_id[10] in ids and obs.product_id[0] in ids  # the stereo pair survives
    assert obs.product_id[11] not in ids  # slewed and not in a pair: only nadir fills the rest
    assert sfs.cap_candidates(obs, aoi, 50) is obs


def test_slewed_images_join_ba_but_not_sfs(tmp_path):
    cfg = SfsConfig(selection="all", max_sfs_slew=5.0)
    p = SfsPipeline(tmp_path / "w", tmp_path / "ldem.tif", cfg)
    p.images = [
        sfs.SfsImage("A", tmp_path / "A.cub", slew_angle=0.4),
        sfs.SfsImage("B", tmp_path / "B.cub", slew_angle=-15.0),
        sfs.SfsImage("C", tmp_path / "C.cub", slew_angle=None),
    ]
    assert p._sfs_eligible() == [0, 2]
    p.config = SfsConfig(selection="all", max_sfs_slew=None)
    assert p._sfs_eligible() == [0, 1, 2]


def test_bundle_adjust_overlap_options_are_exclusive(tmp_path):
    cfg = SfsConfig(registration="residuals", bundle_adjust_options={"auto_overlap_percent": 15})
    _, _, cmds = _run(tmp_path, cfg)
    ba = _cmd(cmds, "parallel_bundle_adjust")
    assert "--auto-overlap-params" in ba and "--overlap-limit" not in ba  # footprint pairs win
    _, _, cmds = _run(tmp_path / "plain", SfsConfig(registration="residuals"))
    ba = _cmd(cmds, "parallel_bundle_adjust")
    assert _opt(ba, "--overlap-limit") == "200" and "--auto-overlap-params" not in ba
