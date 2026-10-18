"""SfS product suite: names, raster math, command sequence and ensemble statistics (offline)."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from lunar_toolkit import sfs_products as sp
from lunar_toolkit.tools import dry_run, find_tool

MOON_SPS = CRS.from_proj4("+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m +no_defs")
RES = 5.0


def _grid(width=40, height=30, res=RES):
    return sp.Grid(MOON_SPS, from_origin(1000.0, 2000.0, res, res), width, height)


def _plane(grid, a=0.1, b=-0.05):
    rows, cols = np.mgrid[0 : grid.height, 0 : grid.width]
    return a * cols * grid.res + b * (-rows * grid.res) + 100.0


def _write(path, array, grid, nodata=np.nan):
    return sp.write_raster(path, np.asarray(array, np.float32), grid, nodata=nodata)


def _images(tmp_path, grid):
    """Three images: two in Sun bin 0 (one shadowed on the left), one in bin 3."""
    lit = np.full((grid.height, grid.width), 0.3)
    half = lit.copy()
    half[:, : grid.width // 2] = 0.0
    paths = [_write(tmp_path / f"im{i}.tif", a, grid) for i, a in enumerate((lit, half, lit))]
    paths[2] = _write(tmp_path / "im2.tif", np.where(np.arange(grid.height)[:, None] < 10, np.nan, lit), grid)
    return [sp.ProductImage(paths[0], 5, 1.0), sp.ProductImage(paths[1], 25, 0.5), (paths[2], 95, 2.0)]


def test_product_name():
    assert sp.product_name("SR01", "ELEV", 5) == "SR01_GLDELEV_5.tif"
    assert sp.product_name("SR01", "HILL", 5, 315, 45) == "SR01_GLDHILL_5_315_45.tif"
    assert sp.product_name("SR01", "TSLP", 0.5, 20) == "SR01_GLDTSLP_0.5_20.tif"


def test_sun_bin_and_baseline_step():
    assert [sp.sun_bin(a) for a in (0, 29.9, 30, 359, -10, 365)] == [0, 0, 1, 11, 11, 0]
    assert sp.baseline_step(None, 5) == 1
    assert sp.baseline_step(5, 5) == 1
    assert sp.baseline_step(20, 5) == 2
    assert sp.baseline_step(100, 5) == 10


def test_slope_of_plane():
    grid = _grid()
    z = _plane(grid, 0.1, -0.05)
    expected = np.degrees(np.arctan(np.hypot(0.1, 0.05)))
    for step in (1, 3):
        s = sp.slope_deg(z, grid.res, step)
        inner = s[step:-step, step:-step]
        assert np.allclose(inner, expected)
        assert np.isnan(s[0]).all() and np.isnan(s[:, -1]).all()


def test_vrm_plane_is_flat_and_rough_is_not():
    grid = _grid()
    z = _plane(grid)
    v = sp.vrm(z, grid.res)
    assert np.allclose(v[2:-2, 2:-2], 0, atol=1e-12)
    rough = z + np.random.default_rng(0).normal(0, 3, z.shape)
    vr = sp.vrm(rough, grid.res, 2)
    assert np.nanmin(vr) > 0 and np.nanmax(vr) < 1
    assert np.isnan(vr[:3]).all()


def test_box_sum_matches_loop():
    a = np.arange(36, dtype=float).reshape(6, 6)
    a[0, 0] = np.nan
    s = sp._box_sum(a, 1)
    assert s[2, 2] == a[1:4, 1:4].sum()
    assert np.isnan(s[1, 1]) and np.isnan(s[0, 3])


def test_illumination_maps(tmp_path):
    grid = _grid()
    maps = sp.illumination_maps(sp.as_product_images(_images(tmp_path, grid)), grid)
    left, right = (5, 0), (5, grid.width - 1)
    low_left, low_right = (20, 0), (20, grid.width - 1)
    assert maps.count[left] == 1 and maps.count[right] == 2
    assert maps.count[low_left] == 2 and maps.count[low_right] == 3
    assert maps.solar_bins[left] == 1 and maps.solar_bins[low_left] == 2 and maps.solar_bins[low_right] == 2
    assert maps.best_resolution is not None
    assert maps.best_resolution[left] == 1.0 and maps.best_resolution[right] == 0.5


def test_illumination_without_resolutions(tmp_path):
    grid = _grid()
    ims = [sp.ProductImage(im.path, im.sun_azimuth) for im in sp.as_product_images(_images(tmp_path, grid))]
    assert sp.illumination_maps(ims, grid).best_resolution is None
    out = sp.illumination_products(ims, grid, tmp_path / "p", region_id="R", resolution=5)
    assert set(out) == {"mask", "solar_bins"}


def test_make_products_dry_run_sequence(tmp_path):
    images = [(tmp_path / "a.map.tif", 10.0), (tmp_path / "b.map.tif", 200.0)]
    with dry_run() as cmds:
        out = sp.make_products(
            tmp_path / "sfs-DEM.tif",
            tmp_path / "ldem.tif",
            tmp_path / "products",
            images,
            region_id="SR01",
            resolution=5,
            height_error=tmp_path / "sfs-height-error.tif",
        )
    assert [c[0] for c in cmds] == [
        "gdal_translate",
        "gdal_translate",
        "geodiff",
        "hillshade",
        "dem_mosaic",
        "gdalwarp",
    ]
    hill = cmds[3]
    assert hill[hill.index("--azimuth") + 1] == "315" and hill[hill.index("--elevation") + 1] == "45"
    assert "--max" in cmds[4] and str(tmp_path / "a.map.tif") in cmds[4]
    warp = cmds[5]
    assert warp[warp.index("-tr") + 1 :][:2] == ["5", "5"]
    assert out["elevation"].name == "SR01_GLDELEV_5.tif"
    assert out["uncertainty"].name == "SR01_GLDISGM_5.tif"
    assert out["diff"].name == "SR01_GLDDIFF_5.tif"
    assert out["hillshade"].name == "SR01_GLDHILL_5_315_45.tif"
    assert out["orthomosaic"].name == "SR01_GLDOMOS_5.tif"
    assert "slope" not in out
    assert list(tmp_path.iterdir()) == []  # a dry run writes nothing


def test_make_products_on_existing_dem(tmp_path):
    grid = _grid()
    dem = _write(tmp_path / "sfs-DEM.tif", _plane(grid), grid)
    with dry_run() as cmds:
        out = sp.make_products(
            dem, tmp_path / "ldem.tif", tmp_path / "p", _images(tmp_path, grid), region_id="R", resolution=5
        )
    assert "uncertainty" not in out and "slope" not in out
    warp = next(c for c in cmds if c[0] == "gdalwarp")
    assert warp[warp.index("-ts") + 1 :][:2] == [str(grid.width), str(grid.height)]
    assert [float(v) for v in warp[warp.index("-te") + 1 :][:4]] == [1000, 1850, 1200, 2000]


def test_raster_products(tmp_path):
    grid = _grid()
    dem = _write(tmp_path / "sfs-DEM.tif", _plane(grid), grid)
    out = sp.terrain_products(dem, tmp_path / "p", region_id="R", resolution=5, baselines=(None, 20))
    images = sp.as_product_images(_images(tmp_path, grid))
    out.update(sp.illumination_products(images, grid, tmp_path / "p", region_id="R", resolution=5))
    for key, name in [
        ("slope", "R_GLDTSLP_5_5.tif"),
        ("roughness", "R_GLDTRGH_5_5.tif"),
        ("slope_20", "R_GLDTSLP_5_20.tif"),
        ("roughness_20", "R_GLDTRGH_5_20.tif"),
        ("mask", "R_GLDMASK_5.tif"),
        ("solar_bins", "R_GLDSBCT_5.tif"),
        ("best_resolution", "R_GLDBRES_5.tif"),
    ]:
        assert out[key].name == name and out[key].exists()
    slope, g = sp.read_raster(out["slope"])
    assert (g.transform, g.width, g.height) == (grid.transform, grid.width, grid.height)
    assert np.nanmax(slope) == pytest.approx(np.degrees(np.arctan(np.hypot(0.1, 0.05))), rel=1e-5)
    assert sp.read_raster(out["mask"])[0].max() == 3
    stamp = out["slope"].stat().st_mtime_ns
    sp.terrain_products(dem, tmp_path / "p", region_id="R", resolution=5)
    assert out["slope"].stat().st_mtime_ns == stamp


def test_products_are_reused(tmp_path):
    grid = _grid()
    dem = _write(tmp_path / "sfs-DEM.tif", _plane(grid), grid)
    names = ("ELEV", "DIFF", "HILL", "OMOS")
    for code in names:
        suffix = (315, 45) if code == "HILL" else ()
        (tmp_path / "p").mkdir(exist_ok=True)
        (tmp_path / "p" / sp.product_name("R", code, 5, *suffix)).touch()
    with dry_run() as cmds:
        sp.make_products(dem, dem, tmp_path / "p", [(dem, 0.0)], region_id="R", resolution=5)
    assert cmds == []


def test_ensemble_subsets_overlap():
    pool = list(range(20))
    subsets = sp.ensemble_subsets(pool, 5, 8, seed=1)
    assert len(subsets) == 5
    assert all(len(s) == 8 and s == sorted(s) for s in subsets)
    for i, a in enumerate(subsets):
        for b in subsets[i + 1 :]:
            assert len(set(a) & set(b)) <= 3
    assert sp.ensemble_subsets(pool, 5, 8, seed=1) == subsets
    assert len(sp.ensemble_subsets(list(range(5)), 4, 5, max_tries=50)) == 1
    with pytest.raises(ValueError):
        sp.ensemble_subsets(pool, 2, 21)


def test_ensemble_stats_and_files(tmp_path):
    grid = _grid()
    z = _plane(grid)
    members = [_write(tmp_path / f"m{i}.tif", z + i, grid) for i in range(3)]
    tilted = z + 0.05 * np.arange(grid.width) * grid.res
    members.append(
        _write(tmp_path / "m3.tif", np.where(np.arange(grid.height)[:, None] < 5, np.nan, z), grid)
    )
    stats = sp.ensemble_stats(members[:3])
    assert np.allclose(stats.elevation_std, 1.0)
    assert np.allclose(stats.slope_std[1:-1, 1:-1], 0, atol=1e-6)
    stats = sp.ensemble_stats([members[0], members[3]])
    assert np.isnan(stats.elevation_std[0, 0]) and stats.elevation_std[10, 10] == 0
    assert stats.count[0, 0] == 1 and stats.count[10, 10] == 2
    _write(tmp_path / "t.tif", tilted, grid)
    stats = sp.ensemble_stats([members[0], tmp_path / "t.tif"])
    assert np.nanmean(stats.slope_std) > 0
    out = sp.ensemble_uncertainty(members[:3], tmp_path / "ens", region_id="R", resolution=5)
    assert out["elevation_std"].name == "R_GLDISGM_5_ENS.tif"
    assert out["slope_std"].name == "R_GLDSSGM_5_ENS.tif"
    assert np.allclose(sp.read_raster(out["elevation_std"])[0], 1.0)
    with pytest.raises(ValueError):
        sp.ensemble_stats(members[:1])


def test_ensemble_on_reference_grid(tmp_path):
    fine = _grid(40, 30, 5.0)
    coarse = _grid(20, 15, 10.0)
    members = [_write(tmp_path / f"m{i}.tif", _plane(fine) + 2 * i, fine) for i in range(2)]
    ref = _write(tmp_path / "ref.tif", np.zeros((15, 20)), coarse)
    stats = sp.ensemble_stats(members, sp.Grid.from_raster(ref))
    assert stats.elevation_std.shape == (15, 20)
    assert np.allclose(stats.elevation_std, np.sqrt(2))


def test_compare_to_points(tmp_path):
    prefix = tmp_path / "val" / "lola"
    with dry_run() as cmds:
        empty = sp.compare_to_points(tmp_path / "dem.tif", tmp_path / "lola.csv", prefix)
    assert cmds[0][0] == "geodiff"
    assert cmds[0][cmds[0].index("--csv-format") + 1] == "1:lon 2:lat 3:height_above_datum"
    assert empty.stats() == {"count": 0}
    prefix.parent.mkdir()
    (tmp_path / "val" / "lola-diff.csv").write_text(
        "# longitude,latitude, height diff (m)\n# Max difference: 2 meters\n"
        "15.3,-9.6,1.0\n15.4,-9.6,-1.0\n15.5,-9.7,2.0\n"
    )
    pc = sp.compare_to_points(tmp_path / "dem.tif", tmp_path / "lola.csv", prefix)
    st = pc.stats()
    assert st["count"] == 3 and st["min"] == -1 and st["max"] == 2 and st["median"] == 1
    assert st["rms"] == pytest.approx(np.sqrt(2))
    assert pc.lon.tolist() == [15.3, 15.4, 15.5]


@pytest.mark.tools
@pytest.mark.skipif(find_tool("dem_mosaic") is None, reason="ASP tools not found")
def test_make_products_with_tools(tmp_path):
    grid = _grid()
    z = _plane(grid) + np.random.default_rng(0).normal(0, 0.2, (grid.height, grid.width))
    dem = _write(tmp_path / "sfs-DEM.tif", z, grid, nodata=-1e6)
    prior = _write(tmp_path / "prior.tif", _plane(grid), grid, nodata=-1e6)
    out = sp.make_products(dem, prior, tmp_path / "p", _images(tmp_path, grid), region_id="R", resolution=5)
    for key in ("elevation", "diff", "hillshade", "orthomosaic", "slope", "roughness", "mask", "solar_bins"):
        assert out[key].exists(), key
    og = sp.Grid.from_raster(out["orthomosaic"])
    assert (og.transform, og.width, og.height) == (grid.transform, grid.width, grid.height)
    diff = sp.read_raster(out["diff"])[0]
    assert abs(np.nanmean(diff)) < 0.1
    ortho = sp.read_raster(out["orthomosaic"])[0]
    assert np.nanmax(ortho) == pytest.approx(0.3, rel=1e-5)
