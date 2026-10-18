"""ISIS / ASP wrappers run for real on the ASP LRONAC example (pytest -m tools)."""

import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lunar_toolkit import asp, isis
from lunar_toolkit.tools import find_tool, run, tool_env

pytestmark = pytest.mark.tools("gdalinfo")


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    """ISIS writes print.prt and ASP its run files to the working directory."""
    monkeypatch.chdir(tmp_path)


EXAMPLE = Path(os.environ.get("LUNAR_ASP_EXAMPLE", "LRONAC_example"))
LEFT, RIGHT = "M181058717LE", "M181073012LE"
MOON_RADIUS_M = 1737400.0
DEM_CENTRE = (-9.6148, 15.3134, 259.0)  # lat, lon, height of run/run-DEM.tif
CROP = (-200, 200, 200, -200)  # gdal projwin (ulx uly lrx lry) around the DEM centre, metres


def _info(path: Path) -> dict:
    return json.loads(run("gdalinfo", ["-json", "-stats", path], capture=True).stdout)


def _band(path: Path) -> dict:
    return _info(path)["bands"][0]


def _probe(tool: str, args: list, what: str) -> None:
    """Skip unless *tool* exits cleanly (it segfaults on a broken camera stack)."""
    path = find_tool(tool)
    if path is None:
        pytest.skip(f"{tool} not found")
    try:
        res = subprocess.run(
            [str(path), *map(str, args)],
            env=tool_env(path),
            capture_output=True,
            text=True,
            timeout=600,
            cwd=tempfile.mkdtemp(prefix="lunar-probe-"),
        )
    except subprocess.TimeoutExpired:
        pytest.skip(f"{tool} timed out loading {what}")
    if res.returncode != 0:
        rc = res.returncode
        status = f"killed by {signal.Signals(-rc).name}" if rc < 0 else f"exit {rc}"
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-1:]
        pytest.skip(f"cannot load {what}: {tool} {status}{': ' + tail[0] if tail else ''}")


@pytest.fixture(scope="session")
def example() -> Path:
    if not (EXAMPLE / "run" / "run-DEM.tif").is_file():
        pytest.skip(f"ASP LRONAC example not found at {EXAMPLE} (set LUNAR_ASP_EXAMPLE)")
    return EXAMPLE


@pytest.fixture(scope="session")
def dem(example) -> Path:
    return example / "run" / "run-DEM.tif"


@pytest.fixture(scope="session")
def images(example) -> list[Path]:
    return [example / f"{LEFT}_crop.cub", example / f"{RIGHT}_crop.cub"]


@pytest.fixture(scope="session")
def cameras(example) -> list[Path]:
    return [example / f"{LEFT}.json", example / f"{RIGHT}.json"]


@pytest.fixture(scope="session")
def dem_up5(dem, tmp_path_factory) -> Path:
    """The example DEM raised by 5 m."""
    if find_tool("image_calc") is None:
        pytest.skip("image_calc not found")
    out = tmp_path_factory.mktemp("up5") / "up5.tif"
    run("image_calc", ["-c", "var_0 + 5", dem, "-o", out], capture=True)
    return out


@pytest.fixture(scope="session")
def csm_cameras(images, cameras) -> list[Path]:
    """The example CSM cameras, or skip with the reason ASP cannot load them."""
    _probe(
        "cam_test",
        ["--image", images[0], "--cam1", cameras[0], "--cam2", cameras[0]]
        + ["--session1", "csm", "--session2", "csm", "--sample-rate", 1000],
        f"CSM camera {cameras[0].name}",
    )
    return cameras


@pytest.fixture(scope="session")
def isis_camera(images) -> Path:
    """A cube whose ISIS camera loads, or skip with the reason it does not."""
    _probe("campt", [f"from={images[0]}"], f"ISIS camera of {images[0].name}")
    return images[0]


@pytest.mark.tools("gdal_translate")
def test_gdal_translate_crop_and_resample(dem, tmp_path):
    crop = asp.gdal_translate(dem, tmp_path / "crop.tif", projwin=CROP)
    assert _info(crop)["size"] == [95, 95]
    coarse = asp.gdal_translate(dem, tmp_path / "sub" / "r10.tif", tr=10, r="average")
    info = _info(coarse)
    assert info["geoTransform"][1] == pytest.approx(10)
    assert info["size"] == [97, 96]


@pytest.mark.tools("gdal_translate", "gdalsrsinfo")
def test_raster_helpers(dem):
    assert asp.raster_bounds(dem) == pytest.approx((-482.154, 469.410, 486.402, -494.898), abs=1e-3)
    assert "+proj=stere" in asp.srs_proj4(dem)
    grid = asp.read_raster(dem, projwin=CROP, outsize=(10, 10))
    assert grid is not None and grid.data.shape == (10, 10) and grid.dx == pytest.approx(40)
    assert 200 < np.nanmean(grid.data) < 320


@pytest.mark.tools("dem_mosaic", "geodiff", "image_calc")
def test_dem_mosaic_max(dem, dem_up5, tmp_path):
    mosaic = asp.dem_mosaic([dem, dem_up5], tmp_path / "mos.tif", max=True)
    diff = _band(asp.geodiff(mosaic, dem_up5, tmp_path / "gd" / "run"))
    assert abs(diff["minimum"]) < 1e-3 and abs(diff["maximum"]) < 1e-3


@pytest.mark.tools("geodiff", "image_calc")
def test_geodiff(dem, dem_up5, tmp_path):
    out = asp.geodiff(dem_up5, dem, tmp_path / "gd" / "run")
    assert out == tmp_path / "gd" / "run-diff.tif"
    band = _band(out)
    assert band["minimum"] == pytest.approx(5, abs=1e-3)
    assert band["maximum"] == pytest.approx(5, abs=1e-3)


@pytest.mark.tools("hillshade")
def test_hillshade(dem, tmp_path):
    out = asp.hillshade(dem, tmp_path / "hs.tif", azimuth=300, elevation=20)
    assert _info(out)["size"] == _info(asp.gdal_translate(dem, tmp_path / "copy.tif"))["size"]
    assert _band(out)["maximum"] > _band(out)["minimum"]


@pytest.mark.tools("point2dem")
def test_point2dem(example, tmp_path):
    out = asp.point2dem(example / "run" / "run-PC.tif", o=tmp_path / "p2d" / "run", tr=8)
    assert out == tmp_path / "p2d" / "run-DEM.tif"
    info = _info(out)
    assert info["geoTransform"][1] == pytest.approx(8)
    assert 200 < info["bands"][0]["mean"] < 320


@pytest.mark.tools("pc_align", "image_calc")
def test_pc_align_recovers_vertical_shift(dem, dem_up5, tmp_path):
    transform = asp.pc_align(dem, dem_up5, tmp_path / "align" / "run", max_displacement=20)
    t = np.loadtxt(transform)
    lat, lon, h = np.radians(DEM_CENTRE[0]), np.radians(DEM_CENTRE[1]), DEM_CENTRE[2]
    p = (MOON_RADIUS_M + h) * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    shift = t[:3, :3] @ p + t[:3, 3] - p
    assert np.linalg.norm(shift) == pytest.approx(5, abs=0.1)
    assert shift @ p / np.linalg.norm(p) == pytest.approx(-5, abs=0.1)  # moved down


@pytest.mark.tools("reduce", "catlab")
def test_isis_reduce_and_catlab(images, tmp_path):
    small = isis.reduce(images[0], tmp_path / "small.cub", scale=8)
    isis.isis("catlab", from_=small, to=tmp_path / "small.pvl")
    label = (tmp_path / "small.pvl").read_text()
    assert "Group = AlphaCube" in label and "NACL" in label


@pytest.mark.tools("isd_generate")
def test_isd_generate(images, tmp_path):
    out = isis.isd_generate(images[1], tmp_path / "right.json")
    isd = json.loads(out.read_text())
    assert isd["name_model"] == "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL"
    assert isd["name_sensor"].endswith("NARROW ANGLE CAMERA LEFT")


@pytest.mark.tools("isd_generate")
def test_prepare_lronac_from_cube(images, tmp_path):
    cube, camera = isis.prepare_lronac(images[0], tmp_path)
    assert cube == images[0] and camera == tmp_path / f"{images[0].stem}.json"
    assert camera is not None and json.loads(camera.read_text())["image_lines"] > 0
    assert isis.prepare_lronac(images[0], tmp_path) == (cube, camera)


@pytest.mark.tools("sfs")
def test_sun_angles(dem, images, csm_cameras, tmp_path):
    angles = asp.sun_angles(dem, images, csm_cameras, out_prefix=tmp_path / "q" / "run")
    assert [Path(a.image).name for a in angles] == [p.name for p in images]
    assert all(0 < a.elevation < 90 and -360 <= a.azimuth <= 360 for a in angles)


@pytest.mark.tools("mapproject")
def test_mapproject(dem, images, csm_cameras, tmp_path):
    out = asp.mapproject(
        dem, images[0], tmp_path / "mp" / "left.tif", csm_cameras[0], tr=4, t_projwin=(-200, -200, 200, 200)
    )
    info = _info(out)
    assert info["size"] == [101, 101]  # t_projwin bounds are pixel centres
    assert info["bands"][0]["maximum"] > info["bands"][0]["minimum"]


@pytest.mark.tools("bundle_adjust")
def test_bundle_adjust(images, csm_cameras, tmp_path):
    prefix = asp.bundle_adjust(
        images, tmp_path / "ba" / "run", csm_cameras, ip_per_image=1000, num_iterations=5
    )
    for image, camera in zip(images, csm_cameras):
        assert asp.bundle_adjust_file_name(prefix, image, camera).is_file()
    stats = asp.read_residual_stats(Path(f"{prefix}-final_residuals_stats.txt"))
    assert set(stats) == {p.name for p in images}
    assert all(s.count > 0 and s.median < 2 for s in stats.values())


@pytest.mark.tools("sfs", "gdal_translate")
def test_sfs(dem, images, csm_cameras, tmp_path):
    tiny = asp.gdal_translate(dem, tmp_path / "tiny.tif", projwin=(-50, 50, 50, -50))
    out = asp.sfs(
        tiny,
        images[:1],
        tmp_path / "sfs" / "run",
        csm_cameras[:1],
        max_iterations=1,
        smoothness_weight=0.08,
        crop_input_images=True,
    )
    assert out == tmp_path / "sfs" / "run-DEM-final.tif"
    assert _info(out)["size"] == _info(tiny)["size"]


@pytest.mark.tools("reduce", "mosrange", "cam2map")
def test_isis_map_project(isis_camera, tmp_path):
    small = isis.reduce(isis_camera, tmp_path / "small.cub", scale=8)
    out = isis.map_project(small, tmp_path / "small.map.cub")
    isis.isis("catlab", from_=out, to=tmp_path / "map.pvl")
    assert "Group = Mapping" in (tmp_path / "map.pvl").read_text()
    assert not (tmp_path / "small.lis").exists()
