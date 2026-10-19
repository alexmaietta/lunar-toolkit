"""Tool runner, option builders and ASP / ISIS wrappers (offline)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lunar_toolkit import asp, isis
from lunar_toolkit.tools import (
    ToolError,
    asp_options,
    dry_run,
    find_tool,
    gdal_options,
    isis_params,
    run,
    tool_env,
)


def test_asp_options():
    args = asp_options(
        {
            "tr": 5.0,
            "save_sparingly": True,
            "float_albedo": False,
            "nodata_value": None,
            "t_projwin": (0, 10, 10.5, 0),
            "t_srs": "+proj=stere",
            "o": "run/run",
            "remove_outliers_params": "75 3 100 100",
        }
    )
    assert args == [
        "--tr",
        "5",
        "--save-sparingly",
        "--t_projwin",
        "0",
        "10",
        "10.5",
        "0",
        "--t_srs",
        "+proj=stere",
        "-o",
        "run/run",
        "--remove-outliers-params",
        "75 3 100 100",
    ]


def test_gdal_and_isis_options():
    assert gdal_options({"tr": (5, 5), "a_nodata": 0, "scale": True, "r": None}) == [
        "-tr",
        "5",
        "5",
        "-a_nodata",
        "0",
        "-scale",
    ]
    assert isis_params({"from_": "a.cub", "web": True, "spksmithed": False, "x": None}) == [
        "from=a.cub",
        "web=true",
        "spksmithed=false",
    ]


def test_dry_run_records():
    with dry_run() as cmds:
        isis.spiceinit(Path("a.cub"))
        asp.hillshade(Path("dem.tif"), Path("hs.tif"))
    assert cmds == [
        ["spiceinit", "from=a.cub", "spksmithed=true", "web=true"],
        ["hillshade", "dem.tif", "--azimuth", "300", "--elevation", "20", "-o", "hs.tif"],
    ]


def test_missing_tool(monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("ASP_ROOT", raising=False)
    monkeypatch.delenv("ISISROOT", raising=False)
    assert find_tool("definitely_not_a_tool") is None
    with pytest.raises(ToolError, match="not found"):
        run("definitely_not_a_tool")


def _fake_env(tmp_path: Path) -> Path:
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "share" / "proj").mkdir(parents=True)
    (prefix / "share" / "proj" / "proj.db").touch()
    (prefix / "IsisPreferences").touch()
    tool = prefix / "bin" / "faketool"
    tool.write_text('#!/bin/sh\necho "$ISISROOT $1"\nexit ${2:-0}\n')
    tool.chmod(0o755)
    return prefix


def test_tool_env_defaults(tmp_path, monkeypatch):
    prefix = _fake_env(tmp_path)
    for var in ("ISISROOT", "ISISDATA", "PROJ_DATA", "GDAL_DATA"):
        monkeypatch.delenv(var, raising=False)
    env = tool_env(prefix / "bin" / "faketool")
    assert env["ISISROOT"] == str(prefix)
    assert env["PROJ_DATA"] == str(prefix / "share" / "proj")
    assert "GDAL_DATA" not in env
    assert env["PATH"].split(os.pathsep)[0] == str(prefix / "bin")


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell script")
def test_run_real_process(tmp_path, monkeypatch):
    prefix = _fake_env(tmp_path)
    monkeypatch.setenv("ASP_ROOT", str(prefix))
    monkeypatch.delenv("ISISROOT", raising=False)
    res = run("faketool", ["hello"], capture=True)
    assert res.stdout.strip() == f"{prefix} hello"
    with pytest.raises(ToolError) as exc:
        run("faketool", ["x", "3"], capture=True)
    assert exc.value.returncode == 3


QUERY_OUT = """\
Sun position for: /w/images/A.cal.echo.cub is Vector3(1.4e11,-2.0e10,3.1e9)
Sun azimuth and elevation for: /w/images/A.cal.echo.cub are 172.49712345678901 and 1.25 degrees.
Sun azimuth and elevation for: /w/images/B.cal.echo.cub are 73.1771 and -0.5 degrees.
"""


def test_parse_sun_angles():
    angles = asp.parse_sun_angles(QUERY_OUT)
    assert [(Path(a.image).name, a.azimuth, a.elevation) for a in angles] == [
        ("A.cal.echo.cub", 172.49712345678901, 1.25),
        ("B.cal.echo.cub", 73.1771, -0.5),
    ]


def test_sun_angles_runs_query(tmp_path, monkeypatch):
    seen = {}

    def fake_run(tool, args, **kw):
        seen["cmd"] = [tool, *map(str, args)]
        return subprocess.CompletedProcess(args, 0, QUERY_OUT, "")

    monkeypatch.setattr(asp, "run", fake_run)
    angles = asp.sun_angles(
        Path("d.tif"), [Path("A.cub")], [Path("A.json")], out_prefix=tmp_path / "q" / "run"
    )
    assert seen["cmd"][:5] == ["sfs", "-i", "d.tif", "A.cub", "A.json"]
    assert "--query" in seen["cmd"]
    assert len(angles) == 2


def test_bundle_adjust_mapprojected_data(tmp_path):
    with dry_run() as cmds:
        asp.bundle_adjust(
            [Path("a.cub"), Path("b.cub")],
            tmp_path / "ba" / "run",
            mapprojected=[Path("a.tif"), Path("b.tif")],
            dem=Path("dem.tif"),
            parallel=True,
            min_matches=1,
        )
    assert cmds[0][0] == "parallel_bundle_adjust"
    i = cmds[0].index("--mapprojected-data")
    assert cmds[0][i + 1] == "a.tif b.tif dem.tif"
    assert cmds[0][-2:] == ["-o", str(tmp_path / "ba" / "run")]
    assert not (tmp_path / "ba").exists()


def test_bundle_adjust_fails_without_adjustments(tmp_path, monkeypatch):
    prefix = tmp_path / "ba" / "run"
    stale = asp.bundle_adjust_file_name(prefix, Path("b.cub"), Path("b.json"))
    written = []

    def fake_run(tool, args, **kw):
        for name in written:
            asp.bundle_adjust_file_name(prefix, Path(f"{name}.cub"), Path(f"{name}.json")).write_text("")
        return subprocess.CompletedProcess(args, 0, "No points to optimize. Cannot continue.", "")

    monkeypatch.setattr(asp, "run", fake_run)
    images, cams = [Path("a.cub"), Path("b.cub")], [Path("a.json"), Path("b.json")]
    prefix.parent.mkdir(parents=True)
    stale.write_text("")
    os.utime(stale, (0, 0))
    with pytest.raises(ToolError, match="no adjustment for 2 of 2 images"):
        asp.bundle_adjust(images, prefix, cams)
    written[:] = ["a", "b"]
    assert asp.bundle_adjust(images, prefix, cams) == prefix


def test_point2dem_output_name():
    with dry_run():
        assert asp.point2dem(Path("st/run-PC.tif")) == Path("st/run-DEM.tif")
        assert asp.point2dem(Path("al/run-trans_source.tif")) == Path("al/run-trans_source-DEM.tif")
        assert asp.point2dem(Path("x.tif"), o="out/run") == Path("out/run-DEM.tif")


def test_calibrate_lronac_sequence(tmp_path):
    with dry_run() as cmds:
        out = isis.calibrate_lronac(tmp_path / "M1LE.IMG", tmp_path / "w")
    assert out == tmp_path / "w" / "M1LE.cal.echo.cub"
    assert [c[0] for c in cmds] == ["lronac2isis", "spiceinit", "lronaccal", "lronacecho"]
    out.parent.mkdir(exist_ok=True)
    out.touch()
    with dry_run() as cmds:
        isis.calibrate_lronac(tmp_path / "M1LE.IMG", tmp_path / "w")
    assert cmds == []


def test_prepare_lronac(tmp_path):
    with dry_run() as cmds:
        cube, cam = isis.prepare_lronac(tmp_path / "M1LE.IMG", tmp_path / "w")
        assert isis.prepare_lronac(tmp_path / "M2LE.cub", tmp_path / "w", csm=False) == (
            tmp_path / "M2LE.cub",
            None,
        )
    assert (cube, cam) == (tmp_path / "w" / "M1LE.cal.echo.cub", tmp_path / "w" / "M1LE.json")
    assert [c[0] for c in cmds] == ["lronac2isis", "spiceinit", "lronaccal", "lronacecho", "isd_generate"]
    assert not (tmp_path / "w").exists()


def test_bundle_adjust_file_name():
    assert asp.bundle_adjust_file_name(Path("ba/run"), Path("x/A.cal.echo.cub")) == Path(
        "ba/run-A.cal.echo.adjust"
    )
    assert asp.bundle_adjust_file_name(Path("ba/run"), Path("A.cub"), Path("i/A.json")) == Path(
        "ba/run-A.adjust"
    )


def test_read_bundle_adjust_reports(tmp_path):
    (tmp_path / "off.txt").write_text(
        "# Percentiles of distances between mapprojected matching pixels in an image and the others.\n"
        "# image_name 25% 50% 75% 85% 95% count\n"
        "/w/images/A.cal.echo.cub 0.2 0.4 0.9 1.2 2.5 5120\n"
        "/w/images/B.cal.echo.cub -1 -1 -1 -1 -1 0\n"
    )
    off = asp.read_mapproj_offset_stats(tmp_path / "off.txt")
    assert off["A.cal.echo.cub"].p85 == 1.2 and off["A.cal.echo.cub"].count == 5120
    assert off["B.cal.echo.cub"].count == 0
    (tmp_path / "stats.txt").write_text(
        "# Pixel reprojection error per camera\n# Image, mean, median, count\n"
        "/w/images/A.cal.echo.cub, 0.41, 0.33, 812\n/w/images/B.cal.echo.cub, nan, nan, 0\n"
    )
    st = asp.read_residual_stats(tmp_path / "stats.txt")
    assert st["A.cal.echo.cub"].median == 0.33 and st["B.cal.echo.cub"].count == 0
    (tmp_path / "raw.txt").write_text("/w/A.cub, 2\n3, 4\n0.1, 0\n/w/B.cub, 0\n/w/C.cub, 1\n-1, 0\n")
    norms = asp.read_residual_norms(tmp_path / "raw.txt")
    assert list(norms["A.cub"]) == [5.0, 0.1] and len(norms["B.cub"]) == 0 and list(norms["C.cub"]) == [1.0]


def test_parse_aaigrid():
    text = (
        "ncols        3\nnrows        2\nxllcorner    100.0\nyllcorner    -20.0\n"
        "dx           5.0\ndy           10.0\nNODATA_value -9999\n"
        " 1 2 3\n 4 -9999 6\n"
        'PROJCS["Stereographic_MOON"]\n'
    )
    r = asp.parse_aaigrid(text)
    assert r.data.shape == (2, 3) and np.isnan(r.data[1, 1]) and r.data[0, 2] == 3
    assert (r.x0, r.y0, r.dx, r.dy) == (100.0, 0.0, 5.0, -10.0)


def test_read_raster_command(tmp_path):
    with dry_run() as cmds:
        assert asp.read_raster(tmp_path / "a.tif", projwin=(0, 10, 10, 0), outsize=(4, 4)) is None
        assert asp.raster_bounds(tmp_path / "a.tif") is None
    assert cmds[0][:4] == ["gdal_translate", "-q", "-of", "AAIGrid"]
    assert cmds[0][-2:] == [str(tmp_path / "a.tif"), "/vsistdout/"]
    assert "-outsize" in cmds[0] and "average" in cmds[0]
    assert cmds[1][:2] == ["gdalinfo", "-json"]


def test_image_subset_parses_output(tmp_path, monkeypatch):
    out = tmp_path / "sel" / "subset.txt"

    def fake_run(tool, args, **kw):
        assert tool == "image_subset"
        assert args[args.index("--threshold") + 1] == 0.002
        out.write_text("/m/b.tif 900\n/m/a.tif 12\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(asp, "run", fake_run)
    got = asp.image_subset([Path("/m/a.tif"), Path("/m/b.tif")], out, threshold=0.002)
    assert got == [("/m/b.tif", 900), ("/m/a.tif", 12)]
    assert (tmp_path / "sel" / "subset.txt.in.txt").read_text() == f"{Path('/m/a.tif')}\n{Path('/m/b.tif')}\n"


def test_spiceinit_falls_back_to_ellipsoid(monkeypatch):
    monkeypatch.setattr(isis, "system_shape_available", lambda: False)
    with dry_run() as cmds:
        isis.spiceinit(Path("a.cub"))
        isis.spiceinit(Path("b.cub"), shape="system")
        isis.spiceinit(Path("c.cub"), shape="user", model="dem.cub")
    assert "shape=ellipsoid" in cmds[0] and "shape=ellipsoid" in cmds[1]
    assert "shape=user" in cmds[2]
    monkeypatch.setattr(isis, "system_shape_available", lambda: True)
    with dry_run() as cmds:
        isis.spiceinit(Path("d.cub"))
    assert not any(a.startswith("shape=") for a in cmds[0])


def test_lronaccal_falls_back_to_nearest_dark(tmp_path, monkeypatch):
    calls = []

    def fake_run(tool, args, **kw):
        calls.append([tool, *map(str, args)])
        if tool == "lronaccal" and not any(a.startswith("darkfiletype=") for a in map(str, args)):
            raise isis.ToolError("lronaccal failed (exit 2).")

    monkeypatch.setattr(isis, "run", fake_run)
    monkeypatch.setattr(isis, "system_shape_available", lambda: True)
    isis.calibrate_lronac(tmp_path / "M1LE.IMG", tmp_path / "w")
    cal = [c for c in calls if c[0] == "lronaccal"]
    assert len(cal) == 2 and "darkfiletype=NEAREST" in cal[1]
    assert [c[0] for c in calls][-1] == "lronacecho"
