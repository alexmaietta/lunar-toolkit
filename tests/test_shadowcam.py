"""ShadowCam: ODE file selection and the SPICE / CSM command chain (offline)."""

import argparse
import json

import pytest
from conftest import FIXTURES

from lunar_toolkit.ode import extract_products, parse_product, to_dataframe
from lunar_toolkit.shadowcam.sc import (
    ScProcessor,
    ScProduct,
    download_product,
    has_spice,
    is_sc_product_id,
    product_id_for,
    sc_products,
    select_sc_files,
)
from lunar_toolkit.tools import dry_run

PID = "M013178466SE"
BASE = "https://pds.shadowcam.im-ldi.com/observation/2023/004/M013178466S/"
LABEL = """Object = IsisCube
  Group = Kernels
    NaifFrameCode = -155151
{pointing}  End_Group
End_Object
{history}End
"""
HISTORY = "Object = History\n  Name = IsisCube\n  StartByte = {start:6d}\n  Bytes = {size}\nEnd_Object\n"
PAD = b"\0" * 64


def _df(name):
    res = json.loads((FIXTURES / name).read_text())["ODEResults"]
    return to_dataframe([parse_product(p, "ShadowCam") for p in extract_products(res)])


@pytest.fixture
def cube(tmp_path):
    c = tmp_path / f"{PID}.cub"
    c.write_bytes(LABEL.format(pointing="", history="").encode() + PAD)
    return c


def _spiceinit(cube, program=b"spiceinit"):
    """Label with SPICE tables and a History blob recording *program*."""
    blob = b"Object = " + program + b"\nEnd_Object\n"
    pointing = "    InstrumentPointing = Table\n"
    size = len(LABEL.format(pointing=pointing, history=HISTORY.format(start=0, size=len(blob))))
    history = HISTORY.format(start=size + len(PAD) + 1, size=len(blob))
    cube.write_bytes(LABEL.format(pointing=pointing, history=history).encode() + PAD + blob)


def test_product_ids():
    assert is_sc_product_id("M013178466S")
    assert is_sc_product_id("m013178466se") and is_sc_product_id("M013178466SC")
    assert not is_sc_product_id("M1096757719LE")
    assert not is_sc_product_id("TC1W2B0_01_02936N034E0938")
    assert product_id_for("M013178466S") == "M013178466SE"
    assert product_id_for("m013178466s", "cal") == "M013178466SC"
    assert product_id_for("M013178466SC", "raw") == "M013178466SC"
    with pytest.raises(ValueError):
        product_id_for("M013178466S", "cmos")
    with pytest.raises(ValueError):
        product_id_for("M1096757719LE")


def test_sc_products_cal():
    products = sc_products(_df("sc_cal.json"))
    assert [p.product_id for p in products] == ["M013142801SC", "M013164210SC", "M013178466SC"]
    p = products[2]
    assert p.cube_url == BASE + "M013178466SC.cub"  # not the _map_raw / _cog TIFs
    assert p.label_url == BASE + "M013178466SC.xml"
    assert p.cube_kbytes == 70656
    assert p.is_calibrated and p.observation_id == "M013178466S"


def test_sc_products_raw():
    (p,) = sc_products(_df("sc_raw.json"))
    assert p.product_id == PID and not p.is_calibrated
    assert p.cube_url == BASE + "M013178466SE.cub"
    assert p.cube_kbytes == 18432


def test_select_sc_files_rules():
    files = [
        {"FileName": "X_MAP_RAW.TIF", "Type": "Referenced", "URL": BASE + "x_map_raw.tif"},
        {"FileName": "X_SMEAR_FULL.CUB", "Type": "Referenced", "URL": BASE + "x_smear_full.cub"},
        {"FileName": "X.XML", "Type": "Product", "URL": BASE + "M013178466SC.xml"},
    ]
    assert select_sc_files(files) is None  # no product cube
    files.append({"FileName": "X.CUB", "Type": "Product", "URL": BASE + "M013178466SC.cub", "KBytes": "?"})
    p = select_sc_files(files)
    assert p.product_id == "M013178466SC" and p.cube_kbytes is None
    assert select_sc_files([]) is None


def test_sc_products_skips_mosaics():
    df = _df("sc_cal.json")
    df.loc[0, "product_id"] = "SC_CMOS_SP_0001"
    assert [p.product_id for p in sc_products(df)] == ["M013164210SC", "M013178466SC"]


def test_has_spice(cube, tmp_path):
    assert not has_spice(cube)
    _spiceinit(cube, b"crop")  # SPICE tables left by a failed spiceinit
    assert not has_spice(cube)
    _spiceinit(cube)
    assert has_spice(cube)
    assert not has_spice(tmp_path / "missing.cub")


def test_command_chain(cube, tmp_path):
    out = tmp_path / "out"
    with dry_run() as cmds:
        res = ScProcessor(out, validate=True).process_cube(cube)
    assert [c[0] for c in cmds] == ["spiceinit", "isd_generate", "cam_test"]
    spice, isd, cam = cmds
    assert {f"from={cube}", "web=false", "shape=system"} <= set(spice)
    assert isd == ["isd_generate", "-k", str(cube), "-o", str(out / f"{PID}.json"), str(cube)]
    assert cam[cam.index("--cam2") + 1] == str(out / f"{PID}.json")
    assert {"isis", "csm", "50"} <= set(cam)
    assert res.cube == cube and res.camera == out / f"{PID}.json"
    assert res.cam_test == out / f"{PID}.cam_test.txt" and res.geotiff is None
    assert not out.exists()  # nothing written in a dry run


def test_mapproject_variants(cube, tmp_path):
    with dry_run() as cmds:
        res = ScProcessor(tmp_path, mapproject=True, resolution=1.7).process_cube(cube)
    assert [c[0] for c in cmds] == ["spiceinit", "isd_generate", "mosrange", "cam2map", "gdal_translate"]
    assert {"pixres=mpp", "resolution=1.7"} <= set(cmds[3])
    assert cmds[4][-1] == str(tmp_path / f"{PID}.tif") == str(res.geotiff)

    dem = tmp_path / "ref.tif"
    proj = "+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m"
    with dry_run() as cmds:
        ScProcessor(tmp_path, mapproject=True, dem=dem, resolution=50, t_srs=proj).process_cube(cube)
    mp = cmds[-1]
    assert mp[0] == "mapproject"
    assert mp[-4:] == [str(dem), str(cube), str(tmp_path / f"{PID}.json"), str(tmp_path / f"{PID}.tif")]
    assert mp[mp.index("--tr") + 1] == "50" and mp[mp.index("--t_srs") + 1] == proj


def test_resume_skips_existing_outputs(cube, tmp_path):
    _spiceinit(cube)
    with dry_run() as cmds:
        ScProcessor(tmp_path).process_cube(cube)
    assert [c[0] for c in cmds] == ["isd_generate"]
    (tmp_path / f"{PID}.json").touch()
    with dry_run() as cmds:
        ScProcessor(tmp_path).process_cube(cube)
    assert cmds == []
    with dry_run() as cmds:
        ScProcessor(tmp_path, force=True).process_cube(cube)
    assert [c[0] for c in cmds] == ["spiceinit", "isd_generate"]


def test_download_product_resumes(tmp_path, monkeypatch):
    from lunar_toolkit.shadowcam import sc

    fetched: list[str] = []

    def fake_download(url, dest, show_progress=True, **kw):
        dest.write_bytes(b"x")
        fetched.append(url)
        return dest

    monkeypatch.setattr(sc, "download_file", fake_download)
    product = ScProduct(PID, BASE + f"{PID}.cub", BASE + f"{PID}.xml")
    assert download_product(product, tmp_path) == tmp_path / f"{PID}.cub"
    assert (tmp_path / f"{PID}.xml").exists()
    download_product(product, tmp_path)
    assert len(fetched) == 2


def test_cli_dry_run(capsys, tmp_path, monkeypatch):
    from lunar_toolkit.cli import shadowcam as cli
    from lunar_toolkit.shadowcam import sc

    seen: list[str] = []

    def fake_lookup(ids, client=None, product="raw"):
        seen.extend(product_id_for(i, product) for i in ids)
        return _df("sc_raw.json")

    monkeypatch.setattr(sc, "lookup_products", fake_lookup)
    sub = argparse.ArgumentParser().add_subparsers(dest="cmd")
    cli.add_parser(sub)
    parser = sub.choices[cli.COMMAND]
    args = parser.parse_args(["M013178466S", "-o", str(tmp_path), "--dry-run", "--validate"])
    assert cli.run(args) == 0
    assert seen == [PID]
    out = capsys.readouterr().out
    assert f"spiceinit from={tmp_path / PID}.cub" in out
    assert "isd_generate" in out and "cam_test" in out
    assert not list(tmp_path.iterdir())  # nothing written
