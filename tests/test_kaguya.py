"""Kaguya TC: ODE file selection and the ISIS import command chain (offline)."""

import argparse
import gzip
import json

import pytest
from conftest import FIXTURES

from lunar_toolkit.kaguya.tc import (
    TcProcessor,
    TcProduct,
    download_product,
    image_pointer,
    is_tc_product_id,
    select_tc_files,
    tc_products,
)
from lunar_toolkit.ode import extract_products, parse_product, to_dataframe
from lunar_toolkit.tools import dry_run, format_command

PID = "TC1W2B0_01_01601N336E0940"
LABEL = f"""PDS_VERSION_ID = "PDS3"
FILE_NAME = "{PID}.img"
^IMAGE = ("{PID}.img", 1 <BYTES>)
PRODUCT_ID = "{PID}"
"""


@pytest.fixture
def stereo_df():
    res = json.loads((FIXTURES / "kaguya_tc_stereo.json").read_text())["ODEResults"]
    return to_dataframe([parse_product(p, "Kaguya TC") for p in extract_products(res)])


@pytest.fixture
def label(tmp_path):
    lbl = tmp_path / f"{PID}.lbl"
    lbl.write_text(LABEL)
    (tmp_path / f"{PID}.img").write_bytes(b"\0" * 16)
    return lbl


def test_product_id_regex():
    assert is_tc_product_id("TC1W2B0_01_02936N034E0938")
    assert is_tc_product_id("tc2s2b0_01_03005n096e0199")
    assert not is_tc_product_id("TCO_MAP_02_N33E093N30E096SC")
    assert not is_tc_product_id("M1096757719LE")


def test_tc_products_from_ode(stereo_df):
    products = tc_products(stereo_df)
    assert [p.product_id for p in products] == [PID, "TC1W2B0_01_01601N350E0940"]
    p = products[0]
    assert p.image_url.endswith(f"/data/{PID}.img.gz")
    assert p.label_url.endswith(f"/data/{PID}.lbl")
    assert p.image_kbytes == 23001
    assert p.is_stereo and p.sensor == "TC1"


def test_select_tc_files_rules():
    base = "https://example.org/data/"
    files = [
        {"FileName": "X.JPG", "Type": "Browse", "URL": base + "x.jpg"},
        {"FileName": "X.LBL", "Type": "Product", "URL": base + "TC2S2B0_01_00001N000E0000.lbl"},
        {"FileName": "X.IMG.GZ", "Type": "Product", "URL": base + "TC2S2B0_01_00001N000E0000.img.gz"},
        {"FileName": "X.IMG", "Type": "Product", "URL": base + "TC2S2B0_01_00001N000E0000.img"},
        {"FileName": "X_LBL.KML", "Type": "Derived", "URL": base + "x_lbl.kml"},
    ]
    p = select_tc_files(files)
    assert p.product_id == "TC2S2B0_01_00001N000E0000"
    assert p.image_url.endswith(".img")  # uncompressed preferred
    assert not p.is_stereo
    assert select_tc_files(files[:2]) is None  # no data file
    assert select_tc_files([]) is None


def test_tc_products_skips_non_level2b0(stereo_df):
    df = stereo_df.copy()
    df.loc[0, "product_id"] = "TCO_MAP_02_N33E093N30E096SC"
    assert [p.product_id for p in tc_products(df)] == ["TC1W2B0_01_01601N350E0940"]


def test_image_pointer(label):
    assert image_pointer(label) == f"{PID}.img"


def test_import_command_chain(label, tmp_path):
    out = tmp_path / "out"
    with dry_run() as cmds:
        res = TcProcessor(out).import_label(label)
    assert [c[0] for c in cmds] == ["kaguyatc2isis", "spiceinit", "isd_generate"]
    k2i, spice, isd = cmds
    assert f"from={label}" in k2i and f"to={out / PID}.cub" in k2i
    assert {"setnullrange=false", "sethrsrange=false", "setlisrange=false"} <= set(k2i)
    assert not any(a.startswith("attach=") for a in spice)  # ISIS: not with web=true
    assert {"web=true", "ckrecon=true", "spksmithed=true", "spkpredicted=false", "shape=system"} <= set(spice)
    assert "--only_naif_spice" not in isd  # web SPICE → use the attached tables
    assert isd[-1] == str(out / f"{PID}.cub")
    assert res.cube == out / f"{PID}.cub" and res.camera == out / f"{PID}.json"
    assert res.geotiff is None
    assert (tmp_path / f"{PID}.img").exists()  # nothing deleted in a dry run


def test_import_local_kernels_and_mapproject(label, tmp_path):
    with dry_run() as cmds:
        res = TcProcessor(tmp_path, web=False, mapproject=True, resolution=10).import_label(label)
    assert [c[0] for c in cmds] == [
        "kaguyatc2isis",
        "spiceinit",
        "isd_generate",
        "mosrange",
        "cam2map",
        "gdal_translate",
    ]
    assert {"web=false", "attach=true"} <= set(cmds[1])
    assert "--only_naif_spice" in cmds[2]  # ASP recipe with local kernels
    assert {"pixres=mpp", "resolution=10"} <= set(cmds[4])
    assert cmds[5][-1] == str(tmp_path / f"{PID}.tif") == str(res.geotiff)


def test_resume_skips_existing_outputs(label, tmp_path):
    (tmp_path / f"{PID}.cub").touch()
    with dry_run() as cmds:
        TcProcessor(tmp_path).import_label(label)
    assert [c[0] for c in cmds] == ["isd_generate"]
    (tmp_path / f"{PID}.json").touch()
    with dry_run() as cmds:
        TcProcessor(tmp_path, csm=True).import_label(label)
    assert cmds == []
    with dry_run() as cmds:
        TcProcessor(tmp_path, force=True).import_label(label)
    assert [c[0] for c in cmds] == ["kaguyatc2isis", "spiceinit", "isd_generate"]


def test_failed_spiceinit_removes_cube(label, tmp_path, monkeypatch):
    from lunar_toolkit.kaguya import tc

    cube = tmp_path / f"{PID}.cub"
    monkeypatch.setattr(tc, "kaguyatc2isis", lambda *a, **k: cube.touch())

    def fail(*a, **k):
        raise RuntimeError("no SPICE")

    monkeypatch.setattr(tc, "spiceinit", fail)
    with pytest.raises(RuntimeError):
        TcProcessor(tmp_path, csm=False).import_label(label)
    assert not cube.exists()


def test_download_product_gunzips_to_label_name(tmp_path, monkeypatch):
    from lunar_toolkit.kaguya import tc

    payload = b"TC pixels"

    def fake_download(url, dest, show_progress=True, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".lbl"):
            dest.write_text(LABEL)
        else:
            dest.write_bytes(gzip.compress(payload))
        fetched.append(url)
        return dest

    fetched: list[str] = []
    monkeypatch.setattr(tc, "download_file", fake_download)
    product = TcProduct(PID, f"https://x/data/{PID.lower()}.img.gz", f"https://x/data/{PID}.lbl")
    lbl = download_product(product, tmp_path)
    assert lbl == tmp_path / f"{PID}.lbl"
    assert (tmp_path / f"{PID}.img").read_bytes() == payload
    assert not (tmp_path / f"{PID.lower()}.img.gz").exists()
    download_product(product, tmp_path)  # resume: nothing fetched again
    assert len(fetched) == 2


def test_cli_dry_run(capsys, tmp_path, monkeypatch, stereo_df):
    from lunar_toolkit.cli import kaguya as cli
    from lunar_toolkit.kaguya import tc

    monkeypatch.setattr(tc, "lookup_products", lambda ids, client=None: stereo_df)
    sub = argparse.ArgumentParser().add_subparsers(dest="cmd")
    cli.add_parser(sub)
    parser = sub.choices[cli.COMMAND]
    args = parser.parse_args([PID, "-o", str(tmp_path), "--dry-run", "--mapproject"])
    assert cli.run(args) == 0
    out = capsys.readouterr().out
    assert format_command(["kaguyatc2isis", f"from={tmp_path / PID}.lbl"]) in out
    assert "cam2map" in out and "gdal_translate" in out
    assert not list(tmp_path.iterdir())  # nothing written
