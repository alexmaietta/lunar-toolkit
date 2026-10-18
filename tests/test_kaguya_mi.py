"""Kaguya MI: ODE file selection and the ISIS import command chain (offline)."""

import argparse
import gzip
import json

import pytest
from conftest import FIXTURES

from lunar_toolkit.kaguya.mi import (
    MiProcessor,
    MiProduct,
    download_mi_product,
    is_mi_product_id,
    mi_products,
    select_mi_files,
)
from lunar_toolkit.ode import INSTRUMENTS, extract_products, parse_product, resolve_instrument, to_dataframe
from lunar_toolkit.tools import dry_run, format_command

VIS = "MVA_2B2_01_01935N340E0939"
NIR = "MNA_2B2_01_01935N339E0939"
LABEL = f"""PDS_VERSION_ID = "PDS3"
FILE_NAME = "{VIS}.img"
^IMAGE = ("{VIS}.img", 1 <BYTES>)
PRODUCT_ID = "{VIS}"
"""


@pytest.fixture
def mi_df():
    res = json.loads((FIXTURES / "kaguya_mi_2b2.json").read_text())["ODEResults"]
    return to_dataframe([parse_product(p, "Kaguya MI") for p in extract_products(res)])


@pytest.fixture
def label(tmp_path):
    lbl = tmp_path / f"{VIS}.lbl"
    lbl.write_text(LABEL)
    (tmp_path / f"{VIS}.img").write_bytes(b"\0" * 16)
    return lbl


def test_registry():
    key, entry, pt = resolve_instrument("mi")
    assert key == "kaguya_mi" and pt == "VIS2B"
    assert (entry["ihid"], entry["iid"]) == ("SLN", "MI")
    assert resolve_instrument("kaguya_mi", "nir")[2] == "NIR2B"
    assert INSTRUMENTS["kaguya_mi"]["label"] == "Kaguya MI"


def test_product_id_regex():
    assert is_mi_product_id(VIS) and is_mi_product_id(NIR.lower())
    assert not is_mi_product_id("MVA_2C2_01_01935N340E0939")
    assert not is_mi_product_id("MIA_3C5_03_01935N340E0939SC")
    assert not is_mi_product_id("TC1W2B0_01_02936N034E0938")


def test_mi_products_from_ode(mi_df):
    products = mi_products(mi_df)
    assert [p.product_id for p in products] == [VIS, NIR]
    vis, nir = products
    assert vis.image_url.endswith(f"/sln-l-mi-3-vis-level2b2-v1.0/20080315/data/{VIS}.img.gz")
    assert vis.label_url.endswith(f"/data/{VIS}.lbl")
    assert (vis.image_kbytes, nir.image_kbytes) == (6001, 514)
    assert (vis.sensor, nir.sensor) == ("VIS", "NIR")


def test_select_mi_files_rules(mi_df):
    files = mi_df["files"][0]
    assert {f["Type"] for f in files} == {"Product", "Browse", "Derived"}
    assert select_mi_files(files).product_id == VIS
    assert select_mi_files([f for f in files if not f["FileName"].endswith(".LBL")]) is None


def test_mi_products_skips_other_levels(mi_df):
    df = mi_df.copy()
    df.loc[0, "product_id"] = "MIA_3C5_03_01935N340E0939SC"
    assert [p.product_id for p in mi_products(df)] == [NIR]


def test_import_command_chain(label, tmp_path):
    out = tmp_path / "out"
    with dry_run() as cmds:
        res = MiProcessor(out).import_label(label)
    assert [c[0] for c in cmds] == ["kaguyami2isis", "spiceinit"]  # web → no CSM by default
    k2i, spice = cmds
    assert f"from={label}" in k2i and f"to={out / VIS}.cub" in k2i
    assert {"setnullrange=false", "setlisrange=false"} <= set(k2i)
    assert {"web=true", "ckrecon=true", "spkrecon=true", "shape=system"} <= set(spice)
    assert not any(a.startswith("attach=") for a in spice)
    assert res.cube == out / f"{VIS}.cub" and res.camera is None and res.geotiff is None
    assert (tmp_path / f"{VIS}.img").exists()  # nothing deleted in a dry run


def test_import_local_kernels_csm_and_mapproject(label, tmp_path):
    with dry_run() as cmds:
        res = MiProcessor(tmp_path, web=False, mapproject=True, resolution=20).import_label(label)
    assert [c[0] for c in cmds] == [
        "kaguyami2isis",
        "spiceinit",
        "isd_generate",
        "mosrange",
        "cam2map",
        "gdal_translate",
    ]
    assert "web=false" in cmds[1]
    assert "--only_naif_spice" in cmds[2] and cmds[2][-1] == str(tmp_path / f"{VIS}.cub")
    assert {"pixres=mpp", "resolution=20"} <= set(cmds[4])
    assert res.camera == tmp_path / f"{VIS}.json"
    assert cmds[5][-1] == str(tmp_path / f"{VIS}.tif") == str(res.geotiff)


def test_resume_skips_existing_outputs(label, tmp_path):
    (tmp_path / f"{VIS}.cub").touch()
    with dry_run() as cmds:
        MiProcessor(tmp_path, csm=True).import_label(label)
    assert [c[0] for c in cmds] == ["isd_generate"]
    (tmp_path / f"{VIS}.json").touch()
    with dry_run() as cmds:
        MiProcessor(tmp_path, csm=True).import_label(label)
    assert cmds == []
    with dry_run() as cmds:
        MiProcessor(tmp_path, force=True).import_label(label)
    assert [c[0] for c in cmds] == ["kaguyami2isis", "spiceinit"]


def test_failed_spiceinit_removes_cube(label, tmp_path, monkeypatch):
    from lunar_toolkit.kaguya import mi

    cube = tmp_path / f"{VIS}.cub"
    monkeypatch.setattr(mi, "isis", lambda *a, **k: cube.touch())

    def fail(*a, **k):
        raise RuntimeError("no SPICE")

    monkeypatch.setattr(mi, "spiceinit", fail)
    with pytest.raises(RuntimeError):
        MiProcessor(tmp_path).import_label(label)
    assert not cube.exists()


def test_download_gunzips_to_label_name(tmp_path, monkeypatch):
    from lunar_toolkit.kaguya import tc

    fetched: list[str] = []

    def fake_download(url, dest, show_progress=True, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".lbl"):
            dest.write_text(LABEL)
        else:
            dest.write_bytes(gzip.compress(b"MI pixels"))
        fetched.append(url)
        return dest

    monkeypatch.setattr(tc, "download_file", fake_download)
    product = MiProduct(VIS, f"https://x/data/{VIS}.img.gz", f"https://x/data/{VIS}.lbl")
    assert download_mi_product(product, tmp_path) == tmp_path / f"{VIS}.lbl"
    assert (tmp_path / f"{VIS}.img").read_bytes() == b"MI pixels"
    download_mi_product(product, tmp_path)
    assert len(fetched) == 2


def test_cli_dry_run_infers_mi(capsys, tmp_path, monkeypatch, mi_df):
    from lunar_toolkit.cli import kaguya as cli
    from lunar_toolkit.kaguya import mi

    monkeypatch.setattr(mi, "lookup_mi_products", lambda ids, client=None: mi_df)
    sub = argparse.ArgumentParser().add_subparsers(dest="cmd")
    cli.add_parser(sub)
    args = sub.choices[cli.COMMAND].parse_args([VIS, NIR, "-o", str(tmp_path), "--dry-run", "--csm"])
    assert cli.run(args) == 0
    out = capsys.readouterr().out
    assert "2 Kaguya MI product(s)" in out
    assert format_command(["kaguyami2isis", f"from={tmp_path / VIS}.lbl"]) in out
    assert format_command(["kaguyami2isis", f"from={tmp_path / NIR}.lbl"]) in out
    assert "isd_generate" in out and "kaguyatc2isis" not in out
    assert not list(tmp_path.iterdir())
