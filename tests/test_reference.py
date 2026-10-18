"""Reference data: Artemis III regions, PGDA LOLA 5 m index, xyzi point clouds, Bertone et al. zips."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES

from lunar_toolkit import reference as ref
from lunar_toolkit.cli import reference as cli
from lunar_toolkit.geometry import aoi_mask, aoi_region, ps_xy

PGDA_HTML = FIXTURES / "pgda_lola_5mpp_products78.html"
ZENODO_JSON = FIXTURES / "zenodo_17954508.json"
MEMBERS_JSON = FIXTURES / "a3clr22_11_members.json"


@pytest.fixture(scope="module")
def index():
    return ref.parse_pgda_index(PGDA_HTML.read_text())


def test_thirteen_regions_with_unique_keys():
    assert [r.id for r in ref.REGIONS] == list(range(13))
    assert len({r.slug for r in ref.REGIONS}) == 13
    assert len({r.zenodo_file for r in ref.REGIONS}) == 13
    for r in ref.REGIONS:
        assert r.zenodo_file.startswith(f"A3CLR22_{r.id}_")
        assert r.pgda_site in ref.PGDA_SITES
        w, s, e, n = r.bbox
        assert r.center == pytest.approx(((w + e) / 2, (s + n) / 2), abs=1e-5)


@pytest.mark.parametrize(
    "key",
    [
        11,
        "11",
        "malapert",
        "Malapert Massif",
        "MALAPERT-MASSIF",
        "malapert_massif",
        "Site23",
        "A3CLR22_11_Malapert_Massif",
    ],
)
def test_get_region_aliases(key):
    assert ref.get_region(key).id == 11


def test_get_region_exact_beats_partial():
    assert ref.get_region("de Gerlache Rim").id == 9
    assert ref.get_region("de gerlache rim 2").id == 4
    assert ref.get_region("connecting ridge").id == 6
    assert ref.get_region("shackleton").id == 8
    with pytest.raises(KeyError, match="ambiguous"):
        ref.get_region("nobile")
    with pytest.raises(KeyError):
        ref.get_region("tycho")
    with pytest.raises(KeyError):
        ref.get_region(13)


def test_zenodo_fixture_matches_table():
    files = {f["key"]: f for f in ref.parse_zenodo_files(json.loads(ZENODO_JSON.read_text()))}
    assert len(files) == 14  # 13 zips + README.md
    for r in ref.REGIONS:
        f = files[r.zenodo_file]
        assert (f["size"], f["md5"]) == (r.zenodo_size, r.zenodo_md5)
        assert f["url"] == r.zenodo_url
    assert ref.get_region(11).zenodo_url.endswith("/A3CLR22_11_Malapert_Massif.zip?download=1")


def test_zenodo_files_cached(tmp_path):
    (tmp_path / f"zenodo_{ref.ZENODO_RECORD}.json").write_text(ZENODO_JSON.read_text())
    assert len(ref.zenodo_files(cache_dir=tmp_path)) == 14


def test_ps_latlon_inverts_ps_xy():
    for lat, lon in [(-85.98, -0.256), (-89.46, -136.68), (-84.2, 69.3), (-88.8, 124.7)]:
        assert ref.ps_latlon(*ps_xy(lat, lon, "S")) == pytest.approx((lat, lon), abs=1e-9)


def test_roi_bbox_is_lonlat_envelope_of_ps_square():
    # The Zenodo bboxes are the lon/lat envelopes of the PGDA site squares (vertex-based for lat).
    for r in ref.REGIONS:
        ring = r.roi.ring_lonlat(step_m=50)
        los, las = [p[0] for p in ring], [p[1] for p in ring]
        w, s, e, n = r.bbox
        assert (min(los), max(los), max(las)) == pytest.approx((w, e, n), abs=2e-3)
        assert s - 0.04 < min(las) <= s + 1e-3


def test_malapert_roi_and_projwin():
    r = ref.get_region("malapert")
    assert r.pgda_site == "Site23"
    assert r.roi.bounds == (-11000, 111000, 10000, 132000)
    assert r.projwin == (-11000, 132000, 10000, 111000)
    ulx, uly, lrx, lry = r.bbox_projwin
    assert ulx < -11000 and uly > 132000 and lrx > 10000 and lry < 111000


def test_region_aoi_is_square_in_ps():
    r = ref.get_region(6)  # Connecting Ridge: widest longitude span
    aoi = r.aoi()
    reg = aoi_region(aoi)
    x0, x1, y0, y1 = reg["ps"]["S"]
    assert (x0, y0) == pytest.approx(r.roi.bounds[:2], abs=110) and (x1, y1) == pytest.approx(
        r.roi.bounds[2:], abs=110
    )
    lat, lon = r.roi.center_latlon
    corner = ref.ps_latlon(r.roi.x0 + 50, r.roi.y0 + 50)
    outside = ref.ps_latlon(r.roi.x0 - 500, r.roi.y0 + 500)
    df = pd.DataFrame(
        {
            "FP_WKT": [
                f"POLYGON(({lon} {lat}, {lon + 1e-4} {lat}, {lon} {lat + 1e-4}, {lon} {lat}))",
                f"POLYGON(({corner[1]} {corner[0]}, {corner[1] + 1e-4} {corner[0]}, {corner[1]} {corner[0] + 1e-5}, {corner[1]} {corner[0]}))",
                f"POLYGON(({outside[1]} {outside[0]}, {outside[1] + 1e-4} {outside[0]}, {outside[1]} {outside[0] + 1e-5}, {outside[1]} {outside[0]}))",
            ]
        }
    )
    assert aoi_mask(df, aoi) == [True, True, False]


def test_bbox_aoi_is_densified_wedge():
    r = ref.get_region(11)
    ring = r.bbox_aoi().polygons[0][0]
    assert len(ring) > 100
    assert aoi_region(r.bbox_aoi())["ps"]["S"][0] == pytest.approx(r.bbox_projwin[0], abs=200)


def test_crop_window_default_center_and_snap():
    box = ref.crop_window("malapert")
    assert (box.width, box.height) == (2500, 2500)
    assert box.center_xy == pytest.approx((-500, 121500), abs=2.5)
    assert all(v % 5 == 0 for v in box.bounds)
    assert ref.get_region(11).roi.contains(box)
    assert ref.PGDA_SITES["Site23"].extent.contains(box)


def test_crop_window_center_latlon_and_aoi():
    box = ref.crop_window(11, size_km=1.0, center=(-86.0, 1.0), snap_m=None)
    assert box.center_xy == pytest.approx(ps_xy(-86.0, 1.0, "S"))
    assert box.center_latlon == pytest.approx((-86.0, 1.0))
    assert box.aoi().polygons
    assert box.wkt().startswith("POLYGON((")
    assert box.wkt_ps().startswith("POLYGON((")


def test_crop_window_outside_roi_warns(caplog):
    ref.crop_window(11, 30.0)
    assert "outside" in caplog.text


def test_pole_box_has_no_lonlat_ring():
    with pytest.raises(ValueError, match="pole"):
        ref.PsBox(-10, -10, 10, 10).ring_lonlat()


def test_parse_pgda_index(index):
    assert set(index) >= {r.pgda_site for r in ref.REGIONS}
    assert set(index) == set(ref.PGDA_SITES)
    s23 = index["Site23"]
    assert set(s23) == {"surf", "ldec", "toterr", "slp", "slperr", "xyzi", "clones"}
    assert s23["surf"] == ["https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site23/Site23_final_adj_5mpp_surf.tif"]
    assert s23["xyzi"] == ["https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site23/Site23_final_adj.xyzi"]
    assert len(s23["clones"]) == 100 and s23["clones"][0].endswith(
        "Clones/Site23_final_adj_5mpp_0001_err.tif"
    )
    assert index["Site01"]["rois"] == ["https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site01/Site01_ROIs.tgz"]
    assert "rois" not in index["Site23"]
    assert len(index["DM2"]["clones"]) == 100  # relative duplicate of clone 1 is merged


def test_parse_pgda_relative_links():
    html = '<a href="Site23/Site23_final_adj_5mpp_ldec.tif">x</a> <a href=/data/LOLA_5mpp/DM2/DM2_final_adj.xyzi>y</a>'
    idx = ref.parse_pgda_index(html)
    assert idx["Site23"]["ldec"] == [ref.PGDA_ROOT + "Site23/Site23_final_adj_5mpp_ldec.tif"]
    assert idx["DM2"]["xyzi"] == [ref.PGDA_ROOT + "DM2/DM2_final_adj.xyzi"]


def test_pgda_index_cache(tmp_path, index):
    (tmp_path / "pgda_lola_5mpp_index.json").write_text(json.dumps(index))
    assert ref.pgda_index(cache_dir=tmp_path) == index
    assert ref.pgda_index(html=PGDA_HTML.read_text()) == index


def test_pgda_urls(index):
    urls = ref.pgda_urls("malapert", ["surf", "toterr", "clones"], clones=[1, 50], index=index)
    assert [u.rsplit("/", 1)[1] for u in urls["clones"]] == [
        "Site23_final_adj_5mpp_0001_err.tif",
        "Site23_final_adj_5mpp_0050_err.tif",
    ]
    assert len(ref.pgda_urls("Site23", ["clones"], clones=3, index=index)["clones"]) == 3
    assert urls["toterr"][0].endswith("Site23_final_adj_5mpp_toterr.tif")
    with pytest.raises(ValueError, match="Unknown"):
        ref.pgda_urls(11, ["dem"], index=index)


def test_download_pgda_layout_and_reuse(tmp_path, index, monkeypatch):
    calls = []

    def fake_download(url, dest, show_progress=True, expected_md5=None):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(ref, "download_file", fake_download)
    out = ref.download_pgda(11, ["ldec", "clones"], tmp_path, clones=2, index=index, show_progress=False)
    assert out["ldec"] == [tmp_path / "Site23" / "Site23_final_adj_5mpp_ldec.tif"]
    assert out["clones"][1] == tmp_path / "Site23" / "Clones" / "Site23_final_adj_5mpp_0002_err.tif"
    assert len(calls) == 3
    ref.download_pgda(11, ["ldec"], tmp_path, index=index)
    assert len(calls) == 3
    ref.download_pgda(11, ["ldec"], tmp_path, index=index, force=True)
    assert len(calls) == 4


def test_fetch_retries_throttling(tmp_path, monkeypatch):
    import urllib.error

    attempts = []

    def flaky(url, dest, show_progress=True, expected_md5=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)  # type: ignore[arg-type]
        dest.write_bytes(b"ok")
        return dest

    monkeypatch.setattr(ref, "download_file", flaky)
    monkeypatch.setattr(ref.time, "sleep", lambda s: None)
    assert ref.fetch("https://example/x", tmp_path / "x").read_bytes() == b"ok"
    assert len(attempts) == 3


def _xyzi(path, rows):
    np.asarray(rows, dtype="<f8").tofile(path)
    return path


def test_read_xyzi_and_asp_csv(tmp_path):
    # Values as in the first rows of Site23_final_adj.xyzi (km; Z is height above 1737.4 km)
    rows = [
        [-8.39048968166, 131.543520054, 1.67282089909, 91941154.0],
        [-0.5, 121.5, 4.25858925171, 182130625.0],
        [9.9, 111.1, 2.0, 100010000.0],
    ]
    f = _xyzi(tmp_path / "t.xyzi", rows)
    df = ref.read_xyzi(f)
    assert list(df.columns) == ["x", "y", "height", "rdr_id"]
    assert df.x[0] == pytest.approx(-8390.48968166) and df.height[0] == pytest.approx(1672.82089909)
    assert df.rdr_id.tolist() == [91941154, 182130625, 100010000]
    box = ref.crop_window(11)
    assert len(ref.read_xyzi(f, box)) == 1

    out = ref.write_asp_csv(f, tmp_path / "t.csv", box)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("#") and len(lines) == 2
    assert lines[1] == "-500.000,121500.000,4258.589,182130625"
    out2 = ref.write_asp_csv(df, tmp_path / "all.csv")
    assert len(out2.read_text().splitlines()) == 4


def test_read_xyzi_rejects_truncated(tmp_path):
    p = tmp_path / "bad.xyzi"
    np.zeros(5, dtype="<f8").tofile(p)
    with pytest.raises(ValueError):
        ref.read_xyzi(p)


@pytest.fixture
def fake_zip(tmp_path):
    names = [n for n, _, _ in json.loads(MEMBERS_JSON.read_text())]
    path = tmp_path / "A3CLR22_11_Malapert_Massif.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for n in names:
            if n.endswith("/"):
                zf.writestr(n, b"")
            elif n.endswith(("GLDELEV_005.tif", "GLDSIGM_005.tif", "README.txt", "_100.tif")) or "/." in n:
                zf.writestr(n, n.encode())
    return path


def test_members_fixture_contents():
    names = [n for n, _, _ in json.loads(MEMBERS_JSON.read_text())]
    top = "A3CLR22_11_Malapert_Massif/"
    for product in ("GLDELEV", "GLDMASK", "GLDSIGM", "GLDDIFF", "GLDHILL_005_315_45", "GLDOMOS", "GLDBRES"):
        assert any(n.startswith(f"{top}A311_{product}") for n in names)
    assert sum(n.startswith(f"{top}LROC-NAC-maps/") and n.endswith(".tif") for n in names) == 265


def test_select_members():
    names = [n for n, _, _ in json.loads(MEMBERS_JSON.read_text())]
    assert ref.select_members(names, ["GLDELEV"]) == ["A3CLR22_11_Malapert_Massif/A311_GLDELEV_005.tif"]
    slopes = ref.select_members(names, ["SLOPE_MAPS/*_100.tif"])
    assert slopes == ["A3CLR22_11_Malapert_Massif/SLOPE_MAPS/A311_GLDTSLP_005_100.tif"]
    assert ref.select_members(names, ["readme"]) == ["A3CLR22_11_Malapert_Massif/README.txt"]
    assert not any("MACOSX" in n for n in ref.select_members(names, ["*"]))


def test_sdem_members_and_extract_local(fake_zip, tmp_path):
    members = ref.sdem_members(fake_zip)
    assert not members.name.str.contains("MACOSX|DS_Store").any()
    assert not members.name.str.endswith("/").any()
    out = ref.extract_sdem(fake_zip, ["GLDELEV", "SLOPE_MAPS/*_100.tif"], tmp_path / "x")
    assert out == [
        tmp_path / "x" / "A311_GLDELEV_005.tif",
        tmp_path / "x" / "SLOPE_MAPS" / "A311_GLDTSLP_005_100.tif",
    ]
    assert out[0].read_bytes() == b"A3CLR22_11_Malapert_Massif/A311_GLDELEV_005.tif"
    out[0].write_bytes(b"kept")
    ref.extract_sdem(fake_zip, ["GLDELEV"], tmp_path / "x")
    assert out[0].read_bytes() == b"kept"
    ref.extract_sdem(fake_zip, ["GLDELEV"], tmp_path / "x", force=True)
    assert out[0].read_bytes() != b"kept"
    with pytest.raises(KeyError):
        ref.extract_sdem(fake_zip, ["NOPE"], tmp_path / "x")


def test_http_range_file_reads_zip(fake_zip, monkeypatch):
    data = fake_zip.read_bytes()

    class Resp:
        status = 206

        def __init__(self, req):
            a, b = req.headers["Range"].split("=")[1].split("-")
            self.body = data[int(a) : int(b) + 1]
            self.headers = {"Content-Range": f"bytes {a}-{b}/{len(data)}"}

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ref.urllib.request, "urlopen", lambda req, timeout=None: Resp(req))
    members = ref.sdem_members(11)
    assert "A3CLR22_11_Malapert_Massif/README.txt" in set(members.name)
    with zipfile.ZipFile(ref.HttpRangeFile(ref.get_region(11).zenodo_url)) as zf:
        assert zf.read("A3CLR22_11_Malapert_Massif/README.txt") == b"A3CLR22_11_Malapert_Massif/README.txt"


def test_download_sdem_uses_md5(tmp_path, monkeypatch):
    seen = {}

    def fake_download(url, dest, show_progress=True, expected_md5=None):
        seen.update(url=url, md5=expected_md5)
        dest.write_bytes(b"z")
        return dest

    monkeypatch.setattr(ref, "download_file", fake_download)
    p = ref.download_sdem("malapert", tmp_path, show_progress=False)
    assert p == tmp_path / "A3CLR22_11_Malapert_Massif.zip"
    assert seen == {"url": ref.get_region(11).zenodo_url, "md5": "772a578b8f2ebdb078c5a04a3c07182a"}


def test_regions_table():
    t = ref.regions_table()
    assert len(t) == 13 and t.loc[11, "pgda_site"] == "Site23"


def _cli(*argv):
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    cli.add_parser(sub)
    return cli.run(parser.parse_args([cli.COMMAND, *argv]))


def test_cli_list_show_crop(capsys):
    assert _cli("list") == 0
    out = capsys.readouterr().out
    assert "Malapert Massif" in out and "Site23" in out and len(out.splitlines()) == 14
    assert _cli("show", "malapert", "--json", "--crop-km", "2.5") == 0
    info = json.loads(capsys.readouterr().out)
    assert info["pgda_site"] == "Site23" and info["roi_projwin"] == [-11000, 132000, 10000, 111000]
    assert info["crop_projwin"] == [-1750, 122750, 750, 120250]
    assert _cli("crop", "11", "--crop-km", "2.5") == 0
    assert capsys.readouterr().out.split() == ["-1750", "122750", "750", "120250"]
    assert _cli("crop", "11", "--format", "geojson") == 0
    assert json.loads(capsys.readouterr().out)["type"] == "Polygon"
    assert _cli("show", "nobile") == 1
    assert "ambiguous" in capsys.readouterr().err


def test_cli_pgda_list(capsys, monkeypatch, index):
    monkeypatch.setattr(ref, "pgda_index", lambda refresh=False: index)
    assert _cli("pgda", "malapert", "--kind", "surf", "xyzi", "--list") == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[1].endswith("Site23_final_adj.xyzi")


def test_cli_sdem_local(capsys, fake_zip, tmp_path):
    assert _cli("sdem", "malapert", "-o", str(tmp_path / "o")) == 1
    assert "--download" in capsys.readouterr().err
    assert (
        _cli("sdem", "malapert", "--zip", str(fake_zip), "--member", "GLDELEV", "-o", str(tmp_path / "o"))
        == 0
    )
    path = capsys.readouterr().out.strip()
    assert path.endswith("A3CLR22_11_Malapert_Massif/A311_GLDELEV_005.tif")
    assert _cli("sdem", "malapert", "--zip", str(fake_zip), "--list") == 0
    assert "GLDSIGM" in capsys.readouterr().out


CAMERA = FIXTURES / "A311_GLDACAM_005_M176388393LE.json"
STATUS = FIXTURES / "image_status_11_mal_head.csv"
MAP_HEAD = FIXTURES / "A311_GLDIMGP_005_M176388393LE_head.tif"  # first 4 KB of the real map


@pytest.fixture
def published_zip(tmp_path):
    """A region-11 zip with the real member names; one real camera, status table and map header."""
    names = [n for n, _, _ in json.loads(MEMBERS_JSON.read_text())]
    path = tmp_path / "A3CLR22_11_Malapert_Massif.zip"
    head = MAP_HEAD.read_bytes()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in names:
            if n.endswith("/"):
                continue
            if n.endswith("_GLDACAM_005_M176388393LE.json"):
                zf.writestr(n, CAMERA.read_bytes())
            elif "_GLDACAM_" in n:
                zf.writestr(n, n.encode())
            elif n.endswith("image_status_11_mal.csv"):
                zf.writestr(n, STATUS.read_bytes())
            elif n.endswith("_GLDIMGP_005_M176388393LE.tif") or n.endswith("_GLDIMGP_005_M176395141LE.tif"):
                zf.writestr(n, head)
    return path


def test_parse_tiff_extent():
    box = ref.parse_tiff_extent(MAP_HEAD.read_bytes())
    assert box.bounds == pytest.approx((-317.5, 127957.5 - 3392 * 5, -317.5 + 1044 * 5, 127957.5))
    with pytest.raises(ValueError):
        ref.parse_tiff_extent(b"GIF89a" + bytes(10))
    with pytest.raises(ValueError, match="beyond"):
        ref.parse_tiff_extent(MAP_HEAD.read_bytes()[:600])


def test_parse_image_status():
    df = ref.parse_image_status(STATUS.read_text())
    assert list(df.columns) == ["product_id", "status"] and len(df) == 19
    assert df.attrs["summary"] == {
        "total_images": 2761,
        "available_not_selected": 2435,
        "not_aligned": 61,
        "used_for_sfs": 265,
    }
    assert set(df.status) == {"used_for_sfs", "not_aligned", "available_not_selected"}


def test_published_members_and_status(published_zip, tmp_path):
    cams = ref.published_members(published_zip)
    assert len(cams) == 265 and cams.product_id.str.match(r"M\d+[LR]E$").all()
    maps = ref.published_members(published_zip, "GLDIMGP")
    assert set(maps.product_id) == {"M176388393LE", "M176395141LE"}
    st = ref.image_status(published_zip, tmp_path / "st")
    assert st.attrs["summary"]["used_for_sfs"] == 265
    assert (tmp_path / "st" / "extra" / "image_status_11_mal.csv").is_file()


def test_published_extents_and_covering(published_zip, tmp_path):
    cache = tmp_path / "ext.json"
    ext = ref.published_extents(published_zip, cache=cache)
    assert set(ext.product_id) == {"M176388393LE", "M176395141LE"}
    assert ext.iloc[0][["x0", "y1"]].tolist() == pytest.approx([-317.5, 127957.5])
    assert cache.is_file() and ref.published_extents(tmp_path / "missing.zip", cache=cache).equals(ext)
    inside = ref.PsBox(0, 115000, 2500, 117500)
    outside = ref.PsBox(-9000, 115000, -6500, 117500)
    assert set(ref.images_covering(ext, inside)) == {"M176388393LE", "M176395141LE"}
    assert ref.images_covering(ext, outside) == []
    assert ref.images_covering(ext, ref.PsBox(4000, 115000, 6000, 117000), min_fraction=0.5) == []


def test_published_cameras(published_zip, tmp_path):
    out = ref.published_cameras(11, tmp_path / "c", images=["m176388393le"], source=published_zip)
    assert list(out) == ["M176388393LE"]
    assert out["M176388393LE"] == tmp_path / "c" / "LROC-NAC-adjust" / "A311_GLDACAM_005_M176388393LE.json"
    win = ref.published_cameras(
        11, tmp_path / "c", window=ref.PsBox(0, 115000, 2500, 117500), source=published_zip
    )
    assert list(win) == ["M176388393LE", "M176395141LE"]
    assert (tmp_path / "c" / "gldimgp_extents_11.json").is_file()
    assert ref.published_cameras(11, tmp_path / "c", images=["M999LE"], source=published_zip) == {}


def test_published_camera_factor_and_native():
    from lunar_toolkit.csm import read_camera

    cam = read_camera(CAMERA)
    assert ref.published_reduce_factor(cam) == 4 and ref.published_reduce_factor(CAMERA) == 4
    native = ref.published_camera_native(cam, (2532, 15360))
    d = native.data
    assert native.size == (2532, 15360)
    assert d["m_detectorSampleSumming"] == pytest.approx(0.5)
    assert d["m_intTimes"][0] == pytest.approx(cam.data["m_intTimes"][0] / 4)
    assert d["m_gsd"] == cam.data["m_gsd"]
    assert d["m_focalLength"] == cam.data["m_focalLength"]  # native sample 4 x maps to reduced sample x
