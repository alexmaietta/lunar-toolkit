"""The static browser page stays in sync with the Python tables; serve's Range, images and relay."""

import re
from pathlib import Path

import lunar_toolkit
from lunar_toolkit.cli.serve import _byte_range, _image_path, _list_images, _relay_allowed
from lunar_toolkit.constants import FOOTPRINT_CORNERS, MOON_RADIUS_KM, POLAR_LAT_THRESHOLD
from lunar_toolkit.lroc.cumindex import _MACROS
from lunar_toolkit.lroc.nac import METADATA_LABELS
from lunar_toolkit.ode import INSTRUMENTS

PAGE = (Path(lunar_toolkit.__file__).parent / "static" / "index.html").read_text()


def _js_block(name: str) -> str:
    start = PAGE.index(f"const {name} = ")
    return PAGE[start : PAGE.index("};" if name == "ODE_INSTRUMENTS" else "];", start)]


def test_labels_match_metadata_labels():
    js = re.findall(r"\['([A-Z_]+)',\s+'([^']+)'\]", _js_block("LABELS"))
    assert js == [(c, label) for c, label in METADATA_LABELS if c != "PDS_URL"]


def test_ode_instruments_match_registry():
    block = _js_block("ODE_INSTRUMENTS")
    for key, entry in INSTRUMENTS.items():
        m = re.search(rf"{key}:\s*{{(.*?)types: {{(.*?)}}", block, re.S)
        assert m, key
        head, types = m.groups()
        assert f"ihid: '{entry['ihid']}'" in head and f"iid: '{entry['iid']}'" in head
        assert f"default: '{entry['default']}'" in head
        assert dict(re.findall(r"(\w+): '(\w+)'", types)) == entry["product_types"]


def _js_const(name: str) -> str:
    m = re.search(rf"const {name} = (.+?);\n", PAGE)
    assert m, name
    return m.group(1)


def test_geometry_constants_match_python():
    assert float(_js_const("MOON_RADIUS_KM")) == MOON_RADIUS_KM
    assert float(_js_const("POLAR_LAT_THRESHOLD")) == POLAR_LAT_THRESHOLD
    assert tuple(re.findall(r"'(\w+)'", _js_const("FOOTPRINT_CORNERS"))) == FOOTPRINT_CORNERS


def test_macros_match_cumindex():
    js = re.findall(r"'(CREATE OR REPLACE MACRO [^']+)'", _js_block("MACROS"))
    assert {m.split("(")[0] for m in js} == {
        "CREATE OR REPLACE MACRO _lon360",
        "CREATE OR REPLACE MACRO _arc_overlap",
    }
    assert all(m in _MACROS for m in js)


def test_js_element_ids_exist():
    ids = set(re.findall(r'\bid="([\w-]+)"', PAGE))
    used = set(re.findall(r"\$\('([\w-]+)'\)", PAGE))
    assert used and used <= ids, sorted(used - ids)
    # $(`${p}-...`) with p the search form prefix
    for suffix in set(re.findall(r"\$\(`\$\{p\}-([\w-]+)`\)", PAGE)):
        assert any(f"{p}-{suffix}" in ids for p in ("s",)), suffix


def test_byte_range():
    assert _byte_range("", 100) == ()
    assert _byte_range("bytes=0-9", 100) == (0, 10)
    assert _byte_range("bytes=-10", 100) == (90, 10)
    assert _byte_range("bytes=90-", 100) == (90, 10)
    assert _byte_range("bytes=95-200", 100) == (95, 5)
    assert _byte_range("bytes=100-", 100) is None
    assert _byte_range("bytes=x-", 100) is None


def test_cdn_modules_are_pinned():
    for m in re.finditer(r"https://cdn\.jsdelivr\.net/npm/([@\w./-]+?)@([^/'\"]+)/", PAGE):
        assert re.fullmatch(r"\d+\.\d+\.\d+", m.group(2)), m.group(0)
    assert "geotiff@" in PAGE


def test_image_path_stays_inside_root(tmp_path):
    root = tmp_path / "images"
    (root / "sfs").mkdir(parents=True)
    (root / "sfs" / "dem.tif").write_bytes(b"II*\0")
    (root / "notes.txt").write_text("x")
    (root / ".hidden.tif").write_bytes(b"II*\0")
    (tmp_path / "secret.tif").write_bytes(b"II*\0")
    (root / "link.tif").symlink_to(tmp_path / "secret.tif")
    assert _image_path(root, "sfs/dem.tif") == (root / "sfs" / "dem.tif").resolve()
    assert _image_path(root, "sfs%2Fdem.tif") == (root / "sfs" / "dem.tif").resolve()
    for bad in (
        "",
        "../secret.tif",
        "sfs/../../secret.tif",
        "%2E%2E/secret.tif",
        "/etc/passwd",
        "notes.txt",
        ".hidden.tif",
        "sfs//dem.tif",
        "link.tif",
        "sfs",
        "missing.tif",
        "..%5Csecret.tif",
    ):
        assert _image_path(root, bad) is None, bad
    assert _list_images(root) == ["sfs/dem.tif"]


def test_relay_only_pds_image_files():
    ok = "https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/x/M1_pyr.tif"
    assert _relay_allowed(ok)
    assert _relay_allowed("https://data.darts.isas.jaxa.jp/pub/pds3/x/browse/large/TC1.jpg")
    assert _relay_allowed("https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_20MPP_ADJ.TIF")
    for bad in (
        "https://pgda.gsfc.nasa.gov/other/a.tif",
        ok.replace("https:", "http:"),
        "https://example.com/a.tif",
        "https://pds.mcp.nasa.gov.evil.com/a.tif",
        "https://user@pds.mcp.nasa.gov/a.tif",
        "https://pds.mcp.nasa.gov:8443/a.tif",
        "https://pds.mcp.nasa.gov:x/a.tif",
        "https://pds.mcp.nasa.gov/data/a.IMG",
        "file:///etc/passwd",
        "",
    ):
        assert not _relay_allowed(bad), bad


def test_settings_view_and_job_api_in_page():
    for needle in (
        "dockview-core@8.4.0",
        'id="view-settings"',
        'id="jobs-summary"',
        "api/jobs/events",
        "api/status",
    ):
        assert needle in PAGE, needle
    assert PAGE.isascii()
