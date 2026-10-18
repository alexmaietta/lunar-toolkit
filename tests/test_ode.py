"""ODE REST client: parameter building and parsing of saved responses."""

import json

import pytest
from conftest import FIXTURES

from lunar_toolkit.observation import OBSERVATION_COLUMNS
from lunar_toolkit.ode import (
    OdeClient,
    extract_products,
    parse_product,
    resolve_instrument,
    select_file_urls,
    to_dataframe,
)


def _load(name, label):
    res = json.loads((FIXTURES / name).read_text())["ODEResults"]
    return to_dataframe([parse_product(p, label) for p in extract_products(res)])


def test_parse_lroc_edr():
    df = _load("lroc_edr.json", "LROC NAC")
    assert list(df.columns[: len(OBSERVATION_COLUMNS)]) == OBSERVATION_COLUMNS
    assert list(df["product_id"]) == ["M1402266078LE", "M1402266078RE"]
    r = df.iloc[0]
    assert r.product_type == "EDRNAC4"
    assert r.incidence == pytest.approx(10.92)
    assert r.pixel_scale == pytest.approx(0.82)
    assert r.lon_west == pytest.approx(0.32)
    # legacy LROC host rewritten to the PDS store
    assert r.url.startswith("https://pds.mcp.nasa.gov/") and r.url.endswith("M1402266078LE.IMG")
    assert r.label_url.endswith("M1402266078LE.xml")
    assert r.source == "ode"


def test_parse_shadowcam_cal():
    df = _load("sc_cal.json", "ShadowCam")
    assert len(df) == 3
    r = df.iloc[0]
    assert r.product_id == "M013142801SC"
    assert r.instrument == "ShadowCam"
    assert r.product_type == "CAL"
    assert r.center_lat == pytest.approx(-89.4531)
    assert r.incidence == pytest.approx(87.948)
    assert r.pixel_scale == pytest.approx(1.8025)
    assert r.lon_west > r.lon_east  # footprint crosses 0/360
    assert r.url.endswith("M013142801SC.cub")
    assert r.label_url.endswith("M013142801SC.xml")
    assert str(r.start_time).startswith("2023-01-04 01:54:42")


def test_select_files():
    df = _load("sc_cal.json", "ShadowCam")
    assert all(u.endswith(".cub") for u in select_file_urls(df, "product"))
    assert all(u.endswith(".xml") for u in select_file_urls(df, "label"))
    cogs = select_file_urls(df, "browse", match="_cog.tif")
    assert len(cogs) == 3


def test_extract_products_shapes():
    assert extract_products({"Products": "No Products Found"}) == []
    assert len(extract_products({"Products": {"Product": {"pdsid": "x"}}})) == 1


def test_resolve_instrument():
    assert resolve_instrument("shadowcam")[2] == "CAL"
    assert resolve_instrument("ShadowCam", "dtm")[2] == "DTM"
    assert resolve_instrument("lroc", "cdr")[2] == "CDRNAC4"
    assert resolve_instrument("kaguya")[1]["ihid"] == "SLN"
    with pytest.raises(ValueError):
        resolve_instrument("hirise")


def test_build_params_bbox_and_angles():
    p = OdeClient().build_params("shadowcam", "cal", lat=(-90, -89), lon=(350, 10), incidence=(80, 90))
    assert p["ihid"] == "KPLO" and p["iid"] == "ShadowCam" and p["pt"] == "CAL"
    assert (p["westernlon"], p["easternlon"]) == ("350", "10")
    assert (p["minlat"], p["maxlat"]) == ("-90", "-89")
    assert (p["mininangle"], p["maxinangle"]) == ("80", "90")
    assert p["query"] == "product" and p["output"] == "JSON" and p["target"] == "moon"


def test_build_params_point_over_pole():
    p = OdeClient().build_params("shadowcam", point=(-89.7, 129.8), radius_km=20)
    assert (p["westernlon"], p["easternlon"]) == ("0", "360")
    assert p["minlat"] == "-90"


def test_search_pages(monkeypatch):
    res = json.loads((FIXTURES / "sc_cal.json").read_text())["ODEResults"]
    calls = []

    def fake_request(self, params):
        calls.append(params)
        return res if params["offset"] == "0" else {"Products": "No Products Found"}

    monkeypatch.setattr(OdeClient, "request", fake_request)
    df = OdeClient(page_size=3).search("shadowcam", lat=(-90, -89), max_results=10)
    assert len(df) == 3
    assert [c["offset"] for c in calls] == ["0", "3"]


@pytest.mark.network
def test_live_shadowcam_shackleton():
    df = OdeClient().search("shadowcam", "cal", point=(-89.67, 129.8), radius_km=5, max_results=20)
    assert len(df) > 0
