"""PDS ODE REST client (ShadowCam, LROC NAC, Kaguya TC) returning the observation schema."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import modernize_pds_url
from .downloader import _USER_AGENT, PdsDownloader
from .geometry import aoi_mask, aoi_region, load_aoi, query_region, region_bbox
from .observation import OBSERVATION_COLUMNS, normalize_observations

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

log = logging.getLogger(__name__)

ODE_REST_URL = "https://oderest.rsl.wustl.edu/live2/"

# Instrument registry: CLI/API key → ODE instrument host id, instrument id and
# friendly product-type aliases (raw ODE ``pt`` values are accepted too).
INSTRUMENTS: dict[str, dict[str, Any]] = {
    "shadowcam": {
        "label": "ShadowCam",
        "ihid": "KPLO",
        "iid": "ShadowCam",
        "product_types": {"raw": "RAW", "cal": "CAL", "cmos": "CMOS", "umos": "UMOS", "dtm": "DTM"},
        "default": "cal",
    },
    "lroc": {
        "label": "LROC NAC",
        "ihid": "LRO",
        "iid": "LROC",
        "product_types": {"edr": "EDRNAC4", "cdr": "CDRNAC4"},
        "default": "edr",
    },
    "kaguya_tc": {
        "label": "Kaguya TC",
        "ihid": "SLN",
        "iid": "TC",
        "product_types": {
            "stereo": "TCWL2B",  # Stereo Level 2B0
            "mono": "TCSL2B",  # Mono Level 2B0
            "dtm": "TCDTMO",  # DTM ortho
            "ortho": "TCORT",  # Ortho map
            "morning": "TCMORM",
            "evening": "TCEVEM",
        },
        "default": "stereo",
    },
    "kaguya_mi": {
        "label": "Kaguya MI",
        "ihid": "SLN",
        "iid": "MI",
        "product_types": {
            "vis": "VIS2B",  # VIS Level 2B2 (kaguyami2isis)
            "nir": "NIR2B",  # NIR Level 2B2 (kaguyami2isis)
            "vis2c": "VIS2C",  # VIS Level 2C2
            "nir2c": "NIR2C",  # NIR Level 2C2
            "l3c5": "MI3C",  # Level 3C5 reflectance, map-projected
            "map": "MIMAP",  # Map tiles
        },
        "default": "vis",
    },
}

_PRODUCT_QUERY = {"query": "product", "output": "JSON", "target": "moon"}

_ALIASES = {
    "sc": "shadowcam",
    "kplo": "shadowcam",
    "nac": "lroc",
    "lroc_nac": "lroc",
    "kaguya": "kaguya_tc",
    "tc": "kaguya_tc",
    "selene": "kaguya_tc",
    "mi": "kaguya_mi",
}


def resolve_instrument(instrument: str, product_type: str | None = None) -> tuple[str, dict, str]:
    """Return ``(key, registry_entry, ode_pt)`` for user-facing names."""
    key = instrument.strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument {instrument!r}. Valid: {sorted(INSTRUMENTS)}")
    entry = INSTRUMENTS[key]
    pt_key = (product_type or entry["default"]).strip()
    pts = entry["product_types"]
    if pt_key.lower() in pts:
        return key, entry, pts[pt_key.lower()]
    if pt_key.upper() in {v.upper() for v in pts.values()} or product_type:
        return key, entry, pt_key.upper()
    raise ValueError(f"Unknown product type {product_type!r} for {key}: {sorted(pts)}")


class OdeError(RuntimeError):
    """ODE returned an error status."""


class OdeClient:
    """Query ODE REST and return `OBSERVATION_COLUMNS` DataFrames.

    Parameters
    ----------
    base_url : str
        ODE REST endpoint (default: ``live2``).
    timeout : float
        Per-request timeout in seconds — ODE footprint queries can take
        30 s or more.
    page_size : int
        Products per request (ODE ``limit``); results are paged with ``offset``.
    """

    def __init__(
        self,
        base_url: str = ODE_REST_URL,
        timeout: float = 180.0,
        page_size: int = 100,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.page_size = page_size
        self.retries = retries

    # Low-level request

    def build_params(
        self,
        instrument: str,
        product_type: str | None = None,
        lat: tuple[float, float] | None = None,
        lon: tuple[float, float] | None = None,
        point: tuple[float, float] | None = None,
        radius_km: float = 0.0,
        incidence: tuple[float, float] | None = None,
        emission: tuple[float, float] | None = None,
        phase: tuple[float, float] | None = None,
        time_range: tuple[str, str] | None = None,
        loc: str | None = None,
        results: str = "fmp",
        aoi=None,
    ) -> dict[str, str]:
        """Translate search arguments into ODE query parameters.

        An ``aoi`` (see [`lunar_toolkit.geometry.load_aoi`][lunar_toolkit.geometry.load_aoi]) is sent as
        its bounding box; [`search`][lunar_toolkit.ode.OdeClient.search] refines the results exactly.
        """
        _, entry, pt = resolve_instrument(instrument, product_type)
        params: dict[str, str] = {
            **_PRODUCT_QUERY,
            "results": results,
            "ihid": entry["ihid"],
            "iid": entry["iid"],
            "pt": pt,
        }
        if aoi is not None or point is not None or lat is not None or lon is not None:
            if aoi is not None:
                if point is not None or lat is not None or lon is not None:
                    raise ValueError("Pass either aoi or lat/lon/point, not both")
                region = aoi_region(aoi)
            else:
                region = query_region(lat=lat, lon=lon, point=point, radius_km=radius_km)
            west, south, east, north = region_bbox(region)
            params.update(
                {
                    "minlat": _fmt(south),
                    "maxlat": _fmt(north),
                    # ODE: positive east 0–360, boxes may cross the meridian
                    "westernlon": _fmt(west),
                    "easternlon": _fmt(east),
                }
            )
        for (lo, hi), rng in (
            (("mininangle", "maxinangle"), incidence),
            (("minemangle", "maxemangle"), emission),
            (("minphangle", "maxphangle"), phase),
        ):
            if rng is not None:
                params[lo], params[hi] = _fmt(rng[0]), _fmt(rng[1])
        if time_range is not None:
            params["minobtime"], params["maxobtime"] = str(time_range[0]), str(time_range[1])
        if loc:
            params["loc"] = loc
        return params

    def request(self, params: dict[str, str]) -> dict:
        """GET the ODE endpoint and return the decoded ``ODEResults`` dict."""
        url = self.base_url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8", "replace"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        else:
            raise OdeError(f"ODE request failed: {last}") from last
        res = data.get("ODEResults", data)
        if str(res.get("Status", "")).lower() == "error":
            raise OdeError(res.get("Error") or json.dumps(res)[:500])
        return res

    # Search

    def search(
        self,
        instrument: str,
        product_type: str | None = None,
        lat: tuple[float, float] | None = None,
        lon: tuple[float, float] | None = None,
        point: tuple[float, float] | None = None,
        radius_km: float = 0.0,
        incidence: tuple[float, float] | None = None,
        emission: tuple[float, float] | None = None,
        phase: tuple[float, float] | None = None,
        time_range: tuple[str, str] | None = None,
        max_results: int | None = 1000,
        loc: str | None = None,
        aoi=None,
    ) -> pd.DataFrame:
        """Search ODE and return the common observation schema.

        Region arguments follow [`CumIndex.search`][lunar_toolkit.lroc.cumindex.CumIndex.search]:
        ``lat=(min, max)`` and ``lon=(west, east)`` (``west > east`` crosses 0/360), or
        ``point=(lat, lon)`` with ``radius_km`` (converted to a bounding box;
        circles reaching a pole become a full-longitude cap).  ODE matches the
        box against product footprints (``loc=f``, its default).  With
        ``aoi`` (GeoJSON / WKT path, WKT string or
        [`Aoi`][lunar_toolkit.geometry.Aoi]) the AOI's
        bounding box is queried and each ``footprint_wkt`` is then
        intersected with the polygon; paging continues until
        ``max_results`` products survive that refinement.

        Extra columns: ``files`` (list of ODE product-file dicts) and
        ``ode_url`` (ODE product page).
        """
        key, entry, _ = resolve_instrument(instrument, product_type)
        if aoi is not None:
            aoi = load_aoi(aoi)
        params = self.build_params(
            instrument,
            product_type,
            lat=lat,
            lon=lon,
            point=point,
            radius_km=radius_km,
            incidence=incidence,
            emission=emission,
            phase=phase,
            time_range=time_range,
            loc=loc,
            aoi=aoi,
        )
        records: list[dict] = []
        offset = 0
        while True:
            limit = self.page_size
            if max_results is not None:
                remaining = max_results - len(records)
                if remaining <= 0:
                    break
                if aoi is None:
                    limit = min(limit, remaining)
            res = self.request({**params, "limit": str(limit), "offset": str(offset)})
            products = extract_products(res)
            page = [parse_product(p, instrument_label=entry["label"]) for p in products]
            if aoi is not None and page:
                page = [r for r, keep in zip(page, aoi_mask(to_dataframe(page), aoi)) if keep]
            records += page
            if len(products) < limit:
                break
            offset += len(products)
        if max_results is not None:
            records = records[:max_results]
        return to_dataframe(records)

    def lookup(self, product_ids: Iterable[str], instrument_label: str | None = None) -> pd.DataFrame:
        """Observations for exact product IDs, one ODE ``productid`` request each.

        Unknown IDs are skipped with a warning.
        """
        records: list[dict] = []
        for pid in product_ids:
            pid = pid.strip().upper()
            products = extract_products(self.request({**_PRODUCT_QUERY, "results": "fmp", "productid": pid}))
            if not products:
                log.warning("ODE has no product %s", pid)
            records += [parse_product(p, instrument_label=instrument_label) for p in products]
        return to_dataframe(records)

    def count(self, instrument: str, product_type: str | None = None, **kwargs) -> int:
        """Number of matching products (``results=c``)."""
        params = self.build_params(instrument, product_type, results="c", **kwargs)
        res = self.request(params)
        return int(res.get("Count", 0) or 0)

    # Download

    def download(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        kind: str = "product",
        match: str | None = None,
        workers: int = 4,
        force: bool = False,
    ) -> dict[str, Path | None]:
        """Download files listed in the ``files`` column of a search result.

        Parameters
        ----------
        kind : str
            ODE file ``Type`` to select: ``"product"`` (default; data files
            only, labels skipped), ``"label"``, ``"browse"``, ``"referenced"``,
            ``"derived"`` or ``"all"``.
        match : str, optional
            Case-insensitive substring the file name must contain
            (e.g. ``"_cog.tif"``).
        force : bool
            Download again even when the file already exists in *output_dir*.
        """
        urls = select_file_urls(df, kind=kind, match=match)
        if not urls:
            log.warning("No files selected for download.")
            return {}
        log.info("Downloading %d file(s) to %s", len(urls), output_dir)
        return PdsDownloader(max_workers=workers).download_urls(urls, Path(output_dir), force=force)


# Response parsing (pure functions — unit-tested against saved JSON)


def extract_products(res: dict) -> list[dict]:
    """Return the product list from an ``ODEResults`` dict (handles 0/1/N)."""
    prods = res.get("Products")
    if not isinstance(prods, dict):  # "No Products Found" string or missing
        return []
    p = prods.get("Product")
    if p is None:
        return []
    return p if isinstance(p, list) else [p]


def _files(prod: dict) -> list[dict]:
    pf = (prod.get("Product_files") or {}).get("Product_file") or []
    return pf if isinstance(pf, list) else [pf]


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_product(prod: dict, instrument_label: str | None = None) -> dict:
    """Map one ODE product dict to the observation schema (+ extras)."""
    pdsid = str(prod.get("pdsid", "")).strip()
    product_id = pdsid.rsplit(".", 1)[-1].upper() if pdsid else None
    files = _files(prod)
    for f in files:
        if f.get("URL"):
            f["URL"] = modernize_pds_url(f["URL"])

    data_files = [f for f in files if str(f.get("Type", "")).lower() == "product"]
    url = next(
        (f["URL"] for f in data_files if not str(f.get("FileName", "")).upper().endswith((".XML", ".LBL"))),
        None,
    )
    label_url = prod.get("LabelURL") or next(
        (f["URL"] for f in data_files if str(f.get("FileName", "")).upper().endswith((".XML", ".LBL"))),
        None,
    )
    if label_url:
        label_url = modernize_pds_url(label_url)

    sub_solar_az = None
    for k in ("Sub_solar_azimuth", "Subsolar_azimuth", "Solar_azimuth"):
        if k in prod:
            sub_solar_az = _num(prod[k])
            break

    # ODE echoes the LROC CUMINDEX row in ODE_notes ("NOTE: Index Record:"
    # followed by the CSV record).  Its 83 columns follow CUMINDEX.LBL, so
    # 0-based index = COLUMN_NUMBER - 1; 999.99-style N/A constants → None.
    geom: dict[str, float | None] = {}
    notes = (prod.get("ODE_notes") or {}).get("ODE_note") or []
    notes = notes if isinstance(notes, list) else [notes]
    for i, note in enumerate(notes[:-1]):
        if str(note).startswith("NOTE: Index Record"):
            import csv
            import io

            row = next(csv.reader(io.StringIO(str(notes[i + 1]))), [])
            if len(row) == 83:
                for key, col in (
                    ("slew_angle", 9),
                    ("north_azimuth", 61),
                    ("sub_solar_azimuth", 62),
                    ("sub_spacecraft_lat", 65),
                    ("sub_spacecraft_lon", 66),
                    ("spacecraft_altitude", 79),
                ):
                    v = _num(row[col].strip())
                    geom[key] = None if v is None or abs(v) >= 999.0 else v
            break
    if sub_solar_az is None:
        sub_solar_az = geom.get("sub_solar_azimuth")

    return {
        "product_id": product_id,
        "instrument": instrument_label or f"{prod.get('ihid', '')} {prod.get('iid', '')}".strip(),
        "product_type": prod.get("pt"),
        "start_time": prod.get("UTC_start_time") or prod.get("Observation_time"),
        "center_lat": _num(prod.get("Center_latitude")),
        "center_lon": _num(prod.get("Center_longitude")),
        "min_lat": _num(prod.get("Minimum_latitude")),
        "max_lat": _num(prod.get("Maximum_latitude")),
        "lon_west": _num(prod.get("Westernmost_longitude")),
        "lon_east": _num(prod.get("Easternmost_longitude")),
        "footprint_wkt": prod.get("Footprint_geometry") or prod.get("Footprint_C0_geometry"),
        "incidence": _num(prod.get("Incidence_angle")),
        "emission": _num(prod.get("Emission_angle")),
        "phase": _num(prod.get("Phase_angle")),
        "slew_angle": geom.get("slew_angle"),
        "sub_solar_azimuth": sub_solar_az,
        "north_azimuth": geom.get("north_azimuth"),
        "sub_spacecraft_lat": geom.get("sub_spacecraft_lat"),
        "sub_spacecraft_lon": geom.get("sub_spacecraft_lon"),
        "spacecraft_altitude": geom.get("spacecraft_altitude"),
        "pixel_scale": _num(prod.get("Map_resolution") or prod.get("Map_scale")),
        "url": url,
        "label_url": label_url,
        "source": "ode",
        "files": files,
        "ode_url": prod.get("ProductURL"),
    }


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Observation DataFrame with the ``files`` / ``ode_url`` extras appended."""
    df = normalize_observations(records)
    df["files"] = [r.get("files", []) for r in records] if records else []
    df["ode_url"] = [r.get("ode_url") for r in records] if records else []
    return df[OBSERVATION_COLUMNS + ["files", "ode_url"]]


def select_file_urls(df: pd.DataFrame, kind: str = "product", match: str | None = None) -> list[str]:
    """Pick file URLs from the ``files`` column (see
    [`OdeClient.download`][lunar_toolkit.ode.OdeClient.download])."""
    kind = kind.lower()
    urls: list[str] = []
    for files in df.get("files", []):
        for f in files or []:
            name = str(f.get("FileName", ""))
            ftype = str(f.get("Type", "")).lower()
            is_label = name.upper().endswith((".XML", ".LBL"))
            if kind == "label":
                ok = ftype == "product" and is_label
            elif kind == "product":
                ok = ftype == "product" and not is_label
            elif kind == "all":
                ok = True
            else:
                ok = ftype == kind
            if ok and match and match.lower() not in name.lower():
                ok = False
            if ok and f.get("URL") and f["URL"] not in urls:
                urls.append(f["URL"])
    return urls


def _fmt(x: float) -> str:
    return f"{float(x):.6g}"
