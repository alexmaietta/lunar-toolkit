"""Reference data for the Artemis III candidate regions: PGDA LOLA 5 m site DEMs and Bertone et al. SfS DEMs."""

from __future__ import annotations

import fnmatch
import io
import json
import logging
import math
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from .constants import APP_NAME, APP_VERSION, DEFAULT_CACHE_DIR, MOON_RADIUS_KM
from .csm import CsmCamera, read_camera, transform_camera
from .downloader import download_file, url_filename
from .geometry import Aoi, load_aoi, ps_xy
from .tools import reuse_existing

log = logging.getLogger(__name__)

PS_SRS = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
"""South polar stereographic on the 1737.4 km sphere (PGDA and Bertone et al. rasters)."""

ASP_CSV_FORMAT = "1:easting 2:northing 3:height_above_datum"
"""``--csv-format`` for :func:`write_asp_csv` output (use with ``--csv-srs`` :data:`PS_SRS`)."""

ZENODO_RECORD = 17954508
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ZENODO_FILE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{{name}}?download=1"

PGDA_PRODUCT_PAGE = "https://pgda.gsfc.nasa.gov/products/78"
PGDA_ROOT = "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/"
PGDA_KINDS = ("surf", "ldec", "toterr", "slp", "slperr", "xyzi", "clones", "rois")

REFERENCE_CACHE_DIR = DEFAULT_CACHE_DIR / "reference"

_USER_AGENT = f"{APP_NAME}/{APP_VERSION} (python-urllib)"
_RETRY_CODES = (403, 429, 500, 502, 503, 504)  # PGDA answers 403 to bursts of requests


@dataclass(frozen=True)
class PsBox:
    """Axis-aligned box in south polar stereographic metres (:data:`PS_SRS`).

    Parameters
    ----------
    x0, y0, x1, y1 : float
        West, south, east and north edges (``x0 < x1``, ``y0 < y1``).
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)``."""
        return self.x0, self.y0, self.x1, self.y1

    @property
    def projwin(self) -> tuple[float, float, float, float]:
        """GDAL / ASP ``projwin`` order: ``(ulx, uly, lrx, lry)``."""
        return self.x0, self.y1, self.x1, self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_xy(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    @property
    def center_latlon(self) -> tuple[float, float]:
        """Centre as ``(lat, lon)`` degrees, lon in −180..180."""
        return ps_latlon(*self.center_xy)

    def contains(self, other: PsBox) -> bool:
        return self.x0 <= other.x0 and self.y0 <= other.y0 and other.x1 <= self.x1 and other.y1 <= self.y1

    def intersects(self, other: PsBox) -> bool:
        return self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1

    def ring_lonlat(self, step_m: float = 100.0) -> list[tuple[float, float]]:
        """Closed ``(lon, lat)`` ring of the box edges, sampled every *step_m* metres."""
        if self.x0 < 0.0 < self.x1 and self.y0 < 0.0 < self.y1:
            raise ValueError("Box contains the pole; it has no simple lon/lat outline")
        corners = [(self.x0, self.y0), (self.x1, self.y0), (self.x1, self.y1), (self.x0, self.y1)]
        pts: list[tuple[float, float]] = []
        for (xa, ya), (xb, yb) in zip(corners, corners[1:] + corners[:1]):
            n = max(1, math.ceil(math.hypot(xb - xa, yb - ya) / step_m))
            pts += [(xa + (xb - xa) * i / n, ya + (yb - ya) * i / n) for i in range(n)]
        ring = [ps_latlon(x, y)[::-1] for x, y in pts]
        return ring + ring[:1]

    def aoi(self, step_m: float = 100.0) -> Aoi:
        """The box as a lon/lat :class:`~lunar_toolkit.geometry.Aoi` (edges straight in PS).

        Parameters
        ----------
        step_m : float
            Vertex spacing along the edges; the AOI matches the box to well
            under a metre once :mod:`~lunar_toolkit.geometry` projects it back.
        """
        geom = {"type": "Polygon", "coordinates": [self.ring_lonlat(step_m)]}
        aoi = load_aoi(geom, densify_deg=None)
        aoi.source = f"PS box {self.bounds}"
        return aoi

    def wkt(self, step_m: float = 100.0) -> str:
        """``POLYGON`` WKT of :meth:`ring_lonlat` (lon/lat degrees)."""
        return "POLYGON((" + ", ".join(f"{lo:.7f} {la:.7f}" for lo, la in self.ring_lonlat(step_m)) + "))"

    def wkt_ps(self) -> str:
        """``POLYGON`` WKT of the corners in PS metres."""
        x0, y0, x1, y1 = self.bounds
        return f"POLYGON(({x0:g} {y0:g}, {x1:g} {y0:g}, {x1:g} {y1:g}, {x0:g} {y1:g}, {x0:g} {y0:g}))"


def ps_latlon(x: float, y: float) -> tuple[float, float]:
    """Inverse of south-hemisphere :func:`~lunar_toolkit.geometry.ps_xy`: ``(lat, lon)`` degrees."""
    rho = math.hypot(x, y)
    lat = math.degrees(2.0 * math.atan(rho / (2.0 * MOON_RADIUS_KM * 1000.0))) - 90.0
    return lat, math.degrees(math.atan2(x, y))


@dataclass(frozen=True)
class PgdaSite:
    """A PGDA LOLA 5 m/px site DEM (Barker et al. 2021): name, description and PS extent."""

    name: str
    description: str
    extent: PsBox

    @property
    def url(self) -> str:
        return f"{PGDA_ROOT}{self.name}/"


# Extents read from each site's *_surf.tif (all 5 m/px, PS_SRS, float32, NaN nodata).
PGDA_SITES: dict[str, PgdaSite] = {
    s[0]: PgdaSite(s[0], s[1], PsBox(*s[2]))
    for s in [
        ("Site01", "Connecting ridge", (-19000, -20000, -3000, -4000)),
        ("Site04", "Shackleton rim", (-9000, -15000, 7000, 1000)),
        ("Site06", "Nobile rim 1", (74000, 100000, 94000, 120000)),
        ("Site07", "Peak near Shackleton", (22000, -28000, 38000, -12000)),
        ("Site11", "de Gerlache rim", (-45000, 7000, -29000, 23000)),
        ("Site20", "Leibnitz beta plateau", (65000, 110000, 81000, 126000)),
        ("Site20v2", "Leibnitz beta plateau, extended boundaries", (60000, 105000, 86000, 131000)),
        ("Site23", "Malapert massif", (-11000, 111000, 10000, 132000)),
        ("Site42", "de Gerlache-Kocher massif", (-123400, -66100, -103400, -46100)),
        ("Haworth", "Haworth", (-52900, 75600, -23100, 105400)),
        ("Shoemaker", "Shoemaker", (66000, 29000, 86000, 49000)),
        ("DM1", "Amundsen rim", (154000, 51500, 174000, 71500)),
        ("DM2", "Nobile rim 2", (147000, 85000, 167000, 105000)),
        ("SL2", "de Gerlache rim 2", (-58500, 13000, -38500, 33000)),
        ("SL3", "Connecting ridge extension", (-38700, -16100, -18700, 3900)),
        ("NPA", "Cabeus exterior wall 1", (-98500, 19500, -78500, 39500)),
        ("NPB", "Amundsen 1", (175000, -8500, 195000, 11500)),
        ("NPC", "Idel'son L crater 1", (122500, -109500, 142500, -89500)),
        ("NPD", "Malapert crater 1", (13000, 146000, 33000, 166000)),
        ("LM1", "Shackleton Rim B", (2000, -2000, 22000, 18000)),
        ("LM2", "Shoemaker Rim A", (2000, 47000, 22000, 67000)),
        ("LM3", "Shoemaker Rim B", (4000, 37000, 24000, 57000)),
        ("LM4", "Shoemaker Rim C", (9000, 15000, 29000, 35000)),
        ("LM5", "Shoemaker Rim D", (30000, 1000, 50000, 21000)),
        ("LM6", "Shoemaker Rim E", (49000, 10000, 69000, 30000)),
        ("LM7", "Faustini Rim A", (54000, -10000, 74000, 10000)),
        ("LM8", "Shoemaker Rim F", (60000, 30000, 80000, 50000)),
    ]
}


@dataclass(frozen=True)
class Region:
    """A 2022 Artemis III candidate landing region as processed by Bertone et al. (2026).

    ``bbox`` (``min_lon, min_lat, max_lon, max_lat``) and ``center``
    (``lon, lat``, the bbox midpoint) come from the Zenodo README; ``roi`` is
    the polar stereographic square the SfS DEM covers, which is the extent of
    the matching PGDA LOLA 5 m site DEM (``pgda_site``).
    """

    id: int
    name: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    pgda_site: str | None
    zenodo_file: str
    zenodo_size: int
    zenodo_md5: str

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

    @property
    def roi(self) -> PsBox:
        if self.pgda_site is None:
            return bbox_ps_box(self.bbox)
        return PGDA_SITES[self.pgda_site].extent

    @property
    def projwin(self) -> tuple[float, float, float, float]:
        """``(ulx, uly, lrx, lry)`` of :attr:`roi` in PS metres."""
        return self.roi.projwin

    @property
    def bbox_projwin(self) -> tuple[float, float, float, float]:
        """``projwin`` of the PS envelope of the lon/lat :attr:`bbox` (a wedge; larger than :attr:`roi`)."""
        return bbox_ps_box(self.bbox).projwin

    @property
    def zenodo_url(self) -> str:
        return ZENODO_FILE_URL.format(name=self.zenodo_file)

    def aoi(self, step_m: float = 100.0) -> Aoi:
        """:attr:`roi` as an :class:`~lunar_toolkit.geometry.Aoi`."""
        aoi = self.roi.aoi(step_m)
        aoi.source = f"A3CLR22 #{self.id} {self.name}"
        return aoi

    def bbox_aoi(self, densify_deg: float = 0.05) -> Aoi:
        """The lon/lat :attr:`bbox` as an AOI, densified so it stays a wedge in PS."""
        w, s, e, n = self.bbox
        aoi = load_aoi(f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))", densify_deg=densify_deg)
        aoi.source = f"A3CLR22 #{self.id} {self.name} bbox"
        return aoi


# Zenodo README table + file sizes / MD5s from the record API (id, label, center lon/lat,
# bbox min lon/lat max lon/lat, PGDA site, zip, size, md5).
_REGION_ROWS = [
    (0, "Amundsen Rim", (69.303844, -84.225586), (65.095231, -84.648844, 73.512457, -83.802328), "DM1",
     "A3CLR22_0_Amundsen_Rim.zip", 1775016989, "d1e12877277cf9715381b8795c0873cb"),
    (1, "Nobile Rim 2", (58.743523, -83.953082), (54.462322, -84.404616, 63.024724, -83.501549), "DM2",
     "A3CLR22_1_Nobile_Rim_2.zip", 1839070131, "964310cb7f2a5033e2e12cd34e7aa7b1"),
    (2, "Haworth", (-23.671814, -86.752966), (-34.981846, -87.393534, -12.361782, -86.112398), "Haworth",
     "A3CLR22_2_Haworth.zip", 4353812518, "e83a923c41c00be51396d06ab7cf6e9e"),
    (3, "Faustini Rim A", (90.0, -87.863451), (79.508523, -88.189067, 100.491477, -87.537834), "LM7",
     "A3CLR22_3_Faustini_Rim_A.zip", 1936729230, "161db71236036f7c9b98b501d353482a"),
    (4, "de Gerlache Rim 2", (-63.434949, -88.222637), (-77.471192, -88.659986, -49.398705, -87.785288), "SL2",
     "A3CLR22_4_deGerlache_Rim_2.zip", 2187384204, "52921a134793a6b35b0eea6f98ad79d8"),
    (5, "Connecting Ridge Extension", (-104.473348, -88.993919), (-130.727219, -89.370051, -78.219476, -88.617786),
     "SL3", "A3CLR22_5_Connecting_Ridge_Extension.zip", 2225897760, "9141ec5d298a3aebab4d3671f7f38a79"),
    (6, "Connecting Ridge", (-136.678946, -89.462697), (-171.469234, -89.835111, -101.888658, -89.090284),
     "Site01", "A3CLR22_6_Connecting_Ridge.zip", 2206485514, "0ad982cf42319a300f2a2cff69d39322"),
    (7, "Nobile Rim 1", (37.444641, -85.43775), (31.660751, -85.899218, 43.22853, -84.976283), "Site06",
     "A3CLR22_7_Nobile_Rim_1.zip", 1962014939, "a8bcb60f432253f5fa97ff07899bd02a"),
    (8, "Peak Near Shackleton", (124.684171, -88.808537), (107.525568, -89.173591, 141.842773, -88.443484),
     "Site07", "A3CLR22_8_Peak_Near_Shackleton.zip", 1360387336, "9ac25cc18b1290e679dfad9b1eaabad7"),
    (9, "de Gerlache Rim", (-66.370065, -88.674855), (-81.158185, -89.016199, -51.581945, -88.33351), "Site11",
     "A3CLR22_9_deGerlache_Rim.zip", 1340667387, "a34e603e9d2c2c9a2796fc237d24b1b4"),
    (10, "Leibnitz Beta Plateau", (31.827284, -85.425819), (27.287988, -85.788335, 36.36658, -85.063304),
     "Site20", "A3CLR22_10_Leibnitz_Beta_Plateau.zip", 1050441742, "b0a606e6a4ce73faf2ea68c91ce0b93e"),
    (11, "Malapert Massif", (-0.255799, -85.979917), (-5.659482, -86.325889, 5.147885, -85.633944), "Site23",
     "A3CLR22_11_Malapert_Massif.zip", 2213137537, "772a578b8f2ebdb078c5a04a3c07182a"),
    (12, "de Gerlache-Kocher Massif", (-116.537066, -85.826913), (-122.589357, -86.267855, -110.484776, -85.385972),
     "Site42", "A3CLR22_12_deGerlache_Kocher_Massif.zip", 2108633565, "78b8a042d0c6be2923abdd288e436c23"),
]  # fmt: skip

REGIONS: tuple[Region, ...] = tuple(
    Region(id=i, name=name, center=c, bbox=b, pgda_site=site, zenodo_file=f, zenodo_size=size, zenodo_md5=md5)
    for i, name, c, b, site, f, size, md5 in _REGION_ROWS
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def get_region(key: Region | int | str) -> Region:
    """Look up a region by id, name, slug or PGDA site (case-insensitive).

    ``11``, ``"11"``, ``"Malapert Massif"``, ``"malapert-massif"``,
    ``"malapert"`` and ``"Site23"`` all give region 11.  A partial name must
    match exactly one region.
    """
    if isinstance(key, Region):
        return key
    if isinstance(key, int) or str(key).strip().isdigit():
        i = int(key)
        for r in REGIONS:
            if r.id == i:
                return r
        raise KeyError(f"No region with id {i} (0–{len(REGIONS) - 1})")
    k = _norm(str(key))
    if not k:
        raise KeyError("Empty region key")
    for r in REGIONS:
        if k in (
            _norm(r.name),
            _norm(r.slug),
            _norm(r.pgda_site or ""),
            _norm(r.zenodo_file.removesuffix(".zip")),
        ):
            return r
    hits = [r for r in REGIONS if k and k in _norm(r.name)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise KeyError(f"{key!r} is ambiguous: " + ", ".join(f"{r.id} {r.name}" for r in hits))
    raise KeyError(f"No region matches {key!r}; known: " + ", ".join(r.name for r in REGIONS))


def regions_table() -> pd.DataFrame:
    """All regions as a table (one row per region)."""
    rows = []
    for r in REGIONS:
        rows.append(
            {
                "id": r.id,
                "name": r.name,
                "slug": r.slug,
                "center_lon": r.center[0],
                "center_lat": r.center[1],
                "min_lon": r.bbox[0],
                "min_lat": r.bbox[1],
                "max_lon": r.bbox[2],
                "max_lat": r.bbox[3],
                "pgda_site": r.pgda_site,
                "roi_projwin": r.projwin,
                "zenodo_file": r.zenodo_file,
                "zenodo_size": r.zenodo_size,
                "zenodo_md5": r.zenodo_md5,
            }
        )
    return pd.DataFrame(rows)


def bbox_ps_box(bbox: Sequence[float], samples: int = 200) -> PsBox:
    """PS envelope of a lon/lat ``(min_lon, min_lat, max_lon, max_lat)`` box (south)."""
    w, s, e, n = bbox
    pts = []
    for i in range(samples + 1):
        f = i / samples
        lo, la = w + (e - w) * f, s + (n - s) * f
        pts += [ps_xy(s, lo, "S"), ps_xy(n, lo, "S"), ps_xy(la, w, "S"), ps_xy(la, e, "S")]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return PsBox(min(xs), min(ys), max(xs), max(ys))


def crop_window(
    region: Region | int | str,
    size_km: float = 2.5,
    center: tuple[float, float] | None = None,
    snap_m: float | None = 5.0,
) -> PsBox:
    """Square PS window of *size_km* centred on a region.

    Parameters
    ----------
    region : Region, int or str
        Anything :func:`get_region` accepts.
    size_km : float
        Side length (2.5 km is the Bertone et al. uncertainty-ensemble crop).
    center : (lat, lon), optional
        Centre in degrees; default the centre of the region's :attr:`Region.roi`.
    snap_m : float, optional
        Snap the edges to multiples of this many metres so the window lines
        up with the 5 m PGDA / SfS pixel grid; ``None`` keeps them exact.

    Returns
    -------
    PsBox
        Use ``.projwin`` for ASP / GDAL and ``.aoi()`` for searches.
    """
    reg = get_region(region)
    cx, cy = ps_xy(center[0], center[1], "S") if center is not None else reg.roi.center_xy
    side = size_km * 1000.0
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    if snap_m:
        x0, y0 = round(x0 / snap_m) * snap_m, round(y0 / snap_m) * snap_m
    box = PsBox(x0, y0, x0 + side, y0 + side)
    if not reg.roi.contains(box):
        log.warning("Crop window %s extends outside the %s ROI %s", box.bounds, reg.name, reg.roi.bounds)
    return box


def _site_name(site_or_region: Region | int | str) -> str:
    if isinstance(site_or_region, str) and site_or_region in PGDA_SITES:
        return site_or_region
    reg = get_region(site_or_region)
    if reg.pgda_site is None:
        raise KeyError(f"Region {reg.id} {reg.name} has no PGDA LOLA 5 m site")
    return reg.pgda_site


def _classify_pgda(path: str) -> str | None:
    name = path.rsplit("/", 1)[-1]
    if "/Clones/" in path:
        return "clones"
    if name.endswith("_ROIs.tgz"):
        return "rois"
    if name.endswith(".xyzi"):
        return "xyzi"
    m = re.search(r"_(surf|ldec|toterr|slp|slperr)\.tif$", name)
    return m.group(1) if m else None


def parse_pgda_index(html: str) -> dict[str, dict[str, list[str]]]:
    """``site → kind → [url, ...]`` from the PGDA product page HTML.

    Links may be absolute, root-relative or relative to :data:`PGDA_ROOT`
    (``DM2/Clones/DM2_final_adj_5mpp_0001_err.tif``); clones are sorted by
    number.
    """
    index: dict[str, dict[str, list[str]]] = {}
    host = "{0.scheme}://{0.netloc}".format(urllib.parse.urlparse(PGDA_ROOT))
    for href in re.findall(r"""href\s*=\s*["']?([^"'\s>]+)""", html, flags=re.I):
        if href.startswith(("http://", "https://")):
            url = href.replace("http://", "https://", 1)
        elif href.startswith("/"):
            url = host + href
        else:
            url = PGDA_ROOT + href
        if not url.startswith(PGDA_ROOT):
            continue
        rel = url[len(PGDA_ROOT) :]
        kind = _classify_pgda(rel)
        if kind is None or "/" not in rel:
            continue
        urls = index.setdefault(rel.split("/", 1)[0], {}).setdefault(kind, [])
        if url not in urls:
            urls.append(url)
    for kinds in index.values():
        kinds.get("clones", []).sort()
    return index


def _get(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0, retries: int = 4) -> bytes:
    """GET *url* with retries on throttling / server errors."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_CODES or attempt == retries:
                raise
        except urllib.error.URLError:
            if attempt == retries:
                raise
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def pgda_index(
    refresh: bool = False, cache_dir: Path = REFERENCE_CACHE_DIR, html: str | None = None
) -> dict[str, dict[str, list[str]]]:
    """PGDA LOLA 5 m file index, parsed from the product page and cached.

    Parameters
    ----------
    refresh : bool
        Re-fetch :data:`PGDA_PRODUCT_PAGE` even when a cached index exists.
    cache_dir : Path
        Where ``pgda_lola_5mpp_index.json`` is kept.
    html : str, optional
        Parse this page text instead of fetching it (nothing is cached).
    """
    if html is not None:
        return parse_pgda_index(html)
    cache = Path(cache_dir) / "pgda_lola_5mpp_index.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    index = parse_pgda_index(_get(PGDA_PRODUCT_PAGE).decode("utf-8", "replace"))
    if not index:
        raise RuntimeError(f"No LOLA 5 m files found on {PGDA_PRODUCT_PAGE}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index, indent=1))
    return index


def _clone_number(url: str) -> int:
    m = re.search(r"_(\d{4})_err\.tif$", url)
    return int(m.group(1)) if m else 0


def pgda_urls(
    site_or_region: Region | int | str,
    kinds: Iterable[str] = ("surf",),
    clones: int | Iterable[int] | None = None,
    index: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, list[str]]:
    """``kind → [url, ...]`` for one PGDA site.

    Parameters
    ----------
    site_or_region : Region, int or str
        A PGDA site name (``"Site23"``) or anything :func:`get_region` accepts.
    kinds : iterable of str
        Members of :data:`PGDA_KINDS`.
    clones : int or iterable of int, optional
        With ``"clones"``: the first *n* clones, or these clone numbers
        (1-based); default all 100.
    index : dict, optional
        A :func:`pgda_index` result (default: the cached index).
    """
    site = _site_name(site_or_region)
    kinds = list(kinds)
    bad = [k for k in kinds if k not in PGDA_KINDS]
    if bad:
        raise ValueError(f"Unknown PGDA kind(s) {bad}; choose from {', '.join(PGDA_KINDS)}")
    files = (index if index is not None else pgda_index()).get(site)
    if not files:
        raise KeyError(f"PGDA site {site} not in the file index")
    out: dict[str, list[str]] = {}
    for kind in kinds:
        urls = list(files.get(kind, []))
        if kind == "clones" and clones is not None:
            if isinstance(clones, int):
                urls = urls[:clones]
            else:
                wanted = set(clones)
                urls = [u for u in urls if _clone_number(u) in wanted]
        if not urls:
            log.warning("PGDA site %s has no %s files", site, kind)
        out[kind] = urls
    return out


def fetch(
    url: str,
    dest: Path,
    expected_md5: str | None = None,
    force: bool = False,
    show_progress: bool = True,
    retries: int = 4,
) -> Path:
    """Resumable download of *url* to *dest*, skipped when *dest* exists.

    Throttling replies (403 / 429) and server errors are retried with
    back-off; the ``.part`` file is resumed each time.
    """
    dest = Path(dest)
    if reuse_existing(dest, force):
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    if force and part.exists():
        part.unlink()
    for attempt in range(retries + 1):
        try:
            return download_file(url, dest, show_progress=show_progress, expected_md5=expected_md5)
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_CODES or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        log.info("Retrying %s", url)
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def download_pgda(
    site_or_region: Region | int | str,
    kinds: Iterable[str] = ("surf",),
    out_dir: Path = Path("."),
    clones: int | Iterable[int] | None = None,
    force: bool = False,
    show_progress: bool = True,
    index: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, list[Path]]:
    """Download PGDA LOLA 5 m files of one site by kind (sequentially; resumable).

    Files land in ``out_dir/<site>/`` (clones in ``out_dir/<site>/Clones/``);
    existing ones are kept unless *force*.  Arguments as in :func:`pgda_urls`.

    Returns
    -------
    dict
        ``kind → [local path, ...]``.
    """
    site = _site_name(site_or_region)
    out: dict[str, list[Path]] = {}
    for kind, urls in pgda_urls(site, kinds, clones, index).items():
        sub = Path(out_dir) / site / ("Clones" if kind == "clones" else "")
        out[kind] = [fetch(u, sub / url_filename(u), force=force, show_progress=show_progress) for u in urls]
    return out


def read_xyzi(path: Path, box: PsBox | None = None) -> pd.DataFrame:
    """Read a PGDA ``.xyzi`` LOLA point cloud.

    The file is little-endian float64 rows of ``X, Y, Z, RDRid`` with X/Y the
    polar stereographic coordinates and Z the height above the 1737.4 km
    sphere, all in km (the same Z as the ``*_surf.tif`` metres), and RDRid the
    LOLA RDR ``YYDOYHHMM``.

    Parameters
    ----------
    path : Path
        The ``.xyzi`` file.
    box : PsBox, optional
        Keep only points inside this window.

    Returns
    -------
    pandas.DataFrame
        Columns ``x``, ``y``, ``height`` (metres) and ``rdr_id`` (int64).
    """
    raw = np.fromfile(path, dtype="<f8")
    if raw.size % 4:
        raise ValueError(f"{path}: size is not a multiple of 4 float64 values")
    pts = raw.reshape(-1, 4)
    if box is not None:
        x, y = pts[:, 0] * 1000.0, pts[:, 1] * 1000.0
        pts = pts[(x >= box.x0) & (x <= box.x1) & (y >= box.y0) & (y <= box.y1)]
    return pd.DataFrame(
        {
            "x": pts[:, 0] * 1000.0,
            "y": pts[:, 1] * 1000.0,
            "height": pts[:, 2] * 1000.0,
            "rdr_id": pts[:, 3].astype(np.int64),
        }
    )


def write_asp_csv(points: pd.DataFrame | Path, out: Path, box: PsBox | None = None) -> Path:
    """Write LOLA points as an ASP CSV of ``easting, northing, height_above_datum``.

    Use with ``--csv-format`` :data:`ASP_CSV_FORMAT` and ``--csv-srs``
    :data:`PS_SRS` (the 1737.4 km sphere is then the datum), e.g.
    ``pc_align --csv-format '1:easting 2:northing 3:height_above_datum'
    --csv-srs '+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m' dem.tif lola.csv``.

    Parameters
    ----------
    points : DataFrame or Path
        A :func:`read_xyzi` table or an ``.xyzi`` path.
    out : Path
        CSV to write; a ``#`` header line (ignored by ASP) names the columns.
    box : PsBox, optional
        Keep only points inside this window.
    """
    df = read_xyzi(Path(points), box) if not isinstance(points, pd.DataFrame) else points
    if box is not None and isinstance(points, pd.DataFrame):
        df = df[(df.x >= box.x0) & (df.x <= box.x1) & (df.y >= box.y0) & (df.y <= box.y1)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# easting_m,northing_m,height_above_datum_m,rdr_id  csv-srs: {PS_SRS}\n")
        np.savetxt(
            f,
            df[["x", "y", "height", "rdr_id"]].to_numpy(),
            fmt=["%.3f", "%.3f", "%.3f", "%d"],
            delimiter=",",
        )
    return out


def parse_zenodo_files(record: dict[str, Any]) -> list[dict[str, Any]]:
    """``[{"key", "size", "md5", "url"}, ...]`` from a Zenodo record API response."""
    files = []
    for f in record.get("files", []):
        algo, _, digest = str(f.get("checksum", "")).partition(":")
        files.append(
            {
                "key": f["key"],
                "size": int(f.get("size", 0)),
                "md5": digest if algo == "md5" else None,
                "url": ZENODO_FILE_URL.format(name=f["key"]),
            }
        )
    return sorted(files, key=lambda f: f["key"])


def zenodo_files(refresh: bool = False, cache_dir: Path = REFERENCE_CACHE_DIR) -> list[dict[str, Any]]:
    """Files of the Bertone et al. Zenodo record (fetched once, then cached)."""
    cache = Path(cache_dir) / f"zenodo_{ZENODO_RECORD}.json"
    if cache.exists() and not refresh:
        record = json.loads(cache.read_text())
    else:
        record = json.loads(_get(ZENODO_API_URL))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"id": record.get("id"), "files": record.get("files", [])}, indent=1))
    return parse_zenodo_files(record)


def download_sdem(
    region: Region | int | str,
    out_dir: Path = Path("."),
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download a region's Bertone et al. zip (1–4 GB; resumable, MD5-checked)."""
    reg = get_region(region)
    return fetch(
        reg.zenodo_url,
        Path(out_dir) / reg.zenodo_file,
        expected_md5=reg.zenodo_md5,
        force=force,
        show_progress=show_progress,
    )


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP Range requests.

    Lets :class:`zipfile.ZipFile` read a remote zip's central directory or a
    single member without downloading the archive.
    """

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self.url, self.timeout, self.pos = url, timeout, 0
        self.requests = 0
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rng = resp.headers.get("Content-Range", "")
            if resp.status != 206 or "/" not in rng:
                raise OSError(f"{url} does not support HTTP Range requests")
            self.size = int(rng.rsplit("/", 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b: Any) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = _get(self.url, {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=self.timeout)[:n]
        self.requests += 1
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _open_zip(source: Region | int | str | Path, buffer_size: int = 4 << 20) -> zipfile.ZipFile:
    """Local zip path, or a region's remote Zenodo zip (read via HTTP Range, *buffer_size* per request)."""
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
        return zipfile.ZipFile(source)
    fh: IO[bytes] = io.BufferedReader(HttpRangeFile(get_region(source).zenodo_url), buffer_size=buffer_size)
    return zipfile.ZipFile(fh)


def _is_junk(name: str) -> bool:
    return name.startswith("__MACOSX/") or name.endswith((".DS_Store", "/"))


def sdem_members(source: Region | int | str | Path) -> pd.DataFrame:
    """Members of a Bertone et al. zip: a local path, or a region (remote, central directory only).

    Returns
    -------
    pandas.DataFrame
        ``name`` (path inside the zip), ``size`` and ``compressed_size``
        (bytes); macOS metadata and directories are left out.
    """
    with _open_zip(source) as zf:
        rows = [
            {"name": i.filename, "size": i.file_size, "compressed_size": i.compress_size}
            for i in zf.infolist()
            if not _is_junk(i.filename)
        ]
    return pd.DataFrame(rows, columns=["name", "size", "compressed_size"])


def select_members(names: Iterable[str], patterns: Iterable[str]) -> list[str]:
    """Zip member names matching any pattern.

    A pattern with ``*``, ``?`` or ``[`` is a glob against the full name and
    the base name (``"SLOPE_MAPS/*_100.tif"``); otherwise it is a
    case-insensitive substring of the base name (``"GLDELEV"``, ``"README"``).
    """
    pats = list(patterns)
    out = []
    for name in names:
        if _is_junk(name):
            continue
        base = name.rsplit("/", 1)[-1]
        for p in pats:
            if any(c in p for c in "*?["):
                hit = fnmatch.fnmatch(name, p) or fnmatch.fnmatch(base, p) or fnmatch.fnmatch(name, f"*/{p}")
            else:
                hit = p.lower() in base.lower()
            if hit:
                out.append(name)
                break
    return out


def extract_sdem(
    source: Region | int | str | Path,
    members: Iterable[str] = ("GLDELEV",),
    out_dir: Path = Path("."),
    force: bool = False,
) -> list[Path]:
    """Extract selected members of a Bertone et al. zip.

    Parameters
    ----------
    source : Path, Region, int or str
        A downloaded zip, or a region whose Zenodo zip is then read remotely
        member by member (HTTP Range; only the selected members are fetched).
    members : iterable of str
        :func:`select_members` patterns, e.g. ``("GLDELEV", "GLDSIGM", "README")``.
    out_dir : Path
        Destination; paths below the zip's top-level folder are kept
        (``SLOPE_MAPS/A311_GLDTSLP_005_100.tif``).
    force : bool
        Re-extract members that already exist.

    Returns
    -------
    list of Path
    """
    out_dir = Path(out_dir)
    paths = []
    with _open_zip(source) as zf:
        names = select_members(zf.namelist(), members)
        if not names:
            raise KeyError(f"No zip member matches {list(members)}")
        for name in names:
            rel = name.split("/", 1)[1] if "/" in name else name
            dest = out_dir / rel
            if not reuse_existing(dest, force):
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with zf.open(name) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                tmp.replace(dest)
                log.info("Extracted %s", dest)
            paths.append(dest)
    return paths


NAC_FOCAL_MM = {"LEFT": 699.62, "RIGHT": 701.57}
"""LROC NAC focal lengths (mm) of the ISIS / ALE camera models, by ``name_sensor`` suffix."""

_PID_RE = re.compile(r"_(M\d+[LR]E)\.(?:json|tif)$")


def _member_pid(name: str) -> str | None:
    m = _PID_RE.search(name)
    return m.group(1) if m else None


def published_members(source: Region | int | str | Path, kind: str = "GLDACAM") -> pd.DataFrame:
    """Per-image members of a Bertone et al. zip: cameras (``GLDACAM``) or mapprojected images (``GLDIMGP``).

    Returns
    -------
    pandas.DataFrame
        ``product_id``, ``name`` and ``size``; regions 0, 7, 8 and 10 have
        no ``LROC-NAC-adjust/`` cameras (the table is then empty).
    """
    members = sdem_members(source)
    hit = members[members["name"].str.contains(f"_{kind}_", regex=False)].copy()
    hit.insert(0, "product_id", hit["name"].map(_member_pid))
    return hit.dropna(subset=["product_id"])[["product_id", "name", "size"]].reset_index(drop=True)


def parse_image_status(text: str) -> pd.DataFrame:
    """Parse ``extra/image_status_*.csv``: ``product_id`` and ``status``.

    Statuses are ``used_for_sfs`` (the SfS inputs, one ``GLDACAM`` camera
    each), ``not_aligned`` (dropped by bundle adjustment) and
    ``available_not_selected``.  The ``# key: value`` header lines go to
    ``attrs["summary"]`` (counts as ints).
    """
    summary: dict[str, int] = {}
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, sep, val = line[1:].partition(":")
            if sep and val.strip().isdigit():
                summary[key.strip()] = int(val)
            continue
        pid, _, status = line.partition(",")
        if pid != "img_name":
            rows.append((pid.strip(), status.strip()))
    df = pd.DataFrame(rows, columns=["product_id", "status"])
    df.attrs["summary"] = summary
    return df


def image_status(
    source: Region | int | str | Path, out_dir: Path = REFERENCE_CACHE_DIR / "published", force: bool = False
) -> pd.DataFrame:
    """A region's image status table (:func:`parse_image_status`), extracted once to *out_dir*."""
    (path,) = extract_sdem(source, ["extra/image_status_*.csv"], Path(out_dir), force=force)
    return parse_image_status(path.read_text())


def parse_tiff_extent(head: bytes) -> PsBox:
    """Extent of a north-up GeoTIFF from its first bytes (header, IFD and georeferencing tags).

    Reads ``ImageWidth`` / ``ImageLength`` with ``ModelPixelScale`` and
    ``ModelTiepoint`` (or ``ModelTransformation``) from a classic TIFF whose
    IFD and tag values lie inside *head* (GDAL writes them first, so 16 KB
    is plenty for the Bertone et al. ``GLDIMGP`` maps).
    """
    import struct

    bo = {b"II": "<", b"MM": ">"}.get(head[:2])
    if bo is None or struct.unpack(bo + "H", head[2:4])[0] != 42:
        raise ValueError("Not a classic TIFF")
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    codes = {3: "H", 4: "I", 8: "h", 9: "i", 11: "f", 12: "d"}
    (ifd,) = struct.unpack(bo + "I", head[4:8])
    (n,) = struct.unpack(bo + "H", head[ifd : ifd + 2])
    tags: dict[int, tuple[Any, ...]] = {}
    for i in range(n):
        tag, typ, cnt = struct.unpack(bo + "HHI", head[ifd + 2 + 12 * i : ifd + 10 + 12 * i])
        if typ not in codes:
            continue
        size = sizes[typ] * cnt
        raw = head[ifd + 10 + 12 * i : ifd + 14 + 12 * i]
        if size > 4:
            (off,) = struct.unpack(bo + "I", raw)
            if off + size > len(head):
                raise ValueError(f"TIFF tag {tag} lies beyond the first {len(head)} bytes")
            raw = head[off : off + size]
        tags[tag] = struct.unpack(bo + codes[typ] * cnt, raw[:size])
    w, h = tags[256][0], tags[257][0]
    if 34264 in tags:
        t = tags[34264]
        sx, sy, x0, y0 = t[0], -t[5], t[3], t[7]
    elif 33550 in tags and 33922 in tags:
        sx, sy = tags[33550][:2]
        i, j, _, x, y, _ = tags[33922][:6]
        x0, y0 = x - i * sx, y + j * sy
    else:
        raise ValueError("TIFF has no GeoTIFF georeferencing tags")
    return PsBox(x0, y0 - h * sy, x0 + w * sx, y0)


def _member_head(zf: zipfile.ZipFile, info: zipfile.ZipInfo, n: int) -> bytes:
    """First *n* uncompressed bytes of a stored or deflated member, in one read of the archive."""
    import struct
    import zlib

    fp = zf.fp
    assert fp is not None
    fp.seek(info.header_offset)
    raw = fp.read(30 + len(info.filename.encode()) + 1024 + 2 * n)
    if raw[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"Bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    data = raw[30 + name_len + extra_len :][: info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        return data[:n]
    if info.compress_type != zipfile.ZIP_DEFLATED:
        raise NotImplementedError(f"Compression {info.compress_type} of {info.filename}")
    return zlib.decompressobj(-15).decompress(data, n)


def published_extents(
    source: Region | int | str | Path,
    cache: Path | None = None,
    head_bytes: int = 16384,
) -> pd.DataFrame:
    """PS extents of a region's mapprojected input images (``LROC-NAC-maps/*_GLDIMGP_*.tif``).

    Only the start of each member is read (remotely: about 2 small HTTP
    Range requests per image), so this is a cheap way to find the images
    over a crop window.  The maps are cropped to the valid pixels' bounding
    box within the ROI, so a diagonal strip's extent over-covers.

    Parameters
    ----------
    source : Path, Region, int or str
        A downloaded zip or a region (remote).
    cache : Path, optional
        JSON file of the result, written once and reused.
    head_bytes : int
        Bytes decompressed from each member.

    Returns
    -------
    pandas.DataFrame
        ``product_id``, ``x0``, ``y0``, ``x1``, ``y1`` (PS metres).
    """
    cols = ["product_id", "x0", "y0", "x1", "y1"]
    if cache is not None and Path(cache).is_file():
        return pd.DataFrame(json.loads(Path(cache).read_text()), columns=cols)
    rows = []
    with _open_zip(source, buffer_size=4 << 10) as zf:
        for info in zf.infolist():
            pid = _member_pid(info.filename)
            if pid is None or "_GLDIMGP_" not in info.filename or _is_junk(info.filename):
                continue
            rows.append([pid, *parse_tiff_extent(_member_head(zf, info, head_bytes)).bounds])
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(rows))
    return pd.DataFrame(rows, columns=cols)


def images_covering(extents: pd.DataFrame, box: PsBox, min_fraction: float = 0.0) -> list[str]:
    """Product IDs whose extent overlaps *box* by more than *min_fraction* of the box area."""
    ix = np.clip(np.minimum(extents.x1, box.x1) - np.maximum(extents.x0, box.x0), 0, None)
    iy = np.clip(np.minimum(extents.y1, box.y1) - np.maximum(extents.y0, box.y0), 0, None)
    frac = ix * iy / (box.width * box.height)
    return list(extents.product_id[(frac > min_fraction) & (ix * iy > 0)])


def published_cameras(
    region: Region | int | str,
    out_dir: Path,
    images: Iterable[str] | None = None,
    window: PsBox | None = None,
    source: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Extract the Bertone et al. bundle-adjusted NAC cameras (``LROC-NAC-adjust/*_GLDACAM_*.json``).

    Each file is an ASP / USGS CSM linescan model state (model-name line,
    then JSON) with the bundle adjustment and LOLA registration baked in.
    It describes the *full* image ISIS-``reduce``-d by an integer factor to
    about 5 m/px (see :func:`published_reduce_factor` and
    :func:`published_camera_native`).

    Parameters
    ----------
    region : Region, int or str
        Anything :func:`get_region` accepts.
    out_dir : Path
        Destination; files land in ``out_dir/LROC-NAC-adjust/``.
    images : iterable of str, optional
        Only these product IDs.
    window : PsBox, optional
        Only images whose mapprojected extent overlaps this box
        (:func:`published_extents`, cached in ``out_dir``).
    source : Path, optional
        A downloaded zip instead of the remote Zenodo file.
    force : bool
        Re-extract existing files.

    Returns
    -------
    dict
        ``product_id -> camera path`` (sorted by product ID).
    """
    reg = get_region(region)
    src: Region | Path = Path(source) if source is not None else reg
    cams = published_members(src)
    if cams.empty:
        raise KeyError(f"Region {reg.id} {reg.name} has no published cameras (LROC-NAC-adjust/)")
    wanted = set(cams.product_id)
    if images is not None:
        wanted &= {str(p).upper() for p in images}
    if window is not None:
        ext = published_extents(src, cache=Path(out_dir) / f"gldimgp_extents_{reg.id}.json")
        wanted &= set(images_covering(ext, window))
    names = sorted(n.rsplit("/", 1)[-1] for n in cams.name[cams.product_id.isin(wanted)])
    if not names:
        return {}
    paths = extract_sdem(src, names, Path(out_dir), force=force)
    return dict(sorted((p.name.rsplit("_", 1)[-1].removesuffix(".json"), p) for p in paths))


def published_reduce_factor(camera: CsmCamera | Path) -> int:
    """``reduce`` factor of a published camera: NAC focal length / its (scaled) ``m_focalLength``."""
    cam = camera if isinstance(camera, CsmCamera) else read_camera(camera)
    side = "LEFT" if str(cam.data.get("m_sensorName", "")).upper().endswith("LEFT") else "RIGHT"
    return max(1, round(NAC_FOCAL_MM[side] / float(cam.data["m_focalLength"])))


def published_camera_native(camera: CsmCamera | Path, native_size: tuple[int, int]) -> CsmCamera:
    """Published camera re-mapped to the full-resolution image (``(samples, lines)`` of the EDR / cube).

    The published states keep the detector summing and divide the focal
    length and detector-sample origin by the reduce factor *f*; their
    ``m_nLines`` / ``m_nSamples`` are the native size divided by ``f**2``
    (not ``f``), which only affects usgscsm's initial ground-to-image guess.
    The result is a state for the native image with consistent sizes; use
    :func:`lunar_toolkit.csm.camera_for_cube` or
    :func:`lunar_toolkit.csm.isis_reduce_with_csm` to get any crop or
    reduction of it.
    """
    cam = camera if isinstance(camera, CsmCamera) else read_camera(camera)
    f = published_reduce_factor(cam)
    out = transform_camera(cam, 1.0 / f, 0.0, 1.0 / f, 0.0, native_size)
    if "m_gsd" in cam.data:  # already the native ground sample distance
        out.data["m_gsd"] = cam.data["m_gsd"]
        out.data["m_halfSwath"] = cam.data["m_gsd"] * native_size[0] / 2.0
    return out
