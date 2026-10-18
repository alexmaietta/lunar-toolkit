"""Search regions and AOI polygons (lon/lat, 0–360 east; polar stereographic past ±60°)."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import shapely.geometry
import shapely.wkt
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid

from .constants import FOOTPRINT_CORNERS, MOON_RADIUS_KM, POLAR_LAT_THRESHOLD

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def ps_xy(lat: float, lon: float, hemi: str) -> tuple[float, float]:
    """Polar stereographic (x, y) in metres; mirrors the SQL ``_ps_*`` macros."""
    sgn = 1.0 if hemi == "S" else -1.0
    rho = 2.0 * MOON_RADIUS_KM * 1000.0 * math.tan(math.pi / 4 + sgn * math.radians(lat) / 2)
    return rho * math.sin(math.radians(lon)), sgn * rho * math.cos(math.radians(lon))


def _ps_hemis(s: float, n: float) -> list[str]:
    """Hemispheres (``'S'`` / ``'N'``) a latitude range reaches poleward of ±60°."""
    return [h for h, past in (("S", s < -POLAR_LAT_THRESHOLD), ("N", n > POLAR_LAT_THRESHOLD)) if past]


def query_region(
    lat: tuple[float, float] | None = None,
    lon: tuple[float, float] | None = None,
    point: tuple[float, float] | None = None,
    radius_km: float = 0.0,
) -> dict:
    """Normalise a bbox or point+radius into lat range, lon arc and PS boxes."""
    if point is not None:
        plat, plon = float(point[0]), float(point[1])
        delta = math.degrees(radius_km / MOON_RADIUS_KM)
        s, n = plat - delta, plat + delta
        if n >= 90 or s <= -90 or radius_km <= 0 and abs(plat) >= 90:
            west, span = 0.0, 360.0
        else:
            ratio = math.sin(math.radians(delta)) / math.cos(math.radians(plat))
            dlon = 90.0 if ratio >= 1 else math.degrees(math.asin(ratio))
            west, span = (plon - dlon) % 360.0, min(2 * dlon, 360.0)
        s, n = max(s, -90.0), min(n, 90.0)
        ps: dict[str, tuple[float, float, float, float]] = {}
        for hemi in _ps_hemis(s, n):
            x, y = ps_xy(plat, plon, hemi)
            # PS scale k = 2 / (1 + sin|φ|); use the largest k in the circle
            edge = min(abs(s), abs(n)) if s * n > 0 else 0.0
            k = 2.0 / (1.0 + math.sin(math.radians(edge)))
            r = radius_km * 1000.0 * k * 1.01 + 1.0
            ps[hemi] = (x - r, x + r, y - r, y + r)
        return {"lat": (s, n), "west": west, "span": span, "ps": ps}

    s, n = (float(lat[0]), float(lat[1])) if lat is not None else (-90.0, 90.0)
    if s > n:
        s, n = n, s
    if lon is None or float(lon[1]) - float(lon[0]) >= 360.0:
        west, span = 0.0, 360.0
    else:
        west = float(lon[0]) % 360.0
        span = (float(lon[1]) - float(lon[0])) % 360.0
    ps = {hemi: _ps_bbox_of_box(s, n, west, span, hemi) for hemi in _ps_hemis(s, n)}
    return {"lat": (s, n), "west": west, "span": span, "ps": ps}


def _ps_bbox_of_box(
    s: float, n: float, west: float, span: float, hemi: str, samples: int = 181
) -> tuple[float, float, float, float]:
    """Projected bbox of a lat/lon box by sampling its boundary."""
    # Only the part of the box on this hemisphere's side of ±60° matters for
    # rows that carry PS coordinates, but clipping at the equator is enough.
    if hemi == "S":
        s2, n2 = s, min(n, 0.0)
    else:
        s2, n2 = max(s, 0.0), n
    pts = []
    for i in range(samples):
        f = i / (samples - 1)
        lo = west + span * f
        la = s2 + (n2 - s2) * f
        pts += [(s2, lo), (n2, lo), (la, west), (la, west + span)]
    xy = [ps_xy(a, b, hemi) for a, b in pts]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    pole_lat = -90.0 if hemi == "S" else 90.0
    if span >= 360.0 and s2 <= pole_lat <= n2:
        xs.append(0.0)
        ys.append(0.0)
    pad = 0.005 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def region_bbox(region: dict) -> tuple[float, float, float, float]:
    """``(west, south, east, north)`` of a :func:`query_region` (east may be < west)."""
    s, n = region["lat"]
    if region["span"] >= 360.0:
        return 0.0, s, 360.0, n
    west = region["west"]
    east = (west + region["span"]) % 360.0
    return west, s, east, n


Ring = list[tuple[float, float]]  # open ring of (lon, lat) vertices, degrees
AoiLike: TypeAlias = "Aoi | dict | str | Path"  # anything load_aoi accepts
Projection = Callable[[Ring], list[tuple[float, float]]]  # lon/lat ring -> planar (x, y) ring

_WKT_COLUMNS = ("FP_WKT", "footprint_wkt")


@dataclass
class Aoi:
    """Area of interest: one or more lon/lat polygons (degrees, positive east).

    ``polygons`` holds, per polygon, its exterior ring followed by any holes.
    Rings are open (no repeated closing vertex), densified and longitude
    unwrapped so consecutive vertices never jump across the 0/360 seam
    (values may lie outside 0..360).  A ring whose longitudes wind once
    around the globe encloses a pole.
    """

    polygons: list[list[Ring]]
    source: str = ""

    @property
    def wkt(self) -> str:
        """``MULTIPOLYGON`` WKT of the (unwrapped) rings."""

        def ring_wkt(r: Ring) -> str:
            return "(" + ", ".join(f"{lo:.6f} {la:.6f}" for lo, la in r + r[:1]) + ")"

        return (
            "MULTIPOLYGON("
            + ", ".join("(" + ", ".join(ring_wkt(r) for r in poly) + ")" for poly in self.polygons)
            + ")"
        )


_WKT_RE = re.compile(
    r"^\s*((MULTI)?(POINT|LINESTRING|POLYGON)|GEOMETRYCOLLECTION)\s*(Z|M|ZM)?\s*(\(|EMPTY)", re.IGNORECASE
)


def _is_file(text: str) -> bool:
    """True when *text* names an existing file (WKT / GeoJSON strings never do)."""
    if not text or "\n" in text or text.startswith("{") or _WKT_RE.match(text):
        return False
    try:
        return Path(text).is_file()
    except (OSError, ValueError):
        return False


def load_aoi(source: AoiLike, densify_deg: float | None = 1.0) -> Aoi:
    """Load an area of interest from GeoJSON, WKT or a box / circle region.

    Parameters
    ----------
    source : str, Path, dict or Aoi
        A ``.geojson`` / ``.json`` file (Feature, FeatureCollection or bare
        Geometry), a ``.wkt`` (or any text) file, a raw WKT string, a GeoJSON
        mapping, a region dict ``{"point": (lat, lon), "radius_km": r}``
        (spherical cap) or ``{"lat": (south, north), "lon": (west, east)}``
        (``west > east`` crosses 0/360; omit ``lon`` for all longitudes), or
        an existing :class:`Aoi` (returned unchanged).  Only
        Polygon / MultiPolygon parts are used (also inside
        GeometryCollections); coordinates are lon/lat degrees, positive east,
        in either −180..180 or 0..360.  Polygons may cross the 0/360 (or
        ±180) seam or wind around a pole.
    densify_deg : float, optional
        Insert vertices so no edge spans more than this many degrees of
        latitude or longitude.  Edges are taken as straight in lon/lat (as in
        GeoJSON); densifying keeps them that way once projected to polar
        stereographic.  ``None`` keeps the input vertices.

    Returns
    -------
    Aoi
    """
    if isinstance(source, Aoi):
        return source
    label = ""
    if isinstance(source, dict) and "type" not in source:
        raw = [[_region_ring(source)]]
        label = repr(source)
    elif isinstance(source, dict):
        raw = _shapely_rings(_geojson_geometries(source))
    else:
        text = str(source).strip()
        if _is_file(text):
            label = text
            text = Path(text).read_text(encoding="utf-8").strip()
        if text.startswith("{"):
            geoms = _geojson_geometries(json.loads(text))
        else:
            try:
                geoms = [shapely.wkt.loads(text)]
            except Exception as exc:
                raise ValueError(f"Not a GeoJSON/WKT file or WKT string: {text[:80]!r}") from exc
        raw = _shapely_rings(geoms)
        label = label or text[:80]

    polygons: list[list[Ring]] = []
    for rings in raw:
        rings = [r for r in (_open_ring(r) for r in rings) if len(r) >= 3]
        if not rings:
            continue
        if densify_deg:
            rings = [_densify_ring(r, densify_deg) for r in rings]
        rings = [_unwrap_ring(r) for r in rings]
        out = _shift_to_0_360(rings[:1])
        ext_mid = _lon_mid(out[0])
        for h in rings[1:]:
            kh = round((_lon_mid(h) - ext_mid) / 360.0)
            out.append([(lo - 360.0 * kh, la) for lo, la in h])
        polygons.append(out)
    if not polygons:
        raise ValueError("AOI contains no Polygon or MultiPolygon geometry")
    for poly in polygons:
        for r in poly:
            if any(abs(la) > 90.0 for _, la in r):
                raise ValueError("AOI latitude outside ±90° — coordinates must be (lon, lat)")
    return Aoi(polygons=polygons, source=label)


def _region_ring(region: dict, n: int = 72) -> Ring:
    """Boundary ring of a ``{"point", "radius_km"}`` cap or ``{"lat", "lon"}`` box."""
    if "point" in region:
        lat, lon = map(math.radians, region["point"])
        d = float(region.get("radius_km", 0.0)) / MOON_RADIUS_KM
        if d <= 0:
            raise ValueError("point AOI needs radius_km > 0")
        ring = []
        for k in range(n):
            b = 2 * math.pi * k / n
            la = math.asin(math.sin(lat) * math.cos(d) + math.cos(lat) * math.sin(d) * math.cos(b))
            dlo = math.atan2(
                math.sin(b) * math.sin(d) * math.cos(lat), math.cos(d) - math.sin(lat) * math.sin(la)
            )
            ring.append((math.degrees(lon + dlo), math.degrees(la)))
        return ring
    if "lat" not in region and "lon" not in region:
        raise ValueError(f"Region dict needs 'point' + 'radius_km' or 'lat' / 'lon', got {sorted(region)}")
    r = query_region(lat=region.get("lat"), lon=region.get("lon"))
    (s, n), west, span = r["lat"], r["west"], r["span"]
    return _box_ring(s, n, west, span)


def _shapely_rings(geoms: list) -> list[list[Ring]]:
    """Per polygon part of *geoms*: exterior then hole rings, as (lon, lat) lists."""
    return [
        [[(float(p[0]), float(p[1])) for p in r.coords] for r in (poly.exterior, *poly.interiors)]
        for g in geoms
        for poly in _polygon_parts(g)
    ]


def _geojson_geometries(obj: dict) -> list:
    kind = obj.get("type")
    if kind == "FeatureCollection":
        return [g for f in obj.get("features", []) for g in _geojson_geometries(f)]
    if kind == "Feature":
        return _geojson_geometries(obj["geometry"]) if obj.get("geometry") else []
    if kind == "GeometryCollection":
        return [g for sub in obj.get("geometries", []) for g in _geojson_geometries(sub)]
    return [shapely.geometry.shape(obj)]


def _polygon_parts(geom) -> list:
    kind = geom.geom_type
    if kind == "Polygon":
        return [] if geom.is_empty else [geom]
    if kind in ("MultiPolygon", "GeometryCollection"):
        return [p for g in geom.geoms for p in _polygon_parts(g)]
    return []


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def _open_ring(ring: Ring) -> Ring:
    if len(ring) > 1 and ring[0] == ring[-1]:
        return ring[:-1]
    return ring


def _densify_ring(ring: Ring, step: float) -> Ring:
    """Subdivide edges (seam-aware) so none exceeds ``step`` degrees."""
    out: Ring = []
    for i, (lo0, la0) in enumerate(ring):
        lo1, la1 = ring[(i + 1) % len(ring)]
        dlo, dla = _wrap180(lo1 - lo0), la1 - la0
        n = max(1, math.ceil(max(abs(dlo), abs(dla)) / step))
        out += [(lo0 + dlo * j / n, la0 + dla * j / n) for j in range(n)]
    return out


def _unwrap_ring(ring: Ring) -> Ring:
    """Make longitudes continuous along the ring (no ±360 jumps)."""
    out = [ring[0]]
    for lo, la in ring[1:]:
        out.append((out[-1][0] + _wrap180(lo - out[-1][0]), la))
    return out


def _winding(ring: Ring) -> float:
    """Net longitude swept by the closed ring (≈ ±360 if it encloses a pole)."""
    return sum(_wrap180(ring[(i + 1) % len(ring)][0] - ring[i][0]) for i in range(len(ring)))


def _encloses_pole(ring: Ring) -> bool:
    return abs(_winding(ring)) > 180.0


def _shift_to_0_360(rings: list[Ring]) -> list[Ring]:
    """Shift all rings by the multiple of 360° that puts the first one's west edge in [0, 360)."""
    k = math.floor(min(lo for lo, _ in rings[0]) / 360.0)
    return [[(lo - 360.0 * k, la) for lo, la in r] for r in rings]


def _lon_mid(ring: Ring) -> float:
    los = [lo for lo, _ in ring]
    return (min(los) + max(los)) / 2.0


def _covering_arc(arcs: list[tuple[float, float]]) -> tuple[float, float]:
    """Smallest ``(west, span)`` arc covering all ``(west, span)`` arcs."""
    pieces = []
    for w, sp in arcs:
        if sp >= 360.0:
            return 0.0, 360.0
        w %= 360.0
        if w + sp > 360.0:
            pieces += [(w, 360.0), (0.0, w + sp - 360.0)]
        else:
            pieces.append((w, w + sp))
    pieces.sort()
    merged = [list(pieces[0])]
    for a, b in pieces[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    # Largest uncovered gap (including the one wrapping past 360)
    gaps = [(merged[i + 1][0] - merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
    gaps.append((merged[0][0] + 360.0 - merged[-1][1], merged[0][0]))
    size, east_of_gap = max(gaps)
    if size <= 0.0:
        return 0.0, 360.0
    return east_of_gap % 360.0, 360.0 - size


def aoi_region(aoi: AoiLike) -> dict:
    """Bounding region of an AOI, in :func:`query_region` form.

    Returns ``{"lat": (s, n), "west", "span", "ps"}`` so the bbox prefilter
    used for boxes and circles applies unchanged; refine the prefiltered
    rows with :func:`filter_aoi`.
    """
    aoi = load_aoi(aoi)
    lats = [la for poly in aoi.polygons for la in (p[1] for p in poly[0])]
    s, n = min(lats), max(lats)
    arcs = []
    for poly in aoi.polygons:
        ext = poly[0]
        if _encloses_pole(ext):
            if sum(la for _, la in ext) < 0:
                s = -90.0
            else:
                n = 90.0
            arcs.append((0.0, 360.0))
        else:
            los = [lo for lo, _ in ext]
            arcs.append((min(los), max(los) - min(los)))
    west, span = _covering_arc(arcs)
    ps = {}
    for hemi in _ps_hemis(s, n):
        same_side = all((la <= 0.0) if hemi == "S" else (la >= 0.0) for la in lats)
        if not same_side:
            ps[hemi] = _ps_bbox_of_box(s, n, west, span, hemi)
            continue
        xy = [ps_xy(la, lo, hemi) for poly in aoi.polygons for lo, la in poly[0]]
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        pad = 0.005 * max(max(xs) - min(xs), max(ys) - min(ys)) + 1.0
        ps[hemi] = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
    return {"lat": (s, n), "west": west, "span": span, "ps": ps}


def _polar_hemi(region: dict) -> str | None:
    """``'S'`` / ``'N'`` when the whole region is poleward of ±60°."""
    s, n = region["lat"]
    if n < -POLAR_LAT_THRESHOLD:
        return "S"
    if s > POLAR_LAT_THRESHOLD:
        return "N"
    return None


def ps_projection(hemi: str) -> Projection:
    """Ring projection to polar stereographic (:func:`ps_xy`) for hemisphere *hemi*."""
    # Clamp away from the opposite pole, where the projection diverges
    lim = 89.0
    return lambda ring: [ps_xy(min(la, lim) if hemi == "S" else max(la, -lim), lo, hemi) for lo, la in ring]


def _ll_ring(ring: Ring) -> Ring:
    """Unwrapped lon/lat ring; pole-enclosing rings are closed via the pole."""
    r = _unwrap_ring(ring)
    if _encloses_pole(ring):
        pole = -90.0 if sum(la for _, la in r) < 0 else 90.0
        lo_end = r[0][0] + _winding(ring)
        r = r + [(lo_end, r[0][1]), (lo_end, pole), (r[0][0], pole)]
    return r


def _make_polygon(rings: list[list[tuple[float, float]]]):
    """Valid shapely geometry from rings (degenerate rings become hulls)."""
    if len(rings[0]) < 3:
        return MultiPoint(rings[0]).convex_hull
    poly = Polygon(rings[0], [h for h in rings[1:] if len(h) >= 3])
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.area == 0.0:
        return MultiPoint(rings[0]).convex_hull
    return poly


def aoi_geometry(aoi: AoiLike, proj: Projection | None = None):
    """AOI as one shapely geometry: rings projected by *proj*, else seam-replicated lon/lat."""
    aoi = load_aoi(aoi)
    if proj is not None:
        return unary_union([_make_polygon([proj(r) for r in poly]) for poly in aoi.polygons])
    g = unary_union([_make_polygon(_shift_to_0_360([_ll_ring(r) for r in poly])) for poly in aoi.polygons])
    return unary_union([g, translate(g, xoff=360.0), translate(g, xoff=-360.0)])


def footprint_geometry(rings_list: list[list[Ring]], proj: Projection | None):
    """Footprint polygons (per polygon: lon/lat exterior, *holes) as shapely geometry.

    Rings are projected by *proj* (e.g. :func:`ps_projection`), else kept in
    lon/lat (pole-enclosing rings closed via the pole) starting in 0..360.
    """
    parts = []
    for rings in rings_list:
        rings = [_open_ring(r) for r in rings]
        if not rings or len(rings[0]) < 1:
            continue
        if proj is not None:
            parts.append(_make_polygon([proj(r) for r in rings]))
            continue
        parts.append(_make_polygon(_shift_to_0_360([_ll_ring(r) if len(r) >= 3 else r for r in rings])))
    return unary_union(parts) if parts else None


def _box_ring(s: float, n: float, west: float, span: float, step: float = 1.0) -> Ring:
    """Densified lon/lat box ring (already unwrapped; span may be 360)."""
    m = max(1, math.ceil(span / step))
    k = max(1, math.ceil((n - s) / step))
    bottom = [(west + span * i / m, s) for i in range(m + 1)]
    right = [(west + span, s + (n - s) * j / k) for j in range(1, k + 1)]
    top = [(west + span * i / m, n) for i in range(m - 1, -1, -1)]
    left = [(west, n - (n - s) * j / k) for j in range(1, k)]
    return bottom + right + top + left


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def footprint_rings(row: dict, wkt_column: str | None = None) -> list[list[Ring]] | None:
    """Footprint of a result row as lon/lat rings (per polygon: exterior, *holes).

    Source: the *wkt_column* WKT, else the four corner lat/lon columns, else
    the ``FP_*`` / ``min_lat``…``lon_east`` bounding box; ``None`` if none.
    """
    wkt = row.get(wkt_column) if wkt_column else None
    if isinstance(wkt, str) and wkt.strip():
        try:
            rings = _shapely_rings([shapely.wkt.loads(wkt)])
        except Exception:
            rings = []  # unparsable WKT: fall back to corners / bbox
        if rings:
            return rings
    corners = [(_num(row.get(f"{c}_LONGITUDE")), _num(row.get(f"{c}_LATITUDE"))) for c in FOOTPRINT_CORNERS]
    ring_ll = [(lo, la) for lo, la in corners if lo is not None and la is not None]
    if len(ring_ll) == len(corners):
        return [[ring_ll]]

    s, n = _num(row.get("FP_MIN_LAT")), _num(row.get("FP_MAX_LAT"))
    west, span = _num(row.get("FP_LON_WEST")), _num(row.get("FP_LON_SPAN"))
    if s is None or n is None or west is None or span is None:
        s, n = _num(row.get("min_lat")), _num(row.get("max_lat"))
        west, east = _num(row.get("lon_west")), _num(row.get("lon_east"))
        if s is None or n is None or west is None or east is None:
            return None
        span = east - west if east >= west else east - west + 360.0
        span = min(span, 360.0)
    return [[_box_ring(s, n, west, span)]]


def row_footprint(row: dict, wkt_column: str | None = None, proj: Projection | None = None):
    """Shapely footprint (:func:`footprint_geometry`) of a result row, or ``None``."""
    rings = footprint_rings(row, wkt_column)
    return None if rings is None else footprint_geometry(rings, proj)


def aoi_mask(df: pd.DataFrame, aoi: AoiLike, wkt_column: str | None = None) -> list[bool]:
    """Per-row flag: does the row's footprint intersect the AOI exactly?

    Footprint source, per row: the WKT column (``FP_WKT`` for CumIndex,
    ``footprint_wkt`` for observations, or ``wkt_column``), else the four
    corner lat/lon columns, else the ``FP_*`` / ``min_lat``…``lon_east``
    bounding box.  Rows with no geometry at all are kept.

    *aoi* is anything :func:`load_aoi` accepts.  Regions entirely poleward
    of ±60° (``POLAR_LAT_THRESHOLD``) are intersected in polar
    stereographic (:func:`ps_xy`, footprint edges straight in the
    projection); others in lon/lat with the AOI replicated at ±360° so
    footprints across the 0/360 seam match.
    """
    aoi = load_aoi(aoi)
    if df is None or len(df) == 0:
        return []
    hemi = _polar_hemi(aoi_region(aoi))
    proj = ps_projection(hemi) if hemi else None
    target = prep(aoi_geometry(aoi, proj))
    if wkt_column is None:
        wkt_column = next((c for c in _WKT_COLUMNS if c in df.columns), None)
    mask = []
    for row in df.to_dict("records"):
        geom = row_footprint(row, wkt_column, proj)
        mask.append(True if geom is None or geom.is_empty else bool(target.intersects(geom)))
    return mask


def filter_aoi(df: pd.DataFrame, aoi: AoiLike, wkt_column: str | None = None) -> pd.DataFrame:
    """Keep only rows whose footprint intersects the AOI (see :func:`aoi_mask`)."""
    if df is None or len(df) == 0:
        return df
    return df[aoi_mask(df, aoi, wkt_column=wkt_column)].reset_index(drop=True)
