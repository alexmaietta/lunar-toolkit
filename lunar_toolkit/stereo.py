"""Stereo pair ranking by convergence, illumination similarity and overlap."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import MOON_RADIUS_KM, parse_nac_id
from .geometry import AoiLike, Ring, aoi_geometry, footprint_geometry, footprint_rings, load_aoi

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

DEFAULT_WEIGHTS: dict[str, float] = {
    "convergence": 0.35,
    "overlap": 0.25,
    "incidence": 0.20,
    "azimuth": 0.10,
    "resolution": 0.10,
}

PAIR_COLUMNS: list[str] = [
    "index_1",
    "index_2",
    "product_id_1",
    "product_id_2",
    "instrument",
    "start_time_1",
    "start_time_2",
    "dt_days",
    "convergence",
    "convergence_min",
    "convergence_max",
    "b_h",
    "incidence_1",
    "incidence_2",
    "incidence_diff",
    "azimuth_1",
    "azimuth_2",
    "azimuth_diff",
    "emission_1",
    "emission_2",
    "pixel_scale_1",
    "pixel_scale_2",
    "resolution_ratio",
    "overlap_km2",
    "overlap_frac",
    "aoi_coverage",
    "overlap_lat",
    "overlap_lon",
    "score",
]


def unit_vector(lat, lon) -> np.ndarray:
    """Unit vector(s) ``(..., 3)`` of planetocentric latitude / longitude (deg)."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def _latlon(v: np.ndarray) -> tuple[float, float]:
    v = v / np.linalg.norm(v)
    return math.degrees(math.asin(max(-1.0, min(1.0, v[2])))), math.degrees(math.atan2(v[1], v[0])) % 360.0


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (deg) between vectors along the last axis (numerically stable)."""
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cross, np.sum(a * b, axis=-1)))


def azimuth_difference(a, b) -> np.ndarray:
    """Smallest circular difference (deg, 0–180) between azimuths."""
    d = np.abs((np.asarray(a, float) - np.asarray(b, float)) % 360.0)
    return np.minimum(d, 360.0 - d)


def _rotate_onto(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply to *v* the minimal rotation taking unit vector *a* onto *b*."""
    k = np.cross(a, b)
    s, c = np.linalg.norm(k), float(np.dot(a, b))
    if s < 1e-15:
        return v
    k = k / s
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)


def convergence_angle(
    ground: tuple[float, float],
    sc1: tuple[float, float, float],
    sc2: tuple[float, float, float],
    centre1: tuple[float, float] | None = None,
    centre2: tuple[float, float] | None = None,
    radius_km: float = MOON_RADIUS_KM,
) -> tuple[float, float]:
    """Convergence angle and base/height ratio for two views of one point.

    Parameters
    ----------
    ground : (lat, lon)
        Ground point (deg) on the sphere of radius *radius_km*.
    sc1, sc2 : (sub_spacecraft_lat, sub_spacecraft_lon, altitude_km)
        Spacecraft positions as sub-spacecraft point + altitude above the
        sphere, at the time the image centre was acquired.
    centre1, centre2 : (lat, lon), optional
        Image centres.  Pushbroom images (NAC, ShadowCam) are tens of km
        long, so the spacecraft was elsewhere when it imaged a ground point
        away from the centre.  When given, each spacecraft position is moved
        by the rotation that takes the image centre onto *ground*, i.e. the
        look direction is held fixed in the local frame (constant pointing
        along a circular orbit).

    Returns
    -------
    (convergence_deg, b_h)
        Angle between the look vectors spacecraft → ground, and the baseline
        ``|S1 - S2|`` divided by the mean slant range to the ground point.
    """
    gu = unit_vector(*ground)
    g = radius_km * gu
    s1 = (radius_km + sc1[2]) * unit_vector(sc1[0], sc1[1])
    s2 = (radius_km + sc2[2]) * unit_vector(sc2[0], sc2[1])
    if centre1 is not None:
        s1 = _rotate_onto(s1, unit_vector(*centre1), gu)
    if centre2 is not None:
        s2 = _rotate_onto(s2, unit_vector(*centre2), gu)
    conv = float(angle_between(g - s1, g - s2))
    h = 0.5 * (np.linalg.norm(s1 - g) + np.linalg.norm(s2 - g))
    return conv, float(np.linalg.norm(s1 - s2) / h)


def _rotate_onto_many(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise :func:`_rotate_onto` of ``(n, 3)`` arrays."""
    k = np.cross(a, b)
    s = np.linalg.norm(k, axis=-1)
    c = np.sum(a * b, axis=-1)
    still = s < 1e-15
    k = k / np.where(still, 1.0, s)[:, None]
    out = (
        v * c[:, None]
        + np.cross(k, v) * s[:, None]
        + k * np.sum(k * v, axis=-1)[:, None] * (1.0 - c)[:, None]
    )
    return np.where(still[:, None], v, out)


def convergence_angles(
    ground: tuple[Any, Any],
    sc1: tuple[Any, Any, Any],
    sc2: tuple[Any, Any, Any],
    centre1: tuple[Any, Any] | None = None,
    centre2: tuple[Any, Any] | None = None,
    radius_km: float = MOON_RADIUS_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`convergence_angle` for arrays: each tuple element is an array over pairs.

    Returns
    -------
    (convergence_deg, b_h) : ndarrays
        NaN where an input is NaN.
    """
    gu = unit_vector(*ground).reshape(-1, 3)
    g = radius_km * gu
    s1 = (radius_km + np.asarray(sc1[2], float)).reshape(-1, 1) * unit_vector(sc1[0], sc1[1]).reshape(-1, 3)
    s2 = (radius_km + np.asarray(sc2[2], float)).reshape(-1, 1) * unit_vector(sc2[0], sc2[1]).reshape(-1, 3)
    if centre1 is not None:
        s1 = _rotate_onto_many(s1, unit_vector(*centre1).reshape(-1, 3), gu)
    if centre2 is not None:
        s2 = _rotate_onto_many(s2, unit_vector(*centre2).reshape(-1, 3), gu)
    conv = angle_between(g - s1, g - s2)
    h = 0.5 * (np.linalg.norm(s1 - g, axis=-1) + np.linalg.norm(s2 - g, axis=-1))
    return conv, np.linalg.norm(s1 - s2, axis=-1) / h


class TangentPlane:
    """Gnomonic projection (km) about a centre point; great circles → lines."""

    def __init__(self, lat: float, lon: float, radius_km: float = MOON_RADIUS_KM) -> None:
        self.c = unit_vector(lat, lon)
        ref = np.array([0.0, 0.0, 1.0]) if abs(self.c[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
        self.e = np.cross(ref, self.c)
        self.e /= np.linalg.norm(self.e)
        self.n = np.cross(self.c, self.e)
        self.r = radius_km

    @classmethod
    def from_vector(cls, v: np.ndarray) -> TangentPlane:
        return cls(*_latlon(np.asarray(v, float)))

    def ring(self, ring: Ring) -> list[tuple[float, float]]:
        """Project a lon/lat ring (deg) to plane (x, y) km (a :data:`~lunar_toolkit.geometry.Projection`)."""
        a = np.asarray(ring, float).reshape(-1, 2)
        v = unit_vector(a[:, 1], a[:, 0])
        d = np.maximum(v @ self.c, 0.05)  # far side (callers prefilter): clamp
        return list(zip((self.r * (v @ self.e) / d).tolist(), (self.r * (v @ self.n) / d).tolist()))

    def unproject(self, x: float, y: float) -> tuple[float, float]:
        return _latlon(self.c + (x / self.r) * self.e + (y / self.r) * self.n)


def observation_footprints(df: pd.DataFrame) -> tuple[list, np.ndarray, np.ndarray]:
    """Footprint rings, centre unit vectors and angular radii (deg) of observation rows.

    Rings are :func:`lunar_toolkit.geometry.footprint_rings` of each row
    (``footprint_wkt``, else the ``min_lat``…``lon_east`` box), ``None``
    (centre 0, radius 0) where a row has no footprint polygon.
    """
    rings = [footprint_rings(r, "footprint_wkt") for r in df.to_dict("records")]
    centres, radii = np.zeros((len(rings), 3)), np.zeros(len(rings))
    for k, rr in enumerate(rings):
        if rr is None or len(rr[0][0]) < 3:
            rings[k] = None
            continue
        centres[k], radii[k] = _cap([poly[0] for poly in rr])
    return rings, centres, radii


def _cap(rings: list) -> tuple[np.ndarray, float]:
    """Centre unit vector and angular radius (deg) of lon/lat rings' vertices."""
    ll = np.array([p for r in rings for p in r], float)
    v = unit_vector(ll[:, 1], ll[:, 0])
    c = v.sum(axis=0)
    c /= np.linalg.norm(c) or 1.0
    return c, float(np.max(angle_between(v, c[None, :])))


def aoi_plane(aoi: AoiLike) -> tuple[TangentPlane, Any, np.ndarray, float]:
    """Tangent plane about an AOI, the AOI in it (km), and its centre / angular radius.

    *aoi* is anything :func:`lunar_toolkit.geometry.load_aoi` accepts,
    e.g. ``{"point": (lat, lon), "radius_km": r}``,
    ``{"lat": (s, n), "lon": (w, e)}``, a GeoJSON / WKT path or string.

    Returns
    -------
    (TangentPlane, shapely geometry, ndarray, float)
    """
    a = load_aoi(aoi)
    centre, radius = _cap([poly[0] for poly in a.polygons])
    plane = TangentPlane.from_vector(centre)
    return plane, aoi_geometry(a, plane.ring), centre, radius


def _column(df: pd.DataFrame, col: str) -> np.ndarray:
    import pandas as pd

    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def _ramp(x: np.ndarray, lo: float, ideal_lo: float, ideal_hi: float, hi: float) -> np.ndarray:
    """1 inside [ideal_lo, ideal_hi], linear to 0 at lo / hi."""
    up = np.clip((x - lo) / max(ideal_lo - lo, 1e-9), 0.0, 1.0)
    down = np.clip((hi - x) / max(hi - ideal_hi, 1e-9), 0.0, 1.0)
    return np.minimum(up, down)


def find_stereo_pairs(
    obs_df: pd.DataFrame,
    aoi: AoiLike | None = None,
    convergence: tuple[float, float] = (10.0, 60.0),
    ideal_convergence: tuple[float, float] = (15.0, 45.0),
    max_incidence_diff: float = 20.0,
    max_azimuth_diff: float = 30.0,
    min_overlap: float = 0.1,
    min_aoi_coverage: float = 0.0,
    max_resolution_ratio: float | None = None,
    same_instrument: bool = True,
    require_geometry: bool = False,
    min_time_sep_s: float = 60.0,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank candidate stereo pairs in an observation table.

    Parameters
    ----------
    obs_df : DataFrame
        Observations in the common schema (extra columns are ignored).
    aoi : dict, str, Path or Aoi, optional
        Area of interest: ``{"point": (lat, lon), "radius_km": r}``,
        ``{"lat": (s, n), "lon": (w, e)}`` or anything
        :func:`lunar_toolkit.geometry.load_aoi` accepts (GeoJSON / WKT file
        or string).  Overlaps are then clipped to the AOI, pairs not
        overlapping inside it are dropped and the convergence angle is
        evaluated at the centroid of the clipped overlap.
    convergence : (min, max)
        Hard limits on the convergence angle (deg).  Pairs without spacecraft
        geometry are kept only if the emission bounds ``[|e1-e2|, e1+e2]``
        (from centre emission angles, so approximate; 5° slack is allowed)
        intersect this range and ``require_geometry`` is False.
    ideal_convergence : (min, max)
        Range scored 1; the score falls linearly to 0 at *convergence* limits.
    max_incidence_diff, max_azimuth_diff : float
        Hard limits (deg) on ``|i1 - i2|`` and the circular sub-solar azimuth
        difference (azimuth is ignored where unknown).
    min_overlap : float
        Minimum overlap area as a fraction of the smaller footprint (with
        an *aoi*: of the smaller footprint-inside-AOI).
    min_aoi_coverage : float
        Minimum fraction of the AOI covered by the overlap (with *aoi*).
    max_resolution_ratio : float, optional
        Maximum ``pixel_scale`` ratio (coarser / finer).
    same_instrument : bool
        Only pair images from the same ``instrument``.
    require_geometry : bool
        Drop pairs whose convergence angle cannot be computed.
    min_time_sep_s : float
        Minimum start-time separation; also, NAC LE/RE halves of one
        observation are never paired (their convergence is ~0).
    weights : dict, optional
        Score weights, keys as :data:`DEFAULT_WEIGHTS`.

    Returns
    -------
    DataFrame
        One row per pair, best first, with ``index_1`` / ``index_2`` (row
        labels in *obs_df*), ``product_id_1`` / ``_2``, ``instrument``,
        ``start_time_1`` / ``_2``, ``dt_days``, ``convergence`` (deg, NaN if
        unknown), ``convergence_min`` / ``convergence_max`` (emission bounds),
        ``b_h`` (baseline / mean slant range, ≈ 2 sin(convergence / 2)),
        ``incidence_1`` / ``_2``, ``incidence_diff``, ``azimuth_1`` / ``_2``, ``azimuth_diff``,
        ``emission_1`` / ``_2``, ``pixel_scale_1`` / ``_2``,
        ``resolution_ratio``, ``overlap_km2``, ``overlap_frac``,
        ``aoi_coverage`` (NaN without *aoi*), ``overlap_lat`` /
        ``overlap_lon`` (where convergence is evaluated) and ``score``.

    Notes
    -----
    ``score`` is a weighted mean of terms in [0, 1]:

    * convergence — 1 inside *ideal_convergence*, linear to 0 at the
      *convergence* limits (0.5 when the angle is unknown);
    * overlap — ``aoi_coverage`` with an AOI, else ``overlap_frac``;
    * incidence — ``1 - incidence_diff / max_incidence_diff``;
    * azimuth — ``1 - azimuth_diff / max_azimuth_diff`` (0.5 if unknown);
    * resolution — finer / coarser ``pixel_scale`` (0.5 if unknown).
    """
    import pandas as pd

    cols = PAIR_COLUMNS
    if obs_df is None or len(obs_df) < 2:
        return pd.DataFrame(columns=cols)

    df = obs_df
    labels = list(df.index)
    n = len(df)
    pids = (
        df["product_id"].astype(str).to_numpy() if "product_id" in df else np.array([str(i) for i in labels])
    )
    keys = np.array([(parse_nac_id(p) or (p.upper(),))[0] for p in pids])  # NAC L/R halves share one
    inst = df["instrument"].astype(str).to_numpy() if "instrument" in df else np.full(n, "")
    t = pd.to_datetime(df["start_time"], errors="coerce") if "start_time" in df else pd.Series([pd.NaT] * n)
    tsec = (t - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy(float)
    inc, emi = _column(df, "incidence"), _column(df, "emission")
    az, ps = _column(df, "sub_solar_azimuth"), _column(df, "pixel_scale")
    sclat, sclon = _column(df, "sub_spacecraft_lat"), _column(df, "sub_spacecraft_lon")
    alt = _column(df, "spacecraft_altitude")
    has_geom = ~(np.isnan(sclat) | np.isnan(sclon) | np.isnan(alt))

    rings, centres, radii = observation_footprints(df)
    valid = np.array([r is not None for r in rings], bool)
    clat, clon = _column(df, "center_lat"), _column(df, "center_lon")
    img_centres = centres.copy()
    known = ~(np.isnan(clat) | np.isnan(clon))
    img_centres[known] = unit_vector(clat[known], clon[known])

    plane = None
    clipped: list = []
    aoi_area = np.nan
    if aoi is not None:
        plane, target, ac, aoi_radius = aoi_plane(aoi)
        aoi_area = target.area
        near = valid & (angle_between(centres, ac[None, :]) < radii + aoi_radius + 1e-6)
        clipped = [
            footprint_geometry(r, plane.ring).intersection(target) if near[k] else None
            for k, r in enumerate(rings)
        ]
        valid &= np.array([g is not None and g.area > 0 for g in clipped], bool)

    out = []
    lo_c, hi_c = convergence
    for i in range(n - 1):
        if not valid[i]:
            continue
        j = np.arange(i + 1, n)
        m = valid[j] & (keys[j] != keys[i])
        if same_instrument:
            m &= inst[j] == inst[i]
        dt = np.abs(tsec[j] - tsec[i])
        m &= ~(dt < min_time_sep_s)
        dinc = np.abs(inc[j] - inc[i])
        m &= ~(dinc > max_incidence_diff)
        daz = azimuth_difference(az[j], az[i])
        m &= ~(daz > max_azimuth_diff)
        cmin, cmax = np.abs(emi[j] - emi[i]), emi[j] + emi[i]
        # Centre emissions only bound the convergence approximately: allow slack.
        m &= ~(cmax < lo_c - 5.0) & ~(cmin > hi_c + 5.0)
        if require_geometry:
            m &= has_geom[j] & has_geom[i]
        if max_resolution_ratio is not None:
            rr = np.maximum(ps[j], ps[i]) / np.minimum(ps[j], ps[i])
            m &= ~(rr > max_resolution_ratio)
        sep = angle_between(centres[j], centres[i][None, :])
        m &= sep <= radii[j] + radii[i] + 1e-6
        for jj in j[m]:
            rec = _pair_record(
                i, jj, rings, centres, img_centres, plane, clipped, aoi_area, has_geom, sclat, sclon, alt
            )
            if rec is None:
                continue
            frac = rec["overlap_frac"]
            if frac < min_overlap or (aoi is not None and rec["aoi_coverage"] < max(min_aoi_coverage, 1e-9)):
                continue
            c = rec["convergence"]
            if not math.isnan(c) and not (lo_c <= c <= hi_c):
                continue
            rec.update(i=i, j=int(jj))
            out.append(rec)

    if not out:
        return pd.DataFrame(columns=cols)

    res = pd.DataFrame(out)
    i_, j_ = res.pop("i").to_numpy(), res.pop("j").to_numpy()
    res.insert(0, "index_1", [labels[k] for k in i_])
    res.insert(1, "index_2", [labels[k] for k in j_])
    res["product_id_1"], res["product_id_2"] = pids[i_], pids[j_]
    res["instrument"] = inst[i_]
    res["start_time_1"], res["start_time_2"] = t.to_numpy()[i_], t.to_numpy()[j_]
    res["dt_days"] = np.abs(tsec[j_] - tsec[i_]) / 86400.0
    res["convergence_min"] = np.abs(emi[j_] - emi[i_])
    res["convergence_max"] = emi[j_] + emi[i_]
    res["incidence_1"], res["incidence_2"] = inc[i_], inc[j_]
    res["incidence_diff"] = np.abs(inc[j_] - inc[i_])
    res["azimuth_1"], res["azimuth_2"] = az[i_], az[j_]
    res["azimuth_diff"] = azimuth_difference(az[j_], az[i_])
    res["emission_1"], res["emission_2"] = emi[i_], emi[j_]
    res["pixel_scale_1"], res["pixel_scale_2"] = ps[i_], ps[j_]
    res["resolution_ratio"] = np.maximum(ps[i_], ps[j_]) / np.minimum(ps[i_], ps[j_])

    res["score"] = score_pairs(
        res,
        convergence=convergence,
        ideal_convergence=ideal_convergence,
        max_incidence_diff=max_incidence_diff,
        max_azimuth_diff=max_azimuth_diff,
        weights=weights,
        aoi=aoi is not None,
    )
    return res[cols].sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def score_pairs(
    pairs: pd.DataFrame,
    *,
    convergence: tuple[float, float] = (10.0, 60.0),
    ideal_convergence: tuple[float, float] = (15.0, 45.0),
    max_incidence_diff: float = 20.0,
    max_azimuth_diff: float = 30.0,
    weights: dict[str, float] | None = None,
    aoi: bool = False,
) -> np.ndarray:
    """The :func:`find_stereo_pairs` ``score`` of pair rows (``convergence``, ``overlap_frac`` or
    ``aoi_coverage`` with *aoi*, ``incidence_diff``, ``azimuth_diff``, ``resolution_ratio``)."""
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    conv = pairs["convergence"].to_numpy(float)
    s_conv = np.where(np.isnan(conv), 0.5, _ramp(conv, convergence[0], *ideal_convergence, convergence[1]))
    s_ovl = pairs["aoi_coverage" if aoi else "overlap_frac"].to_numpy(float)
    s_inc = np.nan_to_num(
        1.0 - pairs["incidence_diff"].to_numpy(float) / max(max_incidence_diff, 1e-9), nan=0.5
    )
    s_az = np.nan_to_num(1.0 - pairs["azimuth_diff"].to_numpy(float) / max(max_azimuth_diff, 1e-9), nan=0.5)
    s_res = np.nan_to_num(1.0 / pairs["resolution_ratio"].to_numpy(float), nan=0.5)
    tot = sum(w.values()) or 1.0
    return (
        w["convergence"] * s_conv
        + w["overlap"] * np.clip(s_ovl, 0, 1)
        + w["incidence"] * np.clip(s_inc, 0, 1)
        + w["azimuth"] * np.clip(s_az, 0, 1)
        + w["resolution"] * s_res
    ) / tot


def _pair_record(
    i, j, rings, centres, img_centres, plane, clipped, aoi_area, has_geom, sclat, sclon, alt
) -> dict | None:
    if plane is None:
        pl = TangentPlane.from_vector(centres[i] + centres[j])
        g1, g2 = footprint_geometry(rings[i], pl.ring), footprint_geometry(rings[j], pl.ring)
    else:
        pl, g1, g2 = plane, clipped[i], clipped[j]  # already clipped to the AOI
    inter = g1.intersection(g2)
    area = inter.area
    if area <= 0:
        return None
    aoi_cov = area / aoi_area if plane is not None and aoi_area > 0 else np.nan
    c = inter.centroid
    glat, glon = pl.unproject(c.x, c.y)
    conv = bh = np.nan
    if has_geom[i] and has_geom[j]:
        conv, bh = convergence_angle(
            (glat, glon),
            (sclat[i], sclon[i], alt[i]),
            (sclat[j], sclon[j], alt[j]),
            _latlon(img_centres[i]),
            _latlon(img_centres[j]),
        )
    return {
        "convergence": conv,
        "b_h": bh,
        "overlap_km2": area,
        "overlap_frac": area / max(min(g1.area, g2.area), 1e-12),
        "aoi_coverage": aoi_cov,
        "overlap_lat": glat,
        "overlap_lon": glon,
    }
