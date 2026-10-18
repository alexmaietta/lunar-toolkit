"""Illumination-diverse image selection for shape-from-shading."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import shapely

from .geometry import AoiLike, footprint_geometry
from .stereo import angle_between, aoi_plane, azimuth_difference, observation_footprints

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import pandas as pd

Prefer: TypeAlias = "str | Callable[[pd.DataFrame], pd.Series]"


def sort_by_sun_azimuth(
    obs_df: pd.DataFrame,
    start: float = 0.0,
    column: str = "sub_solar_azimuth",
) -> pd.DataFrame:
    """Sort rows by Sun azimuth, measured clockwise from *start* degrees.

    Rows with an unknown azimuth go last.  The index is preserved.
    """
    import pandas as pd

    az = pd.to_numeric(obs_df[column], errors="coerce")
    key = (az - start) % 360.0
    return obs_df.assign(_k=key).sort_values("_k", kind="stable", na_position="last").drop(columns="_k")


def aoi_coverage(obs_df: pd.DataFrame, aoi: AoiLike) -> pd.Series:
    """Fraction (0–1) of the AOI covered by each row's footprint.

    *aoi* is ``{"point": (lat, lon), "radius_km": r}``,
    ``{"lat": (s, n), "lon": (w, e)}`` or anything
    :func:`lunar_toolkit.geometry.load_aoi` accepts.
    """
    import pandas as pd

    geoms, target = _aoi_cover(obs_df, aoi)
    area = target.area
    vals = [g.intersection(target).area / area if g is not None and area > 0 else 0.0 for g in geoms]
    return pd.Series(vals, index=obs_df.index, name="aoi_coverage", dtype=float)


def _aoi_cover(obs_df: pd.DataFrame, aoi: AoiLike) -> tuple[list, Any]:
    """Per row the footprint in the AOI's tangent plane (``None`` if absent / disjoint), and the AOI."""
    plane, target, centre, radius = aoi_plane(aoi)
    rings, centres, radii = observation_footprints(obs_df)
    near = angle_between(centres, centre[None, :]) < radii + radius + 1e-6
    geoms = [
        footprint_geometry(r, plane.ring) if r is not None and near[k] else None for k, r in enumerate(rings)
    ]
    return geoms, target


def _candidates(
    obs_df: pd.DataFrame,
    incidence: tuple[float, float] | None,
    max_emission: float | None,
    aoi: AoiLike | None,
    min_aoi_coverage: float,
) -> pd.DataFrame:
    import pandas as pd

    df = obs_df.copy()
    df["sub_solar_azimuth"] = pd.to_numeric(df["sub_solar_azimuth"], errors="coerce")
    df = df[df["sub_solar_azimuth"].notna()]
    if incidence is not None:
        inc = pd.to_numeric(df["incidence"], errors="coerce")
        df = df[inc.between(min(incidence), max(incidence))]
    if max_emission is not None:
        emi = pd.to_numeric(df["emission"], errors="coerce")
        df = df[~(emi > max_emission)]
    if aoi is not None and not df.empty:
        df["aoi_coverage"] = aoi_coverage(df, aoi)
        df = df[(df["aoi_coverage"] > 0) & (df["aoi_coverage"] >= min_aoi_coverage)]
    else:
        df["aoi_coverage"] = np.nan
    return df


def _quality(df: pd.DataFrame, prefer: Prefer) -> pd.Series:
    """Selection score (higher is better); see :func:`select_illumination_diverse`."""
    import pandas as pd

    if callable(prefer):
        return pd.Series(prefer(df), index=df.index, dtype=float)
    ps = (
        pd.to_numeric(df["pixel_scale"], errors="coerce")
        if "pixel_scale" in df
        else pd.Series(np.nan, df.index)
    )
    res = (ps.min() / ps).fillna(0.5) if ps.notna().any() else pd.Series(0.5, index=df.index)
    cov = df["aoi_coverage"].fillna(1.0)
    if prefer == "balanced":
        return cov * np.sqrt(res)
    if prefer == "coverage":
        return cov + 1e-3 * res
    if prefer == "resolution":
        return res + 1e-3 * cov
    raise ValueError(f"prefer must be 'balanced', 'coverage', 'resolution' or a callable, not {prefer!r}")


def _thin_to(df: pd.DataFrame, max_images: int) -> pd.DataFrame:
    """Drop picks until *max_images* remain, always removing the lower-scored
    image of the closest pair in azimuth (keeps the azimuth spread)."""
    df = df.copy()
    while len(df) > max_images:
        az = df["sub_solar_azimuth"].to_numpy(float)
        d = azimuth_difference(az[:, None], az[None, :])
        np.fill_diagonal(d, np.inf)
        a, b = np.unravel_index(np.argmin(d), d.shape)
        sc = df["selection_score"].to_numpy(float)
        drop = a if sc[a] < sc[b] or (sc[a] == sc[b] and a > b) else b
        df = df.iloc[np.delete(np.arange(len(df)), drop)]
    return df


def select_illumination_diverse(
    obs_df: pd.DataFrame,
    azimuth_bin_deg: float = 30.0,
    incidence: tuple[float, float] | None = None,
    per_bin: int = 1,
    max_images: int | None = None,
    prefer: Prefer = "balanced",
    aoi: AoiLike | None = None,
    min_aoi_coverage: float = 0.0,
    incidence_bin_deg: float | None = None,
    max_emission: float | None = None,
    method: str = "bins",
    azimuth_origin: float = 0.0,
) -> pd.DataFrame:
    """Choose images spanning diverse Sun azimuths (for SfS / PSR mapping).

    Parameters
    ----------
    obs_df : DataFrame
        Observations in the common schema.
    azimuth_bin_deg : float
        Bin width (``method="bins"``) or minimum separation
        (``method="greedy"``) in sub-solar azimuth, degrees.
    incidence : (min, max), optional
        Keep only images with incidence in this range (e.g. ``(80, 95)`` at
        the poles; SfS needs lit but low-Sun images).
    per_bin : int
        Images kept per bin.  With ``method="greedy"``: at most this many
        picks may lie within *azimuth_bin_deg* of each other.
    max_images : int, optional
        Cap on the result.  Excess picks are removed by repeatedly dropping
        the lower-scored image of the pair closest in azimuth, so the
        azimuth spread is kept.
    prefer : {"balanced", "coverage", "resolution"} or callable
        How to rank images competing for a bin.  With ``res`` = finest
        ``pixel_scale`` among candidates / own ``pixel_scale`` (0.5 if
        unknown) and ``cov`` = AOI coverage fraction (1 without *aoi*):
        ``"balanced"`` = ``cov * sqrt(res)``; ``"coverage"`` ranks by
        coverage then resolution; ``"resolution"`` by resolution then
        coverage.  A callable gets the candidate DataFrame and returns the
        score (higher is better).
    aoi : dict, str, Path or Aoi, optional
        ``{"point": (lat, lon), "radius_km": r}``, ``{"lat": ..., "lon": ...}``
        or anything :func:`lunar_toolkit.geometry.load_aoi` accepts; images
        not overlapping it are dropped.
    min_aoi_coverage : float
        Minimum fraction of the AOI an image must cover (with *aoi*).
    incidence_bin_deg : float, optional
        Also bin by incidence (``method="bins"`` only), giving azimuth ×
        incidence cells — useful when shadow length diversity matters too.
    max_emission : float, optional
        Drop images with larger emission angles (off-nadir views are
        foreshortened and complicate SfS).
    method : {"bins", "greedy"}
        ``"bins"``: fixed bins ``[origin + k*w, origin + (k+1)*w)``, best
        *per_bin* per bin (two picks in neighbouring bins may still be close
        in azimuth).  ``"greedy"``: candidates in descending score; accept
        one unless *per_bin* accepted images already lie within
        *azimuth_bin_deg* of it, so with ``per_bin=1`` all picks are at
        least *azimuth_bin_deg* apart (the Bertone et al. criterion).
    azimuth_origin : float
        Start of the first azimuth bin, degrees.

    Returns
    -------
    DataFrame
        The selected rows (original index kept) sorted by
        ``sub_solar_azimuth`` (see :func:`sort_by_sun_azimuth`), with extra
        columns ``aoi_coverage`` (NaN without *aoi*), ``azimuth_bin`` (bin
        start, deg; NaN for greedy), ``incidence_bin`` (if binned) and
        ``selection_score``.

    Notes
    -----
    This uses each image's single scene-centre Sun azimuth; for a large AOI
    where per-pixel diversity matters, use :func:`select_for_coverage`.
    """
    import pandas as pd

    if azimuth_bin_deg <= 0 or per_bin < 1:
        raise ValueError("azimuth_bin_deg must be > 0 and per_bin >= 1")
    df = _candidates(obs_df, incidence, max_emission, aoi, min_aoi_coverage)
    if df.empty:
        return df.assign(azimuth_bin=pd.Series(dtype=float), selection_score=pd.Series(dtype=float))
    df["selection_score"] = _quality(df, prefer)
    df = df.sort_values("selection_score", ascending=False, kind="stable")

    if method == "bins":
        rel = (df["sub_solar_azimuth"] - azimuth_origin) % 360.0
        df["azimuth_bin"] = (np.floor(rel / azimuth_bin_deg) * azimuth_bin_deg + azimuth_origin) % 360.0
        keys = ["azimuth_bin"]
        if incidence_bin_deg:
            inc = pd.to_numeric(df["incidence"], errors="coerce")
            df["incidence_bin"] = np.floor(inc / incidence_bin_deg) * incidence_bin_deg
            keys.append("incidence_bin")
        picked = df.groupby(keys, sort=False, dropna=False).head(per_bin)
    elif method == "greedy":
        df["azimuth_bin"] = np.nan
        accepted: list[float] = []
        keep = []
        for idx, a in zip(df.index, df["sub_solar_azimuth"].to_numpy(float)):
            near = int(np.sum(azimuth_difference(np.array(accepted), a) < azimuth_bin_deg)) if accepted else 0
            if near < per_bin:
                accepted.append(a)
                keep.append(idx)
        picked = df.loc[keep]
    else:
        raise ValueError(f"method must be 'bins' or 'greedy', not {method!r}")

    if max_images is not None and len(picked) > max_images:
        picked = _thin_to(picked, max_images)
    return sort_by_sun_azimuth(picked)


def _grid(target, grid_km: float | None, max_cells: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres ``(x, y)`` of a regular grid over *target* that fall inside it."""
    x0, y0, x1, y1 = target.bounds
    if grid_km is None:
        grid_km = max(max(x1 - x0, y1 - y0) / 100.0, 1e-3)
    nx = max(int(math.ceil((x1 - x0) / grid_km)), 1)
    ny = max(int(math.ceil((y1 - y0) / grid_km)), 1)
    if nx * ny > max_cells:
        f = math.sqrt(nx * ny / max_cells)
        nx, ny = max(int(nx / f), 1), max(int(ny / f), 1)
    gx, gy = np.meshgrid(
        x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx, y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    )
    gx, gy = gx.ravel(), gy.ravel()
    inside = shapely.intersects_xy(target, gx, gy)
    return gx[inside], gy[inside]


def select_for_coverage(
    obs_df: pd.DataFrame,
    aoi: AoiLike,
    azimuth_sep_deg: float = 30.0,
    views_per_cell: int = 3,
    incidence: tuple[float, float] | None = None,
    max_images: int | None = None,
    prefer: Prefer = "balanced",
    max_emission: float | None = None,
    grid_km: float | None = None,
    min_gain: int = 1,
) -> pd.DataFrame:
    """Greedy, per-location illumination-diverse selection over an AOI.

    The AOI is sampled on a grid (``grid_km`` spacing; default ~100 cells
    across).  Each grid cell keeps the Sun azimuths of the selected images
    covering it.  A candidate *helps* a cell if it covers the cell, the cell
    has fewer than *views_per_cell* views and the candidate's azimuth is at
    least *azimuth_sep_deg* from every azimuth already there.  At each step
    the image maximising ``helped_cells * (0.5 + 0.5 * score)`` is taken
    (score as in :func:`select_illumination_diverse`, normalised to 0–1),
    until no image helps *min_gain* cells or *max_images* is reached.

    Returns
    -------
    DataFrame
        Selected rows sorted by ``sub_solar_azimuth`` with extra columns
        ``aoi_coverage``, ``selection_score``, ``selection_order`` (0 = first
        pick) and ``cells_gained`` (fraction of AOI cells the pick helped).
        ``attrs["view_fractions"]`` maps a view count *k* to the fraction of
        AOI cells with exactly *k* non-redundant azimuths
        (``views_per_cell`` = fully served), and ``attrs["grid_km"]`` /
        ``attrs["n_cells"]`` describe the grid.
    """
    import pandas as pd

    df = _candidates(obs_df, incidence, max_emission, aoi, 0.0)
    if df.empty:
        out = df.assign(
            selection_score=pd.Series(dtype=float),
            selection_order=pd.Series(dtype=int),
            cells_gained=pd.Series(dtype=float),
        )
        out.attrs["view_fractions"] = {}
        return out
    score = _quality(df, prefer).to_numpy(float)
    smax = np.nanmax(score) if np.isfinite(score).any() else 1.0
    norm = np.nan_to_num(score / (smax or 1.0), nan=0.0)
    df["selection_score"] = score

    geoms, target = _aoi_cover(df, aoi)
    cx, cy = _grid(target, grid_km)
    if not len(cx):
        raise ValueError("AOI grid is empty; pass a smaller grid_km")
    cover = np.stack(
        [shapely.intersects_xy(g, cx, cy) if g is not None else np.zeros(len(cx), bool) for g in geoms]
    )
    n_cells = len(cx)
    az = df["sub_solar_azimuth"].to_numpy(float)
    views = np.full((n_cells, views_per_cell), np.nan)
    count = np.zeros(n_cells, int)
    available = np.ones(len(df), bool)
    order: list[int] = []
    gained: list[float] = []
    while available.any() and (max_images is None or len(order) < max_images):
        best, best_val, best_mask = -1, 0.0, None
        open_cells = count < views_per_cell
        for k in np.flatnonzero(available):
            m = cover[k] & open_cells
            if not m.any():
                continue
            d = azimuth_difference(views[m], az[k])
            ok = np.all(np.isnan(d) | (d >= azimuth_sep_deg), axis=1)
            g = int(ok.sum())
            val = g * (0.5 + 0.5 * norm[k])
            if g >= min_gain and val > best_val:
                best, best_val = int(k), val
                best_mask = np.flatnonzero(m)[ok]
        if best_mask is None:
            break
        views[best_mask, count[best_mask]] = az[best]
        count[best_mask] += 1
        available[best] = False
        order.append(best)
        gained.append(len(best_mask) / n_cells)

    picked = df.iloc[order].copy()
    picked["selection_order"] = np.arange(len(order))
    picked["cells_gained"] = gained
    out = sort_by_sun_azimuth(picked)
    out.attrs["view_fractions"] = {int(k): float(np.mean(count == k)) for k in range(views_per_cell + 1)}
    out.attrs["grid_km"] = float(np.sqrt(target.area / n_cells))
    out.attrs["n_cells"] = int(n_cells)
    return out


def selection_distances(
    coverage: np.ndarray,
    azimuth: np.ndarray,
    weights: tuple[float, float] = (1.0, 1.0),
    chunk: int = 256,
) -> np.ndarray:
    """Image-to-image distance of Bertone et al. (2026, sec. 2) for k-medoids selection.

    ``d_ij = sqrt((w_a * a_ij)**2 + (w_c * c_ij)**2)`` with ``a_ij`` the
    circular Sun azimuth (sub-solar longitude) difference and ``c_ij`` the
    fraction of grid cells covered by exactly one of the two images, each
    divided by its largest value over all pairs (so both span 0-1).

    Parameters
    ----------
    coverage : (n_images, n_cells) bool array
        Per-image coverage (or lit) masks on a common grid.
    azimuth : (n_images,) array
        Sun azimuths, degrees (finite).
    weights : (w_a, w_c)
        Weights of the azimuth and coverage terms.
    chunk : int
        Rows per block of the coverage products (memory).

    Returns
    -------
    (n_images, n_images) float array
    """
    cov = np.asarray(coverage, bool)
    az = np.asarray(azimuth, float)
    n, cells = cov.shape
    a = azimuth_difference(az[:, None], az[None, :])
    f = cov.astype(np.float32)
    counts = f.sum(axis=1)
    c = np.empty((n, n))
    for i in range(0, n, chunk):
        inter = f[i : i + chunk] @ f.T
        c[i : i + chunk] = (counts[i : i + chunk, None] + counts[None, :] - 2.0 * inter) / max(cells, 1)
    np.fill_diagonal(c, 0.0)
    a = a / np.max(a) if np.max(a) > 0 else a
    c = c / np.max(c) if np.max(c) > 0 else c
    return np.hypot(weights[0] * a, weights[1] * c)


def _build(d: np.ndarray, k: int) -> np.ndarray:
    """PAM BUILD: the most central object, then each object that lowers the total distance most."""
    med = [int(np.argmin(d.sum(axis=1)))]
    near = d[:, med[0]].copy()
    for _ in range(1, k):
        gain = np.clip(near[:, None] - d, 0.0, None).sum(axis=0)
        gain[med] = -1.0
        c = int(np.argmax(gain))
        med.append(c)
        near = np.minimum(near, d[:, c])
    return np.sort(med)


def kmedoids(
    dist: np.ndarray,
    k: int,
    init: str = "build",
    seed: int | None = 0,
    n_init: int = 1,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating k-medoids (Park & Jun 2009) on a precomputed distance matrix.

    Assign every object to its nearest medoid, move each medoid to the
    member with the smallest total distance to its cluster, repeat until the
    medoids stop changing.

    Parameters
    ----------
    dist : (n, n) array
        Symmetric distances.
    k : int
        Number of clusters (all objects are medoids when ``k >= n``).
    init : {"build", "park_jun", "random"}
        ``"build"``: the PAM BUILD step (most central object, then greedy
        cost reduction; deterministic and robust to duplicate images);
        ``"park_jun"``: the *k* objects with the smallest
        ``v_j = sum_i d_ij / sum_l d_il`` (Park & Jun's own start, which
        favours central objects); ``"random"``: *k* distinct objects drawn
        with *seed*, best of *n_init* runs.
    seed : int, optional
        Random generator seed (``init="random"``; ties are always broken by
        the lowest index).
    n_init : int
        Random restarts (``init="random"``).
    max_iter : int
        Iteration cap.

    Returns
    -------
    (medoids, labels)
        Sorted medoid indices and each object's cluster (index into
        *medoids*).
    """
    d = np.asarray(dist, float)
    n = len(d)
    if n == 0 or k <= 0:
        return np.empty(0, int), np.full(n, -1)
    if k >= n:
        return np.arange(n), np.arange(n)
    if init == "build":
        starts = [_build(d, k)]
    elif init == "park_jun":
        row = d.sum(axis=1)
        v = (d / np.where(row > 0, row, 1.0)[:, None]).sum(axis=0)
        starts = [np.sort(np.argsort(v, kind="stable")[:k])]
    elif init == "random":
        rng = np.random.default_rng(seed)
        starts = [np.sort(rng.choice(n, k, replace=False)) for _ in range(max(1, n_init))]
    else:
        raise ValueError(f"init must be 'build', 'park_jun' or 'random', not {init!r}")
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for med in starts:
        for _ in range(max_iter):
            labels = np.argmin(d[:, med], axis=1)
            labels[med] = np.arange(k)  # duplicates of a medoid stay in their own cluster
            new = med.copy()
            for j in range(k):
                members = np.flatnonzero(labels == j)
                new[j] = members[np.argmin(d[np.ix_(members, members)].sum(axis=1))]
            new = np.sort(new)
            if np.array_equal(new, med):
                break
            med = new
        labels = np.argmin(d[:, med], axis=1)
        labels[med] = np.arange(k)
        cost = float(d[np.arange(n), med[labels]].sum())
        if best is None or cost < best[0] - 1e-12:
            best = (cost, med, labels)
    assert best is not None
    return best[1], best[2]


@dataclass
class KMedoidsSelection:
    """Result of :func:`select_kmedoids` (indices refer to the input rows).

    Attributes
    ----------
    chosen : list of int
        Selected images in pick order: medoids first, then the coverage and
        illumination refinements.
    stage : list of str
        Per entry of *chosen*: ``"medoid"``, ``"coverage"`` or ``"illumination"``.
    medoids, labels : array
        The k-medoids clustering (``labels[i]`` indexes *medoids*; -1 for
        images without an azimuth).
    directions : (n_cells,) int array
        Distinct illumination directions reached per cell (capped at *views*).
    uncovered, poorly_lit : float
        Fractions of the potentially lit cells with no selected image, and
        with fewer than *views* directions.
    """

    chosen: list[int]
    stage: list[str]
    medoids: np.ndarray
    labels: np.ndarray
    directions: np.ndarray
    uncovered: float
    poorly_lit: float
    history: list[dict[str, Any]] = field(default_factory=list)


def select_kmedoids(
    lit: np.ndarray,
    azimuth: np.ndarray,
    n_clusters: int = 30,
    *,
    coverage: np.ndarray | None = None,
    views: int = 3,
    azimuth_sep: float = 30.0,
    max_uncovered: float = 0.005,
    max_poorly_lit: float = 0.01,
    max_images: int | None = None,
    weights: tuple[float, float] = (1.0, 1.0),
    init: str = "build",
    seed: int | None = 0,
    n_init: int = 1,
) -> KMedoidsSelection:
    """k-medoids image selection with coverage and illumination refinement (Bertone et al. 2026, sec. 2).

    1. Cluster the images with :func:`kmedoids` on
       :func:`selection_distances` (Sun azimuth and coverage differences)
       and take the *n_clusters* medoids.
    2. Add the image lighting the most still-uncovered cells until fewer
       than *max_uncovered* of the potentially lit cells (lit in any image)
       are uncovered.
    3. Add the image giving the most cells a new direction (Sun azimuth at
       least *azimuth_sep* from the cell's directions so far, up to *views*)
       until fewer than *max_poorly_lit* of the potentially lit cells have
       fewer than *views* directions.

    Steps 2-3 stop early when no image helps or *max_images* is reached;
    ties go to the lowest index, so the result is deterministic (for
    ``init="random"`` given *seed*).

    Parameters
    ----------
    lit : (n_images, n_cells) bool array
        Lit (valid and above the shadow threshold) cells per image on a
        common grid, e.g. ``SfsPipeline._lit_masks()``.
    azimuth : (n_images,) array
        Sun azimuths in degrees; images with NaN are never selected.
    n_clusters : int
        Medoids (the paper's ~30 user-defined images).
    coverage : (n_images, n_cells) bool array, optional
        Footprint masks for the distance (default *lit*).
    views, azimuth_sep : int, float
        Directions wanted per cell and their minimum separation (3, 30 deg).
    max_uncovered, max_poorly_lit : float
        Stopping fractions of steps 2 and 3 (0.5 % and 1 %).
    max_images : int, optional
        Cap on the total (medoids are truncated to it first).
    weights, init, seed, n_init
        See :func:`selection_distances` and :func:`kmedoids`.
    """
    lit = np.asarray(lit, bool)
    az = np.asarray(azimuth, float)
    n, cells = lit.shape
    cov = lit if coverage is None else np.asarray(coverage, bool)
    valid = np.flatnonzero(np.isfinite(az))
    labels = np.full(n, -1)
    if len(valid):
        med, lab = kmedoids(
            selection_distances(cov[valid], az[valid], weights), n_clusters, init, seed, n_init
        )
        medoids = valid[med]
        labels[valid] = lab
    else:
        medoids = np.empty(0, int)
    cap = n if max_images is None else max(0, max_images)
    chosen = [int(i) for i in medoids[:cap]]
    stage = ["medoid"] * len(chosen)

    potential = lit[valid].any(axis=0) if len(valid) else np.zeros(cells, bool)
    n_pot = max(int(potential.sum()), 1)
    dirs = np.full((cells, views), np.nan)
    count = np.zeros(cells, int)

    def add(k: int) -> None:
        idx = np.flatnonzero(lit[k] & (count < views))
        if len(idx):
            d = azimuth_difference(dirs[idx], az[k])
            idx = idx[np.all(np.isnan(d) | (d >= azimuth_sep), axis=1)]
            dirs[idx, count[idx]] = az[k]
            count[idx] += 1

    for k in chosen:
        add(k)
    covered = lit[chosen].any(axis=0) if chosen else np.zeros(cells, bool)
    available = np.zeros(n, bool)
    available[valid] = True
    available[chosen] = False
    history: list[dict[str, Any]] = []

    def fractions() -> tuple[float, float]:
        return (
            float(np.sum(potential & ~covered) / n_pot),
            float(np.sum(potential & (count < views)) / n_pot),
        )

    while len(chosen) < cap and fractions()[0] >= max_uncovered and available.any():
        cand = np.flatnonzero(available)
        gain = (lit[cand] & ~covered & potential).sum(axis=1)
        if gain.max() <= 0:
            break
        k = int(cand[np.argmax(gain)])
        chosen.append(k)
        stage.append("coverage")
        covered |= lit[k]
        available[k] = False
        add(k)
        history.append({"image": k, "stage": "coverage", "gain": int(gain.max())})

    while len(chosen) < cap and fractions()[1] >= max_poorly_lit and available.any():
        open_cells = potential & (count < views)
        best, best_gain = -1, 0
        for j in np.flatnonzero(available):
            idx = np.flatnonzero(lit[j] & open_cells)
            if len(idx) <= best_gain:
                continue
            d = azimuth_difference(dirs[idx], az[j])
            g = int(np.all(np.isnan(d) | (d >= azimuth_sep), axis=1).sum())
            if g > best_gain:
                best, best_gain = int(j), g
        if best < 0:
            break
        chosen.append(best)
        stage.append("illumination")
        covered |= lit[best]
        available[best] = False
        add(best)
        history.append({"image": best, "stage": "illumination", "gain": best_gain})

    unc, poor = fractions()
    return KMedoidsSelection(chosen, stage, medoids, labels, count, unc, poor, history)


def coverage_matrix(
    obs_df: pd.DataFrame, aoi: AoiLike, grid_km: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Footprint coverage of the AOI grid cells: ``(cover[n_images, n_cells], x, y)``.

    The grid is the one :func:`select_for_coverage` uses (cell centres in
    the AOI's tangent plane, km).
    """
    geoms, target = _aoi_cover(obs_df, aoi)
    cx, cy = _grid(target, grid_km)
    cover = (
        np.stack(
            [shapely.intersects_xy(g, cx, cy) if g is not None else np.zeros(len(cx), bool) for g in geoms]
        )
        if geoms
        else np.zeros((0, len(cx)), bool)
    )
    return cover, cx, cy


def select_kmedoids_footprints(
    obs_df: pd.DataFrame,
    aoi: AoiLike,
    n_clusters: int = 30,
    incidence: tuple[float, float] | None = None,
    max_emission: float | None = None,
    grid_km: float | None = None,
    **options: Any,
) -> pd.DataFrame:
    """:func:`select_kmedoids` on catalogue footprints over an AOI (footprint = lit, no shadows).

    *options* go to :func:`select_kmedoids`.  Returns the selected rows
    sorted by ``sub_solar_azimuth`` with ``selection_order``,
    ``selection_stage`` and ``cluster`` columns; ``attrs`` holds
    ``uncovered``, ``poorly_lit`` and ``n_cells``.
    """
    import pandas as pd

    df = _candidates(obs_df, incidence, max_emission, aoi, 0.0)
    if df.empty:
        out = df.assign(
            selection_order=pd.Series(dtype=int),
            selection_stage=pd.Series(dtype=str),
            cluster=pd.Series(dtype=int),
        )
        out.attrs.update(uncovered=1.0, poorly_lit=1.0, n_cells=0)
        return out
    cover, cx, _ = coverage_matrix(df, aoi, grid_km)
    res = select_kmedoids(cover, df["sub_solar_azimuth"].to_numpy(float), n_clusters, **options)
    picked = df.iloc[res.chosen].copy()
    picked["selection_order"] = np.arange(len(res.chosen))
    picked["selection_stage"] = res.stage
    picked["cluster"] = res.labels[res.chosen]
    out = sort_by_sun_azimuth(picked)
    out.attrs.update(uncovered=res.uncovered, poorly_lit=res.poorly_lit, n_cells=int(len(cx)))
    return out
