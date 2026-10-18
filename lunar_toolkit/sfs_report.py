"""Validation report of an SfS work dir: metrics, figures and HTML (Bertone et al. 2026, Sec. 3)."""

from __future__ import annotations

import base64
import html
import json
import logging
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import asp
from .sfs_products import Grid, illumination_maps, product_name, read_raster, slope_deg
from .tools import ToolError

log = logging.getLogger(__name__)

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
COVERAGE_COLORS = ("#000000", "#e34948", "#eda100", "#b8b8b8")
COVERAGE_LABELS = ("0", "1", "2", ">=3")
DIFF_EDGES = np.linspace(-5.0, 5.0, 101)
NORM_EDGES = np.linspace(-5.0, 5.0, 101)
SLOPE_EDGES = (0, 5, 10, 15, 20, 25, 30, 35, 40, 90)
BA_STAGES = (("ba", "ba/run"), ("refine", "ba_refine/run"))

SECTIONS = {
    "ba": "Bundle adjustment residuals and match offsets (Fig. 5)",
    "removed": "Removed images (Sec. 2)",
    "coverage": "Illumination coverage (Figs. 3, 6)",
    "prior": "SfS minus prior DEM (Table 2, Fig. 7 left)",
    "lola": "SfS minus LOLA points vs slope (Table 2, Fig. 7)",
    "ensemble": "Ensemble uncertainty (Fig. 8)",
    "uncertainty": "Residuals normalised by the LDEM uncertainty maps (Fig. 9)",
    "published": "Comparison with the published SDEM (GLDELEV)",
}


@dataclass
class ReportInputs:
    """Files and settings of a report; anything None is discovered or its section skipped.

    Parameters
    ----------
    work_dir : Path
        An :class:`~lunar_toolkit.sfs.SfsPipeline` work directory.
    sfs_dem, prior_dem : Path, optional
        Default ``products/<ROI>_GLDELEV_<RES>.tif`` (else ``sfs/run-DEM-final.tif``)
        and ``dem/prior-<RES>m.tif``.
    region_id, resolution : optional
        Product name parts (default: ``config.yaml``, else the work dir name and 5).
    shadow_threshold, max_offset_p85 : float, optional
        Default from ``config.yaml`` (or the Bertone preset).
    ldec, toterr, slperr, ldem_slope : Path, optional
        PGDA LOLA count, elevation / slope uncertainty and slope rasters.
    lola_points : Path, optional
        ASP CSV of LOLA points (:func:`lunar_toolkit.reference.write_asp_csv`).
    lola_xyzi : Path, optional
        PGDA ``.xyzi`` point cloud, converted to *lola_points* when those are not given.
    published_dem : Path, optional
        Published Bertone et al. SDEM (``<ROI>_GLDELEV_005.tif``).
    slope_edges : sequence of float
        Slope bins (deg) for the differences vs slope.
    slope_split : float
        Slope (deg) separating the low / high-slope ensemble regimes.
    preview_size : int
        Longer side (px) of the grid maps and coverage comparisons are drawn on.
    """

    work_dir: Path
    sfs_dem: Path | None = None
    prior_dem: Path | None = None
    region_id: str | None = None
    resolution: float | None = None
    shadow_threshold: float | None = None
    max_offset_p85: float | None = None
    ldec: Path | None = None
    toterr: Path | None = None
    slperr: Path | None = None
    ldem_slope: Path | None = None
    lola_points: Path | None = None
    lola_xyzi: Path | None = None
    published_dem: Path | None = None
    slope_edges: Sequence[float] = SLOPE_EDGES
    slope_split: float = 20.0
    preview_size: int = 800

    @classmethod
    def from_work_dir(cls, work_dir: Path, **overrides: Any) -> ReportInputs:
        """Inputs with defaults read from ``<work_dir>/config.yaml`` and the standard paths."""
        work_dir = Path(work_dir)
        kw: dict[str, Any] = {k: v for k, v in overrides.items() if v is not None}
        cfg = _work_dir_config(work_dir)
        if cfg is not None:
            kw.setdefault("resolution", cfg.resolution)
            kw.setdefault("region_id", cfg.region_id)
            kw.setdefault("shadow_threshold", cfg.options("sfs").get("shadow_threshold"))
            kw.setdefault("max_offset_p85", cfg.max_offset_p85)
        inp = cls(work_dir, **{k: v for k, v in kw.items() if v is not None})
        inp.resolution = inp.resolution or 5.0
        inp.region_id = inp.region_id or work_dir.resolve().name
        inp.shadow_threshold = 0.002 if inp.shadow_threshold is None else inp.shadow_threshold
        inp.max_offset_p85 = 1.8 if inp.max_offset_p85 is None else inp.max_offset_p85
        if inp.sfs_dem is None:
            inp.sfs_dem = _first_existing(inp.product("ELEV"), work_dir / "sfs" / "run-DEM-final.tif")
        if inp.prior_dem is None:
            inp.prior_dem = _first_existing(work_dir / "dem" / f"prior-{inp.resolution:g}m.tif")
        return inp

    def product(self, code: str, *suffix: Any) -> Path:
        """``<work_dir>/products/<ROI>_GLD<code>_<RES>[_suffix].tif``."""
        return (
            self.work_dir
            / "products"
            / product_name(self.region_id or "", code, self.resolution or 5, *suffix)
        )


def _work_dir_config(work_dir: Path) -> Any:
    path = work_dir / "config.yaml"
    if not path.exists():
        return None
    from .config import ConfigError, load_config

    try:
        return load_config(path).sfs
    except ConfigError as exc:
        log.warning("Ignoring %s: %s", path, exc)
        return None


def _first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


@dataclass
class Histogram:
    """Counts of *values* in *edges*; values beyond the edges go to the outer bins."""

    label: str
    edges: np.ndarray
    counts: np.ndarray

    @classmethod
    def of(cls, label: str, values: np.ndarray, edges: np.ndarray | Sequence[float]) -> Histogram:
        e = np.asarray(edges, float)
        v = np.asarray(values, float)
        v = np.clip(v[np.isfinite(v)], e[0], e[-1])
        return cls(label, e, np.histogram(v, e)[0])

    @property
    def total(self) -> int:
        return int(self.counts.sum())


@dataclass
class Metrics:
    """Everything a report shows: JSON summary, tables and histograms per section."""

    inputs: ReportInputs
    summary: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    histograms: dict[str, list[Histogram]] = field(default_factory=dict)
    hist2d: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def stats(values: Any) -> dict[str, float]:
    """Count, mean, median, std, NMAD, RMS, range, 5/95th percentiles and the same of ``|values|``."""
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"count": 0}
    med = float(np.median(v))
    a = np.abs(v)
    return {
        "count": int(v.size),
        "mean": float(v.mean()),
        "median": med,
        "std": float(v.std()),
        "nmad": float(1.4826 * np.median(np.abs(v - med))),
        "rms": float(np.sqrt(np.mean(v * v))),
        "min": float(v.min()),
        "max": float(v.max()),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "abs_mean": float(a.mean()),
        "abs_median": float(np.median(a)),
        "abs_std": float(a.std()),
    }


def split_stats(values: np.ndarray, ldec: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """:func:`stats` of all values and, with a LOLA count raster, of ``LDEC > 0`` / ``LDEC = 0`` pixels."""
    out = {"all": stats(values)}
    if ldec is not None:
        ok = np.isfinite(ldec)
        out["ldec>0"] = stats(values[ok & (ldec > 0)])
        out["ldec=0"] = stats(values[ok & (ldec <= 0)])
    return out


def binned(values: np.ndarray, by: np.ndarray, edges: Sequence[float]) -> pd.DataFrame:
    """Per bin of *by*: count, median, quartiles, mean ``|values|`` and RMS of *values*."""
    v, b = np.asarray(values, float).ravel(), np.asarray(by, float).ravel()
    ok = np.isfinite(v) & np.isfinite(b)
    v, b = v[ok], b[ok]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = v[(b >= lo) & (b < hi)]
        row: dict[str, Any] = {"bin_min": lo, "bin_max": hi, "count": int(x.size)}
        if x.size:
            p25, p50, p75 = np.percentile(x, [25, 50, 75])
            row.update(
                median=p50,
                p25=p25,
                p75=p75,
                abs_mean=float(np.abs(x).mean()),
                rms=float(np.sqrt(np.mean(x * x))),
            )
        rows.append(row)
    return pd.DataFrame(
        rows, columns=["bin_min", "bin_max", "count", "median", "p25", "p75", "abs_mean", "rms"]
    )


def normal_summary(z: np.ndarray) -> dict[str, float]:
    """How close normalised residuals are to a unit Gaussian: std, NMAD and fractions within 1-3 sigma."""
    z = np.asarray(z, float).ravel()
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {"count": 0}
    a = np.abs(z)
    return {
        "count": int(z.size),
        "mean": float(z.mean()),
        "std": float(z.std()),
        "nmad": float(1.4826 * np.median(np.abs(z - np.median(z)))),
        "within_1": float(np.mean(a < 1)),
        "within_2": float(np.mean(a < 2)),
        "within_3": float(np.mean(a < 3)),
    }


def coverage_fractions(solar_bins: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    """Fractions of pixels lit from 0, 1, 2 and >=3 non-redundant Sun directions (and >=1, >=2)."""
    b = np.asarray(solar_bins)
    b = b[valid] if valid is not None else b.ravel()
    n = b.size
    if n == 0:
        return {"pixels": 0}
    f = [float(np.mean(b == k)) for k in (0, 1, 2)] + [float(np.mean(b >= 3))]
    return {
        "pixels": int(n),
        "frac_0": f[0],
        "frac_1": f[1],
        "frac_2": f[2],
        "frac_3plus": f[3],
        "frac_ge1": f[1] + f[2] + f[3],
        "frac_ge2": f[2] + f[3],
    }


def read_ba_reports(prefix: Path) -> pd.DataFrame | None:
    """Per-image bundle-adjustment report of ``<prefix>-*``: residuals before / after and match offsets.

    Columns ``image``, ``initial_mean`` / ``initial_median`` / ``initial_count``,
    ``final_*`` (pixels) and ``offset_p25`` ... ``offset_p95``, ``offset_count`` (m).
    None when none of the reports exist.
    """
    files = {
        "initial": Path(f"{prefix}-initial_residuals_stats.txt"),
        "final": Path(f"{prefix}-final_residuals_stats.txt"),
        "offsets": Path(f"{prefix}-mapproj_match_offset_stats.txt"),
    }
    if not any(p.exists() for p in files.values()):
        return None
    rows: dict[str, dict[str, Any]] = {}
    for key in ("initial", "final"):
        if files[key].exists():
            for name, s in asp.read_residual_stats(files[key]).items():
                rows.setdefault(name, {}).update(
                    {f"{key}_mean": s.mean, f"{key}_median": s.median, f"{key}_count": s.count}
                )
    if files["offsets"].exists():
        for name, o in asp.read_mapproj_offset_stats(files["offsets"]).items():
            rows.setdefault(name, {}).update(
                {f"offset_p{p}": getattr(o, f"p{p}") for p in (25, 50, 75, 85, 95)}, offset_count=o.count
            )
    cols = ["image"] + [f"{k}_{s}" for k in ("initial", "final") for s in ("mean", "median", "count")]
    cols += [f"offset_p{p}" for p in (25, 50, 75, 85, 95)] + ["offset_count"]
    df = pd.DataFrame([{"image": k, **v} for k, v in sorted(rows.items())], columns=cols)
    offsets = [c for c in cols if c.startswith("offset_p")]
    df[offsets] = df[offsets].where(df[offsets] >= 0)
    return df


def ba_summary(table: pd.DataFrame, resolution: float, max_offset_p85: float) -> dict[str, Any]:
    """Fig. 5 numbers: per-image median residual mean +- std before / after, offset percentiles."""
    out: dict[str, Any] = {"images": int(len(table))}
    for key in ("initial", "final"):
        med = table[f"{key}_median"].dropna()
        if len(med):
            out[f"{key}_median_px"] = {
                "mean": float(med.mean()),
                "std": float(med.std(ddof=0)),
                "median": float(med.median()),
            }
    p85, p95 = table["offset_p85"].dropna(), table["offset_p95"].dropna()
    if len(p85):
        out["offset_p85_m"] = {"median": float(p85.median()), "max": float(p85.max())}
        out["offset_p85_over_threshold"] = int((p85 > max_offset_p85).sum())
        out["offset_threshold_m"] = max_offset_p85
    if len(p95):
        out["offset_p95_m"] = {"median": float(p95.median()), "max": float(p95.max())}
        out["frac_p95_below_resolution"] = float((p95 < resolution).mean())
    return out


def read_convergence(path: Path) -> pd.DataFrame:
    """``<prefix>-convergence_angles.txt``: ``left``, ``right``, ``p25``, ``p50``, ``p75`` (deg), ``matches``."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6 and not parts[0].startswith("#"):
            rows.append(
                [Path(parts[0]).name, Path(parts[1]).name, *map(float, parts[2:5]), int(float(parts[5]))]
            )
    return pd.DataFrame(rows, columns=["left", "right", "p25", "p50", "p75", "matches"])


def read_pointmap_residuals(path: Path) -> np.ndarray:
    """``mean_residual`` column (pixels) of a ``*_residuals_pointmap.csv`` (GCP rows included)."""
    df = pd.read_csv(path, comment="#", header=None, usecols=[3], skipinitialspace=True)
    return pd.to_numeric(df[3], errors="coerce").to_numpy(float)


def removed_images(work_dir: Path) -> pd.DataFrame:
    """``selection/removed.csv`` with a ``cause`` column (matches, offset, residuals, shadow, other)."""
    path = Path(work_dir) / "selection" / "removed.csv"
    cols = ["product_id", "stage", "reason", "cause"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str).fillna("")
    df["cause"] = df["reason"].map(_cause)
    return df[cols]


def _cause(reason: str) -> str:
    for key, cause in (
        ("matches", "matches"),
        ("offset", "offset"),
        ("residuals", "residuals"),
        ("shadow", "shadow"),
    ):
        if key in reason:
            return cause
    return "other"


def selection_table(work_dir: Path) -> pd.DataFrame | None:
    """``selection/selection.csv`` (every kept image; ``selected`` as bool), or None."""
    path = Path(work_dir) / "selection" / "selection.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["selected"] = df["selected"].astype(str).str.lower().isin(("true", "1"))
    return df


def preview_grid(grid: Grid, max_size: int) -> Grid:
    """*grid* coarsened by an integer factor so its longer side is at most *max_size*."""
    f = max(1, math.ceil(max(grid.width, grid.height) / max_size))
    if f == 1:
        return grid
    from rasterio.transform import Affine

    t = grid.transform
    coarse = Affine(t.a * f, t.b * f, t.c, t.d * f, t.e * f, t.f)
    return Grid(grid.crs, coarse, math.ceil(grid.width / f), math.ceil(grid.height / f))


def solar_bins_from_images(
    paths: Sequence[Path],
    azimuths: Sequence[float],
    grid: Grid,
    shadow_threshold: float,
    bin_width: float = 30.0,
) -> np.ndarray:
    """Non-redundant Sun directions per pixel of *grid* from mapprojected images (``illumination_maps``)."""
    from .sfs_products import ProductImage

    ims = [ProductImage(Path(p), float(a)) for p, a in zip(paths, azimuths)]
    return illumination_maps(ims, grid, shadow_threshold=shadow_threshold, bin_width=bin_width).solar_bins


def sample(array: np.ndarray, grid: Grid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nearest-pixel values of *array* on *grid* at map coordinates; NaN outside."""
    inv = ~grid.transform
    x, y = np.asarray(x, float), np.asarray(y, float)
    col, row = inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f
    c, r = np.floor(col).astype(int), np.floor(row).astype(int)
    ok = (c >= 0) & (c < grid.width) & (r >= 0) & (r < grid.height)
    out = np.full(c.shape, np.nan)
    out[ok] = array[r[ok], c[ok]]
    return out


def lola_differences(
    dem: Path, points_csv: Path, out_prefix: Path, z: np.ndarray | None = None, grid: Grid | None = None
) -> pd.DataFrame:
    """DEM minus LOLA points (``x``, ``y`` polar stereographic m; ``diff`` m).

    Uses ``geodiff`` (:func:`~lunar_toolkit.sfs_products.compare_to_points`);
    without ASP, the nearest DEM pixel of each point (*z* on *grid*, read when not given).
    """
    from . import reference as ref
    from .geometry import ps_xy
    from .sfs_products import compare_to_points

    try:
        cmp = compare_to_points(
            dem, points_csv, out_prefix, csv_format=ref.ASP_CSV_FORMAT, csv_srs=ref.PS_SRS
        )
    except ToolError as exc:
        log.info("geodiff unavailable (%s); sampling the DEM at the points", exc)
    else:
        xy = [ps_xy(la, lo, "S") for lo, la in zip(cmp.lon, cmp.lat)]
        x = np.array([p[0] for p in xy], float)
        y = np.array([p[1] for p in xy], float)
        return pd.DataFrame({"x": x, "y": y, "diff": cmp.diff})
    pts = np.loadtxt(points_csv, delimiter=",", comments="#", ndmin=2)
    if z is None or grid is None:
        z, grid = read_raster(dem)
    return pd.DataFrame(
        {"x": pts[:, 0], "y": pts[:, 1], "diff": sample(z, grid, pts[:, 0], pts[:, 1]) - pts[:, 2]}
    )


class _Terrain:
    """The SfS DEM, prior and derived arrays on the SfS grid, read once (float32)."""

    def __init__(self, inputs: ReportInputs):
        self.inputs = inputs
        self._cache: dict[str, np.ndarray | None] = {}
        self.grid = Grid.from_raster(inputs.sfs_dem) if inputs.sfs_dem else None

    def get(self, name: str, make: Callable[[], np.ndarray | None]) -> np.ndarray | None:
        if name not in self._cache:
            self._cache[name] = make()
        return self._cache[name]

    def raster(self, path: Path | None) -> np.ndarray | None:
        if path is None or self.grid is None or not Path(path).exists():
            return None
        return read_raster(path, self.grid)[0].astype(np.float32)

    @property
    def z(self) -> np.ndarray | None:
        return self.get("z", lambda: self.raster(self.inputs.sfs_dem))

    @property
    def prior(self) -> np.ndarray | None:
        return self.get("prior", lambda: self.raster(self.inputs.prior_dem))

    @property
    def diff(self) -> np.ndarray | None:
        return self.get("diff", lambda: None if self.z is None or self.prior is None else self.z - self.prior)

    @property
    def slope(self) -> np.ndarray | None:
        def make() -> np.ndarray | None:
            res = self.inputs.resolution or 5
            product = self.raster(_first_existing(self.inputs.product("TSLP", res)))
            if product is not None:
                return product
            return None if self.z is None or self.grid is None else _slope(self.z, self.grid.res)

        return self.get("slope", make)


def _slope(z: np.ndarray, res: float) -> np.ndarray:
    return slope_deg(z.astype(np.float64), res).astype(np.float32)


def _quiet(func: Callable[..., Any], *args: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(*args)


def _ba_section(m: Metrics, t: _Terrain) -> None:
    inp = m.inputs
    ws = open_workspace(inp.work_dir)
    try:
        for stage, rel in BA_STAGES:
            prefix = inp.work_dir / rel
            run = Path(rel).parent.name
            from_ws = ws is not None and run in ws.runs()
            table = ws.ba_report(run) if ws is not None and from_ws else read_ba_reports(prefix)
            if table is None:
                continue
            m.tables[f"ba_{stage}"] = table
            summary = ba_summary(table, inp.resolution or 5, inp.max_offset_p85 or 1.8)
            summary["source"] = "workspace" if from_ws else "files"
            conv = Path(f"{prefix}-convergence_angles.txt")
            angles = None
            if ws is not None and from_ws:
                angles = ws.convergence(run)
            elif conv.exists():
                angles = read_convergence(conv)
            if angles is not None and len(angles):
                summary["convergence_deg"] = {"pairs": int(len(angles)), **_med(angles["p50"])}
            m.summary.setdefault("ba", {})[stage] = summary
            for key in ("initial", "final"):
                pm = Path(f"{prefix}-{key}_residuals_pointmap.csv")
                values = None
                if ws is not None and from_ws:
                    values = ws.point_residuals(run, key)
                    values = values if len(values) else None
                elif pm.exists():
                    values = read_pointmap_residuals(pm)
                if values is not None:
                    m.histograms.setdefault(f"pointmap_{stage}", []).append(
                        Histogram.of(key, values, np.logspace(-2, 2, 81))
                    )
    finally:
        if ws is not None:
            ws.close()
    if "ba" not in m.summary:
        m.skipped["ba"] = "no bundle-adjustment reports under ba/ or ba_refine/"


def open_workspace(work_dir: Path) -> Any:
    """The work dir's ``lunar.duckdb`` :class:`~lunar_toolkit.workspace.Workspace` (read-only), or None.

    None when there is none or it cannot be opened (e.g. locked by a running pipeline).
    """
    import duckdb

    from .workspace import Workspace, workspace_path

    path = workspace_path(work_dir)
    if not path.exists():
        return None
    try:
        return Workspace(path, read_only=True, spatial=False)
    except duckdb.Error as exc:
        log.warning("Not reading %s: %s", path, exc)
        return None


def _med(s: pd.Series) -> dict[str, float]:
    s = s.dropna()
    return {"median": float(s.median()), "max": float(s.max())} if len(s) else {}


def _removed_section(m: Metrics, t: _Terrain) -> None:
    wd = m.inputs.work_dir
    removed = removed_images(wd)
    sel = selection_table(wd)
    ba = m.tables.get("ba_ba")
    counts: dict[str, Any] = {}
    if ba is not None:
        counts["bundle_adjusted"] = int(len(ba))
    if sel is not None:
        counts["kept"] = int(len(sel))
        counts["selected"] = int(sel["selected"].sum())
    if removed.empty and not counts:
        m.skipped["removed"] = "no selection/removed.csv or selection.csv"
        return
    m.tables["removed"] = removed
    counts["removed"] = int(len(removed))
    counts["by_stage"] = {str(k): int(v) for k, v in removed["stage"].value_counts().items()}
    counts["by_cause"] = {str(k): int(v) for k, v in removed["cause"].value_counts().items()}
    m.summary["removed"] = counts


def _coverage_section(m: Metrics, t: _Terrain) -> None:
    inp = m.inputs
    out: dict[str, Any] = {}
    product = inp.product("SBCT")
    if product.exists():
        bins = read_raster(product)[0]
        out["selected_full_res"] = coverage_fractions(np.nan_to_num(bins, nan=0.0))
    sel = selection_table(inp.work_dir)
    if sel is not None and t.grid is not None:
        sel = sel[sel["mapprojected"].map(lambda p: isinstance(p, str) and Path(p).exists())]
        sel = sel[np.isfinite(pd.to_numeric(sel["sun_azimuth"], errors="coerce"))]
        coarse = preview_grid(t.grid, inp.preview_size)
        groups = {"kept": sel, "selected": sel[sel["selected"]]}
        for name, df in groups.items():
            if len(df):
                bins = solar_bins_from_images(
                    df["mapprojected"].tolist(),
                    df["sun_azimuth"].astype(float).tolist(),
                    coarse,
                    inp.shadow_threshold or 0.0,
                )
                out[name] = {"images": int(len(df)), **coverage_fractions(bins)}
    if not out:
        m.skipped["coverage"] = "no solar-bin product (GLDSBCT) or mapprojected SfS images"
        return
    m.summary["coverage"] = out


def _prior_section(m: Metrics, t: _Terrain) -> None:
    inp = m.inputs
    if t.diff is None:
        m.skipped["prior"] = "SfS DEM or prior DEM missing"
        return
    ldec = t.get("ldec", lambda: t.raster(inp.ldec))
    m.summary["prior"] = split_stats(t.diff, ldec)
    hists = [Histogram.of("SfS - prior (all)", t.diff, DIFF_EDGES)]
    if ldec is not None:
        hists += [
            Histogram.of("LDEC > 0", t.diff[np.isfinite(ldec) & (ldec > 0)], DIFF_EDGES),
            Histogram.of("LDEC = 0", t.diff[np.isfinite(ldec) & (ldec <= 0)], DIFF_EDGES),
        ]
    m.histograms["prior_diff"] = hists
    if t.slope is not None:
        m.tables["prior_vs_slope"] = binned(t.diff, t.slope, inp.slope_edges)


def _lola_section(m: Metrics, t: _Terrain, work: Path) -> None:
    inp = m.inputs
    points = inp.lola_points
    if points is None and inp.lola_xyzi is not None and t.grid is not None and Path(inp.lola_xyzi).exists():
        from . import reference as ref

        x0, y0, x1, y1 = t.grid.bounds
        points = ref.write_asp_csv(Path(inp.lola_xyzi), work / "lola_points.csv", ref.PsBox(x0, y0, x1, y1))
    if points is None or not Path(points).exists() or inp.sfs_dem is None or t.z is None or t.grid is None:
        m.skipped["lola"] = "no LOLA points (--lola-points / --lola-xyzi) or SfS DEM"
        return
    pts = lola_differences(inp.sfs_dem, Path(points), work / "lola", t.z, t.grid)
    m.summary["lola"] = stats(pts["diff"])
    m.histograms["lola_diff"] = [Histogram.of("SfS - LOLA", pts["diff"].to_numpy(), DIFF_EDGES)]
    if t.slope is not None:
        slope = sample(t.slope, t.grid, pts["x"].to_numpy(), pts["y"].to_numpy())
        m.tables["lola_vs_slope"] = binned(pts["diff"].to_numpy(), slope, inp.slope_edges)
        ok = np.isfinite(slope) & np.isfinite(pts["diff"].to_numpy())
        counts, xe, ye = np.histogram2d(
            slope[ok],
            np.clip(pts["diff"].to_numpy()[ok], DIFF_EDGES[0], DIFF_EDGES[-1]),
            [np.linspace(0, 45, 46), DIFF_EDGES],
        )
        m.hist2d["lola_vs_slope"] = (xe, ye, counts)


def _ensemble_section(m: Metrics, t: _Terrain) -> None:
    inp = m.inputs
    paths = {k: inp.product(c, "ENS") for k, c in (("elevation_std", "ISGM"), ("slope_std", "SSGM"))}
    if not paths["elevation_std"].exists():
        m.skipped["ensemble"] = "no ensemble products (*_GLDISGM_*_ENS.tif)"
        return
    esd, grid = read_raster(paths["elevation_std"])
    ssd = read_raster(paths["slope_std"], grid)[0] if paths["slope_std"].exists() else None
    out: dict[str, Any] = {"elevation_std_m": stats(esd)}
    if ssd is not None:
        out["slope_std_deg"] = stats(ssd)
    m.histograms["ensemble"] = [Histogram.of("elevation std (m)", esd, np.linspace(0, 5, 101))]
    slope = None
    if inp.sfs_dem is not None:
        z = read_raster(inp.sfs_dem, grid)[0]
        slope = slope_deg(z, grid.res)
    if slope is not None:
        lo, hi = slope < inp.slope_split, slope >= inp.slope_split
        out["by_slope"] = {
            f"slope<{inp.slope_split:g}": stats(esd[lo]),
            f"slope>={inp.slope_split:g}": stats(esd[hi]),
        }
        m.tables["ensemble_vs_slope"] = binned(esd, slope, inp.slope_edges)
    height_error = inp.product("ISGM")
    if height_error.exists():
        he = read_raster(height_error, grid)[0]
        ratio = _quiet(np.divide, esd, np.where(he > 0, he, np.nan))
        out["ensemble_over_height_error"] = stats(ratio)
        m.histograms["ensemble"].append(Histogram.of("ASP height error (m)", he, np.linspace(0, 5, 101)))
    m.summary["ensemble"] = out


def _uncertainty_section(m: Metrics, t: _Terrain) -> None:
    inp = m.inputs
    out: dict[str, Any] = {}
    hists = []
    toterr = t.raster(inp.toterr)
    if toterr is not None and t.diff is not None:
        z = _quiet(np.divide, t.diff, np.where(toterr > 0, toterr, np.nan))
        out["elevation"] = normal_summary(z)
        hists.append(Histogram.of("elevation", z, NORM_EDGES))
    slperr = t.raster(inp.slperr)
    if slperr is not None and t.slope is not None:
        ldem_slope = t.raster(inp.ldem_slope)
        if ldem_slope is None and t.prior is not None and t.grid is not None:
            ldem_slope = _slope(t.prior, t.grid.res)
        if ldem_slope is not None:
            z = _quiet(np.divide, t.slope - ldem_slope, np.where(slperr > 0, slperr, np.nan))
            out["slope"] = normal_summary(z)
            hists.append(Histogram.of("slope", z, NORM_EDGES))
    if not out:
        m.skipped["uncertainty"] = "no PGDA toterr / slperr maps (--toterr / --slperr / --region)"
        return
    m.summary["uncertainty"] = out
    m.histograms["normalised"] = hists


def _published_section(m: Metrics, t: _Terrain) -> None:
    inp = m.inputs
    pub = t.raster(inp.published_dem)
    if pub is None or t.z is None or t.grid is None:
        m.skipped["published"] = "no published SDEM (--published-dem)"
        return
    diff = t.z - pub
    t._cache["published_diff"] = diff
    out = {"elevation": stats(diff)}
    if t.slope is not None:
        out["slope"] = stats(t.slope - _slope(pub, t.grid.res))
    m.summary["published"] = out
    m.histograms["published_diff"] = [Histogram.of("SfS - published", diff, DIFF_EDGES)]


def metrics(
    inputs: ReportInputs | Path, work: Path | None = None, sections: Iterable[str] | None = None
) -> Metrics:
    """All report metrics; sections whose inputs are missing are listed in ``skipped``.

    Parameters
    ----------
    inputs : ReportInputs or Path
        Inputs, or a work dir (:meth:`ReportInputs.from_work_dir`).
    work : Path, optional
        Scratch directory for converted LOLA points and ``geodiff`` output
        (default: ``<work_dir>/report/work``).
    sections : iterable of str, optional
        Subset of :data:`SECTIONS` (default: all).
    """
    inp = inputs if isinstance(inputs, ReportInputs) else ReportInputs.from_work_dir(inputs)
    work = Path(work) if work else inp.work_dir / "report" / "work"
    m = Metrics(inp)
    t = _Terrain(inp)
    runners: dict[str, Callable[[], None]] = {
        "ba": lambda: _ba_section(m, t),
        "removed": lambda: _removed_section(m, t),
        "coverage": lambda: _coverage_section(m, t),
        "prior": lambda: _prior_section(m, t),
        "lola": lambda: _lola_section(m, t, work),
        "ensemble": lambda: _ensemble_section(m, t),
        "uncertainty": lambda: _uncertainty_section(m, t),
        "published": lambda: _published_section(m, t),
    }
    wanted = list(sections) if sections is not None else list(SECTIONS)
    unknown = set(wanted) - set(runners)
    if unknown:
        raise ValueError(f"Unknown report sections {sorted(unknown)}; choose from {', '.join(SECTIONS)}")
    for name in wanted:
        log.info("Report: %s", SECTIONS[name])
        runners[name]()
    m.summary["inputs"] = {f.name: _jsonable(getattr(inp, f.name)) for f in fields(inp)}
    m.summary["skipped"] = dict(m.skipped)
    return m


def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_tables(m: Metrics, out_dir: Path) -> dict[str, Path]:
    """``tables/<name>.csv`` for every table, plus ``metrics.json`` (summary and histograms)."""
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in m.tables.items():
        paths[name] = out_dir / "tables" / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    for name, df in summary_tables(m).items():
        paths[name] = out_dir / "tables" / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    doc = {
        **_jsonable(m.summary),
        "histograms": {
            k: [{"label": h.label, "edges": h.edges.tolist(), "counts": h.counts.tolist()} for h in v]
            for k, v in m.histograms.items()
        },
    }
    paths["metrics"] = out_dir / "metrics.json"
    paths["metrics"].write_text(json.dumps(doc, indent=1))
    return paths


def summary_tables(m: Metrics) -> dict[str, pd.DataFrame]:
    """Table 2-style rows (one per comparison) and the coverage fractions as tables."""
    out = {}
    rows = []
    for label, s in (
        *((f"SfS - prior ({k})", v) for k, v in m.summary.get("prior", {}).items()),
        *((("SfS - LOLA points", m.summary["lola"]),) if "lola" in m.summary else ()),
        *((("SfS - published", m.summary["published"]["elevation"]),) if "published" in m.summary else ()),
    ):
        rows.append({"comparison": label, **s})
    if rows:
        out["elevation_differences"] = pd.DataFrame(rows)
    cov = m.summary.get("coverage", {})
    if cov:
        out["coverage"] = pd.DataFrame([{"images_set": k, **v} for k, v in cov.items()])
    return out


def _figure(nrows: int = 1, ncols: int = 1, size: tuple[float, float] = (6.4, 4.2)) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True)
    return fig, ax


def _save(fig: Any, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _style(ax: Any) -> None:
    ax.grid(True, color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _hist_figure(
    hists: Sequence[Histogram],
    xlabel: str,
    path: Path,
    *,
    density: bool = True,
    logx: bool = False,
    gaussians: Sequence[tuple[float, str]] = (),
) -> Path:
    fig, ax = _figure()
    for h, color in zip(hists, SERIES):
        y = h.counts / max(h.total, 1) / np.diff(h.edges) if density else h.counts
        ax.stairs(y, h.edges, color=color, linewidth=1.6, label=f"{h.label} (n={h.total:,})")
    for sigma, color in gaussians:
        x = np.linspace(hists[0].edges[0], hists[0].edges[-1], 400)
        ax.plot(
            x,
            np.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)),
            "--",
            color=color,
            linewidth=1.2,
            label=f"Gaussian sigma={sigma:.2f}",
        )
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "count")
    _style(ax)
    if len(hists) + len(gaussians) > 1:
        ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def _map_figure(
    array: np.ndarray,
    grid: Grid,
    path: Path,
    label: str,
    *,
    cmap: str = "cividis",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    fig, ax = _figure(size=(6.0, 5.0))
    x0, y0, x1, y1 = grid.bounds
    finite = array[np.isfinite(array)]
    if symmetric and finite.size:
        lim = float(np.percentile(np.abs(finite), 98)) or 1.0
        vmin, vmax = -lim, lim
    elif finite.size and vmin is None:
        vmin, vmax = (float(v) for v in np.percentile(finite, [2, 98]))
    im = ax.imshow(
        array,
        extent=(x0 / 1e3, x1 / 1e3, y0 / 1e3, y1 / 1e3),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    _map_axes(ax)
    return _save(fig, path)


def _map_axes(ax: Any) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("x (km, polar stereographic)")
    ax.set_ylabel("y (km)")


def hillshade_array(z: np.ndarray, res: float, azimuth: float = 315.0, elevation: float = 45.0) -> np.ndarray:
    """Lambertian hillshade (0-1) of *z* with the Sun at *azimuth* / *elevation* (deg)."""
    from .sfs_products import gradient

    p, q = gradient(z, res)
    az, el = math.radians(azimuth), math.radians(elevation)
    sx, sy, sz = math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)
    return np.clip((-p * sx - q * sy + sz) / np.sqrt(1 + p * p + q * q), 0, 1)


def figures(m: Metrics, out_dir: Path) -> dict[str, Path]:
    """PNG figures of every section with data, in ``<out_dir>/figures``; returns name -> path."""
    import matplotlib.colors as mcolors

    inp = m.inputs
    fig_dir = Path(out_dir) / "figures"
    out: dict[str, Path] = {}
    if inp.sfs_dem is not None and Path(inp.sfs_dem).exists():
        grid = preview_grid(Grid.from_raster(inp.sfs_dem), inp.preview_size)
        z = read_raster(inp.sfs_dem, grid)[0]
        out["hillshade"] = _map_figure(
            hillshade_array(z, grid.res),
            grid,
            fig_dir / "hillshade.png",
            "hillshade (Sun az 315, el 45 deg)",
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        out["slope_map"] = _map_figure(
            slope_deg(z, grid.res),
            grid,
            fig_dir / "slope_map.png",
            "slope (deg)",
            cmap="cividis",
            vmin=0,
            vmax=40,
        )
        if inp.prior_dem is not None and Path(inp.prior_dem).exists():
            d = z - read_raster(inp.prior_dem, grid)[0]
            out["prior_diff_map"] = _map_figure(
                d, grid, fig_dir / "prior_diff_map.png", "SfS - prior (m)", cmap="RdBu_r", symmetric=True
            )
        if inp.published_dem is not None and Path(inp.published_dem).exists():
            d = z - read_raster(inp.published_dem, grid)[0]
            out["published_diff_map"] = _map_figure(
                d,
                grid,
                fig_dir / "published_diff_map.png",
                "SfS - published SDEM (m)",
                cmap="RdBu_r",
                symmetric=True,
            )
    sbct = inp.product("SBCT")
    if sbct.exists():
        g = preview_grid(Grid.from_raster(sbct), inp.preview_size)
        bins = np.nan_to_num(_nearest(sbct, g), nan=0.0)
        cmap = mcolors.ListedColormap(COVERAGE_COLORS)
        fig, ax = _figure(size=(6.0, 5.0))
        x0, y0, x1, y1 = g.bounds
        im = ax.imshow(
            np.minimum(bins, 3),
            extent=(x0 / 1e3, x1 / 1e3, y0 / 1e3, y1 / 1e3),
            cmap=cmap,
            norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4),
            interpolation="nearest",
        )
        cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.85, label="non-redundant Sun directions")
        cb.ax.set_yticklabels(COVERAGE_LABELS)
        _map_axes(ax)
        out["coverage_map"] = _save(fig, fig_dir / "coverage_map.png")
    cov = {k: v for k, v in m.summary.get("coverage", {}).items() if v.get("pixels")}
    if cov:
        out["coverage_bars"] = _coverage_bars(cov, fig_dir / "coverage_bars.png")
    for stage in ("ba", "refine"):
        table = m.tables.get(f"ba_{stage}")
        if table is not None:
            out.update(_ba_figures(table, stage, inp, fig_dir))
        if f"pointmap_{stage}" in m.histograms:
            out[f"pointmap_{stage}"] = _hist_figure(
                m.histograms[f"pointmap_{stage}"],
                "triangulated point mean residual (px)",
                fig_dir / f"pointmap_residuals_{stage}.png",
                logx=True,
            )
    diffs = m.histograms.get("prior_diff", []) + m.histograms.get("lola_diff", [])
    if diffs:
        out["diff_histograms"] = _hist_figure(
            diffs[:3], "elevation difference (m)", fig_dir / "diff_histograms.png"
        )
    if "lola_diff" in m.histograms and len(diffs) > 3:
        out["lola_histogram"] = _hist_figure(
            m.histograms["lola_diff"], "SfS - LOLA (m)", fig_dir / "lola_histogram.png"
        )
    for name in ("lola_vs_slope", "prior_vs_slope", "ensemble_vs_slope"):
        if name in m.tables:
            out[name] = _slope_figure(m, name, fig_dir / f"{name}.png")
    if "ensemble" in m.histograms:
        out["ensemble_histogram"] = _hist_figure(
            m.histograms["ensemble"], "elevation uncertainty (m)", fig_dir / "ensemble_histogram.png"
        )
        esd = inp.product("ISGM", "ENS")
        g = preview_grid(Grid.from_raster(esd), inp.preview_size)
        out["ensemble_map"] = _map_figure(
            read_raster(esd, g)[0],
            g,
            fig_dir / "ensemble_map.png",
            "ensemble elevation std (m)",
            cmap="cividis",
        )
    for h in m.histograms.get("normalised", []):
        sigma = m.summary["uncertainty"][h.label]["std"]
        out[f"normalised_{h.label}"] = _hist_figure(
            [h],
            f"(SfS - LDEM) / LDEM uncertainty, {h.label}",
            fig_dir / f"normalised_{h.label}.png",
            gaussians=[(1.0, "#e34948"), (sigma, "#008300")],
        )
    return out


def _nearest(path: Path, grid: Grid) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        data = src.read(1, out_shape=(grid.height, grid.width), resampling=Resampling.nearest, masked=True)
    return data.astype(np.float64).filled(np.nan)


def _coverage_bars(cov: dict[str, dict[str, float]], path: Path) -> Path:
    fig, ax = _figure(size=(6.4, 2.2 + 0.5 * len(cov)))
    names = list(cov)
    left = np.zeros(len(names))
    keys = ("frac_3plus", "frac_2", "frac_1", "frac_0")
    for key, color, label in zip(keys, COVERAGE_COLORS[::-1], ("3+", "2", "1", "0")):
        vals = np.array([100 * cov[n].get(key, 0.0) for n in names])
        ax.barh(
            names, vals, left=left, color=color, edgecolor="white", linewidth=1, label=f"{label} directions"
        )
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("pixels (%)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    _style(ax)
    return _save(fig, path)


def _ba_figures(table: pd.DataFrame, stage: str, inp: ReportInputs, fig_dir: Path) -> dict[str, Path]:
    out = {}
    med = [table[c].dropna().to_numpy() for c in ("initial_median", "final_median")]
    if all(len(x) for x in med):
        edges = np.logspace(-2, 2, 61)
        hists = [Histogram.of(k, x, edges) for k, x in zip(("before BA", "after BA"), med)]
        out[f"ba_residuals_{stage}"] = _hist_figure(
            hists,
            "per-image median reprojection error (px)",
            fig_dir / f"ba_residuals_{stage}.png",
            density=False,
            logx=True,
        )
        both = table.dropna(subset=["initial_median", "final_median"])
        fig, ax = _figure(size=(5.0, 4.6))
        ax.scatter(both["initial_median"], both["final_median"], s=10, color=SERIES[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("median residual before BA (px)")
        ax.set_ylabel("median residual after BA (px)")
        _style(ax)
        out[f"ba_before_after_{stage}"] = _save(fig, fig_dir / f"ba_before_after_{stage}.png")
    p85, p95 = table["offset_p85"].dropna(), table["offset_p95"].dropna()
    if len(p85):
        hi = max(float(np.nanmax(p95)) if len(p95) else 0.0, (inp.resolution or 5) * 1.2, 2.0)
        edges = np.linspace(0, hi, 61)
        fig, ax = _figure()
        for s, color, label in ((p85, SERIES[0], "p85"), (p95, SERIES[1], "p95")):
            ax.stairs(
                np.histogram(np.clip(s, 0, hi), edges)[0], edges, color=color, linewidth=1.6, label=label
            )
        ax.axvline(
            inp.max_offset_p85 or 1.8,
            color="#52514e",
            linestyle="--",
            linewidth=1,
            label=f"p85 threshold {inp.max_offset_p85:g} m",
        )
        ax.axvline(
            inp.resolution or 5,
            color="#52514e",
            linestyle=":",
            linewidth=1,
            label=f"SfS resolution {inp.resolution:g} m",
        )
        ax.set_xlabel("per-image mapprojected match offset (m)")
        ax.set_ylabel("images")
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
        out[f"match_offsets_{stage}"] = _save(fig, fig_dir / f"match_offsets_{stage}.png")
    return out


def _slope_figure(m: Metrics, name: str, path: Path) -> Path:
    t = m.tables[name].dropna(subset=["median"])
    fig, ax = _figure()
    if name in m.hist2d:
        xe, ye, counts = m.hist2d[name]
        ax.pcolormesh(
            xe, ye, np.where(counts.T > 0, counts.T, np.nan), cmap="Greys", norm="log", shading="flat"
        )
    mid = (t["bin_min"] + np.minimum(t["bin_max"], 45)) / 2
    ax.fill_between(mid, t["p25"], t["p75"], color=SERIES[0], alpha=0.25, linewidth=0, label="interquartile")
    ax.plot(mid, t["median"], "-o", color=SERIES[0], markersize=4, linewidth=1.6, label="median")
    ylabel = {
        "lola_vs_slope": "SfS - LOLA (m)",
        "prior_vs_slope": "SfS - prior (m)",
        "ensemble_vs_slope": "ensemble elevation std (m)",
    }[name]
    ax.set_xlabel("slope (deg)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _save(fig, path)


_CSS = """
body { background: #ffffff; color: #111111; font-family: Georgia, 'Times New Roman', serif;
       max-width: 60em; margin: 2em auto; padding: 0 16px; line-height: 1.45; }
h1, h2, h3 { font-weight: normal; }
h2 { border-bottom: 1px solid #999999; padding-bottom: 0.2em; margin-top: 2em; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 0.85em; display: block; overflow-x: auto; }
th, td { border-top: 1px solid #cccccc; border-bottom: 1px solid #cccccc; padding: 0.25em 0.6em; text-align: right; }
th { background: #f2f2f2; }
td:first-child, th:first-child { text-align: left; }
figure { margin: 1em 0; }
figure img { max-width: 100%; height: auto; border: 1px solid #dddddd; }
figcaption, .note { font-size: 0.85em; color: #444444; }
code { font-size: 0.9em; }
"""


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.4g}" if abs(v) < 1e5 else f"{v:,.0f}"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return f"{v:,}"
    return html.escape(str(v))


def _table_html(df: pd.DataFrame, max_rows: int = 40) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
        for row in df.head(max_rows).itertuples(index=False)
    )
    more = (
        f'<p class="note">First {max_rows} of {len(df)} rows; see the CSV.</p>' if len(df) > max_rows else ""
    )
    return f"<table><tr>{head}</tr>{body}</table>{more}"


def _dict_table(d: dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        if isinstance(v, dict) and all(not isinstance(x, dict) for x in v.values()):
            v = ", ".join(f"{a}={_fmt(b)}" for a, b in v.items())
            rows.append(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>")
        elif not isinstance(v, dict):
            rows.append(f"<tr><td>{html.escape(k)}</td><td>{_fmt(v)}</td></tr>")
    return f"<table>{''.join(rows)}</table>" if rows else ""


def _stats_rows(d: dict[str, dict[str, Any]]) -> str:
    rows = [{"subset": k, **v} for k, v in d.items() if isinstance(v, dict)]
    return _table_html(pd.DataFrame(rows)) if rows else ""


def _img(path: Path, caption: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


_SECTION_FIGURES = {
    "ba": (
        "ba_residuals_ba",
        "ba_before_after_ba",
        "match_offsets_ba",
        "pointmap_ba",
        "ba_residuals_refine",
        "ba_before_after_refine",
        "match_offsets_refine",
        "pointmap_refine",
    ),
    "coverage": ("coverage_map", "coverage_bars"),
    "prior": ("hillshade", "slope_map", "prior_diff_map", "diff_histograms", "prior_vs_slope"),
    "lola": ("lola_histogram", "lola_vs_slope"),
    "ensemble": ("ensemble_map", "ensemble_histogram", "ensemble_vs_slope"),
    "uncertainty": ("normalised_elevation", "normalised_slope"),
    "published": ("published_diff_map",),
}
_SECTION_TABLES = {
    "ba": ("ba_ba", "ba_refine"),
    "removed": ("removed",),
    "prior": ("prior_vs_slope",),
    "lola": ("lola_vs_slope",),
    "ensemble": ("ensemble_vs_slope",),
}
_CAPTIONS = {
    "ba_residuals": "Per-image median reprojection error before and after bundle adjustment (cf. Fig. 5 top right).",
    "ba_before_after": "Median reprojection error per image, before vs after bundle adjustment.",
    "match_offsets": "Per-image 85th / 95th percentile mapprojected match offsets (cf. Fig. 5 bottom right).",
    "pointmap": "Mean reprojection error of the triangulated points, before and after bundle adjustment.",
    "coverage_map": "Non-redundant Sun directions (30 deg bins) per pixel of the SfS inputs (cf. Fig. 6).",
    "coverage_bars": "Pixels lit from 0, 1, 2 and >=3 directions, kept vs selected images (cf. Fig. 3).",
    "hillshade": "Hillshade of the SfS DEM (preview grid).",
    "slope_map": "Slope of the SfS DEM on the preview grid (baseline = 2 preview pixels).",
    "prior_diff_map": "SfS DEM minus prior DEM.",
    "diff_histograms": "Elevation differences (cf. Fig. 7 left); values beyond the axis go to the outer bins.",
    "lola_histogram": "SfS DEM minus LOLA points.",
    "lola_vs_slope": "SfS minus LOLA points vs slope: median and interquartile range (cf. Fig. 7 right).",
    "prior_vs_slope": "SfS minus prior DEM vs slope: median and interquartile range.",
    "ensemble_map": "Ensemble elevation standard deviation (cf. Fig. 8 top left).",
    "ensemble_histogram": "Ensemble elevation std and ASP height error (cf. Fig. 8 bottom right).",
    "ensemble_vs_slope": "Ensemble elevation std vs slope (cf. Fig. 8 bottom left).",
    "normalised": "SfS - LDEM residuals over the LDEM clone uncertainty, with a unit Gaussian (red) and "
    "a Gaussian of the measured sigma (green) (cf. Fig. 9).",
    "published_diff_map": "SfS DEM minus the published Bertone et al. SDEM.",
}


def _caption(name: str) -> str:
    for key in sorted(_CAPTIONS, key=len, reverse=True):
        if name.startswith(key):
            return _CAPTIONS[key]
    return name


def write_html(m: Metrics, figs: dict[str, Path], out: Path, tables: dict[str, Path] | None = None) -> Path:
    """Self-contained ``report.html``: summary tables, embedded figures, skipped sections."""
    inp = m.inputs
    out = Path(out)
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>SfS report {html.escape(inp.region_id or '')}</title><style>{_CSS}</style></head><body>",
        f"<h1>SfS DEM report: {html.escape(inp.region_id or '')}</h1>",
        f"<p class='note'>Work dir <code>{html.escape(str(inp.work_dir))}</code>; SfS DEM "
        f"<code>{html.escape(str(inp.sfs_dem))}</code>; prior <code>{html.escape(str(inp.prior_dem))}</code>; "
        f"{inp.resolution:g} m/px. Metrics follow Bertone et al. (2026, PSJ 7:118); "
        "tables are in <code>tables/</code> and <code>metrics.json</code>.</p>",
    ]
    summary_tabs = summary_tables(m)
    if "elevation_differences" in summary_tabs:
        parts += [
            "<h2>Elevation differences (Table 2)</h2>",
            _table_html(summary_tabs["elevation_differences"]),
        ]
    for key, title in SECTIONS.items():
        parts.append(f"<h2>{html.escape(title)}</h2>")
        if key in m.skipped:
            parts.append(f"<p class='note'>Skipped: {html.escape(m.skipped[key])}.</p>")
            continue
        s = m.summary.get(key, {})
        if key == "coverage" and "coverage" in summary_tabs:
            parts.append(_table_html(summary_tabs["coverage"]))
        elif key in ("prior", "uncertainty", "published") or (key == "ensemble" and "by_slope" in s):
            parts.append(_stats_rows({k: v for k, v in s.items() if k != "by_slope"}))
            if "by_slope" in s:
                parts.append(_stats_rows(s["by_slope"]))
        elif key == "ba":
            for stage, st in s.items():
                parts += [f"<h3>{html.escape(stage)}</h3>", _dict_table(st)]
        else:
            parts.append(_dict_table(s) if key != "lola" else _stats_rows({"SfS - LOLA": s}))
        for name in _SECTION_FIGURES.get(key, ()):
            if name in figs:
                parts.append(_img(figs[name], _caption(name)))
        for name in _SECTION_TABLES.get(key, ()):
            if name in m.tables and len(m.tables[name]):
                parts += [f"<h3>{html.escape(name)}</h3>", _table_html(m.tables[name])]
    if tables:
        links = "".join(
            f"<li><a href='{html.escape(p.relative_to(out.parent).as_posix())}'>{html.escape(p.name)}</a></li>"
            for p in tables.values()
            if p.is_relative_to(out.parent)
        )
        parts += ["<h2>Files</h2>", f"<ul>{links}</ul>"]
    parts.append("</body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="ascii", errors="xmlcharrefreplace")
    return out


def make_report(
    inputs: ReportInputs | Path, out_dir: Path | None = None, sections: Iterable[str] | None = None
) -> Path:
    """Metrics, tables, figures and ``report.html`` in *out_dir* (default ``<work_dir>/report``)."""
    inp = inputs if isinstance(inputs, ReportInputs) else ReportInputs.from_work_dir(inputs)
    out_dir = Path(out_dir) if out_dir else inp.work_dir / "report"
    m = metrics(inp, out_dir / "work", sections)
    tables = write_tables(m, out_dir)
    figs = figures(m, out_dir)
    return write_html(m, figs, out_dir / "report.html", tables)
