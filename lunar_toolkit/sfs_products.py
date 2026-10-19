"""Standard SfS DEM products and uncertainty estimates (Bertone et al. 2026, PSJ 7:118)."""

from __future__ import annotations

import logging
import math
import os
import random
import warnings
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import rasterio

from . import asp
from .tools import is_dry_run, reuse_existing, run

log = logging.getLogger(__name__)

T = TypeVar("T", bound=Hashable)

# Table 1 of Bertone et al. (2026): product key -> GLD code.
PRODUCT_CODES: dict[str, str] = {
    "elevation": "ELEV",
    "mask": "MASK",
    "uncertainty": "ISGM",
    "diff": "DIFF",
    "slope": "TSLP",
    "roughness": "TRGH",
    "hillshade": "HILL",
    "orthomosaic": "OMOS",
    "best_resolution": "BRES",
    "solar_bins": "SBCT",
}


def product_name(region_id: str, code: str, resolution: float, *suffix: Any) -> str:
    """Table 1 file name, e.g. ``product_name("SR01", "TSLP", 5, 10)`` → ``SR01_GLDTSLP_5_10.tif``."""
    parts = [region_id, f"GLD{code}", _label(resolution), *(_label(s) for s in suffix)]
    return "_".join(parts) + ".tif"


def _label(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


@dataclass(frozen=True)
class Grid:
    """A georeferenced raster grid (CRS, affine transform, size)."""

    crs: Any
    transform: Any
    width: int
    height: int

    @classmethod
    def from_raster(cls, path: Path) -> Grid:
        with rasterio.open(path) as src:
            return cls(src.crs, src.transform, src.width, src.height)

    @property
    def res(self) -> float:
        """Pixel size (map units); pixels are assumed square."""
        return abs(float(self.transform.a))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)``."""
        t = self.transform
        x0, y0 = t.c, t.f
        x1, y1 = x0 + t.a * self.width, y0 + t.e * self.height
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def read_raster(path: Path, grid: Grid | None = None) -> tuple[np.ndarray, Grid]:
    """Band 1 of *path* as float64 with NaN for nodata, optionally warped onto *grid*.

    Warping averages when the source is finer than *grid* and is bilinear otherwise.
    """
    with rasterio.open(path) as src:
        if grid is None:
            data = src.read(1, masked=True)
            return data.astype(np.float64).filled(np.nan), Grid(src.crs, src.transform, src.width, src.height)
        with _warped(src, grid) as vrt:
            data = vrt.read(1, masked=True)
    return data.astype(np.float64).filled(np.nan), grid


def write_raster(path: Path, array: np.ndarray, grid: Grid, nodata: float | None = np.nan) -> Path:
    """Write *array* as a one-band LZW-compressed GeoTIFF on *grid*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": array.dtype.name,
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "lzw",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path


def _warped(src: Any, grid: Grid) -> Any:
    finer = abs(src.transform.a) < grid.res
    return rasterio.vrt.WarpedVRT(
        src,
        crs=grid.crs,
        transform=grid.transform,
        width=grid.width,
        height=grid.height,
        resampling=rasterio.enums.Resampling.average if finer else rasterio.enums.Resampling.bilinear,
    )


def baseline_step(baseline: float | None, res: float) -> int:
    """Neighbour offset *k* (pixels) for a slope / roughness baseline of ``2 k res``; at least 1."""
    if baseline is None:
        return 1
    return max(1, round(baseline / (2 * res)))


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``out[i, j] = a[i + dy, j + dx]``, NaN outside."""
    out = np.full_like(a, np.nan)
    h, w = a.shape
    out[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)] = a[
        max(0, dy) : h - max(0, -dy), max(0, dx) : w - max(0, -dx)
    ]
    return out


def gradient(z: np.ndarray, res: float, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Horn (1981) gradient ``(dz/dx, dz/dy)`` with neighbours *step* pixels away (``gdaldem`` at 1).

    Rows run north to south, so ``dz/dy`` is positive uphill to the north.  Edge pixels are NaN.
    """
    k = step

    def s(dy: int, dx: int) -> np.ndarray:
        return _shift(z, dy, dx)

    dzdx = ((s(-k, k) + 2 * s(0, k) + s(k, k)) - (s(-k, -k) + 2 * s(0, -k) + s(k, -k))) / (8 * k * res)
    dzdy = ((s(-k, -k) + 2 * s(-k, 0) + s(-k, k)) - (s(k, -k) + 2 * s(k, 0) + s(k, k))) / (8 * k * res)
    return dzdx, dzdy


def slope_deg(z: np.ndarray, res: float, step: int = 1) -> np.ndarray:
    """Slope (degrees) over a baseline of ``2 step res``; ``step=1`` matches ``gdaldem slope``."""
    dzdx, dzdy = gradient(z, res, step)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def _box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over the ``(2k+1)²`` window around each pixel; NaN if any value in it is NaN."""
    n = 2 * k + 1

    def window(x: np.ndarray) -> np.ndarray:
        c = np.pad(np.pad(x, k).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]

    valid = np.isfinite(a)
    total = window(np.where(valid, a, 0.0))
    full = window(valid.astype(np.float64)) > n * n - 0.5
    return np.where(full, total, np.nan)


def vrm(z: np.ndarray, res: float, step: int = 1) -> np.ndarray:
    """Vector ruggedness measure (Sappington et al. 2007) over a ``(2 step + 1)²`` window.

    Unit surface normals come from the pixel-scale Horn gradient; VRM is
    ``1 - |Σ n| / N`` over the window: 0 for a plane, up to 1 for rugged terrain.
    """
    p, q = gradient(z, res, 1)
    norm = np.sqrt(1 + p * p + q * q)
    sums = [_box_sum(c, step) for c in (-p / norm, -q / norm, 1 / norm)]
    n = (2 * step + 1) ** 2
    return 1 - np.sqrt(sum(s * s for s in sums)) / n


@dataclass(frozen=True)
class ProductImage:
    """A mapprojected SfS input image, its Sun azimuth (deg from north) and native GSD (m/px)."""

    path: Path
    sun_azimuth: float
    resolution: float | None = None


def as_product_images(images: Iterable[ProductImage | Sequence[Any]]) -> list[ProductImage]:
    """Accept [`ProductImage`][lunar_toolkit.sfs_products.ProductImage] or
    ``(path, sun_azimuth[, resolution])`` tuples."""
    out = []
    for im in images:
        if isinstance(im, ProductImage):
            out.append(im)
        else:
            path, az, *rest = im
            out.append(ProductImage(Path(path), float(az), float(rest[0]) if rest and rest[0] else None))
    return out


def sun_bin(azimuth: float, bin_width: float = 30.0) -> int:
    """Index of the Sun-azimuth bin (``[0, w)``, ``[w, 2w)``, …) holding *azimuth*."""
    return int((azimuth % 360.0) // bin_width)


@dataclass
class IlluminationMaps:
    """Per-pixel illumination statistics of the SfS inputs on the DEM grid.

    Attributes
    ----------
    count : ndarray (uint16)
        Number of lit images (value above the shadow threshold).
    solar_bins : ndarray (uint8)
        Number of Sun-azimuth bins with at least one lit image.
    best_resolution : ndarray (float32) or None
        Finest native GSD among the lit images (NaN where none); None when
        some image has no resolution.
    grid : Grid
    """

    count: np.ndarray
    solar_bins: np.ndarray
    best_resolution: np.ndarray | None
    grid: Grid


def illumination_maps(
    images: Sequence[ProductImage],
    grid: Grid,
    *,
    shadow_threshold: float = 0.005,
    bin_width: float = 30.0,
    block_rows: int = 1024,
) -> IlluminationMaps:
    """Lit-image count, solar-bin count and best input resolution per pixel of *grid*.

    Each image is warped onto *grid* in blocks of *block_rows* rows, so memory
    stays at a few arrays of the grid size.
    """
    nbins = math.ceil(360.0 / bin_width)
    if nbins > 64:
        raise ValueError(f"bin_width {bin_width} gives {nbins} bins; at most 64 are supported")
    shape = (grid.height, grid.width)
    count = np.zeros(shape, np.uint16)
    bits = np.zeros(shape, np.uint64)
    with_res = all(im.resolution for im in images)
    best = np.full(shape, np.inf, np.float32) if with_res else None
    for im in images:
        bit = np.uint64(1) << np.uint64(sun_bin(im.sun_azimuth, bin_width))
        with rasterio.open(im.path) as src, _warped(src, grid) as vrt:
            for row in range(0, grid.height, block_rows):
                rows = slice(row, min(row + block_rows, grid.height))
                win = rasterio.windows.Window(0, row, grid.width, rows.stop - row)
                data = vrt.read(1, window=win, masked=True)
                lit = np.ma.filled(data > shadow_threshold, False)
                count[rows] += lit
                bits[rows] |= np.where(lit, bit, np.uint64(0))
                if best is not None:
                    best[rows] = np.where(lit, np.minimum(best[rows], np.float32(im.resolution)), best[rows])
    bins = np.zeros(shape, np.uint8)
    for b in range(nbins):
        bins += ((bits >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
    if best is not None:
        best[np.isinf(best)] = np.nan
    return IlluminationMaps(count, bins, best, grid)


def copy_elevation(sfs_dem: Path, out: Path, *, force: bool = False) -> Path:
    """GLD01: the SfS DEM, LZW-compressed under its product name."""
    if not reuse_existing(out, force):
        asp.gdal_translate(sfs_dem, out)
    return Path(out)


def copy_height_error(height_error: Path, out: Path, *, force: bool = False) -> Path:
    """GLD03: ASP's ``sfs --estimate-height-errors`` map under its product name."""
    if not reuse_existing(out, force):
        asp.gdal_translate(height_error, out)
    return Path(out)


def difference(sfs_dem: Path, prior_dem: Path, out: Path, *, force: bool = False) -> Path:
    """GLD04: ``sfs_dem - prior_dem`` on the SfS grid (``geodiff``)."""
    out = Path(out)
    if reuse_existing(out, force):
        return out
    prefix = out.with_suffix("")
    diff = asp.geodiff(sfs_dem, prior_dem, prefix)
    if not is_dry_run():
        os.replace(diff, out)
    return out


def hillshade(
    dem: Path, out: Path, *, azimuth: float = 315, elevation: float = 45, force: bool = False
) -> Path:
    """GLD07: ASP ``hillshade`` (Table 1 default: Sun azimuth 315°, elevation 45°)."""
    if not reuse_existing(out, force):
        asp.hillshade(dem, out, azimuth=azimuth, elevation=elevation)
    return Path(out)


def orthomosaic(
    images: Sequence[Path],
    out: Path,
    *,
    grid: Grid | None = None,
    resolution: float | None = None,
    work_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """GLD08: max-lit ``dem_mosaic --max`` of the mapprojected images, warped onto *grid*.

    ``dem_mosaic`` cannot reproduce an arbitrary target grid exactly, so the
    mosaic is made on the inputs' grid and ``gdalwarp -r average`` puts it on
    the DEM grid (or at *resolution* when *grid* is unknown).
    """
    out = Path(out)
    if reuse_existing(out, force):
        return out
    work_dir = Path(work_dir) if work_dir else out.parent / "work"
    max_lit = work_dir / "max_lit.tif"
    if not reuse_existing(max_lit, force):
        asp.dem_mosaic(images, max_lit, max=True)
    if grid is not None:
        target = ["-te", *grid.bounds, "-ts", grid.width, grid.height]
    elif resolution is not None:
        target = ["-tr", resolution, resolution]
    else:
        raise ValueError("orthomosaic needs a grid or a resolution")
    run("gdalwarp", ["-overwrite", "-r", "average", *target, "-co", "COMPRESS=LZW", max_lit, out])
    return out


def terrain_products(
    sfs_dem: Path,
    out_dir: Path,
    *,
    region_id: str,
    resolution: float,
    baselines: Sequence[float | None] = (None,),
    force: bool = False,
) -> dict[str, Path]:
    """GLD05 slope (deg) and GLD06 VRM roughness at each baseline.

    ``None`` is the pixel scale (3×3 kernel), named with *resolution* as the
    baseline.  A baseline *b* (m) uses neighbours ``k = round(b / 2 res)``
    pixels away ([`baseline_step`][lunar_toolkit.sfs_products.baseline_step]), so baselines up to ``2 res``
    equal the pixel scale.  Keys are ``slope`` / ``roughness`` for the pixel scale and
    ``slope_<b>`` / ``roughness_<b>`` otherwise.
    """
    out: dict[str, Path] = {}
    grid = Grid.from_raster(sfs_dem)
    z: np.ndarray | None = None
    for baseline in baselines:
        k = baseline_step(baseline, grid.res)
        key = "" if baseline is None else f"_{baseline:g}"
        label = resolution if baseline is None else baseline
        for name, func in (("slope", slope_deg), ("roughness", vrm)):
            path = Path(out_dir) / product_name(region_id, PRODUCT_CODES[name], resolution, label)
            out[name + key] = path
            if reuse_existing(path, force):
                continue
            if z is None:
                z = read_raster(sfs_dem)[0]
            write_raster(path, func(z, grid.res, k).astype(np.float32), grid)
    return out


def illumination_products(
    images: Sequence[ProductImage],
    grid: Grid,
    out_dir: Path,
    *,
    region_id: str,
    resolution: float,
    shadow_threshold: float = 0.005,
    bin_width: float = 30.0,
    force: bool = False,
) -> dict[str, Path]:
    """GLD02 lit-image count, GLD10 solar-bin count and GLD09 best input resolution (if known)."""
    names = {
        k: Path(out_dir) / product_name(region_id, PRODUCT_CODES[k], resolution)
        for k in ("mask", "solar_bins", "best_resolution")
    }
    with_res = all(im.resolution for im in images)
    if not with_res:
        log.info("Skipping best-resolution product: not every image has a native resolution")
        del names["best_resolution"]
    if all(reuse_existing(p, force) for p in names.values()):
        return names
    maps = illumination_maps(images, grid, shadow_threshold=shadow_threshold, bin_width=bin_width)
    write_raster(names["mask"], maps.count, grid, nodata=None)
    write_raster(names["solar_bins"], maps.solar_bins, grid, nodata=None)
    if maps.best_resolution is not None:
        write_raster(names["best_resolution"], maps.best_resolution, grid)
    return names


def make_products(
    sfs_dem: Path,
    prior_dem: Path,
    out_dir: Path,
    images: Sequence[ProductImage | Sequence[Any]],
    *,
    region_id: str,
    resolution: float,
    height_error: Path | None = None,
    shadow_threshold: float = 0.005,
    bin_width: float = 30.0,
    baselines: Sequence[float | None] = (None,),
    hillshade_sun: tuple[float, float] = (315, 45),
    force: bool = False,
) -> dict[str, Path]:
    """Table 1 products GLD01–GLD10 of an SfS DEM, named ``<region_id>_GLD<code>_<resolution>…``.

    Parameters
    ----------
    sfs_dem : Path
        The SfS DEM (e.g. ``sfs/run-DEM-final.tif``).
    prior_dem : Path
        Reference DEM the SfS DEM is differenced against (e.g. the LOLA LDEM).
    out_dir : Path
        Destination; intermediates go to ``out_dir/work``.
    images : sequence
        Mapprojected SfS input images, as [`ProductImage`][lunar_toolkit.sfs_products.ProductImage] or
        ``(path, sun_azimuth[, native_resolution])``.
    region_id : str
        ``ROIID`` of the file names.
    resolution : float
        ``RES`` of the file names (m/px).
    height_error : Path, optional
        ``<prefix>-height-error.tif`` from ``sfs --estimate-height-errors``;
        GLD03 is skipped without it.
    shadow_threshold : float
        Image values at or below this are shadow (as ``sfs --shadow-threshold``).
    bin_width : float
        Sun-azimuth bin width (deg) for the solar-bin count.
    baselines : sequence
        Slope / roughness baselines (m); ``None`` is the pixel scale.
    hillshade_sun : (azimuth, elevation)
        Hillshade illumination (deg).
    force : bool
        Recompute products that already exist.

    Returns
    -------
    dict
        Product key (`PRODUCT_CODES`, plus ``slope_<b>`` / ``roughness_<b>``)
        → path.  Under [`dry_run`][lunar_toolkit.tools.dry_run] only the CLI-tool
        products are planned; the raster-math ones need the real DEM.
    """
    out_dir = Path(out_dir)
    if not is_dry_run():
        out_dir.mkdir(parents=True, exist_ok=True)
    ims = as_product_images(images)

    def name(key: str, *suffix: Any) -> Path:
        return out_dir / product_name(region_id, PRODUCT_CODES[key], resolution, *suffix)

    out: dict[str, Path] = {"elevation": copy_elevation(sfs_dem, name("elevation"), force=force)}
    if height_error is not None and (is_dry_run() or Path(height_error).exists()):
        out["uncertainty"] = copy_height_error(height_error, name("uncertainty"), force=force)
    else:
        log.info("Skipping elevation uncertainty: no SfS height-error map")
    out["diff"] = difference(sfs_dem, prior_dem, name("diff"), force=force)
    az, el = hillshade_sun
    out["hillshade"] = hillshade(sfs_dem, name("hillshade", az, el), azimuth=az, elevation=el, force=force)
    grid = Grid.from_raster(sfs_dem) if Path(sfs_dem).exists() else None
    if ims:
        out["orthomosaic"] = orthomosaic(
            [im.path for im in ims],
            name("orthomosaic"),
            grid=grid,
            resolution=resolution,
            work_dir=out_dir / "work",
            force=force,
        )
    if is_dry_run() or grid is None:
        log.info("Skipping raster-math products (dry run or missing SfS DEM)")
        return out
    out.update(
        terrain_products(
            sfs_dem, out_dir, region_id=region_id, resolution=resolution, baselines=baselines, force=force
        )
    )
    if ims:
        out.update(
            illumination_products(
                ims,
                grid,
                out_dir,
                region_id=region_id,
                resolution=resolution,
                shadow_threshold=shadow_threshold,
                bin_width=bin_width,
                force=force,
            )
        )
    return out


def crop_window(x: float, y: float, size: float = 2500.0) -> tuple[float, float, float, float]:
    """``(ulx, uly, lrx, lry)`` of a *size* × *size* square centred on ``(x, y)``."""
    h = size / 2
    return x - h, y + h, x + h, y - h


def ensemble_subsets(
    pool: Sequence[T],
    n_subsets: int,
    size: int,
    *,
    max_overlap: float = 0.4,
    seed: int = 0,
    max_tries: int = 10000,
) -> list[list[T]]:
    """Random subsets of *pool* sharing at most ``max_overlap * size`` items pairwise.

    Bertone et al. (2026, §3.2) limit the overlap of any two SfS ensemble
    subsets to 40% of their images.  Subsets keep the order of *pool* (e.g.
    sorted by Sun azimuth).  Fewer than *n_subsets* are returned, with a
    warning, if none more fit within *max_tries* draws.
    """
    if size > len(pool):
        raise ValueError(f"subset size {size} exceeds the pool of {len(pool)}")
    rng = random.Random(seed)
    limit = math.floor(max_overlap * size + 1e-9)
    chosen: list[set[int]] = []
    for _ in range(max_tries):
        if len(chosen) == n_subsets:
            break
        cand = set(rng.sample(range(len(pool)), size))
        if all(len(cand & c) <= limit for c in chosen):
            chosen.append(cand)
    if len(chosen) < n_subsets:
        log.warning("Only %d of %d subsets with <= %d shared images", len(chosen), n_subsets, limit)
    return [[pool[i] for i in sorted(c)] for c in chosen]


@dataclass
class EnsembleStats:
    """Per-pixel spread of an ensemble of SfS DEMs on a common grid.

    Attributes
    ----------
    elevation_std, slope_std : ndarray
        Sample standard deviation (``ddof=1``) of elevation (m) and pixel-scale
        slope (deg); NaN where fewer than two members have data.
    count : ndarray
        Members with a valid elevation at each pixel.
    grid : Grid
    """

    elevation_std: np.ndarray
    slope_std: np.ndarray
    count: np.ndarray
    grid: Grid


def ensemble_stats(members: Sequence[Path], grid: Grid | None = None) -> EnsembleStats:
    """Elevation and slope spread of ensemble DEMs, warped onto *grid* (default: the first's)."""
    if len(members) < 2:
        raise ValueError("an ensemble needs at least two DEMs")
    grid = grid or Grid.from_raster(members[0])
    z = np.stack([read_raster(m, grid)[0] for m in members])
    s = np.stack([slope_deg(zi, grid.res) for zi in z])

    def std(a: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanstd(a, axis=0, ddof=1)

    return EnsembleStats(std(z), std(s), np.isfinite(z).sum(0).astype(np.uint16), grid)


def ensemble_uncertainty(
    members: Sequence[Path],
    out_dir: Path,
    *,
    region_id: str,
    resolution: float,
    reference: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Ensemble elevation / slope uncertainty maps (Bertone et al. 2026, §3.2).

    Parameters
    ----------
    members : sequence of Path
        SfS DEMs of the same crop (e.g. 2.5 km, [`crop_window`][lunar_toolkit.sfs_products.crop_window]) from
        image subsets ([`ensemble_subsets`][lunar_toolkit.sfs_products.ensemble_subsets]).
    out_dir : Path
    region_id : str
    resolution : float
        ``RES`` of the file names (m/px).
    reference : Path, optional
        Raster whose grid the members are warped onto (e.g. the cropped prior
        DEM); default the first member's grid.
    force : bool

    Returns
    -------
    dict
        ``elevation_std`` → ``<ROI>_GLDISGM_<RES>_ENS.tif`` (m) and
        ``slope_std`` → ``<ROI>_GLDSSGM_<RES>_ENS.tif`` (deg).
    """
    out = {
        "elevation_std": Path(out_dir) / product_name(region_id, "ISGM", resolution, "ENS"),
        "slope_std": Path(out_dir) / product_name(region_id, "SSGM", resolution, "ENS"),
    }
    if all(reuse_existing(p, force) for p in out.values()):
        return out
    stats = ensemble_stats(members, Grid.from_raster(reference) if reference else None)
    write_raster(out["elevation_std"], stats.elevation_std.astype(np.float32), stats.grid)
    write_raster(out["slope_std"], stats.slope_std.astype(np.float32), stats.grid)
    return out


@dataclass
class PointComparison:
    """DEM minus reference points, as written by ``geodiff`` (lon, lat in degrees; diff in m)."""

    lon: np.ndarray
    lat: np.ndarray
    diff: np.ndarray

    def stats(self) -> dict[str, float]:
        d = self.diff[np.isfinite(self.diff)]
        if d.size == 0:
            return {"count": 0}
        med = float(np.median(d))
        return {
            "count": int(d.size),
            "mean": float(d.mean()),
            "median": med,
            "std": float(d.std()),
            "rms": float(np.sqrt(np.mean(d * d))),
            "nmad": float(1.4826 * np.median(np.abs(d - med))),
            "min": float(d.min()),
            "max": float(d.max()),
        }


def compare_to_points(
    dem: Path,
    points_csv: Path,
    out_prefix: Path,
    *,
    csv_format: str = "1:lon 2:lat 3:height_above_datum",
    csv_srs: str | None = None,
    force: bool = False,
) -> PointComparison:
    """Validate *dem* against altimetry points (e.g. LOLA RDR) with ``geodiff --csv-format``.

    Points off the DEM are dropped by ``geodiff``.  Returns an empty
    comparison under [`dry_run`][lunar_toolkit.tools.dry_run].
    """
    diff_csv = Path(f"{out_prefix}-diff.csv")
    if not reuse_existing(diff_csv, force):
        asp.geodiff(dem, points_csv, out_prefix, csv_format=csv_format, csv_srs=csv_srs)
    if is_dry_run():
        empty = np.empty(0)
        return PointComparison(empty, empty, empty)
    data = np.loadtxt(diff_csv, delimiter=",", comments="#", ndmin=2)
    return PointComparison(data[:, 0], data[:, 1], data[:, 2])
