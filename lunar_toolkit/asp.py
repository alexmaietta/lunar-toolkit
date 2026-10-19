"""Ames Stereo Pipeline wrappers: positional inputs as arguments, other options as keywords."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .tools import ToolError, asp_options, gdal_options, is_dry_run, run


def _paths(items: Sequence[Any]) -> list[str]:
    return [str(p) for p in items]


def _parent(prefix: Path) -> None:
    if not is_dry_run():
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)


def mapproject(
    dem: Path, image: Path, out: Path, camera: Path | None = None, *, cwd: Path | None = None, **options: Any
) -> Path:
    """Orthorectify *image* onto *dem* (``tr=``, ``t_projwin=``, ``bundle_adjust_prefix=``)."""
    _parent(out)
    cams = [camera] if camera else []
    run("mapproject", [*asp_options(options), dem, image, *cams, out], cwd=cwd)
    return Path(out)


def bundle_adjust(
    images: Sequence[Path],
    out_prefix: Path,
    cameras: Sequence[Path] = (),
    *,
    mapprojected: Sequence[Path] | None = None,
    dem: Path | None = None,
    parallel: bool = False,
    cwd: Path | None = None,
    **options: Any,
) -> Path:
    """Run ``bundle_adjust`` (or ``parallel_bundle_adjust`` with *parallel*).

    *mapprojected* (same order as *images*) and the *dem* they were projected
    onto become ``--mapprojected-data``, which finds matches between images of
    very different illumination.  Returns *out_prefix*.

    Raises :class:`~lunar_toolkit.tools.ToolError` when the run writes no fresh
    ``.adjust`` file for some image: ASP exits 0 even when it finds nothing to
    optimize ("No points to optimize ... Cannot continue").
    """
    _parent(out_prefix)
    if mapprojected is not None:
        options["mapprojected_data"] = " ".join(_paths([*mapprojected, *([dem] if dem else [])]))
    tool = "parallel_bundle_adjust" if parallel else "bundle_adjust"
    start = time.time()
    run(tool, [*_paths(images), *_paths(cameras), *asp_options(options), "-o", out_prefix], cwd=cwd)
    if not is_dry_run():
        prefix = Path(cwd or ".") / out_prefix
        cams: Sequence[Path | None] = cameras if len(cameras) == len(images) else [None] * len(images)
        missing = [
            p
            for p in (bundle_adjust_file_name(prefix, im, cam) for im, cam in zip(images, cams))
            if not p.is_file() or p.stat().st_mtime < start - 2
        ]
        if missing:
            raise ToolError(
                f"{tool} wrote no adjustment for {len(missing)} of {len(images)} images "
                f"(e.g. {missing[0].name}); see {out_prefix}-log-bundle_adjust-*.txt"
            )
    return Path(out_prefix)


def apply_transform(
    images: Sequence[Path],
    transform: Path,
    out_prefix: Path,
    cameras: Sequence[Path] = (),
    *,
    input_adjustments_prefix: Path | None = None,
    cwd: Path | None = None,
    **options: Any,
) -> Path:
    """Apply a ``pc_align`` 4×4 transform to (adjusted) cameras, without optimizing."""
    return bundle_adjust(
        images,
        out_prefix,
        cameras,
        initial_transform=transform,
        input_adjustments_prefix=input_adjustments_prefix,
        apply_initial_transform_only=True,
        cwd=cwd,
        **options,
    )


def sfs(
    dem: Path,
    images: Sequence[Path],
    out_prefix: Path,
    cameras: Sequence[Path] = (),
    *,
    parallel: bool = False,
    cwd: Path | None = None,
    **options: Any,
) -> Path:
    """Run ``sfs`` (or ``parallel_sfs``); returns ``<out_prefix>-DEM-final.tif``."""
    _parent(out_prefix)
    tool = "parallel_sfs" if parallel else "sfs"
    run(
        tool, ["-i", dem, *_paths(images), *_paths(cameras), *asp_options(options), "-o", out_prefix], cwd=cwd
    )
    return Path(f"{out_prefix}-DEM-final.tif")


@dataclass(frozen=True)
class SunAngles:
    image: str
    azimuth: float
    elevation: float


_SUN_RE = re.compile(r"Sun azimuth and elevation for: (?P<image>\S+) are (?P<az>\S+) and (?P<el>\S+) degrees")


def parse_sun_angles(text: str) -> list[SunAngles]:
    """Parse ``sfs --query`` output into per-image Sun azimuth / elevation."""
    return [SunAngles(m["image"], float(m["az"]), float(m["el"])) for m in _SUN_RE.finditer(text)]


def sun_angles(
    dem: Path,
    images: Sequence[Path],
    cameras: Sequence[Path] = (),
    *,
    out_prefix: Path = Path("sfs_query/run"),
    cwd: Path | None = None,
    **options: Any,
) -> list[SunAngles]:
    """Sun azimuth (from north) and elevation per image at the DEM centre.

    Runs ``sfs --query``; results come back in input order.  Sort by azimuth
    before bundle adjustment and SfS.
    """
    _parent(out_prefix)
    res = run(
        "sfs",
        ["-i", dem, *_paths(images), *_paths(cameras), "--query", *asp_options(options), "-o", out_prefix],
        cwd=cwd,
        capture=True,
    )
    return parse_sun_angles(res.stdout or "")


def pc_align(
    reference: Path,
    source: Path,
    out_prefix: Path,
    *,
    max_displacement: float,
    cwd: Path | None = None,
    **options: Any,
) -> Path:
    """Align *source* to *reference*; returns ``<out_prefix>-transform.txt``."""
    _parent(out_prefix)
    run(
        "pc_align",
        ["--max-displacement", max_displacement, *asp_options(options), reference, source, "-o", out_prefix],
        cwd=cwd,
    )
    return Path(f"{out_prefix}-transform.txt")


def point2dem(cloud: Path, *, cwd: Path | None = None, **options: Any) -> Path:
    """Grid a point cloud; returns ``<prefix>-DEM.tif`` (prefix from ``o=`` or the cloud)."""
    run("point2dem", [*asp_options(options), cloud], cwd=cwd)
    prefix = options.get("o") or options.get("output_prefix")
    if prefix is None:
        stem = str(cloud)
        prefix = stem[: -len("-PC.tif")] if stem.endswith("-PC.tif") else str(Path(stem).with_suffix(""))
    return Path(f"{prefix}-DEM.tif")


def stereo(
    left: Path,
    right: Path,
    out_prefix: Path,
    left_camera: Path | None = None,
    right_camera: Path | None = None,
    *,
    dem: Path | None = None,
    parallel: bool = True,
    cwd: Path | None = None,
    **options: Any,
) -> Path:
    """Run ``parallel_stereo`` (or ``stereo``); returns ``<out_prefix>-PC.tif``.

    Pass *dem* when the inputs are mapprojected onto it.
    """
    _parent(out_prefix)
    cams = [c for c in (left_camera, right_camera) if c]
    tail = [dem] if dem else []
    tool = "parallel_stereo" if parallel else "stereo"
    run(tool, [*asp_options(options), left, right, *cams, out_prefix, *tail], cwd=cwd)
    return Path(f"{out_prefix}-PC.tif")


def dem_mosaic(inputs: Sequence[Path], out: Path, *, cwd: Path | None = None, **options: Any) -> Path:
    """Mosaic DEMs or images (``max=True`` for a maximally-lit mosaic)."""
    _parent(out)
    run("dem_mosaic", [*_paths(inputs), *asp_options(options), "-o", out], cwd=cwd)
    return Path(out)


def hillshade(
    dem: Path,
    out: Path | None = None,
    *,
    azimuth: float = 300,
    elevation: float = 20,
    cwd: Path | None = None,
    **options: Any,
) -> Path:
    out = Path(out) if out else Path(dem).with_name(Path(dem).stem + "_hillshade.tif")
    run(
        "hillshade",
        [dem, "--azimuth", azimuth, "--elevation", elevation, *asp_options(options), "-o", out],
        cwd=cwd,
    )
    return out


def geodiff(dem1: Path, dem2: Path, out_prefix: Path, *, cwd: Path | None = None, **options: Any) -> Path:
    """``dem1 - dem2`` (``absolute=True`` for |diff|); returns ``<out_prefix>-diff.tif``."""
    _parent(out_prefix)
    run("geodiff", [*asp_options(options), dem1, dem2, "-o", out_prefix], cwd=cwd)
    return Path(f"{out_prefix}-diff.tif")


def gdal_translate(
    src: Path,
    dst: Path,
    *,
    projwin: Sequence[float] | None = None,
    tr: float | None = None,
    r: str | None = None,
    cwd: Path | None = None,
    **options: Any,
) -> Path:
    """Crop (``projwin`` = ulx uly lrx lry) and / or resample (``tr`` metres, ``r`` method)."""
    _parent(dst)
    opts = {
        "projwin": projwin,
        "tr": (tr, tr) if tr is not None else None,
        "r": r,
        "of": "GTiff",
        "co": "COMPRESS=LZW",
        **options,
    }
    run("gdal_translate", [*gdal_options(opts), src, dst], cwd=cwd)
    return Path(dst)


def bundle_adjust_file_name(prefix: Path, image: Path, camera: Path | None = None) -> Path:
    """``<prefix>-<stem>.adjust`` as ASP names it (stem of *camera*, else *image*, up to the last dot)."""
    name = Path(camera or image).name
    stem = name[: name.rfind(".")] if "." in name else name
    return Path(f"{prefix}-{stem}.adjust")


@dataclass(frozen=True)
class MatchOffsetStats:
    """One row of ``<prefix>-mapproj_match_offset_stats.txt`` (metres; -1 when no matches)."""

    image: str
    p25: float
    p50: float
    p75: float
    p85: float
    p95: float
    count: int


def read_mapproj_offset_stats(path: Path) -> dict[str, MatchOffsetStats]:
    """Parse ``bundle_adjust --mapproj-dem`` per-image offsets, keyed by image file name.

    Format (ASP ``saveMapprojOffsets``): ``# image_name 25% 50% 75% 85% 95% count``.
    """
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#") or len(parts) < 7:
            continue
        p25, p50, p75, p85, p95 = map(float, parts[1:6])
        out[Path(parts[0]).name] = MatchOffsetStats(parts[0], p25, p50, p75, p85, p95, int(float(parts[6])))
    return out


@dataclass(frozen=True)
class ResidualStats:
    """One row of ``<prefix>-final_residuals_stats.txt`` (pixels)."""

    image: str
    mean: float
    median: float
    count: int


def read_residual_stats(path: Path) -> dict[str, ResidualStats]:
    """Parse per-camera reprojection errors (``# Image, mean, median, count``), keyed by file name."""
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts[0] or parts[0].startswith("#") or len(parts) < 4:
            continue
        out[Path(parts[0]).name] = ResidualStats(parts[0], float(parts[1]), float(parts[2]), int(parts[3]))
    return out


def read_residual_norms(path: Path) -> dict[str, np.ndarray]:
    """Per-image reprojection error norms (pixels) from ``<prefix>-final_residuals_raw_pixels.txt``.

    The file has, per image, a ``name, count`` line followed by *count* ``ex, ey`` lines.
    """
    out: dict[str, np.ndarray] = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        head = [p.strip() for p in lines[i].split(",")]
        i += 1
        if len(head) != 2 or not head[0] or head[0].startswith("#"):
            continue
        n = int(head[1])
        xy = np.array([[float(v) for v in ln.split(",")] for ln in lines[i : i + n]]).reshape(-1, 2)
        out[Path(head[0]).name] = np.hypot(xy[:, 0], xy[:, 1])
        i += n
    return out


def image_subset(
    images: Sequence[Path], out: Path, *, threshold: float, cwd: Path | None = None, **options: Any
) -> list[tuple[str, int]]:
    """Run ``image_subset``; returns ``(image, contributing pixels)`` best first.

    Writes the image list to ``<out>.in.txt``; ``t_projwin=(xmin, ymin, xmax, ymax)`` limits
    the evaluated extent.  Empty in a dry run.
    """
    _parent(out)
    image_list = Path(f"{out}.in.txt")
    if not is_dry_run():
        image_list.write_text("".join(f"{p}\n" for p in images))
    run(
        "image_subset",
        ["--image-list", image_list, "--threshold", threshold, *asp_options(options), "-o", out],
        cwd=cwd,
    )
    if is_dry_run() or not Path(out).exists():
        return []
    return [
        (parts[0], int(float(parts[1])))
        for parts in (ln.split() for ln in Path(out).read_text().splitlines())
        if len(parts) >= 2 and not parts[0].startswith("#")
    ]


def raster_info(path: Path) -> dict:
    """``gdalinfo -json`` of *path* (empty in a dry run)."""
    res = run("gdalinfo", ["-json", path], capture=True)
    return json.loads(res.stdout) if res.stdout else {}


def raster_bounds(path: Path) -> tuple[float, float, float, float] | None:
    """``(ulx, uly, lrx, lry)`` of a north-up raster, or None in a dry run."""
    info = raster_info(path)
    if not info:
        return None
    x0, dx, _, y0, _, dy = info["geoTransform"]
    nx, ny = info["size"]
    return x0, y0, x0 + nx * dx, y0 + ny * dy


def srs_proj4(path: Path) -> str:
    """PROJ string of a raster's spatial reference (``gdalsrsinfo -o proj4``; empty in a dry run)."""
    res = run("gdalsrsinfo", ["-o", "proj4", path], capture=True)
    return (res.stdout or "").strip().strip("'")


@dataclass(frozen=True)
class Raster:
    """Band 1 of a north-up raster: ``data[row, col]`` (NaN = nodata), top-left corner and pixel size."""

    data: np.ndarray
    x0: float
    y0: float
    dx: float
    dy: float


def read_raster(
    path: Path,
    *,
    projwin: Sequence[float] | None = None,
    outsize: tuple[int, int] | None = None,
    r: str = "average",
) -> Raster | None:
    """Read (a resampled window of) a raster through ``gdal_translate -of AAIGrid /vsistdout/``.

    *projwin* (``ulx uly lrx lry``) may extend past the raster (filled with nodata) and
    *outsize* ``(columns, rows)`` resamples with *r*, so several images read with the
    same window share one grid.  Meant for small grids; returns None in a dry run.
    """
    opts = {
        "q": True,
        "of": "AAIGrid",
        "ot": "Float32",
        "b": 1,
        "projwin": projwin,
        "outsize": outsize,
        "r": r if outsize else None,
    }
    res = run("gdal_translate", [*gdal_options(opts), path, "/vsistdout/"], capture=True)
    return parse_aaigrid(res.stdout) if res.stdout else None


def parse_aaigrid(text: str) -> Raster:
    """Parse ESRI ASCII grid text (as GDAL's AAIGrid driver writes it)."""
    lines = text.splitlines()
    head: dict[str, float] = {}
    k = 0
    while k < len(lines) and lines[k][:1].isalpha():
        key, value = lines[k].split()[:2]
        head[key.lower()] = float(value)
        k += 1
    nx, ny = int(head["ncols"]), int(head["nrows"])
    dx = head.get("dx", head.get("cellsize"))
    dy = head.get("dy", head.get("cellsize"))
    if dx is None or dy is None:
        raise ValueError("AAIGrid header lacks cellsize / dx / dy")
    x0 = head["xllcorner"] if "xllcorner" in head else head["xllcenter"] - dx / 2
    yll = head["yllcorner"] if "yllcorner" in head else head["yllcenter"] - dy / 2
    # GDAL also writes the .prj (WKT) to /vsistdout/ after the cells.
    data = np.array(" ".join(lines[k:]).split()[: nx * ny], dtype=float).reshape(ny, nx)
    if "nodata_value" in head:
        data[data == head["nodata_value"]] = np.nan
    return Raster(data, x0, yll + ny * dy, dx, -dy)
