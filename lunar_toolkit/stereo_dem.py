"""Stereo DEM from one LRO NAC pair with ISIS and ASP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import asp, isis
from .tools import reuse_existing

# ASP's recommended correlator settings for NAC stereo.
STEREO_DEFAULTS: dict[str, Any] = {
    "stereo_algorithm": "asp_mgm",
    "subpixel_mode": 9,
}

# bundle_adjust of a lone pair before stereo.
PAIR_BUNDLE_ADJUST_DEFAULTS: dict[str, Any] = {
    "ip_per_image": 20000,
    "min_matches": 1,
    "camera_weight": 0,
    "tri_weight": 0.1,
    "tri_robust_threshold": 0.1,
    "remove_outliers_params": "75.0 3.0 3.0 5.0",
    "datum": "D_MOON",
}


def stereo_dem(
    left: Path,
    right: Path,
    work_dir: Path,
    *,
    resolution: float = 2.0,
    prior_dem: Path | None = None,
    image_resolution: float = 1.0,
    align: bool = True,
    max_displacement: float = 200.0,
    csm: bool = True,
    web_spice: bool = True,
    cameras: tuple[Path, Path] | None = None,
    mapprojected: tuple[Path, Path] | None = None,
    bundle_adjust_prefix: Path | None = None,
    processes: int | None = None,
    threads: int | None = None,
    stereo_options: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    """Build a DEM from a NAC stereo pair; returns the final DEM path.

    Parameters
    ----------
    left, right : Path
        NAC EDR ``.IMG`` files or calibrated ``.cub`` files.
    resolution : float
        DEM grid spacing (m/px); about 3× the image resolution is typical.
    prior_dem : Path, optional
        Reference DEM: inputs are mapprojected onto it at *image_resolution*
        before correlation, and the result is aligned to it when *align*.
    cameras : (Path, Path), optional
        Existing CSM cameras for *left* / *right* (with *csm*; default: written by
        ``isd_generate`` into ``<work_dir>/images``).
    mapprojected : (Path, Path), optional
        Existing images mapprojected onto *prior_dem* (any camera / adjustments:
        ASP reads those from the geoheader), instead of mapprojecting here.
    bundle_adjust_prefix : Path, optional
        Adjustments to triangulate with, instead of bundle-adjusting the pair here.
    """
    work_dir = Path(work_dir)
    img_dir = work_dir / "images"

    cubes: list[Path] = []
    cams: list[Path] = []
    for k, src in enumerate((Path(left), Path(right))):
        cube, cam = isis.prepare_lronac(src, img_dir, csm=csm and not cameras, web=web_spice, force=force)
        if csm and cameras:
            cam = Path(cameras[k])
        cubes.append(cube)
        if cam:
            cams.append(cam)

    ba = Path(bundle_adjust_prefix) if bundle_adjust_prefix else work_dir / "ba" / "run"
    if not bundle_adjust_prefix and not reuse_existing(Path(f"{ba}-final_residuals_pointmap.csv"), force):
        asp.bundle_adjust(cubes, ba, cams, threads=threads, **PAIR_BUNDLE_ADJUST_DEFAULTS)

    opts = {
        **STEREO_DEFAULTS,
        "processes": processes,
        "threads": threads,
        "bundle_adjust_prefix": ba,
        **(stereo_options or {}),
    }
    out = work_dir / "stereo" / "run"
    pc = Path(f"{out}-PC.tif")
    pair_cams: list[Path | None] = [*cams] if cams else [None, None]
    if not reuse_existing(pc, force):
        if prior_dem:
            maps = [Path(m) for m in mapprojected] if mapprojected else []
            if not maps:
                for cube, camera in zip(cubes, pair_cams):
                    m = work_dir / "prj" / f"{cube.name.split('.')[0]}.map.tif"
                    if not reuse_existing(m, force):
                        asp.mapproject(
                            prior_dem,
                            cube,
                            m,
                            camera,
                            tr=image_resolution,
                            bundle_adjust_prefix=ba,
                            threads=threads,
                        )
                    maps.append(m)
            asp.stereo(
                maps[0],
                maps[1],
                out,
                *(cubes if not cams else cams),
                dem=prior_dem,
                alignment_method="none",
                **opts,
            )
        else:
            asp.stereo(cubes[0], cubes[1], out, *pair_cams, alignment_method="local_epipolar", **opts)

    dem = Path(f"{out}-DEM.tif")
    if not reuse_existing(dem, force):
        asp.point2dem(pc, tr=resolution, datum="D_MOON", errorimage=True)
    if not (prior_dem and align):
        return dem

    align_prefix = work_dir / "align" / "run"
    trans = Path(f"{align_prefix}-trans_source.tif")
    if not reuse_existing(trans, force):
        asp.pc_align(
            prior_dem,
            pc,
            align_prefix,
            max_displacement=max_displacement,
            save_transformed_source_points=True,
            highest_accuracy=True,
        )
    aligned = Path(f"{align_prefix}-trans_source-DEM.tif")
    if not reuse_existing(aligned, force):
        asp.point2dem(trans, tr=resolution, datum="D_MOON")
    return aligned
