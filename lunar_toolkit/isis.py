"""ISIS command wrappers (plus ALE ``isd_generate``), returning output paths."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .constants import DEFAULT_PROJECTION
from .tools import ToolError, find_tool, is_dry_run, isis_params, reuse_existing, run, tool_env

log = logging.getLogger(__name__)


def isis(program: str, cwd: Path | None = None, **params: Any) -> None:
    """Run any ISIS *program* with ``key=value`` *params*."""
    run(program, isis_params(params), cwd=cwd)


def lronac2isis(from_img: Path, to_cub: Path, **params: Any) -> Path:
    isis("lronac2isis", from_=from_img, to=to_cub, **params)
    return Path(to_cub)


def kaguyatc2isis(from_img: Path, to_cub: Path, **params: Any) -> Path:
    isis("kaguyatc2isis", from_=from_img, to=to_cub, **params)
    return Path(to_cub)


MOON_SHAPE_DEM = "base/dems/ldem_128ppd_Mar2011_clon180_radius_pad.cub"


def spiceinit(from_cub: Path, spksmithed: bool = True, web: bool = True, **params: Any) -> Path:
    """Attach SPICE; ``web=True`` uses the USGS SPICE web service.

    ISIS's default lunar shape model is the 2 GB LOLA DEM in ``$ISISDATA`` (not in a
    minimal ISISDATA).  When it is missing and no other shape is asked for, the
    ellipsoid is used instead: ASP projects onto its own prior DEM and CSM cameras do
    not use the ISIS shape model.
    """
    if params.get("shape") in (None, "system") and not system_shape_available():
        log.warning("No ISIS lunar shape model ($ISISDATA/%s); spiceinit shape=ellipsoid", MOON_SHAPE_DEM)
        params["shape"] = "ellipsoid"
    isis("spiceinit", from_=from_cub, spksmithed=spksmithed, web=web, **params)
    return Path(from_cub)


def system_shape_available() -> bool:
    """Whether ``$ISISDATA`` (as the ISIS tools see it) has the default lunar shape model.

    True when ``spiceinit`` cannot be found or under [`dry_run`][lunar_toolkit.tools.dry_run]
    (nothing to decide then).
    """
    tool = find_tool("spiceinit")
    if tool is None or is_dry_run():
        return True
    data = tool_env(tool).get("ISISDATA")
    return data is not None and (Path(data) / MOON_SHAPE_DEM).exists()


def lronaccal(from_cub: Path, to_cub: Path, **params: Any) -> Path:
    isis("lronaccal", from_=from_cub, to=to_cub, **params)
    return Path(to_cub)


def lronacecho(from_cub: Path, to_cub: Path, **params: Any) -> Path:
    isis("lronacecho", from_=from_cub, to=to_cub, **params)
    return Path(to_cub)


def reduce(from_cub: Path, to_cub: Path, scale: float, algorithm: str = "average", **params: Any) -> Path:
    """Downsample by *scale* in both directions (``sscale`` = ``lscale``)."""
    isis(
        "reduce",
        from_=from_cub,
        to=to_cub,
        algorithm=algorithm,
        mode="scale",
        sscale=scale,
        lscale=scale,
        **params,
    )
    return Path(to_cub)


def mosrange(fromlist: Path, projection: str, to_map: Path, **params: Any) -> Path:
    isis("mosrange", fromlist=fromlist, projection=projection, to=to_map, **params)
    return Path(to_map)


def cam2map(
    from_cub: Path,
    to_cub: Path,
    map: Path,
    warpalgorithm: str = "forwardpatch",
    patchsize: int = 50,
    **params: Any,
) -> Path:
    isis(
        "cam2map",
        from_=from_cub,
        to=to_cub,
        map=map,
        warpalgorithm=warpalgorithm,
        patchsize=patchsize,
        **params,
    )
    return Path(to_cub)


def map_project(
    from_cub: Path,
    to_cub: Path,
    projection: str = DEFAULT_PROJECTION,
    resolution: float | None = None,
    keep: bool = False,
) -> Path:
    """``mosrange`` → ``cam2map``: project a cube onto a map fitted to its footprint.

    Parameters
    ----------
    from_cub, to_cub : Path
        spiceinit'ed input cube and map-projected output cube.
    projection : str
        ISIS projection for ``mosrange``.
    resolution : float, optional
        Map resolution in m/pixel (default: ``mosrange``'s estimate).
    keep : bool
        Keep the ``<id>.lis`` / ``<id>.map`` files written next to *to_cub*.
    """
    to_cub = Path(to_cub)
    pid = Path(from_cub).name.split(".")[0]
    fromlist, map_file = to_cub.with_name(f"{pid}.lis"), to_cub.with_name(f"{pid}.map")
    if not is_dry_run():
        fromlist.write_text(f"{from_cub}\n")
    mosrange(fromlist, projection, map_file)
    params: dict[str, Any] = {"pixres": "mpp", "resolution": resolution} if resolution else {}
    cam2map(from_cub, to_cub, map_file, **params)
    if not keep and not is_dry_run():
        fromlist.unlink(missing_ok=True)
        map_file.unlink(missing_ok=True)
    return to_cub


def isd_generate(
    cub: Path, out: Path | None = None, kernel: Path | None = None, only_naif_spice: bool = False
) -> Path:
    """Write a CSM camera (ISD ``.json``) for a spiceinit'ed cube.

    *kernel* defaults to the cube itself, so its attached SPICE is used.
    Returns the JSON path (``<cub stem>.json`` next to the cube by default).
    """
    cub = Path(cub)
    out = Path(out) if out else cub.with_suffix(".json")
    args = ["-k", kernel or cub, "-o", out]
    if only_naif_spice:
        args.append("--only_naif_spice")
    run("isd_generate", [*args, cub])
    return out


def calibrate_lronac(
    img: Path,
    out_dir: Path | None = None,
    *,
    web: bool = True,
    keep: bool = False,
    force: bool = False,
) -> Path:
    """lronac2isis → spiceinit → lronaccal → lronacecho for one NAC EDR.

    Writes ``<id>.cal.echo.cub`` in *out_dir* (default: next to *img*) and
    returns it; an existing output is reused unless *force*.  The raw and
    calibrated cubes are removed unless *keep*.
    """
    img = Path(img)
    out_dir = Path(out_dir) if out_dir else img.parent
    pid = img.name.split(".")[0]
    raw, cal, echo = (out_dir / f"{pid}{ext}" for ext in (".cub", ".cal.cub", ".cal.echo.cub"))
    if reuse_existing(echo, force):
        return echo
    if not is_dry_run():
        out_dir.mkdir(parents=True, exist_ok=True)
    lronac2isis(img, raw)
    spiceinit(raw, web=web)
    try:
        lronaccal(raw, cal)
    except ToolError:
        # Some dates have no pair of darks within lronaccal's 45-day window.
        log.warning("lronaccal with paired darks failed for %s; retrying with the nearest dark", pid)
        lronaccal(raw, cal, darkfiletype="NEAREST")
    lronacecho(cal, echo)
    if not keep and not is_dry_run():
        raw.unlink(missing_ok=True)
        cal.unlink(missing_ok=True)
    return echo


def prepare_lronac(
    src: Path, out_dir: Path, *, csm: bool = True, web: bool = True, force: bool = False
) -> tuple[Path, Path | None]:
    """Calibrated cube and optional CSM camera for one NAC image.

    A ``.cub`` *src* is used as is; anything else (an EDR ``.IMG``) goes through
    [`calibrate_lronac`][lunar_toolkit.isis.calibrate_lronac] into *out_dir*.  With *csm*, ``<id>.json`` is
    written in *out_dir* by [`isd_generate`][lunar_toolkit.isis.isd_generate].  Existing outputs are reused
    unless *force*.

    Returns
    -------
    (cube, camera)
        *camera* is None without *csm*.
    """
    src, out_dir = Path(src), Path(out_dir)
    cube = src if src.suffix.lower() == ".cub" else calibrate_lronac(src, out_dir, web=web, force=force)
    if not csm:
        return cube, None
    camera = out_dir / f"{src.name.split('.')[0]}.json"
    if not reuse_existing(camera, force):
        if not is_dry_run():
            out_dir.mkdir(parents=True, exist_ok=True)
        isd_generate(cube, camera)
    return cube, camera
