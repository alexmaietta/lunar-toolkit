"""Flat YAML run configuration for ``lunar sfs``: load, validate, merge, resolve regions and dump."""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import logging
import os
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml

from .sfs import SfsConfig

log = logging.getLogger(__name__)

PGDA = "pgda"
RUNS_DIR_ENV = "LUNAR_RUNS_DIR"
"""``dem: pgda`` (or no ``dem`` with a ``region``): the region's PGDA LOLA 5 m ``surf`` DEM."""

OPTION_KEYS = ("bundle_adjust_options", "refine_options", "sfs_options")
FLOW_KEYS = ("projwin", "stereo_convergence", "crop_center")


class ConfigError(ValueError):
    """An invalid configuration: unknown key, wrong type or inconsistent settings."""


@dataclass
class RunConfig:
    """Everything a ``lunar sfs`` run needs: inputs, region and the
    [`SfsConfig`][lunar_toolkit.sfs.SfsConfig].

    In YAML the run keys below and the [`SfsConfig`][lunar_toolkit.sfs.SfsConfig] fields share one flat
    mapping.  Relative paths are relative to the current directory.

    Parameters
    ----------
    work_dir : Path
        Run directory; a relative path is taken under *runs_dir*.
    runs_dir : Path, optional
        Where relative ``work_dir`` paths live (default: ``$LUNAR_RUNS_DIR``, else the
        current directory).
        Working / output directory.
    dem : str, optional
        Prior DEM path, or ``"pgda"`` for the region's PGDA LOLA 5 m DEM (the
        default with a *region*).
    region : str or int, optional
        Artemis III region ([`lunar_toolkit.reference.get_region`][lunar_toolkit.reference.get_region]): sets
        the ``projwin`` (its ROI, or a *crop_km* window), ``region_id`` and the
        image search AOI.
    crop_km : float, optional
        Side of a square window at the region centre ([`crop_window`][lunar_toolkit.reference.crop_window]).
    crop_center : (lat, lon), optional
        Centre of that window (default: the ROI centre).
    aoi : str, optional
        AOI (GeoJSON / WKT file or WKT) for the image search and, without
        ``projwin``, the crop.
    sources : list of str
        NAC product IDs or ``.IMG`` / ``.cub`` paths (default: search the AOI).
    from_file : Path, optional
        Text file of more sources, one per line.
    max_slew, max_pixel_scale : float, optional
        Image search cuts (deg, m/px).  No slew cut by default: slewed images give the
        stereo convergence for registration, and ``max_sfs_slew`` keeps them out of SfS.
    max_images : int, optional
        Cap on the searched images before download (stereo pairs + illumination coverage;
        [`lunar_toolkit.sfs.cap_candidates`][lunar_toolkit.sfs.cap_candidates]).
    pgda_dir : Path, optional
        Where PGDA files are downloaded (default: the reference cache).
    data_dir : Path, optional
        Parquet index directory.
    checksums : bool
        MD5 the input files (under 2 GB) in ``provenance.json``.
    sfs : SfsConfig
    """

    work_dir: Path = Path("sfs_work")
    runs_dir: Path | None = None
    dem: str | None = None
    region: str | int | None = None
    crop_km: float | None = None
    crop_center: tuple[float, float] | None = None
    aoi: str | None = None
    sources: list[str] = field(default_factory=list)
    from_file: Path | None = None
    max_slew: float | None = None
    max_pixel_scale: float | None = 1.5
    max_images: int | None = None
    pgda_dir: Path | None = None
    data_dir: Path | None = None
    checksums: bool = True
    sfs: SfsConfig = field(default_factory=SfsConfig)


def _hints(cls: type) -> dict[str, Any]:
    names = [f.name for f in dataclasses.fields(cls)]
    hints = typing.get_type_hints(cls)
    return {n: hints[n] for n in names if n != "sfs"}


RUN_KEYS: dict[str, Any] = _hints(RunConfig)
SFS_KEYS: dict[str, Any] = _hints(SfsConfig)
KEYS: dict[str, Any] = {**RUN_KEYS, **SFS_KEYS}


def _describe(hint: Any) -> str:
    if typing.get_origin(hint) in (Union, types.UnionType):
        return " or ".join(_describe(a) for a in typing.get_args(hint))
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is tuple:
        return f"list of {len(args)} {_describe(args[0])}s"
    if origin is list:
        return f"list of {_describe(args[0])}s"
    if origin is dict:
        return "mapping"
    names = {bool: "boolean", int: "integer", float: "number", str: "string", Path: "path"}
    return "null" if hint is type(None) else names.get(hint, str(hint))


def _coerce(key: str, value: Any, hint: Any) -> Any:
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if hint is Any:
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for arm in args:
            if arm is not type(None):
                try:
                    return _coerce(key, value, arm)
                except ConfigError:
                    pass
    elif hint is bool and isinstance(value, bool):
        return value
    elif hint is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    elif hint is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    elif hint is str and isinstance(value, str):
        return value
    elif hint is Path and isinstance(value, (str, Path)):
        return Path(value)
    elif origin is tuple and isinstance(value, (list, tuple)) and len(value) == len(args):
        return tuple(_coerce(key, v, a) for v, a in zip(value, args))
    elif origin is list and isinstance(value, (list, tuple)):
        return [_coerce(key, v, args[0]) for v in value]
    elif origin is dict and isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
        return {k.replace("-", "_"): v for k, v in value.items()}
    raise ConfigError(f"{key}: expected {_describe(hint)}, got {value!r}")


def from_dict(data: Mapping[str, Any], source: str = "config") -> RunConfig:
    """A validated [`RunConfig`][lunar_toolkit.config.RunConfig] from a flat mapping (keys as in `KEYS`).

    Raises
    ------
    ConfigError
        For unknown keys (with suggestions), wrong types and invalid choices.
    """
    if not isinstance(data, Mapping):
        raise ConfigError(f"{source}: expected a mapping of settings, got {type(data).__name__}")
    norm = {str(k).replace("-", "_"): v for k, v in data.items()}
    for key in norm:
        if key not in KEYS:
            close = difflib.get_close_matches(key, list(KEYS), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise ConfigError(f"{source}: unknown key {key!r}{hint}")
    try:
        values = {k: _coerce(k, v, KEYS[k]) for k, v in norm.items()}
    except ConfigError as exc:
        raise ConfigError(f"{source}: {exc}") from None
    try:
        sfs = SfsConfig(**{k: v for k, v in values.items() if k in SFS_KEYS})
    except ValueError as exc:
        raise ConfigError(f"{source}: {exc}") from None
    return RunConfig(**{k: v for k, v in values.items() if k in RUN_KEYS}, sfs=sfs)


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def to_dict(cfg: RunConfig) -> dict[str, Any]:
    """Flat, YAML / JSON-ready mapping of every setting (run keys first)."""
    out = {k: _plain(getattr(cfg, k)) for k in RUN_KEYS}
    out.update({k: _plain(getattr(cfg.sfs, k)) for k in SFS_KEYS})
    return out


def merge(cfg: RunConfig, overrides: Mapping[str, Any], source: str = "overrides") -> RunConfig:
    """*cfg* with *overrides* applied; option dicts are merged key by key, other values replaced."""
    data = to_dict(cfg)
    for key, value in overrides.items():
        key = key.replace("-", "_")
        if key in OPTION_KEYS and isinstance(value, Mapping) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return from_dict(data, source)


def load_config(path: Path) -> RunConfig:
    """Read and validate a YAML config (an empty file gives the defaults)."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc.strerror or exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    return from_dict(data if data is not None else {}, str(path))


def dump_config(cfg: RunConfig, path: Path | None = None) -> str:
    """YAML text of every setting (written to *path* when given)."""
    from .constants import APP_VERSION

    lines = [f"# lunar sfs configuration (lunar-toolkit {APP_VERSION})\n"]
    for key, value in to_dict(cfg).items():
        if key == next(iter(SFS_KEYS)):
            lines.append("\n# SfS pipeline settings (lunar_toolkit.sfs.SfsConfig)\n")
        if key in FLOW_KEYS or (isinstance(value, (list, dict)) and not value):
            flow = yaml.safe_dump(value, default_flow_style=True, width=10**6).strip()
            lines.append(f"{key}: {flow.removesuffix('...').strip()}\n")
        else:
            lines.append(yaml.safe_dump({key: value}, default_flow_style=False, sort_keys=False, width=110))
    text = "".join(lines)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text


def config_hash(cfg: RunConfig) -> str:
    """SHA-256 of the settings (canonical JSON of [`to_dict`][lunar_toolkit.config.to_dict])."""
    return hashlib.sha256(json.dumps(to_dict(cfg), sort_keys=True).encode()).hexdigest()


def pgda_file(region: Any, kind: str, out_dir: Path | None = None, download: bool = True) -> Path:
    """Local path of a region's PGDA LOLA 5 m raster (``surf``, ``ldec``, ``toterr``, ...).

    Downloaded into ``out_dir/<site>/`` when missing and *download*; otherwise the
    expected path (``<site>_final_adj_5mpp_<kind>.tif``, ``<site>_final_adj.xyzi``)
    is returned without touching the network or the disk.
    """
    from . import reference as ref

    site = ref._site_name(region)
    out_dir = Path(out_dir) if out_dir else ref.REFERENCE_CACHE_DIR / "pgda"
    name = f"{site}_final_adj.xyzi" if kind == "xyzi" else f"{site}_final_adj_5mpp_{kind}.tif"
    expected = out_dir / site / name
    if expected.exists() or not download:
        return expected
    files = ref.download_pgda(site, [kind], out_dir)[kind]
    if not files:
        raise ConfigError(f"PGDA site {site} has no {kind} file")
    return files[0]


def region_id(region: Any) -> str:
    """Bertone et al. product prefix of a region, e.g. ``A311`` for Malapert Massif."""
    from . import reference as ref

    return f"A3{ref.get_region(region).id:02d}"


def resolve(cfg: RunConfig, download: bool = True) -> RunConfig:
    """Fill what *region* implies: ``projwin``, ``region_id`` and the PGDA prior DEM.

    An explicit ``projwin`` wins, then ``aoi`` (the crop is computed from it at run
    time), then a ``crop_km`` window, then the region ROI.  ``dem: pgda`` (or none)
    is the region's PGDA ``surf`` DEM, downloaded when missing and *download*.

    Raises
    ------
    ConfigError
        For a ``crop_km`` / ``dem: pgda`` without a region, or an unknown region.
    """
    from . import reference as ref

    runs_dir = cfg.runs_dir or (Path(os.environ[RUNS_DIR_ENV]) if os.environ.get(RUNS_DIR_ENV) else None)
    if runs_dir is not None and not Path(cfg.work_dir).is_absolute():
        cfg = merge(cfg, {"work_dir": str(Path(runs_dir).expanduser() / cfg.work_dir), "runs_dir": None})
    if cfg.region is None:
        if cfg.crop_km is not None or cfg.crop_center is not None:
            raise ConfigError("crop_km / crop_center need a region")
        if cfg.dem == PGDA:
            raise ConfigError("dem: pgda needs a region")
        return cfg
    try:
        reg = ref.get_region(cfg.region)
    except KeyError as exc:
        raise ConfigError(f"region: {exc.args[0]}") from None
    changes: dict[str, Any] = {}
    if cfg.sfs.projwin is None and cfg.aoi is None:
        if cfg.crop_km is not None:
            box = ref.crop_window(reg, cfg.crop_km, cfg.crop_center)
        else:
            box = reg.roi
        changes["projwin"] = box.projwin
    if cfg.sfs.region_id is None:
        changes["region_id"] = region_id(reg)
    if cfg.dem in (None, PGDA):
        try:
            changes["dem"] = str(pgda_file(reg, "surf", cfg.pgda_dir, download=download))
        except KeyError as exc:
            raise ConfigError(f"dem: {exc.args[0]}") from None
    return merge(cfg, changes, "region") if changes else cfg


def search_aoi(cfg: RunConfig) -> str | None:
    """Image-search AOI: ``aoi``, else the region crop (``projwin`` as lon/lat WKT), else None."""
    from . import reference as ref

    if cfg.aoi is not None:
        return cfg.aoi
    if cfg.region is None or cfg.sfs.projwin is None:
        return None
    ulx, uly, lrx, lry = cfg.sfs.projwin
    return ref.PsBox(ulx, lry, lrx, uly).wkt()


def load_sources(cfg: RunConfig) -> list[str]:
    """``sources`` plus the non-comment lines of ``from_file``."""
    sources = list(cfg.sources)
    if cfg.from_file is not None:
        try:
            text = Path(cfg.from_file).read_text()
        except OSError as exc:
            raise ConfigError(f"from_file: cannot read {cfg.from_file}: {exc.strerror or exc}") from None
        sources += [s.strip() for s in text.splitlines() if s.strip() and not s.strip().startswith("#")]
    return sources
