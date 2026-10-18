"""Shared helpers for CLI commands."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def load_sources(args) -> list[str]:
    """Collect source strings from positional args and/or --from-file."""
    sources: list[str] = list(getattr(args, "sources", []) or [])
    from_file = getattr(args, "from_file", None)
    if from_file:
        p = Path(from_file)
        if not p.exists():
            print(f"Error: file not found: {from_file}", file=sys.stderr)
            sys.exit(1)
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sources.append(line)
    return sources


def extract_product_id(source: str) -> str:
    """Derive a product ID from a product ID string, local path, or URL."""
    s = source.strip()
    if s.startswith(("http://", "https://")):
        return Path(urlparse(s).path).stem.upper()
    if not _looks_like_id(s):
        return Path(s).stem.upper()
    return s.upper()


def _looks_like_id(s: str) -> bool:
    from ..constants import NAC_ID_RE, WAC_ID_RE

    return bool(NAC_ID_RE.match(s) or WAC_ID_RE.match(s))


def resolve_images(sources: list[str], raw_dir: Path, data_dir=None, download: bool = True) -> list[Path]:
    """Local .IMG / .cub paths as-is; NAC product IDs downloaded (EDR) to *raw_dir*.

    With ``download=False`` IDs map to ``<raw_dir>/<ID>.IMG`` without fetching.
    """
    from ..lroc.nac import NacImageCollection

    paths = [Path(s) for s in sources if Path(s).exists()]
    ids = [s for s in sources if not Path(s).exists()]
    if ids and not download:
        return paths + [Path(raw_dir) / f"{i.upper()}.IMG" for i in ids]
    if ids:
        found = NacImageCollection.from_sources(ids, data_dir=data_dir)
        missing = {i.upper() for i in ids} - {img.product_id for img in found}
        if missing:
            raise SystemExit(
                f"Not in the local index: {', '.join(sorted(missing))} "
                "(run `lunar build-db edr` or pass .IMG paths)"
            )
        downloaded = found.download_edrs(raw_dir)
        failed = [pid for pid, p in downloaded.items() if p is None]
        if failed:
            raise SystemExit(f"Download failed: {', '.join(failed)}")
        paths += [p for p in (downloaded[img.product_id] for img in found) if p is not None]
    return paths


def region_args(args: argparse.Namespace) -> dict[str, Any]:
    """Search-region keywords from ``--bbox`` / ``--point`` + ``--radius-km`` / ``--aoi`` (empty if none)."""
    if args.bbox:
        west, south, east, north = args.bbox
        return {"lat": (south, north), "lon": (west, east)}
    if args.point:
        return {"point": tuple(args.point), "radius_km": args.radius_km}
    if getattr(args, "aoi", None):
        return {"aoi": args.aoi}
    return {}


def ode_filters(args: argparse.Namespace) -> dict[str, Any]:
    """:meth:`OdeClient.search` filters from ``--inc`` / ``--emi`` / ``--phase`` / ``--time``."""
    return {
        "incidence": tuple(args.inc) if args.inc else None,
        "emission": tuple(args.emi) if args.emi else None,
        "phase": tuple(args.phase) if args.phase else None,
        "time_range": tuple(args.time) if args.time else None,
    }


def print_dry_run(func: Callable[[], object]) -> None:
    """Call *func* with tool commands recorded instead of run, then print them."""
    from ..tools import dry_run

    with dry_run() as cmds:
        func()
    for c in cmds:
        print(shlex.join(c))


def parse_options(items: list[str] | None) -> dict:
    """``["max_iterations=5", "float_albedo=true", "x=none"]`` -> typed dict."""
    out: dict = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"Expected KEY=VALUE, got {item!r}")
        out[key.strip().replace("-", "_")] = _typed(value.strip())
    return out


def _typed(value: str):
    low = value.lower()
    if low in ("true", "yes", "on", ""):
        return True
    if low in ("false", "no", "off"):
        return False
    if low == "none":
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value
