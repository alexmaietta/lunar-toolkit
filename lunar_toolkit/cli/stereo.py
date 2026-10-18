"""stereo - DEM from one LROC NAC stereo pair (ISIS + ASP)."""

import argparse
import sys
from pathlib import Path

COMMAND = "stereo"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Build a DEM from a NAC stereo pair (bundle_adjust, parallel_stereo, point2dem).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar stereo M181058717LE M181073012LE --res 2 --work-dir pair1
  lunar stereo L.IMG R.IMG --dem ldem.tif --res 2      # mapproject first, align to the DEM
  lunar stereo L.cub R.cub --dry-run

Needs ISIS and ASP on PATH (or ASP_ROOT / ISISROOT).
""",
    )
    p.add_argument("left", help="Left image: NAC EDR product ID, .IMG or .cub path.")
    p.add_argument("right", help="Right image: NAC EDR product ID, .IMG or .cub path.")
    p.add_argument("--work-dir", default="stereo_work", metavar="DIR")
    p.add_argument("--res", type=float, default=2.0, metavar="M", help="DEM m/px (default: 2).")
    p.add_argument("--dem", default=None, metavar="TIF", help="Prior DEM to mapproject onto and align to.")
    p.add_argument(
        "--image-res",
        type=float,
        default=1.0,
        metavar="M",
        help="Mapprojected image m/px when --dem is set (default: 1).",
    )
    p.add_argument("--no-align", action="store_true", help="Skip pc_align to --dem.")
    p.add_argument("--max-displacement", type=float, default=200.0, metavar="M")
    p.add_argument("--no-csm", action="store_true", help="Use ISIS cameras instead of CSM.")
    p.add_argument("--local-spice", action="store_true")
    p.add_argument("--processes", type=int, default=None, metavar="N")
    p.add_argument("--threads", type=int, default=None, metavar="N")
    p.add_argument(
        "--stereo-opt",
        action="append",
        metavar="KEY=VALUE",
        help="Extra / overriding parallel_stereo option (repeatable).",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    p.add_argument("--data-dir", type=Path, default=None, metavar="DIR")


def run(args: argparse.Namespace) -> int:
    from ..stereo_dem import stereo_dem
    from ..tools import ToolError
    from ._utils import parse_options, print_dry_run, resolve_images

    work_dir = Path(args.work_dir)
    left, right = resolve_images(
        [args.left, args.right], work_dir / "raw", args.data_dir, download=not args.dry_run
    )
    kwargs = dict(
        resolution=args.res,
        prior_dem=Path(args.dem) if args.dem else None,
        image_resolution=args.image_res,
        align=not args.no_align,
        max_displacement=args.max_displacement,
        csm=not args.no_csm,
        web_spice=not args.local_spice,
        processes=args.processes,
        threads=args.threads,
        stereo_options=parse_options(args.stereo_opt),
        force=args.force,
    )
    try:
        if args.dry_run:
            print_dry_run(lambda: stereo_dem(left, right, work_dir, **kwargs))
            return 0
        dem = stereo_dem(left, right, work_dir, **kwargs)
    except ToolError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Stereo DEM -> {dem}")
    return 0
