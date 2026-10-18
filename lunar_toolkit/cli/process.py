"""process - download and run the ISIS + GDAL pipeline on LROC NAC images."""

import argparse
import sys
from pathlib import Path

from ..constants import DEFAULT_PROJECTION

COMMAND = "process"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Run the ISIS + GDAL pipeline on LROC NAC images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar process M1096757719LE
  lunar process M1096757719LE M1096757719RE --workers 2
  lunar process --from-file ids.txt --output-dir ./output
  lunar process /data/M1096757719LE.IMG --keep --resolution 1
  lunar process https://pds.lroc.im-ldi.com/data/.../M1096757719LE.IMG

Everything is written to --output-dir: the GeoTIFF <id>.tif (map-projected I/F,
DEFLATE-compressed) plus intermediates (<id>.IMG, <id>.cal.echo.cub, <id>.map.cub),
which are removed after success unless --keep.  Steps whose output exists are
skipped, so re-running resumes (use --force to redo them).
""",
    )
    p.add_argument(
        "sources",
        nargs="*",
        metavar="SOURCE",
        help="Product IDs (e.g. M1096757719LE), local .IMG paths, or PDS URLs.",
    )
    p.add_argument(
        "--from-file",
        metavar="FILE",
        help="Text file of sources, one per line (#comments ignored).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory (default: current working directory).",
    )
    p.add_argument(
        "--keep",
        action="store_true",
        help="Keep the downloaded EDR and the intermediate ISIS cubes.",
    )
    p.add_argument("--force", action="store_true", help="Redo steps whose output exists.")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for batch processing (default: 1).",
    )
    p.add_argument(
        "--projection",
        default=DEFAULT_PROJECTION,
        metavar="PROJ",
        help=f"ISIS map projection for mosrange (default: {DEFAULT_PROJECTION}).",
    )
    p.add_argument("--resolution", type=float, default=None, metavar="M", help="Map resolution in m/pixel.")
    p.add_argument(
        "--no-web",
        dest="web",
        action="store_false",
        help="Use local SPICE kernels in spiceinit instead of the USGS web service.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override Parquet data directory.",
    )


def run(args: argparse.Namespace) -> int:
    from ..lroc.nac import NacImageCollection, NacProcessor
    from ._utils import load_sources

    sources = load_sources(args)
    if not sources:
        print("No sources provided. Pass product IDs, paths, URLs, or --from-file.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Route: product IDs -> NacImageCollection; existing paths / URLs -> processor directly
    direct_sources: list[str] = []
    id_sources: list[str] = []
    for s in sources:
        s = s.strip()
        if s.startswith(("http://", "https://")) or Path(s).exists():
            direct_sources.append(s)
        else:
            id_sources.append(s)

    opts = {
        "projection": args.projection,
        "resolution": args.resolution,
        "web": args.web,
        "keep": args.keep,
        "force": args.force,
    }
    ok = total = 0

    if id_sources:
        collection = NacImageCollection.from_sources(id_sources, data_dir=args.data_dir)
        results = collection.process(output_dir, workers=args.workers, **opts)
        for pid, tif in results.items():
            total += 1
            if tif:
                print(f"  {pid}  ->  {tif}")
                ok += 1
            else:
                print(f"  {pid}  -  FAILED", file=sys.stderr)

    if direct_sources:
        proc = NacProcessor(output_dir, **opts)
        tifs = proc.process_batch(direct_sources, max_workers=args.workers)
        for src, tif in zip(direct_sources, tifs):
            total += 1
            label = Path(src).name if not src.startswith("http") else src
            if tif:
                print(f"  {label}  ->  {tif}")
                ok += 1
            else:
                print(f"  {label}  -  FAILED", file=sys.stderr)

    return 0 if ok == total else 1
