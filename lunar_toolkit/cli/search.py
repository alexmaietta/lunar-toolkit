"""search - footprint search of the local PDS index (poles and 0/360 safe)."""

import argparse
import sys
from pathlib import Path

COMMAND = "search"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Find images whose footprint intersects a box, point+radius or AOI polygon (local index).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar search --bbox 0 -90 360 -85 --nac-only --inc 80 95
  lunar search --bbox 355 -5 5 5               # crosses the 0/360 seam
  lunar search --point -89.67 129.8 --radius-km 15 --csv shackleton.csv
  lunar search --aoi aoi.geojson --nac-only    # QuickMap-exported polygon
  lunar search --aoi "POLYGON((350 -5, 10 -5, 10 5, 350 5, 350 -5))"

--bbox is WEST SOUTH EAST NORTH (degrees, positive east; WEST > EAST crosses 0/360).
--aoi takes a GeoJSON / WKT file or a WKT string (lon lat, degrees, positive
east); bbox matches are refined by exact polygon intersection.
""",
    )
    p.add_argument("--dataset", choices=["edr", "cdr"], default="edr", help="Index to search (default: edr).")
    region = p.add_mutually_exclusive_group(required=True)
    region.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    region.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"))
    region.add_argument(
        "--aoi", metavar="FILE_OR_WKT", help="Area of interest: GeoJSON / WKT file or WKT polygon string."
    )
    p.add_argument("--radius-km", type=float, default=1.0, metavar="KM")
    p.add_argument("--inc", nargs=2, type=float, metavar=("MIN", "MAX"), help="Incidence angle range.")
    p.add_argument("--emi", nargs=2, type=float, metavar=("MIN", "MAX"), help="Emission angle range.")
    p.add_argument("--phase", nargs=2, type=float, metavar=("MIN", "MAX"), help="Phase angle range.")
    p.add_argument("--nac-only", action="store_true", help="Only NAC products.")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    p.add_argument("--csv", metavar="FILE", help="Write matching rows to CSV.")
    p.add_argument("--wkt", action="store_true", help="Include an FP_WKT footprint column.")
    p.add_argument(
        "--data-dir", type=Path, default=None, metavar="DIR", help="Override Parquet data directory."
    )


def run(args: argparse.Namespace) -> int:
    from ..lroc.cumindex import CumIndex
    from ._utils import region_args

    ci = CumIndex(dataset=args.dataset, data_dir=args.data_dir)
    try:
        df = ci.search(
            **region_args(args),
            incidence_angle=tuple(args.inc) if args.inc else None,
            emission_angle=tuple(args.emi) if args.emi else None,
            phase_angle=tuple(args.phase) if args.phase else None,
            nac_only=args.nac_only,
            wkt=args.wkt,
            limit=args.limit,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"{len(df)} image(s)")
    if not df.empty:
        cols = [
            c
            for c in (
                "PRODUCT_ID",
                "START_TIME",
                "CENTER_LATITUDE",
                "CENTER_LONGITUDE",
                "INCIDENCE_ANGLE",
                "EMISSION_ANGLE",
                "RESOLUTION",
            )
            if c in df.columns
        ]
        print(df[cols].to_string(index=False, max_rows=40))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")
    return 0
