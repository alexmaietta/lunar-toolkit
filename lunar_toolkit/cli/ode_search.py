"""ode-search - search ShadowCam / LROC / Kaguya TC products via ODE REST."""

import argparse
import sys
from pathlib import Path

COMMAND = "ode-search"


def add_parser(sub: argparse._SubParsersAction) -> None:
    from ..ode import INSTRUMENTS

    types = "; ".join(f"{k}: {', '.join(v['product_types'])}" for k, v in INSTRUMENTS.items())
    p = sub.add_parser(
        COMMAND,
        help="Search lunar products (ShadowCam, LROC NAC, Kaguya TC) via the ODE REST API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples
--------
  lunar ode-search --instrument shadowcam --bbox 0 -90 360 -89 --inc 80 90
  lunar ode-search --instrument shadowcam --point -89.67 129.8 --radius-km 15
  lunar ode-search --instrument lroc --product-type cdr --bbox 355 -1 5 1 --csv hits.csv
  lunar ode-search --instrument shadowcam --point -89.67 129.8 --radius-km 5 \\
      --download ./shadowcam --kind product
  lunar ode-search --instrument shadowcam --aoi aoi.geojson   # QuickMap polygon

Product types - {types}
(raw ODE pt codes such as EDRNAC4 are accepted too).
--bbox is WEST SOUTH EAST NORTH, longitudes positive east; WEST > EAST crosses 0/360.
--aoi takes a GeoJSON / WKT file or a WKT string; ODE is queried with its bounding
box and footprints are then intersected with the polygon.
""",
    )
    p.add_argument(
        "--instrument",
        "-i",
        default="shadowcam",
        help=f"Instrument: {', '.join(INSTRUMENTS)} (default: shadowcam).",
    )
    p.add_argument(
        "--product-type",
        "-t",
        default=None,
        metavar="PT",
        help="Product type alias or ODE pt code (default: instrument's default).",
    )
    region = p.add_mutually_exclusive_group()
    region.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in degrees.",
    )
    region.add_argument(
        "--point",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        help="Search around a point (use with --radius-km).",
    )
    region.add_argument(
        "--aoi", metavar="FILE_OR_WKT", help="Area of interest: GeoJSON / WKT file or WKT polygon string."
    )
    p.add_argument(
        "--radius-km", type=float, default=1.0, metavar="KM", help="Radius for --point (default: 1 km)."
    )
    p.add_argument("--inc", nargs=2, type=float, metavar=("MIN", "MAX"), help="Incidence angle range.")
    p.add_argument("--emi", nargs=2, type=float, metavar=("MIN", "MAX"), help="Emission angle range.")
    p.add_argument("--phase", nargs=2, type=float, metavar=("MIN", "MAX"), help="Phase angle range.")
    p.add_argument(
        "--time",
        nargs=2,
        metavar=("START", "END"),
        help="Observation time range (UTC, e.g. 2023-01-01 2023-12-31).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=200,
        metavar="N",
        help="Maximum number of products to return (default: 200).",
    )
    p.add_argument("--count", action="store_true", help="Only print the number of matches.")
    p.add_argument("--csv", metavar="FILE", help="Write results (common schema) to CSV.")
    p.add_argument("--download", metavar="DIR", help="Download the selected files to DIR.")
    p.add_argument(
        "--kind",
        default="product",
        help="Files to download: product, label, browse, referenced, derived, all.",
    )
    p.add_argument(
        "--match",
        default=None,
        metavar="TEXT",
        help="Only download files whose name contains TEXT (e.g. _cog.tif).",
    )
    p.add_argument(
        "--workers", type=int, default=4, metavar="N", help="Parallel download workers (default: 4)."
    )


def run(args: argparse.Namespace) -> int:
    from ..ode import OdeClient, OdeError
    from ._utils import ode_filters, region_args

    region, filters = region_args(args), ode_filters(args)

    client = OdeClient()
    try:
        if args.count:
            print(client.count(args.instrument, args.product_type, **region, **filters))
            return 0
        df = client.search(args.instrument, args.product_type, max_results=args.limit, **region, **filters)
    except (OdeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"{len(df)} product(s)")
    if not df.empty:
        cols = [
            "product_id",
            "product_type",
            "start_time",
            "center_lat",
            "center_lon",
            "incidence",
            "emission",
            "phase",
            "pixel_scale",
        ]
        print(df[cols].to_string(index=False, max_rows=50))
    if args.csv:
        df.drop(columns=["files"]).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")
    if args.download and not df.empty:
        results = client.download(
            df, Path(args.download), kind=args.kind, match=args.match, workers=args.workers
        )
        failed = sum(1 for p in results.values() if p is None)
        print(f"Downloaded {len(results) - failed}/{len(results)} file(s).")
        return 1 if failed else 0
    return 0
