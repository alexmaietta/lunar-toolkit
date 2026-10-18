"""shadowcam - download KPLO ShadowCam cubes and prepare them for ASP (SPICE + CSM camera)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COMMAND = "shadowcam"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Download ShadowCam cubes, spiceinit them and write CSM cameras for ASP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar shadowcam M074289249S M074296291S -o ./sc
  lunar shadowcam M013178466SC --validate -o ./sc
  lunar shadowcam --point -89.67 129.8 --radius-km 5 --inc 80 90 --limit 4 -o ./sc
  lunar shadowcam --aoi malapert.geojson --product cal --download-only -o ./sc
  lunar shadowcam ./sc/M013178466SE.cub --mapproject --dem ref.tif --resolution 1.7
  lunar shadowcam M013178466S --dry-run

Follows the ASP 3.7 ShadowCam example, which needs ASP's custom ISIS / ALE
build and the KPLO kernels in $ISISDATA/kplo/kernels (rsync.im-ldi.com::kplo).
Observation IDs (M...S) map to the raw cube M...SE (used in ASP's example) or,
with --product cal, the calibrated M...SC.  Per product, <output-dir> gets
<id>.cub (spiceinit'ed in place), <id>.json (isd_generate), with --validate
<id>.cam_test.txt and with --mapproject <id>.tif (ASP mapproject with --dem,
else ISIS cam2map).  Steps whose output exists are skipped (--force redoes them).
--bbox is WEST SOUTH EAST NORTH, longitudes positive east.
""",
    )
    p.add_argument(
        "sources",
        nargs="*",
        metavar="SOURCE",
        help="Observation / product IDs (M013178466S, M013178466SE, M013178466SC) or local .cub files.",
    )
    p.add_argument(
        "--from-file", metavar="FILE", help="Text file of sources, one per line (#comments ignored)."
    )
    p.add_argument(
        "--output-dir",
        "-o",
        default=None,
        metavar="DIR",
        help="Output directory (default: current working directory).",
    )
    p.add_argument(
        "--product",
        "-t",
        choices=("raw", "cal"),
        default="raw",
        help="raw (M...SE, ASP's example, default) or cal (M...SC) for IDs and searches.",
    )

    search = p.add_argument_group("ODE search (instead of / in addition to SOURCE)")
    region = search.add_mutually_exclusive_group()
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
    search.add_argument(
        "--radius-km", type=float, default=1.0, metavar="KM", help="Radius for --point (default: 1 km)."
    )
    search.add_argument("--inc", nargs=2, type=float, metavar=("MIN", "MAX"), help="Incidence angle range.")
    search.add_argument("--emi", nargs=2, type=float, metavar=("MIN", "MAX"), help="Emission angle range.")
    search.add_argument("--phase", nargs=2, type=float, metavar=("MIN", "MAX"), help="Phase angle range.")
    search.add_argument("--time", nargs=2, metavar=("START", "END"), help="Observation time range (UTC).")
    search.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of products from the search (default: 10).",
    )

    proc = p.add_argument_group("Processing")
    proc.add_argument(
        "--web",
        action="store_true",
        help="spiceinit with the USGS SPICE web service (isd_generate still needs local kernels).",
    )
    proc.add_argument(
        "--shape",
        default="system",
        metavar="SHAPE",
        help="spiceinit shape: system (LOLA DEM in $ISISDATA/base/dems, default) or ellipsoid.",
    )
    proc.add_argument(
        "--no-csm",
        dest="csm",
        action="store_false",
        help="Do not write a CSM camera (.json) with isd_generate.",
    )
    proc.add_argument(
        "--validate",
        action="store_true",
        help="Compare ISIS and CSM cameras with cam_test (<id>.cam_test.txt).",
    )
    proc.add_argument("--mapproject", action="store_true", help="Also write a map-projected <id>.tif.")
    proc.add_argument("--dem", metavar="DEM", help="DEM for ASP mapproject (default: ISIS cam2map).")
    proc.add_argument(
        "--resolution",
        type=float,
        default=None,
        metavar="M",
        help="Map resolution in m/pixel for --mapproject.",
    )
    proc.add_argument(
        "--t-srs", default=None, metavar="PROJ", help="PROJ string for ASP mapproject with --dem."
    )
    proc.add_argument(
        "--projection",
        default="equirectangular",
        metavar="PROJ",
        help="ISIS projection for --mapproject without --dem (default: equirectangular).",
    )
    proc.add_argument("--keep", action="store_true", help="Keep the ISIS map-projected cube.")
    proc.add_argument("--force", action="store_true", help="Redo processing steps whose output exists.")
    proc.add_argument("--download-only", action="store_true", help="Download only; no processing.")
    proc.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ISIS / ASP commands without downloading or running anything.",
    )
    proc.add_argument(
        "--workers", type=int, default=1, metavar="N", help="Products processed in parallel (default: 1)."
    )


def run(args: argparse.Namespace) -> int:
    from ..ode import OdeClient, OdeError
    from ..shadowcam.sc import (
        ScProcessor,
        ScProduct,
        download_product,
        lookup_products,
        product_id_for,
        sc_products,
    )
    from ._utils import load_sources, ode_filters, print_dry_run, region_args

    sources = load_sources(args)
    region = region_args(args)
    if not sources and not region:
        print(
            "No sources. Pass IDs, .cub paths, --from-file, --bbox, --point or --aoi.",
            file=sys.stderr,
        )
        return 1

    cubes = [Path(s) for s in sources if s.lower().endswith(".cub")]
    ids = [s for s in sources if not s.lower().endswith(".cub")]
    client = OdeClient()
    try:
        wanted = {product_id_for(i, args.product) for i in ids}
        products: list[ScProduct] = sc_products(lookup_products(ids, client, args.product)) if ids else []
        if region:
            products += sc_products(
                client.search(
                    "shadowcam",
                    args.product,
                    max_results=args.limit,
                    **region,
                    **ode_filters(args),
                )
            )
    except (OdeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    products = list({p.product_id: p for p in products}.values())
    for missing in sorted(wanted - {p.product_id for p in products}):
        print(f"  {missing}  -  not found on ODE", file=sys.stderr)

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    print(f"{len(products) + len(cubes)} product(s) -> {output_dir}")
    for p in products:
        size = f"  ({p.cube_kbytes / 1024:.0f} MB)" if p.cube_kbytes else ""
        print(f"  {p.product_id}{size}")

    if args.download_only:
        failed = 0
        for p in products:
            try:
                print(f"  {p.product_id}  ->  {download_product(p, output_dir)}")
            except Exception as exc:
                print(f"  {p.product_id}  -  FAILED: {exc}", file=sys.stderr)
                failed += 1
        return 1 if failed else 0

    proc = ScProcessor(
        output_dir,
        product=args.product,
        web=args.web,
        csm=args.csm,
        validate=args.validate,
        mapproject=args.mapproject,
        dem=Path(args.dem) if args.dem else None,
        resolution=args.resolution,
        t_srs=args.t_srs,
        projection=args.projection,
        keep=args.keep,
        force=args.force,
        spiceinit_params={"shape": args.shape},
    )
    if args.dry_run:

        def process_all() -> None:
            for p in products:
                proc.process_cube(proc.cube_path(p))
            for cub in cubes:
                proc.process_cube(cub)

        print_dry_run(process_all)
        return 0

    sources_all: list = [*products, *cubes]
    results = proc.process_batch(sources_all, max_workers=args.workers)
    ok = 0
    for src, out in zip(sources_all, results):
        name = src.product_id if isinstance(src, ScProduct) else src.name
        if out is None:
            print(f"  {name}  -  FAILED", file=sys.stderr)
            continue
        ok += 1
        files = [f for f in (out.cube, out.camera, out.cam_test, out.geotiff) if f]
        print(f"  {name}  ->  {', '.join(str(f) for f in files)}")
    return 0 if ok == len(sources_all) else 1
