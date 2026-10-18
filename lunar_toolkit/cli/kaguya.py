"""kaguya - download Kaguya TC (Level 2B0) or MI (Level 2B2) products and import them into ISIS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

COMMAND = "kaguya"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Download Kaguya TC / MI images and import them into ISIS (cube + CSM camera).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar kaguya TC1W2B0_01_02936N034E0938 TC2W2B0_01_02936N036E0938 -o ./tc
  lunar kaguya --point 34.0 93.8 --radius-km 2 --limit 4 -o ./tc
  lunar kaguya --bbox 93.5 33.9 94.0 34.1 --product-type mono --inc 40 70 -o ./tc
  lunar kaguya ./tc/TC1W2B0_01_02936N034E0938.lbl --mapproject --resolution 10
  lunar kaguya TC1W2B0_01_02936N034E0938 --download-only
  lunar kaguya TC1W2B0_01_02936N034E0938 --dry-run
  lunar kaguya MVA_2B2_01_01935N340E0939 MNA_2B2_01_01935N339E0939 -o ./mi
  lunar kaguya --instrument mi --point 34.0 93.9 --product-type nir --limit 2 -o ./mi

Per product, <output-dir> gets <id>.lbl, <id>.cub (kaguyatc2isis / kaguyami2isis
+ spiceinit), <id>.json (CSM camera from isd_generate) and, with --mapproject,
<id>.tif.  MI cubes hold all VIS (5) or NIR (4) bands; ALE's MI drivers need
local kernels, so MI CSM cameras default to --no-web runs only (--csm forces).
The instrument is inferred from the product IDs (MI: M[VN]A_2B2_...) unless
--instrument is given.
Steps whose output exists are skipped (use --force to redo them).
--bbox is WEST SOUTH EAST NORTH, longitudes positive east.
""",
    )
    p.add_argument(
        "sources",
        nargs="*",
        metavar="SOURCE",
        help="TC / MI product IDs (e.g. TC1W2B0_01_02936N034E0938) or local .lbl files.",
    )
    p.add_argument(
        "--instrument",
        "-i",
        choices=("tc", "mi"),
        default=None,
        help="Terrain Camera or Multiband Imager (default: from SOURCE, else tc).",
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
    search.add_argument(
        "--radius-km", type=float, default=1.0, metavar="KM", help="Radius for --point (default: 1 km)."
    )
    search.add_argument(
        "--product-type",
        "-t",
        default=None,
        metavar="PT",
        help="TC: stereo (TCWL2B, default) or mono (TCSL2B); MI: vis (VIS2B, default) or nir (NIR2B).",
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

    isis = p.add_argument_group("ISIS import")
    isis.add_argument(
        "--no-web",
        dest="web",
        action="store_false",
        help="spiceinit with local kernels ($ISISDATA/kaguya) instead of the web service.",
    )
    isis.add_argument(
        "--shape",
        default="system",
        metavar="SHAPE",
        help="spiceinit shape: system (LOLA DEM in $ISISDATA/base/dems, default) or ellipsoid.",
    )
    isis.add_argument(
        "--csm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write a CSM camera (.json) with isd_generate (default: TC always, MI with --no-web).",
    )
    isis.add_argument(
        "--only-naif-spice",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="TC: pass --only_naif_spice to isd_generate (default: with --no-web; MI always).",
    )
    isis.add_argument("--mapproject", action="store_true", help="Also cam2map and write <id>.tif.")
    isis.add_argument(
        "--projection",
        default="equirectangular",
        metavar="PROJ",
        help="ISIS projection for --mapproject (default: equirectangular).",
    )
    isis.add_argument(
        "--resolution",
        type=float,
        default=None,
        metavar="M",
        help="Map resolution in m/pixel for --mapproject.",
    )
    isis.add_argument("--keep", action="store_true", help="Keep the PDS .img and the map-projected cube.")
    isis.add_argument("--force", action="store_true", help="Redo steps whose output exists.")
    isis.add_argument(
        "--download-only", action="store_true", help="Download (and gunzip) only; no ISIS processing."
    )
    isis.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ISIS commands without downloading or running anything.",
    )
    isis.add_argument(
        "--workers", type=int, default=1, metavar="N", help="Products processed in parallel (default: 1)."
    )


def run(args: argparse.Namespace) -> int:
    from ..downloader import url_filename
    from ..kaguya import mi, tc
    from ..ode import OdeClient, OdeError
    from ._utils import load_sources, ode_filters, print_dry_run, region_args

    sources = load_sources(args)
    if not sources and not (args.bbox or args.point):
        print("No sources. Pass product IDs, .lbl paths, --from-file, --bbox or --point.", file=sys.stderr)
        return 1

    instrument = args.instrument or (
        "mi" if any(mi.is_mi_product_id(Path(s).name.split(".")[0]) for s in sources) else "tc"
    )
    steps: dict[str, tuple[Any, Any, Any]] = {
        "mi": (mi.lookup_mi_products, mi.mi_products, mi.download_mi_product),
        "tc": (tc.lookup_products, tc.tc_products, tc.download_product),
    }
    lookup, select, download = steps[instrument]

    labels = [Path(s) for s in sources if s.lower().endswith(".lbl")]
    ids = [s for s in sources if not s.lower().endswith(".lbl")]
    client = OdeClient()
    try:
        products: list[Any] = select(lookup(ids, client)) if ids else []
        if args.bbox or args.point:
            products += select(
                client.search(
                    f"kaguya_{instrument}",
                    args.product_type,
                    max_results=args.limit,
                    **region_args(args),
                    **ode_filters(args),
                )
            )
    except (OdeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    products = list({p.product_id: p for p in products}.values())
    seen = {p.product_id for p in products}
    for missing in sorted({i.upper() for i in ids} - seen):
        print(f"  {missing}  -  not found on ODE", file=sys.stderr)

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    print(f"{len(products) + len(labels)} Kaguya {instrument.upper()} product(s) -> {output_dir}")
    for p in products:
        size = f"  ({p.image_kbytes / 1024:.0f} MB)" if p.image_kbytes else ""
        print(f"  {p.product_id}{size}")

    if args.download_only:
        failed = 0
        for p in products:
            try:
                print(f"  {p.product_id}  ->  {download(p, output_dir, force=args.force)}")
            except Exception as exc:
                print(f"  {p.product_id}  -  FAILED: {exc}", file=sys.stderr)
                failed += 1
        return 1 if failed else 0

    common: dict[str, Any] = {
        "web": args.web,
        "mapproject": args.mapproject,
        "projection": args.projection,
        "resolution": args.resolution,
        "keep": args.keep,
        "force": args.force,
        "spiceinit_params": {"shape": args.shape},
    }
    proc: tc.TcProcessor | mi.MiProcessor
    if instrument == "mi":
        proc = mi.MiProcessor(output_dir, csm=args.csm, **common)
    else:
        proc = tc.TcProcessor(
            output_dir, csm=args.csm is not False, only_naif_spice=args.only_naif_spice, **common
        )
    if args.dry_run:

        def import_all() -> None:
            for p in products:
                proc.import_label(output_dir / url_filename(p.label_url))
            for lbl in labels:
                proc.import_label(lbl)

        print_dry_run(import_all)
        return 0

    sources_all: list = [*products, *labels]
    results = proc.process_batch(sources_all, max_workers=args.workers)
    ok = 0
    for src, out in zip(sources_all, results):
        name = src.name if isinstance(src, Path) else src.product_id
        if out is None:
            print(f"  {name}  -  FAILED", file=sys.stderr)
            continue
        ok += 1
        files = [f for f in (out.cube, out.camera, out.geotiff) if f]
        print(f"  {name}  ->  {', '.join(str(f) for f in files)}")
    return 0 if ok == len(sources_all) else 1
