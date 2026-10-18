"""sfs-report - validation metrics, figures and an HTML report of an SfS work dir."""

import argparse
import sys
from pathlib import Path

COMMAND = "sfs-report"

PGDA_REPORT_KINDS = ("ldec", "toterr", "slp", "slperr", "xyzi")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Metrics, figures and an HTML report of an SfS work dir (Bertone et al. 2026 validation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar sfs-report malapert_crop                       # work dir files only
  lunar sfs-report malapert_crop --region malapert     # + PGDA LDEC, uncertainty maps, LOLA points
  lunar sfs-report malapert_crop --region malapert --fetch-published   # + published SDEM
  lunar sfs-report w --toterr Site23_final_adj_5mpp_toterr.tif --lola-points lola.csv -o w/report2

Sections (skipped when their inputs are missing): bundle-adjustment residuals before /
after and match offsets (Fig. 5); removed images; illumination coverage with 0/1/2/>=3
non-redundant Sun directions (Figs. 3, 6); SfS minus prior DEM, split by LOLA count
(Table 2); SfS minus LOLA points vs slope (Fig. 7); ensemble uncertainty (Fig. 8);
SfS - LDEM residuals over the LDEM uncertainty maps (Fig. 9); SfS minus the published SDEM.
Writes OUTPUT_DIR/report.html (figures embedded), metrics.json, tables/*.csv, figures/*.png.
""",
    )
    p.add_argument("work_dir", type=Path, metavar="WORK_DIR", help="lunar sfs work directory.")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Destination (default: WORK_DIR/report).",
    )
    p.add_argument("--sfs-dem", type=Path, metavar="TIF", help="SfS DEM (default: products GLDELEV or sfs/).")
    p.add_argument("--prior-dem", type=Path, metavar="TIF", help="Prior DEM (default: dem/prior-<RES>m.tif).")
    p.add_argument("--region-id", metavar="ID", help="Product name prefix (default: from config.yaml).")
    p.add_argument(
        "--res", type=float, metavar="M", help="Product resolution label (default: from config.yaml)."
    )
    p.add_argument("--ldec", type=Path, metavar="TIF", help="PGDA LOLA count raster (*_ldec.tif).")
    p.add_argument("--toterr", type=Path, metavar="TIF", help="PGDA elevation uncertainty (*_toterr.tif).")
    p.add_argument("--slperr", type=Path, metavar="TIF", help="PGDA slope uncertainty (*_slperr.tif).")
    p.add_argument("--ldem-slope", type=Path, metavar="TIF", help="PGDA LDEM slope (*_slp.tif).")
    p.add_argument(
        "--lola-points",
        type=Path,
        metavar="CSV",
        help="LOLA points as an ASP CSV (easting, northing, height).",
    )
    p.add_argument("--lola-xyzi", type=Path, metavar="XYZI", help="PGDA LOLA point cloud (*.xyzi).")
    p.add_argument(
        "--published-dem", type=Path, metavar="TIF", help="Published Bertone et al. SDEM (GLDELEV)."
    )
    p.add_argument(
        "--region",
        metavar="REGION",
        help="Artemis III region: download its PGDA files (see --pgda) for the inputs not given.",
    )
    p.add_argument(
        "--pgda",
        nargs="+",
        default=list(PGDA_REPORT_KINDS),
        choices=PGDA_REPORT_KINDS,
        metavar="KIND",
        help=f"PGDA files fetched with --region (default: {' '.join(PGDA_REPORT_KINDS)}).",
    )
    p.add_argument(
        "--fetch-published",
        action="store_true",
        help="With --region: extract the published GLDELEV from the Zenodo zip (HTTP range reads).",
    )
    p.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Download directory for --region files (default: the reference cache).",
    )
    p.add_argument("--sections", nargs="+", metavar="NAME", help="Only these sections (default: all).")
    p.add_argument(
        "--preview-size",
        type=int,
        default=800,
        metavar="PX",
        help="Longer side of the map figures (default: 800).",
    )


def _region_inputs(args: argparse.Namespace) -> dict:
    from .. import reference as ref
    from ..config import pgda_file

    reg = ref.get_region(args.region)
    names = {"ldec": "ldec", "toterr": "toterr", "slp": "ldem_slope", "slperr": "slperr", "xyzi": "lola_xyzi"}
    out = {}
    for kind in args.pgda:
        if getattr(args, names[kind]) is None:
            out[names[kind]] = pgda_file(
                reg, kind, args.reference_dir / "pgda" if args.reference_dir else None
            )
    if args.fetch_published and args.published_dem is None:
        base = args.reference_dir or ref.REFERENCE_CACHE_DIR
        dest = base / "sdem" / reg.zenodo_file.removesuffix(".zip")
        out["published_dem"] = ref.extract_sdem(reg, ["GLDELEV"], dest)[0]
    return out


def run(args: argparse.Namespace) -> int:
    from .. import sfs_report

    if not args.work_dir.is_dir():
        print(f"Error: {args.work_dir} is not a directory", file=sys.stderr)
        return 1
    given = {
        k: getattr(args, k)
        for k in (
            "sfs_dem",
            "prior_dem",
            "region_id",
            "ldec",
            "toterr",
            "slperr",
            "ldem_slope",
            "lola_points",
            "lola_xyzi",
            "published_dem",
        )
    }
    try:
        if args.region:
            given.update(_region_inputs(args))
        inputs = sfs_report.ReportInputs.from_work_dir(
            args.work_dir, resolution=args.res, preview_size=args.preview_size, **given
        )
        out = sfs_report.make_report(inputs, args.output_dir, args.sections)
    except (KeyError, ValueError, OSError) as exc:
        print(f"Error: {exc.args[0] if isinstance(exc, KeyError) and exc.args else exc}", file=sys.stderr)
        return 1
    print(f"Report -> {out}")
    return 0
