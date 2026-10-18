"""pairs - rank NAC stereo pairs, or pick illumination-diverse images for SfS."""

import argparse
import sys
from pathlib import Path

COMMAND = "pairs"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Rank NAC stereo pairs (or select SfS images with --sfs) over a box, point+radius or AOI polygon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar pairs --point -89.67 129.8 --radius-km 3 --top 20
  lunar pairs --bbox 355 -5 5 5 --conv 15 45 --csv pairs.csv
  lunar pairs --point -89.67 129.8 --radius-km 3 --sfs --inc 80 95 --list sfs_images.txt
  lunar pairs --point -89.67 129.8 --radius-km 3 --sfs --coverage --views 3
  lunar pairs --aoi aoi.geojson --sfs --coverage
  lunar pairs --point -89.5 0 --radius-km 30 --engine duckdb --workspace work/pole

--bbox is WEST SOUTH EAST NORTH (degrees, positive east; WEST > EAST crosses 0/360).
--aoi takes a GeoJSON / WKT file or a WKT string (lon lat, degrees, positive east).
The search region is also the AOI that overlaps / coverage are measured in.
--sfs output (and --list) is sorted by sub-solar azimuth, as ASP SfS expects.
--engine duckdb ranks pairs with a DuckDB spatial self-join (same pairs and metrics as
the Python engine); auto uses it from 300 images when the spatial extension loads.
--workspace keeps the images and pairs tables in DIR/lunar.duckdb (see lunar workspace).
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
    p.add_argument("--emi-max", type=float, default=None, metavar="DEG", help="Maximum emission angle.")
    p.add_argument(
        "--conv",
        nargs=2,
        type=float,
        default=(10.0, 60.0),
        metavar=("MIN", "MAX"),
        help="Convergence angle limits (default: 10 60).",
    )
    p.add_argument("--max-inc-diff", type=float, default=20.0, metavar="DEG")
    p.add_argument(
        "--max-az-diff",
        type=float,
        default=30.0,
        metavar="DEG",
        help="Maximum sub-solar azimuth difference within a pair.",
    )
    p.add_argument("--min-overlap", type=float, default=0.1, metavar="FRAC")
    p.add_argument(
        "--sfs", action="store_true", help="Select illumination-diverse images instead of stereo pairs."
    )
    p.add_argument(
        "--az-bin",
        type=float,
        default=30.0,
        metavar="DEG",
        help="--sfs: azimuth bin width / minimum separation (default: 30).",
    )
    p.add_argument("--per-bin", type=int, default=1, metavar="N")
    p.add_argument("--max-images", type=int, default=None, metavar="N")
    p.add_argument(
        "--greedy",
        action="store_true",
        help="--sfs: enforce --az-bin separation between all picks instead of fixed bins.",
    )
    p.add_argument(
        "--coverage", action="store_true", help="--sfs: per-location greedy selection over an AOI grid."
    )
    p.add_argument(
        "--views",
        type=int,
        default=3,
        metavar="N",
        help="--sfs --coverage: target distinct Sun azimuths per location (default: 3).",
    )
    p.add_argument(
        "--engine",
        choices=["auto", "python", "duckdb"],
        default="auto",
        help="Stereo pair engine (default: auto = duckdb from 300 images).",
    )
    p.add_argument(
        "--workspace",
        type=Path,
        metavar="DIR",
        help="Write the images and pairs tables to DIR/lunar.duckdb (implies --engine duckdb).",
    )
    p.add_argument("--top", type=int, default=30, metavar="N", help="Rows to print (default: 30).")
    p.add_argument("--csv", metavar="FILE", help="Write the full result to CSV.")
    p.add_argument("--list", metavar="FILE", help="--sfs: write selected product IDs, one per line.")
    p.add_argument(
        "--data-dir", type=Path, default=None, metavar="DIR", help="Override Parquet data directory."
    )


def run(args: argparse.Namespace) -> int:
    from ..lroc.cumindex import CumIndex
    from ._utils import region_args

    ci = CumIndex(dataset=args.dataset, data_dir=args.data_dir)
    region = region_args(args)
    aoi = region.get("aoi", region)
    try:
        df = ci.search(
            **region,
            incidence_angle=tuple(args.inc) if args.inc else None,
            emission_angle=(0.0, args.emi_max) if args.emi_max is not None else None,
            nac_only=True,
            wkt=True,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    obs = ci.to_observations(df)
    print(f"{len(obs)} image(s) in region")

    if args.sfs:
        from ..selection import select_for_coverage, select_illumination_diverse

        if args.coverage:
            res = select_for_coverage(
                obs,
                aoi,
                azimuth_sep_deg=args.az_bin,
                views_per_cell=args.views,
                max_images=args.max_images,
            )
            vf = res.attrs.get("view_fractions", {})
            if vf:
                print(
                    "AOI fraction by number of distinct Sun azimuths: "
                    + ", ".join(f"{k}: {v:.0%}" for k, v in vf.items())
                )
        else:
            res = select_illumination_diverse(
                obs,
                azimuth_bin_deg=args.az_bin,
                per_bin=args.per_bin,
                max_images=args.max_images,
                aoi=aoi,
                method="greedy" if args.greedy else "bins",
            )
        print(f"{len(res)} image(s) selected (sorted by Sun azimuth)")
        cols = [
            c
            for c in (
                "product_id",
                "start_time",
                "sub_solar_azimuth",
                "incidence",
                "emission",
                "pixel_scale",
                "aoi_coverage",
                "selection_score",
            )
            if c in res.columns
        ]
        if args.list:
            Path(args.list).write_text("".join(f"{p}\n" for p in res["product_id"]))
            print(f"Wrote {args.list}")
    else:
        res = _stereo_pairs(
            obs,
            args,
            aoi=aoi,
            convergence=tuple(args.conv),
            max_incidence_diff=args.max_inc_diff,
            max_azimuth_diff=args.max_az_diff,
            min_overlap=args.min_overlap,
        )
        if res is None:
            return 1
        print(f"{len(res)} pair(s)")
        cols = [
            "product_id_1",
            "product_id_2",
            "convergence",
            "b_h",
            "incidence_diff",
            "azimuth_diff",
            "resolution_ratio",
            "aoi_coverage",
            "dt_days",
            "score",
        ]

    if not res.empty:
        print(res[cols].head(args.top).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if args.csv:
        res.drop(columns=[c for c in ("footprint_wkt",) if c in res.columns]).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")
    return 0


def _stereo_pairs(obs, args: argparse.Namespace, **kwargs):
    """find_stereo_pairs, or its DuckDB engine (--engine / --workspace); None on error."""
    from ..stereo import find_stereo_pairs
    from ..workspace import (
        PAIRS_ENGINE_THRESHOLD,
        Workspace,
        WorkspaceError,
        find_stereo_pairs_duckdb,
        spatial_available,
        workspace_path,
    )

    engine = "duckdb" if args.workspace else args.engine
    if engine == "auto":
        engine = "duckdb" if len(obs) >= PAIRS_ENGINE_THRESHOLD and spatial_available() else "python"
    if engine == "python":
        return find_stereo_pairs(obs, **kwargs)
    try:
        if args.workspace:
            with Workspace(args.workspace) as ws:
                res = find_stereo_pairs_duckdb(obs, workspace=ws, **kwargs)
            print(f"Wrote images and pairs to {workspace_path(args.workspace)}")
            return res
        return find_stereo_pairs_duckdb(obs, **kwargs)
    except WorkspaceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
