"""build-db - download CUMINDEX files and build local Parquet databases."""

import argparse
import sys
from pathlib import Path

from ..constants import DATASETS, DEFAULT_CACHE_DIR

COMMAND = "build-db"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Build the local Parquet index database(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar build-db              # EDR only (default)
  lunar build-db edr cdr      # EDR and CDR
  lunar build-db --all        # EDR, CDR, and RDR
  lunar build-db edr --force  # rebuild even if exists
  lunar build-db edr --update # append PDS volumes newer than the built release
  lunar build-db edr --release LROLRC_0066C --volume-only   # one volume (~20k rows)
""",
    )
    p.add_argument(
        "datasets",
        nargs="*",
        metavar="DATASET",
        help="Datasets to build: edr, cdr, rdr (default: edr if none given).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Build all three datasets: edr, cdr, rdr.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the Parquet already exists.",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Append newer per-volume INDEX.TABs to an existing database.",
    )
    p.add_argument(
        "--release",
        default=None,
        metavar="VOLUME",
        help="Build from this PDS volume, e.g. LROLRC_0066C (default: newest on the server).",
    )
    p.add_argument(
        "--volume-only",
        action="store_true",
        help="Use the volume's own INDEX/INDEX.TAB instead of the cumulative CUMINDEX.TAB.",
    )
    p.add_argument(
        "--wkt",
        action="store_true",
        help="Also store an FP_WKT footprint polygon column.",
    )
    p.add_argument(
        "--memory-limit",
        default="4GB",
        metavar="SIZE",
        help="DuckDB memory limit; sorting spills to disk beyond it (default: 4GB).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="DuckDB worker threads (default: all cores).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Where to write Parquet files (default: LUNAR_DATA_DIR or the user data dir).",
    )
    p.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        metavar="DIR",
        help=f"Where to cache raw TAB/LBL downloads (default: {DEFAULT_CACHE_DIR}).",
    )


def run(args: argparse.Namespace) -> int:
    from ..lroc.cumindex import CumIndex

    if args.all:
        datasets = list(DATASETS)
    elif args.datasets:
        invalid = [d for d in args.datasets if d not in DATASETS]
        if invalid:
            print(
                f"Unknown dataset(s): {', '.join(invalid)}. Choose from: {', '.join(DATASETS)}",
                file=sys.stderr,
            )
            return 1
        datasets = args.datasets
    else:
        datasets = ["edr"]

    cache_dir = Path(args.cache_dir)

    for dataset in datasets:
        ds = DATASETS[dataset]
        print(f"\n{ds['label']}")
        ci = CumIndex(dataset=dataset, data_dir=args.data_dir, cache_dir=cache_dir)
        print(f"   Parquet : {ci.parquet_path}")
        print(f"   Cache   : {cache_dir}")
        if args.update and ci.parquet_path.exists() and not args.force:
            ci.update_db(memory_limit=args.memory_limit, threads=args.threads)
            continue
        ci.build_db(
            force=args.force,
            release=args.release,
            volume_only=args.volume_only,
            memory_limit=args.memory_limit,
            threads=args.threads,
            wkt=args.wkt,
        )

    return 0
