"""query - print metadata for LROC NAC images from the Parquet index."""

import argparse
import sys
from pathlib import Path

COMMAND = "query"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Print metadata for LROC NAC images from the Parquet index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar query M1096757719LE
  lunar query M1096757719LE M1096757719RE
  lunar query --from-file ids.txt
  lunar query /path/to/M1096757719LE.IMG
""",
    )
    p.add_argument(
        "sources",
        nargs="*",
        metavar="SOURCE",
        help="Product IDs, local .IMG paths, or a text file of IDs.",
    )
    p.add_argument(
        "--from-file",
        metavar="FILE",
        help="Text file of product IDs or paths, one per line.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override Parquet data directory.",
    )


def run(args: argparse.Namespace) -> int:
    from ..lroc.nac import NacImageCollection
    from ._utils import extract_product_id, load_sources

    sources = load_sources(args)
    if not sources:
        print("No sources provided. Pass product IDs or --from-file.", file=sys.stderr)
        return 1

    product_ids = [extract_product_id(s) for s in sources]

    collection = NacImageCollection.from_sources(product_ids, data_dir=args.data_dir)
    found = {img.product_id: img for img in collection}

    for pid in product_ids:
        img = found.get(pid)
        print(pid)
        if img is None:
            print("  (not found in index)")
        else:
            items = img.metadata_items()
            width = max((len(label) for label, _ in items), default=30) + 2
            for label, value in items:
                print(f"  {label:<{width}} {value}")
        print()

    return 0
