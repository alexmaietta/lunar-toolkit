"""download - download LROC image files by product ID, text file, or URL."""

import argparse
import sys
from pathlib import Path

COMMAND = "download"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Download LROC image files to a local directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar download M1096757719LE M1096757719RE
  lunar download M1096757719LE --output-dir ./raw
  lunar download --from-file ids.txt --output-dir ./raw
  lunar download https://pds.lroc.im-ldi.com/data/.../M1096757719LE.IMG
  lunar download M1096757719LC --dataset cdr  # CDR image

Files already in --output-dir are skipped (use --force to download them again);
interrupted downloads resume from their .part file.
""",
    )
    p.add_argument(
        "sources",
        nargs="*",
        metavar="SOURCE",
        help="Product IDs (e.g. M1096757719LE) or PDS URLs.",
    )
    p.add_argument(
        "--from-file",
        metavar="FILE",
        help="Text file of product IDs or URLs, one per line (#comments ignored).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Destination directory (default: current working directory).",
    )
    p.add_argument(
        "--dataset",
        choices=["edr", "cdr"],
        default="edr",
        help="Download each image's raw EDR or calibrated CDR file (default: edr); "
        "IDs are looked up in the index matching their LE/RE or LC/RC suffix.",
    )
    p.add_argument("--force", action="store_true", help="Download again even if the file exists.")
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Parallel download workers (default: 8).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override Parquet data directory.",
    )


def run(args: argparse.Namespace) -> int:
    from ..downloader import PdsDownloader
    from ..lroc.nac import NacImageCollection
    from ._utils import load_sources

    sources = load_sources(args)
    if not sources:
        print("No sources provided. Pass product IDs, URLs, or --from-file.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_urls = [s.strip() for s in sources if s.strip().startswith(("http://", "https://"))]
    id_sources = [s for s in sources if not s.strip().startswith(("http://", "https://"))]

    ok = fail = 0

    if id_sources:
        collection = NacImageCollection.from_sources(id_sources, data_dir=args.data_dir)
        if not collection:
            print("No product IDs found in the index.", file=sys.stderr)
        else:
            if args.dataset == "cdr":
                results = collection.download_cdrs(output_dir, workers=args.workers, force=args.force)
            else:
                results = collection.download_edrs(output_dir, workers=args.workers, force=args.force)
            ok += sum(1 for p in results.values() if p is not None)
            fail += sum(1 for p in results.values() if p is None)

    if raw_urls:
        print(f"Downloading {len(raw_urls)} URL(s) to {output_dir} ...")
        dl = PdsDownloader(max_workers=args.workers)
        url_results = dl.download_urls(raw_urls, output_dir, force=args.force)
        ok += sum(1 for p in url_results.values() if p is not None)
        fail += sum(1 for p in url_results.values() if p is None)

    if ok == 0 and fail == 0:
        print("Nothing to download.", file=sys.stderr)
        return 1

    print(f"Done: {ok} succeeded{f', {fail} failed' if fail else ''}.")
    return 0 if fail == 0 else 1
