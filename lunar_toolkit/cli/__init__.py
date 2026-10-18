"""``lunar`` CLI; each subcommand module exports COMMAND, add_parser(sub) and run(args)."""

import argparse
import logging
import sys

from . import (
    app,
    build_db,
    download,
    kaguya,
    ode_search,
    pairs,
    process,
    query,
    reference,
    search,
    serve,
    sfs,
    sfs_report,
    shadowcam,
    stereo,
    workspace,
)

_COMMANDS = [
    build_db,
    download,
    query,
    search,
    ode_search,
    pairs,
    reference,
    process,
    kaguya,
    shadowcam,
    stereo,
    sfs,
    sfs_report,
    workspace,
    serve,
    app,
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunar",
        description=(
            "Lunar imagery toolkit - build LROC index databases, search footprints, "
            "query ODE (ShadowCam, LROC, Kaguya TC and MI), download and import images into ISIS, "
            "build stereo and shape-from-shading DEMs with ISIS + ASP, and fetch Artemis III "
            "reference data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in _COMMANDS:
        cmd.add_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    for cmd in _COMMANDS:
        if args.cmd == cmd.COMMAND:
            return cmd.run(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
