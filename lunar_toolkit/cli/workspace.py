"""workspace - ad-hoc SQL on a work dir's DuckDB workspace (lunar.duckdb)."""

import argparse
import sys
from pathlib import Path

COMMAND = "workspace"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Query a work dir's DuckDB workspace (images, pairs, bundle-adjustment tables and views).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar workspace work/malapert                      # tables and views with row counts
  lunar workspace work/malapert --sql "SELECT * FROM ba_run_summary"
  lunar workspace work/malapert --sql "SELECT run, rule, count(*) FROM image_rejections GROUP BY ALL"
  lunar workspace work/malapert --sql "SELECT * FROM image_rejections_at(1.5, 0.8, 1.0, 1)"
  lunar workspace work/malapert --export ba_image_stats ba_stats.parquet

WORK_DIR is a lunar sfs work directory (its lunar.duckdb) or a .duckdb file; it is
opened read-only.  Tables: images, pairs, ba_runs, ba_residual_stats,
ba_pixel_residuals, ba_match_offsets, ba_points, ba_convergence, ba_camera_offsets,
rejection_params.  Views: ba_image_stats, ba_run_summary, image_rejections.
--export writes Parquet for .parquet paths, else CSV (geometries as WKT).
""",
    )
    p.add_argument("work_dir", type=Path, metavar="WORK_DIR", help="Work directory or .duckdb file.")
    p.add_argument("--sql", action="append", metavar="SQL", help="Run SQL and print the result (repeatable).")
    p.add_argument("--tables", action="store_true", help="List tables and views with row counts.")
    p.add_argument(
        "--export", nargs=2, metavar=("TABLE", "PATH"), help="Write a table or view to Parquet / CSV."
    )
    p.add_argument("--csv", metavar="FILE", help="Write the last --sql result to CSV.")
    p.add_argument("--max-rows", type=int, default=50, metavar="N", help="Rows to print (default: 50).")
    p.add_argument(
        "--memory-limit", default="1GB", metavar="SIZE", help="DuckDB memory_limit (default: 1GB)."
    )
    p.add_argument("--threads", type=int, metavar="N", help="DuckDB threads (default: all cores).")


def run(args: argparse.Namespace) -> int:
    import duckdb

    from ..workspace import Workspace, WorkspaceError, workspace_path

    path = workspace_path(args.work_dir)
    if not path.exists():
        print(f"Error: no workspace at {path}", file=sys.stderr)
        return 1
    try:
        ws = Workspace(path, read_only=True, memory_limit=args.memory_limit, threads=args.threads)
    except duckdb.Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    with ws:
        try:
            if args.tables or not (args.sql or args.export):
                print(ws.tables().to_string(index=False))
            res = None
            for sql in args.sql or []:
                res = ws.query(sql)
                geom = [
                    c for c in res.columns if res[c].map(lambda v: isinstance(v, bytes | bytearray)).any()
                ]
                shown = res.drop(columns=geom)
                if len(shown.columns):
                    with_more = len(res) > args.max_rows
                    print(shown.head(args.max_rows).to_string(index=False))
                    if with_more:
                        print(f"... {len(res)} rows")
            if args.csv and res is not None:
                res.to_csv(args.csv, index=False)
                print(f"Wrote {args.csv}")
            if args.export:
                out = ws.export(args.export[0], args.export[1])
                print(f"Wrote {out}")
        except (duckdb.Error, WorkspaceError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    return 0
