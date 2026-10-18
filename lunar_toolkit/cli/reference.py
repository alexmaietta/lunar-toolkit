"""reference - Artemis III candidate regions, PGDA LOLA 5 m site DEMs and Bertone et al. (2026) SfS DEMs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMAND = "reference"


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Reference data for the Artemis III regions: crop windows, PGDA LOLA 5 m DEMs, Bertone SfS DEMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  lunar reference list
  lunar reference show malapert                 # bbox, ROI projwin, PGDA site, Zenodo zip
  lunar reference crop 11 --crop-km 2.5         # 2.5 km window at the ROI centre
  lunar reference crop malapert --center -85.98 -0.26 --format wkt
  lunar reference pgda malapert --kind surf toterr -o ref
  lunar reference pgda malapert --kind clones --clones 10 -o ref
  lunar reference pgda malapert --kind xyzi --csv --crop-km 2.5 -o ref
  lunar reference sdem malapert --list           # zip members (remote central directory)
  lunar reference sdem malapert --remote --member GLDELEV GLDSIGM -o ref
  lunar reference sdem malapert --download -o ref --member GLDELEV

REGION is an id (0-12), name, slug, unique part of a name or PGDA site
(11, "Malapert Massif", malapert-massif, malapert, Site23).  The ROI is the
polar stereographic square the SfS DEM covers (= the PGDA site extent);
projwin is ULX ULY LRX LRY in metres (+proj=stere +lat_0=-90 +R=1737400).
Existing files are skipped (--force to redo); downloads resume.
""",
    )
    sp = p.add_subparsers(dest="ref_cmd", required=True)

    sp.add_parser("list", help="List the 13 regions.").add_argument(
        "--json", action="store_true", help="Print JSON instead of a table."
    )

    show = sp.add_parser("show", help="Show one region.")
    show.add_argument("region", metavar="REGION")
    show.add_argument("--crop-km", type=float, default=None, metavar="KM", help="Also print a crop window.")
    show.add_argument("--json", action="store_true", help="Print JSON.")

    crop = sp.add_parser("crop", help="Print a square crop window centred on a region.")
    crop.add_argument("region", metavar="REGION")
    crop.add_argument("--crop-km", type=float, default=2.5, metavar="KM", help="Side length (default: 2.5).")
    crop.add_argument(
        "--center", nargs=2, type=float, metavar=("LAT", "LON"), help="Centre (default: ROI centre)."
    )
    crop.add_argument(
        "--format",
        choices=("projwin", "bounds", "wkt", "geojson"),
        default="projwin",
        help="projwin (ULX ULY LRX LRY), bounds (XMIN YMIN XMAX YMAX), lon/lat WKT or GeoJSON.",
    )

    pg = sp.add_parser("pgda", help="Download a region's PGDA LOLA 5 m site files.")
    pg.add_argument("region", metavar="REGION")
    pg.add_argument(
        "--kind",
        nargs="+",
        default=["surf"],
        metavar="KIND",
        help="surf ldec toterr slp slperr xyzi clones rois (default: surf).",
    )
    pg.add_argument("--clones", type=int, default=None, metavar="N", help="Only the first N clones.")
    pg.add_argument("--csv", action="store_true", help="Also write each .xyzi as an ASP CSV.")
    pg.add_argument("--crop-km", type=float, default=None, metavar="KM", help="Crop the --csv points.")
    pg.add_argument("--list", action="store_true", help="Print the URLs only.")
    pg.add_argument("--refresh", action="store_true", help="Re-read the PGDA product page.")
    _common(pg)

    sd = sp.add_parser("sdem", help="List / download / extract a region's Bertone et al. Zenodo zip.")
    sd.add_argument("region", metavar="REGION")
    sd.add_argument(
        "--member",
        nargs="+",
        default=["GLDELEV"],
        metavar="PAT",
        help="Members to extract: product code / substring or glob (default: GLDELEV).",
    )
    mode = sd.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List zip members (reads only the directory).")
    mode.add_argument("--remote", action="store_true", help="Extract members remotely (no full zip).")
    mode.add_argument(
        "--download", action="store_true", help="Download the whole zip (1-4 GB), then extract."
    )
    sd.add_argument("--zip", type=Path, default=None, metavar="ZIP", help="Extract from this local zip.")
    _common(sd)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--output-dir", "-o", type=Path, default=Path("."), metavar="DIR", help="Destination (default: .)."
    )
    p.add_argument("--force", action="store_true", help="Download / extract again even if present.")


def _region_info(reg, crop_km: float | None = None) -> dict:
    from .. import reference as ref

    roi = reg.roi
    info = {
        "id": reg.id,
        "name": reg.name,
        "slug": reg.slug,
        "center_lonlat": reg.center,
        "bbox": reg.bbox,
        "roi_projwin": roi.projwin,
        "roi_size_km": (roi.width / 1000.0, roi.height / 1000.0),
        "bbox_projwin": tuple(round(v, 1) for v in reg.bbox_projwin),
        "pgda_site": reg.pgda_site,
        "pgda_url": ref.PGDA_SITES[reg.pgda_site].url if reg.pgda_site else None,
        "zenodo_file": reg.zenodo_file,
        "zenodo_size_gb": round(reg.zenodo_size / 1e9, 2),
        "zenodo_md5": reg.zenodo_md5,
        "zenodo_url": reg.zenodo_url,
    }
    if crop_km:
        info["crop_projwin"] = ref.crop_window(reg, crop_km).projwin
    return info


def run(args: argparse.Namespace) -> int:
    from .. import reference as ref

    try:
        if args.ref_cmd == "list":
            return _list(args)
        reg = ref.get_region(args.region)
        if args.ref_cmd == "show":
            info = _region_info(reg, args.crop_km)
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                for k, v in info.items():
                    print(f"{k:15s} {' '.join(f'{x:g}' for x in v) if isinstance(v, tuple) else v}")
            return 0
        if args.ref_cmd == "crop":
            return _crop(args, reg)
        if args.ref_cmd == "pgda":
            return _pgda(args, reg)
        return _sdem(args, reg)
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1


def _list(args: argparse.Namespace) -> int:
    from .. import reference as ref

    if args.json:
        print(json.dumps([_region_info(r) for r in ref.REGIONS], indent=2))
        return 0
    print(f"{'id':>2}  {'name':27s} {'pgda':9s} {'zip GB':>6}  roi projwin (ULX ULY LRX LRY, m)")
    for r in ref.REGIONS:
        pw = " ".join(f"{v:.0f}" for v in r.projwin)
        print(f"{r.id:>2}  {r.name:27s} {r.pgda_site or '-':9s} {r.zenodo_size / 1e9:6.2f}  {pw}")
    return 0


def _crop(args: argparse.Namespace, reg) -> int:
    from .. import reference as ref

    box = ref.crop_window(reg, args.crop_km, tuple(args.center) if args.center else None)
    if args.format == "projwin":
        print(" ".join(f"{v:.3f}".rstrip("0").rstrip(".") for v in box.projwin))
    elif args.format == "bounds":
        print(" ".join(f"{v:.3f}".rstrip("0").rstrip(".") for v in box.bounds))
    elif args.format == "wkt":
        print(box.wkt())
    else:
        print(json.dumps({"type": "Polygon", "coordinates": [box.ring_lonlat()]}))
    return 0


def _pgda(args: argparse.Namespace, reg) -> int:
    from .. import reference as ref

    index = ref.pgda_index(refresh=args.refresh)
    if args.list:
        for kind, urls in ref.pgda_urls(reg, args.kind, args.clones, index).items():
            for u in urls:
                print(f"{kind:7s} {u}")
        return 0
    paths = ref.download_pgda(reg, args.kind, args.output_dir, args.clones, args.force, index=index)
    for kind, files in paths.items():
        for f in files:
            print(f"{kind:7s} {f}")
    if args.csv:
        box = ref.crop_window(reg, args.crop_km) if args.crop_km else None
        for f in paths.get("xyzi", []):
            out = f.with_suffix(f"{'_crop' if box else ''}.csv")
            ref.write_asp_csv(f, out, box)
            print(f"csv     {out}  (--csv-format '{ref.ASP_CSV_FORMAT}' --csv-srs '{ref.PS_SRS}')")
    return 0


def _sdem(args: argparse.Namespace, reg) -> int:
    from .. import reference as ref

    if args.list:
        members = ref.sdem_members(args.zip or reg)
        for row in members.itertuples():
            print(f"{row.size:>12,}  {row.name}")
        print(f"{len(members)} members, {members['size'].sum() / 1e9:.2f} GB uncompressed")
        return 0
    if args.remote:
        source = reg
    elif args.zip:
        source = args.zip
    elif args.download:
        source = ref.download_sdem(reg, args.output_dir, force=args.force)
    else:
        local = args.output_dir / reg.zenodo_file
        if not local.exists():
            print(
                f"{local} not found: pass --download (whole zip, {reg.zenodo_size / 1e9:.1f} GB), "
                "--remote (selected members only) or --zip",
                file=sys.stderr,
            )
            return 1
        source = local
    out_dir = args.output_dir / reg.zenodo_file.removesuffix(".zip")
    for path in ref.extract_sdem(source, args.member, out_dir, force=args.force):
        print(path)
    return 0
