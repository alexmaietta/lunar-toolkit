"""sfs - shape-from-shading DEM from LROC NAC images (ISIS + ASP)."""

import argparse
import sys
from pathlib import Path

from ..constants import CAMERA_SOURCES, REGISTRATION_MODES, SELECTION_METHODS, SFS_CAMERA_MODELS

COMMAND = "sfs"

_OPTION_FLAGS = {"ba_opt": "bundle_adjust_options", "refine_opt": "refine_options", "sfs_opt": "sfs_options"}


def _reduce(value: str):
    low = value.lower()
    if low == "auto":
        return "auto"
    if low == "none":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, auto or none, got {value!r}") from None


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        COMMAND,
        help="Build a shape-from-shading DEM from NAC images and a prior DEM (ISIS + ASP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
        epilog="""
Examples
--------
  lunar sfs --config examples/malapert_crop.yaml
  lunar sfs --config run.yaml --res 2 --sfs-opt smoothness_weight=0.05   # flags override the file
  lunar sfs --config run.yaml --dump-config > effective.yaml             # print every setting
  lunar sfs --config run.yaml --check          # compare tools / config / inputs with the work dir
  lunar sfs --region malapert --crop-km 2.5 --work-dir malapert_crop   # PGDA prior DEM, ROI search
  lunar sfs --aoi site.geojson --dem LDEM_87S_5MPP_ADJ.TIF --work-dir malapert
  lunar sfs M1234567890LE M1234567891RE ... --dem LDEM_87S_5MPP_ADJ.TIF \\
      --projwin -10000 50000 0 40000 --res 5 --work-dir malapert
  lunar sfs raw/*.IMG --dem ldem.tif --registration residuals --processes 8
  lunar sfs --aoi "POLYGON((...))" --dem ldem.tif --dry-run   # print the commands only

Config files are flat YAML mappings of the same settings (keys as printed by
--dump-config); command-line flags override them.  Each run writes the effective
settings to WORK_DIR/config.yaml and appends its environment, tool versions,
input checksums and per-step commands to WORK_DIR/provenance.json.

Steps (Bertone et al. 2026, PSJ 7:118): [search + slew/resolution cut + download] ->
calibrate + CSM cameras -> crop/resample prior DEM -> sort by Sun azimuth -> mapproject at
--ba-res -> bundle adjust -> drop badly registered images -> register to the prior DEM
(stereo DEMs or BA points, translation only) -> refine with --heights-from-dem -> drop
again -> mapproject at --res -> max-lit QA mosaics (all + 30 deg Sun bins) -> select images ->
reduce cubes to --res -> parallel_sfs -> products.
Re-running resumes: steps whose outputs exist are skipped (--force to redo).
Needs ISIS and ASP on PATH (or ASP_ROOT / ISISROOT).
""",
    )
    p.add_argument(
        "sources",
        nargs="*",
        default=[],
        metavar="SOURCE",
        help="NAC EDR product IDs (downloaded), .IMG or calibrated .cub paths "
        "(default with --aoi / --region: search the local EDR index).",
    )
    p.add_argument("--config", metavar="YAML", help="Run configuration file (flags override it).")
    p.add_argument(
        "--dump-config", action="store_true", help="Print the effective configuration as YAML and exit."
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Compare tool versions, settings and input files with WORK_DIR/provenance.json and exit "
        "(status 1 when they differ).",
    )
    p.add_argument("--from-file", metavar="FILE", help="Text file of sources, one per line.")
    p.add_argument(
        "--dem", metavar="TIF", help="Prior DEM (e.g. LOLA LDEM GeoTIFF), or pgda (default with --region)."
    )
    p.add_argument(
        "--region",
        metavar="REGION",
        help="Artemis III region (id, name or PGDA site, see lunar reference list): sets the crop, "
        "product prefix and image search, and the PGDA LOLA 5 m prior DEM.",
    )
    p.add_argument(
        "--crop-km", type=float, metavar="KM", help="Square crop of this side at the region centre."
    )
    window = p.add_mutually_exclusive_group()
    window.add_argument(
        "--aoi",
        metavar="FILE_OR_WKT",
        help="Area of interest (GeoJSON / WKT file or WKT, lon/lat): crops a polar stereographic "
        "prior DEM and, without SOURCEs, selects the images.",
    )
    window.add_argument(
        "--projwin",
        nargs=4,
        type=float,
        metavar=("ULX", "ULY", "LRX", "LRY"),
        help="Crop window in the prior DEM's projected units.",
    )
    p.add_argument("--work-dir", metavar="DIR", help="Working / output directory (default: sfs_work).")
    p.add_argument("--res", dest="resolution", type=float, metavar="M", help="SfS DEM m/px (default: 5).")
    p.add_argument("--preset", choices=["bertone", "asp"], help="Option defaults (default: bertone).")
    p.add_argument(
        "--ba-res",
        dest="ba_resolution",
        type=float,
        metavar="M",
        help="m/px of the images mapprojected for bundle adjustment (default: 1, NAC native).",
    )
    p.add_argument(
        "--reduce",
        type=_reduce,
        metavar="N|auto|none",
        help="ISIS reduce factor of the SfS cubes (default: auto = target / native m/px).",
    )
    p.add_argument("--no-csm", dest="csm", action="store_false", help="Use ISIS cameras instead of CSM.")
    p.add_argument(
        "--local-spice",
        dest="web_spice",
        action="store_false",
        help="spiceinit with local kernels instead of the SPICE web service.",
    )
    p.add_argument(
        "--registration",
        choices=list(REGISTRATION_MODES),
        help="Register cameras to the prior DEM from stereo DEMs, BA points or not at all (default: stereo).",
    )
    p.add_argument("--stereo-pairs", type=int, metavar="N", help="Max stereo pairs (default: 30).")
    p.add_argument(
        "--max-displacement",
        type=float,
        metavar="M",
        help="pc_align --max-displacement for registration (default: 400).",
    )
    p.add_argument(
        "--no-refine", dest="refine", action="store_false", help="Skip the --heights-from-dem refinement."
    )
    p.add_argument(
        "--max-offset-p85",
        type=float,
        metavar="M",
        help="Drop images whose 85th-percentile mapprojected match offset exceeds this (default: 1.8).",
    )
    p.add_argument(
        "--selection",
        choices=list(SELECTION_METHODS),
        help="SfS image selection (default: coverage = 3 Sun directions 30 deg apart per pixel; "
        "kmedoids = Bertone et al. k-medoids seeding + coverage refinement).",
    )
    p.add_argument(
        "--selection-clusters", type=int, metavar="N", help="Medoids for --selection kmedoids (default: 30)."
    )
    p.add_argument(
        "--sfs-cameras",
        choices=list(SFS_CAMERA_MODELS),
        help="Cameras of the reduced SfS cubes: re-mapped CSM (default) or ISIS approximate models.",
    )
    p.add_argument(
        "--cameras-from",
        choices=list(CAMERA_SOURCES),
        help="ba (default) or published: Bertone et al. adjusted cameras, skipping bundle adjustment.",
    )
    p.add_argument("--max-sfs-images", type=int, metavar="N", help="Cap on SfS images.")
    p.add_argument(
        "--max-slew",
        type=float,
        metavar="DEG",
        help="Search cut on |slew| (default: none; see --max-sfs-slew).",
    )
    p.add_argument(
        "--max-sfs-slew",
        type=float,
        metavar="DEG",
        help="Only images with |slew| below this go into SfS (default: 5; slewed images still help BA / stereo).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        metavar="N",
        help="Cap on searched images before download: stereo pairs + illumination coverage (default: none).",
    )
    p.add_argument(
        "--max-pixel-scale", type=float, metavar="M", help="Search cut on the image m/px (default: 1.5)."
    )
    p.add_argument(
        "--ensemble",
        dest="ensemble_members",
        type=int,
        metavar="N",
        help="SfS ensemble members (default: 0).",
    )
    p.add_argument("--seed", type=int, metavar="N", help="Random seed of the ensemble subsets (default: 0).")
    p.add_argument(
        "--region-id", metavar="ID", help="Product name prefix (default: region or work dir name)."
    )
    p.add_argument("--processes", type=int, metavar="N")
    p.add_argument("--threads", type=int, metavar="N")
    for flag, tool in (
        ("--ba-opt", "bundle_adjust"),
        ("--refine-opt", "refinement bundle_adjust"),
        ("--sfs-opt", "parallel_sfs"),
    ):
        p.add_argument(
            flag,
            action="append",
            metavar="KEY=VALUE",
            help=f"Extra / overriding {tool} option (repeatable; VALUE none drops it).",
        )
    p.add_argument("--force", action="store_true", help="Recompute steps whose outputs exist.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    p.add_argument("--data-dir", type=Path, metavar="DIR", help="Override Parquet data directory.")


def _overrides(args: argparse.Namespace) -> dict:
    """Config keys given on the command line."""
    from ..config import KEYS
    from ._utils import parse_options

    given = vars(args)
    out = {k: v for k, v in given.items() if k in KEYS and k != "sources"}
    for flag, key in _OPTION_FLAGS.items():
        if flag in given:
            out[key] = parse_options(given[flag])
    if args.sources or "from_file" in given:
        out["sources"] = list(args.sources)
        out.setdefault("from_file", None)
    return out


def _effective(args: argparse.Namespace):
    from ..config import RunConfig, load_config, merge

    cfg = load_config(Path(args.config)) if getattr(args, "config", None) else RunConfig()
    return merge(cfg, _overrides(args), "command line")


def run(args: argparse.Namespace) -> int:
    from .. import config as cfgmod
    from ..sfs import PipelineError, SfsPipeline, aoi_projwin, search_images
    from ..tools import ToolError
    from ._utils import print_dry_run, resolve_images

    dry = getattr(args, "dry_run", False)
    try:
        cfg = _effective(args)
        if getattr(args, "dump_config", False):
            print(cfgmod.dump_config(cfgmod.resolve(cfg, download=False)), end="")
            return 0
        if getattr(args, "check", False):
            return _check(cfgmod.resolve(cfg, download=False), args)
        cfg = cfgmod.resolve(cfg, download=not dry)
        sources = cfgmod.load_sources(cfg)
    except (cfgmod.ConfigError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if cfg.dem is None:
        print("No prior DEM: pass --dem, or --region for the PGDA LOLA DEM.", file=sys.stderr)
        return 1
    dem = Path(cfg.dem)
    aoi = cfgmod.search_aoi(cfg)
    observations = None
    try:
        if cfg.aoi and cfg.sfs.projwin is None:
            projwin = aoi_projwin(cfg.aoi, dem, snap=cfg.sfs.resolution)
            print(f"AOI -> projwin {' '.join(f'{v:g}' for v in projwin)}")
            cfg = cfgmod.merge(cfg, {"projwin": projwin})
        if not sources and aoi:
            observations = search_images(
                aoi,
                max_slew=cfg.max_slew,
                max_resolution=cfg.max_pixel_scale,
                max_images=cfg.max_images,
                config=cfg.sfs,
                data_dir=cfg.data_dir,
            )
            sources = observations["product_id"].astype(str).tolist()
            print(f"{len(sources)} NAC images selected over the AOI")
            cfg = cfgmod.merge(cfg, {"sources": sources})
    except (ToolError, ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if not sources:
        print(
            "No sources. Pass product IDs, paths, --from-file, or --aoi / --region to search.",
            file=sys.stderr,
        )
        return 1

    work_dir = Path(cfg.work_dir)
    images = resolve_images(sources, work_dir / "raw", cfg.data_dir, download=not dry)
    pipeline = SfsPipeline(
        work_dir,
        dem,
        cfg.sfs,
        force=getattr(args, "force", False),
        observations=observations,
        aoi=aoi,
        data_dir=cfg.data_dir,
        run_config=cfg,
    )

    try:
        if dry:
            print_dry_run(lambda: pipeline.run(images))
            return 0
        out = pipeline.run(images)
    except (ToolError, PipelineError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"SfS DEM -> {out}")
    if pipeline.removed:
        print(f"{len(pipeline.removed)} images removed (see {work_dir / 'selection' / 'removed.csv'})")
    print(f"{len(pipeline.selected)} SfS images, {len(pipeline.unselected)} unselected")
    return 0


def _check(cfg, args: argparse.Namespace) -> int:
    from .. import config as cfgmod
    from .. import provenance
    from ._utils import resolve_images

    work_dir = Path(cfg.work_dir)
    sources = cfgmod.load_sources(cfg)
    inputs = None
    if sources and cfg.dem:
        inputs = [*resolve_images(sources, work_dir / "raw", cfg.data_dir, download=False), Path(cfg.dem)]
    try:
        diffs = provenance.check(work_dir, cfg, inputs, checksums=cfg.checksums)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not diffs:
        print(f"No differences from the last run in {work_dir / provenance.FILE_NAME}")
        return 0
    print(f"{len(diffs)} differences from the last run in {work_dir / provenance.FILE_NAME}:")
    for d in diffs:
        print(f"  {d}")
    return 1
