# DEMs with ISIS + ASP

`lunar_toolkit` wraps the ISIS and Ames Stereo Pipeline (ASP) tools used to build lunar DEMs, and chains them into two pipelines: a stereo DEM from one NAC pair, and a shape-from-shading (SfS) DEM from many NAC images.

## Setup

Install the Ames Stereo Pipeline by following its [installation guide](https://stereopipeline.readthedocs.io/en/latest/installation.html). ASP ships its own ISIS, so a separate [ISIS install](https://astrogeology.usgs.gov/docs/how-to-guides/environment-setup-and-maintenance/installing-isis-via-anaconda/) is only needed without ASP. Then set up the ISIS data area for LRO (base and lro, see the [ISIS data area docs](https://astrogeology.usgs.gov/docs/how-to-guides/environment-setup-and-maintenance/isis-data-area/)) and set `ISISDATA`. The [Docker image](install.md#docker) has all of this ready.

`spiceinit` uses the USGS SPICE web service by default (`web=True`), so the large LRO SPK and CK kernels are not needed, and `--local-spice` / `web_spice=False` uses local kernels. Check the camera stack with `cam_test` or `ASP_ROOT=... pytest -m tools`.

Tools are looked up in `$ASP_ROOT/bin`, then `$ISISROOT/bin`, then `PATH`. The env does not need to be activated: its `conda env config vars` (`conda-meta/state`) are read, `ISISROOT`, `PROJ_DATA` and `GDAL_DATA` default to the tool's env prefix, and its `bin/` is put first on `PATH` for ASP's Python wrapper scripts.

```bash
export ASP_ROOT=~/miniforge3/envs/isis
```

## Wrappers

Positional inputs are arguments; every other option is a keyword argument (`max_iterations=5` -> `--max-iterations 5`, `save_sparingly=True` -> `--save-sparingly`, `False` / `None` dropped, tuples expand to several values, `t_*` GDAL-style options keep their underscores). ISIS programs take `key=value` parameters (`from_` for `from`).

```python
from lunar_toolkit import asp, isis

cube = isis.calibrate_lronac("raw/M1234567890LE.IMG", "work")  # .cal.echo.cub
cam = isis.isd_generate(cube)  # CSM camera (.json)
asp.mapproject("dem.tif", cube, "work/M1234567890LE.map.tif", cam, tr=5)
asp.bundle_adjust([cube], "ba/run", [cam], mapprojected=[...], dem="dem.tif", parallel=True)
asp.sfs("dem.tif", [cube], "sfs/run", [cam], parallel=True, smoothness_weight=0.08)
angles = asp.sun_angles("dem.tif", [cube], [cam])  # sfs --query
```

Any other ISIS program runs through `isis.isis("program", **params)` and any tool through `tools.run(tool, args)`.

### Dry runs

Inside `tools.dry_run()` commands are recorded instead of executed, so a whole pipeline can be inspected first:

```python
import shlex
from lunar_toolkit.tools import dry_run

with dry_run() as cmds:
    pipeline.run(images)
print("\n".join(shlex.join(c) for c in cmds))
```

The CLI equivalent is `--dry-run` on `lunar stereo`, `lunar sfs` and `lunar kaguya`.

## Stereo DEM

```bash
lunar stereo M181058717LE M181073012LE --res 2 --work-dir pair1
lunar stereo L.IMG R.IMG --dem ldem.tif --res 2 --image-res 1
```

Calibrate -> CSM cameras -> `bundle_adjust` -> `parallel_stereo` (`asp_mgm`, subpixel mode 9) -> `point2dem`. With a prior DEM the images are mapprojected onto it first (better on steep terrain) and the result is aligned to it with `pc_align`. Use `lunar pairs` to choose the pair.

## Shape-from-shading DEM

`SfsPipeline` reproduces the workflow of Bertone et al. (2026, PSJ 7:118) behind the 5 m/px SfS DEMs of the 13 Artemis III candidate regions (`preset="bertone"`, the default); `preset="asp"` gives ASP's large-area LRO NAC recipe.

| Step | Bertone et al. (2026) | Config / CLI |
|---|---|---|
| Image search | < 1.5 m/px, NAC-L and NAC-R separate; slewed images kept for BA / stereo; capped before download | `--aoi`, `--max-pixel-scale`, `--max-images` |
| SfS images | nadir (slew < 5 deg) | `max_sfs_slew`, `--max-sfs-slew` |
| Cameras | CSM | `csm`, `--no-csm` |
| Mapproject for BA | native ~1 m/px onto the LDEM | `ba_resolution`, `--ba-res` |
| Bundle adjust | next 200 images in Sun order, 100 m camera position uncertainty, intrinsics fixed | `PRESETS["bertone"]["bundle_adjust"]`, `--ba-opt` |
| Image rejection | match-offset p85 > 1.8 m (on the registered, refined cameras, as ASP's `ba_align_ref`), no matches, > 85 % of residuals > 1 px; stops if fewer than `min_images` remain | `max_offset_p85`, `max_large_residual_fraction`, `min_matches`, `min_images` |
| QA | max-lit mosaics, all + 30 deg bins | `qa/` |
| Registration | stereo DEMs from up to 30 pairs (convergence >= 5 deg), translation to the LDEM | `registration`, `--stereo-pairs` |
| Refinement | quick re-alignment | `refine`, `--refine-opt` |
| SfS selection | k-medoids seeding, then < 0.5 % uncovered and < 1 % with < 3 directions >= 30 deg apart; drop >= 95 % shadowed | `selection="kmedoids"` (or `"coverage"`), `selection_clusters`, `selection_views`, `max_shadow_fraction` |
| Camera downsampling | reduce to the target resolution, CSM cameras patched to match | `reduce="auto"`, `sfs_cameras="csm"` (`lunar_toolkit.csm`, matches ISIS `reduce` to ~0.001 px median) |
| SfS | mu 0.05-0.1, lambda 1e-3, 5 iterations, shadow threshold 0.002, fixed albedo | `PRESETS["bertone"]["sfs"]`, `--sfs-opt` |
| Products | Table 1 (GLD01-GLD10) | `products/` |
| Uncertainty | ensemble of 4-6 subsets on 2.5 km crops, <= 40 % overlap | `ensemble_members`, `--ensemble` |

```bash
lunar sfs --aoi site.geojson --dem LDEM_87S_5MPP_ADJ.TIF --work-dir malapert --processes 8
lunar sfs M1234567890LE ... --dem ldem.tif --projwin -10000 50000 0 40000 --res 5 --preset asp
lunar sfs --aoi site.geojson --dem ldem.tif --dry-run
```

```python
from lunar_toolkit.sfs import SfsConfig, SfsPipeline

cfg = SfsConfig(
    resolution=5, projwin=(-10000, 50000, 0, 40000), preset="bertone", ensemble_members=4, processes=8
)
pipeline = SfsPipeline(
    "malapert", "LDEM_87S_5MPP_ADJ.TIF", cfg, observations=obs
)  # obs: common schema, for stereo pairs
dem = pipeline.run(images)
print(pipeline.removed, pipeline.selected, pipeline.unselected)
```

Differences from the paper: the CSM re-mapping for reduced cubes is our own implementation (`lunar_toolkit.csm`, validated with `cam_test` against ISIS `reduce`); the default selection is greedy coverage (`kmedoids` follows the paper); QA bins use the `sfs --query` Sun azimuth rather than sub-solar longitude. `cameras_from="published"` uses their published adjusted cameras instead of running bundle adjustment.

Work directory:

| Path | Contents |
|---|---|
| `images/` | calibrated cubes, CSM cameras, reduced cubes |
| `dem/` | cropped and resampled prior DEM |
| `prj/`, `prj_sfs/` | images mapprojected for BA (native) and for SfS (target resolution) |
| `ba/`, `ba_align/`, `ba_refine/` | bundle adjustment, registered and refined cameras |
| `stereo/`, `align/` | registration stereo DEMs, mosaic and transform |
| `qa/` | max-lit mosaics, all and per 30 deg Sun-azimuth bin |
| `selection/` | `removed.csv`, `selection.csv`, `selected.txt`, `unselected.txt` |
| `sfs_cameras/`, `sfs/` | SfS camera adjustments, `run-DEM-final.tif` |
| `products/`, `ensemble/` | product suite, ensemble members |

Steps whose outputs exist are skipped, so an interrupted run resumes (`force=True` / `--force` recomputes).

## Reproducible runs

```bash
lunar sfs --config examples/malapert_crop.yaml
lunar sfs --config run.yaml --dump-config > effective.yaml
lunar sfs --config run.yaml --check
lunar sfs-report malapert_crop --region malapert --fetch-published
```

`examples/malapert_crop.yaml` and `examples/asp_recipe.yaml` are complete starting points. A config holds the run keys (`work_dir`, `dem` or `dem: pgda`, `region`, `crop_km`, `crop_center`, `aoi`, `sources`, `from_file`, `max_slew`, `max_pixel_scale`, `checksums`), every `SfsConfig` field (including `seed`) and the three option-override dicts. Window precedence: `projwin`, then `aoi`, then `crop_km`, then the region. Each run saves `config.yaml` and appends to `provenance.json` (schema in `lunar_toolkit.provenance`); `--check` reports any difference in tool versions, settings or input checksums before a resume or re-run.

`lunar sfs-report` turns a work dir into `report.html`, `metrics.json`, CSV tables and PNG figures covering Figs. 3, 5-9 and Table 2 of Bertone et al.

## SfS products

`lunar_toolkit.sfs_products.make_products` writes the Table 1 suite of Bertone et al., named `<ROI>_GLD<code>_<RES>`: `ELEV` (elevation), `MASK` (lit-image count), `ISGM` (ASP height error, when given), `DIFF` (minus the prior), `TSLP` / `TRGH` (slope and VRM roughness at selectable baselines), `HILL_<RES>_315_45`, `OMOS` (max-lit orthomosaic), `BRES` (best input resolution) and `SBCT` (30 deg Sun-azimuth bins with a lit image). `ensemble_uncertainty` writes the per-pixel standard deviation of elevation and slope across ensemble members, and `compare_to_points` compares a DEM with LOLA points (`geodiff --csv-format`).

## Reproducing Bertone et al. (2026)

`lunar reference` provides the regions and reference data:

```bash
lunar reference list
lunar reference pgda malapert --kind surf toterr ldec xyzi -o ref        # a priori LDEM, uncertainty, LOLA counts, points
lunar reference pgda malapert --kind xyzi --csv --crop-km 2.5 -o ref     # LOLA points for validation
lunar reference sdem malapert --remote --member GLDELEV GLDSIGM -o ref   # their published SDEM members
lunar sfs --aoi "$(lunar reference crop malapert --crop-km 2.5 --format wkt)" \
    --dem ref/Site23/Site23_final_adj_5mpp_surf.tif --res 5 --work-dir malapert_crop
```

Start with a 2.5 km crop (a few dozen images, hours) before a full region (hundreds of images; the paper reports 3-5 days of bundle adjustment and about a day of SfS on 36 CPUs). A full region needs several hundred GB for calibrated cubes. The published zips also contain the bundle-adjusted NAC cameras (`LROC-NAC-adjust/`), which can be used to start from their registration.
