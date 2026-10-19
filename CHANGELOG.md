# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- LROC PDS index: DuckDB build of CUMINDEX -> Parquet, with incremental per-volume `--update`.
- Footprint search that handles the poles and the 0/360 seam (`lunar search`, `CumIndex.search`).
- Area-of-interest search from GeoJSON / WKT with exact polygon refinement (`--aoi`).
- ODE search and download for ShadowCam, LROC NAC and Kaguya TC (`lunar ode-search`, `OdeClient`).
- Common observation schema shared by all search backends.
- Resumable, MD5-checked PDS downloads (`lunar download`).
- LROC NAC ISIS pipeline to GeoTIFF (`lunar process`).
- Kaguya TC (Level 2B0) and MI (Level 2B2) download and ISIS import with CSM cameras (`lunar kaguya`, `lunar_toolkit.kaguya`).
- ShadowCam cube download, SPICE and CSM cameras after the ASP 3.7 example (`lunar shadowcam`, `lunar_toolkit.shadowcam`).
- ISIS, ASP and GDAL wrappers with dry-run recording; tools run from un-activated conda envs (`lunar_toolkit.isis`, `.asp`, `.tools`).
- Stereo DEM from a NAC pair (`lunar stereo`, `stereo_dem`).
- Shape-from-shading pipeline reproducing Bertone et al. (2026): native-resolution bundle adjustment, post-BA image rejection, stereo registration to the prior DEM, QA mosaics, illumination-coverage image selection, camera downsampling, presets (`lunar sfs`, `SfsPipeline`).
- SfS product suite (Table 1 of Bertone et al.), VRM roughness, ensemble uncertainty and LOLA point comparison (`sfs_products`).
- Artemis III regions, PGDA LOLA 5 m site DEMs and published SfS DEMs for reproduction and validation (`lunar reference`, `reference`).
- Config-driven, reproducible SfS runs: YAML configs with region presets and validation (`lunar sfs --config / --dump-config / --check`, `--region`, `--crop-km`, `--seed`), `config.yaml` and `provenance.json` in each work dir, example configs (`examples/*.yaml`).
- SfS report reproducing the evaluation of Bertone et al. (2026): metrics, tables, figures and `report.html` (`lunar sfs-report`, `sfs_report`).
- DuckDB workspace per run (`lunar.duckdb`): polar stereographic footprints, SQL spatial join for stereo pairs (`lunar pairs --engine duckdb`), bundle-adjust tables and views implementing the image rejection rules (`lunar workspace`, `lunar_toolkit.workspace`).
- CSM cameras re-mapped to reduced / cropped cubes so SfS runs on CSM at the target resolution (`lunar_toolkit.csm`, `sfs_cameras`), k-medoids SfS image selection (`selection: kmedoids`), and the published Bertone et al. adjusted cameras, image status and map extents (`cameras_from: published`, `reference.published_cameras`).
- Polite, patient downloads: per-host concurrency and pacing, shared and persisted `Retry-After` cooldowns, retries that resume `.part` files (`LUNAR_MAX_PER_HOST`, `LUNAR_MIN_INTERVAL`, `LUNAR_MAX_WAIT`).
- `runs_dir` / `LUNAR_RUNS_DIR` for relative run directories.
- Image search keeps slewed images for bundle adjustment and stereo registration, with the nadir cut applied only to SfS selection (`max_sfs_slew`), and caps candidates before download (`max_images`: stereo pairs + illumination coverage).
- Stereo pair ranking and illumination-diverse image selection, with `--aoi` (`lunar pairs`, `stereo`, `selection`).
- Browser page with one search panel for LROC, ShadowCam and Kaguya (local LROC index or ODE, chosen by instrument), box / point / AOI search matching `lunar search`, a footprint map, and a classic desktop look with light and dark themes, via `lunar serve` or GitHub Pages.
- Image previews and a deep-zoom Image viewer in the browser (LROC browse pyramids, ShadowCam COGs, Kaguya JPEGs, local GeoTIFFs; geotiff.js 3.0.5), a PDS image relay and `--images DIR` in `lunar serve`.
- Basemaps (LROC WAC, LOLA hillshade, south pole LOLA 5 m hillshade and LROC NAC 1 m mosaic) and elevation contours under the browser's footprint map: a bundled offline overview plus streamed USGS COG tiles (`docker/basemaps.sh`).
- More robust ISIS calibration: `spiceinit` falls back to the ellipsoid without the ISIS LOLA shape model, `lronaccal` falls back to the nearest dark when no dark pair brackets the image.
- Desktop app (`lunar app`, pywebview with Chromium `--app` and browser fallbacks) and a Settings tab: indexes, caches, reference data, runs, ISIS / ASP environment, and build / download jobs with live progress (`/api/status`, `/api/jobs`, SSE); remote clients read-only unless they hold `--admin-token`; the image relay uses the per-host download throttle.
- MkDocs documentation with a CLI reference generated from argparse, offline test suite, opt-in ISIS / ASP integration tests (`pytest -m tools`) and coverage report.
- CI: pre-commit (ruff, mypy), tests on Linux / macOS / Windows, sdist + wheel build, docs and coverage on GitHub Pages, PyPI trusted publishing.
- Docker image with ISIS 10, ASP 3.7, a minimal ISISDATA and lunar_toolkit, published to `ghcr.io/alexmaietta/lunar-toolkit`.
- Python 3.10-3.14 support.
- BSD-3-Clause license.
