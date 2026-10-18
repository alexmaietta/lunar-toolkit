# Lunar Toolkit

Data-layer tools for lunar orbital imagery: a local, searchable copy of the LROC PDS index, footprint search that handles the poles and the 0/360 seam, area-of-interest search from QuickMap GeoJSON / WKT polygons, ODE search and download for ShadowCam, LROC and Kaguya TC, an ISIS + GDAL pipeline for LROC NAC images, ShadowCam and Kaguya TC / MI import, stereo and shape-from-shading DEM pipelines built on Python wrappers for ISIS and the Ames Stereo Pipeline (reproducing the Bertone et al. 2026 Artemis III SfS workflow and products), and reference data for validation (PGDA LOLA DEMs, published SfS DEMs). All search backends can return one common observation schema.

!!! tip "Desktop app"
    `lunar app` opens the toolkit in a desktop window: search the LROC index and ODE, footprints on a map, full-resolution image viewing, and a Settings tab for indexes, downloads and runs with progress bars. See [Desktop app and browser](app.md); a [hosted copy](https://alexmaietta.github.io/lunar-toolkit/browser/) runs in any browser.

## Install

```bash
pip install lunar-toolkit
```

Python 3.10+. See [Install](install.md) for extras, Docker and data locations, and [DEMs with ISIS + ASP](dems.md#setup) for the ISIS / ASP setup.

## Quickstart

```bash
lunar build-db edr --release LROLRC_0066C --volume-only   # small index, ~18 MB download
lunar search --bbox 0 -90 360 -85 --nac-only
lunar serve
```

See [Getting started](getting-started.md), [DEMs with ISIS + ASP](dems.md), the [CLI reference](cli.md) and the [coverage report](https://alexmaietta.github.io/lunar-toolkit/coverage/).
