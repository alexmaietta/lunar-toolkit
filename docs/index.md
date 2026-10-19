# Lunar Toolkit

Lunar Toolkit is a Python tool suite that finds, downloads and processes lunar orbital imagery and turns it into DEMs. It keeps a searchable local copy of the LROC PDS index whose footprint search works at the poles and across the 0/360 seam, takes QuickMap GeoJSON or WKT areas of interest, and finds ShadowCam, LROC and Kaguya products on the PDS Orbital Data Explorer. It imports images into [ISIS](https://isis.astrogeology.usgs.gov/) and builds stereo and shape-from-shading DEMs with the [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/) through configurable, reproducible pipelines, with PGDA LOLA DEMs and published SfS DEMs as reference data for validation.

!!! tip "Desktop app"
    `lunar app` opens the toolkit in a desktop window with search, footprint maps, full-resolution images and a Settings panel for indexes and downloads. See [Desktop app and browser](app.md), or try the [hosted copy](https://alexmaietta.github.io/lunar-toolkit/browser/) in any browser.

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
