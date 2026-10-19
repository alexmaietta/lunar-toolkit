# API

The Python API mirrors the CLI. Each page documents one module.

## Search and data access

- [LROC index (CumIndex)](cumindex.md) keeps the LROC PDS CUMINDEX as a local DuckDB / Parquet database with footprint search.
- [ODE client](ode.md) searches ShadowCam, LROC NAC and Kaguya products on the PDS Orbital Data Explorer.
- [Observation schema](observation.md) is the common record every search backend returns.
- [Geometry and AOIs](geometry.md) builds search regions and AOI polygons, polar stereographic past +/-60 deg.
- [Downloads](downloader.md) fetches PDS files in parallel, politely and with resume.

## Instruments

- [LROC NAC](nac.md) covers NAC metadata, download and the ISIS + GDAL processing pipeline.
- [Kaguya TC and MI](kaguya.md) selects, downloads and imports Terrain Camera and Multiband Imager products.
- [ShadowCam](shadowcam.md) prepares ShadowCam cubes with SPICE and CSM cameras for ASP.

## ISIS and ASP tools

- [Tool runner](tools.md) finds and runs ISIS, ASP and GDAL tools, with dry runs.
- [ISIS wrappers](isis.md) and [ASP wrappers](asp.md) wrap the command-line tools, returning output paths.
- [CSM cameras](csm.md) reads, writes and resamples CSM linescan cameras.

## DEM pipelines

- [Stereo pairs](stereo.md) ranks NAC stereo pairs and [Stereo DEM](stereo_dem.md) builds a DEM from one pair.
- [Image selection](selection.md) picks illumination-diverse images for shape-from-shading.
- [Shape-from-shading](sfs.md) runs the configurable SfS pipeline, with [SfS products](sfs_products.md) and the [SfS report](sfs_report.md).
- [Workspace](workspace.md) is the run's DuckDB database of footprints, overlaps and bundle-adjustment statistics.

## Configuration and reference data

- [Run configuration](config.md) loads and validates YAML run configs, and [Provenance](provenance.md) records each run.
- [Reference data](reference.md) gets the Artemis III regions, PGDA LOLA DEMs and published SfS DEMs.
- [Constants](constants.md) holds data directories, archive URLs and shared defaults.
