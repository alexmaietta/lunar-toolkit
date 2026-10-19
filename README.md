[![pipeline](https://img.shields.io/github/actions/workflow/status/alexmaietta/lunar-toolkit/ci.yml?branch=main&label=pipeline)](https://github.com/alexmaietta/lunar-toolkit/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https://alexmaietta.github.io/lunar-toolkit/coverage/badge.json)](https://alexmaietta.github.io/lunar-toolkit/coverage/)
[![pypi](https://img.shields.io/pypi/v/lunar-toolkit)](https://pypi.org/project/lunar-toolkit/)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-blue)](LICENSE)
[![app](https://img.shields.io/badge/app-open_in_browser-yellow)](https://alexmaietta.github.io/lunar-toolkit/browser/)

# Lunar Toolkit

![Lunar Toolkit app](https://raw.githubusercontent.com/alexmaietta/lunar-toolkit/main/docs/images/hero.png)

Lunar Toolkit finds, downloads and processes lunar orbital imagery and turns it into DEMs.

It builds a searchable index of LROC images whose footprint search also works near the poles, finds ShadowCam, LROC and Kaguya products through the PDS Orbital Data Explorer, imports images into [ISIS](https://isis.astrogeology.usgs.gov/), and builds stereo and shape-from-shading (SfS) DEMs with the [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/) through configurable, reproducible pipelines. A desktop app browses the index, ODE, footprints and full-resolution images.

See our [latest documentation](https://alexmaietta.github.io/lunar-toolkit/) for guides, the CLI reference and the API.

## Install

```bash
pip install lunar-toolkit
```

Python 3.10+. DEM and image-processing commands also need ISIS and the Ames Stereo Pipeline ([setup](https://alexmaietta.github.io/lunar-toolkit/dems/#setup)), or use the all-in-one Docker image (`docker pull ghcr.io/alexmaietta/lunar-toolkit`, see [Install](https://alexmaietta.github.io/lunar-toolkit/install/)).

## Quick start

```bash
lunar build-db edr # local LROC EDR index (~3.8 GB download)
lunar search --point -85.98 -0.26 --radius-km 2 --nac-only
lunar download M165993756LE --output-dir raw
lunar app # desktop app with search, footprints, images and settings
```

## Commands

| Command | Description |
|---|---|
| `build-db` | Build or update the local LROC index (EDR, CDR, RDR) |
| `search` | Footprint search of the index by box, point + radius or AOI polygon |
| `query` | Print metadata for product IDs |
| `download` | Download LROC EDR / CDR images |
| `ode-search` | Search ShadowCam, LROC and Kaguya products on ODE |
| `pairs` | Rank NAC stereo pairs or pick illumination-diverse SfS images |
| `process` | Calibrate and map-project NAC images with ISIS |
| `kaguya` | Import Kaguya TC / MI images into ISIS with CSM cameras |
| `shadowcam` | Prepare ShadowCam cubes with SPICE and CSM cameras |
| `stereo` | DEM from a NAC stereo pair |
| `sfs` | Shape-from-shading DEM from a YAML config or flags |
| `sfs-report` | Metrics, figures and an HTML report for an SfS run |
| `workspace` | SQL over a run's DuckDB workspace (footprints, bundle-adjust stats) |
| `reference` | Artemis III regions, PGDA LOLA DEMs and published SfS DEMs |
| `app` / `serve` | The desktop app, or the same page served to a browser |

Every option is in the [CLI reference](https://alexmaietta.github.io/lunar-toolkit/cli/).

## Python

```python
from lunar_toolkit import CumIndex, NacImage

ci = CumIndex()  # the local EDR index
df = ci.search(point=(-85.98, -0.26), radius_km=2, nac_only=True)
print(len(df), "NAC images within 2 km of Malapert")

img = NacImage(df.iloc[0]["PRODUCT_ID"])
print(img.product_id, img.start_time, img.pixel_scale, img.incidence_angle, img.sub_solar_azimuth)
print(img.edr_url)
```

More in the [API reference](https://alexmaietta.github.io/lunar-toolkit/api/cumindex/).

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## Acknowledgements

This toolkit only wraps and indexes the work of others.

| What | Provided by |
|---|---|
| LROC NAC and WAC images | [Lunar Reconnaissance Orbiter Camera](https://www.lroc.asu.edu/) team, Arizona State University |
| ShadowCam images | [ShadowCam](https://www.shadowcam.asu.edu/) team (Arizona State University and Malin Space Science Systems), a NASA instrument on KARI's Korea Pathfinder Lunar Orbiter (Danuri) |
| Kaguya Terrain Camera and Multiband Imager images | JAXA [SELENE (Kaguya)](https://www.selene.jaxa.jp/) teams, distributed through JAXA's [DARTS](https://darts.isas.jaxa.jp/) archive |
| Data archives and search | NASA Planetary Data System [Imaging Node](https://pds-imaging.jpl.nasa.gov/), and the Geosciences Node's [Orbital Data Explorer](https://ode.rsl.wustl.edu/moon/) |
| LOLA 5 m site DEMs | LOLA team, NASA GSFC [Planetary Geodesy Data Archive](https://pgda.gsfc.nasa.gov/products/78) |
| Published SfS DEMs and the SfS workflow | Bertone et al. and the NASA GSFC LuNaMaps / Artemis III work, on [Zenodo](https://doi.org/10.5281/zenodo.17954508) |
| [ISIS](https://isis.astrogeology.usgs.gov/) and [ALE](https://github.com/DOI-USGS/ale) | USGS Astrogeology Science Center |
| [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/) | NASA Ames Research Center |
| [SPICE](https://naif.jpl.nasa.gov/) | NASA Navigation and Ancillary Information Facility (NAIF) |
| [GDAL](https://gdal.org/) and [DuckDB](https://duckdb.org/) | their open-source communities |

Please cite the instruments, data sets and software below when you publish results made with them.

## References

### Software

- Beyer, R. A., Alexandrov, O., & McMichael, S. (2018). The Ames Stereo Pipeline: NASA's open source software for deriving and processing terrain data. *Earth and Space Science*, 5, 537-548. [doi:10.1029/2018EA000409](https://doi.org/10.1029/2018EA000409)
- Alexandrov, O., & Beyer, R. A. (2018). Multiview shape-from-shading for planetary images. *Earth and Space Science*, 5, 652-666. [doi:10.1029/2018EA000390](https://doi.org/10.1029/2018EA000390)
- Alexandrov, O., McMichael, S., Broxton, M., et al. (2024). NeoGeographyToolkit/StereoPipeline: ASP 3.4.0. Zenodo. [doi:10.5281/zenodo.12176190](https://doi.org/10.5281/zenodo.12176190) (cite the version you use)
- Laura, J., Acosta, A., Addair, T., et al. (2023). Integrated Software for Imagers and Spectrometers (ISIS) 7.2.0. Zenodo. [doi:10.5281/zenodo.7644616](https://doi.org/10.5281/zenodo.7644616) (cite the version you use)
- Acton, C. H. (1996). Ancillary data services of NASA's Navigation and Ancillary Information Facility. *Planetary and Space Science*, 44, 65-70. [doi:10.1016/0032-0633(95)00107-7](<https://doi.org/10.1016/0032-0633(95)00107-7>)
- Rouault, E., Warmerdam, F., Schwehr, K., et al. (2025). GDAL. Zenodo. [doi:10.5281/zenodo.5884351](https://doi.org/10.5281/zenodo.5884351)
- Raasveldt, M., & Muhleisen, H. (2019). DuckDB: an embeddable analytical database. *SIGMOD '19*, 1981-1984. [doi:10.1145/3299869.3320212](https://doi.org/10.1145/3299869.3320212)

### Instruments and missions

- Vondrak, R., Keller, J., Chin, G., & Garvin, J. (2010). Lunar Reconnaissance Orbiter (LRO): observations for lunar exploration and science. *Space Science Reviews*, 150, 7-22. [doi:10.1007/s11214-010-9631-5](https://doi.org/10.1007/s11214-010-9631-5)
- Robinson, M. S., Brylow, S. M., Tschimmel, M., et al. (2010). Lunar Reconnaissance Orbiter Camera (LROC) instrument overview. *Space Science Reviews*, 150, 81-124. [doi:10.1007/s11214-010-9634-2](https://doi.org/10.1007/s11214-010-9634-2)
- Humm, D. C., Tschimmel, M., Brylow, S. M., et al. (2016). Flight calibration of the LROC Narrow Angle Camera. *Space Science Reviews*, 200, 431-473. [doi:10.1007/s11214-015-0201-8](https://doi.org/10.1007/s11214-015-0201-8)
- Smith, D. E., Zuber, M. T., Jackson, G. B., et al. (2010). The Lunar Orbiter Laser Altimeter investigation on the Lunar Reconnaissance Orbiter mission. *Space Science Reviews*, 150, 209-241. [doi:10.1007/s11214-009-9512-y](https://doi.org/10.1007/s11214-009-9512-y)
- Robinson, M. S., Brylow, S. M., Caplinger, M. A., et al. (2023). ShadowCam instrument and investigation overview. *Journal of Astronomy and Space Sciences*, 40(4), 149-171. [doi:10.5140/JASS.2023.40.4.149](https://doi.org/10.5140/JASS.2023.40.4.149)
- Kato, M., Sasaki, S., Takizawa, Y., & the Kaguya project team (2010). The Kaguya mission overview. *Space Science Reviews*, 154, 3-19. [doi:10.1007/s11214-010-9678-3](https://doi.org/10.1007/s11214-010-9678-3)
- Haruyama, J., Matsunaga, T., Ohtake, M., et al. (2008). Global lunar-surface mapping experiment using the Lunar Imager/Spectrometer on SELENE. *Earth, Planets and Space*, 60, 243-255. [doi:10.1186/BF03352788](https://doi.org/10.1186/BF03352788)
- Ohtake, M., Haruyama, J., Matsunaga, T., et al. (2008). Performance and scientific objectives of the SELENE (KAGUYA) Multiband Imager. *Earth, Planets and Space*, 60, 257-264. [doi:10.1186/BF03352789](https://doi.org/10.1186/BF03352789)

### Topography

- Mazarico, E., Neumann, G. A., Smith, D. E., Zuber, M. T., & Torrence, M. H. (2011). Illumination conditions of the lunar polar regions using LOLA topography. *Icarus*, 211, 1066-1081. [doi:10.1016/j.icarus.2010.10.030](https://doi.org/10.1016/j.icarus.2010.10.030) (PGDA site numbering)
- Barker, M. K., Mazarico, E., Neumann, G. A., et al. (2021). Improved LOLA elevation maps for south pole landing sites: error estimates and their impact on illumination conditions. *Planetary and Space Science*, 203, 105119. [doi:10.1016/j.pss.2020.105119](https://doi.org/10.1016/j.pss.2020.105119)
- Henriksen, M. R., Manheim, M. R., Burns, K. N., et al. (2017). Extracting accurate and precise topography from LROC narrow angle camera stereo observations. *Icarus*, 283, 122-137. [doi:10.1016/j.icarus.2016.05.012](https://doi.org/10.1016/j.icarus.2016.05.012)
- Bertone, S., Barker, M., Mazarico, E., McKenna, T., & Petro, N. (2026). Enhanced topography models for selected lunar south pole regions with shape-from-shading. *The Planetary Science Journal*, 7, 118. [doi:10.3847/PSJ/ae5b70](https://doi.org/10.3847/PSJ/ae5b70). Data: [doi:10.5281/zenodo.17954508](https://doi.org/10.5281/zenodo.17954508)

## License

BSD 3-Clause. See [LICENSE](LICENSE).
