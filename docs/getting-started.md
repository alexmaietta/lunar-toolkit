# Getting started

## Build and update the LROC index

```bash
lunar build-db edr                 # newest cumulative index (~3.8 GB TAB -> Parquet)
lunar build-db edr --update        # later: append newer PDS volumes only
```

The build reads the fixed-width `CUMINDEX.TAB` directly with DuckDB, types each column from the label, derives footprint columns (`FP_*`) and writes a ZSTD Parquet sorted by `CENTER_LATITUDE, PRODUCT_ID`. A `*.meta.json` sidecar records the release, row count and MD5. For a small test database use one volume's own index:

```bash
lunar build-db edr --release LROLRC_0066C --volume-only
```

Databases go to the user data dir (`~/.local/share/lunar_toolkit` on Linux), or `LUNAR_DATA_DIR` if set.

## Search footprints

```python
from lunar_toolkit import CumIndex

ci = CumIndex()
south = ci.search(lat=(-90, -85), nac_only=True, incidence_angle=(80, 95))
seam = ci.search(lat=(-5, 5), lon=(355, 5))  # west > east crosses 0/360
crater = ci.search(point=(-89.67, 129.8), radius_km=10, wkt=True)
obs = ci.to_observations(crater)  # common schema
```

!!! note
    Box and point searches are bounding-box based (lat range, seam-aware longitude arc and, poleward of +/-60 deg, a polar stereographic box). They never drop a true hit for convex footprints but can include a few false positives. AOI searches refine those candidates by exact polygon intersection.

## Search by area of interest

Export a polygon from QuickMap (GeoJSON or WKT) and pass the file, or a WKT string. Longitudes may be -180..180 or 0..360; polygons may cross the seam or enclose a pole.

```python
ci.search(aoi="malapert.geojson", nac_only=True)
```

```bash
lunar search --aoi malapert.geojson --nac-only
lunar ode-search -i shadowcam -t cal --aoi malapert.geojson
```

## Stereo pairs and SfS image selection

```python
from lunar_toolkit.stereo import find_stereo_pairs
from lunar_toolkit.selection import select_illumination_diverse, select_for_coverage

aoi = {"point": (-89.67, 129.8), "radius_km": 3}
obs = ci.to_observations(ci.search(**aoi, nac_only=True, wkt=True))
pairs = find_stereo_pairs(obs, aoi=aoi)  # best first
sfs = select_illumination_diverse(obs, incidence=(80, 95), aoi=aoi, method="greedy")
cover = select_for_coverage(obs, aoi, azimuth_sep_deg=30, views_per_cell=3)
```

```bash
lunar pairs --point -89.67 129.8 --radius-km 3 --top 20
lunar pairs --point -89.67 129.8 --radius-km 3 --sfs --inc 80 95 --list sfs_images.txt
```

## ShadowCam and other instruments via ODE

```python
from lunar_toolkit.ode import OdeClient

ode = OdeClient()
sc = ode.search("shadowcam", "cal", point=(-89.67, 129.8), radius_km=5, incidence=(80, 90))
ode.download(sc.head(3), "./shadowcam", kind="product")  # .cub files
ode.download(sc.head(3), "./shadowcam", kind="browse", match="_cog.tif")
```

```bash
lunar ode-search -i shadowcam -t cal --point -89.67 129.8 --radius-km 5 --count
lunar ode-search -i kaguya_tc -t stereo --bbox 10 20 11 21 --csv tc.csv
```

## Download and process LROC NAC images

```bash
lunar download M1096757719LE M1096757719RE --output-dir ./raw
lunar process M1096757719LE --output-dir ./output
```

Downloads resume from `.part` files (HTTP Range) and are MD5-verified when the label carries a checksum. `process` runs lronac2isis -> spiceinit -> lronaccal -> lronacecho -> cam2map -> GeoTIFF.

## Kaguya and ShadowCam

```bash
lunar kaguya TC1W2B0_01_02936N034E0938 TC2W2B0_01_02936N036E0938 -o ./tc   # Terrain Camera, Level 2B0
lunar kaguya MVA_2B2_01_01935N340E0939 -o ./mi                             # Multiband Imager, Level 2B2
lunar shadowcam --point -89.67 129.8 --radius-km 5 --inc 80 90 --limit 4 -o ./sc
```

`kaguya` downloads from ODE (JAXA DARTS) and imports into ISIS (`kaguyatc2isis` / `kaguyami2isis` -> `spiceinit`) with a CSM camera from `isd_generate`. `shadowcam` downloads the delivered ISIS cubes, attaches SPICE and writes CSM cameras following ASP's ShadowCam example.

## Artemis III reference data

```bash
lunar reference list                                   # 13 Artemis III regions (Bertone et al. 2026)
lunar reference pgda malapert --kind surf -o ref       # PGDA LOLA 5 m DEM for the region
lunar reference crop malapert --crop-km 2.5            # 2.5 km test window (projwin)
```

Continue with [DEMs with ISIS + ASP](dems.md).
