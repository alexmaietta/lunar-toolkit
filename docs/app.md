# Desktop app and browser

The Lunar Toolkit app is a GUI to search the LROC index and the PDS Orbital Data Explorer, map footprints, view full-resolution images, and manage indexes and downloads. A search panel sits on the left, the footprint map, image viewer and Settings on the right, and results and details below. Run it as a desktop window, serve it to others, or open the [hosted copy](https://alexmaietta.github.io/lunar-toolkit/browser/).

```bash
lunar app # desktop window (server on 127.0.0.1, random port)
lunar serve # http://localhost:8080 in your browser
lunar serve --images ~/lunar_runs # also browse local GeoTIFFs (SfS products)
lunar serve --host 0.0.0.0 --admin-token auto # share it; remote clients are read-only without the token
```

`lunar app` uses pywebview (WebView2 on Windows, WebKit on macOS; on Linux Qt WebEngine via `pip install "lunar-toolkit[app]"`), else a Chrome / Chromium / Edge `--app` window, else a browser tab. Its server stops when the window closes; running jobs are cancelled and their `.part` files kept.

## Search

Choose an instrument (LROC NAC, LROC WAC, LROC RDR, ShadowCam, Kaguya TC or Kaguya MI) and a product, then search by bounding box, point + radius or AOI polygon (load or paste GeoJSON / WKT), incidence, emission and date. The source follows the instrument. LROC products come from the local index when it has them, and LROC NAC falls back to the PDS Orbital Data Explorer otherwise; ShadowCam and Kaguya always use ODE. Local searches run in the browser with DuckDB-WASM on the index Parquet files and match `lunar search` exactly (bounding-box prefilter, polar stereographic poleward of +/-60 deg, exact polygon intersection for AOIs). Look up product IDs finds them in every loaded index.

The Local index box lists the indexes that `lunar app` and `lunar serve` found in the data dir. Without one it offers to open `cumindex_*.parquet` files from disk, and Settings can build them.

Results are drawn on the footprint map (lon/lat, or polar stereographic near the poles; scroll to zoom, drag to pan, click a footprint for its details).

The Basemap menu draws the LROC WAC mosaic or a LOLA hillshade under the footprints, and near the south pole also the LOLA 5 m hillshade and the LROC NAC 1 m polar mosaic. A small overview bundled with the page (`lunar_toolkit/static/basemaps`, about 14 MB, rebuilt by `docker/basemaps.sh`) shows when zoomed out or offline, and the USGS Astrogeology COGs stream tile by tile once you zoom in past it. Contours draws elevation contours of the current view at a 1-2-5 interval picked from its relief, from the LOLA global DEM, the LOLA 5 m south pole DEM or the PGDA 20 m 80S DEM (through `lunar serve` only), falling back to the bundled DEM overview.

## Image viewer

Selecting a product shows a quick look in the Details pane (drag to pan, wheel to zoom, double click to reset); the Image tab opens it in a deep-zoom viewer that reads only the visible tiles over HTTP Range:

| Product | Source |
|---|---|
| LROC NAC / WAC EDR and CDR | browse pyramid `<VOL>/EXTRAS/BROWSE/<yyyyddd>/<ID>_pyr.tif` (full resolution, tiled JPEG, 8-9 levels) |
| ShadowCam | `<id>_cog.tif` (COG), map-projected `_map_stretched.tif` / `_map_raw.tif`, `_browse.png` |
| Kaguya TC / MI | browse JPEGs (small / large) |
| Local | GeoTIFF / COG / PNG / JPEG via "Open file...", `?image=URL`, or `--images DIR` |

Wheel zoom, drag pan, `+` `-` `0` (1:1) `F` (fit) `W` (fit width), rotation by any angle (`R` / `Shift+R` 90 deg, `[` `]` 5 deg, or the Angle box), Min / Max / Gamma contrast with an automatic 2-98 % stretch, and a sample / line / value readout in the status bar; decoded tiles are cached up to 150 MB. The PDS store sends no CORS headers to browsers, so `lunar serve` relays the PDS image hosts (`/remote`, allowlisted, Range-forwarding, throttled like downloads); the hosted copy cannot read them, so it says so and shows ShadowCam, Kaguya and local images only.

## Settings

The Settings tab shows the data, cache, reference and runs dirs (`--runs-dir` or `$LUNAR_RUNS_DIR`), the built indexes (rows, release, size, build time), downloaded files and sizes, the ISIS / ASP / ISISDATA setup and tool versions, paused hosts and free disk space. It runs index builds and updates, PGDA and published SfS DEM downloads, and EDR / CDR downloads as background jobs with progress bars, rate, ETA and Cancel; downloads resume from `.part` files. The page reloads its index after a build finishes.

## Access

Admin features (Settings actions, local paths, run reports) are allowed from this machine and for requests with the admin token (`Authorization: Bearer TOKEN` or `?token=TOKEN`; `--admin-token auto` prints one). Other network clients can browse, search and view, but cannot start jobs or see local paths. Behind a reverse proxy on the same host every client looks local, so use the token there.

## Hosted copy

The [GitHub Pages copy](https://alexmaietta.github.io/lunar-toolkit/browser/) has no server: open your `cumindex_*.parquet` files from disk (read in place, nothing is uploaded) or pass hosted copies with `?edr=URL&cdr=URL&rdr=URL` (the host must allow CORS and Range requests). Settings and the PDS image relay need `lunar app` or `lunar serve`.
