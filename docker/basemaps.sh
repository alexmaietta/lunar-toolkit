#!/usr/bin/env bash
# Rebuild the browser's offline basemap bundle (lunar_toolkit/static/basemaps) from the USGS COGs
# with GDAL, reading only their overviews over HTTP.  Usage: basemaps.sh [GDAL_BIN_DIR] [OUT_DIR]
set -euo pipefail

bin=${1:-$HOME/miniforge3/envs/isis/bin}
out=${2:-$(dirname "$0")/../lunar_toolkit/static/basemaps}
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR GDAL_PAM_ENABLED=NO CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif
export PROJ_DATA=${PROJ_DATA:-$bin/../share/proj}
W=/vsicurl/https://asc-pds-services.s3.us-west-2.amazonaws.com/wms_basemaps3/Moon
R=609600                      # polar half-width (m): poleward of ~70 deg
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"
jpeg=(-q -of JPEG -co QUALITY=80 -r average)
cog=(-q -of COG -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BLOCKSIZE=256 -co OVERVIEWS=NONE)
stere() { echo "+proj=stere +lat_0=$1 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"; }

"$bin/gdal_translate" "${jpeg[@]}" -outsize 5120 2560 "$W/LRO_WAC_Mosaic_Global_303ppd_v3_cog.tif" "$out/wac_global.jpg"
"$bin/gdal_translate" "${jpeg[@]}" -outsize 4096 2048 "$W/LRO_LOLA_Shade_Global_128ppd_v04_cog.tif" "$out/lola_shade_global.jpg"
"$bin/gdal_translate" "${cog[@]}" -r average -outsize 1440 720 "$W/LRO_LOLA_DEM_Global_256ppd_v06_16bit_cog.tif" "$out/lola_dem_global.tif"

for pole in south north; do
    lat=$([ $pole = south ] && echo -90 || echo 90)
    hp=$([ $pole = south ] && echo S || echo N)
    "$bin/gdal_translate" "${jpeg[@]}" -projwin -$R $R $R -$R -outsize 3048 3048 \
        "$W/${pole}_pole/WAC_morphology_${hp}P_June2013_100m_8bit_cog.tif" "$out/wac_$pole.jpg"
    "$bin/gdalwarp" -q -overwrite -ovr 1 -t_srs "$(stere $lat)" -te -$R -$R $R $R -tr 400 400 -r bilinear \
        -ot Int16 -dstnodata -32768 "$W/LRO_LOLA_DEM_Global_256ppd_v06_16bit_cog.tif" "$tmp/dem_$pole.tif"
    "$bin/gdaldem" hillshade -q -multidirectional -z 2 "$tmp/dem_$pole.tif" "$tmp/shade_$pole.tif"
    "$bin/gdal_translate" "${jpeg[@]}" -outsize 1524 1524 "$tmp/shade_$pole.tif" "$out/lola_shade_$pole.jpg"
    "$bin/gdal_translate" "${cog[@]}" -r average -outsize 762 762 "$tmp/dem_$pole.tif" "$out/lola_dem_$pole.tif"
done

# JPEG bounds [xmin, ymin, xmax, ymax]: lon / lat degrees (eqc) or polar stereographic metres
cat > "$out/manifest.json" <<EOF
{
  "wac_global": {"file": "wac_global.jpg", "proj": "eqc", "bounds": [-180, -90, 180, 90]},
  "lola_shade_global": {"file": "lola_shade_global.jpg", "proj": "eqc", "bounds": [-180, -90, 180, 90]},
  "lola_dem_global": {"file": "lola_dem_global.tif", "proj": "eqc"},
  "wac_south": {"file": "wac_south.jpg", "proj": "S", "bounds": [-$R, -$R, $R, $R]},
  "wac_north": {"file": "wac_north.jpg", "proj": "N", "bounds": [-$R, -$R, $R, $R]},
  "lola_shade_south": {"file": "lola_shade_south.jpg", "proj": "S", "bounds": [-$R, -$R, $R, $R]},
  "lola_shade_north": {"file": "lola_shade_north.jpg", "proj": "N", "bounds": [-$R, -$R, $R, $R]},
  "lola_dem_south": {"file": "lola_dem_south.tif", "proj": "S"},
  "lola_dem_north": {"file": "lola_dem_north.tif", "proj": "N"}
}
EOF
ls -l "$out"
