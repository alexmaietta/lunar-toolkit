#!/usr/bin/env bash
# Minimal ISISDATA for LRO NAC with the USGS SPICE web service (~150 MB), plus the
# optional 2 GB LOLA shape model.  Usage: isisdata.sh ISISDATA RCLONE_CONF [with_shape]
set -euo pipefail

data=$1
conf=$2
with_shape=${3:-0}
rc=(rclone --config "$conf" --retries 5 --low-level-retries 20 --transfers 4)

"${rc[@]}" copy base: "$data/base" --filter "+ kernels/{lsk,pck,fk}/**" --filter "- **"
if [ "$with_shape" = "1" ]; then
    "${rc[@]}" copy base: "$data/base" --filter "+ dems/ldem_128ppd_Mar2011_clon180_radius_pad.cub" --filter "- **"
fi

sclk=$("${rc[@]}" lsf lro:kernels/sclk | grep -E '^lro_clkcor_[0-9]{7}_v00\.tsc$' | sort | tail -1)
"${rc[@]}" copy lro: "$data/lro" \
    --filter "+ kernels/{fk,iak,ik,pck,tspk,mk}/**" \
    --filter "+ kernels/sclk/kernels.*.db" \
    --filter "+ kernels/sclk/$sclk" \
    --filter "- **"
"${rc[@]}" copy lro: "$data/lro" --filter "+ calibration/NAC*" --filter "+ calibration/nac_darks/**" --filter "- **"

echo "ISISDATA: $(du -sh "$data" | cut -f1) (sclk $sclk, shape model: $with_shape)"
