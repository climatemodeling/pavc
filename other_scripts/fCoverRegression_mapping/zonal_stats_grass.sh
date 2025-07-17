#!/usr/bin/env bash

# directory where you want the outputs
OUT_DIR="/mnt/poseidon/remotesensing/arctic/data/training/Test_06/zonalstats"

# your zone raster
ZONE_MAP="raster_cavm_v1_2019_clipped"

# get a space-separated list of all your _mosaic_clipped rasters
# (using space is easiest for bash loops)
read -r -a MAPS <<< "$(g.list type=raster pattern="*_mosaic_clipped" separator=space)"

echo "Calculating zonal statistics for ${#MAPS[@]} maps..."

for map in "${MAPS[@]}"; do
  echo "  $map"
  # write each map's stats to its own CSV:
  r.univar --overwrite -e -t \
    map="$map" \
    zones="$ZONE_MAP" \
    percentile=90 \
    separator=comma \
    output="${OUT_DIR}/${map}_zonal_stats_cavmclip.csv"
done

echo "All done."