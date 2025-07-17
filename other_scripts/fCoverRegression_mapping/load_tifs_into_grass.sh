#!/usr/bin/env bash

grasscr wgs84 pavc

DATA_DIR="/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test_06-19-2025"
MEM=8000

echo "Importing all *_mosaic_clipped.tif from ${DATA_DIR} with ${MEM} MB memory..."

# 2) Loop over the files
for tif in "${DATA_DIR}"/*_mosaic_clipped.tif; do
    name=$(basename "$tif" .tif)
    echo "  Importing ${name}"
    r.in.gdal --overwrite -e input="$tif" output="$name" memory="$MEM"
done

echo "Done."