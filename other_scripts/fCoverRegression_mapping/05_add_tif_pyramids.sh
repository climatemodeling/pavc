#!/usr/bin/env bash

# this script adds pyramids to the tiff files to make it faster to render on QGIS

# switch into your data directory
cd /mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test_06-19-2025

# loop all *_mosaic.tif and build 2–64× overviews
for tif in *_mosaic.tif; do
  echo "Building pyramids for ${tif}…"
  gdaladdo -r average "${tif}" 2 4 8 16 32 64
done