#!/usr/bin/env bash

# this script applies gdal_calc to 4 files at once

#——— CONFIGURATION ——————————————————————————————————————————————————
BASEDIR="/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test_06-19-2025"
WATER_MASK="${BASEDIR}/water_mask_clipped.tif"
TARGET_NODATA=-9999

# Maximum number of simultaneous gdal_calc jobs
MAXJOBS=4

#——— PARALLEL MASKING LOOP ———————————————————————————————————————————
for SRC in "${BASEDIR}"/*_mosaic_clipped.tif; do
  BASENAME=$(basename "$SRC" .tif)
  DST="${BASEDIR}/${BASENAME}_masked.tif"

  # launch the masking in the background
  gdal_calc.py \
    --quiet \
    -A "$SRC" \
    -B "$WATER_MASK" \
    --outfile="$DST" \
    --calc="numpy.where(B==1, -9999, A)" \
    --NoDataValue=$TARGET_NODATA \
    --type=Float32 \
    --creation-option="TILED=YES" \
    --creation-option="COMPRESS=DEFLATE" \
    --creation-option="PREDICTOR=3" \
    --creation-option="ZLEVEL=6" \
    --creation-option="BIGTIFF=YES" \
    --overwrite &

  echo "Exported to ${DST}"

  # throttle: wait if too many jobs are running
  while (( $(jobs -r | wc -l) >= MAXJOBS )); do
    sleep 1
  done
done

# wait for remaining background jobs to finish
wait
echo "All mosaics have been masked."
