#!/usr/bin/env bash

# this script runs gdalwarp for 4 files at a time

#——— PARAMETERS —————————————————————————————————————————————————————
shapefile="/mnt/poseidon/remotesensing/arctic/alaska_pft_fcover_harmonization/data/supporting_data/tundra_alaska_latlon/tundra_alaska_wgs84.shp"
layer="$(basename "${shapefile%.*}")"
basedir="/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test_06-19-2025"
xres="0.00017963752199490085"
yres="0.0001796375219949009"

# list your mosaics and their desired output data types
files=(
  # "bryophyte_30M-5P-IQR3_mosaic.tif"
  # "deciduous_shrub_30M-3C-IQR2_mosaic.tif"
  # "evergreen_shrub_30M-4P-IQR2.5_mosaic.tif"
  # "forb_55M-5C-IQR1.5_mosaic.tif"
  # "graminoid_30M-3P-IQR2_mosaic.tif"
  # "lichen_55M-3P-IQR2.5_mosaic.tif"
  # "litter_55M-2C-IQR1_mosaic.tif"
  # "litter_55M-4C-IQR2_mosaic.tif"
  # "litter_30M-5P-IQR2_mosaic.tif"
  # "litter_30M-2P-IQR3_mosaic.tif"
  # "non-vascular_30M-2P-IQR1.5_mosaic.tif"
  "water_mask.tif"
  # "raster_cavm_v1_2019.tif"
)
dtypes=(
  # "Float32" 
  # "Float32" 
  # "Float32" 
  # "Float32" 
  # "Float32"
  # "Float32" 
  # "Float32" 
  # "Float32" 
  # "Float32"
  # "Float32"
  # "Float32" 
  "Byte"
  # "Int16"
)

# how many gdalwarp jobs to run in parallel
MAXJOBS=4

#——— LOOP AND CLIP ——————————————————————————————————————————————————
for idx in "${!files[@]}"; do
  file="${files[idx]}"
  dtype="${dtypes[idx]}"
  predictor=$([[ "$dtype" == "Float32" ]] && echo 3 || echo 2)

  src="${basedir}/${file}"
  dst="${basedir}/${file%.tif}_clipped.tif"

  gdalwarp \
    -overwrite \
    -cutline "$shapefile" \
    -cl "$layer" \
    -crop_to_cutline \
    -ot "$dtype" \
    -tr "$xres" "$yres" \
    -tap \
    -multi \
    -wm 500 \
    -co TILED=YES \
    -co COMPRESS=DEFLATE \
    -co PREDICTOR="$predictor" \
    -co ZLEVEL=6 \
    -co BIGTIFF=YES \
    "$src" "$dst" &

  # wait until we have fewer than MAXJOBS running
  while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do
    sleep 1
  done
done

# wait for all background jobs to finish
wait

echo "All clips complete."