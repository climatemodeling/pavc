# create log file
BAND=non-vascular_30M-2P-IQR1.5
# set mosaic parameters
TIF_DIR=/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test_06-19-2025
OUT_DIR=/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test_06-19-2025
MOSAIC_NAME=${BAND}_mosaic.tif
TIF_LIST=${BAND}_tif_list.txt
BASE_FILE=*_${BAND}.tif
NODATA=-9999

# use GDAL to create mosaic and clip
cd "$TIF_DIR"
ls -1 $BASE_FILE > "$TIF_LIST"  # overwrites existing list

# Step 1: Create VRT (Virtual Raster Table)
gdalbuildvrt -input_file_list "$TIF_LIST" "${BAND}_merged.vrt"

# Step 2: Convert VRT to GeoTIFF with compression
gdal_translate \
  -of GTiff \
  -ot Float32 \
  -co TILED=YES \
  -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 \
  -co COMPRESS=DEFLATE \
  -co ZLEVEL=9 \
  -co PREDICTOR=3 \
  -co BIGTIFF=YES \
  -co NUM_THREADS=ALL_CPUS \
  -a_nodata "$NODATA" \
  "${BAND}_merged.vrt" \
  "$OUT_DIR/$MOSAIC_NAME"