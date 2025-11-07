# set mosaic parameters
BAND=B2
TIF_DIR=/mnt/poseidon/remotesensing/arctic/data/rasters/s2_sr_tiled/ak_arctic_summer/B2/2019-06-01_to_2019-08-31
OUT_DIR=/mnt/poseidon/remotesensing/arctic/data/rasters/s2_sr_tiled/ak_arctic_summer/B2/2019-06-01_to_2019-08-31
MOSAIC_NAME=${BAND}_mosaic.tif
TIF_LIST=${BAND}_tif_list.txt
NODATA=-9999
TIF_FILES=GRIDCELL_*.tif

# use GDAL to create mosaic and clip
cd "$TIF_DIR"
ls -1 ${TIF_FILES} | awk -F'[_.]' '$2 >= 1077 && $2 <= 4594' > "$TIF_LIST"  # overwrites existing list
echo "Number of tiles in list: $(wc -l < "$TIF_LIST")"

# Step 1: Create VRT (Virtual Raster Table)
gdalbuildvrt -input_file_list "$TIF_LIST" "${BAND}_merged.vrt"

# Step 2: Convert VRT to GeoTIFF with compression
gdal_translate \
  -of GTiff \
  -tr 0.000179663056824 0.000179663056824 \
  -r average \
  -a_srs EPSG:4326 \
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