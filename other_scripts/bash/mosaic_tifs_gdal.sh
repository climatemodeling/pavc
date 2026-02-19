# create log file
BAND=rh95_rh95-21nov25
# set mosaic parameters
TIF_DIR=/mnt/poseidon/remotesensing/arctic/data/rasters/height_results_tiled_11-21-2025
OUT_DIR=/mnt/poseidon/remotesensing/arctic/data/rasters/height_results_tiled_11-21-2025
POLY=/mnt/poseidon/remotesensing/arctic/data/vectors/supplementary/tundra_alaska_polygon_latlon/tundra_alaska_wgs84.shp
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
  -tr 0.000179663056824 0.000179663056824 \
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

# Step 3: Clip to polygon
gdalwarp \
  -srcnodata "$NODATA" \
  -dstnodata "$NODATA" \
  -crop_to_cutline -cutline "$POLY" \
  -tr 0.000179663056824 0.000179663056824 \
  -tap \
  -te -173.0516512 58.5499761 -141.0005449 71.3703468 \  # extent of PAVC gridded v1.1 files
  -co BIGTIFF=YES \
  -wo CUTLINE_ALL_TOUCHED=TRUE \
  -wo NUM_THREADS=ALL_CPUS \
  "$OUT_DIR/$MOSAIC_NAME" \
  "$OUT_DIR/${BAND}_mosaic_clipped.tif"