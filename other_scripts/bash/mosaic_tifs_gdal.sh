# create log file
BAND=graminoid
LOGDIR=/mnt/poseidon/remotesensing/arctic/scripts/scripts_bash/logs
LOGPATH=${LOGDIR}/${BAND}_mosaic.log
mkdir -p "$LOGDIR"  # Ensure log directory exists
exec 2> >(tee "$LOGPATH")

# set mosaic parameters
TIF_DIR=/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test07
OUT_DIR=/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test07
MOSAIC_NAME=${BAND}_mosaic.tif
TIF_LIST=${BAND}_tif_list.txt
BASE_FILE=*_${BAND}.tif
NODATA=-9999

# (optional) grass parameters
MAPSET=morgan
COLOR=elevation

# (optional) cut mosaic into grid cells
SHAPEFILES=/mnt/poseidon/remotesensing/arctic/data/vectors/supplementary/tundra_alaska_grid_cells

# helper function
removeExtension() {
    file="$1"
    echo "$(echo "$file" | sed -r 's/\.[^\.]*$//')"
}

# use GDAL to create mosaic and clip
cd "$TIF_DIR"
ls -1 $BASE_FILE > "$TIF_LIST"  # overwrites existing list

# Step 1: Create VRT (Virtual Raster Table)
gdalbuildvrt -input_file_list "$TIF_LIST" merged.vrt

# Step 2: Convert VRT to GeoTIFF with compression
gdal_translate -co COMPRESS=DEFLATE -co BIGTIFF=YES -a_nodata "$NODATA" merged.vrt "$OUT_DIR/$MOSAIC_NAME"

# clip data but maintain entire raster extent (otherwise -crop_to_cutline)
# echo Cutting to polygons ...
# for F in ${SHAPEFILES}/*.shp; do
#     echo Working on "$(basename ${F})"
#     gdalwarp -cutline $F ${MOSAIC_NAME}.tif "${OUT_DIR}/$(removeExtension "$(basename ${F})")_${BAND}".tif
# done
# echo COMPLETE!

# import into GRASS
# grasscr global_latlon $MAPSET
# r.in.gdal --o -e input=${MOSAIC_NAME}.tif output=$MOSAIC_NAME
# r.colors map=$MOSAIC_NAME color=$COLOR