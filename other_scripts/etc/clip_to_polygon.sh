#!/bin/bash

# parameters
BAND=dem_aspect
SRC_NODATA=-9999.0 #hillshade 0.0, dem=-9999.0, slope=-9999.0, aspect=-9999.0
DST_NODATA=-9999.0 
GRID_SHP=/mnt/poseidon/remotesensing/arctic/data/vectors/supplementary/test_bounds_smaller/test_bounds_smaller.shp
RASTERS=/mnt/poseidon/remotesensing/arctic/data/rasters/arctic_dem/*${BAND}.tif
OUT_PATH=/mnt/poseidon/remotesensing/arctic/data/rasters/arctic_dem_tiled

# extent function
function ogr_extent() {
	if [ -z "$1" ]; then 
		echo "Missing arguments. Syntax:"
		echo "  ogr_extent <input_vector>"
    	return
	fi
	EXTENT=$(ogrinfo -al -so $1 |\
		grep Extent |\
		sed 's/Extent: //g' |\
		sed 's/(//g' |\
		sed 's/)//g' |\
		sed 's/ - /, /g')
	EXTENT=`echo $EXTENT | awk -F ',' '{print $1 " " $2 " " $3 " " $4}'` #xmin ymin xmax ymax
	echo -n "$EXTENT"
}

# create vrt and subset data to study area-ish
extent=$(ogr_extent $GRID_SHP)
vrt_name=${OUT_PATH}/${BAND}_mosaic_test_bounds.vrt
gdalbuildvrt $vrt_name $RASTERS -te $extent

# clip to polygon
filename=$(basename -- "$GRID_SHP")
name="${filename%.*}"
gdalwarp -overwrite \
-srcnodata $SRC_NODATA \
-dstnodata $DST_NODATA \
-crop_to_cutline -cutline $GRID_SHP \
-t_srs "+proj=longlat +datum=WGS84 +no_defs" \
-wo CUTLINE_ALL_TOUCHED=TRUE \
-wo NUM_THREADS=ALL_CPUS \
-co BIGTIFF=YES \
$vrt_name \
${OUT_PATH}/${BAND}_mosaic_test_bounds.tif