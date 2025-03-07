#!/bin/bash

# set parameters
BAND=VV
OUT_PATH=/mnt/poseidon/remotesensing/arctic/data/rasters/s1_grd_tiled
RASTERS=/mnt/poseidon/remotesensing/arctic/data/rasters/s1_grd_preprocessed/*.data/*${BAND}.img
GRIDCELLS=/mnt/poseidon/remotesensing/arctic/data/vectors/supplementary/tundra_alaska_grid_cells_aoi_latlon/GRIDCELL_*.shp
SRC_NODATA=0 #unknown
DST_NODATA='float64'

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

#--------------------------------------------------------#

# create vrt and subset data to study area-ish
# vrt_name=${OUT_PATH}/${BAND}_mosaic_aoi.vrt
# gdalbuildvrt $vrt_name $RASTERS

#--------------------------------------------------------#
#--------------------------------------------------------#
# Mask code below first.
# Run.
# At the top of _mosaic_aoi.vrt, add/replace with:

#   <VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">
#     <PixelFunctionType>average</PixelFunctionType>
#     <PixelFunctionLanguage>Python</PixelFunctionLanguage>
#     <PixelFunctionCode><![CDATA[
# import numpy as np
# def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
#     x = np.ma.masked_equal(in_ar, -3.4e+38)
#     np.nanmean(x, axis = 0,out = out_ar, dtype = 'float64')
#     mask = np.all(x.mask,axis = 0)
#     out_ar[mask]=-3.4e+38
# ]]>
#     </PixelFunctionCode>

# After doing that, mask code except for vrt_name above and unmask below.
# Run.
#--------------------------------------------------------#
#--------------------------------------------------------#

# try https://askubuntu.com/questions/1447474/how-to-use-sed-for-replacing-xml-tag-in-multiple-lines-xml-content-on-bash

med_vrt_name=${OUT_PATH}/${BAND}_mosaic_aoi.vrt
export GDAL_VRT_ENABLE_PYTHON=YES

# clip to polygons
for GRIDCELL in $GRIDCELLS; do

    echo Working on gridcell ${GRIDCELL}
    filename=$(basename -- "$GRIDCELL")
    name="${filename%.*}"
    gdalwarp -overwrite \
	-srcnodata $SRC_NODATA \
	-dstnodata $DST_NODATA \
	-crop_to_cutline -cutline $GRIDCELL \
	-t_srs "+proj=longlat +datum=WGS84 +no_defs" \
	-wo CUTLINE_ALL_TOUCHED=TRUE \
	-wo NUM_THREADS=ALL_CPUS \
	$med_vrt_name \
	${OUT_PATH}/${name}_${BAND}.tif

done
