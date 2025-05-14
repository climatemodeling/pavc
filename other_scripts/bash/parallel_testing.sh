#!/bin/bash

# set parameters
BAND=VV
GRID_SHP=/mnt/poseidon/remotesensing/arctic/data/vectors/supplementary/tundra_alaska_grid_aoi_latlon/tundra_alaska_grid_aoi.shp
OUT_PATH=/mnt/poseidon/remotesensing/arctic/data/rasters/s1_grd_tiled
RASTERS=/mnt/poseidon/remotesensing/arctic/data/rasters/s1_grd_preprocessed/*.data/*.img
GRIDCELLS=/mnt/poseidon/remotesensing/arctic/data/vectors/supplementary/tundra_alaska_grid_cells_aoi_latlon/GRIDCELL_*.shp
GRIDCELLS=`ls -v1 ${GRIDCELLS}`
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
# extent=$(ogr_extent $GRID_SHP)
vrt_name=${OUT_PATH}/${BAND}_mosaic_aoi.vrt
# gdalbuildvrt $vrt_name $RASTERS -te $extent

#--------------------------------------------------------#
#--------------------------------------------------------#
# Mask code below first.
# Run.
# At the top of _mosaic_aoi.vrt, add/replace with:
#
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
#
# After doing that, mask code except for vrt_name above and unmask below.
# Run.
#--------------------------------------------------------#
#--------------------------------------------------------#

# try https://askubuntu.com/questions/1447474/how-to-use-sed-for-replacing-xml-tag-in-multiple-lines-xml-content-on-bash


run_gdal() {

	# accepted params
	vrt_name=$1
	med_vrt_name=$2
	gridcells=$3
	src_nodata=$4
	dst_nodata=$5
	out_path=$6
	band=$7

	# create virtual raster
	export GDAL_VRT_ENABLE_PYTHON=YES
	gdal_translate $vrt_name $med_vrt_name

	# clip to polygons
	for gridcell in $gridcells; do

		echo Working on gridcell ${gridcell}
		filename=$(basename -- "$gridcell")
		name="${filename%.*}"
		gdalwarp -overwrite \
		-srcnodata $src_nodata \
		-dstnodata $dst_nodata \
		-crop_to_cutline -cutline $gridcell \
		-t_srs "+proj=longlat +datum=WGS84 +no_defs" \
		-wo CUTLINE_ALL_TOUCHED=TRUE \
		-wo NUM_THREADS=ALL_CPUS \
		$med_vrt_name \
		${outpath}/${name}_${band}.tif

	done
}

med_vrt_name=${OUT_PATH}/${BAND}_mosaic_aoi_median.vrt
n_cpus=10
range=$((4594 - 1077))
steps=$(($range / $n_cpus))
seq 1077 4594 | xargs -n 352 echo

for listofgridcells in listoflistsofgridcells
do
  run_gdal $listofgridcells $infile param 
  break
done

listoflistsofgridcells=(({107..1376}) ({1377..1676}) ({1677..1976}) ({1977..2276}) ({2277..2576}) ({2577..2876}) ({2877..3176}) ({3177..3476}) ({3477..3776}) ({3777..4076}) ({4077..4376}) ({4377..4594}))
for sequence in "${listoflistsofgridcells[@]}"
do 
  declare -n list=$sequence
  echo $list
done