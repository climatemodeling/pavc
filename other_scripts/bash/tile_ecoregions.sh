#!/bin/bash

# set parameters
BAND=ecoregion100
OUT_PATH=/mnt/poseidon/remotesensing/arctic/data/rasters/hoffman_ecoregions_100k_tiled
ECOREGIONS100=/mnt/poseidon/remotesensing/arctic/data/Hoffman_LandscapeEcology_2013/ecoregions/alaska_2000_2009_dem_Feb2012.100.nc
GRIDCELLS=/mnt/poseidon/remotesensing/arctic/data/vectors/supplementary/tundra_alaska_grid_cells_akaea/GRIDCELL_*.shp
SRC_NODATA=-99999
DST_NODATA=0

# clip to polygons
for GRIDCELL in $GRIDCELLS; do

    echo Working on gridcell ${GRIDCELL}
    filename=$(basename -- "$GRIDCELL")
    name="${filename%.*}"
    gdalwarp -overwrite \
	-if netCDF \
	-ot UInt16 \
	-srcnodata $SRC_NODATA \
	-dstnodata $DST_NODATA \
	-crop_to_cutline -cutline $GRIDCELL \
	-t_srs "+proj=longlat +datum=WGS84 +no_defs" \
	-wo CUTLINE_ALL_TOUCHED=TRUE \
	-wo NUM_THREADS=ALL_CPUS \
	$ECOREGIONS100 \
	${OUT_PATH}/${name}_${BAND}.tif

done
