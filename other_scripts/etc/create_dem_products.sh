#!/bin/bash

LOGPATH=/mnt/poseidon/remotesensing/arctic/scripts/etc/logs/arctic_dem_products_011824.log
exec 2> >(tee $LOGPATH)

sourceDirectory=/mnt/poseidon/remotesensing/arctic/data/rasters/arctic_dem

removeExtension() {
    file="$1"
    echo "$(echo "$file" | sed -r 's/\.[^\.]*$//')"
}

# loop through DEMs
for F in $(ls -1 "${sourceDirectory}"/*_dem.tif); do

  sourceFile="$(realpath "$F")"
  targetFile="${sourceDirectory}/$(removeExtension "$(basename ${F})")"

  # gdaldem hillshade -az 270 -alt 45 ${sourceFile} ${targetFile}_hillshade.tif
  # gdaldem slope ${sourceFile} ${targetFile}_slope.tif
  gdaldem aspect ${sourceFile} ${targetFile}_aspect.tif

done
