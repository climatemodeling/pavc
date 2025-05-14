# build list of file names between range of numbers
# for ((i=1077; i<=4594; i++)); do echo GRIDCELL_${i}.tif >> tifs_aoi.txt; done

# build vrt


# for many tifs
# path=/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled
# cd ${path}_01
# for file in ${path}/*.tif
# do 
#     # adjust NS resolution 
#     base="$(basename "$file" .tif)"
#     base="${base// /_}"
#     gdalwarp -overwrite -t_srs EPSG:4326 "${file}" "${base}.tif"
# done

# fancier gdalwarp (best for warping vrt into tif)
# gdalwarp \
#     -ot Int8 \
#     -of GTiff \
#     -t_srs "+proj=longlat +datum=WGS84 +no_defs" \
#     -co BIGTIFF=YES \
#     -co COMPRESS=DEFLATE \
#     $inrast \
#     $outrast