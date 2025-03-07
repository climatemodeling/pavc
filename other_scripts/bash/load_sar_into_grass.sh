MAPSET=morgan #mapset must already be created
BANDS=(VV VH)
INPATH=/mnt/poseidon/remotesensing/arctic/data/rasters/S1GRD
COLOR=rainbow

grasscr global_latlon ${MAPSET}

# loop through band folders
for b in "${BANDS[@]}"
do

    for D in ${INPATH}/*/; do

        dirname=$(basename -- "$D")
        dir="${dirname%.*}"

        for F in ${D}*${b}.img; do

            # create grass map name
            bandname=$(basename -- "$F")
            band="${bandname%.*}"

            # import tif into grass
            echo ${dir}_${band}
            r.in.gdal --o -e input=$F output=${dir}_${band}
            r.colors map=${dir}_${band} color=$COLOR
        
        done

    done

done