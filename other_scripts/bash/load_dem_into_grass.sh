MAPSET=morgan #mapset must already be created
BANDS=(dem)
INPATH=/mnt/poseidon/remotesensing/arctic/data/rasters/arctic_dem
COLOR=rainbow

grasscr global_latlon ${MAPSET}

cd $INPATH
# loop through band folders
for b in "${BANDS[@]}"
do

    for F in ${INPATH}/*_${b}.tif; do

        # create grass map name
        filename=$(basename -- "$F")
        name="${filename%.*}"
        # printf "\n${name}\n"

        # covert to wgs84
        outfile=${INPATH}/${name}_01.tif
        if test -f  $outfile; then
            printf "GDALWARP not necessary. Skipping ...\n"
        else
            gdalwarp \
            -multi \
            -wm "50%" \
            -co NUM_THREADS=ALL_CPUS \
            -t_srs "+proj=longlat +datum=WGS84 +no_defs" \
            $F \
            ${INPATH}/${name}_01.tif
        fi

        # import tif into grass
        # r.in.gdal -e input=${INPATH}/${name}_01.tif output=$name memory=1000
        # r.colors map=$name color=$COLOR

    done

done