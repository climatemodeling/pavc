MAPSET=morgan #mapset must already be created
BANDS=(B5 B6 B7 B8A B11 B12)
MONTHS=("06")
INPATH=/mnt/poseidon/remotesensing/arctic/data/rasters/S2SR/ak_arctic_summer
START=0 #first tile of interest
END=4594 #last tile of interest

grasscr global_latlon ${MAPSET}


# loop through band folders
for b in "${BANDS[@]}"
do

  band_folder=${INPATH}/${b}

  # loop through month folders
  for m in "${MONTHS[@]}"
  do

    month_folder=`find ${band_folder}/ -type d -name 2019-${m}*`

    # loop through GRIDCELL tifs
    for(( i=$START; i<=$END; i++ ))
    do

      # create grass map name
      tif_file=${month_folder}/GRIDCELL_${i}.tif
      filename=$(basename -- "$tif_file")
      name="${filename%.*}"
      output_name=${name}_2019-${m}_${b}

      # import tif into grass
      r.in.gdal -e input=${tif_file} output=${output_name}
      r.colors map=${output_name} color=grey

      # grow region extent from first image
      if [ $i -eq 0 ]; then
        g.region rast=${output_name}
      fi

    done

  done

done