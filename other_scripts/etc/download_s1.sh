#!/bin/bash

datetime=date +%s
LOGPATH=/mnt/poseidon/remotesensing/arctic/scripts/etc/logs/S1_Download_${datetime}.log
exec 2> >(tee $LOGPATH)

USERNAME=msteckler98
PASSWORD=E@rthd@t@1998!
BASE_PATH=/mnt/poseidon/remotesensing/arctic/data/rasters/s1_grd_zips
TXT_PATH=/mnt/poseidon/remotesensing/arctic/scripts/etc/s1_arctic_files.txt
OVERWRITE=no #yes or no

# loop through text file lines
cat $TXT_PATH | while read line
do

    zipfile="${BASE_PATH}/${line##*/}"

    # if you don't want to overwrite
    if [ $OVERWRITE = "no" ]; then

        # check if file already exists
        if [ -f $zipfile ]; then
            printf "\nSKIPPING ${zipfile}. File already exists.\n"
        # otherwise, download
        else
            printf "\nFILE DOES NOT EXIST. DOWNLOADING $zipfile\n"
            cd $BASE_PATH
            wget --user ${USERNAME} --password ${PASSWORD} $line
        fi

    # if you want to overwrite, go ahead and download
    elif [ $OVERWRITE = "yes" ]; then
        printf "\nOVERWRITING $zipfile ...\n"
        cd $BASE_PATH
        wget --user ${USERNAME} --password ${PASSWORD} $line

    # if yes/no is not specified
    else
        printf "Choose OVERWRITE=(yes/no)"
    fi
done