#!/bin/bash
# enable next line for debugging purpose
# set -x

syst_date=$(date +"%d-%m-%y-%H%M")
LOGPATH=/mnt/poseidon/remotesensing/arctic/scripts/snap_esa/logs/S1_Preprocessing_${syst_date}.log
exec 2> >(tee $LOGPATH)
TOTAL_START_TIME=$(date +%s)

############################################
# Script Information
############################################

# source: https://senbox.atlassian.net/wiki/spaces/SNAP/pages/70503475/Bulk+Processing+with+GPT
# usage: loops through a dir of images and applies a "graph" (snap workflow .xml) to each image
# output: processed images in a specified directory

############################################
# User Configuration
############################################

# path to snap and gpt executable
export PATH=/home/6ru/snap/bin:$PATH
gptPath="gpt"

############################################
# Command line handling
############################################

# first parameter is a path to the graph xml
graphXmlPath="$1"

# second parameter is a path to a parameter file (.properties) for graph parameters
parameterFilePath="$2"

# use third parameter for path to source products
sourceDirectory="$3"

# use fourth parameter for path to target products
targetDirectory="$4"

# the fifth parameter is a file prefix for the target product name, typically indicating the type of processing
# Orb_NR_Cal_TC
targetFilePrefix="$5"

############################################
# Helper functions
############################################
removeExtension() {
    file="$1"
    echo "$(echo "$file" | sed -r 's/\.[^\.]*$//')"
}

############################################
# Main processing
############################################

# Create the target directory if needed
if [ -d $targetDirectory ]
then
  echo TARGET DIRECTORY $targetDirectory EXISTS
else
  echo CREATING TARGET DIRECTORY $targetDirectory
  # mkdir -p "${targetDirectory}"
fi

# loop over zip files in directory
for file in ${sourceDirectory}/*.zip
do

  # create target file name
  sourceFile="$(realpath "$file")"
  targetFile="${targetDirectory}/${targetFilePrefix}_$(removeExtension "$(basename ${file})").dim"
  
  # do not overwrite existing files
  if [ -f $targetFile ]
  then
    echo FILE $targetFile ALREADY EXISTS
  else
    echo WORKING ON $sourceFile
    START_TIME=$(date +%s)
    
    # execute graph xml
    $gptPath $graphXmlPath -e -p $parameterFilePath -t $targetFile $sourceFile
    
    END_TIME=$(date +%s)
    SECS=$(($END_TIME - $START_TIME))
    echo TIME PASSED: $(($SECS / 60)) MINUTES
    echo EXPORTED TO $targetFile
  fi

done

TOTAL_END_TIME=$(date +%s)
echo TOTAL ELAPSED TIME: $(($TOTAL_END_TIME - $TOTAL_START_TIME / 60 / 60)) HOURS