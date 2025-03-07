#!/usr/bin/python3
# conda activate gee
# python3 S2_timeseries_merge.py

import os
import pandas as pd
import glob
import numpy as np

BUF=30
PC='child'
VI = 'ndvi'

# set input vars
UID = 'Site Code'
DIR = f'/mnt/poseidon/remotesensing/arctic/data/training/Test_05/temp/raw_{VI.lower()}_{PC}_{BUF}m'
DIR2 = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/fcover'

# set output vars
OUT = f'/mnt/poseidon/remotesensing/arctic/data/training/Test_05/features/{PC}'
NAME = f'{PC}_{VI.lower()}'
DATE = f'2019_med_{BUF}m'
BAND = None # specify band name ('ndvi') or derive from .csv base name (None)\
STAT = None # how you aggregated the timeseries or None

# vars
datecol = 'date_first'

##########################################################################################
# Create band.csv of entire year
##########################################################################################

# create lists
date_files = sorted(glob.glob(f'{DIR}/{VI.upper()}/*.csv'))
dataframes = []

# loop through dates in band file
for f in date_files:

    name = f.split('/')[-1]
    name = name[:-4] # drop ".csv"
    date = name.split('_')[0]
    
    # set up dataframe: reset index, drop date, rename ndvi
    df = pd.read_csv(f, index_col=0)
    df.set_index(UID, inplace=True)
    df.drop(columns=[datecol], inplace=True)
    bandcol = f'{VI.lower()}_median'
    df.rename(columns={bandcol:date}, inplace=True)

    dataframes.append(df)

# concat dataframes column-wise
final_df = pd.concat(dataframes, axis=1)
outpath = f'{OUT}/{NAME}_{DATE}.csv'
    
##########################################################################################
# Extract source information
##########################################################################################
    
# get source info
coverfiles = [f'{DIR2}/AVA_fcover_{PC}.csv', 
              f'{DIR2}/VEG_fcover_{PC}.csv', 
              f'{DIR2}/NEO_fcover_{PC}.csv',
              f'{DIR2}/ABR_fcover_{PC}.csv',
              f'{DIR2}/SP_fcover_{PC}.csv']

coverdfs = []
for coverfile in coverfiles:
    coverdf = pd.read_csv(coverfile, index_col=0)
    coverdf.index = coverdf.index.map(str)
    coverdfs.append(coverdf)

sources_dfs = []
for coverdf in coverdfs:
    source = coverdf[['source']]
    sources_dfs.append(source)

##########################################################################################
# Add source column to each band.csv
########################################################################################## 
    
# merge info
all_sources = pd.concat(sources_dfs)
final_df.index = final_df.index.map(str)
outdf = pd.concat([final_df, all_sources], axis=1)
outdf.to_csv(outpath)

