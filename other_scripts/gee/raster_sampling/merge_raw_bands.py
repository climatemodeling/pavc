#!/usr/bin/python3
# conda activate gee
# python3 S2_timeseries_merge.py

import os
import pandas as pd
import glob
import numpy as np

BUF=55
PC='parent'
SRC = 'topo'

# set input vars
UID = 'Site Code'
DIR = f'/mnt/poseidon/remotesensing/arctic/data/training/Test_05/temp/raw_s19_{SRC}bands_{PC}_{BUF}m'
DIR2 = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/fcover'

# set output vars
OUT = f'/mnt/poseidon/remotesensing/arctic/data/training/Test_05/features/{PC}'
NAME = f'{PC}_{SRC}'
DATE = f'summer_2019_med_{BUF}m'
BAND = None # specify band name ('ndvi') or derive from .csv base name (None)\
STAT = None # how you aggregated the timeseries or None

# vars
datecol = 'date'

##########################################################################################
# Create band.csv of entire year
##########################################################################################

# loop through folders in raw band dir and append dataframes
dfs = []
bandfolders = sorted(glob.glob(f'{DIR}/*'))

# loop through band name folders
for folder in bandfolders:
    
    if BAND == None and STAT == None:
        band = os.path.basename(folder)
    elif BAND == None and STAT != None:
        band = os.path.basename(folder) + f'_{STAT}'
    elif BAND != None and STAT == None:
        band = BAND
    else:
        band = f'{BAND}_{STAT}'
    
    # create lists
    datefolder = sorted(glob.glob(f'{folder}/*.csv'))
    ds = []
    uids = []
    
    # loop through dates in band file
    for f in datefolder:
        
        # read data for date
        csv = pd.read_csv(f, index_col=0)
        cols = csv.columns.to_list()
        print(cols)
        
        # if date_first column is present:
        if datecol in cols:

            # if all null date values, create null band column
            if csv[datecol].isnull().values.all():
                csv[band] = np.nan
                csv.drop(columns=[datecol], inplace=True)
            
            # if date column present, drop it and keep only band column
            else:
                csv.drop(columns=[datecol], inplace=True)
        
        # if no date column at all:
        else:
            csv[band] = np.nan

        ds.append(csv[[band]])
        uids.append(csv[UID])
    
    # create dataframe
    df = pd.concat(ds, axis=1)
    if not BAND:
        df = df
    else:
        dates = [os.path.basename(x).split('_')[0] for x in datefolder]
        df.columns = pd.to_datetime(dates)
    df.index = uids[0]
    df.index.astype('str')
    dfs.append(df)

if not BAND:
    df_final = pd.concat(dfs, axis=1)
    outpath = f'{OUT}/{NAME}_{DATE}.csv'
else:
    df_final = pd.concat(dfs, axis=1)
    df_final = df_final.groupby(df_final.columns, axis=1).sum()
    outpath = f'{OUT}/{NAME}_{band}_{DATE}.csv'
    
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
df_final.index = df_final.index.map(str)
outdf = pd.concat([df_final, all_sources], axis=1)
outdf.to_csv(outpath)

