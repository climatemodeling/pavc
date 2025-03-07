#!/usr/bin/python3
# mpiexec -n 4 python3 northslope_pft_mapping.py

import numpy as np
import pandas as pd
import os
import glob
import rioxarray as rxr
import xarray as xr
import pickle
from mpi4py import MPI
import logging
from osgeo import gdal
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H%M%S")

#########################################################################
# Parameters
#########################################################################

# General Params
OVERWRITE = True
BASE = '/mnt/poseidon/remotesensing/arctic/data'
OUT_DIR = f'{BASE}/rasters/model_results_tiled_test07'
DATA_DIR = f'{BASE}/rasters'
CELL_LIST = list(range(1077,4595))
REF_RAST = f'{DATA_DIR}/s2_sr_tiled/ak_arctic_summer/B11/2019-06-01_to_2019-08-31'
MODEL = f'{BASE}/training/Test_06/results/ModelTuning_FeatureImportance'
PFTS = ['evergreen shrub', 'forb', 'graminoid', 'litter']

# Sensor-specific Params
S2_DIR = f'{DATA_DIR}/s2_sr_tiled/ak_arctic_summer/*/*'
S1_DIR = f'{DATA_DIR}/s1_grd_tiled'
DEM_DIR = f'{DATA_DIR}/acrtic_dem_tiled'
print('Number of gridcells to work on:', len(CELL_LIST))

# parallel processing
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#########################################################################
# Definitions
#########################################################################

# function that returns list of paths to gridcell
def gridcell_rast_path(gridcell, directory):
    path = sorted(glob.glob(f'{directory}/GRIDCELL_{gridcell}*'))
    return path

# function to stack sensor bands for one gridcell
# will need to loop through each sensor and gridcell
def stack_bands(sensor, cell_num,
                resample_bands, ref_rast, scale_factor=None):
    
    """
    Creates an xarray with each band recorded as a variable.
    sensor         : [str] sensor source of data (s2_sr, s1_grd, or dem)
    cell_num       : [int] gridcell number to analyze
    resample_bands : [list] bands that need to be rescaled to 20-meters
    ref_rast       : [xr.Dataset] raster used as the model resolution/crs
    scale_factor   : [float or int] number multiplied to rescale data
    Returns an xr.Dataset with x,y,band dimensions for one gridcell with 
    each band as a data variable that matches the resolution/scale of the
    reference raster.
    """

    raster_bands = []
    for band_path in cell_num:

        # get band name from file path
        if sensor == 's2_sr':
            b_name = band_path.split('/')[-3]
        elif sensor == 'dem':
            b_name = band_path.split('/')[-1]
            b_name = b_name.split('.')[0]
            b_name = b_name.split('_')[-1]
        elif sensor == 's1_grd':
            b_name = band_path.split('/')[-1]
            b_name = b_name.split('.')[0]
            b_name = b_name.split('_')[-1]
        else:
            print('Incorrect sensor choice. Try dem, s2_sr, or s1_grd.')
            logging.critical('Incorrect sensor choice. Try dem, s2_sr, or s1_grd.')
            break
        
        # open raster in xarray
        raster = rxr.open_rasterio(band_path)
        raster.name = b_name
        
        # resample and rescale if necessary
        # if b_name in resample_bands:
            # print(f'Rescaling {b_name}...')
        raster = raster.rio.reproject_match(ref_rast)
        raster = raster.where(raster != 0, -9999)
        if scale_factor is not None:
            raster = raster * scale_factor
            
        # append to band list
        raster_bands.append(raster)

    merged = xr.merge(raster_bands)
    # drop pixel if any band is NA
    merged = merged.dropna(dim='band', how='any')
    return merged
    
# function that creates new veg idx data variables for an xr
def calc_veg_idx_s2(xrd):
    
    """
    Creates new data attributes for an s2_sr xr.Dataset with bands
    B2, B3, B4, B5, B6, B8, B8A, B11, and B12. Second step after 
    stack_bands. S2_sr data must be scaled from 0 to 1; can set
    scale factor in stack_bands function if necessary.
    xrd : [xr.Dataset] s2_sr xarray dataset
    Returns: xarray dataset with new vegetation indices
    """
    
    xrd = xrd.assign(ndwi1 = lambda x: (x.nir - x.swir1)/(x.nir + x.swir2))
    xrd = xrd.assign(ndwi2 = lambda x: (x.nir - x.swir2)/(x.nir + x.swir2))
    xrd = xrd.assign(msavi = lambda x: (2*x.nir + 1 -  ((2*x.nir + 1)**2 - 8*(x.nir - x.red))**0.5) * 0.5)
    xrd = xrd.assign(vari = lambda x: (x.green - x.red)/(x.green + x.red - x.blue))
    xrd = xrd.assign(rvi = lambda x: x.nir/x.red)
    xrd = xrd.assign(osavi = lambda x: 1.16 * (x.nir - x.red)/(x.nir + x.red + 0.16))
    xrd = xrd.assign(tgi = lambda x: (120 * (x.red - x.blue) - 190 * (x.red - x.green))*0.5)
    xrd = xrd.assign(gli = lambda x: (2 * x.green - x.red - x.blue)/(2 * x.green + x.red + x.blue))
    xrd = xrd.assign(ngrdi = lambda x: (x.green - x.red)/(x.green + x.red))
    xrd = xrd.assign(ci_g = lambda x: x.nir/x.green - 1)
    xrd = xrd.assign(gNDVI = lambda x: (x.nir - x.green)/(x.nir + x.green))
    xrd = xrd.assign(cvi = lambda x: (x.nir * x.red)/(x.green ** 2))
    xrd = xrd.assign(mtvi2 = lambda x: 1.5*(1.2*(x.nir - x.green) - 2.5*(x.red - x.green))/(((2*x.nir + 1)**2 - (6*x.nir - 5*(x.red**0.5))-0.5)**0.5))
    xrd = xrd.assign(brightness = lambda x: 0.3037 * x.blue +0.2793 * x.green +0.4743 * x.red +0.5585 * x.nir +0.5082 * x.swir1 + 0.1863 * x.swir2)
    xrd = xrd.assign(greenness = lambda x: 0.7243 * x.nir +0.0840 * x.swir1 - 0.2848 * x.blue - 0.2435 * x.green - 0.5436 * x.red - 0.1800 * x.swir2)
    xrd = xrd.assign(wetness = lambda x: 0.1509 * x.blue+0.1973* x.green+0.3279*x.red+0.3406*x.nir-0.7112*x.swir1 - 0.4572*x.swir2)
    xrd = xrd.assign(tcari = lambda x: 3 * ((x.redEdge1 - x.red)-0.2 * (x.redEdge1 - x.green)*(x.redEdge1/x.red)))
    xrd = xrd.assign(tci = lambda x: 1.2 * (x.redEdge1 - x.green)- 1.5 * (x.red - x.green)*((x.redEdge1/x.red)**0.5))
    xrd = xrd.assign(nari = lambda x: (1/x.green - 1/x.redEdge1)/(1/x.green + 1/x.redEdge1))

    return xrd


#########################################################################
# Parallelism
#########################################################################

# create output directory
processes = np.array(CELL_LIST)
split_processes = np.array_split(processes, size) # split array into x pieces
for p_idx in range(len(split_processes)):
    if p_idx == rank:
        grid_list = split_processes[p_idx] # select current list of gridcells
    else:
        pass

logging.basicConfig(level = logging.INFO,
                    filename=f'{OUT_DIR}/std_{grid_list[0]}_to_{grid_list[-1]}_{dt_string}.log', filemode='w',
                    format='%(asctime)s >>> %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')

print(f'CURRENT RANK: {rank}')
print(f'GRIDCELLS: {grid_list[0]} to {grid_list[-1]}')
logging.info(f'CURRENT RANK: {rank}')
logging.info(f'GRIDCELLS: {grid_list[0]} to {grid_list[-1]}')
    
#########################################################################
# Begin modeling
#########################################################################
    
# example model output dataframe for reference creating df to give to model
demo_df = pd.read_csv(f'{MODEL}/FeatureDemo_non-vascular.csv', index_col=0)
col_order = demo_df.columns.tolist() # this var is used near line 266

# loop through gridcells
for gridcell in grid_list:
    
    # overwrite or don't overwrite
    last_pft = f"{OUT_DIR}/GRIDCELL_{gridcell}_non-vascular.tif"
    
    if OVERWRITE == False:
        # don't overwrite, stop here for current gridcell
        if os.path.isfile(last_pft):
            print(f'All files exist for a gridcell, skipping {gridcell}...')
            continue
        # otherwise, do the next steps
        else:
            print(f'Output does not exist, so overwriting existing {gridcell}...')
            pass       
    elif OVERWRITE == True:
        # overwrite existing tifs
        pass
    else:
        logging.critical('Choose OVERWRITE = True or OVERWRITE = False.')
        quit()

    # set loop vars
    reference = f'{REF_RAST}/GRIDCELL_{gridcell}.tif'
    scale_factor = 0.0001 # for rescaling S2 band values
    # reference raster is from S2, hence the scaling for good measure
    reference_raster = rxr.open_rasterio(reference)
    reference_raster = reference_raster.where(reference_raster != 0, -9999)
    reference_raster = reference_raster * scale_factor

    #########################################################################
    # Sentinel 2 (nodata = -9999)
    #########################################################################

    # create 20-m xarray raster
    rast_path = gridcell_rast_path(gridcell, S2_DIR)
    rescale_bands = ['B2', 'B3', 'B4', 'B8'] # these are 10-m bands
    s2_stacked_raster = stack_bands('s2_sr', rast_path, 
                                    rescale_bands, reference_raster, scale_factor)
    
    # rename bands to something legible
    s2_stacked_raster = s2_stacked_raster.rename({'B2':'blue', 
                                                  'B3':'green', 
                                                  'B4':'red', 
                                                  'B5':'redEdge1', 
                                                  'B6':'redEdge2', 
                                                  'B7':'redEdge3', 
                                                  'B8A':'redEdge4', 
                                                  'B8':'nir',
                                                  'B11':'swir1',
                                                  'B12':'swir2'})
    
    # calculate vegetation indices
    s2_stacked_raster_veg = calc_veg_idx_s2(s2_stacked_raster)


    #########################################################################
    # Sentinel 1 (nodata = -9999)
    #########################################################################

    # create 20-m xarray raster
    s1_rast_path = gridcell_rast_path(gridcell, S1_DIR)
    rescale_bands = ['VV', 'VH']
    s1_stacked_raster = stack_bands('s1_grd', s1_rast_path,
                                    rescale_bands, reference_raster)
    s1_stacked_raster = s1_stacked_raster.where(s1_stacked_raster[rescale_bands] != 0, -9999)


    #########################################################################
    # Arctic DEM (nodata = -9999)
    #########################################################################

    # create 20-m xarray raster
    dem_rast_path = gridcell_rast_path(gridcell, DEM_DIR)
    rescale_bands = ['aspect', 'dem', 'hillshade', 'slope']
    dem_stacked_raster = stack_bands('dem', dem_rast_path, 
                                     rescale_bands, reference_raster)
    dem_stacked_raster = dem_stacked_raster.rename({'dem':'elevation'})
    
    # set nodata value
    rescale_bands2 = ['aspect', 'elevation', 'hillshade', 'slope']
    # dem_stacked_raster = dem_stacked_raster.where(dem_stacked_raster[rescale_bands2] != -9999., 0.)


    #########################################################################
    # Combine into one xarray
    #########################################################################

    # make sure pandas df features are in the right order
    stacked_raster = xr.merge([s2_stacked_raster_veg, 
                               s1_stacked_raster, 
                               dem_stacked_raster])

    # get coordinate information from raster as df
    df = stacked_raster.to_dataframe()
    coords = df.reset_index()
    coords = coords[['x', 'y']]

    # get raster data as df
    df = df.droplevel([1, 2]).reset_index(drop=True)
    df = df.iloc[:,1:]
    # df = df.astype("float32")
    
    # find any bands that were divided by 0 and produced an inf value
    bad_idx_list = df[np.isinf(df.values)].index.tolist()
    df.drop(index=bad_idx_list, inplace=True)
    coords.drop(index=bad_idx_list, inplace=True)

    # remove straggling nans
    nan_idx_list = df[np.isnan(df.values)].index.tolist()
    df.drop(index=nan_idx_list, inplace=True)
    coords.drop(index=nan_idx_list, inplace=True)

    #########################################################################
    # Apply model
    #########################################################################
    
    for PFT in PFTS:
        
        try:

            ## Load the pickled model from the file
            model_file_path = f'{MODEL}/tunedModel_{PFT}.pkl'
            # sort columns to match what is expected by pkl file
            df = df[col_order] # col_order is near line 179
            with open(model_file_path, 'rb') as model_file:
                model = pickle.load(model_file)

            # --- prediction directly using the model
            fcover = model.predict(df) # fcover is a 1 by n 
            
        except Exception as e:
            print(f'EXCEPTION WAS RAISED WHILE MODELING {gridcell}: {e}')
            logging.error(f'EXCEPTION WAS RAISED WHILE MODELING {gridcell}: ', exc_info=True)
            continue

        
        #########################################################################
        # Export modeled tif
        #########################################################################

        # set up df for xarray
        results = coords.copy()
        results['fcover'] = fcover
        results['band'] = 1
        
        # export xarray as tif
        try:
            PFT = PFT.replace(" ", "_")
            results_xr = xr.Dataset.from_dataframe(results.set_index(['band', 'y', 'x']))
            xr_band = results_xr.isel(band=0)
            xr_band = xr_band.rio.write_crs('EPSG:4326')
            out_path = f"{OUT_DIR}/GRIDCELL_{gridcell}_{PFT}.tif"
            xr_band.rio.to_raster(out_path)
            print(f"EXPORTED {out_path}")
            logging.info(f"EXPORTED {out_path}")
            
        except Exception as e:
            print(f'EXCEPTION WAS RAISED WHILE EXPORTING {gridcell}: {e}')
            logging.error(f'EXCEPTION WAS RAISED WHILE EXPORTING {gridcell}: ', exc_info=True)
            continue
            
        # set crs
        nodata_value = -9999
        opts = gdal.WarpOptions(format='GTiff', dstSRS='EPSG:4326', dstNodata=nodata_value)
        gdal.Warp(out_path, out_path, options=opts)