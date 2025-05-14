#!/usr/bin/python3
# mpiexec -n 4 python3 water_mask.py

import numpy as np
import os
import glob
import rioxarray as rxr
import xarray as xr
from osgeo import gdal
from osgeo import gdalconst
from skimage import filters
from skimage import exposure
from datetime import datetime
from mpi4py import MPI
import logging

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H%M%S")

#########################################################################
# Parameters
#########################################################################

# General Params
BASE = '/mnt/poseidon/remotesensing/arctic/data'
OUT_DIR = f'{BASE}/rasters/model_results_tiled_test06'
DATA_DIR = f'{BASE}/rasters'
CELL_LIST = list(range(1077,4595))
REF_RAST = f'{DATA_DIR}/s2_sr_tiled/ak_arctic_summer/B11/2019-06-01_to_2019-08-31'
OVERWRITE = True

# Sensor-specific Params
S2_DIR = f'{DATA_DIR}/s2_sr_tiled/ak_arctic_summer/*/*'

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
        if scale_factor is not None:
            raster = raster * scale_factor
            
        # append to band list
        raster_bands.append(raster)

    merged = xr.merge(raster_bands)
    # drop pixel if any band is NA
    merged = merged.dropna(dim='band', how='any')
    return merged

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
# Loop and apply definitions to get water mask
#########################################################################

# loop through gridcells
for gridcell in grid_list:
    
    if OVERWRITE == False:
        # don't overwrite, stop here for current gridcell
        if os.path.isfile(f'{OUT_DIR}/GRIDCELL_{gridcell}_ndwimask.tif'):
            print(f'File exists. Skipping gridcell {gridcell}...')
            continue
        # otherwise, do the next steps
        else:
            print(f'Output does not exist. Overwriting existing gridcell {gridcell}...')
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
    reference_raster = rxr.open_rasterio(reference) * scale_factor
    
    #########################################################################
    # Sentinel 2
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
    # calc ndwi
    rast = s2_stacked_raster.assign(ndwi = lambda x: (x.green - x.nir)/(x.green + x.nir))
    rast = rast.assign(ndvi = lambda x: (x.nir - x.red)/(x.nir + x.red))
    rast = rast.assign(wetness = lambda x: 0.1509 * x.blue+0.1973* x.green+0.3279*x.red+0.3406*x.nir-0.7112*x.swir1 - 0.4572*x.swir2)
    
    # img = rast.isel(band=0).ndwi.to_numpy()
    # ndwi_val = filters.threshold_otsu(img)
    
    # create water mask
    # water_mask = rast.where(((rast.ndwi > 0) & (rast.ndvi < 0.3)), 0)
    # mask = water_mask.isel(band=0).ndwi
    # mask = mask.where((mask == 0), 1)
    
#     def exp_height_equation(x, a, b):
#         return a * np.exp(b * x)
    
#     rast = rast.astype(np.float32) # make sure i'm not saving at float64
#     mask = rast.where((rast.ndvi > 0.3), np.nan)
#     mask = mask.assign(height_cm = lambda x: exp_height_equation(x.wetness, 276.53, 31.2436))
#     mask = mask.isel(band=0).height_cm
    
#     # export xarray as tif
#     try:
#         mask = mask.rio.write_crs('EPSG:4326')
#         out_path = f'{OUT_DIR}/GRIDCELL_{gridcell}_height_cm.tif'
#         mask.rio.to_raster(out_path)
#         print(f"EXPORTED {out_path}")
#         logging.info(f"EXPORTED {out_path}")

#     except Exception as e:
#         print(f'EXCEPTION WAS RAISED WHILE EXPORTING {gridcell}: {e}')
#         logging.error(f'EXCEPTION WAS RAISED WHILE EXPORTING {gridcell}: ', exc_info=True)
#         continue

    bands = ['wetness', 'ndvi', 'ndwi']
    for b in bands:
        try:
            out_rast = rast[b]
            out_rast = out_rast.rio.write_crs('EPSG:4326')
            out_path = f'{OUT_DIR}/GRIDCELL_{gridcell}_{b}.tif'
            out_rast.rio.to_raster(out_path, dtype=np.float32)
            print(f"EXPORTED {out_path}")
            logging.info(f"EXPORTED {out_path}")
        except Exception as e:
            print(f'EXCEPTION WAS RAISED WHILE EXPORTING {gridcell}_{b}: {e}')
            logging.error(f'EXCEPTION WAS RAISED WHILE EXPORTING {gridcell}_{b}: ', exc_info=True)
            continue
            
        # set crs (doing it in xarray isn't enough for some reason)
        opts = gdal.WarpOptions(format='GTiff', dstSRS='EPSG:4326', outputType=gdalconst.GDT_Float32)
        gdal.Warp(out_path, out_path, options=opts)