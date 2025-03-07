#!/home/6ru/anaconda3/envs/gee/bin/python3
# conda activate gee
# mpiexec -n 4 python3 S2_Importer.py

import ee
ee.Initialize()

import geemap
import os
import time
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import pandas as pd
import numpy as np
from mpi4py import MPI
import requests
import logging

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

##########################################################################################
# Set Parameters
##########################################################################################

# cloud filter params 
CLOUD_FILTER = 100

#----------------------------------------------------------------------------------------#

# (ROI) region of interest params
# choose bounding area format ('STATE', 'COUNTRY', 'BBOX', 'HUC', 'SHP'):
ROI = 'SHP'

# if ROI = BBOX or SHP (path to .geojson or .shp, otherwise ''):
IN_PATH = '/mnt/poseidon/remotesensing/arctic/data/vectors/supplementary/tundra_alaska_polygon_latlon/tundra_alaska_wgs84.shp'
# if ROI = STATE or COUNTRY (administrative boundaries, otherwise None):
COUNTRY = None
# if ROI = HUC, state abbreviation for HUC, if STATE, fulls state name:
STATE = None # 'AK' 
# if ROI = HUC (list of HUC6 units):
HUCLIST = None # must be list: ['190604', '190603', '190602']

#----------------------------------------------------------------------------------------#

# size (in m) of grid to superimpose on ROI
GRIDSIZE = 18000

#----------------------------------------------------------------------------------------#

# output file
DIR_PATH = '/mnt/poseidon/remotesensing/arctic/data/rasters/s2_sr_tiled/ak_arctic_summer'
if os.path.isdir(DIR_PATH) == False:
    os.mkdir(DIR_PATH)

#----------------------------------------------------------------------------------------#

# data Information
IDCOL = 'Site Code'
SCALE = 10
BANDS = ['B7']
start_date = date(2019, 6, 1)# Y-M-D (2019, 1, 1)
end_date = date(2019, 8, 31) # Y-M-D minus 5 for even 'days' intervals (6 days for 2020)
TIMESTEP = None # 'months', 'days', or None
DAYS = '' # if TIMESTEP = days
MONTHS = '' # if TIMESTEP = 'months': years * months
OVERWRITE = True # True to overwrite .tiff, False to pass existing .tiff

#b2POLY_LIST = [1643,2817,2818,2828,2982,3047,3140,3180,3527,3539,3785,3788,3837,3877,3901,3904,4242,4247,4272,4279,4317,4570,817,92] # list of ints or None
#b3POLY_LIST = [1993,2108,2526,2757,2761,3037,3044,3046,3558,3572,3786,3840,3881,3908,4048,4221,4232,4243,4246,4280,4321,4324,4326,4578,4580,95]
#b4POLY_LIST = [106,1706,2192,2410,2411,2560,2711,2751,2822,2996,3033,3040,3497,3543,3831,3837,4103,4132,4137,4240,4321,4439,4578]
#b7POLY_LIST = [310,313,318,350,374,376,377,378,380,389,390,391,568,569]
#b8POLY_LIST = [1381,2300,2301,2735,3255,3526,3528,3558,3571,3600,3601,3679,3837,4242,4243,4246,4288,4289,4309,4322,4334,4366,4367]
#b8aPOLY_LIST = [1658,2605,311,312,315,4325]
#POLY_LIST = [2499,328,356,359]
POLY_LIST = [3658]

# logging info
LOGNAME = f"AK_Arctic_S2_{str(SCALE)}m_dwnld_{'_'.join(BANDS)}"
logging.basicConfig(level = logging.INFO,
                    filename=f'{DIR_PATH}/{LOGNAME}.log', filemode='w',
                    format='%(asctime)s >>> %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')

##########################################################################################
# Create ee_to_df function for exporting
##########################################################################################

def ee_to_df(ee_object, col_names, sort_columns=False):
    if isinstance(ee_object, ee.Feature):
        ee_object = ee.FeatureCollection([ee_object])

    if not isinstance(ee_object, ee.FeatureCollection):
        raise TypeError("ee_object must be an ee.FeatureCollection")

    try:
        property_names = ee_object.first().propertyNames().sort().getInfo()
        #data = ee_object.map(lambda f: ee.Feature(None, f.toDictionary(property_names)))
        data = ee_object
        data = [x["properties"] for x in data.getInfo()["features"]]
        df = pd.DataFrame(data)

        if col_names is None:
            col_names = property_names
            col_names.remove("system:index")
        elif not isinstance(col_names, list):
            raise TypeError("col_names must be a list")

        df = df[col_names]

        if sort_columns:
            df = df.reindex(sorted(df.columns), axis=1)

        return df
    
    except Exception as e:
        raise Exception(e)

##########################################################################################
# Set Date Ranges
##########################################################################################

# days
if TIMESTEP == 'days':

    def create_list_of_dates(start_date, end_date):
        dates = []
        delta = end_date - start_date   # returns timedelta

        for i in range(delta.days + 1):
            day = start_date + timedelta(days=i)
            dates.append(day)
        return dates

    def create_time_intervals(dates_list, Interval):
        time_df = pd.DataFrame({'Date': dates_list}).astype('datetime64[ns]')
        interval = timedelta(Interval)
        grouped_cr = time_df.groupby(pd.Grouper(key='Date', freq=interval))
        date_ranges = []
        for i in grouped_cr:
            date_ranges.append(((str(i[1].min()[0]).split(' ')[0]), 
                                (str(i[1].max()[0]).split(' ')[0])))
        return date_ranges

    date_ranges = create_time_intervals(create_list_of_dates(start_date, 
                                                             end_date), 
                                        DAYS)

#----------------------------------------------------------------------------------------#

# months
elif TIMESTEP == 'months':

    def create_list_of_dates(start_date, end_date):

        dates = []
        end_date = end_date - relativedelta(months=MONTHS-1)
        for i in range(MONTHS):
            delta = relativedelta(months=i)
            month_start = start_date + delta
            month_end = end_date + delta
            dates.append((month_start.strftime('%Y-%m-%d'), 
                          month_end.strftime('%Y-%m-%d')))
        return dates

    date_ranges = create_list_of_dates(start_date, end_date)

#----------------------------------------------------------------------------------------#

# no step
elif TIMESTEP == None:

    date_ranges = [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]

else:

    logging.critical("Invalid TIMESTEP selection. Use 'days', 'months', or None.")
    quit

##########################################################################################
# Set Ranks by Number of bands
##########################################################################################

# create band directories
allbands = np.array(BANDS)
allbands = np.array_split(allbands, size) # split array into x pieces
for r in range(len(allbands)):
    if r == rank:
        CURRENTBANDS = BANDS[r] # select object from list (could be str or lst)
        PATH = f'{DIR_PATH}/{CURRENTBANDS}'
    else:
        pass

# create timestamp directories within each main DIR_PATH
if os.path.isdir(PATH) == False:
    os.mkdir(PATH)

##########################################################################################
# Set GEE Vector Bounds
##########################################################################################

# Import admin data and select country to create grid around
if ROI == 'STATE':
    grid_location_ee = (ee.FeatureCollection("FAO/GAUL/2015/level1")
                        .filterMetadata('ADM0_NAME', 'equals', COUNTRY)
                        .filterMetadata('ADM1_NAME', 'equals', STATE))

elif ROI == 'COUNTRY':
    grid_location_ee = (ee.FeatureCollection("FAO/GAUL/2015/level1")
                        .filterMetadata('ADM0_NAME', 'equals', COUNTRY))
	
elif ROI == 'BBOX':
	grid_location_ee = geemap.geojson_to_ee(IN_PATH)
    
elif ROI == 'HUC':
    grid_location_ee = (ee.FeatureCollection("USGS/WBD/2017/HUC06")
                        .filter(ee.Filter.inList('huc6', HUCLIST)))
    
elif ROI == 'SHP':
    geodataframe = gpd.read_file(IN_PATH)
    grid_location_ee = geemap.geopandas_to_ee(geodataframe)
    
else:
    logging.critical('Invalid region of interest. Check ROI parameter.')
    quit
		
##########################################################################################
# Create grid superimposed on HUCs
##########################################################################################

# Create grid
# https://developers.google.com/earth-engine/tutorials/community/drawing-tools

grid = grid_location_ee.geometry().coveringGrid(proj='EPSG:4326', scale=GRIDSIZE)

# Create dictionary of grid coordinates
grid_dict = grid.getInfo()
feats = grid_dict['features']

# Create a list of several ee.Geometry.Polygons
polys = []
for d in feats:
    coords = d['geometry']['coordinates']
    poly = ee.Geometry.Polygon(coords)
    polys.append(poly)

num_km = GRIDSIZE / 1000
logging.info(f'{len(polys)} {num_km}-km grid cells superimposed on {ROI}.')

##########################################################################################
# Set Cloud Mask Function for S2-SR
##########################################################################################

def mask_s2_clouds(image):
    # select bitmask band
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0).And
            (qa.bitwiseAnd(cirrus_bit_mask).eq(0)))

    return image.updateMask(mask)

def add_variables(image):
    # Compute time in fractional years since the epoch.
    date = ee.Date(image.get('system:time_start')).millis()
    # Return the image with the added bands.
    return (image
            .addBands(ee.Image(date).rename('date').float())
           )

##########################################################################################
# Sample Raster and Export Table
##########################################################################################

start = time.time()

# get band names
# ee.reduce(reducer) adds '_reducer' to band name
b_list = [CURRENTBANDS]
b_list = [f'{b}_median' for b in b_list]
logging.info('Rank {rank} bands: {b_list}')

# Loop through date ranges and export sampled composites
for RANGE in date_ranges:

    date_dir = f'{PATH}/{RANGE[0]}_to_{RANGE[1]}'
    if os.path.isdir(date_dir) == False:
        os.mkdir(date_dir)

    if POLY_LIST != None:

        print(f'Working with specific list of polys: {POLY_LIST}')

        polys = [ polys[i] for i in POLY_LIST]
        for grid_num, poly in zip(POLY_LIST, polys):

            logging.info(f'Rank: {rank} \nGrid number: {grid_num}\nDate range: {RANGE[0]} to {RANGE[1]}')
            print(f'Rank: {rank} \nGrid number: {grid_num}\nDate range: {RANGE[0]} to {RANGE[1]}')

            # set export file name
            FILE = f'{date_dir}/GRIDCELL_{grid_num}.tif'

            if OVERWRITE == False:
                if os.path.isfile(FILE):
                    logging.info(f'{FILE} ALREADY EXISTS. SKIPPED.')
                    print(f'{FILE} ALREADY EXISTS. SKIPPED.')
                    continue # go to next loop iteration 

            # apply cloud mask and date band
            s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(poly)
                        .filterDate(str(RANGE[0]), str(RANGE[1]))
                        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
                    )

            # apply cloud mask & select current rank bands
            s2_sr = (s2_sr_col.map(mask_s2_clouds).map(add_variables)).select(CURRENTBANDS)

            # IF IMAGE COLLECTION HAS IMAGES:
            if s2_sr.size().getInfo() != 0:

                # create median composite for time step
                composite = (s2_sr.select(CURRENTBANDS)).reduce(ee.Reducer.median())
                composite = composite.clip(poly)

                # if no overwriting allowed and file exists, move on to next loop iteration
                if OVERWRITE == False:
                    if os.path.isfile(FILE):
                        logging.info(f'{FILE} ALREADY EXISTS. SKIPPED.')
                        continue # go to next loop iteration            

                try:
                        
                    # export multi-band GeoTIFF file of composite
                    url = composite.getDownloadUrl({'bands': b_list,
                                                    'region': poly,
                                                    'scale': SCALE,
                                                    'format': 'GEO_TIFF',
                                                    'filePerBand': False})

                    # download
                    response = requests.get(url)
                    with open(FILE, 'wb') as fd:
                        fd.write(response.content)
                        logging.info(f'SUCCESS: {FILE}')
                        print(f'SUCCESS: {FILE}')

                # if GEE raises an error (probably user memory limit)
                except Exception as e:
                    logging.error('EXCEPTION RAISED. SKIPPING ...', exc_info=True)
                    print(f'EXCEPTION RAISED FOR: {FILE} SKIPPING ...')

            # IF IMAGE COLLECTION DOESNT HAVE IMAGES:
            else:

                # create dummy composite for bands
                composite = ee.Image(-999).rename(b_list[0])
                if len(b_list) > 0:
                    for b in b_list[1:]:
                        composite = composite.addBands(ee.Image(-999)).rename(b_list[b])
                            
                try:

                    # download dummy tif file
                    composite = composite.clip(poly)
                    url = composite.getDownloadUrl({'bands': b_list,
                                                    'region': poly,
                                                    'scale': SCALE,
                                                    'format': 'GEO_TIFF',
                                                    'filePerBand': False})
                    # download
                    response = requests.get(url)
                    with open(FILE, 'wb') as fd:
                        fd.write(response.content)
                        logging.info(f'NO SENTINAL DATA TO CREATE COMPOSITE. SAVED DUMMY AT: {FILE}')
                        print(f'NO SENTINAL DATA TO CREATE COMPOSITE. SAVED DUMMY AT: {FILE}')
                    
                # if the dummy tif fails for some  GEE reason, give up.
                except Exception as e:
                    logging.error('EXCEPTION RAISED. SKIPPING ...', exc_info=True)
                    print(f'EXCEPTION RAISED. SKIPPING ...')

    else:

        print(f'Looping through all polys...')

        for grid_num, poly in enumerate(polys):

            logging.info(f'Rank: {rank} \nGrid number: {grid_num}\nDate range: {RANGE[0]} to {RANGE[1]}')
            print(f'Rank: {rank} \nGrid number: {grid_num}\nDate range: {RANGE[0]} to {RANGE[1]}')

            # set export file name
            FILE = f'{date_dir}/GRIDCELL_{grid_num}.tif'

            if OVERWRITE == False:
                if os.path.isfile(FILE):
                    logging.info(f'{FILE} ALREADY EXISTS. SKIPPED.')
                    print(f'{FILE} ALREADY EXISTS. SKIPPED.')
                    continue # go to next loop iteration 

            # apply cloud mask and date band
            s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(poly)
                        .filterDate(str(RANGE[0]), str(RANGE[1]))
                        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
                    )

            # apply cloud mask & select current rank bands
            s2_sr = (s2_sr_col.map(mask_s2_clouds).map(add_variables)).select(CURRENTBANDS)

            # IF IMAGE COLLECTION HAS IMAGES:
            if s2_sr.size().getInfo() != 0:

                # create median composite for time step
                composite = (s2_sr.select(CURRENTBANDS)).reduce(ee.Reducer.median())
                composite = composite.clip(poly)

                # if no overwriting allowed and file exists, move on to next loop iteration
                if OVERWRITE == False:
                    if os.path.isfile(FILE):
                        logging.info(f'{FILE} ALREADY EXISTS. SKIPPED.')
                        continue # go to next loop iteration            

                try:
                        
                    # export multi-band GeoTIFF file of composite
                    url = composite.getDownloadUrl({'bands': b_list,
                                                    'region': poly,
                                                    'scale': SCALE,
                                                    'format': 'GEO_TIFF',
                                                    'filePerBand': False})

                    # download
                    response = requests.get(url)
                    with open(FILE, 'wb') as fd:
                        fd.write(response.content)
                        logging.info(f'SUCCESS: {FILE}')
                        print(f'SUCCESS: {FILE}')

                # if GEE raises an error (probably user memory limit)
                except Exception as e:
                    logging.error('EXCEPTION RAISED. SKIPPING ...', exc_info=True)
                    print(f'EXCEPTION RAISED FOR: {FILE} SKIPPING ...')

            # IF IMAGE COLLECTION DOESNT HAVE IMAGES:
            else:

                # create dummy composite for bands
                composite = ee.Image(-999).rename(b_list[0])
                if len(b_list) > 0:
                    for b in b_list[1:]:
                        composite = composite.addBands(ee.Image(-999)).rename(b_list[b])
                            
                try:

                    # download dummy tif file
                    composite = composite.clip(poly)
                    url = composite.getDownloadUrl({'bands': b_list,
                                                    'region': poly,
                                                    'scale': SCALE,
                                                    'format': 'GEO_TIFF',
                                                    'filePerBand': False})
                    # download
                    response = requests.get(url)
                    with open(FILE, 'wb') as fd:
                        fd.write(response.content)
                        logging.info(f'NO SENTINAL DATA TO CREATE COMPOSITE. SAVED DUMMY AT: {FILE}')
                        print(f'NO SENTINAL DATA TO CREATE COMPOSITE. SAVED DUMMY AT: {FILE}')
                    
                # if the dummy tif fails for some  GEE reason, give up.
                except Exception as e:
                    logging.error('EXCEPTION RAISED. SKIPPING ...', exc_info=True)
                    print(f'EXCEPTION RAISED. SKIPPING ...')

stop = time.time()
time_taken = (stop - start)/3600
print(f'PROCESS COMPLETE! IT TOOK {round(time_taken, 3)} HOURS')