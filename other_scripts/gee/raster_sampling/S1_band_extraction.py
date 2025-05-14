#!/usr/bin/python3
# conda activate gee
# mpiexec -n 4 python3 S2_band_extraction.py

import ee
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

import geemap
import os
import time
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import pandas as pd
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

##########################################################################################
# Set Parameters
##########################################################################################

# area of interest params
# choose bounding area format ('STATE', 'COUNTRY', 'BBOX', 'HUC', 'SHP'):
ROI = 'SHP'

# if ROI = BBOX or SHP (path to .geojson or .shp, otherwise ''):
b = '/mnt/poseidon/remotesensing/arctic/data/vectors/supplementary/tundra_alaska'
IN_PATH = f'{b}/tundra_alaska.shp'
# if ROI = STATE or COUNTRY (administrative boundaries, otherwise None):
COUNTRY = None
# if ROI = HUC, state abbreviation for HUC, if STATE, fulls state name:
STATE = None # 'AK' 
# if ROI = HUC (list of HUC6 units):
HUCLIST = None # must be list: ['190604', '190603', '190602']

##########################################################################################

# buffer around point to find median of intersecting pixel values
POINTBUFFER = 30 # meters
PC = 'child'

##########################################################################################

# output file
b2 = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/temp'
DIR_PATH = f'{b2}/raw_s19_topobands_{PC}_{POINTBUFFER}m'
try:
    if os.path.isdir(DIR_PATH) == False:
        os.mkdir(DIR_PATH)
except Exception as e:
    pass

##########################################################################################

# data Information
IDCOL = 'Site Code'
SCALE = 2
BANDS = ['slope', 'aspect', 'hillshade']
start_date = date(2019, 6, 1)# Y-M-D (2019, 1, 1)
end_date = date(2019, 8, 31) # Y-M-D minus 5 for even 'days' intervals (6 days for 2020)
TIMESTEP = None # 'months', 'days', or None
DAYS = '' # if TIMESTEP = days
MONTHS = '' # if TIMESTEP = 'months': years * 12

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
    print(date_ranges)

##########################################################################################
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
    print(date_ranges)

##########################################################################################
# no step

elif TIMESTEP == None:

    date_ranges = [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]

else:

    print("Invalid TIMESTEP selection. Use 'days', 'months', or None.")

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

print('RANK:', rank, CURRENTBANDS, flush = True)

# create timestamp directories within each huc (rank)
if os.path.isdir(PATH):
    pass
else:
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
    print('Invalid region of interest. Check STATE, COUNTRY, HUC')
    quit
    
##########################################################################################
# Get Sampling Points
##########################################################################################

##########################################################################################
# AKVEG test 04
di = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/fcover/'
fi = f'VEG_fcover_{PC}.csv'
obs_data = pd.read_csv(di + fi)

# extract geometry and unique ID
akv_geom = obs_data[['latitude', 
                     'longitude', 
                     'Site Code']]
print(len(akv_geom))
akv_geom.columns = ['latitude', 'longitude', 'Site Code']

##########################################################################################
# ABR_RS test 04
di = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/fcover/'
fi = f'ABR_fcover_{PC}.csv'
obs_data = pd.read_csv(di + fi)

# extract geometry and unique ID
abr_geom = obs_data[['latitude', 
                     'longitude', 
                     'Site Code']]
print(len(abr_geom))
abr_geom.columns = ['latitude', 'longitude', 'Site Code']

##########################################################################################
# AKAVA test 04
di = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/fcover/'
fi = f'AVA_fcover_{PC}.csv'
obs_data = pd.read_csv(di + fi)

# extract geometry and unique ID
ava_geom = obs_data[['latitude', 
                     'longitude', 
                     'Site Code']]
print(len(ava_geom))
ava_geom.columns = ['latitude', 'longitude', 'Site Code']

##########################################################################################
# NEON
di = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/fcover/'
fi = f'NEO_fcover_{PC}.csv'
obs_data = pd.read_csv(di + fi)

# extract geometry and unique ID
neo_geom = obs_data[['latitude', 
                     'longitude', 
                     'Site Code']]
print(len(neo_geom))
neo_geom.columns = ['latitude', 'longitude', 'Site Code']

##########################################################################################
# Seward test 04
di = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/fcover/'
fi = f'SP_fcover_{PC}.csv'
obs_data = pd.read_csv(di + fi)

# extract geometry and unique ID
nge_geom = obs_data[['latitude', 
                     'longitude', 
                     'Site Code']]
print(len(nge_geom))
nge_geom.columns = ['latitude', 'longitude', 'Site Code']

##########################################################################################
# combine
obs_geom = pd.concat([akv_geom, abr_geom, ava_geom, neo_geom, nge_geom], 
                     axis=0, 
                     ignore_index=True)
print(len(obs_geom))

# create ee object (feature collection)
obs_geom = obs_geom.reset_index()
obs_points = geemap.df_to_ee(obs_geom,
                             latitude='latitude',
                             longitude='longitude')
print(obs_points.size().getInfo())

##########################################################################################
#sub-select points and extract geometry

# select points that intercept HUC
samplepoints = obs_points.filterBounds(grid_location_ee)

# create dictionary of grid coordinates
points_dict = samplepoints.getInfo()
feats = points_dict['features']

# get ID column
unique_ids = []
for f in feats:
    id = f['properties'][IDCOL]
    unique_ids.append(id)

# Create a list of several ee.Geometry.Polygons
points = []
for f in feats:
    coords = f['geometry']['coordinates']
    point = ee.Geometry.Point(coords)
    # create buffer around point for later reduce regions
    buffered = point.buffer(POINTBUFFER)
    points.append(buffered)

# Make a feature collection for export purposes
points_ee = ee.FeatureCollection(points)
print(f'{len(points)} {POINTBUFFER}-meter buffered points.')

##########################################################################################
# Set Cloud Mask Function for S2-SR
##########################################################################################

def mask_edge(image):
    edge = image.lt(-30.0)
    masked_image = image.mask().And(edge.Not())
    return image.updateMask(masked_image)

def add_variables(image):
    # Compute time in fractional years since the epoch.
    date = ee.Date(image.get('system:time_start')).millis()
    # Return the image with the added bands.
    return (image
            .addBands(ee.Image(date).rename('date').float())
           )

##########################################################################################
# Apply Cloud Mask, NDVI, and Date Functions
##########################################################################################
s1_grd_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(samplepoints)
              .filterMetadata('instrumentMode', 'equals', 'IW')
              .filterDate(str(start_date), str(end_date)))

# choose standard pass direction to avoid noise issues
passdirection = s1_grd_col.aggregate_array('orbitProperties_pass').getInfo()
desc = len([d for d in passdirection if d == 'DESCENDING'])
asc = len([d for d in passdirection if d == 'ASCENDING'])
if desc > asc:
    direction = 'DESCENDING'
else:
    direction = 'ASCENDING'

# filter by pass direction
s1_grd_col = (s1_grd_col.filterMetadata('orbitProperties_pass', 
                                        'equals',
                                        direction))

everyband = [CURRENTBANDS]
everyband = everyband + ['date']
s1_grd = (s1_grd_col.map(mask_edge).map(add_variables)).select(everyband)

print("Cloud and shadow mask applied to tiles. Date band created.")

##########################################################################################
# Sample Raster and Export Table
##########################################################################################

start = time.time()

# Loop through date ranges and export sampled composites
for RANGE in date_ranges:
    
    # select cloud-filtered sentinel 2 imagery for time step
    print(RANGE[0], RANGE[1])
    s1_by_date = s1_grd.filterDate(RANGE[0], RANGE[1])
    sentinel1 = s1_by_date.filterBounds(points_ee)
    
    # get band names
    b_list = [CURRENTBANDS]
    b_list = [f'{b}_median' for b in b_list]
    b_list = b_list + ['date_first']
    print(b_list)

    # if image collection has images:
    if sentinel1.size().getInfo() != 0:

        # create composite for time step
        # ee.reduce(reducer) adds _reducer to band name
        composite = (sentinel1.select(CURRENTBANDS)).reduce(ee.Reducer.median())
        composite_date = (sentinel1.select('date')).reduce(ee.Reducer.first())
        composite = composite.addBands(composite_date)

        # sample composite using point buffers (returns feature collection)
        # gets median of all pixels in buffer
        print('Sampling...')
        sampled = composite.reduceRegions(points_ee,
                                          scale = SCALE,
                                          reducer = ee.Reducer.median(),
                                          crs = 'EPSG:4326')
        
        # export feature collection as csv
        FILE = f'{PATH}/{RANGE[0]}_to_{RANGE[1]}.csv'
        if os.path.isfile(FILE):
            print('FILE:', FILE, ' ALREADY EXISTS', flush=True)
        else:
            
            # export to dataframe
            df = ee_to_df(sampled, col_names=b_list)
            df[IDCOL] = unique_ids
            df.to_csv(FILE)
            print(f'Saved to {FILE}')

    # if image collection doesn't have images:
    else:
        
        FILE = f'{PATH}/{RANGE[0]}_to_{RANGE[1]}.csv'
        if os.path.isfile(FILE):
            print('FILE:', FILE, ' ALREADY EXISTS', flush=True)
        else:
            # create dataframe from original point data
            df = pd.DataFrame(unique_ids, columns=[IDCOL])
            df.loc[:, b_list] = np.nan
            df.to_csv(FILE)
            print(f'No images in collection. Saved dummy data to {FILE}')

stop = time.time()
time_taken = (stop - start)/3600
print('PROCESS COMPLETE! TOOK', round(time_taken, 3), 'HOURS')