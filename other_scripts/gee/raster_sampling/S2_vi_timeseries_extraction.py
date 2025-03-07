#!/usr/bin/python3
# conda activate gee
# mpiexec -n 3 python3 S2_timeseries_extraction.py

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
FCOVER_PATH = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/fcover'
POINTBUFFER = 30 # meters
PC = 'child'

##########################################################################################

##########################################################################################

# output file
dp = '/mnt/poseidon/remotesensing/arctic/data/training/Test_05/temp'
DIR_PATH = f'{dp}/raw_ndvi_{PC}_{POINTBUFFER}m'
try:
    if os.path.isdir(DIR_PATH) == False:
        os.mkdir(DIR_PATH)
except Exception as e:
    pass

##########################################################################################

# data Information
IDCOL = 'Site Code'
SCALE = 10 # (int) scale in meters
BANDS = ['B4', 'B8'] # (list) band list
VI = 'ndvi'
NORM_DIFF_BANDS = ['B8', 'B4']
start_date = date(2019, 1, 1) # Y-M-D (2019, 1, 1)
end_date = date(2019, 12, 31) # Y-M-D minus 5 for even 'days' intervals (6 days for 2020)
TIMESTEP = 'months' # (str) 'months', 'days', or None
DAYS = '' # (int) if TIMESTEP = days
MONTHS = 12 # (int) if TIMESTEP = 'months': years * 12

# create timestamp directories within each huc (rank)
PATH = f'{DIR_PATH}/{VI.upper()}'
if os.path.isdir(PATH):
    pass
else:
    os.mkdir(PATH)

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
# AKVEG test 05
fi = f'VEG_fcover_{PC}.csv'
obs_data = pd.read_csv(f"{FCOVER_PATH}/{fi}")

# extract geometry and unique ID
akv_geom = obs_data[['latitude', 
                     'longitude', 
                     IDCOL]]
print(len(akv_geom))
akv_geom.columns = ['latitude', 'longitude', IDCOL]

##########################################################################################
# ABR_RS test 05
fi = f'ABR_fcover_{PC}.csv'
obs_data = pd.read_csv(f"{FCOVER_PATH}/{fi}")

# extract geometry and unique ID
abr_geom = obs_data[['latitude', 
                     'longitude', 
                     IDCOL]]
print(len(abr_geom))
abr_geom.columns = ['latitude', 'longitude', IDCOL]

##########################################################################################
# AKAVA test 05
fi = f'AVA_fcover_{PC}.csv'
obs_data = pd.read_csv(f"{FCOVER_PATH}/{fi}")

# extract geometry and unique ID
ava_geom = obs_data[['latitude', 
                     'longitude', 
                     IDCOL]]
print(len(ava_geom))
ava_geom.columns = ['latitude', 'longitude', IDCOL]

##########################################################################################
# NEON
fi = f'NEO_fcover_{PC}.csv'
obs_data = pd.read_csv(f"{FCOVER_PATH}/{fi}")

# extract geometry and unique ID
neo_geom = obs_data[['latitude', 
                     'longitude', 
                     IDCOL]]
print(len(neo_geom))
neo_geom.columns = ['latitude', 'longitude', IDCOL]

##########################################################################################
# Seward test 04
fi = f'SP_fcover_{PC}.csv'
obs_data = pd.read_csv(f"{FCOVER_PATH}/{fi}")

# extract geometry and unique ID
nge_geom = obs_data[['latitude', 
                     'longitude', 
                     IDCOL]]
print(len(nge_geom))
nge_geom.columns = ['latitude', 'longitude', IDCOL]

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
            .addBands(image.normalizedDifference(NORM_DIFF_BANDS).rename(VI))
            .addBands(ee.Image(date).rename('date').float())
           )


##########################################################################################
# Apply Cloud Mask, VI, and Date Functions
##########################################################################################
s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(samplepoints)
                .filterDate(str(start_date), str(end_date))
                #.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
            )

s2_sr = (s2_sr_col.map(mask_s2_clouds).map(add_variables)).select([VI, 'date'])

print(f"Cloud and shadow mask applied to tiles. {VI.title()} and Date bands created.")

##########################################################################################
# Sample Raster and Export Table
##########################################################################################

start = time.time()

# Loop through date ranges and export sampled composites
for RANGE in date_ranges:
    
    # select cloud-filtered sentinel 2 imagery for time step
    print(RANGE[0], RANGE[1])
    s2_by_date = s2_sr.filterDate(RANGE[0], RANGE[1])
    sentinel2 = s2_by_date.filterBounds(points_ee)

    # set band list
    vi_med = f'{VI}_median'
    b_list = [vi_med, 'date_first']

    # if image collection has images:
    if sentinel2.size().getInfo() != 0:

        # create composite for time step
        composite = (sentinel2.select(VI)).reduce(ee.Reducer.median())
        composite_date = (sentinel2.select('date')).reduce(ee.Reducer.first())
        composite = composite.addBands(composite_date)

        # sample composite with points (returns feature collection)
        sampled = composite.reduceRegions(collection = points_ee,
                                          reducer = ee.Reducer.median(),
                                          scale = SCALE,
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