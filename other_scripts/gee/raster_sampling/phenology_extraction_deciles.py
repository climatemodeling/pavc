#!/usr/bin/python3

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# set vars
DIR = '/mnt/poseidon/remotesensing/arctic/data/training/Test_03/harmonic_analysis/'
FILE = 'harmonicFitted_ndvi_NEON.csv'
PATH = DIR + FILE
SDATE = "2019-04-01"
EDATE = "2019-10-31"
P_RANGE = np.arange(.10,1.01,.10)

##########################################################################################
# Extract timeseries data
##########################################################################################

# read and extract timeseries columns
harmonic_orig = pd.read_csv(PATH)
harmonic = harmonic_orig.filter(regex='_fitted')

# set datetime objects and rescale NDVI
harmonic.columns = pd.to_datetime(harmonic.columns, format='%Y_%m_fitted')
harmonic = harmonic * .0001

##########################################################################################
# Apply cubic spline
##########################################################################################

# orusa et al spline function
def function1(timeseries, dates):

    steplist = np.arange(1, len(timeseries)-2, 1).tolist()

    def function(ii):

        p0, d0 = timeseries[ii-1], dates[ii-1] # ndvi/date at index 0
        p1, d1 = timeseries[ii], dates[ii]     # ndvi/date at index 1
        p2, d2 = timeseries[ii+1], dates[ii+1] # ndvi/date at index 2
        p3, d3 = timeseries[ii+2], dates[ii+2] # ndvi/date at index 3

        # difference in days between dates
        diff01 = (d1 - d0).astype('timedelta64[D]') #in days
        diff01 = int(diff01 / np.timedelta64(1, 'D'))
        diff12 = (d2 - d1).astype('timedelta64[D]') #in days
        diff12 = int(diff12 / np.timedelta64(1, 'D'))
        diff23 = (d3 - d2).astype('timedelta64[D]') #in days
        diff23 = int(diff23 / np.timedelta64(1, 'D'))

        # normalize days
        diff01nor = diff01 / diff12 # 1 - 0 / 2 - 1
        diff12nor = diff12 / diff12 # 2 - 1 / 2 - 1
        diff23nor = diff23 / diff12 # 3 - 2 / 2 - 1

        # prep vars
        f0 = p1
        f1 = p2
        f0p = (p2 - p0) / (diff01nor + diff12nor) # ndvi 2 - ndvi 0 / days
        f1p = (p3 - p1) / (diff12nor + diff23nor) # ndvi 3 - ndvi 1 / days

        # spline vars
        a = (f0*2) - (f1*2) + f0p + f1p
        b = (f0*-3) + (f1*3) - (f0p*2) - f1p
        c = f0p
        d = f0

        # create spline function
        xValues = np.arange(0, diff12-1, 1) # daily step
        xDates = pd.date_range(d1, d2, freq='D') # daily step
        # xValues = np.arange(0, (diff12-1)*24, 1) # hourly step
        # xDates = pd.date_range(d1, d2, freq='H')
        
        def daily_spline(x): 
            val = x / diff12
            return ((val**3) * a) + ((val**2) * b) + (val*c) + d, xDates[x]

        interpolated = []
        interpolated_dates = []
        # apply spline            
        for x in xValues:
            interp, date = daily_spline(x)
            interpolated.append(interp)
            interpolated_dates.append(date)

        return interpolated, interpolated_dates
    
    interp = []
    interp_dates = []
    for step in steplist:
        i, d = function(step) # list of interp images
        interp.append(i)
        interp_dates.append(d)
        
    def flatten(l):
        return [item for sublist in l for item in sublist]
        
    return flatten(interp), flatten(interp_dates)

# apply cubic spline (daily interpolation)
dates = harmonic.columns.to_numpy()
result = harmonic.apply(lambda row: function1(row, dates), 
                        axis=1, 
                        result_type='expand')

# add dates as columns and extract year
splined = pd.DataFrame(result[0].tolist(), index=harmonic.index)
splined.columns = result[1].tolist()[0]

##########################################################################################
# Extract Phenology
##########################################################################################

# get data from specified time period
def select_period(data, sdate, edate):
    transposed = data.T
    transposed.index.name = 'timestamp'
    query = f'timestamp.between("{sdate}", "{edate}")'
    result = transposed.query(query)
    result = result.T
    return result
splined_sel = select_period(splined, SDATE, EDATE)

# calculate percentiles
percentiles = P_RANGE
splined_sel_pc = splined_sel.quantile(percentiles, 
                                      interpolation='nearest', 
                                      method='single', 
                                      axis=1).T
splined_sel_pc.columns = [str(round(val, 1)) for val in percentiles]

# extra distribution-based phenometrics
stats = splined_sel.agg(['min', 'max', 'median', 
                         'mean', 'skew', 'std', 
                         'kurtosis'],
                        axis=1)
final = pd.concat([splined_sel_pc, stats], axis=1)

# export
first = FILE.split('_')[0]
last = FILE.split('_')[-1]
name = f'{first}_decile_phenology_{last}'
final.to_csv(DIR + name)