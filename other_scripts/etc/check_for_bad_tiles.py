from osgeo import gdal
import sys
import os
import glob
import logging
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

# params
band = 'B5'
d = f'/mnt/poseidon/remotesensing/arctic/data/rasters/S1GRD/'

# logging info
logpath = '/mnt/poseidon/remotesensing/arctic/scripts/etc/logs'
logname = f'bad_tiles_s1_{band}'
logging.basicConfig(level = logging.INFO,
                    filename=f'{logpath}/{logname}_01.log', filemode='w',
                    format='%(asctime)s >>> %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')

# find tiles with issues
files = sorted(glob.glob(f'{d}/*.tif'))
total = len(files)
for i, file in enumerate(files):
    sys.stdout.write(f"\r{i}/{total-1}")
    sys.stdout.flush()
    try:
        gtif = gdal.Open(file)
    except RuntimeError as e:
        filename = os.path.splitext(os.path.basename(file))[0]
        logging.critical(f'CRITICAL: Gdal unable to open {filename} \nERROR: {e}')