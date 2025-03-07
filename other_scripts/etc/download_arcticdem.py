import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os

# set parameters
bucket_name = 'pgc-opendata-dems'
folder_name = 'arcticdem/mosaics/v4.1/10m'
outpath = '/mnt/poseidon/remotesensing/arctic/data/rasters/arctic_dem'

# download files
s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
bucket = s3.Bucket(bucket_name)
for obj in bucket.objects.filter(Prefix=folder_name):
    filename = obj.key
    if '_dem.tif' in filename:
        outname = filename.split('/')[-1]
        outname = outname.replace('v4.1', 'v4-1')
        if not os.path.isfile(f'{outpath}/{outname}'):
            print(f'Saving file to: {outpath}/{outname}')
            bucket.download_file(filename, f'{outpath}/{outname}')
        else:
            continue