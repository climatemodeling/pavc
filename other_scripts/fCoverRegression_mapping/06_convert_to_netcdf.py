import datetime
import glob
import os

import cftime as cf
import numpy as np
import xarray as xr
import rioxarray as rxr
import os, shlex, subprocess
from osgeo import gdal

# set parameters
dirpath = "/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test_06-19-2025"
sdate = datetime.datetime(2019, 6, 1)
edate = datetime.datetime(2019, 9, 30)
filepaths = sorted(glob.glob(f"{dirpath}/*_mosaic_clipped_masked.tif"))
chunk_size = 2048
xres = 0.000179663056824
yres = 0.000179663056824

# function to extract PFT name from filename
def extract_pft(fp):
    base = os.path.splitext(os.path.basename(fp))[0]
    parts = base.split('_')
    p = []
    for seg in parts:
        if seg and seg[0].isdigit():
            break
        p.append(seg)
    return '_'.join(p)

# start by loading the water mask
mask = rxr.open_rasterio(f"{dirpath}/water_mask_clipped.tif", band_as_variable=True)
mask = mask.rename({"y": "lat", "x": "lon", "band_1": "water_mask"})
del mask["spatial_ref"]
for attr in ("scale_factor", "add_offset", "_FillValue"):
    del mask["water_mask"].attrs[attr]
del mask.attrs["AREA_OR_POINT"]

# clear and set mask attributes
mask["water_mask"].encoding.clear()
mask["water_mask"].attrs.update({
    "long_name": "Water Mask",
    "standard_name": "status_flag",
    "valid_range": [np.byte(0), np.byte(1)],
    "flag_values": [np.byte(0), np.byte(1)],
    "flag_meanings": "not_water water",
})

# loop through each tif
for filepath in filepaths:

    # set pft name and output nc file name
    variable_name = extract_pft(filepath)
    nc_file = f"{dirpath}/{variable_name}_pavc-raster_arctic-alaska_summer-2019_v1-1.nc"

    # read the tif as an xr dataset
    print(f'Loading {filepath} into xarray and building NetCDF...')
    ds = rxr.open_rasterio(filepath, band_as_variable=True)
    ds = ds.chunk({"y": chunk_size, "x": chunk_size})
    ds = ds.rename({
        "x":"lon", 
        "y": "lat",
        "band_1":"cover"
    })
    # convert nodata of -9999 to 255
    ds["cover"] = ds["cover"].where(ds["cover"] != -9999, 255)
    # multiply values except 255 by 100 to convert to percent
    ds["cover"] = ds["cover"].where(ds["cover"] == 255, ds["cover"] * 100)
    ds["cover"] = ds["cover"].astype(np.uint8)

    # add long_name to cover and remove some attrs
    ds["cover"].attrs["long_name"] = f"percent total cover of {variable_name}s"
    for attr in ("scale_factor", "add_offset", "_FillValue"):
        try:
            del ds["cover"].attrs[attr]
        except KeyError:
            pass
    del ds.attrs["AREA_OR_POINT"]
    del ds["spatial_ref"]

    # set cover encoding
    cover_encoding = {
        'dtype': 'uint8',
        'zlib': True,
        'complevel': 6,
        'shuffle': True,
        '_FillValue': np.uint8(255),
        'chunksizes': (1, chunk_size, chunk_size)
    }

    # create time bounds xr data array
    print('Creating time bounds variable...')
    tb_arr = np.asarray(
        [
            [cf.DatetimeNoLeap(sdate.year, sdate.month, sdate.day)],
            [cf.DatetimeNoLeap(edate.year, edate.month, edate.day)],
        ]
    ).T
    tb_da = xr.DataArray(tb_arr, dims=("time", "nv"))
    time_mid = tb_da.mean(dim="nv")

    # add time and time bounds
    print('Creating time dimension...')
    ds = ds.expand_dims(time=time_mid)
    ds["time_bounds"] = tb_da

    # add the water mask
    print('Adding water mask...')
    ds["water_mask"] = mask["water_mask"]

    # clean up time
    print('Editing attributes...')
    ds["time"].attrs = {
        "axis": "T", 
        "long_name": "time",
        "standard_name": "time"}
    ds["time"].encoding["units"] = f"days since {sdate.strftime('%Y-%m-%d %H:%M:%S')}"
    ds["time"].encoding["calendar"] = "noleap"
    ds["time"].encoding["bounds"] = "time_bounds"

    # clean up lat and lon
    ds = ds.reindex(lat=list(reversed(ds.lat))) # must do this BEFORE setting attrs
    ds["lat"].attrs = {
        "axis": "Y", 
        "long_name": "latitude",
        "standard_name": "latitude", 
        "units": "degrees_north"}
    ds["lon"].attrs = {
        "axis": "X", 
        "long_name": "longitude", 
        "standard_name": "longitude",
        "units": "degrees_east"}

    # set _FillValue to None for time, lat, lon
    for var in list(ds.coords) + [v for v in ds.data_vars if v != "cover"]:
        ds[var].encoding["_FillValue"] = None

    # add global attributes
    print('Formatting global attributes...')
    ds.attrs = {
        "title": f"Pan-Arctic Vegetation Cover (PAVC) Gridded: {variable_name.replace('_',' ')}",
        "version": "v1.1",
        "institution": "Oak Ridge National Laboratory",
        "source": ("Total fractional cover estimated by combining 20m Sentinel and ArcticDEM-derived "
                "predictors with high-quality plot samples based on fine-tuned random forest regression models"),
        "references": "",
        "comment": "The water mask identifies pixels where ndwi > 0 and ndvi <0.3",
        "Conventions": "CF-1.12",
    }

    # set all encoding
    print('Encoding...')
    encoding = {
        'cover': cover_encoding,
        'time': ds['time'].encoding,
        'lat': ds['lat'].encoding,
        'lon': ds['lon'].encoding,
        'time_bounds': ds['time_bounds'].encoding,
        'water_mask': dict(dtype="byte", _FillValue=np.int8(-1))
    }

    # export
    print(f'Exporting to {nc_file}...')
    ds.to_netcdf(
        nc_file,
        format="NETCDF4",
        engine="netcdf4",
        encoding=encoding
    )

    ds.close()
    del ds
    print('Export complete')