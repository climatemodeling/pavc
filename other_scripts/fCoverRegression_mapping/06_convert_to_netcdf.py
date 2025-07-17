import datetime
import glob
import os
import time

import cftime as cf
import numpy as np
import xarray as xr
from osgeo import gdal
import rioxarray as rxr

# parameters and paths
dirpath = "/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test_06-19-2025"
sdate = datetime.datetime(2019, 6, 1, 0, 0, 0)
edate = datetime.datetime(2019, 9, 30, 23, 59, 59)
filepaths = sorted(glob.glob(f"{dirpath}/*_mosaic_clipped_masked.tif"))

def extract_pft(fp):
    """
    Given a path like
      ".../deciduous_shrub_30M-3C-IQR2_mosaic_clipped_masked.tif"
    return "deciduous_shrub".
    """
    base = os.path.splitext(os.path.basename(fp))[0]
    parts = base.split('_')
    pft_parts = []
    for seg in parts:
        # stop once you hit the measurement segment,
        # which always starts with a digit (e.g. "30M-3C-IQR2")
        if seg and seg[0].isdigit():
            break
        pft_parts.append(seg)
    return '_'.join(pft_parts)

for filepath in filepaths:

    # convert tif to netcdf
    variable_name = extract_pft(filepath)
    nc_file = f"{dirpath}/{variable_name}_pavc-raster_arctic-alaska_summer-2019_v1-1.nc"
    print(f'Converting {filepath} to {nc_file}...')
    translate_options = gdal.TranslateOptions(format="netCDF", creationOptions=["FORMAT=NC4"])
    gdal.UseExceptions()
    gdal.Translate(nc_file, filepath, options=translate_options)

    # load the new netcdf
    print(f'Loading {nc_file} into xarray ...')
    ds = xr.open_dataset(nc_file)
    ds = ds.rename({'Band1': 'cover'})
    ds['cover'] = ds['cover'].astype('float32')
    ds.rio.write_crs(4326, inplace=True)

    # create time bounds
    print('Creating time bounds variable...')
    tb_arr = np.asarray(
        [
            [cf.DatetimeNoLeap(sdate.year, sdate.month, sdate.day)],
            [cf.DatetimeNoLeap(edate.year, edate.month, edate.day)],
        ]
    ).T
    tb_da = xr.DataArray(tb_arr, dims=("time", "nv"))

    # create time dimension from bounds
    print('Creating time dimension...')
    ds = ds.expand_dims(time=tb_da.mean(dim="nv"))
    ds["time_bounds"] = tb_da

    # set attributes lat/lon/time/time_bounds attrs
    print('Editing attributes...')
    ds["time"].attrs = {"axis": "T", "long_name": "time"}
    ds["lat"].attrs = {"axis": "Y", "long_name": "latitude", "units": "degrees_north"}
    ds["lat"] = ds["lat"].astype("float64")
    ds["lon"].attrs = {"axis": "X", "long_name": "longitude", "units": "degrees_east"}
    ds["lon"] = ds["lon"].astype("float64")

    # sort lat/lon values
    ds = ds.reindex(lat=list(reversed(ds.lat)))

    # set _FillValue to None for time, lat, lon
    for var in list(ds.coords) + [v for v in ds.data_vars if v != "cover"]:
        ds[var].attrs["_FillValue"] = None

    # add global attributes
    print('Formatting the global attributes')
    generate_stamp = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(os.path.getmtime(f"{dirpath}/{variable_name}.tif")),
    )
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

    # add water mask
    print(f'Opening water mask...')
    mask = rxr.open_rasterio(f"{dirpath}/water_mask_clipped.tif", masked=True)
    mask = mask.sel(band=1).drop_vars("band")
    mask = mask.where(mask != 255)
    mask = mask.rename({"y": "lat", "x": "lon"})
    mask = mask.assign_coords(lat=ds.lat, lon=ds.lon)
    mask = mask.astype("uint8")
    ds["water_mask"] = mask

    # set mask attributes
    ds["water_mask"].attrs.update({
        "long_name": "Water Mask",
        "standard_name": "status_flag",
        "valid_range": [np.byte(0), np.byte(1)],
        "flag_values": [np.byte(0), np.byte(1)],
        "flag_meanings": "not_water water",
        "_FillValue": np.byte(255)
    })

    # set time encoding
    print('Encoding time information...')
    ds["time"].encoding["units"] = f"days since {sdate.strftime('%Y-%m-%d %H:%M:%S')}"
    ds["time"].encoding["calendar"] = "noleap"
    ds["time"].encoding["bounds"] = "time_bounds"

    # set compression encoding
    print('Encoding compression information...')
    cover_encoding = {
        'dtype': 'float32',
        'zlib': True,
        'complevel': 9,
        'shuffle': True,           # helps compression
        '_FillValue': np.float32(-9999)
    }

    # set all encoding
    encoding = {
        'cover': cover_encoding,
        'time': ds['time'].encoding,
        'lat': ds['lat'].encoding,
        'lon': ds['lon'].encoding,
        'time_bounds': ds['time_bounds'].encoding,
        'water_mask': dict(dtype="uint8", _FillValue=255)
    }

    # export
    print(f'Exporting to {nc_file}...')
    ds.to_netcdf(
        nc_file,
        format="NETCDF4",
        encoding=encoding,
    )

    ds.close()
    del ds
    print('Export complete')