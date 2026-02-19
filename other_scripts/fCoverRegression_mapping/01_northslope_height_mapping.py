#!/usr/bin/python3
# mpiexec -n 4 python3 northslope_pft_mapping.py
# 4 is the number of PFTs I'm processing (see line 33)

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
import sys
from pathlib import Path
import traceback

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H%M%S")


# General Params
OVERWRITE = True
BASE = '/mnt/poseidon/remotesensing/arctic/data'
OUT_DIR = f'{BASE}/rasters/height_results_tiled_11-21-2025'
DATA_DIR = f'{BASE}/rasters'
CELL_LIST = list(range(1077,4595))
REF_RAST = f'{DATA_DIR}/s2_sr_tiled/ak_arctic_summer/B11/2019-06-01_to_2019-08-31'
MODEL = f'{BASE}/training/height_test_01'
PFTS = ['rh95']
EXPORT_FEAT_TIF = False

pft_file_map = {
    "rh95": {
        "model": "RF_model_rh95.pkl",
        "outfile_suffix": "rh95-21nov25",
    },
}

# Sensor-specific Params
S2_DIR = f"{DATA_DIR}/s2_sr_tiled/ak_arctic_summer"
S1_DIR = f"{DATA_DIR}/s1_grd_tiled"
DEM_DIR = f"{DATA_DIR}/arctic_dem_tiled"
print('Number of gridcells to work on:', len(CELL_LIST))

# parallel processing
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# logging configuration
os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    level    = logging.INFO,
    filename = f'{OUT_DIR}/std_{dt_string}.log',
    filemode = 'w',
    format   = '%(asctime)s >>> %(message)s',
    datefmt  = '%d-%b-%y %H:%M:%S'
)


def s2_gridcell_paths(gridcell: int, s2_root: str | Path) -> list[str]:
    """
    S2 layout (your REF_RAST suggests):
      <s2_root>/<band>/<date_range>/GRIDCELL_<id>.tif
    e.g. .../s2_sr_tiled/ak_arctic_summer/B11/2019-06-01_to_2019-08-31/GRIDCELL_1516.tif
    """
    s2_root = str(s2_root)
    pattern = os.path.join(
        s2_root,
        "*",  # band (B02, B03, B11, ...)
        "*",  # date range (2019-06-01_to_2019-08-31, etc.)
        f"GRIDCELL_{gridcell}.tif",
    )
    paths = sorted(glob.glob(pattern))
    return paths


def s1_gridcell_paths(gridcell: int, s1_root: str | Path) -> list[str]:
    """
    S1 layout (example you gave):
      <s1_root>/GRIDCELL_<id>_VV.tif
      <s1_root>/GRIDCELL_<id>_VH.tif
    """
    s1_root = str(s1_root)
    pattern = os.path.join(
        s1_root,
        f"GRIDCELL_{gridcell}_*.tif",
    )
    paths = sorted(glob.glob(pattern))
    return paths


def dem_gridcell_paths(gridcell: int, dem_root: str | Path) -> list[str]:
    """
    DEM layout (assuming similar to S1, tweak if needed):
      <dem_root>/GRIDCELL_<id>_*.tif
    """
    dem_root = str(dem_root)
    pattern = os.path.join(
        dem_root,
        f"GRIDCELL_{gridcell}_*.tif",
    )
    paths = sorted(glob.glob(pattern))
    return paths


# function that returns list of paths to gridcell
def gridcell_rast_path(gridcell, directory):
    path = sorted(glob.glob(f'{directory}/GRIDCELL_{gridcell}_*.tif'))
    return path


# stack each band as variables in an xarray Dataset
def stack_bands(
    sensor: str,
    band_paths,
    resample_bands,
    ref_rast,
    scale_factor=None,
) -> xr.Dataset:
    """
    Stack all raster bands for one gridcell into a Dataset with one variable per band.
    """
    vars_dict: dict[str, xr.DataArray] = {}

    for band_path in band_paths:
        band_path = str(band_path)

        # --- band name parsing per sensor ---
        if sensor == "s2_sr":
            # directory like 'B02','B03','B8A','B11',...
            band_dir = band_path.split("/")[-3]

            # Map S2 directory names -> your feature names
            s2_name_map = {
                "B02": "B2",
                "B03": "B3",
                "B04": "B4",
                "B05": "B5",
                "B06": "B6",
                "B07": "B7",
                "B08": "B8",
                "B8A": "B8A",
                "B11": "B11",
                "B12": "B12",
            }
            var_name = s2_name_map.get(band_dir, band_dir)

        elif sensor == "s1_grd":
            # e.g. GRIDCELL_1516_VV.tif -> 'VV'
            fname = os.path.basename(band_path)
            base = os.path.splitext(fname)[0]
            b_name = base.split("_")[-1]
            var_name = b_name

        elif sensor == "dem":
            fname = os.path.basename(band_path)
            base = os.path.splitext(fname)[0]
            parts = base.split("_")
            if len(parts) <= 2:
                # fallback, but this would be weird
                band_token = parts[-1]
            else:
                band_token = "_".join(parts[2:])
            var_name = band_token

        else:
            msg = f"Incorrect sensor choice {sensor!r}. Try 'dem', 's2_sr', or 's1_grd'."
            print(msg)
            logging.critical(msg)
            raise ValueError(msg)

        # open raster
        raster = rxr.open_rasterio(band_path)

        # rioxarray default: (band, y, x) with band size 1
        if "band" in raster.dims and raster.sizes.get("band", 1) == 1:
            raster = raster.squeeze("band", drop=True)

        # match reference raster resolution / CRS
        raster = raster.rio.reproject_match(ref_rast)

        # set nodata placeholder
        raster = raster.where(raster != 0, -9999)

        # optional scaling (e.g. for S2 reflectance)
        if scale_factor is not None:
            raster = raster * scale_factor

        raster.name = var_name
        vars_dict[var_name] = raster

    if not vars_dict:
        raise ValueError(
            f"No bands were loaded in stack_bands for sensor={sensor}. "
            f"Check band_paths and directory layout."
        )

    # Build Dataset with one variable per band (all on y,x)
    ds = xr.Dataset(vars_dict)

    # remove any pixels with NaNs in *any* band
    all_vars = xr.concat(list(ds.data_vars.values()), dim="__band_tmp__")
    mask = all_vars.isnull().any(dim="__band_tmp__")   # (y,x)
    ds = ds.where(~mask)

    return ds


# create output directory
processes = np.array(CELL_LIST)
split_processes = np.array_split(processes, size) # split array into x pieces
for p_idx in range(len(split_processes)):
    if p_idx == rank:
        grid_list = split_processes[p_idx] # select current list of gridcells
    else:
        pass
print(f'CURRENT RANK: {rank}')
print(f'GRIDCELLS: {grid_list[0]} to {grid_list[-1]}')
logging.info(f'CURRENT RANK: {rank}')
logging.info(f'GRIDCELLS: {grid_list[0]} to {grid_list[-1]}')


# loop through gridcells
for gridcell in grid_list:

    # 1) Build the exact list of expected outputs—*with* suffix
    expected = [
        os.path.join(
            OUT_DIR,
            f"GRIDCELL_{gridcell}_{p.replace(' ', '_')}_{pft_file_map[p]['outfile_suffix']}.tif"
        )
        for p in PFTS
    ]

    # 2) If NONE need writing, skip all the heavy raster work
    if not OVERWRITE and all(os.path.isfile(path) for path in expected):
        msg = f"SKIPPING GRIDCELL {gridcell}: all {len(PFTS)} files exist"
        print(msg)
        logging.info(msg)
        continue

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
    rast_path = s2_gridcell_paths(gridcell, S2_DIR)
    print(f"GRIDCELL {gridcell}: found {len(rast_path)} S2 bands")
    rescale_bands = ['B2', 'B3', 'B4', 'B8'] # these are 10-m bands
    s2_stacked_raster = stack_bands(
        "s2_sr",
        rast_path,
        rescale_bands,
        reference_raster,
        scale_factor,
    )
    
    #########################################################################
    # Sentinel 1 (nodata = -9999)
    #########################################################################

    # create 20-m xarray raster
    s1_paths = s1_gridcell_paths(gridcell, S1_DIR)
    rescale_bands = ['VV', 'VH']
    s1_stacked_raster = stack_bands(
        "s1_grd",
        s1_paths,
        resample_bands=None,
        ref_rast=reference_raster,
        scale_factor=None,
    )

    #########################################################################
    # Arctic DEM (nodata = -9999)
    #########################################################################

    # create 20-m xarray raster
    dem_paths = dem_gridcell_paths(gridcell, DEM_DIR)
    rescale_bands = ['aspect', 'dem', 'hillshade', 'slope']
    dem_stacked_raster = stack_bands(
        "dem",
        dem_paths,
        resample_bands=None,
        ref_rast=reference_raster,
        scale_factor=None,
    )

    #########################################################################
    # Combine into one xarray
    #########################################################################

    # make sure pandas df features are in the right order
    stacked_raster = xr.merge(
        [s2_stacked_raster, s1_stacked_raster, dem_stacked_raster],
        compat="override",
    )
    
    # optionally export feature raster
    if EXPORT_FEAT_TIF:
        feat_out_path = os.path.join(OUT_DIR, f'GRIDCELL_{gridcell}_features.tif')
        try:
            stacked_raster.rio.to_raster(feat_out_path)
            msg = f"EXPORTED FEATURE RASTER {feat_out_path}"
            print(msg)
            logging.info(msg)
        except Exception as e:
            msg = f"EXCEPTION WHILE EXPORTING FEATURE RASTER FOR GRIDCELL {gridcell}: {e}"
            print(msg)
            logging.error(msg, exc_info=True)
            continue

    # get full dataframe: index (y,x), columns = bands
    df = stacked_raster.to_dataframe()

    # save coordinates
    coords = df.reset_index()[['x', 'y']]

    # flatten index and drop coordinate columns, keep only band variables
    df = df.reset_index(drop=True)

    # find any bands that were divided by 0 and produced an inf value
    bad_idx_list = df[np.isinf(df.values)].index.tolist()
    df.drop(index=bad_idx_list, inplace=True)
    coords.drop(index=bad_idx_list, inplace=True)

    # remove straggling nans
    nan_idx_list = df[np.isnan(df.values)].index.tolist()
    df.drop(index=nan_idx_list, inplace=True)
    coords.drop(index=nan_idx_list, inplace=True)

    # add lat and lon as columns to df
    df['lon'] = coords['x'].values
    df['lat'] = coords['y'].values

    # sort the dataframe so the model input order is correct
    feature_cols = [
        'lon','lat',
        'B11','B12','B2','B3','B4','B5','B6','B7','B8A','B8',
        'VH','VV',
        'dem_aspect','dem_hillshade','dem','dem_slope',
    ]
    df = df[feature_cols]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected feature columns: {missing}")

    if df.empty:
        raise RuntimeError("No samples left after filtering NaN/inf")

    #########################################################################
    # Apply model
    #########################################################################
    
    for PFT in PFTS:

        entry = pft_file_map.get(PFT)
        if entry is None:
            logging.error(f"No mapping for PFT '{PFT}', skipping.")
            continue
        
        try:

            # get pickle path
            model_file_path = os.path.join(MODEL, entry["model"])
            if not os.path.isfile(model_file_path):
                logging.critical(f"Model file not found: {model_file_path}. Exiting.")
                sys.exit(1)

            # 2) Load the pickled model
            with open(model_file_path, "rb") as f:
                model = pickle.load(f)

            # 3) Reorder df and predict
            ordered_df = df[feature_cols]
            predicted = model["model"].predict(ordered_df)  # predicted is 1 × n

            # 4) Build output filename with suffix
            pft_slug = PFT.replace(" ", "_")
            tag = entry["outfile_suffix"]
            out_name = f"GRIDCELL_{gridcell}_{pft_slug}_{tag}.tif"
            out_path = os.path.join(OUT_DIR, out_name)
            
        except Exception as e:
            print(f"ERROR MODELLING FAILED FOR GRIDCELL {gridcell}, PFT '{PFT}': {e}")
            traceback.print_exc()

        
        #########################################################################
        # Export modeled tif
        #########################################################################

        # set up df for xarray
        results = coords.copy()
        results['fcover'] = predicted
        results['band'] = 1
        
        # export xarray as tif
        try:
            results_xr = xr.Dataset.from_dataframe(results.set_index(['band', 'y', 'x']))
            xr_band = results_xr.isel(band=0).rio.write_crs('EPSG:4326')
            xr_band.rio.to_raster(out_path)

            msg = f"EXPORTED {out_path}"
            print(msg)
            logging.info(msg)
            
        except Exception as e:
            msg = f"EXCEPTION WHILE EXPORTING FOR GRIDCELL {gridcell}, PFT '{PFT}': {e}"
            print(msg)
            logging.error(msg, exc_info=True)
            continue
            
        # set crs
        nodata_value = -9999
        opts = gdal.WarpOptions(format='GTiff', dstSRS='EPSG:4326', dstNodata=nodata_value)
        gdal.Warp(out_path, out_path, options=opts) 