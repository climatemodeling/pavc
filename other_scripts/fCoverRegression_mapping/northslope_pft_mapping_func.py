#!/usr/bin/env python3
"""
northslope_pft_mapping.py

Usage examples:

# 1) Query how many MPI ranks you need, based on your PFT list:
python3 northslope_pft_mapping.py \
    --pfts evergreen_shrub forb graminoid litter \
    --print-n-pfts
# → prints "4"

# 2) Launch under MPI automatically with that many ranks:
mpiexec -n $(python3 northslope_pft_mapping.py \
                 --pfts evergreen_shrub forb graminoid litter \
                 --print-n-pfts) \
         python3 northslope_pft_mapping.py \
             --overwrite \
             --base /mnt/poseidon/remotesensing/arctic/data \
             --outdir /mnt/poseidon/remotesensing/arctic/data/rasters/model_results \
             --model-dir /mnt/poseidon/remotesensing/arctic/data/training/Test_06/results/03 \
             --pfts evergreen_shrub forb graminoid litter
"""

import os
import sys
import glob
import pickle
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from mpi4py import MPI
from osgeo import gdal

def parse_args():
    p = argparse.ArgumentParser(
        description="Spatial PFT fractional‐cover mapping over Arctic gridcells with MPI"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing outputs"
    )
    p.add_argument(
        "--base",
        "-b",
        required=True,
        help="Base directory (e.g. /mnt/poseidon/remotesensing/arctic/data)"
    )
    p.add_argument(
        "--outdir",
        "-o",
        required=True,
        help="Output directory for geotiffs"
    )
    p.add_argument(
        "--data-dir",
        "-d",
        help="Data directory (defaults to base/rasters)"
    )
    p.add_argument(
        "--model-dir",
        "-m",
        help="Directory containing your pickled models"
    )
    p.add_argument(
        "--pfts",
        "-p",
        nargs="+",
        required=True,
        help="List of PFT names (must match keys in the internal pft_file_map)"
    )
    p.add_argument(
        "--cell-list",
        nargs="+",
        type=int,
        help="List of gridcell IDs to process (defaults to 1077–4594)"
    )
    p.add_argument(
        "--ref-rast",
        help="Path to reference raster directory (defaults under data-dir)"
    )
    p.add_argument(
        "--print-n-pfts",
        action="store_true",
        help="Print the number of PFTs and exit (for MPI launcher)"
    )
    return p.parse_args()


def stack_bands(sensor, cell_paths, resample_bands, ref_rast, scale_factor=None):
    rasters = []
    for band_path in cell_paths:
        # derive a short name for each band
        if sensor == "s2_sr":
            b_name = os.path.basename(os.path.dirname(os.path.dirname(band_path)))
        elif sensor == "dem":
            b_name = os.path.splitext(os.path.basename(band_path))[0].split("_")[-1]
        elif sensor == "s1_grd":
            b_name = os.path.splitext(os.path.basename(band_path))[0].split("_")[-1]
        else:
            raise ValueError(f"Unknown sensor: {sensor}")

        r = rxr.open_rasterio(band_path)
        r.name = b_name
        # reproject to match
        r = r.rio.reproject_match(ref_rast)
        r = r.where(r != 0, -9999)
        if scale_factor is not None:
            r = r * scale_factor
        rasters.append(r)

    merged = xr.merge(rasters)
    # drop pixels where any band is NA
    merged = merged.dropna(dim="band", how="any")
    return merged


def calc_veg_idx_s2(xrd):
    x = xrd
    # NB: assume bands have been renamed already to: blue, green, red, redEdge1, etc.
    x = x.assign(ndwi1=lambda x: (x.nir - x.swir1) / (x.nir + x.swir2))
    x = x.assign(ndwi2=lambda x: (x.nir - x.swir2) / (x.nir + x.swir2))
    x = x.assign(
        msavi=lambda x: (
            2 * x.nir
            + 1
            - ((2 * x.nir + 1) ** 2 - 8 * (x.nir - x.red)) ** 0.5
        )
        * 0.5
    )
    # … add the rest of your indices exactly as before …
    return x


def main():
    args = parse_args()

    # if user only wants the count of PFTs, print & exit
    if args.print_n_pfts:
        print(len(args.pfts))
        sys.exit(0)

    # resolve directories
    BASE = args.base
    DATA_DIR = args.data_dir or os.path.join(BASE, "rasters")
    OUT_DIR = args.outdir
    MODEL_DIR = args.model_dir or os.path.join(BASE, "training", "Test_06", "results", "03")
    os.makedirs(OUT_DIR, exist_ok=True)

    # reference raster folder
    REF_RAST = args.ref_rast or os.path.join(
        DATA_DIR,
        "s2_sr_tiled",
        "ak_arctic_summer",
        "B11",
        "2019-06-01_to_2019-08-31",
    )

    # gridcells to process
    CELL_LIST = args.cell_list if args.cell_list else list(range(1077, 4595))

    # your PFT list
    PFTS = args.pfts

    # quick sanity check: if you launch under mpiexec, make sure -n matches
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size != len(PFTS):
        if size == 1:
            # probably not launched under mpiexec -n …
            sys.stderr.write(
                f"WARNING: MPI size=1 but {len(PFTS)} PFTs requested.\n"
                f"  → re-launch with: mpiexec -n {len(PFTS)} python3 {sys.argv[0]} {' '.join(sys.argv[1:])}\n"
            )
        else:
            sys.stderr.write(
                f"ERROR: MPI size={size} does not match #PFTs={len(PFTS)}.  Exiting.\n"
            )
            sys.exit(1)

    # logging
    now = datetime.now().strftime("%d-%m-%Y-%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(OUT_DIR, f"std_{now}.log"),
        filemode="w",
        format="%(asctime)s >>> %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    # mapping from PFT names → pickle and suffix
    pft_file_map = {
        "bryophyte": {
            "model": "bryophyte_30m_parent_sources5_IQR3.pkl",
            "outfile_suffix": "30M-5P-IQR3",
        },
        "lichen": {
            "model": "lichen_55m_parent_sources3_IQR2.5.pkl",
            "outfile_suffix": "55M-3P-IQR2.5",
        },
        "deciduous_shrub": {
            "model": "deciduous shrub_30m_child_sources3_IQR2.pkl",
            "outfile_suffix": "30M-3C-IQR2",
        },
        "evergreen_shrub": {
            "model": "evergreen shrub_30m_parent_sources4_IQR2.5.pkl",
            "outfile_suffix": "30M-4P-IQR2.5",
        },
        "forb": {
            "model": "forb_55m_child_sources5_IQR1.5.pkl",
            "outfile_suffix": "55M-5C-IQR1.5",
        },
        "graminoid": {
            "model": "graminoid_30m_parent_sources3_IQR2.pkl",
            "outfile_suffix": "30M-3P-IQR2",
        },
        "non-vascular": {
            "model": "non-vascular_30m_parent_sources2_IQR1.5.pkl",
            "outfile_suffix": "30M-2P-IQR1.5",
        },
        "litter": {
            "model": "litter_55m_child_sources4_IQR2.pkl",
            "outfile_suffix": "55M-4C-IQR2",
        },
    }

    print(f"PROCESS RANK {comm.Get_rank()} of {size}, PFTs: {PFTS}")

    # split up the gridcells
    processes = np.array(CELL_LIST)
    split = np.array_split(processes, size)
    grid_list = split[comm.Get_rank()]

    # begin loop
    for gridcell in grid_list:
        expected = []
        for p in PFTS:
            m = pft_file_map.get(p)
            if not m:
                logging.error(f"No mapping for PFT '{p}', skipping on rank {comm.Get_rank()}")
                continue
            fn = f"GRIDCELL_{gridcell}_{p}_{m['outfile_suffix']}.tif"
            expected.append(os.path.join(OUT_DIR, fn))

        if not args.overwrite and all(os.path.isfile(fp) for fp in expected):
            msg = f"SKIP {gridcell}: all {len(PFTS)} outputs exist"
            print(msg)
            logging.info(msg)
            continue

        # load reference raster
        ref_r = rxr.open_rasterio(os.path.join(REF_RAST, f"GRIDCELL_{gridcell}.tif"))
        ref_r = ref_r.where(ref_r != 0, -9999) * 0.0001

        # stack S2, calc indices
        s2_paths = sorted(glob.glob(os.path.join(DATA_DIR, "s2_sr_tiled", "*", "*", f"GRIDCELL_{gridcell}*")))
        s2 = stack_bands("s2_sr", s2_paths, ["B2","B3","B4","B8"], ref_r, scale_factor=0.0001)
        s2 = s2.rename({
            "B2":"blue","B3":"green","B4":"red","B5":"redEdge1","B6":"redEdge2",
            "B7":"redEdge3","B8A":"redEdge4","B8":"nir","B11":"swir1","B12":"swir2"
        })
        s2 = calc_veg_idx_s2(s2)

        # stack S1
        s1_paths = sorted(glob.glob(os.path.join(DATA_DIR, "s1_grd_tiled", f"*GRIDCELL_{gridcell}*")))
        s1 = stack_bands("s1_grd", s1_paths, ["VV","VH"], ref_r)
        s1 = s1.where(s1[["VV","VH"]] != 0, -9999)

        # stack DEM
        dem_paths = sorted(glob.glob(os.path.join(DATA_DIR, "arctic_dem_tiled", f"*GRIDCELL_{gridcell}*")))
        dem = stack_bands("dem", dem_paths, ["aspect","dem","hillshade","slope"], ref_r)
        dem = dem.rename({"dem":"elevation"})

        # merge sensors
        stacked = xr.merge([s2, s1, dem])
        df = stacked.to_dataframe().droplevel([1,2]).reset_index()
        coords = df[["x","y"]].copy()
        df = df.drop(columns=["x","y"]).reset_index(drop=True)

        # drop inf/nan
        bad = np.isinf(df.values).any(axis=1)
        nan = np.isnan(df.values).any(axis=1)
        mask = ~(bad|nan)
        df, coords = df[mask], coords[mask]

        # loop PFTs
        for p in PFTS:
            entry = pft_file_map[p]
            model_fp = os.path.join(MODEL_DIR, "tunedModel_" + entry["model"])
            if not os.path.isfile(model_fp):
                logging.critical(f"Model file not found: {model_fp}")
                sys.exit(1)

            with open(model_fp, "rb") as f:
                mdl = pickle.load(f)

            df2 = df[mdl.feature_names_in_]
            preds = mdl.predict(df2)

            out_name = f"GRIDCELL_{gridcell}_{p}_{entry['outfile_suffix']}.tif"
            out_fp   = os.path.join(OUT_DIR, out_name)

            # build xarray and write
            res = coords.copy()
            res["fcover"] = preds
            res["band"]   = 1
            ds = xr.Dataset.from_dataframe(res.set_index(["band","y","x"]))
            ds = ds.isel(band=0).rio.write_crs("EPSG:4326")
            ds.rio.to_raster(out_fp)

            # finalize with gdal warp to enforce nodata & SRS
            opts = gdal.WarpOptions(format="GTiff", dstSRS="EPSG:4326", dstNodata=-9999)
            gdal.Warp(out_fp, out_fp, options=opts)

            msg = f"EXPORTED {out_fp}"
            print(msg)
            logging.info(msg)


if __name__ == "__main__":
    main()
