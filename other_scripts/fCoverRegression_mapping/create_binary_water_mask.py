#!/usr/bin/env python3
import os
import glob
import re
import subprocess
from tqdm import tqdm
import argparse

# ─── PARSE ARGS ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Build VRTs, compute NDWI/NDVI, and create a water mask with optional overwrite behavior"
)
parser.add_argument(
    "--overwrite", "-f",
    action="store_true",
    help="Force overwrite of existing output files"
)
args = parser.parse_args()
overwrite = args.overwrite

# ─── PARAMETERS ──────────────────────────────────────────────────────────────
BASE    = '/mnt/poseidon/remotesensing/arctic/data'
DATA    = os.path.join(BASE, 'rasters')
OUT     = os.path.join(BASE, 'rasters', 'model_results_tiled_test_06-19-2025')
os.makedirs(OUT, exist_ok=True)

DATE_RANGE = '2019-06-01_to_2019-08-31'
BAND_DIRS  = {
    'green': os.path.join(DATA, 's2_sr_tiled/ak_arctic_summer/B3', DATE_RANGE),
    'red':   os.path.join(DATA, 's2_sr_tiled/ak_arctic_summer/B4', DATE_RANGE),
    'nir':   os.path.join(DATA, 's2_sr_tiled/ak_arctic_summer/B8', DATE_RANGE),
}

# output VRTs & TIFs
green_vrt      = os.path.join(OUT, 'green.vrt')
red_vrt        = os.path.join(OUT, 'red.vrt')
nir_vrt        = os.path.join(OUT, 'nir.vrt')
ndwi_tif       = os.path.join(OUT, 'ndwi.tif')
ndvi_tif       = os.path.join(OUT, 'ndvi.tif')
water_mask_tif = os.path.join(OUT, 'water_mask.tif')

# common creation options for gdal_calc
COMMON_CRE_OPT = [
    '--creation-option', 'TILED=YES',
    '--creation-option', 'BLOCKXSIZE=512',
    '--creation-option', 'BLOCKYSIZE=512',
    '--creation-option', 'COMPRESS=DEFLATE',
    '--creation-option', 'ZLEVEL=9',
    '--creation-option', 'NUM_THREADS=ALL_CPUS',
    '--creation-option', 'BIGTIFF=YES',
]

# ─── 1) BUILD THE VRT MOSAICS ────────────────────────────────────────────────
print("1/5 Building VRTs…")
for name, vrt in [('green', green_vrt), ('red', red_vrt), ('nir', nir_vrt)]:
    if not overwrite and os.path.exists(vrt):
        print(f"   • {os.path.basename(vrt)} exists, skipping")
        continue

    band_dir = BAND_DIRS[name]
    all_files = sorted(glob.glob(os.path.join(band_dir, 'GRIDCELL_*.tif')))
    files = []
    for f in all_files:
        m = re.search(r'GRIDCELL_(\d+)', os.path.basename(f))
        if m and 1077 <= int(m.group(1)) <= 4595:
            files.append(f)
    if not files:
        raise FileNotFoundError(f"No files for {name} in {band_dir} within 1077-4595")

    cmd = ['gdalbuildvrt', '-resolution', 'highest'] + (['-overwrite'] if overwrite else []) + [vrt] + files
    print(f"   {name.upper():>5} VRT → {os.path.basename(vrt)} (using {len(files)} tiles)")
    subprocess.run(cmd, check=True)

# ─── 2) CALCULATE NDWI ────────────────────────────────────────────────────────
print("\n2/5 Computing NDWI…")
if overwrite or not os.path.exists(ndwi_tif):
    cmd = [
        'gdal_calc.py', 
        '--quiet',
        '-A', green_vrt, '-B', nir_vrt,
        '--outfile', ndwi_tif,
        '--calc', '(A.astype(float)-B)/(A+B)',
        '--NoDataValue', '-9999',
        '--type', 'Float32',
    ] + COMMON_CRE_OPT + ['--creation-option', 'PREDICTOR=3'] + (['--overwrite'] if overwrite else [])
    print("   Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"   NDWI written to {ndwi_tif}")
else:
    print(f"   {os.path.basename(ndwi_tif)} exists, skipping")

# ─── 3) CALCULATE NDVI ────────────────────────────────────────────────────────
print("\n3/5 Computing NDVI…")
if overwrite or not os.path.exists(ndvi_tif):
    cmd = [
        'gdal_calc.py', 
        '--quiet',
        '-A', nir_vrt, '-B', red_vrt,
        '--outfile', ndvi_tif,
        '--calc', '(A.astype(float)-B)/(A+B)',
        '--NoDataValue', '-9999',
        '--type', 'Float32',
    ] + COMMON_CRE_OPT + ['--creation-option', 'PREDICTOR=3'] + (['--overwrite'] if overwrite else [])
    print("   Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"   NDVI written to {ndvi_tif}")
else:
    print(f"   {os.path.basename(ndvi_tif)} exists, skipping")

# ─── 4) BUILD WATER/ICE MASK ─────────────────────────────────────────────────
print("\n4/5 Computing NDVI…")
# ensure old mask is removed so gdal_calc won't complain
if os.path.exists(water_mask_tif):
    os.remove(water_mask_tif)

# always rebuild the mask: 1=water, 0=land, nodata=-9999
cmd = [
    'gdal_calc.py',
    '--quiet',
    '-A', ndwi_tif,
    '-B', ndvi_tif,
    '--outfile', water_mask_tif,
    '--calc', '((A>0)&(B<0.3))*1',
    '--NoDataValue', '255',
    '--type', 'Byte',
    '--overwrite',
] + COMMON_CRE_OPT + ['--creation-option', 'PREDICTOR=2']

print("   Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"   Water mask written to {water_mask_tif}")

# ─── 5) CLEAN UP VRTs ─────────────────────────────────────────────────────────
print("\n5/5 Cleaning up intermediate VRTs…")
for vrt in (green_vrt, red_vrt, nir_vrt):
    if os.path.exists(vrt):
        os.remove(vrt)
        print(f"   Removed {os.path.basename(vrt)}")

print("\nAll done.")