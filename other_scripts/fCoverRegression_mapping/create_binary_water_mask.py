#!/usr/bin/env python3
import os
import glob
import re
import subprocess
from tqdm import tqdm

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

# nodata & creation options (for gdal_calc.py)
NODATA  = '-9999'
CRE_OPT = [
    '--creation-option', 'TILED=YES',
    '--creation-option', 'BLOCKXSIZE=512',
    '--creation-option', 'BLOCKYSIZE=512',
    '--creation-option', 'COMPRESS=DEFLATE',
    '--creation-option', 'ZLEVEL=9',
    '--creation-option', 'PREDICTOR=3',
    '--creation-option', 'NUM_THREADS=ALL_CPUS',
    '--creation-option', 'BIGTIFF=YES',
]

# ─── 1) BUILD THE VRT MOSAICS ────────────────────────────────────────────────
print("1/5 ▶ Building VRTs…")
for name, vrt in [('green', green_vrt), ('red', red_vrt), ('nir', nir_vrt)]:
    band_dir = BAND_DIRS[name]
    # find all tiles...
    all_files = sorted(glob.glob(os.path.join(band_dir, 'GRIDCELL_*.tif')))
    # ...but only keep those with cell IDs 1077–4595
    files = []
    for f in all_files:
        m = re.search(r'GRIDCELL_(\d+)', os.path.basename(f))
        if m:
            cell = int(m.group(1))
            if 1077 <= cell <= 4595:
                files.append(f)
    if not files:
        raise FileNotFoundError(f"No files for {name} in {band_dir} within 1077–4595")
    cmd = ['gdalbuildvrt', '-overwrite', '-resolution', 'highest', vrt] + files
    print(f"   • {name.upper():>5} VRT → {os.path.basename(vrt)} (using {len(files)} tiles)")
    subprocess.run(cmd, check=True)

# ─── 2) CALCULATE NDWI ────────────────────────────────────────────────────────
print("\n2/5 ▶ Computing NDWI…")
cmd = [
    'gdal_calc.py',
    '--quiet',
    '-A', green_vrt, '-B', nir_vrt,
    '--outfile', ndwi_tif,
    '--calc', '(A.astype(float)-B)/(A+B)',
    '--NoDataValue', NODATA,
    '--type', 'Float32',
] + CRE_OPT + ['--overwrite']
print("   • Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"   ✔ NDWI written to {ndwi_tif}")

# ─── 3) CALCULATE NDVI ────────────────────────────────────────────────────────
print("\n3/5 ▶ Computing NDVI…")
cmd = [
    'gdal_calc.py',
    '--quiet',
    '-A', nir_vrt, '-B', red_vrt,
    '--outfile', ndvi_tif,
    '--calc', '(A.astype(float)-B)/(A+B)',
    '--NoDataValue', NODATA,
    '--type', 'Float32',
] + CRE_OPT + ['--overwrite']
print("   • Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"   ✔ NDVI written to {ndvi_tif}")

# ─── 4) BUILD WATER/ICE MASK ─────────────────────────────────────────────────
print("\n4/5 ▶ Building water/ice mask…")
mask_expr = '((A>0)&(B<0.3))*1'
cmd = [
    'gdal_calc.py',
    '-A', ndwi_tif, '-B', ndvi_tif,
    '--outfile', water_mask_tif,
    '--calc', mask_expr,
    '--NoDataValue', '0',
    '--type', 'Byte',
] + CRE_OPT + ['--overwrite']
print("   • Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"   ✔ Water mask → {water_mask_tif}")

# ─── 5) CLEAN UP VRTs ─────────────────────────────────────────────────────────
print("\n5/5 ▶ Cleaning up intermediate VRTs…")
for vrt in (green_vrt, red_vrt, nir_vrt):
    os.remove(vrt)
    print(f"   • removed {os.path.basename(vrt)}")

print("\nAll done.")