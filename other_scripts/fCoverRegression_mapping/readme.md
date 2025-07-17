# Step-by-step of how to run modeling and post-analysis

---
## Create water mask
1. Run `create_binary_water_mask.py` to create a mosaicked binary water mask raster (water = 1, not water = 0)
    - Threshold to get water = 1: '((NDWI>0)&(NDVI<0.3))*1'
    - Nodata = -9999
2. Run `03_clip_to_polygon.sh` to clip the water mask to our study area (alaska tundra)

---
## Modeling
1. Run `01_northslope_pft_mapping.py` to generate per-GRIDCELL PFT maps using pkl models
2. Run `02_mosaic_tifs_gdal.sh` to mosaic the GRIDCELL maps into single PFT map mosaics
3. Run `03_clip_to_polygon.sh` to clip the mosiacs to our study area (alaska tundra)
4. (Optionally) run `04_apply_water_bitmask.sh` to set water areas in mosaics to -9999
    - I don't do this for the final map products because I just add a bitmask layer to the NetCDF
    - But you might want to do this when doing Zonal Stats
        - You can also just mask temporarily during Zonal Stats
5. (Optionally) run `05_add_tif_pyramids.sh` so that these big files render quickly in QGIS
6. Run `convert_to_netcdf.py` to create PFT mosaic NetCDFs with a water bitmask and global attrs assigned
    - These are the files that are uploaded to ESS-Dive
    - Be sure to indicate the version of the maps in the global attributes!

---
## Post-analysis
### A. PFT distributions per CAVM zone (Plus Macander comparison)
1. Clip Macander maps to our study area (they have values out in the ocean we need to remove)
    - In QGIS raster calculator: `if (("one-of-our-clipped-rasters" = -9999), 255, "macander-raster")`
        - Output: `macander-raster-clipped-to-ours`
        - Macander no-data is `255`, our no-data is `-9999`, and cavm no-data is `127`
2. Clip our maps to Macander study area (their study area is much smaller)
    - In QGIS raster calcularor: `if (("macander-raster-clipped-to-ours" = 255), -9999, "our-raster")`
        - Output: `our-raster-clipped-to-macander`
3. Clip CAVM zone raster to clipped Macander study area
    - In QGIS raster calculator: `if (("macander-raster-clipped-to-ours" = 255), 127, "cavm-raster")`
        - Output: `cavm-raster-clipped-to-macander`
4. Clip water mask to clipped Macander study area
    - In QGIS raster calculator: `if (("one-of-macander-clipped-rasters" ))
4. Run `load_tifs_into_grass.sh` to import "water-mask-clipped", "macander-raster-clipped-to-ours" PFT maps, "our-raster-clipped-to-macander" PFT maps, and the "cavm-raster-clipped-to-macander" map into GRASS
5. Run `g.copy raster="water-mask-clipped",MASK` to create water mask (1=water)
6. Run `zonal_stats_grass` to calculate zonal statistics of our rasters within cavm