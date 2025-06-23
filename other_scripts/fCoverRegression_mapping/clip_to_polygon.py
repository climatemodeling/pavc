# this worked on qgis
from osgeo import gdal
import os

shapefile_path = '/mnt/poseidon/remotesensing/arctic/alaska_pft_fcover_harmonization/data/supporting_data/tundra_alaska_latlon/tundra_alaska_wgs84.shp'
shapefile_name_wext = os.path.basename(shapefile_path)
shapefile_name = os.path.splitext(shapefile_name_wext)[0]
mosaic_base_path = '/mnt/poseidon/remotesensing/arctic/data/rasters/model_results_tiled_test_06-19-2025'
mosaic_list = ['bryophyte_30M-5P-IQR3_mosaic', 
               'deciduous_shrub_30M-3C-IQR2_mosaic',
               ]

for mosaic in mosaic_list:
    str = f"""
    gdalwarp 
        -overwrite 
        -s_srs EPSG:4326 
        -t_srs EPSG:4326 
        -of GTiff 
        -tr 0.00017963752199490085 -0.0001796375219949009 
        -tap 
        -cutline {shapefile_path}
        -cl {shapefile_name}
        -multi 
        -co COMPRESS=DEFLATE 
        -co PREDICTOR=2 
        -co ZLEVEL=9 
        -ot Float32 
        -co BIGTIFF=YES 
        {mosaic_base_path}/{mosaic}.tif
        {mosaic_base_path}/{mosaic}_clipped.tif
    """