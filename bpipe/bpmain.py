import os 
from glob import glob
from geotile import mosaic
from gutils import build_tileindex,build_vrt
from uvars import RSPROX_DPATH,BLOCKS_DPATH


tilenames_tls = [
 'N11E104',
 'N11E105',
 'N12E103',
 'N12E104',
 'N12E105',
 'N13E103',
 'N13E104',
 'N13E105']

roi = 'TLS'
blocks_dpath = f"{BLOCKS_DPATH}/{roi}"
os.makedirs(blocks_dpath, exist_ok=True)

print(
    """ 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #deva ::: GENERATE AUXILIARY FILES 
    """
)

grid_filesi = glob(f"{RSPROX_DPATH}/OUT_TILES/TILES12/*/*EDEM_GRID.tif")
egm_filesi = glob(f"{RSPROX_DPATH}/OUT_TILES/TILES12/*/*EGM08.tif")

tdemf_filesi = glob(f"{RSPROX_DPATH}/OUT_TILES/TILES12/*/vfill/*DEM__Fw_mlinterps.tif")
tdemv_filesi = glob(f"{RSPROX_DPATH}/OUT_TILES/TILES12/*/*tdem_DEM__Fw.tif")

gdemv_filesi = glob(f"{RSPROX_DPATH}/GEDI_GRID/tiles/*/*elev_lowestmode_mean_2019108_2022019_002_03_EPSG4326.tif")
gdemf_filesi = glob(f'{RSPROX_DPATH}/GEDI_GRID/tiles/*/*elev_lowestmode_mean_2019108_2022019_002_03_EPSG4326_mlinterps.tif')

gdsmf_filesi = glob(f"{RSPROX_DPATH}/GEDI_GRID/tiles/*/*rh100_mean_2019108_2022019_002_03_EPSG4326_mlinterps.tif")
gdsmv_filesi = glob(f"{RSPROX_DPATH}/GEDI_GRID/tiles/*/*rh100_mean_2019108_2022019_002_03_EPSG4326.tif")

gdemf_files = [i for i in gdemf_filesi for t in tilenames_tls if t in i]
gdemv_files = [i for i in gdemv_filesi for t in tilenames_tls if t in i]
tdemv_files = [i for i in tdemv_filesi for t in tilenames_tls if t in i]
tdemf_files = [i for i in tdemf_filesi for t in tilenames_tls if t in i]
gdsmf_files = [i for i in gdsmf_filesi for t in tilenames_tls if t in i]
gdsmv_files = [i for i in gdsmv_filesi for t in tilenames_tls if t in i]

assert len(tdemf_files) == len(tdemv_files) == len(gdemf_files) == len(gdemv_files),'files list != length'
assert len(gdsmv_files) == len(gdsmf_files) == len(gdemf_files) == len(gdemv_files),'files list != length'
grid_files = [i for i in grid_filesi for t in tilenames_tls if t in i]
egm_files = [i for i in egm_filesi for t in tilenames_tls if t in i]

assert len(tdemf_files) == len(tdemv_files) == len(grid_files) == len(egm_files),'files list != length'

filelist = [grid_files, egm_files, gdemf_files, gdemv_files,
            tdemf_files,tdemv_files,gdsmf_files,gdsmv_files]
namelist = ['grid', 'egm08','gdemf','gdemv',
            'tdemf','tdemv','gdsmf','gdsmv']

assert len(namelist) == len(filelist), 'list length different'
for name, files in zip(namelist, filelist):
    aux_dpath = f'{blocks_dpath}/auxfiles'
    print(aux_dpath)
    os.makedirs(aux_dpath, exist_ok=True)
    vrt = f'{aux_dpath}/{name}.vrt'
    print(vrt)
    build_vrt(files,vrt)

gpkg = f'{aux_dpath}/{roi}_tindex.gpkg'
build_tileindex(tdemf_files, gpkg)

print(
    """ 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #devb ::: GENERATE DONWX FILES 
    """
)

from uvars import var1_dpath,var2_dpath
os.makedirs(var1_dpath, exist_ok=True)
os.makedirs(var2_dpath,exist_ok=True)
from gutils import file_exists
from uvars import libd_rsdx_dpath
from uvars import gdemf_fn,gdsmf_fn,tdemf_fn,gdemd_fn,gdsmd_fn
import sys 
sys.path.append(libd_rsdx_dpath)
from utils import gwr_grid_downscaling # block tls 10min

file_exists(gdemf_fn)
file_exists(gdsmf_fn)
file_exists(tdemf_fn)

rfpath = gdemd_fn
hfpath = tdemf_fn
lfpath = gdemf_fn
epsg = 4326#4979

if not os.path.isfile(rfpath):
    gwr_grid_downscaling(xpath=hfpath, ypath=lfpath, 
                             opath=rfpath, oaux=False,
                             epsg_code=epsg,clean=True)
else:
    print('already created')



rfpath = gdsmd_fn
lfpath = gdsmf_fn
hfpath = tdemf_fn
if not os.path.isfile(rfpath):
    gwr_grid_downscaling(xpath=hfpath, ypath=lfpath, 
                             opath=rfpath, oaux=False,
                             epsg_code=epsg,clean=True)
else:
    print('already created')


print(
    """ 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #devb ::: GENERATE h2H FILES 
    """
)
from uvars import grid_fn, egm_fn, gdemdH_fn,gdemd_fn, gdsmdH_fn, gdsmf_fn
from gutils import cleanxmlfiles
from uvars import libb_prox_dpath
sys.path.append(libb_prox_dpath)
from rtransforms import raster_calc
# WORK OUT THE ORDER IN raster_calc , WHCICH WAY GYM @6PM 

cleanxmlfiles()
# import os
# import rasterio
# import numpy as np

# def clip_raster_to_match(reference_raster, target_raster):
#     """Clips `target_raster` to match the shape and extent of `reference_raster`."""
#     if target_raster.shape != reference_raster.shape:
#         print("Resizing target raster to match reference...")
#         min_height = min(reference_raster.shape[0], target_raster.shape[0])
#         min_width = min(reference_raster.shape[1], target_raster.shape[1])
#         target_raster = target_raster[:min_height, :min_width]
#         reference_raster = reference_raster[:min_height, :min_width]
#     return reference_raster, target_raster

# def raster_calc(mpath, apath, operation, output_path):
#     """Performs raster calculation ensuring rasters have the same shape."""
#     with rasterio.open(mpath) as m, rasterio.open(apath) as a:
#         m_data = m.read(1)
#         a_data = a.read(1)

#         # Ensure both rasters have the same shape
#         m_data, a_data = clip_raster_to_match(m_data, a_data)
        
#         # Perform the specified operation
#         if operation == '+':
#             result_data = m_data + a_data
#         elif operation == '-':
#             result_data = m_data - a_data
#         else:
#             raise ValueError("Invalid operation. Use '+' or '-'.")

#         # Save the result
#         profile = m.profile
#         profile.update({"height": result_data.shape[0], "width": result_data.shape[1]})

#         with rasterio.open(output_path, "w", **profile) as dst:
#             dst.write(result_data, 1)



operation = '+'  # Change to '-' if subtraction is needed
apath = grid_fn
mpath = gdemd_fn
opath = gdemdH_fn

if not os.path.isfile(opath):
    raster_calc(mpath, apath, operation='add', output_path=opath)
else:
    print('already created')

# if os.path.isfile(opath):
#     print(f"Skipping processing: {opath} already exists.")
# else:
#     print(f"Processing {opath}...")
#     raster_calc(mpath, apath, operation=operation, output_path=opath)

##########################################################################
# VAR 2 
##########################################################################



mpath = gdsmd_fn
opath = gdsmdH_fn
operation = '+'  # Change to '-' if needed
if not os.path.isfile(opath):
    raster_calc(mpath, apath, operation='add', output_path=opath)
else:
    print('already created')

# if os.path.isfile(opath):
#     print(f"Skipping processing: {opath} already exists.")
# else:
#     print(f"Processing {opath}...")
#     raster_calc(mpath, apath, operation=operation, output_path=opath)


print(
    """ 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #devd ::: GENERATE ZCORR FILES 
    """
)
