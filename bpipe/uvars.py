
import os 
from glob import glob

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# deva  
tilenames_tls = [
 'N11E104',
 'N11E105',
 'N12E103',
 'N12E104',
 'N12E105',
 'N13E103',
 'N13E104',
 'N13E105']

RSPROX_DPATH = "/media/ljp238/12TBWolf/RSPROX"
BLOCKS_DPATH = "/media/ljp238/12TBWolf/RSPROX/OUT_TILES/BLOCKS"

grid_filesi = glob(f"{RSPROX_DPATH}/OUT_TILES/TILES12/*/*EDEM_GRID.tif")
egm_filesi = glob(f"{RSPROX_DPATH}/OUT_TILES/TILES12/*/*EGM08.tif")

tdemf_filesi = glob(f"{RSPROX_DPATH}/OUT_TILES/TILES12/*/vfill/*DEM__Fw_mlinterps.tif")
tdemv_filesi = glob(f"{RSPROX_DPATH}/OUT_TILES/TILES12/*/*tdem_DEM__Fw.tif")

gdemv_filesi = glob(f"{RSPROX_DPATH}/GEDI_GRID/tiles/*/*elev_lowestmode_mean_2019108_2022019_002_03_EPSG4326.tif")
gdemf_filesi = glob(f'{RSPROX_DPATH}/GEDI_GRID/tiles/*/*elev_lowestmode_mean_2019108_2022019_002_03_EPSG4326_mlinterps.tif')

gdsmf_filesi = f"{RSPROX_DPATH}/GEDI_GRID/tiles/*/rh100_mean_2019108_2022019_002_03_EPSG4326_mlinterps.tif"
gdsmv_filesi = f"{RSPROX_DPATH}/GEDI_GRID/tiles/*/*rh100_mean_2019108_2022019_002_03_EPSG4326.tif"

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# devb  
libd_rsdx_dpath = '/home/ljp238/Documents/UoE/libd_rsdx'
libb_prox_dpath = '/home/ljp238/Documents/UoE/libb_prox'
roi = 'TLS'

gdemf_fn = f'{BLOCKS_DPATH}/{roi}/auxfiles/gdemf.vrt'
gdsmf_fn = f"{BLOCKS_DPATH}/{roi}/auxfiles/gdsmf.vrt"
tdemf_fn = f'{BLOCKS_DPATH}/{roi}/auxfiles/tdemf.vrt'

grid_fn = f"{BLOCKS_DPATH}/{roi}/auxfiles/grid.vrt"
egm_fn = f'{BLOCKS_DPATH}/{roi}/auxfiles/egm08.vrt'

var1 = 'DTM'
var2 = 'DSM'

var1_dpath = f'{BLOCKS_DPATH}/{roi}/DOWNX/{var1}'
var2_dpath = f'{BLOCKS_DPATH}/{roi}/DOWNX/{var2}'

var1_dpath,var2_dpath


gdemd_fn = f'{var1_dpath}/{roi}_{var1}_h.tif'
gdsmd_fn = f'{var2_dpath}/{roi}_{var2}_h.tif'

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# devc 
gdemdH_fn = f'{var1_dpath}/{roi}_{var1}_H.tif'
gdsmdH_fn = f'{var2_dpath}/{roi}_{var1}_H.tif'

