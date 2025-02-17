import os 
import sys 
from upaths import libb_prox_dpath
from glob import glob

sys.path.append(libb_prox_dpath)
from rtransforms import raster_calc

grid_pattern = "/media/ljp238/12TBWolf/RSPROX/OUT_TILES/TILES12/*/*_EDEM_GRID.tif" 
dowx_pattern = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/DEMDOWNX/TILE12/*/*_GEDI03_dtm_WGS.tif"

dfiles = glob(dowx_pattern)
gfiles = glob(grid_pattern)
assert len(gfiles) == len(dfiles), 'files donnot match'

for i in range(len(dfiles)):
    #if i > 0 : break
    dfpath = dfiles[i]
    gfpath = gfiles[i]
    #print(f'{hfpath}\n{lfpath}')
    if os.path.isfile(gfpath) & os.path.isfile(dfpath):
        rfpath = dfpath.replace('WGS','EGM')
    if not os.path.isfile(rfpath):
        raster_calc(dfpath, gfpath, operation='add', output_path=rfpath)
    else:
        print('already created')