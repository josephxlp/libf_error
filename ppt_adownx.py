import os 
import sys 
from glob import glob 

from upaths import libd_rsdx_dpath 
sys.path.append(libd_rsdx_dpath)

from utils import gwr_grid_downscaling
epsg = 4326

def downx_gwr(hfpath,lfpath,outdir,dset,epsg):
    if os.path.isfile(hfpath) & os.path.isfile(lfpath):
        print('lfpath and hfpath exist' )
        tilename = lfpath.split('/')[-2]
        tile_dopath = os.path.join(outdir, tilename)
        os.makedirs(tile_dopath,exist_ok=True)
    rfpath = os.path.join(tile_dopath, f'{tilename}_{dset}_WGS.tif')
    if not os.path.isfile(rfpath):
        gwr_grid_downscaling(xpath=hfpath, ypath=lfpath, 
                             opath=rfpath, oaux=False,
                             epsg_code=epsg,clean=True)
    else:
        print('already created')

bpattern = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/DEMVFILL/TILES12/*/*_mlinterps.tif"
tpattern = "/media/ljp238/12TBWolf/RSPROX/GEDI_GRID/tiles/*/*GEDI03_elev_lowestmode_mean_2019108_2022019_002_03_EPSG4326_mlinterps.tif"
bfiles = glob(bpattern)
tfiles = glob(tpattern)
assert len(bfiles) == len(tfiles), 'files donnot match'
print(len(bfiles))
print(len(tfiles))
outdir = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/DEMDOWNX/TILE12/"
dset = 'GEDI03_dtm'

for i in range(len(tfiles)):
    #if i > 0 : break
    lfpath = tfiles[i]
    hfpath = bfiles[i]
    downx_gwr(hfpath,lfpath,outdir,dset,epsg)


