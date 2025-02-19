import subprocess
import os 
from glob import glob 

def cleanxmlfiles(path=None):
    if path is None:
        path = "/media/ljp238/12TBWolf/RSPROX/OUT_TILES/BLOCKS/*/*/*.aux.xml"
    files = glob(path)
    for fo in files:
        if os.path.isfile(fo):  # Check if it's a file
            print(f'Removing {fo}...')
            os.remove(fo)
        else:
            print(f'Skipping directory: {fo}')

def file_exists(path):
    if os.path.isfile(path):
        print('YES')

def list2txt(txt_file,filelist):
    with open(txt_file, 'w') as f:
        for tif in filelist:
            f.write(f"{tif}\n")
    print('list2txt')
    
def build_vrt(filelist,vrt_file):

    # g
    txt_file = vrt_file.replace('.vrt', '.txt')
   # gpkg_file = vrt_file.replace('.vrt', '.gpkg')

    if not os.path.isfile(vrt_file):
        list2txt(txt_file,filelist)
        subprocess.run(['gdalbuildvrt', '-input_file_list', txt_file, vrt_file])

    print(f"VRT created: {vrt_file}")
# do the other vrt python function too []

def build_tileindex(filelist, gkpg_file):
    txt_file = gkpg_file.replace('.gpkg', '.txt')
    if not os.path.isfile(gkpg_file):
        list2txt(txt_file,filelist)
        subprocess.run(['gdaltindex', gkpg_file,'--optfile', txt_file])
        print(f"GPKG created: {gkpg_file}")

        