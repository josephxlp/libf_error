import os
import rasterio
import rasterio.mask
import rasterio.features
from rasterio.mask import mask
import numpy as np 


import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from pathlib import Path

def crop_raster_by_vector(output_path: str, raster_path: str, vector_path: str):
    """
    Recorta um raster com base em um vetor.

    Parâmetros:
    output_path (str): Caminho do arquivo de saída.
    raster_path (str): Caminho do raster de entrada.
    vector_path (str): Caminho do vetor de entrada.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Arquivo {output_path} já existe. Pulando processamento.")
        return
    
    # Carregar o vetor e o raster
    gdf = gpd.read_file(vector_path)
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            print(f"O raster {raster_path} não possui um CRS definido. Atribuindo EPSG:4979.")
            raster_crs = "EPSG:4979"
        else:
            raster_crs = src.crs
        
        gdf = gdf.to_crs(raster_crs)  # Converter o CRS do vetor para o do raster
        nodata_value = src.nodata
        
        # Obter envelope do vetor
        bounding_box = gdf.geometry.unary_union.envelope
        
        # Recortar o raster
        out_image, out_transform = mask(src, [bounding_box], crop=True, all_touched=True, nodata=nodata_value)
        
        # Atualizar metadados
        out_meta = src.meta.copy()
        out_meta.update({"transform": out_transform, "nodata": nodata_value, "height": out_image.shape[1], "width": out_image.shape[2], "crs": raster_crs})
        
    # Salvar o raster recortado
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(out_image)
    
    print(f"Arquivo salvo em: {output_path}")

def clip_raster_percentile(input_path, output_path,lper=2,hper=80):
    output_path = output_path.replace('.tif', f'{lper}_{hper}.tif')

    if os.path.exists(output_path):
        print(f"Output raster '{output_path}' already exists. Operation not performed.")
        return
    
    with rasterio.open(input_path) as src:
        profile = src.profile
        data = src.read(1).astype(np.float32)  # Read first band
        
        # Set the src.nodata also to np.nan
        nodata_value = src.nodata
        data[data == nodata_value] = np.nan

        # Compute percentiles ignoring NaNs
        p2, p80 = np.nanpercentile(data, [lper, hper])
        
        # Mask values outside the range
        data[(data < p2) | (data > p80)] = np.nan
        
        # Replace NaN values with nodata value for saving
        data[np.isnan(data)] = nodata_value

        # Update the profile to handle NaNs
        profile.update(dtype='float32', nodata=nodata_value)

        # Save the output raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)


def adjust_for_geoid(input_raster, output_raster, constant, operation):
    """
    Apply an arithmetic operation between a raster and a constant value.
    
    Parameters:
    - input_raster (str): Path to the input raster file.
    - output_raster (str): Path to save the output raster file.
    - constant (float): The constant value to apply.
    - operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').
    """
    
    # Check if output raster already exists
    if os.path.exists(output_raster):
        print(f"Output raster '{output_raster}' already exists. Operation not performed.")
        return
    
    # Open input raster
    with rasterio.open(input_raster) as src:
        profile = src.profile
        data = src.read(1)  # Read first band
        
        # Perform the specified operation
        if operation == 'add':
            result = data + constant
        elif operation == 'subtract':
            result = data - constant
        elif operation == 'multiply':
            result = data * constant
        elif operation == 'divide':
            result = np.where(constant != 0, data / constant, data)  # Avoid division by zero
        else:
            raise ValueError("Invalid operation. Choose from 'add', 'subtract', 'multiply', or 'divide'.")
        
        # Ensure no data type overflow
        result = result.astype(profile['dtype'])
        
        # Save the result as a new raster
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(result, 1)
    
    print(f"Operation '{operation}' applied and saved to {output_raster}")


def mask_raster_by_vector(outdir, rpath,gdf,polyname,nvd = -9999):
    bname = os.path.basename(rpath).replace('.tif','')
    fopath = os.path.join(outdir, f'{bname}__{polyname}_masked.tif')

    src_srtm = rasterio.open(rpath)
    zion = gdf.to_crs(src_srtm.crs)
    out_image_mask, out_transform_mask = rasterio.mask.mask(
        src_srtm, 
        zion.geometry, 
        crop=False, 
        nodata=nvd
    )
    dst_kwargs = src_srtm.meta
    dst_kwargs.update(nodata=nvd)
    dst_kwargs
    new_dataset = rasterio.open(fopath, 'w', **dst_kwargs)
    new_dataset.write(out_image_mask)
    new_dataset.close()

def crop_raster_by_vetcor(outdir, rpath,gdf,polyname,nvd = -9999):
    bname = os.path.basename(rpath).replace('.tif','')
    fopath = os.path.join(outdir, f'{bname}__{polyname}_croped.tif')
    src_srtm = rasterio.open(rpath)
    zion = gdf.to_crs(src_srtm.crs)

    try:
        bb = zion.union_all().envelope
    except:
        bb = zion.geometry.unary_union

    out_image_crop, out_transform_crop = rasterio.mask.mask(
        src_srtm, 
        [bb], 
        crop=True, 
        all_touched=True, 
        nodata=nvd
    )
    dst_kwargs = src_srtm.meta
    dst_kwargs.update(nodata=nvd)
    dst_kwargs
    new_dataset = rasterio.open(fopath, 'w', **dst_kwargs)
    new_dataset.write(out_image_crop)
    new_dataset.close()

def cutline_raster(outdir, rpath, gdf, polyname):
    """
    Crops a raster using a vector mask (GeoDataFrame) and saves the result.

    Parameters:
    - outdir (str): Path to the output directory.
    - rpath (str): Path to the input raster file.
    - gdf (GeoDataFrame): GeoDataFrame containing the vector mask geometry.
    - polyname (str): Identifier for the output file name.
    - nvd (int/float/None): NoData value to assign to cropped raster. If None, a suitable value is inferred.

    Returns:
    - str: Path to the saved output raster.
    """
    # Generate output file path
    base_name = os.path.basename(rpath).replace('.tif', '')
    output_path = os.path.join(outdir, f'{base_name}__{polyname}_cutline.tif')

    # Open the input raster
    with rasterio.open(rpath) as src:
        # Reproject GeoDataFrame to match raster CRS
        gdf_aligned = gdf.to_crs(src.crs)
        # Apply the vector mask to crop the raster
        cropped_image, cropped_transform = mask(
            src, gdf_aligned.geometry, crop=True#, nodata=nvd
        )

        # Update metadata for the cropped raster
        metadata = src.meta.copy()
        metadata.update({
            #'nodata': nvd,
            'transform': cropped_transform,
            'width': cropped_image.shape[2],
            'height': cropped_image.shape[1],
            'driver': 'GTiff'
        })

    # Save the cropped raster to the output file
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(cropped_image)

    return output_path


   