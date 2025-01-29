import os
import rasterio
import rasterio.mask
import rasterio.features
from rasterio.mask import mask


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


   