# finding values or surface for adj 

# value for ajd by optuna 
import time 
import rasterio
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score

def load_raster(filepath):
    """Load raster data and apply preprocessing (handle nodata, NaNs)."""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        
        # Handle nodata values properly
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        
        # Apply value filters
        data[(data > 1000) | (data < -30)] = np.nan
    
    return data

def load_mask(mask_path):
    """Load binary mask where 1 = valid area, 0 = exclude."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)
    
    return mask  # Returns 2D array with values 0 and 1

def optimize_adj_value(gedi_egm_file, reference_lidar_file, lwm_fpath):
    ti = time.perf_counter()
    """Optimize the adjustment value using Optuna to minimize RMSE, considering a mask."""
    
    # Load rasters
    gedi_data = load_raster(gedi_egm_file)
    lidar_data = load_raster(reference_lidar_file)
    mask = load_mask(lwm_fpath)  # Binary mask (0: exclude, 1: include)
    
    # Ensure mask is binary (0 or 1)
    mask = (mask == 1)  # Converts it to a boolean array

    # Define optimization function
    def objective(trial):
        adj = trial.suggest_float("adj", 0.8, 2.0)  # Search range
        adjusted_gedi = gedi_data - adj
        
        # Create valid mask (valid values & included in mask)
        valid_mask = ~np.isnan(adjusted_gedi) & ~np.isnan(lidar_data) & mask
        
        if np.sum(valid_mask) == 0:  # Avoid errors if no valid data points
            return float("inf")
        
        rmse = mean_squared_error(lidar_data[valid_mask], adjusted_gedi[valid_mask], squared=False)
        
        return rmse  # Optuna minimizes this value
    
    # Run optimization
    #study = optuna.create_study(direction="minimize")
    
    #study.optimize(objective, n_trials=20)  # Adjust trials as needed
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50, timeout=300) # # Run for 50 trials or 5 minutes
    
    best_adj = study.best_params["adj"]
    best_rmse = study.best_value
    
    # Compute R2 for best adj
    best_adjusted_gedi = gedi_data - best_adj
    valid_mask = ~np.isnan(best_adjusted_gedi) & ~np.isnan(lidar_data) & mask
    
    best_r2 = r2_score(lidar_data[valid_mask], best_adjusted_gedi[valid_mask]) if np.sum(valid_mask) > 0 else None
    tf = time.perf_counter() - ti 
    print(f'RUN.TIME {tf/60} min(s)')
    return {"Best Adj": best_adj, "Best RMSE": best_rmse, "Best R2": best_r2}





