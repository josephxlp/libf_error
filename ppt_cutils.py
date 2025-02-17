"""

Final Recommendation
🚀 Best approach depends on your dataset size & computational power

Use patch-based Optuna if you want local adjustments without too much compute cost.
Use ML regression if you have a large dataset & want fast per-pixel predictions.
Use pixel-wise gradient descent if you need fine-tuned per-pixel adjustments.
Would you like me to implement a patch-based Optuna approach or a machine learning-based prediction model? 🚀

https://chatgpt.com/share/67b33339-d064-800e-9b8d-6498cdf7ad9f
"""


# finding values or surface for adj 
# add wanb logging 
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



"""
Challenges & Considerations
Dimensionality Explosion

1.Instead of 1 scalar adj value, we now have N pixels to optimize independently.
If the image is 1000x1000, this means optimizing 1 million separate adj values.
This is computationally expensive and impractical with traditional optimization techniques like Optuna.
Memory & Compute Constraints

2.Running a separate optimization for each pixel is infeasible on large rasters.
A direct per-pixel Optuna optimization would require millions of trials, which is impractical.
Alternative Approaches

3.Grid-based optimization: Optimize adj values for small patches instead of individual pixels.
Machine learning regression: Train a model (e.g., CatBoost, Random Forest) to predict the best adj per pixel based on image features.
Pixel-wise gradient descent: Use a per-pixel loss function and optimize adj via gradient-based methods.

"""

"""

✅ 1. Patch-Based Optimization (Compromise)
Instead of optimizing every pixel independently, divide the image into small tiles (e.g., 32x32 or 64x64 patches) and find an optimal adj per patch.

patch_size = 32  # Define patch size (e.g., 32x32)
height, width = gedi_data.shape
adj_map = np.zeros((height, width))  # Store per-patch adjustments

for i in range(0, height, patch_size):
    for j in range(0, width, patch_size):
        patch_gedi = gedi_data[i:i+patch_size, j:j+patch_size]
        patch_lidar = lidar_data[i:i+patch_size, j:j+patch_size]

        # Run Optuna optimization for this patch
        best_adj = run_optuna_for_patch(patch_gedi, patch_lidar)
        adj_map[i:i+patch_size, j:j+patch_size] = best_adj  # Apply to the whole patch

        
✅ Computationally feasible
✅ Captures spatial variations in adj
✅ More accurate than a single global adj
"""

"""
✅ 2. Machine Learning-Based Adjustment Prediction
Use a ML model (e.g., CatBoost, Random Forest, or CNN) to predict adj for each pixel based on its DEM, Sentinel-1, Sentinel-2, and other features.

# Train a model to predict adj based on pixel features
X_train = extract_features(gedi_data, lidar_data, sentinel_data)
y_train = compute_optimal_adj(X_train)

model = train_regressor(X_train, y_train)  # CatBoost, Random Forest, etc.
predicted_adj = model.predict(X_test)

✅ Fast inference once trained
✅ Can generalize across images
✅ Better than brute-force optimization

"""


"""
✅ 3. Pixel-Wise Gradient Descent Optimization
Instead of using Optuna, apply a per-pixel loss function and use gradient descent to optimize adj for each pixel.

learning_rate = 0.01
adj_map = np.ones_like(gedi_data)  # Start with uniform adj values

for _ in range(100):  # Iterate to optimize adj per pixel
    adjusted_gedi = gedi_data - adj_map
    gradient = compute_loss_gradient(adjusted_gedi, lidar_data)
    adj_map -= learning_rate * gradient  # Update per-pixel adj

✅ Fine-tunes adj at pixel level
✅ Computationally efficient compared to brute-force Optuna 
"""

"""
Final Recommendation
🚀 Best approach depends on your dataset size & computational power

Use patch-based Optuna if you want local adjustments without too much compute cost.
Use ML regression if you have a large dataset & want fast per-pixel predictions.
Use pixel-wise gradient descent if you need fine-tuned per-pixel adjustments.
Would you like me to implement a patch-based Optuna approach or a machine learning-based prediction model? 🚀

"""




