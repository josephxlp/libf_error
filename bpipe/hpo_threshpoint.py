"""

Final Recommendation
🚀 Best approach depends on your dataset size & computational power

Use patch-based Optuna if you want local adjustments without too much compute cost.
Use ML regression if you have a large dataset & want fast per-pixel predictions.
Use pixel-wise gradient descent if you need fine-tuned per-pixel adjustments.
Would you like me to implement a patch-based Optuna approach or a machine learning-based prediction model? 🚀


"""

import time
import rasterio
import numpy as np
import optuna
import os
import datetime
from sklearn.metrics import mean_squared_error, r2_score

#add parameters to the logger like ntrila, direction and pbla bal bla 
# so basicilly anything worth tracking to the log file, aslo latter to be tracked with wandb

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Generate timestamp for log file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = os.path.join(log_dir, f"{timestamp}.log")

def log_message(message, verbose=True):
    """Logs a message to the console and the log file."""
    if verbose:
        print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

def load_raster(filepath, gridshape=None, verbose=True):
    """Load raster data, handle nodata/NaNs, and ensure shape consistency."""
    log_message(f"Loading raster: {filepath}", verbose)
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        data[(data > 1000) | (data < -30)] = np.nan

        if gridshape:
            data = crop_to_shape(data, gridshape, filepath, verbose)
    
    return data

def crop_to_shape(data, gridshape, filepath, verbose=True):
    """Ensure the raster has the correct shape by cropping if necessary."""
    if data.shape == gridshape:
        return data

    log_message(f"Cropping {filepath} from {data.shape} to {gridshape}", verbose)
    min_rows = min(data.shape[0], gridshape[0])
    min_cols = min(data.shape[1], gridshape[1])

    return data[:min_rows, :min_cols]

def load_mask(mask_path, gridshape=None, verbose=True):
    """Load binary mask and crop if needed."""
    log_message(f"Loading mask: {mask_path}", verbose)
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)
    
    return crop_to_shape(mask, gridshape, mask_path, verbose) if gridshape else mask

def optimize_adj_value(mod_file, ref_file, lwm_fpath=None, 
                       gridshape=(9001,9001), direction="minimize",
                       n_trials=50, timeout=300, verbose=True):
    """
    Optimize the adjustment value using Optuna to minimize RMSE.
    Ensures all rasters are aligned to `gridshape`.
    """
    log_message("Starting optimization process...", verbose)
    ti = time.perf_counter()
    
    # Load rasters with enforced shape
    mod_data = load_raster(mod_file, gridshape, verbose)
    ref_data = load_raster(ref_file, gridshape, verbose)

    # Load mask if provided, otherwise use all valid pixels
    if lwm_fpath:
        mask = load_mask(lwm_fpath, gridshape, verbose)
        mask = (mask == 1)
    else:
        mask = np.ones_like(mod_data, dtype=bool)

    def objective(trial):
        adj = trial.suggest_float("adj", 0.8, 2.0)
        adjusted_mod_data = mod_data - adj

        valid_mask = ~np.isnan(adjusted_mod_data) & ~np.isnan(ref_data) & mask

        if np.sum(valid_mask) == 0:
            return float("inf")
        
        rmse = mean_squared_error(ref_data[valid_mask], adjusted_mod_data[valid_mask], squared=False)
        return rmse

    # Run optimization
    log_message(f"Running Optuna with {n_trials} trials and {timeout} seconds timeout...", verbose)
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_adj = study.best_params["adj"]
    best_rmse = study.best_value

    # Compute R2 for best adj
    best_adjusted_mod_data = mod_data - best_adj
    valid_mask = ~np.isnan(best_adjusted_mod_data) & ~np.isnan(ref_data) & mask

    best_r2 = r2_score(ref_data[valid_mask], best_adjusted_mod_data[valid_mask]) if np.sum(valid_mask) > 0 else None

    tf = time.perf_counter() - ti
    log_message(f"Optimization complete. Runtime: {tf/60:.2f} min(s)", verbose)

    results = {"Best Adj": best_adj, "Best RMSE": best_rmse, "Best R2": best_r2}
    log_message(f"Results: {results}", verbose)

    return results

# # Example Usage
# if __name__ == "__main__":
#     mod_fn = "path/to/mod.tif"
#     ref_fn = "path/to/ref.tif"
#     lwm_fn = None  # Can be set to "path/to/mask.tif"

#     minimize = optimize_adj_value(mod_fn, ref_fn, lwm_fpath=lwm_fn, 
#                                   gridshape=(9001,9001),
#                                   direction="minimize", 
#                                   n_trials=50, timeout=100, verbose=True)



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
https://chatgpt.com/share/67b33339-d064-800e-9b8d-6498cdf7ad9f
"""




