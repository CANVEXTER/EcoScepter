import numpy as np
from scripts.masking import valid_data_mask
from scripts.indices import compute_ndvi, compute_mndwi, compute_ndbi
from scripts.change import normalize


def assess_vegetation(arr: np.ndarray) -> dict:
    """
    Per-image vegetation assessment (signals only).
    """
    valid_mask = valid_data_mask(arr)

    ndvi = compute_ndvi(arr)
    mndwi = compute_mndwi(arr)
    ndbi = compute_ndbi(arr)

    ndvi = np.where(valid_mask, ndvi, np.nan)
    mndwi = np.where(valid_mask, mndwi, np.nan)
    ndbi = np.where(valid_mask, ndbi, np.nan)

    veg_score = normalize(ndvi)

    return {
        "ndvi": ndvi,
        "mndwi": mndwi,
        "ndbi": ndbi,
        "veg_score": veg_score,
    }

def otsu_threshold(data: np.ndarray) -> float:
    """
    Computes the optimal threshold using Otsu's method (bimodal).
    Returns the threshold value that separates the two dominant classes.
    """
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0
        
    # Create histogram
    hist, bin_edges = np.histogram(data, bins=256, range=(-1, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram
    hist = hist.astype(float) / hist.sum()
    
    # Cumulative sum and mean
    weight1 = np.cumsum(hist)
    mean1 = np.cumsum(hist * bin_centers)
    
    # Global mean
    global_mean = mean1[-1]
    
    # Inter-class variance
    # Avoid division by zero
    valid_mask = (weight1 > 0) & (weight1 < 1)
    
    # Otsu variance maximization
    numerator = (global_mean * weight1[valid_mask] - mean1[valid_mask]) ** 2
    denominator = weight1[valid_mask] * (1.0 - weight1[valid_mask])
    
    if denominator.size == 0:
        return 0.0
        
    variance = numerator / denominator
    
    # Find max variance
    idx = np.argmax(variance)
    
    # Map back to real threshold value
    real_idx = np.where(valid_mask)[0][idx]
    return float(bin_centers[real_idx])

def auto_tune_assessment_thresholds(ndvi: np.ndarray, mndwi: np.ndarray) -> dict:
    """
    Advanced Auto-Tuner using Otsu's Method.
    Finds the NATURAL separation between classes.
    """
    # 1. Water Threshold (MNDWI)
    # Otsu is perfect for Water vs Land
    water_t = otsu_threshold(mndwi)
    # Sanity clamp: Water is rarely below -0.2 or above 0.3 in MNDWI
    water_t = float(np.clip(water_t, -0.2, 0.3))
    
    water_mask = mndwi > water_t
    
    # 2. Land Separation (NDVI)
    ndvi_land = ndvi[~water_mask]
    ndvi_land = ndvi_land[np.isfinite(ndvi_land)]
    
    if ndvi_land.size < 100:
        return {"water_t": water_t, "clear_t": 0.2, "veg_low": 0.4, "veg_high": 0.8}

    # Run Otsu on Land to separate "Soil/Urban" from "Vegetation"
    otsu_val = otsu_threshold(ndvi_land)
    
    # Otsu gives us the split center. We derive ranges from it.
    # If split is 0.4, then Clear < 0.4 and Veg > 0.4
    
    # We add a small buffer/transition zone around the Otsu threshold
    buffer = 0.05
    clear_t = float(otsu_val - buffer)
    veg_low = float(otsu_val + buffer)
    
    # Sanity Clamps (Physical Reality Check)
    # Even if Otsu says 0.1, we know dense veg isn't 0.1.
    clear_t = float(np.clip(clear_t, 0.1, 0.5))
    veg_low = float(np.clip(veg_low, 0.2, 0.6))
    
    # High vegetation is usually the 95th percentile of the "Vegetation" cluster
    veg_cluster = ndvi_land[ndvi_land > otsu_val]
    if veg_cluster.size > 0:
        veg_high = float(np.percentile(veg_cluster, 95))
    else:
        veg_high = 0.9

    return {
        "water_t": water_t,
        "clear_t": clear_t,
        "veg_low": veg_low,
        "veg_high": veg_high,
    }