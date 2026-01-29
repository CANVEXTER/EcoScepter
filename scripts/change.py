import numpy as np

# --- HELPER (Restored to fix import error in assessment.py) ---
def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Min-Max normalization to [0, 1].
    Used by assessment.py for visualization, NOT used for change detection logic.
    """
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)

# --- CORE LOGIC ---

def delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape")
    return b - a

def ndvi_slope(years: np.ndarray, ndvi_stack: np.ndarray) -> np.ndarray:
    t = years[:, None, None]
    y = ndvi_stack.astype(np.float32)

    valid = ~np.isnan(y)
    count = np.sum(valid, axis=0)

    t_mean = np.nanmean(t, axis=0)
    y_mean = np.nanmean(y, axis=0)

    cov = np.nanmean((t - t_mean) * (y - y_mean), axis=0)
    var = np.nanmean((t - t_mean) ** 2, axis=0)

    slope = cov / (var + 1e-6)
    slope[count < 3] = np.nan

    return slope

def composite_change_score(
    d_ndvi: np.ndarray,
    d_ndbi: np.ndarray,
    ndvi_baseline: np.ndarray,
    w_ndvi: float = 0.7,
    w_ndbi: float = 0.3,
) -> np.ndarray:
    """
    Calculates a raw physical change score.
    Positive Score = Likely Degradation.
    """
    if d_ndvi.shape != d_ndbi.shape:
        raise ValueError("Input arrays must have the same shape")

    # Weight NDVI loss by how much vegetation was there originally
    ndvi_weight = np.clip(ndvi_baseline, 0.0, 1.0)

    # d_ndvi is (Current - Baseline). Negative means loss.
    # Invert so "Loss" is positive
    ndvi_loss = -d_ndvi 
    
    # Calculate weighted loss
    weighted_loss = ndvi_loss * ndvi_weight
    
    # NDBI Gain: Positive means more built-up
    ndbi_gain = d_ndbi 

    # Composite Score (Physical range, approx -1 to 1)
    # Clip NDBI to avoid extreme noise
    score = (w_ndvi * weighted_loss) + (w_ndbi * np.clip(ndbi_gain, -0.5, 0.5))

    return score

# EcoScepter/scripts/change.py
import numpy as np

# ... [Keep existing delta, ndvi_slope, composite_change_score functions exactly as they are] ...

# --- ADD THIS NEW FUNCTION ---
def compute_extended_stats(
    change_arr: np.ndarray, 
    mask: np.ndarray, 
    loss_mask: np.ndarray, 
    gain_mask: np.ndarray
) -> dict:
    """
    Computes intensity statistics for reporting.
    """
    # Extract valid pixels only
    valid_change = change_arr[mask]
    
    # Loss Intensity (how bad is the loss where it is happening?)
    loss_vals = change_arr[mask & loss_mask]
    avg_loss = np.mean(loss_vals) if loss_vals.size > 0 else 0.0
    
    # Gain Intensity (how strong is the recovery?)
    gain_vals = change_arr[mask & gain_mask]
    avg_gain = np.mean(gain_vals) if gain_vals.size > 0 else 0.0
    
    # Overall Distribution
    return {
        "avg_loss_val": float(avg_loss),
        "avg_gain_val": float(avg_gain),
        "std_dev": float(np.std(valid_change)) if valid_change.size > 0 else 0.0,
        "min_val": float(np.min(valid_change)) if valid_change.size > 0 else 0.0,
        "max_val": float(np.max(valid_change)) if valid_change.size > 0 else 0.0
    }