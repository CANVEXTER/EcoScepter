import numpy as np

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