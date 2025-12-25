# scripts/change.py
import numpy as np
from numpy.polynomial.polynomial import polyfit


def delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape")
    return b - a


def ndvi_slope(years: np.ndarray, ndvi_stack: np.ndarray) -> np.ndarray:
    T, H, W = ndvi_stack.shape
    slope = np.full((H, W), np.nan, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            y = ndvi_stack[:, i, j]
            if np.count_nonzero(~np.isnan(y)) >= 3:
                b, m = polyfit(years, y, 1)
                slope[i, j] = m

    return slope


def normalize(arr: np.ndarray) -> np.ndarray:
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)


def composite_change_score(
    d_ndvi: np.ndarray,
    d_ndbi: np.ndarray,
    ndvi_baseline: np.ndarray,
    w_ndvi: float = 0.7,
    w_ndbi: float = 0.3,
) -> np.ndarray:
    """
    NDVI loss is weighted by baseline vegetation density.
    """
    if d_ndvi.shape != d_ndbi.shape:
        raise ValueError("Input arrays must have the same shape")

    ndvi_weight = np.clip(ndvi_baseline, 0.0, 1.0)

    ndvi_loss = normalize((-d_ndvi) * ndvi_weight)
    ndbi_gain = normalize(d_ndbi)

    return (w_ndvi * ndvi_loss) + (w_ndbi * ndbi_gain)
