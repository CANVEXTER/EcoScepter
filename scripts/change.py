# scripts/change.py
import numpy as np
from numpy.polynomial.polynomial import polyfit


def delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape")
    return b - a


def ndvi_slope(years: np.ndarray, ndvi_stack: np.ndarray) -> np.ndarray:
    t = years[:, None, None]              # (T, 1, 1)
    y = ndvi_stack.astype(np.float32)     # (T, H, W)

    valid = ~np.isnan(y)
    count = np.sum(valid, axis=0)

    t_mean = np.nanmean(t, axis=0)
    y_mean = np.nanmean(y, axis=0)

    cov = np.nanmean((t - t_mean) * (y - y_mean), axis=0)
    var = np.nanmean((t - t_mean) ** 2, axis=0)

    slope = cov / (var + 1e-6)
    slope[count < 3] = np.nan

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
