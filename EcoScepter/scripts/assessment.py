# scripts/assessment.py
import numpy as np
from scipy.ndimage import gaussian_filter1d

from scripts.indices import compute_ndvi, compute_mndwi, compute_ndbi
from scripts.masking import valid_data_mask
from scripts.change import normalize


def assess_vegetation(arr: np.ndarray) -> dict:
    """
    Per-image vegetation assessment (signals only).
    NO thresholds, NO classification, NO visualization.
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


def auto_tune_assessment_thresholds(
    ndvi: np.ndarray,
    mndwi: np.ndarray,
) -> dict:
    """
    Scene-adaptive threshold tuning.

    - Works for any geography
    - No fixed NDVI assumptions
    - Relative to scene statistics
    - Stable across land-cover extremes
    """

    # -------------------------------
    # 1. Water detection (robust)
    # -------------------------------
    water_t = float(np.nanpercentile(mndwi, 90))
    water_mask = mndwi > water_t

    # -------------------------------
    # 2. Land NDVI extraction
    # -------------------------------
    ndvi_land = ndvi[~water_mask]
    ndvi_land = ndvi_land[np.isfinite(ndvi_land)]

    # Fallback for tiny scenes
    if ndvi_land.size < 500:
        return {
            "water_t": water_t,
            "clear_t": 0.2,
            "veg_low": 0.4,
            "veg_high": 0.8,
        }

    # -------------------------------
    # 3. Quantile-based tuning
    # -------------------------------
    q20 = np.nanpercentile(ndvi_land, 20)
    q50 = np.nanpercentile(ndvi_land, 50)
    q80 = np.nanpercentile(ndvi_land, 80)
    q95 = np.nanpercentile(ndvi_land, 95)

    # Clear / bare land
    clear_t = float(q20)

    # Vegetation range adapts to scene
    veg_low = float(q50)
    veg_high = float(q95)

    # -------------------------------
    # 4. Physical sanity clamps
    # -------------------------------
    clear_t = float(np.clip(clear_t, -0.1, 0.6))
    veg_low = float(np.clip(veg_low, clear_t + 0.05, 0.85))
    veg_high = float(np.clip(veg_high, veg_low + 0.05, 0.95))

    return {
        "water_t": water_t,
        "clear_t": clear_t,
        "veg_low": veg_low,
        "veg_high": veg_high,
    }

