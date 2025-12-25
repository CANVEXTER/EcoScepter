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
    Relative, distribution-aware auto tuning for vegetation assessment.

    Anchors thresholds to dominant NDVI ranges (modes),
    mimicking expert visual tuning rather than percentile slicing.
    """

    # --------------------------------------------------
    # 1. Water threshold (unchanged, robust)
    # --------------------------------------------------
    water_t = float(np.nanpercentile(mndwi, 90))
    water_mask = mndwi > water_t

    ndvi_land = ndvi[~water_mask]
    ndvi_land = ndvi_land[np.isfinite(ndvi_land)]

    # Safety fallback
    if ndvi_land.size < 100:
        return {
            "water_t": water_t,
            "clear_t": 0.2,
            "veg_low": 0.4,
            "veg_high": 0.7,
        }

    # --------------------------------------------------
    # 2. NDVI histogram (relative logic)
    # --------------------------------------------------
    bins = np.linspace(-0.2, 1.0, 256)
    hist, edges = np.histogram(ndvi_land, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth histogram to suppress noise spikes
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)

    # --------------------------------------------------
    # 3. Detect dominant NDVI modes
    # --------------------------------------------------
    # Low / sparse vegetation peak
    low_region = centers < 0.5
    low_peak = centers[low_region][
        np.argmax(hist_smooth[low_region])
    ]

    # Dense / healthy vegetation peak
    high_region = centers >= 0.5
    high_peak = centers[high_region][
        np.argmax(hist_smooth[high_region])
    ]

    # --------------------------------------------------
    # 4. Derive thresholds RELATIVE to peaks
    # --------------------------------------------------
    clear_t = float(low_peak - 0.05)

    veg_low = float(
        low_peak + 0.15 * (high_peak - low_peak)
    )
    veg_high = float(
        low_peak + 0.70 * (high_peak - low_peak)
    )

    # Clamp to sane NDVI bounds
    clear_t = float(np.clip(clear_t, -0.1, 0.6))
    veg_low = float(np.clip(veg_low, clear_t + 0.05, 0.9))
    veg_high = float(np.clip(veg_high, veg_low + 0.05, 0.95))

    return {
        "water_t": water_t,
        "clear_t": clear_t,
        "veg_low": veg_low,
        "veg_high": veg_high,
    }
