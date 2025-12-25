# scripts/masking.py
import numpy as np

def valid_data_mask(arr: np.ndarray) -> np.ndarray:
    """
    Generates a validity mask for multi-band arrays.
    True = valid pixel, False = invalid (NaN or inf in any band).
    """
    if arr.ndim != 3:
        raise ValueError("Expected array shape (bands, H, W)")
    mask = np.ones(arr.shape[1:], dtype=bool)
    for b in range(arr.shape[0]):
        band = arr[b]
        mask &= np.isfinite(band)
    return mask

def water_mask_from_mndwi(mndwi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Generates a water mask from MNDWI.
    True = water pixel, False = non-water.
    Default threshold aligns with common practice.
    """
    return mndwi > threshold

def apply_masks(*arrays: np.ndarray, mask: np.ndarray) -> list:
    """
    Applies a boolean mask to one or more arrays.
    Masked pixels are set to NaN.

    Parameters
    ----------
    arrays : np.ndarray
        Arrays with shape (H, W)
    mask : np.ndarray
        Boolean mask where True = keep, False = mask out

    Returns
    -------
    list[np.ndarray]
        Masked arrays
    """
    out = []
    for arr in arrays:
        if arr.shape != mask.shape:
            raise ValueError("Mask and array shape mismatch")
        masked = arr.copy()
        masked[~mask] = np.nan
        out.append(masked)
    return out
