# scripts/indices.py
import numpy as np

def safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """
    Safe division that returns NaN where denominator is zero.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom == 0, np.nan, numer / denom)

def compute_ndvi(arr: np.ndarray) -> np.ndarray:
    """
    NDVI = (NIR - R) / (NIR + R)
    Assumes band order: R,G,B,NIR,SWIR
    """
    R = arr[0]
    NIR = arr[3]
    return safe_div(NIR - R, NIR + R)

def compute_mndwi(arr: np.ndarray) -> np.ndarray:
    """
    MNDWI = (G - SWIR) / (G + SWIR)
    Used ONLY for water masking.
    """
    G = arr[1]
    SWIR = arr[4]
    return safe_div(G - SWIR, G + SWIR)

def compute_ndbi(arr: np.ndarray) -> np.ndarray:
    """
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    Supporting indicator only.
    """
    NIR = arr[3]
    SWIR = arr[4]
    return safe_div(SWIR - NIR, SWIR + NIR)
