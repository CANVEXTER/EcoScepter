# scripts/io.py
import rasterio
import numpy as np

BAND_ORDER = {
    "R": 0,
    "G": 1,
    "B": 2,
    "NIR": 3,
    "SWIR": 4,
}

def read_bands(path: str) -> np.ndarray:
    """
    Reads a multi-band GeoTIFF.

    Assumed band order:
    1: Red
    2: Green
    3: Blue
    4: NIR
    5: SWIR

    Returns
    -------
    np.ndarray
        Array of shape (bands, height, width), dtype float32
    """
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr
