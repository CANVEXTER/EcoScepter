# scripts/io.py
import rasterio
import numpy as np
from rasterio.warp import transform_bounds

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
    """
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr

def read_metadata(path: str) -> dict:
    """
    Extracts rich metadata including WGS84 bounds, Center Point, and Tags.
    """
    with rasterio.open(path) as src:
        # 1. Basic properties
        meta = {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs_raw": src.crs.to_string() if src.crs else "Undefined",
            "nodata": src.nodata,
            "tags": src.tags()  # Reads textual metadata (Date, Software, etc.)
        }

        # 2. Get Native Bounds
        bounds = src.bounds
        meta["bounds_native"] = {
            "left": bounds.left,
            "bottom": bounds.bottom,
            "right": bounds.right,
            "top": bounds.top
        }

        # 3. Convert to Lat/Lon (WGS84) if possible
        try:
            # transform_bounds(src_crs, dst_crs, left, bottom, right, top)
            # EPSG:4326 is the standard Lat/Lon coordinate system
            if src.crs:
                wgs_b = transform_bounds(src.crs, "EPSG:4326", 
                                         bounds.left, bounds.bottom, 
                                         bounds.right, bounds.top)
                
                meta["bounds_wgs84"] = {
                    "min_lon": wgs_b[0], # West
                    "min_lat": wgs_b[1], # South
                    "max_lon": wgs_b[2], # East
                    "max_lat": wgs_b[3]  # North
                }
                
                # Calculate Center Point
                meta["center"] = {
                    "lat": (wgs_b[1] + wgs_b[3]) / 2,
                    "lon": (wgs_b[0] + wgs_b[2]) / 2
                }
            else:
                meta["error"] = "CRS Undefined - Cannot determine Lat/Lon"
        except Exception as e:
            meta["error"] = str(e)

        return meta