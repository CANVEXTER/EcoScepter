# scripts/visualize.py
import numpy as np
from matplotlib import cm

def stretch(img: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    """
    Percentile stretch for visualization only.

    Parameters
    ----------
    img : np.ndarray
        2D or 3D image
    pmin : float
        Lower percentile
    pmax : float
        Upper percentile

    Returns
    -------
    np.ndarray
        Stretched image in range [0, 1]
    """
    lo, hi = np.nanpercentile(img, (pmin, pmax))
    if hi == lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1)

def index_to_rgb(
    index: np.ndarray,
    cmap_name: str,
    pmin: float = 2,
    pmax: float = 98,
) -> np.ndarray:
    """
    Notebook-faithful spectral index visualization.

    - Percentile stretch (same as notebook)
    - Matplotlib colormap
    - Returns RGB float32 image in [0,1]
    """
    stretched = stretch(index, pmin, pmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(stretched)[..., :3]  # drop alpha
    return rgb.astype(np.float32)
