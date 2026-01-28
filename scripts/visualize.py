# scripts/visualize.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colormaps

def stretch(img: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    """
    Percentile stretch for visualization only.
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
    """
    stretched = stretch(index, pmin, pmax)
    cmap = colormaps[cmap_name]
    rgb = cmap(stretched)[..., :3]  # drop alpha
    return rgb.astype(np.float32)

def plot_with_grid(
    img: np.ndarray,
    bounds: dict,
    title: str = None
) -> plt.Figure:
    """
    Wraps the image in a Matplotlib figure with Lat/Lon coordinates on axes.
    Moves ticks to Right and Bottom as requested.
    """
    # Create figure with darker background to match Streamlit theme
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    # Extent order for imshow: [left, right, bottom, top]
    # We expect bounds to have keys: min_lon, max_lon, min_lat, max_lat
    extent = [
        bounds["min_lon"], 
        bounds["max_lon"], 
        bounds["min_lat"], 
        bounds["max_lat"]
    ]

    # Plot
    ax.imshow(img, extent=extent)

    # Configure Axes Colors
    ax.tick_params(axis='x', colors='#a1a1aa', labelsize=8)
    ax.tick_params(axis='y', colors='#a1a1aa', labelsize=8)
    
    # --- KEY REQUEST: TICKS ON RIGHT AND BOTTOM ---
    ax.yaxis.tick_right()
    ax.xaxis.tick_bottom()
    
    # Format labels (4 decimal places for precision)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    
    # Stylize Spines (Borders)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30333d')

    if title:
        ax.set_title(title, color='#e4e4e7', pad=15, fontsize=10)

    return fig