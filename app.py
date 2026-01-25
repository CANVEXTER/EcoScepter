import os
import glob
import streamlit as st
import numpy as np
from matplotlib import colormaps
from scipy.ndimage import binary_opening

from scripts.io import read_bands
from scripts.visualize import stretch, index_to_rgb
from scripts.indices import compute_ndvi, compute_mndwi, compute_ndbi
from scripts.masking import valid_data_mask, water_mask_from_mndwi, apply_masks
from scripts.change import delta, composite_change_score
from scripts.thresholds import aggressiveness_to_threshold, apply_threshold
from scripts.assessment import (
    assess_vegetation,
    auto_tune_assessment_thresholds,
)

DATA_DIR = "data"

REFERENCE_THRESHOLDS = {
    "water_t": 0.0,
    "clear_t": 0.60,
    "veg_low": 0.60,
    "veg_high": 0.90,
}

# Page config - force dark theme
st.set_page_config(
    layout="wide",
    page_title="Vegetation & Change Detection Analysis",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark mode styling
st.markdown("""
<style>
    /* Force dark theme colors */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4ade80;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #334155;
        color: #fafafa;
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #1e3a28;
        border-left: 4px solid #4ade80;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #e5e7eb;
    }
    .warning-box {
        background-color: #3a2e1e;
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #e5e7eb;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #22c55e;
        color: #0e1117;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #16a34a;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4);
        color: #0e1117;
    }
    
    /* Expander */
    div[data-testid="stExpander"] {
        background-color: #1e293b;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    div[data-testid="stExpander"] summary {
        color: #fafafa;
    }
    
    /* Legend */
    .legend-container {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-top: 1rem;
    }
    .legend-title {
        margin-top: 0;
        color: #fafafa;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: #e5e7eb;
    }
    .legend-color {
        width: 30px;
        height: 20px;
        border-radius: 4px;
        margin-right: 10px;
        border: 1px solid #475569;
    }
    
    /* Stats card */
    .stats-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #4ade80;
        color: #fafafa;
        margin-top: 1rem;
    }
    .stats-title {
        margin: 0 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4ade80;
    }
    .stat-label {
        opacity: 0.8;
        color: #9ca3af;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af;
        background-color: transparent;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #334155;
        color: #4ade80;
    }
    
    /* Select boxes and inputs */
    .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {
        color: #e5e7eb !important;
        font-weight: 500;
    }
    
    /* Divider */
    hr {
        border-color: #334155;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4ade80;
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af;
    }
            
    /* Hide Streamlit warning / deprecation banners */
    div[data-testid="stAlert"] {
        display: none !important;
    }

</style>
""", unsafe_allow_html=True)


def sanitize_image(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def adjust_image_display(img: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    """
    Adjust brightness and contrast of image for display purposes.
    
    Parameters
    ----------
    img : np.ndarray
        Image array in range [0, 1]
    brightness : float
        Brightness multiplier (0.0 to 2.0, default 1.0)
    contrast : float
        Contrast multiplier (0.0 to 2.0, default 1.0)
    
    Returns
    -------
    np.ndarray
        Adjusted image
    """
    # Apply contrast
    img = (img - 0.5) * contrast + 0.5
    # Apply brightness
    img = img * brightness
    # Clip to valid range
    return np.clip(img, 0.0, 1.0)


# Header
st.markdown('<p class="main-header">Vegetation & Change Detection Analysis Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Monitor vegetation health and detect land cover changes using satellite imagery analysis</p>', unsafe_allow_html=True)

# Check for data files
tif_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif")))
if not tif_files:
    st.error("⚠ No GeoTIFF files found in the 'data/' directory. Please add satellite imagery to begin analysis.")
    st.info("📁 Expected file format: Multi-band GeoTIFF with bands in order: Red, Green, Blue, NIR, SWIR")
    st.stop()

# Show data status
with st.expander("▸ Data Status", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Images", len(tif_files))
    with col2:
        change_ready = "✓ Ready" if len(tif_files) >= 2 else "✗ Need 2+ images"
        st.metric("Change Detection", change_ready)
    
    st.write("**Loaded files:**")
    for i, f in enumerate(tif_files, 1):
        st.text(f"  {i}. {os.path.basename(f)}")

st.divider()

# Main tabs
tab_assess, tab_change, tab_help = st.tabs([
    "▸ Vegetation Assessment", 
    "▸ Change Detection",
    "▸ Help & Info"
])

# ===============================================================
# TAB 1 – VEGETATION ASSESSMENT
# ===============================================================
with tab_assess:
    st.markdown("### Analyze Vegetation Health from Single Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tif_name = st.selectbox(
            "◆ Select satellite image to analyze",
            options=[os.path.basename(f) for f in tif_files],
            help="Choose a GeoTIFF file containing multi-spectral satellite data"
        )
        tif_path = os.path.join(DATA_DIR, tif_name)
    
    with col2:
        st.write("")  # spacing
        st.write("")
        analyze_btn = st.button("▸ Run Analysis", type="primary", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("Processing satellite imagery..."):
            arr = read_bands(tif_path)
            result = assess_vegetation(arr)
            auto = auto_tune_assessment_thresholds(
                result["ndvi"], result["mndwi"]
            )
            
            st.session_state["assess"] = {
                "arr": arr,
                **result,
                **auto,
            }
            st.success("✓ Analysis complete!")

    if "assess" in st.session_state:
        a = st.session_state["assess"]
        
        st.divider()
        
        # Visualization controls in sidebar-style columns
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            st.markdown("#### ◆ Visualization")
            
            view_mode = st.radio(
                "Display Layer",
                [
                    "Vegetation Assessment",
                    "RGB (True Color)",
                    "NDVI (Vegetation)",
                    "MNDWI (Water)",
                    "NDBI (Built-up)",
                ],
                help="Choose which analysis layer to display"
            )
            
            # Image adjustment controls
            with st.expander("▸ Display Adjustments"):
                brightness = st.slider(
                    "Brightness",
                    0.0, 2.0, 1.0, 0.05,
                    help="Adjust image brightness"
                )
                contrast = st.slider(
                    "Contrast",
                    0.0, 2.0, 1.0, 0.05,
                    help="Adjust image contrast"
                )
            
            # Advanced controls
            with st.expander("▸ Advanced Settings"):
                use_reference = False
                if view_mode == "Vegetation Assessment":
                    use_reference = st.checkbox(
                        "Use standard thresholds",
                        value=True,
                        help="Uses standardized vegetation classification"
                    )
                    
                    if not use_reference:
                        st.markdown("**Custom Thresholds**")
                        water_t = st.slider(
                            "Water (MNDWI)",
                            -1.0, 1.0, float(a["water_t"]), 0.01
                        )
                        clear_t = st.slider(
                            "Clear Land (NDVI)",
                            -1.0, 1.0, float(a["clear_t"]), 0.01
                        )
                        veg_low, veg_high = st.slider(
                            "Vegetation Range",
                            -1.0, 1.0,
                            (float(a["veg_low"]), float(a["veg_high"])),
                            0.01
                        )
                    else:
                        water_t = REFERENCE_THRESHOLDS["water_t"]
                        clear_t = REFERENCE_THRESHOLDS["clear_t"]
                        veg_low = REFERENCE_THRESHOLDS["veg_low"]
                        veg_high = REFERENCE_THRESHOLDS["veg_high"]
                else:
                    water_t = a["water_t"]
                    clear_t = a["clear_t"]
                    veg_low = a["veg_low"]
                    veg_high = a["veg_high"]
        
        with col_right:
            # Generate visualization
            water = a["mndwi"] > water_t
            clear = (~water) & (a["ndvi"] < clear_t)
            veg_mask = (~water) & (~clear)
            
            rgb = sanitize_image(
                stretch(np.stack(a["arr"][:3], axis=-1))
            )
            
            # Display based on mode
            if view_mode == "RGB (True Color)":
                adjusted_rgb = adjust_image_display(rgb, brightness, contrast)
                st.image(adjusted_rgb, use_column_width=True, caption="True Color RGB Composite")
                
            elif view_mode == "NDVI (Vegetation)":
                ndvi_vis = adjust_image_display(
                    sanitize_image(index_to_rgb(a["ndvi"], "RdYlGn")),
                    brightness, contrast
                )
                st.image(
                    ndvi_vis,
                    use_column_width=True,
                    caption="NDVI - Normalized Difference Vegetation Index"
                )
                st.markdown("""
                <div class="info-box">
                <b>NDVI Scale:</b> Red = No vegetation, Yellow = Sparse, Green = Dense vegetation
                </div>
                """, unsafe_allow_html=True)
                
            elif view_mode == "MNDWI (Water)":
                mndwi_vis = adjust_image_display(
                    sanitize_image(index_to_rgb(a["mndwi"], "Blues")),
                    brightness, contrast
                )
                st.image(
                    mndwi_vis,
                    use_column_width=True,
                    caption="MNDWI - Modified Normalized Difference Water Index"
                )
                st.markdown("""
                <div class="info-box">
                <b>MNDWI Scale:</b> Darker blue = More water content
                </div>
                """, unsafe_allow_html=True)
                
            elif view_mode == "NDBI (Built-up)":
                ndbi_vis = adjust_image_display(
                    sanitize_image(index_to_rgb(a["ndbi"], "inferno")),
                    brightness, contrast
                )
                st.image(
                    ndbi_vis,
                    use_column_width=True,
                    caption="NDBI - Normalized Difference Built-up Index"
                )
                st.markdown("""
                <div class="info-box">
                <b>NDBI Scale:</b> Brighter colors = More urban/built-up areas
                </div>
                """, unsafe_allow_html=True)
                
            else:  # Vegetation Assessment
                overlay = rgb.copy()
                overlay[water] = [0.0, 0.3, 0.8]
                overlay[clear] = [0.55, 0.4, 0.25]
                
                ndvi = a["ndvi"].copy()
                ndvi[~veg_mask] = np.nan
                
                veg_norm = (ndvi - veg_low) / (veg_high - veg_low + 1e-6)
                veg_norm = np.clip(veg_norm, 0.0, 1.0)
                
                veg_rgb = colormaps["YlGn"](veg_norm)[..., :3]
                overlay[veg_mask] = veg_rgb[veg_mask]
                
                adjusted_overlay = adjust_image_display(overlay, brightness, contrast)
                st.image(
                    sanitize_image(adjusted_overlay),
                    use_column_width=True,
                    caption="Classified Vegetation Health Map"
                )
                
                # Legend
                st.markdown("""
                <div class="legend-container">
                    <h4 class="legend-title">◆ Land Cover Classification</h4>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(0, 77, 204);"></div>
                        <span><b>Water Bodies</b> - Rivers, lakes, flooded areas</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(140, 102, 64);"></div>
                        <span><b>Bare/Clear Land</b> - Soil, urban areas, cleared ground</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(200, 230, 130);"></div>
                        <span><b>Sparse Vegetation</b> - Grassland, crops, low density</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(120, 200, 80);"></div>
                        <span><b>Moderate Vegetation</b> - Mixed forests, mature crops</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(34, 139, 34);"></div>
                        <span><b>Dense Vegetation</b> - Mature forests, dense tree cover</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ===============================================================
# TAB 2 – CHANGE DETECTION
# ===============================================================
with tab_change:
    st.markdown("### Detect Land Cover Changes Between Two Time Periods")
    
    if len(tif_files) < 2:
        st.warning("⚠ At least two satellite images are required for change detection.")
        st.info("▸ Add more GeoTIFF files to the 'data/' directory to enable this feature.")
        st.stop()
    
    st.markdown(f"""
    <div class="info-box">
    <b>Comparison:</b> {os.path.basename(tif_files[0])} (baseline) → {os.path.basename(tif_files[-1])} (current)
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        compute_btn = st.button("▸ Compute Change Analysis", type="primary", use_container_width=True)
    
    if compute_btn:
        with st.spinner("Analyzing changes between images..."):
            arr1 = read_bands(tif_files[0])
            arr2 = read_bands(tif_files[-1])
            
            ndvi1 = compute_ndvi(arr1)
            ndvi2 = compute_ndvi(arr2)
            ndbi1 = compute_ndbi(arr1)
            ndbi2 = compute_ndbi(arr2)
            mndwi = compute_mndwi(arr2)
            
            valid_mask = valid_data_mask(arr2)
            water_mask = ~water_mask_from_mndwi(mndwi)
            mask = valid_mask & water_mask
            
            ndvi1_masked, ndvi2_masked, ndbi1_masked, ndbi2_masked = apply_masks(
                ndvi1, ndvi2, ndbi1, ndbi2, mask=mask
            )
            
            # Compute NDVI change
            ndvi_change = delta(ndvi1_masked, ndvi2_masked)
            
            # Compute composite score for degradation
            score = composite_change_score(
                ndvi_change,
                delta(ndbi1_masked, ndbi2_masked),
                ndvi_baseline=ndvi1_masked,
            )
            
            st.session_state["change_data"] = {
                "score": score,
                "ndvi_change": ndvi_change,
                "arr2": arr2,
            }
            st.success("✓ Change analysis complete!")
    
    if "change_data" in st.session_state:
        st.divider()
        
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            st.markdown("#### ◆ Detection Controls")
            
            # Change type selection
            st.markdown("**Change Types to Display**")
            show_degradation = st.checkbox("Vegetation Loss / Degradation", value=True, help="Show areas where vegetation decreased (purple)")
            show_improvement = st.checkbox("Vegetation Gain / Improvement", value=True, help="Show areas where vegetation increased (lime green)")
            
            st.divider()
            
            # Sensitivity control
            aggressiveness = st.slider(
                "Sensitivity Level",
                0.0, 1.0, 0.5, 0.05,
                help="Lower = Detect only major changes | Higher = Detect subtle changes"
            )
            
            sensitivity_label = (
                "● Conservative" if aggressiveness < 0.35 else
                "● Moderate" if aggressiveness < 0.65 else
                "● Aggressive"
            )
            st.markdown(f"**Current Mode:** {sensitivity_label}")
            
            st.divider()
            
            # Image adjustment controls
            with st.expander("▸ Display Adjustments"):
                brightness = st.slider(
                    "RGB Brightness",
                    0.0, 2.0, 1.0, 0.05,
                    help="Adjust background image brightness",
                    key="change_brightness"
                )
                contrast = st.slider(
                    "RGB Contrast",
                    0.0, 2.0, 1.0, 0.05,
                    help="Adjust background image contrast",
                    key="change_contrast"
                )
                overlay_alpha = st.slider(
                    "Overlay Opacity",
                    0.0, 1.0, 0.7, 0.05,
                    help="Transparency of change overlay"
                )
            
            st.write("")
            apply_btn = st.button("▸ Apply Detection", use_container_width=True)
            
            if apply_btn:
                change_data = st.session_state["change_data"]
                
                # Compute degradation mask
                thr = aggressiveness_to_threshold(
                    change_data["score"],
                    aggressiveness,
                )
                degradation_mask = apply_threshold(change_data["score"], thr)
                degradation_mask = binary_opening(degradation_mask, structure=np.ones((3, 3)))
                
                # Compute improvement mask based on NDVI change
                ndvi_change = change_data["ndvi_change"]
                # Use percentile-based threshold for improvement
                improvement_thr = np.nanpercentile(ndvi_change, 85 - aggressiveness * 15)
                improvement_mask = ndvi_change > improvement_thr
                improvement_mask = binary_opening(improvement_mask, structure=np.ones((3, 3)))
                
                st.session_state["change_masks"] = {
                    "degradation": degradation_mask,
                    "improvement": improvement_mask,
                    "brightness": brightness,
                    "contrast": contrast,
                    "overlay_alpha": overlay_alpha,
                }
                st.success("Detection applied!")
        
        with col_right:
            if "change_masks" in st.session_state:
                masks = st.session_state["change_masks"]
                change_data = st.session_state["change_data"]
                
                rgb = sanitize_image(
                    stretch(np.stack(change_data["arr2"][:3], axis=-1))
                )
                
                # Apply brightness and contrast adjustments
                rgb = adjust_image_display(rgb, masks["brightness"], masks["contrast"])
                overlay = rgb.copy()
                
                # Define colors
                degradation_color = np.array([0.6, 0.2, 0.8], dtype=np.float32)  # Purple
                improvement_color = np.array([0.7, 0.95, 0.3], dtype=np.float32)  # Lime green
                alpha = masks["overlay_alpha"]
                
                # Apply overlays based on checkbox selection
                if show_degradation:
                    deg_mask = masks["degradation"]
                    overlay[deg_mask] = (
                        (1 - alpha) * overlay[deg_mask]
                        + alpha * degradation_color
                    )
                
                if show_improvement:
                    imp_mask = masks["improvement"]
                    overlay[imp_mask] = (
                        (1 - alpha) * overlay[imp_mask]
                        + alpha * improvement_color
                    )
                
                st.image(
                    sanitize_image(overlay),
                    use_column_width=True,
                    caption="Detected Vegetation Changes"
                )
                
                # Statistics
                total_pixels = masks["degradation"].size
                
                if show_degradation:
                    deg_pixels = np.sum(masks["degradation"])
                    deg_pct = (deg_pixels / total_pixels) * 100
                else:
                    deg_pixels = 0
                    deg_pct = 0.0
                
                if show_improvement:
                    imp_pixels = np.sum(masks["improvement"])
                    imp_pct = (imp_pixels / total_pixels) * 100
                else:
                    imp_pixels = 0
                    imp_pct = 0.0
                
                net_change_pct = imp_pct - deg_pct
                
                st.markdown(f"""
                <div class="stats-card">
                    <h4 class="stats-title">◆ Change Statistics</h4>
                    <div class="stats-grid">
                        <div>
                            <div class="stat-value" style="color: #9d6dd6;">{deg_pixels:,}</div>
                            <div class="stat-label">Degradation Pixels</div>
                            <div class="stat-label">{deg_pct:.2f}% of area</div>
                        </div>
                        <div>
                            <div class="stat-value" style="color: #b8f34c;">{imp_pixels:,}</div>
                            <div class="stat-label">Improvement Pixels</div>
                            <div class="stat-label">{imp_pct:.2f}% of area</div>
                        </div>
                        <div>
                            <div class="stat-value" style="color: {'#b8f34c' if net_change_pct > 0 else '#9d6dd6'};">{net_change_pct:+.2f}%</div>
                            <div class="stat-label">Net Change</div>
                            <div class="stat-label">{'Positive' if net_change_pct > 0 else 'Negative'}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Legend
                st.markdown("""
                <div class="legend-container">
                    <h4 class="legend-title">◆ Change Type Legend</h4>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(153, 51, 204);"></div>
                        <span><b>Vegetation Loss</b> - Deforestation, land clearing, degradation</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(184, 243, 76);"></div>
                        <span><b>Vegetation Gain</b> - Reforestation, regrowth, greening</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("▸ Adjust sensitivity and click 'Apply Detection' to view results")

# ===============================================================
# TAB 3 – HELP & INFO
# ===============================================================
with tab_help:
    st.markdown("### User Guide & Information")
    
    with st.expander("▸ What is this tool?", expanded=True):
        st.markdown("""
        This platform analyzes satellite imagery to monitor vegetation health and detect land cover changes over time.
        
        **Key Features:**
        - **Vegetation Assessment**: Classify land cover and measure vegetation density from single images
        - **Change Detection**: Identify areas where vegetation has increased or decreased between two time periods
        - **Multi-spectral Analysis**: Uses Red, Green, Blue, NIR, and SWIR bands
        - **Customizable Display**: Adjust brightness, contrast, and overlay opacity for optimal viewing
        """)
    
    with st.expander("▸ Data Requirements"):
        st.markdown("""
        **Required Format:**
        - Multi-band GeoTIFF files (.tif)
        - 5 bands in this order: Red, Green, Blue, NIR (Near-Infrared), SWIR (Short-wave Infrared)
        - Place files in the `data/` directory
        
        **Recommended:**
        - Satellite imagery from Landsat, Sentinel-2, or similar sensors
        - Cloud-free or low-cloud images
        - Same geographic area and spatial resolution for change detection
        - Images from similar seasons for accurate change analysis
        """)
    
    with st.expander("▸ Indices Explained"):
        st.markdown("""
        **NDVI** (Normalized Difference Vegetation Index)
        - Measures vegetation greenness and health
        - Range: -1 to +1 (higher = more vegetation)
        - Primary indicator for vegetation change detection
        
        **MNDWI** (Modified Normalized Difference Water Index)
        - Detects water bodies
        - Helps exclude water from vegetation analysis
        - Range: -1 to +1 (higher = more water)
        
        **NDBI** (Normalized Difference Built-up Index)
        - Identifies urban and built-up areas
        - Used in change detection scoring
        - Range: -1 to +1 (higher = more built-up)
        """)
    
    with st.expander("▸ Understanding Change Detection"):
        st.markdown("""
        **Vegetation Loss (Purple Overlay)**
        - Indicates areas where vegetation health has declined
        - Can represent: deforestation, land clearing, drought stress, fire damage
        - Uses composite scoring based on NDVI decrease and NDBI increase
        
        **Vegetation Gain (Lime Green Overlay)**
        - Indicates areas where vegetation health has improved
        - Can represent: reforestation, natural regrowth, crop maturation, greening
        - Based on positive NDVI change between time periods
        
        **Sensitivity Control**
        - Conservative: Detects only major, obvious changes
        - Moderate: Balanced detection of significant changes
        - Aggressive: Detects subtle changes, may include more noise
        """)
    
    with st.expander("▸ Tips for Best Results"):
        st.markdown("""
        1. **Use images from similar seasons** for accurate change detection
        2. **Start with moderate sensitivity** (0.5) and adjust as needed
        3. **Adjust display settings** to improve visibility of changes on different terrains
        4. **Toggle change types** to focus on specific changes (loss vs gain)
        5. **Verify results** by comparing with RGB imagery
        6. **Consider time span** - longer periods show more dramatic changes
        7. **Use standard thresholds** for consistent vegetation classification
        """)
    
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem;">
        <p>Built with Streamlit • Powered by Remote Sensing Science</p>
        <p style="font-size: 0.9rem;">For technical support or questions, please refer to the documentation</p>
    </div>
    """, unsafe_allow_html=True)