import os
import glob
import streamlit as st
import numpy as np
import re
from datetime import datetime

# --- 1. ROBUST DATE PARSING ---
# Handles dd_mm_yyyy, yyyy-mm-dd, and yyyymmdd
DATE_RE_1 = re.compile(r"(\d{2})_(\d{2})_(\d{4})")  # dd_mm_yyyy
DATE_RE_2 = re.compile(r"(\d{4})-(\d{2})-(\d{2})")  # yyyy-mm-dd
DATE_RE_3 = re.compile(r"(\d{4})(\d{2})(\d{2})")    # yyyymmdd

def extract_date(path: str) -> datetime:
    name = os.path.basename(path)
    
    m1 = DATE_RE_1.search(name)
    if m1:
        dd, mm, yyyy = m1.groups()
        return datetime(int(yyyy), int(mm), int(dd))
        
    m2 = DATE_RE_2.search(name)
    if m2:
        yyyy, mm, dd = m2.groups()
        return datetime(int(yyyy), int(mm), int(dd))

    m3 = DATE_RE_3.search(name)
    if m3:
        yyyy, mm, dd = m3.groups()
        return datetime(int(yyyy), int(mm), int(dd))
    
    raise ValueError(f"No valid date found in filename: {name}")

from matplotlib import colormaps
from scipy.ndimage import binary_opening

from scripts.io import read_bands
from scripts.visualize import stretch, index_to_rgb
from scripts.indices import compute_ndvi, compute_mndwi, compute_ndbi
from scripts.masking import valid_data_mask, water_mask_from_mndwi, apply_masks
from scripts.change import (
    delta,
    composite_change_score,
    ndvi_slope,
)
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

# Page config
st.set_page_config(
    layout="wide",
    page_title="Vegetation & Change Detection Analysis",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #4ade80; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #9ca3af; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 1.5rem; border-radius: 10px; border: 1px solid #334155; color: #fafafa; margin-bottom: 1rem; }
    .info-box { background-color: #1e3a28; border-left: 4px solid #4ade80; padding: 1rem; border-radius: 4px; margin: 1rem 0; color: #e5e7eb; }
    .stButton>button { width: 100%; background-color: #22c55e; color: #0e1117; border-radius: 8px; font-weight: 600; border: none; transition: all 0.3s; }
    .stButton>button:hover { background-color: #16a34a; color: #0e1117; }
    div[data-testid="stExpander"] { background-color: #1e293b; border-radius: 8px; border: 1px solid #334155; }
    .legend-container { background: #1e293b; padding: 1.5rem; border-radius: 8px; border: 1px solid #334155; margin-top: 1rem; }
    .legend-item { display: flex; align-items: center; margin: 0.5rem 0; font-size: 0.95rem; color: #e5e7eb; }
    .legend-color { width: 30px; height: 20px; border-radius: 4px; margin-right: 10px; border: 1px solid #475569; }
    .stats-card { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 1.5rem; border-radius: 10px; border: 1px solid #4ade80; color: #fafafa; margin-top: 1rem; }
    .stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
    .stat-value { font-size: 2rem; font-weight: bold; }
    .stat-label { opacity: 0.8; color: #9ca3af; }
    div[data-testid="stAlert"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

def sanitize_image(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def adjust_image_display(img: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    img = (img - 0.5) * contrast + 0.5
    img = img * brightness
    return np.clip(img, 0.0, 1.0)

# Header
st.markdown('<p class="main-header">Vegetation & Change Detection Analysis Platform</p>', unsafe_allow_html=True)

# Check for data files
tif_files = glob.glob(os.path.join(DATA_DIR, "*.tif"))
try:
    tif_files = sorted(tif_files, key=extract_date)
except ValueError as e:
    st.error(f"Error parsing dates: {e}")
    st.stop()

if not tif_files:
    st.error("⚠ No GeoTIFF files found in the 'data/' directory.")
    st.stop()

# Main tabs
tab_assess, tab_change, tab_help = st.tabs(["▸ Vegetation Assessment", "▸ Change Detection", "▸ Help & Info"])

# ===============================================================
# TAB 1 – VEGETATION ASSESSMENT (Unchanged Logic)
# ===============================================================
with tab_assess:
    st.markdown("### Analyze Vegetation Health")
    col1, col2 = st.columns([2, 1])
    with col1:
        tif_name = st.selectbox("◆ Select image", options=[os.path.basename(f) for f in tif_files])
        tif_path = os.path.join(DATA_DIR, tif_name)
    with col2:
        st.write(""); st.write("")
        analyze_btn = st.button("▸ Run Analysis", type="primary")
    
    if analyze_btn:
        with st.spinner("Processing..."):
            arr = read_bands(tif_path)
            result = assess_vegetation(arr)
            auto = auto_tune_assessment_thresholds(result["ndvi"], result["mndwi"])
            st.session_state["assess"] = {"arr": arr, **result, **auto}
            st.success("✓ Analysis complete!")

    if "assess" in st.session_state:
        a = st.session_state["assess"]
        st.divider()
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            st.markdown("#### ◆ Visualization")
            view_mode = st.radio("Display Layer", ["Vegetation Assessment", "RGB (True Color)", "NDVI (Vegetation)", "MNDWI (Water)", "NDBI (Built-up)"])
            with st.expander("▸ Display Adjustments"):
                brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.05)
                contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.05)
            
            # Use reference logic
            water_t = REFERENCE_THRESHOLDS["water_t"]
            clear_t = REFERENCE_THRESHOLDS["clear_t"]
            veg_low = REFERENCE_THRESHOLDS["veg_low"]
            veg_high = REFERENCE_THRESHOLDS["veg_high"]
        
        with col_right:
            water = a["mndwi"] > water_t
            clear = (~water) & (a["ndvi"] < clear_t)
            veg_mask = (~water) & (~clear)
            rgb = sanitize_image(stretch(np.stack(a["arr"][:3], axis=-1)))
            
            if view_mode == "RGB (True Color)":
                st.image(adjust_image_display(rgb, brightness, contrast), use_column_width=True)
            elif view_mode == "Vegetation Assessment":
                overlay = rgb.copy()
                overlay[water] = [0.0, 0.3, 0.8]
                overlay[clear] = [0.55, 0.4, 0.25]
                ndvi = a["ndvi"].copy()
                ndvi[~veg_mask] = np.nan
                veg_norm = np.clip((ndvi - veg_low) / (veg_high - veg_low + 1e-6), 0, 1)
                veg_rgb = colormaps["YlGn"](veg_norm)[..., :3]
                overlay[veg_mask] = veg_rgb[veg_mask]
                st.image(sanitize_image(adjust_image_display(overlay, brightness, contrast)), use_column_width=True)

# ===============================================================
# TAB 2 – CHANGE DETECTION (Fixed & Reactive)
# ===============================================================
with tab_change:
    st.markdown("### Detect Land Cover Changes")
    
    if len(tif_files) < 2:
        st.warning("⚠ Need at least 2 images.")
        st.stop()
    
    dates = [extract_date(f) for f in tif_files]
    labels = [d.strftime("%Y-%m-%d") for d in dates]

    start_idx, end_idx = st.select_slider(
        "Select time range",
        options=list(range(len(tif_files))),
        value=(0, len(tif_files) - 1),
        format_func=lambda i: labels[i]
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # This button is ONLY for the heavy lifting (loading files, math)
        compute_btn = st.button("▸ Compute Change Analysis", type="primary", use_container_width=True)
    
    if compute_btn:
        with st.spinner("Analyzing changes..."):
            selected_files = tif_files[start_idx:end_idx + 1]
            selected_dates = [extract_date(f) for f in selected_files]
            arrays = [read_bands(f) for f in selected_files]
            
            # Compute Stack
            ndvi_stack = np.stack([compute_ndvi(a) for a in arrays])
            
            # Safe Mask (based on last image)
            mndwi_last = compute_mndwi(arrays[-1])
            mask = valid_data_mask(arrays[-1]) & (~water_mask_from_mndwi(mndwi_last))
            ndvi_stack = np.where(mask, ndvi_stack, np.nan)

            if len(selected_files) == 2:
                # Delta Mode
                d_ndvi = delta(ndvi_stack[0], ndvi_stack[1]) # b - a
                ndbi1 = compute_ndbi(arrays[0])
                ndbi2 = compute_ndbi(arrays[1])
                d_ndbi = delta(ndbi1, ndbi2) # b - a
                
                # Use updated composite score (Physical logic)
                score = composite_change_score(d_ndvi, d_ndbi, ndvi_baseline=ndvi_stack[0])
                ndvi_change = d_ndvi
            else:
                # Trend Mode
                years = np.array([d.year + d.timetuple().tm_yday / 365.25 for d in selected_dates])
                slope = ndvi_slope(years, ndvi_stack)
                # Negative slope = Degradation (Score > 0)
                score = -slope 
                ndvi_change = slope

            st.session_state["change_data"] = {
                "score": score,
                "ndvi_change": ndvi_change,
                "arr2": arrays[-1],
                "mask": mask
            }
            st.success("✓ Analysis complete!")

    # --- REACTIVE UI SECTION ---
    if "change_data" in st.session_state:
        st.divider()
        cd = st.session_state["change_data"]
        
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            st.markdown("#### ◆ Visualization")
            
            # 1. Display Controls (Immediate update)
            show_degradation = st.checkbox("Show Degradation (Purple)", value=True)
            show_improvement = st.checkbox("Show Improvement (Green)", value=True)
            
            brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.05, key="ch_bright")
            contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.05, key="ch_cont")
            overlay_alpha = st.slider("Overlay Opacity", 0.0, 1.0, 0.7, 0.05, key="ch_alpha")
            
            st.divider()
            
            # 2. Threshold Controls (Immediate calculation)
            aggressiveness = st.slider(
                "Threshold Sensitivity", 0.0, 1.0, 0.5, 0.05,
                help="Left: Only major changes | Right: Detects subtle changes"
            )
            
            # --- REAL-TIME CALCULATION ---
            # Calculate thresholds on the fly
            thr = aggressiveness_to_threshold(cd["score"], aggressiveness)
            deg_mask = apply_threshold(cd["score"], thr)
            deg_mask = binary_opening(deg_mask, structure=np.ones((3, 3)))
            
            # Improvement (Simple percentile of positive NDVI change)
            # Higher aggressiveness = lower percentile required
            imp_thr = np.nanpercentile(cd["ndvi_change"], 85 - (aggressiveness * 15))
            # Ensure we are only looking at positive change > noise
            imp_thr = max(imp_thr, 0.05) 
            imp_mask = cd["ndvi_change"] > imp_thr
            imp_mask = binary_opening(imp_mask, structure=np.ones((3, 3)))

        with col_right:
            # --- RENDER ---
            rgb = sanitize_image(stretch(np.stack(cd["arr2"][:3], axis=-1)))
            rgb = adjust_image_display(rgb, brightness, contrast)
            overlay = rgb.copy()
            
            # Define Colors
            c_deg = np.array([0.6, 0.2, 0.8], dtype=np.float32) 
            c_imp = np.array([0.7, 0.95, 0.3], dtype=np.float32)
            
            # Apply Masks
            if show_degradation:
                overlay[deg_mask] = (1 - overlay_alpha) * overlay[deg_mask] + overlay_alpha * c_deg
            
            if show_improvement:
                overlay[imp_mask] = (1 - overlay_alpha) * overlay[imp_mask] + overlay_alpha * c_imp
            
            st.image(sanitize_image(overlay), use_column_width=True, caption="Change Detection Result")
            
            # --- STATS ---
            total = np.sum(cd["mask"]) # Only count valid land pixels
            if total > 0:
                deg_px = np.sum(deg_mask & cd["mask"]) if show_degradation else 0
                imp_px = np.sum(imp_mask & cd["mask"]) if show_improvement else 0
                
                deg_pct = (deg_px / total) * 100
                imp_pct = (imp_px / total) * 100
                net = imp_pct - deg_pct
                
                st.markdown(f"""
                <div class="stats-card"><div class="stats-grid">
                    <div><div class="stat-value" style="color:#9d6dd6">{deg_px:,}</div><div class="stat-label">Loss ({deg_pct:.1f}%)</div></div>
                    <div><div class="stat-value" style="color:#b8f34c">{imp_px:,}</div><div class="stat-label">Gain ({imp_pct:.1f}%)</div></div>
                    <div><div class="stat-value">{net:+.1f}%</div><div class="stat-label">Net Change</div></div>
                </div></div>
                """, unsafe_allow_html=True)