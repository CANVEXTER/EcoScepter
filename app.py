import os
import glob
import streamlit as st
import numpy as np
import re
from datetime import datetime

# ==============================================================================
# 1. CORE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="EcoScepter Analytics",
    initial_sidebar_state="expanded"
)

# Robust Date Parsing (Fixed Version)
DATE_RE_1 = re.compile(r"(\d{2})_(\d{2})_(\d{4})")  # dd_mm_yyyy
DATE_RE_2 = re.compile(r"(\d{4})-(\d{2})-(\d{2})")  # yyyy-mm-dd
DATE_RE_3 = re.compile(r"(\d{4})(\d{2})(\d{2})")    # yyyymmdd

def extract_date(path: str) -> datetime:
    name = os.path.basename(path)
    m1 = DATE_RE_1.search(name)
    if m1: return datetime(int(m1.group(3)), int(m1.group(2)), int(m1.group(1)))
    m2 = DATE_RE_2.search(name)
    if m2: return datetime(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
    m3 = DATE_RE_3.search(name)
    if m3: return datetime(int(m3.group(1)), int(m3.group(2)), int(m3.group(3)))
    raise ValueError(f"ERR_DATE_PARSE: {name}")

# Backend Imports
from matplotlib import colormaps
from scipy.ndimage import binary_opening
from scripts.io import read_bands
from scripts.visualize import stretch, index_to_rgb
from scripts.indices import compute_ndvi, compute_mndwi, compute_ndbi
from scripts.masking import valid_data_mask, water_mask_from_mndwi
from scripts.change import delta, composite_change_score, ndvi_slope
from scripts.thresholds import aggressiveness_to_threshold, apply_threshold
from scripts.assessment import assess_vegetation, auto_tune_assessment_thresholds

DATA_DIR = "data"

# Professional Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Reset */
    .stApp {
        background-color: #09090b; /* Zinc-950 */
        color: #e4e4e7; /* Zinc-200 */
        font-family: 'Inter', sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.025em;
        font-weight: 600;
        color: #fafafa;
    }
    
    .main-title {
        font-size: 1.5rem;
        border-bottom: 1px solid #27272a;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Technical Data Cards */
    .stat-card {
        background-color: #18181b; /* Zinc-900 */
        border: 1px solid #27272a; /* Zinc-800 */
        border-radius: 4px;
        padding: 1.25rem;
        transition: border-color 0.2s;
    }
    .stat-card:hover {
        border-color: #3f3f46;
    }
    
    .stat-label {
        color: #a1a1aa;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f4f4f5;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0c0e;
        border-right: 1px solid #27272a;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        border-bottom: 1px solid #27272a;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background-color: transparent;
        border: none;
        color: #71717a;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #fafafa;
        border-bottom: 2px solid #22c55e; /* Primary Green */
    }

    /* Buttons */
    .stButton > button {
        background-color: #27272a;
        color: #fafafa;
        border: 1px solid #3f3f46;
        border-radius: 4px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #3f3f46;
        border-color: #52525b;
    }
    .stButton > button:active {
        background-color: #22c55e;
        color: #000;
        border-color: #22c55e;
    }

    /* Primary Action Buttons */
    div[data-testid="stVerticalBlock"] > .stButton > button[kind="primary"] {
        background-color: #22c55e;
        color: #052e16;
        border: none;
        font-weight: 600;
    }

    /* Legend / Info Box */
    .legend-box {
        background: #18181b;
        border-left: 3px solid #3f3f46;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def sanitize_image(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def adjust_display(img: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    """Post-processing for display only."""
    img = (img - 0.5) * contrast + 0.5
    img = img * brightness
    return np.clip(img, 0.0, 1.0)

# ==============================================================================
# 3. SIDEBAR & NAVIGATION
# ==============================================================================

with st.sidebar:
    st.markdown("## ECO_SCEPTER")
    st.markdown('<div style="font-size: 0.75rem; color: #71717a; margin-top: -1rem; margin-bottom: 2rem;">REMOTE SENSING ANALYTICS // V2.0</div>', unsafe_allow_html=True)
    
    # Data Loader
    st.markdown("### :: DATA SOURCE")
    tif_files = glob.glob(os.path.join(DATA_DIR, "*.tif"))
    
    try:
        tif_files = sorted(tif_files, key=extract_date)
        file_count = len(tif_files)
        status_color = "#22c55e" if file_count >= 2 else "#ef4444"
        status_text = "READY" if file_count >= 2 else "INSUFFICIENT DATA"
        
        st.markdown(f"""
        <div style="background: #18181b; padding: 0.75rem; border-radius: 4px; border: 1px solid #27272a; margin-bottom: 1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#a1a1aa; font-size:0.8rem;">FILES LOADED</span>
                <span style="font-family:'JetBrains Mono'; font-weight:bold;">{file_count}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem;">
                <span style="color:#a1a1aa; font-size:0.8rem;">SYSTEM STATUS</span>
                <span style="color:{status_color}; font-size:0.7rem; font-weight:bold; border:1px solid {status_color}; padding: 1px 4px; border-radius: 2px;">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except ValueError as e:
        st.error(f"Date Parsing Error: {e}")
        st.stop()

    st.markdown("---")
    st.markdown("### :: GLOBAL SETTINGS")
    
    with st.expander("DISPLAY PARAMETERS", expanded=False):
        st.caption("These settings affect all visualizations.")
        global_gamma = st.slider("GAMMA CORRECTION", 0.5, 2.0, 1.0, 0.1)

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================

# Header
st.markdown('<div class="main-title"><span>Target Area Analysis</span><span style="margin-left:auto; font-size:0.8rem; font-family:\'JetBrains Mono\'; color:#52525b;">SESSION_ID: 0X8291A</span></div>', unsafe_allow_html=True)

if not tif_files:
    st.info("DATA DIRECTORY EMPTY. PLEASE UPLOAD GEOTIFF IMAGERY.")
    st.stop()

# Tabs
tab_analysis, tab_change, tab_docs = st.tabs(["VEGETATION INDICES", "CHANGE DETECTION", "DOCUMENTATION"])

# ------------------------------------------------------------------------------
# TAB 1: SINGLE IMAGE ANALYSIS
# ------------------------------------------------------------------------------
with tab_analysis:
    col_ctrl, col_view = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("#### INPUT SELECTION")
        selected_file = st.selectbox("SATELLITE IMAGE", options=[os.path.basename(f) for f in tif_files], label_visibility="collapsed")
        file_path = os.path.join(DATA_DIR, selected_file)
        
        st.markdown("#### CONFIGURATION")
        process_btn = st.button("EXECUTE ANALYSIS", type="primary", use_container_width=True)
        
        if process_btn:
            with st.spinner("PROCESSING SPECTRAL BANDS..."):
                arr = read_bands(file_path)
                res = assess_vegetation(arr)
                # Auto-tune thresholds
                auto_thr = auto_tune_assessment_thresholds(res["ndvi"], res["mndwi"])
                st.session_state["analysis_result"] = {"arr": arr, **res, **auto_thr}
                st.toast("Processing Complete", icon="⚡")

        if "analysis_result" in st.session_state:
            st.markdown("---")
            st.markdown("#### LAYER CONTROL")
            layer_mode = st.radio(
                "ACTIVE LAYER",
                ["CLASSIFICATION", "TRUE COLOR (RGB)", "NDVI (VEGETATION)", "MNDWI (WATER)", "NDBI (URBAN)"],
                label_visibility="collapsed"
            )
            
            st.markdown("#### IMAGE TUNING")
            b_val = st.slider("BRIGHTNESS", 0.5, 2.0, 1.0, 0.1, key="an_b")
            c_val = st.slider("CONTRAST", 0.5, 2.0, 1.0, 0.1, key="an_c")

    with col_view:
        if "analysis_result" in st.session_state:
            res = st.session_state["analysis_result"]
            
            # Prepare Base RGB
            base_rgb = sanitize_image(stretch(np.stack(res["arr"][:3], axis=-1)))
            base_rgb = adjust_display(base_rgb, b_val, c_val)
            
            # Logic for Layers
            display_img = base_rgb
            caption_text = "TRUE COLOR COMPOSITE"
            
            if layer_mode == "NDVI (VEGETATION)":
                display_img = sanitize_image(index_to_rgb(res["ndvi"], "RdYlGn"))
                caption_text = "NORMALIZED DIFFERENCE VEGETATION INDEX"
                
            elif layer_mode == "MNDWI (WATER)":
                display_img = sanitize_image(index_to_rgb(res["mndwi"], "Blues"))
                caption_text = "MODIFIED NORMALIZED DIFFERENCE WATER INDEX"
                
            elif layer_mode == "NDBI (URBAN)":
                display_img = sanitize_image(index_to_rgb(res["ndbi"], "inferno"))
                caption_text = "NORMALIZED DIFFERENCE BUILT-UP INDEX"
                
            elif layer_mode == "CLASSIFICATION":
                # Apply Classification Logic
                water_mask = res["mndwi"] > res["water_t"]
                clear_mask = (~water_mask) & (res["ndvi"] < res["clear_t"])
                veg_mask = (~water_mask) & (~clear_mask)
                
                # Create Overlay
                overlay = base_rgb.copy()
                overlay[water_mask] = [0.0, 0.3, 0.8]   # Deep Blue
                overlay[clear_mask] = [0.6, 0.5, 0.4]   # Brown/Grey
                
                # Vegetation Gradient
                ndvi_veg = np.where(veg_mask, res["ndvi"], np.nan)
                veg_norm = np.clip((ndvi_veg - res["veg_low"]) / (res["veg_high"] - res["veg_low"] + 1e-6), 0, 1)
                veg_colors = colormaps["YlGn"](veg_norm)[..., :3]
                
                overlay[veg_mask] = veg_colors[veg_mask]
                display_img = sanitize_image(adjust_display(overlay, b_val, c_val))
                caption_text = "LAND COVER CLASSIFICATION"
                
                # Professional Legend
                st.markdown("""
                <div style="display: flex; gap: 1.5rem; background: #18181b; padding: 1rem; border: 1px solid #27272a; margin-bottom: 1rem;">
                    <div style="display:flex; align-items:center; gap:0.5rem;"><div style="width:12px; height:12px; background:#004dcc;"></div><span style="font-size:0.8rem;">WATER</span></div>
                    <div style="display:flex; align-items:center; gap:0.5rem;"><div style="width:12px; height:12px; background:#998066;"></div><span style="font-size:0.8rem;">BARREN/URBAN</span></div>
                    <div style="display:flex; align-items:center; gap:0.5rem;"><div style="width:12px; height:12px; background:#aadd66;"></div><span style="font-size:0.8rem;">VEGETATION (LOW)</span></div>
                    <div style="display:flex; align-items:center; gap:0.5rem;"><div style="width:12px; height:12px; background:#228b22;"></div><span style="font-size:0.8rem;">VEGETATION (HIGH)</span></div>
                </div>
                """, unsafe_allow_html=True)

            st.image(display_img, use_column_width=True, channels="RGB")
            st.caption(f"RENDERING: {caption_text}")
            
        else:
            st.markdown("""
            <div style="height: 400px; display: flex; align-items: center; justify-content: center; border: 1px dashed #3f3f46; border-radius: 4px; color: #52525b;">
                AWAITING INPUT EXECUTION
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: CHANGE DETECTION
# ------------------------------------------------------------------------------
with tab_change:
    if len(tif_files) < 2:
        st.warning("INSUFFICIENT DATA FOR TEMPORAL ANALYSIS")
        st.stop()

    # Timeline Control
    st.markdown("#### TEMPORAL RANGE SELECTION")
    dates = [extract_date(f) for f in tif_files]
    labels = [d.strftime("%Y-%m-%d") for d in dates]
    
    start_i, end_i = st.select_slider(
        "RANGE",
        options=list(range(len(tif_files))),
        value=(0, len(tif_files) - 1),
        format_func=lambda i: labels[i],
        label_visibility="collapsed"
    )
    
    # Action Bar
    col_act_1, col_act_2 = st.columns([3, 1])
    with col_act_1:
        st.caption(f"ANALYSIS VECTOR: {labels[start_i]}  ➔  {labels[end_i]}")
    with col_act_2:
        calc_btn = st.button("COMPUTE DELTA", type="primary", use_container_width=True)

    if calc_btn:
        with st.spinner("CALCULATING SPECTRAL DIFFERENCES..."):
            subset = tif_files[start_i:end_i+1]
            sub_dates = [extract_date(f) for f in subset]
            arrays = [read_bands(f) for f in subset]
            
            # --- MATH CORE ---
            ndvi_stack = np.stack([compute_ndvi(a) for a in arrays])
            # Masking based on latest image
            mask_ref = valid_data_mask(arrays[-1]) & (~water_mask_from_mndwi(compute_mndwi(arrays[-1])))
            ndvi_stack = np.where(mask_ref, ndvi_stack, np.nan)
            
            if len(subset) == 2:
                # Delta
                d_ndvi = delta(ndvi_stack[0], ndvi_stack[1])
                d_ndbi = delta(compute_ndbi(arrays[0]), compute_ndbi(arrays[1]))
                score = composite_change_score(d_ndvi, d_ndbi, ndvi_stack[0])
                raw_change = d_ndvi
            else:
                # Slope
                years = np.array([d.year + d.timetuple().tm_yday/365.25 for d in sub_dates])
                slope = ndvi_slope(years, ndvi_stack)
                score = -slope # Negative slope = degradation (positive score)
                raw_change = slope
                
            st.session_state["cd_result"] = {
                "score": score,
                "change_val": raw_change,
                "img_latest": arrays[-1],
                "mask": mask_ref
            }
            st.toast("Delta Calculation Complete", icon="⚡")

    # Reactive Dashboard
    if "cd_result" in st.session_state:
        cd = st.session_state["cd_result"]
        st.markdown("---")
        
        c_dash, c_map = st.columns([1, 2.5])
        
        with c_dash:
            st.markdown("#### THRESHOLD CONTROL")
            st.info("Adjust sensitivity to filter noise.")
            sens = st.slider("SENSITIVITY", 0.0, 1.0, 0.5, 0.05)
            
            st.markdown("#### VISUAL FILTERS")
            show_loss = st.checkbox("DEGRADATION (LOSS)", value=True)
            show_gain = st.checkbox("RECOVERY (GAIN)", value=True)
            
            st.markdown("#### OVERLAY OPACITY")
            alpha = st.slider("ALPHA", 0.0, 1.0, 0.6, 0.1, label_visibility="collapsed")

            # --- LIVE CALCULATION ---
            # Degradation
            deg_thr = aggressiveness_to_threshold(cd["score"], sens)
            deg_mask = apply_threshold(cd["score"], deg_thr)
            deg_mask = binary_opening(deg_mask, structure=np.ones((3,3)))
            
            # Improvement
            imp_thr = np.nanpercentile(cd["change_val"], 85 - (sens * 15))
            imp_thr = max(imp_thr, 0.05)
            imp_mask = cd["change_val"] > imp_thr
            imp_mask = binary_opening(imp_mask, structure=np.ones((3,3)))
            
            # Stats Logic
            px_total = np.sum(cd["mask"])
            px_loss = np.sum(deg_mask & cd["mask"]) if show_loss else 0
            px_gain = np.sum(imp_mask & cd["mask"]) if show_gain else 0
            
            p_loss = (px_loss / px_total * 100) if px_total > 0 else 0
            p_gain = (px_gain / px_total * 100) if px_total > 0 else 0
            
            st.markdown("#### STATISTICS")
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">DETECTED LOSS</div>
                <div class="stat-value" style="color: #c084fc;">{p_loss:.2f}%</div>
                <div style="height: 4px; background: #3f3f46; margin-top: 5px; border-radius: 2px;">
                    <div style="height: 100%; width: {min(p_loss, 100)}%; background: #c084fc; border-radius: 2px;"></div>
                </div>
            </div>
            <div class="stat-card" style="margin-top: 1rem;">
                <div class="stat-label">DETECTED GAIN</div>
                <div class="stat-value" style="color: #bef264;">{p_gain:.2f}%</div>
                <div style="height: 4px; background: #3f3f46; margin-top: 5px; border-radius: 2px;">
                    <div style="height: 100%; width: {min(p_gain, 100)}%; background: #bef264; border-radius: 2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_map:
            # Render
            rgb_base = sanitize_image(stretch(np.stack(cd["img_latest"][:3], axis=-1)))
            # Gamma/Brightness from sidebar
            rgb_base = adjust_display(rgb_base, brightness=1.1, contrast=1.1)
            
            out_img = rgb_base.copy()
            
            # Colors: Purple (Loss), Lime (Gain)
            c_loss = np.array([0.75, 0.5, 0.98]) # Violet
            c_gain = np.array([0.74, 0.94, 0.39]) # Lime
            
            if show_loss:
                out_img[deg_mask] = (1-alpha)*out_img[deg_mask] + alpha*c_loss
            if show_gain:
                out_img[imp_mask] = (1-alpha)*out_img[imp_mask] + alpha*c_gain
                
            st.image(sanitize_image(out_img), use_column_width=True, caption="CHANGE DETECTION RENDER [COMPOSITE]")

# ------------------------------------------------------------------------------
# TAB 3: DOCUMENTATION
# ------------------------------------------------------------------------------
with tab_docs:
    st.markdown("""
    ### SYSTEM DOCUMENTATION
    
    #### 1. SPECTRAL INDICES
    * **NDVI:** Normalized Difference Vegetation Index. Primary indicator of biomass.
    * **MNDWI:** Modified Normalized Difference Water Index. Used for automated water masking.
    * **NDBI:** Normalized Difference Built-up Index. Used to differentiate urban expansion from soil.

    #### 2. CHANGE DETECTION ALGORITHMS
    * **Dual-Image Mode:** Uses direct spectral delta subtraction (ΔNDVI) weighted by baseline vegetation density.
    * **Multi-Temporal Mode:** Uses linear regression (Theil-Sen estimator) to derive the slope of the vegetation trend line over time.
    
    #### 3. THRESHOLDING LOGIC
    * **Adaptive Sensitivity:** The sensitivity slider interpolates between statistical outliers (percentiles) and physical magnitude limits to separate signal from sensor noise.
    """)