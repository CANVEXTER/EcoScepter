import os
import glob
import streamlit as st
import numpy as np
import re
from datetime import datetime

# ==============================================================================
# 1. CORE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="EcoScepter | Remote Sensing Analytics",
    initial_sidebar_state="expanded"
)

# Robust Date Parsing
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

# PHYSICAL CONSTANTS (EXPLICIT MODE)
REFERENCE_THRESHOLDS = {
    "water_t": 0.0,   
    "clear_t": 0.60,  
    "veg_low": 0.60,  
    "veg_high": 0.90  
}

# Advanced CSS for Professional UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Base Theme */
    .stApp { background-color: #09090b; color: #e4e4e7; font-family: 'Inter', sans-serif; }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; font-weight: 600; color: #fafafa; }
    
    /* Custom Header Bar */
    .header-bar {
        border-bottom: 1px solid #27272a;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .app-brand { font-size: 1.25rem; font-weight: 700; color: #fafafa; letter-spacing: -0.03em; }
    .session-tag { font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #52525b; background: #18181b; padding: 4px 8px; border-radius: 4px; border: 1px solid #27272a; }

    /* Custom Metric Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 6px;
        padding: 1rem;
        position: relative;
        overflow: hidden;
    }
    .metric-box::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: #3f3f46;
    }
    .metric-box.loss::before { background: #c084fc; }
    .metric-box.gain::before { background: #bef264; }
    
    .metric-label { font-size: 0.7rem; text-transform: uppercase; color: #a1a1aa; letter-spacing: 0.05em; margin-bottom: 0.25rem; }
    .metric-value { font-family: 'JetBrains Mono'; font-size: 1.5rem; font-weight: 700; color: #f4f4f5; }

    /* Legend Box */
    .legend-box {
        display: flex; 
        gap: 1rem; 
        background: #18181b; 
        padding: 0.75rem; 
        border: 1px solid #27272a; 
        border-radius: 6px;
        margin-bottom: 1rem;
        width: 100%;
        flex-wrap: wrap;
        font-size: 0.8rem;
    }
    .l-item { display:flex; align-items:center; gap:0.5rem; }
    .l-swatch { width:10px; height:10px; border-radius:2px; }

    /* Streamlit Components Overrides */
    section[data-testid="stSidebar"] { background-color: #0c0c0e; border-right: 1px solid #27272a; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; border-bottom: 1px solid #27272a; }
    .stTabs [data-baseweb="tab"] { height: 3rem; color: #71717a; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #fafafa; border-bottom: 2px solid #22c55e; }
    
    div[data-testid="stExpander"] { background: #121214; border: 1px solid #27272a; border-radius: 6px; }
    
    /* Button Styling */
    .stButton > button { background: #27272a; color: #fafafa; border: 1px solid #3f3f46; border-radius: 6px; transition: all 0.2s; }
    .stButton > button:hover { border-color: #71717a; background: #3f3f46; }
    div[data-testid="stVerticalBlock"] > .stButton > button[kind="primary"] { background: #22c55e; color: #052e16; border: none; font-weight: 600; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def sanitize_image(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def adjust_display(img: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    img = (img - 0.5) * contrast + 0.5
    img = img * brightness
    return np.clip(img, 0.0, 1.0)

# ==============================================================================
# 3. SIDEBAR: STATUS ONLY
# ==============================================================================

with st.sidebar:
    st.markdown("## EcoScepter")
    st.markdown('<div style="font-size: 0.75rem; color: #71717a; margin-top: -1rem; margin-bottom: 2rem;">V3.2 // ANALYTICS SUITE</div>', unsafe_allow_html=True)
    
    raw_files = glob.glob(os.path.join(DATA_DIR, "*.tif"))
    
    if not raw_files:
        st.error("NO DATA FOUND.")
        st.stop()
        
    try:
        raw_files = sorted(raw_files, key=extract_date)
        
        st.markdown("### :: SYSTEM STATUS")
        st.markdown(f"""
        <div style="background: #18181b; padding: 0.8rem; border-radius: 6px; border: 1px solid #27272a;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#a1a1aa;">
                <span>TOTAL SCENES</span><span>{len(raw_files)}</span>
            </div>
            <div style="font-size:0.7rem; font-weight:700; color:#22c55e; letter-spacing:0.05em; margin-top:0.5rem;">
                SYSTEM ONLINE
            </div>
        </div>
        """, unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"Date Parsing Error: {e}")
        st.stop()

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================

st.markdown(f"""
<div class="header-bar">
    <div class="app-brand">Target Area Analysis</div>
    <div class="session-tag">ID: 0X8291A</div>
</div>
""", unsafe_allow_html=True)

tab_analysis, tab_change, tab_docs = st.tabs(["VEGETATION INDICES", "CHANGE DETECTION", "DOCUMENTATION"])

# ------------------------------------------------------------------------------
# TAB 1: SINGLE IMAGE ANALYSIS
# ------------------------------------------------------------------------------
with tab_analysis:
    col_ctrl, col_view = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("#### INPUT SELECTION")
        selected_file = st.selectbox("SATELLITE IMAGE", options=[os.path.basename(f) for f in raw_files], label_visibility="collapsed")
        file_path = os.path.join(DATA_DIR, selected_file)
        
        st.markdown("#### EXECUTION")
        process_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
        
        if process_btn:
            with st.spinner("PROCESSING..."):
                arr = read_bands(file_path)
                res = assess_vegetation(arr)
                auto_thr = auto_tune_assessment_thresholds(res["ndvi"], res["mndwi"])
                st.session_state["analysis_result"] = {"arr": arr, **res, "auto_stats": auto_thr}
                st.toast("Processing Complete")

        if "analysis_result" in st.session_state:
            st.divider()
            
            st.markdown("**VISUALIZATION**")
            layer_mode = st.radio(
                "Layer Mode",
                ["CLASSIFICATION", "TRUE COLOR (RGB)", "NDVI (VEGETATION)", "MNDWI (WATER)", "NDBI (CLEARINGS/BUILDUPS)"],
                label_visibility="collapsed"
            )
            
            # CONTROL DECK B: CLASSIFICATION
            with st.expander(":: CLASSIFICATION LOGIC", expanded=False):
                ar = st.session_state["analysis_result"]
                tuning_mode = st.radio("Method", ["EXPLICIT (PHYSICAL)", "ADAPTIVE (OTSU)", "MANUAL"], label_visibility="collapsed")
                
                if tuning_mode == "EXPLICIT (PHYSICAL)":
                    st.caption("Standard physics-based thresholds.")
                    t_water = REFERENCE_THRESHOLDS["water_t"]
                    t_clear = REFERENCE_THRESHOLDS["clear_t"]
                    t_vlow  = REFERENCE_THRESHOLDS["veg_low"]
                    t_vhigh = REFERENCE_THRESHOLDS["veg_high"]
                elif tuning_mode == "ADAPTIVE (OTSU)":
                    st.caption("Statistical separation based on histogram valleys.")
                    stats = ar["auto_stats"]
                    t_water, t_clear, t_vlow, t_vhigh = stats["water_t"], stats["clear_t"], stats["veg_low"], stats["veg_high"]
                else:
                    t_water = st.slider("Water Max", -1.0, 1.0, 0.0, 0.05, key="an_t_water")
                    t_clear = st.slider("Barren Max", -1.0, 1.0, 0.6, 0.05, key="an_t_clear")
                    t_vlow, t_vhigh = st.slider("Veg Range", -1.0, 1.0, (0.6, 0.9), 0.05, key="an_t_veg")

            # CONTROL DECK C: OPTICS
            with st.expander(":: IMAGE AESTHETICS", expanded=False):
                st.caption("Dynamic Range Clipping")
                p_min, p_max = st.slider("Histogram Clip %", 0, 100, (2, 98), 1, key="an_clip")
                st.caption("Post-Processing")
                b_val = st.slider("Brightness", 0.1, 3.0, 1.0, 0.1, key="an_bright")
                c_val = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1, key="an_cont")

    with col_view:
        if "analysis_result" in st.session_state:
            res = st.session_state["analysis_result"]
            
            # Base Processing
            base_rgb = sanitize_image(stretch(np.stack(res["arr"][:3], axis=-1), pmin=p_min, pmax=p_max))
            base_rgb = adjust_display(base_rgb, b_val, c_val)
            
            display_img = base_rgb
            caption = "TRUE COLOR COMPOSITE"
            
            # Layer Logic
            if layer_mode == "NDVI (VEGETATION)":
                display_img = sanitize_image(index_to_rgb(res["ndvi"], "RdYlGn", pmin=p_min, pmax=p_max))
                caption = "NDVI (VEGETATION HEALTH)"
            elif layer_mode == "MNDWI (WATER)":
                display_img = sanitize_image(index_to_rgb(res["mndwi"], "Blues", pmin=p_min, pmax=p_max))
                caption = "MNDWI (WATER CONTENT)"
            elif layer_mode == "NDBI (CLEARINGS/BUILDUPS)":
                display_img = sanitize_image(index_to_rgb(res["ndbi"], "inferno", pmin=p_min, pmax=p_max))
                caption = "NDBI (URBAN/BAREN)"
            elif layer_mode == "CLASSIFICATION":
                water_mask = res["mndwi"] > t_water
                clear_mask = (~water_mask) & (res["ndvi"] < t_clear)
                veg_mask = (~water_mask) & (~clear_mask)
                
                overlay = base_rgb.copy()
                overlay[water_mask] = [0.0, 0.3, 0.8]
                overlay[clear_mask] = [0.6, 0.5, 0.4]
                ndvi_veg = np.where(veg_mask, res["ndvi"], np.nan)
                veg_norm = np.clip((ndvi_veg - t_vlow) / (t_vhigh - t_vlow + 1e-6), 0, 1)
                overlay[veg_mask] = colormaps["YlGn"](veg_norm)[..., :3][veg_mask]
                display_img = sanitize_image(overlay)
                caption = "LAND COVER CLASSIFICATION"
                
                # Legend
                st.markdown("""
                <div class="legend-box">
                    <div class="l-item"><div class="l-swatch" style="background:#004dcc;"></div><span>WATER</span></div>
                    <div class="l-item"><div class="l-swatch" style="background:#998066;"></div><span>CLEARING/URBAN</span></div>
                    <div class="l-item"><div class="l-swatch" style="background:#aadd66;"></div><span>VEG (LOW)</span></div>
                    <div class="l-item"><div class="l-swatch" style="background:#228b22;"></div><span>VEG (HIGH)</span></div>
                </div>""", unsafe_allow_html=True)
            
            # Main Render (UPDATED: use_container_width)
            st.image(display_img, use_container_width=True, channels="RGB")
            st.caption(f"VIEW: {caption} | CLIP: {p_min}-{p_max}%")
            
            if layer_mode == "NDBI (CLEARINGS/BUILDUPS)":
                st.info("ℹ NOTE: NDBI highlights non-vegetated areas. Use Classification for precise separation.")

# ------------------------------------------------------------------------------
# TAB 2: CHANGE DETECTION
# ------------------------------------------------------------------------------
with tab_change:
    # 1. RANGE FILTER
    all_dates = [extract_date(f) for f in raw_files]
    all_labels = [d.strftime("%Y-%m-%d") for d in all_dates]
    
    st.markdown("#### TEMPORAL RANGE")
    
    start_i, end_i = st.select_slider(
        "TIMELINE WINDOW",
        options=list(range(len(raw_files))),
        value=(0, len(raw_files) - 1),
        format_func=lambda i: all_labels[i],
        label_visibility="collapsed"
    )

    # 2. SPECIFIC INCLUSION (KEY FIXED)
    range_indices = list(range(start_i, end_i + 1))
    range_labels = [all_labels[i] for i in range_indices]
    
    with st.expander(":: ACTIVE OBSERVATIONS (EXCLUDE DATA)", expanded=True):
        selected_labels = st.multiselect(
            "Uncheck bad observations (clouds/artifacts) to exclude them from calculation:",
            options=range_labels,
            default=range_labels,
            label_visibility="collapsed",
            key="cd_exclude_multiselect" 
        )
    
    final_indices = [i for i in range_indices if all_labels[i] in selected_labels]
    subset_files = [raw_files[i] for i in final_indices]
    subset_labels = [all_labels[i] for i in final_indices]
    
    # 3. TIMELINE VISUALIZER
    if not subset_labels:
        st.error("No dates selected.")
        st.stop()

    timeline_str = "  >>  ".join(subset_labels) if len(subset_labels) <= 4 else \
                   f"{subset_labels[0]}  >>  ... ({len(subset_labels)-2} frames) ...  >>  {subset_labels[-1]}"
    
    st.markdown(f"""
    <div style="background:#121214; padding:0.5rem 1rem; border-radius:4px; border:1px solid #27272a; font-family:'JetBrains Mono'; font-size:0.8rem; color:#a1a1aa; text-align:center; margin-bottom:1rem;">
        {timeline_str}
    </div>
    """, unsafe_allow_html=True)

    if len(subset_files) < 2:
        st.warning("PLEASE SELECT AT LEAST 2 DATES FOR ANALYSIS.")
        st.stop()
    
    # 4. ACTION
    col_act_1, col_act_2 = st.columns([3, 1])
    with col_act_1:
        mode = "DELTA (A vs B)" if len(subset_files) == 2 else f"TREND SLOPE ({len(subset_files)} Scenes)"
        st.caption(f"MODE: {mode}")
    with col_act_2:
        calc_btn = st.button("COMPUTE TREND", type="primary", use_container_width=True)

    if calc_btn:
        with st.spinner("CALCULATING SPECTRAL DIFFERENCES..."):
            sub_dates = [extract_date(f) for f in subset_files]
            arrays = [read_bands(f) for f in subset_files]
            
            # Math
            ndvi_stack = np.stack([compute_ndvi(a) for a in arrays])
            ref_idx = -1
            mask_ref = valid_data_mask(arrays[ref_idx]) & (~water_mask_from_mndwi(compute_mndwi(arrays[ref_idx])))
            ndvi_stack = np.where(mask_ref, ndvi_stack, np.nan)
            
            if len(subset_files) == 2:
                d_ndvi = delta(ndvi_stack[0], ndvi_stack[1])
                d_ndbi = delta(compute_ndbi(arrays[0]), compute_ndbi(arrays[1]))
                score = composite_change_score(d_ndvi, d_ndbi, ndvi_stack[0])
                raw_change = d_ndvi
            else:
                years = np.array([d.year + d.timetuple().tm_yday/365.25 for d in sub_dates])
                slope = ndvi_slope(years, ndvi_stack)
                score = -slope 
                raw_change = slope
                
            st.session_state["cd_result"] = {
                "score": score,
                "change_val": raw_change,
                "img_latest": arrays[ref_idx],
                "mask": mask_ref
            }
            st.toast("Calculation Complete")

    # --- RESULTS DASHBOARD ---
    if "cd_result" in st.session_state:
        st.divider()
        cd = st.session_state["cd_result"]
        
        c_dash, c_map = st.columns([1, 3])
        
        with c_dash:
            st.markdown("##### SETTINGS")
            
            with st.expander(":: SENSITIVITY", expanded=True):
                sens = st.slider("Threshold Stringency", 0.0, 1.0, 0.5, 0.05, help="Uses Rosin's Corner Method", key="cd_sens")
                st.caption("Filters")
                show_loss = st.checkbox("Show Loss", value=True, key="cd_loss")
                show_gain = st.checkbox("Show Gain", value=True, key="cd_gain")
            
            with st.expander(":: MAP STYLE", expanded=False):
                alpha = st.slider("Opacity", 0.0, 1.0, 0.6, 0.1, key="cd_alpha")
                cp_min, cp_max = st.slider("Clip %", 0, 100, (2, 98), 1, key="cd_clip")
                map_b = st.slider("Brightness", 0.1, 3.0, 1.0, 0.1, key="cd_bright")
                map_c = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1, key="cd_cont")

            # Live Math
            deg_thr = aggressiveness_to_threshold(cd["score"], sens)
            deg_mask = apply_threshold(cd["score"], deg_thr)
            deg_mask = binary_opening(deg_mask, structure=np.ones((3,3)))
            
            imp_thr = np.nanpercentile(cd["change_val"], 85 - (sens * 15))
            imp_thr = max(imp_thr, 0.05)
            imp_mask = cd["change_val"] > imp_thr
            imp_mask = binary_opening(imp_mask, structure=np.ones((3,3)))
            
            # Stats
            px_total = np.sum(cd["mask"])
            px_loss = np.sum(deg_mask & cd["mask"]) if show_loss else 0
            px_gain = np.sum(imp_mask & cd["mask"]) if show_gain else 0
            
            p_loss = (px_loss / px_total * 100) if px_total > 0 else 0
            p_gain = (px_gain / px_total * 100) if px_total > 0 else 0
            
        with c_map:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-box loss">
                    <div class="metric-label">DETECTED LOSS</div>
                    <div class="metric-value" style="color:#c084fc;">{p_loss:.2f}%</div>
                </div>
                <div class="metric-box gain">
                    <div class="metric-label">DETECTED GAIN</div>
                    <div class="metric-value" style="color:#bef264;">{p_gain:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">NET CHANGE</div>
                    <div class="metric-value" style="color:#fafafa;">{p_gain - p_loss:+.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            rgb_base = sanitize_image(stretch(np.stack(cd["img_latest"][:3], axis=-1), pmin=cp_min, pmax=cp_max))
            rgb_base = adjust_display(rgb_base, brightness=map_b, contrast=map_c)
            
            out_img = rgb_base.copy()
            c_loss = np.array([0.75, 0.5, 0.98])
            c_gain = np.array([0.74, 0.94, 0.39])
            
            if show_loss: out_img[deg_mask] = (1-alpha)*out_img[deg_mask] + alpha*c_loss
            if show_gain: out_img[imp_mask] = (1-alpha)*out_img[imp_mask] + alpha*c_gain
            
            # UPDATED: use_container_width
            st.image(sanitize_image(out_img), use_container_width=True, caption=f"CHANGE MAP | SENSITIVITY: {sens}")

# ------------------------------------------------------------------------------
# TAB 3: DOCS
# ------------------------------------------------------------------------------
with tab_docs:
    st.markdown("""
    ### SYSTEM DOCUMENTATION
    
    #### 1. DATA HYGIENE
    * **Active Observations:** Use the 'Exclude Data' dropdown in the Timeline settings to remove specific dates (e.g., cloudy scenes) from the trend calculation without reloading the dataset.
    
    #### 2. CLASSIFICATION LOGIC
    * **Explicit (Physical):** Uses fixed values (Water < 0.0, Veg > 0.6). Best for consistency.
    * **Adaptive (Otsu):** Calculates optimal thresholds based on histogram valleys. Best for maximizing contrast in difficult lighting.
    
    #### 3. CHANGE ALGORITHMS
    * **Delta:** Direct subtraction (B - A).
    * **Trend (Theil-Sen):** Robust slope calculation for 3+ points.
    """)