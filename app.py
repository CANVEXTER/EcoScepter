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
    "water_t": 0.0,   # MNDWI > 0.0 is Water
    "clear_t": 0.60,  # NDVI < 0.60 is Barren/Urban
    "veg_low": 0.60,  # Vegetation starts at 0.60
    "veg_high": 0.90  # Dense vegetation saturates at 0.90
}

# Professional Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;500;700&display=swap');
    .stApp { background-color: #09090b; color: #e4e4e7; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.025em; font-weight: 600; color: #fafafa; }
    .main-title { font-size: 1.5rem; border-bottom: 1px solid #27272a; padding-bottom: 1rem; margin-bottom: 2rem; display: flex; align-items: center; gap: 0.75rem; }
    .stat-card { background-color: #18181b; border: 1px solid #27272a; border-radius: 4px; padding: 1.25rem; transition: border-color 0.2s; }
    .stat-label { color: #a1a1aa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #f4f4f5; }
    section[data-testid="stSidebar"] { background-color: #0c0c0e; border-right: 1px solid #27272a; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; background-color: transparent; border-bottom: 1px solid #27272a; padding-bottom: 0; }
    .stTabs [data-baseweb="tab"] { height: 3rem; background-color: transparent; border: none; color: #71717a; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #fafafa; border-bottom: 2px solid #22c55e; }
    .stButton > button { background-color: #27272a; color: #fafafa; border: 1px solid #3f3f46; border-radius: 4px; font-weight: 500; transition: all 0.2s; }
    .stButton > button:hover { background-color: #3f3f46; border-color: #52525b; }
    div[data-testid="stVerticalBlock"] > .stButton > button[kind="primary"] { background-color: #22c55e; color: #052e16; border: none; font-weight: 600; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    
    /* Legend Styling Fix for Width */
    .legend-box {
        display: flex; 
        gap: 1.5rem; 
        background: #18181b; 
        padding: 1rem; 
        border: 1px solid #27272a; 
        margin-bottom: 1rem;
        width: 100%;
        box-sizing: border-box;
        flex-wrap: wrap;
    }
    .legend-item { display:flex; align-items:center; gap:0.5rem; }
    .legend-swatch { width:12px; height:12px; }
    .legend-label { font-size:0.8rem; }
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
# 3. SIDEBAR & NAVIGATION
# ==============================================================================

with st.sidebar:
    st.markdown("## EcoScepter")
    st.markdown('<div style="font-size: 0.75rem; color: #71717a; margin-top: -1rem; margin-bottom: 2rem;">REMOTE SENSING ANALYTICS // V2.4</div>', unsafe_allow_html=True)
    
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

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================

st.markdown('<div class="main-title"><span>Target Area Analysis</span><span style="margin-left:auto; font-size:0.8rem; font-family:\'JetBrains Mono\'; color:#52525b;">SESSION_ID: 0X8291A</span></div>', unsafe_allow_html=True)

if not tif_files:
    st.info("DATA DIRECTORY EMPTY. PLEASE UPLOAD GEOTIFF IMAGERY.")
    st.stop()

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
                # Auto-tune thresholds (Otsu Method)
                auto_thr = auto_tune_assessment_thresholds(res["ndvi"], res["mndwi"])
                st.session_state["analysis_result"] = {"arr": arr, **res, "auto_stats": auto_thr}
                st.toast("Processing Complete", icon="⚡")

        if "analysis_result" in st.session_state:
            st.markdown("---")
            st.markdown("#### LAYER CONTROL")
            # TERM FIX: NDBI Renamed
            layer_mode = st.radio(
                "ACTIVE LAYER",
                ["CLASSIFICATION", "TRUE COLOR (RGB)", "NDVI (VEGETATION)", "MNDWI (WATER)", "NDBI (CLEARINGS/BUILDUPS)"],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            st.markdown("#### CLASSIFICATION RULES")
            ar = st.session_state["analysis_result"]
            
            tuning_mode = st.radio("THRESHOLD LOGIC", ["EXPLICIT (STANDARD)", "AUTO-ADAPTIVE (OTSU/STATS)", "MANUAL"], label_visibility="collapsed")
            
            if tuning_mode == "EXPLICIT (STANDARD)":
                st.caption("Using fixed physical constants. Best for consistency across images.")
                t_water = REFERENCE_THRESHOLDS["water_t"]
                t_clear = REFERENCE_THRESHOLDS["clear_t"]
                t_vlow  = REFERENCE_THRESHOLDS["veg_low"]
                t_vhigh = REFERENCE_THRESHOLDS["veg_high"]
                
            elif tuning_mode == "AUTO-ADAPTIVE (OTSU/STATS)":
                st.caption("Derived from image statistics (Otsu's method). Good for maximizing contrast.")
                stats = ar["auto_stats"]
                t_water = stats["water_t"]
                t_clear = stats["clear_t"]
                t_vlow  = stats["veg_low"]
                t_vhigh = stats["veg_high"]
                
            else: # MANUAL
                st.caption("Override classification boundaries:")
                t_water = st.slider("WATER (MNDWI)", -1.0, 1.0, REFERENCE_THRESHOLDS["water_t"], 0.05)
                t_clear = st.slider("CLEAR LAND (NDVI)", -1.0, 1.0, REFERENCE_THRESHOLDS["clear_t"], 0.05)
                t_vlow, t_vhigh = st.slider("VEGETATION RANGE", -1.0, 1.0, (REFERENCE_THRESHOLDS["veg_low"], REFERENCE_THRESHOLDS["veg_high"]), 0.05)
                
            st.markdown("---")
            st.markdown("#### IMAGE SETTINGS")
            with st.expander("DYNAMIC RANGE (CLIP)", expanded=True):
                p_min = st.slider("SHADOW CLIP (%)", 0, 10, 2, 1)
                p_max = st.slider("HIGHLIGHT CLIP (%)", 90, 100, 98, 1)
                
            b_val = st.slider("BRIGHTNESS", 0.1, 3.0, 1.0, 0.1, key="an_b")
            c_val = st.slider("CONTRAST", 0.1, 3.0, 1.0, 0.1, key="an_c")

    with col_view:
        if "analysis_result" in st.session_state:
            res = st.session_state["analysis_result"]
            
            # Dynamic Stretch applied to RGB
            base_rgb = sanitize_image(stretch(np.stack(res["arr"][:3], axis=-1), pmin=p_min, pmax=p_max))
            base_rgb = adjust_display(base_rgb, b_val, c_val)
            
            display_img = base_rgb
            caption_text = "TRUE COLOR COMPOSITE"
            
            if layer_mode == "NDVI (VEGETATION)":
                display_img = sanitize_image(index_to_rgb(res["ndvi"], "RdYlGn", pmin=p_min, pmax=p_max))
                display_img = adjust_display(display_img, b_val, c_val)
                caption_text = "NORMALIZED DIFFERENCE VEGETATION INDEX"
                
            elif layer_mode == "MNDWI (WATER)":
                display_img = sanitize_image(index_to_rgb(res["mndwi"], "Blues", pmin=p_min, pmax=p_max))
                display_img = adjust_display(display_img, b_val, c_val)
                caption_text = "MODIFIED NORMALIZED DIFFERENCE WATER INDEX"
                
            elif layer_mode == "NDBI (CLEARINGS/BUILDUPS)":
                display_img = sanitize_image(index_to_rgb(res["ndbi"], "inferno", pmin=p_min, pmax=p_max))
                display_img = adjust_display(display_img, b_val, c_val)
                caption_text = "NORMALIZED DIFFERENCE BUILT-UP INDEX"
                
            elif layer_mode == "CLASSIFICATION":
                #Explicit Logic
                water_mask = res["mndwi"] > t_water
                clear_mask = (~water_mask) & (res["ndvi"] < t_clear)
                veg_mask = (~water_mask) & (~clear_mask)
                
                overlay = base_rgb.copy()
                overlay[water_mask] = [0.0, 0.3, 0.8]   # Deep Blue
                overlay[clear_mask] = [0.6, 0.5, 0.4]   # Brown/Grey
                
                # Vegetation Gradient
                ndvi_veg = np.where(veg_mask, res["ndvi"], np.nan)
                veg_norm = np.clip((ndvi_veg - t_vlow) / (t_vhigh - t_vlow + 1e-6), 0, 1)
                veg_colors = colormaps["YlGn"](veg_norm)[..., :3]
                overlay[veg_mask] = veg_colors[veg_mask]
                
                display_img = sanitize_image(overlay)
                caption_text = f"LAND COVER CLASSIFICATION (Water > {t_water:.2f}, Veg > {t_vlow:.2f})"
                
                # LAYOUT FIX: Updated CSS class and structure
                st.markdown("""
                <div class="legend-box">
                    <div class="legend-item"><div class="legend-swatch" style="background:#004dcc;"></div><span class="legend-label">WATER</span></div>
                    <div class="legend-item"><div class="legend-swatch" style="background:#998066;"></div><span class="legend-label">CLEARING/URBAN</span></div>
                    <div class="legend-item"><div class="legend-swatch" style="background:#aadd66;"></div><span class="legend-label">VEG (LOW)</span></div>
                    <div class="legend-item"><div class="legend-swatch" style="background:#228b22;"></div><span class="legend-label">VEG (HIGH)</span></div>
                </div>
                """, unsafe_allow_html=True)

            st.image(display_img, use_column_width=True, channels="RGB")
            st.caption(f"RENDERING: {caption_text} | RGB RANGE: {p_min}%-{p_max}%")
            
            # TERM FIX: NDBI Note
            if layer_mode == "NDBI (CLEARINGS/BUILDUPS)":
                st.info("NOTE: NDBI highlights non-vegetated areas (soil, concrete). Raw colormaps can be ambiguous due to spectral similarity; rely on the 'CLASSIFICATION' layer for definitive land cover separation.")
            
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
            
            ndvi_stack = np.stack([compute_ndvi(a) for a in arrays])
            mask_ref = valid_data_mask(arrays[-1]) & (~water_mask_from_mndwi(compute_mndwi(arrays[-1])))
            ndvi_stack = np.where(mask_ref, ndvi_stack, np.nan)
            
            if len(subset) == 2:
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
                "img_latest": arrays[-1],
                "mask": mask_ref
            }
            st.toast("Delta Calculation Complete", icon="⚡")

    if "cd_result" in st.session_state:
        cd = st.session_state["cd_result"]
        st.markdown("---")
        
        # LAYOUT FIX: Consistent [1, 3] ratio with Tab 1
        c_dash, c_map = st.columns([1, 3])
        
        with c_dash:
            st.markdown("#### THRESHOLD CONTROL")
            # Using Rosin's Method via the backend script
            sens = st.slider("SENSITIVITY (ROSIN'S METHOD)", 0.0, 1.0, 0.5, 0.05)
            
            st.markdown("#### VISUAL FILTERS")
            show_loss = st.checkbox("DEGRADATION (LOSS)", value=True)
            show_gain = st.checkbox("RECOVERY (GAIN)", value=True)
            
            st.markdown("#### OVERLAY OPACITY")
            alpha = st.slider("ALPHA", 0.0, 1.0, 0.6, 0.1, label_visibility="collapsed")
            
            st.markdown("#### MAP SETTINGS")
            with st.expander("DYNAMIC RANGE (CLIP)", expanded=True):
                cp_min = st.slider("SHADOW CLIP (%)", 0, 10, 2, 1, key="cd_pmin")
                cp_max = st.slider("HIGHLIGHT CLIP (%)", 90, 100, 98, 1, key="cd_pmax")

            map_b = st.slider("BRIGHTNESS", 0.1, 3.0, 1.0, 0.1, key="cd_b")
            map_c = st.slider("CONTRAST", 0.1, 3.0, 1.0, 0.1, key="cd_c")

            deg_thr = aggressiveness_to_threshold(cd["score"], sens)
            deg_mask = apply_threshold(cd["score"], deg_thr)
            deg_mask = binary_opening(deg_mask, structure=np.ones((3,3)))
            
            imp_thr = np.nanpercentile(cd["change_val"], 85 - (sens * 15))
            imp_thr = max(imp_thr, 0.05)
            imp_mask = cd["change_val"] > imp_thr
            imp_mask = binary_opening(imp_mask, structure=np.ones((3,3)))
            
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
            # Stretch with CLIP params
            rgb_base = sanitize_image(stretch(np.stack(cd["img_latest"][:3], axis=-1), pmin=cp_min, pmax=cp_max))
            rgb_base = adjust_display(rgb_base, brightness=map_b, contrast=map_c)
            
            out_img = rgb_base.copy()
            c_loss = np.array([0.75, 0.5, 0.98])
            c_gain = np.array([0.74, 0.94, 0.39])
            
            if show_loss:
                out_img[deg_mask] = (1-alpha)*out_img[deg_mask] + alpha*c_loss
            if show_gain:
                out_img[imp_mask] = (1-alpha)*out_img[imp_mask] + alpha*c_gain
                
            st.image(sanitize_image(out_img), use_column_width=True, caption=f"CHANGE DETECTION | RGB RANGE: {cp_min}%-{cp_max}%")

with tab_docs:
    st.markdown("""
    ### SYSTEM DOCUMENTATION
    #### SPECTRAL INDICES
    * **NDVI:** Normalized Difference Vegetation Index. Primary indicator of biomass.
    * **MNDWI:** Modified Normalized Difference Water Index. Used for automated water masking.
    * **NDBI (Clearings/Buildups):** Normalized Difference Built-up Index. Highlights clearings, bare soil, and built-up areas. **Note:** Raw NDBI colormaps can be ambiguous due to spectral similarities between different land cover types. It is best used as an input for the definitive 'Classification' layer rather than a standalone visual.
    
    #### CHANGE ALGORITHMS
    * **Dual-Image Mode:** Uses direct spectral delta subtraction (ΔNDVI).
    * **Multi-Temporal Mode:** Uses linear regression (Theil-Sen estimator).
    
    #### THRESHOLDING LOGIC
    * **Vegetation Assessment:** Offers Explicit (physical constants) or Auto-Adaptive (Otsu's Method) thresholding for robust classification.
    * **Change Detection:** Uses **Rosin's Corner Method** ("The Elbow") to statistically determine the optimal separation between noise and significant change.
    """)