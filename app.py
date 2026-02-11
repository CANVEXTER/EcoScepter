import os
import glob
import streamlit as st
import numpy as np
import pandas as pd
import re
import tempfile
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Backend logic imports
from scripts.change import compute_extended_stats, delta, composite_change_score, ndvi_slope
from scripts.reporting import generate_change_report
from scripts.io import read_bands, read_metadata
from scripts.visualize import stretch, index_to_rgb, plot_with_grid 
from scripts.indices import compute_ndvi, compute_mndwi, compute_ndbi
from scripts.masking import valid_data_mask, water_mask_from_mndwi
from scripts.thresholds import aggressiveness_to_threshold, apply_threshold
from scripts.assessment import assess_vegetation, auto_tune_assessment_thresholds
from matplotlib import colormaps
from scipy.ndimage import binary_opening

# ==============================================================================
# 1. CORE CONFIGURATION & THEME LOADER
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="EcoScepter | Remote Sensing Analytics",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if os.path.exists("style.css"):
    load_css("style.css")

# Robust Date Parsing Logic
DATE_RE_1 = re.compile(r"(\d{2})_(\d{2})_(\d{4})")
DATE_RE_2 = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
DATE_RE_3 = re.compile(r"(\d{4})(\d{2})(\d{2})")

def extract_date(path: str) -> datetime:
    name = os.path.basename(path)
    m1 = DATE_RE_1.search(name)
    if m1: return datetime(int(m1.group(3)), int(m1.group(2)), int(m1.group(1)))
    m2 = DATE_RE_2.search(name)
    if m2: return datetime(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
    m3 = DATE_RE_3.search(name)
    if m3: return datetime(int(m3.group(1)), int(m3.group(2)), int(m3.group(3)))
    raise ValueError(f"ERR_DATE_PARSE: {name}")

# Constants
REFERENCE_THRESHOLDS = {
    "water_t": 0.0,   
    "clear_t": 0.60,  
    "veg_low": 0.60,  
    "veg_high": 0.90  
}

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
# 3. SIDEBAR: DATA INGESTION
# ==============================================================================
with st.sidebar:
    st.markdown("## EcoScepter")
    st.markdown('<div style="font-size: 0.75rem; color: #71717a; margin-top: -1rem; margin-bottom: 2rem;">V4.3 // PRO ANALYTICS</div>', unsafe_allow_html=True)
    
    st.markdown("### :material/cloud_upload: DATA IMPORT")
    uploaded_files = st.file_uploader("Drop Satellite Images Here", type=["tif", "tiff"], accept_multiple_files=True)
    
    if not uploaded_files:
        st.info("ℹ️ Waiting for data...")
        st.stop()
        
    if "temp_data_dir" not in st.session_state:
        st.session_state["temp_data_dir"] = tempfile.mkdtemp()
    
    DATA_DIR = st.session_state["temp_data_dir"]
    current_upload_names = {f.name for f in uploaded_files}
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    
    existing_files = set(os.listdir(DATA_DIR))
    for filename in existing_files:
        if filename not in current_upload_names:
            os.remove(os.path.join(DATA_DIR, filename))

    raw_files = glob.glob(os.path.join(DATA_DIR, "*.tif"))
    if not raw_files:
        st.error("Error processing files."); st.stop()
        
    try:
        raw_files = sorted(raw_files, key=extract_date)
        st.markdown("### :material/analytics: STATUS")
        st.markdown(f"""
        <div style="background: #18181b; padding: 1rem; border-radius: 8px; border: 1px solid #27272a;">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span style="font-size:0.8rem; color:#e4e4e7;">LOADED SCENES</span>
                <span style="font-family:'JetBrains Mono'; font-size:1.2rem; color:#fafafa; font-weight:700;">{len(raw_files)}</span>
            </div>
            <div style="margin-top:0.75rem; display:flex; gap:0.5rem; align-items:center;">
                <div style="width:8px; height:8px; background:#22c55e; border-radius:50%; box-shadow: 0 0 8px #22c55e40;"></div>
                <span style="font-size:0.7rem; font-weight:600; color:#22c55e; letter-spacing:0.05em;">SYSTEM READY</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Date Parsing Error: {e}"); st.stop()

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================
st.markdown(f"""<div class="header-bar"><div class="app-brand">Target Area Analysis</div><div class="session-tag">ID: 0X8291A</div></div>""", unsafe_allow_html=True)
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
        
        # Reactive Logic
        if "an_cache_key" not in st.session_state or st.session_state["an_cache_key"] != selected_file:
            with st.status(f"Processing {selected_file}...", expanded=True) as status:
                st.write("Reading raster bands...")
                arr = read_bands(file_path)
                meta_run = read_metadata(file_path)
                
                st.write("Computing indices & masks...")
                res = assess_vegetation(arr)
                
                st.write("Optimizing threshold logic...")
                auto_thr = auto_tune_assessment_thresholds(res["ndvi"], res["mndwi"])
                
                st.session_state["analysis_result"] = {"arr": arr, "bounds": meta_run.get("bounds_wgs84"), **res, "auto_stats": auto_thr}
                st.session_state["an_cache_key"] = selected_file
                status.update(label="Analysis Complete", state="complete", expanded=False)
        
        if "analysis_result" in st.session_state:
            st.divider()
            st.markdown("#### VISUALIZATION")
            layer_mode = st.radio("Layer Mode", ["CLASSIFICATION", "TRUE COLOR (RGB)", "NDVI (VEGETATION)", "MNDWI (WATER)", "NDBI (CLEARINGS)"], label_visibility="collapsed")
            
            # --- UPDATE: ALWAYS VISIBLE (Removed 'if layer_mode == ...' check) ---
            with st.expander(":: CLASSIFICATION LOGIC", expanded=True):
                ar = st.session_state["analysis_result"]
                tuning_mode = st.radio("Method", ["EXPLICIT (PHYSICAL)", "ADAPTIVE (OTSU)", "MANUAL"], label_visibility="collapsed")
                if tuning_mode == "EXPLICIT (PHYSICAL)":
                    t_water, t_clear, t_vlow, t_vhigh = REFERENCE_THRESHOLDS.values()
                elif tuning_mode == "ADAPTIVE (OTSU)":
                    stats = ar["auto_stats"]
                    t_water, t_clear, t_vlow, t_vhigh = stats["water_t"], stats["clear_t"], stats["veg_low"], stats["veg_high"]
                else:
                    t_water = st.slider("Water", -1.0, 1.0, 0.0, 0.05, key="an_t_water")
                    t_clear = st.slider("Barren", -1.0, 1.0, 0.6, 0.05, key="an_t_clear")
                    t_vlow, t_vhigh = st.slider("Veg", -1.0, 1.0, (0.6, 0.9), 0.05, key="an_t_veg")

            with st.expander(":: IMAGE AESTHETICS", expanded=False):
                p_min, p_max = st.slider("Histogram Clip %", 0, 100, (2, 98), 1, key="an_clip")
                b_val = st.slider("Brightness", 0.1, 3.0, 1.0, 0.1, key="an_bright")
                c_val = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1, key="an_cont")

    with col_view:
        if "analysis_result" in st.session_state:
            # FIX 1: CLEAR ALL STATE
            plt.close('all')
            plt.clf()
            
            res = st.session_state["analysis_result"]
            base_rgb = sanitize_image(stretch(np.stack(res["arr"][:3], axis=-1), pmin=p_min, pmax=p_max))
            base_rgb = adjust_display(base_rgb, b_val, c_val)
            display_img, caption = base_rgb, "TRUE COLOR COMPOSITE"
            
            legend_slot = st.empty()
            
            # Layer Logic
            if layer_mode == "NDVI (VEGETATION)":
                display_img = sanitize_image(index_to_rgb(res["ndvi"], "RdYlGn", pmin=p_min, pmax=p_max))
                caption = "NDVI (VEGETATION HEALTH)"
                legend_slot.empty()
            elif layer_mode == "MNDWI (WATER)":
                display_img = sanitize_image(index_to_rgb(res["mndwi"], "Blues", pmin=p_min, pmax=p_max))
                caption = "MNDWI (WATER CONTENT)"
                legend_slot.empty()
            elif layer_mode == "NDBI (CLEARINGS)":
                display_img = sanitize_image(index_to_rgb(res["ndbi"], "inferno", pmin=p_min, pmax=p_max))
                caption = "NDBI (URBAN/BAREN)"
                legend_slot.empty()
            elif layer_mode == "CLASSIFICATION":
                # Fallback safety (though now usually redundant since controls are always active)
                if 't_water' not in locals(): 
                    t_water, t_clear, t_vlow, t_vhigh = REFERENCE_THRESHOLDS.values()

                water_mask = res["mndwi"] > t_water
                clear_mask = (~water_mask) & (res["ndvi"] < t_clear)
                veg_mask = (~water_mask) & (~clear_mask)
                overlay = base_rgb.copy()
                overlay[water_mask] = [0.0, 0.3, 0.8]
                overlay[clear_mask] = [0.6, 0.5, 0.4]
                veg_norm = np.clip((np.where(veg_mask, res["ndvi"], np.nan) - t_vlow) / (t_vhigh - t_vlow + 1e-6), 0, 1)
                overlay[veg_mask] = colormaps["YlGn"](veg_norm)[..., :3][veg_mask]
                display_img = sanitize_image(overlay)
                caption = "LAND COVER CLASSIFICATION"
                
                # FIX 2: EXPLICIT LEGEND
                legend_slot.markdown("""<div class="legend-box"><div class="l-item"><div class="l-swatch" style="background:#004dcc;"></div><span>WATER</span></div><div class="l-item"><div class="l-swatch" style="background:#998066;"></div><span>CLEARING/URBAN</span></div><div class="l-item"><div class="l-swatch" style="background:#aadd66;"></div><span>VEG (LOW)</span></div><div class="l-item"><div class="l-swatch" style="background:#228b22;"></div><span>VEG (HIGH)</span></div></div>""", unsafe_allow_html=True)
            else:
                 legend_slot.empty()

            # FIX 3: DOUBLE CONTAINER SWAP
            slot_plot = st.empty()
            slot_img = st.empty()
            
            if res.get("bounds"):
                slot_img.empty()
                fig = plot_with_grid(display_img, res["bounds"], title=f"GEO-REFERENCED: {caption}")
                slot_plot.pyplot(fig, use_container_width=True)
                plt.close(fig) 
            else:
                slot_plot.empty()
                slot_img.image(display_img, use_container_width=True, channels="RGB", caption=caption)

# ------------------------------------------------------------------------------
# TAB 2: CHANGE DETECTION
# ------------------------------------------------------------------------------
with tab_change:
    all_dates = [extract_date(f) for f in raw_files]
    all_labels = [d.strftime("%Y-%m-%d") for d in all_dates]
    
    st.markdown("#### TEMPORAL RANGE")
    start_i, end_i = st.select_slider("TIMELINE WINDOW", options=list(range(len(raw_files))), value=(0, len(raw_files) - 1), format_func=lambda i: all_labels[i], label_visibility="collapsed")
    subset_indices = list(range(start_i, end_i + 1))
    subset_files = [raw_files[i] for i in subset_indices]
    subset_labels = [all_labels[i] for i in subset_indices]
    
    timeline_str = " >> ".join(subset_labels) if len(subset_labels) <= 4 else f"{subset_labels[0]} >> ... ({len(subset_labels)-2} frames) ... >> {subset_labels[-1]}"
    st.markdown(f'<div style="background:#121214; padding:0.5rem 1rem; border-radius:4px; border:1px solid #27272a; font-family:\'JetBrains Mono\'; font-size:0.8rem; color:#a1a1aa; text-align:center; margin-bottom:1rem;">{timeline_str}</div>', unsafe_allow_html=True)
    
    if len(subset_files) < 2: st.warning("SELECT > 1 DATE"); st.stop()
    
    current_range_key = (start_i, end_i)
    if "cd_cache_key" not in st.session_state or st.session_state["cd_cache_key"] != current_range_key:
        with st.status("Computing Temporal Trends...", expanded=True) as status:
            st.write("Optimizing Memory Footprint...")
            sub_dates = [extract_date(f) for f in subset_files]
            
            DS = 2 
            def load_optimized(fpath):
                raw = read_bands(fpath)
                if raw.ndim == 3 and raw.shape[1] > 500:
                    return raw[:, ::DS, ::DS]
                return raw

            st.write("Streaming Raster Processing...")
            if len(subset_files) == 2:
                img0 = load_optimized(subset_files[0])
                img1 = load_optimized(subset_files[1])
                
                ndvi0, ndvi1 = compute_ndvi(img0), compute_ndvi(img1)
                ndbi0, ndbi1 = compute_ndbi(img0), compute_ndbi(img1)
                mndwi_ref = compute_mndwi(img1)
                
                mask_ref = valid_data_mask(img1) & (~water_mask_from_mndwi(mndwi_ref))
                d_ndvi, d_ndbi = delta(ndvi0, ndvi1), delta(ndbi0, ndbi1)
                
                score, raw_change = composite_change_score(d_ndvi, d_ndbi, ndvi0)
                
                img_start, img_end = img0, img1
                ndvi_stack_masked = np.stack([np.where(mask_ref, ndvi0, np.nan), np.where(mask_ref, ndvi1, np.nan)])
                
            else:
                ndvi_list = []
                img_start, img_end = None, None
                for idx, f in enumerate(subset_files):
                    current_img = load_optimized(f)
                    if idx == 0: img_start = current_img
                    elif idx == len(subset_files) - 1: img_end = current_img
                    ndvi = compute_ndvi(current_img)
                    ndvi_list.append(ndvi)
                    if idx != 0 and idx != len(subset_files) - 1:
                        del current_img
                
                ndvi_stack = np.stack(ndvi_list)
                mndwi_ref = compute_mndwi(img_end)
                mask_ref = valid_data_mask(img_end) & (~water_mask_from_mndwi(mndwi_ref))
                ndvi_stack_masked = np.where(mask_ref, ndvi_stack, np.nan)
                years = np.array([d.year + d.timetuple().tm_yday/365.25 for d in sub_dates])
                slope = ndvi_slope(years, ndvi_stack_masked)
                score, raw_change = -slope, slope

            st.session_state["cd_result"] = {
                "score": score, "change_val": raw_change, 
                "img_start": img_start, "img_end": img_end,
                "ndvi_stack": ndvi_stack_masked, "dates": sub_dates,
                "mask": mask_ref, "bounds": read_metadata(subset_files[-1]).get("bounds_wgs84")
            }
            st.session_state["cd_cache_key"] = current_range_key
            status.update(label="Trend Analysis Ready", state="complete", expanded=False)

    if "cd_result" in st.session_state:
        cd = st.session_state["cd_result"]
        
        # 1. VISUAL VALIDATION
        with st.expander(":: VISUAL VALIDATION (BEFORE VS AFTER)", expanded=True):
            col_v1, col_v2 = st.columns(2)
            img_a = adjust_display(sanitize_image(stretch(np.stack(cd["img_start"][:3], axis=-1), pmin=2, pmax=98)))
            img_b = adjust_display(sanitize_image(stretch(np.stack(cd["img_end"][:3], axis=-1), pmin=2, pmax=98)))
            with col_v1:
                st.caption(f"INITIAL STATE: {subset_files[0].split(os.sep)[-1]}")
                st.image(img_a, use_container_width=True)
            with col_v2:
                st.caption(f"FINAL STATE: {subset_files[-1].split(os.sep)[-1]}")
                st.image(img_b, use_container_width=True)

        st.divider()

        # Stats
        sens = st.session_state.get("cd_sens", 0.5) 
        deg_mask = binary_opening(apply_threshold(cd["score"], aggressiveness_to_threshold(cd["score"], sens)), structure=np.ones((3,3)))
        imp_mask = binary_opening(cd["change_val"] > max(np.nanpercentile(cd["change_val"], 85 - (sens * 15)), 0.05), structure=np.ones((3,3)))
        px_total = np.sum(cd["mask"])
        p_loss = (np.sum(deg_mask & cd["mask"]) / px_total * 100) if px_total > 0 else 0
        p_gain = (np.sum(imp_mask & cd["mask"]) / px_total * 100) if px_total > 0 else 0
        ext_stats = compute_extended_stats(cd["change_val"], cd["mask"], deg_mask, imp_mask)

        # 2. METRICS
        st.markdown(f'''
        <div class="metric-container" style="margin-bottom: 2rem;">
            <div class="metric-box loss">
                <div class="metric-label">VEGETATION REGRESSION</div>
                <div class="metric-value" style="color:#c084fc;">{p_loss:.2f}%</div>
                <div class="metric-label" style="margin-top:0.2rem;">INTENSITY: {ext_stats["avg_loss_val"]:.3f}</div>
            </div>
            <div class="metric-box gain">
                <div class="metric-label">VEGETATION ACCRETION</div>
                <div class="metric-value" style="color:#bef264;">{p_gain:.2f}%</div>
                <div class="metric-label" style="margin-top:0.2rem;">INTENSITY: {ext_stats["avg_gain_val"]:.3f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">NET SPECTRAL DELTA</div>
                <div class="metric-value" style="color:#fafafa;">{p_gain - p_loss:+.2f}%</div>
                <div class="metric-label" style="margin-top:0.2rem;">COMPOSITE SCORE</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # 3. SPLIT VIEW
        c_dash, c_map = st.columns([1, 3], gap="large")
        
        with c_dash:
            st.markdown("##### SETTINGS")
            with st.expander(":: SENSITIVITY", expanded=True):
                sens = st.slider("Threshold Stringency", 0.0, 1.0, 0.5, 0.05, key="cd_sens")
                show_loss = st.checkbox("Show Regression", True)
                show_gain = st.checkbox("Show Accretion", True)
            
            with st.expander(":: MAP AESTHETICS", expanded=True):
                alpha = st.slider("Overlay Opacity", 0.0, 1.0, 0.6, 0.1, key="cd_alpha")
                cp_min, cp_max = st.slider("Base Clip %", 0, 100, (2, 98), 1, key="cd_clip")
                map_b = st.slider("Base Brightness", 0.1, 3.0, 0.30, 0.05, key="cd_bright")
                map_c = st.slider("Base Contrast", 0.1, 3.0, 1.0, 0.1, key="cd_cont")
            
        with c_map:
            rgb_base = adjust_display(sanitize_image(stretch(np.stack(cd["img_end"][:3], axis=-1), pmin=cp_min, pmax=cp_max)), brightness=map_b, contrast=map_c)
            out_img = rgb_base.copy()
            c_loss = np.array([0.75, 0.5, 0.98]) 
            c_gain = np.array([0.74, 0.94, 0.39]) 
            
            if show_loss: out_img[deg_mask] = (1-alpha)*out_img[deg_mask] + alpha*c_loss
            if show_gain: out_img[imp_mask] = (1-alpha)*out_img[imp_mask] + alpha*c_gain
            
            cd_slot_plot = st.empty()
            cd_slot_img = st.empty()
            
            if cd.get("bounds"):
                cd_slot_img.empty()
                plt.close('all')
                fig = plot_with_grid(sanitize_image(out_img), cd["bounds"], title=f"CHANGE MAP | SENSITIVITY: {sens}")
                cd_slot_plot.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                cd_slot_plot.empty()
                cd_slot_img.image(sanitize_image(out_img), use_container_width=True, caption=f"CHANGE MAP | SENSITIVITY: {sens}")

        # 4. CHART
        if np.sum(deg_mask) > 0:
            st.divider()
            st.markdown("##### SPECTRAL PROFILING")
            stack = cd["ndvi_stack"]
            deg_profile = np.nanmean(stack[:, deg_mask], axis=1) 
            stable_mask = cd["mask"] & (~deg_mask)
            stable_profile = np.nanmean(stack[:, stable_mask], axis=1)

            df_trend = pd.DataFrame({
                "Date": cd["dates"],
                "Regression Zone": deg_profile,
                "Stable Zone": stable_profile
            })
            
            fig = px.line(df_trend, x="Date", y=["Regression Zone", "Stable Zone"], 
                          color_discrete_map={"Regression Zone": "#ef4444", "Stable Zone": "#22c55e"})
            fig.update_layout(plot_bgcolor="#18181b", paper_bgcolor="#18181b", font_color="#a1a1aa", 
                              height=350, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant change detected to profile.")

# ------------------------------------------------------------------------------
# TAB 3: DOCS
# ------------------------------------------------------------------------------
with tab_docs:
    st.markdown("""### SYSTEM DOCUMENTATION\n#### 1. ANALYSIS WORKFLOW\n* **Automated Computation:** The engine now proactively calculates spectral trends upon data ingestion.\n* **Smart Caching:** Visual adjustments (Opacity, Contrast) are decoupled from the analytical backend for zero-latency updates.\n#### 2. TERMINOLOGY\n* **Vegetation Regression:** Area where biomass density has statistically significantly decreased.\n* **Vegetation Accretion:** Area where biomass density has increased.\n* **Net Spectral Delta:** The composite balance of ecological gain versus loss.""")