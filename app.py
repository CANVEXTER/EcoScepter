import os
import glob
import streamlit as st
import numpy as np
import pandas as pd
import re
import tempfile
import matplotlib.pyplot as plt
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

# Load styles from external file
if os.path.exists("style.css"):
    load_css("style.css")

# Robust Date Parsing Logic
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
    st.markdown('<div style="font-size: 0.75rem; color: #71717a; margin-top: -1rem; margin-bottom: 2rem;">V3.4 // ANALYTICS SUITE</div>', unsafe_allow_html=True)
    
    st.markdown("### :material/cloud_upload: DATA IMPORT")
    
    uploaded_files = st.file_uploader(
        "Drop Satellite Images Here", 
        type=["tif", "tiff"], 
        accept_multiple_files=True,
        help="Select all files in your folder (Ctrl+A) and drag them here."
    )
    
    if not uploaded_files:
        st.info("ℹ️ Waiting for data...")
        st.caption("Drag & drop .tif files to begin analysis.")
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
        st.error("Error processing files.")
        st.stop()
        
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
                <span style="font-size:0.7rem; font-weight:600; color:#22c55e; letter-spacing:0.05em;">READY FOR ANALYSIS</span>
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
        
        if os.path.exists(file_path):
            try:
                meta = read_metadata(file_path)
                with st.expander(":: META DATA", expanded=False):
                    if "center" in meta:
                        st.caption("📍 LOCATION PIN")
                        df_map = pd.DataFrame([meta["center"]])
                        st.map(df_map, zoom=10, size=20, use_container_width=True)
                        st.code(f"LAT: {meta['center']['lat']:.4f}\nLON: {meta['center']['lon']:.4f}", language="yaml")
                    st.divider()
                    st.caption(f"CRS: {meta['crs_raw']}")
                    st.caption(f"DIM: {meta['width']}x{meta['height']} | BANDS: {meta['count']}")
            except Exception as e:
                st.warning(f"Metadata Error: {e}")
        
        st.markdown("#### EXECUTION")
        process_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
        
        if process_btn:
            with st.spinner("PROCESSING..."):
                arr = read_bands(file_path)
                meta_run = read_metadata(file_path)
                bounds_run = meta_run.get("bounds_wgs84", None)
                res = assess_vegetation(arr)
                auto_thr = auto_tune_assessment_thresholds(res["ndvi"], res["mndwi"])
                st.session_state["analysis_result"] = {"arr": arr, "bounds": bounds_run, **res, "auto_stats": auto_thr}
                st.toast("Processing Complete")

        if "analysis_result" in st.session_state:
            st.divider()
            st.markdown("**VISUALIZATION**")
            layer_mode = st.radio("Layer Mode", ["CLASSIFICATION", "TRUE COLOR (RGB)", "NDVI (VEGETATION)", "MNDWI (WATER)", "NDBI (CLEARINGS/BUILDUPS)"], label_visibility="collapsed")
            
            with st.expander(":: CLASSIFICATION LOGIC", expanded=False):
                ar = st.session_state["analysis_result"]
                tuning_mode = st.radio("Method", ["EXPLICIT (PHYSICAL)", "ADAPTIVE (OTSU)", "MANUAL"], label_visibility="collapsed")
                if tuning_mode == "EXPLICIT (PHYSICAL)":
                    t_water, t_clear, t_vlow, t_vhigh = REFERENCE_THRESHOLDS.values()
                elif tuning_mode == "ADAPTIVE (OTSU)":
                    stats = ar["auto_stats"]
                    t_water, t_clear, t_vlow, t_vhigh = stats["water_t"], stats["clear_t"], stats["veg_low"], stats["veg_high"]
                else:
                    t_water = st.slider("Water Max", -1.0, 1.0, 0.0, 0.05, key="an_t_water")
                    t_clear = st.slider("Barren Max", -1.0, 1.0, 0.6, 0.05, key="an_t_clear")
                    t_vlow, t_vhigh = st.slider("Veg Range", -1.0, 1.0, (0.6, 0.9), 0.05, key="an_t_veg")

            with st.expander(":: IMAGE AESTHETICS", expanded=False):
                p_min, p_max = st.slider("Histogram Clip %", 0, 100, (2, 98), 1, key="an_clip")
                b_val = st.slider("Brightness", 0.1, 3.0, 1.0, 0.1, key="an_bright")
                c_val = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1, key="an_cont")

    with col_view:
        if "analysis_result" in st.session_state:
            res = st.session_state["analysis_result"]
            base_rgb = sanitize_image(stretch(np.stack(res["arr"][:3], axis=-1), pmin=p_min, pmax=p_max))
            base_rgb = adjust_display(base_rgb, b_val, c_val)
            display_img, caption = base_rgb, "TRUE COLOR COMPOSITE"
            
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
                veg_norm = np.clip((np.where(veg_mask, res["ndvi"], np.nan) - t_vlow) / (t_vhigh - t_vlow + 1e-6), 0, 1)
                overlay[veg_mask] = colormaps["YlGn"](veg_norm)[..., :3][veg_mask]
                display_img = sanitize_image(overlay)
                caption = "LAND COVER CLASSIFICATION"
                st.markdown("""<div class="legend-box"><div class="l-item"><div class="l-swatch" style="background:#004dcc;"></div><span>WATER</span></div><div class="l-item"><div class="l-swatch" style="background:#998066;"></div><span>CLEARING/URBAN</span></div><div class="l-item"><div class="l-swatch" style="background:#aadd66;"></div><span>VEG (LOW)</span></div><div class="l-item"><div class="l-swatch" style="background:#228b22;"></div><span>VEG (HIGH)</span></div></div>""", unsafe_allow_html=True)
            
            if res.get("bounds"):
                fig = plot_with_grid(display_img, res["bounds"], title=f"GEO-REFERENCED: {caption}")
                st.pyplot(fig, use_container_width=True)
            else:
                st.image(display_img, use_container_width=True, channels="RGB")
                st.caption(f"VIEW: {caption} (No Geodata)")

# ------------------------------------------------------------------------------
# TAB 2: CHANGE DETECTION
# ------------------------------------------------------------------------------
with tab_change:
    all_dates = [extract_date(f) for f in raw_files]
    all_labels = [d.strftime("%Y-%m-%d") for d in all_dates]
    st.markdown("#### TEMPORAL RANGE")
    start_i, end_i = st.select_slider("TIMELINE WINDOW", options=list(range(len(raw_files))), value=(0, len(raw_files) - 1), format_func=lambda i: all_labels[i], label_visibility="collapsed")
    range_indices = list(range(start_i, end_i + 1))
    range_labels = [all_labels[i] for i in range_indices]
    
    with st.expander(":: ACTIVE OBSERVATIONS (EXCLUDE DATA)", expanded=True):
        selected_labels = st.multiselect("Uncheck bad observations:", options=range_labels, default=range_labels, label_visibility="collapsed", key="cd_exclude_multiselect")
    
    subset_files = [raw_files[i] for i in range_indices if all_labels[i] in selected_labels]
    subset_labels = [all_labels[i] for i in range_indices if all_labels[i] in selected_labels]
    
    if not subset_labels: st.error("No dates selected."); st.stop()
    timeline_str = " >> ".join(subset_labels) if len(subset_labels) <= 4 else f"{subset_labels[0]} >> ... ({len(subset_labels)-2} frames) ... >> {subset_labels[-1]}"
    st.markdown(f'<div style="background:#121214; padding:0.5rem 1rem; border-radius:4px; border:1px solid #27272a; font-family:\'JetBrains Mono\'; font-size:0.8rem; color:#a1a1aa; text-align:center; margin-bottom:1rem;">{timeline_str}</div>', unsafe_allow_html=True)

    if len(subset_files) < 2: st.warning("PLEASE SELECT AT LEAST 2 DATES."); st.stop()
    
    col_act_1, col_act_2 = st.columns([3, 1])
    with col_act_1:
        mode = "DELTA (A vs B)" if len(subset_files) == 2 else f"TREND SLOPE ({len(subset_files)} Scenes)"
        st.caption(f"MODE: {mode}")
    with col_act_2:
        calc_btn = st.button("COMPUTE TREND", type="primary", use_container_width=True)

    if calc_btn:
        with st.spinner("CALCULATING..."):
            sub_dates = [extract_date(f) for f in subset_files]
            arrays = [read_bands(f) for f in subset_files]
            ref_file, meta_ref = subset_files[-1], read_metadata(subset_files[-1])
            bounds_ref = meta_ref.get("bounds_wgs84", None)
            ndvi_stack = np.stack([compute_ndvi(a) for a in arrays])
            mask_ref = valid_data_mask(arrays[-1]) & (~water_mask_from_mndwi(compute_mndwi(arrays[-1])))
            ndvi_stack = np.where(mask_ref, ndvi_stack, np.nan)
            
            if len(subset_files) == 2:
                d_ndvi, d_ndbi = delta(ndvi_stack[0], ndvi_stack[1]), delta(compute_ndbi(arrays[0]), compute_ndbi(arrays[1]))
                score, raw_change = composite_change_score(d_ndvi, d_ndbi, ndvi_stack[0]), d_ndvi
            else:
                years = np.array([d.year + d.timetuple().tm_yday/365.25 for d in sub_dates])
                slope = ndvi_slope(years, ndvi_stack)
                score, raw_change = -slope, slope
                
            st.session_state["cd_result"] = {"score": score, "change_val": raw_change, "img_latest": arrays[-1], "mask": mask_ref, "bounds": bounds_ref}
            st.toast("Calculation Complete")

    if "cd_result" in st.session_state:
        st.divider()
        cd = st.session_state["cd_result"]
        c_dash, c_map = st.columns([1, 3])
        
        with c_dash:
            st.markdown("##### SETTINGS")
            with st.expander(":: SENSITIVITY", expanded=True):
                sens = st.slider("Threshold Stringency", 0.0, 1.0, 0.5, 0.05, key="cd_sens")
                show_loss, show_gain = st.checkbox("Show Loss", True, key="cd_loss"), st.checkbox("Show Gain", True, key="cd_gain")
            with st.expander(":: MAP STYLE", expanded=False):
                alpha = st.slider("Opacity", 0.0, 1.0, 0.6, 0.1, key="cd_alpha")
                cp_min, cp_max = st.slider("Clip %", 0, 100, (2, 98), 1, key="cd_clip")
                map_b, map_c = st.slider("Brightness", 0.1, 3.0, 0.30, 0.05, key="cd_bright"), st.slider("Contrast", 0.1, 3.0, 1.0, 0.1, key="cd_cont")

            deg_mask = binary_opening(apply_threshold(cd["score"], aggressiveness_to_threshold(cd["score"], sens)), structure=np.ones((3,3)))
            imp_mask = binary_opening(cd["change_val"] > max(np.nanpercentile(cd["change_val"], 85 - (sens * 15)), 0.05), structure=np.ones((3,3)))
            
            px_total = np.sum(cd["mask"])
            p_loss = (np.sum(deg_mask & cd["mask"]) / px_total * 100) if px_total > 0 else 0
            p_gain = (np.sum(imp_mask & cd["mask"]) / px_total * 100) if px_total > 0 else 0
            ext_stats = compute_extended_stats(cd["change_val"], cd["mask"], deg_mask, imp_mask)
            full_stats = {"p_loss": p_loss, "p_gain": p_gain, "net_change": p_gain - p_loss, **ext_stats}

            st.divider()
            st.markdown("##### EXPORT")
            with st.expander(":: GENERATE REPORT", expanded=False):
                rep_inc_stats, rep_inc_map = st.checkbox("Executive Summary", True), st.checkbox("Change Map", True)
                rep_inc_hist, rep_inc_meta = st.checkbox("Data Distribution", False), st.checkbox("Metadata", True)
                if st.button("COMPILE PDF", use_container_width=True):
                    with st.spinner("Compiling..."):
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            rgb_base = adjust_display(sanitize_image(stretch(np.stack(cd["img_latest"][:3], axis=-1), pmin=cp_min, pmax=cp_max)), brightness=map_b, contrast=map_c)
                            out_img_rep, c_loss, c_gain = rgb_base.copy(), np.array([0.75, 0.5, 0.98]), np.array([0.74, 0.94, 0.39])
                            if show_loss: out_img_rep[deg_mask] = (1-alpha)*out_img_rep[deg_mask] + alpha*c_loss
                            if show_gain: out_img_rep[imp_mask] = (1-alpha)*out_img_rep[imp_mask] + alpha*c_gain
                            map_path = os.path.join(tmpdirname, "map.png")
                            plt.imsave(map_path, sanitize_image(out_img_rep))
                            img_paths = {'map': map_path}
                            if rep_inc_hist:
                                hist_path = os.path.join(tmpdirname, "hist.png")
                                fig_h, ax_h = plt.subplots(figsize=(6, 3))
                                ax_h.hist(cd["change_val"][cd["mask"]], bins=50, color='#52525b')
                                fig_h.savefig(hist_path, dpi=150, bbox_inches='tight'); plt.close(fig_h)
                                img_paths['hist'] = hist_path
                            pdf_bytes = generate_change_report(stats=full_stats, meta_info={"timeline": timeline_str, "sensitivity": sens, "file_count": len(subset_files), "algorithm": mode, "brightness": map_b, "contrast": map_c, "clip": f"{cp_min}-{cp_max}%", "opacity": alpha}, images=img_paths, selections={"include_stats": rep_inc_stats, "include_map": rep_inc_map, "include_hist": rep_inc_hist, "include_meta": rep_inc_meta})
                            st.download_button(label="DOWNLOAD REPORT (.PDF)", data=pdf_bytes, file_name=f"EcoScepter_Report_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", type="primary")

        with c_map:
            st.markdown(f'<div class="metric-container"><div class="metric-box loss"><div class="metric-label">LOSS ({p_loss:.1f}%)</div><div class="metric-value" style="color:#c084fc; font-size:1.1rem;">INT: {ext_stats["avg_loss_val"]:.3f}</div></div><div class="metric-box gain"><div class="metric-label">GAIN ({p_gain:.1f}%)</div><div class="metric-value" style="color:#bef264; font-size:1.1rem;">INT: {ext_stats["avg_gain_val"]:.3f}</div></div><div class="metric-box"><div class="metric-label">NET CHANGE</div><div class="metric-value" style="color:#fafafa;">{p_gain - p_loss:+.2f}%</div></div></div>', unsafe_allow_html=True)
            rgb_base = adjust_display(sanitize_image(stretch(np.stack(cd["img_latest"][:3], axis=-1), pmin=cp_min, pmax=cp_max)), brightness=map_b, contrast=map_c)
            out_img, c_loss, c_gain = rgb_base.copy(), np.array([0.75, 0.5, 0.98]), np.array([0.74, 0.94, 0.39])
            if show_loss: out_img[deg_mask] = (1-alpha)*out_img[deg_mask] + alpha*c_loss
            if show_gain: out_img[imp_mask] = (1-alpha)*out_img[imp_mask] + alpha*c_gain
            if cd.get("bounds"):
                st.pyplot(plot_with_grid(sanitize_image(out_img), cd["bounds"], title=f"CHANGE MAP (GEO-REF) | SENSITIVITY: {sens}"), use_container_width=True)
            else:
                st.image(sanitize_image(out_img), use_container_width=True, caption=f"CHANGE MAP | SENSITIVITY: {sens}")

# ------------------------------------------------------------------------------
# TAB 3: DOCS
# ------------------------------------------------------------------------------
with tab_docs:
    st.markdown("""### SYSTEM DOCUMENTATION\n#### 1. DATA HYGIENE\n* **Active Observations:** Use the 'Exclude Data' dropdown to remove specific dates.\n#### 2. CLASSIFICATION LOGIC\n* **Explicit (Physical):** Fixed values (Water < 0.0, Veg > 0.6).\n* **Adaptive (Otsu):** Optimal thresholds based on histogram valleys.\n#### 3. CHANGE ALGORITHMS\n* **Delta:** Direct subtraction (B - A).\n* **Trend (Theil-Sen):** Robust slope calculation for 3+ points.""")