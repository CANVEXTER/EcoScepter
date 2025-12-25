import os
import glob
import streamlit as st
import numpy as np
from matplotlib import cm
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

# -------------------------------
# VISUAL REFERENCE (SEMANTIC ONLY)
# -------------------------------
REFERENCE_THRESHOLDS = {
    "water_t": 0.0,
    "clear_t": 0.60,
    "veg_low": 0.60,
    "veg_high": 0.90,
}

st.set_page_config(layout="wide")
st.title("Vegetation Assessment & Deforestation Detection")


def sanitize_image(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


tif_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif")))
if not tif_files:
    st.warning("No GeoTIFFs found in data/")
    st.stop()

tab_assess, tab_deforest = st.tabs(
    ["Vegetation Assessment", "Deforestation Detection"]
)

# ===============================================================
# TAB 1 — VEGETATION ASSESSMENT
# ===============================================================
with tab_assess:
    st.subheader("Per-Image Vegetation Health Assessment")

    tif_name = st.selectbox(
        "Select GeoTIFF",
        options=[os.path.basename(f) for f in tif_files],
    )
    tif_path = os.path.join(DATA_DIR, tif_name)

    if st.button("Run Assessment"):
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

    if "assess" in st.session_state:
        a = st.session_state["assess"]

        view_mode = st.selectbox(
            "Select visualization layer",
            [
                "Vegetation Assessment",
                "RGB",
                "NDVI (raw)",
                "MNDWI (raw)",
                "NDBI (raw)",
            ],
        )

        # ----------------------------------
        # Visualization control policy
        # ----------------------------------
        use_reference = False
        if view_mode == "Vegetation Assessment":
            use_reference = st.checkbox(
                "Use visual reference thresholds",
                value=True,
                help="Stable, human-interpretable vegetation classes",
            )

        if view_mode == "Vegetation Assessment" and not use_reference:
            water_t = st.slider(
                "Water threshold (MNDWI)",
                -1.0, 1.0, float(a["water_t"]), 0.01
            )
            clear_t = st.slider(
                "Clear land threshold (NDVI)",
                -1.0, 1.0, float(a["clear_t"]), 0.01
            )
            veg_low, veg_high = st.slider(
                "Vegetation NDVI range",
                -1.0, 1.0,
                (float(a["veg_low"]), float(a["veg_high"])),
                0.01
            )
        elif view_mode == "Vegetation Assessment" and use_reference:
            water_t = REFERENCE_THRESHOLDS["water_t"]
            clear_t = REFERENCE_THRESHOLDS["clear_t"]
            veg_low = REFERENCE_THRESHOLDS["veg_low"]
            veg_high = REFERENCE_THRESHOLDS["veg_high"]
        else:
            water_t = a["water_t"]
            clear_t = a["clear_t"]
            veg_low = a["veg_low"]
            veg_high = a["veg_high"]

        # ----------------------------------
        # Masks
        # ----------------------------------
        water = a["mndwi"] > water_t
        clear = (~water) & (a["ndvi"] < clear_t)
        veg_mask = (~water) & (~clear)

        rgb = sanitize_image(
            stretch(np.stack(a["arr"][:3], axis=-1))
        )

        # ----------------------------------
        # RAW SCIENTIFIC HEATMAPS (UNCHANGED)
        # ----------------------------------
        if view_mode == "RGB":
            st.image(rgb, width="stretch")

        elif view_mode == "NDVI (raw)":
            st.image(
                sanitize_image(index_to_rgb(a["ndvi"], "RdYlGn")),
                width="stretch",
            )

        elif view_mode == "MNDWI (raw)":
            st.image(
                sanitize_image(index_to_rgb(a["mndwi"], "Blues")),
                width="stretch",
            )

        elif view_mode == "NDBI (raw)":
            st.image(
                sanitize_image(index_to_rgb(a["ndbi"], "inferno")),
                width="stretch",
            )

        # ----------------------------------
        # VEGETATION ASSESSMENT (SEMANTIC)
        # ----------------------------------
        else:
            overlay = rgb.copy()

            overlay[water] = [0.0, 0.3, 0.8]       # Water
            overlay[clear] = [0.55, 0.4, 0.25]    # Bare / clear land

            ndvi = a["ndvi"].copy()
            ndvi[~veg_mask] = np.nan

            veg_norm = (ndvi - veg_low) / (veg_high - veg_low + 1e-6)
            veg_norm = np.clip(veg_norm, 0.0, 1.0)

            veg_rgb = cm.get_cmap("YlGn")(veg_norm)[..., :3]
            overlay[veg_mask] = veg_rgb[veg_mask]

            st.image(
                sanitize_image(overlay),
                width="stretch",
                caption="Vegetation Health Assessment (visual semantics)",
            )

            # ----------------------------
            # Visual reference legend
            # ----------------------------
            with st.expander("Vegetation color reference"):
                st.markdown(
                    """
                    **Blue** — Water bodies (rivers, lakes, flooded areas)  
                    **Brown** — Bare soil / built-up land  
                    **Light green** — Sparse vegetation (grassland, crops)  
                    **Green** — Moderate vegetation  
                    **Dark green** — Dense vegetation / tree cover  

                    This visualization is **relative** and intended for
                    **human interpretation**, not biomass quantification.
                    """
                )

# ===============================================================
# TAB 2 — DEFORESTATION DETECTION (UNCHANGED)
# ===============================================================
with tab_deforest:
    st.subheader("Deforestation Detection")

    if len(tif_files) < 2:
        st.warning("At least two GeoTIFFs are required.")
        st.stop()

    if st.button("Compute Change Score"):
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

        ndvi1, ndvi2, ndbi1, ndbi2 = apply_masks(
            ndvi1, ndvi2, ndbi1, ndbi2, mask=mask
        )

        score = composite_change_score(
            delta(ndvi1, ndvi2),
            delta(ndbi1, ndbi2),
            ndvi_baseline=ndvi1,
        )

        st.session_state["deforest_score"] = score

    if "deforest_score" in st.session_state:
        aggressiveness = st.slider(
            "Detection aggressiveness",
            0.0, 1.0, 0.5, 0.05
        )

        if st.button("Apply Threshold"):
            thr = aggressiveness_to_threshold(
                st.session_state["deforest_score"],
                aggressiveness,
            )
            mask = apply_threshold(
                st.session_state["deforest_score"], thr
            )
            mask = binary_opening(mask, structure=np.ones((3, 3)))
            st.session_state["deforest_mask"] = mask
            st.session_state["deforest_thr"] = thr

    if "deforest_mask" in st.session_state:
        rgb = sanitize_image(
            stretch(np.stack(read_bands(tif_files[-1])[:3], axis=-1))
        )
        overlay = rgb.copy()
        mask = st.session_state["deforest_mask"]
        color = np.array([0.55, 0.05, 0.30], dtype=np.float32)
        alpha = 0.65

        overlay[mask] = (
            (1 - alpha) * overlay[mask]
            + alpha * color
        )

        st.image(
            overlay,
            width="stretch",
            caption="Deforestation Hotspots",
        )
