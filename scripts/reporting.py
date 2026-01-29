# EcoScepter/scripts/reporting.py
import os
from datetime import datetime
from fpdf import FPDF
import pandas as pd

class EcoReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(200, 200, 200)
        self.cell(0, 10, 'EcoScepter | Analytics Report', 0, 1, 'R')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_change_report(
    stats: dict, 
    meta_info: dict,
    images: dict, 
    selections: dict
) -> bytes:
    """
    Generates a PDF report based on user selections.
    images dict should contain paths to temporary image files: {'map': 'path/to/map.png', 'hist': '...'}
    """
    pdf = EcoReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- TITLE SECTION ---
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 15, "Change Detection Analysis", ln=True)
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 5, f"Session ID: {meta_info.get('session_id', 'N/A')}", ln=True)
    pdf.ln(10)

    # --- 1. GLOBAL STATISTICS ---
    if selections.get("include_stats", True):
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0)
        pdf.cell(0, 10, "1. Executive Summary", ln=True)
        
        pdf.set_font("Arial", "", 11)
        pdf.set_fill_color(245, 245, 245)
        
        # Simple Table
        metrics = [
            ("Total Loss Detected", f"{stats['p_loss']:.2f}%"),
            ("Total Gain Detected", f"{stats['p_gain']:.2f}%"),
            ("Net Ecological Shift", f"{stats['net_change']:+.2f}%"),
            ("Avg. Degradation Intensity", f"{stats['avg_loss_val']:.4f}"),
            ("Avg. Recovery Intensity", f"{stats['avg_gain_val']:.4f}"),
        ]
        
        for label, value in metrics:
            pdf.cell(95, 8, label, 1, 0, 'L', 1)
            pdf.cell(95, 8, value, 1, 1, 'R', 0)
        pdf.ln(10)

    # --- 2. CHANGE MAP VISUALIZATION ---
    if selections.get("include_map", True) and "map" in images:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "2. Geospatial Change Map", ln=True)
        pdf.ln(2)
        
        # Image fits within page width (approx 190mm)
        try:
            pdf.image(images['map'], x=10, w=190)
            pdf.ln(5)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(0, 5, f"Figure 1: Spatial distribution of change. Purple indicates degradation/loss, Green indicates gain/recovery. Sensitivity Setting: {meta_info.get('sensitivity', 'N/A')}")
        except Exception as e:
            pdf.cell(0, 10, f"Error loading map image: {str(e)}", ln=True)
        pdf.ln(10)

    # --- 3. DISTRIBUTION ANALYSIS ---
    if selections.get("include_hist", True) and "hist" in images:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "3. Change Magnitude Distribution", ln=True)
        pdf.ln(2)
        try:
            pdf.image(images['hist'], x=30, w=150)
            pdf.ln(5)
            pdf.set_font("Arial", "I", 9)
            pdf.multi_cell(0, 5, "Figure 2: Histogram of spectral change values. Skewness towards negative values indicates prevalent degradation.")
        except Exception as e:
             pdf.cell(0, 10, f"Error loading histogram: {str(e)}", ln=True)
        pdf.ln(10)

    # --- 4. METADATA (UPDATED) ---
    if selections.get("include_meta", True):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "4. Technical & Visual Metadata", ln=True)
        
        pdf.set_font("Courier", "", 9)
        # UPDATED: Added Visual Parameters to the text block
        meta_txt = f"""
        -- DATA PARAMETERS --
        Timeline:       {meta_info.get('timeline', 'N/A')}
        Algorithm:      {meta_info.get('algorithm', 'N/A')}
        Input Scenes:   {meta_info.get('file_count', 0)}
        
        -- ANALYSIS PARAMETERS --
        Sensitivity:    {meta_info.get('sensitivity', 0.5)}
        
        -- VISUAL PARAMETERS --
        Brightness:     {meta_info.get('brightness', 'N/A')}
        Contrast:       {meta_info.get('contrast', 'N/A')}
        Clip Range:     {meta_info.get('clip', 'N/A')}
        Opacity:        {meta_info.get('opacity', 'N/A')}
        """
        pdf.multi_cell(0, 5, meta_txt)

    return bytes(pdf.output(dest='S'))

    # Output to byte string (Already bytes in your version)
    return bytes(pdf.output(dest='S'))