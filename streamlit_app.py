import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import time
from PIL import Image
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

# Configure page to force sidebar always expanded
st.set_page_config(
    page_title="IVF Embryo Score Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force sidebar to always be visible with CSS
st.markdown("""
<style>
    /* Force sidebar to always be visible */
    .css-1d391kg, .css-1lcbmhc, .css-1outpf7, .css-1y4p8pa, 
    /* Align all headings to the left */
    h1, h2, h3, h4, h5, h6 {
        text-align: left !important;
    }
    
    /* Specifically target plotly chart titles and graph headings */
    .js-plotly-plot .plotly .gtitle {
        text-anchor: start !important;
    }
    
    /* Target streamlit metric labels and chart titles */
    [data-testid="metric-container"] > div > div > div {
        text-align: left !important;
    }
    
    .css-17eq0hr, .css-1544g2n, .css-1lcbmhc, .css-1outpf7,
    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        width: 300px !important;
        margin-left: 0px !important;
        transform: translateX(0px) !important;
        visibility: visible !important;
        display: block !important;
    }
    
    /* Hide the sidebar collapse button */
    button[kind="header"] {
        display: none !important;
    }
    
    /* Ensure main content adjusts properly */
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
    }
    
    /* Hide any collapse controls */
    .css-1lcbmhc .css-1outpf7 button,
    .css-1y4p8pa button[aria-label*="collapse"],
    .css-1y4p8pa button[aria-label*="Close"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = True

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom toggle button */
    .sidebar-toggle-btn {
        position: fixed;
        top: 60px;
        left: 15px;
        z-index: 999999;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: 2px solid rgba(255,255,255,0.2);
        padding: 10px 15px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        text-decoration: none;
        display: inline-block;
    }
    
    .sidebar-toggle-btn:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.6);
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-color: rgba(255,255,255,0.3);
    }
    
    .sidebar-toggle-btn:active {
        transform: translateY(-1px) scale(1.02);
    }
    
    /* Adjust button position when sidebar is collapsed */
    .sidebar-collapsed .sidebar-toggle-btn {
        left: 15px;
    }
    
    .sidebar-expanded .sidebar-toggle-btn {
        left: 315px;
    }
    
    /* Enhanced main content styling */
    .gradient-header {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 25%, #d946ef 50%, #f97316 75%, #fb923c 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15), 0 0 0 1px rgba(255,255,255,0.1) inset;
        position: relative;
        overflow: hidden;
    }
    
    .gradient-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .gradient-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 600;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .gradient-header .dna-icon {
        font-size: 3rem;
        animation: pulse 2s infinite, rotate 10s linear infinite;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    
    .gradient-header .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        font-weight: 300;
        margin-top: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05), 0 0 0 1px rgba(255,255,255,0.8) inset;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 0 2px 2px 0;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.9) inset;
    }
    
    .metric-card h3 {
        color: #475569;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: #1e293b;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Enhanced Prediction Result */
    .prediction-result {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 50%, #a7f3d0 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #10b981;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-result::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }
    
    .prediction-result h2 {
        color: #065f46;
        font-weight: 600;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .prediction-result h1 {
        position: relative;
        z-index: 1;
    }
    
    .threshold-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .threshold-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .threshold-value {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 1rem;
        padding: 0.5rem;
        background: rgba(255,255,255,0.8);
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e2e8f0;
        position: relative;
    }
    
    .sub-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 2px;
    }
    
    /* Enhanced Info Boxes */
    .error-message {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 4px solid #ef4444;
        color: #dc2626;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }
    
    .image-prediction-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #10b981;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .image-prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .image-prediction-box h3 {
        position: relative;
        z-index: 1;
        color: #065f46;
        margin-bottom: 1rem;
    }
    
    .image-prediction-box p {
        position: relative;
        z-index: 1;
        margin: 0.5rem 0;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Enhanced Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Enhanced File Uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader:hover {
        border-color: #3b82f6 !important;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
    }
    
    /* Enhanced Selectbox */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Enhanced Slider */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    /* Enhanced Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Enhanced Metrics */
    .css-1xarl3l {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Enhanced Plotly Charts */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        overflow: hidden;
    }
    
    /* Footer Enhancement */
    .footer {
        background: linear-gradient(135deg, #1e293