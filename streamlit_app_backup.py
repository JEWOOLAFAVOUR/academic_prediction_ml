# Academic Performance Prediction System - Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os

# Add the model_and_others directory to the path
sys.path.append('model_and_others')

# Import prediction functions
from prediction_functions import (
    load_all_models, 
    predict_single_student, 
    predict_batch_students,
    validate_input_data,
    get_feature_importance
)

# Page configuration
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple Light Theme CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 1rem;
        background: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #ffffff;
    }
    
    /* Headers */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: #2d3748;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 500;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    /* Simple Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .form-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    .metric-card {
        background: #ffffff;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        border: 2px solid #e2e8f0;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.15);
        border-color: #4f46e5;
    }
    
    .metric-card h3 {
        color: #374151;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card h2 {
        color: #111827;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card p {
        color: #4b5563;
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Prediction Cards */
    .prediction-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        border: 2px solid #e5e7eb;
        margin: 1.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card h3 {
        color: #111827;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
    }
    
    .prediction-card h2 {
        color: #111827;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0.5rem 0;
    }
    
    .prediction-card p {
        color: #374151;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0.3rem 0;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        width: 6px;
        background: #10b981;
    }
    
    .risk-high {
        border-color: #fca5a5;
        background: #fef2f2;
    }
    
    .risk-high::before {
        background: #ef4444;
    }
    
    .risk-high h3, .risk-high h2, .risk-high p {
        color: #991b1b;
    }
    
    .risk-medium {
        border-color: #fed7aa;
        background: #fffbeb;
    }
    
    .risk-medium::before {
        background: #f59e0b;
    }
    
    .risk-medium h3, .risk-medium h2, .risk-medium p {
        color: #92400e;
    }
    
    .risk-low {
        border-color: #bbf7d0;
        background: #f0fdf4;
    }
    
    .risk-low::before {
        background: #10b981;
    }
    
    .risk-low h3, .risk-low h2, .risk-low p {
        color: #065f46;
    }
    
    /* Form Sections */
    .form-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        margin: 1rem 0;
        border: 2px solid #e5e7eb;
    }
    
    .form-section:hover {
        border-color: #d1d5db;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .form-section h4 {
        color: #111827;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .form-section p, .form-section li, .form-section span {
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        line-height: 1.6;
    }
    
    .form-section strong {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    /* Clean form section spacing */
    .form-section .stSelectbox, .form-section .stNumberInput, .form-section .stSlider {
        margin: 6px 0 16px 0 !important;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(79, 70, 229, 0.5);
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Feature Importance Cards */
    .feature-importance {
        background: #ffffff;
        padding: 1.8rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 24px rgba(0,0,0,0.08);
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .feature-importance:hover {
        transform: translateX(8px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        border-color: #6366f1;
    }
    
    .feature-importance strong {
        color: #111827;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .feature-importance span {
        color: #374151;
        font-weight: 500;
    }
    
    .feature-importance small {
        color: #6b7280;
        font-weight: 600;
    }
    
    /* Clean Form Controls */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        min-height: 42px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1) !important;
    }
    
    /* Selectbox text */
    .stSelectbox [data-baseweb="select"] > div {
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* Clean dropdown */
    .stSelectbox [role="listbox"] {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    .stSelectbox [role="option"] {
        color: #374151 !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
    }
    
    .stSelectbox [role="option"]:hover {
        background: #f9fafb !important;
        color: #111827 !important;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
        transition: all 0.2s ease !important;
        background: #ffffff !important;
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
        min-height: 42px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1) !important;
        outline: none !important;
    }
    
    /* Clean number input buttons */
    .stNumberInput button {
        color: #6b7280 !important;
        font-weight: 500 !important;
        background: #f9fafb !important;
        border: 1px solid #d1d5db !important;
        border-radius: 4px !important;
    }
    
    .stNumberInput button:hover {
        background: #f3f4f6 !important;
        color: #374151 !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #f9fafb !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    
    /* Clean Form Labels */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        font-family: 'Inter', sans-serif !important;
        margin-bottom: 4px !important;
    }
    
    /* Help text styling */
    .stSelectbox .help, .stNumberInput .help, .stSlider .help,
    .stSelectbox small, .stNumberInput small, .stSlider small {
        color: #6b7280 !important;
        font-weight: 400 !important;
        font-size: 12px !important;
    }
    
    /* Slider improvements */
    .stSlider > div > div > div > div {
        background: #4f46e5 !important;
    }
    
    .stSlider > div > div > div[role="slider"] {
        background: #4f46e5 !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3) !important;
    }
    
    /* Comprehensive Sidebar Enhancements */
    .css-1d391kg, .css-6qob1r, .css-1cypcdb, .css-17eq0hr, 
    .stSidebar, .stSidebar > div, section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background: #ffffff !important;
        border-right: 2px solid #d1d5db !important;
    }
    
    /* All sidebar text improvements */
    .css-1d391kg *, .css-6qob1r *, .css-1cypcdb *, .css-17eq0hr *,
    .stSidebar *, section[data-testid="stSidebar"] *,
    .stSelectbox div[data-baseweb="select"] span {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar headers */
    .css-1d391kg h1, .css-6qob1r h1, .css-1cypcdb h1, .css-17eq0hr h1,
    .css-1d391kg h2, .css-6qob1r h2, .css-1cypcdb h2, .css-17eq0hr h2,
    .css-1d391kg h3, .css-6qob1r h3, .css-1cypcdb h3, .css-17eq0hr h3,
    .css-1d391kg h4, .css-6qob1r h4, .css-1cypcdb h4, .css-17eq0hr h4,
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4,
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar selectbox improvements */
    .stSidebar .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #ffffff !important;
        border: 2px solid #374151 !important;
        color: #111827 !important;
    }
    
    /* Sidebar selectbox text */
    .stSidebar .stSelectbox div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar markdown content */
    .stSidebar .stMarkdown, 
    section[data-testid="stSidebar"] .stMarkdown {
        color: #374151 !important;
    }
    
    .stSidebar .stMarkdown p, 
    section[data-testid="stSidebar"] .stMarkdown p,
    .stSidebar .stMarkdown span, 
    section[data-testid="stSidebar"] .stMarkdown span,
    .stSidebar .stMarkdown li, 
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    .stSidebar .stMarkdown strong, 
    section[data-testid="stSidebar"] .stMarkdown strong {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    /* Data Display Tables */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Navigation Pills */
    .nav-pill {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        color: #4a5568;
        text-decoration: none;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .nav-pill:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 4px solid #48bb78;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left: 4px solid #f56565;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border-left: 4px solid #4299e1;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef5e7 100%);
        border-left: 4px solid #ed8936;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Loading Spinner */
    .stSpinner > div > div {
        border-top-color: #667eea !important;
    }
    
    /* Clean Button styling */
    .stButton button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
        min-height: 44px !important;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.2) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    .stRadio label {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > div {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    .stCheckbox label {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }
    
    /* Clean element containers */
    .element-container {
        margin-bottom: 8px;
    }
    
    .block-container {
        padding: 1rem;
    }
    
    /* Clean text input styling */
    .stTextInput > div > div > input {
        color: #374151 !important;
        font-weight: 500 !important;
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Force all sidebar text to be visible */
    [data-testid="stSidebar"] * {
        color: #374151 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] strong {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    /* Selectbox in sidebar */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #ffffff !important;
        border: 2px solid #374151 !important;
        color: #111827 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
    }
    
    [data-testid="stSidebar"] > div {
        background: #ffffff !important;
        border-right: 2px solid #d1d5db !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.components = None

# Load models function
@st.cache_resource
def initialize_models():
    """Load models and cache them"""
    try:
        components = load_all_models()
        if components is not None:
            return components, True
        else:
            return None, False
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, False

# Main app
def main():
    # Enhanced Header with subtitle
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 class="main-header">🎓 Academic Performance Predictor</h1>
        <p style="font-size: 1.3rem; color: #4b5563; font-family: 'Inter', sans-serif; 
                  font-weight: 600; margin-top: -1rem; animation: slideInLeft 1s ease-out;">
            AI-Powered Student Success Analytics & Early Risk Detection
        </p>
        <div style="width: 80px; height: 4px; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
                    margin: 1rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models with enhanced UI
    if not st.session_state.models_loaded:
        with st.spinner("🤖 Loading AI models and preparing the system..."):
            components, success = initialize_models()
            if success:
                st.session_state.components = components
                st.session_state.models_loaded = True
                st.balloons()
                st.success("✅ Models loaded successfully! Ready for predictions.")
            else:
                st.error("❌ Failed to load models. Please check if all model files are present.")
                st.info("💡 Try running the fix_models.py script to regenerate compatible models.")
                st.stop()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; background: #ffffff;">
            <h2 style="color: #111827 !important; font-family: 'Inter', sans-serif; font-weight: 800; margin-bottom: 0.5rem; font-size: 1.5rem;">
                🎯 Navigation
            </h2>
            <p style="color: #4b5563 !important; font-size: 1rem; margin-bottom: 2rem; font-weight: 600;">
                Academic Performance Predictor
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <style>
            .sidebar-selectbox {
                background: #ffffff !important;
                border: 2px solid #374151 !important;
                border-radius: 8px !important;
                padding: 0.5rem !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        page = st.selectbox(
            "🚀 Choose a page:",
            ["🏠 Home", "👤 Single Student Prediction", "📊 Batch Prediction", "📈 Model Analytics", "ℹ️ About"],
            index=0,
            key="page_selector"
        )
        
        st.markdown("---")
        
        # Quick stats in sidebar
        st.markdown("""
        <div style="background: #ffffff; border: 2px solid #e5e7eb;
                    padding: 1.2rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <h4 style="color: #111827; font-size: 1rem; margin-bottom: 0.8rem; font-weight: 700;">📊 Quick Stats</h4>
            <p style="color: #374151; font-size: 0.9rem; margin: 0.4rem 0; font-weight: 600;">✅ Model Accuracy: 83.5%</p>
            <p style="color: #374151; font-size: 0.9rem; margin: 0.4rem 0; font-weight: 600;">👥 Students Analyzed: 909</p>
            <p style="color: #374151; font-size: 0.9rem; margin: 0.4rem 0; font-weight: 600;">🔍 Features: 14</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tips section
        st.markdown("""
        <div style="background: #dbeafe; border: 2px solid #93c5fd;
                    padding: 1.2rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <h4 style="color: #1e40af; font-size: 1rem; margin-bottom: 0.8rem; font-weight: 700;">💡 Pro Tips</h4>
            <p style="color: #1e3a8a; font-size: 0.9rem; margin: 0.4rem 0; font-weight: 600;">• Use realistic values for better predictions</p>
            <p style="color: #1e3a8a; font-size: 0.9rem; margin: 0.4rem 0; font-weight: 600;">• Check feature importance in Analytics</p>
            <p style="color: #1e3a8a; font-size: 0.9rem; margin: 0.4rem 0; font-weight: 600;">• Download CSV template for batch uploads</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "👤 Single Student Prediction":
        single_prediction_page()
    elif page == "📊 Batch Prediction":
        batch_prediction_page()
    elif page == "📈 Model Analytics":
        analytics_page()
    elif page == "ℹ️ About":
        about_page()

def home_page():
    """Home page with overview and quick stats"""
    col1, col2, col3 = st.columns(3)
    
    metadata = st.session_state.components['metadata']
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Model Accuracy</h3>
            <h2>83.5%</h2>
            <p>Logistic Regression Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dataset Size</h3>
            <h2>909</h2>
            <p>Students Successfully Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Features Used</h3>
            <h2>14</h2>
            <p>Comprehensive Student Attributes</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="sub-header">🎯 What This System Does</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="form-section" style="margin-bottom: 2rem;">
            <div style="display: grid; gap: 1rem;">
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">🎯</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Predicts</strong> student academic performance categories</span>
                </div>
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">⚡</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Identifies</strong> at-risk students early</span>
                </div>
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">📊</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Analyzes</strong> study habits and lifestyle factors</span>
                </div>
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">💡</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Provides</strong> actionable insights for improvement</span>
                </div>
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">🔄</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Supports</strong> both individual and batch predictions</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="sub-header">🚀 Key Features</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="form-section">
            <div style="display: grid; gap: 1rem;">
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">⚡</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Real-time Predictions:</strong> Instant results for individual students</span>
                </div>
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">📁</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Batch Processing:</strong> Upload CSV files for multiple students</span>
                </div>
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">🚨</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Risk Assessment:</strong> Categorizes students by risk level</span>
                </div>
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">🔍</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Feature Importance:</strong> Shows which factors matter most</span>
                </div>
                <div style="display: flex; align-items: center; padding: 0.5rem 0;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">📈</span>
                    <span style="color: #4a5568; font-size: 1rem;"><strong>Interactive Visualizations:</strong> Easy-to-understand charts</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Performance distribution chart
        distribution = metadata['target_variable']['distribution']
        
        fig = px.pie(
            values=list(distribution.values()),
            names=list(distribution.keys()),
            title="Student Performance Distribution",
            color_discrete_map={
                'Good': '#28a745',
                'Average': '#ffc107', 
                'Poor': '#dc3545'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance chart
        importance_data = get_feature_importance()
        features = [item['feature'].replace('_', ' ').title() for item in importance_data]
        importances = [item['importance'] for item in importance_data]
        
        fig2 = px.bar(
            x=importances,
            y=features,
            orientation='h',
            title="Top 5 Most Important Features",
            color=importances,
            color_continuous_scale="viridis"
        )
        fig2.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

def single_prediction_page():
    """Single student prediction page"""
    st.markdown('<h2 class="sub-header">👤 Single Student Prediction</h2>', unsafe_allow_html=True)
    
    # Load feature descriptions
    with open('model_and_others/feature_descriptions.json', 'r') as f:
        feature_descriptions = json.load(f)
    
    # Create enhanced input form
    with st.form("student_form"):
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem; padding: 1rem; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;">
            <h3 style="color: #1a202c; font-family: 'Inter', sans-serif; font-weight: 600; margin: 0;">
                📝 Student Information Form
            </h3>
            <p style="color: #4a5568; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Complete all fields for accurate prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <h4 style="color: #1a202c; margin: 0 0 0.5rem 0; font-weight: 600; font-size: 1rem;">👤 Personal Information</h4>
            </div>
            """, unsafe_allow_html=True)
            age = st.number_input("🎂 Age", min_value=17, max_value=24, value=20, help=feature_descriptions['age'])
            gender = st.selectbox("⚧ Gender", ["Male", "Female"], help=feature_descriptions['gender'])
            part_time_job = st.selectbox("💼 Part-time Job", ["No", "Yes"], help=feature_descriptions['part_time_job'])
            parental_education = st.selectbox(
                "🎓 Parental Education Level", 
                ["High School", "Bachelor", "Master"],
                help=feature_descriptions['parental_education_level']
            )
            extracurricular = st.selectbox(
                "🏆 Extracurricular Activities", 
                ["No", "Yes"],
                help=feature_descriptions['extracurricular_participation']
            )
        
        with col2:
            st.markdown("""
            <div style="background: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <h4 style="color: #1a202c; margin: 0 0 0.5rem 0; font-weight: 600; font-size: 1rem;">📚 Study & Screen Time</h4>
            </div>
            """, unsafe_allow_html=True)
            study_hours = st.slider(
                "📖 Study Hours per Day", 
                min_value=0.0, max_value=8.3, value=4.0, step=0.1,
                help=feature_descriptions['study_hours_per_day']
            )
            social_media_hours = st.slider(
                "📱 Social Media Hours per Day", 
                min_value=0.0, max_value=7.2, value=2.5, step=0.1,
                help=feature_descriptions['social_media_hours']
            )
            netflix_hours = st.slider(
                "📺 Netflix/Streaming Hours per Day", 
                min_value=0.0, max_value=5.4, value=1.8, step=0.1,
                help=feature_descriptions['netflix_hours']
            )
            attendance = st.slider(
                "📅 Attendance Percentage", 
                min_value=56.0, max_value=100.0, value=84.0, step=1.0,
                help=feature_descriptions['attendance_percentage']
            )
        
        with col3:
            st.markdown("""
            <div style="background: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <h4 style="color: #1a202c; margin: 0 0 0.5rem 0; font-weight: 600; font-size: 1rem;">🏥 Health & Lifestyle</h4>
            </div>
            """, unsafe_allow_html=True)
            sleep_hours = st.slider(
                "😴 Sleep Hours per Day", 
                min_value=3.2, max_value=10.0, value=6.5, step=0.1,
                help=feature_descriptions['sleep_hours']
            )
            exercise_frequency = st.selectbox(
                "🏃 Exercise Frequency (per week)", 
                [0, 1, 2, 3, 4, 5, 6],
                index=3,
                help=feature_descriptions['exercise_frequency']
            )
            diet_quality = st.selectbox(
                "🥗 Diet Quality", 
                ["Poor", "Fair", "Good"],
                index=1,
                help=feature_descriptions['diet_quality']
            )
            mental_health = st.slider(
                "🧠 Mental Health Rating", 
                min_value=1, max_value=10, value=5,
                help=feature_descriptions['mental_health_rating']
            )
            internet_quality = st.selectbox(
                "🌐 Internet Quality", 
                ["Poor", "Average", "Good"],
                index=1,
                help=feature_descriptions['internet_quality']
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Academic Performance", use_container_width=True)
    
    if submitted:
        # Prepare student data
        student_data = {
            'age': age,
            'gender': gender,
            'study_hours_per_day': study_hours,
            'social_media_hours': social_media_hours,
            'netflix_hours': netflix_hours,
            'part_time_job': part_time_job,
            'attendance_percentage': attendance,
            'sleep_hours': sleep_hours,
            'diet_quality': diet_quality,
            'exercise_frequency': exercise_frequency,
            'parental_education_level': parental_education,
            'internet_quality': internet_quality,
            'mental_health_rating': mental_health,
            'extracurricular_participation': extracurricular
        }
        
        # Make prediction
        with st.spinner("Analyzing student data..."):
            try:
                results = predict_single_student(student_data, st.session_state.components)
                
                # Display results
                st.markdown("---")
                st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Primary prediction
                    primary = results['primary_prediction']
                    risk_class = f"risk-{results['risk_level'].split()[0].lower()}"
                    
                    st.markdown(f"""
                    <div class="prediction-card {risk_class}">
                        <h3>🎯 Primary Prediction (Logistic Regression)</h3>
                        <h2>{primary['prediction']}</h2>
                        <p><strong>Confidence:</strong> {primary['confidence_percentage']}</p>
                        <p><strong>Risk Level:</strong> {results['risk_level']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Secondary prediction
                    secondary = results['secondary_prediction']
                    agreement_icon = "✅" if results['agreement'] else "⚠️"
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🔍 Secondary Prediction (SVM)</h3>
                        <h2>{secondary['prediction']}</h2>
                        <p><strong>Confidence:</strong> {secondary['confidence_percentage']}</p>
                        <p><strong>Model Agreement:</strong> {agreement_icon}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown('<h3 class="sub-header">📈 Detailed Probability Breakdown</h3>', unsafe_allow_html=True)
                
                prob_data = {
                    'Performance Category': list(primary['probabilities'].keys()),
                    'Logistic Regression': [f"{prob:.1%}" for prob in primary['probabilities'].values()],
                    'SVM': [f"{prob:.1%}" for prob in secondary['probabilities'].values()]
                }
                
                prob_df = pd.DataFrame(prob_data)
                st.dataframe(prob_df, use_container_width=True)
                
                # Visualization
                categories = list(primary['probabilities'].keys())
                lr_probs = list(primary['probabilities'].values())
                svm_probs = list(secondary['probabilities'].values())
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Logistic Regression', x=categories, y=lr_probs, marker_color='#667eea'))
                fig.add_trace(go.Bar(name='SVM', x=categories, y=svm_probs, marker_color='#764ba2'))
                
                fig.update_layout(
                    title='Model Predictions Comparison',
                    xaxis_title='Performance Category',
                    yaxis_title='Probability',
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                show_recommendations(student_data, results)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def batch_prediction_page():
    """Batch prediction page for CSV uploads"""
    st.markdown('<h2 class="sub-header">📊 Batch Student Prediction</h2>', unsafe_allow_html=True)
    
    # Instructions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="form-section">
            <h3 style="color: #2d3748; font-family: 'Inter', sans-serif; margin-bottom: 1rem;">📋 Instructions</h3>
            <ol style="color: #4a5568; font-size: 1rem; line-height: 1.6;">
                <li><strong>Download</strong> the sample template below</li>
                <li><strong>Fill in</strong> your student data following the same format</li>
                <li><strong>Upload</strong> your CSV file using the uploader</li>
                <li><strong>View</strong> predictions and download results</li>
            </ol>
            <div style="background: #fef3c7; border: 2px solid #f59e0b; padding: 1.2rem; border-radius: 10px; margin-top: 1rem;">
                <strong style="color: #92400e; font-weight: 700; font-size: 1rem;">📌 Important:</strong>
                <span style="color: #78350f; font-weight: 600;"> All 14 features must be present in your CSV file.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Download sample template
        with open('model_and_others/sample_upload_template.csv', 'r') as f:
            sample_csv = f.read()
        
        st.download_button(
            label="📥 Download Sample Template",
            data=sample_csv,
            file_name="student_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # File uploader
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📤 Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file with student data following the template format"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} students.")
            
            # Validate the data
            is_valid, missing_cols, extra_cols = validate_input_data(df)
            
            if not is_valid:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Please ensure your CSV file has all required columns as shown in the template.")
            else:
                st.success("✅ Data validation passed!")
                
                # Show preview
                st.markdown('<h3 class="sub-header">👀 Data Preview</h3>', unsafe_allow_html=True)
                st.dataframe(df.head(), use_container_width=True)
                
                # Prediction button
                if st.button("🔮 Generate Predictions", use_container_width=True):
                    with st.spinner(f"Processing {len(df)} students..."):
                        try:
                            results_df = predict_batch_students(df, st.session_state.components)
                            
                            st.markdown("---")
                            st.markdown('<h2 class="sub-header">📊 Batch Prediction Results</h2>', unsafe_allow_html=True)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            total_students = len(results_df)
                            high_risk = len(results_df[results_df['Risk_Level'] == 'High Risk'])
                            medium_risk = len(results_df[results_df['Risk_Level'] == 'Medium Risk'])
                            low_risk = len(results_df[results_df['Risk_Level'] == 'Low Risk'])
                            
                            with col1:
                                st.metric("👥 Total Students", total_students)
                            with col2:
                                st.metric("🔴 High Risk", high_risk, f"{high_risk/total_students:.1%}")
                            with col3:
                                st.metric("🟡 Medium Risk", medium_risk, f"{medium_risk/total_students:.1%}")
                            with col4:
                                st.metric("🟢 Low Risk", low_risk, f"{low_risk/total_students:.1%}")
                            
                            # Results visualization
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Risk distribution pie chart
                                risk_counts = results_df['Risk_Level'].value_counts()
                                fig1 = px.pie(
                                    values=risk_counts.values,
                                    names=risk_counts.index,
                                    title="Risk Level Distribution",
                                    color_discrete_map={
                                        'Low Risk': '#28a745',
                                        'Medium Risk': '#ffc107',
                                        'High Risk': '#dc3545'
                                    }
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                # Prediction distribution
                                pred_counts = results_df['LR_Prediction'].value_counts()
                                fig2 = px.bar(
                                    x=pred_counts.index,
                                    y=pred_counts.values,
                                    title="Performance Category Distribution",
                                    color=pred_counts.index,
                                    color_discrete_map={
                                        'Good': '#28a745',
                                        'Average': '#ffc107',
                                        'Poor': '#dc3545'
                                    }
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # Detailed results table
                            st.markdown('<h3 class="sub-header">📋 Detailed Results</h3>', unsafe_allow_html=True)
                            
                            # Filter options
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                risk_filter = st.selectbox("Filter by Risk Level:", ["All"] + list(results_df['Risk_Level'].unique()))
                            with col2:
                                pred_filter = st.selectbox("Filter by Prediction:", ["All"] + list(results_df['LR_Prediction'].unique()))
                            with col3:
                                agreement_filter = st.selectbox("Model Agreement:", ["All", True, False])
                            
                            # Apply filters
                            filtered_df = results_df.copy()
                            if risk_filter != "All":
                                filtered_df = filtered_df[filtered_df['Risk_Level'] == risk_filter]
                            if pred_filter != "All":
                                filtered_df = filtered_df[filtered_df['LR_Prediction'] == pred_filter]
                            if agreement_filter != "All":
                                filtered_df = filtered_df[filtered_df['Model_Agreement'] == agreement_filter]
                            
                            st.dataframe(filtered_df, use_container_width=True, height=400)
                            
                            # Download results
                            csv_results = filtered_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv_results,
                                file_name=f"prediction_results_{len(filtered_df)}_students.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"Error processing predictions: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format.")

def analytics_page():
    """Model analytics and insights page"""
    st.markdown('<h2 class="sub-header">📈 Model Analytics & Insights</h2>', unsafe_allow_html=True)
    
    metadata = st.session_state.components['metadata']
    
    # Model performance section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="sub-header">🎯 Model Performance</h3>', unsafe_allow_html=True)
        
        # Performance metrics
        perf_data = {
            'Model': ['Logistic Regression', 'Support Vector Machine'],
            'Accuracy': [83.5, 78.6],
            'Status': ['Primary (Recommended)', 'Secondary']
        }
        
        perf_df = pd.DataFrame(perf_data)
        
        fig = px.bar(
            perf_df, 
            x='Model', 
            y='Accuracy',
            text='Accuracy',
            title='Model Accuracy Comparison',
            color='Accuracy',
            color_continuous_scale='viridis'
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(height=400, yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        st.markdown("""
        **🔍 Model Details:**
        - **Algorithm**: Logistic Regression (Primary)
        - **Accuracy**: 83.5%
        - **Best at**: Identifying poor performers
        - **Model Agreement**: 100% on test cases
        - **Training Data**: 909 students, 14 features
        """)
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 Dataset Distribution</h3>', unsafe_allow_html=True)
        
        # Dataset distribution
        distribution = metadata['target_variable']['distribution']
        
        fig2 = px.pie(
            values=list(distribution.values()),
            names=list(distribution.keys()),
            title="Student Performance Distribution in Training Data",
            color_discrete_map={
                'Good': '#28a745',
                'Average': '#ffc107',
                'Poor': '#dc3545'
            },
            hole=0.4
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Dataset stats
        total = sum(distribution.values())
        st.markdown(f"""
        **📈 Dataset Statistics:**
        - **Total Students**: {total:,}
        - **Good Performers**: {distribution['Good']} ({distribution['Good']/total:.1%})
        - **Average Performers**: {distribution['Average']} ({distribution['Average']/total:.1%})
        - **Poor Performers**: {distribution['Poor']} ({distribution['Poor']/total:.1%})
        """)
    
    # Feature importance analysis
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🔍 Feature Importance Analysis</h3>', unsafe_allow_html=True)
    
    importance_data = get_feature_importance()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Feature importance chart
        features = [item['feature'].replace('_', ' ').title() for item in importance_data]
        importances = [item['importance'] for item in importance_data]
        impacts = [item['impact'] for item in importance_data]
        
        colors = ['#dc3545' if impact == 'negative' else '#28a745' for impact in impacts]
        
        fig3 = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{imp:.3f}" for imp in importances],
            textposition='auto'
        ))
        
        fig3.update_layout(
            title="Feature Importance (Top 5)",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Key Insights:**")
        
        for item in importance_data:
            impact_icon = "📉" if item['impact'] == 'negative' else "📈"
            impact_color = "red" if item['impact'] == 'negative' else "green"
            
            st.markdown(f"""
            <div class="feature-importance">
                <strong>{item['feature'].replace('_', ' ').title()}</strong> {impact_icon}<br>
                <span style="color: {impact_color};">Impact: {item['impact'].title()}</span><br>
                <small>Importance: {item['importance']:.3f}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations based on insights
    st.markdown("---")
    st.markdown('<h3 class="sub-header">💡 Key Recommendations</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📚 For Students:**
        - **Reduce social media usage** - Most negative factor
        - **Increase daily study hours** - Second most important
        - **Maintain good diet quality** - Significant positive impact
        - **Focus on mental health** - Strong correlation with performance
        - **Join extracurricular activities** - Helps with overall development
        """)
    
    with col2:
        st.markdown("""
        **🏫 For Educators:**
        - **Early identification** of at-risk students (social media habits)
        - **Promote study habits** through structured programs
        - **Mental health support** programs
        - **Nutrition awareness** campaigns
        - **Balanced lifestyle** education
        """)

def about_page():
    """About page with project information"""
    st.markdown('<h2 class="sub-header">ℹ️ About the Academic Performance Predictor</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        The **Academic Performance Prediction System** is an AI-powered tool designed to help educational institutions 
        identify students who may be at risk of poor academic performance. By analyzing study habits, lifestyle factors, 
        and personal characteristics, the system provides early warnings and actionable insights.
        
        ### 🔬 Methodology
        
        **Machine Learning Approach:**
        - **Primary Algorithm**: Logistic Regression (83.5% accuracy)
        - **Secondary Algorithm**: Support Vector Machine (78.6% accuracy)  
        - **Training Dataset**: 909 students with 14 behavioral features
        - **Validation**: Stratified train-test split (80-20)
        
        **Key Features Analyzed:**
        1. **Study Patterns** - Daily study hours, attendance percentage
        2. **Digital Habits** - Social media usage, streaming hours
        3. **Lifestyle Factors** - Sleep, exercise, diet quality
        4. **Personal Context** - Age, family background, mental health
        
        ### 📊 Performance Categories
        
        - **Good (Low Risk)** - Exam scores 80-100%
        - **Average (Medium Risk)** - Exam scores 60-79%
        - **Poor (High Risk)** - Exam scores below 60%
        
        ### 🎯 Use Cases
        
        **For Educational Institutions:**
        - Early identification of at-risk students
        - Resource allocation for student support programs
        - Data-driven intervention strategies
        - Academic counseling prioritization
        
        **For Students & Parents:**
        - Self-assessment of study habits
        - Understanding factors affecting academic success
        - Guidance for lifestyle improvements
        - Performance prediction for goal setting
        """)
    
    with col2:
        # Technical specifications
        st.markdown("""
        ### 🛠️ Technical Specifications
        
        **Model Architecture:**
        - Logistic Regression (Primary)
        - Support Vector Machine (Secondary)
        - Standard Scaler for normalization
        - Label encoding for categories
        
        **Performance Metrics:**
        - **Accuracy**: 83.5%
        - **Precision**: High for poor performers
        - **Model Agreement**: 100% on test cases
        - **Training Time**: < 1 second
        
        **Technology Stack:**
        - **Backend**: Python, Scikit-learn
        - **Frontend**: Streamlit
        - **Visualization**: Plotly
        - **Data Processing**: Pandas, NumPy
        
        **Data Security:**
        - No personal identification stored
        - Anonymized predictions
        - Local processing only
        """)
        
        # Model metadata
        metadata = st.session_state.components['metadata']
        
        st.markdown(f"""
        ### 📈 Model Information
        
        **Version**: {metadata['project_info']['version']}
        **Created**: {metadata['project_info']['created_date'][:10]}
        **Dataset**: {metadata['project_info']['dataset_size']}
        **Features**: {metadata['features']['total_features']}
        **Accuracy**: {metadata['model_performance']['primary_model']['accuracy_percentage']}
        """)
    
    # Disclaimers and limitations
    st.markdown("---")
    st.markdown('<h3 class="sub-header">⚠️ Important Disclaimers</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Model Limitations:**
        - Predictions are probabilistic, not definitive
        - Based on historical patterns, individual results may vary
        - Cultural and contextual factors not fully captured
        - Should supplement, not replace, human judgment
        """)
    
    with col2:
        st.markdown("""
        **📋 Usage Guidelines:**
        - Use as an early warning system, not final assessment
        - Combine with qualitative observations
        - Consider individual student circumstances
        - Regular model updates recommended with new data
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 📞 Support & Contact
    
    For technical support, feature requests, or academic collaboration opportunities, 
    please contact the development team through your institutional channels.
    
    **Remember**: This tool is designed to support educational success, not to label or limit students. 
    Every student has the potential to improve with the right support and intervention.
    """)

def show_recommendations(student_data, results):
    """Show personalized recommendations based on student data and prediction"""
    st.markdown('<h3 class="sub-header">💡 Personalized Recommendations</h3>', unsafe_allow_html=True)
    
    recommendations = []
    
    # Analyze key factors and provide recommendations
    if student_data['social_media_hours'] > 3.0:
        recommendations.append("📱 **Reduce social media usage** - Currently high at {:.1f}h/day. Try limiting to 2 hours or less.".format(student_data['social_media_hours']))
    
    if student_data['study_hours_per_day'] < 3.0:
        recommendations.append("📚 **Increase study time** - Currently {:.1f}h/day. Aim for at least 4-5 hours for better performance.".format(student_data['study_hours_per_day']))
    
    if student_data['attendance_percentage'] < 80:
        recommendations.append("🎯 **Improve attendance** - Currently {:.1f}%. Aim for 85%+ attendance for better outcomes.".format(student_data['attendance_percentage']))
    
    if student_data['sleep_hours'] < 6:
        recommendations.append("😴 **Get more sleep** - Currently {:.1f}h/night. Aim for 7-8 hours for optimal cognitive function.".format(student_data['sleep_hours']))
    
    if student_data['mental_health_rating'] < 5:
        recommendations.append("🧠 **Focus on mental health** - Current rating: {}/10. Consider counseling or stress management techniques.".format(student_data['mental_health_rating']))
    
    if student_data['exercise_frequency'] < 2:
        recommendations.append("🏃 **Increase physical activity** - Currently {}x/week. Aim for 3-4 sessions per week.".format(student_data['exercise_frequency']))
    
    if student_data['diet_quality'] == 'Poor':
        recommendations.append("🥗 **Improve diet quality** - Good nutrition supports cognitive function and energy levels.")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.success("🎉 Great habits! Keep maintaining your current lifestyle for continued success.")
    
    # Risk-specific advice
    if results['risk_level'] == 'High Risk':
        st.warning("""
        **🔴 High Risk Alert**: This student shows patterns associated with poor academic performance. 
        Immediate intervention recommended focusing on study habits and lifestyle factors.
        """)
    elif results['risk_level'] == 'Medium Risk':
        st.info("""
        **🟡 Medium Risk**: Some concerning patterns detected. 
        Proactive support and monitoring recommended to prevent decline.
        """)
    else:
        st.success("""
        **🟢 Low Risk**: Student shows positive patterns for academic success. 
        Continue current practices and maintain balance.
        """)

if __name__ == "__main__":
    main()