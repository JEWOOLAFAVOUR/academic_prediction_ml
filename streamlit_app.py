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

# Beautiful Dark Mode CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Dark Theme Global Styles */
    .main {
        padding-top: 2rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        min-height: 100vh;
        color: #f1f5f9;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    }
    
    /* Override Streamlit's default text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #e2e8f0 !important;
    }
    
    /* Headers with Glow Effect */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(96, 165, 250, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(96, 165, 250, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(167, 139, 250, 0.8)); }
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
        padding-bottom: 0.5rem;
    }
    
    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(96, 165, 250, 0.2);
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
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
        height: 3px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 50px rgba(96, 165, 250, 0.3);
        border-color: rgba(96, 165, 250, 0.5);
    }
    
    .metric-card h3 {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .metric-card h2 {
        color: #f1f5f9;
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card p {
        color: #cbd5e1;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Prediction Cards with Neon Effect */
    .prediction-card {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(167, 139, 250, 0.3);
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #10b981, #06d6a0, #118ab2);
        border-radius: 0 0 0 20px;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(167, 139, 250, 0.2);
    }
    
    .prediction-card h3 {
        color: #a78bfa;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }
    
    .prediction-card h2 {
        color: #f1f5f9;
        font-weight: 800;
        font-size: 2.5rem;
        margin: 1rem 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.5);
    }
    
    .prediction-card p {
        color: #cbd5e1;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    
    /* Form Sections with Dark Glass Effect */
    .form-section {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }
    
    .form-section:hover {
        border-color: rgba(96, 165, 250, 0.4);
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
    }
    
    .form-section h4 {
        color: #60a5fa;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Dark Theme Form Controls */
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 2px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        color: #f1f5f9 !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.2) !important;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 2px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.2) !important;
    }
    
    /* Slider Dark Theme */
    .stSlider > div > div > div {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 10px !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #60a5fa, #a78bfa) !important;
    }
    
    .stSlider > div > div > div[role="slider"] {
        background: #f1f5f9 !important;
        border: 3px solid #60a5fa !important;
        box-shadow: 0 4px 15px rgba(96, 165, 250, 0.4) !important;
    }
    
    /* Form Labels */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Button with Neon Effect */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Sidebar Dark Theme */
    .stSidebar {
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(96, 165, 250, 0.2) !important;
    }
    
    .stSidebar .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        color: #f1f5f9 !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #60a5fa !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    .stSidebar .stMarkdown {
        color: #cbd5e1 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
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
    # Epic Dark Theme Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; padding: 2rem 0;">
        <h1 class="main-header">🎓 Academic Performance Predictor</h1>
        <p style="font-size: 1.4rem; color: #94a3b8; font-family: 'Inter', sans-serif; 
                  font-weight: 600; margin-top: 1rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
            🤖 AI-Powered Student Success Analytics & Future Prediction
        </p>
        <div style="width: 100px; height: 4px; 
                    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6); 
                    margin: 1.5rem auto; border-radius: 2px;
                    box-shadow: 0 2px 10px rgba(96, 165, 250, 0.5);"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models..."):
            components, success = initialize_models()
            if success:
                st.session_state.components = components
                st.session_state.models_loaded = True
                st.success("✅ Models loaded successfully!")
            else:
                st.error("❌ Failed to load models.")
                st.stop()
    
    # Epic Dark Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
            <h2 style="color: #60a5fa; font-weight: 800; font-size: 1.5rem; margin: 0;">
                Academic Prediction
            </h2>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                Explore AI Predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.selectbox(
            "🎯 Choose Your Destination:",
            ["🏠 Home", "👤 Single Prediction", "📊 Batch Prediction", "📈 Analytics", "ℹ️ About"],
            help="Navigate through different sections of the app"
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; 
                    background: rgba(96, 165, 250, 0.1); border-radius: 10px;
                    border: 1px solid rgba(96, 165, 250, 0.2);">
            <h4 style="color: #60a5fa; margin: 0 0 0.5rem 0;">💡 Quick Info</h4>
            <p style="color: #cbd5e1; font-size: 0.8rem; margin: 0;">
                This AI system uses advanced ML algorithms to predict student performance with 83.5% accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Route to pages
    if page == "🏠 Home":
        home_page()
    elif page == "👤 Single Prediction":
        single_prediction_page()
    elif page == "📊 Batch Prediction":
        batch_prediction_page()
    elif page == "📈 Analytics":
        analytics_page()
    elif page == "ℹ️ About":
        about_page()

def home_page():
    """Home page with overview"""
    st.markdown('<h2 class="sub-header">Welcome to Academic Performance Predictor</h2>', unsafe_allow_html=True)
    
    # Simple metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>83.5%</h2>
            <p>Logistic Regression Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Models</h3>
            <h2>2</h2>
            <p>Machine Learning Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Features</h3>
            <h2>14</h2>
            <p>Student Attributes</p>
        </div>
        """, unsafe_allow_html=True)

def single_prediction_page():
    """Single student prediction page"""
    st.markdown('<h2 class="sub-header">👤 Single Student Prediction</h2>', unsafe_allow_html=True)
    
    # Load feature descriptions
    with open('model_and_others/feature_descriptions.json', 'r') as f:
        feature_descriptions = json.load(f)
    
    # Simple form
    with st.form("student_form"):
        st.markdown("### Enter Student Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Personal Information")
            age = st.number_input("Age", min_value=17, max_value=24, value=20)
            gender = st.selectbox("Gender", ["Male", "Female"])
            part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])
            parental_education = st.selectbox("Parental Education Level", ["High School", "Bachelor", "Master"])
            extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
        
        with col2:
            st.markdown("#### Study & Screen Time")
            study_hours = st.slider("Study Hours per Day", min_value=0.0, max_value=8.3, value=4.0, step=0.1)
            social_media_hours = st.slider("Social Media Hours per Day", min_value=0.0, max_value=7.2, value=2.5, step=0.1)
            netflix_hours = st.slider("Netflix/Streaming Hours per Day", min_value=0.0, max_value=5.4, value=1.8, step=0.1)
            attendance = st.slider("Attendance Percentage", min_value=56.0, max_value=100.0, value=84.0, step=1.0)
        
        with col3:
            st.markdown("#### Health & Lifestyle")
            sleep_hours = st.slider("Sleep Hours per Day", min_value=3.2, max_value=10.0, value=6.5, step=0.1)
            exercise_frequency = st.selectbox("Exercise Frequency (per week)", [0, 1, 2, 3, 4, 5, 6], index=3)
            diet_quality = st.selectbox("Diet Quality", ["Poor", "Fair", "Good"], index=1)
            mental_health = st.slider("Mental Health Rating", min_value=1, max_value=10, value=5)
            internet_quality = st.selectbox("Internet Quality", ["Poor", "Average", "Good"], index=1)
        
        submitted = st.form_submit_button("🔮 Predict Academic Performance")
    
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
                st.markdown("### Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    primary = results['primary_prediction']
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🎯 Primary Prediction (Logistic Regression)</h3>
                        <h2>{primary['prediction']}</h2>
                        <p><strong>Confidence:</strong> {primary['confidence_percentage']}</p>
                        <p><strong>Risk Level:</strong> {results['risk_level']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
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
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def batch_prediction_page():
    """Batch prediction page"""
    st.markdown('<h2 class="sub-header">📊 Batch Prediction</h2>', unsafe_allow_html=True)
    
    # Instructions and template
    st.markdown("""
    <div class="form-section">
        <h4>📋 Instructions</h4>
        <p>Upload a CSV file with student data. Make sure your CSV contains all required columns with exact names and valid values.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample template
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    with col2:
        st.markdown("### 📄 Download Sample Template")
        # Create sample data
        sample_data = {
            'age': [20, 21, 19],
            'gender': ['Male', 'Female', 'Male'],
            'study_hours_per_day': [5.0, 6.2, 4.1],
            'social_media_hours': [2.5, 1.8, 3.2],
            'netflix_hours': [1.5, 0.8, 2.1],
            'part_time_job': ['No', 'Yes', 'No'],
            'attendance_percentage': [85.0, 92.5, 78.3],
            'sleep_hours': [7.2, 6.8, 7.5],
            'diet_quality': ['Good', 'Fair', 'Poor'],
            'exercise_frequency': [4, 3, 2],
            'parental_education_level': ['Bachelor', 'Master', 'High School'],
            'internet_quality': ['Good', 'Average', 'Poor'],
            'mental_health_rating': [7, 8, 6],
            'extracurricular_participation': ['Yes', 'No', 'Yes']
        }
        sample_df = pd.DataFrame(sample_data)
        csv_data = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_data,
            file_name="sample_student_data.csv",
            mime="text/csv",
            help="Download this template and fill with your data"
        )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head())
            
            # Validate data
            is_valid, missing_cols, extra_cols = validate_input_data(df)
            
            if not is_valid:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV has all required columns. Download the sample template above.")
            else:
                if extra_cols:
                    st.warning(f"⚠️ Extra columns found (will be ignored): {', '.join(extra_cols)}")
                
                st.success("✅ Data format looks good!")
                
                if st.button("🚀 Run Batch Predictions", type="primary"):
                    with st.spinner("🤖 Processing batch predictions..."):
                        try:
                            results = predict_batch_students(df, st.session_state.components)
                            
                            st.success("🎉 Batch predictions completed!")
                            
                            # Display results
                            st.markdown("### 📈 Prediction Results")
                            st.dataframe(results)
                            
                            # Download results
                            results_csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=results_csv,
                                file_name="prediction_results.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                good_count = sum(results['LR_Prediction'] == 'Good')
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>🟢 Good Performance</h3>
                                    <h2>{good_count}</h2>
                                    <p>{good_count/len(results)*100:.1f}% of students</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                avg_count = sum(results['LR_Prediction'] == 'Average')
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>🟡 Average Performance</h3>
                                    <h2>{avg_count}</h2>
                                    <p>{avg_count/len(results)*100:.1f}% of students</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                poor_count = sum(results['LR_Prediction'] == 'Poor')
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>🔴 At-Risk Students</h3>
                                    <h2>{poor_count}</h2>
                                    <p>{poor_count/len(results)*100:.1f}% need support</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"❌ Batch prediction failed: {str(e)}")
                            st.info("💡 Please check your data format and try again. Make sure categorical values match the expected format.")
                            
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")

def analytics_page():
    """Analytics and insights page"""
    st.markdown('<h2 class="sub-header">📈 Analytics</h2>', unsafe_allow_html=True)
    st.info("Analytics features will be added here.")

def about_page():
    """About page"""
    st.markdown('<h2 class="sub-header">ℹ️ About</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Academic Performance Prediction System
    
    This application uses machine learning to predict student academic performance based on various factors.
    
    ### Features:
    - **Single Student Prediction**: Get predictions for individual students
    - **Batch Processing**: Upload CSV files for multiple predictions
    - **Dual Model Approach**: Uses both Logistic Regression and SVM
    - **Risk Assessment**: Identifies at-risk students early
    
    ### Models:
    - **Logistic Regression**: Primary model with 83.5% accuracy
    - **Support Vector Machine**: Secondary model for validation
    
    ### Input Features:
    - Age, Gender, Study habits
    - Screen time (Social media, Netflix)
    - Health factors (Sleep, Exercise, Diet)
    - Academic factors (Attendance, Parental education)
    - Lifestyle factors (Part-time job, Extracurriculars)
    """)

if __name__ == "__main__":
    main()