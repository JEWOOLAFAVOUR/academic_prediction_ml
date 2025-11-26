# Academic Performance Prediction System

A comprehensive machine learning application for predicting student academic performance using study habits and lifestyle factors.

## 🎯 Features

- **Single Student Prediction**: Individual student performance analysis
- **Batch Prediction**: Upload CSV files for multiple student predictions
- **Model Analytics**: Detailed insights and feature importance analysis
- **Interactive Visualizations**: Charts and graphs for easy understanding
- **Risk Assessment**: Categorizes students by risk levels
- **Personalized Recommendations**: Actionable insights for improvement

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Required packages (see requirements.txt)

### Installation

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**

   ```bash
   streamlit run streamlit_app.py
   ```

   Or use the batch file on Windows:

   ```bash
   run_app.bat
   ```

3. **Open your browser** to http://localhost:8501

## 📊 Model Performance

- **Primary Model**: Logistic Regression (83.5% accuracy)
- **Secondary Model**: Support Vector Machine (78.6% accuracy)
- **Training Data**: 909 students with 14 behavioral features
- **Best At**: Identifying at-risk students with high precision

## 🔍 Key Features Analyzed

1. **Study Patterns**

   - Daily study hours
   - Class attendance percentage

2. **Digital Habits**

   - Social media usage hours
   - Netflix/streaming hours

3. **Lifestyle Factors**

   - Sleep hours per day
   - Exercise frequency
   - Diet quality

4. **Personal Context**
   - Age and gender
   - Mental health rating
   - Parental education level
   - Internet quality
   - Extracurricular participation

## 📈 Performance Categories

- **Good (Low Risk)**: Expected exam scores 80-100%
- **Average (Medium Risk)**: Expected exam scores 60-79%
- **Poor (High Risk)**: Expected exam scores below 60%

## 📁 Project Structure

```
academic_prediction/
│
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── run_app.bat                   # Windows launcher script
├── README.md                     # This file
├── academic-performance.ipynb    # Jupyter notebook for model development
│
└── model_and_others/            # Model files and data
    ├── logistic_regression_model.pkl
    ├── svm_model.pkl
    ├── scaler.pkl
    ├── target_label_encoder.pkl
    ├── feature_label_encoders.pkl
    ├── feature_info.pkl
    ├── model_metadata.json
    ├── feature_descriptions.json
    ├── prediction_functions.py
    ├── sample_upload_template.csv
    └── empty_template.csv
```

## 🎯 Usage Guide

### Single Student Prediction

1. Navigate to "👤 Single Student Prediction"
2. Fill in the student information form
3. Click "🔮 Predict Performance"
4. View results and personalized recommendations

### Batch Prediction

1. Navigate to "📊 Batch Prediction"
2. Download the sample CSV template
3. Fill in your student data following the template format
4. Upload your CSV file
5. Generate predictions and download results

### Model Analytics

1. Navigate to "📈 Model Analytics"
2. View model performance metrics
3. Analyze feature importance
4. Understand dataset distribution

## 💡 Key Insights

**Most Important Factors:**

1. **Social Media Hours** (Negative impact) - Reduce usage for better performance
2. **Study Hours per Day** (Positive impact) - Increase for better outcomes
3. **Diet Quality** (Positive impact) - Good nutrition supports learning
4. **Mental Health Rating** (Positive impact) - Well-being affects performance
5. **Extracurricular Participation** (Positive impact) - Balanced development

## ⚠️ Important Notes

- Predictions are probabilistic, not definitive
- Use as an early warning system, not final assessment
- Combine with qualitative observations
- Consider individual student circumstances
- Regular model updates recommended

## 🛠️ Technical Details

- **Backend**: Python, Scikit-learn
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Joblib

## 📞 Support

For technical issues or questions about the model, please check the "ℹ️ About" section in the application for detailed information about methodology and limitations.

## 🎓 Academic Use

This tool is designed to support educational success by:

- Providing early identification of at-risk students
- Supporting data-driven intervention strategies
- Helping with resource allocation for student support programs
- Enabling proactive academic counseling

**Remember**: This tool is designed to support, not replace, human judgment in educational settings.
