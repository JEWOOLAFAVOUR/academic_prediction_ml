import joblib
import pandas as pd
import numpy as np
import json

def load_all_models():
    """
    Load all trained models and preprocessing components
    Returns: Dictionary with all components
    """
    try:
        import os
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        components = {
            'lr_model': joblib.load(os.path.join(current_dir, 'logistic_regression_model.pkl')),
            'svm_model': joblib.load(os.path.join(current_dir, 'svm_model.pkl')),
            'scaler': joblib.load(os.path.join(current_dir, 'scaler.pkl')),
            'target_encoder': joblib.load(os.path.join(current_dir, 'target_label_encoder.pkl')),
            'feature_encoders': joblib.load(os.path.join(current_dir, 'feature_label_encoders.pkl')),
            'feature_info': joblib.load(os.path.join(current_dir, 'feature_info.pkl'))
        }
        
        # Load metadata
        with open(os.path.join(current_dir, 'model_metadata.json'), 'r') as f:
            components['metadata'] = json.load(f)
            
        return components
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def preprocess_input_data(data, feature_encoders, feature_info):
    """
    Preprocess input data for prediction
    
    Args:
        data: DataFrame with student features
        feature_encoders: Dictionary of label encoders for categorical features
        feature_info: Dictionary with feature information
    
    Returns:
        Preprocessed DataFrame ready for scaling
    """
    processed_data = data.copy()
    
    # Encode categorical variables
    categorical_columns = feature_info['categorical_columns']
    
    for col in categorical_columns:
        if col in feature_encoders and col in processed_data.columns:
            try:
                processed_data[col] = feature_encoders[col].transform(processed_data[col])
            except ValueError:
                # Handle unseen categories by using the most frequent category (0)
                print(f"Warning: Unseen category in {col}, using default encoding")
                processed_data[col] = 0
    
    return processed_data

def predict_single_student(student_data, components):
    """
    Predict performance for a single student
    
    Args:
        student_data: Dictionary with student features
        components: Dictionary with loaded models and preprocessors
    
    Returns:
        Dictionary with predictions and probabilities
    """
    # Convert to DataFrame
    df = pd.DataFrame([student_data])
    
    # Preprocess
    df_processed = preprocess_input_data(df, components['feature_encoders'], components['feature_info'])
    
    # Scale features
    df_scaled = components['scaler'].transform(df_processed)
    
    # Make predictions
    lr_pred = components['lr_model'].predict(df_scaled)[0]
    lr_prob = components['lr_model'].predict_proba(df_scaled)[0]
    
    svm_pred = components['svm_model'].predict(df_scaled)[0]
    svm_prob = components['svm_model'].predict_proba(df_scaled)[0]
    
    # Get class labels
    classes = components['target_encoder'].classes_
    
    return {
        'primary_prediction': {
            'model': 'Logistic Regression',
            'prediction': classes[lr_pred],
            'confidence': float(max(lr_prob)),
            'confidence_percentage': f"{max(lr_prob):.1%}",
            'probabilities': {classes[i]: float(prob) for i, prob in enumerate(lr_prob)}
        },
        'secondary_prediction': {
            'model': 'SVM',
            'prediction': classes[svm_pred],
            'confidence': float(max(svm_prob)),
            'confidence_percentage': f"{max(svm_prob):.1%}",
            'probabilities': {classes[i]: float(prob) for i, prob in enumerate(svm_prob)}
        },
        'agreement': lr_pred == svm_pred,
        'risk_level': get_risk_level(classes[lr_pred])
    }

def predict_batch_students(data_df, components):
    """
    Predict performance for multiple students from CSV
    
    Args:
        data_df: DataFrame with student features
        components: Dictionary with loaded models and preprocessors
    
    Returns:
        DataFrame with predictions
    """
    # Preprocess
    df_processed = preprocess_input_data(data_df, components['feature_encoders'], components['feature_info'])
    
    # Scale features
    df_scaled = components['scaler'].transform(df_processed)
    
    # Make predictions
    lr_pred = components['lr_model'].predict(df_scaled)
    lr_prob = components['lr_model'].predict_proba(df_scaled)
    
    svm_pred = components['svm_model'].predict(df_scaled)
    svm_prob = components['svm_model'].predict_proba(df_scaled)
    
    # Get class labels
    classes = components['target_encoder'].classes_
    
    # Create results DataFrame
    results = data_df.copy()
    results['LR_Prediction'] = [classes[pred] for pred in lr_pred]
    results['LR_Confidence'] = [f"{max(prob):.1%}" for prob in lr_prob]
    results['SVM_Prediction'] = [classes[pred] for pred in svm_pred]
    results['SVM_Confidence'] = [f"{max(prob):.1%}" for prob in svm_prob]
    results['Model_Agreement'] = lr_pred == svm_pred
    results['Risk_Level'] = [get_risk_level(classes[pred]) for pred in lr_pred]
    
    return results

def get_risk_level(prediction):
    """
    Convert prediction to risk level
    """
    risk_mapping = {
        'Good': 'Low Risk',
        'Average': 'Medium Risk', 
        'Poor': 'High Risk'
    }
    return risk_mapping.get(prediction, 'Unknown')

def get_feature_importance():
    """
    Return feature importance information for display
    """
    return [
        {'feature': 'social_media_hours', 'importance': 0.178, 'impact': 'negative'},
        {'feature': 'study_hours_per_day', 'importance': 0.153, 'impact': 'positive'},
        {'feature': 'diet_quality', 'importance': 0.134, 'impact': 'positive'},
        {'feature': 'mental_health_rating', 'importance': 0.085, 'impact': 'positive'},
        {'feature': 'extracurricular_participation', 'importance': 0.057, 'impact': 'positive'}
    ]

def validate_input_data(data_df):
    """
    Validate that input data has required columns
    
    Args:
        data_df: Input DataFrame
        
    Returns:
        Tuple (is_valid, missing_columns, extra_columns)
    """
    required_columns = set(['age', 'gender', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 'part_time_job', 'attendance_percentage', 'sleep_hours', 'diet_quality', 'exercise_frequency', 'parental_education_level', 'internet_quality', 'mental_health_rating', 'extracurricular_participation'])
    input_columns = set(data_df.columns)
    
    missing_columns = required_columns - input_columns
    extra_columns = input_columns - required_columns
    
    is_valid = len(missing_columns) == 0
    
    return is_valid, list(missing_columns), list(extra_columns)
