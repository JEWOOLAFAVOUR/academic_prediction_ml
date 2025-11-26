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
    
    # Get categorical columns (fallback if not in feature_info)
    if 'categorical_columns' in feature_info:
        categorical_columns = feature_info['categorical_columns']
    else:
        # Define categorical columns based on data types and known categoricals
        categorical_columns = ['gender', 'part_time_job', 'diet_quality', 
                             'parental_education_level', 'internet_quality', 
                             'extracurricular_participation']
    
    # Encode categorical variables
    for col in categorical_columns:
        if col in processed_data.columns:
            if col in feature_encoders:
                try:
                    # Get unique values to check
                    unique_values = processed_data[col].unique()
                    encoder_classes = feature_encoders[col].classes_
                    
                    # Check for unseen categories
                    unseen_categories = set(unique_values) - set(encoder_classes)
                    
                    if unseen_categories:
                        print(f"Warning: Unseen categories in {col}: {unseen_categories}")
                        # Replace unseen categories with the first known category
                        for unseen_cat in unseen_categories:
                            processed_data[col] = processed_data[col].replace(unseen_cat, encoder_classes[0])
                    
                    # Now encode safely
                    processed_data[col] = feature_encoders[col].transform(processed_data[col])
                    
                except ValueError as e:
                    print(f"Error encoding {col}: {e}")
                    # Fallback to default encoding (0)
                    processed_data[col] = 0
            else:
                print(f"Warning: No encoder found for {col}, skipping")
    
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
    try:
        # Validate input data first
        is_valid, missing_cols, extra_cols = validate_input_data(data_df)
        
        if not is_valid:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create a copy of the data to avoid modifying original
        df_copy = data_df.copy()
        
        # Validate and clean categorical data
        categorical_mappings = {
            'gender': ['Male', 'Female'],
            'part_time_job': ['Yes', 'No'],
            'diet_quality': ['Poor', 'Fair', 'Good'],
            'parental_education_level': ['High School', 'Bachelor', 'Master'],
            'internet_quality': ['Poor', 'Average', 'Good'],
            'extracurricular_participation': ['Yes', 'No']
        }
        
        # Clean and validate categorical columns
        for col, valid_values in categorical_mappings.items():
            if col in df_copy.columns:
                # Convert to string and strip whitespace
                df_copy[col] = df_copy[col].astype(str).str.strip()
                
                # Check for invalid values
                invalid_mask = ~df_copy[col].isin(valid_values)
                if invalid_mask.any():
                    invalid_values = df_copy.loc[invalid_mask, col].unique()
                    print(f"Warning: Invalid values in {col}: {invalid_values}")
                    # Replace invalid values with the first valid value
                    df_copy.loc[invalid_mask, col] = valid_values[0]
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 
                          'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating']
        
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                
                # Fill any NaN values with median
                if df_copy[col].isna().any():
                    median_val = df_copy[col].median()
                    df_copy[col].fillna(median_val, inplace=True)
                    print(f"Warning: Filled NaN values in {col} with median: {median_val}")
        
        # Preprocess the cleaned data
        df_processed = preprocess_input_data(df_copy, components['feature_encoders'], components['feature_info'])
        
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
        results = data_df.copy()  # Use original data for display
        results['LR_Prediction'] = [classes[pred] for pred in lr_pred]
        results['LR_Confidence'] = [f"{max(prob):.1%}" for prob in lr_prob]
        results['SVM_Prediction'] = [classes[pred] for pred in svm_pred]
        results['SVM_Confidence'] = [f"{max(prob):.1%}" for prob in svm_prob]
        results['Model_Agreement'] = lr_pred == svm_pred
        results['Risk_Level'] = [get_risk_level(classes[pred]) for pred in lr_pred]
        
        return results
        
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")
        raise e

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
