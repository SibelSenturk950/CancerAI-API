"""
CancerAI: Flask API Backend
Author: Sibel Senturk
Date: November 2025

This API provides endpoints for cancer survival prediction using trained ML models.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Load models at startup
print("Loading models...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
survival_model = joblib.load(os.path.join(MODEL_PATH, 'survival_prediction_model.pkl'))
drug_model = joblib.load(os.path.join(MODEL_PATH, 'drug_response_model.pkl'))
encoders = joblib.load(os.path.join(MODEL_PATH, 'label_encoders.pkl'))
print("✓ Models loaded successfully!")

def encode_features(data, encoders):
    """Encode patient data for prediction"""
    features = []
    
    # Numerical features
    features.append(float(data['age']))
    features.append(float(data['tumor_size_cm']))
    
    # Categorical features
    categorical_cols = ['sex', 'cancer_type', 'stage', 'grade', 'treatment', 'performance_status']
    
    for col in categorical_cols:
        try:
            encoded_value = encoders[col].transform([data[col]])[0]
            features.append(encoded_value)
        except KeyError:
            # Use default value if key not found
            encoded_value = 0
            features.append(encoded_value)
    
    return np.array(features).reshape(1, -1)

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'name': 'CancerAI API',
        'version': '1.0.0',
        'author': 'Sibel Senturk',
        'description': 'AI-powered cancer survival prediction API',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict/survival': 'Predict 5-year survival',
            '/predict/drug-response': 'Predict drug response'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'survival_model': 'Gradient Boosting (85.0% accuracy)',
        'drug_model': 'Random Forest (85.0% accuracy)'
    })

@app.route('/predict/survival', methods=['POST'])
def predict_survival():
    """
    Predict 5-year survival probability
    
    Expected JSON input:
    {
        "age": 58,
        "sex": "Female",
        "cancer_type": "Breast Cancer",
        "stage": "II",
        "grade": "Moderately Differentiated",
        "tumor_size_cm": 3.2,
        "treatment": "Surgery + Chemotherapy",
        "performance_status": "Good"
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'sex', 'cancer_type', 'stage', 'grade', 
                          'tumor_size_cm', 'treatment', 'performance_status']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Encode features
        features = encode_features(data, encoders)
        
        # Make prediction
        prediction = survival_model.predict(features)[0]
        probability = survival_model.predict_proba(features)[0]
        
        # Prepare response
        result = {
            'prediction': {
                'survived': bool(prediction),
                'survival_probability': float(probability[1]),
                'death_probability': float(probability[0]),
                'confidence': float(max(probability)),
                'risk_category': 'Low' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'High'
            },
            'patient': {
                'age': data['age'],
                'sex': data['sex'],
                'cancer_type': data['cancer_type'],
                'stage': data['stage'],
                'tumor_size_cm': data['tumor_size_cm']
            },
            'model': {
                'name': 'Gradient Boosting Classifier',
                'accuracy': '85.0%',
                'cross_validation': '85.5% (±4.9%)'
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/predict/drug-response', methods=['POST'])
def predict_drug_response():
    """
    Predict drug response probability
    
    Expected JSON input: Same as survival prediction
    """
    try:
        data = request.get_json()
        
        # Encode features
        features = encode_features(data, encoders)
        
        # Make prediction
        prediction = drug_model.predict(features)[0]
        probability = drug_model.predict_proba(features)[0]
        
        # Map to response categories
        response_types = ['No Response', 'Partial Response', 'Complete Response']
        if probability[1] > 0.8:
            response = 'Complete Response'
        elif probability[1] > 0.5:
            response = 'Partial Response'
        else:
            response = 'No Response'
        
        result = {
            'prediction': {
                'response_type': response,
                'response_probability': float(probability[1]),
                'confidence': float(max(probability))
            },
            'patient': {
                'cancer_type': data['cancer_type'],
                'stage': data['stage'],
                'treatment': data['treatment']
            },
            'model': {
                'name': 'Random Forest Classifier',
                'accuracy': '85.0%'
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/cancer-types', methods=['GET'])
def get_cancer_types():
    """Get list of supported cancer types"""
    cancer_types = [
        'Breast Cancer',
        'Lung Cancer',
        'Prostate Cancer',
        'Colorectal Cancer',
        'Melanoma',
        'Pancreatic Cancer',
        'Leukemia',
        'Ovarian Cancer'
    ]
    return jsonify({'cancer_types': cancer_types})

@app.route('/stages', methods=['GET'])
def get_stages():
    """Get list of cancer stages"""
    stages = ['I', 'II', 'III', 'IV']
    return jsonify({'stages': stages})

@app.route('/treatments', methods=['GET'])
def get_treatments():
    """Get list of treatment options"""
    treatments = [
        'Surgery',
        'Chemotherapy',
        'Radiation',
        'Immunotherapy',
        'Surgery + Chemotherapy',
        'Surgery + Radiation',
        'Chemotherapy + Radiation',
        'Multimodal'
    ]
    return jsonify({'treatments': treatments})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
