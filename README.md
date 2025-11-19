# CancerAI API - Deployment Guide

**Flask API for Cancer Survival Prediction**

This API provides real-time cancer survival predictions using trained machine learning models.

---

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py

# API will be available at http://localhost:5000
```

### Test API

```bash
curl -X POST http://localhost:5000/predict/survival \
  -H "Content-Type: application/json" \
  -d '{
    "age": 58,
    "sex": "Female",
    "cancer_type": "Breast Cancer",
    "stage": "II",
    "grade": "Moderately Differentiated",
    "tumor_size_cm": 3.2,
    "treatment": "Surgery + Chemotherapy",
    "performance_status": "Good"
  }'
```

---

## 📡 API Endpoints

### GET /
API information and available endpoints

### GET /health
Health check - returns model status

### POST /predict/survival
Predict 5-year survival probability

**Request Body:**
```json
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
```

**Response:**
```json
{
  "prediction": {
    "survived": true,
    "survival_probability": 0.997,
    "death_probability": 0.003,
    "confidence": 0.997,
    "risk_category": "Low"
  },
  "patient": {
    "age": 58,
    "sex": "Female",
    "cancer_type": "Breast Cancer",
    "stage": "II",
    "tumor_size_cm": 3.2
  },
  "model": {
    "name": "Gradient Boosting Classifier",
    "accuracy": "85.0%",
    "cross_validation": "85.5% (±4.9%)"
  }
}
```

### POST /predict/drug-response
Predict drug response probability

### GET /cancer-types
Get list of supported cancer types

### GET /stages
Get list of cancer stages

### GET /treatments
Get list of treatment options

---

## 🌐 Deployment Options

### Option 1: Heroku (Recommended)

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create cancerai-api

# Deploy
git init
git add .
git commit -m "Initial commit"
git push heroku master

# Your API will be at: https://cancerai-api.herokuapp.com
```

### Option 2: Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize
railway init

# Deploy
railway up

# Your API will be at: https://cancerai-api.railway.app
```

### Option 3: Render

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect GitHub repository
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Deploy

### Option 4: PythonAnywhere

1. Go to https://www.pythonanywhere.com
2. Create account (free tier available)
3. Upload files
4. Configure WSGI
5. Your API will be at: https://yourusername.pythonanywhere.com

---

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set custom port
export PORT=5000

# Optional: Enable debug mode (development only)
export FLASK_DEBUG=1
```

### CORS Configuration

The API allows cross-origin requests from any domain. To restrict:

```python
# In app.py
CORS(app, origins=["https://yourdomain.com"])
```

---

## 📦 Files

```
CancerAI-API/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku configuration
├── README.md             # This file
└── models/
    ├── survival_prediction_model.pkl
    ├── drug_response_model.pkl
    └── label_encoders.pkl
```

---

## 🧪 Testing

### Python Test

```python
import requests

url = "http://localhost:5000/predict/survival"
data = {
    "age": 58,
    "sex": "Female",
    "cancer_type": "Breast Cancer",
    "stage": "II",
    "grade": "Moderately Differentiated",
    "tumor_size_cm": 3.2,
    "treatment": "Surgery + Chemotherapy",
    "performance_status": "Good"
}

response = requests.post(url, json=data)
print(response.json())
```

### JavaScript Test

```javascript
const API_URL = 'http://localhost:5000';

fetch(`${API_URL}/predict/survival`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    age: 58,
    sex: 'Female',
    cancer_type: 'Breast Cancer',
    stage: 'II',
    grade: 'Moderately Differentiated',
    tumor_size_cm: 3.2,
    treatment: 'Surgery + Chemotherapy',
    performance_status: 'Good'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 🔒 Security Notes

- This is a demo API for educational purposes
- For production use:
  - Add authentication (API keys, JWT)
  - Implement rate limiting
  - Add input validation and sanitization
  - Use HTTPS only
  - Monitor and log requests

---

## 📝 License

MIT License - See LICENSE file

---

## 👤 Author

**Sibel Senturk**  
Master's Thesis Project  
North American University  
November 2025

---

## 📧 Support

For issues or questions:
- GitHub: [@sibelsturk](https://github.com/sibelsturk)
- Email: sibel.senturk@example.com
