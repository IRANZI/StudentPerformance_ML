import numpy as np
import joblib
import os
from django.shortcuts import render
from tensorflow import keras
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'nn_model.keras')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def index(request):
    """
    Serve the main prediction page.
    """
    return render(request, 'performance/index.html')

@api_view(['POST'])
def predict_performance(request):
    data = request.data

    required = ['hours_studied', 'previous_scores', 'extracurricular', 'sleep_hours', 'sample_papers']
    if not all(k in data for k in required):
        return Response({"error": "Missing fields"}, status=400)

    try:
        hours_studied = float(data['hours_studied'])
        previous_scores = float(data['previous_scores'])
        extracurricular = 1 if data['extracurricular'] in [True, 'true', 'yes', 'Yes', 1] else 0
        sleep_hours = float(data['sleep_hours'])
        sample_papers = float(data['sample_papers'])

        # Logical Validation
        if hours_studied + sleep_hours > 24:
            return Response({"error": "Total study and sleep hours cannot exceed 24 hours."}, status=400)

        # Feature Engineering (must match training order)
        # 1. Study_Sleep_Interaction
        study_sleep_interaction = hours_studied * sleep_hours
        
        # 2. Study_Efficiency
        study_efficiency = (hours_studied * previous_scores) / 100
        
        # 3. Effort_Consistency
        effort_consistency = hours_studied * sample_papers

        features = np.array([[
            hours_studied,
            previous_scores,
            extracurricular,
            sleep_hours,
            sample_papers,
            study_sleep_interaction,
            study_efficiency,
            effort_consistency
        ]])
    except Exception as e:
        return Response({"error": f"Invalid numeric values: {str(e)}"}, status=400)

    features_scaled = scaler.transform(features)

    # Predict
    prediction_raw = model.predict(features_scaled, verbose=0)[0][0]
    
    # Clip prediction to meaningful range [0, 100]
    prediction_val = max(0, min(100, float(prediction_raw)))

    return Response({
        'predicted_performance_index': round(prediction_val, 2),
        'fairness_statement': "This prediction is based solely on objective academic metrics and study habits. No demographic or sensitive personal data is used, ensuring a bias-free assessment."
    })