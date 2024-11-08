import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import os

# Load the model
model_path = os.path.join(settings.BASE_DIR, 'gradient_boosting_model.pkl') 
model = joblib.load(model_path)

@csrf_exempt  # Disable CSRF just for testing; not recommended for production
def predict_temperature(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        
        # Create a DataFrame for the input data
        df = pd.DataFrame([data])

        # Ensure all features match the training set
        expected_features = ['precipitation', 'snow_depth', 'tmax', 'tmin', 'year', 'month', 'day', 'country_encoded']
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0  # Fill missing columns with default values

        # Predict with the loaded model
        prediction = model.predict(df)[0]

        # Return the result as JSON
        return JsonResponse({'predicted_temperature': prediction})
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=400)
