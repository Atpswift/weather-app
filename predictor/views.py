import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Load the model
model = joblib.load('gradient_boosting_model.pkl')

@csrf_exempt  # Disable CSRF just for testing; not recommended for production
def predict_temperature(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        
        # Create a DataFrame for the input data
        df = pd.DataFrame([data])

        # Ensure all features match the training set
        expected_features = ['year', 'month', 'day', 'precipitation', 'tmin', 'tmax', 'snow_depth', 'country_encoded']
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0  # Fill missing columns with default values

        # Predict with the loaded model
        prediction = model.predict(df)[0]

        # Return the result as JSON
        return JsonResponse({'predicted_temperature': prediction})
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=400)
