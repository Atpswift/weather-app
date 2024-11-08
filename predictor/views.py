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

@csrf_exempt
def predict_temperature(request):
    if request.method == 'POST':
        try:
            # Print the request body to verify it in Heroku logs
            print("Request body:", request.body)
            
            data = json.loads(request.body)
            
            # Further debug statements to inspect the parsed data
            print("Parsed data:", data)

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
        
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        
        except Exception as e:
            print("Error:", e)
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST requests are allowed'}, status=400)
