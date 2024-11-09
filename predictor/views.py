from django.shortcuts import render
from joblib import load
import os
import pandas as pd  # Import pandas to create DataFrame
from django.conf import settings

# Load the model
model_path = os.path.join(settings.BASE_DIR, "savedModels", "model.joblib")
model = load(model_path)

# Define feature names (adjust according to your model's features)
feature_names = ["precipitation", "snow_depth", "tmax", "tmin", "year", "month", "day", "country_encoded"]

def predictor(request):
    if request.method == 'POST':
        try:
            # Retrieve and convert input data from POST request
            precipitation = float(request.POST.get('precipitation', 0))
            snow_depth = float(request.POST.get('snow_depth', 0))
            tmax = float(request.POST.get('tmax', 0))
            tmin = float(request.POST.get('tmin', 0))
            year = int(request.POST.get('year', 0))
            month = int(request.POST.get('month', 0))
            day = int(request.POST.get('day', 0))
            country_encoded = int(request.POST.get('country_encoded', 0))

            # Create a DataFrame with feature names
            input_data = pd.DataFrame([[precipitation, snow_depth, tmax, tmin, year, month, day, country_encoded]], columns=feature_names)

            # Make a prediction using the model
            y_pred = model.predict(input_data)
            print(y_pred)
            days = range(1, 32)
            return render(request, 'main.html', {'prediction': y_pred[0]})

             
        except Exception as e:
            # Print error to the console and optionally render an error page
            print("Error during prediction:", e)
            return render(request, 'error.html', {'error': str(e)})

    return render(request, "main.html")
