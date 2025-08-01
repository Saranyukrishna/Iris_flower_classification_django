from django.shortcuts import render
import joblib
import numpy as np

# Load model once
model = joblib.load('iris_model.joblib')

def index(request):
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.POST['sepal_length']),
                float(request.POST['sepal_width']),
                float(request.POST['petal_length']),
                float(request.POST['petal_width'])
            ]
            result = model.predict([features])[0]
            classes = ['Setosa', 'Versicolor', 'Virginica']
            prediction = classes[result]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render(request, 'iris_app/index.html', {'prediction': prediction})
