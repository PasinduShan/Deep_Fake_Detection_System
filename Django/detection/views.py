import os
import tensorflow as tf
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from tensorflow.keras.preprocessing import image
import numpy as np
import io

def homepage(request):
    return render(request, 'homepage.html')
  # Render the homepage template

def result(request):
    return render(request, 'result.html')
  # Render the homepage template



# Load the model (use the path to your saved .keras file)
model_path = os.path.join(settings.BASE_DIR, 'models', 'dfd_testcode.keras')
model = tf.keras.models.load_model(model_path)

result_image = "Unknown"

def predict_image(request):
    if request.method == 'POST' and 'fileToUpload' in request.FILES:
        # Get the uploaded file
        uploaded_file = request.FILES['fileToUpload']

        # Convert the InMemoryUploadedFile to a byte stream using io.BytesIO
        img_bytes = uploaded_file.read()
        img_stream = io.BytesIO(img_bytes)

        # Load and preprocess the image from the byte stream
        img = image.load_img(img_stream, target_size=(128, 128))  # Adjust size for your model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the pixel values

        # Make a prediction using the model
        prediction = model.predict(img_array)
        
        # Extract the probability and determine the result
        probability = float(prediction[0][0])  # Get probability for binary classification
        result_image = "Real" if probability > 0.5 else "Fake"

        # Render the result.html template with the prediction result and probability
        return render(request, 'result.html', {
            'predicted_class': result_image,
            'probability': (1- probability)*100,
            'threshold': probability,
        })

    return HttpResponse("Invalid request", status=400)



