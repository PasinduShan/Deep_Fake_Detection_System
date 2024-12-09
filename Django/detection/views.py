import os
import tensorflow as tf
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import cv2

def homepage(request):
    return render(request, 'homepage.html')
  # Render the homepage template

def result(request):
    return render(request, 'result.html')
  # Render the homepage template

def test(request):
    return render(request, 'test.html')
  # Render the homepage template
# Load the model (use the path to your saved .keras file)
model_path = os.path.join(settings.BASE_DIR, 'models', 'dfd_testcode.keras')
model = tf.keras.models.load_model(model_path)

# Define the image size (should match the size used during training)
IMAGE_SIZE = (128, 128)

def preprocess_image(image_path, image_size):
    img = cv2.imread(image_path)  # Read the image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, image_size)  # Resize to the input size
    img = img / 255.0  # Scale pixel values to [0, 1]
    return img

# Create Saliency Map
def compute_saliency_map(model, input_tensor):
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)  # Watch the input tensor
        predictions = model(input_tensor)  # Make predictions
        class_idx = tf.argmax(predictions[0])  # Get the predicted class
        loss = predictions[:, class_idx]  # Compute the loss for the predicted class

    # Calculate gradients of the loss w.r.t. the input
    grads = tape.gradient(loss, input_tensor)

    # Take the absolute maximum across the color channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]

    # Normalize the saliency map to [0, 1]
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))
    return saliency

# Function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

from PIL import Image  # Import the Image module from PIL




from PIL import Image  # Import the Image module from PIL

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

        # Generate the saliency map
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_array[0], axis=0), dtype=tf.float32)
        saliency_map = compute_saliency_map(model, input_tensor)

        # Increase saliency intensity by applying a scaling factor
        saliency_map_np = saliency_map.numpy()  # Convert saliency map to numpy array
        saliency_map_np = saliency_map_np * 3  # Scale by 3x (or adjust factor for more visibility)
        saliency_map_np = np.clip(saliency_map_np, 0, 1)  # Ensure values stay in range [0, 1]

        # Apply a higher contrast color map to make spots stand out more
        saliency_map_img = np.uint8(saliency_map_np * 255)  # Scale to [0, 255] for image display
        saliency_map_img = cv2.applyColorMap(saliency_map_img, cv2.COLORMAP_JET)  # Use a high-contrast colormap

        # Combine original image and saliency map overlay with increased alpha blending
        combined_image = cv2.addWeighted(np.uint8(img_array[0] * 255), 0.6, saliency_map_img, 0.4, 0)  # Increase alpha

        # Convert both images to base64 for HTML embedding
        original_img_base64 = image_to_base64(img)
        saliency_map_base64 = image_to_base64(Image.fromarray(combined_image))  # Convert combined image to base64

        # Render the result.html template with the prediction result, images, and probability
        return render(request, 'result.html', {
            'predicted_class': result_image,
            'probability': (1 - probability) * 100,
            'threshold': probability,
            'original_image': original_img_base64,
            'saliency_map': saliency_map_base64,
        })

    return HttpResponse("Invalid request", status=400)

