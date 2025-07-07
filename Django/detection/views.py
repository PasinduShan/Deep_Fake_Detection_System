import os
import tensorflow as tf
from django.conf import settings
from django.http import JsonResponse, HttpResponse

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import cv2

from django.shortcuts import render, redirect

from django.core.files.storage import FileSystemStorage

from torch.utils.data import Dataset, DataLoader
from torch import nn

from facenet_pytorch import MTCNN, InceptionResnetV1


import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from PIL import Image
import tempfile
from PIL import Image  # Import the Image module from PIL

def index(request):
    return render(request, 'index.html')
  # Render the homepage template

def p_image(request):
    return render(request, 'image.html')
  # Render the image upload template

def i_result(request):
    return render(request, 'i_result.html')
  # Render the image result template

def p_video(request):
    return render(request, 'p_video.html')
    # Render the video upload template

def vi_result(request):
    return render(request, 'vi_result.html')
  # Render the video result template

def p_voice(request):
    return render(request, 'voice.html')
    # Voice upload

def test(request):
    return render(request, 'test.html')
  # Render the test template

def vo_result(request):
    return render(request, 'vo_result.html')
    # Render the voice detection result template

def p_super(request):
    return render(request, 'p_super.html')
  # Render the super resolution template

# Load the image detection model
IMAGE_MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'dfd_testcode.keras')
image_model = load_model(IMAGE_MODEL_PATH)

# Load the voice detection model
VOICE_MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'deep_voice_detection_model.h5')
model = load_model(VOICE_MODEL_PATH)

# Initialize label encoder
encoder = LabelEncoder()
encoder.classes_ = np.array(['real', 'fake'])  # Set the classes based on your model's output

# Define the image size (should match the size used during training)
IMAGE_SIZE = (128, 128)

def preprocess_image(image_path, image_size):
    img = cv2.imread(image_path)  # Read the image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, image_size)  # Resize to the input size
    img = img / 255.0  # Scale pixel values to [0, 1]
    return img

# Create Saliency Map
def compute_saliency_map(image_model, input_tensor):
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)  # Watch the input tensor
        predictions = image_model(input_tensor)  # Make predictions
        class_idx = tf.argmax(predictions[0])  # Get the predicted class
        loss = predictions[:, class_idx]  # Compute the loss for the predicted class

    # Calculate gradients of the loss w.r.t. the input
    grads = tape.gradient(loss, input_tensor)

    # Take the absolute maximum across the color channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]

    # Normalize the saliency map to [0, 1]
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_max(saliency))
    return saliency

# Function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str



def predict_image(request):
    if request.method == 'POST' and 'imageToUpload' in request.FILES:
        # Get the uploaded file
        uploaded_file = request.FILES['imageToUpload']

        # Convert the InMemoryUploadedFile to a byte stream using io.BytesIO
        img_bytes = uploaded_file.read()
        img_stream = io.BytesIO(img_bytes)

        # Load and preprocess the image from the byte stream
        img = image.load_img(img_stream, target_size=(128, 128))  # Adjust size for your model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the pixel values

        # Use MTCNN to detect faces
        mtcnn = MTCNN()
        face = mtcnn.detect(cv2.cvtColor(np.uint8(img_array[0] * 255), cv2.COLOR_RGB2BGR))

        # Check if no face is detected
        if face[0] is None:
            return render(request, 'image.html', {
                'error_message': 'Face Not Detected'
            })

        # Make a prediction using the model
        prediction = image_model.predict(img_array)

        # Extract the probability and determine the result
        probability = float(prediction[0][0])  # Get probability for binary classification
        result_image = "Real" if probability > 0.5 else "Fake"

        # Generate the saliency map
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_array[0], axis=0), dtype=tf.float32)
        saliency_map = compute_saliency_map(image_model, input_tensor)

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
        return render(request, 'i_result.html', {
            'predicted_class': result_image,
            'probability': (1 - probability) * 100,
            'threshold': probability,
            'original_image': original_img_base64,
            'saliency_map': saliency_map_base64,
        })

    return HttpResponse("Invalid request", status=400)

# Function to convert OpenCV frame to base64-encoded string
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    img_bytes = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_bytes}"




def process_voice(p_voice):
    try:
        # Save uploaded file temporarily
        temp_path = 'temp_audio.wav'
        with open(temp_path, 'wb+') as destination:
            for chunk in p_voice.chunks():
                destination.write(chunk)
        
        # Extract features
        audio, sr = librosa.load(temp_path)
        features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features = np.mean(features.T, axis=0)
        features = np.expand_dims(features, axis=0)
        
        # Make prediction
        prediction = model.predict(features)
        result = encoder.inverse_transform([np.argmax(prediction)])
        
        # Clean up temporary file
        import os
        os.remove(temp_path)
        
        return {
            'status': 'success',
            'prediction': result[0],
            'confidence': float(np.max(prediction))
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

def predict_voice(request):
    if request.method == 'POST' and 'audioToUpload' in request.FILES:
        uploaded_file = request.FILES['audioToUpload']
        result = process_voice(uploaded_file)
        
        if result['status'] == 'success':
            return render(request, 'vo_result.html', {
                'predicted_class': result['prediction'],
                'confidence': result['confidence'] * 100
            })
        else:
            return render(request, 'voice.html', {
                'error_message': result['message']
            })
            
    return HttpResponse("Invalid request", status=400)

class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepFakeDetector, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        for param in self.facenet.parameters():
            param.requires_grad = False
            
        self.lstm = nn.LSTM(512, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        embeddings = self.facenet(x)
        embeddings = embeddings.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = torch.mean(lstm_out, dim=1)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

def predict_video(request):
    if request.method == 'POST' and 'videoToUpload' in request.FILES:
        try:
            uploaded_file = request.FILES['videoToUpload']
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                for chunk in uploaded_file.chunks():
                    temp_video.write(chunk)
                temp_video_path = temp_video.name

            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = DeepFakeDetector()
            model_path = os.path.join(settings.BASE_DIR, 'models', 'video_detector.pth')
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model.to(device)

            # Process video and get frames
            mtcnn = MTCNN(keep_all=True, device=device)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])

            # Extract frames and faces with random selection
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_process = min(150, total_frames)  # Limit to 150 frames
            
            # Generate random frame indices for sampling
            random_frame_indices = np.random.choice(
                total_frames, 
                size=frames_to_process, 
                replace=False if total_frames >= frames_to_process else True
            )
            random_frame_indices.sort()  # Sort to process frames in order
            
            frames = []
            original_frames = []
            processed_faces = []
            current_frame = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_frame in random_frame_indices:
                    original_frames.append(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Detect faces
                    boxes, _ = mtcnn.detect(frame_pil)
                    if boxes is not None:
                        for box in boxes:
                            try:
                                # Extract and process face
                                x1, y1, x2, y2 = map(int, box)
                                face = frame_rgb[y1:y2, x1:x2]
                                face = cv2.resize(face, (112, 112))
                                processed_faces.append(face)
                                
                                # Convert to tensor
                                face_tensor = transforms.ToTensor()(face)
                                face_tensor = transforms.Resize((160, 160))(face_tensor)
                                face_tensor = transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                )(face_tensor)
                                
                                frames.append(face_tensor)
                            except Exception as e:
                                print(f"Error processing face: {e}")
                                continue
                
                current_frame += 1

            cap.release()

            if len(frames) == 0:
                return render(request, 'p_video.html', {
                    'error_message': 'No faces detected in video'
                })

            # Change the n_display_frames variable from 5 to 12
            # Randomly select display frames
            n_display_frames = 12  # Changed from 5 to 12 frames to display
            if len(original_frames) > n_display_frames:
                display_indices = np.random.choice(
                    len(original_frames), 
                    size=n_display_frames, 
                    replace=False
                )
            else:
                display_indices = range(len(original_frames))
            
            original_frames_b64 = []
            processed_faces_b64 = []
            
            for idx in display_indices:
                if idx < len(original_frames):
                    orig_frame = cv2.cvtColor(original_frames[idx], cv2.COLOR_BGR2RGB)
                    original_frames_b64.append(frame_to_base64(orig_frame))
                
                if idx < len(processed_faces):
                    proc_face = cv2.cvtColor(processed_faces[idx], cv2.COLOR_BGR2RGB)
                    processed_faces_b64.append(frame_to_base64(proc_face))

            # Process each frame through image model
            frame_results = []
            for idx in display_indices:
                if idx < len(original_frames):
                    orig_frame = original_frames[idx]
                    # Preprocess frame for image model
                    frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (128, 128))
                    frame_array = np.expand_dims(frame_resized / 255.0, axis=0)
                    
                    # Get image model prediction
                    img_prediction = image_model.predict(frame_array)
                    probability = float(img_prediction[0][0])
                    
                    frame_results.append({
                        'frame': frame_to_base64(frame_rgb),
                        'face': frame_to_base64(cv2.cvtColor(processed_faces[idx], cv2.COLOR_BGR2RGB)) if idx < len(processed_faces) else None,
                        'prediction': 'Real' if probability > 0.5 else 'Fake',
                        'confidence': (1 - probability) * 100 if probability > 0.5 else probability * 100
                    })

            # Prepare model input
            frames_tensor = torch.stack(frames[:10]).unsqueeze(0).to(device)  # Take first 10 frames

            # Make prediction
            with torch.no_grad():
                outputs = model(frames_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                pred_prob, prediction = torch.max(probabilities, dim=1)

            # Clean up temporary file
            os.unlink(temp_video_path)

            # Prepare results
            result = {
                'prediction': 'Real' if prediction.item() == 1 else 'Fake',
                'confidence': pred_prob.item() * 100,
                'fake_probability': probabilities[0][0].item() * 100,
                'real_probability': probabilities[0][1].item() * 100,
                'original_frames': original_frames_b64,
                'processed_faces': processed_faces_b64
            }

            # Update the result dictionary
            result.update({
                'frame_results': frame_results,
            })

            return render(request, 'multi_result.html', result)

        except Exception as e:
            return HttpResponse(f"<script>alert('Error processing video: {str(e)}'); window.location.href = '/p_super';</script>")

    return HttpResponse("Invalid request", status=400)

