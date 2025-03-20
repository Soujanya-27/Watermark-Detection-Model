import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load Trained Model
model = load_model(r"C:\Users\souja\Desktop\watermark_detection_project\models\watermark_detector.h5", 
                   custom_objects={"mse": MeanSquaredError()})

# Function to Preprocess Image for Testing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_height, original_width = img.shape[:2]  # Store original size
    img_resized = cv2.resize(img, (256, 256))  # Resize for model input
    img_resized = img_resized / 255.0  # Normalize
    return np.expand_dims(img_resized, axis=0), (original_width, original_height)

# Function to Recover Image
def recover_image(watermarked_img_path, output_path):
    watermarked_img, (original_width, original_height) = preprocess_image(watermarked_img_path)
    recovered_img = model.predict(watermarked_img)
    recovered_img = (recovered_img[0] * 255).astype(np.uint8)  # Convert back to 0-255 scale

    # Resize output back to original dimensions
    recovered_img_resized = cv2.resize(recovered_img, (original_width, original_height))
    cv2.imwrite(output_path, recovered_img_resized)
    print(f"✅ Recovered image saved: {output_path}")

# Automatically Select a Test Image
folder_path = "../dataset/watermarked"
images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

if not images:
    raise FileNotFoundError("❌ No images found in the dataset folder!")

# Pick the first image in the folder
test_image_path = os.path.join(folder_path, images[0])
output_path = "../dataset/recovered_sample.jpg"

print(f"✅ Using image: {test_image_path}")
recover_image(test_image_path, output_path)
