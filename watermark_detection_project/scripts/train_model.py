import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os


# Define Image Dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Define Model Architecture
def build_model():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    
    outputs = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

# Load and Preprocess Dataset
def load_images(path):
    images = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        images.append(img / 255.0)
    return np.array(images)

# Paths (Update with actual dataset paths)
original_images_path = "../dataset/original/"
watermarked_images_path = "../dataset/watermarked/"

original_images = load_images(original_images_path)
watermarked_images = load_images(watermarked_images_path)

# Train the Model
model = build_model()
model.summary()

epochs = 50
batch_size = 8
model.fit(watermarked_images, original_images, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Save the Model
model.save("../models/watermark_detector.h5")
print("Model training complete and saved successfully!")


