import os
import cv2
import numpy as np

# Define Image Dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Function to Add Synthetic Watermark using DCT
def add_watermark_dct(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(img_gray))
    watermark = np.zeros_like(dct)
    h, w = dct.shape
    watermark[h//2-10:h//2+10, w//2-10:w//2+10] = 50  # Adding a hidden pattern
    dct_watermarked = dct + watermark
    watermarked_img = cv2.idct(dct_watermarked)
    watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(watermarked_img, cv2.COLOR_GRAY2BGR)

# Generate Synthetic Dataset
def generate_synthetic_dataset(original_path, watermarked_path):
    if not os.path.exists(watermarked_path):
        os.makedirs(watermarked_path)
    
    for file in os.listdir(original_path):
        img_path = os.path.join(original_path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        watermarked_img = add_watermark_dct(img)
        cv2.imwrite(os.path.join(watermarked_path, file), watermarked_img)

# Paths
original_images_path = "../dataset/original/"
watermarked_images_path = "../dataset/watermarked/"

# Run the function
generate_synthetic_dataset(original_images_path, watermarked_images_path)
print("Synthetic watermarked images generated successfully!")
