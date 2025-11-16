#!/usr/bin/env python3
"""
Train Facial Recognition Model
Extracts features from team member images and trains a Random Forest classifier.
Saves the trained model to ../models/facial_recognition_model.pkl
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Feature extraction functions
def extract_histogram_features(image, bins=32):
    """Extract color histogram features from image."""
    features = []
    for i in range(3):  # For each channel (B, G, R)
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        features.extend(hist.flatten())
    return np.array(features)

def extract_hog_features(image):
    """Extract HOG (Histogram of Oriented Gradients) features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    features = hog.compute(cv2.resize(gray, (64, 64)))
    return features.flatten()

def extract_color_moments(image):
    """Extract color moment features."""
    features = []
    for i in range(3):  # For each channel
        channel = image[:, :, i].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-6)
        features.extend([mean, std, skewness])
    return np.array(features)

def extract_lbp_features(image, num_points=8, radius=1):
    """Extract Local Binary Pattern (LBP) features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def lbp_computation(img, p=num_points, r=radius):
        rows, cols = img.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)
        
        for i in range(r, rows - r):
            for j in range(r, cols - r):
                center = img[i, j]
                lbp_val = 0
                
                for k in range(p):
                    angle = 2 * np.pi * k / p
                    x = int(r * np.cos(angle))
                    y = int(r * np.sin(angle))
                    neighbor = img[i + y, j + x]
                    lbp_val = (lbp_val << 1) | (1 if neighbor >= center else 0)
                
                lbp[i, j] = lbp_val
        
        return lbp
    
    lbp = lbp_computation(gray, num_points, radius)
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    return hist.flatten()

def extract_all_features(image):
    """Extract all features from an image."""
    features = []
    features.extend(extract_histogram_features(image))
    features.extend(extract_hog_features(image))
    features.extend(extract_color_moments(image))
    features.extend(extract_lbp_features(image))
    return np.array(features)

# Augmentation functions
def rotate_image(image, angle=15):
    """Rotate image by specified angle."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def flip_image(image, axis=1):
    """Flip image horizontally or vertically."""
    return cv2.flip(image, axis)

def convert_to_grayscale(image):
    """Convert image to grayscale then back to BGR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def main():
    print("="*70)
    print("FACIAL RECOGNITION MODEL TRAINING")
    print("="*70)
    
    # Paths
    raw_images_path = "../data/raw/images"
    models_path = "../models"
    processed_path = "../data/processed"
    
    # Create directories
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)
    
    # Team members
    members = ['Alliance', 'Ange', 'Elissa', 'Terry']
    
    # Prepare data
    print("\nLoading and processing images...")
    X_list = []
    y_list = []
    label_encoder = {}
    reverse_label_encoder = {}
    
    for idx, member in enumerate(members):
        label_encoder[member] = idx
        reverse_label_encoder[idx] = member
    
    for member_id in members:
        print(f"Processing {member_id}...")
        member_path = os.path.join(raw_images_path, member_id)
        
        if not os.path.exists(member_path):
            print(f"  Warning: {member_path} not found, skipping...")
            continue
        
        # Load images
        for filename in os.listdir(member_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(member_path, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                # Resize to standard size
                image_resized = cv2.resize(image, (224, 224))
                
                # Extract features from original
                features = extract_all_features(image_resized)
                X_list.append(features)
                y_list.append(label_encoder[member_id])
                
                # Apply augmentations
                augmented = [
                    rotate_image(image_resized, 15),
                    flip_image(image_resized),
                    convert_to_grayscale(image_resized)
                ]
                
                for aug_img in augmented:
                    aug_img = cv2.resize(aug_img, (224, 224))
                    features = extract_all_features(aug_img)
                    X_list.append(features)
                    y_list.append(label_encoder[member_id])
    
    # Convert to arrays
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nTraining data shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[OK] Model training completed!")
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE")
    print("="*70)
    print(f"Training Accuracy: {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
    print(f"Testing Accuracy:  {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"Training F1-Score: {f1_train:.4f}")
    print(f"Testing F1-Score:  {f1_test:.4f}")
    print("="*70)
    
    # Save model
    model_path = os.path.join(models_path, 'facial_recognition_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder': label_encoder,
            'reverse_label_encoder': reverse_label_encoder
        }, f)
    print(f"\n[OK] Model saved to: {model_path}")
    
    # Save features
    features_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    features_df['label'] = y
    features_df['member'] = features_df['label'].map(reverse_label_encoder)
    
    csv_path = os.path.join(processed_path, 'image_features.csv')
    features_df.to_csv(csv_path, index=False)
    print(f"[OK] Features saved to: {csv_path}")
    print(f"    Shape: {features_df.shape}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()
