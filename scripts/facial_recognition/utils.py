"""
Utility functions for facial recognition pipeline.
"""

import os
import csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def create_image_features_csv(raw_images_path, output_csv_path, feature_extractor):
    """
    Create image_features.csv with all extracted features.
    
    Args:
        raw_images_path (str): Path to raw images
        output_csv_path (str): Path to save CSV file
        feature_extractor (function): Function to extract features
    """
    data = []
    members = [d for d in os.listdir(raw_images_path) if os.path.isdir(os.path.join(raw_images_path, d))]
    
    for member in members:
        member_path = os.path.join(raw_images_path, member)
        
        for filename in os.listdir(member_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(member_path, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    features = feature_extractor(image)
                    data.append({
                        'member': member,
                        'filename': filename,
                        'features': features
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Features saved to {output_csv_path}")
    
    return df


def ensure_directory(directory):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory (str): Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_augmented_image(image, output_path, augmentation_name):
    """
    Save an augmented image to disk.
    
    Args:
        image (ndarray): Image to save
        output_path (str): Base output path
        augmentation_name (str): Name of augmentation
    """
    ensure_directory(output_path)
    filename = f"{augmentation_name}.jpg"
    filepath = os.path.join(output_path, filename)
    cv2.imwrite(filepath, image)
    print(f"Saved: {filepath}")


def get_image_paths(directory):
    """
    Get all image paths in a directory recursively.
    
    Args:
        directory (str): Root directory
        
    Returns:
        list: List of image file paths
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    
    return image_paths


def print_dataset_info(X, y, label_encoder):
    """
    Print information about the dataset.
    
    Args:
        X (ndarray): Feature matrix
        y (ndarray): Labels
        label_encoder (dict): Label encoder mapping
    """
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    print(f"Total samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Number of classes: {len(label_encoder)}")
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        member = list(label_encoder.keys())[label]
        print(f"  {member}: {count} samples")
    print("="*50 + "\n")


def visualize_feature_importance(model, feature_names=None, top_n=20):
    """
    Visualize feature importance from Random Forest model.
    
    Args:
        model: Trained Random Forest model
        feature_names (list): Names of features
        top_n (int): Number of top features to display
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        selected_names = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices]
        selected_importances = importances[indices]
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(selected_importances)), selected_importances)
        plt.yticks(range(len(selected_importances)), selected_names)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()
