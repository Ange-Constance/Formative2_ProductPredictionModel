"""
Image Feature Extraction Module
Extract facial features such as embeddings, histograms, and landmarks.
"""

import cv2
import numpy as np
from pathlib import Path


def extract_histogram_features(image, bins=32):
    """
    Extract color histogram features from image.
    
    Args:
        image (ndarray): Input image in BGR format
        bins (int): Number of histogram bins
        
    Returns:
        ndarray: Flattened histogram features
    """
    features = []
    for i in range(3):  # For each channel (B, G, R)
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        features.extend(hist.flatten())
    
    return np.array(features)


def extract_edge_features(image):
    """
    Extract edge features using Canny edge detection.
    
    Args:
        image (ndarray): Input image
        
    Returns:
        ndarray: Edge features (flattened)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges.flatten()


def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract HOG (Histogram of Oriented Gradients) features.
    
    Args:
        image (ndarray): Input image
        orientations (int): Number of orientation bins
        pixels_per_cell (tuple): Pixels per cell
        cells_per_block (tuple): Cells per block
        
    Returns:
        ndarray: HOG features
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        (64, 64),
        (16, 16),
        (8, 8),
        (8, 8),
        orientations
    )
    features = hog.compute(cv2.resize(gray, (64, 64)))
    return features.flatten()


def extract_color_moments(image):
    """
    Extract color moment features (mean, standard deviation, skewness per channel).
    
    Args:
        image (ndarray): Input image in BGR format
        
    Returns:
        ndarray: Color moment features
    """
    features = []
    
    for i in range(3):  # For each channel
        channel = image[:, :, i].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-6)
        
        features.extend([mean, std, skewness])
    
    return np.array(features)


def extract_lbp_features(image, num_points=8, radius=1):
    """
    Extract Local Binary Pattern (LBP) features.
    
    Args:
        image (ndarray): Input image
        num_points (int): Number of sample points
        radius (int): Radius of the circular neighborhood
        
    Returns:
        ndarray: LBP histogram features
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simple LBP computation
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


def extract_all_features(image, feature_names=None):
    """
    Extract all available features from an image.
    
    Args:
        image (ndarray): Input image
        feature_names (list): Specific features to extract. If None, extract all.
        
    Returns:
        dict: Dictionary with feature names as keys and feature arrays as values
    """
    available_features = {
        'histogram': extract_histogram_features,
        'edges': extract_edge_features,
        'hog': extract_hog_features,
        'color_moments': extract_color_moments,
        'lbp': extract_lbp_features
    }
    
    if feature_names is None:
        feature_names = list(available_features.keys())
    
    features = {}
    for name in feature_names:
        if name in available_features:
            features[name] = available_features[name](image)
    
    return features


def concatenate_features(features_dict):
    """
    Concatenate multiple feature arrays into a single feature vector.
    
    Args:
        features_dict (dict): Dictionary with feature arrays
        
    Returns:
        ndarray: Concatenated feature vector
    """
    all_features = []
    for feature_array in features_dict.values():
        all_features.extend(feature_array)
    
    return np.array(all_features)


if __name__ == "__main__":
    # Example usage
    from image_collection import load_image
    
    # Load sample image
    sample_image = load_image("../../data/raw/images/member1/neutral.jpg")
    
    # Extract features
    features = extract_all_features(sample_image)
    
    # Print feature information
    for feature_name, feature_array in features.items():
        print(f"{feature_name}: {feature_array.shape}")
    
    # Concatenate all features
    concatenated = concatenate_features(features)
    print(f"Total features: {len(concatenated)}")
