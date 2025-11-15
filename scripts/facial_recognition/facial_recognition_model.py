"""
Facial Recognition Model
Train and evaluate a facial recognition model using scikit-learn.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from image_collection import load_member_images
from image_augmentation import apply_augmentations, rotate_image, flip_image, convert_to_grayscale
from feature_extraction_image import extract_all_features, concatenate_features


class FacialRecognitionModel:
    """
    Facial Recognition Model for user authentication.
    """
    
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize the facial recognition model.
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'logistic_regression', etc.)
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the classifier based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
    
    def prepare_training_data(self, raw_images_path, processed_path, members=['member1', 'member2', 'member3', 'member4']):
        """
        Prepare training data from raw images.
        
        Args:
            raw_images_path (str): Path to raw images directory
            processed_path (str): Path to save processed features
            members (list): List of member identifiers
            
        Returns:
            tuple: (X, y) - Features and labels
        """
        X_list = []
        y_list = []
        
        # Create label encoder
        for idx, member in enumerate(members):
            self.label_encoder[member] = idx
            self.reverse_label_encoder[idx] = member
        
        print("Loading and processing images...")
        
        for member_id in members:
            print(f"Processing {member_id}...")
            
            try:
                # Load member's images
                member_images = load_member_images(member_id, raw_images_path)
                
                for img_name, image in member_images.items():
                    # Extract features from original image
                    features_dict = extract_all_features(image)
                    feature_vector = concatenate_features(features_dict)
                    X_list.append(feature_vector)
                    y_list.append(self.label_encoder[member_id])
                    
                    # Apply augmentations and extract features
                    augmentations = [
                        lambda img: rotate_image(img, 15),
                        lambda img: flip_image(img),
                        lambda img: convert_to_grayscale(img)
                    ]
                    
                    augmented_images = apply_augmentations(image, augmentations)
                    
                    for aug_img in augmented_images[1:]:  # Skip original
                        if len(aug_img.shape) == 2:  # Grayscale
                            aug_img = np.stack([aug_img] * 3, axis=-1)
                        
                        features_dict = extract_all_features(aug_img)
                        feature_vector = concatenate_features(features_dict)
                        X_list.append(feature_vector)
                        y_list.append(self.label_encoder[member_id])
            
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Training data shape: {X.shape}")
        print(f"Number of samples: {len(y)}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """
        Train the facial recognition model.
        
        Args:
            X (ndarray): Feature matrix
            y (ndarray): Labels
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'accuracy_train': accuracy_score(y_train, y_pred_train),
            'accuracy_test': accuracy_score(y_test, y_pred_test),
            'f1_train': f1_score(y_train, y_pred_train, average='weighted'),
            'f1_test': f1_score(y_test, y_pred_test, average='weighted'),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
        
        return metrics
    
    def predict(self, image):
        """
        Predict the user from an image.
        
        Args:
            image (ndarray): Input image
            
        Returns:
            dict: Prediction result with member ID and confidence
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        features_dict = extract_all_features(image)
        feature_vector = concatenate_features(features_dict).reshape(1, -1)
        
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        confidence = np.max(probabilities)
        
        return {
            'member': self.reverse_label_encoder[prediction],
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'reverse_label_encoder': self.reverse_label_encoder
            }, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to load the model from
        """
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.reverse_label_encoder = data['reverse_label_encoder']
        print(f"Model loaded from {model_path}")
    
    def evaluate_and_plot(self, metrics):
        """
        Evaluate and plot model performance.
        
        Args:
            metrics (dict): Dictionary with evaluation metrics
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Training Accuracy: {metrics['accuracy_train']:.4f}")
        print(f"Testing Accuracy:  {metrics['accuracy_test']:.4f}")
        print(f"Training F1-Score: {metrics['f1_train']:.4f}")
        print(f"Testing F1-Score:  {metrics['f1_test']:.4f}")
        print("="*50 + "\n")
        
        # Confusion Matrix
        cm = confusion_matrix(metrics['y_test'], metrics['y_pred_test'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=list(self.reverse_label_encoder.values()),
                    yticklabels=list(self.reverse_label_encoder.values()))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(metrics['y_test'], metrics['y_pred_test'],
                                  target_names=list(self.reverse_label_encoder.values())))


if __name__ == "__main__":
    # Example usage
    raw_images_path = "../../data/raw/images"
    processed_path = "../../data/processed"
    model_path = "../../models/facial_recognition_model.pkl"
    
    # Initialize model
    model = FacialRecognitionModel(model_type='random_forest')
    
    # Prepare data
    X, y = model.prepare_training_data(raw_images_path, processed_path)
    
    # Train model
    metrics = model.train(X, y, test_size=0.2)
    
    # Evaluate
    model.evaluate_and_plot(metrics)
    
    # Save model
    model.save_model(model_path)
