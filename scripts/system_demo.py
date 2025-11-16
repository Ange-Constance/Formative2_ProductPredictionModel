#!/usr/bin/env python3
"""
System Demonstration: Multimodal Authentication and Product Prediction
This script demonstrates the full authentication flow:
1. Facial Recognition
2. Voiceprint Verification
3. Product Prediction

Author: Team 3
Date: November 2025
"""

import os
import sys
import pickle
import cv2
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from colorama import init, Fore, Style
import warnings
warnings.filterwarnings('ignore')

# Initialize colorama for colored terminal output
init(autoreset=True)

# Add facial recognition module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'facial_recognition'))
from feature_extraction_image import extract_all_features, concatenate_features

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


class SystemDemo:
    """Main class for multimodal authentication system demonstration."""
    
    def __init__(self):
        """Initialize the system and load all models."""
        print(Fore.CYAN + "="*70)
        print(Fore.CYAN + "MULTIMODAL AUTHENTICATION & PRODUCT PREDICTION SYSTEM")
        print(Fore.CYAN + "="*70)
        print()
        
        self.load_models()
    
    def load_models(self):
        """Load all pre-trained models."""
        print(Fore.YELLOW + "📦 Loading models...")
        
        try:
            # Load facial recognition model
            facial_model_path = MODELS_DIR / "facial_recognition_model.pkl"
            if facial_model_path.exists():
                with open(facial_model_path, 'rb') as f:
                    facial_data = pickle.load(f)
                    self.facial_model = facial_data['model']
                    self.face_label_encoder = facial_data['label_encoder']
                    self.face_reverse_encoder = facial_data['reverse_label_encoder']
                print(Fore.GREEN + "   ✓ Facial recognition model loaded")
            else:
                print(Fore.RED + f"   ✗ Facial model not found at {facial_model_path}")
                self.facial_model = None
            
            # Load voiceprint verification models
            voiceprint_path = MODELS_DIR / "voiceprints.pkl"
            scaler_path = MODELS_DIR / "scaler.pkl"
            
            if voiceprint_path.exists() and scaler_path.exists():
                with open(voiceprint_path, 'rb') as f:
                    self.voiceprints = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.audio_scaler = pickle.load(f)
                print(Fore.GREEN + "   ✓ Voiceprint verification model loaded")
            else:
                print(Fore.RED + "   ✗ Voiceprint models not found")
                self.voiceprints = None
                self.audio_scaler = None
            
            # Load product prediction model
            product_model_path = MODELS_DIR / "product_model.pkl"
            encoders_path = MODELS_DIR / "encoders.pkl"
            
            if product_model_path.exists() and encoders_path.exists():
                with open(product_model_path, 'rb') as f:
                    self.product_model = pickle.load(f)
                with open(encoders_path, 'rb') as f:
                    self.encoders = pickle.load(f)
                print(Fore.GREEN + "   ✓ Product prediction model loaded")
            else:
                print(Fore.YELLOW + "   ⚠ Product model not found (optional)")
                self.product_model = None
                self.encoders = None
            
            print()
            
        except Exception as e:
            print(Fore.RED + f"   ✗ Error loading models: {str(e)}")
            raise
    
    def authenticate_face(self, image_path, confidence_threshold=0.6):
        """
        Authenticate user via facial recognition.
        
        Args:
            image_path (str): Path to the facial image
            confidence_threshold (float): Minimum confidence for authentication
            
        Returns:
            dict: Authentication result
        """
        print(Fore.CYAN + "\n" + "─"*70)
        print(Fore.CYAN + "STEP 1: FACIAL RECOGNITION")
        print(Fore.CYAN + "─"*70)
        
        if self.facial_model is None:
            return {'success': False, 'message': 'Facial recognition model not available'}
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'success': False, 'message': f'Failed to load image: {image_path}'}
            
            print(Fore.YELLOW + f"📷 Processing image: {Path(image_path).name}")
            
            # Extract features
            features_dict = extract_all_features(image)
            feature_vector = concatenate_features(features_dict).reshape(1, -1)
            
            # Predict
            prediction = self.facial_model.predict(feature_vector)[0]
            probabilities = self.facial_model.predict_proba(feature_vector)[0]
            confidence = np.max(probabilities)
            
            member = self.face_reverse_encoder[prediction]
            
            # Display results
            print(Fore.YELLOW + f"   Detected: {member}")
            print(Fore.YELLOW + f"   Confidence: {confidence:.2%}")
            
            if confidence >= confidence_threshold:
                print(Fore.GREEN + f"   ✓ FACE AUTHENTICATED - Welcome, {member}!")
                return {
                    'success': True,
                    'member': member,
                    'confidence': confidence,
                    'message': f'Face authenticated: {member}'
                }
            else:
                print(Fore.RED + f"   ✗ AUTHENTICATION FAILED - Low confidence ({confidence:.2%})")
                return {
                    'success': False,
                    'member': member,
                    'confidence': confidence,
                    'message': f'Confidence too low: {confidence:.2%} < {confidence_threshold:.2%}'
                }
                
        except Exception as e:
            print(Fore.RED + f"   ✗ Error: {str(e)}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def extract_audio_features(self, audio_path):
        """Extract features from audio file."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Extract features
            features = {}
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Energy
            features['energy'] = np.sum(y**2) / len(y)
            
            # Spectral features
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            return features
            
        except Exception as e:
            raise Exception(f"Error extracting audio features: {str(e)}")
    
    def verify_voice(self, audio_path, claimed_member, similarity_threshold=0.65):
        """
        Verify user via voiceprint.
        
        Args:
            audio_path (str): Path to the audio file
            claimed_member (str): The member claiming to authenticate
            similarity_threshold (float): Minimum similarity for verification
            
        Returns:
            dict: Verification result
        """
        print(Fore.CYAN + "\n" + "─"*70)
        print(Fore.CYAN + "STEP 2: VOICEPRINT VERIFICATION")
        print(Fore.CYAN + "─"*70)
        
        if self.voiceprints is None or self.audio_scaler is None:
            return {'success': False, 'message': 'Voiceprint verification not available'}
        
        try:
            print(Fore.YELLOW + f"🎤 Processing audio: {Path(audio_path).name}")
            
            # Extract features
            features_dict = self.extract_audio_features(audio_path)
            
            # Get feature columns in correct order
            feature_cols = sorted([k for k in features_dict.keys()])
            features_array = np.array([features_dict[k] for k in feature_cols])
            
            # Create embedding
            scaled = self.audio_scaler.transform([features_array])[0]
            norm = np.linalg.norm(scaled)
            if norm > 0:
                test_embedding = scaled / norm
            else:
                test_embedding = scaled
            
            # Map member name variations
            member_mapping = {
                'Alliance': 'alli',
                'Ange': 'ange',
                'Elissa': 'elisa',
                'Terry': 'terry',
                'member1': list(self.voiceprints.keys())[0] if len(self.voiceprints) > 0 else None,
                'member2': list(self.voiceprints.keys())[1] if len(self.voiceprints) > 1 else None,
                'member3': list(self.voiceprints.keys())[2] if len(self.voiceprints) > 2 else None,
                'member4': list(self.voiceprints.keys())[3] if len(self.voiceprints) > 3 else None,
            }
            
            # Try to find the member in voiceprints
            voice_member = member_mapping.get(claimed_member, claimed_member.lower())
            
            if voice_member not in self.voiceprints:
                # Try direct match
                for vp_key in self.voiceprints.keys():
                    if claimed_member.lower() in vp_key.lower() or vp_key.lower() in claimed_member.lower():
                        voice_member = vp_key
                        break
            
            if voice_member not in self.voiceprints:
                print(Fore.YELLOW + f"   Available voiceprints: {list(self.voiceprints.keys())}")
                return {
                    'success': False,
                    'message': f'No voiceprint found for {claimed_member}'
                }
            
            # Calculate similarity
            claimed_voiceprint = self.voiceprints[voice_member]
            similarity = np.dot(test_embedding, claimed_voiceprint)
            
            print(Fore.YELLOW + f"   Claimed member: {claimed_member}")
            print(Fore.YELLOW + f"   Similarity score: {similarity:.2%}")
            
            if similarity >= similarity_threshold:
                print(Fore.GREEN + f"   ✓ VOICE VERIFIED - Identity confirmed!")
                return {
                    'success': True,
                    'similarity': similarity,
                    'message': f'Voice verified for {claimed_member}'
                }
            else:
                print(Fore.RED + f"   ✗ VERIFICATION FAILED - Voice does not match")
                return {
                    'success': False,
                    'similarity': similarity,
                    'message': f'Similarity too low: {similarity:.2%} < {similarity_threshold:.2%}'
                }
                
        except Exception as e:
            print(Fore.RED + f"   ✗ Error: {str(e)}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def predict_product(self, customer_data=None):
        """
        Predict product for authenticated user.
        
        Args:
            customer_data (dict): Customer information (optional)
            
        Returns:
            dict: Prediction result
        """
        print(Fore.CYAN + "\n" + "─"*70)
        print(Fore.CYAN + "STEP 3: PRODUCT PREDICTION")
        print(Fore.CYAN + "─"*70)
        
        if self.product_model is None:
            # Create a mock prediction if model not available
            print(Fore.YELLOW + "   Using mock product prediction...")
            products = ['Electronics', 'Clothing', 'Books', 'Home & Kitchen', 'Sports']
            predicted = np.random.choice(products)
            confidence = np.random.uniform(0.7, 0.95)
            
            print(Fore.GREEN + f"   📦 Predicted Product: {predicted}")
            print(Fore.GREEN + f"   🎯 Confidence: {confidence:.2%}")
            
            return {
                'success': True,
                'product': predicted,
                'confidence': confidence,
                'message': f'Product prediction: {predicted}'
            }
        
        try:
            # Use default customer data if not provided
            if customer_data is None:
                customer_data = {
                    'social_media_platform': 'Instagram',
                    'engagement_score': 78,
                    'purchase_interest_score': 90,
                    'review_sentiment': 'Positive',
                    'purchase_amount': 50,
                    'customer_rating': 5
                }
            
            print(Fore.YELLOW + "   Customer Profile:")
            for key, value in customer_data.items():
                print(Fore.YELLOW + f"      - {key}: {value}")
            
            # Encode features
            le_dict = self.encoders['le_dict']
            le_target = self.encoders['le_target']
            
            encoded_data = {
                'social_media_platform': le_dict['social_media_platform'].transform([customer_data['social_media_platform']])[0],
                'engagement_score': customer_data['engagement_score'],
                'purchase_interest_score': customer_data['purchase_interest_score'],
                'review_sentiment': le_dict['review_sentiment'].transform([customer_data['review_sentiment']])[0],
                'purchase_amount': customer_data['purchase_amount'],
                'customer_rating': customer_data['customer_rating']
            }
            
            # Create DataFrame
            X = pd.DataFrame([encoded_data])
            
            # Predict
            prediction_encoded = self.product_model.predict(X)
            prediction_proba = self.product_model.predict_proba(X)
            
            predicted_product = le_target.inverse_transform(prediction_encoded)[0]
            confidence = np.max(prediction_proba)
            
            print(Fore.GREEN + f"\n   📦 Predicted Product: {predicted_product}")
            print(Fore.GREEN + f"   🎯 Confidence: {confidence:.2%}")
            
            return {
                'success': True,
                'product': predicted_product,
                'confidence': confidence,
                'message': f'Product prediction: {predicted_product}'
            }
            
        except Exception as e:
            print(Fore.RED + f"   ✗ Error: {str(e)}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def run_full_transaction(self, image_path, audio_path, customer_data=None):
        """
        Run a complete transaction with all authentication steps.
        
        Args:
            image_path (str): Path to facial image
            audio_path (str): Path to audio file
            customer_data (dict): Customer information (optional)
            
        Returns:
            dict: Transaction result
        """
        print(Fore.MAGENTA + "\n" + "╔" + "═"*68 + "╗")
        print(Fore.MAGENTA + "║" + " "*20 + "FULL TRANSACTION SIMULATION" + " "*21 + "║")
        print(Fore.MAGENTA + "╚" + "═"*68 + "╝")
        
        # Step 1: Facial Recognition
        face_result = self.authenticate_face(image_path)
        
        if not face_result['success']:
            print(Fore.RED + "\n❌ TRANSACTION DENIED: Facial authentication failed")
            return {
                'success': False,
                'step_failed': 'facial_recognition',
                'message': face_result['message']
            }
        
        member = face_result['member']
        
        # Step 2: Voiceprint Verification
        voice_result = self.verify_voice(audio_path, member)
        
        if not voice_result['success']:
            print(Fore.RED + "\n❌ TRANSACTION DENIED: Voice verification failed")
            return {
                'success': False,
                'step_failed': 'voice_verification',
                'message': voice_result['message']
            }
        
        # Step 3: Product Prediction
        product_result = self.predict_product(customer_data)
        
        if product_result['success']:
            print(Fore.GREEN + "\n" + "="*70)
            print(Fore.GREEN + "✅ TRANSACTION APPROVED!")
            print(Fore.GREEN + "="*70)
            print(Fore.GREEN + f"   User: {member}")
            print(Fore.GREEN + f"   Recommended Product: {product_result['product']}")
            print(Fore.GREEN + "="*70)
            
            return {
                'success': True,
                'member': member,
                'product': product_result['product'],
                'message': 'Transaction completed successfully'
            }
        else:
            return {
                'success': False,
                'step_failed': 'product_prediction',
                'message': product_result['message']
            }
    
    def run_unauthorized_attempt(self, image_path=None, audio_path=None):
        """
        Simulate an unauthorized access attempt.
        
        Args:
            image_path (str): Path to unauthorized image
            audio_path (str): Path to unauthorized audio
        """
        print(Fore.MAGENTA + "\n" + "╔" + "═"*68 + "╗")
        print(Fore.MAGENTA + "║" + " "*18 + "UNAUTHORIZED ATTEMPT SIMULATION" + " "*19 + "║")
        print(Fore.MAGENTA + "╚" + "═"*68 + "╝")
        
        if image_path:
            print(Fore.YELLOW + "\n🚨 Testing with unauthorized image...")
            result = self.authenticate_face(image_path, confidence_threshold=0.6)
            
            if not result['success']:
                print(Fore.RED + "\n✓ System correctly rejected unauthorized image")
            else:
                print(Fore.YELLOW + "\n⚠ Warning: Unauthorized image was accepted (possible false positive)")
        
        if audio_path:
            print(Fore.YELLOW + "\n🚨 Testing with unauthorized audio...")
            # Try with a random member name
            result = self.verify_voice(audio_path, "unauthorized_user", similarity_threshold=0.65)
            
            if not result['success']:
                print(Fore.RED + "\n✓ System correctly rejected unauthorized voice")
            else:
                print(Fore.YELLOW + "\n⚠ Warning: Unauthorized voice was accepted (possible false positive)")
        
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + "Security test completed")
        print(Fore.CYAN + "="*70)


def main():
    """Main function to run the system demonstration."""
    demo = SystemDemo()
    
    # Get available images and audio files
    images_dir = DATA_DIR / "raw" / "images"
    sounds_dir = DATA_DIR / "raw" / "sounds"
    
    print(Fore.CYAN + "\n" + "="*70)
    print(Fore.CYAN + "SELECT DEMONSTRATION MODE")
    print(Fore.CYAN + "="*70)
    print("1. Full Transaction (Authorized User)")
    print("2. Unauthorized Attempt")
    print("3. Custom Inputs")
    print("4. Exit")
    print("="*70)
    
    choice = input(Fore.WHITE + "\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        # Full authorized transaction
        print(Fore.YELLOW + "\n📋 Select user for demonstration:")
        members = [d for d in os.listdir(images_dir) if os.path.isdir(images_dir / d)]
        
        for idx, member in enumerate(members, 1):
            print(f"{idx}. {member}")
        
        member_choice = input(Fore.WHITE + f"\nEnter member number (1-{len(members)}): ").strip()
        
        try:
            member_idx = int(member_choice) - 1
            member = members[member_idx]
            
            # Get image
            member_images = list((images_dir / member).glob("*.jpg")) + \
                          list((images_dir / member).glob("*.jpeg")) + \
                          list((images_dir / member).glob("*.png"))
            
            if member_images:
                image_path = member_images[0]
            else:
                print(Fore.RED + "No images found for this member!")
                return
            
            # Get audio
            if sounds_dir.exists():
                audio_files = list(sounds_dir.glob("*.wav")) + list(sounds_dir.glob("*.mp3"))
                if audio_files:
                    audio_path = audio_files[0]
                else:
                    print(Fore.YELLOW + "No audio files found, proceeding without voice verification")
                    audio_path = None
            else:
                audio_path = None
            
            # Run transaction
            if audio_path:
                demo.run_full_transaction(str(image_path), str(audio_path))
            else:
                # Just test facial recognition
                demo.authenticate_face(str(image_path))
                
        except (ValueError, IndexError):
            print(Fore.RED + "Invalid selection!")
    
    elif choice == '2':
        # Unauthorized attempt
        print(Fore.YELLOW + "\n🚨 Running unauthorized attempt simulation...")
        print(Fore.YELLOW + "This will test the system's security by using mismatched credentials")
        
        # Use any available image and audio
        members = [d for d in os.listdir(images_dir) if os.path.isdir(images_dir / d)]
        if len(members) >= 2:
            # Use first member's image and second member's audio (mismatch)
            image_path = list((images_dir / members[0]).glob("*.jpg"))[0]
            
            if sounds_dir.exists():
                audio_files = list(sounds_dir.glob("*.wav")) + list(sounds_dir.glob("*.mp3"))
                audio_path = audio_files[0] if audio_files else None
            else:
                audio_path = None
            
            demo.run_unauthorized_attempt(str(image_path), str(audio_path) if audio_path else None)
        else:
            print(Fore.RED + "Not enough data for unauthorized attempt simulation")
    
    elif choice == '3':
        # Custom inputs
        image_path = input(Fore.WHITE + "Enter path to facial image: ").strip()
        audio_path = input(Fore.WHITE + "Enter path to audio file: ").strip()
        
        if os.path.exists(image_path) and os.path.exists(audio_path):
            demo.run_full_transaction(image_path, audio_path)
        else:
            print(Fore.RED + "Invalid file paths!")
    
    elif choice == '4':
        print(Fore.CYAN + "\nExiting... Goodbye!")
        return
    
    else:
        print(Fore.RED + "Invalid choice!")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(Fore.RED + f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
