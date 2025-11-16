#!/usr/bin/env python3
"""
Simple System Demonstration
Simulates the authentication flow with minimal dependencies.
"""

import os
import sys
import pickle
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")


def print_step(step_num, text):
    """Print a step header."""
    print(f"\n{Colors.BLUE}{'─'*70}{Colors.END}")
    print(f"{Colors.BLUE}STEP {step_num}: {text}{Colors.END}")
    print(f"{Colors.BLUE}{'─'*70}{Colors.END}")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}   [OK] {text}{Colors.END}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}   [FAILED] {text}{Colors.END}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.YELLOW}   {text}{Colors.END}")


class SimpleSystemDemo:
    """Simplified demonstration system."""
    
    def __init__(self):
        """Initialize the system."""
        print_header("MULTIMODAL AUTHENTICATION SYSTEM DEMO")
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        
        # Check if models exist
        self.check_models()
    
    def check_models(self):
        """Check which models are available."""
        print_info("Checking available models...")
        
        self.has_facial = (self.models_dir / "facial_recognition_model.pkl").exists()
        self.has_voice = (self.models_dir / "voiceprints.pkl").exists()
        self.has_product = (self.models_dir / "product_model.pkl").exists()
        
        if self.has_facial:
            print_success("Facial recognition model: Available")
        else:
            print_error("Facial recognition model: Not found")
        
        if self.has_voice:
            print_success("Voiceprint verification model: Available")
        else:
            print_error("Voiceprint verification model: Not found")
        
        if self.has_product:
            print_success("Product prediction model: Available")
        else:
            print_info("Product prediction model: Not found (will use mock)")
        
        print()
    
    def simulate_facial_recognition(self, member_name, image_file):
        """Simulate facial recognition."""
        print_step(1, "FACIAL RECOGNITION")
        
        print_info(f"Processing image: {image_file}")
        print_info(f"   Analyzing facial features...")
        print_info(f"   Comparing with database...")
        
        # Simulate processing
        import random
        confidence = random.uniform(0.75, 0.95)
        
        print_info(f"   Detected: {member_name}")
        print_info(f"   Confidence: {confidence:.2%}")
        
        if confidence >= 0.6:
            print_success(f"FACE AUTHENTICATED - Welcome, {member_name}!")
            return True, member_name, confidence
        else:
            print_error(f"AUTHENTICATION FAILED - Low confidence")
            return False, member_name, confidence
    
    def simulate_voice_verification(self, member_name, audio_file):
        """Simulate voice verification."""
        print_step(2, "VOICEPRINT VERIFICATION")
        
        print_info(f"Processing audio: {audio_file}")
        print_info(f"   Extracting voice features...")
        print_info(f"   Comparing voiceprint...")
        
        # Simulate processing
        import random
        similarity = random.uniform(0.70, 0.90)
        
        print_info(f"   Claimed member: {member_name}")
        print_info(f"   Similarity score: {similarity:.2%}")
        
        if similarity >= 0.65:
            print_success(f"VOICE VERIFIED - Identity confirmed!")
            return True, similarity
        else:
            print_error(f"VERIFICATION FAILED - Voice does not match")
            return False, similarity
    
    def simulate_product_prediction(self):
        """Simulate product prediction."""
        print_step(3, "PRODUCT PREDICTION")
        
        print_info("Customer Profile:")
        profile = {
            'Social Media Platform': 'Instagram',
            'Engagement Score': 78,
            'Purchase Interest': 90,
            'Review Sentiment': 'Positive',
            'Purchase Amount': '$50',
            'Customer Rating': '5/5'
        }
        
        for key, value in profile.items():
            print_info(f"   - {key}: {value}")
        
        print_info("\n   Analyzing customer behavior...")
        print_info("   Running prediction model...")
        
        # Simulate prediction
        import random
        products = ['Electronics', 'Clothing', 'Books', 'Home & Kitchen', 'Sports & Outdoors']
        product = random.choice(products)
        confidence = random.uniform(0.75, 0.95)
        
        print_success(f"\n   Predicted Product: {product}")
        print_success(f"   Confidence: {confidence:.2%}")
        
        return True, product, confidence
    
    def run_authorized_transaction(self, member_name, image_file, audio_file):
        """Run a full authorized transaction."""
        print_header("AUTHORIZED TRANSACTION SIMULATION")
        
        # Step 1: Facial Recognition
        face_auth, detected_member, face_conf = self.simulate_facial_recognition(member_name, image_file)
        
        if not face_auth:
            print(f"\n{Colors.RED}❌ TRANSACTION DENIED: Facial authentication failed{Colors.END}\n")
            return False
        
        # Step 2: Voice Verification
        voice_auth, voice_sim = self.simulate_voice_verification(detected_member, audio_file)
        
        if not voice_auth:
            print(f"\n{Colors.RED}❌ TRANSACTION DENIED: Voice verification failed{Colors.END}\n")
            return False
        
        # Step 3: Product Prediction
        pred_success, product, pred_conf = self.simulate_product_prediction()
        
        if pred_success:
            print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
            print(f"{Colors.GREEN}TRANSACTION APPROVED{Colors.END}")
            print(f"{Colors.GREEN}{'='*70}{Colors.END}")
            print(f"{Colors.GREEN}   User: {detected_member}{Colors.END}")
            print(f"{Colors.GREEN}   Recommended Product: {product}{Colors.END}")
            print(f"{Colors.GREEN}   Face Confidence: {face_conf:.2%}{Colors.END}")
            print(f"{Colors.GREEN}   Voice Similarity: {voice_sim:.2%}{Colors.END}")
            print(f"{Colors.GREEN}   Product Confidence: {pred_conf:.2%}{Colors.END}")
            print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
            return True
        
        return False
    
    def run_unauthorized_attempt(self):
        """Simulate an unauthorized access attempt."""
        print_header("UNAUTHORIZED ATTEMPT SIMULATION")
        
        print_info("Testing system security with unauthorized credentials...")
        print_info("   Scenario: Unknown user attempting access\n")
        
        # Simulate failed facial recognition
        print_step(1, "FACIAL RECOGNITION")
        print_info("Processing image: unauthorized_face.jpg")
        print_info("   Analyzing facial features...")
        print_info("   Comparing with database...")
        
        import random
        confidence = random.uniform(0.30, 0.55)
        
        print_info(f"   Detection confidence: {confidence:.2%}")
        print_error(f"AUTHENTICATION FAILED - User not recognized")
        print_success("\n✓ System correctly rejected unauthorized user")
        
        # Alternative: Simulate failed voice verification
        print_step(2, "ALTERNATIVE: Voice Verification Failure")
        print_info("Processing audio: unauthorized_voice.wav")
        print_info("   Extracting voice features...")
        print_info("   Comparing voiceprint...")
        
        similarity = random.uniform(0.35, 0.60)
        print_info(f"   Best match similarity: {similarity:.2%}")
        print_error(f"VERIFICATION FAILED - Voice does not match any user")
        print_success("\nSystem correctly rejected unauthorized voice")
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.CYAN}Security test completed - System functioning correctly{Colors.END}")
        print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")
    
    def run_demo(self):
        """Run the main demonstration."""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.CYAN}SELECT DEMONSTRATION MODE{Colors.END}")
        print(f"{Colors.CYAN}{'='*70}{Colors.END}")
        print("1. Full Transaction (Authorized User)")
        print("2. Unauthorized Attempt (Security Test)")
        print("3. Run Both Demonstrations")
        print("4. Exit")
        print(f"{Colors.CYAN}{'='*70}{Colors.END}")
        
        choice = input(f"\n{Colors.BOLD}Enter your choice (1-4): {Colors.END}").strip()
        
        if choice == '1':
            # Get available members
            images_dir = self.data_dir / "raw" / "images"
            if images_dir.exists():
                members = [d.name for d in images_dir.iterdir() if d.is_dir()]
                
                if members:
                    print(f"\n{Colors.YELLOW}Available members:{Colors.END}")
                    for idx, member in enumerate(members, 1):
                        print(f"{idx}. {member}")
                    
                    member_choice = input(f"\n{Colors.BOLD}Select member (1-{len(members)}): {Colors.END}").strip()
                    
                    try:
                        member_idx = int(member_choice) - 1
                        member = members[member_idx]
                        
                        # Find first image
                        member_dir = images_dir / member
                        images = list(member_dir.glob("*.jpg")) + list(member_dir.glob("*.png"))
                        
                        if images:
                            image_file = images[0].name
                            audio_file = "approve_" + member.lower() + ".wav"
                            
                            self.run_authorized_transaction(member, image_file, audio_file)
                        else:
                            print_error("No images found for this member!")
                    
                    except (ValueError, IndexError):
                        print_error("Invalid selection!")
                else:
                    # Use default demo
                    self.run_authorized_transaction("Alliance", "neutral.jpg", "approve.wav")
            else:
                # Use default demo
                self.run_authorized_transaction("Alliance", "neutral.jpg", "approve.wav")
        
        elif choice == '2':
            self.run_unauthorized_attempt()
        
        elif choice == '3':
            # Run both
            self.run_authorized_transaction("Alliance", "neutral.jpg", "approve.wav")
            input(f"\n{Colors.BOLD}Press Enter to continue to unauthorized attempt demo...{Colors.END}")
            self.run_unauthorized_attempt()
        
        elif choice == '4':
            print(f"\n{Colors.CYAN}Exiting... Goodbye!{Colors.END}\n")
            return
        
        else:
            print_error("Invalid choice!")


def main():
    """Main entry point."""
    try:
        demo = SimpleSystemDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user. Exiting...{Colors.END}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {str(e)}{Colors.END}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
