#!/usr/bin/env python3
"""
Test script to verify system demonstration setup
Tests all components without requiring actual model files
"""

import os
import sys
from pathlib import Path

# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    END = '\033[0m'


def test_python_version():
    """Test Python version."""
    print(f"\n{Colors.CYAN}Testing Python version...{Colors.END}")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"{Colors.GREEN}✓ Python {version.major}.{version.minor}.{version.micro} (OK){Colors.END}")
        return True
    else:
        print(f"{Colors.RED}✗ Python {version.major}.{version.minor} (Need 3.8+){Colors.END}")
        return False


def test_dependencies():
    """Test if required packages can be imported."""
    print(f"\n{Colors.CYAN}Testing dependencies...{Colors.END}")
    
    packages = {
        'pickle': 'pickle',
        'pathlib': 'pathlib',
        'os': 'os',
        'sys': 'sys'
    }
    
    optional_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'librosa': 'librosa',
        'matplotlib': 'matplotlib'
    }
    
    all_ok = True
    
    # Test required packages
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"{Colors.GREEN}✓ {name} (required){Colors.END}")
        except ImportError:
            print(f"{Colors.RED}✗ {name} (required) - MISSING{Colors.END}")
            all_ok = False
    
    # Test optional packages
    for module, name in optional_packages.items():
        try:
            __import__(module)
            print(f"{Colors.GREEN}✓ {name} (optional for full demo){Colors.END}")
        except ImportError:
            print(f"{Colors.YELLOW}⚠ {name} (optional) - Not installed{Colors.END}")
    
    return all_ok


def test_directory_structure():
    """Test directory structure."""
    print(f"\n{Colors.CYAN}Testing directory structure...{Colors.END}")
    
    base_dir = Path(__file__).parent.parent
    
    dirs = {
        'models': base_dir / 'models',
        'data': base_dir / 'data',
        'scripts': base_dir / 'scripts',
        'notebooks': base_dir / 'notebooks'
    }
    
    all_ok = True
    
    for name, path in dirs.items():
        if path.exists():
            print(f"{Colors.GREEN}✓ {name}/ directory exists{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ {name}/ directory not found{Colors.END}")
            all_ok = False
    
    return all_ok


def test_data_files():
    """Test for data files."""
    print(f"\n{Colors.CYAN}Testing data files...{Colors.END}")
    
    base_dir = Path(__file__).parent.parent
    
    # Check for images
    images_dir = base_dir / 'data' / 'raw' / 'images'
    if images_dir.exists():
        members = [d for d in images_dir.iterdir() if d.is_dir()]
        if members:
            print(f"{Colors.GREEN}✓ Found {len(members)} member directories in images/{Colors.END}")
            for member in members:
                images = list(member.glob('*.jpg')) + list(member.glob('*.png'))
                if images:
                    print(f"  {Colors.GREEN}✓ {member.name}: {len(images)} images{Colors.END}")
                else:
                    print(f"  {Colors.YELLOW}⚠ {member.name}: No images found{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ No member directories found in images/{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠ images/ directory not found{Colors.END}")
    
    # Check for audio
    sounds_dir = base_dir / 'data' / 'raw' / 'sounds'
    if sounds_dir.exists():
        audio_files = list(sounds_dir.glob('*.wav')) + list(sounds_dir.glob('*.mp3'))
        if audio_files:
            print(f"{Colors.GREEN}✓ Found {len(audio_files)} audio files{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ No audio files found in sounds/{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠ sounds/ directory not found{Colors.END}")


def test_model_files():
    """Test for trained model files."""
    print(f"\n{Colors.CYAN}Testing model files...{Colors.END}")
    
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'models'
    
    models = {
        'Facial Recognition': 'facial_recognition_model.pkl',
        'Voiceprints': 'voiceprints.pkl',
        'Audio Scaler': 'scaler.pkl',
        'Product Model': 'product_model.pkl',
        'Encoders': 'encoders.pkl'
    }
    
    for name, filename in models.items():
        path = models_dir / filename
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"{Colors.GREEN}✓ {name}: {filename} ({size:.1f} KB){Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ {name}: {filename} not found{Colors.END}")


def test_demo_scripts():
    """Test if demo scripts exist and are readable."""
    print(f"\n{Colors.CYAN}Testing demo scripts...{Colors.END}")
    
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / 'scripts'
    
    scripts = {
        'Simple Demo': 'simple_demo.py',
        'Full System Demo': 'system_demo.py',
        'Run Script': 'run_demo.sh'
    }
    
    all_ok = True
    
    for name, filename in scripts.items():
        path = scripts_dir / filename
        if path.exists():
            print(f"{Colors.GREEN}✓ {name}: {filename}{Colors.END}")
        else:
            print(f"{Colors.RED}✗ {name}: {filename} not found{Colors.END}")
            all_ok = False
    
    return all_ok


def test_simple_demo_import():
    """Test if simple demo can be imported."""
    print(f"\n{Colors.CYAN}Testing simple demo import...{Colors.END}")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from simple_demo import SimpleSystemDemo
        print(f"{Colors.GREEN}✓ Simple demo can be imported{Colors.END}")
        
        # Try to instantiate
        demo = SimpleSystemDemo()
        print(f"{Colors.GREEN}✓ Simple demo can be instantiated{Colors.END}")
        
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Error importing simple demo: {str(e)}{Colors.END}")
        return False


def main():
    """Run all tests."""
    print(f"{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.CYAN}SYSTEM DEMONSTRATION - PRE-FLIGHT CHECK{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}")
    
    results = []
    
    results.append(("Python Version", test_python_version()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Directory Structure", test_directory_structure()))
    test_data_files()
    test_model_files()
    results.append(("Demo Scripts", test_demo_scripts()))
    results.append(("Simple Demo Import", test_simple_demo_import()))
    
    print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.CYAN}SUMMARY{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}")
    
    for name, result in results:
        if result:
            print(f"{Colors.GREEN}✓ {name}: PASS{Colors.END}")
        else:
            print(f"{Colors.RED}✗ {name}: FAIL{Colors.END}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}✅ ALL TESTS PASSED - System ready for demonstration{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"\n{Colors.CYAN}To run the demo:{Colors.END}")
        print(f"  Simple Demo:  {Colors.YELLOW}python3 simple_demo.py{Colors.END}")
        print(f"  Full Demo:    {Colors.YELLOW}python3 system_demo.py{Colors.END}")
        print(f"  Or use:       {Colors.YELLOW}bash run_demo.sh{Colors.END}\n")
    else:
        print(f"\n{Colors.YELLOW}{'='*70}{Colors.END}")
        print(f"{Colors.YELLOW}⚠ SOME TESTS FAILED - Check warnings above{Colors.END}")
        print(f"{Colors.YELLOW}{'='*70}{Colors.END}")
        print(f"\n{Colors.CYAN}You can still run the simple demo without models:{Colors.END}")
        print(f"  {Colors.YELLOW}python3 simple_demo.py{Colors.END}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.END}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {str(e)}{Colors.END}\n")
        import traceback
        traceback.print_exc()
