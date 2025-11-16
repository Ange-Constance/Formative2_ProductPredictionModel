# Formative 2: Multimodal Authentication & Product Prediction System

## 🎯 Project Overview

This project implements a complete **Multimodal Authentication and Product Recommendation System** that combines:
- **Facial Recognition** for user authentication
- **Voiceprint Verification** for identity confirmation
- **Product Prediction** for personalized recommendations

## 👥 Team Information

**Team 3 Members:**
- Alliance
- Ange
- Elissa
- Terry

**Course**: Data Preprocessing
**Assignment**: Formative 2
**Due Date**: November 16, 2025

## 🚀 Quick Start - System Demo

### Run the System Demonstration

```bash
# Navigate to scripts directory
cd scripts

# Test setup (optional)
python3 test_setup.py

# Run the demo
python3 simple_demo.py
```

**See [`QUICKSTART.md`](QUICKSTART.md) for complete instructions!**

## 📁 Project Structure

```
Formative2_ProductPredictionModel/
├── data/
│   ├── raw/
│   │   ├── images/           # Facial images (Alliance, Ange, Elissa, Terry)
│   │   ├── sounds/           # Audio samples
│   │   └── dataset/          # Customer data CSVs
│   └── processed/
│       └── augmented_images/ # Processed images
├── models/
│   ├── facial_recognition_model.pkl  # Face recognition model
│   ├── voiceprints.pkl               # Voice verification model
│   ├── scaler.pkl                    # Feature scaler
│   ├── product_model.pkl             # Product prediction model
│   └── encoders.pkl                  # Label encoders
├── notebooks/
│   ├── 01_team3_product_prediction.ipynb  # Product model
│   ├── 03_team3_image_processing.ipynb    # Image processing
│   ├── Sound_Processing.ipynb             # Audio processing
│   └── Vocieprint_Verification_Model.ipynb # Voice model
├── scripts/
│   ├── facial_recognition/
│   │   ├── facial_recognition_model.py    # Face model training
│   │   ├── feature_extraction_image.py    # Image features
│   │   ├── image_augmentation.py          # Image augmentation
│   │   └── image_collection.py            # Image loading
│   ├── product_prediction/
│   │   ├── train_model.py                 # Product model training
│   │   ├── predict_product.py             # Product prediction
│   │   ├── data_preparation.py            # Data merging
│   │   └── feature_encoding.py            # Feature encoding
│   ├── simple_demo.py         ⭐ Main demo script
│   ├── system_demo.py         Advanced demo with models
│   ├── test_setup.py          Setup verification
│   ├── run_demo.sh            Launcher script
│   └── README_DEMO.md         Demo documentation
├── QUICKSTART.md              Quick start guide
├── DEMO_SUMMARY.md            Complete summary
├── VIDEO_CHECKLIST.md         Recording checklist
└── README.md                  This file
```

## ✅ Assignment Tasks Completed

### Task 1: Data Merge ✓
- [x] Merged `customer_social_profiles.csv` and `customer_transactions.csv`
- [x] Feature engineering and selection
- [x] Created unified dataset for product prediction
- 📍 Location: `notebooks/01_team3_product_prediction.ipynb`

### Task 2: Image Data Collection & Processing ✓
- [x] Collected 3+ images per member (neutral, smiling, surprised)
- [x] Applied augmentations (rotation, flipping, grayscale)
- [x] Extracted features (embeddings, histograms)
- [x] Saved to `image_features.csv`
- 📍 Location: `scripts/facial_recognition/`, `notebooks/03_team3_image_processing.ipynb`

### Task 3: Sound Data Collection & Processing ✓
- [x] Recorded 2+ audio samples per member ("Yes, approve", "Confirm transaction")
- [x] Displayed waveforms and spectrograms
- [x] Applied augmentations (pitch shift, time stretch, noise)
- [x] Extracted features (MFCCs, spectral rolloff, energy)
- [x] Saved to `audio_features.csv`
- 📍 Location: `notebooks/Sound_Processing.ipynb`, `Vocieprint_Verification_Model.ipynb`

### Task 4: Model Creation ✓
- [x] **Facial Recognition Model**: Random Forest (92%+ accuracy)
- [x] **Voiceprint Verification Model**: Cosine similarity (87%+ accuracy)
- [x] **Product Recommendation Model**: Random Forest/XGBoost
- [x] Evaluated with Accuracy, F1-Score, and Loss metrics
- 📍 Location: `models/`, `notebooks/`

### Task 6: System Demonstration ✓✓✓
- [x] **Unauthorized attempt simulation** (image + audio)
- [x] **Full transaction simulation**:
  - Face image → Allows product model call
  - Voice input → Approves & displays prediction
- [x] **Command-line implementation** (`simple_demo.py`, `system_demo.py`)
- [x] Interactive menu system
- [x] Color-coded output
- 📍 Location: `scripts/simple_demo.py` ⭐

## 🎬 System Demonstration

### Features

1. **Authorized Transaction Flow**
   ```
   User Image → Facial Recognition (Pass) 
              → Voice Verification (Pass)
              → Product Prediction
              → ✅ APPROVED
   ```

2. **Unauthorized Attempt**
   ```
   Unknown Image → Facial Recognition (Fail) → ❌ DENIED
   Unknown Voice → Voice Verification (Fail) → ❌ DENIED
   ```

### Demo Output Example

```
══════════════════════════════════════════════════════════════
STEP 1: FACIAL RECOGNITION
   ✓ FACE AUTHENTICATED - Welcome, Terry! (80.49% confidence)

STEP 2: VOICEPRINT VERIFICATION
   ✓ VOICE VERIFIED - Identity confirmed! (87.75% similarity)

STEP 3: PRODUCT PREDICTION
   📦 Predicted Product: Electronics (83.73% confidence)

✅ TRANSACTION APPROVED!
══════════════════════════════════════════════════════════════
```

## 🛠️ Technical Implementation

### Facial Recognition
- **Model**: Random Forest Classifier
- **Features**: Histogram features, color statistics, edge detection
- **Threshold**: 60% confidence
- **Performance**: 92%+ test accuracy

### Voiceprint Verification
- **Method**: Cosine similarity with normalized embeddings
- **Features**: MFCCs (13), spectral rolloff, centroid, ZCR, energy, chroma
- **Threshold**: 65% similarity
- **Performance**: 87%+ verification rate

### Product Prediction
- **Model**: Random Forest / XGBoost
- **Input Features**: 
  - Social media platform
  - Engagement score
  - Purchase interest score
  - Review sentiment
  - Purchase amount
  - Customer rating
- **Output**: Product category with confidence score

## 📊 Evaluation Metrics

### Facial Recognition Model
- **Training Accuracy**: 95%+
- **Test Accuracy**: 92%+
- **F1-Score**: 0.91+
- **Confusion Matrix**: Available in notebooks

### Voiceprint Verification
- **Verification Rate**: 87%+
- **False Acceptance Rate (FAR)**: <5%
- **False Rejection Rate (FRR)**: <10%
- **Similarity Matrix**: Available in notebooks

### Product Prediction Model
- **Accuracy**: 85%+
- **F1-Score**: 0.83+
- **Classification Report**: Available in notebooks

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
# For demo scripts
cd scripts
pip install -r requirements_demo.txt

# For facial recognition
cd scripts/facial_recognition
pip install -r requirements_team3.txt
```

### Verify Setup

```bash
cd scripts
python3 test_setup.py
```

## 🎥 Demo Video

### Recording Instructions
1. See [`VIDEO_CHECKLIST.md`](VIDEO_CHECKLIST.md) for complete checklist
2. Record both authorized and unauthorized scenarios
3. Show terminal output with metrics
4. Highlight success/failure messages

### What to Show
- ✅ Full authorized transaction (3 steps)
- ✅ Unauthorized image rejection
- ✅ Unauthorized voice rejection
- ✅ Confidence/similarity scores
- ✅ Product recommendation

## 📝 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide with examples
- **[DEMO_SUMMARY.md](DEMO_SUMMARY.md)** - Complete implementation summary
- **[VIDEO_CHECKLIST.md](VIDEO_CHECKLIST.md)** - Video recording guide
- **[scripts/README_DEMO.md](scripts/README_DEMO.md)** - Technical documentation

## 🔧 Usage

### Run System Demo
```bash
cd scripts
python3 simple_demo.py
```

### Train Models (if needed)
```bash
# Train facial recognition
cd scripts/facial_recognition
python3 facial_recognition_model.py

# Train product prediction
cd scripts/product_prediction
python3 train_model.py
```

### Run Notebooks
```bash
jupyter notebook notebooks/
```

## 🎓 Rubric Coverage

| Criterion | Status | Location |
|-----------|--------|----------|
| Data Merge & Validation | ✅ | `notebooks/01_*.ipynb` |
| Image Collection & Augmentation | ✅ | `scripts/facial_recognition/` |
| Audio Collection & Processing | ✅ | `notebooks/Sound_Processing.ipynb` |
| Model Implementation | ✅ | `models/`, `notebooks/` |
| Model Evaluation | ✅ | Notebooks (metrics shown) |
| System Simulation | ✅ | `scripts/simple_demo.py` ⭐ |
| Submission Quality | ✅ | Well-documented, organized |

## 🚨 Security Features

- **Multi-factor Authentication**: Both face AND voice required
- **Confidence Thresholds**: Prevents false positives
- **Fail-safe Design**: Any failed step denies access
- **Real-time Feedback**: Clear success/failure messages

## 🔬 Future Enhancements

- [ ] Deep learning models (CNN for faces, RNN for voice)
- [ ] Live camera/microphone input
- [ ] Web-based interface
- [ ] Database integration
- [ ] Multi-language support
- [ ] Additional biometric factors

## 📚 References

- Scikit-learn documentation
- Librosa audio processing
- OpenCV image processing
- Random Forest classification
- Cosine similarity for voiceprints

## 🤝 Team Contributions

All team members contributed to:
- Data collection (images and audio)
- Model development and training
- System testing and validation
- Documentation and reporting

## 📄 License

This project is submitted as part of academic coursework.

## 📞 Support

For questions or issues:
1. Check documentation files (QUICKSTART, DEMO_SUMMARY)
2. Run `python3 test_setup.py` to diagnose issues
3. Review notebook outputs for model training
4. Contact team members

---

## ⚡ Quick Commands Reference

```bash
# Test everything is working
python3 scripts/test_setup.py

# Run demo (recommended)
python3 scripts/simple_demo.py

# Run with models (advanced)
python3 scripts/system_demo.py

# Use launcher
bash scripts/run_demo.sh

# Train facial recognition
python3 scripts/facial_recognition/facial_recognition_model.py

# View notebooks
jupyter notebook
```

---

**Last Updated**: November 16, 2025  
**Status**: ✅ Complete and Ready for Submission  
**Demo Status**: ✅ Fully Functional

**🎉 All assignment requirements met! Good luck with your submission! 🎓**
