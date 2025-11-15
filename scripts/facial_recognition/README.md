# Team 3: Image Processing & Facial Recognition Model - README

## Task Overview
This folder contains all scripts and utilities for **Task 3: Image Data Processing and Facial Recognition Model**.

## Responsibilities
- Collect facial images (neutral, smiling, surprised) from all 4 team members
- Apply image augmentations (rotation, flipping, grayscale)
- Extract facial features (histograms, HOG, edges, color moments, LBP)
- Build and train a facial recognition model using Random Forest
- Evaluate model performance using Accuracy, F1-Score, and confusion matrix
- Save features to `image_features.csv`

## Files Structure

```
team3_facial_recognition/
├── image_collection.py              # Load and display images
├── image_augmentation.py            # Apply augmentations to images
├── feature_extraction_image.py      # Extract features from images
├── facial_recognition_model.py      # Train and evaluate model
├── utils.py                         # Utility functions
├── requirements_team3.txt           # Python dependencies
└── README.md                        # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements_team3.txt
```

## Usage

### Step 1: Prepare Image Data
Place your facial images in:
```
data/raw/images/
├── member1/
│   ├── neutral.jpg
│   ├── smiling.jpg
│   └── surprised.jpg
├── member2/
├── member3/
└── member4/
```

### Step 2: Load and Display Images
```python
from image_collection import load_member_images, display_images

member1_images = load_member_images("member1", "../../data/raw/images")
display_images(member1_images, "Member 1 - Facial Images")
```

### Step 3: Apply Augmentations
```python
from image_augmentation import apply_augmentations, rotate_image, flip_image

augmentations = [
    lambda img: rotate_image(img, 15),
    lambda img: flip_image(img),
]
augmented = apply_augmentations(image, augmentations)
```

### Step 4: Extract Features
```python
from feature_extraction_image import extract_all_features, concatenate_features

features_dict = extract_all_features(image)
feature_vector = concatenate_features(features_dict)
```

### Step 5: Train Model
```python
from facial_recognition_model import FacialRecognitionModel

model = FacialRecognitionModel(model_type='random_forest')
X, y = model.prepare_training_data("../../data/raw/images", "../../data/processed")
metrics = model.train(X, y, test_size=0.2)
model.evaluate_and_plot(metrics)
model.save_model("../../models/facial_recognition_model.pkl")
```

## Output Files

- `image_features.csv` - Extracted features from all images (saved to `data/processed/`)
- `facial_recognition_model.pkl` - Trained model (saved to `models/`)
- Model evaluation metrics and confusion matrix plots

## Model Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness of predictions
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows true positives, false positives, etc.

## Feature Types Extracted

1. **Histogram Features**: Color distribution across channels
2. **Edge Features**: Canny edge detection
3. **HOG Features**: Histogram of Oriented Gradients
4. **Color Moments**: Mean, standard deviation, skewness per channel
5. **LBP Features**: Local Binary Pattern descriptors

## Dependencies

- `opencv-python`: Image processing
- `scikit-learn`: Machine learning models
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Visualization
- `seaborn`: Statistical visualizations
- `albumentations`: Image augmentation

## Notes

- Ensure all team members have 3 images (neutral, smiling, surprised) in their respective folders
- Images should be clear and well-lit for best results
- Model training time depends on number of images and augmentations applied
- Augmentations increase training data size for better model generalization

## Team Coordination

This task depends on:
- ✅ Team 1: No dependency (can work independently)
- ✅ Team 2: No dependency (can work independently)
- ⚠️ Team 4: Requires this task to be completed (needs image_features.csv and facial recognition model)

## Questions or Issues?

Contact team member 3 for facial recognition related questions.
