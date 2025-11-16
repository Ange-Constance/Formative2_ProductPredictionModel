#!/usr/bin/env python3
"""
Quick script to rename existing model files to match demo requirements.
This copies rf_model.pkl -> product_model.pkl and label_encoders.pkl -> encoders.pkl
"""

import shutil
import os

models_dir = '../models'

# Check if source files exist
rf_model = os.path.join(models_dir, 'rf_model.pkl')
label_encoders = os.path.join(models_dir, 'label_encoders.pkl')

if os.path.exists(rf_model):
    shutil.copy(rf_model, os.path.join(models_dir, 'product_model.pkl'))
    print("[OK] Created product_model.pkl from rf_model.pkl")
else:
    print("[FAILED] rf_model.pkl not found")

if os.path.exists(label_encoders):
    shutil.copy(label_encoders, os.path.join(models_dir, 'encoders.pkl'))
    print("[OK] Created encoders.pkl from label_encoders.pkl")
else:
    print("[FAILED] label_encoders.pkl not found")

print("\nModel files ready for demo system!")
