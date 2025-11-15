"""
feature_encoding.py

This script encodes categorical features in the merged dataset,
prepares features (X) and target (y), and saves the encoders
for future predictions.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load merged dataset
merged_df = pd.read_csv('merged_dataset.csv')

# Initialize dictionary to save LabelEncoders
le_dict = {}

# Encode categorical features (except target)
for col in merged_df.select_dtypes(include=['object']).columns:
    if col != 'product_category':
        le = LabelEncoder()
        merged_df[col] = le.fit_transform(merged_df[col].astype(str))
        le_dict[col] = le

# Encode target variable
le_target = LabelEncoder()
merged_df['product_category'] = le_target.fit_transform(merged_df['product_category'])

# Prepare features and target
X = merged_df.drop(columns=['product_category'])
y = merged_df['product_category']

# Save encoders and dataset for model training
with open('encoders.pkl', 'wb') as f:
    pickle.dump({'le_dict': le_dict, 'le_target': le_target}, f)

X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
print("Features, target, and encoders saved.")
