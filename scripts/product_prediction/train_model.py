"""
train_model.py

This script trains a Random Forest Classifier on the encoded dataset
and saves the trained model for future predictions.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load features and target
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').values.ravel()  # flatten to 1D

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print("Model Evaluation Report:\n")
print(classification_report(y_test, y_pred))

# Save trained model
with open('product_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Trained model saved as 'product_model.pkl'.")
