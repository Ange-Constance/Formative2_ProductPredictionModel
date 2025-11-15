"""
predict_product.py

This script loads the trained model and encoders, takes new customer
and transaction input, and predicts the product category.
"""

import pandas as pd
import pickle

# Load trained model and encoders
with open('product_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

le_dict = encoders['le_dict']
le_target = encoders['le_target']

# Example new customer data
new_data = pd.DataFrame([{
    'social_media_platform': le_dict['social_media_platform'].transform(['Instagram'])[0],
    'engagement_score': 78,
    'purchase_interest_score': 90,
    'review_sentiment': le_dict['review_sentiment'].transform(['Positive'])[0],
    'transaction_id': 0,
    'purchase_amount': 50,
    'purchase_date': 0,
    'customer_rating': 5
}])

# Predict product category
prediction_encoded = model.predict(new_data)
prediction = le_target.inverse_transform(prediction_encoded)
print("Predicted Product Category:", prediction[0])
