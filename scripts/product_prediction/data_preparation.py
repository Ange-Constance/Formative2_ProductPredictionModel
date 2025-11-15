"""
data_preparation.py

This script loads the CSV files containing customer social profiles
and transactions, handles mismatched IDs, missing values, and
creates a combined dataset ready for model training.
"""

import pandas as pd

# Load CSV files
social_df = pd.read_csv('../../data/raw/dataset/customer_social_profiles.csv')
transactions_df = pd.read_csv('../../data/raw/dataset/customer_transactions.csv')

# Rename ID columns for consistency
social_df = social_df.rename(columns={'customer_id_new': 'customer_id'})
transactions_df = transactions_df.rename(columns={'customer_id_legacy': 'customer_id'})

# Drop IDs because they don't match
if 'customer_id' in social_df.columns:
    social_df = social_df.drop(columns=['customer_id'])
if 'customer_id' in transactions_df.columns:
    transactions_df = transactions_df.drop(columns=['customer_id'])

# Handle missing values
social_df.fillna({'social_media_platform': 'Unknown', 'review_sentiment': 'Neutral'}, inplace=True)
transactions_df.fillna({'product_category': 'Unknown', 'purchase_date': 0, 'transaction_id': 0}, inplace=True)

# Create cross join to combine datasets
social_df['key'] = 1
transactions_df['key'] = 1
merged_df = pd.merge(social_df, transactions_df, on='key').drop(columns=['key'])

# Save merged dataset for next step
merged_df.to_csv('merged_dataset.csv', index=False)
print("Merged dataset saved as 'merged_dataset.csv'. Shape:", merged_df.shape)
