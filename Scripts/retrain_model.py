import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Import your data loader
from data_loader import load_all_data

def main():
    print("Loading data...")
    books, ratings, users = load_all_data()

    if books.empty or ratings.empty or users.empty:
        print("Warning: One or more data files are missing or empty. Exiting early.")
        return

    # Merge ratings with users and books data
    try:
        merged = ratings.merge(users, on='user_id', how='left').merge(books, on='book_id', how='left')
        # Fix columns after merge (genre_x and genre_y appear)
        print("✅ Columns in merged:", merged.columns.tolist())
    except Exception as e:
        print(f"Warning: Error merging data: {e}")
        return

    # Select features - safely check for columns
    try:
        # Adjust column name to genre_x based on your merge printout
        features = merged[['user_id', 'age', 'gender', 'occupation', 'city', 'genre_x']]
    except KeyError as e:
        print(f"Warning: Missing columns in merged DataFrame: {e}")
        print("Available columns:", merged.columns.tolist())
        return

    # Rename genre_x to genre for easier processing
    features = features.rename(columns={'genre_x': 'genre'})

    # Target variable (rating)
    y = merged['rating'] if 'rating' in merged.columns else None
    if y is None:
        print("Warning: 'rating' column missing in merged data.")
        return

    # Simple train test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Warning: Train-test split failed: {e}")
        return

    # Encode target labels if needed
    label_encoder = LabelEncoder()
    try:
        y_train_encoded = label_encoder.fit_transform(y_train)
    except Exception as e:
        print(f"Warning: Label encoding failed: {e}")
        return

    # Build a simple pipeline (example: just RandomForestClassifier here)
    pipeline = Pipeline([
        # Add your transformers here (e.g., OneHotEncoder, etc.) as needed
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Fit pipeline safely
    try:
        pipeline.fit(X_train, y_train_encoded)
        print("Model training completed successfully.")
    except Exception as e:
        print(f"Warning: Pipeline fitting error: {e}")

if __name__ == "__main__":
    main()
