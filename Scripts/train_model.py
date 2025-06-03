import pandas as pd
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Paths
RAW_DIR = os.path.join("data", "raw")
ARTIFACTS_DIR = os.path.join("Scripts", "Artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Load data
ratings_df = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))
users_df = pd.read_csv(os.path.join(RAW_DIR, "users.csv"))
books_df = pd.read_csv(os.path.join(RAW_DIR, "books.csv"))

# Convert IDs to strings
ratings_df["user_id"] = ratings_df["user_id"].astype(str)
ratings_df["book_id"] = ratings_df["book_id"].astype(str)
users_df["user_id"] = users_df["user_id"].astype(str)
books_df["book_id"] = books_df["book_id"].astype(str)

# Merge
merged_df = ratings_df.merge(users_df, on="user_id", how="inner").merge(books_df, on="book_id", how="inner")

# Features and target
X = merged_df[["user_id", "age", "gender", "occupation", "city", "language", "genre"]]
y = merged_df["rating"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Preprocessing pipeline for categorical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Full pipeline including preprocessing, scaling, and classifier
full_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model pipeline
full_pipeline.fit(X_train, y_train)

# Save the whole pipeline as model artifact
with open(os.path.join(ARTIFACTS_DIR, "best_classifier.pkl"), "wb") as f:
    pickle.dump(full_pipeline, f)

# Save label encoder separately
with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)
