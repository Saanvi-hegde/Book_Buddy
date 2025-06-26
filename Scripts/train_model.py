import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === PATH SETUP ===
RAW_DIR = r"C:\Users\Saanvi Hegde\OneDrive\Desktop\mlops\BookBuddy\Data\raw"
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD DATASETS ===
users_df = pd.read_csv(os.path.join(RAW_DIR, "users.csv"))
books_df = pd.read_csv(os.path.join(RAW_DIR, "books.csv"))
ratings_df = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))

# === RENAME TO AVOID COLUMN COLLISIONS ===
users_df = users_df.rename(columns={"genre": "preferred_genre"})
books_df = books_df.rename(columns={"genre": "book_genre"})

# === MERGE ALL DATA ===
merged_df = ratings_df.merge(users_df, on="user_id", how="left")
merged_df = merged_df.merge(books_df, on="book_id", how="left")

# === LABEL ENCODING ===
le_user = LabelEncoder()
le_gender = LabelEncoder()
le_occupation = LabelEncoder()
le_city = LabelEncoder()
le_language = LabelEncoder()
le_genre = LabelEncoder()

merged_df["user_id_encoded"] = le_user.fit_transform(merged_df["user_id"])
merged_df["gender_encoded"] = le_gender.fit_transform(merged_df["gender"])
merged_df["occupation_encoded"] = le_occupation.fit_transform(merged_df["occupation"])
merged_df["city_encoded"] = le_city.fit_transform(merged_df["city"])
merged_df["language_encoded"] = le_language.fit_transform(merged_df["language"])
merged_df["book_genre_encoded"] = le_genre.fit_transform(merged_df["book_genre"])

# === FEATURES & TARGET ===
X = merged_df[[
    "user_id_encoded", "age", "gender_encoded",
    "occupation_encoded", "city_encoded", "book_genre_encoded"  # ❌ Removed language_encoded
]]

y = merged_df["book_id"]

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === TRAIN MODEL ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === EVALUATION ===
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# === SAVE MODEL ===
joblib.dump(model, os.path.join(MODEL_DIR, "recommender_model.pkl"))

# === SAVE ENCODERS ===
joblib.dump(le_user, os.path.join(MODEL_DIR, "le_user.pkl"))
joblib.dump(le_gender, os.path.join(MODEL_DIR, "le_gender.pkl"))
joblib.dump(le_occupation, os.path.join(MODEL_DIR, "le_occupation.pkl"))
joblib.dump(le_city, os.path.join(MODEL_DIR, "le_city.pkl"))
joblib.dump(le_language, os.path.join(MODEL_DIR, "le_language.pkl"))
joblib.dump(le_genre, os.path.join(MODEL_DIR, "le_genre.pkl"))  # ⬅️ renamed from le_book_genre

# === OPTIONAL: Save metadata for reference ===
with open(os.path.join(MODEL_DIR, "training_info.txt"), "w") as f:
    f.write("Model: RandomForestClassifier\n")
    f.write("Features: user_id_encoded, age, gender_encoded, occupation_encoded, city_encoded, language_encoded, book_genre_encoded\n")
    f.write("Target: book_id\n")
    f.write("Encoders: le_user, le_gender, le_occupation, le_city, le_language, le_genre\n")
    f.write("Train/Test split: 80/20\n")

print("✅ Model training and saving completed.")
