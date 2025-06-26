import streamlit as st
import pandas as pd
import os
from joblib import load

# === PATH SETUP ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "Scripts", "models")
RAW_DIR = os.path.join(BASE_DIR, "Data", "raw")

# === MODEL & ENCODER PATHS ===
MODEL_PATH = os.path.join(MODEL_DIR, "recommender_model.pkl")
LE_USER_PATH = os.path.join(MODEL_DIR, "le_user.pkl")
LE_GENDER_PATH = os.path.join(MODEL_DIR, "le_gender.pkl")
LE_OCCUPATION_PATH = os.path.join(MODEL_DIR, "le_occupation.pkl")
LE_CITY_PATH = os.path.join(MODEL_DIR, "le_city.pkl")
LE_GENRE_PATH = os.path.join(MODEL_DIR, "le_book_genre.pkl")  # ✅ Rename matched to training

# === LOAD MODEL ===
try:
    model = load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# === LOAD ENCODERS ===
try:
    le_user = load(LE_USER_PATH)
    le_gender = load(LE_GENDER_PATH)
    le_occupation = load(LE_OCCUPATION_PATH)
    le_city = load(LE_CITY_PATH)
    le_genre = load(LE_GENRE_PATH)
except Exception as e:
    st.error(f"❌ Could not load encoders: {e}")
    st.stop()

# === LOAD RAW DATA ===
try:
    users_df = pd.read_csv(os.path.join(RAW_DIR, "users.csv"))
    users_df = users_df.rename(columns={"genre": "preferred_genre"})
    books_df = pd.read_csv(os.path.join(RAW_DIR, "books.csv"))
    books_df = books_df.rename(columns={"genre": "book_genre"})  # ✅ Match training step
except Exception as e:
    st.error(f"❌ Could not load data files: {e}")
    st.stop()

# === STREAMLIT UI ===
st.title("📚 BookBuddy - Smart Book Recommender")
st.sidebar.header("Enter User Details")

user_id = st.sidebar.selectbox("User ID", users_df["user_id"].unique())
age = st.sidebar.slider("Age", 10, 80, 25)
gender = st.sidebar.selectbox("Gender", users_df["gender"].unique())
occupation = st.sidebar.selectbox("Occupation", users_df["occupation"].unique())
city = st.sidebar.selectbox("City", users_df["city"].unique())
genre = st.sidebar.selectbox("Preferred Genre", users_df["preferred_genre"].unique())

if st.sidebar.button("📖 Recommend Book"):
    try:
        input_data = {
            "user_id_encoded": le_user.transform([user_id])[0],
            "age": age,
            "gender_encoded": le_gender.transform([gender])[0],
            "occupation_encoded": le_occupation.transform([occupation])[0],
            "city_encoded": le_city.transform([city])[0],
            "book_genre_encoded": le_genre.transform([genre])[0]
        }

        input_df = pd.DataFrame([input_data])
        predicted_book_id = model.predict(input_df)[0]

        book = books_df[books_df["book_id"] == int(predicted_book_id)]

        if not book.empty:
            book = book.iloc[0]
            st.success(f"✅ We recommend: **{book['title']}** by *{book['author']}*")
            st.info(f"Genre: {book['book_genre']}")
        else:
            st.warning("⚠️ No matching book found for the predicted ID.")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
