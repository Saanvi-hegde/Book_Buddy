import streamlit as st
import pandas as pd
import os
import pickle
from dotenv import load_dotenv

from helper_functions import log_info, log_error
from ml_functions import recommend_books

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR', '../Artifacts'))

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

@st.cache_data
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log_error(f"Error loading {path}: {str(e)}")
        return None

model = load_pickle(MODEL_PATH)
label_encoder = load_pickle(LABEL_ENCODER_PATH)

st.title("📚 BookBuddy")
st.markdown("Your personalized book recommender engine powered by machine learning.")

col1, col2 = st.columns(2)

with col1:
    user_id = st.text_input("Enter your User ID", "")
    age = st.slider("Select your Age", 12, 80, 25)
    gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])
    occupation = st.selectbox("Select Occupation", ["Student", "Engineer", "Teacher", "Artist", "Other"])

with col2:
    city = st.text_input("Enter your City")
    language = st.selectbox("Preferred Language", ["English", "Spanish", "Hindi", "French", "Other"])
    genre = st.multiselect("Favorite Genres", ["Fiction", "Science", "History", "Biography", "Comics", "Mystery"])

if st.button("📖 Recommend Books"):
    if not model or not label_encoder:
        st.error("Error: Model artifacts not properly loaded.")
    elif not user_id:
        st.warning("Please enter User ID.")
    else:
        # Pass only user features expected by the model
        user_input = {
            "user_id": user_id,
            "age": age,
            "gender": gender,
            "occupation": occupation,
            "city": city
        }
        try:
            recommendations = recommend_books(user_input, model, label_encoder)
            if recommendations.empty:
                st.warning("No recommendations found for the given preferences.")
            else:
                st.success("Top Book Recommendations:")
                st.dataframe(recommendations)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
