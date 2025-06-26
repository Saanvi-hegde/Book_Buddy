import pandas as pd
import logging
import traceback
from data_loader import load_all_data

logger = logging.getLogger(__name__)

def recommend_books(user_input: dict, model, label_encoder) -> pd.DataFrame:
    """
    Recommend books based on user input using the trained model pipeline.

    Args:
        user_input: dict containing user features
        model: trained sklearn Pipeline including preprocessing and classifier
        label_encoder: fitted LabelEncoder for decoding predicted ratings

    Returns:
        DataFrame of recommended books
    """
    try:
        logger.info("Generating book recommendations...")

        input_df = pd.DataFrame([user_input])

        # Predict the rating using trained model
        predicted_label = model.predict(input_df)
        decoded_rating = label_encoder.inverse_transform(predicted_label)[0]

        # Load books data
        books, _, _ = load_all_data()

        if books.empty or 'rating' not in books.columns:
            print("⚠️ Books data missing or does not include 'rating'. Returning dummy data.")
            return pd.DataFrame({
                'book_id': ['B1', 'B2'],
                'title': ['Alice in Wonderland', 'The Hunger Games'],
                'author': ['Author A', 'Author B'],
                'genre': ['Fiction', 'Fantasy']
            })

        # Recommend books with that predicted rating
        recommended_books = books[books['rating'] == decoded_rating]

        if recommended_books.empty:
            print("⚠️ No books found with predicted rating. Showing top-rated books instead.")
            recommended_books = books.sort_values(by='rating', ascending=False).head(5)

        return recommended_books[['book_id', 'title', 'author', 'genre']]

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame({
            'book_id': ['B1', 'B2'],
            'title': ['Alice in Wonderland', 'The Hunger Games'],
            'author': ['Author A', 'Author B'],
            'genre': ['Fiction', 'Non-Fiction']
        })
