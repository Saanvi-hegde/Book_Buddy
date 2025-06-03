import pandas as pd
import logging
import traceback

logger = logging.getLogger(__name__)

def preprocess_user_input(user_input: dict) -> pd.DataFrame:
    """
    Prepare user input DataFrame with minimal processing needed before passing to the pipeline.
    Specifically, convert 'genre' list into a comma-separated string if needed.
    """
    try:
        logger.info("Preprocessing user input...")

        df = pd.DataFrame([user_input])

        if 'genre' in df.columns and isinstance(df['genre'].iloc[0], list):
            df['genre'] = df['genre'].apply(lambda x: ','.join(x))

        logger.info("User input preprocessing done.")
        return df

    except Exception as e:
        logger.error(f"Error preprocessing user input: {str(e)}")
        raise ValueError(f"Error preprocessing user input: {str(e)}")


def recommend_books(user_input: dict, model, label_encoder) -> pd.DataFrame:
    """
    Recommend books based on user input using the full pipeline model.

    Args:
        user_input: dict containing user features
        model: sklearn Pipeline with preprocessing and classifier
        label_encoder: fitted LabelEncoder for decoding predicted labels

    Returns:
        DataFrame of recommended books (book_id, title, author, genre, language)
    """
    try:
        logger.info("Generating book recommendations...")

        # Preprocess user input minimally (handle genres)
        input_df = preprocess_user_input(user_input)

        # Use pipeline model to predict encoded book ratings or ids
        predictions = model.predict(input_df)

        # Decode prediction labels to original book ids or rating labels (depending on what label_encoder encodes)
        top_book_ids = label_encoder.inverse_transform(predictions)

        # Load book data
        from data_loader import load_all_data
        books, _, _ = load_all_data()

        print("DEBUG: Books columns in recommend_books:", books.columns.tolist())
        print("DEBUG: Books data preview:\n", books.head())
        print("DEBUG: top_book_ids:", top_book_ids)

        # Filter books that match predicted book_ids
        # But if required columns missing, print dummy recommendations instead of error
        required_cols = {'book_id', 'title', 'author', 'genre'}
        if not required_cols.issubset(books.columns):
            print("⚠️ Required book columns missing! Showing dummy recommendations instead.")
            dummy_recs = pd.DataFrame({
                'book_id': ['B1', 'B2'],
                'title': ['Alice in Wonderland', 'The Hunger Games'],
                'author': ['Author A', 'Author B'],
                'genre': ['Fiction', 'Non-Fiction']
            })
            return dummy_recs

        recommended_books = books[books['book_id'].isin(top_book_ids)]

        logger.info("Book recommendations generated successfully.")
        return recommended_books[['book_id', 'title', 'author', 'genre']]

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        print("⚠️ An error occurred while generating recommendations.")
        print("Showing placeholder recommended books instead.")
        dummy_recs = pd.DataFrame({
            'book_id': ['B1', 'B2'],
            'title': ['Alice in Wonderland', 'The Hunger Games'],
            'author': ['Author A', 'Author B'],
            'genre': ['Fiction', 'Non-Fiction']
        })
        return dummy_recs
