import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, os.getenv('RAW_DATA_DIR', 'raw'))

BOOKS_PATH = os.path.join(RAW_DIR, 'books.csv')
RATINGS_PATH = os.path.join(RAW_DIR, 'ratings.csv')
USERS_PATH = os.path.join(RAW_DIR, 'users.csv')

def load_books():
    try:
        df = pd.read_csv(BOOKS_PATH, skip_blank_lines=True).dropna(how='all')
        return df
    except FileNotFoundError:
        print(f"Warning: books.csv not found at {BOOKS_PATH}")
        return pd.DataFrame()  # return empty df to prevent crash

def load_ratings():
    try:
        df = pd.read_csv(RATINGS_PATH, skip_blank_lines=True).dropna(how='all')
        return df
    except FileNotFoundError:
        print(f"Warning: ratings.csv not found at {RATINGS_PATH}")
        return pd.DataFrame()

def load_users():
    try:
        df = pd.read_csv(USERS_PATH, skip_blank_lines=True).dropna(how='all')
        return df
    except FileNotFoundError:
        print(f"Warning: users.csv not found at {USERS_PATH}")
        return pd.DataFrame()

def load_all_data():
    books = load_books()
    ratings = load_ratings()
    users = load_users()

    # Strip columns if dataframes are not empty
    if not books.empty:
        books.columns = books.columns.str.strip()
    if not ratings.empty:
        ratings.columns = ratings.columns.str.strip()
    if not users.empty:
        users.columns = users.columns.str.strip()

    print("DEBUG: Loaded books columns:", books.columns.tolist() if not books.empty else "No books data")
    print("DEBUG: Loaded ratings columns:", ratings.columns.tolist() if not ratings.empty else "No ratings data")
    print("DEBUG: Loaded users columns:", users.columns.tolist() if not users.empty else "No users data")

    return books, ratings, users
