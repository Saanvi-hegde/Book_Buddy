from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from helper_functions import log_info, log_error
import pandas as pd
from typing import Tuple


def create_and_fit_pipeline(X: pd.DataFrame) -> Pipeline:
    try:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        log_info(f"Identified categorical features: {categorical_features}")
        log_info(f"Identified numerical features: {numerical_features}")

        transformers = []

        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
            ])
            transformers.append(("cat", categorical_transformer, categorical_features))

        if numerical_features:
            numerical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", numerical_transformer, numerical_features))

        if not transformers:
            raise ValueError("No features to process. Check input dataframe.")

        preprocessor = ColumnTransformer(transformers=transformers)

        pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
        pipeline.fit(X)

        log_info("Pipeline successfully created and fitted.")
        return pipeline

    except Exception as e:
        log_error(f"Pipeline creation and fitting failed: {e}")
        raise


def save_pipeline(pipeline: Pipeline, filename: str = "pipeline.pkl") -> None:
    try:
        joblib.dump(pipeline, filename)
        log_info(f"Pipeline saved successfully as {filename}")
    except Exception as e:
        log_error(f"Failed to save pipeline: {e}")
        raise


def encode_response_variable(y: pd.Series) -> pd.Series:
    try:
        y_encoded = y.astype("category").cat.codes
        log_info("Response variable encoded successfully.")
        return y_encoded
    except Exception as e:
        log_error(f"Failed to encode target variable: {e}")
        raise


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        log_info(f"Data split into train and test sets with test size = {test_size}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        log_error(f"Failed to split data: {e}")
        raise
