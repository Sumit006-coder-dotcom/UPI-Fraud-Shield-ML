"""
Data loader for the UPI fraud dataset.

Replaces the old synthetic generator.  All training data now comes from
fraud_dataset.csv so that model accuracy reflects a real-world distribution
rather than a hand-crafted synthetic one.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_engineering import load_and_clean_dataset, get_X_y

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_HERE, "..", "..", "dataset", "fraud_dataset.csv")

# UI option lists (used by the prediction form in app.py)
SESSION_SOURCES        = ["app", "web", "sms-link"]
AUTH_METHODS           = ["pin", "biometric", "pattern"]
TRANSACTION_TYPES      = ["payment", "request", "collect"]
RELATIONSHIP_OPTIONS   = ["known", "unknown", "business"]
BUSINESS_MATCH_OPTIONS = ["match", "mismatch", "none"]
SOCIAL_OPTIONS         = ["verified", "unverified", "none"]


def _find_csv(csv_path: str = None) -> str:
    candidates = [
        csv_path,
        _DEFAULT_CSV,
        os.path.join("dataset", "fraud_dataset.csv"),
        "fraud_dataset.csv",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "fraud_dataset.csv not found. "
        "Place it at dataset/fraud_dataset.csv relative to the app directory."
    )


def load_dataset(csv_path: str = None) -> pd.DataFrame:
    return load_and_clean_dataset(_find_csv(csv_path))


def get_train_test_data(csv_path: str = None, test_size: float = 0.2, random_state: int = 42):
    df = load_dataset(csv_path)
    X, y, label_encoders, feature_cols = get_X_y(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, label_encoders, feature_cols
