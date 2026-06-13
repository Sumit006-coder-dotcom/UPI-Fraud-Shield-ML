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


from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]

def _find_csv(csv_path=None):
    candidates = [
        csv_path,
        BASE_DIR / "dataset" / "fraud_dataset.csv",
        BASE_DIR / "fraud_dataset.csv"
    ]

    for p in candidates:
        if p and os.path.exists(str(p)):
            return str(p)

    raise FileNotFoundError(
        f"fraud_dataset.csv not found.\n"
        f"Looked in: {candidates}"
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
