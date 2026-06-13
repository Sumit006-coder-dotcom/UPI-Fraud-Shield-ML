from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


MODEL_REGISTRY = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=42),
        "needs_scaling": True,
        "color": "#636EFA",
        "description": "Baseline linear model. Fast and interpretable. Uses log-odds to score fraud risk.",
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42),
        "needs_scaling": False,
        "color": "#EF553B",
        "description": "Rule-based tree. Highly interpretable. max_depth=6 prevents overfitting.",
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "needs_scaling": False,
        "color": "#00CC96",
        "description": "Ensemble of 200 trees. Robust to outliers. Best overall trade-off on fraud data.",
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4,
            min_samples_leaf=15, subsample=0.8, random_state=42
        ),
        "needs_scaling": False,
        "color": "#AB63FA",
        "description": "Sequential boosting. High accuracy on tabular data. Subsample=0.8 reduces overfitting.",
    },
    "Neural Network": {
        "model": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), max_iter=300, alpha=0.01,
            random_state=42, early_stopping=True, validation_fraction=0.15
        ),
        "needs_scaling": True,
        "color": "#FFA15A",
        "description": "3-layer MLP. Learns non-linear patterns. alpha=0.01 regularisation prevents overfit.",
    },
}


def train_model(name: str, X_train, y_train, scaler: StandardScaler = None):
    import copy
    cfg = MODEL_REGISTRY[name]
    model = copy.deepcopy(cfg["model"])
    X = scaler.transform(X_train) if cfg["needs_scaling"] and scaler else X_train
    model.fit(X, y_train)
    return model


def predict_model(name: str, model, X_test, scaler: StandardScaler = None):
    cfg = MODEL_REGISTRY[name]
    X = scaler.transform(X_test) if cfg["needs_scaling"] and scaler else X_test
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


def get_feature_importance(name: str, model, feature_names: list) -> dict:
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    if importance is None:
        return {}
    return dict(sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True))
