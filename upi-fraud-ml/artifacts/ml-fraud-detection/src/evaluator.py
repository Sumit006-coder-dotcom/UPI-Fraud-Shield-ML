import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision":       round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall":          round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1 Score":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "AUC-ROC":         round(float(roc_auc_score(y_true, y_prob)), 4),
        "Avg Precision":   round(float(average_precision_score(y_true, y_prob)), 4),
        "True Positives":  int(tp),
        "True Negatives":  int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
    }


def get_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist()


def get_pr_curve(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision.tolist(), recall.tolist(), thresholds.tolist()


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).tolist()


def metrics_dataframe(all_metrics: dict) -> pd.DataFrame:
    rows = [
        {
            "Model":        name,
            "Accuracy":     m["Accuracy"],
            "Precision":    m["Precision"],
            "Recall":       m["Recall"],
            "F1 Score":     m["F1 Score"],
            "AUC-ROC":      m["AUC-ROC"],
        }
        for name, m in all_metrics.items()
    ]
    return pd.DataFrame(rows).set_index("Model")
