from __future__ import annotations
from typing import Any

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
    }