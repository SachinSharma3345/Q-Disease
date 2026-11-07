"""
Evaluation metrics - computed from actual confusion matrices.
Never hardcoded.
"""
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)
from typing import Any, Dict, List, Optional, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: Optional[np.ndarray] = None,
                    positive_class: int = 1,
                    label_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    sensitivity = recall for positive class
    specificity = recall for negative class
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    acc = float(accuracy_score(y_true, y_pred))
    avg = "binary" if len(classes) == 2 else "macro"

    prec = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    # Sensitivity / Specificity for binary
    sensitivity = None
    specificity = None
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = None
    roc_data = None
    if y_proba is not None:
        try:
            if len(classes) == 2:
                scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                roc_auc = float(roc_auc_score(y_true, scores))
                fpr, tpr, thresholds = roc_curve(y_true, scores)
                # Downsample to 100 points for JSON
                step = max(1, len(fpr) // 100)
                roc_data = {
                    "fpr": fpr[::step].tolist(),
                    "tpr": tpr[::step].tolist(),
                }
        except Exception:
            pass

    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_curve": roc_data,
        "n_test": int(len(y_true)),
        "positive_class": positive_class,
    }


def timed_predict(model, X) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Run prediction and return (y_pred, y_proba, inference_time_s)."""
    t0 = time.perf_counter()
    y_pred = model.predict(X)
    elapsed = time.perf_counter() - t0
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            pass
    return np.array(y_pred), y_proba, elapsed
