import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.evaluation.metrics import compute_metrics

def test_perfect_metrics():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    m = compute_metrics(y_true, y_pred)
    assert m['accuracy'] == 1.0
    assert m['sensitivity'] == 1.0
    assert m['specificity'] == 1.0
    assert m['f1_score'] == 1.0

def test_confusion_matrix_derivation():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert m['sensitivity'] == 0.5
    assert m['specificity'] == 0.5
    assert m['accuracy'] == 0.5

def test_roc_auc():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m['roc_auc'] == 1.0
