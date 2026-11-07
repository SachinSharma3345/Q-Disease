import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.classical.models import build_classical_model, train_classical
from app.evaluation.metrics import compute_metrics


@pytest.fixture
def training_data():
    data = load_breast_cancer()
    X, y = data.data[:400], data.target[:400]
    return X[:320], X[320:], y[:320], y[320:]


def test_logistic_regression(training_data):
    X_train, X_test, y_train, y_test = training_data
    model = build_classical_model('logistic_regression', {})
    training_time = train_classical(model, X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert training_time > 0


def test_random_forest(training_data):
    X_train, X_test, y_train, y_test = training_data
    model = build_classical_model('random_forest', {'n_estimators': 10})
    train_classical(model, X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_svm(training_data):
    X_train, X_test, y_train, y_test = training_data
    model = build_classical_model('svm', {})
    train_classical(model, X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
