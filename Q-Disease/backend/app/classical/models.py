"""Classical ML models."""
import warnings
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict


def build_classical_model(model_type: str, params: Dict[str, Any]):
    """Build a sklearn classifier."""
    rs = params.get("random_state", 42)
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=rs,
        )
    elif model_type == "svm":
        # SVC(probability=True) is deprecated in sklearn 1.9 (removed in 1.11).
        # Use CalibratedClassifierCV to get calibrated predict_proba support.
        base_svc = SVC(
            C=float(params.get("C", 1.0)),
            kernel=params.get("kernel", "rbf"),
            random_state=rs,
        )
        return CalibratedClassifierCV(base_svc, ensemble=False)
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth"),
            random_state=rs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_classical(model, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Train and return elapsed seconds."""
    start = time.perf_counter()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"scipy\.optimize: The `disp` and `iprint` options.*",
            category=DeprecationWarning,
        )
        model.fit(X_train, y_train)

    return max(time.perf_counter() - start, 1e-9)

def get_feature_importance(model, feature_names) -> Dict[str, float]:
    """Extract feature importances where available."""
    if hasattr(model, "feature_importances_"):
        return {f: float(v) for f, v in zip(feature_names, model.feature_importances_)}
    if hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        return {f: float(abs(v)) for f, v in zip(feature_names, coef)}
    return {}
