
"""
Explainability: permutation feature importance.
Optimized for classical and quantum classifiers.
"""
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Any, Dict, List
from app.classical.models import get_feature_importance


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Measure accuracy drop when individual features are shuffled."""
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2 or len(X) == 0 or len(y) != len(X):
        return {}

    rng = np.random.RandomState(random_state)
    baseline_acc = accuracy_score(y, model.predict(X))
    importances = {}

    for j, name in enumerate(feature_names):
        if j >= X.shape[1]:
            break

        drops = []

        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_acc = accuracy_score(y, model.predict(X_perm))
            drops.append(baseline_acc - perm_acc)

        importances[name] = float(np.mean(drops))

    return importances


def explain_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_type: str = "classical",
) -> Dict[str, Any]:
    """Generate feature explanations with reduced quantum computation."""
    result: Dict[str, Any] = {}
    is_quantum = model_type in ("quantum_kernel_svm", "vqc")

    if model_type in ("logistic_regression", "svm", "random_forest"):
        native = get_feature_importance(model, feature_names)
        if native:
            result["feature_importance"] = dict(
                sorted(native.items(), key=lambda item: item[1], reverse=True)
            )

    # Quantum predictions can be expensive, so use one shuffle per feature.
    repeats = 1 if is_quantum else 5

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        feature_names,
        n_repeats=repeats,
    )

    result["permutation_importance"] = dict(
        sorted(perm.items(), key=lambda item: item[1], reverse=True)
    )

    result["method_note"] = (
        f"Permutation importance: mean accuracy drop over {repeats} "
        "shuffle(s) per feature. Feature importance describes model "
        "behavior, not medical causation."
    )

    if is_quantum:
        result["quantum_note"] = (
            "Permutation importance uses the quantum prediction pipeline. "
            "One shuffle per feature is used to reduce simulation time. "
            "Importance estimates may vary between runs."
        )

    return result
