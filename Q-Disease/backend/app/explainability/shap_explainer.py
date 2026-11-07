import numpy as np
import logging
from typing import Dict, Any, List
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not available, using permutation importance only")


def compute_feature_importance(model, X_test: np.ndarray, y_test: np.ndarray,
                                feature_names: List[str], model_type: str,
                                random_seed: int = 42) -> Dict[str, Any]:
    result = {
        'method': None,
        'feature_names': feature_names[:X_test.shape[1]],
        'importance_values': [],
        'is_quantum': model_type in ['quantum_kernel_svm', 'vqc'],
    }

    if model_type in ['quantum_kernel_svm', 'vqc']:
        try:
            perm = permutation_importance(model, X_test, y_test,
                                          n_repeats=10, random_state=random_seed)
            result['method'] = 'permutation_importance'
            result['importance_values'] = perm.importances_mean.tolist()
            result['note'] = (
                "Permutation-based feature importance computed using the full quantum "
                "prediction pipeline. Indicates model sensitivity to each feature."
            )
        except Exception as e:
            logger.error(f"Permutation importance failed: {e}")
            result['method'] = 'unavailable'
            result['importance_values'] = [0.0] * len(result['feature_names'])

    elif SHAP_AVAILABLE and model_type in ['logistic_regression', 'random_forest']:
        try:
            if model_type == 'random_forest':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):
                    importances = np.abs(shap_values[1]).mean(axis=0).tolist()
                else:
                    importances = np.abs(shap_values).mean(axis=0).tolist()
            else:
                explainer = shap.LinearExplainer(model, X_test)
                shap_values = explainer.shap_values(X_test)
                importances = np.abs(shap_values).mean(axis=0).tolist()
            result['method'] = 'shap'
            result['importance_values'] = importances
        except Exception as e:
            logger.warning(f"SHAP failed, using permutation: {e}")
            perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=random_seed)
            result['method'] = 'permutation_importance'
            result['importance_values'] = perm.importances_mean.tolist()
    else:
        try:
            if hasattr(model, 'feature_importances_'):
                result['method'] = 'model_feature_importance'
                result['importance_values'] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                result['method'] = 'coefficient_magnitude'
                result['importance_values'] = np.abs(model.coef_[0]).tolist()
            else:
                perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=random_seed)
                result['method'] = 'permutation_importance'
                result['importance_values'] = perm.importances_mean.tolist()
        except Exception as e:
            logger.error(f"Feature importance failed: {e}")
            result['method'] = 'unavailable'
            result['importance_values'] = [0.0] * len(result['feature_names'])

    n = len(result['feature_names'])
    if len(result['importance_values']) > n:
        result['importance_values'] = result['importance_values'][:n]
    elif len(result['importance_values']) < n:
        result['importance_values'].extend([0.0] * (n - len(result['importance_values'])))

    return result
