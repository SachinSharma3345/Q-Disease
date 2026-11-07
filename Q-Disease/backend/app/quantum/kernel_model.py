
"""
Quantum Kernel SVM using numpy statevector simulation.
Pipeline: classical preprocessing → PCA to n_qubits dims → quantum kernel → SVM
"""
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Optional

from app.quantum.simulator import compute_kernel_matrix


class QuantumKernelSVM:
    """
    Hybrid quantum-classical SVM using a simulated quantum kernel.

    Input features are PCA-reduced, scaled using training-derived anchors,
    and encoded with a ZZFeatureMap statevector simulation. A classical SVM
    is trained using the resulting kernel matrix.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        C: float = 1.0,
        feature_map: str = "zz",
        random_state: int = 42,
    ):
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if C <= 0:
            raise ValueError("C must be greater than 0")

        self.n_qubits = n_qubits
        self.C = C
        self.feature_map = feature_map
        self.random_state = random_state

        self.svc = SVC(
            kernel="precomputed",
            C=C,
            probability=True,
            random_state=random_state,
        )

        self.X_train_reduced: Optional[np.ndarray] = None
        self.scaler = StandardScaler()

        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_qubits, random_state=random_state)

        self.training_time: float = 0.0
        self._is_fitted = False

        # Scaling anchors learned from the training data.
        self._pi_min: Optional[np.ndarray] = None
        self._pi_max: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = False,
    ) -> "QuantumKernelSVM":
        t0 = time.perf_counter()

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array")
        if X_train.shape[0] < 2:
            raise ValueError("At least two training samples are required")
        if X_train.shape[1] < 1:
            raise ValueError("At least one input feature is required")
        if not np.isfinite(X_train).all():
            raise ValueError("X_train contains NaN or infinite values")
        if y_train.ndim != 1 or len(y_train) != len(X_train):
            raise ValueError("y_train must be 1D and match the number of samples")
        if np.unique(y_train).size < 2:
            raise ValueError("Quantum Kernel SVM requires at least two classes")

        # PCA cannot use more components than min(samples, features).
        n_comp = min(self.n_qubits, X_train.shape[1], X_train.shape[0])
        self.pca.set_params(n_components=n_comp)
        X_r = self.pca.fit_transform(X_train)

        # Standardize using training data only.
        self.scaler.fit(X_r)
        X_std = self.scaler.transform(X_r)

        # Store the training distribution for all future predictions.
        self._pi_min = X_std.min(axis=0)
        self._pi_max = X_std.max(axis=0)

        X_r = self._scale_to_pi(
            X_std, self._pi_min, self._pi_max
        )
        self.X_train_reduced = X_r

        if verbose:
            print(f"Computing kernel matrix {len(X_r)}×{len(X_r)}...")

        K_train = compute_kernel_matrix(
            X_r, X_r, n_comp, self.feature_map, verbose=verbose
        )
        self.svc.fit(K_train, y_train)

        self.training_time = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if (
            not self._is_fitted
            or self.X_train_reduced is None
            or self._pi_min is None
            or self._pi_max is None
        ):
            raise RuntimeError("QuantumKernelSVM must be fitted before prediction")

    def _transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("Prediction input must be a 2D array")
        if X.shape[1] != self.pca.n_features_in_:
            raise ValueError(
                f"Expected {self.pca.n_features_in_} input features, "
                f"received {X.shape[1]}"
            )
        if not np.isfinite(X).all():
            raise ValueError("Prediction input contains NaN or infinite values")

        X_r = self.pca.transform(X)
        X_std = self.scaler.transform(X_r)

        return self._scale_to_pi(X_std, self._pi_min, self._pi_max)

    @staticmethod
    def _scale_to_pi(
        X: np.ndarray,
        mn: Optional[np.ndarray] = None,
        mx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Map features to [0, π] using fixed training-derived anchors."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Scaling input must be a 2D array")
        if not np.isfinite(X).all():
            raise ValueError("Scaling input contains NaN or infinite values")

        # The batch fallback is retained for compatibility with callers
        # that use this utility directly. Prediction uses saved fit anchors.
        if mn is None or mx is None:
            mn = X.min(axis=0)
            mx = X.max(axis=0)

        mn = np.asarray(mn, dtype=float)
        mx = np.asarray(mx, dtype=float)

        rng = mx - mn
        safe_rng = np.where(rng == 0, 1.0, rng)

        scaled = (X - mn) / safe_rng * np.pi

        # Constant training features map to zero; clip extrapolated values.
        scaled = np.where(rng == 0, 0.0, scaled)
        return np.clip(scaled, 0.0, np.pi)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_r = self._transform(X_test)
        K_test = compute_kernel_matrix(
            X_r,
            self.X_train_reduced,
            self.X_train_reduced.shape[1],
            self.feature_map,
        )
        return self.svc.predict(K_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_r = self._transform(X_test)
        K_test = compute_kernel_matrix(
            X_r,
            self.X_train_reduced,
            self.X_train_reduced.shape[1],
            self.feature_map,
        )
        return self.svc.predict_proba(K_test)

    def description(self) -> Dict[str, Any]:
        return {
            "model_type": "quantum_kernel_svm",
            "n_qubits": self.n_qubits,
            "feature_map": self.feature_map,
            "svm_C": self.C,
            "note": (
                "Features are PCA-reduced and angle-encoded with a ZZFeatureMap. "
                "The kernel K(x,y)=|<φ(x)|φ(y)>|² is calculated by classical "
                "NumPy statevector simulation, followed by a classical SVM. "
                "No quantum hardware is used, and quantum advantage is not claimed."
            ),
        }
