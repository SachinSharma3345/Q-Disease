
"""
Variational Quantum Classifier using numpy statevector simulation.
Trained via ADAM gradient descent using finite-difference gradients.
"""
import time
import numpy as np
from typing import Any, Dict, Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from app.quantum.simulator import vqc_predict_proba


class VQCClassifier:
    """
    Variational Quantum Classifier using classical statevector simulation.

    Pipeline:
    1. PCA reduces input features to n_qubits dimensions.
    2. StandardScaler is fitted on the training subset.
    3. Fixed training-derived anchors map features to [0, pi].
    4. A parameterized circuit is optimized using ADAM and finite differences.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 2,
        feature_map: str = "zz",
        optimizer: str = "adam",
        max_iter: int = 50,
        learning_rate: float = 0.01,
        random_state: int = 42,
        max_train_samples: int = 80,
    ):
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if reps < 0:
            raise ValueError("reps must be non-negative")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if max_train_samples < 2:
            raise ValueError("max_train_samples must be at least 2")

        self.n_qubits = n_qubits
        self.reps = reps
        self.feature_map = feature_map
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_train_samples = max_train_samples

        self.pca = PCA(n_components=n_qubits, random_state=random_state)
        self.scaler = StandardScaler()

        self.params: Optional[np.ndarray] = None
        self.loss_history: list = []
        self.training_time: float = 0.0
        self._is_fitted = False
        self._single_class = False

        # Fixed angle-scaling anchors learned from training data.
        self._pi_min: Optional[np.ndarray] = None
        self._pi_max: Optional[np.ndarray] = None

    def _n_params(self) -> int:
        return self.n_qubits * (self.reps + 1)

    def _loss(
        self, X: np.ndarray, y: np.ndarray, params: np.ndarray
    ) -> float:
        proba = vqc_predict_proba(
            X, params, self.n_qubits, self.feature_map, self.reps
        )
        eps = 1e-8
        p = np.clip(proba[:, 1], eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _grad(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray,
        eps: float = 0.1,
    ) -> np.ndarray:
        """Central finite-difference approximation to the loss gradient."""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += eps

            p_minus = params.copy()
            p_minus[i] -= eps

            grad[i] = (
                self._loss(X, y, p_plus) - self._loss(X, y, p_minus)
            ) / (2 * eps)

        return grad

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> "VQCClassifier":
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

        classes = np.unique(y_train)
        if len(classes) == 0:
            raise ValueError("y_train must contain at least one class.")

        # A single-class dataset cannot train a binary decision boundary.
        # Keep a constant-class fallback so downstream evaluation remains finite.
        if len(classes) == 1:
            self.classes_ = classes
            self._single_class = True

            n_comp = min(self.n_qubits, X_train.shape[1], X_train.shape[0])
            self.pca.set_params(n_components=n_comp)
            X_r = self.pca.fit_transform(X_train)

            self.scaler.fit(X_r)
            X_std = self.scaler.transform(X_r)
            self._pi_min = X_std.min(axis=0)
            self._pi_max = X_std.max(axis=0)

            self.n_qubits = n_comp
            self.params = None
            self.loss_history = []
            self.training_time = time.perf_counter() - t0
            self._is_fitted = True
            return self

        # Map arbitrary binary labels to 0 and 1, and retain the mapping.
        self._single_class = False
        self.classes_ = classes
        y_binary = (y_train == classes[1]).astype(float)

        rng = np.random.RandomState(self.random_state)

        # Subsample while ensuring both classes are represented.
        n = min(self.max_train_samples, len(X_train))
        class_0_idx = np.flatnonzero(y_binary == 0)
        class_1_idx = np.flatnonzero(y_binary == 1)

        if n < 2:
            raise ValueError("At least two training samples are required.")

        # Allocate at least one sample to each class.
        n_0 = min(
            max(1, int(round(n * len(class_0_idx) / len(y_binary)))),
            len(class_0_idx),
            n - 1,
        )
        n_1 = min(n - n_0, len(class_1_idx))

        # Fill any remaining slots from available samples.
        remaining = n - n_0 - n_1
        if remaining > 0:
            extra_0 = min(remaining, len(class_0_idx) - n_0)
            n_0 += extra_0
            remaining -= extra_0

        if remaining > 0:
            extra_1 = min(remaining, len(class_1_idx) - n_1)
            n_1 += extra_1
            remaining -= extra_1

        idx_0 = rng.choice(class_0_idx, n_0, replace=False)
        idx_1 = rng.choice(class_1_idx, n_1, replace=False)
        idx = np.concatenate([idx_0, idx_1])
        rng.shuffle(idx)

        X_s, y_s = X_train[idx], y_binary[idx]

        if len(np.unique(y_s)) != 2:
            raise ValueError("Training subset must contain both classes.")

        # PCA components cannot exceed min(samples, features).
        n_comp = min(self.n_qubits, X_s.shape[1], X_s.shape[0])
        self.pca.set_params(n_components=n_comp)
        X_r = self.pca.fit_transform(X_s)

        self.scaler.fit(X_r)
        X_std = self.scaler.transform(X_r)

        # Save anchors from the training subset and reuse them at prediction.
        self._pi_min = X_std.min(axis=0)
        self._pi_max = X_std.max(axis=0)
        X_r = self._scale(X_std, self._pi_min, self._pi_max)

        # Keep circuit dimensionality consistent with PCA output.
        self.n_qubits = n_comp

        self.params = rng.uniform(-np.pi, np.pi, self._n_params())

        # ADAM optimizer.
        lr = self.learning_rate
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        m = np.zeros_like(self.params)
        v = np.zeros_like(self.params)

        self.loss_history = []

        for t in range(1, self.max_iter + 1):
            loss = self._loss(X_r, y_s, self.params)
            self.loss_history.append(float(loss))

            g = self._grad(X_r, y_s, self.params)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            self.params -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            if t % 10 == 0:
                print(
                    f"  VQC iter {t}/{self.max_iter}, loss={loss:.4f}"
                )

        self.training_time = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if (
            not self._is_fitted
            or (self.params is None and not self._single_class)
            or self._pi_min is None
            or self._pi_max is None
        ):
            raise RuntimeError("VQCClassifier must be fitted before prediction")

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

        return self._scale(X_std, self._pi_min, self._pi_max)

    @staticmethod
    def _scale(
        X: np.ndarray,
        mn: Optional[np.ndarray] = None,
        mx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Map features to [0, pi] using fixed training-derived anchors."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Scaling input must be a 2D array")
        if not np.isfinite(X).all():
            raise ValueError("Scaling input contains NaN or infinite values")

        # Retain a fallback for direct utility callers. Prediction always
        # supplies the anchors saved by fit().
        if mn is None or mx is None:
            mn = X.min(axis=0)
            mx = X.max(axis=0)

        mn = np.asarray(mn, dtype=float)
        mx = np.asarray(mx, dtype=float)
        rng = mx - mn
        safe_rng = np.where(rng == 0, 1.0, rng)

        scaled = (X - mn) / safe_rng * np.pi
        scaled = np.where(rng == 0, 0.0, scaled)

        return np.clip(scaled, 0.0, np.pi)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._single_class:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            self._transform(X)  # Validate dimensions, values, and fitted state.

            probabilities = np.zeros((len(X), 2), dtype=float)
            class_label = self.classes_[0]
            # Probability columns correspond to class 0 and class 1.
            if class_label == 0:
                probabilities[:, 0] = 1.0
            else:
                probabilities[:, 1] = 1.0
            return probabilities

        X_r = self._transform(X)

        # The simulator returns probabilities for binary encoded classes 0, 1.
        return vqc_predict_proba(
            X_r, self.params, self.n_qubits, self.feature_map, self.reps
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if self._single_class:
            return np.full(probabilities.shape[0], self.classes_[0])
        encoded = (probabilities[:, 1] >= 0.5).astype(int)
        return self.classes_[encoded]

    def description(self) -> Dict[str, Any]:
        return {
            "model_type": "vqc",
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "feature_map": self.feature_map,
            "optimizer": self.optimizer,
            "max_iter": self.max_iter,
            "n_parameters": self._n_params(),
            "note": (
                "Features are PCA-reduced and encoded with a ZZFeatureMap. "
                "A parameterized ansatz is optimized using ADAM and "
                "finite-difference gradients on a classical NumPy statevector "
                "simulation. No quantum hardware is used, and quantum "
                "advantage is not claimed."
            ),
        }
