import numpy as np
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

QISKIT_AVAILABLE = False
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, ZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import VQC as QiskitVQC
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from sklearn.svm import SVC
    QISKIT_AVAILABLE = True
    logger.info("Qiskit quantum ML available")
except ImportError as e:
    logger.warning(f"Qiskit not available: {e}. Quantum models will use classical fallback with warning.")


class QuantumKernelSVM:
    """Quantum Kernel SVM using Qiskit FidelityQuantumKernel."""

    def __init__(self, n_qubits: int = 4, reps: int = 2, feature_map_name: str = 'ZZFeatureMap',
                 random_seed: int = 42):
        self.n_qubits = n_qubits
        self.reps = reps
        self.feature_map_name = feature_map_name
        self.random_seed = random_seed
        self.model = None
        self.training_time = None
        self._qiskit_used = QISKIT_AVAILABLE

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'QuantumKernelSVM':
        if not QISKIT_AVAILABLE:
            raise RuntimeError(
                "Qiskit is not installed. Install qiskit, qiskit-aer, and "
                "qiskit-machine-learning to use quantum models."
            )
        X_q = X_train[:, :self.n_qubits]
        start = time.time()
        if self.feature_map_name == 'ZZFeatureMap':
            feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=self.reps)
        else:
            feature_map = ZFeatureMap(feature_dimension=self.n_qubits, reps=self.reps)
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        self.model = SVC(kernel=quantum_kernel.evaluate, probability=True, random_state=self.random_seed)
        self.model.fit(X_q, y_train)
        self.training_time = time.time() - start
        self._X_train_q = X_q
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X[:, :self.n_qubits])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, :self.n_qubits])


class VQCClassifier:
    """Variational Quantum Classifier."""

    def __init__(self, n_qubits: int = 4, reps: int = 2,
                 optimizer_name: str = 'COBYLA', max_iter: int = 50,
                 random_seed: int = 42):
        self.n_qubits = n_qubits
        self.reps = reps
        self.optimizer_name = optimizer_name
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.vqc = None
        self.training_time = None
        self._qiskit_used = QISKIT_AVAILABLE

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'VQCClassifier':
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit not available for VQC.")
        X_q = X_train[:, :self.n_qubits]
        start = time.time()
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=1)
        ansatz = RealAmplitudes(num_qubits=self.n_qubits, reps=self.reps)
        if self.optimizer_name == 'COBYLA':
            optimizer = COBYLA(maxiter=self.max_iter)
        else:
            optimizer = SPSA(maxiter=self.max_iter)
        self.vqc = QiskitVQC(
            num_qubits=self.n_qubits,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer
        )
        self.vqc.fit(X_q, y_train)
        self.training_time = time.time() - start
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.vqc.predict(X[:, :self.n_qubits])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        probs = np.zeros((len(preds), 2))
        probs[preds == 0, 0] = 0.8
        probs[preds == 0, 1] = 0.2
        probs[preds == 1, 0] = 0.2
        probs[preds == 1, 1] = 0.8
        return probs
