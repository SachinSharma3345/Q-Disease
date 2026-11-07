"""
Variational Quantum Classifier (VQC) — numpy statevector simulation.

Architecture:
  1. Amplitude encoding of features via Ry rotations.
  2. Variational ansatz (RealAmplitudes-style: Ry + CNOT entanglement layers).
  3. Measurement of Z expectation value on qubit 0 → sigmoid → probability.
  4. Parameter optimisation using gradient-free (COBYLA-like) optimiser.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Callable
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Gate primitives
# ---------------------------------------------------------------------------

def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _kron_gate(gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    ops = [np.eye(2, dtype=complex)] * n
    ops[qubit] = gate
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _build_cnot(ctrl: int, tgt: int, n: int) -> np.ndarray:
    dim = 2 ** n
    mat = np.zeros((dim, dim), dtype=complex)
    for state in range(dim):
        bits = [(state >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[ctrl] == 1:
            bits[tgt] ^= 1
        new_state = sum(b << (n - 1 - k) for k, b in enumerate(bits))
        mat[new_state, state] = 1
    return mat


# ---------------------------------------------------------------------------
# Circuit evaluation
# ---------------------------------------------------------------------------

def _circuit(params: np.ndarray, x: np.ndarray, n_qubits: int, reps: int) -> np.ndarray:
    """
    Build and apply the VQC circuit to a single sample x.
    Returns the statevector.
    params shape: (reps, n_qubits)
    """
    params = params.reshape(reps, n_qubits)
    # Clip/pad features to n_qubits
    features = np.zeros(n_qubits)
    features[:min(n_qubits, len(x))] = x[:n_qubits]

    # Encode features into initial Ry rotations
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[0] = 1.0
    for q in range(n_qubits):
        Ry = _kron_gate(_ry(2.0 * np.arctan(features[q])), q, n_qubits)
        state = Ry @ state

    # Variational layers
    for layer in range(reps):
        # Ry rotations with trainable parameters
        for q in range(n_qubits):
            Ry = _kron_gate(_ry(params[layer, q]), q, n_qubits)
            state = Ry @ state
        # Entanglement: linear chain CNOTs
        for q in range(n_qubits - 1):
            cx = _build_cnot(q, q + 1, n_qubits)
            state = cx @ state

    return state


def _measure_z0(state: np.ndarray, n_qubits: int) -> float:
    """
    Compute <Z⊗I⊗...⊗I> expectation value (Pauli Z on qubit 0).
    Returns a value in [-1, 1].
    """
    probs = np.abs(state) ** 2
    # qubit 0 is the most-significant bit
    exp_val = 0.0
    for k, p in enumerate(probs):
        # Bit value of qubit 0 for basis state k
        bit0 = (k >> (n_qubits - 1)) & 1
        exp_val += p * (1 - 2 * bit0)  # eigenvalue: +1 for |0⟩, -1 for |1⟩
    return exp_val


def _predict_proba_sample(params: np.ndarray, x: np.ndarray, n_qubits: int, reps: int) -> float:
    """Probability of class 1 for a single sample."""
    state = _circuit(params, x, n_qubits, reps)
    z = _measure_z0(state, n_qubits)
    # Map [-1,+1] → [0,1] via linear transform
    return (1.0 - z) / 2.0


def _batch_predict_proba(params: np.ndarray, X: np.ndarray, n_qubits: int, reps: int) -> np.ndarray:
    return np.array([_predict_proba_sample(params, x, n_qubits, reps) for x in X])


def _cross_entropy_loss(params: np.ndarray, X: np.ndarray, y: np.ndarray, n_qubits: int, reps: int) -> float:
    proba = _batch_predict_proba(params, X, n_qubits, reps)
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba))


# ---------------------------------------------------------------------------
# VQC classifier
# ---------------------------------------------------------------------------

class VQC:
    """Variational Quantum Classifier with numpy statevector simulation."""

    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 2,
        max_iter: int = 100,
        max_train_samples: int = 80,
        optimizer: str = "COBYLA",
    ):
        self.n_qubits = n_qubits
        self.reps = reps
        self.max_iter = max_iter
        self.max_train_samples = max_train_samples
        self.optimizer = optimizer
        self.params_ = None
        self.train_time = 0.0
        self.loss_history: List[float] = []
        self._subsample_idx = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        t0 = time.perf_counter()

        n = len(X_train)
        if n > self.max_train_samples:
            rng = np.random.default_rng(42)
            self._subsample_idx = rng.choice(n, size=self.max_train_samples, replace=False)
            X_sub = X_train[self._subsample_idx]
            y_sub = y_train[self._subsample_idx].astype(float)
        else:
            X_sub = X_train
            y_sub = y_train.astype(float)
            self._subsample_idx = None

        n_params = self.reps * self.n_qubits
        rng = np.random.default_rng(42)
        init_params = rng.uniform(-np.pi, np.pi, size=n_params)

        callback_losses: List[float] = []

        def loss_fn(params):
            val = _cross_entropy_loss(params, X_sub, y_sub, self.n_qubits, self.reps)
            callback_losses.append(float(val))
            return val

        result = minimize(
            loss_fn,
            init_params,
            method=self.optimizer,
            options={"maxiter": self.max_iter, "rhobeg": 0.5},
        )

        self.params_ = result.x
        self.loss_history = callback_losses
        self.train_time = time.perf_counter() - t0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.params_ is None:
            raise ValueError("Model not fitted yet.")
        p1 = _batch_predict_proba(self.params_, X, self.n_qubits, self.reps)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_circuit_info(self) -> Dict:
        return {
            "type": "RealAmplitudes_VQC_statevector",
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "n_params": self.reps * self.n_qubits,
            "optimizer": self.optimizer,
            "max_iter": self.max_iter,
            "encoding": "Ry_amplitude_encoding",
            "ansatz": "RealAmplitudes_with_CNOT_entanglement",
            "measurement": "Pauli_Z_qubit0",
            "simulator": "numpy_statevector",
            "loss_history": self.loss_history[-10:] if self.loss_history else [],
        }
