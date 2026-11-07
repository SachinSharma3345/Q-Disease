"""
Quantum Kernel SVM — genuine quantum circuit simulation using NumPy.

Implements the ZZFeatureMap quantum kernel as described in:
  Havlíček et al. (2019) "Supervised learning with quantum-enhanced feature spaces"
  Nature 567, 209–212.

The kernel K(x, x') = |<φ(x)|φ(x')>|² is computed by:
  1. Encoding each sample x into a quantum statevector |φ(x)⟩ via a parameterised
     unitary circuit (ZZ-feature map).
  2. Computing the inner product of statevectors analytically.
This is exactly what Qiskit Aer's statevector_simulator does.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Quantum gate primitives (unitary matrices)
# ---------------------------------------------------------------------------

def _rz(theta: float) -> np.ndarray:
    """Single-qubit Rz rotation."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=complex)


def _rx(theta: float) -> np.ndarray:
    """Single-qubit Rx rotation."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _h() -> np.ndarray:
    """Hadamard gate."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _cx() -> np.ndarray:
    """CNOT gate (2-qubit, control=0, target=1)."""
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)


def _kron_gate(gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed single-qubit gate at position `qubit` in n_qubit system."""
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = gate
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _zz_interaction(qubit_i: int, qubit_j: int, n_qubits: int, theta: float) -> np.ndarray:
    """
    ZZ interaction: exp(-i θ/2 Z⊗Z) = CNOT (Rz(2θ) ⊗ I) CNOT.
    Applied as a 2^n × 2^n unitary on the full state space.
    """
    dim = 2 ** n_qubits
    full = np.eye(dim, dtype=complex)

    # CNOT(i → j)
    cx = _build_cnot(qubit_i, qubit_j, n_qubits)
    rz_j = _kron_gate(_rz(2 * theta), qubit_j, n_qubits)

    return cx @ rz_j @ cx


def _build_cnot(ctrl: int, tgt: int, n_qubits: int) -> np.ndarray:
    """Build full CNOT matrix for arbitrary qubit positions."""
    dim = 2 ** n_qubits
    mat = np.zeros((dim, dim), dtype=complex)
    for state in range(dim):
        bits = [(state >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
        if bits[ctrl] == 1:
            bits[tgt] ^= 1
        new_state = sum(b << (n_qubits - 1 - k) for k, b in enumerate(bits))
        mat[new_state, state] = 1
    return mat


# ---------------------------------------------------------------------------
# ZZ Feature Map statevector encoding
# ---------------------------------------------------------------------------

def _encode_sample(x: np.ndarray, n_qubits: int, reps: int = 2) -> np.ndarray:
    """
    Encode a classical feature vector x into a quantum statevector |φ(x)⟩
    using the ZZ feature map (Pauli expansion, depth=reps).

    x is truncated / zero-padded to n_qubits features.
    """
    # Clip feature vector to n_qubits dimensions
    features = np.zeros(n_qubits)
    features[:min(n_qubits, len(x))] = x[:n_qubits]

    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0...0⟩

    for _ in range(reps):
        # Layer of Hadamards
        for q in range(n_qubits):
            H = _kron_gate(_h(), q, n_qubits)
            state = H @ state

        # Layer of Rz rotations (single-qubit phase)
        for q in range(n_qubits):
            angle = 2.0 * features[q]
            Rz = _kron_gate(_rz(angle), q, n_qubits)
            state = Rz @ state

        # Layer of ZZ interactions (nearest-neighbor)
        for q in range(n_qubits - 1):
            angle = 2.0 * (np.pi - features[q]) * (np.pi - features[q + 1])
            ZZ = _zz_interaction(q, q + 1, n_qubits, angle)
            state = ZZ @ state

    return state


def _compute_kernel_matrix(X1: np.ndarray, X2: np.ndarray, n_qubits: int, reps: int = 2) -> np.ndarray:
    """
    Compute the quantum kernel matrix K[i,j] = |<φ(x1_i)|φ(x2_j)>|²
    """
    sv1 = np.array([_encode_sample(x, n_qubits, reps) for x in X1])
    sv2 = np.array([_encode_sample(x, n_qubits, reps) for x in X2])

    # K[i,j] = |sv1[i] · conj(sv2[j])|²
    overlap = np.abs(sv1 @ sv2.conj().T) ** 2
    return overlap


# ---------------------------------------------------------------------------
# QuantumKernelSVM
# ---------------------------------------------------------------------------

class QuantumKernelSVM:
    """SVM classifier using a quantum kernel computed via statevector simulation."""

    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 2,
        C: float = 1.0,
        max_train_samples: int = 150,
    ):
        self.n_qubits = n_qubits
        self.reps = reps
        self.C = C
        self.max_train_samples = max_train_samples
        self._svm = SVC(kernel="precomputed", C=C, probability=True)
        self._X_train = None
        self.train_time = 0.0
        self._subsample_idx = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        t0 = time.perf_counter()

        # Subsample for manageable kernel computation
        n = len(X_train)
        if n > self.max_train_samples:
            rng = np.random.default_rng(42)
            self._subsample_idx = rng.choice(n, size=self.max_train_samples, replace=False)
            X_sub = X_train[self._subsample_idx]
            y_sub = y_train[self._subsample_idx]
        else:
            X_sub = X_train
            y_sub = y_train
            self._subsample_idx = None

        self._X_train = X_sub
        K_train = _compute_kernel_matrix(X_sub, X_sub, self.n_qubits, self.reps)
        self._svm.fit(K_train, y_sub)
        self.train_time = time.perf_counter() - t0
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        K_test = _compute_kernel_matrix(X_test, self._X_train, self.n_qubits, self.reps)
        return self._svm.predict(K_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        K_test = _compute_kernel_matrix(X_test, self._X_train, self.n_qubits, self.reps)
        return self._svm.predict_proba(K_test)

    def get_circuit_info(self) -> Dict:
        return {
            "type": "ZZ_feature_map_statevector",
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "kernel": "quantum_kernel_|<phi(x)|phi(x')>|^2",
            "simulator": "numpy_statevector",
            "reference": "Havlicek et al. (2019) Nature 567, 209-212",
            "train_samples_used": len(self._X_train) if self._X_train is not None else 0,
        }
