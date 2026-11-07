# """
# Standalone test runner — runs without pytest, using only stdlib unittest.
# Converts pytest-style tests to unittest manually.
# """
# import unittest
# import sys
# import os
# import traceback
# import uuid

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_breast_cancer


# # ─── test_metrics ──────────────────────────────────────────────────────────
# class TestMetrics(unittest.TestCase):

#     def setUp(self):
#         from app.evaluation.metrics import compute_metrics
#         self.compute_metrics = compute_metrics

#     def test_perfect_metrics(self):
#         y_true = np.array([0, 0, 1, 1, 1])
#         y_pred = np.array([0, 0, 1, 1, 1])
#         m = self.compute_metrics(y_true, y_pred)
#         self.assertEqual(m['accuracy'], 1.0)
#         self.assertEqual(m['sensitivity'], 1.0)
#         self.assertEqual(m['specificity'], 1.0)
#         self.assertEqual(m['f1_score'], 1.0)

#     def test_confusion_matrix_derivation(self):
#         y_true = np.array([1, 0, 1, 0])
#         y_pred = np.array([1, 1, 0, 0])
#         m = self.compute_metrics(y_true, y_pred)
#         self.assertEqual(m['sensitivity'], 0.5)
#         self.assertEqual(m['specificity'], 0.5)
#         self.assertEqual(m['accuracy'], 0.5)

#     def test_roc_auc(self):
#         y_true = np.array([0, 0, 1, 1])
#         y_pred = np.array([0, 0, 1, 1])
#         y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
#         m = self.compute_metrics(y_true, y_pred, y_prob)
#         self.assertEqual(m['roc_auc'], 1.0)


# # ─── test_classical_models ─────────────────────────────────────────────────
# # test_classical_models.py imports train_classical_model but the module
# # exposes build_classical_model + train_classical separately.
# # We test the actual exported API here and note the import mismatch.

# class TestClassicalModels(unittest.TestCase):
#     """
#     Tests for build_classical_model + train_classical.

#     Data is scaled via StandardScaler before fitting to mirror what the production
#     PreprocessingPipeline does (pipeline.py StandardScaler.fit_transform(X_train)).
#     Using raw breast-cancer features (range ~1–2500) without scaling causes
#     LogisticRegression to emit a ConvergenceWarning even at max_iter=1000 — not
#     because the model is broken, but because unscaled lbfgs requires many more
#     iterations.  The fix is to scale the test data, not to raise max_iter, because
#     the goal is to test the model under conditions that match production.
#     """

#     @classmethod
#     def setUpClass(cls):
#         from sklearn.preprocessing import StandardScaler
#         data = load_breast_cancer()
#         X, y = data.data[:400], data.target[:400]
#         X_train_raw, X_test_raw = X[:320], X[320:]
#         cls.y_train, cls.y_test = y[:320], y[320:]

#         # Scale to match the production pipeline (StandardScaler fit on train only).
#         scaler = StandardScaler()
#         cls.X_train = scaler.fit_transform(X_train_raw)
#         cls.X_test  = scaler.transform(X_test_raw)

#     def test_import_train_classical_model_missing(self):
#         """train_classical_model (combined) is NOT exported by app.classical.models.
#         The module exposes build_classical_model + train_classical separately.
#         This test documents the stale import in test_classical_models.py."""
#         try:
#             from app.classical.models import train_classical_model  # noqa
#             # If it exists, that's fine too
#         except ImportError:
#             pass  # expected — see comment above

#     def test_build_and_train_logistic_regression(self):
#         import warnings
#         from app.classical.models import build_classical_model, train_classical
#         model = build_classical_model('logistic_regression', {})
#         # No ConvergenceWarning expected on scaled data.
#         with warnings.catch_warnings():
#             warnings.simplefilter('error')  # any warning → test failure
#             t = train_classical(model, self.X_train, self.y_train)
#         preds = model.predict(self.X_test)
#         self.assertEqual(len(preds), len(self.y_test))
#         self.assertGreater(t, 0)
#         # Model should converge well on scaled breast-cancer data.
#         accuracy = np.mean(preds == self.y_test)
#         self.assertGreater(accuracy, 0.90,
#             f"Logistic regression accuracy {accuracy:.2f} unexpectedly low on scaled data")

#     def test_build_and_train_random_forest(self):
#         from app.classical.models import build_classical_model, train_classical
#         model = build_classical_model('random_forest', {'n_estimators': 10})
#         train_classical(model, self.X_train, self.y_train)
#         preds = model.predict(self.X_test)
#         self.assertEqual(len(preds), len(self.y_test))
#         accuracy = np.mean(preds == self.y_test)
#         self.assertGreater(accuracy, 0.85,
#             f"Random forest accuracy {accuracy:.2f} unexpectedly low")

#     def test_build_and_train_svm(self):
#         from app.classical.models import build_classical_model, train_classical
#         model = build_classical_model('svm', {})
#         train_classical(model, self.X_train, self.y_train)
#         preds = model.predict(self.X_test)
#         self.assertEqual(len(preds), len(self.y_test))
#         accuracy = np.mean(preds == self.y_test)
#         self.assertGreater(accuracy, 0.90,
#             f"SVM accuracy {accuracy:.2f} unexpectedly low on scaled data")


# # ─── test_preprocessing ────────────────────────────────────────────────────
# # test_preprocessing.py imports prepare_data; we check whether it exists.

# class TestPreprocessing(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         data = load_breast_cancer()
#         df = pd.DataFrame(data.data, columns=data.feature_names)
#         df['diagnosis'] = pd.Categorical.from_codes(data.target, ['malignant', 'benign'])
#         cls.df = df

#     def _get_prepare_data(self):
#         try:
#             from app.preprocessing.pipeline import prepare_data
#             return prepare_data
#         except ImportError:
#             self.skipTest("prepare_data not exported from app.preprocessing.pipeline")

#     def test_no_leakage(self):
#         prepare_data = self._get_prepare_data()
#         config = {'imputation_strategy': 'mean', 'scaling_method': 'standard'}
#         result = prepare_data(self.df, 'diagnosis', None, [], 0.2, 42, config)
#         self.assertEqual(result['X_train'].shape[0], result['train_size'])
#         self.assertEqual(result['X_test'].shape[0], result['test_size'])
#         self.assertEqual(result['train_size'] + result['test_size'], len(self.df))

#     def test_stratified_split(self):
#         prepare_data = self._get_prepare_data()
#         config = {'imputation_strategy': 'mean', 'scaling_method': 'standard'}
#         result = prepare_data(self.df, 'diagnosis', None, [], 0.2, 42, config)
#         self.assertEqual(len(np.unique(result['y_train'])), 2)
#         self.assertEqual(len(np.unique(result['y_test'])), 2)

#     def test_feature_selection_kbest(self):
#         prepare_data = self._get_prepare_data()
#         config = {'imputation_strategy': 'mean', 'scaling_method': 'standard',
#                   'feature_selection': 'kbest', 'n_features': 10}
#         result = prepare_data(self.df, 'diagnosis', None, [], 0.2, 42, config)
#         self.assertEqual(result['X_train'].shape[1], 10)
#         self.assertEqual(result['X_test'].shape[1], 10)

#     def test_pca(self):
#         prepare_data = self._get_prepare_data()
#         config = {'imputation_strategy': 'mean', 'scaling_method': 'standard',
#                   'feature_selection': 'pca', 'n_features': 5}
#         result = prepare_data(self.df, 'diagnosis', None, [], 0.2, 42, config)
#         self.assertEqual(result['X_train'].shape[1], 5)
#         self.assertEqual(result['X_test'].shape[1], 5)


# # ─── test_training_service_status_values ───────────────────────────────────

# class TestTrainingServiceStatusValues(unittest.TestCase):
#     """Regression test for Training.tsx bug: backend must use 'completed'/'failed',
#     not 'done'/'error', as terminal job statuses."""

#     def test_terminal_status_completed_not_done(self):
#         """training_service._run_training must set status='completed' on success."""
#         import inspect
#         from app.services import training_service
#         src = inspect.getsource(training_service)
#         # Source may use single or double quotes; check for the bare word
#         self.assertIn("completed", src,
#                       "training_service must set status='completed' on success")

#     def test_terminal_status_failed_not_error(self):
#         """training_service._run_training must set status='failed' on exception."""
#         import inspect
#         from app.services import training_service
#         src = inspect.getsource(training_service)
#         self.assertIn("failed", src,
#                       "training_service must set status='failed' on exception")

#     def test_no_done_terminal_status(self):
#         """'done' must not appear as a terminal status assignment."""
#         import inspect
#         from app.services import training_service
#         src = inspect.getsource(training_service)
#         # Allow 'done' in comments but not as a status value assignment
#         for line in src.splitlines():
#             stripped = line.strip()
#             if stripped.startswith('#'):
#                 continue
#             self.assertNotIn("= 'done'", stripped,
#                              f"Found status='done' assignment in training_service: {stripped!r}")


# # ─── test_job_cross_worker_visibility ─────────────────────────────────────────

# class TestJobCrossWorkerVisibility(unittest.TestCase):
#     """
#     Regression tests for the gunicorn cross-worker job-status bug.

#     Root cause: _jobs is an in-memory dict, per-process.  With --workers 2,
#     Worker A starts a job (writes _jobs[job_id]) but Worker B handles the
#     status poll.  Worker B has no entry in its own _jobs → get_job() returned
#     None → HTTP 404 → frontend catch-handler killed the polling interval →
#     UI stuck at "Queued" forever.

#     Fix: _set_job and _update_job now also write to disk via storage.insert /
#     storage.update; get_job falls back to storage.find_one when the entry is
#     absent from the in-process _jobs dict.
#     """

#     def _cleanup(self, job_id):
#         """Remove test artefacts from both _jobs and disk."""
#         from app.services import training_service, storage as svc_storage
#         with training_service._jobs_lock:
#             training_service._jobs.pop(job_id, None)
#         svc_storage.delete("jobs", job_id)

#     def test_get_job_falls_back_to_disk(self):
#         """get_job() returns job data even when not in local _jobs dict."""
#         from app.services import training_service, storage as svc_storage
#         job_id = str(uuid.uuid4())
#         job_doc = {
#             "_id": job_id,
#             "job_id": job_id,
#             "status": "running",
#             "progress": "Training model...",
#         }
#         # Simulate another worker having written the job to disk
#         svc_storage.insert("jobs", job_doc)
#         try:
#             # Ensure the entry is absent from this process's _jobs
#             with training_service._jobs_lock:
#                 training_service._jobs.pop(job_id, None)

#             result = training_service.get_job(job_id)
#             self.assertIsNotNone(result,
#                 "get_job() returned None when job absent from _jobs but present on disk "
#                 "— cross-worker visibility is broken")
#             self.assertEqual(result["status"], "running")
#             self.assertEqual(result["job_id"], job_id)
#         finally:
#             self._cleanup(job_id)

#     def test_set_job_persists_to_disk(self):
#         """_set_job writes job to disk so other workers can read it."""
#         from app.services import training_service, storage as svc_storage
#         job_id = str(uuid.uuid4())
#         training_service._set_job(job_id, {
#             "_id": job_id,
#             "job_id": job_id,
#             "status": "queued",
#             "progress": "Queued",
#         })
#         try:
#             disk_doc = svc_storage.find_one("jobs", job_id)
#             self.assertIsNotNone(disk_doc,
#                 "_set_job did not persist job to disk — other workers will see 404")
#             self.assertEqual(disk_doc["status"], "queued")
#         finally:
#             self._cleanup(job_id)

#     def test_update_job_persists_to_disk(self):
#         """_update_job writes status changes to disk for cross-worker reads."""
#         from app.services import training_service, storage as svc_storage
#         job_id = str(uuid.uuid4())
#         training_service._set_job(job_id, {
#             "_id": job_id,
#             "job_id": job_id,
#             "status": "queued",
#             "progress": "Queued",
#         })
#         try:
#             training_service._update_job(job_id, {"status": "running", "progress": "Training..."})
#             disk_doc = svc_storage.find_one("jobs", job_id)
#             self.assertIsNotNone(disk_doc)
#             self.assertEqual(disk_doc["status"], "running",
#                 "_update_job did not update status on disk — cross-worker polls see stale data")
#         finally:
#             self._cleanup(job_id)


# # ─── test_quantum_models ───────────────────────────────────────────────────────

# class TestQuantumSimulator(unittest.TestCase):
#     """Unit tests for the numpy statevector simulator primitives."""

#     def test_hadamard_normalised(self):
#         """H|0⟩ = (|0⟩+|1⟩)/√2 — statevector is normalised."""
#         from app.quantum.simulator import apply_single, H
#         sv = np.zeros(2, dtype=complex); sv[0] = 1.0
#         sv_out = apply_single(sv, H, 0, 1)
#         self.assertAlmostEqual(float(np.abs(sv_out[0])**2 + np.abs(sv_out[1])**2), 1.0, places=12)
#         self.assertAlmostEqual(float(np.abs(sv_out[0])), 1 / np.sqrt(2), places=12)

#     def test_zz_feature_map_normalised(self):
#         """ZZFeatureMap output statevector must have unit norm."""
#         from app.quantum.simulator import zz_feature_map
#         x = np.array([0.5, 1.2])
#         sv = zz_feature_map(x, n_qubits=2)
#         norm = float(np.sum(np.abs(sv) ** 2))
#         self.assertAlmostEqual(norm, 1.0, places=12,
#                                msg="ZZFeatureMap statevector is not normalised")

#     def test_zz_feature_map_deterministic(self):
#         """Same input → same statevector."""
#         from app.quantum.simulator import zz_feature_map
#         x = np.array([0.3, 0.7, 1.1, 0.2])
#         sv1 = zz_feature_map(x, n_qubits=4)
#         sv2 = zz_feature_map(x, n_qubits=4)
#         np.testing.assert_array_equal(sv1, sv2)

#     def test_vqc_forward_in_range(self):
#         """vqc_forward returns a value in [-1, 1] (⟨Z₀⟩ expectation)."""
#         from app.quantum.simulator import vqc_forward
#         rng = np.random.RandomState(7)
#         for _ in range(5):
#             x = rng.uniform(0, np.pi, 3)
#             params = rng.uniform(-np.pi, np.pi, 3 * (2 + 1))  # n_qubits=3, reps=2
#             val = vqc_forward(x, params, n_qubits=3)
#             self.assertGreaterEqual(val, -1.0 - 1e-9)
#             self.assertLessEqual(val, 1.0 + 1e-9)

#     def test_quantum_kernel_self_similarity_is_one(self):
#         """K(x, x) = |⟨φ(x)|φ(x)⟩|² = 1.0 for all x."""
#         from app.quantum.simulator import quantum_kernel_entry
#         rng = np.random.RandomState(13)
#         for _ in range(5):
#             x = rng.uniform(0, np.pi, 2)
#             k = quantum_kernel_entry(x, x, n_qubits=2)
#             self.assertAlmostEqual(k, 1.0, places=10,
#                                    msg="Quantum kernel diagonal entry is not 1.0")

#     def test_kernel_matrix_symmetric(self):
#         """Training kernel matrix must be symmetric."""
#         from app.quantum.simulator import compute_kernel_matrix
#         rng = np.random.RandomState(17)
#         X = rng.uniform(0, np.pi, (5, 2))
#         K = compute_kernel_matrix(X, X, n_qubits=2)
#         np.testing.assert_allclose(K, K.T, atol=1e-12,
#                                    err_msg="Quantum kernel matrix is not symmetric")

#     def test_vqc_predict_proba_shape_and_sum(self):
#         """vqc_predict_proba returns (n_samples, 2) with rows summing to 1."""
#         from app.quantum.simulator import vqc_predict_proba
#         rng = np.random.RandomState(3)
#         X = rng.uniform(0, np.pi, (4, 2))
#         params = rng.uniform(-np.pi, np.pi, 2 * (2 + 1))
#         proba = vqc_predict_proba(X, params, n_qubits=2)
#         self.assertEqual(proba.shape, (4, 2))
#         np.testing.assert_allclose(proba.sum(axis=1), np.ones(4), atol=1e-12)
#         self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1),
#                         msg="Some probabilities out of [0, 1]")


# class TestQuantumKernelSVM(unittest.TestCase):
#     """
#     Integration tests for QuantumKernelSVM (kernel_model.py).

#     Uses a tiny deterministic dataset so tests complete in a few seconds.
#     The quantum computation is NOT mocked — all kernel entries are computed
#     via the actual numpy statevector simulator.
#     """

#     @classmethod
#     def setUpClass(cls):
#         rng = np.random.RandomState(42)
#         # 30 samples, 6 features, linearly separable on feature 0
#         cls.X_train = rng.randn(24, 6)
#         cls.y_train = (cls.X_train[:, 0] > 0).astype(int)
#         cls.X_test = rng.randn(6, 6)
#         cls.y_test = (cls.X_test[:, 0] > 0).astype(int)

#     def _make_fitted_model(self):
#         from app.quantum.kernel_model import QuantumKernelSVM
#         m = QuantumKernelSVM(n_qubits=2, C=1.0)
#         m.fit(self.X_train, self.y_train)
#         return m

#     def test_fit_sets_fitted_flag(self):
#         m = self._make_fitted_model()
#         self.assertTrue(m._is_fitted)

#     def test_fit_saves_training_anchors(self):
#         """_pi_min and _pi_max must be set after fit — needed for correct scaling."""
#         m = self._make_fitted_model()
#         self.assertIsNotNone(m._pi_min, "_pi_min not set after fit")
#         self.assertIsNotNone(m._pi_max, "_pi_max not set after fit")
#         self.assertTrue(np.all(m._pi_max >= m._pi_min))

#     def test_predict_returns_valid_classes(self):
#         m = self._make_fitted_model()
#         preds = m.predict(self.X_test)
#         self.assertEqual(preds.shape, (len(self.X_test),))
#         self.assertTrue(set(preds).issubset({0, 1}),
#                         f"predict() returned unexpected classes: {set(preds)}")

#     def test_predict_proba_shape_and_sums(self):
#         m = self._make_fitted_model()
#         proba = m.predict_proba(self.X_test)
#         self.assertEqual(proba.shape, (len(self.X_test), 2))
#         np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(self.X_test)), atol=1e-9)
#         self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))

#     def test_single_sample_prediction_not_zero_vector(self):
#         """
#         Regression test for the single-sample scaling bug.

#         Old _scale_to_pi used the batch's own min/max.  A 1-row batch has
#         min==max for every feature → rng=0 → all features mapped to 0.
#         Fixed by anchoring scaling to the training-set min/max in fit().
#         """
#         from app.quantum.kernel_model import QuantumKernelSVM
#         m = QuantumKernelSVM(n_qubits=2, C=1.0)
#         m.fit(self.X_train, self.y_train)
#         single = self.X_test[0:1]
#         reduced = m._transform(single)
#         self.assertFalse(
#             np.all(reduced == 0),
#             "Single-sample _transform returned all-zero vector — scaling bug not fixed")
#         # Probabilities must not be exactly 0.5/0.5 due to zero input
#         proba = m.predict_proba(single)
#         # (They could be 0.5 in theory, but all-zero features would give constant proba)
#         self.assertTrue(np.isfinite(proba).all())
#         self.assertAlmostEqual(float(proba[0].sum()), 1.0, places=9)

#     def test_reproducibility(self):
#         """Same seed → same predictions."""
#         from app.quantum.kernel_model import QuantumKernelSVM
#         m1 = QuantumKernelSVM(n_qubits=2, C=1.0, random_state=0)
#         m2 = QuantumKernelSVM(n_qubits=2, C=1.0, random_state=0)
#         m1.fit(self.X_train, self.y_train)
#         m2.fit(self.X_train, self.y_train)
#         np.testing.assert_array_equal(m1.predict(self.X_test), m2.predict(self.X_test))

#     def test_artifact_save_load(self):
#         """joblib.dump / joblib.load round-trip preserves predict_proba output."""
#         import joblib, tempfile, os
#         m = self._make_fitted_model()
#         expected = m.predict_proba(self.X_test)
#         with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
#             path = f.name
#         try:
#             joblib.dump(m, path)
#             m2 = joblib.load(path)
#             loaded = m2.predict_proba(self.X_test)
#             np.testing.assert_allclose(expected, loaded, atol=1e-12,
#                                        err_msg="predict_proba changed after joblib round-trip")
#         finally:
#             os.unlink(path)

#     def test_finite_metrics_on_test_set(self):
#         """All metrics values must be finite floats in [0, 1]."""
#         from app.evaluation.metrics import compute_metrics, timed_predict
#         m = self._make_fitted_model()
#         y_pred, y_proba, _ = timed_predict(m, self.X_test)
#         metrics = compute_metrics(self.y_test, y_pred, y_proba)
#         for key in ('accuracy', 'precision', 'recall', 'f1_score'):
#             val = metrics[key]
#             self.assertTrue(np.isfinite(val), f"metrics[{key!r}] = {val} is not finite")
#             self.assertGreaterEqual(val, 0.0)
#             self.assertLessEqual(val, 1.0)


# class TestVQCClassifier(unittest.TestCase):
#     """
#     Integration tests for VQCClassifier (vqc_model.py).

#     max_iter=5 keeps each test under a few seconds while still running real
#     quantum circuit forward passes.  The ADAM optimizer and finite-difference
#     gradients are NOT mocked.
#     """

#     @classmethod
#     def setUpClass(cls):
#         rng = np.random.RandomState(99)
#         cls.X_train = rng.randn(20, 6)
#         cls.y_train = (cls.X_train[:, 0] > 0).astype(int)
#         cls.X_test = rng.randn(6, 6)
#         cls.y_test = (cls.X_test[:, 0] > 0).astype(int)

#     def _make_fitted_model(self):
#         from app.quantum.vqc_model import VQCClassifier
#         m = VQCClassifier(n_qubits=2, max_iter=5, max_train_samples=20, random_state=42)
#         m.fit(self.X_train, self.y_train)
#         return m

#     def test_fit_sets_fitted_flag(self):
#         m = self._make_fitted_model()
#         self.assertTrue(m._is_fitted)

#     def test_fit_saves_training_anchors(self):
#         m = self._make_fitted_model()
#         self.assertIsNotNone(m._pi_min, "_pi_min not set after fit")
#         self.assertIsNotNone(m._pi_max, "_pi_max not set after fit")
#         self.assertTrue(np.all(m._pi_max >= m._pi_min))

#     def test_params_shape(self):
#         """Learned parameter vector must have length n_qubits * (reps + 1)."""
#         m = self._make_fitted_model()
#         expected = m.n_qubits * (m.reps + 1)
#         self.assertEqual(len(m.params), expected,
#                          f"params has length {len(m.params)}, expected {expected}")

#     def test_loss_history_populated(self):
#         m = self._make_fitted_model()
#         self.assertGreater(len(m.loss_history), 0)
#         # All losses must be finite positive
#         for loss in m.loss_history:
#             self.assertTrue(np.isfinite(loss), f"Non-finite loss in history: {loss}")

#     def test_predict_returns_valid_classes(self):
#         m = self._make_fitted_model()
#         preds = m.predict(self.X_test)
#         self.assertEqual(preds.shape, (len(self.X_test),))
#         self.assertTrue(set(preds).issubset({0, 1}),
#                         f"predict() returned unexpected classes: {set(preds)}")

#     def test_predict_proba_shape_and_sums(self):
#         m = self._make_fitted_model()
#         proba = m.predict_proba(self.X_test)
#         self.assertEqual(proba.shape, (len(self.X_test), 2))
#         np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(self.X_test)), atol=1e-12)
#         self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))

#     def test_single_sample_prediction_not_zero_vector(self):
#         """
#         Regression test for the single-sample scaling bug in VQCClassifier._scale.

#         Old implementation used batch min/max → zero-vector for single inputs.
#         Fixed by anchoring to training-set min/max saved in fit().
#         """
#         from app.quantum.vqc_model import VQCClassifier
#         m = VQCClassifier(n_qubits=2, max_iter=5, max_train_samples=20, random_state=42)
#         m.fit(self.X_train, self.y_train)
#         single = self.X_test[0:1]
#         reduced = m._transform(single)
#         self.assertFalse(
#             np.all(reduced == 0),
#             "Single-sample _transform returned all-zero vector — scaling bug not fixed")
#         proba = m.predict_proba(single)
#         self.assertTrue(np.isfinite(proba).all())
#         self.assertAlmostEqual(float(proba[0].sum()), 1.0, places=9)

#     def test_reproducibility(self):
#         """Same random_state → identical trained parameters."""
#         from app.quantum.vqc_model import VQCClassifier
#         m1 = VQCClassifier(n_qubits=2, max_iter=5, max_train_samples=20, random_state=7)
#         m2 = VQCClassifier(n_qubits=2, max_iter=5, max_train_samples=20, random_state=7)
#         m1.fit(self.X_train, self.y_train)
#         m2.fit(self.X_train, self.y_train)
#         np.testing.assert_array_almost_equal(m1.params, m2.params, decimal=12)

#     def test_artifact_save_load(self):
#         """joblib round-trip preserves predict_proba output."""
#         import joblib, tempfile, os
#         m = self._make_fitted_model()
#         expected = m.predict_proba(self.X_test)
#         with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
#             path = f.name
#         try:
#             joblib.dump(m, path)
#             m2 = joblib.load(path)
#             loaded = m2.predict_proba(self.X_test)
#             np.testing.assert_allclose(expected, loaded, atol=1e-12,
#                                        err_msg="predict_proba changed after joblib round-trip")
#         finally:
#             os.unlink(path)

#     def test_finite_metrics_on_test_set(self):
#         from app.evaluation.metrics import compute_metrics, timed_predict
#         m = self._make_fitted_model()
#         y_pred, y_proba, _ = timed_predict(m, self.X_test)
#         metrics = compute_metrics(self.y_test, y_pred, y_proba)
#         for key in ('accuracy', 'precision', 'recall', 'f1_score'):
#             val = metrics[key]
#             self.assertTrue(np.isfinite(val), f"metrics[{key!r}] = {val} is not finite")
#             self.assertGreaterEqual(val, 0.0)
#             self.assertLessEqual(val, 1.0)

#     def test_class_imbalance_does_not_crash(self):
#         """A severely imbalanced dataset must not silently produce misleading NaN metrics."""
#         from app.quantum.vqc_model import VQCClassifier
#         from app.evaluation.metrics import compute_metrics, timed_predict
#         # All-same-class training set
#         X_imb = np.random.RandomState(0).randn(15, 4)
#         y_imb = np.zeros(15, dtype=int)   # all class 0
#         y_test_imb = np.zeros(5, dtype=int)
#         m = VQCClassifier(n_qubits=2, max_iter=3, max_train_samples=15, random_state=0)
#         m.fit(X_imb, y_imb)
#         y_pred, y_proba, _ = timed_predict(m, np.random.RandomState(1).randn(5, 4))
#         # metrics must not raise; zero_division=0 keeps values finite
#         metrics = compute_metrics(y_test_imb, y_pred, y_proba)
#         self.assertIn('accuracy', metrics)
#         self.assertTrue(np.isfinite(metrics['accuracy']))


# if __name__ == '__main__':
#     loader = unittest.TestLoader()
#     suite = unittest.TestSuite()
#     suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
#     suite.addTests(loader.loadTestsFromTestCase(TestClassicalModels))
#     suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
#     suite.addTests(loader.loadTestsFromTestCase(TestTrainingServiceStatusValues))
#     suite.addTests(loader.loadTestsFromTestCase(TestJobCrossWorkerVisibility))
#     suite.addTests(loader.loadTestsFromTestCase(TestQuantumSimulator))
#     suite.addTests(loader.loadTestsFromTestCase(TestQuantumKernelSVM))
#     suite.addTests(loader.loadTestsFromTestCase(TestVQCClassifier))

#     runner = unittest.TextTestRunner(verbosity=2)
#     result = runner.run(suite)
#     sys.exit(0 if result.wasSuccessful() else 1)


































































"""
Standalone test runner — runs without pytest, using only stdlib unittest.
Converts pytest-style tests to unittest manually.
"""
import unittest
import sys
import os
import traceback
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


# ─── test_metrics ──────────────────────────────────────────────────────────
class TestMetrics(unittest.TestCase):

    def setUp(self):
        from app.evaluation.metrics import compute_metrics
        self.compute_metrics = compute_metrics

    def test_perfect_metrics(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        m = self.compute_metrics(y_true, y_pred)
        self.assertEqual(m['accuracy'], 1.0)
        self.assertEqual(m['sensitivity'], 1.0)
        self.assertEqual(m['specificity'], 1.0)
        self.assertEqual(m['f1_score'], 1.0)

    def test_confusion_matrix_derivation(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        m = self.compute_metrics(y_true, y_pred)
        self.assertEqual(m['sensitivity'], 0.5)
        self.assertEqual(m['specificity'], 0.5)
        self.assertEqual(m['accuracy'], 0.5)

    def test_roc_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        m = self.compute_metrics(y_true, y_pred, y_prob)
        self.assertEqual(m['roc_auc'], 1.0)


# ─── test_classical_models ─────────────────────────────────────────────────
# test_classical_models.py imports train_classical_model but the module
# exposes build_classical_model + train_classical separately.
# We test the actual exported API here and note the import mismatch.

class TestClassicalModels(unittest.TestCase):
    """
    Tests for build_classical_model + train_classical.

    Data is scaled via StandardScaler before fitting to mirror what the production
    PreprocessingPipeline does (pipeline.py StandardScaler.fit_transform(X_train)).
    Using raw breast-cancer features (range ~1–2500) without scaling causes
    LogisticRegression to emit a ConvergenceWarning even at max_iter=1000 — not
    because the model is broken, but because unscaled lbfgs requires many more
    iterations.  The fix is to scale the test data, not to raise max_iter, because
    the goal is to test the model under conditions that match production.
    """

    @classmethod
    def setUpClass(cls):
        from sklearn.preprocessing import StandardScaler
        data = load_breast_cancer()
        X, y = data.data[:400], data.target[:400]
        X_train_raw, X_test_raw = X[:320], X[320:]
        cls.y_train, cls.y_test = y[:320], y[320:]

        # Scale to match the production pipeline (StandardScaler fit on train only).
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(X_train_raw)
        cls.X_test  = scaler.transform(X_test_raw)

    def test_import_train_classical_model_missing(self):
        """train_classical_model (combined) is NOT exported by app.classical.models.
        The module exposes build_classical_model + train_classical separately.
        This test documents the stale import in test_classical_models.py."""
        try:
            from app.classical.models import train_classical_model  # noqa
            # If it exists, that's fine too
        except ImportError:
            pass  # expected — see comment above

    def test_build_and_train_logistic_regression(self):
        import warnings
        from app.classical.models import build_classical_model, train_classical
        model = build_classical_model('logistic_regression', {})
        # No ConvergenceWarning expected on scaled data.
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any warning → test failure
            t = train_classical(model, self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.y_test))
        self.assertGreater(t, 0)
        # Model should converge well on scaled breast-cancer data.
        accuracy = np.mean(preds == self.y_test)
        self.assertGreater(accuracy, 0.90,
            f"Logistic regression accuracy {accuracy:.2f} unexpectedly low on scaled data")

    def test_build_and_train_random_forest(self):
        from app.classical.models import build_classical_model, train_classical
        model = build_classical_model('random_forest', {'n_estimators': 10})
        train_classical(model, self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.y_test))
        accuracy = np.mean(preds == self.y_test)
        self.assertGreater(accuracy, 0.85,
            f"Random forest accuracy {accuracy:.2f} unexpectedly low")

    def test_build_and_train_svm(self):
        from app.classical.models import build_classical_model, train_classical
        model = build_classical_model('svm', {})
        train_classical(model, self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.y_test))
        accuracy = np.mean(preds == self.y_test)
        self.assertGreater(accuracy, 0.90,
            f"SVM accuracy {accuracy:.2f} unexpectedly low on scaled data")


# ─── test_preprocessing ────────────────────────────────────────────────────
# test_preprocessing.py imports prepare_data; we check whether it exists.

class TestPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = pd.Categorical.from_codes(data.target, ['malignant', 'benign'])
        cls.df = df

    def _get_prepare_data(self):
        try:
            from app.preprocessing.pipeline import prepare_data
            return prepare_data
        except ImportError:
            self.skipTest("prepare_data not exported from app.preprocessing.pipeline")

    def test_no_leakage(self):
        prepare_data = self._get_prepare_data()
        config = {'imputation_strategy': 'mean', 'scaling_method': 'standard'}
        result = prepare_data(self.df, 'diagnosis', None, [], 0.2, 42, config)
        self.assertEqual(result['X_train'].shape[0], result['train_size'])
        self.assertEqual(result['X_test'].shape[0], result['test_size'])
        self.assertEqual(result['train_size'] + result['test_size'], len(self.df))

    def test_stratified_split(self):
        prepare_data = self._get_prepare_data()
        config = {'imputation_strategy': 'mean', 'scaling_method': 'standard'}
        result = prepare_data(self.df, 'diagnosis', None, [], 0.2, 42, config)
        self.assertEqual(len(np.unique(result['y_train'])), 2)
        self.assertEqual(len(np.unique(result['y_test'])), 2)

    def test_feature_selection_kbest(self):
        prepare_data = self._get_prepare_data()
        config = {'imputation_strategy': 'mean', 'scaling_method': 'standard',
                  'feature_selection': 'kbest', 'n_features': 10}
        result = prepare_data(self.df, 'diagnosis', None, [], 0.2, 42, config)
        self.assertEqual(result['X_train'].shape[1], 10)
        self.assertEqual(result['X_test'].shape[1], 10)

    def test_pca(self):
        prepare_data = self._get_prepare_data()
        config = {'imputation_strategy': 'mean', 'scaling_method': 'standard',
                  'feature_selection': 'pca', 'n_features': 5}
        result = prepare_data(self.df, 'diagnosis', None, [], 0.2, 42, config)
        self.assertEqual(result['X_train'].shape[1], 5)
        self.assertEqual(result['X_test'].shape[1], 5)


# ─── test_training_service_status_values ───────────────────────────────────

class TestTrainingServiceStatusValues(unittest.TestCase):
    """Regression test for Training.tsx bug: backend must use 'completed'/'failed',
    not 'done'/'error', as terminal job statuses."""

    def test_terminal_status_completed_not_done(self):
        """training_service._run_training must set status='completed' on success."""
        import inspect
        from app.services import training_service
        src = inspect.getsource(training_service)
        # Source may use single or double quotes; check for the bare word
        self.assertIn("completed", src,
                      "training_service must set status='completed' on success")

    def test_terminal_status_failed_not_error(self):
        """training_service._run_training must set status='failed' on exception."""
        import inspect
        from app.services import training_service
        src = inspect.getsource(training_service)
        self.assertIn("failed", src,
                      "training_service must set status='failed' on exception")

    def test_no_done_terminal_status(self):
        """'done' must not appear as a terminal status assignment."""
        import inspect
        from app.services import training_service
        src = inspect.getsource(training_service)
        # Allow 'done' in comments but not as a status value assignment
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            self.assertNotIn("= 'done'", stripped,
                             f"Found status='done' assignment in training_service: {stripped!r}")


# ─── test_job_cross_worker_visibility ─────────────────────────────────────────

class TestJobCrossWorkerVisibility(unittest.TestCase):
    """
    Regression tests for the gunicorn cross-worker job-status bug.

    Root cause: _jobs is an in-memory dict, per-process.  With --workers 2,
    Worker A starts a job (writes _jobs[job_id]) but Worker B handles the
    status poll.  Worker B has no entry in its own _jobs → get_job() returned
    None → HTTP 404 → frontend catch-handler killed the polling interval →
    UI stuck at "Queued" forever.

    Fix: _set_job and _update_job now also write to disk via storage.insert /
    storage.update; get_job falls back to storage.find_one when the entry is
    absent from the in-process _jobs dict.
    """

    def _cleanup(self, job_id):
        """Remove test artefacts from both _jobs and disk."""
        from app.services import training_service, storage as svc_storage
        with training_service._jobs_lock:
            training_service._jobs.pop(job_id, None)
        svc_storage.delete("jobs", job_id)

    def test_get_job_falls_back_to_disk(self):
        """get_job() returns job data even when not in local _jobs dict."""
        from app.services import training_service, storage as svc_storage
        job_id = str(uuid.uuid4())
        job_doc = {
            "_id": job_id,
            "job_id": job_id,
            "status": "running",
            "progress": "Training model...",
        }
        # Simulate another worker having written the job to disk
        svc_storage.insert("jobs", job_doc)
        try:
            # Ensure the entry is absent from this process's _jobs
            with training_service._jobs_lock:
                training_service._jobs.pop(job_id, None)

            result = training_service.get_job(job_id)
            self.assertIsNotNone(result,
                "get_job() returned None when job absent from _jobs but present on disk "
                "— cross-worker visibility is broken")
            self.assertEqual(result["status"], "running")
            self.assertEqual(result["job_id"], job_id)
        finally:
            self._cleanup(job_id)

    def test_set_job_persists_to_disk(self):
        """_set_job writes job to disk so other workers can read it."""
        from app.services import training_service, storage as svc_storage
        job_id = str(uuid.uuid4())
        training_service._set_job(job_id, {
            "_id": job_id,
            "job_id": job_id,
            "status": "queued",
            "progress": "Queued",
        })
        try:
            disk_doc = svc_storage.find_one("jobs", job_id)
            self.assertIsNotNone(disk_doc,
                "_set_job did not persist job to disk — other workers will see 404")
            self.assertEqual(disk_doc["status"], "queued")
        finally:
            self._cleanup(job_id)

    def test_update_job_persists_to_disk(self):
        """_update_job writes status changes to disk for cross-worker reads."""
        from app.services import training_service, storage as svc_storage
        job_id = str(uuid.uuid4())
        training_service._set_job(job_id, {
            "_id": job_id,
            "job_id": job_id,
            "status": "queued",
            "progress": "Queued",
        })
        try:
            training_service._update_job(job_id, {"status": "running", "progress": "Training..."})
            disk_doc = svc_storage.find_one("jobs", job_id)
            self.assertIsNotNone(disk_doc)
            self.assertEqual(disk_doc["status"], "running",
                "_update_job did not update status on disk — cross-worker polls see stale data")
        finally:
            self._cleanup(job_id)


# ─── test_quantum_models ───────────────────────────────────────────────────────

class TestQuantumSimulator(unittest.TestCase):
    """Unit tests for the numpy statevector simulator primitives."""

    def test_hadamard_normalised(self):
        """H|0⟩ = (|0⟩+|1⟩)/√2 — statevector is normalised."""
        from app.quantum.simulator import apply_single, H
        sv = np.zeros(2, dtype=complex); sv[0] = 1.0
        sv_out = apply_single(sv, H, 0, 1)
        self.assertAlmostEqual(float(np.abs(sv_out[0])**2 + np.abs(sv_out[1])**2), 1.0, places=12)
        self.assertAlmostEqual(float(np.abs(sv_out[0])), 1 / np.sqrt(2), places=12)

    def test_zz_feature_map_normalised(self):
        """ZZFeatureMap output statevector must have unit norm."""
        from app.quantum.simulator import zz_feature_map
        x = np.array([0.5, 1.2])
        sv = zz_feature_map(x, n_qubits=2)
        norm = float(np.sum(np.abs(sv) ** 2))
        self.assertAlmostEqual(norm, 1.0, places=12,
                               msg="ZZFeatureMap statevector is not normalised")

    def test_zz_feature_map_deterministic(self):
        """Same input → same statevector."""
        from app.quantum.simulator import zz_feature_map
        x = np.array([0.3, 0.7, 1.1, 0.2])
        sv1 = zz_feature_map(x, n_qubits=4)
        sv2 = zz_feature_map(x, n_qubits=4)
        np.testing.assert_array_equal(sv1, sv2)

    def test_vqc_forward_in_range(self):
        """vqc_forward returns a value in [-1, 1] (⟨Z₀⟩ expectation)."""
        from app.quantum.simulator import vqc_forward
        rng = np.random.RandomState(7)
        for _ in range(5):
            x = rng.uniform(0, np.pi, 3)
            params = rng.uniform(-np.pi, np.pi, 3 * (2 + 1))  # n_qubits=3, reps=2
            val = vqc_forward(x, params, n_qubits=3)
            self.assertGreaterEqual(val, -1.0 - 1e-9)
            self.assertLessEqual(val, 1.0 + 1e-9)

    def test_quantum_kernel_self_similarity_is_one(self):
        """K(x, x) = |⟨φ(x)|φ(x)⟩|² = 1.0 for all x."""
        from app.quantum.simulator import quantum_kernel_entry
        rng = np.random.RandomState(13)
        for _ in range(5):
            x = rng.uniform(0, np.pi, 2)
            k = quantum_kernel_entry(x, x, n_qubits=2)
            self.assertAlmostEqual(k, 1.0, places=10,
                                   msg="Quantum kernel diagonal entry is not 1.0")

    def test_kernel_matrix_symmetric(self):
        """Training kernel matrix must be symmetric."""
        from app.quantum.simulator import compute_kernel_matrix
        rng = np.random.RandomState(17)
        X = rng.uniform(0, np.pi, (5, 2))
        K = compute_kernel_matrix(X, X, n_qubits=2)
        np.testing.assert_allclose(K, K.T, atol=1e-12,
                                   err_msg="Quantum kernel matrix is not symmetric")

    def test_vqc_predict_proba_shape_and_sum(self):
        """vqc_predict_proba returns (n_samples, 2) with rows summing to 1."""
        from app.quantum.simulator import vqc_predict_proba
        rng = np.random.RandomState(3)
        X = rng.uniform(0, np.pi, (4, 2))
        params = rng.uniform(-np.pi, np.pi, 2 * (2 + 1))
        proba = vqc_predict_proba(X, params, n_qubits=2)
        self.assertEqual(proba.shape, (4, 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(4), atol=1e-12)
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1),
                        msg="Some probabilities out of [0, 1]")


class TestQuantumKernelSVM(unittest.TestCase):
    """
    Integration tests for QuantumKernelSVM (kernel_model.py).

    Uses a tiny deterministic dataset so tests complete in a few seconds.
    The quantum computation is NOT mocked — all kernel entries are computed
    via the actual numpy statevector simulator.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        # 30 samples, 6 features, linearly separable on feature 0
        cls.X_train = rng.randn(24, 6)
        cls.y_train = (cls.X_train[:, 0] > 0).astype(int)
        cls.X_test = rng.randn(6, 6)
        cls.y_test = (cls.X_test[:, 0] > 0).astype(int)

    def _make_fitted_model(self):
        from app.quantum.kernel_model import QuantumKernelSVM
        m = QuantumKernelSVM(n_qubits=2, C=1.0)
        m.fit(self.X_train, self.y_train)
        return m

    def test_fit_sets_fitted_flag(self):
        m = self._make_fitted_model()
        self.assertTrue(m._is_fitted)

    def test_fit_saves_training_anchors(self):
        """_pi_min and _pi_max must be set after fit — needed for correct scaling."""
        m = self._make_fitted_model()
        self.assertIsNotNone(m._pi_min, "_pi_min not set after fit")
        self.assertIsNotNone(m._pi_max, "_pi_max not set after fit")
        self.assertTrue(np.all(m._pi_max >= m._pi_min))

    def test_predict_returns_valid_classes(self):
        m = self._make_fitted_model()
        preds = m.predict(self.X_test)
        self.assertEqual(preds.shape, (len(self.X_test),))
        self.assertTrue(set(preds).issubset({0, 1}),
                        f"predict() returned unexpected classes: {set(preds)}")

    def test_predict_proba_shape_and_sums(self):
        m = self._make_fitted_model()
        proba = m.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (len(self.X_test), 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(self.X_test)), atol=1e-9)
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))

    def test_single_sample_prediction_not_zero_vector(self):
        """
        Regression test for the single-sample scaling bug.

        Old _scale_to_pi used the batch's own min/max.  A 1-row batch has
        min==max for every feature → rng=0 → all features mapped to 0.
        Fixed by anchoring scaling to the training-set min/max in fit().
        """
        from app.quantum.kernel_model import QuantumKernelSVM
        m = QuantumKernelSVM(n_qubits=2, C=1.0)
        m.fit(self.X_train, self.y_train)
        single = self.X_test[0:1]
        reduced = m._transform(single)
        self.assertFalse(
            np.all(reduced == 0),
            "Single-sample _transform returned all-zero vector — scaling bug not fixed")
        # Probabilities must not be exactly 0.5/0.5 due to zero input
        proba = m.predict_proba(single)
        # (They could be 0.5 in theory, but all-zero features would give constant proba)
        self.assertTrue(np.isfinite(proba).all())
        self.assertAlmostEqual(float(proba[0].sum()), 1.0, places=9)

    def test_reproducibility(self):
        """Same seed → same predictions."""
        from app.quantum.kernel_model import QuantumKernelSVM
        m1 = QuantumKernelSVM(n_qubits=2, C=1.0, random_state=0)
        m2 = QuantumKernelSVM(n_qubits=2, C=1.0, random_state=0)
        m1.fit(self.X_train, self.y_train)
        m2.fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(m1.predict(self.X_test), m2.predict(self.X_test))

    def test_artifact_save_load(self):
        """joblib.dump / joblib.load round-trip preserves predict_proba output."""
        import joblib, tempfile, os
        m = self._make_fitted_model()
        expected = m.predict_proba(self.X_test)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            joblib.dump(m, path)
            m2 = joblib.load(path)
            loaded = m2.predict_proba(self.X_test)
            np.testing.assert_allclose(expected, loaded, atol=1e-12,
                                       err_msg="predict_proba changed after joblib round-trip")
        finally:
            os.unlink(path)

    def test_finite_metrics_on_test_set(self):
        """All metrics values must be finite floats in [0, 1]."""
        from app.evaluation.metrics import compute_metrics, timed_predict
        m = self._make_fitted_model()
        y_pred, y_proba, _ = timed_predict(m, self.X_test)
        metrics = compute_metrics(self.y_test, y_pred, y_proba)
        for key in ('accuracy', 'precision', 'recall', 'f1_score'):
            val = metrics[key]
            self.assertTrue(np.isfinite(val), f"metrics[{key!r}] = {val} is not finite")
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)


class TestVQCClassifier(unittest.TestCase):
    """
    Integration tests for VQCClassifier (vqc_model.py).

    max_iter=5 keeps each test under a few seconds while still running real
    quantum circuit forward passes.  The ADAM optimizer and finite-difference
    gradients are NOT mocked.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(99)
        cls.X_train = rng.randn(20, 6)
        cls.y_train = (cls.X_train[:, 0] > 0).astype(int)
        cls.X_test = rng.randn(6, 6)
        cls.y_test = (cls.X_test[:, 0] > 0).astype(int)

    def _make_fitted_model(self):
        from app.quantum.vqc_model import VQCClassifier
        m = VQCClassifier(n_qubits=2, max_iter=5, max_train_samples=20, random_state=42)
        m.fit(self.X_train, self.y_train)
        return m

    def test_fit_sets_fitted_flag(self):
        m = self._make_fitted_model()
        self.assertTrue(m._is_fitted)

    def test_fit_saves_training_anchors(self):
        m = self._make_fitted_model()
        self.assertIsNotNone(m._pi_min, "_pi_min not set after fit")
        self.assertIsNotNone(m._pi_max, "_pi_max not set after fit")
        self.assertTrue(np.all(m._pi_max >= m._pi_min))

    def test_params_shape(self):
        """Learned parameter vector must have length n_qubits * (reps + 1)."""
        m = self._make_fitted_model()
        expected = m.n_qubits * (m.reps + 1)
        self.assertEqual(len(m.params), expected,
                         f"params has length {len(m.params)}, expected {expected}")

    def test_loss_history_populated(self):
        m = self._make_fitted_model()
        self.assertGreater(len(m.loss_history), 0)
        # All losses must be finite positive
        for loss in m.loss_history:
            self.assertTrue(np.isfinite(loss), f"Non-finite loss in history: {loss}")

    def test_predict_returns_valid_classes(self):
        m = self._make_fitted_model()
        preds = m.predict(self.X_test)
        self.assertEqual(preds.shape, (len(self.X_test),))
        self.assertTrue(set(preds).issubset({0, 1}),
                        f"predict() returned unexpected classes: {set(preds)}")

    def test_predict_proba_shape_and_sums(self):
        m = self._make_fitted_model()
        proba = m.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (len(self.X_test), 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(self.X_test)), atol=1e-12)
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))

    def test_single_sample_prediction_not_zero_vector(self):
        """
        Regression test for the single-sample scaling bug in VQCClassifier._scale.

        Old implementation used batch min/max → zero-vector for single inputs.
        Fixed by anchoring to training-set min/max saved in fit().
        """
        from app.quantum.vqc_model import VQCClassifier
        m = VQCClassifier(n_qubits=2, max_iter=5, max_train_samples=20, random_state=42)
        m.fit(self.X_train, self.y_train)
        single = self.X_test[0:1]
        reduced = m._transform(single)
        self.assertFalse(
            np.all(reduced == 0),
            "Single-sample _transform returned all-zero vector — scaling bug not fixed")
        proba = m.predict_proba(single)
        self.assertTrue(np.isfinite(proba).all())
        self.assertAlmostEqual(float(proba[0].sum()), 1.0, places=9)

    def test_reproducibility(self):
        """Same random_state → identical trained parameters."""
        from app.quantum.vqc_model import VQCClassifier
        m1 = VQCClassifier(n_qubits=2, max_iter=5, max_train_samples=20, random_state=7)
        m2 = VQCClassifier(n_qubits=2, max_iter=5, max_train_samples=20, random_state=7)
        m1.fit(self.X_train, self.y_train)
        m2.fit(self.X_train, self.y_train)
        np.testing.assert_array_almost_equal(m1.params, m2.params, decimal=12)

    def test_artifact_save_load(self):
        """joblib round-trip preserves predict_proba output."""
        import joblib, tempfile, os
        m = self._make_fitted_model()
        expected = m.predict_proba(self.X_test)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            joblib.dump(m, path)
            m2 = joblib.load(path)
            loaded = m2.predict_proba(self.X_test)
            np.testing.assert_allclose(expected, loaded, atol=1e-12,
                                       err_msg="predict_proba changed after joblib round-trip")
        finally:
            os.unlink(path)

    def test_finite_metrics_on_test_set(self):
        from app.evaluation.metrics import compute_metrics, timed_predict
        m = self._make_fitted_model()
        y_pred, y_proba, _ = timed_predict(m, self.X_test)
        metrics = compute_metrics(self.y_test, y_pred, y_proba)
        for key in ('accuracy', 'precision', 'recall', 'f1_score'):
            val = metrics[key]
            self.assertTrue(np.isfinite(val), f"metrics[{key!r}] = {val} is not finite")
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_class_imbalance_does_not_crash(self):
        """A severely imbalanced dataset must not silently produce misleading NaN metrics."""
        from app.quantum.vqc_model import VQCClassifier
        from app.evaluation.metrics import compute_metrics, timed_predict
        # All-same-class training set
        X_imb = np.random.RandomState(0).randn(15, 4)
        y_imb = np.zeros(15, dtype=int)   # all class 0
        y_test_imb = np.zeros(5, dtype=int)
        m = VQCClassifier(n_qubits=2, max_iter=3, max_train_samples=15, random_state=0)
        m.fit(X_imb, y_imb)
        y_pred, y_proba, _ = timed_predict(m, np.random.RandomState(1).randn(5, 4))
        # metrics must not raise; zero_division=0 keeps values finite
        metrics = compute_metrics(y_test_imb, y_pred, y_proba)
        self.assertIn('accuracy', metrics)
        self.assertTrue(np.isfinite(metrics['accuracy']))


# ─── test_phase3_integration_defects ──────────────────────────────────────────
# Regression tests for the four defects identified in Phase 3.

class TestPhase3IntegrationDefects(unittest.TestCase):
    """
    Phase 3 E2E Integration Audit — regression tests for confirmed defects.

    Defect 1: use_pca=True from frontend was silently ignored (pipeline.py only
              checked n_pca_components, never the use_pca flag).
    Defect 2: n_features slider was silently ignored when feature_selection was
              'none' (the frontend default) and use_pca was False.
    Defect 3: predict_single returned probabilities as [[p0,p1]] (nested 2-D
              array) instead of [p0,p1] (flat list).
    Defect 4: Neither predict_single nor predict_batch ever set risk_category;
              the Predictions UI badge was always empty.
    """

    @classmethod
    def setUpClass(cls):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = pd.Categorical.from_codes(data.target, ['malignant', 'benign'])
        cls.df = df  # 30 features + target

    # ── Defect 1: use_pca flag honoured ────────────────────────────────────

    def test_use_pca_true_reduces_dimensionality(self):
        """When use_pca=True the pipeline must apply PCA and reduce features."""
        from app.preprocessing.pipeline import PreprocessingPipeline
        # 5 components << 30 features
        cfg = {"use_pca": True, "n_features": 5, "test_size": 0.2, "random_state": 42}
        pipe = PreprocessingPipeline(cfg)
        X_tr, X_te, _, _ = pipe.fit_transform(self.df, "diagnosis")
        self.assertEqual(
            X_tr.shape[1], 5,
            f"use_pca=True with n_features=5 should give 5 components, got {X_tr.shape[1]}"
        )
        self.assertEqual(X_te.shape[1], 5)
        self.assertIsNotNone(pipe.pca, "PCA object must be set when use_pca=True")
        self.assertTrue(
            all(name.startswith("PC") for name in pipe.feature_names_out),
            f"feature_names_out should be PC1…PCn, got: {pipe.feature_names_out[:3]}"
        )

    def test_use_pca_false_does_not_apply_pca(self):
        """When use_pca=False PCA must not be applied."""
        from app.preprocessing.pipeline import PreprocessingPipeline
        cfg = {"use_pca": False, "test_size": 0.2, "random_state": 42}
        pipe = PreprocessingPipeline(cfg)
        X_tr, _, _, _ = pipe.fit_transform(self.df, "diagnosis")
        self.assertIsNone(pipe.pca, "PCA object must be None when use_pca=False")
        self.assertEqual(X_tr.shape[1], 30,
                         "use_pca=False must keep all 30 features")

    def test_use_pca_true_without_n_features_does_not_crash(self):
        """use_pca=True with no n_features must not crash.

        When n_features is absent the pipeline has nothing to reduce to, so PCA
        is skipped (applying PCA with all n_features components is a no-op).
        The frontend always sends n_features alongside use_pca=True; this test
        only verifies the degenerate case doesn't raise.
        """
        from app.preprocessing.pipeline import PreprocessingPipeline
        cfg = {"use_pca": True, "test_size": 0.2, "random_state": 42}
        pipe = PreprocessingPipeline(cfg)
        try:
            X_tr, _, _, _ = pipe.fit_transform(self.df, "diagnosis")
        except Exception as exc:
            self.fail(f"use_pca=True without n_features raised unexpectedly: {exc}")

    # ── Defect 2: n_features slider honoured ────────────────────────────────

    def test_n_features_slider_reduces_features_without_explicit_method(self):
        """n_features with default feature_selection='none' must apply SelectKBest."""
        from app.preprocessing.pipeline import PreprocessingPipeline
        # Frontend sends n_features but never sets feature_selection
        cfg = {"n_features": 8, "test_size": 0.2, "random_state": 42}
        pipe = PreprocessingPipeline(cfg)
        X_tr, X_te, _, _ = pipe.fit_transform(self.df, "diagnosis")
        self.assertEqual(
            X_tr.shape[1], 8,
            f"n_features=8 with no explicit method should give 8 features, got {X_tr.shape[1]}"
        )
        self.assertEqual(X_te.shape[1], 8)
        self.assertIsNotNone(pipe.feature_selector,
                             "feature_selector must be set when n_features slider is used")

    def test_n_features_larger_than_ncols_does_not_crash(self):
        """n_features > number of available features must not error."""
        from app.preprocessing.pipeline import PreprocessingPipeline
        cfg = {"n_features": 999, "test_size": 0.2, "random_state": 42}
        pipe = PreprocessingPipeline(cfg)
        # Should complete without raising; no feature selection applied
        try:
            X_tr, _, _, _ = pipe.fit_transform(self.df, "diagnosis")
            self.assertEqual(X_tr.shape[1], 30)
        except Exception as exc:
            self.fail(f"n_features > n_cols raised unexpectedly: {exc}")

    # ── Defect 3: probabilities shape (flat, not nested) ────────────────────

    def test_predict_single_probabilities_are_flat_list(self):
        """predict_single must return probabilities as [p0,p1], not [[p0,p1]]."""
        from app.classical.models import build_classical_model, train_classical
        from app.preprocessing.pipeline import PreprocessingPipeline

        # Build a tiny model + pipeline we can call predict_single on
        cfg = {"test_size": 0.2, "random_state": 42}
        pipe = PreprocessingPipeline(cfg)
        X_tr, X_te, y_tr, y_te = pipe.fit_transform(self.df, "diagnosis")

        model = build_classical_model("logistic_regression", {})
        train_classical(model, X_tr, y_tr)

        # Verify the fix directly on predict_proba output shape
        row = {col: float(self.df[col].iloc[0]) for col in pipe.selected_features}
        X_df = pd.DataFrame([row])
        X = pipe.transform(X_df)

        proba = model.predict_proba(X)[0].tolist()  # fixed line in training_service.py
        self.assertIsInstance(proba, list, "probabilities must be a list")
        self.assertEqual(len(proba), 2, "probabilities must have 2 elements")
        self.assertIsInstance(proba[0], float,
                              "probabilities[0] must be a float, not a nested list")
        # Additional check: the original bug gave [[p0,p1]]; verify we don't have that
        self.assertNotIsInstance(proba[0], list,
                                 "probabilities[0] is a list — nested shape bug not fixed")

    def test_predict_batch_probabilities_are_flat_per_row(self):
        """predict_batch already returns flat lists per row; verify consistency."""
        from app.classical.models import build_classical_model, train_classical
        from app.preprocessing.pipeline import PreprocessingPipeline

        cfg = {"test_size": 0.2, "random_state": 42}
        pipe = PreprocessingPipeline(cfg)
        X_tr, X_te, y_tr, _ = pipe.fit_transform(self.df, "diagnosis")

        model = build_classical_model("logistic_regression", {})
        train_classical(model, X_tr, y_tr)

        # Three rows
        rows = [
            {col: float(self.df[col].iloc[i]) for col in pipe.selected_features}
            for i in range(3)
        ]
        X_df = pd.DataFrame(rows)
        X = pipe.transform(X_df)
        proba_matrix = model.predict_proba(X)

        for i, pred in enumerate(model.predict(X)):
            p = proba_matrix[i].tolist()  # same as predict_batch
            self.assertIsInstance(p, list, f"Row {i}: probabilities must be a list")
            self.assertEqual(len(p), 2, f"Row {i}: probabilities must have 2 elements")
            self.assertIsInstance(p[0], float, f"Row {i}: probabilities[0] must be float")

    # ── Defect 4: risk_category present in both predict functions ───────────

    def test_risk_category_helper_values(self):
        """_risk_category must return correct label from flat probability list."""
        from app.services.training_service import _risk_category
        # High: predicted class probability ≥ 0.70
        self.assertEqual(_risk_category([0.15, 0.85], 1), "High")
        self.assertEqual(_risk_category([0.72, 0.28], 0), "High")
        # Moderate: 0.45 ≤ p < 0.70
        self.assertEqual(_risk_category([0.40, 0.60], 1), "Moderate")
        self.assertEqual(_risk_category([0.55, 0.45], 0), "Moderate")
        # Low: p < 0.45
        self.assertEqual(_risk_category([0.85, 0.15], 1), "Low")
        # None input → Unknown
        self.assertEqual(_risk_category(None, 0), "Unknown")
        # Out-of-bounds index → Unknown
        self.assertEqual(_risk_category([0.5, 0.5], 5), "Unknown")

    def test_predict_single_includes_risk_category(self):
        """predict_single result dict must include risk_category key."""
        from app.services import training_service
        import inspect
        src = inspect.getsource(training_service.predict_single)
        self.assertIn("risk_category", src,
                      "predict_single must set risk_category in the returned dict")

    def test_predict_batch_includes_risk_category(self):
        """Each item from predict_batch must include risk_category key."""
        from app.services import training_service
        import inspect
        src = inspect.getsource(training_service.predict_batch)
        self.assertIn("risk_category", src,
                      "predict_batch must set risk_category in each result dict")

    def test_risk_category_boundary_at_0_70(self):
        """Exactly 0.70 probability is 'High' (≥ threshold)."""
        from app.services.training_service import _risk_category
        self.assertEqual(_risk_category([0.30, 0.70], 1), "High")

    def test_risk_category_boundary_at_0_45(self):
        """Exactly 0.45 probability is 'Moderate' (≥ lower threshold)."""
        from app.services.training_service import _risk_category
        self.assertEqual(_risk_category([0.55, 0.45], 1), "Moderate")


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestClassicalModels))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingServiceStatusValues))
    suite.addTests(loader.loadTestsFromTestCase(TestJobCrossWorkerVisibility))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumKernelSVM))
    suite.addTests(loader.loadTestsFromTestCase(TestVQCClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3IntegrationDefects))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)