"""
End-to-end training service.
Coordinates: dataset → preprocessing → training → evaluation → persistence.
"""
import os
import time
import uuid
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import threading

from app.core.config import settings
from app.services import storage
from app.services.dataset_service import get_dataset
from app.preprocessing.pipeline import PreprocessingPipeline
from app.classical.models import build_classical_model, train_classical
from app.evaluation.metrics import compute_metrics, timed_predict
from app.explainability.importance import explain_model

from pathlib import Path
from app.core.config import settings

ARTIFACTS_DIR = Path(settings.ARTIFACTS_DIR)
REPORTS_DIR = Path(settings.REPORTS_DIR)

# In-memory job tracking (per-process)
# NOTE: gunicorn runs multiple workers, each with its own _jobs dict.
# get_job() falls back to disk so any worker can read jobs started elsewhere.
_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()

def _risk_category(probabilities, positive_class_index=1):
    """Convert positive-class probability into an application risk band."""
    if probabilities is None:
        return "Unknown"

    try:
        values = np.asarray(probabilities, dtype=float).flatten()

        if not 0 <= positive_class_index < len(values):
            return "Unknown"

        probability = float(values[positive_class_index])

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            return "Unknown"

        if probability < 0.45:
            return "Low"
        elif probability < 0.70:
            return "Moderate"
        return "High"

    except (TypeError, ValueError, IndexError):
        return "Unknown"


def _positive_class_probability(probabilities) -> Optional[float]:
    """Extract the positive-class probability from predict_proba output."""
    if probabilities is None:
        return None

    values = np.asarray(probabilities, dtype=float)

    if values.ndim == 2:
        if values.shape[0] == 0 or values.shape[1] == 0:
            return None
        values = values[0]

    if values.ndim != 1 or len(values) == 0:
        return None

    # For binary classification, class index 1 is treated as the
    # positive class. For a single probability, use that value.
    probability = values[1] if len(values) >= 2 else values[0]

    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None

    return float(probability)


def get_job(job_id: str) -> Optional[Dict]:
    """Return job dict, checking in-memory first then disk.

    The disk fallback is essential when gunicorn routes the status-poll
    request to a different worker than the one that started the job.
    Without it, get_job() returns None → HTTP 404 → frontend polls stop
    and the UI is permanently stuck at "Queued".
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return job
    # Cross-worker fallback: read the status written to disk by the worker
    # that owns the training thread.
    return storage.find_one("jobs", job_id)


def _set_job(job_id: str, data: Dict):
    """Write job to in-memory dict AND to disk for cross-worker visibility."""
    with _jobs_lock:
        _jobs[job_id] = data
    doc = dict(data)
    doc.setdefault("_id", job_id)
    storage.insert("jobs", doc)


def _update_job(job_id: str, updates: Dict):
    """Update job in in-memory dict AND on disk.

    The training thread always runs on the worker that called
    start_training_job(), so job_id is always in _jobs here.
    The disk write keeps other workers' disk reads up to date.
    """
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)
    storage.update("jobs", job_id, updates)


def _run_training(job_id: str, dataset_id: str, model_type: str,
                  model_params: Dict, preproc_config: Dict):
    """Background training thread."""
    try:
        _update_job(job_id, {"status": "running", "progress": "Loading dataset..."})
        ds = get_dataset(dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found.")

                # Resolve dataset paths saved on another operating system.
        stored_path = ds.get("filepath")
        dataset_path = Path(stored_path) if stored_path else None

        if dataset_path is None or not dataset_path.is_file():
            filename = Path(stored_path).name if stored_path else ""
            local_path = Path(settings.DATASETS_DIR) / filename

            if not filename or not local_path.is_file():
                raise FileNotFoundError(
                    f"Dataset file not found. Stored path: {stored_path}. "
                    f"Checked local path: {local_path}"
                )

            dataset_path = local_path

        df = pd.read_csv(dataset_path)
        target_col = ds["target_col"]
        exclude_cols = ds.get("exclude_cols", [])

        _update_job(job_id, {"progress": "Preprocessing..."})
        pipeline = PreprocessingPipeline(preproc_config)
        X_train, X_test, y_train, y_test = pipeline.fit_transform(df, target_col, exclude_cols)

        label_names = [str(c) for c in pipeline.label_encoder.classes_]
        feature_names = pipeline.feature_names_out

        _update_job(job_id, {"progress": "Training model..."})

        if model_type in ("logistic_regression", "svm", "random_forest"):
            model = build_classical_model(model_type, model_params)
            train_time = train_classical(model, X_train, y_train)
        elif model_type == "quantum_kernel_svm":
            from app.quantum.kernel_model import QuantumKernelSVM
            n_qubits = int(model_params.get("n_qubits", 4))
            # Subsample for speed
            max_samples = int(model_params.get("max_train_samples", 100))
            if len(X_train) > max_samples:
                idx = np.random.RandomState(42).choice(len(X_train), max_samples, replace=False)
                X_tr2, y_tr2 = X_train[idx], y_train[idx]
            else:
                X_tr2, y_tr2 = X_train, y_train
            model = QuantumKernelSVM(
                n_qubits=n_qubits,
                C=float(model_params.get("C", 1.0)),
                feature_map=model_params.get("feature_map", "zz"),
            )
            t0 = time.perf_counter()
            _update_job(job_id, {"progress": f"Computing quantum kernel ({len(X_tr2)} samples)..."})
            model.fit(X_tr2, y_tr2, verbose=False)
            train_time = time.perf_counter() - t0
        elif model_type == "vqc":
            from app.quantum.vqc_model import VQCClassifier
            model = VQCClassifier(
                n_qubits=int(model_params.get("n_qubits", 4)),
                reps=int(model_params.get("reps", 2)),
                feature_map=model_params.get("feature_map", "zz"),
                max_iter=int(model_params.get("max_iter", 30)),
                learning_rate=float(model_params.get("learning_rate", 0.01)),
                max_train_samples=int(model_params.get("max_train_samples", 60)),
            )
            _update_job(job_id, {"progress": "Training VQC (this may take a minute)..."})
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        _update_job(job_id, {"progress": "Evaluating..."})
        y_pred, y_proba, inference_time = timed_predict(model, X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba, label_names=label_names)

        _update_job(job_id, {"progress": "Computing feature importance..."})
        try:
            explanations = explain_model(model, X_test, y_test, feature_names, model_type)
        except Exception as e:
            explanations = {"error": str(e)}

        # Persist model artifact
        model_id = str(uuid.uuid4())
        artifact_path = str(ARTIFACTS_DIR / f"{model_id}.pkl")
        pipeline_path = str(ARTIFACTS_DIR / f"{model_id}_pipeline.pkl")
        joblib.dump(model, artifact_path)
        pipeline.save(pipeline_path)

        model_doc = {
            "_id": model_id,
            "model_type": model_type,
            "dataset_id": dataset_id,
            "artifact_path": artifact_path,
            "pipeline_path": pipeline_path,
            "feature_names": feature_names,
            "label_names": label_names,
            "target_mapping": pipeline.target_mapping,
            "metrics": metrics,
            "training_time_s": train_time,
            "inference_time_s": inference_time,
            "model_params": model_params,
            "preproc_config": preproc_config,
            "explanations": explanations,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "status": "completed",
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        storage.insert("models", model_doc)

        # Experiment doc
        exp_doc = {
            "_id": job_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "model_type": model_type,
            "model_params": model_params,
            "preproc_config": preproc_config,
            "status": "completed",
            "metrics": metrics,
            "training_time_s": train_time,
            "inference_time_s": inference_time,
            "feature_names": feature_names,
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
        storage.insert("experiments", exp_doc)

        _update_job(job_id, {
            "status": "completed",
            "progress": "Done",
            "model_id": model_id,
            "metrics": metrics,
            "training_time_s": train_time,
            "feature_names": feature_names,
            "explanations": explanations,
        })

    except Exception as exc:
        import traceback
        _update_job(job_id, {
            "status": "failed",
            "progress": "Failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


def start_training_job(dataset_id: str, model_type: str,
                        model_params: Dict, preproc_config: Dict) -> str:
    job_id = str(uuid.uuid4())
    _set_job(job_id, {
        "job_id": job_id,
        "status": "queued",
        "progress": "Queued",
        "model_type": model_type,
        "dataset_id": dataset_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
    })
    t = threading.Thread(
        target=_run_training,
        args=(job_id, dataset_id, model_type, model_params, preproc_config),
        daemon=True,
    )
    t.start()
    return job_id


def list_models() -> List[Dict]:
    return storage.find_all("models")


def get_model_doc(model_id: str) -> Optional[Dict]:
    return storage.find_one("models", model_id)



def _resolve_artifact_path(stored_path: Optional[str], model_id: str,
                           suffixes: List[str]) -> Path:
    """Resolve a saved artifact path on the current operating system."""
    candidates = []

    # First, try the path stored in the database.
    if stored_path:
        stored = Path(stored_path)

        if stored.is_file():
            return stored

        # Recover the filename from paths saved on another OS.
        candidates.append(ARTIFACTS_DIR / stored.name)

    # Then try the standard local artifact filenames.
    for suffix in suffixes:
        candidates.append(ARTIFACTS_DIR / f"{model_id}{suffix}")

    # Return the first existing local file.
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Artifact not found for model {model_id}. "
        f"Checked: {expected}"
    )


def load_model(model_id: str):
    """Load a model and its preprocessing pipeline from local artifacts."""
    doc = get_model_doc(model_id)

    if not doc:
        raise ValueError(f"Model {model_id} not found.")

    artifact_path = _resolve_artifact_path(
        doc.get("artifact_path"),
        model_id,
        [".pkl", ".joblib"],
    )

    pipeline_path = _resolve_artifact_path(
        doc.get("pipeline_path"),
        model_id,
        ["_pipeline.pkl", "_pipeline.joblib"],
    )

    model = joblib.load(artifact_path)
    pipeline = PreprocessingPipeline.load(str(pipeline_path))

    return model, pipeline, doc



def predict_single(model_id: str, feature_values: Dict[str, float]) -> Dict:
    model, pipeline, doc = load_model(model_id)

    X_df = pd.DataFrame([feature_values])
    X = pipeline.transform(X_df)

    y_pred = model.predict(X)
    label = (
        doc["label_names"][int(y_pred[0])]
        if doc.get("label_names")
        else str(y_pred[0])
    )

    proba = None
    risk_probability = None

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X).tolist()
            risk_probability = _positive_class_probability(proba)
        except Exception:
            pass

    pred_doc = {
        "model_id": model_id,
        "predicted_class": label,
        "predicted_index": int(y_pred[0]),
        "probabilities": proba,
        "risk_category": _risk_category([risk_probability]) if risk_probability is not None else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "Experimental model. Not for clinical use.",
    }

    storage.insert("predictions", pred_doc)
    return pred_doc


def predict_batch(model_id: str, rows: List[Dict[str, float]]) -> List[Dict]:
    model, pipeline, doc = load_model(model_id)

    X_df = pd.DataFrame(rows)
    X = pipeline.transform(X_df)

    y_pred = model.predict(X)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            pass

    results = []

    for i, pred in enumerate(y_pred):
        label = (
            doc["label_names"][int(pred)]
            if doc.get("label_names")
            else str(pred)
        )

        p = proba[i].tolist() if proba is not None else None
        risk_probability = _positive_class_probability(p)

        results.append({
            "row": i,
            "predicted_class": label,
            "predicted_index": int(pred),
            "probabilities": p,
            "risk_category": _risk_category([risk_probability]) if risk_probability is not None else None,
        })

    return results
