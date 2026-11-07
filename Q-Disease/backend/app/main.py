"""
Q-Disease Flask API
Hybrid Quantum-Classical ML Platform for Early Disease Detection
"""
import os
import json
import csv
import io
import traceback
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response

from app.core.config import settings

app = Flask(__name__)

@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin", "")
    allowed = [settings.FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"]
    if origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.route("/api/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return "", 204

DISCLAIMER = (
    "Q-Disease is an experimental research and educational platform. "
    "It is not a clinically validated medical device and must not be used "
    "to diagnose, treat, or make medical decisions for patients."
)


def ok(data=None, **kwargs):
    payload = {"success": True}
    if data is not None:
        payload["data"] = data
    payload.update(kwargs)
    return jsonify(payload)


def err(msg: str, code: int = 400):
    return jsonify({"success": False, "error": msg}), code


# ─── Health ──────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    from app.services import storage
    return ok({
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "storage": "file-based",
        "disclaimer": DISCLAIMER,
        "datasets": storage.count("datasets"),
        "models": storage.count("models"),
        "experiments": storage.count("experiments"),
    })


@app.get("/api/stats")
def stats():
    from app.services import storage
    return ok({
        "total_datasets": storage.count("datasets"),
        "total_models": storage.count("models"),
        "total_experiments": storage.count("experiments"),
        "total_predictions": storage.count("predictions"),
    })


# ─── Datasets ────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets_route():
    from app.services.dataset_service import list_datasets
    return ok(list_datasets())


@app.post("/api/datasets/upload")
def upload_dataset():
    from app.services.dataset_service import save_dataset_file, register_dataset, analyze_dataset
    if "file" not in request.files:
        return err("No file provided.")
    f = request.files["file"]
    if not f.filename:
        return err("Empty filename.")
    content = f.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        return err(f"File too large. Max {settings.MAX_UPLOAD_SIZE_MB} MB.")
    try:
        filepath = save_dataset_file(f.filename, content)
    except ValueError as e:
        return err(str(e))

    target_col = request.form.get("target_col", "")
    exclude_cols_raw = request.form.get("exclude_cols", "[]")
    try:
        exclude_cols = json.loads(exclude_cols_raw)
    except Exception:
        exclude_cols = []
    name = request.form.get("name", Path(f.filename).stem)

    try:
        analysis = analyze_dataset(filepath, target_col or None)
    except ValueError as e:
        # Clean up saved file if analysis fails
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception:
            pass
        return err(str(e))
    except Exception as e:
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception:
            pass
        return err(f"Failed to parse CSV: {e}")

    if target_col:
        # Fully register in one step
        try:
            ds_id = register_dataset(name, filepath, target_col, exclude_cols)
        except Exception as e:
            return err(str(e))
        return ok({
            "_id": ds_id,
            "registered": True,
            "analysis": analysis,
            "target_col": target_col,
        })
    else:
        # Step 1 of 2: return analysis so frontend can show column picker
        return ok({
            "filepath": filepath,
            "name": name,
            "registered": False,
            "analysis": analysis,
            "column_names": [c["name"] for c in analysis["columns"]],
        })


@app.post("/api/datasets/register")
def register_dataset_route():
    from app.services.dataset_service import register_dataset
    body = request.get_json() or {}
    filepath = body.get("filepath")
    target_col = body.get("target_col")
    if not filepath or not target_col:
        return err("filepath and target_col required.")
    exclude_cols = body.get("exclude_cols", [])
    name = body.get("name", Path(filepath).name)
    try:
        ds_id = register_dataset(name, filepath, target_col, exclude_cols)
        return ok({"_id": ds_id})
    except Exception as e:
        return err(str(e))


@app.get("/api/datasets/<dataset_id>")
def get_dataset_route(dataset_id):
    from app.services.dataset_service import get_dataset, analyze_dataset
    doc = get_dataset(dataset_id)
    if not doc:
        return err("Dataset not found.", 404)
    return ok(doc)


@app.delete("/api/datasets/<dataset_id>")
def delete_dataset_route(dataset_id):
    from app.services.dataset_service import delete_dataset
    ok_ = delete_dataset(dataset_id)
    if not ok_:
        return err("Dataset not found.", 404)
    return ok({"deleted": dataset_id})


@app.get("/api/datasets/<dataset_id>/summary")
def dataset_summary(dataset_id):
    from app.services.dataset_service import get_dataset, analyze_dataset
    doc = get_dataset(dataset_id)
    if not doc:
        return err("Dataset not found.", 404)
    analysis = analyze_dataset(doc["filepath"], doc.get("target_col"))
    return ok({**doc, **analysis})


@app.get("/api/datasets/<dataset_id>/distributions")
def dataset_distributions(dataset_id):
    from app.services.dataset_service import get_feature_distributions
    data = get_feature_distributions(dataset_id)
    return ok(data)


@app.get("/api/datasets/<dataset_id>/correlation")
def dataset_correlation(dataset_id):
    from app.services.dataset_service import get_correlation_matrix
    data = get_correlation_matrix(dataset_id)
    return ok(data)


# ─── Preprocessing preview ───────────────────────────────────────

@app.post("/api/preprocessing/preview")
def preprocessing_preview():
    import pandas as pd
    from app.preprocessing.pipeline import PreprocessingPipeline
    from app.services.dataset_service import get_dataset
    body = request.get_json() or {}
    dataset_id = body.get("dataset_id")
    config = body.get("config", {})
    doc = get_dataset(dataset_id)
    if not doc:
        return err("Dataset not found.", 404)
    try:
        df = pd.read_csv(doc["filepath"])
        pipeline = PreprocessingPipeline(config)
        X_tr, X_te, y_tr, y_te = pipeline.fit_transform(
            df, doc["target_col"], doc.get("exclude_cols", [])
        )
        return ok({
            "train_samples": int(len(X_tr)),
            "test_samples": int(len(X_te)),
            "n_features_out": int(X_tr.shape[1]),
            "feature_names_out": pipeline.feature_names_out,
            "target_mapping": pipeline.target_mapping,
            "label_names": [str(c) for c in pipeline.label_encoder.classes_],
        })
    except Exception as e:
        return err(str(e))


# ─── Model Training ───────────────────────────────────────────────

@app.post("/api/models/train")
def train_model():
    from app.services.training_service import start_training_job
    body = request.get_json() or {}
    dataset_id = body.get("dataset_id")
    model_type = body.get("model_type", "logistic_regression")
    model_params = body.get("model_params", {})
    preproc_config = body.get("preproc_config", {})
    if not dataset_id:
        return err("dataset_id required.")
    job_id = start_training_job(dataset_id, model_type, model_params, preproc_config)
    return ok({"job_id": job_id, "status": "queued"})


@app.post("/api/models/quantum/train")
def train_quantum():
    return train_model()  # same endpoint handles quantum types


@app.get("/api/models/jobs/<job_id>")
def job_status(job_id):
    from app.services.training_service import get_job
    job = get_job(job_id)
    if not job:
        return err("Job not found.", 404)
    # Never expose internal stack traces in API responses (security).
    # The full traceback is kept in-memory for server-side debugging only.
    safe_job = {k: v for k, v in job.items() if k != "traceback"}
    return ok(safe_job)


@app.get("/api/models")
def list_models_route():
    from app.services.training_service import list_models
    return ok(list_models())


@app.get("/api/models/<model_id>")
def get_model_route(model_id):
    from app.services.training_service import get_model_doc
    doc = get_model_doc(model_id)
    if not doc:
        return err("Model not found.", 404)
    return ok(doc)


@app.get("/api/models/<model_id>/metrics")
def model_metrics(model_id):
    from app.services.training_service import get_model_doc
    doc = get_model_doc(model_id)
    if not doc:
        return err("Model not found.", 404)
    return ok({"metrics": doc.get("metrics", {}), "training_time_s": doc.get("training_time_s")})


@app.get("/api/models/<model_id>/explanations")
def model_explanations(model_id):
    from app.services.training_service import get_model_doc
    doc = get_model_doc(model_id)
    if not doc:
        return err("Model not found.", 404)
    raw = doc.get("explanations", {})
    # Prefer permutation_importance (more meaningful); fall back to feature_importance
    imp_dict = raw.get("permutation_importance") or raw.get("feature_importance") or {}
    method_note = raw.get("method_note", "Feature importance")
    method = "permutation_importance" if "permutation_importance" in raw else "feature_importance"
    is_quantum = doc.get("model_type", "").startswith("quantum") or doc.get("model_type") == "vqc"
    # Sort by absolute importance descending
    sorted_items = sorted(imp_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    feature_names = [k for k, _ in sorted_items]
    importance_values = [v for _, v in sorted_items]
    return ok({
        "feature_names": feature_names,
        "importance_values": importance_values,
        "method": method,
        "method_note": method_note,
        "is_quantum": is_quantum,
        "shap_available": False,
        "raw": raw,
    })


# ─── Experiments ─────────────────────────────────────────────────

@app.get("/api/experiments")
def list_experiments():
    from app.services import storage
    return ok(storage.find_all("experiments"))


@app.get("/api/experiments/<exp_id>")
def get_experiment(exp_id):
    from app.services import storage
    doc = storage.find_one("experiments", exp_id)
    if not doc:
        return err("Experiment not found.", 404)
    return ok(doc)


@app.get("/api/experiments/<exp_id>/comparison")
def experiment_comparison(exp_id):
    from app.services import storage
    from app.services.training_service import get_model_doc
    # Return all models for same dataset
    exp = storage.find_one("experiments", exp_id)
    if not exp:
        return err("Experiment not found.", 404)
    all_models = storage.find_all("models")
    same_ds = [m for m in all_models if m.get("dataset_id") == exp.get("dataset_id")]
    comparison = []
    for m in same_ds:
        comparison.append({
            "model_id": m["_id"],
            "model_type": m["model_type"],
            "metrics": m.get("metrics", {}),
            "training_time_s": m.get("training_time_s"),
            "inference_time_s": m.get("inference_time_s"),
            "train_samples": m.get("train_samples"),
            "test_samples": m.get("test_samples"),
            "created_at": m.get("created_at"),
        })
    return ok({"dataset_id": exp["dataset_id"], "models": comparison})


# ─── Predictions ─────────────────────────────────────────────────

@app.post("/api/predict")
def predict_single_route():
    from app.services.training_service import predict_single
    body = request.get_json() or {}
    model_id = body.get("model_id")
    features = body.get("features", {})
    if not model_id:
        return err("model_id required.")
    try:
        result = predict_single(model_id, features)
        return ok(result)
    except Exception as e:
        return err(str(e))


@app.post("/api/predict/batch")
def predict_batch_route():
    from app.services.training_service import predict_batch
    body = request.get_json() or {}
    model_id = body.get("model_id")
    rows = body.get("rows", [])
    if not model_id or not rows:
        return err("model_id and rows required.")
    try:
        results = predict_batch(model_id, rows)
        return ok(results)
    except Exception as e:
        return err(str(e))


@app.get("/api/predictions")
def list_predictions():
    from app.services import storage
    return ok(storage.find_all("predictions"))


# ─── Reports ─────────────────────────────────────────────────────

@app.get("/api/reports/<exp_id>/json")
def report_json(exp_id):
    from app.services import storage
    from app.services.training_service import get_model_doc
    from app.services.dataset_service import get_dataset

    # exp_id may be an experiment _id OR a model _id (frontend uses model._id)
    exp = storage.find_one("experiments", exp_id)
    if not exp:
        # Try treating it as a model_id and find matching experiment
        model_doc = get_model_doc(exp_id)
        if model_doc:
            all_exps = storage.find_all("experiments")
            exp = next((e for e in all_exps if e.get("model_id") == exp_id), None)
        if not exp:
            return err("Experiment not found.", 404)
    model = get_model_doc(exp.get("model_id", ""))
    ds = get_dataset(exp.get("dataset_id", ""))

    report = {
        "title": "Q-Disease Experiment Report",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "disclaimer": DISCLAIMER,
        "experiment": exp,
        "model": {k: v for k, v in (model or {}).items()
                  if k not in ("artifact_path", "pipeline_path")},
        "dataset": {k: v for k, v in (ds or {}).items() if k != "filepath"},
        "limitations": [
            "This is a research prototype, not a clinical diagnostic tool.",
            "Quantum simulation runtime is not comparable to real quantum hardware.",
            "Metrics are from the held-out test set.",
        ],
    }
    buf = io.BytesIO(json.dumps(report, indent=2, default=str).encode())
    buf.seek(0)
    return send_file(buf, mimetype="application/json",
                     download_name=f"qdisease_report_{exp_id[:8]}.json",
                     as_attachment=True)


@app.get("/api/reports/<exp_id>/csv")
def report_csv(exp_id):
    from app.services import storage
    from app.services.training_service import get_model_doc

    exp = storage.find_one("experiments", exp_id)
    if not exp:
        model_doc = get_model_doc(exp_id)
        if model_doc:
            all_exps = storage.find_all("experiments")
            exp = next((e for e in all_exps if e.get("model_id") == exp_id), None)
        if not exp:
            return err("Experiment not found.", 404)
    model = get_model_doc(exp.get("model_id", ""))
    metrics = (model or {}).get("metrics", {})

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Q-Disease Experiment Report"])
    writer.writerow(["Generated", datetime.utcnow().isoformat()])
    writer.writerow(["Disclaimer", DISCLAIMER])
    writer.writerow([])
    writer.writerow(["Experiment ID", exp_id])
    writer.writerow(["Model Type", exp.get("model_type", "")])
    writer.writerow(["Dataset ID", exp.get("dataset_id", "")])
    writer.writerow(["Status", exp.get("status", "")])
    writer.writerow(["Training Time (s)", exp.get("training_time_s", "")])
    writer.writerow([])
    writer.writerow(["Metric", "Value"])
    for k, v in metrics.items():
        if isinstance(v, (int, float, str)):
            writer.writerow([k, v])
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=qdisease_{exp_id[:8]}.csv"}
    )


# ─── Demo dataset setup ──────────────────────────────────────────

@app.post("/api/demo/load")
def load_demo():
    from sklearn.datasets import load_breast_cancer
    from app.services.dataset_service import register_dataset
    from app.services import storage
    import pandas as pd

    # Check if already loaded
    existing = [d for d in storage.find_all("datasets") if d.get("name") == "Breast Cancer Wisconsin"]
    if existing:
        return ok({"dataset_id": existing[0]["_id"], "message": "Already loaded"})

    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df["diagnosis"] = ["malignant" if t == 1 else "benign" for t in bc.target]
    dest = Path(settings.DATASETS_DIR) / "breast_cancer_wisconsin.csv"
    df.to_csv(dest, index=False)

    ds_id = register_dataset(
        name="Breast Cancer Wisconsin",
        filepath=str(dest),
        target_col="diagnosis",
        exclude_cols=[],
    )
    return ok({
        "dataset_id": ds_id,
        "message": "Breast Cancer Wisconsin dataset loaded",
        "n_rows": len(df),
        "n_features": len(bc.feature_names),
        "provenance": "sklearn.datasets.load_breast_cancer() — UCI ML Repository dataset #17",
        "license": "CC BY 4.0",
    })


if __name__ == "__main__":
    # debug=True is for local development only.
    # In production, run via gunicorn (see Dockerfile CMD).
    # Never set debug=True in a deployed environment.
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=8000, debug=debug)
