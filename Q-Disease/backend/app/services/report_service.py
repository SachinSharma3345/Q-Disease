"""Generate downloadable reports (CSV and JSON)."""
import csv
import json
import io
from datetime import datetime, timezone
from typing import Dict

from ..core import storage
from .training_service import get_model


def generate_json_report(experiment_id: str) -> bytes:
    exp = storage.find_one("experiments", {"_id": experiment_id})
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")

    model_doc = get_model(exp.get("model_id", "")) or {}

    report = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": "Q-Disease v1.0",
        "disclaimer": (
            "Q-Disease is an experimental research and educational platform. "
            "It is not a clinically validated medical device and must not be used "
            "to diagnose, treat, or make medical decisions for patients."
        ),
        "experiment": {
            "id": experiment_id,
            "dataset_id": exp.get("dataset_id"),
            "model_type": exp.get("model_type"),
            "status": exp.get("status"),
            "created_at": exp.get("created_at"),
            "completed_at": exp.get("completed_at"),
        },
        "configuration": {
            "model_params": exp.get("params", {}),
            "preprocessing": exp.get("preprocessing", {}),
            "target_column": exp.get("target_column"),
            "test_size": exp.get("test_size"),
        },
        "evaluation_metrics": model_doc.get("metrics", {}),
        "preprocessing_metadata": model_doc.get("preprocessing_meta", {}),
        "circuit_info": model_doc.get("circuit_info"),
        "explainability_summary": {
            k: v.get("top_features") if isinstance(v, dict) and "top_features" in v else None
            for k, v in (model_doc.get("explainability") or {}).items()
        },
        "limitations": [
            "Results are from a simulator, not quantum hardware.",
            "Metrics reflect the specific train/test split, not population-level performance.",
            "Models have not been clinically validated.",
            "Predicted probabilities may not be calibrated.",
        ],
    }

    return json.dumps(report, indent=2, default=str).encode("utf-8")


def generate_csv_report(experiment_id: str) -> bytes:
    exp = storage.find_one("experiments", {"_id": experiment_id})
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")

    model_doc = get_model(exp.get("model_id", "")) or {}
    metrics = model_doc.get("metrics", {})

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Q-Disease Experiment Report"])
    writer.writerow(["Generated", datetime.now(timezone.utc).isoformat()])
    writer.writerow([])

    writer.writerow(["=== Experiment ==="])
    writer.writerow(["experiment_id", experiment_id])
    writer.writerow(["model_type", exp.get("model_type")])
    writer.writerow(["dataset_id", exp.get("dataset_id")])
    writer.writerow(["status", exp.get("status")])
    writer.writerow(["created_at", exp.get("created_at")])
    writer.writerow([])

    writer.writerow(["=== Evaluation Metrics ==="])
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "classification_report", "roc_auc"):
            writer.writerow([k, v])
    writer.writerow([])

    cm = metrics.get("confusion_matrix")
    if cm:
        writer.writerow(["=== Confusion Matrix ==="])
        for row in cm:
            writer.writerow(row)
        writer.writerow([])

    writer.writerow(["=== Disclaimer ==="])
    writer.writerow(["Q-Disease is an experimental research and educational platform."])
    writer.writerow(["Not for clinical use."])

    return output.getvalue().encode("utf-8")
