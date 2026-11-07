from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import csv
import io
from datetime import datetime
from app.api.models import _experiment_store

router = APIRouter(prefix="/api/reports", tags=["reports"])

@router.get("/{experiment_id}/json")
async def download_report_json(experiment_id: str):
    data = _experiment_store.get(experiment_id)
    if not data:
        raise HTTPException(404, "Experiment not found")

    report = {
        'report_generated': datetime.utcnow().isoformat(),
        'disclaimer': "Q-Disease is experimental. Not for clinical use.",
        'experiment_id': experiment_id,
        'model_type': data.get('model_type'),
        'dataset_id': data.get('dataset_id'),
        'target_column': data.get('target_column'),
        'preprocessing_config': data.get('preprocessing_config'),
        'model_params': data.get('model_params'),
        'train_size': data.get('train_size'),
        'test_size': data.get('test_size_n'),
        'metrics': data.get('metrics'),
        'explanations': {k: v for k, v in (data.get('explanations') or {}).items()},
        'limitations': [
            "This is a research prototype, not a clinical tool.",
            "Quantum simulator performance is not representative of real quantum hardware.",
            "Results should not be used for medical decisions."
        ]
    }

    content = json.dumps(report, indent=2, default=str)
    return StreamingResponse(
        io.StringIO(content),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=report_{experiment_id[:8]}.json"}
    )

@router.get("/{experiment_id}/csv")
async def download_report_csv(experiment_id: str):
    data = _experiment_store.get(experiment_id)
    if not data:
        raise HTTPException(404, "Experiment not found")

    metrics = data.get('metrics', {})
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Field", "Value"])
    writer.writerow(["Experiment ID", experiment_id])
    writer.writerow(["Model Type", data.get('model_type')])
    writer.writerow(["Status", data.get('status')])
    writer.writerow(["Train Size", data.get('train_size')])
    writer.writerow(["Test Size", data.get('test_size_n')])
    writer.writerow(["Accuracy", metrics.get('accuracy', 'N/A')])
    writer.writerow(["Precision", metrics.get('precision', 'N/A')])
    writer.writerow(["Recall", metrics.get('recall', 'N/A')])
    writer.writerow(["F1 Score", metrics.get('f1_score', 'N/A')])
    writer.writerow(["ROC-AUC", metrics.get('roc_auc', 'N/A')])
    writer.writerow(["Sensitivity", metrics.get('sensitivity', 'N/A')])
    writer.writerow(["Specificity", metrics.get('specificity', 'N/A')])
    writer.writerow(["Training Time (s)", metrics.get('training_time', 'N/A')])
    writer.writerow(["Inference Time (s)", metrics.get('inference_time', 'N/A')])
    writer.writerow(["DISCLAIMER", "This model is experimental and not for clinical use."])
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=report_{experiment_id[:8]}.csv"}
    )
