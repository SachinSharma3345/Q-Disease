from fastapi import APIRouter, HTTPException
import uuid
import joblib
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from app.schemas.models import PredictionRequest
from app.api.models import _model_store

router = APIRouter(prefix="/api/predict", tags=["predictions"])

_prediction_store = []

@router.post("")
async def predict(request: PredictionRequest):
    model_data = _model_store.get(request.model_id)
    if not model_data:
        raise HTTPException(404, "Model not found")

    pipeline_path = model_data.get('pipeline_path')
        # Get the stored model artifact path
    artifact_path = model_data.get("artifact_path")

    # Resolve stale paths from the local artifacts directory
    if not artifact_path or not os.path.exists(artifact_path):
        model_id = str(
            model_data.get("_id") or model_data.get("id") or ""
        )

        artifacts_dir = Path(settings.ARTIFACTS_DIR)

        candidates = [
            artifacts_dir / f"{model_id}.pkl",
            artifacts_dir / f"{model_id}.joblib",
        ]

        artifact_path = next(
            (str(path) for path in candidates if path.is_file()),
            None,
        )

    if not artifact_path:
        raise FileNotFoundError(
            f"Model artifact not found for model ID: "
            f"{model_data.get('_id') or model_data.get('id')}"
        )

    model = joblib.load(artifact_path)

    feature_names = model_data.get('feature_names', [])
    try:
        vals = {fn: [request.features.get(fn, 0.0)] for fn in feature_names}
        X_input = pd.DataFrame(vals)
        X_proc = pipeline.transform(X_input)
    except Exception as e:
        raise HTTPException(400, f"Feature processing failed: {str(e)}")

    pred = model.predict(X_proc)[0]
    prob = None
    if hasattr(model, 'predict_proba'):
        prob_arr = model.predict_proba(X_proc)[0]
        prob = float(prob_arr[int(pred)])

    inv_map = {v: k for k, v in pipeline.target_mapping.items()}
    pred_class = inv_map.get(int(pred), str(pred))

    risk = "High Risk" if int(pred) == 1 else "Low Risk"
    if prob is not None:
        if prob > 0.8:
            risk = "Very High Risk" if int(pred) == 1 else "Very Low Risk"
        elif prob > 0.6:
            risk = "High Risk" if int(pred) == 1 else "Low Risk"
        else:
            risk = "Moderate Risk"

    pred_id = str(uuid.uuid4())
    result = {
        'prediction_id': pred_id,
        'model_id': request.model_id,
        'predicted_class': pred_class,
        'probability': prob,
        'risk_category': risk,
        'timestamp': datetime.utcnow(),
        'disclaimer': "This is an experimental research model. Not a clinical diagnosis."
    }
    _prediction_store.append(result)

    return {**result, 'timestamp': result['timestamp'].isoformat()}

@router.get("/history")
async def get_prediction_history():
    safe = []
    for p in _prediction_store[-50:]:
        item = dict(p)
        if hasattr(item.get('timestamp'), 'isoformat'):
            item['timestamp'] = item['timestamp'].isoformat()
        safe.append(item)
    return {"predictions": safe, "total": len(safe)}
