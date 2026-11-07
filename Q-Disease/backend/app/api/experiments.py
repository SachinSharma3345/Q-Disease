from fastapi import APIRouter, HTTPException
from app.api.models import _experiment_store, _model_store
from app.database.mongodb import get_db

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

@router.get("")
async def list_experiments():
    db = get_db()
    if db:
        docs = list(db.experiments.find({}, {"_id": 0}).sort("created_at", -1).limit(50))
    else:
        docs = list(_experiment_store.values())
    safe = []
    for d in docs:
        item = {k: v for k, v in d.items() if k not in ['_id', 'artifact_path', 'pipeline_path']}
        for key in ['created_at', 'completed_at']:
            if key in item and hasattr(item[key], 'isoformat'):
                item[key] = item[key].isoformat()
        safe.append(item)
    return {"experiments": safe, "total": len(safe)}

@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    data = _experiment_store.get(experiment_id)
    if not data:
        raise HTTPException(404, "Experiment not found")
    item = {k: v for k, v in data.items() if k not in ['_id', 'artifact_path', 'pipeline_path']}
    for key in ['created_at', 'completed_at']:
        if key in item and hasattr(item[key], 'isoformat'):
            item[key] = item[key].isoformat()
    return item

@router.get("/{experiment_id}/comparison")
async def get_comparison(experiment_id: str):
    all_exp = []
    for d in _experiment_store.values():
        item = {k: v for k, v in d.items() if k not in ['_id', 'artifact_path', 'pipeline_path']}
        for key in ['created_at', 'completed_at']:
            if key in item and hasattr(item[key], 'isoformat'):
                item[key] = item[key].isoformat()
        all_exp.append(item)
    return {"experiments": all_exp, "total": len(all_exp)}
