from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
import pandas as pd
from app.core.config import settings
from app.services.dataset_service import process_uploaded_csv, save_dataset_metadata, get_dataset_metadata, list_datasets

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@router.get("")
async def get_datasets():
    datasets = list_datasets()
    safe = []
    for d in datasets:
        item = {k: v for k, v in d.items() if k not in ['_id', 'file_path']}
        if 'created_at' in item and hasattr(item['created_at'], 'isoformat'):
            item['created_at'] = item['created_at'].isoformat()
        safe.append(item)
    return {"datasets": safe, "total": len(safe)}

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files are accepted")

    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(400, f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")

    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{dataset_id}.csv")

    with open(file_path, 'wb') as f:
        f.write(content)

    try:
        metadata = process_uploaded_csv(file_path, file.filename)
        metadata['dataset_id'] = dataset_id
        save_dataset_metadata(metadata)

        return {
            "dataset_id": dataset_id,
            "name": file.filename,
            "rows": metadata['rows'],
            "columns": metadata['columns'],
            "column_names": metadata['column_names'],
            "dtypes": metadata['dtypes'],
            "missing_values": metadata['missing_values'],
            "duplicates": metadata['duplicates'],
            "message": "Dataset uploaded successfully"
        }
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(500, f"Failed to process dataset: {str(e)}")

@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    metadata = get_dataset_metadata(dataset_id)
    if not metadata:
        raise HTTPException(404, "Dataset not found")
    item = {k: v for k, v in metadata.items() if k not in ['_id', 'file_path']}
    if 'created_at' in item and hasattr(item['created_at'], 'isoformat'):
        item['created_at'] = item['created_at'].isoformat()
    return item

@router.get("/{dataset_id}/summary")
async def get_dataset_summary(dataset_id: str):
    metadata = get_dataset_metadata(dataset_id)
    if not metadata:
        raise HTTPException(404, "Dataset not found")

    file_path = metadata.get('file_path')
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "Dataset file not found")

    df = pd.read_csv(file_path)
    numeric_df = df.select_dtypes(include=['number'])
    cols = numeric_df.columns[:15].tolist()
    corr = numeric_df[cols].corr().round(3).to_dict() if len(cols) > 1 else {}

    distributions: dict = {}
    for col in cols[:10]:
        counts, bins = pd.cut(df[col].dropna(), bins=20, retbins=True)
        dist = counts.value_counts().sort_index()
        distributions[col] = {
            'counts': dist.values.tolist(),
            'bins': [round(float(b), 4) for b in bins.tolist()]
        }

    return {
        "dataset_id": dataset_id,
        "shape": list(df.shape),
        "correlation_matrix": corr,
        "distributions": distributions,
        "numeric_columns": numeric_df.columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=['number']).columns.tolist(),
    }

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    metadata = get_dataset_metadata(dataset_id)
    if not metadata:
        raise HTTPException(404, "Dataset not found")

    file_path = metadata.get('file_path')
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    from app.database.mongodb import get_db
    db = get_db()
    if db:
        db.datasets.delete_one({"dataset_id": dataset_id})
    else:
        from app.services.dataset_service import _memory_store
        _memory_store.pop(dataset_id, None)

    return {"message": "Dataset deleted successfully"}
