"""Dataset management service."""
import os
import uuid
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.core.config import settings
from app.services import storage

UPLOAD_DIR = Path(settings.UPLOAD_DIR)


def save_dataset_file(filename: str, content: bytes) -> str:
    """Validate and save an uploaded CSV; raise ValueError with a user-readable message on any problem."""
    # Extension check
    ext = Path(filename).suffix.lower()
    if ext != ".csv":
        raise ValueError("Only CSV files are supported. Please upload a .csv file.")

    # Empty file
    if not content or len(content) == 0:
        raise ValueError("The uploaded file is empty.")

    # Encoding check — must be valid UTF-8
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")  # allow common Windows encoding
        except Exception:
            raise ValueError("File encoding is not supported. Please save the file as UTF-8 CSV.")

    # Must have at least one non-blank line
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) == 0:
        raise ValueError("The CSV file contains no data.")
    if len(lines) < 2:
        raise ValueError("The CSV file must have at least a header row and one data row.")

    uid = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{uid}.csv"
    # Write as UTF-8 regardless of original encoding
    dest.write_text(text, encoding="utf-8")
    return str(dest)


def _dtype_name(dt) -> str:
    if pd.api.types.is_integer_dtype(dt):
        return "integer"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    if pd.api.types.is_bool_dtype(dt):
        return "boolean"
    return "string"


def analyze_dataset(filepath: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Load and analyze a CSV dataset. Raises ValueError for malformed CSVs."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")

    if df.empty or len(df.columns) == 0:
        raise ValueError("CSV file has no columns.")
    if len(df) == 0:
        raise ValueError("CSV file has a header but no data rows.")
    if len(df.columns) < 2:
        raise ValueError("CSV must have at least 2 columns (features + target).")

    if target_col and target_col not in df.columns:
        available = ", ".join(df.columns.tolist()[:10])
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {available}"
        )

    n_rows, n_cols = df.shape
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    columns = []
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": _dtype_name(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
            "unique": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
            col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
            col_info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
            col_info["std"] = float(df[col].std()) if not df[col].isnull().all() else None
        columns.append(col_info)

    target_distribution = {}
    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts()
        target_distribution = {str(k): int(v) for k, v in vc.items()}

    preview = df.head(10).fillna("").astype(str).to_dict(orient="records")

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_total": missing,
        "duplicates": duplicates,
        "columns": columns,
        "target_distribution": target_distribution,
        "preview": preview,
    }


def register_dataset(name: str, filepath: str, target_col: str, exclude_cols: List[str] = None) -> str:
    """Register a dataset and return its ID."""
    analysis = analyze_dataset(filepath, target_col)
    doc = {
        "name": name,
        "filepath": filepath,
        "target_col": target_col,
        "exclude_cols": exclude_cols or [],
        "n_rows": analysis["n_rows"],
        "n_cols": analysis["n_cols"],
        "columns": analysis["columns"],
        "missing_total": analysis["missing_total"],
        "duplicates": analysis["duplicates"],
        "target_distribution": analysis["target_distribution"],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    return storage.insert("datasets", doc)


def get_dataset(dataset_id: str) -> Optional[Dict]:
    return storage.find_one("datasets", dataset_id)


def list_datasets() -> List[Dict]:
    return storage.find_all("datasets")


def delete_dataset(dataset_id: str) -> bool:
    doc = storage.find_one("datasets", dataset_id)
    if doc and os.path.exists(doc.get("filepath", "")):
        try:
            os.remove(doc["filepath"])
        except Exception:
            pass
    return storage.delete("datasets", dataset_id)


def get_feature_distributions(dataset_id: str, n_bins: int = 20) -> Dict[str, Any]:
    """Compute histogram data for each numeric feature."""
    doc = get_dataset(dataset_id)
    if not doc:
        return {}
    df = pd.read_csv(doc["filepath"])
    target = doc.get("target_col")
    excludes = set(doc.get("exclude_cols", []) + ([target] if target else []))
    features = [c for c in df.columns if c not in excludes and pd.api.types.is_numeric_dtype(df[c])]

    result = {}
    for feat in features[:30]:  # cap at 30
        vals = df[feat].dropna()
        counts, edges = np.histogram(vals, bins=n_bins)
        result[feat] = {
            "bins": [round((edges[i] + edges[i+1]) / 2, 4) for i in range(len(counts))],
            "counts": counts.tolist(),
        }
    return result


def get_correlation_matrix(dataset_id: str) -> Dict[str, Any]:
    doc = get_dataset(dataset_id)
    if not doc:
        return {}
    df = pd.read_csv(doc["filepath"])
    target = doc.get("target_col")
    excludes = set(doc.get("exclude_cols", []) + ([target] if target else []))
    num_cols = [c for c in df.columns if c not in excludes and pd.api.types.is_numeric_dtype(df[c])][:20]
    corr = df[num_cols].corr().round(3)
    return {
        "columns": num_cols,
        "matrix": corr.values.tolist(),
    }
