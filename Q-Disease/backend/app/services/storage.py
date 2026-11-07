"""
File-based JSON storage service (replaces MongoDB when unavailable).
Each collection is a directory of JSON files, one per document.
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import settings

STORE_ROOT = Path(settings.UPLOAD_DIR) / "db"
STORE_ROOT.mkdir(parents=True, exist_ok=True)


def _col_path(collection: str) -> Path:
    p = STORE_ROOT / collection
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def insert(collection: str, doc: Dict[str, Any]) -> str:
    doc_id = doc.get("_id") or str(uuid.uuid4())
    doc["_id"] = doc_id
    if "created_at" not in doc:
        doc["created_at"] = _now()
    path = _col_path(collection) / f"{doc_id}.json"
    path.write_text(json.dumps(doc, default=str))
    return doc_id


def find_one(collection: str, doc_id: str) -> Optional[Dict]:
    path = _col_path(collection) / f"{doc_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def find_all(collection: str) -> List[Dict]:
    col = _col_path(collection)
    docs = []
    for f in sorted(col.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            docs.append(json.loads(f.read_text()))
        except Exception:
            pass
    return docs


def update(collection: str, doc_id: str, updates: Dict[str, Any]) -> bool:
    doc = find_one(collection, doc_id)
    if doc is None:
        return False
    doc.update(updates)
    doc["updated_at"] = _now()
    path = _col_path(collection) / f"{doc_id}.json"
    path.write_text(json.dumps(doc, default=str))
    return True


def delete(collection: str, doc_id: str) -> bool:
    path = _col_path(collection) / f"{doc_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def count(collection: str) -> int:
    return len(list(_col_path(collection).glob("*.json")))
