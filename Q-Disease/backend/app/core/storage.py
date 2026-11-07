"""File-based JSON storage (MongoDB-like interface). Falls back gracefully when MongoDB unavailable."""
import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import STORAGE_DIR

MONGO_AVAILABLE = False
_mongo_client = None
_mongo_db = None


def _try_mongo():
    global MONGO_AVAILABLE, _mongo_client, _mongo_db
    try:
        from ..core.config import MONGO_URI, MONGO_DB
        if not MONGO_URI:
            return
        import pymongo
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        _mongo_client = client
        _mongo_db = client[MONGO_DB]
        MONGO_AVAILABLE = True
    except Exception:
        MONGO_AVAILABLE = False


_try_mongo()


def _collection_path(collection: str) -> Path:
    p = STORAGE_DIR / f"{collection}.json"
    if not p.exists():
        p.write_text("[]")
    return p


def _load(collection: str) -> List[Dict]:
    p = _collection_path(collection)
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _save(collection: str, docs: List[Dict]):
    _collection_path(collection).write_text(
        json.dumps(docs, indent=2, default=str)
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- Public CRUD ----

def insert_one(collection: str, doc: Dict) -> str:
    if MONGO_AVAILABLE:
        try:
            res = _mongo_db[collection].insert_one(doc)
            return str(res.inserted_id)
        except Exception:
            pass
    doc = dict(doc)
    doc.setdefault("_id", str(uuid.uuid4()))
    doc.setdefault("created_at", _now())
    docs = _load(collection)
    docs.append(doc)
    _save(collection, docs)
    return doc["_id"]


def find_one(collection: str, query: Dict) -> Optional[Dict]:
    if MONGO_AVAILABLE:
        try:
            from bson import ObjectId
            if "_id" in query and isinstance(query["_id"], str):
                try:
                    query = dict(query, _id=ObjectId(query["_id"]))
                except Exception:
                    pass
            doc = _mongo_db[collection].find_one(query)
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception:
            pass
    docs = _load(collection)
    for doc in docs:
        if all(doc.get(k) == v for k, v in query.items()):
            return doc
    return None


def find_all(collection: str, query: Optional[Dict] = None, limit: int = 100) -> List[Dict]:
    if MONGO_AVAILABLE:
        try:
            cursor = _mongo_db[collection].find(query or {}).sort("created_at", -1).limit(limit)
            docs = list(cursor)
            for d in docs:
                d["_id"] = str(d["_id"])
            return docs
        except Exception:
            pass
    docs = _load(collection)
    if query:
        docs = [d for d in docs if all(d.get(k) == v for k, v in query.items())]
    return list(reversed(docs))[:limit]


def update_one(collection: str, query: Dict, update: Dict) -> bool:
    if MONGO_AVAILABLE:
        try:
            res = _mongo_db[collection].update_one(query, {"$set": update})
            return res.modified_count > 0
        except Exception:
            pass
    docs = _load(collection)
    for i, doc in enumerate(docs):
        if all(doc.get(k) == v for k, v in query.items()):
            docs[i].update(update)
            _save(collection, docs)
            return True
    return False


def delete_one(collection: str, query: Dict) -> bool:
    if MONGO_AVAILABLE:
        try:
            res = _mongo_db[collection].delete_one(query)
            return res.deleted_count > 0
        except Exception:
            pass
    docs = _load(collection)
    new_docs = [d for d in docs if not all(d.get(k) == v for k, v in query.items())]
    if len(new_docs) < len(docs):
        _save(collection, new_docs)
        return True
    return False


def count(collection: str, query: Optional[Dict] = None) -> int:
    if MONGO_AVAILABLE:
        try:
            return _mongo_db[collection].count_documents(query or {})
        except Exception:
            pass
    docs = _load(collection)
    if query:
        docs = [d for d in docs if all(d.get(k) == v for k, v in query.items())]
    return len(docs)


def get_status() -> Dict:
    return {
        "mongo_available": MONGO_AVAILABLE,
        "storage_type": "mongodb" if MONGO_AVAILABLE else "file",
        "storage_path": str(STORAGE_DIR) if not MONGO_AVAILABLE else None,
    }
