from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

client = None
db = None

def connect_db():
    global client, db
    try:
        client = MongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=3000)
        client.admin.command('ping')
        db = client[settings.MONGODB_DB]
        db.datasets.create_index([("created_at", -1)])
        db.experiments.create_index([("created_at", -1)])
        db.models.create_index([("created_at", -1)])
        db.predictions.create_index([("created_at", -1)])
        logger.info("MongoDB connected successfully")
        return True
    except Exception as e:
        logger.warning(f"MongoDB unavailable: {e}. Running without persistence.")
        client = None
        db = None
        return False

def get_db():
    return db

def is_connected():
    return db is not None

def disconnect_db():
    global client, db
    if client:
        client.close()
    client = None
    db = None
