import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    APP_NAME: str = "Q-Disease"
    APP_VERSION: str = "1.0.0"
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(BASE_DIR / "storage"))
    ARTIFACTS_DIR: str = os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "artifacts"))
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", str(BASE_DIR / "reports"))
    DATASETS_DIR: str = os.getenv("DATASETS_DIR", str(BASE_DIR / "datasets"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    RANDOM_SEED: int = 42

    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "q_disease")

settings = Settings()

# Ensure directories exist
for d in [settings.UPLOAD_DIR, settings.ARTIFACTS_DIR, settings.REPORTS_DIR, settings.DATASETS_DIR]:
    os.makedirs(d, exist_ok=True)
