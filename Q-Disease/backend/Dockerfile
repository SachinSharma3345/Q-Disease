# Q-Disease backend — Flask API
# Build:  docker build -t q-disease-backend .
# Run:    docker run -p 8000:8000 q-disease-backend

FROM python:3.11-slim

WORKDIR /app

# System deps (for scipy/numpy native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Persistent storage for datasets, models, and predictions.
# Mount this volume in production to survive container restarts:
#   docker run -v q-disease-data:/app/storage -p 8000:8000 q-disease-backend
RUN mkdir -p storage/db/datasets storage/db/models storage/db/experiments \
             storage/db/predictions artifacts

# Non-root user for security
RUN useradd -m -u 1001 qdisease && chown -R qdisease:qdisease /app
USER qdisease

EXPOSE 8000

# Gunicorn for production; falls back to Flask dev server if not installed.
# SHAP requires enough workers memory — use 1-2 workers per CPU.
CMD ["python", "-m", "gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app.main:app"]
