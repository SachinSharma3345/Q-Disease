# Q-Disease

A research-grade web application for training and comparing classical and
quantum-kernel machine learning models on medical datasets.

> **⚠️ Research tool only.** Predictions from this application must **not** be used for
> medical diagnosis, treatment, or clinical decision-making.

---

## Features

- **Dataset management** — Upload CSV datasets, pick a target column, or load the
  built-in breast-cancer Wisconsin demo dataset
- **Data exploration** — Per-column distributions, correlation matrix, summary statistics
- **Preprocessing** — Configurable imputation, scaling, feature selection (k-best / PCA)
- **Model training** — Classical (Logistic Regression, SVM, Random Forest) and quantum
  (Quantum Kernel SVM, VQC) classifiers, all tracked as async jobs
- **Model comparison** — Side-by-side metrics table, bar chart, radar chart, confusion
  matrices
- **Predictions** — Single-sample and batch CSV prediction with probability display
- **Explainability** — Feature importance (coefficients / impurity / permutation)
- **Experiment history** — Searchable log of all training runs

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| npm | ≥ 9 |

Python packages: `flask`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`
(full list in `backend/requirements.txt`)

---

## Setup

### 1. Backend

```bash
# From the project root
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the Flask server (default: http://127.0.0.1:8000)
python app/main.py
```

The backend creates its storage directories automatically on first run:

```
backend/
  storage/
    db/         ← dataset, model, experiment JSON records
  artifacts/    ← trained model .joblib files + preprocessing pipelines
  reports/      ← generated CSV / JSON reports
```

### 2. Frontend

```bash
# From the project root
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server (default: http://localhost:5173)
npm run dev
```

The Vite dev server proxies `/api` requests to `http://localhost:8000`, so both
must be running at the same time.

### Production build

```bash
cd frontend
npm run build        # outputs to frontend/dist/
```

Serve `frontend/dist/` with any static-file server and configure it to forward
`/api/*` to the Flask backend.

---

## Running Tests

Backend tests use Python's built-in `unittest` (no pytest required):

```bash
cd backend
python tests/run_tests.py
```

If pytest is installed:

```bash
cd backend
python -m pytest tests/ -v
```

**Note:** `test_classical_models.py` and `test_preprocessing.py` require
`scikit-learn` and `pandas` to be installed.

---

## Demo Dataset

The app ships with `backend/datasets/breast_cancer_wisconsin.csv`.
Click **Load Demo Dataset** on the Datasets page to register it without uploading
a file.

---

## Security Notes

- Uploaded files are treated as untrusted data — they are never executed
- Trained model artifacts are stored locally and loaded with `joblib`; user-uploaded
  model files are never deserialised
- No credentials, API keys, or secrets are hardcoded
- Stack traces are not exposed in API error responses
- Predicted probabilities are **not** calibrated and must not be interpreted as
  clinical risk scores

---

## Project Structure

```
Q-Disease/
├── backend/
│   ├── app/
│   │   ├── main.py                  ← Flask routes (all under /api/)
│   │   ├── classical/models.py      ← build_classical_model, train_classical
│   │   ├── quantum/                 ← NumPy-based quantum circuit simulator
│   │   ├── preprocessing/pipeline.py
│   │   ├── evaluation/metrics.py
│   │   ├── explainability/
│   │   └── services/
│   │       ├── training_service.py  ← async job queue (in-memory)
│   │       ├── dataset_service.py
│   │       └── storage.py          ← file-based JSON store
│   ├── datasets/                    ← bundled demo CSV
│   └── tests/
│       ├── run_tests.py             ← unittest runner (no pytest needed)
│       ├── test_metrics.py
│       ├── test_classical_models.py
│       └── test_preprocessing.py
└── frontend/
    ├── src/
    │   ├── pages/                   ← one component per route
    │   ├── services/api.ts          ← Axios client (baseURL /api)
    │   └── types.ts
    └── vite.config.ts               ← /api proxy → localhost:8000
```

---

## Known Limitations

- Quantum simulation runtime scales as O(n²) with sample count; keep `max_train_samples`
  ≤ 80 for practical runtimes
- Job status is in-memory only — training jobs are lost on backend restart
- No authentication; intended for single-user local research use
- Temp CSV files from abandoned uploads are not cleaned up automatically
