from fastapi import APIRouter, HTTPException
import os
import uuid
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from app.core.config import settings
from app.schemas.models import TrainingConfig, ModelType
from app.services.dataset_service import get_dataset_metadata
from app.preprocessing.pipeline import prepare_data
from app.classical.models import train_classical_model
from app.quantum.models import QuantumKernelSVM, VQCClassifier, QISKIT_AVAILABLE
from app.evaluation.metrics import compute_metrics, timed_inference
from app.explainability.shap_explainer import compute_feature_importance
from app.database.mongodb import get_db

router = APIRouter(prefix="/api/models", tags=["models"])

os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)

_model_store: Dict[str, Dict] = {}
_experiment_store: Dict[str, Dict] = {}

def save_experiment(exp_data: Dict):
    db = get_db()
    exp_id = exp_data['experiment_id']
    if db:
        try:
            db.experiments.insert_one(dict(exp_data))
            db.models.insert_one(dict(exp_data))
        except Exception:
            pass
    _model_store[exp_data.get('model_id', exp_id)] = exp_data
    _experiment_store[exp_id] = exp_data

@router.get("")
async def list_models():
    db = get_db()
    if db:
        docs = list(db.models.find({}, {"_id": 0}).sort("created_at", -1).limit(50))
    else:
        docs = list(_model_store.values())

    safe = []
    for d in docs:
        item = {k: v for k, v in d.items() if k not in ['_id', 'artifact_path', 'pipeline_path']}
        for key in ['created_at', 'completed_at']:
            if key in item and hasattr(item[key], 'isoformat'):
                item[key] = item[key].isoformat()
        safe.append(item)
    return {"models": safe, "total": len(safe)}

@router.get("/{model_id}")
async def get_model(model_id: str):
    data = _model_store.get(model_id)
    if not data:
        raise HTTPException(404, "Model not found")
    return {k: v for k, v in data.items() if k not in ['_id', 'artifact_path', 'pipeline_path']}

@router.get("/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    data = _model_store.get(model_id)
    if not data:
        raise HTTPException(404, "Model not found")
    return data.get('metrics', {})

@router.get("/{model_id}/explanations")
async def get_model_explanations(model_id: str):
    data = _model_store.get(model_id)
    if not data:
        raise HTTPException(404, "Model not found")
    return data.get('explanations', {})

@router.post("/train")
async def train_model(config: TrainingConfig):
    meta = get_dataset_metadata(config.dataset_id)
    if not meta:
        raise HTTPException(404, "Dataset not found")

    file_path = meta.get('file_path')
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "Dataset file not found")

    df = pd.read_csv(file_path)

    if config.target_column not in df.columns:
        raise HTTPException(400, f"Target column '{config.target_column}' not in dataset")

    prep_config = {
        'imputation_strategy': config.imputation_strategy,
        'scaling_method': config.scaling_method,
        'feature_selection': config.feature_selection,
        'n_features': config.n_features,
    }

    try:
        prepared = prepare_data(
            df, config.target_column, config.feature_columns,
            config.exclude_columns or [], config.test_size,
            config.random_seed, prep_config
        )
    except Exception as e:
        raise HTTPException(400, f"Data preparation failed: {str(e)}")

    X_train = prepared['X_train']
    X_test = prepared['X_test']
    y_train = prepared['y_train']
    y_test = prepared['y_test']
    pipeline = prepared['pipeline']

    is_quantum = config.model_type in [ModelType.QUANTUM_KERNEL_SVM, ModelType.VQC]

    if is_quantum and not QISKIT_AVAILABLE:
        raise HTTPException(422,
            "Qiskit is not installed. Run: pip install qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms. "
            "Quantum models require Qiskit to be installed."
        )

    model_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    trained_model = None
    training_time = 0.0

    try:
        if config.model_type == ModelType.QUANTUM_KERNEL_SVM:
            n_train = min(config.n_training_samples or 100, len(X_train))
            idx = np.random.choice(len(X_train), n_train, replace=False)
            model = QuantumKernelSVM(
                n_qubits=config.n_qubits,
                reps=config.circuit_depth,
                feature_map_name=config.feature_map,
                random_seed=config.random_seed
            )
            model.fit(X_train[idx], y_train[idx])
            trained_model = model
            training_time = model.training_time or 0.0

        elif config.model_type == ModelType.VQC:
            n_train = min(config.n_training_samples or 50, len(X_train))
            idx = np.random.choice(len(X_train), n_train, replace=False)
            model = VQCClassifier(
                n_qubits=config.n_qubits,
                reps=config.circuit_depth,
                optimizer_name=config.optimizer,
                max_iter=config.max_iterations,
                random_seed=config.random_seed
            )
            model.fit(X_train[idx], y_train[idx])
            trained_model = model
            training_time = model.training_time or 0.0

        else:
            result = train_classical_model(
                config.model_type.value,
                X_train, y_train,
                config.model_params or {},
                config.random_seed
            )
            trained_model = result['model']
            training_time = result['training_time']

    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")

    infer = timed_inference(trained_model, X_test)
    y_pred = infer['predictions']
    y_prob = infer['probabilities']
    inf_time = infer['inference_time']

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics['training_time'] = training_time
    metrics['inference_time'] = inf_time
    metrics['test_size'] = len(y_test)
    metrics['evaluation_source'] = "held-out test set"

    inv_map = {v: k for k, v in pipeline.target_mapping.items()}
    metrics['positive_class'] = inv_map.get(1, "class_1")

    feature_names = prepared['feature_names']
    if is_quantum:
        feat_names_used = feature_names[:config.n_qubits]
    else:
        feat_names_used = feature_names[:X_test.shape[1]]

    explanations = compute_feature_importance(
        trained_model, X_test, y_test,
        feat_names_used, config.model_type.value, config.random_seed
    )

    artifact_path = os.path.join(settings.ARTIFACTS_DIR, f"{model_id}.joblib")
    pipeline_path = os.path.join(settings.ARTIFACTS_DIR, f"{model_id}_pipeline.joblib")
    joblib.dump(trained_model, artifact_path)
    pipeline.save(pipeline_path)

    exp_data = {
        'experiment_id': experiment_id,
        'model_id': model_id,
        'dataset_id': config.dataset_id,
        'model_type': config.model_type.value,
        'target_column': config.target_column,
        'feature_names': feature_names,
        'preprocessing_config': prep_config,
        'model_params': config.model_params,
        'metrics': metrics,
        'explanations': explanations,
        'status': 'completed',
        'artifact_path': artifact_path,
        'pipeline_path': pipeline_path,
        'created_at': datetime.utcnow(),
        'completed_at': datetime.utcnow(),
        'train_size': prepared['train_size'],
        'test_size_n': prepared['test_size'],
        'n_features_processed': prepared['n_features_processed'],
        'qiskit_used': QISKIT_AVAILABLE if is_quantum else False,
    }

    save_experiment(exp_data)

    return {
        'experiment_id': experiment_id,
        'model_id': model_id,
        'model_type': config.model_type.value,
        'status': 'completed',
        'metrics': metrics,
        'explanations': explanations,
        'training_time': training_time,
        'message': f"Model trained on {prepared['train_size']} samples, evaluated on {prepared['test_size']} held-out test samples"
    }

@router.post("/quantum/train")
async def train_quantum_model(config: TrainingConfig):
    if config.model_type not in [ModelType.QUANTUM_KERNEL_SVM, ModelType.VQC]:
        config.model_type = ModelType.QUANTUM_KERNEL_SVM
    return await train_model(config)
